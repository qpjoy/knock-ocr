"""knock-ocr demo API

单进程 + 流水线池的最小可用服务，为后续高并发留好接口形状：
  POST /api/ocr      上传图片/PDF -> markdown + 结构化 json
  GET  /api/info     当前配置与后端状态
  GET  /api/metrics  进程内计数与延迟分位
  GET  /healthz      就绪探针
  GET  /             Web 界面

刻意只依赖官方镜像里已有的包（fastapi / uvicorn / starlette），
不引入 python-multipart —— 内网构建时 pip 可能出不去，多一个依赖就多一个卡点。
"""
from __future__ import annotations

import json
import os
import queue
import statistics
import tempfile
import threading
import time
import traceback
import uuid
from contextlib import asynccontextmanager, contextmanager
from email.parser import BytesParser
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse
from starlette.concurrency import run_in_threadpool

APP_DIR = Path(__file__).parent
STATIC = APP_DIR / "static"

DEVICE = os.environ.get("OCR_DEVICE", "cpu")
MODEL = os.environ.get("OCR_MODEL", "PaddleOCR-VL-1.6-0.9B")
# backend 必须是 "vllm-server"（调用远端服务），不是 "vllm"（在本进程内自起引擎）。
# 用 "vllm" 会让这个没有 GPU 的 API 容器自己去加载 0.9B 模型，然后无声卡死。
# 依据是 genai_server 启动时打印的用法提示：
#   --vl_rec_backend vllm-server --vl_rec_server_url http://localhost:8118/v1
VL_BACKEND = os.environ.get("OCR_VL_BACKEND", "vllm-server")


def _normalize_vllm_url(url: str) -> str:
    """服务端要求带 /v1 后缀，统一补齐，避免配置漏写。"""
    url = (url or "").rstrip("/")
    if not url.endswith("/v1"):
        url += "/v1"
    return url


VLLM_URL = _normalize_vllm_url(os.environ.get("OCR_VLLM_URL", "http://127.0.0.1:8118"))
# VLLM_URL 已经以 /v1 结尾，探活地址只需再接 /models，别重复拼 /v1
MODELS_URL = VLLM_URL + "/models"
WORKERS = int(os.environ.get("OCR_WORKERS", "4"))
MAX_MB = int(os.environ.get("OCR_MAX_MB", "50"))
# 单请求排队等流水线的上限；超时直接 503，避免请求堆积拖垮服务
ACQUIRE_TIMEOUT = float(os.environ.get("OCR_ACQUIRE_TIMEOUT", "120"))

ALLOWED = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff", ".pdf"}

# 魔数比文件名可靠：粘贴上来的截图往往没有正经文件名
MAGIC = (
    (b"%PDF", ".pdf"),
    (bytes.fromhex("89504e470d0a1a0a"), ".png"),
    (bytes.fromhex("ffd8ff"), ".jpg"),
    (b"BM", ".bmp"),
    (bytes.fromhex("49492a00"), ".tif"),
    (bytes.fromhex("4d4d002a"), ".tif"),
)

CRLF = bytes.fromhex("0d0a")


def sniff_suffix(blob: bytes, filename: str) -> str:
    if blob[:4] == b"RIFF" and blob[8:12] == b"WEBP":
        return ".webp"
    for magic, suffix in MAGIC:
        if blob.startswith(magic):
            return suffix
    suffix = Path(filename or "").suffix.lower()
    return suffix if suffix in ALLOWED else ""


def parse_upload(content_type: str, body: bytes) -> tuple[str, bytes]:
    """取出上传的文件，返回 (filename, bytes)。

    两种调用方式都支持：
        curl -F 'file=@a.png' http://host/api/ocr
        curl --data-binary @a.png -H 'X-Filename: a.png' http://host/api/ocr
    """
    ctype = (content_type or "").strip()
    if not ctype.lower().startswith("multipart/"):
        return "", body

    header = b"Content-Type: " + ctype.encode("latin-1", "replace") + CRLF + CRLF
    try:
        msg = BytesParser().parsebytes(header + body)
    except Exception:
        raise HTTPException(400, "multipart 解析失败")

    if not msg.is_multipart():
        raise HTTPException(400, "multipart 格式不正确")

    fallback = None
    for part in msg.walk():
        if part.is_multipart():
            continue
        payload = part.get_payload(decode=True)
        if not payload:
            continue
        fname = part.get_filename() or ""
        if part.get_param("name", header="content-disposition") == "file" or fname:
            return fname, payload
        if fallback is None:
            fallback = (fname, payload)
    if fallback:
        return fallback
    raise HTTPException(400, "multipart 中没有找到文件")


# --------------------------------------------------------------- 流水线池
class PipelinePool:
    """预热 N 个 PaddleOCRVL 实例轮流用。

    实例本身不保证线程安全，所以用队列做独占借还，而不是共享单例。
    这也是后面换成多进程 / 多机 worker 时最自然的切分点。
    """

    def __init__(self, size: int):
        self.size = size
        self._q: "queue.Queue" = queue.Queue()
        self.ready = 0
        self.error: str | None = None
        self._lock = threading.Lock()

    def _build(self):
        from paddleocr import PaddleOCRVL

        kwargs = {
            "vl_rec_backend": VL_BACKEND,
            "vl_rec_server_url": VLLM_URL,
            "vl_rec_api_model_name": MODEL,
        }
        if DEVICE:
            kwargs["device"] = DEVICE
        return PaddleOCRVL(**kwargs)

    def warmup(self):
        print(f"[pool] 开始构建 {self.size} 条流水线  backend={VL_BACKEND}  "
              f"url={VLLM_URL}  device={DEVICE}  model={MODEL}", flush=True)
        for i in range(self.size):
            t0 = time.perf_counter()
            print(f"[pool] 构建第 {i + 1}/{self.size} 条…"
                  f"（首次要下载版面分析模型，可能要几分钟）", flush=True)
            try:
                self._q.put(self._build())
                with self._lock:
                    self.ready += 1
                print(f"[pool] 第 {i + 1}/{self.size} 条就绪"
                      f"（{time.perf_counter() - t0:.1f}s）", flush=True)
            except Exception:
                self.error = traceback.format_exc(limit=6)
                print(f"[pool] 第 {i + 1} 条构建失败:\n{self.error}", flush=True)
                return
        print(f"[pool] 全部 {self.ready} 条流水线就绪", flush=True)

    @contextmanager
    def acquire(self, timeout: float):
        try:
            item = self._q.get(timeout=timeout)
        except queue.Empty:
            raise HTTPException(503, "服务繁忙，流水线全部占用中，请重试")
        try:
            yield item
        finally:
            self._q.put(item)

    @property
    def idle(self) -> int:
        return self._q.qsize()


POOL = PipelinePool(WORKERS)


# --------------------------------------------------------------- 指标
class Metrics:
    def __init__(self):
        self.total = 0
        self.failed = 0
        self.lat: list[float] = []
        self._lock = threading.Lock()

    def record(self, ms: float, ok: bool):
        with self._lock:
            self.total += 1
            if not ok:
                self.failed += 1
            self.lat.append(ms)
            if len(self.lat) > 2000:
                del self.lat[:1000]

    def snapshot(self) -> dict:
        with self._lock:
            lat = sorted(self.lat)

        def pct(p):
            if not lat:
                return None
            return round(lat[min(len(lat) - 1, int(len(lat) * p))], 1)

        return {
            "total": self.total,
            "failed": self.failed,
            "p50_ms": pct(0.50),
            "p95_ms": pct(0.95),
            "p99_ms": pct(0.99),
            "mean_ms": round(statistics.fmean(lat), 1) if lat else None,
        }


METRICS = Metrics()


# --------------------------------------------------------------- 结果提取
def _markdown_of(res) -> str:
    """PaddleOCR 各版本 markdown 返回形状不完全一致，按优先级兜底取。"""
    m = getattr(res, "markdown", None)
    if isinstance(m, str):
        return m
    if isinstance(m, dict):
        for key in ("markdown_texts", "markdown_text", "text", "md"):
            v = m.get(key)
            if isinstance(v, str) and v.strip():
                return v
        parts = [v for v in m.values() if isinstance(v, str)]
        if parts:
            return "\n\n".join(parts)
    return ""


def _json_of(res) -> dict:
    j = getattr(res, "json", None)
    if callable(j):
        try:
            j = j()
        except Exception:
            j = None
    if isinstance(j, dict):
        return j.get("res", j)
    return {}


def _run(pipeline, path: str, merge_tables: bool) -> tuple[str, list, int]:
    pages = list(pipeline.predict(path))

    # 多页 PDF 走官方重组，跨页表格能接起来
    if len(pages) > 1 and hasattr(pipeline, "restructure_pages"):
        try:
            merged = pipeline.restructure_pages(pages, merge_tables=merge_tables)
            md = _markdown_of(merged)
            if md:
                return md, [_json_of(p) for p in pages], len(pages)
        except Exception:
            print("[warn] restructure_pages 失败，回落逐页拼接:\n"
                  + traceback.format_exc(limit=3), flush=True)

    md = "\n\n---\n\n".join(filter(None, (_markdown_of(p) for p in pages)))
    return md, [_json_of(p) for p in pages], len(pages)


# --------------------------------------------------------------- 应用
@asynccontextmanager
async def lifespan(_: FastAPI):
    # 后台预热：容器立刻可探活，界面能显示 "流水线 0/4 预热中"
    threading.Thread(target=POOL.warmup, daemon=True, name="pool-warmup").start()
    yield


app = FastAPI(title="knock-ocr demo", version="0.1.0",
              docs_url="/api/docs", lifespan=lifespan)


@app.get("/healthz")
def healthz():
    return {
        "ready": POOL.ready > 0,
        "pool_ready": POOL.ready,
        "pool_size": POOL.size,
        "pool_idle": POOL.idle,
        "error": POOL.error,
    }


@app.get("/api/info")
def info():
    backend = {"url": VLLM_URL, "probe": MODELS_URL, "reachable": False, "models": []}
    try:
        import urllib.request

        with urllib.request.urlopen(MODELS_URL, timeout=3) as r:
            data = json.loads(r.read().decode())
            backend["reachable"] = True
            backend["models"] = [m.get("id") for m in data.get("data", [])]
    except Exception as e:
        backend["error"] = str(e)

    return {
        "model": MODEL,
        "benchmark": "OmniDocBench v1.6 96.3%",
        "layout_device": DEVICE,
        "vl_backend": VL_BACKEND,
        "vlm_backend": backend,
        "workers": WORKERS,
        "pool": {"ready": POOL.ready, "idle": POOL.idle, "size": POOL.size},
        "metrics": METRICS.snapshot(),
    }


@app.get("/api/metrics")
def metrics():
    return {**METRICS.snapshot(), "pool_idle": POOL.idle, "pool_size": POOL.size}


@app.post("/api/ocr")
async def ocr(
    request: Request,
    merge_tables: bool = Query(True, description="多页 PDF 是否合并跨页表格"),
    include_json: bool = Query(True, description="是否返回结构化结果"),
):
    rid = uuid.uuid4().hex[:12]

    raw = await request.body()
    if not raw:
        raise HTTPException(400, "空请求体")
    if len(raw) > MAX_MB * 1024 * 1024:
        raise HTTPException(413, f"请求超过 {MAX_MB}MB")

    filename, blob = parse_upload(request.headers.get("content-type", ""), raw)
    filename = filename or request.headers.get("x-filename", "") or "upload"
    if not blob:
        raise HTTPException(400, "空文件")

    suffix = sniff_suffix(blob, filename)
    if suffix not in ALLOWED:
        raise HTTPException(
            415, f"无法识别的文件类型（文件名 {filename!r}）。支持：{sorted(ALLOWED)}")

    t0 = time.perf_counter()
    ok = False
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as fh:
            fh.write(blob)
            tmp = fh.name

        def work():
            with POOL.acquire(ACQUIRE_TIMEOUT) as pipeline:
                return _run(pipeline, tmp, merge_tables)

        md, pages_json, npages = await run_in_threadpool(work)
        ok = True
        body = {
            "request_id": rid,
            "filename": filename,
            "pages": npages,
            "elapsed_ms": round((time.perf_counter() - t0) * 1000, 1),
            "markdown": md,
        }
        if include_json:
            body["result"] = pages_json
        return JSONResponse(body)

    except HTTPException:
        raise
    except Exception:
        print(f"[{rid}] 识别失败:\n{traceback.format_exc(limit=6)}", flush=True)
        raise HTTPException(500, f"识别失败 (request_id={rid})")
    finally:
        METRICS.record((time.perf_counter() - t0) * 1000, ok)
        if tmp:
            try:
                os.unlink(tmp)
            except OSError:
                pass


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")
