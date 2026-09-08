"""knock-ocr demo API

单进程 + 流水线池的最小可用服务，为后续高并发留好接口形状：
  POST /api/ocr      上传图片/PDF -> markdown + 结构化 json
  GET  /api/info     当前配置与后端状态
  GET  /api/metrics  进程内计数与延迟分位
  GET  /healthz      就绪探针
  GET  /             Web 界面

引擎由 OCR_ENGINE 决定（见 engines.py）：
  vl        PaddleOCR-VL + vLLM，服务器用
  rapidocr  纯 CPU 轻量引擎，本地开发用
两者对外接口完全一致，换引擎不动前端和调用方。

刻意只依赖运行镜像里已有的包（fastapi / uvicorn / starlette），
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

from engines import get_engine

APP_DIR = Path(__file__).parent
STATIC = APP_DIR / "static"

ENGINE = get_engine()
WORKERS = int(os.environ.get("OCR_WORKERS", "4"))
MAX_MB = int(os.environ.get("OCR_MAX_MB", "50"))
# 单请求排队等流水线的上限；超时直接 503，避免请求堆积拖垮服务
ACQUIRE_TIMEOUT = float(os.environ.get("OCR_ACQUIRE_TIMEOUT", "120"))

ALLOWED = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
if ENGINE.supports_pdf:
    ALLOWED = ALLOWED | {".pdf"}

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
    """预热 N 个引擎实例轮流用。

    实例本身不保证线程安全，所以用队列做独占借还，而不是共享单例。
    这也是后面换成多进程 / 多机 worker 时最自然的切分点。
    """

    def __init__(self, size: int):
        self.size = size
        self._q: "queue.Queue" = queue.Queue()
        self.ready = 0
        self.error: str | None = None
        self._lock = threading.Lock()

    def warmup(self):
        print(f"[pool] 开始构建 {self.size} 条流水线  {ENGINE.describe()}", flush=True)
        for i in range(self.size):
            t0 = time.perf_counter()
            print(f"[pool] 构建第 {i + 1}/{self.size} 条…（首次可能要下载模型）", flush=True)
            try:
                self._q.put(ENGINE.build())
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


# --------------------------------------------------------------- 应用
@asynccontextmanager
async def lifespan(_: FastAPI):
    # 后台预热：容器立刻可探活，界面能显示 "流水线 0/N 预热中"
    threading.Thread(target=POOL.warmup, daemon=True, name="pool-warmup").start()
    yield


app = FastAPI(title="knock-ocr demo", version="0.2.0",
              docs_url="/api/docs", lifespan=lifespan)


@app.get("/healthz")
def healthz():
    return {
        "ready": POOL.ready > 0,
        "engine": ENGINE.name,
        "pool_ready": POOL.ready,
        "pool_size": POOL.size,
        "pool_idle": POOL.idle,
        "error": POOL.error,
    }


def _backend_status() -> dict:
    """只有 vl 引擎才有外部推理服务需要探活。"""
    url = getattr(ENGINE, "server_url", None)
    if not url:
        return {"reachable": True, "note": "本引擎无外部依赖"}
    probe = url.rstrip("/") + "/models"
    out = {"url": url, "probe": probe, "reachable": False, "models": []}
    try:
        import urllib.request

        with urllib.request.urlopen(probe, timeout=10) as r:
            data = json.loads(r.read().decode())
            out["reachable"] = True
            out["models"] = [m.get("id") for m in data.get("data", [])]
    except Exception as e:
        out["error"] = str(e)
    return out


@app.get("/api/info")
def info():
    d = ENGINE.describe()
    return {
        "engine": ENGINE.name,
        "model": d.get("model", ENGINE.name),
        "benchmark": "OmniDocBench v1.6 96.3%" if ENGINE.name == "vl" else "轻量 CPU 引擎",
        "layout_device": d.get("layout_device", "cpu"),
        "supports_pdf": ENGINE.supports_pdf,
        "engine_detail": d,
        "vlm_backend": _backend_status(),
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

    # 池子没就绪就立刻拒绝，别让请求傻等 ACQUIRE_TIMEOUT 秒把界面挂住
    if POOL.ready == 0:
        detail = f"服务尚未就绪（流水线 0/{POOL.size}）。"
        detail += ("构建流水线时报错了，用 `manage.sh logs api` 看栈。"
                   if POOL.error else "仍在初始化，请稍候；进度看 `manage.sh logs api`。")
        raise HTTPException(503, detail)

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
            415, f"当前引擎（{ENGINE.name}）不支持这个文件类型（{filename!r}）。"
                 f"支持：{sorted(ALLOWED)}")

    t0 = time.perf_counter()
    ok = False
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as fh:
            fh.write(blob)
            tmp = fh.name

        def work():
            with POOL.acquire(ACQUIRE_TIMEOUT) as handle:
                return ENGINE.run(handle, tmp, merge_tables)

        md, pages_json, npages = await run_in_threadpool(work)
        ok = True
        body = {
            "request_id": rid,
            "engine": ENGINE.name,
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
    except Exception as e:
        print(f"[{rid}] 识别失败:\n{traceback.format_exc(limit=6)}", flush=True)
        raise HTTPException(500, f"识别失败 (request_id={rid}): {type(e).__name__}: {e}")
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
