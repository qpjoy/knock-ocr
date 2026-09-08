"""knock-ocr demo API

单进程 + 流水线池的最小可用服务，为后续高并发留好接口形状：
  POST /api/ocr      上传图片/PDF -> markdown + 结构化 json
  GET  /api/info     当前配置与后端状态
  GET  /api/metrics  进程内计数与延迟分位
  GET  /healthz      就绪探针
  GET  /             Web 界面
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
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from starlette.concurrency import run_in_threadpool

APP_DIR = Path(__file__).parent
STATIC = APP_DIR / "static"

DEVICE = os.environ.get("OCR_DEVICE", "cpu")
MODEL = os.environ.get("OCR_MODEL", "PaddleOCR-VL-1.6-0.9B")
VLLM_URL = os.environ.get("OCR_VLLM_URL", "http://127.0.0.1:8118")
WORKERS = int(os.environ.get("OCR_WORKERS", "4"))
MAX_MB = int(os.environ.get("OCR_MAX_MB", "50"))
# 单请求排队等流水线的上限；超时直接 503，避免请求堆积拖垮服务
ACQUIRE_TIMEOUT = float(os.environ.get("OCR_ACQUIRE_TIMEOUT", "120"))

ALLOWED = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff", ".pdf"}


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
            "vl_rec_backend": "vllm",
            "vl_rec_server_url": VLLM_URL,
            "vl_rec_api_model_name": MODEL,
        }
        if DEVICE:
            kwargs["device"] = DEVICE
        return PaddleOCRVL(**kwargs)

    def warmup(self):
        for i in range(self.size):
            try:
                self._q.put(self._build())
                with self._lock:
                    self.ready += 1
            except Exception:
                self.error = traceback.format_exc(limit=4)
                print(f"[pool] 第 {i + 1} 个流水线构建失败:\n{self.error}", flush=True)
                return
        print(f"[pool] {self.ready} 个流水线就绪 device={DEVICE} vllm={VLLM_URL}", flush=True)

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
    out = pipeline.predict(path)
    pages = list(out)

    # 多页 PDF 走官方重组，跨页表格能接起来
    if len(pages) > 1 and hasattr(pipeline, "restructure_pages"):
        try:
            merged = pipeline.restructure_pages(pages, merge_tables=merge_tables)
            md = _markdown_of(merged)
            if md:
                return md, [_json_of(p) for p in pages], len(pages)
        except Exception:
            print("[warn] restructure_pages 失败，回落逐页拼接:\n" + traceback.format_exc(limit=3),
                  flush=True)

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
    backend = {"url": VLLM_URL, "reachable": False, "models": []}
    try:
        import urllib.request

        with urllib.request.urlopen(f"{VLLM_URL}/v1/models", timeout=3) as r:
            data = json.loads(r.read().decode())
            backend["reachable"] = True
            backend["models"] = [m.get("id") for m in data.get("data", [])]
    except Exception as e:
        backend["error"] = str(e)

    return {
        "model": MODEL,
        "benchmark": "OmniDocBench v1.6 96.3%",
        "layout_device": DEVICE,
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
    file: UploadFile = File(...),
    merge_tables: bool = Query(True, description="多页 PDF 是否合并跨页表格"),
    include_json: bool = Query(True, description="是否返回结构化结果"),
):
    rid = uuid.uuid4().hex[:12]
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED:
        raise HTTPException(415, f"不支持的类型 {suffix or '(空)'}，支持：{sorted(ALLOWED)}")

    blob = await file.read()
    if not blob:
        raise HTTPException(400, "空文件")
    if len(blob) > MAX_MB * 1024 * 1024:
        raise HTTPException(413, f"文件超过 {MAX_MB}MB")

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
        elapsed = (time.perf_counter() - t0) * 1000
        body = {
            "request_id": rid,
            "filename": file.filename,
            "pages": npages,
            "elapsed_ms": round(elapsed, 1),
            "markdown": md,
        }
        if include_json:
            body["result"] = pages_json
        return JSONResponse(body)

    except HTTPException:
        raise
    except Exception:
        tb = traceback.format_exc(limit=6)
        print(f"[{rid}] 识别失败:\n{tb}", flush=True)
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
