"""knock-ocr API —— 一个接口，两种引擎，按需路由。

  POST /api/ocr?engine=fast|quality   上传图片/PDF -> markdown + 结构化 json
  GET  /api/info                      当前档位、引擎、后端状态、实时指标
  GET  /api/metrics                   计数与 P50/P95/P99（分引擎）
  POST /api/config                    在线调参（含并发，会后台重建流水线）
  GET  /healthz                       就绪探针
  GET  /                              Web 界面

部署档由 OCR_TIER 决定：fast | quality | full。
fast 档不装 PaddleOCR-VL，镜像 <1GB，也不需要 GPU。

刻意只依赖运行镜像里已有的包（fastapi / uvicorn / starlette），
不引入 python-multipart —— 内网构建 pip 可能出不去，多一个依赖就多一个卡点。
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

from engines import IMAGE_SUFFIXES, build_engines, default_engine_name

APP_DIR = Path(__file__).parent
STATIC = APP_DIR / "static"

TIER = os.environ.get("OCR_TIER", "quality").strip().lower()
MAX_MB = int(os.environ.get("OCR_MAX_MB", "50"))
# 单请求排队等流水线的上限；超时直接 503，避免请求堆积拖垮服务
ACQUIRE_TIMEOUT = float(os.environ.get("OCR_ACQUIRE_TIMEOUT", "120"))


def _num_env(name, cast=int):
    v = os.environ.get(name, "").strip()
    return cast(v) if v else None


# 运行时可调。per-request 三项每次 predict 传，立即生效；
# vl_rec_max_concurrency 是构造期参数，改了要重建 quality 池。
TUNING = {
    "vl_rec_max_concurrency": _num_env("OCR_VL_CONCURRENCY") or 8,
    "max_pixels": _num_env("OCR_MAX_PIXELS"),
    "max_new_tokens": _num_env("OCR_MAX_NEW_TOKENS"),
    "layout_threshold": _num_env("OCR_LAYOUT_THRESHOLD", float),
}

ENGINES = build_engines(TUNING)
DEFAULT_ENGINE = default_engine_name(ENGINES)

# 快通道便宜，可以多开；VL 通道每条都占后端并发，少开
WORKERS = {
    "fast": int(os.environ.get("OCR_WORKERS_FAST", "8")),
    "quality": int(os.environ.get("OCR_WORKERS_QUALITY",
                                  os.environ.get("OCR_WORKERS", "4"))),
}

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
    """魔数比文件名可靠：粘贴上来的截图往往没有正经文件名。"""
    if blob[:4] == b"RIFF" and blob[8:12] == b"WEBP":
        return ".webp"
    for magic, suffix in MAGIC:
        if blob.startswith(magic):
            return suffix
    suffix = Path(filename or "").suffix.lower()
    return suffix if suffix in (IMAGE_SUFFIXES | {".pdf"}) else ""


def parse_upload(content_type: str, body: bytes) -> tuple[str, bytes]:
    """取出上传的文件。两种调用方式都支持：
        curl -F 'file=@a.png' ...
        curl --data-binary @a.png -H 'X-Filename: a.png' ...
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
    """每个引擎一个池。实例不保证线程安全，用队列做独占借还。

    这也是后面换成多进程 / 多机 worker 时最自然的切分点。
    """

    def __init__(self, engine, size: int):
        self.engine = engine
        self.size = size
        self._q: "queue.Queue" = queue.Queue()
        self.ready = 0
        self.error: str | None = None
        self.rebuilding = False
        self._lock = threading.Lock()

    def warmup(self):
        print("[pool:%s] 开始构建 %d 条  %s" % (self.engine.name, self.size,
                                              self.engine.describe()), flush=True)
        for i in range(self.size):
            t0 = time.perf_counter()
            try:
                self._q.put(self.engine.build())
                with self._lock:
                    self.ready += 1
                print("[pool:%s] %d/%d 就绪（%.1fs）"
                      % (self.engine.name, i + 1, self.size,
                         time.perf_counter() - t0), flush=True)
            except Exception:
                self.error = traceback.format_exc(limit=6)
                print("[pool:%s] 第 %d 条构建失败:" % (self.engine.name, i + 1), flush=True)
                print(self.error, flush=True)
                return
        print("[pool:%s] 全部 %d 条就绪" % (self.engine.name, self.ready), flush=True)

    def rebuild_async(self, note: str = ""):
        """构造期参数改动后重建整池。新实例全建好再原子替换，期间不中断服务。"""
        if self.rebuilding:
            return False
        self.rebuilding = True

        def run():
            try:
                print("[pool:%s] 重建（%s）…" % (self.engine.name, note), flush=True)
                fresh = [self.engine.build() for _ in range(self.size)]
                nq: "queue.Queue" = queue.Queue()
                for x in fresh:
                    nq.put(x)
                with self._lock:
                    self._q = nq          # 原子替换，老实例交给 GC
                    self.ready = len(fresh)
                    self.error = None
                print("[pool:%s] 重建完成" % self.engine.name, flush=True)
            except Exception:
                self.error = traceback.format_exc(limit=6)
                print("[pool:%s] 重建失败:" % self.engine.name, flush=True)
                print(self.error, flush=True)
            finally:
                self.rebuilding = False

        threading.Thread(target=run, daemon=True,
                         name="rebuild-" + self.engine.name).start()
        return True

    @contextmanager
    def acquire(self, timeout: float):
        try:
            item = self._q.get(timeout=timeout)
        except queue.Empty:
            raise HTTPException(503, "%s 引擎繁忙，%d 条流水线全占用中，请重试"
                                     % (self.engine.name, self.size))
        try:
            yield item
        finally:
            self._q.put(item)

    @property
    def idle(self) -> int:
        return self._q.qsize()

    def snapshot(self) -> dict:
        return {"ready": self.ready, "idle": self.idle, "size": self.size,
                "rebuilding": self.rebuilding, "error": self.error}


POOLS = {name: PipelinePool(eng, WORKERS.get(name, 4)) for name, eng in ENGINES.items()}


# --------------------------------------------------------------- 指标
class Metrics:
    def __init__(self):
        self._d: dict = {}
        self._lock = threading.Lock()

    def record(self, engine: str, ms: float, ok: bool):
        with self._lock:
            s = self._d.setdefault(engine, {"total": 0, "failed": 0, "lat": []})
            s["total"] += 1
            if not ok:
                s["failed"] += 1
            s["lat"].append(ms)
            if len(s["lat"]) > 2000:
                del s["lat"][:1000]

    def snapshot(self) -> dict:
        with self._lock:
            data = {k: (v["total"], v["failed"], sorted(v["lat"]))
                    for k, v in self._d.items()}
        out = {}
        for k, (total, failed, lat) in data.items():
            def pct(p):
                return round(lat[min(len(lat) - 1, int(len(lat) * p))], 1) if lat else None
            out[k] = {"total": total, "failed": failed,
                      "p50_ms": pct(0.50), "p95_ms": pct(0.95), "p99_ms": pct(0.99),
                      "mean_ms": round(statistics.fmean(lat), 1) if lat else None}
        return out

    def totals(self) -> dict:
        snap = self.snapshot()
        return {"total": sum(v["total"] for v in snap.values()),
                "failed": sum(v["failed"] for v in snap.values())}


METRICS = Metrics()


# --------------------------------------------------------------- 应用
@asynccontextmanager
async def lifespan(_: FastAPI):
    for pool in POOLS.values():
        threading.Thread(target=pool.warmup, daemon=True,
                         name="warmup-" + pool.engine.name).start()
    yield


app = FastAPI(title="knock-ocr", version="0.3.0",
              docs_url="/api/docs", lifespan=lifespan)


def _vl_backend_status() -> dict:
    eng = ENGINES.get("quality")
    if eng is None:
        return {"required": False, "reachable": True,
                "note": "当前档位（%s）不含 VL 引擎" % TIER}
    probe = eng.server_url.rstrip("/") + "/models"
    out = {"required": True, "url": eng.server_url, "probe": probe,
           "reachable": False, "models": []}
    try:
        import urllib.request

        with urllib.request.urlopen(probe, timeout=10) as r:
            data = json.loads(r.read().decode())
            out["reachable"] = True
            out["models"] = [m.get("id") for m in data.get("data", [])]
    except Exception as e:
        out["error"] = str(e)
    return out


@app.get("/healthz")
def healthz():
    pools = {k: p.snapshot() for k, p in POOLS.items()}
    return {
        "ready": any(p["ready"] > 0 for p in pools.values()),
        "tier": TIER,
        "engines": sorted(ENGINES),
        "default_engine": DEFAULT_ENGINE,
        "pools": pools,
    }


@app.get("/api/info")
def info():
    pools = {k: p.snapshot() for k, p in POOLS.items()}
    q = POOLS.get("quality")
    return {
        "tier": TIER,
        "engines": {k: e.describe() for k, e in ENGINES.items()},
        "default_engine": DEFAULT_ENGINE,
        "pools": pools,
        "tuning": dict(TUNING),
        "vlm_backend": _vl_backend_status(),
        "rebuilding": bool(q and q.rebuilding),
        "metrics": METRICS.snapshot(),
        "totals": METRICS.totals(),
    }


@app.get("/api/metrics")
def metrics():
    return {"tier": TIER, "by_engine": METRICS.snapshot(),
            "pools": {k: p.snapshot() for k, p in POOLS.items()}}


def _pick_pool(engine: str | None) -> PipelinePool:
    name = (engine or DEFAULT_ENGINE).strip().lower()
    if name in ("auto", ""):
        name = DEFAULT_ENGINE
    pool = POOLS.get(name)
    if pool is None:
        raise HTTPException(
            400, "本次部署（TIER=%s）没有 %r 引擎，可用：%s。"
                 "想同时具备请用 TIER=full 部署。" % (TIER, name, sorted(POOLS)))
    if pool.ready == 0:
        detail = "%s 引擎尚未就绪（0/%d）。" % (name, pool.size)
        detail += ("构建时报错了，看 `manage.sh logs api`。" if pool.error
                   else "仍在初始化，请稍候。")
        raise HTTPException(503, detail)
    return pool


@app.post("/api/ocr")
async def ocr(
    request: Request,
    engine: str | None = Query(None, description="fast=快 | quality=准；不传用默认"),
    merge_tables: bool = Query(True, description="多页 PDF 是否合并跨页表格"),
    include_json: bool = Query(True, description="是否返回结构化结果"),
    layout_threshold: float | None = Query(None, ge=0.05, le=0.95),
    max_pixels: int | None = Query(None, ge=100000, le=20000000),
    max_new_tokens: int | None = Query(None, ge=64, le=8192),
):
    rid = uuid.uuid4().hex[:12]
    t_start = time.perf_counter()
    pool = _pick_pool(engine)
    eng = pool.engine

    raw = await request.body()
    t_body = time.perf_counter()
    if not raw:
        raise HTTPException(400, "空请求体")
    if len(raw) > MAX_MB * 1024 * 1024:
        raise HTTPException(413, "请求超过 %dMB" % MAX_MB)

    filename, blob = parse_upload(request.headers.get("content-type", ""), raw)
    filename = filename or request.headers.get("x-filename", "") or "upload"
    if not blob:
        raise HTTPException(400, "空文件")

    suffix = sniff_suffix(blob, filename)
    allowed = IMAGE_SUFFIXES | ({".pdf"} if eng.supports_pdf else set())
    if suffix not in allowed:
        raise HTTPException(415, "%s 引擎不支持该文件类型（%r）。支持：%s"
                                 % (eng.name, filename, sorted(allowed)))

    # 逐请求覆盖；没传的回落到全局 TUNING
    overrides = {k: v for k, v in (
        ("layout_threshold", layout_threshold if layout_threshold is not None
         else TUNING.get("layout_threshold")),
        ("max_pixels", max_pixels if max_pixels is not None else TUNING.get("max_pixels")),
        ("max_new_tokens", max_new_tokens if max_new_tokens is not None
         else TUNING.get("max_new_tokens")),
    ) if k in eng.per_request_keys}

    timings = {"receive_ms": round((t_body - t_start) * 1000, 1)}
    ok = False
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as fh:
            fh.write(blob)
            tmp = fh.name

        # 分段计时，用来回答「慢在哪」：上传？排队？还是推理本身？
        def work():
            q0 = time.perf_counter()
            with pool.acquire(ACQUIRE_TIMEOUT) as handle:
                q1 = time.perf_counter()
                out = eng.run(handle, tmp, merge_tables, overrides)
                timings["queue_ms"] = round((q1 - q0) * 1000, 1)
                timings["infer_ms"] = round((time.perf_counter() - q1) * 1000, 1)
                return out

        md, pages_json, npages = await run_in_threadpool(work)
        ok = True
        timings["total_ms"] = round((time.perf_counter() - t_start) * 1000, 1)
        body = {
            "request_id": rid,
            "engine": eng.name,
            "filename": filename,
            "bytes": len(blob),
            "pages": npages,
            "elapsed_ms": timings["total_ms"],
            "timings": timings,
            "applied": {k: v for k, v in overrides.items() if v is not None},
            "markdown": md,
        }
        if include_json:
            body["result"] = pages_json
        return JSONResponse(body)

    except HTTPException:
        raise
    except Exception as e:
        print("[%s] 识别失败:" % rid, flush=True)
        print(traceback.format_exc(limit=6), flush=True)
        raise HTTPException(500, "识别失败 (request_id=%s): %s: %s"
                                 % (rid, type(e).__name__, e))
    finally:
        METRICS.record(eng.name, (time.perf_counter() - t_start) * 1000, ok)
        if tmp:
            try:
                os.unlink(tmp)
            except OSError:
                pass


@app.post("/api/config")
async def set_config(request: Request):
    """在线调参。

    per-request 三项只改默认值，立即生效；
    vl_rec_max_concurrency 是构造期参数，改了后台重建 quality 池，期间不中断服务。
    """
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(400, "请求体必须是 JSON")
    if not isinstance(body, dict):
        raise HTTPException(400, "请求体必须是 JSON 对象")

    limits = {
        "vl_rec_max_concurrency": (1, 64, int),
        "max_pixels": (100000, 20000000, int),
        "max_new_tokens": (64, 8192, int),
        "layout_threshold": (0.05, 0.95, float),
    }
    changed, needs_rebuild = {}, False
    for k, v in body.items():
        if k not in limits:
            raise HTTPException(400, "未知参数 %s，可选：%s" % (k, sorted(limits)))
        lo, hi, cast = limits[k]
        if v in (None, "", 0):
            val = None                      # 显式清空 = 恢复不限制
        else:
            try:
                val = cast(v)
            except (TypeError, ValueError):
                raise HTTPException(400, "%s 不是合法数值：%r" % (k, v))
            if not (lo <= val <= hi):
                raise HTTPException(400, "%s 超出范围 [%s, %s]" % (k, lo, hi))
        if TUNING.get(k) != val:
            TUNING[k] = val
            changed[k] = val
            if k == "vl_rec_max_concurrency":
                needs_rebuild = True

    q = POOLS.get("quality")
    started = False
    if needs_rebuild and q:
        started = q.rebuild_async("并发数改为 %s" % TUNING["vl_rec_max_concurrency"])
    if started:
        note = "并发数已改，正在后台重建流水线；重建期间沿用旧实例，不中断服务。"
    elif needs_rebuild and not q:
        note = "当前档位没有 quality 引擎，该参数已记录但暂不生效。"
    elif needs_rebuild:
        note = "上一次重建还没结束，稍后再试。"
    else:
        note = "已生效，下一次识别即采用新参数。"
    return {"changed": changed, "tuning": dict(TUNING),
            "rebuilding": bool(q and q.rebuilding), "note": note}


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")
