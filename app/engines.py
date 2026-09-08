"""OCR 引擎抽象。

两种实现，对外产出同一个形状 (markdown, pages_json, npages)：

  vl        PaddleOCR-VL-1.6 + vLLM 远端服务。文档解析最强（OmniDocBench 96.3%），
            需要 GPU 和 20~30GB 官方镜像。服务器用这个。
  rapidocr  RapidOCR（ONNXRuntime）。纯 CPU、模型内置在 wheel 里、镜像不到 1GB，
            本地开发和调试用这个；也是生产「图片快通道」的雏形。

换引擎只改 OCR_ENGINE 环境变量，server.py / 前端 / manage.sh 都不用动。
"""
from __future__ import annotations

import os
from pathlib import Path


class OcrEngine:
    name = "base"
    supports_pdf = False

    def build(self):
        """构造一个可独占使用的实例（放进流水线池）。"""
        raise NotImplementedError

    def run(self, handle, path: str, merge_tables: bool):
        """返回 (markdown, pages_json, npages)。"""
        raise NotImplementedError

    def describe(self) -> dict:
        return {"engine": self.name, "supports_pdf": self.supports_pdf}


# --------------------------------------------------------------- PaddleOCR-VL
class VLEngine(OcrEngine):
    name = "vl"
    supports_pdf = True

    def __init__(self, *, backend: str, server_url: str, model: str, device: str):
        self.backend = backend
        self.server_url = server_url
        self.model = model
        self.device = device

    def build(self):
        from paddleocr import PaddleOCRVL

        kwargs = {
            "vl_rec_backend": self.backend,
            "vl_rec_server_url": self.server_url,
            "vl_rec_api_model_name": self.model,
        }
        if self.device:
            kwargs["device"] = self.device
        return PaddleOCRVL(**kwargs)

    def run(self, handle, path: str, merge_tables: bool):
        import traceback

        pages = list(handle.predict(path))

        if len(pages) > 1 and hasattr(handle, "restructure_pages"):
            try:
                merged = handle.restructure_pages(pages, merge_tables=merge_tables)
                md = _vl_markdown(merged)
                if md:
                    return md, [_vl_json(p) for p in pages], len(pages)
            except Exception:
                print("[warn] restructure_pages 失败，回落逐页拼接:\n"
                      + traceback.format_exc(limit=3), flush=True)

        md = "\n\n---\n\n".join(filter(None, (_vl_markdown(p) for p in pages)))
        return md, [_vl_json(p) for p in pages], len(pages)

    def describe(self) -> dict:
        return {**super().describe(), "backend": self.backend,
                "server_url": self.server_url, "model": self.model,
                "layout_device": self.device}


def _vl_markdown(res) -> str:
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


def _vl_json(res) -> dict:
    j = getattr(res, "json", None)
    if callable(j):
        try:
            j = j()
        except Exception:
            j = None
    if isinstance(j, dict):
        return j.get("res", j)
    return {}


# --------------------------------------------------------------- RapidOCR
class RapidOcrEngine(OcrEngine):
    name = "rapidocr"
    supports_pdf = False

    def build(self):
        # 两个包名都兼容：rapidocr（新）/ rapidocr_onnxruntime（经典），模型都内置在 wheel 里
        try:
            from rapidocr_onnxruntime import RapidOCR
        except ImportError:
            from rapidocr import RapidOCR
        return RapidOCR()

    def run(self, handle, path: str, merge_tables: bool):
        if Path(path).suffix.lower() == ".pdf":
            raise RuntimeError("本地 rapidocr 引擎只支持图片；PDF 请用 vl 引擎（OCR_ENGINE=vl）")

        out = handle(path)
        items = _rapid_items(out)

        # 按行归并：先按 y 排序，y 相近的算同一行，行内再按 x 排
        rows: list[list] = []
        for it in sorted(items, key=lambda i: i["cy"]):
            if rows and abs(it["cy"] - rows[-1][0]["cy"]) <= it["h"] * 0.6:
                rows[-1].append(it)
            else:
                rows.append([it])

        lines = [" ".join(x["text"] for x in sorted(r, key=lambda i: i["cx"])) for r in rows]
        md = "\n\n".join(l for l in lines if l.strip())

        page = {
            "engine": self.name,
            "lines": [
                {"text": it["text"], "score": it["score"], "box": it["box"]}
                for it in items
            ],
        }
        return md, [page], 1

    def describe(self) -> dict:
        return {**super().describe(), "layout_device": "cpu"}


def _rapid_items(out) -> list[dict]:
    """把 RapidOCR 各版本的返回值统一成 [{text, score, box, cx, cy, h}]。"""
    raw = []
    # 新版：对象，带 .txts/.boxes/.scores
    if hasattr(out, "txts") and out.txts is not None:
        boxes = list(getattr(out, "boxes", []) or [])
        scores = list(getattr(out, "scores", []) or [])
        for i, t in enumerate(out.txts):
            raw.append((boxes[i] if i < len(boxes) else None, t,
                        scores[i] if i < len(scores) else None))
    else:
        # 经典版：(result, elapse)，result 是 [[box, text, score], ...]
        res = out[0] if isinstance(out, tuple) else out
        for row in (res or []):
            box = row[0] if len(row) > 0 else None
            text = row[1] if len(row) > 1 else ""
            score = row[2] if len(row) > 2 else None
            raw.append((box, text, score))

    items = []
    for box, text, score in raw:
        if not text:
            continue
        xs, ys = [], []
        try:
            for p in (box or []):
                xs.append(float(p[0]))
                ys.append(float(p[1]))
        except (TypeError, IndexError, ValueError):
            xs, ys = [], []
        cx = sum(xs) / len(xs) if xs else 0.0
        cy = sum(ys) / len(ys) if ys else 0.0
        h = (max(ys) - min(ys)) if ys else 1.0
        items.append({
            "text": str(text),
            "score": round(float(score), 4) if score is not None else None,
            "box": [[float(p[0]), float(p[1])] for p in (box or [])] or None,
            "cx": cx, "cy": cy, "h": max(h, 1.0),
        })
    return items


# --------------------------------------------------------------- 工厂
def get_engine() -> OcrEngine:
    name = os.environ.get("OCR_ENGINE", "vl").strip().lower()
    if name in ("rapidocr", "local", "cpu"):
        return RapidOcrEngine()
    if name == "vl":
        return VLEngine(
            backend=os.environ.get("OCR_VL_BACKEND", "vllm-server"),
            server_url=_normalize_v1(os.environ.get("OCR_VLLM_URL", "http://127.0.0.1:8118")),
            model=os.environ.get("OCR_MODEL", "PaddleOCR-VL-1.6-0.9B"),
            device=os.environ.get("OCR_DEVICE", "cpu"),
        )
    raise ValueError(f"未知引擎 OCR_ENGINE={name!r}，可选：vl | rapidocr")


def _normalize_v1(url: str) -> str:
    url = (url or "").rstrip("/")
    return url if url.endswith("/v1") else url + "/v1"
