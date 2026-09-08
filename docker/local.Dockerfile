# 本地开发用的轻量镜像：纯 CPU、无 GPU 依赖、不到 1GB。
# 跑的是同一份 app/ 代码，只是把引擎换成 RapidOCR（OCR_ENGINE=rapidocr）。
# 目的是在本地就能改 API / 前端 / 脚本并立刻验证，不用拉 20~30GB 的官方镜像。
FROM python:3.11-slim

ARG PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# opencv 需要这两个系统库；字体包让样张能画出中文
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 fonts-wqy-zenhei \
    && rm -rf /var/lib/apt/lists/*

# rapidocr_onnxruntime 的模型内置在 wheel 里，运行时不需要联网下载
RUN python -m pip install --no-cache-dir -i "$PIP_INDEX_URL" \
        "fastapi>=0.110" "uvicorn[standard]>=0.29" \
        "rapidocr-onnxruntime>=1.3" "pillow>=10.0" \
 && python -c "from rapidocr_onnxruntime import RapidOCR; RapidOCR(); print('rapidocr 就绪')"

WORKDIR /app
COPY app/ /app/

ENV OCR_ENGINE=rapidocr
EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=5s --start-period=20s --retries=30 \
    CMD python -c "import urllib.request,sys;sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/healthz',timeout=3).status==200 else 1)"

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
