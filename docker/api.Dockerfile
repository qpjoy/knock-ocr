ARG BASE_IMAGE=ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddleocr-vl:latest-nvidia-gpu
FROM ${BASE_IMAGE}

ARG PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
ENV PIP_INDEX_URL=${PIP_INDEX_URL} \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# 官方镜像已带 paddleocr[doc-parser]，这里只补 Web 层
RUN python -m pip install --no-cache-dir \
        "fastapi>=0.110" "uvicorn[standard]>=0.29" "python-multipart>=0.0.9"

WORKDIR /app
COPY app/ /app/

EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=5s --start-period=30s --retries=60 \
    CMD python -c "import urllib.request,sys;sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/healthz',timeout=3).status==200 else 1)"

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
