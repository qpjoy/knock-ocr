ARG BASE_IMAGE=ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddleocr-vl:latest-nvidia-gpu
FROM ${BASE_IMAGE}

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# 官方镜像里 fastapi / uvicorn / starlette 都已具备，服务端代码也刻意不用
# python-multipart（自己用标准库解析上传），所以这一层完全不需要联网。
# 内网、代理不通、离线机器都能构建。
RUN python -c "import fastapi, uvicorn, starlette; \
print('fastapi', fastapi.__version__, '| uvicorn', uvicorn.__version__, '| starlette', starlette.__version__)"

WORKDIR /app
COPY app/ /app/

EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=5s --start-period=30s --retries=60 \
    CMD python -c "import urllib.request,sys;sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/healthz',timeout=3).status==200 else 1)"

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
