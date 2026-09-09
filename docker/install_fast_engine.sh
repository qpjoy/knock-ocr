#!/bin/sh
# 装 fast 引擎（RapidOCR + PP-OCRv6 ONNX），并做一次真实构造验证。
# 单独成脚本而不是塞进 Dockerfile 的 RUN —— 多行 python 内联在 RUN 里
# 极易被各层转义吃坏，独立文件最稳。
set -e
INDEX="${1:-https://pypi.tuna.tsinghua.edu.cn/simple}"
PIP="python -m pip install --no-cache-dir -i $INDEX"

# rapidocr 3.x 把推理运行时拆成了可选依赖，必须单独装 onnxruntime，
# 否则构造时报 "onnxruntime is not installed."。
# 新版默认模型就是 PP-OCRv6 的 det/rec small。
if $PIP "rapidocr>=3.0" onnxruntime; then
    echo "已安装 rapidocr 3.x + onnxruntime"
else
    echo "rapidocr 3.x 安装失败，退回经典包（模型为 v4/v5 系列，精度低一档）"
    $PIP "rapidocr-onnxruntime>=1.4"
fi

python - <<'PY'
import importlib.util as u
mod = "rapidocr" if u.find_spec("rapidocr") else "rapidocr_onnxruntime"
RapidOCR = __import__(mod, fromlist=["RapidOCR"]).RapidOCR
RapidOCR()          # 真的构造一次，确认模型随包分发、运行时不用联网
print("fast 引擎就绪:", mod)
PY
