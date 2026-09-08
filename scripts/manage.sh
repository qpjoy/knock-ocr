#!/usr/bin/env bash
# knock-ocr demo 一键管理脚本
#   bash scripts/manage.sh deploy      # 幂等：反复跑都行，会自己清理上一次的残留
# 所有配置都有默认值，需要改时在命令前面传：
#   PROXY=http://127.0.0.1:7788 PORT=9000 bash scripts/manage.sh deploy
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# ---------------------------------------------------------------- 配置
PROJECT="${PROJECT:-knock-ocr}"          # 容器/网络/卷名前缀，改它可并存多套
GPU_ID="${GPU_ID:-2}"                    # 只占用这一张卡（默认避开挂显示器的 GPU3）
PORT="${PORT:-8710}"                     # 对外 Web + API 端口；被占用会自动顺延
STRICT_PORT="${STRICT_PORT:-0}"          # =1 则端口被占直接报错，不自动顺延
BIND="${BIND:-0.0.0.0}"                  # 只想本机访问就传 BIND=127.0.0.1
VLLM_PORT="${VLLM_PORT:-8118}"           # 仅容器网络内部使用，不对外发布
MODEL="${MODEL:-PaddleOCR-VL-1.6-0.9B}"  # OmniDocBench v1.6 96.3% 的那个模型
WORKERS="${WORKERS:-4}"                  # API 侧并行流水线数
DEVICE="${DEVICE:-auto}"                 # auto | cpu | gpu:0  —— Paddle 侧（版面分析）跑在哪
WAIT_TIMEOUT="${WAIT_TIMEOUT:-2400}"     # 等服务就绪的秒数（首次要下模型，给足）
VLLM_ARGS="${VLLM_ARGS:-}"               # 透传给 genai_server 的额外参数
PROXY="${PROXY:-}"                       # 出网代理，如 http://127.0.0.1:7788（下模型权重要用）

REGISTRY="${REGISTRY:-ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle}"
VLLM_IMAGE="${VLLM_IMAGE:-$REGISTRY/paddleocr-genai-vllm-server:latest-nvidia-gpu}"
BASE_IMAGE="${BASE_IMAGE:-$REGISTRY/paddleocr-vl:latest-nvidia-gpu}"
API_IMAGE="${API_IMAGE:-$PROJECT/api:local}"

NET="${PROJECT}-net"
C_VLLM="${PROJECT}-vllm"
C_API="${PROJECT}-api"
V_MODELS="${PROJECT}-models"
V_HF="${PROJECT}-hf"
STATE="$ROOT/.deploy"

# ---------------------------------------------------------------- 输出
c_red=$'\033[31m'; c_grn=$'\033[32m'; c_ylw=$'\033[33m'
c_blu=$'\033[36m'; c_dim=$'\033[2m'; c_off=$'\033[0m'
say()  { printf '%s==>%s %s\n' "$c_blu" "$c_off" "$*"; }
ok()   { printf '%s ok %s %s\n' "$c_grn" "$c_off" "$*"; }
warn() { printf '%s  ! %s %s\n' "$c_ylw" "$c_off" "$*"; }
die()  { printf '%s ERR%s %s\n' "$c_red" "$c_off" "$*" >&2; exit 1; }
dim()  { printf '%s%s%s\n' "$c_dim" "$*" "$c_off"; }
rule() { printf '%s%s%s\n' "$c_dim" "────────────────────────────────────────────────────────" "$c_off"; }

# ---------------------------------------------------------------- 工具
have() { command -v "$1" >/dev/null 2>&1; }
container_state() { docker inspect -f '{{.State.Status}}' "$1" 2>/dev/null || echo "absent"; }
container_exit_code() { docker inspect -f '{{.State.ExitCode}}' "$1" 2>/dev/null || echo "?"; }
http_ok() { curl -fsS -m 3 -o /dev/null "$1" 2>/dev/null; }

port_busy() {
  if have ss;        then ss -ltn 2>/dev/null | grep -qE "[:.]$1[[:space:]]" && return 0
  elif have netstat; then netstat -ltn 2>/dev/null | grep -qE "[:.]$1[[:space:]]" && return 0
  fi
  return 1
}

# 从 PORT 开始往后找一个没被占的
free_port() {
  local p="$1"
  for _ in $(seq 0 50); do
    port_busy "$p" || { echo "$p"; return 0; }
    p=$((p + 1))
  done
  return 1
}

# 容器里的 127.0.0.1 是容器自己，不是宿主机。桥接网络下要改走 host-gateway。
proxy_for_container() { sed -E 's#//(127\.0\.0\.1|localhost)([:/]|$)#//host.docker.internal\2#' <<<"$1"; }

proxy_run_args() {
  [ -z "$PROXY" ] && return 0
  local p; p="$(proxy_for_container "$PROXY")"
  printf '%s\n' \
    --add-host "host.docker.internal:host-gateway" \
    -e "http_proxy=$p"  -e "HTTP_PROXY=$p" \
    -e "https_proxy=$p" -e "HTTPS_PROXY=$p" \
    -e "no_proxy=localhost,127.0.0.1,$C_VLLM,$C_API" \
    -e "NO_PROXY=localhost,127.0.0.1,$C_VLLM,$C_API"
}

# ---------------------------------------------------------------- preflight
cmd_preflight() {
  say "环境检查"

  have docker || die "未找到 docker。这个 demo 全部跑在容器里，不会碰宿主机的 Python/CUDA。"
  docker info >/dev/null 2>&1 || die "docker 守护进程不可用（当前用户可能不在 docker 组）。"
  ok "docker 可用"

  have nvidia-smi || die "未找到 nvidia-smi"
  local ngpu; ngpu="$(nvidia-smi -L | wc -l | tr -d ' ')"
  nvidia-smi -i "$GPU_ID" >/dev/null 2>&1 || die "GPU $GPU_ID 不存在（本机共 $ngpu 张卡，编号 0..$((ngpu-1))）"
  ok "GPU $GPU_ID 存在：$(nvidia-smi -i "$GPU_ID" --query-gpu=name,memory.total --format=csv,noheader)"

  local busy; busy="$(nvidia-smi -i "$GPU_ID" --query-compute-apps=pid,used_memory --format=csv,noheader || true)"
  if [ -n "$busy" ]; then
    warn "GPU $GPU_ID 上已有计算进程，demo 会与其共享显存："
    echo "$busy" | sed 's/^/      /'
  else
    ok "GPU $GPU_ID 空闲"
  fi

  if docker info 2>/dev/null | grep -qi 'nvidia'; then
    ok "docker 已注册 nvidia runtime"
  else
    warn "docker info 里没看到 nvidia runtime，若 deploy 失败请检查 nvidia-container-toolkit"
  fi

  # 端口：被自己的旧容器占着不算冲突（deploy 会先清掉）
  if port_busy "$PORT"; then
    if [ "$(container_state "$C_API")" = "running" ]; then
      ok "端口 $PORT 目前被本 demo 自己占着，稍后会重启"
    elif [ "$STRICT_PORT" = "1" ]; then
      die "端口 $PORT 已被占用（STRICT_PORT=1）。换一个：PORT=<n> bash scripts/manage.sh deploy"
    else
      local np; np="$(free_port $((PORT + 1)))" || die "$PORT 起往后 50 个端口都被占用了"
      warn "端口 $PORT 被别的服务占用，自动改用 $np（固定端口请传 PORT=<n>）"
      PORT="$np"
    fi
  fi
  ok "对外端口 $PORT"

  if have firewall-cmd && firewall-cmd --state >/dev/null 2>&1; then
    if firewall-cmd --query-port="$PORT/tcp" >/dev/null 2>&1; then
      ok "firewalld 已放行 $PORT/tcp"
    elif [ "$BIND" = "127.0.0.1" ]; then
      ok "firewalld 运行中；BIND=127.0.0.1 仅本机访问，无需放行"
    else
      warn "firewalld 未放行 $PORT/tcp，外部浏览器可能连不上。放行："
      dim "      firewall-cmd --add-port=$PORT/tcp --permanent && firewall-cmd --reload"
    fi
  fi

  if have getenforce && [ "$(getenforce 2>/dev/null)" = "Enforcing" ]; then
    dim "      SELinux=Enforcing（bind mount 已加 :z，正常情况无需额外处理）"
  fi

  if [ -n "$PROXY" ]; then
    if curl -fsS -m 5 -x "$PROXY" -o /dev/null https://www.baidu.com 2>/dev/null; then
      ok "代理可用：$PROXY（容器内改写为 $(proxy_for_container "$PROXY")）"
    else
      warn "代理 $PROXY 从宿主机测不通，模型下载可能失败"
    fi
  fi

  local free; free="$(df -Pk /var/lib/docker 2>/dev/null | awk 'NR==2{print int($4/1048576)}' || echo 0)"
  if [ "${free:-0}" -lt 40 ]; then
    warn "/var/lib/docker 可用空间约 ${free}GB，镜像约需 20~30GB"
  else
    ok "磁盘空间充足（约 ${free}GB）"
  fi
  echo
}

# ---------------------------------------------------------------- GPU 探测
# 分两件事查，别混为一谈：
#   1) vLLM 侧（PyTorch）能否用这张卡 —— 这决定 demo 能不能跑，是硬要求
#   2) Paddle 侧能否用这张卡 —— 只影响版面分析放 CPU 还是 GPU，降级不影响精度
cmd_gpucheck() {
  mkdir -p "$STATE"
  rule
  say "1/2  vLLM 侧（PyTorch）—— 硬要求"
  local torch_out torch_rc=0
  torch_out="$(docker run --rm --gpus "device=$GPU_ID" "$VLLM_IMAGE" python -c '
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("available", torch.cuda.is_available())
if torch.cuda.is_available():
    cc = torch.cuda.get_device_capability(0)
    print("device", torch.cuda.get_device_name(0), "sm_%d%d" % cc)
    x = torch.ones(1024, 1024, device="cuda", dtype=torch.float16)
    print("matmul ok", float((x @ x)[0, 0]))
' 2>&1)" || torch_rc=$?
  printf '%s\n' "$torch_out" > "$STATE/torch_probe.log"
  if [ $torch_rc -eq 0 ] && grep -q 'matmul ok' <<<"$torch_out"; then
    ok "PyTorch 可以在这张卡上跑核"
    sed 's/^/      /' <<<"$torch_out" | head -4
  else
    warn "PyTorch 在这张卡上跑不起来 —— vLLM 一定会挂"
    sed 's/^/      /' <<<"$torch_out" | tail -12
    dim "      完整日志：$STATE/torch_probe.log"
  fi

  echo
  say "2/2  Paddle 侧（版面分析）—— 失败可降级"
  local pd_out pd_rc=0
  # 不用 paddle.utils.run_check()：它顺带跑分布式自检，容器里常因无关原因失败，
  # 会把 GPU 可用误判成不可用。这里只做最小的真实核启动测试。
  pd_out="$(docker run --rm --gpus "device=$GPU_ID" "$BASE_IMAGE" python -c '
import paddle
print("paddle", paddle.__version__)
n = paddle.device.cuda.device_count()
print("device_count", n)
assert n > 0, "看不到 GPU"
print("capability", paddle.device.cuda.get_device_capability(0))
x = paddle.ones([512, 512], dtype="float32").cuda()
print("matmul ok", float(paddle.matmul(x, x)[0, 0]))
' 2>&1)" || pd_rc=$?
  printf '%s\n' "$pd_out" > "$STATE/paddle_probe.log"
  if [ $pd_rc -eq 0 ] && grep -q 'matmul ok' <<<"$pd_out"; then
    ok "Paddle 可以在这张卡上跑核"
    sed 's/^/      /' <<<"$pd_out" | head -4
    echo "gpu:0" > "$STATE/paddle_gpu"
  else
    if grep -qiE 'no kernel image|sm_120|not compiled with|arch' <<<"$pd_out"; then
      warn "Paddle 不含 sm_120 kernel（Blackwell 已知问题）→ 版面分析走 CPU"
    else
      warn "Paddle GPU 自检未通过 → 版面分析走 CPU"
    fi
    sed 's/^/      /' <<<"$pd_out" | tail -12
    dim "      完整日志：$STATE/paddle_probe.log"
    dim "      注意：这不影响识别精度。决定 96.3% 的 0.9B VLM 跑在 vLLM(GPU) 上，"
    dim "            CPU 只承担轻量版面检测，本机 128 核绰绰有余。"
    echo "cpu" > "$STATE/paddle_gpu"
  fi
  rule
}

resolve_device() {
  mkdir -p "$STATE"
  if [ "$DEVICE" != "auto" ]; then echo "$DEVICE" > "$STATE/paddle_gpu"; fi
  [ -s "$STATE/paddle_gpu" ] || cmd_gpucheck >&2
  cat "$STATE/paddle_gpu"
}

# ---------------------------------------------------------------- 构建 / 启动
ensure_net() {
  docker network inspect "$NET" >/dev/null 2>&1 || docker network create "$NET" >/dev/null
  docker volume inspect "$V_MODELS" >/dev/null 2>&1 || docker volume create "$V_MODELS" >/dev/null
  docker volume inspect "$V_HF"     >/dev/null 2>&1 || docker volume create "$V_HF" >/dev/null
}

cmd_pull() {
  say "确认官方镜像（已有则跳过，不会重复下载）"
  docker image inspect "$VLLM_IMAGE" >/dev/null 2>&1 || docker pull "$VLLM_IMAGE"
  docker image inspect "$BASE_IMAGE" >/dev/null 2>&1 || docker pull "$BASE_IMAGE"
  ok "镜像就绪"
}

cmd_build() {
  say "构建 API 镜像（这一层不需要联网）"
  local net=()
  [ -n "$PROXY" ] && net=(--network host --build-arg "http_proxy=$PROXY" --build-arg "https_proxy=$PROXY")
  docker build "${net[@]}" --build-arg "BASE_IMAGE=$BASE_IMAGE" \
    -f docker/api.Dockerfile -t "$API_IMAGE" .
  ok "API 镜像就绪：$API_IMAGE"
}

start_vllm() {
  docker rm -f "$C_VLLM" >/dev/null 2>&1 || true
  say "启动 VLM 推理服务（GPU $GPU_ID, model=$MODEL）"
  local px=(); mapfile -t px < <(proxy_run_args)
  # shellcheck disable=SC2086
  docker run -d --name "$C_VLLM" --network "$NET" \
    --gpus "device=$GPU_ID" \
    --restart unless-stopped \
    --shm-size 8g \
    "${px[@]}" \
    -e VLLM_FLASH_ATTN_VERSION=2 \
    -v "$V_MODELS":/root/.paddlex \
    -v "$V_HF":/root/.cache/huggingface \
    "$VLLM_IMAGE" \
    paddleocr genai_server --model_name "$MODEL" \
      --host 0.0.0.0 --port "$VLLM_PORT" --backend vllm $VLLM_ARGS >/dev/null
  ok "容器 $C_VLLM 已启动"
}

start_api() {
  local dev="$1"
  docker rm -f "$C_API" >/dev/null 2>&1 || true
  say "启动 API + Web 服务（版面分析设备：$dev）"
  docker run -d --name "$C_API" --network "$NET" \
    --restart unless-stopped \
    -p "$BIND:$PORT:8000" \
    -e "OCR_DEVICE=$dev" \
    -e "OCR_MODEL=$MODEL" \
    -e "OCR_VLLM_URL=http://$C_VLLM:$VLLM_PORT" \
    -e "OCR_WORKERS=$WORKERS" \
    -v "$V_MODELS":/root/.paddlex \
    "$API_IMAGE" >/dev/null
  ok "容器 $C_API 已启动"
}

# ---------------------------------------------------------------- 故障诊断
diagnose() {
  local c="$1" logs
  logs="$(docker logs --tail 400 "$c" 2>&1 || true)"
  echo
  rule
  printf '%s容器 %s 退出（exit=%s），最后 40 行日志：%s\n' \
    "$c_red" "$c" "$(container_exit_code "$c")" "$c_off"
  rule
  tail -40 <<<"$logs" | sed 's/^/  /'
  rule

  say "可能的原因"
  local hit=0
  if grep -qiE 'no kernel image|sm_120|kernel image is available|CUDA capability' <<<"$logs"; then
    hit=1
    warn "GPU 架构不匹配：镜像里的 PyTorch/vLLM 不支持 sm_120（Blackwell）"
    dim "      先跑 bash scripts/manage.sh gpucheck 确认"
    dim "      若确认不支持，需要换支持 Blackwell 的镜像或自行升级 vLLM"
  fi
  if grep -qiE 'proxyerror|max retries|connection refused|connection reset|timed out|temporary failure in name resolution' <<<"$logs"; then
    hit=1
    warn "网络问题：模型权重没下下来"
    dim "      带上代理重试：PROXY=http://127.0.0.1:7788 bash scripts/manage.sh deploy"
    dim "      容器里 127.0.0.1 指容器自己，脚本已自动改写为 host.docker.internal"
  fi
  if grep -qiE 'not a valid|repository not found|no such file or directory|could not find|unknown model|invalid model|does not exist' <<<"$logs"; then
    hit=1
    warn "模型名可能不对：当前用的是 $MODEL"
    dim "      看镜像支持哪些：bash scripts/manage.sh server-help"
    dim "      换一个再试：MODEL=PaddleOCR-VL-0.9B bash scripts/manage.sh deploy"
  fi
  if grep -qiE 'out of memory|CUDA out of memory|OOM' <<<"$logs"; then
    hit=1
    warn "显存不足"
    dim "      换张空卡：GPU_ID=<n> bash scripts/manage.sh deploy"
    dim "      或限制占用：VLLM_ARGS='--gpu-memory-utilization 0.5'（先用 server-help 确认参数名）"
  fi
  if grep -qiE 'address already in use' <<<"$logs"; then
    hit=1
    warn "容器内端口冲突：VLLM_PORT=$VLLM_PORT"
  fi
  if grep -qiE 'flash.?attn' <<<"$logs"; then
    hit=1
    warn "FlashAttention 相关：Blackwell 上 FA3 不可用，已设 VLLM_FLASH_ATTN_VERSION=2"
    dim "      仍失败可试：VLLM_ARGS='--enforce-eager'"
  fi
  [ $hit -eq 0 ] && dim "      没匹配到已知模式，把上面日志发我看看"
  echo
  dim "  完整日志：docker logs $c"
  dim "  改完配置直接重跑：bash scripts/manage.sh deploy（幂等，会自己清理）"
}

wait_ready() {
  say "等待服务就绪（首次需下载模型，可能几分钟到十几分钟）"
  local t0 now spin=0
  local frames=('|' '/' '-' '\')
  t0="$(date +%s)"
  while :; do
    now="$(date +%s)"
    if [ $((now - t0)) -gt "$WAIT_TIMEOUT" ]; then
      echo; warn "等待超时（${WAIT_TIMEOUT}s）"
      dim "  bash scripts/manage.sh logs vllm"
      return 1
    fi
    for c in "$C_VLLM" "$C_API"; do
      if [ "$(container_state "$c")" != "running" ]; then
        echo; diagnose "$c"; return 1
      fi
    done
    if http_ok "http://127.0.0.1:$PORT/healthz" \
       && grep -q '"ready": *true' <<<"$(curl -fsS -m 5 "http://127.0.0.1:$PORT/healthz")"; then
      echo; ok "服务就绪"; return 0
    fi
    spin=$(( (spin + 1) % 4 ))
    printf '\r    %s 已等待 %ss   ' "${frames[$spin]}" "$((now - t0))"
    sleep 3
  done
}

print_endpoints() {
  local ip; ip="$(hostname -I 2>/dev/null | awk '{print $1}')"; ip="${ip:-<服务器IP>}"
  echo
  printf '  %s界面%s   http://%s:%s/\n'             "$c_grn" "$c_off" "$ip" "$PORT"
  printf '  %sAPI %s   POST http://%s:%s/api/ocr\n' "$c_grn" "$c_off" "$ip" "$PORT"
  printf '  %s信息%s   http://%s:%s/api/info\n'     "$c_grn" "$c_off" "$ip" "$PORT"
  echo
  dim "  curl -F 'file=@a.png' http://$ip:$PORT/api/ocr"
  echo
}

# ---------------------------------------------------------------- 命令
# deploy 是幂等的：先清掉上一次的容器，再按当前配置重建重启。
# 反复跑、改了配置再跑、上次跑挂了再跑，都用这一条。
cmd_deploy() {
  say "清理上一次的容器（镜像与模型缓存保留）"
  docker rm -f "$C_API" "$C_VLLM" >/dev/null 2>&1 || true
  ok "已清理"
  echo

  cmd_preflight
  cmd_pull
  cmd_build
  ensure_net

  local dev; dev="$(resolve_device)"
  start_vllm
  start_api "$dev"

  if wait_ready; then
    echo; cmd_status; print_endpoints
  else
    return 1
  fi
}

# 强制重来：额外丢掉探测缓存和自建镜像，但保留官方镜像与模型权重
cmd_reset() {
  say "重置：清掉容器、探测缓存、自建镜像"
  docker rm -f "$C_API" "$C_VLLM" >/dev/null 2>&1 || true
  docker image rm "$API_IMAGE" >/dev/null 2>&1 || true
  rm -f "$STATE/paddle_gpu" "$STATE/paddle_probe.log" "$STATE/torch_probe.log"
  ok "已重置（官方镜像和模型权重都还在，不会重下）"
  echo
  cmd_deploy
}

cmd_down()  { say "停止"; docker rm -f "$C_API" "$C_VLLM" >/dev/null 2>&1 || true; ok "已停止（镜像与模型缓存保留）"; }
cmd_up()    { cmd_deploy; }

cmd_status() {
  printf '  %-22s %s\n' "项目"  "$PROJECT"
  printf '  %-22s %s\n' "模型"  "$MODEL"
  printf '  %-22s %s\n' "GPU"   "$GPU_ID"
  printf '  %-22s %s\n' "端口"  "$BIND:$PORT"
  [ -s "$STATE/paddle_gpu" ] && printf '  %-22s %s\n' "版面分析设备" "$(cat "$STATE/paddle_gpu")"
  for c in "$C_VLLM" "$C_API"; do
    local s; s="$(container_state "$c")"
    if [ "$s" = "running" ]; then printf '  %-22s %s%s%s\n' "$c" "$c_grn" "$s" "$c_off"
    else printf '  %-22s %s%s (exit=%s)%s\n' "$c" "$c_ylw" "$s" "$(container_exit_code "$c")" "$c_off"; fi
  done
  if http_ok "http://127.0.0.1:$PORT/healthz"; then
    curl -fsS -m 5 "http://127.0.0.1:$PORT/healthz" | sed 's/^/  /'; echo
  fi
}

cmd_logs() {
  case "${1:-api}" in
    vllm) docker logs -f --tail 200 "$C_VLLM" ;;
    api)  docker logs -f --tail 200 "$C_API" ;;
    *)    die "logs 参数只能是 api 或 vllm" ;;
  esac
}

cmd_test() {
  local f="${1:-}"
  if [ -n "$f" ]; then
    [ -f "$f" ] || die "文件不存在：$f"
  else
    say "未指定文件，生成一张测试图"
    mkdir -p "$STATE"; f="$STATE/sample.png"
    docker run --rm -v "$STATE":/out:z "$API_IMAGE" python /app/make_sample.py /out/sample.png
  fi
  say "识别 $f"
  local t0 t1; t0="$(date +%s%3N)"
  curl -fsS -m 300 -F "file=@$f" "http://127.0.0.1:$PORT/api/ocr" -o "$STATE/result.json" \
    || die "请求失败，先看 bash scripts/manage.sh logs api"
  t1="$(date +%s%3N)"
  ok "完成，耗时 $((t1 - t0)) ms"
  python3 - "$STATE/result.json" <<'PY' 2>/dev/null || cat "$STATE/result.json"
import json,sys
d=json.load(open(sys.argv[1],encoding="utf-8"))
print("\n--- markdown ---")
print((d.get("markdown") or "")[:2000])
print("\n--- meta ---")
print(json.dumps({k:v for k,v in d.items() if k not in ("markdown","result")},ensure_ascii=False,indent=2))
PY
}

cmd_bench() {
  have python3 || die "需要 python3（只用标准库）"
  python3 scripts/bench.py --url "http://127.0.0.1:$PORT/api/ocr" "$@"
}

cmd_disk() {
  say "镜像（拉过就不再拉，除非手动删）"
  docker images --format '  {{.Size}}\t{{.Repository}}:{{.Tag}}' | grep -E "paddleocr|$PROJECT" || dim "  （还没拉）"
  say "模型缓存卷（down/reset 都不会删）"
  for v in "$V_MODELS" "$V_HF"; do
    if docker volume inspect "$v" >/dev/null 2>&1; then
      local mp sz; mp="$(docker volume inspect -f '{{.Mountpoint}}' "$v")"
      sz="$(du -sh "$mp" 2>/dev/null | cut -f1)"
      printf '  %-8s %s\n' "${sz:-?}" "$v"
    else
      printf '  %-8s %s\n' "-" "$v （未创建）"
    fi
  done
  say "docker 根目录"
  df -h /var/lib/docker 2>/dev/null | sed 's/^/  /'
  echo
  dim "  彻底删干净（下次要重下）："
  dim "    docker volume rm $V_MODELS $V_HF"
  dim "    docker image rm $VLLM_IMAGE $BASE_IMAGE"
  dim "  ⚠ 不要用 docker system prune -a，会把上面全清掉"
}

cmd_server_help() {
  say "genai_server 真实可用参数（用于 MODEL / VLLM_ARGS）"
  docker run --rm "$VLLM_IMAGE" paddleocr genai_server --help
}

cmd_clean() {
  say "清理本 demo 的容器/网络/自建镜像"
  docker rm -f "$C_API" "$C_VLLM" >/dev/null 2>&1 || true
  docker network rm "$NET" >/dev/null 2>&1 || true
  docker image rm "$API_IMAGE" >/dev/null 2>&1 || true
  rm -rf "$STATE"
  ok "已清理"
  dim "  模型缓存卷保留（重装免下载）：docker volume rm $V_MODELS $V_HF"
  dim "  官方基础镜像保留：docker image rm $VLLM_IMAGE $BASE_IMAGE"
}

cmd_help() {
  cat <<EOF
knock-ocr demo

  bash scripts/manage.sh <命令>

命令
  deploy        一条命令搞定部署。幂等 —— 反复跑、改配置再跑、上次挂了再跑都用它
  reset         强制重来（丢探测缓存 + 重建 API 镜像），官方镜像和模型权重保留
  down          停止
  status        当前状态
  gpucheck      单独跑 GPU 探测：先查 vLLM(PyTorch) 再查 Paddle，各自给结论
  logs [api|vllm]
  test [文件]   端到端识别一次（不传文件则自动造一张）
  bench         并发压测，看 QPS / P50 / P95
  disk          镜像与模型缓存占了多少盘
  server-help   查看 genai_server 支持哪些参数（模型名对不对看这个）
  clean         删掉容器/网络/自建镜像（模型权重保留）

可覆盖的环境变量（写在命令前面）
  GPU_ID=$GPU_ID          只占用这一张卡
  PORT=$PORT           对外端口；被别的服务占用会自动顺延，STRICT_PORT=1 可禁止
  BIND=$BIND        只本机访问传 127.0.0.1
  PROXY=                出网代理，如 http://127.0.0.1:7788（下模型权重用）
  WORKERS=$WORKERS            API 并行流水线数
  DEVICE=$DEVICE         auto|cpu|gpu:0，版面分析设备
  MODEL=$MODEL
  VLLM_ARGS=            透传给 genai_server
  PROJECT=$PROJECT     换名字可并存多套

例
  PROXY=http://127.0.0.1:7788 bash scripts/manage.sh deploy
  GPU_ID=1 PORT=9000 bash scripts/manage.sh deploy
EOF
}

case "${1:-deploy}" in
  deploy)  shift; cmd_deploy "$@" ;;
  redeploy) shift; cmd_deploy "$@" ;;
  reset)   shift; cmd_reset "$@" ;;
  up)      shift; cmd_up "$@" ;;
  down)    shift; cmd_down "$@" ;;
  restart) shift; cmd_deploy "$@" ;;
  status)  shift; cmd_status "$@" ;;
  gpucheck) shift; cmd_gpucheck "$@" ;;
  logs)    shift; cmd_logs "$@" ;;
  test)    shift; cmd_test "$@" ;;
  bench)   shift; cmd_bench "$@" ;;
  disk)    shift; cmd_disk "$@" ;;
  pull)    shift; cmd_pull "$@" ;;
  build)   shift; cmd_build "$@" ;;
  preflight) shift; cmd_preflight "$@" ;;
  server-help) shift; cmd_server_help "$@" ;;
  clean)   shift; cmd_clean "$@" ;;
  help|-h|--help) cmd_help ;;
  *) die "未知命令：$1（bash scripts/manage.sh help）" ;;
esac
