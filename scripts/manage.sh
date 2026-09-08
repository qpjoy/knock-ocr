#!/usr/bin/env bash
# knock-ocr demo 一键管理脚本
#   bash scripts/manage.sh deploy
# 所有配置都有默认值，需要改时在命令前面传：
#   GPU_ID=1 PORT=9000 bash scripts/manage.sh deploy
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# ---------------------------------------------------------------- 配置
PROJECT="${PROJECT:-knock-ocr}"          # 容器/网络/卷名前缀，改它可并存多套
GPU_ID="${GPU_ID:-2}"                    # 只占用这一张卡（默认避开挂显示器的 GPU3）
PORT="${PORT:-8710}"                     # 对外 Web + API 端口
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

# ---------------------------------------------------------------- 工具
have() { command -v "$1" >/dev/null 2>&1; }

port_busy() {
  if have ss;   then ss -ltn 2>/dev/null | grep -qE "[:.]$1[[:space:]]" && return 0
  elif have netstat; then netstat -ltn 2>/dev/null | grep -qE "[:.]$1[[:space:]]" && return 0
  fi
  return 1
}

container_state() { docker inspect -f '{{.State.Status}}' "$1" 2>/dev/null || echo "absent"; }

http_ok() { curl -fsS -m 3 -o /dev/null "$1" 2>/dev/null; }

# 容器里的 127.0.0.1 是容器自己，不是宿主机。桥接网络下要改走 host-gateway。
proxy_for_container() { sed -E 's#//(127\.0\.0\.1|localhost)([:/]|$)#//host.docker.internal\2#' <<<"$1"; }

# 输出 docker run 需要的代理相关参数（PROXY 未设则输出空）
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

  # 检查这张卡上有没有别人在跑，避免打扰
  local busy; busy="$(nvidia-smi -i "$GPU_ID" --query-compute-apps=pid,used_memory --format=csv,noheader || true)"
  if [ -n "$busy" ]; then
    warn "GPU $GPU_ID 上已有计算进程，demo 会与其共享显存："
    echo "$busy" | sed 's/^/      /'
    dim "      想换一张卡：GPU_ID=<n> bash scripts/manage.sh deploy"
  else
    ok "GPU $GPU_ID 空闲"
  fi

  # nvidia container runtime
  if docker info 2>/dev/null | grep -qi 'nvidia'; then
    ok "docker 已注册 nvidia runtime"
  else
    warn "docker info 里没看到 nvidia runtime，若 deploy 失败请检查 nvidia-container-toolkit"
  fi

  port_busy "$PORT" && die "端口 $PORT 已被占用。换一个：PORT=<n> bash scripts/manage.sh deploy"
  ok "端口 $PORT 可用"

  # RHEL/CentOS 默认开着 firewalld，外部访问界面前需要放行
  if have firewall-cmd && firewall-cmd --state >/dev/null 2>&1; then
    if firewall-cmd --query-port="$PORT/tcp" >/dev/null 2>&1; then
      ok "firewalld 已放行 $PORT/tcp"
    elif [ "$BIND" = "127.0.0.1" ]; then
      ok "firewalld 运行中；BIND=127.0.0.1 仅本机访问，无需放行"
    else
      warn "firewalld 运行中且未放行 $PORT/tcp，外部浏览器可能连不上。放行："
      dim "      firewall-cmd --add-port=$PORT/tcp --permanent && firewall-cmd --reload"
    fi
  fi

  # SELinux enforcing 下 bind mount 需要重打标签，脚本里已加 :z
  if have getenforce && [ "$(getenforce 2>/dev/null)" = "Enforcing" ]; then
    dim "      SELinux=Enforcing（已按需加 :z，正常情况无需额外处理）"
  fi

  if [ -n "$PROXY" ]; then
    if curl -fsS -m 5 -x "$PROXY" -o /dev/null https://www.baidu.com 2>/dev/null; then
      ok "代理可用：$PROXY（容器内会自动改写为 $(proxy_for_container "$PROXY")）"
    else
      warn "代理 $PROXY 从宿主机测不通，模型下载可能失败"
    fi
  else
    dim "      未设代理。若模型下载卡住，加上：PROXY=http://127.0.0.1:7788"
  fi

  local free; free="$(df -Pk /var/lib/docker 2>/dev/null | awk 'NR==2{print int($4/1048576)}' || echo 0)"
  if [ "${free:-0}" -lt 40 ]; then
    warn "/var/lib/docker 可用空间约 ${free}GB，镜像约需 20~30GB"
  else
    ok "磁盘空间充足（约 ${free}GB）"
  fi
  echo
}

# ---------------------------------------------------------------- Paddle GPU 探测
# RTX 50 系是 Blackwell(sm_120)，官方 paddlepaddle-gpu wheel 长期不含 sm_120 kernel。
# 这里实测一次，结果决定版面分析模型放 CPU 还是 GPU。VLM 主体走 vLLM，不受影响。
probe_paddle_gpu() {
  mkdir -p "$STATE"
  local f="$STATE/paddle_gpu"
  if [ "$DEVICE" != "auto" ]; then echo "$DEVICE" > "$f"; echo "$DEVICE"; return; fi
  if [ -s "$f" ]; then cat "$f"; return; fi

  say "探测 PaddlePaddle 能否用上这张卡（Blackwell sm_120 已知问题）" >&2
  local out rc=0
  out="$(docker run --rm --gpus "device=$GPU_ID" "$BASE_IMAGE" \
        python -c "import paddle;paddle.utils.run_check()" 2>&1)" || rc=$?

  local dev
  if [ $rc -eq 0 ] && ! grep -qiE 'no kernel image|not compiled|sm_120|no CUDA' <<<"$out"; then
    dev="gpu:0"
    ok "PaddlePaddle 可用 GPU，版面分析走 GPU" >&2
  else
    dev="cpu"
    printf '%s\n' "$out" > "$STATE/paddle_probe.log"
    if grep -qiE 'no kernel image|sm_120|not compiled' <<<"$out"; then
      warn "PaddlePaddle 不含 sm_120 kernel（Blackwell 已知问题），版面分析改走 CPU" >&2
    else
      warn "PaddlePaddle GPU 自检未通过，版面分析改走 CPU" >&2
      dim "      若不是 sm_120 问题，可能是 nvidia-container-toolkit，见日志" >&2
    fi
    dim "      这不影响识别精度：决定 96.3% 的 0.9B VLM 跑在 vLLM(GPU) 上，" >&2
    dim "      CPU 只承担轻量版面检测，本机 128 核绰绰有余。" >&2
    dim "      完整日志：$STATE/paddle_probe.log" >&2
  fi
  echo "$dev" > "$f"
  echo "$dev"
}

# ---------------------------------------------------------------- 构建 / 启动
ensure_net() {
  docker network inspect "$NET" >/dev/null 2>&1 || docker network create "$NET" >/dev/null
  docker volume inspect "$V_MODELS" >/dev/null 2>&1 || docker volume create "$V_MODELS" >/dev/null
  docker volume inspect "$V_HF"     >/dev/null 2>&1 || docker volume create "$V_HF" >/dev/null
}

cmd_pull() {
  say "拉取官方镜像（首次约 20~30GB，慢）"
  docker image inspect "$VLLM_IMAGE" >/dev/null 2>&1 || docker pull "$VLLM_IMAGE"
  docker image inspect "$BASE_IMAGE" >/dev/null 2>&1 || docker pull "$BASE_IMAGE"
  ok "镜像就绪"
}

cmd_build() {
  say "构建 API 镜像（这一层不需要联网）"
  local net=()
  # 构建期若配了代理，用 host 网络，这样 127.0.0.1:7788 这类本地代理才通
  [ -n "$PROXY" ] && net=(--network host --build-arg "http_proxy=$PROXY" --build-arg "https_proxy=$PROXY")
  docker build "${net[@]}" \
    --build-arg "BASE_IMAGE=$BASE_IMAGE" \
    -f docker/api.Dockerfile -t "$API_IMAGE" .
  ok "API 镜像就绪：$API_IMAGE"
}

start_vllm() {
  [ "$(container_state "$C_VLLM")" = "running" ] && { ok "vLLM 已在运行"; return; }
  docker rm -f "$C_VLLM" >/dev/null 2>&1 || true
  say "启动 VLM 推理服务（GPU $GPU_ID）"
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
  say "启动 API + Web 服务"
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

wait_ready() {
  say "等待服务就绪（首次需下载模型，可能几分钟到十几分钟）"
  local t0 now spin=0
  t0="$(date +%s)"
  while :; do
    now="$(date +%s)"
    if [ $((now - t0)) -gt "$WAIT_TIMEOUT" ]; then
      echo
      warn "等待超时（${WAIT_TIMEOUT}s）。看日志定位："
      dim "  bash scripts/manage.sh logs vllm"
      dim "  bash scripts/manage.sh logs api"
      return 1
    fi
    for c in "$C_VLLM" "$C_API"; do
      [ "$(container_state "$c")" = "running" ] || {
        echo; die "容器 $c 已退出，查看日志：bash scripts/manage.sh logs ${c##*-}"
      }
    done
    if http_ok "http://127.0.0.1:$PORT/healthz"; then
      local body; body="$(curl -fsS -m 5 "http://127.0.0.1:$PORT/healthz")"
      if grep -q '"ready": *true' <<<"$body"; then echo; ok "服务就绪"; return 0; fi
    fi
    local frames=('|' '/' '-' '\')
    spin=$(( (spin + 1) % 4 ))
    printf '\r    %s 已等待 %ss   ' "${frames[$spin]}" "$((now - t0))"
    sleep 3
  done
}

print_endpoints() {
  local ip; ip="$(hostname -I 2>/dev/null | awk '{print $1}')"; ip="${ip:-<服务器IP>}"
  echo
  printf '  %s界面%s   http://%s:%s/\n'        "$c_grn" "$c_off" "$ip" "$PORT"
  printf '  %sAPI %s   POST http://%s:%s/api/ocr\n' "$c_grn" "$c_off" "$ip" "$PORT"
  printf '  %s信息%s   http://%s:%s/api/info\n' "$c_grn" "$c_off" "$ip" "$PORT"
  echo
  dim "  curl -F 'file=@a.png' http://$ip:$PORT/api/ocr"
  echo
}

# ---------------------------------------------------------------- 命令
cmd_deploy() {
  cmd_preflight
  cmd_pull
  cmd_build
  ensure_net
  local dev; dev="$(probe_paddle_gpu)"
  say "版面分析设备：$dev ；VLM 推理：vLLM @ GPU $GPU_ID"
  start_vllm
  start_api "$dev"
  wait_ready || return 1
  cmd_status
  print_endpoints
}

cmd_up()      { ensure_net; start_vllm; start_api "$(probe_paddle_gpu)"; wait_ready && print_endpoints; }
cmd_down()    { say "停止"; docker rm -f "$C_API" "$C_VLLM" >/dev/null 2>&1 || true; ok "已停止（镜像与模型缓存保留）"; }
cmd_restart() { cmd_down; cmd_up; }

cmd_status() {
  printf '  %-22s %s\n' "项目" "$PROJECT"
  printf '  %-22s %s\n' "模型" "$MODEL"
  printf '  %-22s %s\n' "GPU" "$GPU_ID"
  printf '  %-22s %s\n' "端口" "$BIND:$PORT"
  for c in "$C_VLLM" "$C_API"; do
    local s; s="$(container_state "$c")"
    if [ "$s" = "running" ]; then printf '  %-22s %s%s%s\n' "$c" "$c_grn" "$s" "$c_off"
    else printf '  %-22s %s%s%s\n' "$c" "$c_ylw" "$s" "$c_off"; fi
  done
  if http_ok "http://127.0.0.1:$PORT/healthz"; then
    curl -fsS -m 5 "http://127.0.0.1:$PORT/healthz" | sed 's/^/  /'
    echo
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
    mkdir -p "$STATE"
    f="$STATE/sample.png"
    # :z 是给 RHEL/CentOS 的 SELinux 用的，重新打标签才写得进去；SELinux 关着时无副作用
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

cmd_server_help() {
  say "genai_server 真实可用参数（用于 VLLM_ARGS）"
  docker run --rm "$VLLM_IMAGE" paddleocr genai_server --help
}

cmd_clean() {
  say "清理本 demo 的全部痕迹"
  docker rm -f "$C_API" "$C_VLLM" >/dev/null 2>&1 || true
  docker network rm "$NET" >/dev/null 2>&1 || true
  docker image rm "$API_IMAGE" >/dev/null 2>&1 || true
  rm -rf "$STATE"
  ok "已清理容器/网络/自建镜像"
  dim "  模型缓存卷保留（重装免下载）：docker volume rm $V_MODELS $V_HF"
  dim "  官方基础镜像保留：docker image rm $VLLM_IMAGE $BASE_IMAGE"
}

cmd_help() {
  cat <<EOF
knock-ocr demo

  bash scripts/manage.sh <命令>

命令
  deploy        一键：检查 → 拉镜像 → 构建 → 启动 → 等就绪   ← 从这开始
  up/down/restart
  status        当前状态
  logs [api|vllm]
  test [文件]   端到端识别一次（不传文件则自动造一张）
  bench         并发压测，看 QPS / P50 / P95
  server-help   查看 genai_server 支持哪些参数
  clean         删掉本 demo 的容器/网络/自建镜像

可覆盖的环境变量（写在命令前面）
  GPU_ID=$GPU_ID          只占用这一张卡
  PORT=$PORT           对外端口
  BIND=$BIND        只本机访问传 127.0.0.1
  WORKERS=$WORKERS            API 并行流水线数
  DEVICE=$DEVICE         auto|cpu|gpu:0，版面分析设备
  PROXY=                出网代理，如 http://127.0.0.1:7788（下模型权重用）
                        容器内会自动把 127.0.0.1 改写成 host.docker.internal
  MODEL=$MODEL
  PROJECT=$PROJECT     换名字可并存多套

例
  PROXY=http://127.0.0.1:7788 bash scripts/manage.sh deploy
  GPU_ID=1 PORT=9000 bash scripts/manage.sh deploy
  BIND=127.0.0.1 bash scripts/manage.sh deploy
EOF
}

case "${1:-deploy}" in
  deploy) shift; cmd_deploy "$@" ;;
  up)     shift; cmd_up "$@" ;;
  down)   shift; cmd_down "$@" ;;
  restart) shift; cmd_restart "$@" ;;
  status) shift; cmd_status "$@" ;;
  logs)   shift; cmd_logs "$@" ;;
  test)   shift; cmd_test "$@" ;;
  bench)  shift; cmd_bench "$@" ;;
  pull)   shift; cmd_pull "$@" ;;
  build)  shift; cmd_build "$@" ;;
  preflight) shift; cmd_preflight "$@" ;;
  server-help) shift; cmd_server_help "$@" ;;
  clean)  shift; cmd_clean "$@" ;;
  help|-h|--help) cmd_help ;;
  *) die "未知命令：$1（bash scripts/manage.sh help）" ;;
esac
