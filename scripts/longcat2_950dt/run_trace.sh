#!/usr/bin/env bash
# Two-layer TP8/DP1/EP1 LongCat diagnostic runner; graphs and NEXTN disabled.
set -eo pipefail

platform=${1:-}
case "$platform" in
  A5) fallback=/mnt/share/weight/LongCat-2.0-FP8-cut ;;
  A3) fallback=/home/l00951279/longcat2/LongCat-2.0-INT8-part ;;
  *) echo '用法: bash run_trace.sh A5|A3' >&2; exit 2 ;;
esac

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
SGLANG_SOURCE_ROOT=${SGLANG_SOURCE_ROOT:-$(cd "$script_dir/../.." && pwd)}
MODEL_PATH=${MODEL_PATH:-$fallback}
HTTP_PORT=${HTTP_PORT:-10086}
LOG_DIR=${LOG_DIR:-logs/serve_logs}

[[ -r "$MODEL_PATH/config.json" ]] || {
  echo "找不到 $MODEL_PATH/config.json，请设置 MODEL_PATH" >&2
  exit 1
}
for probe in longcat_activation_trace.py longcat_route_lock.py; do
  [[ -r "$SGLANG_SOURCE_ROOT/python/sglang/srt/debug_utils/$probe" ]] || {
    echo "找不到 $probe，请使用 codex/longcat2-route-lock-full 分支" >&2
    exit 1
  }
done

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY PYTHONPATH
source "${CANN_ENV:-/usr/local/Ascend/ascend-toolkit/set_env.sh}"
source "${ATB_ENV:-/usr/local/Ascend/nnal/atb/set_env.sh}"
export PYTHONPATH="$SGLANG_SOURCE_ROOT/python${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

export TORCH_NPU_DEVICE_CAPABILITY=${TORCH_NPU_DEVICE_CAPABILITY:-8.0}
export HCCL_CONNECT_TIMEOUT=${HCCL_CONNECT_TIMEOUT:-3000}
export HCCL_BUFFSIZE=${HCCL_BUFFSIZE:-200}
export HCCL_OP_EXPANSION_MODE=${HCCL_OP_EXPANSION_MODE:-AIV}
export PYTORCH_NPU_ALLOC_CONF=${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}
export SGLANG_SET_CPU_AFFINITY=1
export SGLANG_ONE_VISIBLE_DEVICE_PER_PROCESS=1
export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=${SGLANG_ENABLE_OVERLAP_PLAN_STREAM:-1}
export STREAMS_PER_DEVICE=${STREAMS_PER_DEVICE:-32}
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-$HOME/.cache/sglang/triton}
unset ASCEND_LAUNCH_BLOCKING

export ASCEND_USE_FIA=1
export SGLANG_NPU_USE_MLAPO=0
export SGLANG_NPU_ARCH35_REQUANT_BLOCK_FP8=0
export SGLANG_DEBUG_LONGCAT_ACTIVATIONS=1

mkdir -p "$LOG_DIR"
LOG_DIR=$(cd "$LOG_DIR" && pwd)
log="$LOG_DIR/longcat_trace_${platform}_$(date +%Y%m%d_%H%M%S)_$$.log"
printf '%s\n' "$log" > "$LOG_DIR/longcat_trace_${platform}.latest"
printf '[LongCat] platform=%s model=%s HCCL=%s log=%s\n' \
  "$platform" "$MODEL_PATH" "$HCCL_OP_EXPANSION_MODE" "$log"
printf '[LongCat] source=%s route_mode=%s routes=%s full=%s\n' \
  "$SGLANG_SOURCE_ROOT" "${LC_ROUTE_MODE:-off}" "${LC_ROUTE_OUT:-}" "${LC_FULL_DUMP_OUT:-}"

python3 -m sglang.launch_server \
  --model-path "$MODEL_PATH" --tokenizer-path "$MODEL_PATH" \
  --served-model-name LongCat-2.0 --trust-remote-code \
  --model-loader-extra-config '{"enable_multithread_load":true}' \
  --device npu --attention-backend ascend \
  --dtype bfloat16 --kv-cache-dtype bf16 \
  --nnodes 1 --node-rank 0 --tp-size 8 --dp-size 1 --attn-dp-size 1 --ep-size 1 \
  --moe-a2a-backend none --disaggregation-mode null \
  --json-model-override-args '{"num_hidden_layers":2}' \
  --cuda-graph-backend-prefill disabled --cuda-graph-backend-decode disabled \
  --disable-radix-cache --skip-server-warmup \
  --page-size 128 --mem-fraction-static 0.85 --chunked-prefill-size 1024 \
  --max-total-tokens 2048 --max-running-requests 1 --watchdog-timeout 9000 \
  --log-level info --host 0.0.0.0 --port "$HTTP_PORT" \
  2>&1 | tee "$log"
