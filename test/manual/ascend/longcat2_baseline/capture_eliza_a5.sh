#!/usr/bin/env bash
# Complete FP8 model, genuine routing, fixed A3 Eliza inputs.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
tool_dir=$PWD
export RUNTIME=${RUNTIME:-$tool_dir/runtime_eliza_a5}
export CAPTURE_ROOT=${CAPTURE_ROOT:-$tool_dir}
export LC_INTERNAL_LAYERS=0,7,37
export LC_INTERNAL_RANKS=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export HCCL_OP_EXPANSION_MODE=AI_CPU
export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1
action=${1:?Usage: capture_eliza_a5.sh prepare SOURCE | start NODE_RANK RUN_ID | replay RUN_ID | verify RUN_ID}
shift
case "$action" in
  prepare)
    source_root=${1:?Supply the source checkout containing the pinned revision}
    python3 prepare_runtime_dp4.py --source "$source_root" --mode trace \
      --trace-topology a5-dp2 --full-boundary-layers {0..37} \
      --with-internal-attention --load-barrier-timeout 3600 --out "$RUNTIME"
    python3 check_capture_runtime.py --runtime "$RUNTIME" --topology a5-dp2
    python3 verify_full_boundary.py
    ;;
  start)
    node_rank=${1:?Supply node rank 0,1,2,3}
    run_id=${2:?Supply the same NEW run ID on all four hosts}
    python3 check_capture_runtime.py --runtime "$RUNTIME" --topology a5-dp2
    # The launcher records the actual checkpoint/config/SDK/env and runtime hashes.
    exec bash run_a5_full_boundary.sh "$node_rank" "$run_id" trace "${3:-}"
    ;;
  replay|verify)
    run_id=${1:?Supply the service run ID}
    [[ "$run_id" =~ ^[a-zA-Z0-9_-]+$ ]] || exit 2
    result="$CAPTURE_ROOT/results/$run_id"
    if [[ "$action" == replay ]]; then
      python3 check_capture_runtime.py --runtime "$RUNTIME" --topology a5-dp2
      [[ ! -e "$result/client_eliza" ]] || { echo 'Existing client evidence: choose a fresh service RUN_ID' >&2; exit 1; }
      mkdir -p "$result"
      url=${BASE_URL:-http://141.61.45.105:10086}
      python3 check_server_dp4.py --url "$url" --trace --topology a5-dp2 \
        --out "$result/server_info.json"
      cp "$RUNTIME/receipt.json" "$result/client_runtime_receipt.json"
      python3 replay_fixed_decode_dp4.py --topology a5-dp2 \
        --fixture "$tool_dir/fixtures/eliza_853.json" --url "$url" \
        --dps 0 --lengths 1 11 --repeat 2 --timeout 3000 \
        --out "$result/client_eliza" | tee "$result/replay.log"
    else
      python3 verify_full_capture.py --topology a5-dp2 \
        --manifest "$result/client_eliza/manifest.json" --dump "$result/dumps" \
        --verify-payloads --workers "${VERIFY_WORKERS:-8}" \
        --out "$result/verification.json" | tee "$result/verification.log"
    fi
    ;;
  *) echo "Unknown action: $action" >&2; exit 2 ;;
esac
