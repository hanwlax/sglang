#!/usr/bin/env bash
set -eo pipefail
# shellcheck source-path=SCRIPTDIR
# shellcheck source=env.sh
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
require_source
export MODEL_PATH=${MODEL_PATH:-/mnt/share/weight/LongCat-2.0-FP8-cut}
test -r "$MODEL_PATH/config.json"
python3 - <<'PY'
import os
import socket
with socket.socket() as sock:
    sock.bind(("0.0.0.0", int(os.environ["HTTP_PORT"])))
PY
mkdir -p "$(dirname -- "$LC_A5_RUN")"
# Existing captures must never be silently mixed or overwritten.
mkdir "$LC_A5_RUN"
git -C "$SGLANG_SOURCE_ROOT" rev-parse HEAD > "$LC_A5_RUN/source_commit.txt"
export LC_ROUTE_MODE=ids_weights
export LC_ROUTE_REFERENCE="$LC_REFERENCE"
export LC_ROUTE_OUT="$LC_A5_RUN/routes"
export LC_FULL_DUMP_OUT="$LC_A5_RUN/full"
export LOG_DIR="$LC_A5_RUN/serve_logs"
export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1
export HCCL_OP_EXPANSION_MODE=AI_CPU
bash "$SGLANG_SOURCE_ROOT/scripts/longcat2_950dt/run_trace.sh" A5 \
  2>&1 | tee "$LC_A5_RUN/serve.log"
