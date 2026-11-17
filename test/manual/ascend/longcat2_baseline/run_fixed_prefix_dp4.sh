#!/usr/bin/env bash
set -eo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
run_id=${1:?Usage: bash run_fixed_prefix_dp4.sh TRACE_SERVER_RUN_ID [CONTROL_JSON]}
[[ "$run_id" =~ ^[a-zA-Z0-9_-]+$ ]] || exit 2
control=${2:-/tmp/longcat-control.json}
BASE_URL=${BASE_URL:-http://192.168.25.209:10096}
mkdir -p "$PWD/results/$run_id"
python3 check_server_dp4.py --url "$BASE_URL" --trace --out "$PWD/results/$run_id/server_info.json"
python3 replay_fixed_decode_dp4.py --control "$control" --url "$BASE_URL" \
  --dps 0 1 2 3 --lengths 1 8 --repeat 2 --timeout 3000 \
  --out "$PWD/results/$run_id/client" \
  | tee "$PWD/results/$run_id/replay.log"
