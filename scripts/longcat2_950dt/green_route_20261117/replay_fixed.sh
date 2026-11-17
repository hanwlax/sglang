#!/usr/bin/env bash
set -eo pipefail
# shellcheck source-path=SCRIPTDIR
# shellcheck source=env.sh
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
require_source
test -s "$LC_A5_RUN/serve.log"
python3 "$SGLANG_SOURCE_ROOT/scripts/longcat2_950dt/route_lock_tools.py" replay \
  --control "$green_dir/reference/control.json" \
  --url "http://127.0.0.1:$HTTP_PORT" --timeout 3000 \
  --log "$LC_A5_RUN/serve.log" \
  --out "$LC_A5_RUN/client" --require-full \
  2>&1 | tee "$LC_A5_RUN/replay.log"
python3 "$SGLANG_SOURCE_ROOT/scripts/longcat2_950dt/route_lock_tools.py" audit \
  --routes "$LC_A5_RUN/routes" --reference "$LC_REFERENCE" \
  2>&1 | tee "$LC_A5_RUN/audit.log"
