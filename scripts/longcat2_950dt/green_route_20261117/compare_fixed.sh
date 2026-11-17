#!/usr/bin/env bash
set -eo pipefail
# shellcheck source-path=SCRIPTDIR
# shellcheck source=env.sh
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
require_source
cmp "$LC_REFERENCE" "$LC_A3_CAPTURE/A3_reference.json"
python3 "$SGLANG_SOURCE_ROOT/scripts/longcat2_950dt/route_lock_tools.py" audit \
  --routes "$LC_A3_CAPTURE/A3_fixed/routes" --reference "$LC_REFERENCE"
python3 "$SGLANG_SOURCE_ROOT/scripts/longcat2_950dt/route_lock_tools.py" audit \
  --routes "$LC_A5_RUN/routes" --reference "$LC_REFERENCE"
test ! -e "$LC_COMPARE_OUT"
mkdir -p "$(dirname -- "$LC_COMPARE_OUT")"
python3 "$SGLANG_SOURCE_ROOT/scripts/longcat2_950dt/compare_full_route_tensors.py" \
  --a3 "$LC_A3_CAPTURE/A3_fixed" --a5 "$LC_A5_RUN" \
  --require-fixed --threads 4 --out "$LC_COMPARE_OUT" \
  2>&1 | tee "$LC_COMPARE_OUT.log"
cat "$LC_COMPARE_OUT/summary.txt"
