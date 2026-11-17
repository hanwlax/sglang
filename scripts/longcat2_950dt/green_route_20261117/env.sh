#!/usr/bin/env bash
# Source from the entry scripts; all paths refer to the current checkout.
# shellcheck disable=SC2034
green_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
SGLANG_SOURCE_ROOT=$(cd "$green_dir/../../.." && pwd)
export SGLANG_SOURCE_ROOT
LC_PROGRESS_ROOT=${LC_PROGRESS_ROOT:-/mnt/share/hanwlax/workspace/progress/longcat2_0}
LC_A3_CAPTURE="$LC_PROGRESS_ROOT/route_runs/A3_route_full_20261117_2110"
LC_A5_RUN=${LC_A5_RUN:-$LC_PROGRESS_ROOT/route_runs/A5_fixed_20261117}
LC_COMPARE_OUT=${LC_COMPARE_OUT:-$LC_PROGRESS_ROOT/route_runs/compare_fixed_20261117}
LC_REFERENCE="$green_dir/reference/A3_reference.json"
export HTTP_PORT=${HTTP_PORT:-10096}

require_source() {
  # The A3 run used these exact runtime files. Delivery helpers may change.
  git -C "$SGLANG_SOURCE_ROOT" diff --exit-code \
    4fd8d9febb95cab8ff679320faa3bcc44ebd7aff -- \
    python/sglang/srt/models/longcat_flash.py \
    python/sglang/srt/debug_utils/longcat_activation_trace.py \
    python/sglang/srt/debug_utils/longcat_route_lock.py \
    scripts/longcat2_950dt/run_trace.sh \
    scripts/longcat2_950dt/route_lock_tools.py \
    scripts/longcat2_950dt/compare_full_route_tensors.py
  (cd "$green_dir/reference" && sha256sum -c SHA256SUMS)
}
