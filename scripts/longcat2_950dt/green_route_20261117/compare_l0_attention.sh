#!/usr/bin/env bash
# Compare the two existing fixed-route captures, entirely on CPU.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
ref=${1:-"$LC_PROGRESS_ROOT/route_runs/a3_l0_qkv.pt.gz"}
sha=77af206bf3997823394679ff8d4f2169b6a700670890665f7df2bc48125473e8
printf '%s  %s\n' "$sha" "$ref" | sha256sum -c -
shopt -s nullglob
extra=()
tails=(/tmp/longcat_attention_tail_rank0_*.pt)
for path in "${tails[@]}"; do extra+=(--tail "$path"); done
qkv=(/tmp/longcat_qkv_fp8_rank0_*.pt)
if [[ -n "${QKV_DUMP:-}" ]]; then
    extra+=(--qkv "$QKV_DUMP")
elif (( ${#qkv[@]} == 1 )); then
    extra+=(--qkv "${qkv[0]}")
elif (( ${#qkv[@]} > 1 )); then
    printf 'Multiple optional QKV dumps. Set QKV_DUMP to check one: %s\n' "${qkv[@]}"
fi
log=$(mktemp "$LC_PROGRESS_ROOT/route_runs/l0_attention.XXXXXX.log")
printf 'LOG %s\n' "$log"
python3 "$SGLANG_SOURCE_ROOT/scripts/longcat2_950dt/replay_l0_attention.py" compare \
    --a3 "$LC_A3_CAPTURE/A3_fixed" --a5 "$LC_A5_RUN" \
    --model "${MODEL_PATH:-/mnt/share/weight/LongCat-2.0-FP8-cut}" \
    --reference "$ref" "${extra[@]}" | tee "$log"
