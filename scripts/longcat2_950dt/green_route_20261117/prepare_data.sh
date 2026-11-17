#!/usr/bin/env bash
set -eo pipefail
# shellcheck source-path=SCRIPTDIR
# shellcheck source=env.sh
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
archive=${1:?用法: bash prepare_data.sh /绝对路径/A3_route_full_20261117_2110.tar.gz}
archive=$(realpath -- "$archive")
expected=$(cut -d ' ' -f 1 "$green_dir/reference/archive.sha256")
printf '%s  %s\n' "$expected" "$archive" | sha256sum -c -
mkdir -p "$LC_PROGRESS_ROOT/route_runs"
if [[ -e "$LC_A3_CAPTURE" ]]; then
  echo "目录已经存在，不覆盖：$LC_A3_CAPTURE" >&2
  echo '若已成功解压，直接校验其中 SHA256SUMS 后执行 compare_fixed.sh。' >&2
  exit 1
fi
tar -xzf "$archive" -C "$LC_PROGRESS_ROOT/route_runs"
(cd "$LC_A3_CAPTURE" && sha256sum --quiet -c SHA256SUMS)
cmp "$LC_REFERENCE" "$LC_A3_CAPTURE/A3_reference.json"
echo "A3_DATA_OK $LC_A3_CAPTURE"
