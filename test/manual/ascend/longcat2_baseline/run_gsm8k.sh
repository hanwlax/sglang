#!/usr/bin/env bash
# Normal Completion evaluation: clean runtime, no forced token / trace hooks.
set -eo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
run_id=${1:?Usage: bash run_gsm8k.sh NEW_EVAL_ID [--dry-run]}
[[ "$run_id" =~ ^[a-zA-Z0-9_-]+$ ]] || exit 2
dry=${2:-}
[[ -z "$dry" || "$dry" == --dry-run ]] || exit 2
RUNTIME=${EVAL_RUNTIME:-$PWD/runtime_dp4_eval}
BASE_URL=${BASE_URL:-http://192.168.25.209:10096}
NUM_EXAMPLES=${NUM_EXAMPLES:-200}
NUM_THREADS=${NUM_THREADS:-1}
NUM_SHOTS=${NUM_SHOTS:-5}
MAX_TOKENS=${MAX_TOKENS:-2048}
export LC_EVAL_OUTPUT_DIR="$PWD/evaluations/$run_id"
export PYTHONPATH="$RUNTIME/python"
export PYTHONUNBUFFERED=1
export OPENAI_API_KEY=${OPENAI_API_KEY:-EMPTY}
# Local inference must not use an inherited proxy; dataset loading may need one.
export NO_PROXY="${NO_PROXY:+$NO_PROXY,}192.168.25.209,192.168.25.212,192.168.25.216,192.168.25.217,localhost,127.0.0.1"
export no_proxy="$NO_PROXY"
python3 - "$RUNTIME" <<'PY'
import json,sys
from pathlib import Path
r=json.loads((Path(sys.argv[1])/'receipt.json').read_text())
assert r['mode']=='eval', 'Use the clean evaluation runtime'
PY
args=(--base-url "$BASE_URL" --model LongCat-2.0 --eval-name gsm8k
  --api completion --num-examples "$NUM_EXAMPLES" --num-threads "$NUM_THREADS"
  --num-shots "$NUM_SHOTS" --max-tokens "$MAX_TOKENS" --temperature 0
  --top-p 1 --repeat 1 --gsm8k-data-path "$LC_EVAL_OUTPUT_DIR/gsm8k_test.jsonl")
printf 'COMMAND python3 %q' "$RUNTIME/run_eval_to_dir.py"
printf ' %q' "${args[@]}"
printf '\n'
[[ "$dry" != --dry-run ]] || exit 0
mkdir -p "$PWD/evaluations"
mkdir "$LC_EVAL_OUTPUT_DIR"
python3 check_server_dp4.py --url "$BASE_URL" --out "$LC_EVAL_OUTPUT_DIR/server_info.json"
python3 - "$LC_EVAL_OUTPUT_DIR" "$RUNTIME" "$MAX_TOKENS" "${args[@]}" <<'PY'
import hashlib,json,os,shutil,sys
from pathlib import Path
from sglang.utils import download_and_cache_file
root,runtime,max_tokens,*args=sys.argv[1:]
root=Path(root)
info=json.loads((root/'server_info.json').read_text())
assert int(info['context_length']) >= 8192, 'Use context 8192 for this baseline; do not silently truncate evaluation'
source=os.environ.get('GSM8K_DATA_PATH')
if not source:
    source=download_and_cache_file('https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl')
target=root/'gsm8k_test.jsonl'
shutil.copyfile(source,target)
lines=[json.loads(s) for s in target.read_text().splitlines() if s.strip()]
num=int(args[args.index('--num-examples')+1]); shots=int(args[args.index('--num-shots')+1])
assert len(lines)>=num+shots and num>0 and shots>=0
receipt={'dataset_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),
         'dataset_source':str(source), 'dataset_rows':len(lines),
         'few_shot_rows':list(range(shots)), 'eval_rows':[shots,shots+num-1],
         'args':args, 'runtime':json.loads((Path(runtime)/'receipt.json').read_text())}
(root/'evaluation_manifest.json').write_text(json.dumps(receipt,indent=2))
print('EVAL_MANIFEST',root/'evaluation_manifest.json')
PY
python3 "$RUNTIME/run_eval_to_dir.py" "${args[@]}" 2>&1 | tee "$LC_EVAL_OUTPUT_DIR/run_eval.log"
