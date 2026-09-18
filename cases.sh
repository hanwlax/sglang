#!/usr/bin/env bash

set -e -o pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# Use the same checkout as the server, and keep CANN's Python dependencies.
# Do not inject the incompatible kernel checkout from /home.
export PYTHONPATH="${SCRIPT_DIR}/python${PYTHONPATH:+:${PYTHONPATH}}"

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

LOG_ROOT="${CASE_LOG_ROOT:-${SCRIPT_DIR}/../runs/bs44_diagnostics}"
HOST="127.0.0.1"
PORT="30000"
DEFAULT_DATASET="/home/zkk/datasets/ShareGPT_V3_unfiltered_cleaned_split.json"
SHARED_128K_DATASET="/home/hanwlax/datasets/shareGPT/sharegpt_natural_shared_128k_32.json"
SHARED_128K_64_DATASET="/home/hanwlax/datasets/shareGPT/sharegpt_natural_shared_128k_64.json"

CASE_NAMES=(
  "gsm8k"
  "single_curl"
  "8k_1k_bs1"
  "8k_1k_bs32"
  "128k_1k_bs1"
  "128k_1k_99cache_bs1"
  "128k_1k_99cache_bs4"
  "128k_1k_99cache_bs32"
  "gpqa"
  "128k_1k_99cache_bs44"
)

usage() {
  echo "Usage: $0 [case_order]"
  echo
  echo "Available cases:"
  local i
  for i in "${!CASE_NAMES[@]}"; do
    printf '%d. %s\n' "$i" "${CASE_NAMES[$i]}"
  done
}

select_case() {
  local selected="${1:-}"
  if [[ -z "$selected" ]]; then
    echo "please enter case order to test:"
    local i
    for i in "${!CASE_NAMES[@]}"; do
      printf '%d. %s\n' "$i" "${CASE_NAMES[$i]}"
    done
    read -r selected
  fi

  if [[ ! "$selected" =~ ^[0-9]+$ ]] || (( selected >= ${#CASE_NAMES[@]} )); then
    echo "Invalid case order: $selected" >&2
    usage >&2
    return 2
  fi

  SELECTED_CASE="$selected"
}

flush_cache() {
  curl --fail --show-error --max-time 120 --location "http://${HOST}:${PORT}/flush_cache" \
    --header 'Content-Type: application/json'
}

run_random_benchmark() {
  local dataset="$1"
  local concurrency="$2"
  local input_len="$3"
  local output_len="$4"
  local prompts="$5"
  shift 5

  local phase="${input_len}_${output_len}_${concurrency}"
  local input_mode_args=()
  if [[ "$SELECTED_CASE" == 9 && "${CASE_TOKENIZE_PROMPT:-0}" == 1 ]]; then
    input_mode_args+=(--tokenize-prompt)
  fi
  date -u '+BENCH_PHASE_BEGIN %Y-%m-%dT%H:%M:%S.%NZ'
  printf 'BENCH_SHAPE input=%s output=%s concurrency=%s prompts=%s\n' "$input_len" "$output_len" "$concurrency" "$prompts"
  printf 'BENCH_INPUT_MODE tokenize_prompt=%s\n' "${CASE_TOKENIZE_PROMPT:-0}"
  curl -fsS --max-time 20 "http://${HOST}:${PORT}/server_info" > "$RUN_DIR/server_before_${phase}.json"

  python -m sglang.bench_serving \
    --dataset-path "$dataset" \
    --dataset-name random \
    --backend sglang \
    --host "$HOST" \
    --port "$PORT" \
    --max-concurrency "$concurrency" \
    --random-input-len "$input_len" \
    --random-output-len "$output_len" \
    --num-prompts "$prompts" \
    --random-range-ratio 1 \
    "${input_mode_args[@]}" \
    "$@" || return $?
  curl -fsS --max-time 20 "http://${HOST}:${PORT}/server_info" > "$RUN_DIR/server_after_${phase}.json"
  date -u '+BENCH_PHASE_END %Y-%m-%dT%H:%M:%S.%NZ'
}

run_case() {
  case "$SELECTED_CASE" in
    0)
      python3 -m sglang.test.few_shot_gsm8k \
        --num-questions 50 --num-shots 5 --host 0.0.0.0 --port "$PORT" \
        --data-path /home/zkk/gsm8k.jsonl
      ;;
    1)
      curl -s "http://${HOST}:${PORT}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d '{
          "model": "/home/weights/Kimi-K3-w4a8-int-moe",
          "messages": [{"role": "user", "content": "The capital of France is"}],
          "max_tokens": 20,
          "temperature": 0
        }'
      echo
      ;;
    2)
      flush_cache
      run_random_benchmark "$DEFAULT_DATASET" 1 8000 1000 1 \
        --disable-ignore-eos --flush-cache --warmup-request 0 \
        --output-details --output-file "$DETAIL_FILE" \
        --extra-request-body '{"routed_dp_rank": 0}'
      ;;
    3)
      flush_cache
      run_random_benchmark "$DEFAULT_DATASET" 32 8000 1000 32 \
        --disable-ignore-eos --flush-cache --warmup-request 0 \
        --output-details --output-file "$DETAIL_FILE"
      ;;
    4)
      flush_cache
      run_random_benchmark "$DEFAULT_DATASET" 1 128000 1000 1 \
        --seed 42 --disable-ignore-eos --warmup-request 0 --flush-cache \
        --output-details --output-file "$DETAIL_FILE" \
        --extra-request-body '{"routed_dp_rank": 0}'
      ;;
    5)
      flush_cache
      echo "Building 99% prefix cache..."
      run_random_benchmark "$DEFAULT_DATASET" 1 126720 1 1 \
        --seed 42 --warmup-requests 0 \
        --extra-request-body '{"routed_dp_rank": 0}'
      echo "Running benchmark..."
      run_random_benchmark "$DEFAULT_DATASET" 1 128000 1000 1 \
        --seed 42 --warmup-requests 0 --cache-report \
        --extra-request-body '{"routed_dp_rank": 0}' \
        --output-details --output-file "$DETAIL_FILE"
      ;;
    6)
      flush_cache
      echo "Building 99% prefix cache..."
      run_random_benchmark "$DEFAULT_DATASET" 4 126720 1 4 \
        --seed 42 --warmup-requests 0
      echo "Running benchmark..."
      run_random_benchmark "$DEFAULT_DATASET" 4 128000 1000 4 \
        --seed 42 --warmup-requests 0 --cache-report \
        --output-details --output-file "$DETAIL_FILE"
      ;;
    7)
      flush_cache
      echo "Building 99% prefix cache..."
      run_random_benchmark "$SHARED_128K_DATASET" 4 126720 1 4 \
        --seed 42 --warmup-requests 0
      echo "Running benchmark..."
      run_random_benchmark "$SHARED_128K_DATASET" 32 128000 1000 32 \
        --seed 42 --warmup-requests 0 --cache-report \
        --output-details --output-file "$DETAIL_FILE"
      ;;
    8)
      evalscope eval \
        --model /home/weights/Kimi-K3-w4a8-int-moe \
        --api-url "http://${HOST}:${PORT}/v1" \
        --api-key EMPTY \
        --work-dir "$RUN_DIR/detail" \
        --no-timestamp \
        --eval-type openai_api \
        --datasets gpqa_diamond \
        --dataset-args '{
          "gpqa_diamond": {
            "local_path": "/home/hanwlax/datasets/gpqa",
            "subset_list": ["gpqa_diamond"],
            "default_subset": "gpqa_diamond"
          }
        }' \
        --generation-config '{
          "max_tokens": 131072,
          "timeout": 10000,
          "temperature": 1.0,
          "top_p": 0.95,
          "extra_body": {"reasoning_effort": "max"}
        }' \
        --eval-batch-size 32 \
        --seed 42
      ;;
    9)
      flush_cache
      echo "Building 99% prefix cache..."
      run_random_benchmark "$SHARED_128K_64_DATASET" 4 126720 1 4 \
        --seed 42 --warmup-requests 0 --output-details \
        --output-file "$RUN_DIR/warmup.jsonl"
      echo "Running benchmark..."
      run_random_benchmark "$SHARED_128K_64_DATASET" 44 128000 1000 44 \
        --seed 42 --warmup-requests 0 --cache-report \
        --output-details --output-file "$DETAIL_FILE"
      ;;
  esac
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

select_case "${1:-}" || exit $?

date_str=$(date +%Y-%m-%d_%H-%M-%S_%N)
RUN_DIR="${LOG_ROOT}/${CASE_NAMES[$SELECTED_CASE]}_${date_str}"
LOG_FILE="${RUN_DIR}/run.log"
DETAIL_FILE="${RUN_DIR}/detail.jsonl"
mkdir -p "$RUN_DIR" || { echo "Cannot create writable result directory: $RUN_DIR" >&2; exit 1; }
cp -- "$SCRIPT_DIR/cases.sh" "$RUN_DIR/cases.sh"
python -c 'import sglang, sys; print(sys.executable); print(sglang.__file__)' > "$RUN_DIR/client_runtime.txt"

echo "Selected case: ${SELECTED_CASE}. ${CASE_NAMES[$SELECTED_CASE]}"
echo "Run log: $LOG_FILE"
if (( (SELECTED_CASE >= 2 && SELECTED_CASE <= 7) || SELECTED_CASE == 9 )); then
  echo "Detail output: $DETAIL_FILE"
elif (( SELECTED_CASE == 8 )); then
  echo "Detail output: $RUN_DIR/detail"
fi

run_case 2>&1 | tee "$LOG_FILE"
exit "${PIPESTATUS[0]}"
