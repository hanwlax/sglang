# hot
curl --location 'http://127.0.0.1:30000/flush_cache' --header 'Content-Type: application/json'
python3 -m sglang.bench_serving \
    --dataset-name generated-shared-prefix \
    --backend sglang --host 192.168.25.209 \
    --port 30000 \
    --max-concurrency 1 \
    --gsp-num-groups 1 \
    --gsp-prompts-per-group 1 \
    --gsp-system-prompt-len 127620 \
    --gsp-question-len 0 \
    --gsp-output-len 1 \
    --warmup-requests 0 \
    --seed 1 \
    --extra-request-body '{"routed_dp_rank": 0}'

python3 -m sglang.bench_serving \
    --dataset-name generated-shared-prefix \
    --backend sglang --host 192.168.25.209 \
    --port 30000 \
    --max-concurrency 1 \
    --gsp-num-groups 1 \
    --gsp-prompts-per-group 1 \
    --gsp-system-prompt-len 126720 \
    --gsp-question-len 1280 \
    --gsp-output-len 1000 \
    --warmup-requests 0 \
    --seed 1 \
    --extra-request-body '{"routed_dp_rank": 0}'

#hot 2
curl --location 'http://127.0.0.1:30000/flush_cache' --header 'Content-Type: application/json'
python -m sglang.bench_serving \
  --dataset-path /home/zkk/datasets/ShareGPT_V3_unfiltered_cleaned_split.json \
  --dataset-name random \
  --backend sglang \
  --host 127.0.0.1 \
  --port 30000 \
  --max-concurrency 1 \
  --random-input-len 126720 \
  --random-output-len 1 \
  --num-prompts 1 \
  --seed 1 \
  --random-range-ratio 1 \
  --warmup-requests 0 \
  --output-details \
  --output-file random_128k.jsonl \
  --extra-request-body '{"routed_dp_rank": 0}'

python -m sglang.bench_serving \
  --dataset-path /home/zkk/datasets/ShareGPT_V3_unfiltered_cleaned_split.json \
  --dataset-name random \
  --backend sglang \
  --host 127.0.0.1 \
  --port 30000 \
  --max-concurrency 1 \
  --random-input-len 128000 \
  --random-output-len 1000 \
  --num-prompts 1 \
  --seed 1 \
  --random-range-ratio 1 \
  --warmup-requests 0 \
  --output-details \
  --output-file random_128k.jsonl \
  --extra-request-body '{"routed_dp_rank": 0}'

# cold 1
curl --location 'http://127.0.0.1:30000/flush_cache' --header 'Content-Type: application/json'
python3 -m sglang.bench_serving \
    --dataset-name generated-shared-prefix \
    --backend sglang --host 192.168.25.209 \
    --port 30000 \
    --max-concurrency 1 \
    --gsp-num-groups 1 \
    --gsp-prompts-per-group 1 \
    --gsp-system-prompt-len 126720 \
    --gsp-question-len 1280 \
    --gsp-output-len 1000 \
    --warmup-requests 0 \
    --seed 1 \
    --extra-request-body '{"routed_dp_rank": 0}'

# cold 2
curl --location 'http://127.0.0.1:30000/flush_cache' --header 'Content-Type: application/json'
python -m sglang.bench_serving \
  --dataset-path /home/zkk/datasets/ShareGPT_V3_unfiltered_cleaned_split.json \
  --dataset-name random \
  --backend sglang \
  --host 127.0.0.1 \
  --port 30000 \
  --max-concurrency 1 \
  --random-input-len 128000 \
  --random-output-len 1000 \
  --num-prompts 1 \
  --seed 1 \
  --random-range-ratio 1 \
  --warmup-requests 0 \
  --output-details \
  --output-file random_128k.jsonl \
  --extra-request-body '{"routed_dp_rank": 0}'

python -m sglang.bench_serving \
  --dataset-path /home/zkk/datasets/ShareGPT_V3_unfiltered_cleaned_split.json \
  --dataset-name random \
  --backend sglang \
  --host 127.0.0.1 \
  --port 30000 \
  --max-concurrency 1 \
  --random-input-len 225000 \
  --random-output-len 1 \
  --num-prompts 1 \
  --seed 40 \
  --random-range-ratio 1 \
  --warmup-requests 0 \
  --output-details \
  --output-file random_128k.jsonl \
  --extra-request-body '{"routed_dp_rank": 0}'




#  构造缓存 seed 1
python -m sglang.bench_serving \
  --dataset-path /home/zkk/datasets/ShareGPT_V3_unfiltered_cleaned_split.json \
  --dataset-name random \
  --backend sglang \
  --host 127.0.0.1 \
  --port 30000 \
  --max-concurrency 1 \
  --random-input-len 126720 \
  --random-output-len 1 \
  --num-prompts 1 \
  --seed 1 \
  --random-range-ratio 1 \
  --warmup-requests 0 \
  --output-details \
  --output-file random_128k_cache.jsonl \
  --extra-request-body '{"routed_dp_rank": 0}'

# 测试 l1 cache
python -m sglang.bench_serving \
  --dataset-path /home/zkk/datasets/ShareGPT_V3_unfiltered_cleaned_split.json \
  --dataset-name random \
  --backend sglang \
  --host 127.0.0.1 \
  --port 30000 \
  --max-concurrency 1 \
  --random-input-len 128000 \
  --random-output-len 1000 \
  --num-prompts 1 \
  --seed 1 \
  --random-range-ratio 1 \
  --warmup-requests 0 \
  --output-details \
  --output-file random_128k_request.jsonl \
  --cache-report \
  --extra-request-body '{"routed_dp_rank": 0}'

# 驱逐 l1 cache seed 11
python3 -m sglang.bench_serving \
    --dataset-name generated-shared-prefix \
    --backend sglang --host 127.0.0.1 \
    --port 30000 \
    --max-concurrency 1 \
    --gsp-num-groups 1 \
    --gsp-prompts-per-group 1 \
    --gsp-system-prompt-len 260000 \
    --gsp-question-len 0 \
    --gsp-output-len 1 \
    --warmup-requests 0 \
    --seed 11 \
    --output-details \
    --output-file random_230k.jsonl \
    --extra-request-body '{"routed_dp_rank": 0}'

# 测试 l2 cache
python -m sglang.bench_serving \
  --dataset-path /home/zkk/datasets/ShareGPT_V3_unfiltered_cleaned_split.json \
  --dataset-name random \
  --backend sglang \
  --host 127.0.0.1 \
  --port 30000 \
  --max-concurrency 1 \
  --random-input-len 128000 \
  --random-output-len 1000 \
  --num-prompts 1 \
  --seed 1 \
  --random-range-ratio 1 \
  --warmup-requests 0 \
  --output-details \
  --cache-report \
  --output-file random_128k_request_l2.jsonl \
  --extra-request-body '{"routed_dp_rank": 0}'