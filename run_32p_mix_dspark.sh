#!/bin/bash

echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
sysctl -w vm.swappiness=10
sysctl -w kernel.numa_balancing=0
sysctl -w kernel.sched_migration_cost_ns=50000
export SGLANG_SET_CPU_AFFINITY=1
export SGLANG_ONE_VISIBLE_DEVICE_PER_PROCESS=1
export SGLANG_NPU_USE_TRITON_PREFIX_KV_CACHE_STORE=1
# export TRITON_ALL_BLOCKS_PARALLEL=1
# export SGLANG_NPU_PROFILING=1
# export SGLANG_DSPARK_DEBUG_PREFIX_CACHE=1
# export SGLANG_DEBUG_DEFERRED_MAMBA_COW_DIFF_MAX_EVENTS=16
# export SGLANG_DEBUG_DEFERRED_MAMBA_COW_DIFF=1
MODEL_PATH=/home/weights/Kimi-K3-w4a8-int-moe
DRAFT_MODEL_PATH=/home/weights/Kimi-K3-DSpark

unset https_proxy
unset http_proxy
unset HTTPS_PROXY
unset HTTP_PROXY
unset ASCEND_LAUNCH_BLOCKING

source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

unset ASCEND_RT_VISIBLE_DEVICES

export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_SOCKET_IFNAME=lo
export GLOO_SOCKET_IFNAME=lo
export STREAMS_PER_DEVICE=32

export DEEP_NORMAL_MODE_USE_INT8_QUANT=1

export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=128
export HCCL_BUFFSIZE=200
export DEEPEP_HCCL_BUFFSIZE=1800
export DEEPEP_NORMAL_LONG_SEQ_ROUND=64
export DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS=512
# export DEEPEP_NORMAL_COMBINE_ENABLE_LONG_SEQ=1

export HCCL_OP_EXPANSION_MODE=AIV

SGLANG_REPO_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export PYTHONPATH="${SGLANG_REPO_DIR}/python:${PYTHONPATH:-}"
# export SGLANG_DSPARK_DEBUG_TRACE=${SGLANG_DSPARK_DEBUG_TRACE:-1}

D_IP=('192.168.25.209' '192.168.25.212' '192.168.25.216' '192.168.25.217')
if [[ "${ENABLE_NPU_GRAPH:-1}" == "1" ]]; then
    GRAPH_ARGS=(--cuda-graph-bs 2 4 16)
else
    GRAPH_ARGS=(--disable-cuda-graph)
fi
LOCAL_HOST1=`hostname -I|awk -F " " '{print$1}'`
LOCAL_HOST2=`hostname -I|awk -F " " '{print$2}'`
echo "${LOCAL_HOST1}"
echo "${LOCAL_HOST2}"

for i in "${!D_IP[@]}";
do
    if [[ "$LOCAL_HOST1" == "${D_IP[$i]}" || "$LOCAL_HOST2" == "${D_IP[$i]}" ]];
    then
        echo "Decode -> ${D_IP[$i]}"

        export HCCL_SOCKET_IFNAME=enp196s0f0
        export GLOO_SOCKET_IFNAME=enp196s0f0
        export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1
        export SGLANG_ENABLE_SPEC_V2=1
        export SGLANG_RAGGED_VERIFY_MODE="${SGLANG_RAGGED_VERIFY_MODE:-static}"

        sglang serve \
            --model-loader-extra-config '{"enable_multithread_load": true}' \
            --dist-init-addr 192.168.25.209:5000 --nnodes 4 --node-rank $i \
            --model-path $MODEL_PATH \
            --tokenizer-path $MODEL_PATH \
            --trust-remote-code \
            --attention-backend ascend \
            --device npu \
            --quantization modelslim \
            --dtype bfloat16 \
            --tp-size 64 \
	          --enable-dp-attention --dp-size 4 --enable-dp-lm-head \
            --mem-fraction-static 0.78 \
            --chunked-prefill-size 8192 \
            "${GRAPH_ARGS[@]}" \
            --reasoning-parser kimi_k3 \
            --max-running-requests 64 \
            --host 0.0.0.0 \
            --port 30000 \
	          --moe-a2a-backend deepep \
            --deepep-mode auto \
            --speculative-algorithm DSPARK \
            --speculative-draft-model-path "$DRAFT_MODEL_PATH" \
            --speculative-dspark-block-size 7 \
            --speculative-draft-attention-backend ascend \
            --speculative-eagle-topk 1 \
            --speculative-draft-model-quantization unquant \
            --watchdog-timeout 9000  2>&1 | tee "logs/run_32p_mix_$(date +%Y-%m-%d_%H-%M-%S).log"
        exit 1
    fi
done

exit 1

# spec options
            --speculative-algorithm DSPARK \
            --speculative-draft-model-path "$DRAFT_MODEL_PATH" \
            --speculative-dspark-block-size 7 \
            --speculative-draft-attention-backend ascend \
            --speculative-eagle-topk 1 \
            --speculative-draft-model-quantization unquant \

python3 -m sglang.test.few_shot_gsm8k --num-questions 50 --num-shots 5 --host 0.0.0.0 --port 30000 --data-path /home/zkk/gsm8k.jsonl

curl --location 'http://0.0.0.0:30000/flush_cache' --header 'Content-Type: application/json'
python -m sglang.bench_serving \
  --dataset-path /home/zkk/datasets/ShareGPT_V3_unfiltered_cleaned_split.json \
  --dataset-name random \
  --backend sglang \
  --host 0.0.0.0 \
  --port 30000 \
  --max-concurrency 16 \
  --random-input-len 8000 \
  --random-output-len 1000 \
  --num-prompts 16 \
  --disable-ignore-eos \
  --random-range-ratio 1 \
  --warmup-request 0


curl -s http://127.0.0.1:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/home/weights/Kimi-K3-w4a8-int-moe",
    "messages": [{"role": "user", "content": "The capital of France is"}],
    "max_tokens": 20,
    "temperature": 0
  }'

# 8k_1k_bs1
curl --location 'http://0.0.0.0:30000/flush_cache' --header 'Content-Type: application/json'
python -m sglang.bench_serving \
  --dataset-path /home/zkk/datasets/ShareGPT_V3_unfiltered_cleaned_split.json \
  --dataset-name random \
  --backend sglang \
  --host 0.0.0.0 \
  --port 30000 \
  --max-concurrency 1 \
  --random-input-len 8000 \
  --random-output-len 1000 \
  --num-prompts 1 \
  --disable-ignore-eos \
  --random-range-ratio 1 \
  --warmup-request 0 2>&1 | tee "logs/8k_1k_bs1_$(date +%Y-%m-%d_%H-%M-%S).log"

# 8k_1k_bs16
curl --location 'http://0.0.0.0:30000/flush_cache' --header 'Content-Type: application/json'
python -m sglang.bench_serving \
  --dataset-path /home/zkk/datasets/ShareGPT_V3_unfiltered_cleaned_split.json \
  --dataset-name random \
  --backend sglang \
  --host 0.0.0.0 \
  --port 30000 \
  --max-concurrency 16 \
  --random-input-len 8000 \
  --random-output-len 1000 \
  --num-prompts 16 \
  --disable-ignore-eos \
  --random-range-ratio 1 \
  --warmup-request 0

# 128k_1k_bs1
curl --location 'http://0.0.0.0:30000/flush_cache' --header 'Content-Type: application/json'
python -m sglang.bench_serving \
  --dataset-path /home/zkk/datasets/ShareGPT_V3_unfiltered_cleaned_split.json \
  --dataset-name random \
  --backend sglang \
  --host 0.0.0.0 \
  --port 30000 \
  --max-concurrency 1 \
  --random-input-len 128000 \
  --random-output-len 1000 \
  --num-prompts 1 \
  --disable-ignore-eos \
  --random-range-ratio 1 \
  --warmup-request 1 \
  --flush-cache

# 128k_1k_99cache_bs1
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
  --warmup-requests 1 \
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
  --warmup-requests 1 \
  --output-details \
  --output-file random_128k.jsonl \
  --extra-request-body '{"routed_dp_rank": 0}'

# 128k_1k_99cache_bs4
curl --location 'http://0.0.0.0:30000/flush_cache' --header 'Content-Type: application/json'
python3 -m sglang.bench_serving \
    --dataset-name generated-shared-prefix \
    --backend sglang --host 192.168.25.209 \
    --port 30000 \
    --max-concurrency 1 \
    --gsp-num-groups 1 \
    --gsp-prompts-per-group 4 \
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
    --gsp-prompts-per-group 4 \
    --gsp-system-prompt-len 127620 \
    --gsp-question-len 1280 \
    --gsp-output-len 1000 \
    --warmup-requests 0 \
    --seed 1 \
    --extra-request-body '{"routed_dp_rank": 0}'