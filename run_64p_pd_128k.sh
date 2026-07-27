#!/bin/bash

echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
sysctl -w vm.swappiness=10
sysctl -w kernel.numa_balancing=0
sysctl -w kernel.sched_migration_cost_ns=50000
export SGLANG_SET_CPU_AFFINITY=1
export SGLANG_ONE_VISIBLE_DEVICE_PER_PROCESS=1

MODEL_PATH=/home/weights/Kimi-K3-w4a8-int-8cards-quarot-all-0722
export ASCEND_MF_STORE_URL="tcp://192.168.25.213:24567"

unset https_proxy
unset http_proxy
unset HTTPS_PROXY
unset HTTP_PROXY
unset ASCEND_LAUNCH_BLOCKING

source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_SOCKET_IFNAME=enp196s0f0
export GLOO_SOCKET_IFNAME=enp196s0f0
export STREAMS_PER_DEVICE=32

export DEEP_NORMAL_MODE_USE_INT8_QUANT=1

export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=64
export HCCL_BUFFSIZE=2000
export DEEPEP_NORMAL_LONG_SEQ_ROUND=64
export DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS=512

export HCCL_OP_EXPANSION_MODE=AIV
export SGLANG_KDA_ASCENDC_CONV1D=1

export PYTHONPATH=/home/fuyong/codes/sglang/python:$PYTHONPATH

P_IP=('192.168.25.213' '192.168.25.214' '192.168.25.215' '192.168.25.218')
D_IP=('192.168.25.209' '192.168.25.212' '192.168.25.216' '192.168.25.217')

export ASCEND_MF_STORE_URL="tcp://192.168.25.213:24669"

LOCAL_HOST1=`hostname -I|awk -F " " '{print$1}'`
LOCAL_HOST2=`hostname -I|awk -F " " '{print$2}'`
echo "${LOCAL_HOST1}"
echo "${LOCAL_HOST2}"

for i in "${!P_IP[@]}";
do
    if [[ "$LOCAL_HOST1" == "${P_IP[$i]}" || "$LOCAL_HOST2" == "${P_IP[$i]}" ]];
    then
        echo "Prefill -> ${P_IP[$i]}"

        sglang serve \
            --model-loader-extra-config '{"enable_multithread_load": true}' \
            --dist-init-addr ${P_IP[0]}:5000 --nnodes 4 --node-rank $i \
            --disaggregation-mode prefill --disaggregation-transfer-backend ascend \
            --disaggregation-bootstrap-port $((8998+$i)) \
            --model-path $MODEL_PATH \
            --tokenizer-path $MODEL_PATH \
            --trust-remote-code \
            --attention-backend ascend \
            --device npu \
            --quantization modelslim \
            --dtype bfloat16 \
            --tp-size 64 \
            --enable-dp-attention --dp-size 4 --enable-dp-lm-head \
            --mem-fraction-static 0.8 \
            --chunked-prefill-size 8192 \
            --max-running-requests 64 \
            --host 0.0.0.0 \
            --port 30000 \
            --moe-a2a-backend deepep \
            --deepep-mode normal \
            --disable-cuda-graph \
            --load-balance-method round_robin


        exit 1
    fi
done

for i in "${!D_IP[@]}";
do
    if [[ "$LOCAL_HOST1" == "${D_IP[$i]}" || "$LOCAL_HOST2" == "${D_IP[$i]}" ]];
    then
        echo "Decode -> ${D_IP[$i]}"

        export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=64
        export HCCL_BUFFSIZE=2000

        sglang serve \
            --model-loader-extra-config '{"enable_multithread_load": true}' \
            --dist-init-addr ${D_IP[0]}:5000 --nnodes 4 --node-rank $i \
            --disaggregation-mode decode --disaggregation-transfer-backend ascend \
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
            --cuda-graph-bs 16 \
            --max-running-requests 64 \
            --host 0.0.0.0 \
            --port 30000 \
            --moe-a2a-backend deepep \
            --deepep-mode low_latency \
            --prefill-round-robin-balance

        exit 1
    fi
done

exit 1

python -m sglang_router.launch_router \
    --pd-disaggregation \
    --prefill http://192.168.25.213:30000 8998 \
    --policy round_robin \
    --decode http://192.168.25.209:30000 \
    --host 0.0.0.0 --port 6688

curl --location 'http://192.168.25.209:8100/flush_cache' --header 'Content-Type: application/json'
python -m sglang.bench_serving \
    --dataset-path /home/zkk/datasets/ShareGPT_V3_unfiltered_cleaned_split.json \
    --dataset-name random \
    --backend sglang \
    --host 192.168.25.209 \
    --port 6688 \
    --max-concurrency 4 \
    --random-input-len 8000 \
    --random-output-len 1000 \
    --num-prompts 4 \
    --disable-ignore-eos \
    --random-range-ratio 1 \
    --warmup-request 0

python3 -m sglang.bench_serving \
    --dataset-name generated-shared-prefix \
    --backend sglang --host 192.168.25.209 \
    --port 6688 \
    --max-concurrency 4 \
    --gsp-num-groups 1 \
    --gsp-prompts-per-group 1 \
    --gsp-system-prompt-len 127620 \
    --gsp-question-len 1280 \
    --gsp-output-len 1000 \
    --warmup-requests 4



# 1st
python3 -m sglang.bench_serving \
    --dataset-name generated-shared-prefix \
    --backend sglang \
    --host 192.168.25.209 \
    --port 6688 \
    --max-concurrency 4 \
    --gsp-num-groups 1 \
    --gsp-prompts-per-group 1 \
    --gsp-system-prompt-len 128329 \
    --gsp-question-len 1 \
    --gsp-output-len 1 \
    --warmup-requests 0 \
    --num-prompts 4 \
    --seed 89 \
    --cache-report \
    --output-details \
    --output-file /home/zcl/kda_ttft_cache_diag_2226.jsonl

# 2nd
python3 -m sglang.bench_serving \
    --dataset-name generated-shared-prefix \
    --backend sglang \
    --host 192.168.25.209 \
    --port 6688 \
    --max-concurrency 4 \
    --gsp-num-groups 1 \
    --gsp-prompts-per-group 1 \
    --gsp-system-prompt-len 128329 \
    --gsp-question-len 1283 \
    --gsp-output-len 1024 \
    --warmup-requests 0 \
    --num-prompts 4 \
    --seed 89 \
    --cache-report \
    --output-details \
    --output-file /home/zcl/kda_ttft_cache_diag_2257.jsonl