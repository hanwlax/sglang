#!/bin/bash

echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
sysctl -w vm.swappiness=10
sysctl -w kernel.numa_balancing=0
sysctl -w kernel.sched_migration_cost_ns=50000
export SGLANG_SET_CPU_AFFINITY=1
export SGLANG_ONE_VISIBLE_DEVICE_PER_PROCESS=1

MODEL_PATH=/mnt/paas/weights/Kimi-K2.7-Code-w4a8
DRAFT_MODEL_PATH=/data/weights/Kimi-K2.7-Code-DFlash

unset https_proxy
unset http_proxy
unset HTTPS_PROXY
unset HTTP_PROXY
unset ASCEND_LAUNCH_BLOCKING

source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

export HCCL_SOCKET_IFNAME=lo
export GLOO_SOCKET_IFNAME=lo
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export STREAMS_PER_DEVICE=32
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=600
export SGLANG_ENABLE_SPEC_V2=1
export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1
export DEEP_NORMAL_MODE_USE_INT8_QUANT=1
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=64
export HCCL_BUFFSIZE=1200
export HCCL_OP_EXPANSION_MODE=AIV

# export SGLANG_NPU_USE_MLAPO=1
# export SGLANG_NPU_USE_MULTI_STREAM=1

export PYTHONPATH=/home/hanwlax/workspace/sglang/python:$PYTHONPATH

sglang serve \
    --model-path $MODEL_PATH \
    --trust-remote-code \
    --attention-backend ascend \
    --device npu \
    --quantization modelslim \
    --dtype bfloat16 \
    --tp-size 16 \
    --mem-fraction-static 0.785 \
    --max-running-requests 16 \
    --prefill-max-requests 1 \
    --chunked-prefill-size 8192 \
    --enable-multimodal \
    --mm-attention-backend ascend_attn \
    --sampling-backend ascend \
    --moe-a2a-backend deepep \
    --deepep-mode auto \
    --cuda-graph-bs-decode 16 \
    --model-loader-extra-config '{"enable_multithread_load": true}' \
    --speculative-algorithm DFLASH \
    --speculative-draft-model-path $DRAFT_MODEL_PATH \
    --speculative-num-steps 4 \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 8 \
    --speculative-draft-model-quantization unquant \
    --device npu \
    --host 0.0.0.0 \
    --port 30000 2>&1 | tee "/home/hanwlax/workspace/progress/kimi_k2.7/final/logs/gitee_sh_${date --iso-8601=ns}.log"

exit 1

curl -s -X POST "http://127.0.0.1:8880/flush_cache?timeout=30"

python3 -m sglang.bench_serving \
    --dataset-name generated-shared-prefix \
    --backend sglang --host 0.0.0.0 \
    --port 30000 \
    --max-concurrency 1 \
    --gsp-num-groups 1 \
    --gsp-prompts-per-group 1 \
    --gsp-system-prompt-len 115200 \
    --gsp-question-len 12800 \
    --gsp-output-len 1000 \
    --warmup-requests 1
