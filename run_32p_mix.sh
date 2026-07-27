#!/bin/bash

echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
sysctl -w vm.swappiness=10
sysctl -w kernel.numa_balancing=0
sysctl -w kernel.sched_migration_cost_ns=50000
export SGLANG_SET_CPU_AFFINITY=1
export SGLANG_ONE_VISIBLE_DEVICE_PER_PROCESS=1

# MODEL_PATH=/home/weights/Kimi-K3-int4
MODEL_PATH=/home/weights/Kimi-K3-w4a8-int-8cards-quarot-all-0722
export SGLANG_NCCL_ALL_GATHER_IN_OVERLAP_SCHEDULER_SYNC_BATCH=1
#export SGLANG_NPU_PROFILING=1

unset https_proxy
unset http_proxy
unset HTTPS_PROXY
unset HTTP_PROXY
unset ASCEND_LAUNCH_BLOCKING

#source /usr/local/Ascend/ascend-toolkit/set_env.sh
#source /usr/local/Ascend/nnal/atb/set_env.sh

source /home/z30071866/cann9.1.0/cann/set_env.sh
export ASCEND_CUSTOM_OPP_PATH=/home/z30071866/cann9.1.0/cann-9.1.0-beta.3/opp/vendors/custom_transformer

export LD_LIBRARY_PATH=${ASCEND_HOME_PATH}/opp/vendors/custom_transformer/op_api/lib/:${LD_LIBRARY_PATH}

export SGLANG_NPU_FUSED_MOE_MODE=3
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=1024


export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_SOCKET_IFNAME=lo
export GLOO_SOCKET_IFNAME=lo
export STREAMS_PER_DEVICE=32

export DEEP_NORMAL_MODE_USE_INT8_QUANT=1

#export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=64
export HCCL_BUFFSIZE=1200
export HCCL_OP_EXPANSION_MODE=AIV
#export MOE_ENABLE_TOPK_NEG_ONE=1
## export SGLANG_KDA_ASCENDC_CONV1D=1
# export PYTHONPATH=/home/zkk/sglang/python:$PYTHONPATH
export PYTHONPATH=/home/l00890003/codes/sglang/python:$PYTHONPATH

export DEEPEP_NORMAL_LONG_SEQ_ROUND=64
export DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS=512
export DEEPEP_NORMAL_COMBINE_ENABLE_LONG_SEQ=1
# export MOE_ENABLE_TOPK_NEG_ONE=1


# export HCCL_EXEC_TIMEOUT=1200

#D_IP=('192.168.25.209' '192.168.25.212' '192.168.25.216' '192.168.25.217')
D_IP=('192.168.25.213' '192.168.25.214' '192.168.25.215' '192.168.25.218')
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

        sglang serve \
            --model-loader-extra-config '{"enable_multithread_load": true}' \
            --dist-init-addr 192.168.25.213:5000 --nnodes 4 --node-rank $i \
            --model-path $MODEL_PATH \
            --tokenizer-path $MODEL_PATH \
            --trust-remote-code \
            --attention-backend ascend \
            --device npu \
            --quantization modelslim \
            --dtype bfloat16 \
            --tp-size 64 \
            --enable-dp-attention --dp-size 4 --enable-dp-lm-head \
            --mem-fraction-static 0.75 \
            --chunked-prefill-size 8192 \
            --cuda-graph-bs 16 \
            --max-total-tokens 1048576 \
            --max-running-requests 4 \
            --host 0.0.0.0 \
            --port 30000 \
            --skip-server-warmup \
	        --moe-a2a-backend ascend_fuseep \
    	    --deepep-mode auto \

        exit 1
    fi
done

exit 1

python -m sglang.bench_serving \
  --dataset-path /home/zkk/datasets/ShareGPT_V3_unfiltered_cleaned_split.json \
  --dataset-name random \
  --backend sglang \
  --host 0.0.0.0 \
  --port 30000 \
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
    --port 30000 \
    --max-concurrency 1 \
    --gsp-num-groups 1 \
    --gsp-prompts-per-group 1 \
    --gsp-system-prompt-len 127620 \
    --gsp-question-len 1280 \
    --gsp-output-len 1000 \
    --warmup-requests 4