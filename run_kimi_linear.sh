#!/bin/bash

unset http_proxy
unset https_proxy
unset HTTP_PROXY
unset HTTPS_PROXY

export PYTHONPATH="/home/rjw/sglang-k3/python:$PYTHONPATH"
export SGLANG_NPU_USE_MULTI_STREAM=1
export USE_VLLM_CUSTOM_ALLREDUCE=1

# export SGLANG_KDA_TORCH_NATIVE_DECODE=1
export SGLANG_KDA_TORCH_NATIVE_EXTEND=1
# export SGLANG_KDA_CONV_TORCH_NATIVE=1

# export SGLANG_KDA_DEBUG=1


python -m sglang.launch_server \
--model-path /home/weights/Kimi-Linear-48B-A3B-Instruct \
--trust-remote-code \
--mem-fraction-static 0.8 \
--host 127.0.0.1 \
--port 8900 \
--tp-size 2 \
--device npu \
--attention-backend ascend \
--disable-radix-cache \
--watchdog-timeout 9000 \
--disable-cuda-graph \
--max-running-requests 16 2>&1 | tee "logs/kimi-linear-$(date --iso-8601=ns).log"




# curl --location 'http://0.0.0.0:8900/generate' --header 'Content-Type: application/json' --data '{
#     "text": "The capital of China is ",
#     "sampling_params": {
#         "temperature": 0,
#         "max_new_tokens": 100
#     }
# }'
