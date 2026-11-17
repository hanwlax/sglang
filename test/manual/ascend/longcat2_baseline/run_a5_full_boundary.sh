#!/usr/bin/env bash
# Green-zone full38 FP8 capture on .105/.113/.121/.122, ranks 0/1/2/3.
set -eo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
base=$PWD
node_rank=${1:?Usage: bash run_a5_full_boundary.sh NODE_RANK RUN_ID [trace] [--dry-run]}
run_id=${2:?Use the SAME new RUN_ID on all four hosts}
[[ "$node_rank" =~ ^[0-3]$ && "$run_id" =~ ^[a-zA-Z0-9_-]+$ ]] || exit 2
mode=${3:-trace}
[[ "$mode" == trace ]] || exit 2
dry=${4:-}
[[ -z "$dry" || "$dry" == --dry-run ]] || exit 2
NODE_IPS=(141.61.45.105 141.61.45.113 141.61.45.121 141.61.45.122)
MODEL_PATH=${MODEL_PATH:-/mnt/share/weight/LongCat-2.0-FP8}
RUNTIME=${RUNTIME:-$base/runtime_full_a5}
HTTP_PORT=${HTTP_PORT:-10086}
DIST_PORT=${DIST_PORT:-5017}
CHUNKED_PREFILL_SIZE=${CHUNKED_PREFILL_SIZE:-2048}
MAX_TOTAL_TOKENS=${MAX_TOTAL_TOKENS:-8192}
MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC:-0.92}
CONTEXT_LENGTH=${CONTEXT_LENGTH:-8192}
CAPTURE_ROOT=${CAPTURE_ROOT:-$base}
RUN_DIR="$CAPTURE_ROOT/results/$run_id"
export LC_INTERNAL_LAYERS=${LC_INTERNAL_LAYERS:-0,7,37}
export LC_INTERNAL_RANKS=${LC_INTERNAL_RANKS:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}
export LONGCAT_FIXED_DECODE_DIR="$RUN_DIR/dumps"
[[ "$CHUNKED_PREFILL_SIZE" -ge 2048 ]] || { echo 'Fixed-prefix trace requires global chunk >=2048' >&2; exit 2; }
# TP32 uses eight logical A5 devices on each host.
export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
python3 - "$MODEL_PATH" "$RUNTIME" "$ASCEND_RT_VISIBLE_DEVICES" "$mode" <<'PY'
import json,sys
from pathlib import Path
m,r,devices,mode=sys.argv[1:]
c=json.loads((Path(m)/'config.json').read_text())
assert c.get('num_hidden_layers', c.get('num_layers'))==38, 'Require full 38-layer config'
assert len(devices.split(','))==len(set(devices.split(',')))==8, 'TP32/four hosts requires 8 logical devices per host'
assert (Path(r)/'receipt.json').is_file(), 'Run prepare_runtime_dp4.py first'
receipt=json.loads((Path(r)/'receipt.json').read_text())
assert receipt['mode']==mode and receipt['topology']=={'tp_size':32,'attn_dp_size':2,'moe_ep_size':32}, 'Wrong runtime mode/topology'
assert receipt.get('capture')=='full_valid_boundaries', 'Require the new full-boundary runtime'
assert c.get('quantization_config',{}).get('quant_method')=='fp8', 'Require FP8 checkpoint'
idx=json.loads((Path(m)/'model.safetensors.index.json').read_text())['weight_map']
assert all(any(k.startswith(f'model.layers.{i}.') for k in idx) for i in range(38))
assert all((Path(m)/s).is_file() for s in set(idx.values())), 'Missing checkpoint shards'
print('BASELINE_MODEL',json.dumps({'path':m,'layers':38,'quantization':'fp8','attention_disabled_quant':c.get('disable_quant_module')}))
PY
args=(--model-path "$MODEL_PATH" --tokenizer-path "$MODEL_PATH"
  --served-model-name LongCat-2.0 --trust-remote-code
  --model-loader-extra-config '{"enable_multithread_load": false}'
  --device npu --attention-backend ascend --quantization fp8
  --dtype bfloat16 --kv-cache-dtype bf16 --nnodes 4 --node-rank "$node_rank"
  --dist-init-addr "${NODE_IPS[0]}:$DIST_PORT" --tp-size 32 --pp-size 1
  --attn-dp-size 2 --enable-dp-lm-head --ep-size 32 --moe-a2a-backend deepep
  --deepep-mode auto --disaggregation-mode null --page-size 128
  --mem-fraction-static "$MEM_FRACTION_STATIC" --max-total-tokens "$MAX_TOTAL_TOKENS"
  --context-length "$CONTEXT_LENGTH" --chunked-prefill-size "$CHUNKED_PREFILL_SIZE"
  --max-running-requests 2 --cuda-graph-backend-prefill disabled
  --cuda-graph-backend-decode disabled --disable-radix-cache --skip-server-warmup
  --watchdog-timeout 9000 --host 0.0.0.0 --port "$HTTP_PORT")
printf 'COMMAND python3 -m sglang.launch_server'
printf ' %q' "${args[@]}"
printf '\n'
[[ "$dry" != --dry-run ]] || exit 0
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY PYTHONPATH
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
export PYTHONPATH="$RUNTIME/python${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export TORCH_NPU_DEVICE_CAPABILITY=${TORCH_NPU_DEVICE_CAPABILITY:-8.0}
export HCCL_SOCKET_IFNAME=${HCCL_SOCKET_IFNAME:-enp34s0f1}
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-data0.3001}
export HCCL_OP_EXPANSION_MODE=${HCCL_OP_EXPANSION_MODE:-AI_CPU}
export HCCL_CONNECT_TIMEOUT=${HCCL_CONNECT_TIMEOUT:-3000}
export HCCL_BUFFSIZE=${HCCL_BUFFSIZE:-200}
export DEEPEP_HCCL_BUFFSIZE=${DEEPEP_HCCL_BUFFSIZE:-3300}
export DEEPEP_HYBRID_DEPLOYMENT=1
export MOE_ENABLE_TOPK_NEG_ONE=1
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=32
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export SGLANG_SET_CPU_AFFINITY=1
export SGLANG_ONE_VISIBLE_DEVICE_PER_PROCESS=1
export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=${SGLANG_ENABLE_OVERLAP_PLAN_STREAM:-1}
export STREAMS_PER_DEVICE=32
export ASCEND_USE_FIA=1
export SGLANG_NPU_USE_MLAPO=0
export SGLANG_NPU_ARCH35_REQUANT_BLOCK_FP8=0
export SGLANG_ENABLE_DELAY_SAMPLE=0
export SGLANG_DEBUG_LONGCAT_ACTIVATIONS=0
unset ASCEND_LAUNCH_BLOCKING
mkdir -p "$RUN_DIR/dumps"
log="$RUN_DIR/dumps/serve_rank${node_rank}.log"
# Never overwrite a previous run's evidence.
(set -C; : > "$log") || { echo 'Choose a new RUN_ID' >&2; exit 1; }
{
  printf 'BASELINE_MODE %s runtime=%s\n' "$mode" "$RUNTIME"
  printf 'BASELINE_CONFIG node=%s layers=38 tp=32 attn_dp=2 ep=32 deepep=auto graph=off nextn=off hccl=%s devices=%s chunk=%s kv=%s mem=%s\n' \
    "$node_rank" "$HCCL_OP_EXPANSION_MODE" "$ASCEND_RT_VISIBLE_DEVICES" "$CHUNKED_PREFILL_SIZE" "$MAX_TOTAL_TOKENS" "$MEM_FRACTION_STATIC"
  python3 - <<'PY'
import hashlib,importlib.metadata,json,os,socket
from pathlib import Path
print('BASELINE_HOST',socket.gethostname())
print('BASELINE_NETWORK',socket.if_nameindex())
for key in ('HCCL_SOCKET_IFNAME','GLOO_SOCKET_IFNAME'):
    assert os.environ[key] in dict((n,i) for i,n in socket.if_nameindex()), f'Missing interface {key}'
print('BASELINE_ENV',json.dumps({k:v for k,v in os.environ.items() if k.startswith(('SGLANG_','HCCL_','DEEPEP_','MOE_','LC_INTERNAL_','ASCEND_RT_VISIBLE'))},sort_keys=True))
for p in ('torch','torch-npu','transformers','deep-ep'):
    try: print('BASELINE_PACKAGE',p,importlib.metadata.version(p))
    except importlib.metadata.PackageNotFoundError: print('BASELINE_PACKAGE',p,'metadata unavailable')
PY
  python3 - "$MODEL_PATH" "$RUNTIME" <<'PY'
import hashlib,json,sys
from pathlib import Path
m,r=map(Path,sys.argv[1:])
for p in (m/'config.json',m/'model.safetensors.index.json',r/'receipt.json'):
    print('BASELINE_FILE',str(p),hashlib.sha256(p.read_bytes()).hexdigest())
print('BASELINE_RUNTIME', (r/'receipt.json').read_text())
PY
  python3 - "$RUN_DIR/launch_rank${node_rank}.json" "$mode" "$RUNTIME" "${args[@]}" <<'PY'
import datetime,json,os,sys
from pathlib import Path
p,mode,runtime,*args=sys.argv[1:]
with Path(p).open('x') as f:
    json.dump({'mode':mode,'runtime':runtime,'args':args,
               'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
               'env':{k:v for k,v in os.environ.items() if k.startswith(('SGLANG_','HCCL_','DEEPEP_','MOE_','LC_','ASCEND_RT_VISIBLE'))}},f,indent=2)
PY
  cp "$RUNTIME/receipt.json" "$RUN_DIR/runtime_rank${node_rank}.json"
  python3 -m sglang.launch_server "${args[@]}"
} 2>&1 | tee -a "$log"
