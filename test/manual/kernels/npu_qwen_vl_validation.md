# Validate Qwen2.5-VL NPU prefill GMM on Ascend 950PR / 950DT

Use a working SGLang NPU environment with the CANN, driver, torch and torch_npu
packages appropriate for the target device. These commands do not install the
A3 image's packages onto a 950 system. The validation uses one card for operator
tests and two free cards for the original VLM suite.

The change selects GMM for unquantized BF16 Qwen2.5-VL EXTEND/SPLIT_PREFILL,
without checking a device-name prefix. Decode and other modes, quantized/custom
methods, adapter wrappers, bias/gather, non-ND storage, unsupported dtype and
layout retain the original linear path. This does not change every model's NPU
linear implementation.

## 1. Check out the exact implementation and select the devices

Run these commands inside the prepared NPU container. Change `NPU_PAIR` to two
free device IDs as displayed inside that container. Keep the same shell for the
following sections.

```bash
set -euo pipefail
git clone --filter=blob:none --single-branch \
  --branch codex/npu-qwen-vl-gmm \
  https://github.com/hanwlax/sglang.git sglang-qwen-gmm
cd sglang-qwen-gmm
export VLM_REPO="$PWD"
export PYTHONPATH="$VLM_REPO/python${PYTHONPATH:+:$PYTHONPATH}"
export NPU_PAIR=0,1
export NPU_SINGLE="${NPU_PAIR%%,*}"
export ASCEND_GLOBAL_LOG_LEVEL=3
export SGLANG_USE_MODELSCOPE=true
export OMP_NUM_THREADS=8
export VLM_RESULTS="$VLM_REPO/validation-results/$(date +%Y%m%d-%H%M%S)"
mkdir -p "$VLM_RESULTS"
git rev-parse HEAD | tee "$VLM_RESULTS/commit.txt"
npu-smi info | tee "$VLM_RESULTS/npu-before.txt"

ASCEND_RT_VISIBLE_DEVICES="$NPU_SINGLE" python - <<'PY' | tee "$VLM_RESULTS/environment.txt"
import os
from pathlib import Path
import torch, torch_npu, sglang
root = Path(os.environ['VLM_REPO']).resolve()
assert Path(sglang.__file__).resolve().is_relative_to(root / 'python')
print('sglang:', sglang.__file__)
print('torch:', torch.__version__, 'torch_npu:', torch_npu.__version__)
print('device:', torch.npu.get_device_properties(0))
PY
```

The source branch is based on runtime commit
`8dbf2b366b690d83843c42498e972745a89072ee`, matching the regression's B100 runtime.
The `PYTHONPATH` assertion prevents accidentally testing an older SGLang
installation. Record the displayed head SHA when comparing machines.

## 2. Operator correctness and fallback contracts

```bash
ASCEND_RT_VISIBLE_DEVICES="$NPU_SINGLE" \
python test/registered/kernels/ops/gemm/test_npu_qwen_vl_prefill.py -v \
  2>&1 | tee "$VLM_RESULTS/operator-tests.log"
```

Require all four tests to pass. These include the original full
`[2526,2048] @ [2048,11008]` shape, output dtype, unchanged weight identity and
contents, multi-request decode, custom methods/wrappers and unsupported inputs.
The non-ND fallback test mocks the backend's format query and compares the real
linear output; it does not claim to run a 950 NZ kernel.

## 3. CPU FP64 precision samples and steady-state operator performance

```bash
ASCEND_RT_VISIBLE_DEVICES="$NPU_SINGLE" \
python test/manual/kernels/bench_npu_qwen_vl_prefill.py \
  --m 1 16 128 512 2526 --k 2048 --n 11008 \
  --seeds 0 42 2026 --warmup 10 --iterations 30 --rounds 3 \
  --output-dir "$VLM_RESULTS/operator-benchmark" \
  2>&1 | tee "$VLM_RESULTS/operator-benchmark.log"
```

Both paths receive identical CPU-seeded BF16 tensors. Every shape/seed checks
sampled complete rows against CPU FP64 (`rtol=0.005`, `atol=0.001`) and checks
three-run full-output repeatability. The sampled rows include historical
partition boundaries and the last row. It reports max absolute error, RMSE,
relative L2, full-output hashes and the number of elements differing between
linear and GMM. This is sampled FP64 coverage, not full-matrix FP64 coverage.

One seed per shape is profiled with `torch_npu.profiler`, ten warmup calls and
three alternating linear/GMM rounds of thirty calls. The script verifies the
number of recorded matmul operations, keeps the generated profiler CSV files
(including `op_statistic.csv`), and reports the median of the three per-round mean device
times. M=1 here is a one-token **prefill**; ordinary decode keeps linear.

These are warmed, repeatedly reused tensors and operator device times. They do
not measure end-to-end throughput, CPU dispatch latency or cold-cache model
performance. Do not infer a model speedup directly from this microbenchmark.

If a trusted real activation/weight capture is available, use the same script:

```bash
# capture.pt must contain CPU BF16 tensors {'x': X[M,K], 'weight': W[N,K]}.
ASCEND_RT_VISIBLE_DEVICES="$NPU_SINGLE" \
python test/manual/kernels/bench_npu_qwen_vl_prefill.py \
  --capture capture.pt --warmup 10 --iterations 30 --rounds 3 \
  --output-dir "$VLM_RESULTS/captured-operator-benchmark" \
  2>&1 | tee "$VLM_RESULTS/captured-operator-benchmark.log"
```

## 4. Prepare the pinned original nine-case suite, images and weights

This base branch does not contain the Ascend fork's original NPU nine-case file.
Fetch the exact file tested in CI, rather than substituting the upstream CUDA
test or an evolving branch head. The original assertions remain unchanged.

```bash
mkdir -p "$VLM_RESULTS/fixtures/images"
curl -fL --retry 3 \
  https://raw.githubusercontent.com/hanwlax/sglang/f1d2c2b4d76ad1afb444eb1c654a108fccd488f7/test/registered/npu/vlm_models/test_npu_vlm_input_format.py \
  -o "$VLM_RESULTS/fixtures/test_npu_vlm_input_format.py"
for name in man_ironing_on_back_of_suv.png sgl_logo.png; do
  curl -fL --retry 3 \
    "https://raw.githubusercontent.com/sgl-project/sgl-test-files/refs/heads/main/images/$name" \
    -o "$VLM_RESULTS/fixtures/images/$name"
done
python - <<'PY'
import hashlib, os
from pathlib import Path
root = Path(os.environ['VLM_RESULTS']) / 'fixtures'
expected = {
    'test_npu_vlm_input_format.py': 'b9728ccd1a81bca5e7db5972e8fcb46c75809f07dae5f88a3cf8d019067a53a9',
    'images/man_ironing_on_back_of_suv.png': 'e06917184a00b14abd70cd8ea0ff5dca9abfbbad29f7b25c02f97133d4cd060e',
    'images/sgl_logo.png': '4af0b4cbf4fe9479c1fc3675e025f2752b67bd9a7c15a0003af11110c6ee4165',
}
for name, digest in expected.items():
    assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest, name
print('Pinned test and image hashes verified')
PY

# Set these to existing checkpoint directories if the weights are already present.
export QWEN_MODEL="${QWEN_MODEL:-$VLM_REPO/validation-assets/models/Qwen2.5-VL-3B-Instruct}"
export GEMMA_MODEL="${GEMMA_MODEL:-$VLM_REPO/validation-assets/models/gemma-3-4b-it}"
export KIMI_MODEL="${KIMI_MODEL:-$VLM_REPO/validation-assets/models/Kimi-VL-A3B-Instruct}"
python - <<'PY'
import json, os
from pathlib import Path
from modelscope import snapshot_download
for env, model in [
    ('QWEN_MODEL', 'Qwen/Qwen2.5-VL-3B-Instruct'),
    ('GEMMA_MODEL', 'google/gemma-3-4b-it'),
    ('KIMI_MODEL', 'moonshotai/Kimi-VL-A3B-Instruct'),
]:
    target = Path(os.environ[env])
    index = target / 'model.safetensors.index.json'
    complete = (target / 'config.json').is_file() and index.is_file()
    if complete:
        shards = set(json.loads(index.read_text())['weight_map'].values())
        complete = all((target / shard).is_file() for shard in shards)
    if not complete:
        snapshot_download(model, local_dir=str(target))
    print(env, target)
PY
```

## 5. Run all nine original assertions on two free cards

```bash
ASCEND_RT_VISIBLE_DEVICES="$NPU_PAIR" \
python test/manual/vlm/run_npu_input_format.py \
  --case "$VLM_RESULTS/fixtures/test_npu_vlm_input_format.py" \
  --gemma-model "$GEMMA_MODEL" \
  --kimi-model "$KIMI_MODEL" \
  --qwen-model "$QWEN_MODEL" \
  --image-dir "$VLM_RESULTS/fixtures/images" \
  --output-dir "$VLM_RESULTS/full-nine" \
  2>&1 | tee "$VLM_RESULTS/full-nine.log"

python - <<'PY'
import json, os
from pathlib import Path
root = Path(os.environ['VLM_RESULTS'])
suite = json.loads((root / 'full-nine/result.json').read_text())
bench = json.loads((root / 'operator-benchmark/result.json').read_text())
assert suite['successful'] and suite['tests_run'] == 9
assert not suite['failures'] and not suite['errors'] and not suite['skipped']
assert bench['successful']
for case in bench['cases']:
    if 'median_device_us' in case:
        print(case['label'], case['median_device_us'],
              'linear/GMM:', case['linear_over_gmm_speedup'])
print('Nine-case and numerical checks passed')
PY
npu-smi info | tee "$VLM_RESULTS/npu-after.txt"
```

Do not require 950 token IDs to match A3. Require the original assertions to
pass, preserve responses for investigation, and compare same-device baseline
and GMM precision/performance. The API's hardware support does not establish
cross-platform bitwise identity or broad model-accuracy equivalence.

Retain the head SHA, environment, test logs, result JSON, response IDs and raw
profiler output. A model OOM or setup failure is not a numerical validation pass.
