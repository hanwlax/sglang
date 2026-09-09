# K3 FIAS V2 prefill: P0 correction

Base: zzx/a5-k3-0828 at `13e5168c684858c3849709c2bf75c5e42564311b`.
Branch: mine/prefill-nz.

## What changed after the first implementation

The first commit `64d1c2ae8d` routed prefill through latent MLA and padded
12 heads to 16. A5 profiling reported FIAS growing from 18.185ms to 101.980ms:
the original expanded Q/K D=192, V D=128 became Q/K D=512+64, V D=512.
That was an algorithm/shape change, not an API-only V1/V2 comparison.
It is removed by this P0 fix, including the unused latent-prefill helper.

The K3 model dispatcher and NPU attention preparation are restored to the base.
The prefill flag now ONLY substitutes the V2 API at the existing expanded-MHA
FIAS call sites in AscendAttnBackend (cached and cold MLA-model prefill).
No head padding, latent Q absorption, new W_vc multiply, or paged-latent
FIAS call is introduced.

- Cached prefill keeps BSND Q/K=192, V=128 and the original head count.
- Cold prefill keeps the original TND split Q/K=128 + suffix64, V=128,
  cumulative sequence lengths, and DP padding handling.
- Q/KV projection, norms, cache writes, prefix gather/unpack and KV expansion
  are the original code paths. There is no new graph ordering or stream.
- Existing decode, target verify, draft, KDA, DeepEP and their V2 controls
  are unchanged. Non-FIAS/native and CP paths are not newly adapted.
- The existing prefill flag also applies to these same expanded MLA-model
  backend call sites for other models; it does not change model routing.

## Startup

Keep the existing launch command. On every node, to test only prefill V2:

```bash
export SGLANG_NPU_USE_FIAS_V2_PREFILL=1
export SGLANG_USE_FIA_NZ=0
export SGLANG_NPU_USE_MLAPO=0
```

Keep `SGLANG_NPU_USE_FIAS_V2_BSND` at your baseline value; it independently
controls the existing verify/draft V2 paths, not this prefill fix.
Setting the prefill flag to 0 selects the original V1 API with the same MHA math.

NZ may separately be enabled with `SGLANG_USE_FIA_NZ=1`. This base already
decouples NZ from MLAPO. Important: NZ is the persistent latent cache layout;
the prefill prefix reader unpacks it and expands K/V before FIAS. Therefore
the corrected prefill FIAS inputs are ND even with NZ cache enabled.
This fix does NOT promise direct NZ prefill FIAS or a performance improvement.
Restart all nodes between changes; do not mix layouts or code versions.

## Verification

CPU regression commands in a fully installed project environment:

```bash
PYTHONPATH=python python test/registered/unit/npu/attention/test_npu_mla_prefill.py
PYTHONPATH=python python test/registered/unit/npu/attention/test_npu_mla_prefill_wiring.py
PYTHONPATH=python python test/registered/unit/npu/attention/test_npu_mla_cache.py
```

Tests cover exact reported shapes using meta tensors, unchanged tensor identity,
keyword mapping, integer cumulative lengths, V1/V2 routing, NZ/ND prefix content,
ragged cold prefill, DP padding and verify/draft isolation. Wiring tests execute
selected production AST bodies with CPU dependencies. CPU tests do not run CANN.

On A5 (no model weights needed):

```bash
PYTHONPATH=python python test/manual/npu/test_k3_mla_prefill_v2.py
```

This compares actual V1/V2 on cold and cached prefill, uses a small FP32 reference,
and includes the reported 12-head, Q=6528, KV=133760 long-prefill shape.
It requires roughly 1.1GB for long-case inputs plus workspaces and temporaries.
These NPU tests have NOT been run in the CPU-only WSL environment.

Then repeat the original model request and inspect matching-layer/chunk FIAS:
Q `[1,6528,12,192]`, K `[1,133760,12,192]`, V `[1,133760,12,128]`,
output `[1,6528,12,128]` for that same cached request.
No `[1,16,6528,512]` should remain on this prefill route.
Compare precision/accept length and full attention stage time, not only FIAS.
No claim is made that latency has returned to 18ms before A5 validation.

The previous pair of traces also had incompatible KV extents:
V1 S=133760 versus V2 page-table capacity 1023*128=130944.
That mismatch was not diagnosed as a truncation bug. Align the exact request,
layer, chunk and effective KV length when comparing; do not claim a separate
metadata defect was fixed by this P0 patch.

Official API references:
- [FIAS V2 Python signature](https://gitcode.com/Ascend/op-plugin/blob/7.3.0/docs/context/torch_npu-npu_fused_infer_attention_score_v2.md)
- [CANN V5 prefill MLA constraints](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/910beta3/API/aolapi/context/ops-transformer/aclnnFusedInferAttentionScoreV5.md)
