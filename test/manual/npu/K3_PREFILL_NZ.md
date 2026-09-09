# K3 paged NZ + FIAS V2 prefill

Base: `zzx/a5-k3-0828` at `13e5168c684858c3849709c2bf75c5e42564311b`.

This opt-in path changes **Kimi-K3 target prefill**, not draft MHA, KDA,
target verify, decode, DeepEP, graph-update ordering, or the NZ scatter kernel.
It preserves the existing prefill Q/KV projection, RMSNorm and pool write.
Instead of expanding the latent prefix with `kv_b_proj`, it computes
`Q_nope @ W_kc`, reads the paged latent cache directly with FIAS V2, then
uses the existing MLA core's `W_vc` and output projection/gate.
The 64-dimensional K3 skip-RoPE suffix still participates in attention.
This moves matrix products and changes floating-point rounding; mathematical
equivalence and CPU tests do not establish model-level numerical parity.

## Enable

Apply on every server node, before launching the existing service script:

```bash
export ASCEND_USE_FIA=1
export SGLANG_USE_FIA_NZ=1
export SGLANG_NPU_USE_FIAS_V2_PREFILL=1
export SGLANG_NPU_USE_FIAS_V2_BSND=1  # existing DSpark verify/draft V2
export SGLANG_NPU_USE_MLAPO=0
```

Unlike older forks discussed in the incident, **this zzx base already decouples
NZ from MLAPO**. Enabling full MLAPO is not required and its generic K3
compatibility is not fixed by this patch. The new prefill preparation bypasses
MLAPO, but existing decode/verify code is intentionally unchanged.

`SGLANG_NPU_USE_FIAS_V2_PREFILL=0` restores the original prefill route while
retaining NZ and the existing verify/draft choice. Setting NZ=0 with the new
prefill flag still exercises V2 over ND latent cache for a layout-only A/B.
Keep all other settings/model weights fixed. Restart between cache-layout changes.

## Layout and scope

- Query per request: BNSD `[1, padded_local_heads, actual_chunk_tokens, 512]`.
  Head padding is zero-initialized and discarded. Query suffix D=64.
- KV: the persistent NZ storage is viewed as
  `[pages, 1, D/16, page_size, 16]`; no full-prefix gather or reorder is needed.
- Per-request KV lengths are **not cumulative**. They include the current
  chunk. Right-aligned causal masking (`sparse_mode=3`) exposes the prefix
  plus tokens up to the current query position, never future tokens.
- Unequal query lengths use separate calls; zero-length rows are skipped,
  and DP token-padding output is zero. No fixed DSPARK block length is used.
- FP16/BF16 query and cache, D=512/suffix64, up to128 local heads; page size
  16-aligned and <=1024. Quantized KV is rejected. MoE weight quantization
  is separate from KV-cache dtype.
- This is ordinary eager/chunked prefill; prefill CP is explicitly rejected.
  No new prefill graph capture support is claimed. Existing decode/verify
  graph behavior is unchanged. Split GGUF attention keeps its original route.
- The pre-existing prefix metadata construction is retained; the expensive
  cached latent gather and KV expansion are bypassed, not all host metadata work.

## Validation

CPU regression files (normal project test environment):

```bash
PYTHONPATH=python python test/registered/unit/npu/attention/test_npu_mla_prefill.py
PYTHONPATH=python python test/registered/unit/npu/attention/test_npu_mla_prefill_wiring.py
PYTHONPATH=python python test/registered/unit/npu/attention/test_npu_mla_cache.py
```

CPU tests use an emulated FIAS call to verify NZ/ND arguments, page ordering,
causality, lengths, empty/padded rows, storage aliasing, and original routing.
They do **not** execute the CANN kernel. The wiring tests compile the selected
production function bodies with AST to avoid importing the full model on CPU.

On the actual A5/CANN/torch_npu deployment (no weights required):

```bash
PYTHONPATH=python python test/manual/npu/test_k3_mla_prefill_v2.py
```

This calls the real V2 operator with cold and 128K-cached ragged shapes, both
ND and NZ, and compares against FP32 attention. It has not been executed in
the CPU-only WSL environment. Hardware/library-specific support must be tested
on the deployment; do not infer A3 support from A5 API documentation.

Then run single-curl, the same cached 128K/1K/BS32 replay, GSM8K/GPQA using
the existing baseline setup, and multiple steady-state performance rounds.
Compare output quality, accept length, TTFT, TPOT, and output throughput.
Absorbed MLA may have different long-query costs; no speedup is promised.
The previous A5 distributed-hang incident is not declared resolved by this work.

Official API reference for A5 constraints:
https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/910beta3/API/aolapi/context/ops-transformer/aclnnFusedInferAttentionScoreV5.md
Python `npu_fused_infer_attention_score_v2` and CANN `aclnn...V5` are different
version namespaces; verify the installed op-plugin mapping, not just a kernel name.
