# Kimi-K3 MLA NZ optimization stages

These changes preserve the explicit PA-NZ cache layout and its logical prefix
reader. They apply to ordinary NPU MLA cache writes with `SGLANG_USE_FIA_NZ=1`.

## Changes and deployment controls

| Stage | Behavior | Selection |
| --- | --- | --- |
| Forward index reuse (`36f3e02bfe`) | Share the latent/RoPE index tensors between layers using the same location tensor, layout and stream. A 24-layer forward needs two index launches instead of 48 under those conditions. | Enabled for eager/NPUGraph Kimi-K3 forwards. |
| Runtime token count (`12f4f29aa7`) | Remove `N` from Triton specialization, including alignment specialization. Different prefill tails reuse the compiled index kernel. | Enabled; layout, location dtype/stride and index width still specialize. |
| Direct block store | One kernel computes addresses and writes both latent and RoPE values. It reads the strided split source directly and writes explicit two-dimensional NZ blocks. | Opt in with `SGLANG_NPU_USE_TRITON_MLA_NZ_STORE=1`; default `0` retains native scatter. |

Set the direct-store variable identically on every rank **before server startup**.
The pool resolves it during construction and graph capture records the selected
path. Changing the environment of an existing server does not change its graphs.
FP16/BF16 storage supports direct writes; other storage dtypes retain native
scatter. This toggle does not change MLAPO's independent write path.

Index reuse is scoped to one model call, including each capture warmup. It never
retains indices from a previous forward or warmup. Graph replay re-executes the
captured producers using current locations. Normal tensors invalidate on version
changes; inference tensors must remain immutable within the model forward.
Different streams do not share producers. `torch.compile` bypasses the Python
cache scope and uses the per-layer builder.

Direct stores use `[dim // 16, 16]` block pointers with strides
`[page_size * 16, 1]`. Keeping the contiguous inner tile explicit matters: the
first prototype with flattened element offsets was much slower on scattered
locations. Destination element addresses use int64, independently of whether
16-element row indices would fit int32. As with native scatter, live locations
must be valid and unique, and source storage must not alias the destination.

## Reproduce local correctness and performance checks

Run from the repository root inside a supported Triton-Ascend/torch_npu image:

```bash
PYTHONPATH=python ASCEND_RT_VISIBLE_DEVICES=0 python \
  test/registered/unit/npu/attention/test_npu_mla_nz_indices.py -v

PYTHONPATH=python python \
  test/registered/unit/npu/attention/test_npu_mla_cache.py -v

PYTHONPATH=python ASCEND_RT_VISIBLE_DEVICES=0 python \
  test/manual/ascend/benchmark_mla_nz_write.py \
  --rows 48 88 128 --output /tmp/mla-nz-contiguous.json

PYTHONPATH=python ASCEND_RT_VISIBLE_DEVICES=0 python \
  test/manual/ascend/benchmark_mla_nz_write.py \
  --rows 48 88 128 --slot-stride 127 --output /tmp/mla-nz-scattered.json
```

The benchmark runs four implementations in rotating order: native index
arithmetic/scatter, per-layer fused indices/scatter, shared indices/scatter, and
direct block stores. Each sample includes 24 layers of latent **and** RoPE writes,
with a split BF16 source, page size 128, 64 cache pages per layer, 3 warmups,
10 groups and 10 repetitions. Both eager and NPUGraph results include sample
distributions. NPU events measure stream intervals; eager intervals can include
host submission gaps. These are local write-chain timings, not serving TPOT.

Tests compare cache contents exactly, including untouched rows, partial-page
overwrites and logical prefix reads. Direct-store cases cover FP16/BF16, page
sizes 16/96/128, non-contiguous locations and values, and token counts
0/1/65/128/4096. Two-layer graph replays update both values and locations. Index
tests additionally cover large addresses, stream separation, forward lifetime,
and one compiled hash across token counts 1/7/48/65/88/128/255.

## A3 measurements, 2026-09-09

Measured on host 209 in `k3-nz-ab-20260906`, using the isolated worktree
`/home/hanwlax/test-codes/k3/.nz-stages-validation-20260909`.
The production experiment checkout was not replaced for these measurements.
Environment: Ascend910_9362, torch 2.10.0+cpu, torch_npu 2.10.0.post4,
Triton-Ascend reporting version 3.2.0.
Values below are median **milliseconds for all 24 layers**, in NPUGraph mode:

| Locations | Tokens | Native indices/scatter | Fused indices/scatter | Shared indices/scatter | Direct block store |
| --- | ---: | ---: | ---: | ---: | ---: |
| Contiguous | 48 | 8.498 | 7.673 | 7.213 | 0.109 |
| Contiguous | 88 | 14.071 | 13.317 | 12.828 | 0.172 |
| Contiguous | 128 | 19.966 | 18.893 | 18.435 | 0.237 |
| Scattered (stride 127) | 48 | 2.339 | 1.501 | 1.035 | 0.117 |
| Scattered (stride 127) | 88 | 2.713 | 1.836 | 1.396 | 0.188 |
| Scattered (stride 127) | 128 | 3.120 | 1.917 | 1.432 | 0.234 |

Raw group samples are in `stage4-final-contiguous.json` and
`stage4-final-scattered.json` under that isolated worktree. The strong dependence
of native scatter on location distribution is a candidate to investigate in A5
variability; these isolated results do not establish its cause.

The dynamic-N check used a separate paired comparison of the old constexpr
kernel and the runtime-N kernel. Graph interval medians (old/runtime, microseconds)
were 14.848/14.920 for N=48, 15.794/15.849 for N=88, 17.470/16.580 for N=128,
and 133.425/129.543 for N=4096. All seven tested token counts shared one compiled
hash for the same layout and input types.

The kernel-verifier precision checks passed. Its benchmark script filtered
`OP Type` by the adapter name and reported invalid zero values; those values are
not used above. The raw profiler CSV did contain the kernels. For N=88, its
single direct-store kernel average fell from 228.553 us in the flattened-address
prototype to 7.905 us with two-dimensional block pointers (50 calls each).

## Serving validation still required

The isolated A3 checks do not establish A5 startup stability or end-to-end gains.
Validate GSM8K and accepted length again for the new commits, then compare TPOT
distributions under the same workload. Keep the first run separately visible.
Compare the default native-scatter path with direct store enabled using the
same commit and otherwise identical launch configuration.

Graph update/submission timing and MoE communication scheduling were not changed.
DSpark target verify already consumes the final CPU sequence lengths. Further
changes require per-rank evidence of the critical path; the older early-update
experiment did not establish an end-to-end gain.
