# Full valid boundary tensors: A3 DP4 / A5 DP2

This opt-in schema-5 capture runs the **complete 38-block model** and saves all
valid rows and all trailing columns of blocks 0 and 1. It does not truncate the
model, change its arithmetic, or establish baseline accuracy. Establish A3's
natural-generation GSM8K baseline using the clean eval runtime first.

## What changed and what is saved

The original DP4 helper saves only first/middle/last live rows (all columns).
The older LONGCAT_ACT JSONL saves still fewer columns. Neither supports a claim
that every activation element was compared.

The new mode writes one CPU `.pt` per boundary/rank/forward and a completed index
`.pt` plus `.json`. It retains native dtype and every column, with all logical
request rows present on that rank; physical padding and inactive DP rows are
excluded. Row-sharded boundaries need all 16 origin Attention TP ranks to cover
all tokens. DP-gathered boundaries select the origin request's contiguous region.
Each record contains physical/logical row indices, token IDs, absolute positions,
shape/dtype, topology, head IDs where applicable, producer sum contract, linear
partition metadata, residual-state semantics, element count and tensor-byte SHA.
CPU copies occur immediately before subsequent in-place operations. Only metadata
is retained between stages; whole-forward NPU activation copies do not accumulate.

Captured points include embedding/OE IDs; both Attentions' input/local output and
available Indexer IDs; router logits/global expert IDs/weights and rewritten
dispatch IDs/weights; ordinary expert output, identity, branch reduction and
combined branch; both dense FFNs' input/local output; independent residuals and
layer contribution. Final norm and last-position **full-vocabulary** logits are
also saved. Only the last position's logits exist at this inference interface;
this is not logits for every prefill token.

The next block's `attn0.input` and newly added `attn0.residual` are also saved, so
for layers 0/1 there are two extra consumer points in layer 2. This observes the
consumer after its normal reduction/residual update without adding a collective.
It does not capture layer 2's Attention computation or later blocks.

`ffn0.finish`/`ffn1.finish` are omitted because their row/transport state is
batch-dependent; use consumer inputs and residuals. `layer.output` is a pending
contribution, not the complete residual hidden state. Identity zero is legal;
`no_live_rows` is different from zero-valued activations. Indexer sentinel/padded
slots and integer IDs require semantic checks, not floating L2 comparisons.

By default full mode does **not** install the old internal Attention/cache hooks: this task
captures boundaries, not every intermediate operator or FP8 operand. It does not
record remote expert-internal tensors. Do not sum origin-group MoE samples as an
all-EP reduction oracle. A3/A5 dense-FFN local contributions have different TP
partitions: compare completed consumer boundaries, not equal global-rank locals.
Attention-local rank/head alignment is valid because both use attention TP16.

Warmup r0 runs the same fixed-prefix protocol but skips full tensor output.
Measured r1/r2 are saved. Missing index means an interrupted capture; never use
partial files as a complete forward. Schema-4 `compare_fixed_decode_dp4.py` is
intentionally incompatible with schema 5. Cross-platform analysis comes after
both datasets exist; do not weaken the old comparator's validation.

## Build new private runtimes

Use the updated versioned tool directory. The generator still extracts inference
revision `c0e9d9a0cc6b`; verify that this is the intended A5 implementation before
using it. It does not merge uncommitted green-zone implementation changes.
Existing runtimes and evidence must not be overwritten. Every participating host
needs the same generated profile and source hashes.

```bash
# From the updated test/manual/ascend/longcat2_baseline directory (or its bundle).
# SOURCE must contain the pinned Git revision; OUT must not exist.
python3 prepare_runtime_dp4.py --source "$SOURCE" --mode trace \
  --trace-topology a3-dp4 --full-boundary-layers 0 1 --out "$OUT"

# A5: different profile; do not copy the A3 runtime/profile unchanged.
python3 prepare_runtime_dp4.py --source "$SOURCE" --mode trace \
  --trace-topology a5-dp2 --full-boundary-layers 0 1 --out "$OUT"
```

The resulting receipt records helper hashes and `longcat_capture_profile.json`,
which controls the expected topology and full capture selection. Merely editing
an outer helper has no effect on an already generated runtime or running worker.

CPU-only capture regression (CPU torch required):

```bash
python3 verify_full_boundary.py
```

The test checks DP2/DP4 gathered offsets, complete union of 16 row shards, decode
ranks with no live row, exclusion of NaN padding, preservation after source
mutation, required-boundary completion and all-column persistence. This does not
certify NPU behavior or A5 deployment.

## Start the capture service

A3: once resources are free and normal GSM8K validates the baseline, use the
existing versioned launcher on all four hosts, with ranks 0/1/2/3 and identical
new RUN_ID. Point RUNTIME at the new **A3** full-boundary runtime:

```bash
RUNTIME=/absolute/new/runtime_full_a3 \
  bash run_a3_full_baseline.sh 0 a3_full_boundary_01 trace
```

A5: use the new capture-only `run_a5_full_boundary.sh` on .105/.113/.121/.122
(node ranks0/1/2/3). It selects full FP8, TP32/DP2/EP32, auto/AI_CPU,
chunk2048, maxrunning2, context/KV8192, no graphs/NEXTN/radix. Defaults match the
previously supplied green interfaces (HCCL enp34s0f1, Gloo data0.3001); override
those variables if the actual container differs. `MOE_ENABLE_TOPK_NEG_ONE=1` and
`DEEPEP_HYBRID_DEPLOYMENT=1` are set, MLAPO/requant are0. It does not stop services.

```bash
# Same NEW run name and runtime version on all four hosts; change node rank.
RUNTIME="$PWD/runtime_full_a5" bash run_a5_full_boundary.sh 0 a5_full_boundary_01 trace --dry-run
RUNTIME="$PWD/runtime_full_a5" bash run_a5_full_boundary.sh 0 a5_full_boundary_01 trace
```

This launcher passed Bash syntax and synthetic model/receipt dry-runs for all
four ranks, including wrong-topology rejection. It has not been run on A5 NPUs.
A5 natural GSM8K eval adaptation is separate; this entry is for capture only.

## Replay a small aligned capture first

Do not send unrelated requests. Start with DP0 and prefix length1 on both sides;
each side makes warmup + two measured requests. All 16 active ranks are necessary.
Length1 means cold prefill of the original 852 IDs, force output token70115, then
one real decode at position852. Both runs use the same fixture/prefix, independent
of naturally generated answers. Use distinct output directories.

```bash
# A3, after full-boundary trace service is ready.
python3 replay_fixed_decode_dp4.py --topology a3-dp4 \
  --control fixtures/control_852.json --url http://192.168.25.209:10096 \
  --dps 0 --lengths 1 --repeat 2 --timeout 3000 --out /absolute/new/a3_client

# A5, run in the green zone with its port and copied fixture.
python3 replay_fixed_decode_dp4.py --topology a5-dp2 \
  --control fixtures/control_852.json --url http://127.0.0.1:10086 \
  --dps 0 --lengths 1 --repeat 2 --timeout 3000 --out /absolute/new/a5_client
```

After the first pair works, use a new output directory and `--lengths 8`, or select
more DPs. Prefix8 prefills 859 fixed tokens and runs one decode at position859;
it is not eight free generation steps. Timeout stops further submissions.

Monitor each host's actual serve log:

```bash
grep -E 'BASELINE_RUNTIME|LONGCAT_FIXED_V4_WORKER|LONGCAT_FIXED_V4_FORCE|LONGCAT_FULL_BOUNDARY|Traceback|RuntimeError' /absolute/serve.log
```

Worker/force log prefixes keep the historical V4 spelling for both profiles;
the worker fields, RID `lcfix4-`/`lcfix2-` and schema5 index determine topology.
For `--dps 0 --lengths 1 --repeat 2`, expect 64 completed indexes per platform:
16 active ranks x 2 measured requests x EXTEND/DECODE. Also retain all 64 A3 or
32 A5 worker receipts. Aggregate same-run files from all hosts without overwriting
collisions. Save client manifest/responses, launch args/env, runtime receipt and
checkpoint config/index hashes. Full tensors can occupy many GiB; measure disk
usage after the first group before expanding. No extra physical KV-cache dump is
produced by this mode.

Green-zone tensor files stay in the green zone. Blue A3 data may be transferred
in through the established sync route. A green path alone is not remotely readable
from blue; comparison will run in green and the user can relay textual summaries.

## Later comparison contract (not implemented here)

Align input IDs, absolute positions, mode, logical token, layer/stage and matching
mathematical partition. Do not compare different local FFN/expert slots blindly.
Report original A3/A5 values plus signed difference, absolute difference,
max/mean/RMS and quantiles of absolute difference, count/fraction unequal, nonfinite
counts, relative L2 with its denominator, and first/max difference logical indices.
Absolute differences are in the activation's original numeric scale, **not percent**;
relative errors are dimensionless. BF16 has magnitude-dependent spacing, so an
absolute 0.001 is not a universal passing tolerance. No comparison threshold or
GSM8K accuracy verdict is encoded in the collector.

Generated probes now live under `python/sglang/srt/debug_utils/` and use fully qualified imports. A same-named helper in the launch working directory cannot override the private runtime copy.

## Coverage switches

| Mode | Block coverage | Row/column coverage | Platforms |
| --- | --- | --- | --- |
| `--full-boundary-layers 0 1` | blocks0/1 + immediate consumer input/residual | all valid rows, all columns at captured boundaries | A3 DP4, A5 DP2 |
| `--full-boundary-layers $(seq 0 37)` | all38 block boundaries | all valid rows, all columns at captured boundaries | A3 DP4, A5 DP2 |
| legacy `--mode trace` without full-boundary flag, `LC_INTERNAL_LAYERS=all` at launch | all38 sampled boundaries + both Attentions per block internally | boundary first/middle/last rows; internal tensors on selected ranks/r1 | A3 DP4 only |

The last mode supports Attention stages a_proj, norm_qb, qk_bmm, rope, sparse,
and tail. It does not expose every MoE/dense-FFN GEMM operand, and the FP8 quantizer
operands/weights needed for a complete CPU replay are not all present. Default
internal ranks are0/16/32/48 (only four local heads per selected Attention rank).
Internal hooks require the explicit generation flag described below;
setting LC_INTERNAL_LAYERS=all alone does not install them in a full-boundary runtime.

See [CAPTURE_STEPS_ZH.md](CAPTURE_STEPS_ZH.md) for source checkout, per-host build,
launch, replay and evidence-file collection commands.

## Full 38-block capture plus selected Attention internals

To collect both in the same forwards, add `--with-internal-attention` when
generating the trace runtime. Select every boundary block with
`--full-boundary-layers $(seq 0 37)`, and launch with
`LC_INTERNAL_LAYERS=0,7,37 LC_INTERNAL_RANKS=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15`
for a DP0 replay. The model still has all 38 blocks. This captures Attention IDs
0/1, 14/15, and 74/75 on all 16 origin ranks for measured repeat r1, alongside
the full boundary r1/r2 data. Later repeats intentionally omit the heavy internal
captures. The six internal stages are `a_proj`, `norm_qb`, `qk_bmm`, `rope`,
`sparse`, and `tail`; both EXTEND and DECODE are expected. Physical KV cache dumps
include unused slots: consume only entries selected by the saved indices, block
tables and sequence lengths. These are Attention internals; remote expert-internal
GEMM operands and every dense-FFN intermediate are not captured. Boundary data
still includes the router, MoE and dense FFN points described above.

The runtime receipt records whether internal hooks were installed. Regenerate
the private runtime and restart after changing this flag; editing only the outer
helper cannot enable hooks in an existing runtime.

## Eliza fixed input and real routing

`fixtures/eliza_853.json` freezes GSM8K test row9 after the original five shots,
using the same SGLang tokenizer loader as the A3 service. Its prompt has853 IDs
and SHA256 `1b8d40fb3f004235b35182563c4883cb4ef8ce14bf6f7ff55484afe8e262196c`.
The explicit11-token comparison prefix is ` Eliza earns 40 * 10 = `.
This was selected after an A5 response showed arithmetic corruption and repeated
text zeros; it is not a recovered original failing output-ID sequence. Do not
re-tokenize mojibake copied from an HTML viewer. The fixture records UTF-8 text,
dataset hash/row, tokenizer file hashes and both complete ID hashes.

```bash
python3 replay_fixed_decode_dp4.py --topology a3-dp4 \
  --fixture fixtures/eliza_853.json --url http://192.168.25.209:10096 \
  --dps 0 --lengths 1 11 --repeat 2 --timeout 3000 --out /absolute/new/eliza_client
```

Length1 targets the second output token; length11 consumes the complete frozen
prefix and samples the following token. This is aligned computation, not a
promise to reproduce a free-generation loop. The existing `--control` input
retains its852-token hash guard. Custom fixtures require full SHA256 validation
and cannot be combined with a prefix override. `make_gsm8k_fixed_fixture.py`
rebuilds fixtures from the frozen dataset and actual model tokenizer.

The collector does not force experts, replace routing weights, or run alternate
operator implementations. It preserves raw router logits, global selected IDs
and scaled routing weights before Identity rewriting, then dispatch IDs/weights
after rewriting. `moe.reduced` is the ordinary expert branch after its declared
sum; `moe.branch_output` additionally includes Identity. Residuals are separate
from pending layer contributions. Compare aligned layer/MoE inputs first, then
logits and routing, then merged outputs. Once selected experts differ, output
differences include routing changes and cannot alone diagnose an expert kernel.
Remote expert-internal tensors remain outside this capture's coverage.
