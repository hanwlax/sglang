# A3 full-layer numerical baseline — TP64 / attention DP4 / EP64

Updated 2026-11-17. Run inside `hanwlax-longcat2.0` on blue hosts
209,212,216,217 (node ranks0,1,2,3) in:
`/home/hanwlax/workspace/progress/longcat2.0/a3_full_baseline_20261117`.

The shared source checkout, existing `run_trace.sh`, old DP2 probe files and old
`runtime/` are unchanged. New commands use `runtime_dp4_eval/` or
`runtime_dp4_trace/`. Backups of replaced launcher/README are in `backup_before_dp4/`.
Do not use the old `*_dp2.py` files for this deployment.

## Deployment and memory

- Full38 target layers, total TP64, attention DP4 / TP16, EP64, MoE TP1.
- Each rank owns12 ordinary experts, each retains full intermediate width2048.
- DeepEP auto, AI_CPU, `MOE_ENABLE_TOPK_NEG_ONE=1`, `DEEPEP_HYBRID_DEPLOYMENT=1`.
- Both graphs disabled, NEXTN off, BF16 activations/KV, radix cache disabled.
- All16 logical NPU devices on each host. Never start over other jobs.
- Global prefill chunk4096 =>1024 per DP; max running requests4 =>1 per DP.
- Context8192 and KV token cap8192; mem_fraction_static0.92; serial model loading.
- Same hardware-independent deployment settings in eval/trace modes. Only tracing
  differs. Model is A3 INT8, attention weights BF16; A5 FP8 need not be bitwise equal.

The previous weight audit implies ordinary experts ~21.39 GiB/rank at EP64,
versus ~42.77 at EP32. Embedding at attention TP16 remains roughly16 GiB/rank.
Attention, dense FFN, replicas, conversions, communication and runtime are extra;
memory fit is NOT established until actual startup. KV/chunk reduction cannot fix
a weight-load OOM. Do not lower max generation length to manufacture a higher score.
For fixed-prefix tracing keep global chunk>=4096 so852/859-token prompts remain one
prefill. Match per-DP chunk geometry when comparing A5. Smaller chunks can be tried
for normal eval, but record that change and use sufficient context for prompt+2048.

## 1. Normal service for GSM8K

Use the same NEW run name on all four hosts. Inside each container:

```bash
cd /home/hanwlax/workspace/progress/longcat2.0/a3_full_baseline_20261117
# 209
bash run_a3_full_baseline.sh 0 a3_dp4_eval_001 eval
# 212
bash run_a3_full_baseline.sh 1 a3_dp4_eval_001 eval
# 216
bash run_a3_full_baseline.sh 2 a3_dp4_eval_001 eval
# 217
bash run_a3_full_baseline.sh 3 a3_dp4_eval_001 eval
```

HTTP10096 on209, rendezvous5017. `eval` selects the pristine inference sources
at c0e9d9a0cc6b; no forced-token, boundary or internal diagnostic hooks are installed.
All arguments/env and runtime/checkpoint metadata hashes are logged. Mode defaults
to eval if omitted. `--dry-run` is the FOURTH argument:
`bash run_a3_full_baseline.sh 0 precheck eval --dry-run`.

After startup, run on216 (or any client with the same environment):

```bash
bash run_gsm8k.sh a3_dp4_gsm8k_200_001
```

This runs the frozen `sglang.test.run_eval` scorer with Completion API,200 examples,
5-shot, temperature0, top_p1, max_tokens2048, one client thread, repeat1.
The evaluator copy ONLY relocates the HTML/JSON report files; it does not alter
requests, prompts, sampling or scoring. No reports in shared `/tmp/gsm8k_*` are overwritten.
It checks the server's effective topology and disabled graph/NEXTN settings first.
Data is copied into `evaluations/ID/gsm8k_test.jsonl`, with SHA and selected row range
recorded in `evaluation_manifest.json`. It uses the upstream GSM8K test file or
`GSM8K_DATA_PATH=/existing/test.jsonl`; the same file/shot count/example subset must
be used for A5 comparison. The first5 rows are shots; the following200 are evaluated.
`NUM_THREADS`, `NUM_EXAMPLES`, `NUM_SHOTS`, `MAX_TOKENS`, `BASE_URL` are explicit overrides.
Default1 thread avoids admission-pressure changes; repeat A5 with the same setting.
The legacy scorer turns failed requests into empty responses; retain error logs and
the per-question HTML, not just the final score. Do not label a run with request errors
as a validated reference. No expected accuracy is asserted for this new deployment.

Output: `evaluations/ID/run_eval.log`, `gsm8k_LongCat-2.0.html`,
`gsm8k_LongCat-2.0.json`, dataset, manifest and `server_info.json`.

## 2. Fixed-prefix / per-layer capture service

After normal precision validation, stop this baseline service through its owning
terminals/process supervisor, then restart all four with a NEW run name and `trace`:

```bash
# On209; use ranks1,2,3 on212,216,217.
bash run_a3_full_baseline.sh 0 a3_dp4_trace_001 trace
```

This uses the private diagnostic runtime. The launcher does not stop or replace
any existing service automatically. Never run GSM8K on the tracing service as the
normal baseline, and do not send other requests during fixed-prefix capture.

On216:

```bash
bash run_fixed_prefix_dp4.sh a3_dp4_trace_001 /tmp/longcat-control.json
```

Equivalent client command:

```bash
python3 replay_fixed_decode_dp4.py \
  --control /tmp/longcat-control.json --url http://192.168.25.209:10096 \
  --dps 0 1 2 3 --lengths 1 8 --repeat 2 --timeout 3000 \
  --out "$PWD/results/a3_dp4_trace_001/client"
```

The prompt is validated as852 tokens/hash07bd4107b296. Four DPs x two prefix
positions x (one warmup + two measured repeats) =24 sequential requests.
Each request prefills a fixed input and forces its first sampled token, then runs
one real decode. Prefix lengths1/8 target decode positions852/859; these are NOT
eight free-running decode steps. Input IDs and positions are saved and checked.
Timeout stops further submissions; resolve the outstanding request before retrying.

Data captured:
- Topology receipts for all64 workers; all16 ranks in the request's origin
  Attention TP group, with real-row/padding mapping for DP0--3.
- Existing Attention/MoE/router/FFN/residual boundary probes in ALL38 layers;
  selected real prefill rows with all columns, complete live decode rows where present;
  final norm and full last-position logits. Remote expert internals are not exhaustive.
- On measured r1 only: full Attention intermediate tensors for layers0,1,37,
  both Attentions, global ranks0,16,32,48 (local Attention TP0 in each DP group).
  Includes A projections, norms/QB, QK BMM, RoPE, sparse inputs/KV/indices/output,
  V BMM and O projection. Cache dumps include unused slots: apply saved lengths,
  tables and indices before comparing. Physical tensor padding is not real input.
- Set `LC_INTERNAL_LAYERS=all` on all four launch commands for every Attention layer,
  or change `LC_INTERNAL_RANKS` for more rank coverage. This increases synchronization,
  CPU RAM, disk and runtime. r1/r2 timings are not a performance comparison.

## 3. Logs and checking

```bash
grep -hE 'BASELINE_(MODE|CONFIG)|LONGCAT_FIXED_V4_|LONGCAT_INTERNAL' \
  results/a3_dp4_trace_001/dumps/serve_rank*.log
```

Keep all four hosts' `results/RUN/dumps/`, launch JSONs, runtime receipts and client
manifest/responses. `/home` is NOT assumed shared. Merge the four dumps directories
into a NEW analysis directory, preserving rank-specific files. Then:

```bash
python3 compare_fixed_decode_dp4.py \
  --manifest results/a3_dp4_trace_001/client/manifest.json \
  --dump-dir /absolute/path/to/merged/dumps \
  --model /home/l00951279/longcat2/LongCat-2.0-INT8-part \
  --out results/a3_dp4_trace_001/comparison.json \
  | tee results/a3_dp4_trace_001/summary.log
```

Checks64-worker topology, full38-layer coverage, request/prefix/DP identity,
padding exclusions, repeat differences and each other DP against the first DP.
This is an A3 repeatability checker, not a cross-quantization accuracy verdict.
For A3/A5, align local Attention TP/global head, global expert ID and logical token,
not equal global TP rank or local expert slot. Compare residual/branch semantics
consistently and distinguish A3/A5 differences from actual-input CPU operator errors.

Preparation and CPU tests are complete only when `verify_dp4.py` passes on the target
container. NPU startup, actual64-rank capture, DeepEP behavior and GSM8K remain untested
until those commands are executed on free devices.
