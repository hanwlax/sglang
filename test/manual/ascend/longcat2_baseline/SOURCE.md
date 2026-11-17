# Versioned sources and reproducible runtimes

This directory is the source of the A3 full-layer TP64 / attention-DP4 / EP64
baseline deployed on 2026-11-17. Read README.md for launch/evaluation/capture commands.
The baseline has passed CPU synthetic checks and argument dry-runs, not an actual
64-rank NPU startup or GSM8K evaluation.

## What is versioned

- The branch inherits SGLang c0e9d9a0cc6b1a37e41677faf0c961fedc668af7.
- The deployment, replay, comparison and runtime-generation Python/Bash files are
  byte-identical to the prepared blue-zone files; deployment_manifest.json records
  their hashes. audit_model.py came from the preceding deployment preparation.
- prepare_runtime_dp4.py contains the exact source edits used for instrumentation.
  Its trace mode changes tp_worker.py, longcat_flash.py, the NPU MLA module and the
  Ascend attention backend, and copies the two diagnostic helper modules.
- eval mode extracts unchanged inference sources at the pinned revision. Both modes
  make a separate evaluator copy changing only the two report output paths.
- Runtime trees, model weights, captures, logs and evaluation outputs are not Git
  artifacts. The one-shot historical install_dp4.py updater is intentionally omitted;
  it only knew how to replace the previous launcher by its old hash.
- fixtures/control_852.json contains the exact prompt token IDs for replay, without
  retaining a model response. The client expects the legacy response/choices nesting.
  fixtures/metadata.json records its ID hash and fixed prefix.

The repository's regular python/ tree does not automatically enable the DP4 hooks.
To reproduce the deployed diagnostic source, run the generator below. Editing a
shared checkout alone does not modify an already generated private runtime.

## Prepare in a new experiment directory

Inside a matching Ascend container with CPU torch available, set SOURCE to this
branch's actual checkout. Do not overwrite an existing results directory.

```bash
SOURCE=/absolute/path/to/this/sglang-checkout
BUNDLE=/home/hanwlax/workspace/progress/longcat2.0/a3_baseline_new
mkdir "$BUNDLE"
cp -a "$SOURCE/test/manual/ascend/longcat2_baseline/." "$BUNDLE/"
cd "$BUNDLE"
python3 prepare_runtime_dp4.py --source "$SOURCE" --mode eval --out runtime_dp4_eval
python3 prepare_runtime_dp4.py --source "$SOURCE" --mode trace --out runtime_dp4_trace
python3 verify_dp4.py
bash run_a3_full_baseline.sh 0 precheck eval --dry-run
bash run_a3_full_baseline.sh 0 precheck trace --dry-run
bash run_gsm8k.sh precheck --dry-run
```

All four hosts need the same bundle and matching private runtimes. Use node ranks
0/1/2/3 on 209/212/216/217. The model path, interfaces, SDK, torch-npu and operator
packages must exist in the container; Git source alone does not recreate them.
The pinned commit must exist locally (a shallow branch checkout may need fetching
more history). The generator deliberately refuses an existing output directory.

Replay can use the bundled input rather than the historical container /tmp file:

```bash
bash run_fixed_prefix_dp4.sh a3_dp4_trace_001 "$PWD/fixtures/control_852.json"
```

## Change ownership and regeneration

| Change | Owning file |
| --- | --- |
| topology, launch env, graph/NEXTN, memory, model path | run_a3_full_baseline.sh |
| dataset, examples, sampling, output locations | run_gsm8k.sh |
| effective server topology validation | check_server_dp4.py |
| fixed IDs, routed DP, length/repeat/timeout | replay_fixed_decode_dp4.py |
| source injection location or captured arguments | prepare_runtime_dp4.py |
| RID binding, forced first token, row mapping, layer boundaries | longcat_fixed_decode_dp4.py |
| internal tensor filtering and immutable CPU snapshots | capture_internal_dp4.py |
| receipt/coverage checks and repeat/DP comparisons | compare_fixed_decode_dp4.py |
| CPU regression checks | verify_dp4.py |

After changing diagnostic helpers or injection logic, build a NEW trace runtime,
run the CPU checks against the intended bundle, and restart only the owned service
with RUNTIME pointing to that new runtime. Both helpers are copied into runtime/python;
editing their bundle copies alone has no effect on an already running service.
verify_dp4.py currently checks the conventional runtime_dp4_eval/runtime_dp4_trace
paths; use a fresh bundle for validation of a revision. Do not loosen assertions to
make an unexplained topology/layout mismatch disappear.

Changing the intended SGLang implementation requires changing the generator's REV,
reviewing its exact-match anchors and testing both clean/trace snapshots. Merely
committing a model file on this branch will not change a generator pinned to c0e9d9a0.

This tooling is A3 DP4-specific. A5 needs TP32/DP2/EP32 routing, receipts, row metadata
and FP8/model/network settings adapted together; changing only --tp-size is insufficient.
Cross-platform numerical comparison of internal dumps is a next step, not a feature
of the same-platform repeat comparator in this directory.
