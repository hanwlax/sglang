# NPU SQ/CQ: target/draft update/replay overlap A/B

Diagnostic branch: `hanwlax/sglang:sq-cq`.
Initial base: upstream `main` at
`8fd181eaf4137b0ba52de8f3a34095f1933b3bc7` (2026-10-30).

## Collect host, device and HAL logs with one command

From the directory containing the existing `run.sh`, run:

```bash
python3 /path/to/sglang/test/manual/ascend/sq_cq_overlap/collect_sqcq.py -- bash ./run.sh
# Forward the existing positional arguments if needed:
python3 /path/to/sglang/test/manual/ascend/sq_cq_overlap/collect_sqcq.py -- bash ./run.sh 10086 4 0 0
```

The script compiles the HAL probe locally, launches `run.sh`, collects logs, and
creates `logs/sqcq_full_*.tar.gz` plus a `.sha256` file. It requires Python 3.9+,
a C compiler and 64-bit glibc Linux. Run with the same privileges/environment as
the reproduction. Reading `/root/ascend`, kernel logs and exporting device logs
usually requires root; unavailable sources are recorded instead of discarded.

The actual plain `python3 -m sglang.launch_server` invocation receives:

```bash
ASCEND_HOST_LOG_FILE_NUM=1000
ASCEND_GLOBAL_LOG_LEVEL=0
ASCEND_SLOG_PRINT_TO_STDOUT=0
ASCEND_PROCESS_LOG_PATH=<new capture directory>/cann
# ASCEND_MODULE_LOG_LEVEL is unset to prevent old INFO overrides.
```

`ASCEND_SLOG_PRINT_TO_STDOUT=0` enables file output; `1` would prevent CANN log
files from being saved. A temporary `BASH_ENV` shell function enforces these
settings even if `run.sh` changes them or clears `LD_PRELOAD`/`LD_AUDIT`.
It keeps the original script, arguments, working directory and overlap flags.
The known `run.sh` uses plain `python3`; an absolute Python executable,
`env python3` or a replacement shell function can bypass this guard. An absent
server launch marker is reported as incomplete coverage.

Default stopping behavior: a nonzero `halStreamTaskFill` return, an actual
SQ/CQ ERROR line, or a scheduler exception triggers a three-second tail capture
and then stops only the process group launched by this script. INFO format
registration and `error=0` do not trigger it. The fallback reproduction timeout
is 900 seconds. Use `--run-timeout 1800` for a longer startup. Ctrl+C stops the
reproduction and continues collection; further Ctrl+C during export/packaging
does not discard the evidence. To stop the reproduction manually without an
error/timeout trigger:

```bash
python3 /path/to/sglang/test/manual/ascend/sq_cq_overlap/collect_sqcq.py \
  --no-auto-stop --run-timeout 0 -- bash ./run.sh
```

The archive retains full file contents, without grep filtering or line limits:

| Path | Evidence |
| --- | --- |
| `cann/` | This run's DEBUG CANN application logs in an independent directory. |
| `host_logs/` | Existing `$HOME/ascend`, `/root/ascend`, `/var/log/npu/slog`, `/var/log/ascend_seclog`, inherited log path and system log files. These can include earlier runs. Source paths and copy failures are recorded. |
| `msnpureport/` | Device export, tool help and complete command output; also snapshots `/var/log/npu/report` when present. |
| `hal_trace/` | Full terminal output, per-PID HAL entry/return JSONL, `aligned.tsv`, `summary.json`, and `problems.jsonl`. Alignment uses PID/device/stream. |
| `metadata/` | Original `run.sh`, launch guard, actual serving Python/packages/import path and Git commit when available, driver/build/firmware information, `npu-smi`, `dmesg` and kernel journal. |
| `collection.json`, `trigger.json` | Exact command, stage results, missing evidence and stopping reason when triggered. |

`msnpureport` is located from PATH or
`/usr/local/Ascend/driver/tools/msnpureport`. Override with
`--msnpureport /path/to/msnpureport`. The script uses the supported `-f` export,
or `report -t 0` if help only advertises the newer interface. Containers detected
via `/.dockerenv` or `/run/.containerenv` receive `--docker` if supported;
`--container yes` forces it when detection is insufficient. The default export
timeout is 180 seconds (`--msnpureport-timeout` overrides it). This command
exports existing device logs; it does not change persistent device log levels.
Add custom host locations with repeatable `--host-log-root /path`.
Only successful help output is used to detect the newer CLI. Loader errors
such as `GLIBC_2.38 not found` are retained and mark device export incomplete;
changing command-line options cannot fix a binary that cannot start. Export
device logs on a host with a compatible runtime, or pass a compatible exporter
with `--msnpureport`. The host/HAL capture remains available in either case.

The collector returns `0` when its coverage checks pass and `2` for incomplete
collection. The model's return code is separate in `collection.json`; a model
failure is expected in this reproduction. Missing tools, export failure/timeout,
copy errors, unreadable kernel logs or missing HAL/CANN coverage still produce
an archive and explicit `incomplete_reasons`. Inspect the recorded export
output for per-device details. A 1000-file limit can still rotate: reaching it
is flagged, and files already rotated before collection cannot be recovered.
No existing logs are deleted. Full DEBUG collection needs adequate free disk
space; disk exhaustion or archive failure is reported, retaining the raw capture
directory and any `.tar.gz.partial` for inspection.

CPU-only self-checks (fake HAL/tools and private log fixtures, no NPU required):

```bash
python3 test/manual/ascend/sq_cq_overlap/self_test_collect_sqcq.py
python3 test/manual/ascend/sq_cq_overlap/trace_hal_stream_mem.py --self-test
```

## Switch semantics

All three variables default to `1`, preserving upstream overlap. They are read
when each `NPUCudaGraphBackend` is constructed; restart the server after changing
them. `ModelRunner.is_draft_worker` selects the role, including DSpark's draft
decode runner. The target switch also controls ordinary non-speculative models.

| Variable | Meaning |
| --- | --- |
| `SGLANG_ENABLE_NPU_GRAPH_UPDATE_REPLAY_OVERLAP` | Global gate: `0` serializes both roles, regardless of role settings. |
| `SGLANG_ENABLE_NPU_TARGET_GRAPH_UPDATE_REPLAY_OVERLAP` | Target role gate, effective when the global gate is `1`. |
| `SGLANG_ENABLE_NPU_DRAFT_GRAPH_UPDATE_REPLAY_OVERLAP` | Draft role gate, effective when the global gate is `1`. |

Effective overlap is `global AND role`. Serial mode waits for the host
`graph.update(...)` call to return before calling `graph.replay()`. Update still
runs on the same device-bound executor. The preceding replay fence and the
following event recording remain unchanged. Direct `replay()` without an input
update is unaffected.

This controls host graph update/replay overlap. Other stream and scheduler
overlap settings retain their existing meaning.

## Prepare run.sh

Use the existing K3 + DSpark `run.sh` with its model paths, TP/DP, attention
options and other launch arguments unchanged. Replace its hardcoded
`export SGLANG_ENABLE_NPU_GRAPH_UPDATE_REPLAY_OVERLAP=0` with:

```bash
export SGLANG_ENABLE_NPU_GRAPH_UPDATE_REPLAY_OVERLAP=${SGLANG_ENABLE_NPU_GRAPH_UPDATE_REPLAY_OVERLAP:-1}
export SGLANG_ENABLE_NPU_TARGET_GRAPH_UPDATE_REPLAY_OVERLAP=${SGLANG_ENABLE_NPU_TARGET_GRAPH_UPDATE_REPLAY_OVERLAP:-1}
export SGLANG_ENABLE_NPU_DRAFT_GRAPH_UPDATE_REPLAY_OVERLAP=${SGLANG_ENABLE_NPU_DRAFT_GRAPH_UPDATE_REPLAY_OVERLAP:-1}
```

Run from the checkout/container where `hanwlax:sq-cq` is installed. Check the
import path and commit once before the comparisons:

```bash
git rev-parse HEAD
python3 -c 'import sglang; print(sglang.__file__)'
```

Keep DSpark enabled for both role comparisons below. Stop the preceding server
and wait for its workers to exit before starting the next case. Preserve the
full startup/driver log for each case separately.

## A: target overlap enabled; draft serialized

```bash
SGLANG_ENABLE_NPU_GRAPH_UPDATE_REPLAY_OVERLAP=1 \
SGLANG_ENABLE_NPU_TARGET_GRAPH_UPDATE_REPLAY_OVERLAP=1 \
SGLANG_ENABLE_NPU_DRAFT_GRAPH_UPDATE_REPLAY_OVERLAP=0 \
bash ./run.sh
```

Expected initialization log settings: `role=target enabled=True` and
`role=draft enabled=False`.

## B: target serialized; draft overlap enabled

```bash
SGLANG_ENABLE_NPU_GRAPH_UPDATE_REPLAY_OVERLAP=1 \
SGLANG_ENABLE_NPU_TARGET_GRAPH_UPDATE_REPLAY_OVERLAP=0 \
SGLANG_ENABLE_NPU_DRAFT_GRAPH_UPDATE_REPLAY_OVERLAP=1 \
bash ./run.sh
```

Expected initialization log settings: `role=target enabled=False` and
`role=draft enabled=True`.

## Retained global A/B

```bash
# Both roles serialized. The global gate overrides both role gates.
SGLANG_ENABLE_NPU_GRAPH_UPDATE_REPLAY_OVERLAP=0 \
SGLANG_ENABLE_NPU_TARGET_GRAPH_UPDATE_REPLAY_OVERLAP=1 \
SGLANG_ENABLE_NPU_DRAFT_GRAPH_UPDATE_REPLAY_OVERLAP=1 \
bash ./run.sh

# Both roles overlap: upstream behavior.
SGLANG_ENABLE_NPU_GRAPH_UPDATE_REPLAY_OVERLAP=1 \
SGLANG_ENABLE_NPU_TARGET_GRAPH_UPDATE_REPLAY_OVERLAP=1 \
SGLANG_ENABLE_NPU_DRAFT_GRAPH_UPDATE_REPLAY_OVERLAP=1 \
bash ./run.sh
```

Initialization logs print the role, effective setting, global setting, role
setting, runner class and device ID. For example:

```text
NPU graph update/replay overlap: role=draft enabled=False global=True role_enabled=False runner=NPUDecodeGraphRunner device=0
```

## Interpretation

The reported incident establishes that global serialization avoids startup
SQ/CQ failure in the user's K3 + DSpark configuration. The role comparisons
above have not yet been executed on that NPU environment.

If A succeeds and B fails, prioritize draft update/replay concurrency. If A
fails and B succeeds, prioritize target concurrency. If both mixed cases fail
while global serialization succeeds, retain cross-graph resource interaction
and independent failures in both roles as hypotheses. These outcomes localize
the trigger; they do not identify a specific CANN/driver root cause.

## Capture the actual HAL stream_mem arguments

From the directory containing your existing `run.sh`, run:

```bash
python3 /path/to/sglang/test/manual/ascend/sq_cq_overlap/trace_hal_stream_mem.py \
  -- bash ./run.sh
```

The wrapper builds its probe on that machine, inherits your overlap/CANN
settings, launches the original script, and saves unfiltered stdout/stderr in
`logs/sqcq_hal_*/terminal.log`. HAL entry/return records go directly to separate
`hal/hal.<PID>.jsonl` files. After failure or a normal Ctrl+C stop, it writes
`aligned.tsv`, `summary.json` and `problems.jsonl`, then returns the launched
command's exit status. If `run.sh` uses a pipeline without `pipefail`, that
status can reflect its final `tee`; inspect HAL return codes in the report.

Requirements: Python 3.9+, a C compiler (`cc`/`gcc`), and 64-bit glibc Linux
(aarch64 or x86_64). No torch import, root privileges or driver-library changes
are required. It uses both `LD_PRELOAD` and GNU `LD_AUDIT` to cover PLT/GOT calls
and explicit `dlsym` lookups, including `RTLD_DEEPBIND`. Existing preload/audit
entries are preserved. A script that clears these variables may prevent
interception; missing coverage is explicitly reported.
If multiple implementations of one HAL API are loaded in the same process,
additional providers are left unwrapped and coverage is marked incomplete.

The public 64-bit `sq_switch_stream_info` ABI is 24 bytes with `sq_id`,
`stream_id`, `sq_depth` at offsets 0/4/8 and `stream_mem` at offset 16. If
`/usr/local/Ascend/driver/include/ascend_hal_define.h` exists, the compiler checks
the installed declaration against these offsets. Otherwise it uses the public
declaration. For headers at another location, use `--header-dir /path/to/include`.
An incompatible installed layout fails compilation before the server starts.

An explicit output directory must be new:

```bash
python3 /path/to/sglang/test/manual/ascend/sq_cq_overlap/trace_hal_stream_mem.py \
  --output "$PWD/logs/hal-overlap-on" -- bash ./run.sh 10086 4 0 0
```

Re-analyze a captured run, optionally selecting a process/device/stream:

```bash
python3 /path/to/sglang/test/manual/ascend/sq_cq_overlap/trace_hal_stream_mem.py \
  --analyze ./logs/hal-overlap-on --pid 98598 --device 0 --stream 22
```

Use the PID of the new run, rather than assuming the historical PID 98598.
Filtered analysis overwrites the generated summaries; it retains all raw logs.
You can analyze from another terminal while the server is still running;
unfinished calls are provisional, so re-analyze after the failure/stop.

`aligned.tsv` compares the entry-time `info[i].stream_mem` of the most recent
observed successful Switch with Fill's third argument, using exactly the same
PID, device and stream. It also records SQ ID/depth, both TIDs, call IDs, return
codes, timing and the Switch array value after return. Correlation is based on
monotonic entry/return timestamps, not file-line order.

- `EQUAL + ORDERED`: the two observed input addresses match.
- `DIFFERENT + ORDERED`: the two observed input addresses differ.
- `UNKNOWN`: missing history, an incomplete call, a failed intervening switch,
  or overlapping calls prevent a definite sequential comparison. Raw records
  and overlapping call IDs remain available.

This compares observed API inputs; it does not read the driver's cached
`res_addr` or track every resource allocation/free. Different inputs can also
represent legal offset fills. If the Switch array differs after return, it
changed during that interval; the probe alone cannot identify which thread
wrote it. A driver error despite `EQUAL` does not establish driver fault without
checking its cached state/lifecycle. Combine the HAL log with the original
`res_addr` error in `terminal.log` or the retained CANN logs.

Logging adds overhead and can change race timing. No probe mutex spans a HAL
call, and arguments/return codes are forwarded unchanged. A run that succeeds
only with tracing does not establish that the original issue is fixed.

The native self-check needs no NPU:

```bash
python3 test/manual/ascend/sq_cq_overlap/trace_hal_stream_mem.py --self-test
```

It checks interception through PLT, GOT, mixed calls and `dlopen`/`dlsym` with DEEPBIND using
a fake HAL. Cases cover equal/different addresses, a deliberately stale fake
driver cache, failed switches, array mutation, batched entries, concurrent
calls and fork/PID isolation. It also checks Python ctypes, multiple providers,
signal/exit forwarding and installed-header ABI rejection. This is not
validation on the actual NPU driver.
