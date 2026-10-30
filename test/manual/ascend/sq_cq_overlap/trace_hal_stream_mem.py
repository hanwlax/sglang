#!/usr/bin/env python3
"""Capture and align HAL stream_mem arguments; no torch/NPU Python imports.

Launch: python3 trace_hal_stream_mem.py -- bash ./run.sh
Read:   python3 trace_hal_stream_mem.py --analyze ./logs/sqcq_hal_...
Check:  python3 trace_hal_stream_mem.py --self-test
"""

import argparse
import csv
import hashlib
import json
import os
import platform
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent


def build_probe(directory, header_dir=None):
    if sys.platform != "linux" or platform.machine() not in ("aarch64", "x86_64"):
        raise RuntimeError("The probe requires 64-bit glibc Linux (aarch64/x86_64)")
    if not os.confstr("CS_GNU_LIBC_VERSION"):
        raise RuntimeError("GNU LD_AUDIT requires glibc")
    compiler = shlex.split(os.environ.get("CC", "cc"))
    if not compiler or shutil.which(compiler[0]) is None:
        raise RuntimeError("Install a C compiler (cc/gcc), or set CC")
    library = directory / "libsqcq_hal_trace.so"
    source = HERE / "hal_stream_mem_trace.c"
    command = compiler + [
        "-std=gnu11",
        "-O2",
        "-Wall",
        "-Wextra",
        "-Werror",
        "-shared",
        "-fPIC",
        "-Wl,-z,now",
    ]
    if header_dir is not None:
        command += ["-DSQ_CQ_USE_DRIVER_HEADERS", "-I", str(header_dir)]
    command += [str(source), "-o", str(library), "-ldl"]
    subprocess.run(command, check=True)
    return library, command


def read_events(directory):
    events = []
    malformed = []
    for path in sorted((directory / "hal").glob("hal.*.jsonl")):
        with path.open(errors="replace") as stream:
            for number, line in enumerate(stream, 1):
                try:
                    event = json.loads(line)
                    if not isinstance(event, dict):
                        raise ValueError("expected a JSON object")
                    event["source"] = f"{path.name}:{number}"
                    events.append(event)
                except (ValueError, TypeError) as error:
                    malformed.append(f"{path.name}:{number}: {error}")
    return events, malformed


def align(events, *, pid=None, device=None, stream=None):
    """Compare entry arguments, preserving uncertainty for overlapping calls.

    File order is not execution order: events carry monotonic entry/return
    timestamps. Failed or unfinished switches cannot establish a binding.
    """

    def call_key(event):
        return (
            event["pid"],
            event.get("probe", "audit"),
            event["call"],
            event.get("index", 0),
        )

    def call_label(event):
        return f"{event.get('probe', 'audit')}:{event['call']}:{event.get('index', 0)}"

    returns = {
        call_key(e): e for e in events if e.get("event") in ("switch_end", "fill_end")
    }
    switches = defaultdict(list)
    for event in events:
        if event.get("event") == "switch_begin":
            switches[(event["pid"], event["device"], event["stream"])].append(event)

    def end_of(event):
        return returns.get(call_key(event))

    rows = []
    fills = sorted(
        (e for e in events if e.get("event") == "fill_begin"),
        key=lambda e: (e["mono_ns"], e["pid"], e["call"]),
    )
    for fill in fills:
        if any(
            wanted is not None and fill[field] != wanted
            for field, wanted in (("pid", pid), ("device", device), ("stream", stream))
        ):
            continue
        key = (fill["pid"], fill["device"], fill["stream"])
        history = switches[key]
        fill_end = end_of(fill)
        finished = [
            switch
            for switch in history
            if (end := end_of(switch)) is not None
            and end["ret"] == 0
            and end["mono_ns"] <= fill["mono_ns"]
        ]
        previous = max(finished, key=lambda e: end_of(e)["mono_ns"], default=None)
        previous_end = end_of(previous) if previous else None
        overlapping = [
            switch
            for switch in history
            if switch["mono_ns"] <= (fill_end["mono_ns"] if fill_end else float("inf"))
            and ((end := end_of(switch)) is None or end["mono_ns"] > fill["mono_ns"])
        ]
        correlation = "ORDERED"
        if fill_end is None:
            correlation = "INCOMPLETE_FILL"
        elif overlapping:
            correlation = "OVERLAPPING_SWITCH"
        elif previous is None:
            correlation = "NO_PRIOR_SUCCESSFUL_SWITCH"
        else:
            # Returning last does not establish which of two concurrent
            # switches took effect last inside the driver.
            if any(
                call_key(other) != call_key(previous)
                and other["mono_ns"] <= previous_end["mono_ns"]
                and (
                    (end := end_of(other)) is None
                    or end["mono_ns"] >= previous["mono_ns"]
                )
                for other in history
            ):
                correlation = "AMBIGUOUS_SWITCH_ORDER"
            elif any(
                (end := end_of(other)) is not None
                and end["ret"] != 0
                and previous_end["mono_ns"] < end["mono_ns"] <= fill["mono_ns"]
                for other in history
            ):
                correlation = "INTERVENING_FAILED_SWITCH"

        relation = "UNKNOWN"
        if correlation == "ORDERED":
            relation = (
                "EQUAL"
                if int(previous["stream_mem"], 16) == int(fill["stream_mem"], 16)
                else "DIFFERENT"
            )
        rows.append(
            {
                "pid": fill["pid"],
                "device": fill["device"],
                "stream": fill["stream"],
                "fill_call": call_label(fill),
                "fill_tid": fill["tid"],
                "fill_wall_ns": fill["wall_ns"],
                "fill_mem": fill["stream_mem"],
                "task_cnt": fill["task_cnt"],
                "fill_ret": fill_end["ret"] if fill_end else None,
                "switch_call": call_label(previous) if previous else None,
                "switch_tid": previous["tid"] if previous else None,
                "switch_sq": previous["sq"] if previous else None,
                "switch_depth": previous["depth"] if previous else None,
                "switch_mem": previous["stream_mem"] if previous else None,
                "switch_mem_after": previous_end["stream_mem"]
                if previous_end
                else None,
                "switch_ret": previous_end["ret"] if previous_end else None,
                "gap_ns": fill["mono_ns"] - previous_end["mono_ns"]
                if previous_end
                else None,
                "relation": relation,
                "correlation": correlation,
                "overlapping_switch_calls": [call_label(e) for e in overlapping],
                "fill_source": fill["source"],
                "switch_source": previous["source"] if previous else None,
            }
        )
    return rows


def analyze(directory, **filters):
    events, malformed = read_events(directory)
    rows = align(events, **filters)
    counts = Counter(e.get("event") for e in events)
    summary = {
        "events": dict(counts),
        "relations": dict(Counter(row["relation"] for row in rows)),
        "correlations": dict(Counter(row["correlation"] for row in rows)),
        "malformed_records": malformed,
        "filters": filters,
        "rows": rows,
    }
    (directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (directory / "aligned.tsv").open("w", newline="") as output:
        if rows:
            writer = csv.DictWriter(output, fieldnames=list(rows[0]), delimiter="\t")
            writer.writeheader()
            writer.writerows(rows)
    with (directory / "problems.jsonl").open("w") as output:
        for row in rows:
            if row["relation"] != "EQUAL" or row["fill_ret"] != 0:
                output.write(json.dumps(row) + "\n")
    print(f"\nHAL capture: {directory}")
    print(f"Switch entries={counts['switch_begin']}; fill calls={counts['fill_begin']}")
    print("Relations:", summary["relations"])
    print("Correlation:", summary["correlations"])
    print("Aligned entries: aligned.tsv; exceptions/errors: problems.jsonl")
    if not rows:
        print("No matching Fill calls; aligned.tsv is empty.")
    for row in [r for r in rows if r["relation"] != "EQUAL" or r["fill_ret"] != 0][:20]:
        print(
            f"pid={row['pid']} dev={row['device']} stream={row['stream']} "
            f"switch_in={row['switch_mem']} switch_out={row['switch_mem_after']} "
            f"fill={row['fill_mem']} "
            f"ret={row['fill_ret']} {row['relation']} {row['correlation']} "
            f"fill_call={row['fill_call']}"
        )
    if malformed:
        print(
            f"WARNING: {len(malformed)} unreadable/truncated records; see summary.json"
        )
    if counts["unsupported_provider"]:
        print(
            "INCOMPLETE COVERAGE: multiple implementations of a HAL API were loaded; "
            "additional implementations were left unwrapped. See raw events."
        )
    if not counts["switch_begin"] or not counts["fill_begin"]:
        print(
            "INCOMPLETE COVERAGE: one or both HAL APIs were not intercepted. "
            "This does not prove that the application did not call them."
        )
    return summary


def run_command(command, directory, library):
    environment = os.environ.copy()
    environment["SQ_CQ_TRACE_DIR"] = str(directory / "hal")
    environment["LD_PRELOAD"] = str(library) + (
        ":" + environment["LD_PRELOAD"] if environment.get("LD_PRELOAD") else ""
    )
    environment["LD_AUDIT"] = str(library) + (
        ":" + environment["LD_AUDIT"] if environment.get("LD_AUDIT") else ""
    )
    process = None
    pending_signals = []

    def forward(signum, _frame):
        if process is None:
            pending_signals.append(signum)
            return
        try:
            os.killpg(process.pid, signum)
        except ProcessLookupError:
            pass

    # Install before announcing/starting the child, including the Popen window.
    old_handlers = {
        number: signal.signal(number, forward)
        for number in (signal.SIGINT, signal.SIGTERM)
    }
    try:
        print("Command:", shlex.join(command), flush=True)
        print("Capture:", directory, flush=True)
        with (directory / "terminal.log").open("wb") as output:
            if pending_signals:
                return -pending_signals[0]
            process = subprocess.Popen(
                command,
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            for signum in pending_signals:
                forward(signum, None)
            while chunk := process.stdout.read1(65536):
                output.write(chunk)
                output.flush()
                try:
                    sys.stdout.buffer.write(chunk)
                    sys.stdout.buffer.flush()
                except BrokenPipeError:
                    pass
        return process.wait()
    finally:
        if process is not None:
            if process.poll() is None:
                forward(signal.SIGTERM, None)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    forward(signal.SIGKILL, None)
                    process.wait()
            process.stdout.close()
        for number, handler in old_handlers.items():
            signal.signal(number, handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="New capture directory")
    parser.add_argument("--analyze", type=Path, help="Analyze an existing capture")
    parser.add_argument(
        "--header-dir", type=Path, help="Directory of ascend_hal_define.h"
    )
    parser.add_argument("--pid", type=int)
    parser.add_argument("--device", type=int)
    parser.add_argument("--stream", type=int)
    parser.add_argument(
        "--self-test", action="store_true", help="Test with a fake HAL; no NPU"
    )
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    filters = {name: getattr(args, name) for name in ("pid", "device", "stream")}
    if args.analyze:
        analyze(args.analyze.resolve(), **filters)
        return 0
    if args.self_test:
        from self_test_hal_stream_mem import self_test

        self_test()
        return 0
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("Pass a command after --, for example: -- bash ./run.sh")
    if args.output:
        directory = args.output.resolve()
        directory.mkdir(parents=True, exist_ok=False)
    else:
        parent = Path.cwd() / "logs"
        parent.mkdir(exist_ok=True)
        directory = Path(tempfile.mkdtemp(prefix="sqcq_hal_", dir=parent)).resolve()
    (directory / "hal").mkdir()
    header_dir = args.header_dir
    default_headers = Path("/usr/local/Ascend/driver/include")
    if header_dir is None and (default_headers / "ascend_hal_define.h").is_file():
        header_dir = default_headers
    if header_dir:
        header_dir = header_dir.resolve()
    library, compile_command = build_probe(directory, header_dir)
    if ":" in str(library) or any(char.isspace() for char in str(library)):
        parser.error("LD_AUDIT library path must not contain whitespace or ':'")
    metadata = {
        "command": command,
        "cwd": str(Path.cwd()),
        "machine": platform.machine(),
        "glibc": os.confstr("CS_GNU_LIBC_VERSION"),
        "compile_command": compile_command,
        "abi_source": str(header_dir)
        if header_dir
        else "public 64-bit HAL ABI snapshot",
        "source_sha256": hashlib.sha256(
            (HERE / "hal_stream_mem_trace.c").read_bytes()
        ).hexdigest(),
    }
    metadata_path = directory / "capture.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    ret = run_command(command, directory, library)
    metadata["command_return_code"] = ret
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    analyze(directory, **filters)
    return ret if ret >= 0 else 128 - ret


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"HAL trace failed: {error}", file=sys.stderr)
        sys.exit(2)
