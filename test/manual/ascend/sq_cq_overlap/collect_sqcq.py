#!/usr/bin/env python3
"""One-command DEBUG + host/device/HAL capture, ending in a tar.gz archive.

python3 collect_sqcq.py -- bash ./run.sh [port tp flashmla needs_cpu_seq_lens]
"""

import argparse
import hashlib
import json
import os
import re
import shlex
import shutil
import signal
import stat
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from trace_hal_stream_mem import analyze, build_probe, run_command

ERROR = re.compile(
    rb"^\[ERROR\][^\r\n]*(?:Fill task exceed stream queue|Build SQ/CQ failed|bind sq cq failed)"
    rb"|^\[[^\r\n]*\]\s+Scheduler hit an exception",
    re.IGNORECASE | re.MULTILINE,
)


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def command_log(command, destination, *, cwd=None, timeout=30):
    """Bounded read-only collection command; retain stderr and any failure."""
    record = {"command": command, "cwd": str(cwd) if cwd else None}
    try:
        with destination.open("wb") as output:
            process = subprocess.Popen(
                command,
                cwd=cwd,
                stdout=output,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                record["return_code"] = process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                record["timeout"] = timeout
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                record["return_code"] = process.wait()
    except OSError as error:
        record["error"] = str(error)
        destination.write_text(str(error) + "\n")
    record["status"] = (
        "ok" if record.get("return_code") == 0 and "timeout" not in record else "failed"
    )
    return record


def snapshot(source, destination, capture_root):
    """Copy all regular files, dereference log symlinks, avoid cycles/self-copy."""
    result = {
        "source": str(source),
        "destination": str(destination),
        "files": 0,
        "bytes": 0,
        "errors": [],
        "skipped": [],
    }
    try:
        if not source.exists():
            result["status"] = "missing"
            return result
    except OSError as error:
        result["status"] = "partial"
        result["errors"].append({"path": str(source), "error": str(error)})
        return result

    def copy(path, target, ancestors):
        try:
            resolved = path.resolve()
            if resolved == capture_root or capture_root in resolved.parents:
                result["skipped"].append(
                    {"path": str(path), "reason": "capture output"}
                )
                return
            info = path.stat()
            identity = (info.st_dev, info.st_ino)
            if stat.S_ISDIR(info.st_mode):
                if identity in ancestors:
                    result["skipped"].append(
                        {"path": str(path), "reason": "symlink cycle"}
                    )
                    return
                target.mkdir(parents=True, exist_ok=True)
                for child in path.iterdir():
                    copy(child, target / child.name, ancestors | {identity})
            elif stat.S_ISREG(info.st_mode):
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)
                result["files"] += 1
                result["bytes"] += target.stat().st_size
            else:
                result["skipped"].append(
                    {"path": str(path), "reason": "not a regular log file"}
                )
        except (OSError, RuntimeError) as error:
            result["errors"].append({"path": str(path), "error": str(error)})

    copy(source, destination, set())
    result["status"] = "partial" if result["errors"] else "ok"
    return result


def write_launch_guard(root, capture_env):
    """Enforce settings at the known run.sh's plain `python3` invocation.

    The shell function survives PATH changes from CANN set_env and does not
    change run.sh itself, its arguments, working directory or overlap settings.
    """
    guard = root / "metadata" / "bash_env.sh"
    previous = os.environ.get("BASH_ENV", "")
    prefix = (
        f"if [[ -r {shlex.quote(os.path.expandvars(os.path.expanduser(previous)))} ]]; then source {shlex.quote(os.path.expandvars(os.path.expanduser(previous)))}; fi\n"
        if previous
        else ""
    )
    assignments = " ".join(
        f"{key}={shlex.quote(value)}" for key, value in capture_env.items()
    )
    helper = root / "metadata" / "python_metadata.py"
    helper.write_text("""import importlib.metadata as m
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
sys.path.insert(0, os.getcwd())
record = {"executable": sys.executable, "python": sys.version, "argv": sys.argv[2:], "packages": {}}
record["logging_environment"] = {key: os.environ.get(key) for key in (
    "ASCEND_GLOBAL_LOG_LEVEL", "ASCEND_HOST_LOG_FILE_NUM",
    "ASCEND_SLOG_PRINT_TO_STDOUT", "ASCEND_PROCESS_LOG_PATH", "ASCEND_MODULE_LOG_LEVEL",
)}
record["overlap_environment"] = {key: os.environ.get(key) for key in (
    "SGLANG_ENABLE_NPU_GRAPH_UPDATE_REPLAY_OVERLAP",
    "SGLANG_ENABLE_NPU_TARGET_GRAPH_UPDATE_REPLAY_OVERLAP",
    "SGLANG_ENABLE_NPU_DRAFT_GRAPH_UPDATE_REPLAY_OVERLAP",
)}
for name in ["torch", "torch-npu", "sglang", "triton-ascend"]:
    try:
        record["packages"][name] = m.version(name)
    except m.PackageNotFoundError:
        record["packages"][name] = None
spec = importlib.util.find_spec("sglang")
record["sglang_origin"] = spec.origin if spec else None
if spec and spec.origin:
    for parent in Path(spec.origin).resolve().parents:
        if (parent / ".git").exists():
            record["sglang_git_root"] = str(parent)
            try:
                record["sglang_commit"] = subprocess.check_output(
                    ["git", "-C", str(parent), "rev-parse", "HEAD"],
                    stderr=subprocess.STDOUT, text=True, timeout=5,
                ).strip()
                dirty = subprocess.run(
                    ["git", "-C", str(parent), "diff", "--quiet", "HEAD"],
                    capture_output=True, timeout=5,
                )
                if dirty.returncode in (0, 1):
                    record["sglang_tracked_changes"] = dirty.returncode == 1
                else:
                    record["sglang_git_error"] = dirty.stderr.decode(errors="replace")
            except (OSError, subprocess.SubprocessError) as error:
                record["sglang_git_error"] = str(error)
            break
with open(sys.argv[1], "a") as out:
    out.write(json.dumps(record) + "\\n")
""")
    marker = root / "metadata" / "launch_guard.log"
    metadata = root / "metadata" / "python.jsonl"
    helper_errors = root / "metadata" / "python_metadata_errors.log"
    probe = shlex.quote(str(root / "hal_trace" / "libsqcq_hal_trace.so"))
    guard.write_text(
        prefix
        + f"""
python3() {{
    local _sq_probe={probe}
    local _sq_preload="${{LD_PRELOAD:-}}" _sq_audit="${{LD_AUDIT:-}}"
    if [[ ":${{_sq_preload// /:}}:" != *":${{_sq_probe}}:"* ]]; then
        _sq_preload="${{_sq_probe}}${{_sq_preload:+:${{_sq_preload}}}}"
    fi
    if [[ ":${{_sq_audit}}:" != *":${{_sq_probe}}:"* ]]; then
        _sq_audit="${{_sq_probe}}${{_sq_audit:+:${{_sq_audit}}}}"
    fi
    if [[ "${{1-}}" == -m && "${{2-}}" == sglang.launch_server ]]; then
        printf 'server_launch=1 DEBUG=0 FILE_NUM=1000 FILE_OUTPUT=0 python=%s\\n' "$(type -P python3)" >> {shlex.quote(str(marker))}
    fi
    command env -u ASCEND_MODULE_LOG_LEVEL {assignments} python3 {shlex.quote(str(helper))} {shlex.quote(str(metadata))} "$@" 2>> {shlex.quote(str(helper_errors))} || true
    command env -u ASCEND_MODULE_LOG_LEVEL {assignments} LD_PRELOAD="$_sq_preload" LD_AUDIT="$_sq_audit" python3 "$@"
}}
"""
    )
    return guard


class ReproMonitor:
    """Watch newly written logs; stop only the process group started by us."""

    def __init__(self, root, *, timeout, post_error, auto_stop):
        self.root = root
        self.timeout = timeout
        self.post_error = post_error
        self.auto_stop = auto_stop
        self.stop = threading.Event()
        self.thread = None
        self.process = None
        self.trigger = None
        self.offsets = {}
        self.tails = {}
        self.manual_signal = None

    def start(self, process):
        self.process = process
        for number in (signal.SIGINT, signal.SIGTERM):
            previous = signal.getsignal(number)

            def relay(signum, frame, handler=previous):
                self.manual_signal = signum
                handler(signum, frame)

            signal.signal(number, relay)
        self.thread = threading.Thread(target=self.watch, args=(process,), daemon=True)
        self.thread.start()

    def scan(self):
        files = [self.root / "hal_trace" / "terminal.log"]
        files += list((self.root / "hal_trace" / "hal").glob("*.jsonl"))
        files += list((self.root / "cann").rglob("*.log"))
        for path in files:
            try:
                position = self.offsets.get(path, 0)
                if path.stat().st_size < position:
                    position = 0
                with path.open("rb") as stream:
                    stream.seek(position)
                    data = stream.read(1024 * 1024)
                    self.offsets[path] = stream.tell()
                chunk = self.tails.get(path, b"") + data
                self.tails[path] = chunk[-512:]
                match = ERROR.search(chunk)
                if match:
                    return {
                        "source": str(path.relative_to(self.root)),
                        "reason": match.group().decode(),
                    }
                if path.suffix == ".jsonl":
                    if re.search(
                        rb'"event":"fill_end"[^\n]*"ret":-?[1-9][0-9]*', chunk
                    ):
                        return {
                            "source": str(path.relative_to(self.root)),
                            "reason": "nonzero HAL Fill return",
                        }
            except OSError:
                continue
        return None

    def watch(self, process):
        started = time.monotonic()
        deadline = None
        sent = 0
        while not self.stop.wait(0.25):
            now = time.monotonic()
            if self.trigger is None:
                found = self.scan() if self.auto_stop else None
                if self.manual_signal is not None:
                    self.trigger = {
                        "reason": "user stopped reproduction",
                        "signal": self.manual_signal,
                    }
                    deadline = now + 10
                    sent = 1
                elif found:
                    self.trigger = found
                    deadline = now + self.post_error
                elif self.timeout and now - started >= self.timeout:
                    self.trigger = {
                        "reason": "reproduction timeout",
                        "seconds": self.timeout,
                    }
                    deadline = now
                if self.trigger:
                    dump(self.root / "trigger.json", self.trigger)
                    print(
                        f"\n[collect] {self.trigger}; preserving trailing logs before stopping this run",
                        flush=True,
                    )
            if deadline is not None and now >= deadline:
                # A shell can exit before a worker holding stdout exits. Keep
                # signaling our group until run_command drains it or it is gone.
                number = (signal.SIGINT, signal.SIGTERM, signal.SIGKILL)[min(sent, 2)]
                try:
                    os.killpg(process.pid, number)
                except ProcessLookupError:
                    return
                sent += 1
                deadline = now + 10

    def finish(self):
        self.stop.set()
        if self.thread is not None:
            self.thread.join()
        if self.process is None:
            return {"status": "not_started"}
        result = {"process_group": self.process.pid, "signals": []}
        try:
            # A pipeline's tee/shell can exit while NPU workers ignore SIGINT.
            # Finish this run's remaining workers before exporting driver logs.
            os.killpg(self.process.pid, signal.SIGTERM)
            result["signals"].append("SIGTERM")
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                os.killpg(self.process.pid, 0)
                time.sleep(0.1)
            os.killpg(self.process.pid, signal.SIGKILL)
            result["signals"].append("SIGKILL")
            result["status"] = "kill_sent"
        except ProcessLookupError:
            result["status"] = "gone"
        except OSError as error:
            result.update(status="failed", error=str(error))
        return result


def collect_host(root, extra_roots):
    sources = [
        Path.home() / "ascend",
        Path("/root/ascend"),
        Path("/var/log/npu/slog"),
        Path("/var/log/ascend_seclog"),
    ]
    inherited = os.environ.get("ASCEND_PROCESS_LOG_PATH")
    if inherited:
        sources.append(Path(inherited).expanduser())
    sources.extend(extra_roots)
    seen = set()
    records = []
    for index, source in enumerate(sources):
        try:
            resolved = source.resolve()
        except OSError:
            resolved = source.absolute()
        if resolved in seen:
            continue
        seen.add(resolved)
        records.append(snapshot(source, root / "host_logs" / f"source_{index}", root))
    log_root = Path("/var/log")
    for prefix in ("syslog", "kern.log", "messages", "dmesg"):
        for path in sorted(log_root.glob(prefix + "*")):
            records.append(
                snapshot(path, root / "host_logs" / "system" / path.name, root)
            )
    return records


def export_device(root, executable, *, container, timeout):
    directory = root / "msnpureport"
    directory.mkdir()
    tool = executable or shutil.which("msnpureport")
    if not tool:
        candidate = Path("/usr/local/Ascend/driver/tools/msnpureport")
        if candidate.is_file():
            tool = str(candidate)
    if not tool:
        return {"status": "missing", "reason": "msnpureport executable not found"}
    tool = str(Path(shutil.which(tool) or tool).resolve())
    help_status = command_log([tool, "--help"], directory / "help.txt", cwd=directory)
    help_text = (directory / "help.txt").read_text(errors="replace")
    flags = ["-f"]
    if (
        "--force" not in help_text
        and not re.search(r"\s-f(?:[ ,\t]|$)", help_text)
        and "report" in help_text
    ):
        flags = ["report", "-t", "0"]
    docker = Path("/.dockerenv").exists() or Path("/run/.containerenv").exists()
    if container == "yes" or (
        container == "auto" and docker and "--docker" in help_text
    ):
        flags.append("--docker")
    print(f"[collect] exporting device logs: {shlex.join([tool] + flags)}", flush=True)
    result = command_log(
        [tool] + flags, directory / "command.log", cwd=directory, timeout=timeout
    )
    result["help"] = help_status
    # Some HDK releases write here rather than the command's working directory.
    result["default_report_snapshot"] = snapshot(
        Path("/var/log/npu/report"), directory / "default_report", root
    )
    payload = [
        p
        for p in directory.rglob("*")
        if p.is_file() and p.name not in ("help.txt", "command.log")
    ]
    result["payload_files"] = len(payload)
    result["payload_bytes"] = sum(p.stat().st_size for p in payload)
    if result["status"] == "ok" and not payload:
        result["status"] = "empty"
    return result


def collect_metadata(root):
    result = {}
    directory = root / "metadata"
    for name, command in {
        "uname": ["uname", "-a"],
        "npu_smi": ["npu-smi", "info"],
        "npu_board_0": ["npu-smi", "info", "-t", "board", "-i", "0"],
        "dmesg": ["dmesg", "-T"],
        "kernel_journal": [
            "journalctl",
            "-k",
            "--since",
            json.loads((root / "collection.json").read_text())["started_utc"],
            "--no-pager",
        ],
    }.items():
        result[name] = command_log(command, directory / (name + ".txt"))
    for source in [
        Path("/etc/ascend_install.info"),
        Path("/usr/local/Ascend/driver/version.info"),
        Path("/usr/local/Ascend/driver/build.info"),
        Path("/usr/local/Ascend/firmware/version.info"),
    ]:
        result[str(source)] = snapshot(
            source, directory / (source.parent.name + "_" + source.name), root
        )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, help="New capture directory; tar.gz is saved beside it"
    )
    parser.add_argument(
        "--run-timeout",
        type=float,
        default=900,
        help="Stop this run after N seconds; 0 disables",
    )
    parser.add_argument("--post-error-seconds", type=float, default=3)
    parser.add_argument(
        "--no-auto-stop",
        action="store_true",
        help="Wait for exit/Ctrl+C instead of stopping on SQ/CQ failure",
    )
    parser.add_argument(
        "--msnpureport",
        help="Path to msnpureport if not installed at its default location",
    )
    parser.add_argument("--msnpureport-timeout", type=float, default=180)
    parser.add_argument("--container", choices=("auto", "yes", "no"), default="auto")
    parser.add_argument("--header-dir", type=Path)
    parser.add_argument("--host-log-root", type=Path, action="append", default=[])
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        command = ["bash", "./run.sh"]
    if command[0] != "bash":
        parser.error("Use -- bash ./run.sh [arguments] so the logging guard can run")
    if (
        min(args.run_timeout, args.post_error_seconds) < 0
        or args.msnpureport_timeout <= 0
    ):
        parser.error(
            "timeouts must be nonnegative; msnpureport timeout must be positive"
        )
    if args.output:
        root = args.output.resolve()
        if any(
            Path(str(root) + suffix).exists()
            for suffix in (".tar.gz", ".tar.gz.partial", ".tar.gz.sha256")
        ):
            parser.error("Output archive already exists; choose a fresh --output")
        root.mkdir(parents=True, exist_ok=False)
    else:
        parent = Path.cwd() / "logs"
        parent.mkdir(exist_ok=True)
        root = Path(tempfile.mkdtemp(prefix="sqcq_full_", dir=parent)).resolve()
    for child in ("metadata", "cann", "host_logs", "hal_trace/hal"):
        (root / child).mkdir(parents=True)
    capture_env = {
        "ASCEND_HOST_LOG_FILE_NUM": "1000",
        "ASCEND_GLOBAL_LOG_LEVEL": "0",
        "ASCEND_SLOG_PRINT_TO_STDOUT": "0",
        "ASCEND_PROCESS_LOG_PATH": str(root / "cann"),
    }
    library_path = root / "hal_trace" / "libsqcq_hal_trace.so"
    enforced_env = {**capture_env, "SQ_CQ_TRACE_DIR": str(root / "hal_trace" / "hal")}
    guard = write_launch_guard(root, enforced_env)
    metadata = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "cwd": str(Path.cwd()),
        "command": command,
        "requested_logging": capture_env,
        "collector_python": sys.executable,
        "stages": {},
        "free_bytes_at_start": shutil.disk_usage(root).free,
    }
    for item in command[1:]:
        source = Path(item)
        if source.is_file() and source.suffix == ".sh":
            shutil.copy2(source, root / "metadata" / source.name)
    dump(root / "collection.json", metadata)
    monitor = ReproMonitor(
        root,
        timeout=args.run_timeout,
        post_error=args.post_error_seconds,
        auto_stop=not args.no_auto_stop,
    )
    try:
        if ":" in str(library_path) or any(
            char.isspace() for char in str(library_path)
        ):
            raise RuntimeError(
                "Capture path must not contain whitespace or ':' for LD_AUDIT"
            )
        headers = args.header_dir
        default_headers = Path("/usr/local/Ascend/driver/include")
        if headers is None and (default_headers / "ascend_hal_define.h").is_file():
            headers = default_headers
        library, build = build_probe(root / "hal_trace", headers)
        metadata["probe_build"] = build
        extra_env = {
            **capture_env,
            "ASCEND_MODULE_LOG_LEVEL": None,
            "BASH_ENV": str(guard),
        }
        metadata["reproduction_return_code"] = run_command(
            command,
            root / "hal_trace",
            library,
            extra_env=extra_env,
            on_start=monitor.start,
        )
    except (
        OSError,
        RuntimeError,
        subprocess.CalledProcessError,
        KeyboardInterrupt,
    ) as error:
        metadata["reproduction_error"] = str(error) or type(error).__name__
    finally:
        # run_command restores signal handlers before returning. Protect the
        # cleanup/export phase from a second interrupt as well.
        for number in (signal.SIGINT, signal.SIGTERM):
            signal.signal(
                number,
                lambda *_: print(
                    "[collect] finishing log export and archive", flush=True
                ),
            )
        metadata["process_group_cleanup"] = monitor.finish()

    print("[collect] reproduction stopped; collecting retained logs", flush=True)
    time.sleep(2)
    stages = metadata["stages"]
    try:
        stages["hal"] = analyze(root / "hal_trace")
        stages["hal"].pop("rows", None)
    except (OSError, ValueError) as error:
        stages["hal"] = {"error": str(error)}
    for name, collect in [
        ("host_logs", lambda: collect_host(root, args.host_log_root)),
        (
            "msnpureport",
            lambda: export_device(
                root,
                args.msnpureport,
                container=args.container,
                timeout=args.msnpureport_timeout,
            ),
        ),
        ("metadata", lambda: collect_metadata(root)),
    ]:
        try:
            stages[name] = collect()
        except (OSError, RuntimeError, ValueError) as error:
            stages[name] = {"status": "failed", "error": str(error)}
    marker = root / "metadata" / "launch_guard.log"
    stages["logging_guard_used"] = (
        marker.exists() and "server_launch=1" in marker.read_text()
    )
    counts = Counter()
    for path in (root / "cann").rglob("*.log"):
        match = re.match(r"(?:plog|device)-(\d+)_", path.name)
        if match:
            counts[(str(path.parent.relative_to(root)), match.group(1))] += 1
    stages["rotation_limit_reached"] = [
        {"directory": directory, "pid": pid, "files": count}
        for (directory, pid), count in counts.items()
        if count >= 1000
    ]
    hal = stages["hal"]
    reasons = []
    if "reproduction_error" in metadata:
        reasons.append(
            "reproduction could not be launched: " + metadata["reproduction_error"]
        )
    if metadata["process_group_cleanup"]["status"] == "failed":
        reasons.append("some reproduction workers could not be stopped; see cleanup")
    if not stages["logging_guard_used"]:
        reasons.append(
            "logging settings were not verified at python3 -m sglang.launch_server"
        )
    if stages["msnpureport"]["status"] != "ok":
        reasons.append("msnpureport: " + stages["msnpureport"]["status"])
    if (
        stages["msnpureport"].get("default_report_snapshot", {}).get("status")
        == "partial"
    ):
        reasons.append("some retained msnpureport files could not be copied")
    if not all(
        hal.get("events", {}).get(event, 0) > 0
        for event in ("switch_begin", "fill_begin")
    ):
        reasons.append("both HAL APIs were not captured")
    if hal.get("malformed_records") or hal.get("events", {}).get(
        "unsupported_provider", 0
    ):
        reasons.append("HAL capture has malformed records or unsupported providers")
    if not isinstance(stages["host_logs"], list) or any(
        record["status"] == "partial" for record in stages["host_logs"]
    ):
        reasons.append("some host logs could not be copied; see stage errors")
    if stages["metadata"].get("dmesg", {}).get("status") != "ok":
        reasons.append("kernel dmesg could not be collected")
    if stages["rotation_limit_reached"]:
        reasons.append(
            "a CANN log directory reached the 1000-file limit; earlier rotation cannot be excluded"
        )
    if not any(
        path.is_file() and path.stat().st_size for path in (root / "cann").rglob("*")
    ):
        reasons.append(
            "no CANN application log file was retained in the per-run directory"
        )
    complete = not reasons
    metadata["incomplete_reasons"] = reasons
    metadata["collection_complete"] = bool(complete)
    metadata["finished_utc"] = datetime.now(timezone.utc).isoformat()
    dump(root / "collection.json", metadata)
    (root / "README.txt").write_text(
        "collection.json: per-stage results; collection_complete is coverage, not model success.\n"
        "cann/: this run's DEBUG CANN host/device application logs, FILE_NUM=1000.\n"
        "host_logs/: retained root/user Ascend and system logs; sources/errors are in collection.json.\n"
        "msnpureport/: device export and command output.\n"
        "hal_trace/: terminal.log, per-PID HAL calls, aligned.tsv and problems.jsonl.\n"
        "metadata/: run.sh, exact Python packages, driver/build/firmware info, npu-smi and dmesg.\n"
        "Different HAL inputs do not alone establish a driver fault; overlapping calls are UNKNOWN.\n"
    )
    archive = Path(str(root) + ".tar.gz")
    print(f"[collect] packaging {archive}", flush=True)
    partial_archive = Path(str(archive) + ".partial")
    with tarfile.open(partial_archive, "w:gz", compresslevel=1) as bundle:
        bundle.add(root, arcname=root.name)
    partial_archive.replace(archive)
    digest = hashlib.sha256()
    with archive.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    Path(str(archive) + ".sha256").write_text(f"{digest.hexdigest()}  {archive.name}\n")
    print(f"ARCHIVE={archive}\nCOLLECTION_COMPLETE={bool(complete)}", flush=True)
    for reason in reasons:
        print("INCOMPLETE:", reason, flush=True)
    # Reproduction failure is expected here. Its original status is retained
    # separately; the collector's exit status reports collection coverage.
    return 0 if complete else 2


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
        print(
            f"Collection failed: {error}; retain the capture directory", file=sys.stderr
        )
        sys.exit(2)
