"""Deterministic native self-check for trace_hal_stream_mem.py; no NPU needed."""

import contextlib
import io
import os
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
from pathlib import Path

from trace_hal_stream_mem import HERE, align, analyze, build_probe, run_command


def self_test():
    with tempfile.TemporaryDirectory(prefix="sqcq-hal-self-test-") as temporary:
        root = Path(temporary)
        library, _ = build_probe(root)
        cc = shlex.split(os.environ.get("CC", "cc"))
        fixture = str(HERE / "hal_stream_mem_fixture.c")
        fake = root / "libfake_hal.so"
        subprocess.run(
            cc
            + [
                "-std=gnu11",
                "-O2",
                "-shared",
                "-fPIC",
                "-DSQ_CQ_FAKE_LIBRARY",
                fixture,
                "-o",
                str(fake),
            ],
            check=True,
        )
        for mode in ("plt", "got", "dlsym_deepbind", "mixed"):
            client = root / mode
            flags = []
            if mode != "dlsym_deepbind":
                flags += [
                    "-DSQ_CQ_FAKE_LINKED",
                    "-L",
                    str(root),
                    "-lfake_hal",
                    f"-Wl,-rpath,{root}",
                ]
            if mode == "got":
                flags += ["-fno-plt"]
            if mode == "mixed":
                flags += ["-DSQ_CQ_FAKE_MIXED"]
            subprocess.run(
                cc
                + ["-std=gnu11", "-O2", fixture, "-o", str(client)]
                + flags
                + ["-ldl", "-pthread"],
                check=True,
            )
            directory = root / (mode + "_capture")
            (directory / "hal").mkdir(parents=True)
            ret = run_command([str(client), str(fake)], directory, library)
            assert ret == 0, (mode, ret)
            with contextlib.redirect_stdout(io.StringIO()):
                summary = analyze(directory)
            rows = summary["rows"]
            assert len(rows) == 9, (mode, len(rows), summary["events"])
            assert not summary["malformed_records"]
            main_pid = rows[0]["pid"]
            main = [row for row in rows if row["pid"] == main_pid]
            stream22 = [r for r in main if r["device"] == 0 and r["stream"] == 22]
            assert [r["relation"] for r in stream22] == ["EQUAL", "EQUAL", "DIFFERENT"]
            assert [r["fill_ret"] for r in stream22] == [0, 14, 0]
            changed = next(r for r in main if r["stream"] == 23)
            assert changed["switch_mem"] == "0x1208bfe00000"
            assert changed["switch_mem_after"] == "0x1208bfe00040"
            failed = next(r for r in main if r["stream"] == 24)
            assert failed["relation"] == "UNKNOWN" and failed["switch_call"] is None
            assert failed["fill_ret"] == 33
            assert (
                len([r for r in main if r["device"] == 1 and r["relation"] == "EQUAL"])
                == 2
            )
            concurrent = next(r for r in main if r["stream"] == 27)
            assert concurrent["relation"] == "UNKNOWN"
            assert concurrent["correlation"] == "OVERLAPPING_SWITCH"
            child = next(r for r in rows if r["pid"] != main_pid)
            assert child["relation"] == "UNKNOWN" and child["switch_call"] is None
            assert len({r["fill_call"] for r in main}) == len(main)
            if mode == "mixed":
                assert {r["fill_call"].split(":")[0] for r in main} == {
                    "audit",
                    "preload",
                }
            with contextlib.redirect_stdout(io.StringIO()):
                empty = analyze(directory, pid=-1)
            assert empty["rows"] == []
            assert (directory / "aligned.tsv").read_text() == ""
            assert (directory / "problems.jsonl").read_text() == ""
            print(
                f"PASS {mode}: real interception, exact arguments, errors, batch stride, fork/PID isolation, concurrent-call ambiguity"
            )
        check_alignment_edges()
        check_multiple_providers(root, fake, library)
        check_cli_and_headers(root)
    print("All self-checks passed. These checks use a fake HAL, not an NPU driver.")


def check_alignment_edges():
    def event(kind, call, stamp, **extra):
        result = dict(
            event=kind,
            call=call,
            mono_ns=stamp,
            wall_ns=stamp,
            pid=10,
            tid=11,
            device=0,
            stream=22,
            index=0,
            sq=38,
            depth=505,
            stream_mem="0x1208bfe00000",
            task_cnt=127,
            source="synthetic",
        )
        result.update(extra)
        return result

    # Out-of-file-order events must still correlate by call intervals.
    base = [
        event("switch_begin", 1, 10),
        event("switch_end", 1, 20, ret=0),
        event("fill_begin", 3, 50),
        event("fill_end", 3, 60, ret=0),
    ]
    assert align(list(reversed(base)))[0]["relation"] == "EQUAL"
    # A later failed switch must not silently re-establish an old binding.
    failed = base + [event("switch_begin", 2, 30), event("switch_end", 2, 40, ret=7)]
    assert align(failed)[0]["correlation"] == "INTERVENING_FAILED_SWITCH"
    assert align(failed)[0]["relation"] == "UNKNOWN"
    # Two successful overlapping switches have an unknown internal order.
    crossed = [
        event("switch_begin", 1, 10),
        event("switch_end", 1, 30, ret=0),
        event("switch_begin", 2, 20),
        event("switch_end", 2, 40, ret=0),
        event("fill_begin", 3, 50),
        event("fill_end", 3, 60, ret=0),
    ]
    assert align(crossed)[0]["correlation"] == "AMBIGUOUS_SWITCH_ORDER"
    assert align(base[:-1])[0]["correlation"] == "INCOMPLETE_FILL"
    assert align(base, device=1) == []
    assert align(base, stream=23) == []
    assert align(base, pid=12) == []
    print(
        "PASS alignment: out-of-order records, failed/reentrant/incomplete calls and PID/device/stream filters"
    )


def check_cli_and_headers(root):
    output = root / "cli_capture"
    command = [
        sys.executable,
        str(HERE / "trace_hal_stream_mem.py"),
        "--output",
        str(output),
        "--",
        "bash",
        "-c",
        "printf 'stdout-marker\\n'; printf 'stderr-marker\\n' >&2; exit 23",
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=10)
    assert result.returncode == 23, result
    assert (output / "terminal.log").read_text() == "stdout-marker\nstderr-marker\n"
    assert "INCOMPLETE COVERAGE" in result.stdout
    assert (output / "summary.json").is_file()

    stopped = root / "signal_capture"
    process = subprocess.Popen(
        [
            sys.executable,
            str(HERE / "trace_hal_stream_mem.py"),
            "--output",
            str(stopped),
            "--",
            sys.executable,
            "-c",
            "import time; print('child-ready', flush=True); time.sleep(60)",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        while line := process.stdout.readline():
            if line.strip() == "child-ready":
                break
        else:
            raise AssertionError("signal fixture did not start")
        process.send_signal(signal.SIGTERM)
        process.communicate(timeout=10)
        assert process.returncode == 143, process.returncode
        assert (stopped / "summary.json").is_file()
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()

    headers = root / "headers"
    headers.mkdir()
    header = headers / "ascend_hal_define.h"
    definition = (
        "#include <stdint.h>\nstruct sq_switch_stream_info {\n"
        "uint32_t sq_id, stream_id, sq_depth, rsv; void *stream_mem;\n};\n"
    )
    header.write_text(definition)
    _, compile_command = build_probe(root, headers)
    header.write_text(
        definition.replace("void *stream_mem;", "void *stream_mem; uint64_t extra;")
    )
    bad = subprocess.run(compile_command, capture_output=True, text=True)
    assert bad.returncode != 0 and "Driver ABI differs" in bad.stderr
    print(
        "PASS CLI: stdout/stderr capture, exit/signal status, missing-coverage warning; installed-header ABI mismatch rejected"
    )


def check_multiple_providers(root, fake, library):
    second = root / "libsecond_hal.so"
    shutil.copyfile(fake, second)
    code = """
import ctypes as c
import os
import sys
class Info(c.Structure):
    _fields_=[('sq', c.c_uint32), ('stream', c.c_uint32),
              ('depth', c.c_uint32), ('rsv', c.c_uint32), ('memory', c.c_void_p)]
for path, memory in [(sys.argv[1], 0x1208bfe00000), (sys.argv[2], 0x1208bfe08000)]:
    h=c.CDLL(path, mode=os.RTLD_NOW | os.RTLD_LOCAL | os.RTLD_DEEPBIND)
    h.halSqSwitchStreamBatch.argtypes=[c.c_uint32, c.POINTER(Info), c.c_uint32]
    h.halStreamTaskFill.argtypes=[c.c_uint32,c.c_uint32,c.c_void_p,c.c_void_p,c.c_uint32]
    info=Info(38,22,505,0,memory)
    assert h.halSqSwitchStreamBatch(0,c.byref(info),1)==0
    assert h.halStreamTaskFill(0,22,memory,0xabc0,127)==0
print('ctypes providers preserved', flush=True)
"""
    directory = root / "multiple_providers"
    (directory / "hal").mkdir(parents=True)
    client = root / "ctypes_providers.py"
    client.write_text(code)
    ret = run_command(
        [sys.executable, str(client), str(fake), str(second)], directory, library
    )
    assert ret == 0
    with contextlib.redirect_stdout(io.StringIO()):
        summary = analyze(directory)
    assert summary["events"]["unsupported_provider"] == 2
    assert len(summary["rows"]) == 1
    print(
        "PASS ctypes/multiple providers: additional HAL implementation remains unchanged, coverage explicitly incomplete"
    )


if __name__ == "__main__":
    self_test()
