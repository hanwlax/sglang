"""CPU-only end-to-end checks for collection/finalization using a fake HAL.

python3 self_test_collect_sqcq.py
No real NPU tools or host log directories are read by these fixtures.
"""

import hashlib
import json
import os
import shlex
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path
from unittest.mock import patch

from collect_sqcq import ReproMonitor, export_device, snapshot

HERE = Path(__file__).resolve().parent


def make_case(root):
    (root / "sglang").mkdir(parents=True)
    (root / "sglang" / "__init__.py").write_text("")
    (root / "old_host" / "debug" / "plog").mkdir(parents=True)
    (root / "old_host" / "debug" / "plog" / "retained.log").write_text(
        "older host log\n"
    )
    (root / "run.sh").write_text("""#!/bin/bash
set -euo pipefail
# Deliberately override the outer environment, like old diagnostic snippets.
export ASCEND_GLOBAL_LOG_LEVEL=3
export ASCEND_HOST_LOG_FILE_NUM=2
export ASCEND_MODULE_LOG_LEVEL='RUNTIME=3:DRV=3'
export ASCEND_SLOG_PRINT_TO_STDOUT=1
export ASCEND_PROCESS_LOG_PATH="$PWD/wrong-log-path"
unset LD_PRELOAD LD_AUDIT
python3 -m sglang.launch_server "$@" 2>&1 | tee server.log
""")
    (root / "sglang" / "launch_server.py").write_text("""import ctypes as c
import json
import os
from pathlib import Path
import signal
import sys
import time
if os.environ['FAKE_LOG_MODE']=='debug':
    assert os.environ['ASCEND_GLOBAL_LOG_LEVEL']=='0'
    assert os.environ['ASCEND_HOST_LOG_FILE_NUM']=='1000'
    assert os.environ['ASCEND_SLOG_PRINT_TO_STDOUT']=='0'
    assert 'ASCEND_MODULE_LOG_LEVEL' not in os.environ
else:
    assert os.environ['ASCEND_GLOBAL_LOG_LEVEL']=='3'
    assert os.environ['ASCEND_HOST_LOG_FILE_NUM']=='2'
    assert os.environ['ASCEND_SLOG_PRINT_TO_STDOUT']=='1'
    assert os.environ['ASCEND_MODULE_LOG_LEVEL']=='RUNTIME=3:DRV=3'
assert sys.argv[1:]==['10086','4','0','0']
if os.environ.get('FAKE_ORPHAN')=='1':
    if os.fork()==0:
        signal.signal(signal.SIGINT,signal.SIG_IGN)
        signal.signal(signal.SIGTERM,signal.SIG_IGN)
        Path('orphan.pid').write_text(str(os.getpid()))
        time.sleep(60)
        os._exit(0)
    while not Path('orphan.pid').exists():
        time.sleep(0.01)
directory=Path(os.environ['ASCEND_PROCESS_LOG_PATH'])/'debug'/'plog'
directory.mkdir(parents=True)
log=directory/('plog-'+str(os.getpid())+'_fixture.log')
log.write_text('[DEBUG] first line retained\\n')
h=c.CDLL(os.environ['FAKE_HAL'],mode=os.RTLD_NOW | os.RTLD_LOCAL)
class Info(c.Structure):
    _fields_=[('sq',c.c_uint32),('stream',c.c_uint32),('depth',c.c_uint32),('rsv',c.c_uint32),('memory',c.c_void_p)]
h.halSqSwitchStreamBatch.argtypes=[c.c_uint32,c.POINTER(Info),c.c_uint32]
h.halStreamTaskFill.argtypes=[c.c_uint32,c.c_uint32,c.c_void_p,c.c_void_p,c.c_uint32]
info=Info(38,22,505,0,0x1208bfe00000)
assert h.halSqSwitchStreamBatch(0,c.byref(info),1)==0
assert h.halStreamTaskFill(0,22,info.memory,0xabc0,127)==0
info.memory=0x1208bfe08000
assert h.halSqSwitchStreamBatch(0,c.byref(info),1)==0
assert h.halStreamTaskFill(0,22,info.memory,0xabc0,127)==14
if os.environ['FAKE_TRACE']=='0':
    assert 'libsqcq_hal_trace.so' not in os.environ.get('LD_PRELOAD','')
    assert 'libsqcq_hal_trace.so' not in os.environ.get('LD_AUDIT','')
    print('[ERROR] RUNTIME: Fill task exceed stream queue',flush=True)
# Do not print a failure: auto-stop must detect the native HAL record.
print('fake model remains alive after HAL failure',flush=True)
time.sleep(60)
""")
    tool = root / "msnpureport"
    tool.write_text("""#!/bin/bash
if [[ "${1-}" == --help ]]; then
    echo 'msnpureport -f, --force; --docker'
    exit 0
fi
printf '%s\\n' "$*" > received_args.txt
if [[ "${FAKE_EXPORT_FAIL:-0}" == 1 ]]; then
    echo 'export denied' >&2
    exit 7
fi
if [[ "${FAKE_EXPORT_HANG:-0}" == 1 ]]; then
    sleep 60
fi
mkdir -p report_fixture/slog
echo 'device diagnostic log' > report_fixture/slog/device.log
""")
    tool.chmod(0o755)
    # Redirect only external collection boundaries to private fixture data.
    wrapper = root / "invoke.py"
    wrapper.write_text(f"""import sys
from pathlib import Path
sys.path.insert(0,{str(HERE)!r})
import collect_sqcq as c
original_snapshot=c.snapshot
def isolated_snapshot(source,destination,capture_root):
    if source==Path('/var/log/npu/report'):
        return {{'source':str(source),'status':'missing','files':0,'bytes':0,'errors':[]}}
    return original_snapshot(source,destination,capture_root)
c.snapshot=isolated_snapshot
c.collect_host=lambda capture,extra:[original_snapshot(Path({str(root / "old_host")!r}),capture/'host_logs'/'fixture',capture)]+[original_snapshot(source,capture/'host_logs'/'discovered'/str(index),capture) for index,source in enumerate(extra)]
c.collect_metadata=lambda capture:{{'dmesg':{{'status':'ok','note':'fake boundary; no real kernel read'}}}}
sys.exit(c.main())
""")
    return wrapper, tool


def check_case(
    root,
    fake,
    fail_export,
    timeout_export=False,
    orphan=False,
    *,
    log_mode="debug",
    trace_hal=True,
):
    wrapper, tool = make_case(root)
    output = root / "capture"
    environment = os.environ.copy()
    environment.update(
        FAKE_HAL=str(fake),
        FAKE_EXPORT_FAIL="1" if fail_export else "0",
        FAKE_EXPORT_HANG="1" if timeout_export else "0",
        FAKE_ORPHAN="1" if orphan else "0",
        FAKE_LOG_MODE=log_mode or "preserve",
        FAKE_TRACE="1" if trace_hal else "0",
    )
    if not trace_hal:
        environment["CC"] = "/bin/false"  # No compiler is needed without the probe.
    result = subprocess.run(
        [
            sys.executable,
            str(wrapper),
            *(["--log-mode", log_mode] if log_mode else []),
            *([] if trace_hal else ["--no-hal-trace"]),
            "--output",
            str(output),
            "--post-error-seconds",
            "0.1",
            "--run-timeout",
            "10",
            "--msnpureport",
            str(tool),
            "--msnpureport-timeout",
            "0.2" if timeout_export else "5",
            "--container",
            "yes",
            "--",
            "bash",
            "./run.sh",
            "10086",
            "4",
            "0",
            "0",
        ],
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
        timeout=25,
    )
    partial = fail_export or timeout_export
    expected = 2 if partial else 0
    assert result.returncode == expected, (
        result.returncode,
        result.stdout,
        result.stderr,
    )
    metadata = json.loads((output / "collection.json").read_text())
    assert metadata["collection_complete"] is (not partial), metadata
    debug = log_mode == "debug"
    assert metadata["log_mode"] == (log_mode or "preserve")
    assert metadata["hal_trace_enabled"] is trace_hal
    if orphan:
        pid = int((root / "orphan.pid").read_text())
        process_stat = Path(f"/proc/{pid}/stat")
        assert not process_stat.exists() or process_stat.read_text().split()[2] == "Z"
        assert "SIGKILL" in metadata["process_group_cleanup"]["signals"]
    assert metadata["stages"]["logging_guard_used"]
    serving_python = json.loads(
        (output / "metadata" / "python.jsonl").read_text().splitlines()[-1]
    )
    assert serving_python["sglang_origin"] == str(root / "sglang" / "__init__.py")
    assert serving_python["argv"][:2] == ["-m", "sglang.launch_server"]
    settings = serving_python["logging_environment"]
    assert settings["ASCEND_GLOBAL_LOG_LEVEL"] == ("0" if debug else "3")
    assert settings["ASCEND_HOST_LOG_FILE_NUM"] == ("1000" if debug else "2")
    assert settings["ASCEND_MODULE_LOG_LEVEL"] == (None if debug else "RUNTIME=3:DRV=3")
    assert (root / "wrong-log-path").exists() is (not debug)
    if debug:
        assert metadata["requested_logging"]["ASCEND_HOST_LOG_FILE_NUM"] == "1000"
    else:
        assert metadata["requested_logging"] == {}
    assert (output / "metadata" / "run.sh").read_bytes() == (
        root / "run.sh"
    ).read_bytes()
    trigger = json.loads((output / "trigger.json").read_text())["reason"]
    assert (
        trigger == "nonzero HAL Fill return"
        if trace_hal
        else "Fill task exceed stream queue" in trigger
    )
    log_root = output / "cann" if debug else output / "host_logs" / "discovered"
    assert "first line retained" in next(log_root.rglob("*.log")).read_text()
    assert (
        output / "host_logs" / "fixture" / "debug" / "plog" / "retained.log"
    ).is_file()
    assert (
        output / "msnpureport" / "received_args.txt"
    ).read_text().strip() == "-f --docker"
    if trace_hal:
        hal = json.loads((output / "hal_trace" / "summary.json").read_text())
        assert any(
            row["fill_ret"] == 14 and row["relation"] == "EQUAL" for row in hal["rows"]
        )
    else:
        assert metadata["stages"]["hal"]["status"] == "disabled"
        assert not (output / "hal_trace" / "libsqcq_hal_trace.so").exists()
        assert not list((output / "hal_trace" / "hal").glob("*.jsonl"))
    archive = Path(str(output) + ".tar.gz")
    assert archive.is_file()
    assert (
        Path(str(archive) + ".sha256").read_text().split()[0]
        == hashlib.sha256(archive.read_bytes()).hexdigest()
    )
    with tarfile.open(archive) as bundle:
        names = bundle.getnames()
        assert (
            any(name.endswith("hal_trace/aligned.tsv") for name in names) is trace_hal
        )
        assert any(name.endswith("collection.json") for name in names)
        if partial:
            assert metadata["stages"]["msnpureport"]["status"] == "failed"
            assert any(name.endswith("msnpureport/command.log") for name in names)
            assert metadata["incomplete_reasons"]
            if timeout_export:
                assert metadata["stages"]["msnpureport"]["timeout"] == 0.2
        else:
            assert any(
                name.endswith("msnpureport/report_fixture/slog/device.log")
                for name in names
            )
    print(
        "PASS collector:",
        "export failure/timeout still packaged, marked partial"
        if partial
        else f"log_mode={log_mode or 'preserve'} hal_trace={trace_hal}; auto-stop, settings, sources and archive verified",
    )


def check_snapshot(root):
    source = root / "source"
    source.mkdir()
    (source / "a.log").write_text("complete content\n")
    outside = root / "target.log"
    outside.write_text("linked content\n")
    (source / "external.log").symlink_to(outside)
    (source / "cycle").symlink_to(source, target_is_directory=True)
    capture = source / "capture"
    capture.mkdir()
    (capture / "own.log").write_text("must not recursively copy this\n")
    result = snapshot(source, capture / "copy", capture.resolve())
    assert result["status"] == "ok" and result["files"] == 2, result
    assert (capture / "copy" / "external.log").read_text() == "linked content\n"
    assert not (capture / "copy" / "capture").exists()
    assert {item["reason"] for item in result["skipped"]} == {
        "symlink cycle",
        "capture output",
    }
    print(
        "PASS snapshot: full file/symlink content, cycles and output recursion excluded"
    )


def check_error_detection(root):
    capture = root / "scan"
    (capture / "hal_trace" / "hal").mkdir(parents=True)
    plog = capture / "cann" / "debug" / "plog" / "plog-fixture.log"
    plog.parent.mkdir(parents=True)
    plog.write_text(
        "[INFO] RegisterFormatErrorMessage: Build SQ/CQ failed\n"
        "[DEBUG] Alloc sq cq error=0\n"
    )
    monitor = ReproMonitor(capture, timeout=0, post_error=0, auto_stop=True)
    assert monitor.scan() is None
    with plog.open("a") as stream:
        stream.write("[ERROR] RUNTIME: Build SQ/CQ failed, model_id=62\n")
    assert "Build SQ/CQ failed" in monitor.scan()["reason"]
    print("PASS detection: INFO registration/error=0 ignored; actual ERROR caught")


def check_export_help(root):
    loader_message = (
        "/usr/local/Ascend/driver/tools/msnpureport: /lib/aarch64-linux-gnu/libc.so.6: "
        "version `GLIBC_2.38' not found (required by msnpureport)"
    )
    for name, help_text, status, expected in (
        ("legacy", "Usage: msnpureport -f, --force", 0, "-f"),
        ("short_legacy", "Usage: msnpureport\n-f\nexport report", 0, "-f"),
        (
            "modern",
            "Usage: msnpureport\nCommands:\n report [options]",
            0,
            "report -t 0",
        ),
        ("basename_only", "Usage: msnpureport [options]", 0, "-f"),
        ("failed_help", "msnpureport: failed to generate report help", 1, "-f"),
        ("loader_failure", loader_message, 127, None),
    ):
        capture = root / name
        capture.mkdir(parents=True)
        tool = capture / "fake_msnpureport"
        tool.write_text(f"""#!/bin/bash
if [[ "${{1-}}" == --help ]]; then
    printf '%s\\n' {shlex.quote(help_text)}
    exit {status}
fi
printf '%s\\n' "$*" > received_args.txt
mkdir -p report_fixture
echo 'exported device data' > report_fixture/device.log
""")
        tool.chmod(0o755)
        with patch(
            "collect_sqcq.snapshot",
            return_value={"status": "missing", "source": "fake boundary"},
        ):
            result = export_device(capture, str(tool), container="no", timeout=2)
        received = capture / "msnpureport" / "received_args.txt"
        if expected is None:
            assert result["status"] == "failed", result
            assert result["export_attempted"] is False
            assert not received.exists()
            assert (
                loader_message in (capture / "msnpureport" / "command.log").read_text()
            )
        else:
            assert result["status"] == "ok", result
            assert received.read_text().strip() == expected
    print(
        "PASS export detection: valid legacy/modern help, basename, failed help and loader error"
    )


def main():
    with tempfile.TemporaryDirectory(prefix="sqcq-collect-self-test-") as temporary:
        root = Path(temporary)
        fake = root / "libfake_hal.so"
        subprocess.run(
            shlex.split(os.environ.get("CC", "cc"))
            + [
                "-std=gnu11",
                "-shared",
                "-fPIC",
                "-DSQ_CQ_FAKE_LIBRARY",
                str(HERE / "hal_stream_mem_fixture.c"),
                "-o",
                str(fake),
            ],
            check=True,
        )
        check_case(root / "successful_export", fake, False)
        check_case(root / "failed_export", fake, True)
        check_case(root / "timed_out_export", fake, False, True)
        check_case(root / "orphan_worker", fake, False, orphan=True)
        check_case(root / "default_preserve", fake, False, log_mode=None)
        check_case(root / "debug_without_probe", fake, False, trace_hal=False)
        check_case(
            root / "preserve_without_probe",
            fake,
            False,
            log_mode="preserve",
            trace_hal=False,
        )
        check_snapshot(root)
        check_error_detection(root)
        check_export_help(root / "export_help")
    print(
        "All collector checks passed with fake tools; no real NPU or host logs were read."
    )


if __name__ == "__main__":
    main()
