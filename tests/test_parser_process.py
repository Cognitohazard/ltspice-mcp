"""Real parser-process containment without simulator discovery or execution."""

from __future__ import annotations

import asyncio
import ctypes
import errno
import json
import os
import struct
import subprocess
import sys
import threading
import time
from dataclasses import replace
from pathlib import Path

import psutil
import pytest

from ltspice_mcp.lib import parser_process, windows_job
from ltspice_mcp.lib.parser_process import (
    JsonValue,
    ParserProcessError,
    ParserProcessLimits,
    run_parser,
    run_parser_sync,
)
from ltspice_mcp.lib.store import parser_file_in
from ltspice_mcp.lib.windows_job import python_launch

_FIXTURE = """
import ctypes, json, os, struct, subprocess, sys
from pathlib import Path
from ltspice_mcp.lib.store import parser_file_in

Path("imported.txt").write_text("imported", encoding="utf-8")

def parse_request(request, work_dir):
    mode = request.get("mode", "ok")
    if mode == "runaway":
        child = subprocess.Popen(
            [sys.executable, "-c", "while True: pass"],
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL, start_new_session=os.name != "nt",
        )
        parser_file_in(work_dir, "started.json").write_text(json.dumps({
            "worker_pid": os.getpid(), "child_pid": child.pid,
        }), encoding="utf-8")
        if sys.platform == "linux":
            ctypes.PyDLL(None).sleep(60)  # Holds the decoder's GIL.
        else:
            ctypes.PyDLL("kernel32").Sleep(60000)
    if mode == "crash":
        os._exit(17)
    if mode == "memory":
        bytearray(1024 * 1024 * 1024)
    if mode == "error":
        raise ValueError("broken " * 100000)
    if mode == "stderr":
        os.write(2, b"x" * request["bytes"])
    if mode == "invalid":
        parser_file_in(work_dir, "result.json").write_text(request["text"], encoding="utf-8")
        return
    if mode == "environment":
        parser_file_in(work_dir, "result.json").write_text(json.dumps({
            name: os.environ.get(name) for name in
            ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
        }), encoding="utf-8")
        return
    if mode == "breakaway":
        try:
            child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"],
                                     creationflags=0x01000000)
        except OSError:
            result = {"escaped": False}
        else:
            result = {"escaped": True, "child_pid": child.pid}
        parser_file_in(work_dir, "result.json").write_text(json.dumps(result), encoding="utf-8")
        return
    parser_file_in(work_dir, "array_0000.bin").write_bytes(struct.pack("<d", 2 / 3))
    parser_file_in(work_dir, "result.json").write_text(json.dumps({
        "value": 2 / 3, "array": "array_0000.bin", "pid": os.getpid(),
    }), encoding="utf-8")
"""


@pytest.fixture
def limits():
    return ParserProcessLimits(
        memory_bytes=128 * 1024 * 1024,
        request_bytes=65536,
        metadata_bytes=4096,
        error_bytes=1024,
        cleanup_grace_s=2,
    )


@pytest.fixture
def call_dir(tmp_path):
    def create(name="call"):
        directory = tmp_path / name
        directory.mkdir()
        parser_file_in(directory, "parser_fixture.py").write_text(_FIXTURE, encoding="utf-8")
        return directory

    return create


async def _started(directory: Path) -> dict[str, int]:
    deadline = time.monotonic() + 5
    path = parser_file_in(directory, "started.json")
    while time.monotonic() < deadline:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        await asyncio.sleep(0.01)
    pytest.fail("The real decoder did not reach its runaway seam")


def _assert_gone(pids):
    for pid in pids:
        assert not psutil.pid_exists(pid), f"Owned process {pid} was not reaped"


async def _success(directory, limits):
    reply = await run_parser(
        {},
        work_dir=directory,
        deadline=time.monotonic() + 5,
        limits=limits,
        _worker_module="parser_fixture",
    )
    assert reply.metadata["value"] == pytest.approx(2 / 3)
    assert reply.metadata["pid"] == reply.worker_pid
    assert struct.unpack("<d", parser_file_in(directory, "array_0000.bin").read_bytes())[
        0
    ] == pytest.approx(2 / 3)
    _assert_gone([reply.worker_pid])


async def test_timeout_reaps_gil_holding_decoder_and_detached_descendant(call_dir, limits):
    directory = call_dir()
    task = asyncio.create_task(
        run_parser(
            {"mode": "runaway"},
            work_dir=directory,
            deadline=time.monotonic() + 1,
            limits=limits,
            _worker_module="parser_fixture",
        )
    )
    started = await _started(directory)
    with pytest.raises(ParserProcessError) as caught:
        await task
    assert caught.value.code == "deadline" and caught.value.reaped
    _assert_gone(started.values())
    await _success(call_dir("fresh"), limits)


async def test_repeated_cancellation_waits_for_tree_reaping(call_dir, limits):
    directory = call_dir()
    task = asyncio.create_task(
        run_parser(
            {"mode": "runaway"},
            work_dir=directory,
            deadline=time.monotonic() + 20,
            limits=limits,
            _worker_module="parser_fixture",
        )
    )
    started = await _started(directory)
    for _ in range(3):
        task.cancel()
        await asyncio.sleep(0)
    with pytest.raises(asyncio.CancelledError):
        await task
    _assert_gone(started.values())
    await _success(call_dir("fresh"), limits)


def test_sync_cancellation_event_reaps_worker(call_dir, limits):
    directory = call_dir()
    cancel = threading.Event()

    def request_cancel():
        deadline = time.monotonic() + 5
        while (
            not parser_file_in(directory, "started.json").exists() and time.monotonic() < deadline
        ):
            time.sleep(0.01)
        cancel.set()

    canceller = threading.Thread(target=request_cancel)
    canceller.start()
    try:
        with pytest.raises(ParserProcessError) as caught:
            run_parser_sync(
                {"mode": "runaway"},
                work_dir=directory,
                deadline=time.monotonic() + 20,
                limits=limits,
                cancel=cancel,
                _worker_module="parser_fixture",
            )
        assert caught.value.code == "cancelled" and caught.value.reaped
        started = json.loads(parser_file_in(directory, "started.json").read_text())
        _assert_gone(started.values())
    finally:
        cancel.set()
        canceller.join(timeout=6)


def test_owner_death_kills_and_reaps_decoder_tree(call_dir, limits):
    directory = call_dir()
    program = """
import time
from pathlib import Path
from ltspice_mcp.lib.parser_process import ParserProcessLimits, run_parser_sync
run_parser_sync({"mode": "runaway"}, work_dir=Path.cwd(),
    deadline=time.monotonic() + 30,
    limits=ParserProcessLimits(134217728, 65536, 4096, 1024, 2),
    _worker_module="parser_fixture")
"""
    executable, env = python_launch()
    owner = subprocess.Popen(
        [executable, "-c", program],
        cwd=directory,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    owned = []
    try:
        started_path = parser_file_in(directory, "started.json")
        deadline = time.monotonic() + 5
        while not started_path.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert started_path.exists()
        owned = list(json.loads(started_path.read_text()).values())
        owner.kill()
        owner.wait(timeout=5)
        deadline = time.monotonic() + 5
        while any(psutil.pid_exists(pid) for pid in owned) and time.monotonic() < deadline:
            time.sleep(0.01)
        _assert_gone(owned)
        if sys.platform == "linux":
            assert (
                json.loads(parser_file_in(directory, "cleanup.json").read_text())["reaped"] is True
            )
    finally:
        if owner.poll() is None:
            owner.kill()
        owner.wait(timeout=5)
        for pid in owned:
            try:
                process = psutil.Process(pid)
                process.kill()
                process.wait(timeout=3)
            except psutil.NoSuchProcess:
                pass


@pytest.mark.parametrize(
    ("mode", "code"),
    [("crash", "worker_crashed"), ("memory", "memory_limit"), ("error", "worker_error")],
)
async def test_worker_failure_is_bounded_and_next_worker_succeeds(call_dir, limits, mode, code):
    directory = call_dir()
    with pytest.raises(ParserProcessError) as caught:
        await run_parser(
            {"mode": mode},
            work_dir=directory,
            deadline=time.monotonic() + 5,
            limits=limits,
            _worker_module="parser_fixture",
        )
    assert caught.value.code == code and caught.value.reaped
    error_path = parser_file_in(directory, "error.json")
    if error_path.exists():
        assert error_path.stat().st_size <= limits.error_bytes
    await _success(call_dir("fresh"), limits)


@pytest.mark.parametrize(
    "text",
    [
        '{"key":1,"key":2}',
        '{"x":NaN}',
        '{"x":' + "[" * 2000 + "0" + "]" * 2000 + "}",
        '{"x":' + "1" * 10000 + "}",
    ],
    ids=["duplicate", "nonfinite", "nesting", "integer"],
)
async def test_invalid_json_result_is_not_published(call_dir, limits, text):
    directory = call_dir()
    with pytest.raises(ParserProcessError) as caught:
        await run_parser(
            {"mode": "invalid", "text": text},
            work_dir=directory,
            deadline=time.monotonic() + 5,
            limits=replace(limits, metadata_bytes=16384),
            _worker_module="parser_fixture",
        )
    assert caught.value.code == "invalid_result" and caught.value.reaped


async def test_oversize_metadata_is_rejected_after_reaping(call_dir, limits):
    with pytest.raises(ParserProcessError) as caught:
        await run_parser(
            {"mode": "invalid", "text": '{"x":"' + "x" * 10000 + '"}'},
            work_dir=call_dir(),
            deadline=time.monotonic() + 5,
            limits=limits,
            _worker_module="parser_fixture",
        )
    assert caught.value.code in {"metadata_limit", "invalid_result"}
    assert caught.value.reaped


async def test_blas_environment_is_clamped_only_in_children(call_dir, limits, monkeypatch):
    names = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
    for name in names:
        monkeypatch.setenv(name, "6")
    reply = await run_parser(
        {"mode": "environment"},
        work_dir=call_dir(),
        deadline=time.monotonic() + 5,
        limits=limits,
        _worker_module="parser_fixture",
    )
    assert reply.metadata == dict.fromkeys(names, "1")
    assert all(os.environ[name] == "6" for name in names)


@pytest.mark.skipif(sys.platform != "linux", reason="Linux guardian startup backpressure seam")
def test_startup_stall_with_full_request_pipe_has_finite_cleanup(call_dir, limits, monkeypatch):
    directory = call_dir()
    parser_file_in(directory, "sitecustomize.py").write_text(
        'import time\nfrom pathlib import Path\nPath("stalled.txt").write_text("ready")\ntime.sleep(60)\n',
        encoding="utf-8",
    )
    spawned = []
    popen = subprocess.Popen
    launch = python_launch

    def fixture_launch():
        executable, environment = launch()
        environment = dict(environment if environment is not None else os.environ)
        environment["PYTHONPATH"] = str(directory) + os.pathsep + environment.get("PYTHONPATH", "")
        return executable, environment

    monkeypatch.setattr(parser_process, "python_launch", fixture_launch)

    def record(*args, **kwargs):
        process = popen(*args, **kwargs)
        spawned.append(process)
        return process

    monkeypatch.setattr(parser_process.subprocess, "Popen", record)
    cancel = threading.Event()
    canceller = threading.Timer(0.5, cancel.set)
    canceller.start()
    started = time.monotonic()
    try:
        with pytest.raises(ParserProcessError) as caught:
            run_parser_sync(
                {"padding": "x" * 1000000},
                work_dir=directory,
                deadline=started + 5,
                limits=replace(limits, request_bytes=2000000, cleanup_grace_s=0.5),
                cancel=cancel,
                _worker_module="parser_fixture",
            )
        assert time.monotonic() - started < 2
        assert parser_file_in(directory, "stalled.txt").exists()
        assert caught.value.code == "cleanup_failed" and not caught.value.reaped
        assert len(spawned) == 1 and spawned[0].poll() is not None
        assert not parser_file_in(directory, "imported.txt").exists()
    finally:
        canceller.cancel()
        canceller.join()
        for process in spawned:
            if process.poll() is None:
                process.kill()
            process.wait(timeout=3)


@pytest.mark.skipif(sys.platform != "win32", reason="Requires native Windows Job enforcement")
async def test_native_windows_parser_cannot_break_away(call_dir, limits):
    reply = await run_parser(
        {"mode": "breakaway"},
        work_dir=call_dir(),
        deadline=time.monotonic() + 5,
        limits=limits,
        _worker_module="parser_fixture",
    )
    try:
        assert reply.metadata["escaped"] is False
    finally:
        pid = reply.metadata.get("child_pid")
        if type(pid) is int and psutil.pid_exists(pid):
            psutil.Process(pid).kill()


@pytest.mark.skipif(sys.platform != "win32", reason="Requires native Windows process launch")
async def test_failed_windows_job_assignment_never_opens_gate(call_dir, limits, monkeypatch):
    directory = call_dir()

    def fail(*args, **kwargs):
        raise OSError("Job assignment failed")

    monkeypatch.setattr(parser_process, "WindowsJob", fail)
    with pytest.raises(ParserProcessError) as caught:
        await run_parser(
            {},
            work_dir=directory,
            deadline=time.monotonic() + 5,
            limits=limits,
            _worker_module="parser_fixture",
        )
    assert caught.value.code == "ownership_failed" and caught.value.reaped
    assert not parser_file_in(directory, "imported.txt").exists()


async def test_stderr_near_bound_is_accepted_and_overflow_is_bounded(call_dir, limits):
    limits = replace(limits, error_bytes=256)
    directory = call_dir()
    await run_parser(
        {"mode": "stderr", "bytes": 200},
        work_dir=directory,
        deadline=time.monotonic() + 5,
        limits=limits,
        _worker_module="parser_fixture",
    )
    assert parser_file_in(directory, "stderr.txt").stat().st_size == 200
    overflow_dir = call_dir("overflow")
    with pytest.raises(ParserProcessError) as caught:
        await run_parser(
            {"mode": "stderr", "bytes": 100000},
            work_dir=overflow_dir,
            deadline=time.monotonic() + 5,
            limits=limits,
            _worker_module="parser_fixture",
        )
    assert caught.value.code == "error_limit" and caught.value.reaped
    assert parser_file_in(overflow_dir, "stderr.txt").stat().st_size == 256


def test_oversize_request_refuses_before_spawn(call_dir, limits, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("An oversized request must not spawn")

    monkeypatch.setattr(parser_process.subprocess, "Popen", forbidden)
    with pytest.raises(ParserProcessError, match="request exceeds") as caught:
        run_parser_sync(
            {"data": "x" * 100000},
            work_dir=call_dir(),
            deadline=time.monotonic() + 5,
            limits=limits,
        )
    assert caught.value.code == "request_limit" and caught.value.reaped


@pytest.mark.parametrize("platform", ["darwin", "freebsd14"])
def test_unsupported_platform_refuses_before_control_files_or_launch(
    call_dir, limits, monkeypatch, platform
):
    directory = call_dir()
    monkeypatch.setattr(sys, "platform", platform)
    with pytest.raises(ParserProcessError, match="containment") as caught:
        run_parser_sync(
            {},
            work_dir=directory,
            deadline=time.monotonic() + 5,
            limits=limits,
            _worker_module="parser_fixture",
        )
    assert caught.value.code == "unsupported_platform" and caught.value.reaped
    assert caught.value.worker_pid is None
    assert {path.name for path in directory.iterdir()} == {"parser_fixture.py"}
    if platform == "darwin":
        assert "memory" in str(caught.value) and "ownership" in str(caught.value)
        assert "owner-death" in str(caught.value)


def _refused_bootstrap(directory, limits, mode, *, platform=None):
    # A real interpreter/gate/import path; changing this string tests admission
    # decisions on other hosts, not native operating-system enforcement.
    program = """
import sys
from ltspice_mcp.lib import parser_bootstrap, store
platform, *arguments = sys.argv[1:]
if platform:
    sys.platform = platform
sys.argv = ["parser_bootstrap", *arguments]
raise SystemExit(parser_bootstrap.main())
"""
    executable, env = python_launch()
    process = subprocess.Popen(
        [
            executable,
            "-c",
            program,
            platform or "",
            mode,
            str(directory),
            str(limits.memory_bytes),
            str(limits.request_bytes),
            str(limits.error_bytes),
            str(time.monotonic() + 5),
            str(limits.cleanup_grace_s),
            "parser_fixture",
            "0",
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        cwd=directory,
        env=env,
    )
    try:
        _, stderr = process.communicate(b'{"version":1,"op":"go","request":{}}\n', timeout=10)
        assert process.returncode != 0
        assert b"containment" in stderr
        assert {path.name for path in directory.iterdir()} == {"parser_fixture.py"}
    finally:
        if process.poll() is None:
            process.kill()
        process.wait(timeout=5)


@pytest.mark.parametrize("platform", ["darwin", "freebsd14"])
@pytest.mark.parametrize("mode", ["worker", "guardian"])
def test_unsupported_bootstrap_refuses_even_an_open_gate(call_dir, limits, platform, mode):
    _refused_bootstrap(call_dir(), limits, mode, platform=platform)


@pytest.mark.skipif(sys.platform != "darwin", reason="Requires a native macOS interpreter")
@pytest.mark.parametrize("mode", ["worker", "guardian"])
def test_native_macos_bootstrap_refuses_even_an_open_gate(call_dir, limits, mode):
    _refused_bootstrap(call_dir(), limits, mode)


@pytest.mark.parametrize("failure", ["json", "directory"])
def test_preparation_failure_confirms_no_child(call_dir, limits, failure):
    directory = call_dir()
    request: dict[str, JsonValue] = {"value": float("nan")} if failure == "json" else {}
    if failure == "directory":
        directory = directory / "missing"
    with pytest.raises(ParserProcessError) as caught:
        run_parser_sync(request, work_dir=directory, deadline=time.monotonic() + 5, limits=limits)
    assert caught.value.code == "supervisor_error" and caught.value.reaped
    assert caught.value.worker_pid is None


def test_stdio_setup_failure_confirms_no_child_and_allows_fresh_worker(
    call_dir, limits, monkeypatch
):
    directory = call_dir()
    with monkeypatch.context() as setup:
        setup.setattr(os, "devnull", str(directory / "missing" / "null"))
        with pytest.raises(ParserProcessError) as caught:
            run_parser_sync(
                {},
                work_dir=directory,
                deadline=time.monotonic() + 5,
                limits=limits,
                _worker_module="parser_fixture",
            )
    assert caught.value.code == "supervisor_error" and caught.value.reaped
    assert isinstance(caught.value.__cause__, FileNotFoundError)
    assert caught.value.worker_pid is None
    assert not parser_file_in(directory, "imported.txt").exists()
    fresh = call_dir("fresh")
    reply = run_parser_sync(
        {},
        work_dir=fresh,
        deadline=time.monotonic() + 5,
        limits=limits,
        _worker_module="parser_fixture",
    )
    assert reply.metadata["value"] == pytest.approx(2 / 3)
    assert parser_file_in(fresh, "imported.txt").exists()
    _assert_gone([reply.worker_pid])


def test_exec_failure_keeps_unconfirmed_service_admission(state_no_sim, work_dir, monkeypatch):
    from ltspice_mcp.lib import services
    from ltspice_mcp.lib.result_cache import ParserCleanupError

    log = work_dir / "pre-spawn.log"
    log.write_text("complete\n", encoding="ascii")
    source = services.AnalysisSource(
        raw=None,
        log=log,
        netlist=None,
        dialect=None,
        identity=None,
        trusted_job_artifact=False,
    )
    monkeypatch.setattr(
        parser_process, "python_launch", lambda: (str(work_dir / "python-missing.exe"), None)
    )
    with pytest.raises(ParserCleanupError) as caught:
        services.load_logs_sync(source, state_no_sim)
    cause = caught.value.__cause__
    assert isinstance(cause, ParserProcessError) and not cause.reaped
    assert isinstance(cause.__cause__, FileNotFoundError)
    assert caught.value.directory.is_dir()
    with pytest.raises(ParserCleanupError) as repeated:
        services.load_logs_sync(source, state_no_sim)
    assert repeated.value.directory == caught.value.directory


@pytest.mark.skipif(sys.platform != "linux", reason="Requires Linux descriptor limits")
def test_real_descriptor_failure_releases_service_admission(call_dir):
    directory = call_dir()
    script = directory / "descriptor_probe.py"
    script.write_text(
        """
import errno, json, os, resource
from pathlib import Path
os.environ['LTSPICE_MCP_DISABLE_SIMULATOR_DETECTION'] = '1'
os.environ['LTSPICE_MCP_HOME'] = str(Path.cwd() / 'session-home')
from ltspice_mcp.config import ServerConfig
from ltspice_mcp.errors import ResultError
from ltspice_mcp.lib import services
from ltspice_mcp.lib.parser_process import ParserProcessError
from ltspice_mcp.state import SessionState
directory = Path.cwd()
state = SessionState.create(ServerConfig(working_dir=directory,
    allowed_paths=[directory], persist_jobs=False), {}, sandbox_pinned=True)
log = directory / 'input.log'
log.write_text('complete\\n', encoding='ascii')
source = services.AnalysisSource(raw=None, log=log, netlist=None, dialect=None,
    identity=None, trusted_job_artifact=False)
original = resource.getrlimit(resource.RLIMIT_NOFILE)
count = len(list(Path('/proc/self/fd').iterdir())) - 1
try:
    resource.setrlimit(resource.RLIMIT_NOFILE, (count + 2, original[1]))
    try:
        services.load_logs_sync(source, state)
    except ResultError as error:
        cause = error.__cause__
        assert isinstance(cause, ParserProcessError) and cause.reaped
        assert isinstance(cause.__cause__, OSError) and cause.__cause__.errno == errno.EMFILE
    else:
        raise AssertionError('Expected actual descriptor exhaustion')
finally:
    resource.setrlimit(resource.RLIMIT_NOFILE, original)
assert len(list(Path('/proc/self/fd').iterdir())) - 1 == count
assert not list((state.store.root / 'parsing').glob('*'))
assert services.load_logs_sync(source, state).scan['complete']
assert not list((state.store.root / 'parsing').glob('*'))
print(json.dumps({'descriptor_failure_reaped': True, 'fresh_call_succeeded': True}))
""",
        encoding="utf-8",
    )
    executable, environment = python_launch()
    result = subprocess.run(
        [executable, str(script)],
        cwd=directory,
        env=environment,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {
        "descriptor_failure_reaped": True,
        "fresh_call_succeeded": True,
    }


def test_error_after_child_creation_remains_unconfirmed(call_dir, limits, monkeypatch):
    directory = call_dir()
    popen = subprocess.Popen
    started = []

    def fail_after_creation(*args, **kwargs):
        started.append(popen(*args, **kwargs))
        raise OSError(errno.EIO, "Lost launch receipt after child creation")

    monkeypatch.setattr(parser_process.subprocess, "Popen", fail_after_creation)
    try:
        with pytest.raises(ParserProcessError) as caught:
            run_parser_sync({}, work_dir=directory, deadline=time.monotonic() + 5, limits=limits)
        assert caught.value.code == "supervisor_error" and not caught.value.reaped
        assert not parser_file_in(directory, "imported.txt").exists()
    finally:
        for process in started:
            process.kill()
            process.wait(timeout=3)


def test_existing_control_files_do_not_claim_empty_tree(call_dir, limits):
    directory = call_dir()
    parser_file_in(directory, "process.json").write_text('{"worker_pid": 1}', encoding="utf-8")
    with pytest.raises(ParserProcessError) as caught:
        run_parser_sync({}, work_dir=directory, deadline=time.monotonic() + 5, limits=limits)
    assert not caught.value.reaped


def test_bootstrap_waits_before_import_or_input_access(call_dir, limits):
    directory = call_dir()
    executable, env = python_launch()
    process = subprocess.Popen(
        [
            executable,
            "-m",
            "ltspice_mcp.lib.parser_bootstrap",
            "guardian" if sys.platform == "linux" else "worker",
            str(directory),
            str(limits.memory_bytes),
            str(limits.request_bytes),
            str(limits.error_bytes),
            str(time.monotonic() + 5),
            "2",
            "parser_fixture",
            "0",
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        cwd=directory,
        env=env,
    )
    try:
        time.sleep(0.2)
        assert process.poll() is None
        assert not (directory / "imported.txt").exists()
        assert process.stdin is not None
        process.stdin.close()
        process.wait(timeout=5)
        assert not (directory / "imported.txt").exists()
    finally:
        if process.poll() is None:
            process.kill()
        process.wait(timeout=5)


def test_windows_job_default_and_parser_limits(monkeypatch):
    captured = []

    class Kernel:
        def CreateJobObjectW(self, *args):
            return 1

        def SetInformationJobObject(self, handle, kind, pointer, size):
            limits = ctypes.cast(pointer, ctypes.POINTER(windows_job._ExtendedLimits)).contents
            captured.append(
                (
                    limits.BasicLimitInformation.LimitFlags,
                    limits.ProcessMemoryLimit,
                    limits.JobMemoryLimit,
                )
            )
            return 1

        def OpenProcess(self, *args):
            return 2

        def AssignProcessToJobObject(self, *args):
            return 1

        def CloseHandle(self, *args):
            return 1

    monkeypatch.setattr(windows_job, "_kernel", Kernel)
    ordinary = windows_job.WindowsJob(1)
    ordinary.close()
    strict = windows_job.WindowsJob(1, allow_breakaway=False, memory_limit_bytes=134217728)
    strict.close()
    assert captured == [(0x2000 | 0x0800, 0, 0), (0x2000 | 0x0100 | 0x0200, 134217728, 134217728)]
