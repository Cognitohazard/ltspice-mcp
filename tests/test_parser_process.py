"""Real parser-process containment without simulator discovery or execution."""

from __future__ import annotations

import asyncio
import contextlib
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
    WarmParser,
    run_parser,
    run_parser_sync,
)
from ltspice_mcp.lib.store import parser_file_in
from ltspice_mcp.lib.windows_job import python_launch
from tests.conftest import (
    LIVENESS_S,
    await_until,
    identify,
    process_running,
    wait_until,
    written,
)

_FIXTURE = """
import ctypes, json, os, struct, subprocess, sys
import psutil
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
    if mode == "stray":
        child = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(60)"],
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL, start_new_session=os.name != "nt",
        )
        parser_file_in(work_dir, "result.json").write_text(json.dumps({
            "value": 2 / 3, "pid": os.getpid(), "child_pid": child.pid,
            "child_created": psutil.Process(child.pid).create_time(),
        }), encoding="utf-8")
        return
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
            result = {"escaped": True, "child_pid": child.pid,
                      "child_created": psutil.Process(child.pid).create_time()}
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


async def _started(directory: Path) -> list[psutil.Process | None]:
    """The runaway decoder's processes, identified while they run."""
    started = await await_until(
        written(parser_file_in(directory, "started.json"), json.loads),
        what="the real decoder to reach its runaway seam",
    )
    return [identify(pid) for pid in started.values()]


def _assert_gone(processes):
    for process in processes:
        assert not process_running(process), f"Owned process {process} was not reaped"


async def _success(directory, limits):
    reply = await run_parser(
        {},
        work_dir=directory,
        deadline=time.monotonic() + LIVENESS_S,
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
            # timing: the deadline under test; the decoder reaches its runaway
            # seam well inside it, and the test lasts as long as it does
            deadline=time.monotonic() + 5,
            limits=limits,
            _worker_module="parser_fixture",
        )
    )
    started = await _started(directory)
    with pytest.raises(ParserProcessError) as caught:
        await task
    assert caught.value.code == "deadline" and caught.value.reaped
    _assert_gone(started)
    await _success(call_dir("fresh"), limits)


async def test_repeated_cancellation_waits_for_tree_reaping(call_dir, limits):
    directory = call_dir()
    task = asyncio.create_task(
        run_parser(
            {"mode": "runaway"},
            work_dir=directory,
            deadline=time.monotonic() + LIVENESS_S,
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
    _assert_gone(started)
    await _success(call_dir("fresh"), limits)


def test_sync_cancellation_event_reaps_worker(call_dir, limits):
    directory = call_dir()
    cancel = threading.Event()
    owned: list[psutil.Process | None] = []

    def request_cancel():
        try:
            started = wait_until(
                written(parser_file_in(directory, "started.json"), json.loads),
                what="the real decoder to reach its runaway seam",
            )
            owned.extend(identify(pid) for pid in started.values())
        finally:
            cancel.set()

    canceller = threading.Thread(target=request_cancel)
    canceller.start()
    try:
        with pytest.raises(ParserProcessError) as caught:
            run_parser_sync(
                {"mode": "runaway"},
                work_dir=directory,
                deadline=time.monotonic() + LIVENESS_S,
                limits=limits,
                cancel=cancel,
                _worker_module="parser_fixture",
            )
        assert caught.value.code == "cancelled" and caught.value.reaped
        canceller.join(timeout=LIVENESS_S)
        assert len(owned) == 2
        _assert_gone(owned)
    finally:
        cancel.set()
        canceller.join(timeout=LIVENESS_S)


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
        started = wait_until(
            written(parser_file_in(directory, "started.json"), json.loads),
            what="the real decoder to reach its runaway seam",
        )
        owned = [identify(pid) for pid in started.values()]
        owner.kill()
        owner.wait(timeout=LIVENESS_S)
        wait_until(
            lambda: not any(process_running(process) for process in owned),
            what="the guardian to reap the decoder tree",
        )
        _assert_gone(owned)
        if sys.platform == "linux":
            # The guardian writes its record after the reap returns, so the
            # processes are gone before it lands.
            cleanup = wait_until(
                written(parser_file_in(directory, "cleanup.json"), json.loads),
                what="the guardian's cleanup record",
            )
            assert cleanup["reaped"] is True
    finally:
        if owner.poll() is None:
            owner.kill()
        owner.wait(timeout=LIVENESS_S)
        for process in owned:
            if process is None:
                continue
            # An identified process refuses a kill once its pid names another.
            with contextlib.suppress(psutil.NoSuchProcess):
                process.kill()
                process.wait(timeout=LIVENESS_S)


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
            deadline=time.monotonic() + LIVENESS_S,
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
            deadline=time.monotonic() + LIVENESS_S,
            limits=replace(limits, metadata_bytes=16384),
            _worker_module="parser_fixture",
        )
    assert caught.value.code == "invalid_result" and caught.value.reaped


async def test_oversize_metadata_is_rejected_after_reaping(call_dir, limits):
    with pytest.raises(ParserProcessError) as caught:
        await run_parser(
            {"mode": "invalid", "text": '{"x":"' + "x" * 10000 + '"}'},
            work_dir=call_dir(),
            deadline=time.monotonic() + LIVENESS_S,
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
        deadline=time.monotonic() + LIVENESS_S,
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
            process.wait(timeout=LIVENESS_S)


@pytest.mark.skipif(sys.platform != "win32", reason="Requires native Windows Job enforcement")
async def test_native_windows_parser_cannot_break_away(call_dir, limits):
    reply = await run_parser(
        {"mode": "breakaway"},
        work_dir=call_dir(),
        deadline=time.monotonic() + LIVENESS_S,
        limits=limits,
        _worker_module="parser_fixture",
    )
    try:
        assert reply.metadata["escaped"] is False
    finally:
        pid = reply.metadata.get("child_pid")
        created = reply.metadata.get("child_created")
        if type(pid) is int and type(created) is float and process_running(pid, created):
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
            deadline=time.monotonic() + LIVENESS_S,
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
        deadline=time.monotonic() + LIVENESS_S,
        limits=limits,
        _worker_module="parser_fixture",
    )
    assert parser_file_in(directory, "stderr.txt").stat().st_size == 200
    overflow_dir = call_dir("overflow")
    with pytest.raises(ParserProcessError) as caught:
        await run_parser(
            {"mode": "stderr", "bytes": 100000},
            work_dir=overflow_dir,
            deadline=time.monotonic() + LIVENESS_S,
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
            deadline=time.monotonic() + LIVENESS_S,
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
            deadline=time.monotonic() + LIVENESS_S,
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
        _, stderr = process.communicate(
            b'{"version":1,"op":"go","request":{}}\n', timeout=LIVENESS_S
        )
        assert process.returncode != 0
        assert b"containment" in stderr
        assert {path.name for path in directory.iterdir()} == {"parser_fixture.py"}
    finally:
        if process.poll() is None:
            process.kill()
        process.wait(timeout=LIVENESS_S)


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
        run_parser_sync(
            request, work_dir=directory, deadline=time.monotonic() + LIVENESS_S, limits=limits
        )
    assert caught.value.code == "supervisor_error" and caught.value.reaped
    assert caught.value.worker_pid is None


def test_stdio_setup_failure_confirms_no_child_and_allows_fresh_worker(
    call_dir, limits, monkeypatch
):
    directory = call_dir()

    def no_pipes(*args, **kwargs):
        raise FileNotFoundError("standard stream setup failed")

    with monkeypatch.context() as setup:
        # Before any child exists: the pipes are made inside Popen, ahead of exec.
        setup.setattr(subprocess.Popen, "_get_handles", no_pipes)
        with pytest.raises(ParserProcessError) as caught:
            run_parser_sync(
                {},
                work_dir=directory,
                deadline=time.monotonic() + LIVENESS_S,
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
        deadline=time.monotonic() + LIVENESS_S,
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
        timeout=LIVENESS_S,
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
            run_parser_sync(
                {}, work_dir=directory, deadline=time.monotonic() + LIVENESS_S, limits=limits
            )
        assert caught.value.code == "supervisor_error" and not caught.value.reaped
        assert not parser_file_in(directory, "imported.txt").exists()
    finally:
        for process in started:
            process.kill()
            process.wait(timeout=LIVENESS_S)


def test_existing_control_files_do_not_claim_empty_tree(call_dir, limits):
    directory = call_dir()
    parser_file_in(directory, "process.json").write_text('{"worker_pid": 1}', encoding="utf-8")
    with pytest.raises(ParserProcessError) as caught:
        run_parser_sync(
            {}, work_dir=directory, deadline=time.monotonic() + LIVENESS_S, limits=limits
        )
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
        # timing: a negative window; nothing may import before the gate opens
        time.sleep(0.2)
        assert process.poll() is None
        assert not (directory / "imported.txt").exists()
        assert process.stdin is not None
        process.stdin.close()
        process.wait(timeout=LIVENESS_S)
        assert not (directory / "imported.txt").exists()
    finally:
        if process.poll() is None:
            process.kill()
        process.wait(timeout=LIVENESS_S)


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


# A process that waits to be put in a job, then starts a descendant holding
# enough memory for its exit to take a moment, and says who they are. Each
# further line it is sent asks it to start one more process.
_JOB_ROOT = """
import json, os, subprocess, sys
from pathlib import Path

sys.stdin.readline()
child = subprocess.Popen(
    [sys.executable, "-c", (
        "import os\\n"
        "ballast = bytearray(64 * 1024 * 1024)\\n"
        "for at in range(0, len(ballast), 4096): ballast[at] = 1\\n"
        "print(os.getpid(), flush=True)\\n"
        "while True: pass"
    )],
    stdout=subprocess.PIPE,
)
pids = [os.getpid(), child.pid, int(child.stdout.readline())]
Path("ready.json").write_text(json.dumps(pids), encoding="utf-8")
for _ in sys.stdin:
    try:
        subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    except OSError:
        Path("answer.txt").write_text("refused", encoding="utf-8")
    else:
        Path("answer.txt").write_text("started", encoding="utf-8")
"""


@contextlib.contextmanager
def _job_with_descendants(directory: Path):
    """A job holding a process and its descendants: (job, root, the processes).

    The processes are identified while they run. On a virtual environment the
    descendant is two: the environment's launcher and the interpreter it starts.
    """
    executable, env = python_launch()
    root = subprocess.Popen(
        [executable, "-c", _JOB_ROOT], stdin=subprocess.PIPE, cwd=directory, env=env, text=True
    )
    owned: list[psutil.Process | None] = []
    job = None
    try:
        job = windows_job.WindowsJob(root.pid, allow_breakaway=False, cleanup_timeout_s=LIVENESS_S)
        assert root.stdin is not None
        root.stdin.write("the job holds you\n")
        root.stdin.flush()
        pids = wait_until(
            written(directory / "ready.json", json.loads), what="the job's processes to start"
        )
        owned.extend(identify(pid) for pid in dict.fromkeys(pids))
        yield job, root, owned
    finally:
        if job is not None:
            job.close()
        if root.poll() is None:
            root.kill()
        root.wait(timeout=LIVENESS_S)


_NEEDS_WINDOWS = pytest.mark.skipif(
    sys.platform != "win32", reason="a Windows Job Object needs Windows"
)


@_NEEDS_WINDOWS
def test_a_closed_windows_job_has_no_process_still_exiting(tmp_path):
    """A close with a cleanup bound returns once the job's processes have exited.

    Windows counts a job's processes as gone the moment it is asked to
    terminate them, while they are still exiting with their files open. The
    tree's first process is waited for by the supervisor; its descendants are
    confirmed by the job alone.
    """
    with _job_with_descendants(tmp_path) as (job, _root, owned):
        assert len(owned) >= 2 and all(process_running(process) for process in owned)
        job.close()
        _assert_gone(owned)


@_NEEDS_WINDOWS
def test_a_sealed_windows_job_admits_no_further_process(tmp_path):
    """A close takes a handle to each process before it ends them, so one
    started after the handles were taken would be ended with the rest and
    waited for by nothing. The job is sealed first, which refuses the start."""
    with _job_with_descendants(tmp_path) as (job, root, owned):
        job.seal()
        assert root.stdin is not None
        root.stdin.write("start another\n")
        root.stdin.flush()
        answer = wait_until(
            written(tmp_path / "answer.txt", str.strip), what="the process to try a start"
        )
        assert answer == "refused"
        assert job.active_processes() == len(owned)
        assert all(process_running(process) for process in owned)


# ---------------------------------------------------------------------------
# A tree kept between calls
# ---------------------------------------------------------------------------


@pytest.fixture
def warm(tmp_path, limits):
    """A warm parser running the fixture worker, and a maker of call directories."""
    home = tmp_path / "warm"
    home.mkdir()
    parser_file_in(home, "parser_fixture.py").write_text(_FIXTURE, encoding="utf-8")
    parser = WarmParser(limits, home, _worker_module="parser_fixture")
    calls = iter(range(1000))

    def call(request: dict[str, JsonValue], **kwargs) -> tuple:
        directory = tmp_path / f"call-{next(calls)}"
        directory.mkdir()
        kwargs.setdefault("deadline", time.monotonic() + LIVENESS_S)
        reply = parser.call(
            request, work_dir=directory, cancel=kwargs.pop("cancel", None), **kwargs
        )
        return reply, directory

    yield parser, call
    parser.close()


def test_a_warm_parser_serves_calls_from_one_worker(warm):
    parser, call = warm
    first, directory = call({})
    second, _ = call({})
    assert first.metadata["value"] == second.metadata["value"] == pytest.approx(2 / 3)
    assert first.worker_pid == second.worker_pid == first.metadata["pid"]
    assert struct.unpack("<d", parser_file_in(directory, "array_0000.bin").read_bytes())[
        0
    ] == pytest.approx(2 / 3)
    assert process_running(first.worker_pid)
    assert parser.close()
    _assert_gone([first.worker_pid])


def test_a_cancelled_warm_call_reaps_the_tree_and_the_next_starts_fresh(warm, tmp_path):
    _, call = warm
    before, _ = call({})
    cancel = threading.Event()
    started: list[psutil.Process | None] = []

    def cancel_once_started():
        try:
            pids = wait_until(
                written(parser_file_in(tmp_path / "call-1", "started.json"), json.loads),
                what="the warm decoder to reach its runaway seam",
            )
            started.extend(identify(pid) for pid in pids.values())
        finally:
            cancel.set()

    canceller = threading.Thread(target=cancel_once_started)
    canceller.start()
    try:
        with pytest.raises(ParserProcessError) as caught:
            call({"mode": "runaway"}, cancel=cancel)
    finally:
        cancel.set()
        canceller.join(timeout=LIVENESS_S)
    assert caught.value.code == "cancelled" and caught.value.reaped
    _assert_gone([*started, before.worker_pid])
    after, _ = call({})
    assert after.worker_pid != before.worker_pid


def test_a_stray_process_ends_the_tree_before_its_call_is_read(warm):
    _, call = warm
    reply, _ = call({"mode": "stray"})
    assert reply.metadata["value"] == pytest.approx(2 / 3)
    # The tree held more than its worker after the reply, so it was closed and
    # reaped before the result was read: neither process outlives the call.
    _assert_gone([reply.worker_pid])
    assert not process_running(reply.metadata["child_pid"], reply.metadata["child_created"])
    assert call({})[0].worker_pid != reply.worker_pid


def test_a_crashed_warm_worker_reports_it_and_the_next_call_starts_fresh(warm):
    _, call = warm
    with pytest.raises(ParserProcessError) as caught:
        call({"mode": "crash"})
    assert caught.value.code == "worker_crashed" and caught.value.reaped
    assert call({})[0].metadata["value"] == pytest.approx(2 / 3)


def test_a_warm_tree_is_replaced_after_its_call_budget(warm, monkeypatch):
    monkeypatch.setattr(parser_process, "WARM_CALLS", 2)
    _, call = warm
    pids = [call({})[0].worker_pid for _ in range(3)]
    assert pids[0] == pids[1] != pids[2]
    _assert_gone([pids[0]])


def test_an_idle_warm_tree_is_closed(warm, monkeypatch):
    monkeypatch.setattr(parser_process, "WARM_IDLE_S", 0.0)
    _, call = warm
    reply, _ = call({})
    wait_until(lambda: not process_running(reply.worker_pid), what="the idle tree to be closed")


def test_a_closed_warm_parser_still_closes_its_next_tree_when_idle(warm, monkeypatch):
    parser, call = warm
    first, _ = call({})
    assert parser.close()
    monkeypatch.setattr(parser_process, "WARM_IDLE_S", 0.0)
    second, _ = call({})
    assert second.worker_pid != first.worker_pid
    wait_until(
        lambda: not process_running(second.worker_pid), what="the reopened tree's idle close"
    )


def test_an_unconfirmed_idle_close_refuses_the_next_call(warm, monkeypatch):
    close = parser_process.ParserTree.close
    closed = threading.Event()

    def unconfirmed(tree):
        assert close(tree)
        closed.set()
        return False

    monkeypatch.setattr(parser_process.ParserTree, "close", unconfirmed)
    monkeypatch.setattr(parser_process, "WARM_IDLE_S", 0.0)
    _, call = warm
    call({})
    assert closed.wait(LIVENESS_S)
    with pytest.raises(ParserProcessError) as caught:
        call({})
    assert caught.value.code == "cleanup_failed" and not caught.value.reaped
