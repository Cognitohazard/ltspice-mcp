"""The macOS parser worker's self-containment, step by step and end to end.

The steps that need no macOS kernel interface (the owner pipe, the owner
thread's signal mask and exit guard, the limit proofs' refusals) run on any
POSIX host, each in a fresh interpreter where it changes process-wide state.
The steps that read Mach thread state or rest on macOS limits run only there.
"""

from __future__ import annotations

import ast
import json
import os
import select
import shutil
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

from ltspice_mcp.lib import parser_bootstrap
from ltspice_mcp.lib.parser_process import (
    ParserProcessError,
    ParserProcessLimits,
    WarmParser,
    run_parser_sync,
)
from ltspice_mcp.lib.store import parser_file_in
from ltspice_mcp.lib.windows_job import python_launch
from tests import _source_scan
from tests.conftest import LIVENESS_S

posix = pytest.mark.skipif(os.name != "posix", reason="POSIX pipes, signals and limits")
macos = pytest.mark.skipif(sys.platform != "darwin", reason="Requires a native macOS kernel")

_FIXTURE = """
import json, os
from pathlib import Path
from ltspice_mcp.lib.store import parser_file_in

Path("imported.txt").write_text("imported", encoding="utf-8")

def parse_request(request, work_dir):
    held = bytearray(request.get("bytes", 0))
    held[::4096] = b"x" * len(held[::4096])
    parser_file_in(work_dir, "result.json").write_text(json.dumps({
        "held": len(held), "pid": os.getpid(),
    }), encoding="utf-8")
"""


def _limits(memory: int = 256 * 1024 * 1024) -> ParserProcessLimits:
    return ParserProcessLimits(
        memory_bytes=memory,
        request_bytes=65536,
        metadata_bytes=4096,
        error_bytes=1024,
        cleanup_grace_s=LIVENESS_S,
    )


@pytest.fixture
def call_dir(tmp_path):
    def create(name="call"):
        directory = tmp_path / name
        directory.mkdir()
        parser_file_in(directory, "parser_fixture.py").write_text(_FIXTURE, encoding="utf-8")
        return directory

    return create


def _python(
    program: str, *, pass_fds: tuple[int, ...] = (), cwd: Path | None = None
) -> subprocess.Popen[bytes]:
    executable, env = python_launch()
    return subprocess.Popen(
        [executable, "-c", textwrap.dedent(program)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        pass_fds=pass_fds,
        cwd=cwd,
    )


def _finish(process: subprocess.Popen[bytes]) -> tuple[int, str, str]:
    try:
        out, err = process.communicate(timeout=LIVENESS_S)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=LIVENESS_S)
    return process.returncode, out.decode() if out else "", err.decode() if err else ""


def _run(
    program: str, *, pass_fds: tuple[int, ...] = (), cwd: Path | None = None
) -> tuple[int, str, str]:
    return _finish(_python(program, pass_fds=pass_fds, cwd=cwd))


# The owner pipe ------------------------------------------------------------


@posix
def test_the_owner_pipe_is_left_full_and_blocking():
    read, write = os.pipe()
    try:
        code, out, err = _run(
            f"""
            import os, select
            from ltspice_mcp.lib import parser_bootstrap
            parser_bootstrap._fill_owner_pipe({write})
            print(os.get_blocking({write}), bool(select.select([], [{write}], [], 0)[1]))
            """,
            pass_fds=(write,),
        )
    finally:
        os.close(write)
    try:
        assert code == 0, err
        # Blocking, and no room for even one byte.
        assert out.split() == ["True", "False"]
        assert select.select([read], [], [], 0)[0]
    finally:
        os.close(read)


@posix
@pytest.mark.parametrize("descriptor", ["file", "read end"])
def test_an_owner_descriptor_that_is_not_a_pipe_write_end_is_refused(tmp_path, descriptor):
    if descriptor == "file":
        fd = os.open(tmp_path / "owner", os.O_WRONLY | os.O_CREAT)
        other = None
    else:
        fd, other = os.pipe()
    try:
        code, _, err = _run(
            f"""
            from ltspice_mcp.lib import parser_bootstrap
            try:
                parser_bootstrap._fill_owner_pipe({fd})
            except parser_bootstrap.ContainmentFailed as error:
                raise SystemExit(str(error))
            """,
            pass_fds=(fd,),
        )
    finally:
        os.close(fd)
        if other is not None:
            os.close(other)
    assert code == 1 and "owner pipe:" in err


# The owner thread -----------------------------------------------------------

_OWNER = """
import ctypes, os, signal, sys, threading, time
from ltspice_mcp.lib import parser_bootstrap
fd = int(sys.argv[1])
parser_bootstrap._fill_owner_pipe(fd)
owner = parser_bootstrap._start_owner_thread(fd, time.monotonic() + 30)
"""


@posix
def test_the_owner_ends_the_worker_while_another_thread_holds_the_gil():
    read, write = os.pipe()
    # A signal sent to the owner thread is held by its mask; without the mask
    # it would end the owner's write, and the owner would then wait for the
    # GIL that the main thread holds, so the pipe's close would end nothing.
    program = (
        _OWNER
        + """
signal.signal(signal.SIGUSR1, lambda *_: None)
if sys.platform == "darwin":
    parser_bootstrap._darwin_owner_proof(owner, time.monotonic() + 30)
else:
    # Wait until the kernel shows the owner thread asleep in the pipe write.
    task = f"/proc/self/task/{owner.native_id}"
    give_up = time.monotonic() + 30
    while not (
        open(f"{task}/stat").read().rsplit(")", 1)[1].split()[0] == "S"
        and "pipe_write" in open(f"{task}/wchan").read()
    ):
        if time.monotonic() >= give_up:
            raise SystemExit("the owner thread was not seen asleep in write")
        time.sleep(0.01)
signal.pthread_kill(owner.ident, signal.SIGUSR1)
print("holding", flush=True)
ctypes.PyDLL(None).sleep(60)
"""
    )
    executable, env = python_launch()
    process = subprocess.Popen(
        [executable, "-c", program, str(write)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        pass_fds=(write,),
    )
    os.close(write)
    try:
        assert process.stdout is not None
        assert process.stdout.readline() == b"holding\n", process.stderr
        os.close(read)
        read = -1
        assert process.wait(timeout=LIVENESS_S) == -13  # SIGPIPE
    finally:
        if read >= 0:
            os.close(read)
        if process.poll() is None:
            process.kill()
        process.wait(timeout=LIVENESS_S)


@posix
def test_an_owner_write_that_returns_ends_the_worker():
    read, write = os.pipe()
    program = _OWNER + 'print("waiting", flush=True)\ntime.sleep(60)\n'
    executable, env = python_launch()
    process = subprocess.Popen(
        [executable, "-c", program, str(write)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        pass_fds=(write,),
    )
    os.close(write)
    try:
        assert process.stdout is not None
        assert process.stdout.readline() == b"waiting\n"
        os.read(read, 65536)  # Room for the owner's byte: its write returns.
        assert process.wait(timeout=LIVENESS_S) == parser_bootstrap._OWNER_EXIT
    finally:
        os.close(read)
        if process.poll() is None:
            process.kill()
        process.wait(timeout=LIVENESS_S)


# The limit proofs -------------------------------------------------------------


@posix
def test_a_limit_that_does_not_read_back_is_refused():
    import resource

    with pytest.raises(parser_bootstrap.ContainmentFailed, match="RLIMIT_NPROC"):
        parser_bootstrap._prove_limit("RLIMIT_NPROC", resource.RLIMIT_NPROC, 0)


@posix
def test_a_mapping_over_the_budget_that_succeeds_is_refused():
    # Here no address-space limit holds, so the mapping succeeds.
    with pytest.raises(parser_bootstrap.ContainmentFailed, match="over the memory budget"):
        parser_bootstrap._prove_over_budget(64 * 1024 * 1024)


@posix
def test_a_trial_fork_that_succeeds_is_refused():
    code, _, err = _run(
        """
        from ltspice_mcp.lib import parser_bootstrap
        try:
            parser_bootstrap._prove_no_fork()
        except parser_bootstrap.ContainmentFailed as error:
            raise SystemExit(str(error))
        """
    )
    assert code == 1 and "trial fork succeeded" in err


@posix
def test_a_mapping_under_the_budget_that_fails_is_refused():
    code, _, err = _run(
        """
        import resource
        from ltspice_mcp.lib import parser_bootstrap
        import sys
        # macOS refuses a limit below the address space already mapped, so
        # there the limit and the budget are measured from that base.
        base = parser_bootstrap._darwin_base() if sys.platform == "darwin" else 0
        _, hard = resource.getrlimit(resource.RLIMIT_AS)
        resource.setrlimit(resource.RLIMIT_AS, (base + (512 << 20), hard))
        try:
            parser_bootstrap._prove_under_budget(base + (1 << 30))
        except parser_bootstrap.ContainmentFailed as error:
            raise SystemExit(str(error))
        """
    )
    assert code == 1 and "under the memory budget" in err, err


# The supervisor ---------------------------------------------------------------


@posix
def test_the_parser_tree_leads_its_own_session(call_dir, monkeypatch):
    started: list[subprocess.Popen[bytes]] = []
    real = subprocess.Popen

    class Recording(real):  # type: ignore[valid-type, misc]
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            started.append(self)

    monkeypatch.setattr(subprocess, "Popen", Recording)
    directory = call_dir()
    parser = WarmParser(_limits(), directory, _worker_module="parser_fixture")
    try:
        reply = parser.call(
            {}, work_dir=directory, deadline=time.monotonic() + LIVENESS_S, cancel=None
        )
        assert reply.metadata["held"] == 0
        (tree,) = started
        assert os.getsid(tree.pid) == tree.pid
    finally:
        parser.close()


def test_macos_refuses_root(monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(os, "getuid", lambda: 1000)
    monkeypatch.setattr(os, "geteuid", lambda: 1000)
    parser_bootstrap.require_containment_platform()
    for name in ("getuid", "geteuid"):
        with monkeypatch.context() as root:
            root.setattr(os, name, lambda: 0)
            with pytest.raises(RuntimeError, match="non-root"):
                parser_bootstrap.require_containment_platform()


def test_a_containment_reply_fits_one_line():
    reply = parser_bootstrap._containment_reply(
        parser_bootstrap.ContainmentFailed("owner proof: " + "x" * 100000)
    )
    line = json.dumps({"version": 1, **reply}, separators=(",", ":"))
    assert len(line) < parser_bootstrap._REPLY_BYTES
    assert reply["code"] == "ownership_failed" and reply["message"].startswith("owner proof")


# The rules the containment rests on ------------------------------------------------


def _close_fds_false(tree: ast.AST) -> list[int]:
    return [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        for keyword in node.keywords
        if keyword.arg == "close_fds"
        and isinstance(keyword.value, ast.Constant)
        and keyword.value.value is False
    ]


def test_the_scan_for_inherited_descriptors_finds_one():
    tree = ast.parse("subprocess.Popen(['x'], close_fds=False)\nsubprocess.run([], close_fds=True)")
    assert _close_fds_false(tree) == [1]


def test_no_process_inherits_the_servers_descriptors():
    # A child that kept every descriptor would hold the owner pipe's read end,
    # so a macOS worker would outlive its supervisor.
    found = {
        path: lines
        for path, tree in _source_scan.modules().items()
        if (lines := _close_fds_false(tree))
    }
    assert not found


# On macOS ---------------------------------------------------------------------


def _parse(directory: Path, *, warm: bool) -> None:
    if not warm:
        run_parser_sync(
            {},
            work_dir=directory,
            deadline=time.monotonic() + LIVENESS_S,
            limits=_limits(),
            _worker_module="parser_fixture",
        )
        return
    parser = WarmParser(_limits(), directory, _worker_module="parser_fixture")
    try:
        parser.call({}, work_dir=directory, deadline=time.monotonic() + LIVENESS_S, cancel=None)
    finally:
        parser.close()


@macos
@pytest.mark.parametrize("warm", [False, True])
def test_a_regular_file_as_owner_fails_before_any_import(call_dir, tmp_path, monkeypatch, warm):
    directory = call_dir()
    real_pipe = os.pipe
    replaced: list[int] = []

    def file_as_owner():
        # Only the owner pipe, the supervisor's first: subprocess makes the
        # rest, and they must stay pipes.
        if replaced:
            return real_pipe()
        read, write = real_pipe()
        os.close(write)
        replaced.append(read)
        return read, os.open(tmp_path / "owner", os.O_WRONLY | os.O_CREAT)

    monkeypatch.setattr(os, "pipe", file_as_owner)
    with pytest.raises(ParserProcessError) as caught:
        _parse(directory, warm=warm)
    assert caught.value.code == "ownership_failed"
    assert "owner pipe" in str(caught.value)
    assert not parser_file_in(directory, "imported.txt").exists()


@macos
def test_the_owner_proof_refuses_a_thread_that_is_not_in_write():
    import threading

    release = threading.Event()
    thread = threading.Thread(target=release.wait, daemon=True)
    thread.start()
    try:
        with pytest.raises(parser_bootstrap.ContainmentFailed, match="not seen asleep in write"):
            # timing: the deadline the proof gives up at, short so the refusal comes quickly
            parser_bootstrap._darwin_owner_proof(thread, time.monotonic() + 0.5)
    finally:
        release.set()
        thread.join(LIVENESS_S)


@macos
def test_an_inherited_lower_address_space_limit_fails_naming_it(call_dir):
    directory = call_dir()
    code, out, err = _run(
        """
        import json, resource, time
        from pathlib import Path
        from ltspice_mcp.lib import parser_bootstrap
        from ltspice_mcp.lib.parser_process import (
            ParserProcessError, ParserProcessLimits, run_parser_sync)
        # The worker starts with another allocator configuration, so its base is
        # not this process's; a budget far beyond the inherited headroom cannot
        # fit under the inherited limit whatever the worker's base is.
        base = parser_bootstrap._darwin_base()
        resource.setrlimit(resource.RLIMIT_AS, (base + (256 << 20), base + (256 << 20)))
        try:
            run_parser_sync({}, work_dir=Path.cwd(), deadline=time.monotonic() + 60,
                limits=ParserProcessLimits(64 << 30, 65536, 4096, 1024, 5),
                _worker_module="parser_fixture")
        except ParserProcessError as error:
            print(json.dumps({"code": error.code, "message": str(error)}))
        else:
            print(json.dumps({"code": None, "message": "the worker ran"}))
        """,
        cwd=directory,
    )
    assert code == 0, err
    failure = json.loads(out)
    assert failure["code"] == "ownership_failed" and "RLIMIT_AS" in failure["message"]
    assert not parser_file_in(directory, "imported.txt").exists()


@macos
def test_a_decode_under_the_budget_less_the_allowance_succeeds(call_dir):
    limits = _limits()
    held = limits.memory_bytes - parser_bootstrap._proof_allowance(limits.memory_bytes)
    reply = run_parser_sync(
        {"bytes": held},
        work_dir=call_dir(),
        deadline=time.monotonic() + LIVENESS_S,
        limits=limits,
        _worker_module="parser_fixture",
    )
    assert reply.metadata["held"] == held


@macos
def test_a_decode_over_the_budget_is_refused(call_dir):
    limits = _limits()
    with pytest.raises(ParserProcessError) as caught:
        run_parser_sync(
            {"bytes": limits.memory_bytes + (1 << 20)},
            work_dir=call_dir(),
            deadline=time.monotonic() + LIVENESS_S,
            limits=limits,
            _worker_module="parser_fixture",
        )
    assert caught.value.code == "memory_limit"


@macos
def test_root_is_refused_by_the_worker_itself(call_dir):
    sudo = shutil.which("sudo")
    if sudo is None or subprocess.run([sudo, "-n", "true"], check=False).returncode != 0:
        pytest.skip("needs passwordless sudo")
    directory = call_dir()
    executable, env = python_launch()
    root = Path(__file__).resolve().parents[1]
    process = subprocess.Popen(
        [
            sudo,
            "-n",
            "env",
            f"PYTHONPATH={root / 'src'}{os.pathsep}{env.get('PYTHONPATH', '') if env else ''}",
            executable,
            "-m",
            "ltspice_mcp.lib.parser_bootstrap",
            "worker",
            str(directory),
            "134217728",
            "65536",
            "1024",
            str(time.monotonic() + LIVENESS_S),
            "2",
            "parser_fixture",
            "0",
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        cwd=directory,
    )
    try:
        _, stderr = process.communicate(
            b'{"version":1,"op":"go","request":{}}\n', timeout=LIVENESS_S
        )
    finally:
        if process.poll() is None:
            process.kill()
        process.wait(timeout=LIVENESS_S)
    assert process.returncode != 0
    assert b"non-root" in stderr
    assert not parser_file_in(directory, "imported.txt").exists()
