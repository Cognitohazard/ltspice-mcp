"""Fixed parser gate, Linux guardian and macOS self-containment.

Imports no decoder before admission. An unsupported host is refused here as
well as by the supervisor, so direct bootstrap execution cannot bypass the
containment requirement.

A worker serves requests, one line each, until its input ends: after each it
writes one reply line (``done`` or ``error``) to the supervisor and reads the
next. The first request is the gate. Nothing a request decodes is imported
before it arrives, and on Windows it arrives only once the Job Object holds the
worker. On Linux the guardian reads that gate, starts the worker inside its
limits, and relays every later request to it. The end of the guardian's input
means the supervisor is done or gone: the guardian kills and reaps the whole
tree, reports the reap on its output, and exits.

On macOS there is no guardian. Before it reads the gate the worker contains
itself (``_darwin_containment``): hard limits on its address space and on
starting processes, so the worker is the whole tree, and an owner thread asleep
in ``write`` on a full pipe whose only read end the supervisor holds, so the
worker dies by ``SIGPIPE`` once the supervisor is gone, even while a decoder
holds the GIL.

``directory`` names a one-call tree's control directory (its process, cleanup
and error records); a tree kept between calls has none and reports on its
output alone.
"""

from __future__ import annotations

import contextlib
import ctypes
import importlib
import json
import mmap
import os
import selectors
import signal
import stat
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

_REPLY_BYTES = 4096
"""The longest reply line a worker or guardian writes to the supervisor."""


_OWNER_EXIT = 70
"""The macOS worker's exit status when its owner thread leaves ``write``."""

_OWNER_PROOF_S = 2.0
"""How long the macOS worker waits to see its owner thread asleep in ``write``."""

_OWNER_POLL_S = 0.01


def require_containment_platform() -> None:
    """Refuse hosts without an established memory and process ownership contract."""
    if sys.platform == "darwin":
        # macOS exempts a real uid of 0 from the process limit the worker's
        # containment rests on.
        if os.getuid() == 0 or os.geteuid() == 0:
            raise RuntimeError(
                "parser containment on macOS needs a non-root user: "
                "root ignores the process limit"
            )
        return
    if sys.platform not in {"linux", "win32"}:
        raise RuntimeError("Parser containment is unavailable on this platform")


def _path(directory: Path, name: str) -> Path:
    from ltspice_mcp.lib.store import parser_file_in

    return parser_file_in(directory, name)


def _write(directory: Path | None, name: str, value: dict[str, Any]) -> None:
    if directory is None:
        return
    _path(directory, name).write_text(
        json.dumps(value, allow_nan=False, separators=(",", ":")), encoding="utf-8"
    )


def _reply(fd: int, value: dict[str, Any]) -> None:
    """One line to the supervisor; it may already be gone, which ends nothing here."""
    line = json.dumps({"version": 1, **value}, separators=(",", ":")).encode("ascii") + b"\n"
    with contextlib.suppress(OSError):
        os.write(fd, line)


def _error(
    directory: Path | None, error: BaseException, limit: int, *, code: str = "worker_error"
) -> None:
    if directory is None:
        return
    # Serialize after truncating, then shrink further for JSON escaping overhead.
    message = f"{type(error).__name__}: {error}"[:limit]
    # An error the caller is told about as it is (a read too large, a signal
    # not in the result) names its own code, and its message is its own words.
    named = getattr(type(error), "parser_code", None)
    if isinstance(named, str):
        code, message = named, str(error)[:limit]
    elif isinstance(error, TimeoutError):
        code = "deadline"
    elif isinstance(error, EOFError):
        code = "cancelled"
    elif isinstance(error, MemoryError):
        code = "memory_limit"
    while True:
        encoded = json.dumps({"code": code, "message": message}, ensure_ascii=True).encode("ascii")
        if len(encoded) <= limit:
            _path(directory, "error.json").write_bytes(encoded)
            return
        message = message[: len(message) // 2]


def _prctl(option: int, value: int) -> None:
    function = ctypes.CDLL(None, use_errno=True).prctl
    function.argtypes = [
        ctypes.c_int,
        ctypes.c_ulong,
        ctypes.c_ulong,
        ctypes.c_ulong,
        ctypes.c_ulong,
    ]
    function.restype = ctypes.c_int
    if function(option, value, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "Parser process ownership setup failed")


def _linux_limits(memory: int) -> None:
    if sys.platform != "linux":
        raise RuntimeError("Linux-only parser containment")
    import resource

    resource.setrlimit(resource.RLIMIT_AS, (memory, memory))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


class ContainmentFailed(RuntimeError):
    """A macOS containment step failed; the message names the step and its values."""

    parser_code = "ownership_failed"


@contextlib.contextmanager
def _step(name: str):
    """Report an ``OSError`` or ``ValueError`` inside a containment step as its failure."""
    try:
        yield
    except ContainmentFailed:
        raise
    except (OSError, ValueError) as error:
        raise ContainmentFailed(f"{name}: {type(error).__name__}: {error}") from error


def _proof_allowance(memory: int) -> int:
    """How far under the budget the mapping that must succeed stays."""
    return memory // 2 if memory < 128 * 1024 * 1024 else 64 * 1024 * 1024


def _darwin_base() -> int:
    """The task's virtual size now, which the address-space limit is set above."""
    if sys.platform != "darwin":
        raise RuntimeError("macOS-only parser containment")
    libc = ctypes.CDLL(None, use_errno=True)
    task = ctypes.c_uint.in_dll(libc, "mach_task_self_")
    task_info = libc.task_info
    task_info.argtypes = [
        ctypes.c_uint,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint),
    ]
    task_info.restype = ctypes.c_int
    info = (ctypes.c_uint64 * 6)()
    count = ctypes.c_uint(12)  # MACH_TASK_BASIC_INFO_COUNT
    status = task_info(task, 20, info, ctypes.byref(count))  # MACH_TASK_BASIC_INFO
    if status != 0:
        raise ContainmentFailed(f"limits: task_info failed with kern_return_t {status}")
    return int(info[0])  # virtual_size


def _set_limit(name: str, which: int, value: int, base: int) -> None:
    import resource

    try:
        resource.setrlimit(which, (value, value))
    except (OSError, ValueError) as error:
        hard = resource.getrlimit(which)[1]
        raise ContainmentFailed(
            f"limits: setting {name} to {value} failed (measured base {base}, "
            f"hard limit {hard}): {error}"
        ) from error


def _darwin_limits(memory: int) -> int:
    """Set the hard limits on core files, address space and processes; the address-space cap.

    The address-space limit covers the mappings the worker makes itself, so the
    budget counts what the worker maps after the base was measured. Memory the
    kernel places in the task on another's behalf (Mach out-of-line copies) is
    outside it. Both values of each limit are set: a soft limit alone could be
    raised back to the hard one.
    """
    import resource

    base = _darwin_base()
    page = mmap.PAGESIZE
    limit = -(-(base + memory) // page) * page
    _set_limit("RLIMIT_CORE", resource.RLIMIT_CORE, 0, base)
    _set_limit("RLIMIT_AS", resource.RLIMIT_AS, limit, base)
    _set_limit("RLIMIT_NPROC", resource.RLIMIT_NPROC, 0, base)
    return limit


def _prove_limit(name: str, which: int, value: int) -> None:
    """The limit reads back as ``value`` for both its soft and hard values."""
    import resource

    current = resource.getrlimit(which)
    if current != (value, value):
        raise ContainmentFailed(f"limit proof: {name} is {current}, not ({value}, {value})")


def _prove_over_budget(memory: int) -> None:
    """A mapping one page over the budget is refused."""
    size = memory + mmap.PAGESIZE
    try:
        mapping = mmap.mmap(-1, size, flags=mmap.MAP_PRIVATE)
    except OSError:
        return
    mapping.close()
    raise ContainmentFailed(
        f"limit proof: a mapping of {size} bytes, over the memory budget of {memory}, succeeded"
    )


def _prove_under_budget(memory: int) -> None:
    """A mapping of the budget less an allowance succeeds; it is released at once."""
    size = memory - _proof_allowance(memory)
    try:
        mapping = mmap.mmap(-1, size, flags=mmap.MAP_PRIVATE)
    except OSError as error:
        raise ContainmentFailed(
            f"limit proof: a mapping of {size} bytes, under the memory budget of {memory}, "
            f"failed: {error}"
        ) from error
    mapping.close()


def _prove_no_fork() -> None:
    """A trial fork is refused; run while the worker has one thread."""
    try:
        pid = os.fork()  # noqa: TID251 - the trial that proves the worker cannot fork
    except OSError:
        return
    if pid == 0:
        os._exit(0)
    os.waitpid(pid, 0)
    raise ContainmentFailed("limit proof: a trial fork succeeded, so RLIMIT_NPROC does not hold")


def _fill_owner_pipe(fd: int) -> None:
    """Check that ``fd`` is the write end of a pipe, fill it, and leave it blocking.

    ``SIGPIPE`` is restored to its default first, so a write to the pipe once
    its last reader is gone ends the process without running any Python.
    """
    import fcntl

    signal.signal(signal.SIGPIPE, signal.SIG_DFL)
    with _step("owner pipe"):
        mode = os.fstat(fd).st_mode
        if not stat.S_ISFIFO(mode):
            raise ContainmentFailed(f"owner pipe: descriptor {fd} is not a pipe (mode {mode:o})")
        access = fcntl.fcntl(fd, fcntl.F_GETFL) & (os.O_WRONLY | os.O_RDWR)
        if access != os.O_WRONLY:
            raise ContainmentFailed(f"owner pipe: descriptor {fd} is not a pipe's write end")
        os.set_blocking(fd, False)
        try:
            chunk = 65536
            while chunk:
                try:
                    os.write(fd, bytes(chunk))
                except BlockingIOError:
                    chunk //= 2
        finally:
            os.set_blocking(fd, True)


def _start_owner_thread(fd: int, deadline: float) -> threading.Thread:
    """Start the thread that sleeps in ``write`` on the full owner pipe.

    It blocks every signal but ``SIGPIPE`` first, so no signal ends its sleep
    and leaves it waiting for the GIL; ``SIGPIPE`` stays deliverable to it.
    If ``write`` ever returns or raises, it ends the worker.
    """
    ready = threading.Event()

    def own() -> None:
        try:
            signal.pthread_sigmask(signal.SIG_BLOCK, signal.valid_signals() - {signal.SIGPIPE})
            ready.set()
            os.write(fd, b"\0")
        finally:
            os._exit(_OWNER_EXIT)

    thread = threading.Thread(target=own, name="parser-owner", daemon=True)
    thread.start()
    if not ready.wait(max(0.0, deadline - time.monotonic())):
        raise ContainmentFailed("owner thread: it did not start within the deadline")
    return thread


class _DlInfo(ctypes.Structure):
    _fields_ = [
        ("dli_fname", ctypes.c_char_p),
        ("dli_fbase", ctypes.c_void_p),
        ("dli_sname", ctypes.c_char_p),
        ("dli_saddr", ctypes.c_void_p),
    ]


def _darwin_owner_proof(thread: threading.Thread, deadline: float) -> None:
    """Wait until the owner thread is seen waiting in the kernel inside ``write``."""
    if sys.platform != "darwin":
        raise RuntimeError("macOS-only parser containment")
    libc = ctypes.CDLL(None)
    mach_thread = libc.pthread_mach_thread_np
    mach_thread.argtypes = [ctypes.c_void_p]
    mach_thread.restype = ctypes.c_uint
    thread_info = libc.thread_info
    thread_info.argtypes = [
        ctypes.c_uint,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint),
    ]
    thread_info.restype = ctypes.c_int
    get_state = libc.thread_get_state
    get_state.argtypes = [
        ctypes.c_uint,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint),
    ]
    get_state.restype = ctypes.c_int
    dladdr = libc.dladdr
    dladdr.argtypes = [ctypes.c_void_p, ctypes.POINTER(_DlInfo)]
    dladdr.restype = ctypes.c_int
    machine = os.uname().machine
    if machine == "arm64":
        flavor, count, pc_index, mask = 6, 68, 32, (1 << 47) - 1  # ARM_THREAD_STATE64
    elif machine == "x86_64":
        flavor, count, pc_index, mask = 4, 42, 16, (1 << 64) - 1  # x86_THREAD_STATE64
    else:
        raise ContainmentFailed(f"owner proof: no thread state layout for {machine}")
    port = mach_thread(thread.ident)
    run_state: object = None
    symbol: object = None
    while True:
        basic = (ctypes.c_int * 10)()
        basic_count = ctypes.c_uint(10)  # THREAD_BASIC_INFO_COUNT
        status = thread_info(port, 3, basic, ctypes.byref(basic_count))  # THREAD_BASIC_INFO
        run_state = basic[6] if status == 0 else f"thread_info failed ({status})"
        state = (ctypes.c_uint64 * (count // 2))()
        state_count = ctypes.c_uint(count)
        status = get_state(port, flavor, state, ctypes.byref(state_count))
        if status == 0:
            info = _DlInfo()
            found = dladdr(state[pc_index] & mask, ctypes.byref(info))
            symbol = info.dli_sname.decode("ascii", "replace") if found and info.dli_sname else None
        else:
            symbol = f"thread_get_state failed ({status})"
        if run_state == 3 and symbol in {"write", "__write_nocancel"}:  # TH_STATE_WAITING
            return
        if time.monotonic() >= deadline:
            raise ContainmentFailed(
                f"owner proof: the owner thread was not seen asleep in write within "
                f"{_OWNER_PROOF_S} s (run state {run_state}, at {symbol})"
            )
        time.sleep(_OWNER_POLL_S)


def _darwin_containment(memory: int, owner_fd: int) -> None:
    """Contain this worker on macOS before any decoder is imported.

    Limits first, proved while the worker has one thread; then the owner pipe
    and its thread, proved asleep in ``write``. Any failure raises
    ``ContainmentFailed`` naming the step.
    """
    import resource

    with _step("limits"):
        limit = _darwin_limits(memory)
    with _step("limit proof"):
        _prove_limit("RLIMIT_AS", resource.RLIMIT_AS, limit)
        _prove_limit("RLIMIT_NPROC", resource.RLIMIT_NPROC, 0)
        _prove_over_budget(memory)
        _prove_under_budget(memory)
        _prove_no_fork()
    _fill_owner_pipe(owner_fd)
    deadline = time.monotonic() + _OWNER_PROOF_S
    with _step("owner thread"):
        thread = _start_owner_thread(owner_fd, deadline)
    with _step("owner proof"):
        _darwin_owner_proof(thread, deadline)


def _containment_reply(error: ContainmentFailed) -> dict[str, Any]:
    """The reply line for a failed containment, its message cut to fit one reply."""
    message = str(error)
    while True:
        reply = {"status": "error", "code": "ownership_failed", "message": message,
                 "pid": os.getpid()}
        if len(json.dumps({"version": 1, **reply}, separators=(",", ":"))) < _REPLY_BYTES:
            return reply
        message = message[: len(message) // 2]


def _read_gate(fd: int, size: int, deadline: float) -> bytes:
    data = bytearray()
    with selectors.DefaultSelector() as selector:
        selector.register(fd, selectors.EVENT_READ)
        while time.monotonic() < deadline:
            if not selector.select(max(0, deadline - time.monotonic())):
                continue
            chunk = os.read(fd, min(4096, size + 1 - len(data)))
            if not chunk:
                raise EOFError("Parser owner closed the request gate")
            data.extend(chunk)
            if len(data) > size:
                raise ValueError("Parser request exceeds its byte limit")
            if b"\n" in data:
                if data[-1:] != b"\n" or data.count(b"\n") != 1:
                    raise ValueError("Parser gate needs exactly one JSON request")
                return bytes(data)
    raise TimeoutError("Parser deadline expired before admission")


def _request(packet: bytes, size: int) -> tuple[dict[str, Any], Path]:
    if len(packet) > size:
        raise ValueError("Parser request exceeds its byte limit")
    request = json.loads(packet)
    if (
        not isinstance(request, dict)
        or request.get("version") != 1
        or request.get("op") != "go"
        or not isinstance(request.get("request"), dict)
        or not isinstance(request.get("directory"), str)
    ):
        raise ValueError("Invalid parser gate")
    directory = Path(request["directory"])
    if not directory.is_absolute() or not directory.is_dir():
        raise ValueError("Parser call directory must be an existing absolute directory")
    return request, directory


def _decoder(
    directory: Path | None,
    memory: int,
    size: int,
    errors: int,
    deadline: float,
    module: str,
    guardian_pid: int,
    owner_fd: int = -1,
) -> int:
    if sys.platform == "linux":
        _prctl(1, signal.SIGKILL)  # PR_SET_PDEATHSIG, before any decoder import.
        if os.getppid() != guardian_pid:
            return 1
        _linux_limits(memory)
    if sys.platform == "darwin":
        try:
            _darwin_containment(memory, owner_fd)
        except ContainmentFailed as error:
            _error(directory, error, errors)
            _reply(1, _containment_reply(error))
            return 1
    # Replies keep their own descriptor; anything a decoder prints goes nowhere.
    replies = os.dup(1)
    silent = os.open(os.devnull, os.O_WRONLY)
    os.dup2(silent, 1)
    os.close(silent)
    worker = None
    first = True
    while True:
        call_directory = directory
        try:
            # Windows pipe selectors are unavailable; the parent gates and supervises
            # this bounded read only after assigning its strict Job Object.
            if sys.platform == "win32":
                packet = sys.stdin.buffer.readline(size + 1)
                if not packet:
                    raise EOFError("Parser owner closed the request gate")
            else:
                packet = _read_gate(sys.stdin.fileno(), size, deadline)
            request, call_directory = _request(packet, size)
            if first:
                _write(directory, "process.json", {"worker_pid": os.getpid()})
                first = False
            if worker is None:
                worker = importlib.import_module(module)
            worker.parse_request(request["request"], call_directory)
        except EOFError as error:
            if first:
                _error(directory, error, errors)
                return 1
            return 0
        except BaseException as error:
            _error(call_directory, error, errors)
            _reply(replies, {"status": "error", "pid": os.getpid()})
            return 1
        _reply(replies, {"status": "done", "pid": os.getpid()})


def _reap_tree(child: subprocess.Popen[bytes], grace: float) -> bool:
    """Kill the worker's group and every descendant adopted here, then reap them.

    SIGCHLD is blocked first, so a child that exits between a scan and the
    wait leaves it pending and the wait returns at once: each pass waits for
    the next exit rather than for a timer.
    """
    if sys.platform != "linux":
        raise RuntimeError("Linux-only parser containment")
    deadline = time.monotonic() + grace
    signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGCHLD})
    # The worker's group, only while the worker is not yet reaped: once
    # ``child.poll()`` reaped it, its pid may name another process. What it
    # left is then this subreaper's to stop, through the scan below.
    from ltspice_mcp.lib.proc_kill import child_start_marker, kill_process_group

    kill_process_group(child.pid, child_start_marker(child))
    children_file = Path(f"/proc/{os.getpid()}/task/{os.getpid()}/children")
    while True:
        # Subreaper adoption includes descendants which started another session.
        with children_file.open("rb") as stream:
            children = stream.read(65537)
        if len(children) > 65536:
            return False
        for pid in children.split():
            with contextlib.suppress(ProcessLookupError):
                os.kill(int(pid), signal.SIGKILL)
        try:
            while pid := os.waitpid(-1, os.WNOHANG)[0]:
                if pid == child.pid:
                    child.returncode = -signal.SIGKILL  # Reaped here, not by Popen.
        except ChildProcessError:
            return True
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        signal.sigtimedwait([signal.SIGCHLD], remaining)


def _exit_watch(child: subprocess.Popen[bytes]) -> int | None:
    """A descriptor that reads as ready once ``child`` exits, where the kernel has one."""
    if sys.platform != "linux":
        raise RuntimeError("Linux-only parser containment")
    try:
        return os.pidfd_open(child.pid)
    except (AttributeError, OSError):
        return None


def _guardian(
    directory: Path | None,
    memory: int,
    size: int,
    errors: int,
    deadline: float,
    grace: float,
    module: str,
) -> int:
    if sys.platform != "linux":
        raise RuntimeError("Linux-only parser containment")
    _linux_limits(memory)
    _prctl(36, 1)  # PR_SET_CHILD_SUBREAPER, before a decoder can create children.
    child = None
    reaped = True
    try:
        packet = _read_gate(sys.stdin.fileno(), size, deadline)
        # Use the same interpreter/environment; no callable or object is pickled.
        child = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "ltspice_mcp.lib.parser_bootstrap",
                "worker",
                str(directory) if directory is not None else "",
                str(memory),
                str(size),
                str(errors),
                str(deadline),
                str(grace),
                module,
                str(os.getpid()),
            ],
            stdin=subprocess.PIPE,
            stderr=None,
            start_new_session=True,
            bufsize=0,
        )
        _write(directory, "process.json", {"worker_pid": child.pid})
        assert child.stdin is not None
        _relay(packet, child.stdin)
        exited = _exit_watch(child)
        try:
            with selectors.DefaultSelector() as selector:
                selector.register(sys.stdin.fileno(), selectors.EVENT_READ)
                if exited is not None:
                    selector.register(exited, selectors.EVENT_READ)
                while child.poll() is None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError("Parser deadline expired")
                    # Without an exit descriptor (an old kernel), look again soon.
                    events = selector.select(
                        remaining if exited is not None else min(remaining, 0.05)
                    )
                    if any(key.fd == sys.stdin.fileno() for key, _ in events):
                        data = os.read(sys.stdin.fileno(), 65536)
                        if not data:
                            break  # The supervisor is done or gone: end the tree.
                        _relay(data, child.stdin)
        finally:
            if exited is not None:
                os.close(exited)
        # Exited on its own, not ended here: without its own record, it crashed.
        if (
            child.returncode not in (None, 0)
            and directory is not None
            and not _path(directory, "error.json").exists()
        ):
            _error(
                directory,
                RuntimeError(f"Parser worker exited with code {child.returncode}"),
                errors,
                code="worker_crashed",
            )
    except BaseException as error:
        _error(directory, error, errors)
    finally:
        if child is not None:
            reaped = _reap_tree(child, grace)
        _write(directory, "cleanup.json", {"reaped": reaped})
        _reply(1, {"cleanup": {"reaped": reaped}})
    return 0 if reaped else 1


def _relay(data: bytes, target: Any) -> None:
    remaining = memoryview(data)
    while remaining:
        written = target.write(remaining)
        if not written:
            raise BrokenPipeError("Parser gate closed before delivery")
        remaining = remaining[written:]


def main() -> int:
    require_containment_platform()
    mode, directory, memory, size, errors, deadline, grace, module, guardian_pid, *rest = (
        sys.argv[1:]
    )
    control = Path(directory) if directory else None
    args = (control, int(memory), int(size), int(errors), float(deadline))
    if mode == "guardian" and sys.platform == "linux":
        return _guardian(*args, float(grace), module)
    if mode == "worker":
        # On macOS the supervisor passes the owner pipe's write end; without
        # one, the owner pipe step fails.
        owner = int(rest[0]) if rest else -1
        return _decoder(*args, module, int(guardian_pid), owner)
    raise ValueError("Invalid parser bootstrap mode")


if __name__ == "__main__":
    raise SystemExit(main())
