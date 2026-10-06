"""Fixed parser gate and Linux guardian; imports no decoder before admission.

An unsupported host is refused here as well as by the supervisor, so direct
bootstrap execution cannot bypass the containment requirement.

A worker serves requests, one line each, until its input ends: after each it
writes one reply line (``done`` or ``error``) to the supervisor and reads the
next. The first request is the gate. Nothing a request decodes is imported
before it arrives, and on Windows it arrives only once the Job Object holds the
worker. On Linux the guardian reads that gate, starts the worker inside its
limits, and relays every later request to it. The end of the guardian's input
means the supervisor is done or gone: the guardian kills and reaps the whole
tree, reports the reap on its output, and exits.

``directory`` names a one-call tree's control directory (its process, cleanup
and error records); a tree kept between calls has none and reports on its
output alone.
"""

from __future__ import annotations

import contextlib
import ctypes
import importlib
import json
import os
import selectors
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

_REPLY_BYTES = 4096
"""The longest reply line a worker or guardian writes to the supervisor."""


def require_containment_platform() -> None:
    """Refuse hosts without an established memory and process ownership contract."""
    if sys.platform == "darwin":
        raise RuntimeError(
            "macOS parser containment is unavailable: hard memory limits, "
            "owner-death cleanup and escaped-child ownership have not been validated"
        )
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
    if isinstance(error, TimeoutError):
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
) -> int:
    if sys.platform == "linux":
        _prctl(1, signal.SIGKILL)  # PR_SET_PDEATHSIG, before any decoder import.
        if os.getppid() != guardian_pid:
            return 1
        _linux_limits(memory)
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
    with contextlib.suppress(ProcessLookupError):
        os.killpg(child.pid, signal.SIGKILL)
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
    mode, directory, memory, size, errors, deadline, grace, module, guardian_pid = sys.argv[1:]
    control = Path(directory) if directory else None
    args = (control, int(memory), int(size), int(errors), float(deadline))
    if mode == "guardian" and sys.platform == "linux":
        return _guardian(*args, float(grace), module)
    if mode == "worker":
        return _decoder(*args, module, int(guardian_pid))
    raise ValueError("Invalid parser bootstrap mode")


if __name__ == "__main__":
    raise SystemExit(main())
