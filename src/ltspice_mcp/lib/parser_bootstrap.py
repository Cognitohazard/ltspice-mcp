"""Fixed parser gate and Linux guardian; imports no decoder before admission.

An unsupported host is refused here as well as by the supervisor, so direct
bootstrap execution cannot bypass the containment requirement.
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


def _write(directory: Path, name: str, value: dict[str, Any]) -> None:
    _path(directory, name).write_text(
        json.dumps(value, allow_nan=False, separators=(",", ":")), encoding="utf-8"
    )


def _error(
    directory: Path, error: BaseException, limit: int, *, code: str = "worker_error"
) -> None:
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
    import resource

    resource.setrlimit(resource.RLIMIT_AS, (memory, memory))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


def _read_gate(fd: int, size: int, deadline: float) -> bytes:
    data = bytearray()
    with selectors.DefaultSelector() as selector:
        selector.register(fd, selectors.EVENT_READ)
        while time.monotonic() < deadline:
            if not selector.select(min(0.05, max(0, deadline - time.monotonic()))):
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


def _decoder(
    directory: Path,
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
    try:
        # Windows pipe selectors are unavailable; the parent gates and supervises
        # this bounded read only after assigning its strict Job Object.
        if sys.platform == "win32":
            packet = sys.stdin.buffer.readline(size + 1)
        else:
            packet = _read_gate(sys.stdin.fileno(), size, deadline)
        if len(packet) > size:
            raise ValueError("Parser request exceeds its byte limit")
        request = json.loads(packet)
        if (
            not isinstance(request, dict)
            or request.get("version") != 1
            or request.get("op") != "go"
            or not isinstance(request.get("request"), dict)
        ):
            raise ValueError("Invalid parser gate")
        _write(directory, "process.json", {"worker_pid": os.getpid()})
        worker = importlib.import_module(module)
        worker.parse_request(request["request"], directory)
        return 0
    except BaseException as error:
        _error(directory, error, errors)
        return 1


def _reap_tree(child: subprocess.Popen[bytes], grace: float) -> bool:
    deadline = time.monotonic() + grace
    with contextlib.suppress(ProcessLookupError):
        os.killpg(child.pid, signal.SIGKILL)
    try:
        child.wait(timeout=max(0.01, deadline - time.monotonic()))
    except subprocess.TimeoutExpired:
        return False
    children_file = Path(f"/proc/{os.getpid()}/task/{os.getpid()}/children")
    while time.monotonic() < deadline:
        # Subreaper adoption includes descendants which started another session.
        with children_file.open("rb") as stream:
            children = stream.read(65537)
        if len(children) > 65536:
            return False
        for pid in children.split():
            with contextlib.suppress(ProcessLookupError):
                os.kill(int(pid), signal.SIGKILL)
        try:
            while os.waitpid(-1, os.WNOHANG)[0]:
                pass
        except ChildProcessError:
            return True
        time.sleep(0.005)
    return False


def _guardian(
    directory: Path,
    memory: int,
    size: int,
    errors: int,
    deadline: float,
    grace: float,
    module: str,
) -> int:
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
                str(directory),
                str(memory),
                str(size),
                str(errors),
                str(deadline),
                str(grace),
                module,
                str(os.getpid()),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=None,
            start_new_session=True,
            bufsize=0,
        )
        _write(directory, "process.json", {"worker_pid": child.pid})
        assert child.stdin is not None
        with child.stdin:
            remaining = memoryview(packet)
            while remaining:
                written = child.stdin.write(remaining)
                if not written:
                    raise BrokenPipeError("Parser gate closed before delivery")
                remaining = remaining[written:]
        with selectors.DefaultSelector() as selector:
            selector.register(sys.stdin.fileno(), selectors.EVENT_READ)
            while child.poll() is None:
                if time.monotonic() >= deadline:
                    raise TimeoutError("Parser deadline expired")
                if selector.select(min(0.01, max(0, deadline - time.monotonic()))):
                    if not os.read(sys.stdin.fileno(), 1):
                        raise EOFError("Parser owner died or cancelled the call")
                    raise ValueError("Unexpected bytes after parser admission")
        if child.returncode != 0 and not _path(directory, "error.json").exists():
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
    return 0 if reaped else 1


def main() -> int:
    require_containment_platform()
    mode, directory, memory, size, errors, deadline, grace, module, guardian_pid = sys.argv[1:]
    args = (Path(directory), int(memory), int(size), int(errors), float(deadline))
    if mode == "guardian" and sys.platform == "linux":
        return _guardian(*args, float(grace), module)
    if mode == "worker":
        return _decoder(*args, module, int(guardian_pid))
    raise ValueError("Invalid parser bootstrap mode")


if __name__ == "__main__":
    raise SystemExit(main())
