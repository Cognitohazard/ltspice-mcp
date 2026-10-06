"""Parser process supervision; decoding stays in a gated, contained child.

A parser process tree serves calls until it is closed: each call sends one
request line and waits for one reply line, and closing the tree kills and reaps
every process in it before anything confirms the close. A one-call tree
(``run_parser_sync``) is closed before its results are read. A session keeps a
``WarmParser`` tree between calls instead, and reads a call's results once the
worker has replied and is again the tree's only process: the worker then waits
for its next request, and nothing else is left that could write.

The caller owns the admitted directory and removes it only after confirmed
cleanup. Numeric files and their manifests are validated by the parser service.
"""

from __future__ import annotations

import asyncio
import atexit
import contextlib
import contextvars
import io
import json
import math
import os
import queue
import subprocess
import sys
import threading
import time
import weakref
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import BinaryIO, TypeAlias

from ltspice_mcp.lib.parser_bootstrap import require_containment_platform
from ltspice_mcp.lib.store import parser_file_in
from ltspice_mcp.lib.windows_job import WindowsJob, python_launch

JsonValue: TypeAlias = bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"] | None
_WORKER_MODULE = "ltspice_mcp.lib.parser_worker"
_KILLED_REAP_WAIT_S = 0.2
"""After a guardian outlives its grace and is killed, how long cleanup waits to
reap it. The call reports the tree unconfirmed either way."""
_REPLY_BYTES = 4096
"""The longest reply line a worker or guardian writes."""
_CANCEL_CHECK_S = 0.05
"""How often a call in flight looks at its cancel event. A reply, an early
exit or excess diagnostics wake it at once; only cancellation waits this long."""
WARM_CALLS = 64
"""Calls one warm tree serves before it is replaced, which bounds what a
long-lived worker can accumulate."""
WARM_IDLE_S = 120.0
"""How long a warm tree may wait for its next call before it is closed."""
WARM_LIFETIME_S = 3600.0
"""The guardian's own bound on a warm tree. The session replaces a tree at half
this age, so a tree in use never meets it."""
_CONTROL_FILES = (
    "request.json",
    "result.json",
    "error.json",
    "process.json",
    "cleanup.json",
    "stderr.txt",
)


@dataclass(frozen=True)
class ParserProcessLimits:
    memory_bytes: int
    request_bytes: int
    metadata_bytes: int
    error_bytes: int
    cleanup_grace_s: float

    def __post_init__(self) -> None:
        for name in ("memory_bytes", "request_bytes", "metadata_bytes", "error_bytes"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.error_bytes < 256:
            raise ValueError("error_bytes must allow a bounded error envelope (256 bytes)")
        if not math.isfinite(self.cleanup_grace_s) or self.cleanup_grace_s <= 0:
            raise ValueError("cleanup_grace_s must be finite and positive")


@dataclass(frozen=True)
class ParserReply:
    metadata: dict[str, JsonValue]
    worker_pid: int


class ParserProcessError(RuntimeError):
    """A failed call, with cleanup evidence needed to retain or remove scratch."""

    def __init__(self, code: str, message: str, *, reaped: bool, worker_pid: int | None = None):
        super().__init__(message)
        self.code = code
        self.reaped = reaped
        self.worker_pid = worker_pid


def _read_json(path: Path, limit: int) -> dict[str, JsonValue]:
    with path.open("rb") as stream:
        data = stream.read(limit + 1)
    if len(data) > limit:
        raise ValueError("Parser JSON exceeds its byte limit")
    text = data.decode("utf-8")
    depth = 0
    quoted = escaped = False
    for character in text:
        if quoted:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                quoted = False
        elif character == '"':
            quoted = True
        elif character in "[{":
            depth += 1
            if depth > 64:
                raise ValueError("Parser JSON exceeds its nesting limit")
        elif character in "]}":
            depth -= 1

    def unique_keys(pairs: list[tuple[str, JsonValue]]) -> dict[str, JsonValue]:
        result: dict[str, JsonValue] = {}
        for name, value in pairs:
            if name in result:
                raise ValueError("Duplicate key in parser JSON")
            result[name] = value
        return result

    def invalid_constant(value: str) -> None:
        raise ValueError(f"Non-finite parser JSON value: {value}")

    def integer(value: str) -> int:
        if len(value) > 256:
            raise ValueError("Parser JSON integer exceeds its digit limit")
        return int(value)

    def floating(value: str) -> float:
        result = float(value)
        if not math.isfinite(result):
            raise ValueError("Non-finite parser JSON number")
        return result

    value = json.loads(
        text,
        object_pairs_hook=unique_keys,
        parse_constant=invalid_constant,
        parse_int=integer,
        parse_float=floating,
    )
    if not isinstance(value, dict):
        raise TypeError("Parser JSON must be an object")
    return value


def _worker_pid(directory: Path) -> int | None:
    try:
        value = _read_json(parser_file_in(directory, "process.json"), 256).get("worker_pid")
        return value if type(value) is int and value > 0 else None
    except (OSError, ValueError, TypeError, RecursionError):
        return None


def _send_request(process: subprocess.Popen[bytes], packet: bytes) -> None:
    assert process.stdin is not None
    try:
        remaining = memoryview(packet)
        while remaining:
            written = process.stdin.write(remaining)
            if not written:
                break
            remaining = remaining[written:]
    except (OSError, ValueError):
        # The supervisor diagnoses worker exit or a deadline, then reaps it.
        pass


def _failed_spawn_reaped(error: Exception) -> bool:
    """Confirm only an OS setup failure before stdlib entered child execution."""
    if not isinstance(error, OSError):
        return False
    constructor = None
    traceback = error.__traceback__
    while traceback is not None:
        frame = traceback.tb_frame
        if frame.f_globals.get("__name__") == "subprocess":
            if frame.f_code.co_qualname == "Popen.__init__":
                constructor = frame.f_locals.get("self")
            elif frame.f_code.co_qualname == "Popen._execute_child":
                return False
        traceback = traceback.tb_next
    return constructor is not None and getattr(constructor, "pid", -1) is None


def _children(pid: int) -> list[int]:
    """Every child of every thread of ``pid`` (Linux)."""
    found: list[int] = []
    for task in Path(f"/proc/{pid}/task").iterdir():
        found.extend(int(child) for child in (task / "children").read_bytes().split())
    return found


def _packet(request: dict[str, JsonValue], directory: Path, limits: ParserProcessLimits) -> bytes:
    packet = (
        json.dumps(
            {
                "version": 1,
                "op": "go",
                "directory": str(directory),
                "request": request,
                "process_limits": asdict(limits),
            },
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )
    if len(packet) > limits.request_bytes:
        raise ParserProcessError(
            "request_limit", "Parser request exceeds its byte limit", reaped=True
        )
    return packet


def _admit_call(deadline: float, cancel: threading.Event | None) -> threading.Event:
    if not math.isfinite(deadline):
        raise ValueError("Parser deadline must be finite")
    cancel = cancel if cancel is not None else threading.Event()
    if cancel.is_set() or time.monotonic() >= deadline:
        code = "cancelled" if cancel.is_set() else "deadline"
        raise ParserProcessError(code, "Parser call ended before spawn", reaped=True)
    return cancel


class _CallFailed(Exception):
    """A call that ended without the worker's reply; ``gone`` means it exited."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


_EOF = object()
_NOISE = object()


class ParserTree:
    """One contained parser process tree whose worker serves calls until closed.

    The first request is the gate: nothing reaches the worker before it, and on
    Windows it is sent only once the Job Object holds the worker. Closing the
    tree ends the guardian's input on Linux, which kills and reaps every
    process the guardian adopted, or terminates the Job Object on Windows; it
    is confirmed only when that is known to have happened.
    """

    def __init__(
        self,
        *,
        limits: ParserProcessLimits,
        cwd: Path,
        control: Path | None,
        lifetime_deadline: float,
        module: str,
    ) -> None:
        self.limits = limits
        self.control = control
        self.calls = 0
        self.started = time.monotonic()
        self.worker_pid: int | None = None
        self.job: WindowsJob | None = None
        self._replies: queue.Queue[object] = queue.Queue()
        self._stdout_closed = threading.Event()
        self._cleanup: object = None
        self._stderr_lock = threading.Lock()
        self._stderr_target: BinaryIO | None = None
        self._stderr_remaining = 0
        self._stderr_overflow = threading.Event()
        self._threads: list[threading.Thread] = []
        self._confirmed: bool | None = None
        executable, env = python_launch()
        env = {
            **(env if env is not None else os.environ),
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "LTSPICE_MCP_DISABLE_SIMULATOR_DETECTION": "1",
        }
        try:
            self.process = subprocess.Popen(
                [
                    executable,
                    "-m",
                    "ltspice_mcp.lib.parser_bootstrap",
                    "guardian" if sys.platform == "linux" else "worker",
                    str(control) if control is not None else "",
                    str(limits.memory_bytes),
                    str(limits.request_bytes),
                    str(limits.error_bytes),
                    str(lifetime_deadline),
                    str(limits.cleanup_grace_s),
                    module,
                    "0",
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=cwd,
                env=env,
                start_new_session=sys.platform == "linux",
                bufsize=0,
            )
        except Exception as exc:
            raise ParserProcessError(
                "supervisor_error",
                str(exc)[: limits.error_bytes],
                reaped=_failed_spawn_reaped(exc),
            ) from exc
        _LIVE_TREES.add(self)
        try:
            if control is not None:
                self._capture_stderr(control)
            self._start(self._read_replies)
            self._start(self._read_stderr)
            if sys.platform == "win32":
                self.job = WindowsJob(
                    self.process.pid,
                    allow_breakaway=False,
                    memory_limit_bytes=limits.memory_bytes,
                    cleanup_timeout_s=limits.cleanup_grace_s,
                )
        except Exception as exc:
            reaped = self.close()
            raise ParserProcessError(
                "ownership_failed", str(exc)[: limits.error_bytes], reaped=reaped
            ) from exc

    def _start(self, target) -> None:
        thread = threading.Thread(target=target, daemon=True)
        self._threads.append(thread)
        thread.start()

    def _read_replies(self) -> None:
        assert self.process.stdout is not None
        stream = io.BufferedReader(self.process.stdout)  # type: ignore[arg-type]
        try:
            while line := stream.readline(_REPLY_BYTES + 1):
                if len(line) > _REPLY_BYTES or not line.endswith(b"\n"):
                    self._replies.put(_NOISE)
                    continue
                try:
                    message = json.loads(line)
                except ValueError:
                    self._replies.put(_NOISE)
                    continue
                if isinstance(message, dict) and "cleanup" in message:
                    self._cleanup = message["cleanup"]
                else:
                    self._replies.put(message)
        except (OSError, ValueError):
            pass
        finally:
            self._stdout_closed.set()
            self._replies.put(_EOF)

    def _read_stderr(self) -> None:
        assert self.process.stderr is not None
        try:
            while chunk := self.process.stderr.read(4096):
                with self._stderr_lock:
                    target = self._stderr_target
                    if target is None:
                        continue  # Between calls: nothing to account it to.
                    accepted = min(len(chunk), self._stderr_remaining)
                    target.write(chunk[:accepted])
                    self._stderr_remaining -= accepted
                    if len(chunk) > accepted and not self._stderr_overflow.is_set():
                        self._stderr_overflow.set()
                        self._replies.put(_NOISE)
        except (OSError, ValueError):
            pass

    def _capture_stderr(self, directory: Path) -> None:
        with self._stderr_lock:
            if self._stderr_target is not None:
                self._stderr_target.close()
            self._stderr_target = parser_file_in(directory, "stderr.txt").open("xb")
            self._stderr_remaining = self.limits.error_bytes
            self._stderr_overflow.clear()

    def _release_stderr(self) -> None:
        with self._stderr_lock:
            if self._stderr_target is not None:
                self._stderr_target.close()
                self._stderr_target = None

    @property
    def stderr_overflowed(self) -> bool:
        return self._stderr_overflow.is_set()

    def call(
        self,
        packet: bytes,
        directory: Path,
        *,
        deadline: float,
        cancel: threading.Event,
    ) -> str:
        """Send one request and wait for its reply: ``done`` or ``error``.

        Raises ``_CallFailed`` when the call ends any other way, with ``gone``
        when the worker exited without replying.
        """
        if self.control is None:
            self._capture_stderr(directory)
        writer = threading.Thread(target=_send_request, args=(self.process, packet), daemon=True)
        self._threads.append(writer)
        writer.start()
        try:
            while True:
                if cancel.is_set():
                    raise _CallFailed("cancelled", "Parser call was cancelled")
                if self._stderr_overflow.is_set():
                    raise _CallFailed("error_limit", "Parser diagnostics exceed their byte limit")
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise _CallFailed("deadline", "Parser call exceeded its deadline")
                try:
                    message = self._replies.get(timeout=min(remaining, _CANCEL_CHECK_S))
                except queue.Empty:
                    continue
                if message is _NOISE:
                    if self._stderr_overflow.is_set():
                        continue
                    raise _CallFailed("supervisor_error", "Parser reply is invalid")
                if message is _EOF:
                    raise _CallFailed("gone", "Parser worker exited without replying")
                if (
                    not isinstance(message, dict)
                    or message.get("version") != 1
                    or message.get("status") not in {"done", "error"}
                    or type(message.get("pid")) is not int
                ):
                    raise _CallFailed("supervisor_error", "Parser reply is invalid")
                self.worker_pid = message["pid"]
                self.calls += 1
                return str(message["status"])
        finally:
            if self.control is None:
                self._release_stderr()

    def contained(self) -> bool:
        """Whether the worker is the only process left in the tree."""
        try:
            if self.job is not None:
                return self.job.active_processes() == 1
            if sys.platform == "linux":
                worker = self.worker_pid
                return (
                    worker is not None
                    and _children(self.process.pid) == [worker]
                    and not _children(worker)
                )
            return True
        except (OSError, ValueError):
            return False

    def close(self) -> bool:
        """Kill and reap the whole tree; True only once that is confirmed."""
        if self._confirmed is not None:
            return self._confirmed
        _LIVE_TREES.discard(self)
        deadline = time.monotonic() + self.limits.cleanup_grace_s
        confirmed = True
        if self.process.stdin is not None:
            with contextlib.suppress(OSError):
                self.process.stdin.close()
        if self.job is not None:
            try:
                self.job.close()
            except (OSError, TimeoutError):
                confirmed = False
        elif sys.platform == "win32":
            # Ownership failed before the gate: this trusted bootstrap has no children.
            with contextlib.suppress(OSError):
                self.process.kill()
        # The tree's last writer to the reply pipe is gone once it reads as closed.
        if self._stdout_closed.wait(max(0.0, deadline - time.monotonic())):
            try:
                self.process.wait(timeout=max(0.01, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                confirmed = self._kill_unconfirmed()
        else:
            confirmed = self._kill_unconfirmed()
        if confirmed and sys.platform == "linux":
            confirmed = self._cleanup == {"reaped": True}
        for thread in self._threads:
            thread.join(timeout=max(0.0, deadline - time.monotonic()))
        if any(thread.is_alive() for thread in self._threads):
            confirmed = False
        self._release_stderr()
        self._confirmed = confirmed
        return confirmed

    def _kill_unconfirmed(self) -> bool:
        if sys.platform == "linux":
            from ltspice_mcp.lib.proc_kill import kill_process_group

            worker = self.worker_pid or (_worker_pid(self.control) if self.control else None)
            if worker:
                kill_process_group(worker)
        with contextlib.suppress(OSError):
            self.process.kill()
        with contextlib.suppress(subprocess.TimeoutExpired):
            self.process.wait(timeout=_KILLED_REAP_WAIT_S)
        return False


def _call_result(
    directory: Path, limits: ParserProcessLimits, worker_pid: int | None, *, replied: bool
) -> ParserReply:
    """What a call left in its directory, read once nothing can still write there."""
    pid = worker_pid if worker_pid is not None else 0
    error_path = parser_file_in(directory, "error.json")
    try:
        if error_path.exists():
            error = _read_json(error_path, limits.error_bytes)
            code = error.get("code", "worker_error")
            if not isinstance(code, str) or code not in {
                "worker_error",
                "worker_crashed",
                "deadline",
                "cancelled",
                "memory_limit",
            }:
                raise ValueError("Invalid parser error code")
            raise ParserProcessError(
                code,
                str(error.get("message", "Parser failed")),
                reaped=True,
                worker_pid=worker_pid,
            )
        if not replied:
            raise ParserProcessError(
                "worker_crashed",
                "Parser exited without replying",
                reaped=True,
                worker_pid=worker_pid,
            )
        metadata = _read_json(parser_file_in(directory, "result.json"), limits.metadata_bytes)
    except (OSError, ValueError, TypeError, RecursionError) as exc:
        raise ParserProcessError(
            "invalid_result",
            "Parser result is missing, oversized or invalid",
            reaped=True,
            worker_pid=worker_pid,
        ) from exc
    return ParserReply(metadata, pid)


def run_parser_sync(
    request: dict[str, JsonValue],
    *,
    work_dir: Path,
    deadline: float,
    limits: ParserProcessLimits,
    cancel: threading.Event | None = None,
    warm: WarmParser | None = None,
    _worker_module: str = _WORKER_MODULE,
) -> ParserReply:
    """Run one call through ownership setup, reply, tree exit and reaping.

    ``deadline`` is absolute monotonic time. With ``warm``, the call goes to
    that session's kept tree instead of a tree of its own. The module override
    is internal test instrumentation, never a field in a parser request, and a
    call carrying one always gets a tree of its own.
    """
    try:
        require_containment_platform()
    except RuntimeError as exc:
        raise ParserProcessError("unsupported_platform", str(exc), reaped=True) from exc
    if warm is not None and _worker_module == _WORKER_MODULE:
        return warm.call(request, work_dir=work_dir, deadline=deadline, cancel=cancel)
    try:
        cancel = _admit_call(deadline, cancel)
        directory = work_dir.resolve(strict=True)
        if not directory.is_dir():
            raise ValueError("Parser work directory must already exist")
        for name in _CONTROL_FILES:
            if parser_file_in(directory, name).exists():
                raise ParserProcessError(
                    "supervisor_error",
                    "Parser control files require an exclusively created directory",
                    reaped=False,
                )
        packet = _packet(request, directory, limits)
        parser_file_in(directory, "request.json").write_bytes(packet)
    except ParserProcessError:
        raise
    except Exception as exc:
        raise ParserProcessError(
            "supervisor_error", str(exc)[: limits.error_bytes], reaped=True
        ) from exc
    tree = ParserTree(
        limits=limits,
        cwd=directory,
        control=directory,
        lifetime_deadline=deadline,
        module=_worker_module,
    )
    failure: _CallFailed | None = None
    status = None
    try:
        status = tree.call(packet, directory, deadline=deadline, cancel=cancel)
    except _CallFailed as exc:
        failure = exc
    except Exception as exc:
        failure = _CallFailed("supervisor_error", str(exc)[: limits.error_bytes])
    finally:
        reaped = tree.close()
    worker_pid = tree.worker_pid or _worker_pid(directory)
    if not reaped:
        raise ParserProcessError(
            "cleanup_failed",
            "Parser tree cleanup was not confirmed",
            reaped=False,
            worker_pid=worker_pid,
        )
    if failure is None and tree.stderr_overflowed:
        failure = _CallFailed("error_limit", "Parser diagnostics exceed their byte limit")
    if failure is not None and failure.code != "gone":
        raise ParserProcessError(failure.code, str(failure), reaped=True, worker_pid=worker_pid)
    return _call_result(directory, limits, worker_pid, replied=status is not None)


class WarmParser:
    """A session's parser tree, kept between calls instead of started for each.

    Calls run one at a time. After each reply the tree must hold only its
    worker; any other process, a failed call or excess diagnostics end the
    tree, and its call's results are then read only after that is confirmed,
    exactly as for a tree of its own. A tree that has served ``WARM_CALLS``
    calls, reached half of ``WARM_LIFETIME_S`` or waited ``WARM_IDLE_S`` is
    closed. A close that cannot be confirmed is reported by the next call,
    which is how the session's admission learns to stay closed.
    """

    def __init__(
        self, limits: ParserProcessLimits, cwd: Path, *, _worker_module: str = _WORKER_MODULE
    ) -> None:
        self.limits = limits
        self.cwd = cwd
        self._module = _worker_module
        self._lock = threading.Condition()
        self._tree: ParserTree | None = None
        self._last_used = 0.0
        self._unconfirmed: int | bool | None = False
        self._closed = False
        self._watcher: threading.Thread | None = None
        _WARM_PARSERS.add(self)

    def call(
        self,
        request: dict[str, JsonValue],
        *,
        work_dir: Path,
        deadline: float,
        cancel: threading.Event | None,
    ) -> ParserReply:
        try:
            cancel = _admit_call(deadline, cancel)
            directory = work_dir.resolve(strict=True)
            if not directory.is_dir():
                raise ValueError("Parser work directory must already exist")
            packet = _packet(request, directory, self.limits)
        except ParserProcessError:
            raise
        except Exception as exc:
            raise ParserProcessError(
                "supervisor_error", str(exc)[: self.limits.error_bytes], reaped=True
            ) from exc
        with self._lock:
            self._raise_unconfirmed()
            self._closed = False  # A call after close() keeps a tree again.
            tree = self._tree
            if tree is not None and (
                tree.calls >= WARM_CALLS or time.monotonic() - tree.started >= WARM_LIFETIME_S / 2
            ):
                self._retire()
                self._raise_unconfirmed()
                tree = None
            if tree is None:
                tree = ParserTree(
                    limits=self.limits,
                    cwd=self.cwd,
                    control=None,
                    lifetime_deadline=time.monotonic() + WARM_LIFETIME_S,
                    module=self._module,
                )
                self._tree = tree
                self._watch()
            failure: _CallFailed | None = None
            status = None
            try:
                status = tree.call(packet, directory, deadline=deadline, cancel=cancel)
            except _CallFailed as exc:
                failure = exc
            except Exception as exc:
                failure = _CallFailed("supervisor_error", str(exc)[: self.limits.error_bytes])
            self._last_used = time.monotonic()
            healthy = failure is None and status == "done" and not tree.stderr_overflowed
            if healthy and tree.contained():
                self._lock.notify_all()
                return _call_result(directory, self.limits, tree.worker_pid, replied=True)
            self._tree = None
            reaped = tree.close()
            if not reaped:
                raise ParserProcessError(
                    "cleanup_failed",
                    "Parser tree cleanup was not confirmed",
                    reaped=False,
                    worker_pid=tree.worker_pid,
                )
            if failure is None and tree.stderr_overflowed:
                failure = _CallFailed("error_limit", "Parser diagnostics exceed their byte limit")
            if failure is not None and failure.code != "gone":
                raise ParserProcessError(
                    failure.code, str(failure), reaped=True, worker_pid=tree.worker_pid
                )
            return _call_result(
                directory, self.limits, tree.worker_pid, replied=status is not None
            )

    def close(self) -> bool:
        """Close the kept tree, if any; True once it is confirmed gone."""
        with self._lock:
            self._closed = True
            self._retire()
            self._lock.notify_all()
            return self._unconfirmed is False

    def _retire(self) -> None:
        tree, self._tree = self._tree, None
        if tree is not None and not tree.close():
            self._unconfirmed = tree.worker_pid

    def _raise_unconfirmed(self) -> None:
        if self._unconfirmed is not False:
            raise ParserProcessError(
                "cleanup_failed",
                "A kept parser tree's cleanup was not confirmed",
                reaped=False,
                worker_pid=self._unconfirmed if isinstance(self._unconfirmed, int) else None,
            )

    def _watch(self) -> None:
        if self._watcher is None:
            self._watcher = threading.Thread(target=self._close_when_idle, daemon=True)
            self._watcher.start()

    def _close_when_idle(self) -> None:
        with self._lock:
            while not self._closed:
                if self._tree is None:
                    self._lock.wait()
                    continue
                idle = self._last_used + WARM_IDLE_S - time.monotonic()
                if idle > 0:
                    self._lock.wait(idle)
                    continue
                self._retire()
            # Still under the lock, so the next tree's _watch() starts a watcher.
            self._watcher = None


_LIVE_TREES: weakref.WeakSet[ParserTree] = weakref.WeakSet()
_WARM_PARSERS: weakref.WeakSet[WarmParser] = weakref.WeakSet()


def close_all() -> bool:
    """Close every parser tree this process still holds; True if all confirmed."""
    confirmed = True
    for warm in list(_WARM_PARSERS):
        confirmed = warm.close() and confirmed
    for tree in list(_LIVE_TREES):
        confirmed = tree.close() and confirmed
    return confirmed


atexit.register(close_all)


async def run_parser(
    request: dict[str, JsonValue],
    *,
    work_dir: Path,
    deadline: float,
    limits: ParserProcessLimits,
    _worker_module: str = _WORKER_MODULE,
) -> ParserReply:
    """Cancellation signals the supervisor and waits for confirmed tree cleanup."""
    cancel = threading.Event()
    context = contextvars.copy_context()

    def run() -> ParserReply:
        return context.run(
            run_parser_sync,
            request,
            work_dir=work_dir,
            deadline=deadline,
            limits=limits,
            cancel=cancel,
            _worker_module=_worker_module,
        )

    future = asyncio.get_running_loop().run_in_executor(None, run)
    cancelled = False
    while not future.done():
        try:
            await asyncio.shield(future)
        except asyncio.CancelledError:
            cancelled = True
            cancel.set()
        except ParserProcessError:
            break
    try:
        result = future.result()
    except ParserProcessError as exc:
        if cancelled and exc.reaped:
            raise asyncio.CancelledError from exc
        raise
    if cancelled:
        raise asyncio.CancelledError
    return result
