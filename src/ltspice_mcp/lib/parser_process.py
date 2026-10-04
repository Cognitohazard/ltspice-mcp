"""One-call parser process supervision; decoding stays in a gated child.

The caller owns the admitted directory and removes it only after confirmed
cleanup. Numeric files and their manifests are validated by the parser service.
"""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import json
import math
import os
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TypeAlias

from ltspice_mcp.lib.parser_bootstrap import require_containment_platform
from ltspice_mcp.lib.store import parser_file_in
from ltspice_mcp.lib.windows_job import WindowsJob, python_launch

JsonValue: TypeAlias = bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"] | None
_WORKER_MODULE = "ltspice_mcp.lib.parser_worker"


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


def _stderr_reader(
    process: subprocess.Popen[bytes], path: Path, limit: int, overflow: threading.Event
):
    assert process.stderr is not None
    try:
        with path.open("xb") as stream:
            remaining = limit
            while chunk := process.stderr.read(4096):
                accepted = min(len(chunk), remaining)
                stream.write(chunk[:accepted])
                remaining -= accepted
                if len(chunk) > accepted:
                    overflow.set()
                    break
    except OSError:
        overflow.set()
    finally:
        process.stderr.close()


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


def _cleanup(
    process: subprocess.Popen[bytes], job: WindowsJob | None, directory: Path, grace: float
) -> bool:
    deadline = time.monotonic() + grace
    if process.stdin is not None:
        with contextlib.suppress(OSError):
            process.stdin.close()
    job_empty = True
    if job is not None:
        try:
            job.close()
        except (OSError, TimeoutError):
            job_empty = False
    elif sys.platform == "win32":
        # Ownership failed before the gate: this trusted bootstrap has no children.
        with contextlib.suppress(OSError):
            process.kill()
    try:
        process.wait(timeout=max(0.01, deadline - time.monotonic()))
    except subprocess.TimeoutExpired:
        if sys.platform == "linux":
            from ltspice_mcp.lib.proc_kill import kill_process_group

            if worker_pid := _worker_pid(directory):
                kill_process_group(worker_pid)
        with contextlib.suppress(OSError):
            process.kill()
        with contextlib.suppress(subprocess.TimeoutExpired):
            process.wait(timeout=0.2)
        return False
    if sys.platform == "win32":
        return job_empty
    try:
        return _read_json(parser_file_in(directory, "cleanup.json"), 256).get("reaped") is True
    except (OSError, ValueError, TypeError, RecursionError):
        return False


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


def run_parser_sync(
    request: dict[str, JsonValue],
    *,
    work_dir: Path,
    deadline: float,
    limits: ParserProcessLimits,
    cancel: threading.Event | None = None,
    _worker_module: str = _WORKER_MODULE,
) -> ParserReply:
    """Run the fixed decoder protocol through ownership setup, exit and reaping.

    ``deadline`` is absolute monotonic time. The module override is internal
    test instrumentation, never a field in a parser request.
    """
    try:
        require_containment_platform()
    except RuntimeError as exc:
        raise ParserProcessError("unsupported_platform", str(exc), reaped=True) from exc
    try:
        if not math.isfinite(deadline):
            raise ValueError("Parser deadline must be finite")
        cancel = cancel if cancel is not None else threading.Event()
        if cancel.is_set() or time.monotonic() >= deadline:
            code = "cancelled" if cancel.is_set() else "deadline"
            raise ParserProcessError(code, "Parser call ended before spawn", reaped=True)
        directory = work_dir.resolve(strict=True)
        if not directory.is_dir():
            raise ValueError("Parser work directory must already exist")
        for name in (
            "request.json",
            "result.json",
            "error.json",
            "process.json",
            "cleanup.json",
            "stderr.txt",
        ):
            if parser_file_in(directory, name).exists():
                raise ParserProcessError(
                    "supervisor_error",
                    "Parser control files require an exclusively created directory",
                    reaped=False,
                )
        packet = (
            json.dumps(
                {"version": 1, "op": "go", "request": request, "process_limits": asdict(limits)},
                allow_nan=False,
                separators=(",", ":"),
            ).encode("utf-8")
            + b"\n"
        )
        if len(packet) > limits.request_bytes:
            raise ParserProcessError(
                "request_limit", "Parser request exceeds its byte limit", reaped=True
            )
        parser_file_in(directory, "request.json").write_bytes(packet)
        executable, env = python_launch()
        env = {
            **(env if env is not None else os.environ),
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "LTSPICE_MCP_DISABLE_SIMULATOR_DETECTION": "1",
        }
    except ParserProcessError:
        raise
    except Exception as exc:
        raise ParserProcessError(
            "supervisor_error", str(exc)[: limits.error_bytes], reaped=True
        ) from exc
    try:
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
                str(deadline),
                str(limits.cleanup_grace_s),
                _worker_module,
                "0",
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            cwd=directory,
            env=env,
            start_new_session=sys.platform == "linux",
            bufsize=0,
        )
    except Exception as exc:
        raise ParserProcessError(
            "supervisor_error", str(exc)[: limits.error_bytes], reaped=_failed_spawn_reaped(exc)
        ) from exc
    job = None
    overflow = threading.Event()
    readers: list[threading.Thread] = []
    failure: tuple[str, str] | None = None
    reaped = False
    try:
        if sys.platform == "win32":
            job = WindowsJob(
                process.pid,
                allow_breakaway=False,
                memory_limit_bytes=limits.memory_bytes,
                cleanup_timeout_s=limits.cleanup_grace_s,
            )
        reader = threading.Thread(
            target=_stderr_reader,
            args=(
                process,
                parser_file_in(directory, "stderr.txt"),
                limits.error_bytes,
                overflow,
            ),
            daemon=True,
        )
        writer = threading.Thread(target=_send_request, args=(process, packet), daemon=True)
        readers.extend((reader, writer))
        reader.start()
        writer.start()  # This request is the gate; ownership now precedes parsing.
        while process.poll() is None:
            if cancel.is_set():
                failure = ("cancelled", "Parser call was cancelled")
                break
            if time.monotonic() >= deadline:
                failure = ("deadline", "Parser call exceeded its deadline")
                break
            if overflow.is_set():
                failure = ("error_limit", "Parser diagnostics exceed their byte limit")
                break
            for name, limit in (
                ("result.json", limits.metadata_bytes),
                ("error.json", limits.error_bytes),
            ):
                path = parser_file_in(directory, name)
                if path.exists() and path.stat().st_size > limit:
                    failure = ("metadata_limit", "Parser JSON exceeds its byte limit")
                    break
            if failure is not None:
                break
            cancel.wait(min(0.01, max(0, deadline - time.monotonic())))
    except Exception as exc:
        failure = (
            "ownership_failed" if not readers else "supervisor_error",
            str(exc)[: limits.error_bytes],
        )
    finally:
        cleanup_deadline = time.monotonic() + limits.cleanup_grace_s
        reaped = _cleanup(process, job, directory, limits.cleanup_grace_s)
        for thread in readers:
            thread.join(timeout=max(0, cleanup_deadline - time.monotonic()))
        if any(thread.is_alive() for thread in readers):
            reaped = False
    worker_pid = _worker_pid(directory)
    if not reaped:
        raise ParserProcessError(
            "cleanup_failed",
            "Parser tree cleanup was not confirmed",
            reaped=False,
            worker_pid=worker_pid,
        )
    if failure is None and overflow.is_set():
        failure = ("error_limit", "Parser diagnostics exceed their byte limit")
    if failure is not None:
        raise ParserProcessError(*failure, reaped=True, worker_pid=worker_pid)
    if worker_pid is None:
        worker_pid = process.pid
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
        if process.returncode != 0:
            raise ParserProcessError(
                "worker_crashed",
                f"Parser exited with code {process.returncode}",
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
    return ParserReply(metadata, worker_pid)


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
