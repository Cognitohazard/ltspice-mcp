"""``run_code`` — a Python snippet run in a warm worker that holds the engine.

The server's side of :mod:`ltspice_mcp.code_worker`: one worker process per
session, spawned on the first call, supervised here. Every call is a fresh
namespace around the worker's one live ``Api``; ``exec_seq`` is owned here
and continues across worker respawns, so a replay after a reset never reads
as a fresh call. One snippet runs at a time — a second call while one runs is
answered ``busy``, never queued, because in a chat client a second send is
usually a correction.

Advertised only when ``[tools] run_code = true``: the snippet runs with the
server process's own file and process authority, not inside
``allowed_paths``, and turning that on is the operator's decision.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mcp import types
from pydantic import Field, model_validator

from ltspice_mcp.code_worker import (
    RESULT_CHARS,
    STDERR_TAIL_CHARS,
    STDOUT_HEAD_CHARS,
    STDOUT_TAIL_CHARS,
    empty_reply,
)
from ltspice_mcp.lib.proc_kill import kill_process_group
from ltspice_mcp.state import SessionState
from ltspice_mcp.tools._base import ToolInput, format_response, registry

#: How long a worker gets to boot its engine before the call fails.
BOOT_TIMEOUT_S = 60.0
#: After an interrupt, how long the worker gets to answer before it is killed.
INTERRUPT_GRACE_S = 5.0
#: After a close, how long the worker gets to exit before it is killed.
CLOSE_GRACE_S = 5.0
#: The longest reply line the pipe reader accepts. A reply is bounded by the
#: output caps (about 28k characters, UTF-8 encoded), so this is far above
#: any real line; the default 64 KiB was not.
_PIPE_LINE_LIMIT = 1 << 20


class RunCodeInput(ToolInput):
    code: str = Field(
        default="",
        description=(
            "Python source. In scope: api (the engine on this working directory: the "
            "same six ops as methods, complete results), np, load_raw, measurements, "
            "reference. The repr of a trailing expression comes back as result."
        ),
    )
    timeout_s: float = Field(
        default=60.0,
        ge=1.0,
        le=600.0,
        description=(
            "Wall-clock bound in seconds; on timeout the snippet is interrupted and a "
            "run it waits on is cancelled. A run longer than this belongs to "
            "api.run_experiments(wait=False) and jobs(wait)."
        ),
    )
    reset: bool = Field(
        default=False,
        description=(
            "Kill the worker first (a wedged snippet, a stale import) and start a "
            "fresh one; with empty code that is all the call does."
        ),
    )

    @model_validator(mode="after")
    def _code_or_reset(self) -> RunCodeInput:
        if not self.code.strip() and not self.reset:
            raise ValueError("code is empty; pass a snippet, or reset: true to restart the worker")
        return self


_ERROR_SCHEMA = {
    "type": ["object", "null"],
    "properties": {
        "type": {"type": "string"},
        "message": {"type": "string"},
        "traceback_tail": {"type": "string"},
    },
    "required": ["type", "message", "traceback_tail"],
}

RUN_CODE_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "status": {
            "type": "string",
            "enum": ["ok", "error", "timeout", "busy", "reset"],
            "description": (
                "ok: ran to the end. error: raised (see error). timeout: interrupted at "
                "timeout_s. busy: another snippet is running (see running). reset: the "
                "worker was restarted and no code ran."
            ),
        },
        "exec_seq": {
            "type": "integer",
            "description": "Server-owned call counter, continuous across worker restarts.",
        },
        "worker_pid": {
            "type": ["integer", "null"],
            "description": "The worker process; jobs it submits carry this owner pid.",
        },
        "stdout": {"type": "string", "description": "print() output; head and tail kept."},
        "stderr": {"type": "string", "description": "Python-level stderr; tail kept."},
        "result": {
            "type": ["string", "null"],
            "description": "repr of a trailing expression, or null.",
        },
        "truncated": {"type": "boolean"},
        "chars_dropped": {"type": "integer"},
        "elapsed_s": {"type": "number"},
        "error": _ERROR_SCHEMA,
        "running": {
            "type": ["object", "null"],
            "description": "On busy: the snippet that is running.",
            "properties": {
                "exec_seq": {"type": "integer"},
                "elapsed_s": {"type": "number"},
                "same_code": {
                    "type": "boolean",
                    "description": "The rejected code is byte-identical to the running one.",
                },
                "phase": {"type": "string", "enum": ["running", "interrupting"]},
            },
            "required": ["exec_seq", "elapsed_s", "same_code", "phase"],
        },
        "worker_restarted": {
            "type": ["object", "null"],
            "description": (
                "Set on the first call after a worker was replaced: jobs the previous "
                "worker owned read as interrupted in jobs(list)."
            ),
            "properties": {
                "previous_pid": {"type": ["integer", "null"]},
                "reason": {"type": "string"},
            },
            "required": ["previous_pid", "reason"],
        },
        "hint": {"type": ["string", "null"]},
    },
    "required": [
        "status",
        "exec_seq",
        "worker_pid",
        *empty_reply("ok"),
        "running",
        "worker_restarted",
        "hint",
    ],
}


#: What a read returns once the worker's reply pipe has closed.
_EOF: dict[str, Any] = {"op": "eof"}

_TIMEOUT_HINT = (
    "Interrupted at timeout_s (max 600). A simulation longer than that: "
    "api.run_experiments(wait=False) then jobs(wait) or api.wait(job_id)."
)
_RESTART_HINT = (
    "The worker was restarted; jobs the previous one owned read as interrupted in jobs(list)."
)


@dataclass
class _Running:
    seq: int
    code: str
    started: float
    phase: str = "running"


class CodeWorker:
    """One session's warm worker: spawn, run, interrupt, kill, respawn."""

    def __init__(self, working_dir: Path, config_path: Path | None) -> None:
        self.working_dir = working_dir
        self.config_path = config_path
        self.process: asyncio.subprocess.Process | None = None
        self.exec_seq = 0
        self.running: _Running | None = None
        #: Why the previous worker went, reported once on the next reply.
        self.restarted: dict[str, Any] | None = None
        self._drain: asyncio.Task[None] | None = None

    # -- lifecycle -----------------------------------------------------------

    @property
    def pid(self) -> int | None:
        return self.process.pid if self.process is not None else None

    def _live(self) -> asyncio.subprocess.Process | None:
        process = self.process
        return process if process is not None and process.returncode is None else None

    async def _spawn(self) -> None:
        self.process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "ltspice_mcp.code_worker",
            str(self.working_dir),
            str(self.config_path or ""),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=None,
            cwd=self.working_dir,
            start_new_session=os.name != "nt",
            limit=_PIPE_LINE_LIMIT,
        )
        ready = await self._read_message(BOOT_TIMEOUT_S)
        if ready is None or ready.get("op") != "ready":
            error = (ready or {}).get("error", "the worker did not report ready in time")
            await self._kill()
            raise RuntimeError(f"run_code worker failed to start: {error}")

    async def _ensure(self) -> None:
        if self._live() is None:
            await self._kill("the worker exited")
            await self._spawn()

    async def _kill(self, reason: str | None = None) -> None:
        """Kill the worker and reap its session; ``reason`` is what the next
        reply says about the worker that went."""
        process = self.process
        if process is None:
            return
        if reason is not None:
            self.restarted = {"previous_pid": process.pid, "reason": reason}
        if process.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                process.kill()
            await process.wait()
        # A child a snippet spawned would otherwise outlive the worker, which
        # was started as a session leader. A simulator LTspice launched over
        # WSL interop is a Windows process and is not reached; its job record,
        # owned by a dead pid, reads as interrupted.
        kill_process_group(process.pid)
        self.process = None

    def _interrupt(self) -> None:
        """Ask the worker to stop the running snippet (POSIX: SIGINT to it alone)."""
        process = self._live()
        if process is None:
            return
        if self.running is not None:
            self.running.phase = "interrupting"
        if os.name == "nt":
            # No wake-capable interrupt exists for a blocked lock wait on
            # Windows, so a Windows worker is killed on timeout and respawned;
            # the graceful path is POSIX-only.
            process.kill()
            return
        with contextlib.suppress(ProcessLookupError):
            os.kill(process.pid, signal.SIGINT)

    async def close(self, reason: str | None = None, grace_s: float = CLOSE_GRACE_S) -> None:
        """Stop the worker: interrupt what runs, ask it to close, then kill."""
        if self._drain is not None:
            self._drain.cancel()
            self._drain = None
        process = self._live()
        if process is not None:
            if self.running is not None:
                self._interrupt()
            await self._send({"op": "close"})
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(process.wait(), grace_s)
        # Exited or not, the worker and whatever its snippets spawned go
        # together: a graceful exit leaves a child it started still running.
        await self._kill(reason)
        self.running = None

    # -- protocol ------------------------------------------------------------

    async def _send(self, message: dict[str, Any]) -> bool:
        if self.process is None or self.process.stdin is None:
            return False
        try:
            self.process.stdin.write((json.dumps(message) + "\n").encode("utf-8"))
            await self.process.stdin.drain()
            return True
        except (BrokenPipeError, ConnectionResetError, OSError):
            return False

    async def _read_message(self, timeout_s: float) -> dict[str, Any] | None:
        """One reply line within ``timeout_s``; None on timeout, ``_EOF`` once
        the worker's pipe closed (it exited).

        ``asyncio.wait`` rather than ``wait_for``: the latter awaits the
        cancelled read, which is fine on a pipe but is not the habit this
        codebase keeps. A read this task abandons (an MCP cancellation) is
        cancelled too — a second ``readline`` on the same stream while one is
        pending is a RuntimeError, and a cancelled one keeps its buffer.
        """
        if self.process is None or self.process.stdout is None:
            return _EOF
        while True:
            reader = asyncio.ensure_future(self.process.stdout.readline())
            try:
                done, _ = await asyncio.wait({reader}, timeout=timeout_s)
            except asyncio.CancelledError:
                reader.cancel()
                raise
            if reader not in done:
                reader.cancel()
                return None
            line = reader.result()
            if not line:
                return _EOF
            try:
                message = json.loads(line)
            except ValueError:
                continue
            if isinstance(message, dict):
                return message

    async def _await_reply(self, seq: int, timeout_s: float) -> dict[str, Any] | None:
        """The reply carrying ``seq``; earlier replies (an interrupted
        predecessor's) are dropped on the way. None on timeout, ``_EOF`` if
        the worker exited first."""
        deadline = time.monotonic() + timeout_s
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            message = await self._read_message(remaining)
            if message is None or message is _EOF:
                return message
            if message.get("op") == "reply" and message.get("seq") == seq:
                return message

    async def _finish_interrupt(self, seq: int) -> None:
        """After an MCP cancellation: hold ``running`` until the interrupted
        reply arrives, so a call in that window is answered busy, not queued."""
        try:
            reply = await self._await_reply(seq, INTERRUPT_GRACE_S)
            if reply is None or reply is _EOF:
                await self._kill(
                    "the worker exited" if reply is _EOF else "killed after a cancellation"
                )
        finally:
            self.running = None
            self._drain = None

    async def _exchange(self, seq: int, code: str, timeout_s: float) -> dict[str, Any] | None:
        """Send one snippet and await its reply: one respawn if the worker died
        between calls, an interrupt at the timeout. None or ``_EOF`` when the
        worker never answered."""
        request = {"op": "run", "seq": seq, "code": code}
        if not await self._send(request):
            await self._ensure()
            if not await self._send(request):
                raise RuntimeError("run_code worker is not accepting requests")
        reply = await self._await_reply(seq, timeout_s)
        if reply is None:
            self._interrupt()
            reply = await self._await_reply(seq, INTERRUPT_GRACE_S)
            if reply is not None and reply is not _EOF:
                reply["status"] = "timeout"
        return reply

    # -- the call ------------------------------------------------------------

    def _reply(self, status: str, **fields: Any) -> dict[str, Any]:
        restarted, self.restarted = self.restarted, None
        base: dict[str, Any] = {
            **empty_reply(status),
            "exec_seq": self.exec_seq,
            "worker_pid": self.pid,
            "running": None,
            "worker_restarted": restarted,
            "hint": None,
        }
        base.update(fields)
        return base

    def _busy(self, code: str) -> dict[str, Any]:
        current = self.running
        assert current is not None
        same = current.code == code
        return self._reply(
            "busy",
            running={
                "exec_seq": current.seq,
                "elapsed_s": round(time.monotonic() - current.started, 3),
                "same_code": same,
                "phase": current.phase,
            },
            hint=(
                "One snippet runs at a time. Wait for it and send again, or send "
                "reset: true to kill it."
                + (" This code is the one already running." if same else "")
            ),
        )

    async def _lost(self, died: bool, timeout_s: float) -> dict[str, Any]:
        """No answer: the worker exited, or it ignored the interrupt. The
        reply is built first, so the restart it causes is reported where the
        contract puts it: on the next call."""
        if died:
            reply = self._reply(
                "error",
                error={
                    "type": "WorkerDied",
                    "message": "the worker exited while running this snippet",
                    "traceback_tail": "",
                },
                hint=_RESTART_HINT,
            )
        else:
            reply = self._reply(
                "timeout",
                elapsed_s=round(timeout_s + INTERRUPT_GRACE_S, 3),
                hint=f"{_TIMEOUT_HINT} {_RESTART_HINT}",
            )
        await self._kill("the worker exited" if died else "killed after a timeout")
        return reply

    async def run(self, code: str, timeout_s: float, reset: bool) -> dict[str, Any]:
        if reset:
            await self.close("reset requested")
            if not code.strip():
                return self._reply("reset", hint=_RESTART_HINT)
        if self.running is not None:
            return self._busy(code)
        try:
            await self._ensure()
        except (OSError, RuntimeError) as exc:
            return self._reply(
                "error",
                error={"type": "WorkerBootFailed", "message": str(exc), "traceback_tail": ""},
                hint="The worker could not start; the server's log has the details.",
            )
        self.exec_seq += 1
        seq = self.exec_seq
        self.running = _Running(seq, code, time.monotonic())
        try:
            reply = await self._exchange(seq, code, timeout_s)
        except asyncio.CancelledError:
            if self._live() is not None:
                self._interrupt()
                self._drain = asyncio.ensure_future(self._finish_interrupt(seq))
            else:
                self.running = None
            raise
        finally:
            if self._drain is None:
                self.running = None
        if reply is None or reply is _EOF:
            return await self._lost(reply is _EOF, timeout_s)
        if reply["status"] == "interrupted":
            reply["status"] = "timeout"
        fields = {key: reply[key] for key in empty_reply("ok") if key != "status"}
        result = self._reply(reply["status"], **fields)
        result["hint"] = _hint(result)
        return result


def _hint(reply: dict[str, Any]) -> str | None:
    parts: list[str] = []
    if reply["status"] == "timeout":
        parts.append(_TIMEOUT_HINT)
    elif reply["status"] == "error":
        parts.append("reference('<op>') lists an op's arguments; the traceback names the line.")
    if reply["truncated"]:
        parts.append(
            f"Output is capped at {STDOUT_HEAD_CHARS + STDOUT_TAIL_CHARS} characters; "
            "print less, or write to a file and read it."
        )
    if reply["worker_restarted"] is not None:
        parts.append(_RESTART_HINT)
    return " ".join(parts) or None


def _render(reply: dict[str, Any]) -> str:
    lines = [f"run_code #{reply['exec_seq']}: {reply['status']}"]
    if reply["stdout"]:
        lines.append(reply["stdout"].rstrip("\n"))
    if reply["result"] is not None:
        lines.append(f"=> {reply['result']}")
    if reply["error"] is not None:
        lines.append(f"{reply['error']['type']}: {reply['error']['message']}")
    if reply["hint"]:
        lines.append(reply["hint"])
    return "\n".join(lines)


def worker_for(state: SessionState) -> CodeWorker:
    if state.code_worker is None:
        state.code_worker = CodeWorker(state.working_dir, state.config.config_path)
    return state.code_worker


@registry.tool(
    name="run_code",
    title="Run Python with the engine",
    description=(
        "Run a Python snippet in a warm worker that holds this server's engine as "
        "`api`. The snippet has the server process's own file and process "
        "authority, not the sandbox, so permission this tool like a shell. For "
        "loops over many runs, numpy on samples, and compute-decide-compute; a "
        "single run or measurement is a tool call. In scope: api (the same six ops "
        "as methods on this working directory, complete results, no paging), np, "
        "load_raw, measurements, reference — reference('run_experiments') lists an "
        "op's arguments, so read it before guessing them. Every call is a fresh "
        "namespace around the same live engine; keep state on disk (a job by "
        f"request_id, a file). stdout keeps the first {STDOUT_HEAD_CHARS} and last "
        f"{STDOUT_TAIL_CHARS} characters of print() output, stderr its last "
        f"{STDERR_TAIL_CHARS}, result the repr of a trailing expression "
        f"({RESULT_CHARS}). Bound: timeout_s, default 60 (max 600); a timeout "
        "interrupts the snippet and cancels a run it waits on. One snippet at a "
        "time: a call during another is answered busy, not queued."
    ),
    input_model=RunCodeInput,
    annotations=types.ToolAnnotations(
        read_only_hint=False,
        destructive_hint=True,
        idempotent_hint=False,
        open_world_hint=True,
    ),
    output_schema=RUN_CODE_OUTPUT_SCHEMA,
    gate="run_code",
)
async def handle_run_code(args: RunCodeInput, state: SessionState) -> types.CallToolResult:
    reply = await worker_for(state).run(args.code, args.timeout_s, args.reset)
    return format_response(_render(reply), reply)
