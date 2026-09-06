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
)
from ltspice_mcp.state import SessionState
from ltspice_mcp.tools._base import ToolInput, format_response, registry

TOOL_NAME = "run_code"
CONFIG_KEY = "tools.run_code"

#: How long a worker gets to boot its engine before the call fails.
BOOT_TIMEOUT_S = 60.0
#: After an interrupt, how long the worker gets to answer before it is killed.
INTERRUPT_GRACE_S = 5.0
#: After a close, how long the worker gets to exit before it is killed.
CLOSE_GRACE_S = 5.0


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
        "stdout",
        "stderr",
        "result",
        "truncated",
        "chars_dropped",
        "elapsed_s",
        "error",
        "running",
        "worker_restarted",
        "hint",
    ],
}


#: What a read returns once the worker's reply pipe has closed.
_EOF: dict[str, Any] = {"op": "eof"}


def _reap_session(pid: int) -> None:
    """Kill what is left of the worker's session (POSIX: it was started as a
    session leader, so its pid is the group id). A child a snippet spawned
    would otherwise outlive the worker; a simulator LTspice launched over WSL
    interop is a Windows process and is not reached, and its job record,
    owned by a dead pid, reads as interrupted."""
    if os.name == "nt":
        return
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(pid, signal.SIGKILL)


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
        self.restarted: dict[str, Any] | None = None
        self._drain: asyncio.Task[None] | None = None

    # -- lifecycle -----------------------------------------------------------

    @property
    def pid(self) -> int | None:
        return self.process.pid if self.process is not None else None

    def _alive(self) -> bool:
        return self.process is not None and self.process.returncode is None

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
        )
        ready = await self._read_message(BOOT_TIMEOUT_S)
        if ready is None or ready.get("op") != "ready":
            error = (ready or {}).get("error", "the worker did not report ready in time")
            await self._kill()
            raise RuntimeError(f"run_code worker failed to start: {error}")

    async def _ensure(self) -> None:
        if not self._alive():
            previous = self.pid
            self.process = None
            await self._spawn()
            if previous is not None and self.restarted is None:
                self.restarted = {"previous_pid": previous, "reason": "the worker exited"}

    async def _kill(self) -> None:
        """Kill the worker and, on POSIX, everything in its session: a child
        the snippet spawned would otherwise outlive it (a simulator LTspice
        launched over WSL interop is a Windows process and is not reached;
        its job record, owned by a dead pid, reads as interrupted)."""
        process = self.process
        if process is None:
            return
        if process.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                process.kill()
            await process.wait()
        _reap_session(process.pid)
        self.process = None

    def _interrupt(self) -> None:
        """Ask the worker to stop the running snippet (POSIX: SIGINT to it alone)."""
        if not self._alive() or self.process is None:
            return
        if os.name == "nt":
            # ponytail: no wake-capable interrupt exists for a blocked lock
            # wait on Windows, so a Windows worker is killed on timeout and
            # respawned; the graceful path is POSIX-only.
            self.process.kill()
            return
        with contextlib.suppress(ProcessLookupError):
            os.kill(self.process.pid, signal.SIGINT)

    async def close(self, grace_s: float = CLOSE_GRACE_S) -> None:
        """Stop the worker: interrupt what runs, ask it to close, then kill."""
        if self._drain is not None:
            self._drain.cancel()
            self._drain = None
        if not self._alive() or self.process is None:
            self.process = None
            return
        if self.running is not None:
            self._interrupt()
        await self._send({"op": "close"})
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(self.process.wait(), grace_s)
        # Exited or not, the worker and whatever its snippets spawned go
        # together: a graceful exit leaves a child it started still running.
        await self._kill()
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
                previous = self.pid
                await self._kill()
                self.restarted = {
                    "previous_pid": previous,
                    "reason": "the worker exited"
                    if reply is _EOF
                    else "killed after a cancellation",
                }
        finally:
            self.running = None
            self._drain = None

    # -- the call ------------------------------------------------------------

    def _reply(self, status: str, **fields: Any) -> dict[str, Any]:
        base: dict[str, Any] = {
            "status": status,
            "exec_seq": self.exec_seq,
            "worker_pid": self.pid,
            "stdout": "",
            "stderr": "",
            "result": None,
            "truncated": False,
            "chars_dropped": 0,
            "elapsed_s": 0.0,
            "error": None,
            "running": None,
            "worker_restarted": None,
            "hint": None,
        }
        base.update(fields)
        return base

    async def run(self, code: str, timeout_s: float, reset: bool) -> dict[str, Any]:
        if reset:
            previous = self.pid
            await self.close()
            self.restarted = {"previous_pid": previous, "reason": "reset requested"}
            if not code.strip():
                restarted, self.restarted = self.restarted, None
                return self._reply(
                    "reset",
                    worker_restarted=restarted,
                    hint="The next call starts a fresh worker; jobs the old one owned read as interrupted in jobs(list).",
                )
        if self.running is not None:
            current = self.running
            return self._reply(
                "busy",
                running={
                    "exec_seq": current.seq,
                    "elapsed_s": round(time.monotonic() - current.started, 3),
                    "same_code": current.code == code,
                    "phase": current.phase,
                },
                hint=(
                    "One snippet runs at a time. Wait for it and send again, or send "
                    "reset: true to kill it."
                    + (" This code is the one already running." if current.code == code else "")
                ),
            )
        await self._ensure()
        self.exec_seq += 1
        seq = self.exec_seq
        self.running = _Running(seq, code, time.monotonic())
        try:
            if not await self._send({"op": "run", "seq": seq, "code": code}):
                # Died between calls: one respawn, one retry.
                await self._kill()
                await self._ensure()
                if not await self._send({"op": "run", "seq": seq, "code": code}):
                    raise RuntimeError("run_code worker is not accepting requests")
            reply = await self._await_reply(seq, timeout_s)
            if reply is None:
                self.running.phase = "interrupting"
                self._interrupt()
                reply = await self._await_reply(seq, INTERRUPT_GRACE_S)
                if reply is not None and reply is not _EOF:
                    reply["status"] = "timeout"
            if reply is None or reply is _EOF:
                # No answer: the worker exited, or it ignored the interrupt.
                died = reply is _EOF
                previous = self.pid
                await self._kill()
                self.restarted = {
                    "previous_pid": previous,
                    "reason": "the worker exited" if died else "killed after a timeout",
                }
                if died:
                    return self._reply(
                        "error",
                        error={
                            "type": "WorkerDied",
                            "message": "the worker exited while running this snippet",
                            "traceback_tail": "",
                        },
                        hint="The next call starts a fresh worker.",
                    )
                return self._reply(
                    "timeout",
                    elapsed_s=round(timeout_s + INTERRUPT_GRACE_S, 3),
                    hint=_TIMEOUT_HINT,
                )
        except asyncio.CancelledError:
            if self._alive():
                self.running.phase = "interrupting"
                self._interrupt()
                self._drain = asyncio.ensure_future(self._finish_interrupt(seq))
            else:
                self.running = None
            raise
        finally:
            if self._drain is None:
                self.running = None
        if reply["status"] == "interrupted":
            reply["status"] = "timeout"
        restarted, self.restarted = self.restarted, None
        return self._reply(
            reply["status"],
            stdout=reply["stdout"],
            stderr=reply["stderr"],
            result=reply["result"],
            truncated=reply["truncated"],
            chars_dropped=reply["chars_dropped"],
            elapsed_s=reply["elapsed_s"],
            error=reply["error"],
            worker_restarted=restarted,
            hint=_hint(reply, restarted),
        )


_TIMEOUT_HINT = (
    "Interrupted at timeout_s (max 600). A simulation longer than that: "
    "api.run_experiments(wait=False) then jobs(wait) or api.wait(job_id)."
)


def _hint(reply: dict[str, Any], restarted: dict[str, Any] | None) -> str | None:
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
    if restarted is not None:
        parts.append("The worker was restarted; jobs it owned read as interrupted in jobs(list).")
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
    name=TOOL_NAME,
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
)
async def handle_run_code(args: RunCodeInput, state: SessionState) -> types.CallToolResult:
    reply = await worker_for(state).run(args.code, args.timeout_s, args.reset)
    return format_response(_render(reply), reply)
