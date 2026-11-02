"""The API profile: a manual tool-use loop between a model and the server.

Manual rather than the SDK's tool runner, because the harness decides exactly
what each tool result looks like (``rendering.py``). The loop only appends to
the conversation, never edits it, so thinking blocks stay valid across turns.

A trial ends when the agent stops calling tools, or when it reaches a
ceiling. Ceilings stop runaways only: nothing the model reads mentions them,
so a trial that stays below them runs as it would with none.
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from collections import Counter
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TextIO

import mcp.types as mcp_types
from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.shared.exceptions import MCPError

from ltspice_eval import LAUNCHER
from ltspice_eval.models import Model, Price, Usage
from ltspice_eval.rendering import Rendering, estimated_tokens, render
from ltspice_eval.variants import Variant
from ltspice_eval.workspace import FILE_TOOL_NAMES, FILE_TOOLS, FileToolError, FileTools, TrialDirs

SERVER_NAME = "spice"
TOOL_PREFIX = f"mcp__{SERVER_NAME}__"

#: Claude Code 2.1.316 shows the model a tool description, and a server's
#: instructions, up to this many characters, and marks the cut.
DESCRIPTION_LIMIT = 4096
TRUNCATION_MARK = "… [truncated]"

#: Claude Code's notice after a response cut off by the output limit.
OUTPUT_LIMIT_NOTICE = (
    "Output token limit hit. Resume directly — no apology, no recap of what you were doing. "
    "Pick up mid-thought if that is where the cut happened. Break remaining work into smaller pieces."
)
NOT_RUN_NOTICE = (
    "Not run: the response was cut off at the output limit before this call was complete."
)

PREAMBLE = """\
You are an agent working in a folder on the user's computer. A SPICE \
circuit-simulation server is connected through the tools whose names start \
with mcp__spice__. You can read, write and edit files in the folder with \
read_file, write_file, edit_file and list_dir. There is no shell, so \
simulations run only through the server.

Do the task in the user's message. When it asks for an answer file, write \
that file as it describes: the file is what is checked, not your reply."""


@dataclass(frozen=True)
class Ceilings:
    """The suite's limits on one trial, far above normal use (see the design,
    "Ceilings, not budgets")."""

    turns: int = 200
    usd: float = 20.0
    wall_clock_s: float = 3600.0

    def as_row(self) -> dict[str, float]:
        return {"turns": self.turns, "usd": self.usd, "wall_clock_s": self.wall_clock_s}


@dataclass(frozen=True)
class Settings:
    model: str
    price: Price
    effort: str | None = None
    rendering: Rendering = "structured"
    max_tokens: int = 64_000
    ceilings: Ceilings = field(default_factory=Ceilings)


class TrialError(Exception):
    """A trial that ended for a reason outside the agent's control. It goes to
    ``errors.jsonl``, not to the results."""

    kind = "harness_error"


class ServerDied(TrialError):
    kind = "server_died"


class ServedModelMismatch(TrialError):
    kind = "served_model_mismatch"


class ApiFailure(TrialError):
    kind = "api_error"


@dataclass
class CallRecord:
    name: str
    #: ``inspect``'s ``kind``, so reference and guide lookups can be counted.
    kind: str | None
    seconds: float
    is_error: bool
    codes: list[str]
    tokens: int
    oversized: bool


@dataclass
class LoopOutcome:
    messages: list[dict[str, Any]]
    #: ``ok`` (the agent stopped), ``capped`` or ``refused``.
    status: str
    capped: str | None
    turns: int
    stop_reason: str | None
    usage: Usage
    usd: float
    model_s: float
    retries: int
    calls: list[CallRecord]
    served_models: list[str]
    max_tokens_stops: int
    final_text: str


@dataclass
class Server:
    session: ClientSession
    instructions: str
    tools: list[mcp_types.Tool]
    server_info: dict[str, Any]


@asynccontextmanager
async def connect(variant: Variant, dirs: TrialDirs, errlog: TextIO) -> AsyncIterator[Server]:
    """The variant's server, started through the launcher in the trial's
    workspace, initialized and listed. Leaving the context stops the
    server's process group.

    The launcher is handed this process's whole environment, as Claude Code
    hands its own to a stdio server, so the launcher alone decides what the
    server keeps and both profiles start it alike. (Left to itself, the MCP
    SDK would pass only its short default list.)
    """
    params = StdioServerParameters(
        command=sys.executable,
        args=[
            "-I",
            str(LAUNCHER),
            "--python",
            str(variant.python),
            "--cwd",
            str(dirs.workspace),
            "--home",
            str(dirs.home),
        ],
        cwd=str(dirs.workspace),
        env=dict(os.environ),
    )
    async with (
        stdio_client(params, errlog=errlog) as (read, write),
        ClientSession(read, write) as session,
    ):
        init = await session.initialize()
        tools: list[mcp_types.Tool] = []
        cursor: str | None = None
        while True:
            listed = await session.list_tools(
                params=mcp_types.PaginatedRequestParams(cursor=cursor) if cursor else None
            )
            tools.extend(listed.tools)
            cursor = listed.next_cursor
            if not cursor:
                break
        yield Server(
            session=session,
            instructions=init.instructions or "",
            tools=tools,
            server_info=init.server_info.model_dump(mode="json", exclude_none=True),
        )


def system_prompt(instructions: str) -> str:
    """The preamble, then the server's instructions where Claude Code puts
    them, under its MCP server instructions heading."""
    text = PREAMBLE
    if instructions:
        text += (
            "\n\n# MCP Server Instructions\n\n"
            "The following MCP servers have provided instructions for how to use their tools "
            f"and resources:\n\n## {SERVER_NAME}\n{_capped(instructions)}"
        )
    return text


def model_tools(server_tools: list[mcp_types.Tool]) -> list[dict[str, Any]]:
    """The harness file tools, then the server's tools in the order it lists
    them, named as Claude Code names them."""
    tools = [dict(tool) for tool in FILE_TOOLS]
    for tool in server_tools:
        entry: dict[str, Any] = {
            "name": TOOL_PREFIX + tool.name,
            "input_schema": tool.input_schema,
        }
        if tool.description:
            entry["description"] = _capped(tool.description)
        tools.append(entry)
    return tools


def first_message(workspace: Path, prompt: str) -> dict[str, Any]:
    """The workspace path goes here rather than in the system prompt, so the
    tools and system prompt are the same, and cache, across trials."""
    return {
        "role": "user",
        "content": [{"type": "text", "text": f"The folder is {workspace}.\n\n{prompt}"}],
    }


async def run_loop(
    model: Model,
    server: Server,
    files: FileTools,
    prompt: str,
    settings: Settings,
    state: Progress | None = None,
) -> LoopOutcome:
    """Run the trial. ``state`` receives the conversation, usage and calls as
    they happen, so a caller still has them when the trial ends in an error."""
    state = state if state is not None else Progress()
    system = system_prompt(server.instructions)
    tools = model_tools(server.tools)
    server_tool_names = {tool.name for tool in server.tools}
    state.system = system
    state.tools = tools
    messages = state.messages
    messages.append(first_message(files.workspace, prompt))
    usage = state.usage

    async def turns() -> None:
        while True:
            if state.turns >= settings.ceilings.turns:
                state.capped = "turns"
                return
            if usage.usd(settings.price) >= settings.ceilings.usd:
                state.capped = "usd"
                return
            reply = await model.create(_request(settings, system, tools, messages))
            state.turns += 1
            state.model_s += reply.seconds
            state.retries += reply.retries
            state.served.append(reply.model)
            usage.add(reply.usage)
            if reply.model != settings.model:
                raise ServedModelMismatch(f"asked for {settings.model}, served {reply.model}")
            messages.append({"role": "assistant", "content": reply.content})
            state.stop_reason = reply.stop_reason
            if reply.stop_reason == "refusal":
                state.refused = True
                return
            uses = [block for block in reply.content if block.get("type") == "tool_use"]
            if reply.stop_reason == "max_tokens":
                state.max_tokens_stops += 1
                results = [_tool_result(block["id"], NOT_RUN_NOTICE, error=True) for block in uses]
                results.append({"type": "text", "text": OUTPUT_LIMIT_NOTICE})
                messages.append({"role": "user", "content": results})
                continue
            if not uses:
                state.final_text = "\n".join(
                    block.get("text", "") for block in reply.content if block.get("type") == "text"
                )
                return
            results = []
            for block in uses:
                result, record = await _execute(
                    block, server, server_tool_names, files, settings.rendering
                )
                results.append(result)
                state.calls.append(record)
            messages.append({"role": "user", "content": results})

    try:
        async with asyncio.timeout(settings.ceilings.wall_clock_s):
            await turns()
    except TimeoutError:
        state.capped = "wall_clock"

    status = "capped" if state.capped else "refused" if state.refused else "ok"
    return LoopOutcome(
        messages=messages,
        status=status,
        capped=state.capped,
        turns=state.turns,
        stop_reason=state.stop_reason,
        usage=usage,
        usd=usage.usd(settings.price),
        model_s=state.model_s,
        retries=state.retries,
        calls=state.calls,
        served_models=sorted(set(state.served)),
        max_tokens_stops=state.max_tokens_stops,
        final_text=state.final_text,
    )


@dataclass
class Progress:
    """A trial as it stands: what the loop has sent, received and spent."""

    system: str = ""
    tools: list[dict[str, Any]] = field(default_factory=list)
    messages: list[dict[str, Any]] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    turns: int = 0
    capped: str | None = None
    refused: bool = False
    stop_reason: str | None = None
    model_s: float = 0.0
    retries: int = 0
    max_tokens_stops: int = 0
    final_text: str = ""
    served: list[str] = field(default_factory=list)
    calls: list[CallRecord] = field(default_factory=list)


def _request(
    settings: Settings, system: str, tools: list[dict[str, Any]], messages: list[dict[str, Any]]
) -> dict[str, Any]:
    """One turn's request. A cache breakpoint on the system prompt (which
    follows the tools in the prefix) gives every trial of a variant a shared
    read point; the top-level automatic breakpoint follows the conversation."""
    request: dict[str, Any] = {
        "model": settings.model,
        "max_tokens": settings.max_tokens,
        "system": [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
        "tools": tools,
        "messages": messages,
        "cache_control": {"type": "ephemeral"},
        "thinking": {"type": "adaptive", "display": "summarized"},
    }
    if settings.effort:
        request["output_config"] = {"effort": settings.effort}
    return request


async def _execute(
    block: dict[str, Any],
    server: Server,
    server_tool_names: set[str],
    files: FileTools,
    rendering: Rendering,
) -> tuple[dict[str, Any], CallRecord]:
    name = block["name"]
    arguments = block.get("input") or {}
    started = time.monotonic()
    tool = name.removeprefix(TOOL_PREFIX)
    kind = arguments.get("kind") if tool == "inspect" and isinstance(arguments, dict) else None
    codes: list[str] = []
    tokens = 0
    oversized = False
    if name in FILE_TOOL_NAMES:
        try:
            text = await asyncio.to_thread(files.call, name, arguments)
            error = False
        except FileToolError as exc:
            text, error = str(exc), True
        content: list[dict[str, Any]] | str = [{"type": "text", "text": text}]
        tokens = estimated_tokens(text)
    elif name.startswith(TOOL_PREFIX) and tool in server_tool_names:
        try:
            result = await server.session.call_tool(tool, arguments)
        except MCPError as exc:
            if exc.code == mcp_types.CONNECTION_CLOSED:
                raise ServerDied(f"the server closed the connection during {tool}") from exc
            content, error = (
                [{"type": "text", "text": f"MCP error {exc.code}: {exc.message}"}],
                True,
            )
        else:
            if not isinstance(result, mcp_types.CallToolResult):
                raise TrialError(f"{tool} returned {type(result).__name__}, not a tool result")
            rendered = render(result, rendering)
            content, error = rendered.content, rendered.is_error
            codes = list(_error_codes(rendered.structured, rendered.is_error))
            tokens, oversized = rendered.tokens, rendered.oversized
    else:
        content, error = [{"type": "text", "text": f"Error: No such tool available: {name}"}], True
    record = CallRecord(
        name=tool if name.startswith(TOOL_PREFIX) else name,
        kind=kind if isinstance(kind, str) else None,
        seconds=time.monotonic() - started,
        is_error=error,
        codes=codes,
        tokens=tokens,
        oversized=oversized,
    )
    return _tool_result(block["id"], content, error=error), record


def _tool_result(
    tool_use_id: str, content: list[dict[str, Any]] | str, *, error: bool
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "type": "tool_result",
        "tool_use_id": tool_use_id,
        "content": content,
    }
    if error:
        result["is_error"] = True
    return result


def _error_codes(structured: Any, is_error: bool) -> list[str]:
    """The error codes in a result: the call's own when it failed; each item's
    ``error.code``, and each ``failures`` row's ``code``, anywhere inside a
    reply that succeeded overall."""
    codes: list[str] = []
    if is_error and isinstance(structured, dict) and isinstance(structured.get("code"), str):
        codes.append(structured["code"])
    stack = [structured]
    while stack:
        node = stack.pop()
        if isinstance(node, dict):
            error = node.get("error")
            if isinstance(error, dict) and isinstance(error.get("code"), str):
                codes.append(error["code"])
            failures = node.get("failures")
            if isinstance(failures, list):
                codes += [
                    row["code"]
                    for row in failures
                    if isinstance(row, dict) and isinstance(row.get("code"), str)
                ]
            stack.extend(node.values())
        elif isinstance(node, list):
            stack.extend(node)
    return codes


def _capped(text: str) -> str:
    if len(text) <= DESCRIPTION_LIMIT:
        return text
    return text[:DESCRIPTION_LIMIT] + TRUNCATION_MARK


def tool_counts(calls: list[CallRecord]) -> dict[str, Any]:
    by_tool = Counter(call.name for call in calls)
    by_kind = Counter(f"{call.name}:{call.kind}" for call in calls if call.kind)
    return {"total": len(calls), "by_tool": dict(by_tool), "by_kind": dict(by_kind)}


def error_counts(calls: list[CallRecord]) -> dict[str, Any]:
    by_code = Counter(code for call in calls for code in call.codes)
    return {"total": sum(call.is_error for call in calls), "by_code": dict(by_code)}
