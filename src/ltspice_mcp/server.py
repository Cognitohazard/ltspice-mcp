"""MCP server instance with lifespan management and tool dispatch."""

import asyncio
import logging
import os
from collections.abc import AsyncIterator, Collection
from contextlib import asynccontextmanager, suppress
from contextvars import ContextVar
from typing import Any

from mcp import types
from mcp.server.caching import CacheHint
from mcp.server.context import ServerRequestContext
from mcp.server.lowlevel import Server
from mcp.shared.exceptions import MCPError
from pydantic import ValidationError

from ltspice_mcp import __version__, prompts
from ltspice_mcp import errors as _err
from ltspice_mcp.api._session import acquire_session_lease, release_session_lease
from ltspice_mcp.config import ServerConfig, generate_default_config
from ltspice_mcp.engine import bootstrap_server_engine
from ltspice_mcp.errors import LTSpiceMCPError, PathSecurityError
from ltspice_mcp.lib import CIRCUIT_EXTENSIONS
from ltspice_mcp.lib.observability import configure_stderr_logging
from ltspice_mcp.lib.pathutil import resolve_safe_path
from ltspice_mcp.lib.simulator import SIMULATOR_DISPLAY, no_simulator_message
from ltspice_mcp.lib.simulator_build import executable_path
from ltspice_mcp.resources import (
    get_resource_templates,
    get_static_resources,
    handle_read_resource,
)
from ltspice_mcp.state import SessionState
from ltspice_mcp.tools._base import RegisteredTool, path_denied_text
from ltspice_mcp.tools.reference_index import validation_error_detail

# Tool argument keys that carry a circuit file path.
_CIRCUIT_PATH_KEYS: tuple[str, ...] = ("path", "netlist")

logger = logging.getLogger(__name__)


def _get_state(ctx: ServerRequestContext) -> SessionState:
    """Extract session state from the request's lifespan context."""
    try:
        return ctx.lifespan_context["state"]
    except (AttributeError, KeyError, TypeError) as e:
        raise RuntimeError(f"Session state not available: {e}") from e


def _extract_circuit_path(arguments: dict | None) -> str | None:
    """Pull a circuit path from raw tool arguments, if one is present."""
    if not isinstance(arguments, dict):
        return None
    for key in _CIRCUIT_PATH_KEYS:
        val = arguments.get(key)
        if isinstance(val, str) and val.strip():
            return val
    return None


async def _notice_circuit(arguments: dict | None, state: SessionState) -> None:
    """Side effects for any tool call that references a circuit file.

    Loads the circuit's persisted jobs (once per session) and bumps it to
    the top of the recent-circuits index. Best-effort — failures don't
    break dispatch. Recent-index writes are debounced per session via
    ``SessionState._touched_recent`` so repeated tool calls on the same
    circuit don't rewrite the file each time.

    The sidecar job load's file read is offloaded (a glob + JSON reads on a
    wedged ``/mnt/c`` would otherwise freeze the whole loop from this common
    dispatch path); its registry mutation stays on the loop. It is awaited so
    a job the handler is about to read is present. The recent-index write is
    a session background task, not awaited here: ``recent.touch`` can poll a
    contended cross-process lock for up to 10 s, and a best-effort
    bookkeeping write must not gate tool dispatch on that. The debounce set is updated before the write's first
    await, so back-to-back calls cannot double-write. A touch still in flight
    at shutdown may be lost — acceptable for best-effort state, and the atomic
    write keeps ``recent.json`` consistent either way.
    """
    raw = _extract_circuit_path(arguments)
    if not raw:
        return
    try:
        resolved = resolve_safe_path(raw, state.allowed_paths())
    except (PathSecurityError, OSError):
        return
    if resolved.suffix.lower() not in CIRCUIT_EXTENSIONS:
        return
    await state.ensure_jobs_loaded_for_async(resolved)
    state.background.spawn(state.note_recent_circuit(resolved))


# Error type → hint appended to error messages. Hints name only tools the
# consolidated surface exposes (the only profile since 0.6.0).
# PathSecurityError is handled separately (needs dynamic allowed_paths).
_ERROR_HINTS: dict[type[LTSpiceMCPError], str] = {
    _err.SimulationError: (
        "Use inspect with a capabilities query to verify simulator availability."
    ),
    _err.NetlistError: (
        "Use verify_circuit to lint the file, or inspect its components — "
        "or read the netlist directly."
    ),
    _err.JobNotFoundError: (
        'Use jobs with action:"list" to see known jobs — the id may be '
        "mistyped, evicted, or from a previous server session."
    ),
    _err.ResultError: (
        'Verify the run reached a terminal state with jobs (action:"status"), '
        "and read signals with analyze_results."
    ),
    _err.LibraryError: (
        'Use inspect with a model query (mode:"search") to find the part in the '
        "simulator's own libraries, or add .lib/.include directives to the netlist."
    ),
}


def _get_error_hint(err_type: type[LTSpiceMCPError]) -> str | None:
    """Get the error hint appended to a failed call's message, if any."""
    return _ERROR_HINTS.get(err_type)


def _configure_server_logging(config: ServerConfig) -> None:
    """Install the server process's stderr logging configuration."""
    configure_stderr_logging(config.log_level)


@asynccontextmanager
async def server_lifespan(server: Server) -> AsyncIterator[dict]:
    """Initialize session state on startup, clean up on shutdown.

    Loads configuration, sets up logging, detects simulators, creates session state.
    Logs a verbose startup summary to stderr for diagnostics.

    Yields:
        dict containing "state" key with SessionState instance

    Raises:
        Various exceptions during config/simulator setup (allowed to propagate)
    """
    lease_owner = object()
    lease_pid = acquire_session_lease(lease_owner)
    try:
        boot = await bootstrap_server_engine(
            on_config_loaded=_configure_server_logging,
            logger=logger,
        )
        state = boot.state
        config = state.config
        available = state.available_simulators
        config_file = config.config_path

        if config_file.exists():
            config_source = str(config_file)
        else:
            # No file: boot on built-in defaults. The default config is written
            # lazily on the first tool call instead of here (see call_tool), so the
            # server doesn't litter directories where its tools are never used.
            config_source = f"{config_file} (defaults; written on first tool use)"

        # Name the actually-detected simulators in the server instructions.
        # Both routes that publish them read this attribute per request — the
        # 2026-07-28 `server/discover` handler directly, and the older
        # `initialize` handshake through the initialization options the runner
        # builds when it answers — so setting it here, before the first request
        # is served, is what a client of either era reads.
        server.instructions = build_instructions(
            available, state.default_simulator, served=state.tool_dispatch
        )

        logger.info("=== LTSpice MCP Server Starting ===")
        logger.info(f"Server name: {server.name}")
        logger.info(f"Config source: {config_source}")
        logger.info(f"Working directory: {state.working_dir}")
        # The listing belongs on this line because it is the operator's answer
        # to "why did my argument descriptions vanish".
        logger.info(f"Tools: {len(state.tool_defs)}, listing: {config.tool_listing}")
        logger.info(f"Log level: {config.log_level}")

        logger.info("Detected simulators:")
        if available:
            for name, cls in available.items():
                is_default = cls == state.default_simulator
                default_marker = " (default)" if is_default else ""
                logger.info(f"  - {name}{default_marker}")
                # The simulator itself, not its launcher: under Wine the
                # launch command starts with "wine".
                exe_path = executable_path(cls)
                if exe_path is not None:
                    logger.info(f"    Executable: {exe_path}")
        else:
            logger.warning(
                "No simulators detected. Circuit editing will work but simulation tools will return errors."
            )
        for selector, cls in state.named_simulators.items():
            logger.info(f"  - {selector} (named): {executable_path(cls)}")

        logger.info(
            f"Default simulator: {state.default_simulator.__name__ if state.default_simulator else 'None'}"
        )

        if state.diagnostics:
            logger.warning("Startup diagnostics:")
            for diag in state.diagnostics:
                logger.warning(f"  - {diag}")

        logger.info("Allowed paths (sandbox):")
        for allowed_path in state.allowed_paths():
            logger.info(f"  - {allowed_path.resolve()}")

        if boot.preloaded_circuits:
            logger.info(
                "Preloaded persisted jobs for %d recent circuit(s)",
                boot.preloaded_circuits,
            )

        logger.info("Startup complete. Server ready for MCP connections.")

        try:
            yield {"state": state}
        finally:
            await state.shutdown()
            logger.info("Server shutdown complete")
    finally:
        release_session_lease(lease_owner, lease_pid)


# Server-level guidance surfaced to the consuming LLM at the MCP initialize
# handshake (forwarded by ``create_initialization_options`` ->
# ``InitializationOptions.instructions``). It is the one text every client
# shows a model without being asked, so it does three things only: send the
# model to the guide (``lib/guide.py``), say when to use Python and when the
# tools, and carry the rules that cost a wrong answer when missed. Everything
# else lives in the guide, which is read on demand. Kept under
# _INSTRUCTIONS_BUDGET including the runtime simulator prefix: Claude Code
# silently truncates server instructions at 2048 chars, and the tail (the
# result-trust and sandbox rules) is the part that must survive.
_INSTRUCTIONS_TEMPLATE = """\
SPICE simulation with LTspice and ngspice, and LTspice .asc schematic editing.

Read the guide first, every session: inspect(queries=[{{"kind": "guide"}}]) returns its core (how to work here, the Python API, the rules that cause silent errors) and an index of topic sections and task playbooks; add "section" to read one. Read the section for a task before starting it.

{python_door} Tools: one sandboxed step per call, with structured, paged replies, charts (plot_waveform), and jobs the server owns. Use Python for anything past a single call; the guide's core compares the two.

Write .cir/.net/.sp decks with your own file tools; change .asc schematics only through edit_schematic. Runs are cheap: simulate instead of reasoning it out. A run can finish with status completed and still hold a degenerate result (a coerced value, a skipped .meas): read observations, warnings and failures. Paths must lie in the sandbox; a refused path names the config line that widens it.
"""

# Claude Code's client truncates MCP server instructions at 2048 characters;
# the runtime prefix (active-simulator line) must fit inside it too.
_INSTRUCTIONS_BUDGET = 2048


#: The Python door, in its two editions: run_code in front of the library when
#: the operator serves it, the library alone when ``[tools] run_code = false``.
_PYTHON_DOOR_TOOL = (
    "Python: run_code runs a snippet with api in scope (in your own Python, from "
    "ltspice_mcp.api import Api), so loops, decisions, trace math and complete results "
    "take one call and far fewer tokens than a tool call per step. run_code has the "
    "server's own file and process authority, outside the sandbox, and a job it owns "
    "stops if its worker restarts unless submitted with detach=True."
)
_PYTHON_DOOR_LIBRARY = (
    "Python: from ltspice_mcp.api import Api gives the same operations in your own "
    "Python, so loops, decisions, trace math and complete results take one call and far "
    "fewer tokens than a tool call per step. A job it owns stops when your process "
    "exits unless submitted with detach=True."
)

#: The instructions as the default configuration serves them (run_code on) —
#: the static default the Server is constructed with, and what the tests pin.
CONSOLIDATED_INSTRUCTIONS = _INSTRUCTIONS_TEMPLATE.format(python_door=_PYTHON_DOOR_TOOL)

#: Sent once, on the first tool reply of a session that has not read the guide.
#: The instructions ask for the read up front, but a client is not obliged to
#: show them, and a model that skipped them meets this instead.
GUIDE_REMINDER = (
    'This session has not read the guide. Read its core first: inspect(queries=[{"kind": '
    '"guide"}]), or api.guide() in Python. Its index names the section for this task.'
)


def build_instructions(
    available: dict[str, type],
    default: type | None,
    *,
    served: Collection[str] = ("run_code",),
) -> str:
    """Prepend a line naming the actually-detected simulators to the static guide.

    The server is named for LTspice, so a client that only has ngspice would
    otherwise read the LTspice-centric name and the "symbols disabled" log as
    degradation. Stating the active engine up front removes that ambiguity.
    ``served`` is the session's tool set (the default configuration's surface
    when not given); the Python clause names run_code only when it is in it.
    """
    instructions = _INSTRUCTIONS_TEMPLATE.format(
        python_door=_PYTHON_DOOR_TOOL if "run_code" in served else _PYTHON_DOOR_LIBRARY
    )
    if not available:
        # The short no-simulator form: the long one plus the guide would
        # overflow the client's 2 KB instruction truncation.
        active = no_simulator_message(short=True)
    else:

        def disp(name: str) -> str:
            return SIMULATOR_DISPLAY.get(name, name)

        if len(available) == 1:
            active = f"Active simulator: {disp(next(iter(available)))}."
        else:
            default_name = next((n for n, c in available.items() if c is default), None)
            parts = [f"{disp(n)} (default)" if n == default_name else disp(n) for n in available]
            active = f"Active simulators: {', '.join(parts)}."
        if "ltspice" not in available:
            active += (
                " (LTspice not detected; .asc editing needs its symbol files "
                "and may be unavailable — simulation and analysis run on the "
                "active engine, unaffected.)"
            )
    return f"{active}\n\n{instructions}"


_client_capabilities: ContextVar[types.ClientCapabilities | None] = ContextVar(
    "mcp_client_capabilities", default=None
)
"""The calling client's capabilities, bound per tool call by ``call_tool``."""


def get_client_capabilities() -> types.ClientCapabilities | None:
    """The calling client's capabilities, or ``None`` if unavailable.

    Bound from the live request before the tool handler runs, so it reads the
    same value whether the client declared its capabilities in the ``initialize``
    handshake or in a 2026-07-28 per-request envelope. ``None`` outside a tool
    call, and when the client declared none. Used to pick the plot delivery
    channel (in-chat ``ui://`` widget vs local open).
    """
    return _client_capabilities.get()


def _tool_error(text: str, structured: dict[str, Any] | None = None) -> types.CallToolResult:
    """A failed tool call: the message on the text channel, ``is_error`` set.

    A tool that fails reports it in its result rather than as a JSON-RPC error,
    which is what lets the calling model read the message and correct itself.
    The SDK turned an exception into this shape for us until MCP SDK 2, which
    raises handler exceptions to the wire instead, so we build it here.
    ``structured`` mirrors what the caller needs to act on into
    structuredContent, which a structured-aware client reads instead of text.
    """
    return types.CallToolResult(
        content=[types.TextContent(type="text", text=text)],
        structured_content=structured,
        is_error=True,
    )


async def list_tools(
    ctx: ServerRequestContext, params: types.PaginatedRequestParams | None
) -> types.ListToolsResult:
    """Return the advertised tool definitions."""
    return types.ListToolsResult(tools=_get_state(ctx).tool_defs)


async def call_tool(
    ctx: ServerRequestContext, params: types.CallToolRequestParams
) -> types.CallToolResult:
    """Dispatch tool calls to registered handlers.

    All handlers return types.CallToolResult (the MCP protocol's canonical
    response type). Data-returning tools populate structuredContent.
    """
    state = _get_state(ctx)
    name = params.name
    arguments = params.arguments

    # Write a default config the first time a tool is actually used in this
    # directory — not at startup, which would litter every unrelated project
    # folder of anyone who has the plugin installed. Attempted at most once per
    # session: the flag also stops a read-only dir from rebuilding+rewriting the
    # config doc on every call. Best-effort — a write failure must not break the
    # tool call. ``write_config`` switches it off entirely.
    if not state.config_write_attempted:
        state.config_write_attempted = True
        cfg_path = state.config.config_path
        if state.config.write_config and not cfg_path.exists():
            with suppress(OSError):
                generate_default_config(cfg_path)

    registered = state.tool_dispatch.get(name)
    if registered is None:
        # A name this server does not serve is a lookup failure, not a tool
        # failure: there is no tool to attribute an error-flagged result to,
        # so it answers a JSON-RPC invalid-params error the way an unknown
        # resource URI does. The message names what the caller asked for and
        # the tools that do exist, which is the whole recovery.
        available = ", ".join(state.tool_dispatch)
        raise MCPError(
            types.INVALID_PARAMS,
            f"Unknown tool: {name}. Available tools: {available}.",
        )

    # Bind the caller's capabilities for the life of this request, so a
    # handler can pick its delivery channel (widget vs local open).
    _client_capabilities.set(ctx.session.client_capabilities)

    # Lazy-load persisted jobs for the circuit this tool is operating on,
    # and bump it in the recent-circuits index. Best-effort; errors swallowed;
    # the index write runs as a background task so it never gates dispatch.
    await _notice_circuit(arguments, state)

    return _with_guide_reminder(await _invoke(registered, name, arguments, state), state)


def _with_guide_reminder(
    result: types.CallToolResult, state: SessionState
) -> types.CallToolResult:
    """Add the read-the-guide reminder to the first reply of a session that has
    not read the guide, once.

    The reminder rides the text channel and ``structuredContent["hint"]`` both,
    because a structured-aware client shows only the latter. The result is
    copied, never edited in place: a replayed receipt's payload can be shared.
    """
    if state.guide_read or state.guide_reminded:
        return result
    state.guide_reminded = True
    update: dict[str, Any] = {
        "content": [*result.content, types.TextContent(type="text", text=GUIDE_REMINDER)]
    }
    structured = result.structured_content
    if structured is not None:
        hint = structured.get("hint")
        update["structured_content"] = {
            **structured,
            "hint": f"{hint} {GUIDE_REMINDER}" if hint else GUIDE_REMINDER,
        }
    return result.model_copy(update=update)


async def _invoke(
    registered: RegisteredTool, name: str, arguments: dict[str, Any] | None, state: SessionState
) -> types.CallToolResult:
    """Run one tool's handler, turning every failure into an error result."""
    # Invoke handler — enrich known errors with actionable guidance, and report
    # every failure as an is_error result rather than a JSON-RPC error, so the
    # calling model reads the message and can act on it.
    # Input validation (Pydantic model_validate) is handled by the registry
    # wrapper in _base.py — no need to validate here.
    try:
        return await registered.handler(arguments or {}, state)
    except ValidationError as e:
        detail = validation_error_detail(name, e, field_owners=state.field_owners)
        return _tool_error(f"Invalid arguments for {name}: {detail}")
    except PathSecurityError as e:
        # The caller reads the refusal in the result; the operator reads it on
        # the server's stderr, which is the only channel left for it.
        logger.warning("Path security violation in %s: %s", name, e)
        # The guidance is the whole recovery, so it rides structuredContent too.
        return _tool_error(
            path_denied_text(e, state),
            {"error": str(e), "code": e.code, "hint": state.sandbox_guidance()},
        )
    except LTSpiceMCPError as e:
        # Errors that already carry precise guidance opt out of the generic
        # per-type hint (show_hint=False) so it doesn't misdirect.
        hint = _get_error_hint(type(e)) if e.show_hint else None
        message = _err.caller_message(e, state.tool_dispatch)
        text = f"{message}\n\n{hint}" if hint else message
        # When the error carries structured suggestions (e.g. fuzzy model
        # matches), return them as structuredContent with is_error=True so
        # clients can parse them without regex'ing the text message.
        if e.suggestions:
            # Mirror the hint into structuredContent (self-sufficiency
            # contract): structured-aware clients drop the text channel, so a
            # text-only hint would be invisible exactly where it's needed.
            structured: dict[str, Any] = {"error": message, "suggestions": e.suggestions}
            if hint:
                structured["hint"] = hint
            return _tool_error(text, structured)
        return _tool_error(text)
    except Exception as e:
        # Surface the actual exception type + message in the response. A bare
        # "check server logs" is a dead end for an MCP client: the traceback
        # lands on the server's stderr, which the calling agent/user can't
        # reach. The concrete cause (e.g. "KeyError: 'PinName'") is what makes
        # an unexpected failure diagnosable. Full traceback still goes to logs.
        logger.exception(f"Unexpected error in tool {name}")
        return _tool_error(f"Internal error in {name}: {type(e).__name__}: {e}")


async def list_resources(
    ctx: ServerRequestContext, params: types.PaginatedRequestParams | None
) -> types.ListResourcesResult:
    """Return all static MCP resources."""
    return types.ListResourcesResult(resources=get_static_resources())


async def list_resource_templates(
    ctx: ServerRequestContext, params: types.PaginatedRequestParams | None
) -> types.ListResourceTemplatesResult:
    """Return all dynamic MCP resource templates."""
    return types.ListResourceTemplatesResult(resource_templates=get_resource_templates())


def _resource_error(message: str) -> MCPError:
    """The JSON-RPC error a failed resource read answers with.

    A read has no result to carry a message, so a failure has to be a protocol
    error. The 2026-07-28 revision dropped the separate resource-not-found code
    that earlier revisions used, so a URI that names nothing this server serves
    is an invalid parameter like any other.

    For failures the caller can act on only: an unknown URI, a denied path, a
    resource that refused the request. A server-side fault gets the
    internal-error code instead, so a client is not told to retry with
    different arguments when nothing it sends would help.
    """
    return MCPError(types.INVALID_PARAMS, message)


async def read_resource(
    ctx: ServerRequestContext, params: types.ReadResourceRequestParams
) -> types.ReadResourceResult:
    """Read a specific resource by URI.

    Dispatches to the appropriate handler based on URI scheme and path.

    Raises:
        MCPError: With the invalid-params code when the URI names no resource
            this server serves, or the resource cannot be read; with the
            internal-error code when the read raised something unexpected,
            which is a fault in this server rather than in the request.
    """
    state = _get_state(ctx)
    uri = params.uri

    try:
        # Resource reads are synchronous and read-only but not cheap: the
        # results/{job}/signals route does a full RawRead parse and the
        # recent route polls a cross-process file lock (time.sleep), so the
        # whole router runs off the loop. It never touches loop-owned
        # mutable state (the editor cache and library sessions stay untouched).
        return await asyncio.to_thread(handle_read_resource, uri, state)
    except PathSecurityError as e:
        # Same sandbox wall as the tool path (e.g. spice://netlists/{outside});
        # enrich it here so every resource route gets the recovery guidance.
        raise _resource_error(path_denied_text(e, state)) from None
    except (LTSpiceMCPError, ValueError) as e:
        raise _resource_error(str(e)) from None
    except Exception as e:
        # Nothing above classified this, so it is a fault in the server, not in
        # the request. Saying invalid-params here tells a client to try other
        # arguments for a failure no argument can avoid.
        logger.exception(f"Unexpected error reading resource {uri}")
        raise MCPError(
            types.INTERNAL_ERROR, f"Internal error reading resource: {type(e).__name__}: {e}"
        ) from e


async def list_prompts(
    ctx: ServerRequestContext, params: types.PaginatedRequestParams | None
) -> types.ListPromptsResult:
    """Return the workflow-starter prompts (registering this advertises the capability)."""
    del ctx, params
    return types.ListPromptsResult(prompts=prompts.list_prompts())


async def get_prompt(
    ctx: ServerRequestContext, params: types.GetPromptRequestParams
) -> types.GetPromptResult:
    """Return a prompt's messages with its arguments interpolated."""
    del ctx
    return prompts.get_prompt(params.name, params.arguments)


# The tool, resource and prompt listings are all built once, during lifespan
# startup, and never change while the process runs — so a client may hold onto
# one instead of re-listing every turn. The scope is private: each listing is
# shaped by this server's own configuration and sandbox, so it must not be
# served from a cache shared with another authorization context. An hour is
# well inside a session and well under any route by which the listings could
# change, since that needs a new server process and therefore a new connection.
_LISTING_CACHE_HINT = CacheHint(ttl_ms=3_600_000, scope="private")

# The name is overridable so the thin alias packages (circuit-mcp, ngspice-mcp)
# can self-identify in the handshake; it defaults to the canonical id. The env
# var must be set before this module is imported. See packaging/aliases/.
_SERVER_NAME = os.environ.get("LTSPICE_MCP_SERVER_NAME", "ltspice-mcp")

server: Server[dict] = Server(
    _SERVER_NAME,
    version=__version__,
    instructions=CONSOLIDATED_INSTRUCTIONS,
    lifespan=server_lifespan,
    cache_hints={
        "tools/list": _LISTING_CACHE_HINT,
        "resources/list": _LISTING_CACHE_HINT,
        "resources/templates/list": _LISTING_CACHE_HINT,
        "prompts/list": _LISTING_CACHE_HINT,
    },
    on_list_tools=list_tools,
    on_call_tool=call_tool,
    on_list_resources=list_resources,
    on_list_resource_templates=list_resource_templates,
    on_read_resource=read_resource,
    on_list_prompts=list_prompts,
    on_get_prompt=get_prompt,
)
