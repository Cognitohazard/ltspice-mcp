"""MCP server instance with lifespan management and tool dispatch."""

import asyncio
import logging
import os
import sys
import warnings
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from contextvars import ContextVar
from typing import Any

from mcp import types
from mcp.server.caching import CacheHint
from mcp.server.context import ServerRequestContext
from mcp.server.lowlevel import Server
from mcp.shared.exceptions import MCPDeprecationWarning, MCPError
from pydantic import ValidationError

from ltspice_mcp import __version__, prompts
from ltspice_mcp import errors as _err
from ltspice_mcp.api._session import acquire_session_lease, release_session_lease
from ltspice_mcp.config import ServerConfig, generate_default_config
from ltspice_mcp.engine import bootstrap_server_engine
from ltspice_mcp.errors import LTSpiceMCPError, PathSecurityError, compact_validation_error
from ltspice_mcp.lib import CIRCUIT_EXTENSIONS
from ltspice_mcp.lib.mcp_logging import mcp_log, set_log_fn
from ltspice_mcp.lib.pathutil import resolve_safe_path
from ltspice_mcp.lib.simulator import no_simulator_message
from ltspice_mcp.resources import (
    get_resource_templates,
    get_static_resources,
    handle_read_resource,
)
from ltspice_mcp.state import SessionState

# Tool argument keys that carry a circuit file path.
_CIRCUIT_PATH_KEYS: tuple[str, ...] = ("path", "netlist")

logger = logging.getLogger(__name__)

# The 2026-07-28 revision deprecates the logging capability, so the SDK warns
# both when a `logging/setLevel` handler is registered and on every log
# notification sent. We keep serving it: clients that negotiate an earlier
# revision still ask for it, and it is the only channel a tool has for progress
# text. The SDK drops a notification the peer never opted into either way, so
# the warning has nothing left to tell us — silence it once, by its message,
# rather than at every call site.
warnings.filterwarnings(
    "ignore",
    message="The logging capability is deprecated",
    category=MCPDeprecationWarning,
)


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


_recent_touch_tasks: set[asyncio.Task[None]] = set()
"""Strong refs to in-flight recent-index writes — ``create_task`` results are
garbage-collectable while pending; each task discards itself when done."""


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
    fire-and-forget: ``recent.touch`` can poll a contended cross-process lock
    for up to 10 s, and a best-effort bookkeeping write must not gate tool
    dispatch on that. The debounce set is updated before the write's first
    await, so back-to-back calls cannot double-write. A touch still in flight
    at shutdown may be lost — acceptable for best-effort state, and the atomic
    write keeps ``recent.json`` consistent either way.
    """
    raw = _extract_circuit_path(arguments)
    if not raw:
        return
    try:
        resolved = resolve_safe_path(raw, state.config.allowed_paths)
    except (PathSecurityError, OSError):
        return
    if resolved.suffix.lower() not in CIRCUIT_EXTENSIONS:
        return
    await state.ensure_jobs_loaded_for_async(resolved)
    task = asyncio.create_task(state.note_recent_circuit(resolved))
    _recent_touch_tasks.add(task)
    task.add_done_callback(_recent_touch_tasks.discard)


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
        'Use inspect with a model query (mode:"enumerate") to see loaded '
        "libraries, or add .lib/.include directives to the netlist directly."
    ),
}


def _get_error_hint(err_type: type[LTSpiceMCPError]) -> str | None:
    """Get the error hint appended to a failed call's message, if any."""
    return _ERROR_HINTS.get(err_type)


def _path_reject_guidance(state: SessionState) -> str:
    """Recovery guidance appended to a PathSecurityError at every agent-facing
    boundary — tool calls AND resource reads. The agent can't widen the sandbox
    itself, so name the knob and the human-escalation/move-the-file fallback or
    it dead-ends. One builder so the two boundaries can't drift."""
    allowed = ", ".join(str(p) for p in state.config.allowed_paths)
    return (
        f"Allowed paths: {allowed}\n"
        "To work on this file, move or copy it into one of those directories, "
        "or ask the user to widen the sandbox: [security] allowed_paths in "
        f"{state.config.config_path} or LTSPICE_MCP_ALLOWED_PATHS (restart "
        "required). An inspect capabilities query shows the full sandbox "
        "configuration."
    )


def _configure_server_logging(config: ServerConfig) -> None:
    """Install the server process's stderr logging configuration."""
    logging.basicConfig(
        level=getattr(logging, config.log_level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stderr)],
        force=True,
    )


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
        server.instructions = build_instructions(available, state.default_simulator)

        logger.info("=== LTSpice MCP Server Starting ===")
        logger.info(f"Server name: {server.name}")
        logger.info(f"Config source: {config_source}")
        logger.info(f"Working directory: {state.working_dir}")
        logger.info(f"Tool profile: {config.tool_profile} ({len(state.tool_defs)} tools)")
        logger.info(f"Log level: {config.log_level}")

        logger.info("Detected simulators:")
        if available:
            for name, cls in available.items():
                is_default = cls == state.default_simulator
                default_marker = " (default)" if is_default else ""
                logger.info(f"  - {name}{default_marker}")
                try:
                    # Try to get executable path if available
                    if hasattr(cls, "spice_exe"):
                        exe_path = (
                            cls.spice_exe[0] if isinstance(cls.spice_exe, list) else cls.spice_exe
                        )
                        logger.info(f"    Executable: {exe_path}")
                except Exception:
                    pass
        else:
            logger.warning(
                "No simulators detected. Circuit editing will work but simulation tools will return errors."
            )

        logger.info(
            f"Default simulator: {state.default_simulator.__name__ if state.default_simulator else 'None'}"
        )

        if state.diagnostics:
            logger.warning("Startup diagnostics:")
            for diag in state.diagnostics:
                logger.warning(f"  - {diag}")

        logger.info("Allowed paths (sandbox):")
        for allowed_path in config.allowed_paths:
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
# ``InitializationOptions.instructions``). Cross-cutting workflow guidance
# only — per-tool detail stays in the individual tool descriptions, which
# remain the contract (client injection of this string is not guaranteed).
# Six tools over three planes; terse, because the client re-reads it each
# turn. Kept under _INSTRUCTIONS_BUDGET including the runtime simulator
# prefix: Claude Code silently truncates server instructions at 2048 chars,
# and the tail (the result-trust paragraph) is the part that must survive.
CONSOLIDATED_INSTRUCTIONS = """\
For any circuit or SPICE task: amplifiers, filters, regulators, schematics. Write .cir/.net/.sp decks with your own file tools; the six tools below run them, analyze results, check circuits, and edit .asc geometry; plot_waveform draws plots. Routing: run quick one-off ngspice jobs yourself and bring the .raw; analyze_results raw_path parses runs this server never executed. Use run_experiments for LTspice (no native automation), sweep/corner/MC matrices, and jobs that outlive a call.

Runs are cheap. Simulate to check instead of reasoning it out.

EXECUTE — run_experiments: staged decks across declared variations (strict assignments plus one random/MC); optional request_id: pass one for a durable, idempotent submission; quick jobs return inline, longer ones a receipt/job_id. jobs: status, wait (long-poll), cancel, list, run pages; by job_id or request_id. Code loops: from ltspice_mcp.api import Api, the same ops in-process.

UNDERSTAND — analyze_results: typed recipes over completed runs/experiments; case/step-attributed values, reductions, spec verdicts. inspect: read-only; capabilities, symbols, net trace, components, models; reference: find a recipe/op/check and its fields by plain words ('phase margin').

AUTHOR — edit_schematic: typed op batch on one .asc sheet; transactional, revision-guarded (expected_sha256); returns geometry facts. verify_circuit: lint, symbols, export, layout, quality, compare, optional render.

A run can finish with status completed and still hold a degenerate result (a coerced value, a skipped .meas): read observations, warnings, and per-item failures. Match the recipe to the run type (.AC vs .tran) or analyze_results errors.
"""

# Claude Code's client truncates MCP server instructions at 2048 characters;
# the runtime prefix (active-simulator line) must fit inside it too.
_INSTRUCTIONS_BUDGET = 2048

# Friendly display names for the detected-simulator line prepended to the
# instructions at runtime (registry keys are lowercase).
_SIM_DISPLAY = {"ltspice": "LTspice", "ngspice": "ngspice", "qspice": "QSPICE", "xyce": "Xyce"}


def build_instructions(available: dict[str, type], default: type | None) -> str:
    """Prepend a line naming the actually-detected simulators to the static guide.

    The server is named for LTspice, so a client that only has ngspice would
    otherwise read the LTspice-centric name and the "symbols disabled" log as
    degradation. Stating the active engine up front removes that ambiguity.
    """
    instructions = CONSOLIDATED_INSTRUCTIONS
    if not available:
        # The short no-simulator form: the long one plus the guide would
        # overflow the client's 2 KB instruction truncation.
        active = no_simulator_message(short=True)
    else:

        def disp(name: str) -> str:
            return _SIM_DISPLAY.get(name, name)

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


def _tool_error(text: str) -> types.CallToolResult:
    """A failed tool call: the message on the text channel, ``is_error`` set.

    A tool that fails reports it in its result rather than as a JSON-RPC error,
    which is what lets the calling model read the message and correct itself.
    The SDK turned an exception into this shape for us until MCP SDK 2, which
    raises handler exceptions to the wire instead, so we build it here.
    """
    return types.CallToolResult(
        content=[types.TextContent(type="text", text=text)],
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
    # tool call.
    if not state.config_write_attempted:
        state.config_write_attempted = True
        cfg_path = state.config.config_path
        if not cfg_path.exists():
            with suppress(OSError):
                generate_default_config(cfg_path)

    registered = state.tool_dispatch.get(name)
    if registered is None:
        return _tool_error(f"Unknown tool: {name}")

    # Set up MCP protocol logging for this request.
    # Handlers and services call mcp_log() which reads this ContextVar —
    # no server/session reference needed downstream. Messages below the
    # client's requested minimum level (logging/setLevel) are not sent.
    session = ctx.session
    _client_capabilities.set(session.client_capabilities)

    async def _log(level: str, msg: str) -> None:
        if _below_client_log_level(level, state.client_log_level):
            return
        await session.send_log_message(level=level, data=msg, logger="ltspice-mcp")  # type: ignore[arg-type]

    set_log_fn(_log)

    # Lazy-load persisted jobs for the circuit this tool is operating on,
    # and bump it in the recent-circuits index. Best-effort; errors swallowed;
    # the index write runs as a background task so it never gates dispatch.
    await _notice_circuit(arguments, state)

    # Invoke handler — enrich known errors with actionable guidance, and report
    # every failure as an is_error result rather than a JSON-RPC error, so the
    # calling model reads the message and can act on it.
    # Input validation (Pydantic model_validate) is handled by the registry
    # wrapper in _base.py — no need to validate here.
    try:
        return await registered.handler(arguments or {}, state)
    except ValidationError as e:
        detail = compact_validation_error(
            e,
            field_owners=state.field_owners,
        )
        return _tool_error(f"Invalid arguments for {name}: {detail}")
    except PathSecurityError as e:
        await mcp_log("warning", f"Path security violation in {name}: {e}")
        return _tool_error(f"{e}\n\n{_path_reject_guidance(state)}")
    except LTSpiceMCPError as e:
        # Errors that already carry precise guidance opt out of the generic
        # per-type hint (show_hint=False) so it doesn't misdirect.
        hint = _get_error_hint(type(e)) if e.show_hint else None
        text = f"{e}\n\n{hint}" if hint else str(e)
        # When the error carries structured suggestions (e.g. fuzzy model
        # matches), return them as structuredContent with is_error=True so
        # clients can parse them without regex'ing the text message.
        if e.suggestions:
            # Mirror the hint into structuredContent (self-sufficiency
            # contract): structured-aware clients drop the text channel, so a
            # text-only hint would be invisible exactly where it's needed.
            structured: dict[str, Any] = {"error": str(e), "suggestions": e.suggestions}
            if hint:
                structured["hint"] = hint
            return types.CallToolResult(
                content=[types.TextContent(type="text", text=text)],
                structured_content=structured,
                is_error=True,
            )
        return _tool_error(text)
    except Exception as e:
        # Surface the actual exception type + message in the response. A bare
        # "check server logs" is a dead end for an MCP client: the traceback
        # lands on the server's stderr, which the calling agent/user can't
        # reach. The concrete cause (e.g. "KeyError: 'PinName'") is what makes
        # an unexpected failure diagnosable. Full traceback still goes to logs.
        logger.exception(f"Unexpected error in tool {name}")
        return _tool_error(f"Internal error in {name}: {type(e).__name__}: {e}")


# MCP log severities, ascending RFC-5424 rank (the protocol's LoggingLevel).
_LOG_SEVERITY = {
    "debug": 0,
    "info": 1,
    "notice": 2,
    "warning": 3,
    "error": 4,
    "critical": 5,
    "alert": 6,
    "emergency": 7,
}


def _below_client_log_level(level: str, client_min: str | None) -> bool:
    """True when ``level`` is below the client's requested minimum.

    No minimum set (client never called logging/setLevel) or an unknown level
    string sends the message — filtering is an opt-in narrowing, never a
    silent drop of something we can't rank.
    """
    if client_min is None:
        return False
    rank = _LOG_SEVERITY.get(level)
    floor = _LOG_SEVERITY.get(client_min)
    if rank is None or floor is None:
        return False
    return rank < floor


async def set_logging_level(
    ctx: ServerRequestContext, params: types.SetLevelRequestParams
) -> types.EmptyResult:
    """Store the client's minimum log level; also declares the logging
    capability (the SDK only advertises it when this handler exists, and
    without it spec-conforming clients drop our notifications/message)."""
    _get_state(ctx).client_log_level = params.level
    return types.EmptyResult()


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
    """
    return MCPError(types.INVALID_PARAMS, message)


async def read_resource(
    ctx: ServerRequestContext, params: types.ReadResourceRequestParams
) -> types.ReadResourceResult:
    """Read a specific resource by URI.

    Dispatches to the appropriate handler based on URI scheme and path.

    Raises:
        MCPError: With the invalid-params code when the URI names no resource
            this server serves, or the resource cannot be read.
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
        raise _resource_error(f"{e}\n\n{_path_reject_guidance(state)}") from None
    except (LTSpiceMCPError, ValueError) as e:
        raise _resource_error(str(e)) from None
    except Exception as e:
        logger.exception(f"Unexpected error reading resource {uri}")
        raise _resource_error(f"Internal error reading resource: {type(e).__name__}: {e}") from e


async def list_prompts(
    ctx: ServerRequestContext, params: types.PaginatedRequestParams | None
) -> types.ListPromptsResult:
    """Return the workflow-starter prompts (registering this advertises the capability)."""
    return types.ListPromptsResult(
        prompts=prompts.list_prompts(_get_state(ctx).config.tool_profile)
    )


async def get_prompt(
    ctx: ServerRequestContext, params: types.GetPromptRequestParams
) -> types.GetPromptResult:
    """Return a prompt's messages with its arguments interpolated."""
    return prompts.get_prompt(params.name, params.arguments, _get_state(ctx).config.tool_profile)


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
    on_set_logging_level=set_logging_level,
)
