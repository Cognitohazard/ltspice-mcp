"""MCP server instance with lifespan management and tool dispatch."""

import asyncio
import logging
import os
import sys
from collections.abc import AsyncIterator, Iterable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import NamedTuple

from mcp import types
from mcp.server.lowlevel import Server
from mcp.server.lowlevel.helper_types import ReadResourceContents
from mcp.server.models import InitializationOptions
from pydantic import AnyUrl, ValidationError

from ltspice_mcp import __version__, prompts
from ltspice_mcp import errors as _err
from ltspice_mcp.config import ServerConfig, generate_default_config
from ltspice_mcp.errors import LTSpiceMCPError, PathSecurityError
from ltspice_mcp.lib import CIRCUIT_EXTENSIONS
from ltspice_mcp.lib.mcp_logging import mcp_log, set_log_fn
from ltspice_mcp.lib.pathutil import resolve_safe_path
from ltspice_mcp.lib.simulator import detect_simulators, no_simulator_message
from ltspice_mcp.resources import (
    get_resource_templates,
    get_static_resources,
    handle_read_resource,
)
from ltspice_mcp.state import SessionState

# Tool argument keys that carry a circuit file path.
_CIRCUIT_PATH_KEYS: tuple[str, ...] = ("path", "netlist")

logger = logging.getLogger(__name__)

# Set by main.py to the InitializationOptions passed to server.run(), so the
# lifespan can rewrite its instructions once simulators are detected.
_dynamic_init_options: InitializationOptions | None = None


def register_init_options(opts: InitializationOptions) -> None:
    """Hand the live initialize options to the lifespan for instruction rewrite."""
    global _dynamic_init_options
    _dynamic_init_options = opts


def _get_state(server_: Server) -> SessionState:
    """Extract session state from the lifespan context."""
    try:
        return server_.request_context.lifespan_context["state"]
    except (AttributeError, KeyError) as e:
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


def _notice_circuit(arguments: dict | None, state: SessionState) -> None:
    """Side effects for any tool call that references a circuit file.

    Loads the circuit's persisted jobs (once per session) and bumps it to
    the top of the recent-circuits index. Best-effort — failures don't
    break dispatch. Recent-index writes are debounced per session via
    ``SessionState._touched_recent`` so repeated tool calls on the same
    circuit don't rewrite the file each time.

    The sidecar job load stays on the loop (small per-circuit JSON files,
    and it mutates the JobRegistry). The recent-index write is fire-and-
    forget: ``recent.touch`` can poll a contended cross-process lock for
    up to 10 s, and a best-effort bookkeeping write must not gate tool
    dispatch on that. The debounce set is updated before the write's first
    await, so back-to-back calls cannot double-write. A touch still in
    flight at shutdown may be lost — acceptable for best-effort state, and
    the atomic write keeps ``recent.json`` consistent either way.
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
    state.ensure_jobs_loaded_for(resolved)
    task = asyncio.create_task(state.note_recent_circuit(resolved))
    _recent_touch_tasks.add(task)
    task.add_done_callback(_recent_touch_tasks.discard)


def _configure_asc_editor(config: ServerConfig, available: dict) -> None:
    """Configure AscEditor library paths for .asc schematic support.

    Schematic editing only needs the ``.asy`` symbol library — NOT a working
    simulator binary — so symbol resolution is deliberately decoupled from
    simulator detection. A WSL box with the symbols present but a mis-pathed
    (or absent) LTspice executable can still edit ``.asc`` files.

    Resolution order:
    1. Explicit config.symbol_paths / LTSPICE_MCP_SYMBOL_PATHS override (any platform)
    2. WSL — resolve symbols via Windows %LOCALAPPDATA%, regardless of whether
       the LTspice *executable* was detected
    3. Windows native / Linux+Wine — spicelib's prepare_for_simulator (needs
       the detected LTspice class)
    4. Otherwise — no symbols available, .asc editing disabled
    """
    from spicelib.editor.asc_editor import AscEditor

    # 1. Explicit config override takes priority on all platforms
    if config.symbol_paths:
        valid = [str(p) for p in config.symbol_paths if p.is_dir()]
        if valid:
            AscEditor.custom_lib_paths = valid
            logger.info(f"AscEditor symbol paths from config: {valid}")
            return
        logger.warning(f"Configured symbol_paths do not exist: {config.symbol_paths}")

    from ltspice_mcp.lib.wsl import get_ltspice_lib_paths, is_wsl

    # 2. WSL — symbol libs live under %LOCALAPPDATA%/LTspice/lib/sym and resolve
    #    independently of simulator-executable detection (spicelib can't find
    #    them via /mnt/c/ on its own). This is the key decoupling: a stale
    #    simulator path must not also disable schematic editing.
    if is_wsl():
        lib_paths = get_ltspice_lib_paths()
        if lib_paths:
            AscEditor.custom_lib_paths = lib_paths
            logger.info(f"AscEditor WSL library paths: {lib_paths}")
            return
        logger.info(
            ".asc schematic graphics editing unavailable on WSL (no LTspice symbol "
            "library found); SPICE simulation and netlist editing are unaffected. "
            "To enable it, set [schematic] symbol_paths in ltspice-mcp.toml or "
            "LTSPICE_MCP_SYMBOL_PATHS env var."
        )
        return

    # 3. Windows native (or Linux with Wine) — needs the detected LTspice class
    ltspice_cls = available.get("ltspice")
    if ltspice_cls is None:
        logger.info(
            ".asc schematic graphics editing unavailable (no LTspice symbol library "
            "found); SPICE simulation and netlist editing are unaffected"
        )
        return

    try:
        AscEditor.prepare_for_simulator(ltspice_cls)
        if AscEditor.simulator_lib_paths or AscEditor.custom_lib_paths:
            logger.info("AscEditor configured via prepare_for_simulator()")
            return
        logger.warning("prepare_for_simulator() found no library paths")
    except Exception as e:
        logger.warning(f"AscEditor prepare_for_simulator failed: {e}")


class _ErrorHint(NamedTuple):
    """Profile-aware error hint: full references MCP tools, agentic gives direct guidance."""

    full: str
    agentic: str


# Error type → profile-aware hint appended to error messages.
# PathSecurityError is handled separately (needs dynamic allowed_paths).
_ERROR_HINTS: dict[type[LTSpiceMCPError], _ErrorHint] = {
    _err.MissingModelError: _ErrorHint(
        full=(
            "Try find_model to fuzzy-match against loaded libraries "
            "(catches typos and near-neighbour part numbers), or load_library "
            "to load a library file containing it."
        ),
        agentic=(
            "Try find_model to fuzzy-match against loaded libraries "
            "(catches typos), or load a library containing it and rerun."
        ),
    ),
    _err.ConvergenceError: _ErrorHint(
        full=(
            "Suggestions:\n"
            "  - Add .OPTIONS (e.g., .OPTIONS reltol=0.003 or .OPTIONS method=gear)\n"
            "  - Use edit_directive to add a .OPTIONS directive\n"
            "  - Check component values for very large/small ratios"
        ),
        agentic=(
            "Suggestions:\n"
            "  - Add a .OPTIONS directive to the netlist "
            "(e.g., .OPTIONS reltol=0.003 or .OPTIONS method=gear)\n"
            "  - Check component values for very large/small ratios"
        ),
    ),
    _err.SingularMatrixError: _ErrorHint(
        full=(
            "This usually means a floating node or short circuit.\n"
            "Use read_circuit to inspect the netlist for connectivity issues."
        ),
        agentic=(
            "This usually means a floating node or short circuit.\n"
            "Inspect the netlist for connectivity issues."
        ),
    ),
    _err.SimulationError: _ErrorHint(
        full="Use server_status to verify simulator availability.",
        agentic="Use server_status to verify simulator availability.",
    ),
    _err.NetlistError: _ErrorHint(
        full=(
            "Use read_circuit to inspect the file, or "
            "list_components to verify component references."
        ),
        agentic=(
            "Inspect the netlist file directly, or use "
            "list_components to verify component references."
        ),
    ),
    _err.JobNotFoundError: _ErrorHint(
        full=(
            "Use check_job with no job_id to list known jobs — the id may be "
            "mistyped, evicted, or from a previous server session."
        ),
        agentic=(
            "Use check_job with no job_id to list known jobs — the id may be "
            "mistyped, evicted, or from a previous server session."
        ),
    ),
    _err.ResultError: _ErrorHint(
        full=(
            "Verify the simulation completed successfully with check_job, "
            "and check signal names with simulation_summary."
        ),
        agentic=(
            "Verify the simulation completed successfully with check_job, "
            "and check signal names with simulation_summary."
        ),
    ),
    _err.LibraryError: _ErrorHint(
        full=("Use list_libraries to see loaded libraries, or load_library to load a new one."),
        agentic=(
            "Use find_model to fuzzy-match against loaded libraries, "
            "or add .lib directives to the netlist manually."
        ),
    ),
}


def _get_error_hint(err_type: type[LTSpiceMCPError], profile: str) -> str | None:
    """Get the appropriate error hint for the active tool profile."""
    hint = _ERROR_HINTS.get(err_type)
    if hint is None:
        return None
    return hint.agentic if profile == "agentic" else hint.full


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
    config = ServerConfig.load()
    config_file = config.config_path

    if config_file.exists():
        config_source = str(config_file)
    else:
        # Generate default config in CWD
        config_file = Path.cwd() / "ltspice-mcp.toml"
        generate_default_config(config_file)
        config_source = f"{config_file} (generated)"

    logging.basicConfig(
        level=getattr(logging, config.log_level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stderr)],
        force=True,  # Override any existing config
    )
    logger = logging.getLogger("ltspice_mcp.server")

    diagnostics: list[str] = []
    available = detect_simulators(config, diagnostics)
    _configure_asc_editor(config, available)

    state = SessionState.create(config, available, diagnostics)

    # Rewrite the initialize instructions to name the actually-detected
    # simulators. main.py stashes the InitializationOptions it passed to
    # server.run() here; lifespan startup completes before the initialize
    # request is answered, and that request reads the same object, so the
    # client sees the dynamic line. Falls back to the static text if unset.
    if _dynamic_init_options is not None:
        _dynamic_init_options.instructions = build_instructions(available, state.default_simulator)

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
        logger.warning("Startup diagnostics (also surfaced via server_status):")
        for diag in state.diagnostics:
            logger.warning(f"  - {diag}")

    logger.info("Allowed paths (sandbox):")
    for allowed_path in config.allowed_paths:
        logger.info(f"  - {allowed_path.resolve()}")

    # Eager-load persisted jobs for the top-N recently-touched circuits so
    # first-tool-call latency on those circuits doesn't surprise the user.
    # Circuits outside this budget fall back to lazy load on first tool call.
    if config.persist_jobs and config.preload_recent_count > 0:
        preloaded = state.job_registry.preload_recent(max_circuits=config.preload_recent_count)
        if preloaded:
            logger.info("Preloaded persisted jobs for %d recent circuit(s)", preloaded)

    logger.info("Startup complete. Server ready for MCP connections.")

    try:
        yield {"state": state}
    finally:
        await state.shutdown()
        logger.info("Server shutdown complete")


# Server-level guidance surfaced to the consuming LLM at the MCP initialize
# handshake (forwarded by ``create_initialization_options`` ->
# ``InitializationOptions.instructions``). Cross-cutting workflow guidance only —
# per-tool detail stays in the individual tool descriptions, which remain the
# contract (client injection of this string is not guaranteed). Kept terse
# (~200 words) since every token is re-read on each LLM turn. The
# "completed can be degenerate" line warns the consuming LLM not to
# equate a completed run with a correct result.
SERVER_INSTRUCTIONS = """\
LTspice-MCP simulates SPICE circuits and edits LTspice .asc schematics.

Prefer the netlist path by default — fewer steps, more reliable: author a .cir/.net netlist, validate_netlist, then run_simulation and the analysis tools. Build or edit .asc schematics only when the task is about schematic graphics/layout, or the user asks. Build .asc with create_schematic/add_component/connect plus apply_schematic_ops for the rest.

Match the analysis tool to the run type or it errors: bode_metrics/resonance/stability_metrics need a .AC run; signal_stats/edge_metrics/timing_between/periodic_metrics/pulse_response need .tran; operating_point needs .op. Scalar results come from .meas directives (failures in failed_measurements); read sweep/Monte-Carlo runs via batch_results or job_id+run_index, aggregates via measurement_stats. To visualize a waveform use plot_waveform (get_waveform for the raw numbers) — do not generate plots externally.

A run can report "completed" yet be degenerate (coerced value, skipped .meas) — check the returned warnings/errors and the `observations` list, don't assume success means correct. `observations` reads the RESULT and does not re-run netlist topology analysis; it surfaces facts worth weighing (the simulator's own error lines, requested .meas/.four that weren't produced, extreme/non-finite node values, and scans that were skipped) — they are facts for you to judge, not a verdict; an empty list means nothing tripped a check, NOT that the result is verified. validate_netlist is the pre-flight gate: topology faults like a floating or capacitive-island node are caught there, not by observations, but it won't catch value typos or undefined models (resolved at run time).

Build or edit .asc with the schematic tools, never by hand — hand-writing the file forfeits connect's orthogonal routing and its pin-collision/junction checks. Pin names and coordinates are symbol-specific (a resistor's are A/B, not 1/2) — read them from add_component/symbol_info. Wire signal nets with connect (orthogonal segments only; waypoints for bends; route outside component bodies); put a ground flag at each ground pin with an apply_schematic_ops add_net_label op (net="0", pin=...); do NOT net-label signal nets — wire them. Ack-only mutations (move/remove a component, set an attribute, add/remove a net label, remove a wire) are apply_schematic_ops ops, not standalone tools; tools that return info you act on (add_component pin geometry, connect routing) are standalone. For a multi-step build use apply_schematic_ops (one transaction). The full schematic-layout playbook (tier alignment, mirror/diff-pair orientations, bus routing) is the spice://guide resource.
"""

# Friendly display names for the detected-simulator line prepended to the
# instructions at runtime (registry keys are lowercase).
_SIM_DISPLAY = {"ltspice": "LTspice", "ngspice": "ngspice", "qspice": "QSPICE", "xyce": "Xyce"}


def build_instructions(available: dict[str, type], default: type | None) -> str:
    """Prepend a line naming the actually-detected simulators to the static guide.

    The server is named for LTspice, so a client that only has ngspice would
    otherwise read the LTspice-centric name and the "symbols disabled" log as
    degradation. Stating the active engine up front removes that ambiguity.
    """
    if not available:
        active = no_simulator_message()
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
                " (LTspice not detected, so .asc schematic editing is unavailable; "
                "simulation and analysis run on the active engine and are unaffected.)"
            )
    return f"{active}\n\n{SERVER_INSTRUCTIONS}"


# The name is overridable so the thin alias packages (circuit-mcp, ngspice-mcp)
# can self-identify in the handshake; it defaults to the canonical id. The env
# var must be set before this module is imported. See packaging/aliases/.
_SERVER_NAME = os.environ.get("LTSPICE_MCP_SERVER_NAME", "ltspice-mcp")
server = Server(_SERVER_NAME, version=__version__, instructions=SERVER_INSTRUCTIONS)
server.lifespan = server_lifespan


def get_client_capabilities() -> types.ClientCapabilities | None:
    """The connected client's capabilities, or ``None`` if unavailable.

    Reads the live MCP session's ``initialize`` params. Returns ``None`` outside a
    request (``LookupError``) or in stateless mode (no ``client_params``). Call
    this from a handler coroutine — the request context is a ``ContextVar`` bound
    to the current task and is NOT propagated into ``asyncio.to_thread`` workers.
    Used to pick the plot delivery channel (in-chat ``ui://`` widget vs local open).
    """
    try:
        params = server.request_context.session.client_params
    except LookupError:
        return None
    return params.capabilities if params is not None else None


@server.list_tools()
async def list_tools() -> list[types.Tool]:
    """Return MCP tools filtered by the active tool profile."""
    return _get_state(server).tool_defs


@server.call_tool()
async def call_tool(name: str, arguments: dict | None):
    """Dispatch tool calls to registered handlers.

    All handlers return types.CallToolResult (the MCP protocol's canonical
    response type). Data-returning tools populate structuredContent.
    """
    state = _get_state(server)

    # Look up handler in profile-filtered dispatch table
    registered = state.tool_dispatch.get(name)
    if registered is None:
        raise ValueError(f"Unknown tool: {name}")

    # Set up MCP protocol logging for this request.
    # Handlers and services call mcp_log() which reads this ContextVar —
    # no server/session reference needed downstream.
    session = server.request_context.session

    async def _log(level: str, msg: str) -> None:
        await session.send_log_message(level=level, data=msg, logger="ltspice-mcp")  # type: ignore[arg-type]

    set_log_fn(_log)

    # Lazy-load persisted jobs for the circuit this tool is operating on,
    # and bump it in the recent-circuits index. Best-effort; errors swallowed;
    # the index write runs as a background task so it never gates dispatch.
    _notice_circuit(arguments, state)

    # Invoke handler — enrich known errors with actionable guidance.
    # Exceptions propagate to the MCP SDK which sets isError=True.
    # Input validation (Pydantic model_validate) is handled by the registry
    # wrapper in _base.py — no need to validate here.
    try:
        return await registered.handler(arguments or {}, state)
    except ValidationError as e:
        raise ValueError(f"Invalid arguments for {name}: {e}") from None
    except PathSecurityError as e:
        await mcp_log("warning", f"Path security violation in {name}: {e}")
        allowed = ", ".join(str(p) for p in state.config.allowed_paths)
        raise PathSecurityError(
            f"{e}\n\nAllowed paths: {allowed}\n"
            f"Use server_status to see full sandbox configuration."
        ) from None
    except LTSpiceMCPError as e:
        # Errors that already carry precise guidance opt out of the generic
        # per-type hint (show_hint=False) so it doesn't misdirect.
        hint = _get_error_hint(type(e), state.config.tool_profile) if e.show_hint else None
        text = f"{e}\n\n{hint}" if hint else str(e)
        # When the error carries structured suggestions (e.g. fuzzy model
        # matches), return them as structuredContent with isError=True so
        # clients can parse them without regex'ing the text message.
        if e.suggestions:
            return types.CallToolResult(
                content=[types.TextContent(type="text", text=text)],
                structuredContent={"error": str(e), "suggestions": e.suggestions},
                isError=True,
            )
        if hint:
            raise type(e)(f"{e}\n\n{hint}") from None
        raise
    except Exception as e:
        # Surface the actual exception type + message in the response. A bare
        # "check server logs" is a dead end for an MCP client: the traceback
        # lands on the server's stderr, which the calling agent/user can't
        # reach. The concrete cause (e.g. "KeyError: 'PinName'") is what makes
        # an unexpected failure diagnosable. Full traceback still goes to logs.
        logger.exception(f"Unexpected error in tool {name}")
        raise RuntimeError(f"Internal error in {name}: {type(e).__name__}: {e}") from e


@server.list_resources()
async def list_resources() -> list[types.Resource]:
    """Return all static MCP resources."""
    return get_static_resources()


@server.list_resource_templates()
async def list_resource_templates() -> list[types.ResourceTemplate]:
    """Return all dynamic MCP resource templates."""
    return get_resource_templates()


@server.read_resource()
async def read_resource(uri: AnyUrl) -> Iterable[ReadResourceContents]:
    """Read a specific resource by URI.

    Dispatches to appropriate handler based on URI scheme and path.
    Converts internal TextResourceContents/BlobResourceContents to the
    SDK's ReadResourceContents format (which uses .content instead of .text).

    Args:
        uri: Resource URI to read (spice://...)

    Returns:
        Iterable of ReadResourceContents entries

    Raises:
        ValueError: If URI is unknown or resource not found
    """
    state = _get_state(server)

    try:
        # Resource reads are synchronous and read-only but not cheap: the
        # results/{job}/signals route does a full RawRead parse and the
        # recent route polls a cross-process file lock (time.sleep), so the
        # whole router runs off the loop. It never touches loop-owned
        # mutable state (the editor cache and library sessions stay untouched).
        result = await asyncio.to_thread(handle_read_resource, str(uri), state)
    except LTSpiceMCPError as e:
        raise ValueError(str(e)) from None
    except Exception as e:
        logger.exception(f"Unexpected error reading resource {uri}")
        raise ValueError(f"Internal error reading resource: {type(e).__name__}: {e}") from e

    # Convert from types.TextResourceContents/BlobResourceContents
    # to the SDK's ReadResourceContents (which has .content not .text)
    converted = []
    for item in result.contents:
        if isinstance(item, types.TextResourceContents):
            converted.append(
                ReadResourceContents(
                    content=item.text,
                    mime_type=item.mimeType,
                )
            )
        elif isinstance(item, types.BlobResourceContents):
            converted.append(
                ReadResourceContents(
                    content=item.blob,
                    mime_type=item.mimeType,
                )
            )
    return converted


@server.list_prompts()
async def list_prompts() -> list[types.Prompt]:
    """Return the workflow-starter prompts (registering this advertises the capability)."""
    return prompts.list_prompts()


@server.get_prompt()
async def get_prompt(name: str, arguments: dict[str, str] | None) -> types.GetPromptResult:
    """Return a prompt's messages with its arguments interpolated."""
    return prompts.get_prompt(name, arguments)
