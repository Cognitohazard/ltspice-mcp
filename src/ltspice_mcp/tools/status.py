"""Server status and diagnostics tools."""

import asyncio
from typing import Any, Literal

from pydantic import Field

from ltspice_mcp.lib import services
from ltspice_mcp.lib.simulator import current_ngbehavior, no_simulator_message
from ltspice_mcp.state import SessionState
from ltspice_mcp.tools._base import (
    FORMAT_DESCRIPTION,
    HINT_SCHEMA,
    RO_ANNOTATIONS,
    ToolInput,
    format_response,
    registry,
)


class ServerStatusInput(ToolInput):
    """Inputs for the server status tool."""

    format: Literal["json", "text"] | None = Field(
        default=None,
        description=FORMAT_DESCRIPTION,
    )


@registry.tool(
    name="server_status",
    description=(
        "Get comprehensive server status including detected simulators, "
        "configuration settings, security sandbox paths, and runtime state. "
        "Use this to check what capabilities are available before attempting operations."
    ),
    input_model=ServerStatusInput,
    annotations=RO_ANNOTATIONS,
    profiles=("full", "agentic"),
    output_schema={
        "type": "object",
        "properties": {
            "simulators": {"type": "object"},
            "default_simulator": {"type": ["string", "null"]},
            "requested_simulator": {"type": ["string", "null"]},
            "simulator_select": {"type": "string"},
            "diagnostics": {"type": "array", "items": {"type": "string"}},
            "tool_profile": {"type": "string"},
            "tool_count": {"type": "integer"},
            "configuration": {"type": "object"},
            "allowed_paths": {"type": "array", "items": {"type": "string"}},
            "runtime": {"type": "object"},
        },
    },
)
async def handle_server_status(args: ServerStatusInput, state: SessionState):
    """Get comprehensive server status information."""
    fmt = args.format

    # config_path is the file config.load() actually resolved (honors
    # LTSPICE_MCP_CONFIG), not a working_dir guess — telling the agent the
    # wrong path is worse than no path.
    config_file = state.config.config_path
    config_file_exists = config_file.exists()

    # The active simulator is config, not a per-call argument: surface the knob
    # so an agent that needs ngspice (or any non-default) knows where to set it
    # instead of guessing. Takes effect on server restart.
    simulator_select = (
        f"Set the active/default simulator in {config_file} under [simulator] default "
        '(e.g. default = "ngspice"), or env LTSPICE_MCP_SIMULATOR=ngspice '
        "(set LTSPICE_MCP_CONFIG to load the config file from another path); "
        "restrict which are exposed with [simulator] enabled. "
        "Changes take effect when the server restarts."
    )

    lines = ["=== LTSpice MCP Server Status ===\n"]

    simulators_data = {}

    lines.append("Simulators:")
    if state.available_simulators:
        for name, cls in state.available_simulators.items():
            is_default = cls == state.default_simulator
            default_marker = " (default)" if is_default else ""
            lines.append(f"  - {name}: available{default_marker}")
            sim_info: dict = {"available": True, "default": is_default}
            try:
                if hasattr(cls, "spice_exe"):
                    exe_path = (
                        cls.spice_exe[0] if isinstance(cls.spice_exe, list) else cls.spice_exe
                    )
                    lines.append(f"    Executable: {exe_path}")
                    sim_info["executable"] = str(exe_path)
            except Exception:
                pass
            simulators_data[name] = sim_info
    else:
        lines.append("  No simulators detected (server running in degraded mode)")
        lines.append(f"  {no_simulator_message()}")

    lines.append(
        f"\nDefault simulator: {state.default_simulator.__name__ if state.default_simulator else 'None'}"
    )
    if state.config.simulator:
        lines.append(f"Requested simulator: {state.config.simulator}")
    lines.append(f"To switch: {simulator_select}")

    if state.diagnostics:
        lines.append("\n⚠ Startup diagnostics:")
        for diag in state.diagnostics:
            lines.append(f"  - {diag}")

    enabled_sims = state.config.enabled_simulators
    lines.append("\nConfiguration:")
    lines.append(f"  Enabled simulators: {enabled_sims if enabled_sims else 'auto-detect all'}")
    lines.append(f"  Tool profile: {state.config.tool_profile}")
    lines.append(f"  Tools exposed: {len(state.tool_defs)}")
    lines.append(f"  Working directory: {state.working_dir}")
    lines.append(f"  Max parallel simulations: {state.config.max_parallel_sims}")
    lines.append(f"  Default timeout: {state.config.default_timeout}s")
    lines.append(f"  Max points returned: {state.config.max_points_returned}")
    lines.append(f"  Log level: {state.config.log_level}")
    if "ngspice" in state.available_simulators:
        # The effective compatibility mode ngspice runs under. It changes how
        # decks parse (e.g. sectioned `.lib file section` handling), so it must
        # be observable for reproducibility — a CLI ngspice run without it can
        # fail differently than the server's.
        lines.append(f"  ngspice behavior mode (ngbehavior): {current_ngbehavior() or 'default'}")
    lines.append(f"  Job persistence: {'on' if state.config.persist_jobs else 'off'}")
    if not state.config.persist_jobs:
        # Distinguish "nothing run yet" from "persistence off" — otherwise an
        # empty `recent` looks like a fresh session, not a config choice.
        lines.append("    (jobs are in-memory only — lost on restart; recent/preload disabled)")

    allowed_paths_list = [str(p) for p in state.config.allowed_paths]
    lines.append("\nSecurity (Sandbox):")
    lines.append("  Allowed paths:")
    for allowed_path in state.config.allowed_paths:
        lines.append(f"    - {allowed_path}")

    if config_file_exists:
        lines.append(f"\n  Config file: {config_file}")
    else:
        lines.append(f"\n  Config file: {config_file} (not found, using defaults)")

    # ``state.jobs`` retains the full lifecycle (running/completed/failed/etc.)
    # so we can answer ``check_job`` lookups after the fact. The status field
    # here is meant for "what's the simulator doing right now" — only the
    # in-flight statuses count.
    _active_statuses = {"queued", "running"}
    active_count = sum(1 for j in state.jobs.values() if j.status in _active_statuses)
    tracked_count = len(state.jobs)

    lines.append("\nRuntime State:")
    lines.append(f"  Active jobs: {active_count} (running)")
    lines.append(f"  Tracked jobs: {tracked_count} (all statuses)")
    lines.append(f"  Cached editors: {len(state.editors)}")
    lines.append(f"  Cached results: {len(state.results)}")
    lines.append(f"  Loaded libraries: {len(state.libraries)}")

    data = {
        "simulators": simulators_data,
        "default_simulator": state.default_simulator.__name__ if state.default_simulator else None,
        "requested_simulator": state.config.simulator,
        "diagnostics": list(state.diagnostics),
        "tool_profile": state.config.tool_profile,
        "tool_count": len(state.tool_defs),
        "simulator_select": simulator_select,
        "configuration": {
            "working_directory": str(state.working_dir),
            "config_file": str(config_file),
            "config_file_exists": config_file_exists,
            "enabled_simulators": list(state.config.enabled_simulators),
            "max_parallel_sims": state.config.max_parallel_sims,
            "default_timeout": state.config.default_timeout,
            "max_points_returned": state.config.max_points_returned,
            "log_level": state.config.log_level,
            "ngbehavior": current_ngbehavior()
            if "ngspice" in state.available_simulators
            else None,
            "persist_jobs": state.config.persist_jobs,
            "preload_recent_count": state.config.preload_recent_count,
        },
        "allowed_paths": allowed_paths_list,
        "runtime": {
            "active_jobs": active_count,
            "tracked_jobs": tracked_count,
            "cached_editors": len(state.editors),
            "cached_results": len(state.results),
            "loaded_libraries": len(state.libraries),
        },
    }
    return format_response("\n".join(lines), data, fmt)


class RecentInput(ToolInput):
    """Inputs for the recent tool."""

    format: Literal["json", "text"] | None = Field(
        default=None,
        description=FORMAT_DESCRIPTION,
    )


@registry.tool(
    name="recent",
    description=(
        "Call on session start to find circuits the user was last working "
        "with, including jobs that were still running when the server "
        "stopped. Needs no inputs.\n\n"
        "Returns a list of recent circuits, each with its absolute path, "
        "whether the file still exists, last-touched timestamp, total "
        "persisted job count, status_counts (completed/failed/"
        "interrupted/etc.), and the IDs of any interrupted jobs.\n\n"
        "'interrupted' means a simulation was in flight when the server "
        "stopped — recovery path is check_job(job_id) to see "
        "whether results are recoverable or the run needs to be "
        "re-kicked. Does NOT start or cancel anything; purely read-only."
    ),
    input_model=RecentInput,
    annotations=RO_ANNOTATIONS,
    profiles=("full", "agentic"),
    output_schema={
        "type": "object",
        "properties": {
            "circuits": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "exists": {"type": "boolean"},
                        "last_touched": {"type": ["string", "null"]},
                        "total_jobs": {"type": "integer"},
                        "total_runs": {"type": "integer"},
                        "status_counts": {"type": "object"},
                        "interrupted_job_ids": {"type": "array", "items": {"type": "string"}},
                    },
                },
            },
            "count": {"type": "integer"},
            "hint": HINT_SCHEMA,
        },
    },
)
async def handle_recent(args: RecentInput, state: SessionState):
    """List recently-touched circuits with persisted job summaries."""
    fmt = args.format

    circuits = await asyncio.to_thread(services.collect_recent_circuits)
    data: dict[str, Any] = {"circuits": circuits, "count": len(circuits)}

    if not circuits:
        # Example tools are membership-checked against the live dispatch table
        # (read_circuit is hidden from the agentic profile) so the referral
        # can't drift from what this session actually exposes.
        example = (
            "run_simulation, read_circuit"
            if "read_circuit" in state.tool_dispatch
            else "run_simulation, validate_netlist"
        )
        text = (
            f"No recent circuits recorded yet. Use any circuit tool to add one (e.g., {example})."
        )
        # Mirrored — see format_response's self-sufficiency contract.
        data["hint"] = text
    else:
        lines = [f"Recent circuits ({len(circuits)}):", ""]
        for c in circuits:
            counts = c.get("status_counts") or {}
            missing = "" if c.get("exists") else " [missing]"
            parts = [f"{k}={v}" for k, v in sorted(counts.items())]
            counts_str = ", ".join(parts) if parts else "no jobs"
            last = c.get("last_touched") or "unknown"
            jobs_n = c.get("total_jobs") or 0
            runs_n = c.get("total_runs") or 0
            run_str = f", {runs_n} runs" if runs_n != jobs_n else ""
            lines.append(f"  {c['path']}{missing}")
            lines.append(f"    last touched: {last}  ·  {counts_str}{run_str}")
            interrupted = c.get("interrupted_job_ids") or []
            if interrupted:
                lines.append(
                    "    interrupted jobs: "
                    + ", ".join(interrupted[:5])
                    + ("" if len(interrupted) <= 5 else f" (+{len(interrupted) - 5} more)")
                )
        text = "\n".join(lines)

    return format_response(text, data, fmt)
