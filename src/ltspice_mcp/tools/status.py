"""Server status and diagnostics tools."""

from pathlib import Path
from typing import Literal

from pydantic import Field

from ltspice_mcp.lib import job_store, recent
from ltspice_mcp.state import SessionState
from ltspice_mcp.tools._base import (
    RO_ANNOTATIONS,
    ToolInput,
    format_response,
    registry,
)


class ServerStatusInput(ToolInput):
    """Inputs for the server status tool."""

    format: Literal["json", "text"] | None = Field(
        default=None,
        description="Response format: 'json' for structured data, 'text' for human-readable",
    )


@registry.tool(
    name="ltspice_server_status",
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

    lines.append(
        f"\nDefault simulator: {state.default_simulator.__name__ if state.default_simulator else 'None'}"
    )

    lines.append("\nConfiguration:")
    lines.append(f"  Tool profile: {state.config.tool_profile}")
    lines.append(f"  Tools exposed: {len(state.tool_defs)}")
    lines.append(f"  Working directory: {state.working_dir}")
    lines.append(f"  Max parallel simulations: {state.config.max_parallel_sims}")
    lines.append(f"  Default timeout: {state.config.default_timeout}s")
    lines.append(f"  Max points returned: {state.config.max_points_returned}")
    lines.append(f"  Log level: {state.config.log_level}")

    allowed_paths_list = [str(p) for p in state.config.allowed_paths]
    lines.append("\nSecurity (Sandbox):")
    lines.append("  Allowed paths:")
    for allowed_path in state.config.allowed_paths:
        lines.append(f"    - {allowed_path}")

    config_file = state.working_dir / "ltspice-mcp.toml"
    if config_file.exists():
        lines.append(f"\n  Config file: {config_file}")
    else:
        lines.append("\n  Config file: Not found (using defaults)")

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
        "tool_profile": state.config.tool_profile,
        "tool_count": len(state.tool_defs),
        "configuration": {
            "working_directory": str(state.working_dir),
            "max_parallel_sims": state.config.max_parallel_sims,
            "default_timeout": state.config.default_timeout,
            "max_points_returned": state.config.max_points_returned,
            "log_level": state.config.log_level,
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
    """Inputs for the ltspice_recent tool."""

    format: Literal["json", "text"] | None = Field(
        default=None,
        description="Response format: 'json' for structured data, 'text' for human-readable",
    )


@registry.tool(
    name="ltspice_recent",
    description=(
        "Call on session start to find circuits the user was last working "
        "with, including jobs that were still running when the server "
        "stopped. Needs no inputs.\n\n"
        "Returns a list of recent circuits, each with its absolute path, "
        "whether the file still exists, last-touched timestamp, total "
        "persisted job count, status_counts (completed/failed/"
        "interrupted/etc.), and the IDs of any interrupted jobs.\n\n"
        "'interrupted' means a simulation was in flight when the server "
        "stopped — recovery path is ltspice_check_job(job_id) to see "
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
                        "status_counts": {"type": "object"},
                        "interrupted_job_ids": {"type": "array", "items": {"type": "string"}},
                    },
                },
            },
            "count": {"type": "integer"},
        },
    },
)
async def handle_recent(args: RecentInput, state: SessionState):
    """List recently-touched circuits with persisted job summaries."""
    del state  # recent.json is user-global; nothing state-scoped is needed
    fmt = args.format
    entries = recent.load(prune_missing=True)
    circuits: list[dict] = []
    for entry in entries:
        raw_path = entry.get("path")
        if not isinstance(raw_path, str):
            continue
        summary = job_store.summarize_circuit(Path(raw_path))
        summary["last_touched"] = entry.get("last_touched")
        circuits.append(summary)

    if not circuits:
        text = (
            "No recent circuits recorded yet. Use any circuit tool to add one "
            "(e.g., ltspice_run_simulation, ltspice_read_circuit)."
        )
    else:
        lines = [f"Recent circuits ({len(circuits)}):", ""]
        for c in circuits:
            counts = c.get("status_counts") or {}
            missing = "" if c.get("exists") else " [missing]"
            parts = [f"{k}={v}" for k, v in sorted(counts.items())]
            counts_str = ", ".join(parts) if parts else "no jobs"
            last = c.get("last_touched") or "unknown"
            lines.append(f"  {c['path']}{missing}")
            lines.append(f"    last touched: {last}  ·  {counts_str}")
            interrupted = c.get("interrupted_job_ids") or []
            if interrupted:
                lines.append(
                    "    interrupted jobs: "
                    + ", ".join(interrupted[:5])
                    + ("" if len(interrupted) <= 5 else f" (+{len(interrupted) - 5} more)")
                )
        text = "\n".join(lines)

    data = {"circuits": circuits, "count": len(circuits)}
    return format_response(text, data, fmt)
