"""Simulation execution tools. (Phase 3)"""

import asyncio
import time
from typing import Literal

from mcp import types
from pydantic import Field

from ltspice_mcp.errors import ResultError, SimulationError
from ltspice_mcp.lib import now, services
from ltspice_mcp.lib.job_lifecycle import transition
from ltspice_mcp.lib.log_parser import extract_error_context, parse_success_summary
from ltspice_mcp.lib.mcp_logging import mcp_log
from ltspice_mcp.lib.sim_runner import SimulationRunner, generate_job_id
from ltspice_mcp.state import NON_TERMINAL_LIVE_STATUSES, SessionState, SimulationJob
from ltspice_mcp.tools._base import (
    MEAS_ERRORS_SCHEMA,
    ToolInput,
    format_meas_errors,
    format_response,
    registry,
    require_simulator,
    resolve_netlist_path,
    resolve_output_folder,
    text_response,
)

# Constants for timeout behavior
SYNC_TIMEOUT_THRESHOLD = 30.0  # Simulations <= 30s run synchronously by default
HARD_MAX_TIMEOUT = 600.0  # 10 minutes - max for wait=true mode


class RunSimulationInput(ToolInput):
    """Inputs for ltspice_run_simulation."""

    netlist: str = Field(description="Path to the netlist file (.cir, .net, .asc)")
    timeout: float | None = Field(
        default=None,
        description=(
            "Timeout in seconds. Simulations exceeding 30s run asynchronously unless wait=true."
        ),
    )
    wait: bool = Field(
        default=False,
        description="Force synchronous execution. Blocks until completion or hard timeout.",
    )
    format: Literal["json", "text"] | None = Field(
        default=None,
        description="Response format: 'json' for structured data, 'text' for human-readable",
    )


class CheckJobInput(ToolInput):
    """Inputs for ltspice_check_job."""

    job_id: str | None = Field(
        default=None,
        description="Job ID returned by ltspice_run_simulation. Omit to list jobs.",
    )
    status: Literal["running", "queued", "completed", "failed", "timeout", "cancelled", "all"] | None = Field(
        default=None,
        description="Filter by status when listing jobs.",
    )
    format: Literal["json", "text"] | None = Field(
        default=None,
        description="Response format: 'json' for structured data, 'text' for human-readable",
    )


class CancelJobInput(ToolInput):
    """Inputs for ltspice_cancel_job."""

    job_id: str = Field(description="Job ID of the running simulation to cancel")


def _get_or_create_runner(state: SessionState) -> SimulationRunner:
    """Get or create a SimulationRunner via the centralized RunnerManager."""
    default_simulator = state.default_simulator
    if default_simulator is None:
        raise SimulationError("No simulator available. Check server status.")
    return state.runners.get_sim_runner(
        loop=asyncio.get_running_loop(),
        simulator_class=default_simulator,
        output_folder=resolve_output_folder(state),
        max_parallel=state.config.max_parallel_sims,
    )


@registry.tool(
    name="ltspice_run_simulation",
    description=(
        "Run a SPICE simulation on a netlist file. "
        "Automatically runs synchronously for short simulations (<=30s timeout) "
        "or asynchronously for longer ones. Use wait=true to force synchronous execution. "
        "Returns raw/log file paths and simulation summary on completion, "
        "or a job ID for async tracking."
    ),
    input_model=RunSimulationInput,
    annotations=types.ToolAnnotations(
        readOnlyHint=False,
        destructiveHint=False,
        idempotentHint=False,
        openWorldHint=False,
    ),
    profiles=("full", "agentic"),
    output_schema={
        "type": "object",
        "properties": {
            "job_id": {"type": "string"},
            "status": {"type": "string"},
            "sim_type": {"type": "string"},
            "duration": {"type": "number"},
            "step_count": {"type": "integer"},
            "raw_file": {"type": "string"},
            "log_file": {"type": "string"},
            "netlist": {"type": "string"},
            "simulator": {"type": "string"},
            "signals": {"type": "array", "items": {"type": "string"}},
            "warnings": {"type": "array", "items": {"type": "string"}},
            "errors": {"type": "array", "items": {"type": "string"}},
            "meas_errors": MEAS_ERRORS_SCHEMA,
            "error": {"type": "string"},
        },
    },
)
async def handle_run_simulation(args: RunSimulationInput, state: SessionState):
    """Run a SPICE simulation synchronously or asynchronously.

    Automatically chooses sync vs async based on timeout threshold (30s).
    Sync mode blocks until completion, async mode returns job ID immediately.
    """
    # Extract args
    netlist_str = args.netlist
    timeout = args.timeout if args.timeout is not None else state.config.default_timeout
    wait = args.wait
    fmt = args.format

    netlist_path = resolve_netlist_path(netlist_str, state)
    require_simulator(state)
    default_simulator = state.default_simulator
    assert default_simulator is not None  # guaranteed by require_simulator

    # Generate job ID and create job
    job_id = generate_job_id()
    job = SimulationJob(
        job_id=job_id,
        netlist=netlist_path,
        simulator=default_simulator.__name__,
        # "queued" until the runner accepts the work; then the
        # runner transitions to "running" and emits 'started'.
        status="queued",
        started_at=now(),
    )
    # Get SimulationRunner before storing job — if this fails, we don't
    # leave an orphaned "running" job with no task to advance it
    runner = _get_or_create_runner(state)
    state.add_job(job)
    await mcp_log("info", f"Simulation started: {netlist_path.name} ({default_simulator.__name__})")
    job.task = asyncio.create_task(runner.start_simulation(netlist_path, job, state))

    # Decide sync vs async
    # If wait=true: force sync with hard max timeout
    # Elif timeout <= threshold: sync
    # Else: async (return job ID immediately)
    if wait:
        effective_timeout = min(timeout, HARD_MAX_TIMEOUT)
        return await _wait_for_completion(job, effective_timeout, runner, state, fmt)
    elif timeout <= SYNC_TIMEOUT_THRESHOLD:
        return await _wait_for_completion(job, timeout, runner, state, fmt)
    else:
        # Async path - return job ID immediately
        data = {
            "job_id": job_id,
            "status": "running",
            "netlist": str(netlist_path),
            "simulator": default_simulator.__name__,
        }
        return format_response(
            f"Simulation started in background\n"
            f"Job ID: {job_id}\n"
            f"Netlist: {netlist_path}\n"
            f"Simulator: {default_simulator.__name__}\n\n"
            f"Use ltspice_check_job('{job_id}') to check status\n"
            f"Use ltspice_check_job() to see all jobs\n"
            f"Use ltspice_cancel_job('{job_id}') to cancel",
            data,
            fmt,
        )


async def _wait_for_completion(
    job: SimulationJob,
    timeout: float,  # noqa: ASYNC109
    runner: SimulationRunner,
    state: SessionState,
    fmt: str | None = None,
):
    """Wait for simulation to complete (sync mode)."""
    start_time = time.time()

    try:
        # Wait for completion with timeout
        await asyncio.wait_for(job.done_event.wait(), timeout=timeout)
    except TimeoutError:
        # Timeout - this is NOT a simulator error, it's a tool-level kill.
        # Kill the spice process first, then record status=timeout (NOT
        # cancelled) so the user sees the real cause.
        duration = time.time() - start_time
        await runner.kill(job.job_id)
        if job.status == "running":
            transition(job, "timeout", state=state, duration_s=duration)

        # Extract log context if available
        log_excerpt = ""
        if job.log_file and job.log_file.exists():
            log_excerpt = f"\n\nLog excerpt:\n{extract_error_context(job.log_file, max_lines=20)}"

        data = {
            "job_id": job.job_id,
            "status": "timeout",
            "duration": duration,
            "netlist": str(job.netlist),
        }
        return format_response(
            f"Simulation timed out after {duration:.1f}s (killed by server)\n"
            f"Job ID: {job.job_id}\n"
            f"Netlist: {job.netlist}{log_excerpt}",
            data,
            fmt,
        )

    # Simulation completed (success or failure)
    duration = time.time() - start_time

    if job.status == "completed":
        # Parse success summary
        if job.raw_file is None or job.log_file is None:
            raise ResultError(
                f"Job {job.job_id} completed but result files are missing.\n"
                f"raw_file: {job.raw_file}, log_file: {job.log_file}"
            )
        summary = parse_success_summary(job.raw_file, job.log_file, duration)
        suggestions = services.suggestions_from_errors(summary.get("errors"), state.libraries)
        if suggestions:
            summary["suggestions"] = suggestions
        await mcp_log("info", f"Simulation completed: {job.netlist.name} ({duration:.1f}s)")
        return _format_success_response(job.job_id, summary, fmt)
    elif job.status == "failed":
        error_msg = job.error or "Unknown error"
        await mcp_log("error", f"Simulation failed: {job.netlist.name} — {job.error or 'unknown'}")
        data = {"job_id": job.job_id, "status": "failed", "duration": duration, "error": job.error}
        if job.log_file and job.log_file.exists():
            log_excerpt = extract_error_context(job.log_file, max_lines=20)
            error_msg = f"{error_msg}\n\nLog excerpt:\n{log_excerpt}"
            error_msg = services.attach_suggestions_to_failure(
                error_msg, data, job.log_file, state.libraries
            )
        return format_response(
            f"Simulation failed\nJob ID: {job.job_id}\nDuration: {duration:.2f}s\n\n{error_msg}",
            data,
            fmt,
        )
    elif job.status == "cancelled":
        data = {"job_id": job.job_id, "status": "cancelled"}
        return format_response(f"Simulation cancelled\nJob ID: {job.job_id}", data, fmt)
    else:
        # Unexpected status
        data = {"job_id": job.job_id, "status": job.status}
        return format_response(f"Simulation ended with unexpected status: {job.status}", data, fmt)


def _format_success_response(job_id: str, summary: dict, fmt: str | None = None):
    """Format simulation success response with structured data."""
    # Format signal list (first 20 signals)
    signals = summary["trace_names"]
    signal_list = []
    for sig in signals[:20]:
        signal_list.append(f"  - {sig}")
    if len(signals) > 20:
        signal_list.append(f"  ... and {len(signals) - 20} more")

    signal_text = "\n".join(signal_list) if signal_list else "  (none)"

    # Format warnings and errors
    warnings = summary.get("warnings", [])
    errors = summary.get("errors", [])
    meas_errors = summary.get("meas_errors", [])
    diagnostics_text = ""
    if errors:
        diagnostics_text += "\n\nErrors:\n" + "\n".join(f"  {e}" for e in errors)
    if warnings:
        diagnostics_text += "\n\nWarnings:\n" + "\n".join(f"  {w}" for w in warnings)
    meas_lines = format_meas_errors(meas_errors)
    if meas_lines:
        diagnostics_text += "\n\n" + "\n".join(meas_lines)

    text = (
        f"Simulation completed successfully\n"
        f"Job ID: {job_id}\n"
        f"Type: {summary['sim_type']}\n"
        f"Duration: {summary['duration']:.2f}s\n"
        f"Steps: {summary['step_count']}\n"
        f"Raw file: {summary['raw_file']}\n"
        f"Log file: {summary['log_file']}\n\n"
        f"Available signals ({len(signals)}):\n{signal_text}{diagnostics_text}"
    )

    data = {
        "job_id": job_id,
        "status": "completed",
        "sim_type": summary["sim_type"],
        "duration": summary["duration"],
        "step_count": summary["step_count"],
        "raw_file": str(summary["raw_file"]),
        "log_file": str(summary["log_file"]),
        "signals": signals,
        "warnings": warnings,
    }
    if errors:
        data["errors"] = errors
    if meas_errors:
        data["meas_errors"] = meas_errors
    return format_response(text, data, fmt)


@registry.tool(
    name="ltspice_check_job",
    description=(
        "Check status of a simulation job by ID, or list all jobs. "
        "Without job_id: lists active jobs (filter with status param). "
        "With job_id: returns detailed status or completion results."
    ),
    input_model=CheckJobInput,
    annotations=types.ToolAnnotations(
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    ),
    profiles=("full", "agentic"),
    output_schema={
        "type": "object",
        "properties": {
            "job_id": {"type": "string"},
            "status": {"type": "string"},
            "netlist": {"type": "string"},
            "simulator": {"type": "string"},
            "elapsed": {"type": "number"},
            "duration": {"type": "number"},
            "sim_type": {"type": "string"},
            "raw_file": {"type": "string"},
            "log_file": {"type": "string"},
            "signals": {"type": "array", "items": {"type": "string"}},
            "error": {"type": "string"},
            "jobs": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "job_id": {"type": "string"},
                        "status": {"type": "string"},
                        "netlist": {"type": "string"},
                        "started_at": {"type": "string"},
                        "duration": {"type": "number"},
                    },
                },
            },
            "count": {"type": "integer"},
        },
    },
)
async def handle_check_job(args: CheckJobInput, state: SessionState):
    """Check status of a simulation job, or list all jobs."""
    job_id = args.job_id
    fmt = args.format

    # If no job_id provided, list jobs
    if not job_id:
        return _list_jobs(args, state, fmt)

    # Look up specific job
    job = services.resolve_simulation_job(job_id, state)

    # Check status
    if job.status in NON_TERMINAL_LIVE_STATUSES:
        elapsed = (now() - job.started_at).total_seconds()
        data = {
            "job_id": job_id,
            "status": job.status,
            "netlist": str(job.netlist),
            "simulator": job.simulator,
            "elapsed": elapsed,
        }
        if job.status == "queued":
            text = (
                f"Job {job_id} is queued (waiting for a runner slot)\n"
                f"Netlist: {job.netlist}\n"
                f"Simulator: {job.simulator}\n"
                f"Elapsed: {elapsed:.1f}s\n\n"
                f"Use ltspice_cancel_job('{job_id}') to cancel"
            )
        else:
            text = (
                f"Job {job_id} is still running\n"
                f"Netlist: {job.netlist}\n"
                f"Simulator: {job.simulator}\n"
                f"Elapsed: {elapsed:.1f}s\n\n"
                f"Use ltspice_cancel_job('{job_id}') to cancel"
            )
        return format_response(text, data, fmt)
    elif job.status == "completed":
        duration = (job.completed_at - job.started_at).total_seconds() if job.completed_at else 0
        if job.raw_file is None or job.log_file is None:
            raise ResultError(
                f"Job {job_id} completed but result files are missing.\n"
                f"raw_file: {job.raw_file}, log_file: {job.log_file}"
            )
        if not job.raw_file.exists() or not job.log_file.exists():
            raise ResultError(
                f"Job {job_id} completed but result files have been removed.\n"
                f"raw: {job.raw_file.exists()}, log: {job.log_file.exists()}"
            )
        summary = parse_success_summary(job.raw_file, job.log_file, duration)
        suggestions = services.suggestions_from_errors(summary.get("errors"), state.libraries)
        if suggestions:
            summary["suggestions"] = suggestions
        return _format_success_response(job_id, summary, fmt)
    elif job.status == "failed":
        duration = (job.completed_at - job.started_at).total_seconds() if job.completed_at else 0
        error_msg = job.error or "Unknown error"
        data = {"job_id": job_id, "status": "failed", "duration": duration, "error": job.error}
        if job.log_file and job.log_file.exists():
            log_excerpt = extract_error_context(job.log_file, max_lines=20)
            error_msg = f"{error_msg}\n\nLog excerpt:\n{log_excerpt}"
            error_msg = services.attach_suggestions_to_failure(
                error_msg, data, job.log_file, state.libraries
            )
        return format_response(
            f"Simulation failed\nJob ID: {job_id}\nDuration: {duration:.2f}s\n\n{error_msg}",
            data,
            fmt,
        )
    elif job.status == "timeout":
        duration = (job.completed_at - job.started_at).total_seconds() if job.completed_at else 0
        log_excerpt = ""
        if job.log_file and job.log_file.exists():
            log_excerpt = f"\n\nLog excerpt:\n{extract_error_context(job.log_file, max_lines=20)}"

        data = {
            "job_id": job_id,
            "status": "timeout",
            "duration": duration,
            "netlist": str(job.netlist),
        }
        return format_response(
            f"Simulation timed out after {duration:.1f}s (killed by server)\n"
            f"Job ID: {job_id}\n"
            f"Netlist: {job.netlist}{log_excerpt}",
            data,
            fmt,
        )
    elif job.status == "cancelled":
        data = {"job_id": job_id, "status": "cancelled", "netlist": str(job.netlist)}
        return format_response(f"Job {job_id} was cancelled\nNetlist: {job.netlist}", data, fmt)
    else:
        data = {"job_id": job_id, "status": job.status}
        return format_response(f"Job {job_id} has unexpected status: {job.status}", data, fmt)


def _list_jobs(arguments: CheckJobInput, state: SessionState, fmt: str | None = None):
    """List simulation jobs with optional status filter."""
    status_filter = arguments.status

    # Determine which jobs to show
    if status_filter == "all":
        jobs_to_show = list(state.jobs.values())
    elif status_filter:
        jobs_to_show = [job for job in state.jobs.values() if job.status == status_filter]
    else:
        jobs_to_show = [job for job in state.jobs.values() if job.status in NON_TERMINAL_LIVE_STATUSES]

    # Sort by started_at (most recent first)
    jobs_to_show.sort(key=lambda j: j.started_at, reverse=True)

    if not jobs_to_show:
        if status_filter == "all" or not status_filter:
            message = "No active jobs" if not status_filter else "No jobs found"
        else:
            message = f"No jobs with status '{status_filter}'"
        return format_response(message, {"jobs": [], "count": 0}, fmt)

    # Build structured data
    jobs_data = []
    lines = [f"Simulation Jobs ({len(jobs_to_show)}):\n"]
    lines.append(f"{'ID':<28} | {'Status':<10} | {'Netlist':<20} | {'Started':<17} | Duration")
    lines.append("-" * 100)

    for job in jobs_to_show:
        if job.completed_at:
            duration = (job.completed_at - job.started_at).total_seconds()
            duration_str = f"{duration:.1f}s"
        else:
            duration = (now() - job.started_at).total_seconds()
            duration_str = f"{duration:.1f}s (running)"

        started_str = job.started_at.strftime("%Y-%m-%d %H:%M")
        netlist_name = job.netlist.name
        if len(netlist_name) > 20:
            netlist_name = netlist_name[:17] + "..."

        lines.append(
            f"{job.job_id:<28} | {job.status:<10} | {netlist_name:<20} | {started_str:<17} | {duration_str}"
        )
        jobs_data.append(
            {
                "job_id": job.job_id,
                "status": job.status,
                "netlist": str(job.netlist),
                "started_at": job.started_at.isoformat(),
                "duration": duration,
            }
        )

    return format_response("\n".join(lines), {"jobs": jobs_data, "count": len(jobs_data)}, fmt)


@registry.tool(
    name="ltspice_cancel_job",
    description="Cancel a running simulation job. Kills the simulator process and marks the job as cancelled.",
    input_model=CancelJobInput,
    annotations=types.ToolAnnotations(
        readOnlyHint=False,
        destructiveHint=True,
        idempotentHint=True,
        openWorldHint=False,
    ),
    profiles=("full", "agentic"),
)
async def handle_cancel_job(args: CancelJobInput, state: SessionState) -> types.CallToolResult:
    """Cancel a running simulation job.

    Args:
        args: Tool args with job_id
        state: Current session state

    Returns:
        List containing TextContent with cancellation result
    """
    job_id = args.job_id

    # Look up job
    job = services.resolve_simulation_job(job_id, state)

    # Check if job is running
    if job.status not in NON_TERMINAL_LIVE_STATUSES:
        raise SimulationError(f"Job {job_id} is not running (status: {job.status})")

    # Cancel the job
    require_simulator(state)
    runner = _get_or_create_runner(state)
    await runner.cancel(job, state)

    return text_response(f"Job {job_id} cancelled")
