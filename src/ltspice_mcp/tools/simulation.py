"""Simulation execution tools. (Phase 3)"""

import asyncio
import logging
import time
from datetime import datetime
from pathlib import Path
from mcp import types

from ltspice_mcp.lib.log_parser import extract_error_context, parse_success_summary
from ltspice_mcp.lib.sim_runner import SimulationRunner, generate_job_id
from ltspice_mcp.state import SessionState, SimulationJob
from ltspice_mcp.tools._base import (
    require_simulator, resolve_netlist_path, resolve_output_folder, text_response,
)

logger = logging.getLogger(__name__)

# Constants for timeout behavior
SYNC_TIMEOUT_THRESHOLD = 30.0  # Simulations <= 30s run synchronously by default
HARD_MAX_TIMEOUT = 600.0  # 10 minutes - max for wait=true mode


def _get_or_create_runner(state: SessionState) -> SimulationRunner:
    """Get or create a SimulationRunner via the centralized RunnerManager."""
    assert state.default_simulator is not None
    return state.runners.get_sim_runner(
        loop=asyncio.get_running_loop(),
        simulator_class=state.default_simulator,
        output_folder=resolve_output_folder(state),
        max_parallel=state.config.max_parallel_sims,
    )


async def handle_run_simulation(arguments: dict, state: SessionState) -> list[types.TextContent]:
    """Run a SPICE simulation synchronously or asynchronously.

    Automatically chooses sync vs async based on timeout threshold (30s).
    Sync mode blocks until completion, async mode returns job ID immediately.

    Args:
        arguments: Tool arguments with netlist, timeout, wait
        state: Current session state

    Returns:
        List containing a single TextContent with results or job ID
    """
    # Extract arguments
    netlist_str = arguments["netlist"]
    timeout = arguments.get("timeout", state.config.default_timeout)
    wait = arguments.get("wait", False)

    netlist_path = resolve_netlist_path(netlist_str, state)
    require_simulator(state)

    # Generate job ID and create job
    job_id = generate_job_id()
    job = SimulationJob(
        job_id=job_id,
        netlist=netlist_path,
        simulator=state.default_simulator.__name__,
        status="running",
        started_at=datetime.now(),
    )
    # Get SimulationRunner before storing job — if this fails, we don't
    # leave an orphaned "running" job with no task to advance it
    runner = _get_or_create_runner(state)
    state.jobs[job_id] = job
    asyncio.create_task(runner.start_simulation(netlist_path, job, state))

    # Decide sync vs async
    # If wait=true: force sync with hard max timeout
    # Elif timeout <= threshold: sync
    # Else: async (return job ID immediately)
    if wait:
        effective_timeout = min(timeout, HARD_MAX_TIMEOUT)
        return await _wait_for_completion(job, effective_timeout, runner)
    elif timeout <= SYNC_TIMEOUT_THRESHOLD:
        return await _wait_for_completion(job, timeout, runner)
    else:
        # Async path - return job ID immediately
        return text_response(
            f"Simulation started in background\n"
            f"Job ID: {job_id}\n"
            f"Netlist: {netlist_path}\n"
            f"Simulator: {state.default_simulator.__name__}\n\n"
            f"Use ltspice_check_job('{job_id}') to check status\n"
            f"Use ltspice_check_job() to see all jobs\n"
            f"Use ltspice_cancel_job('{job_id}') to cancel"
        )


async def _wait_for_completion(
    job: SimulationJob, timeout: float, runner: SimulationRunner
) -> list[types.TextContent]:
    """Wait for simulation to complete (sync mode).

    Args:
        job: SimulationJob to wait for
        timeout: Timeout in seconds
        runner: SimulationRunner for cancellation on timeout

    Returns:
        List containing TextContent with results or error
    """
    start_time = time.time()

    try:
        # Wait for completion with timeout
        await asyncio.wait_for(job.done_event.wait(), timeout=timeout)
    except asyncio.TimeoutError:
        # Timeout - this is NOT a simulator error, it's a tool-level kill
        duration = time.time() - start_time
        job.status = "timeout"
        job.completed_at = datetime.now()

        # Cancel the simulation
        await runner.cancel(job)

        # Extract log context if available
        log_excerpt = ""
        if job.log_file and job.log_file.exists():
            log_excerpt = f"\n\nLog excerpt:\n{extract_error_context(job.log_file, max_lines=20)}"

        return text_response(
            f"Simulation timed out after {duration:.1f}s (killed by server)\n"
            f"Job ID: {job.job_id}\n"
            f"Netlist: {job.netlist}{log_excerpt}"
        )

    # Simulation completed (success or failure)
    duration = time.time() - start_time

    if job.status == "completed":
        # Parse success summary
        assert job.raw_file is not None
        assert job.log_file is not None
        summary = parse_success_summary(job.raw_file, job.log_file, duration)
        return _format_success_response(job.job_id, summary)
    elif job.status == "failed":
        # Extract error context
        error_msg = job.error or "Unknown error"
        if job.log_file and job.log_file.exists():
            log_excerpt = extract_error_context(job.log_file, max_lines=20)
            error_msg = f"{error_msg}\n\nLog excerpt:\n{log_excerpt}"

        return text_response(
            f"Simulation failed\n"
            f"Job ID: {job.job_id}\n"
            f"Duration: {duration:.2f}s\n\n"
            f"{error_msg}"
        )
    elif job.status == "cancelled":
        return text_response(f"Simulation cancelled\nJob ID: {job.job_id}")
    else:
        # Unexpected status
        return text_response(f"Simulation ended with unexpected status: {job.status}")


def _format_success_response(job_id: str, summary: dict) -> list[types.TextContent]:
    """Format simulation success response.

    Args:
        job_id: Job ID
        summary: Parsed summary dict from parse_success_summary

    Returns:
        Single-item TextContent list with formatted success message
    """
    # Format signal list (first 20 signals)
    signals = summary["trace_names"]
    signal_list = []
    for sig in signals[:20]:
        signal_list.append(f"  - {sig}")
    if len(signals) > 20:
        signal_list.append(f"  ... and {len(signals) - 20} more")

    signal_text = "\n".join(signal_list) if signal_list else "  (none)"

    # Format warnings
    warnings = summary.get("warnings", [])
    warning_text = ""
    if warnings:
        warning_text = "\n\nWarnings:\n" + "\n".join(f"  {w}" for w in warnings)

    text = (
        f"Simulation completed successfully\n"
        f"Job ID: {job_id}\n"
        f"Type: {summary['sim_type']}\n"
        f"Duration: {summary['duration']:.2f}s\n"
        f"Steps: {summary['step_count']}\n"
        f"Raw file: {summary['raw_file']}\n"
        f"Log file: {summary['log_file']}\n\n"
        f"Available signals ({len(signals)}):\n{signal_text}{warning_text}"
    )

    return text_response(text)


async def handle_check_job(arguments: dict, state: SessionState) -> list[types.TextContent]:
    """Check status of a simulation job, or list all jobs.

    Without job_id: lists active jobs (filter with status param).
    With job_id: returns detailed status or completion results.

    Args:
        arguments: Tool arguments with optional job_id and status
        state: Current session state

    Returns:
        List containing TextContent with job status, results, or job list
    """
    job_id = arguments.get("job_id")

    # If no job_id provided, list jobs
    if not job_id:
        return _list_jobs(arguments, state)

    # Look up specific job
    job = state.jobs.get(job_id)
    if not job:
        return text_response(f"Job not found: {job_id}\n\nUse ltspice_check_job() with no arguments to see all jobs")

    # Check status
    if job.status == "running":
        # Calculate elapsed time
        elapsed = (datetime.now() - job.started_at).total_seconds()
        return text_response(
            f"Job {job_id} is still running\n"
            f"Netlist: {job.netlist}\n"
            f"Simulator: {job.simulator}\n"
            f"Elapsed: {elapsed:.1f}s\n\n"
            f"Use ltspice_cancel_job('{job_id}') to cancel"
        )
    elif job.status == "completed":
        # Return same format as sync completion
        duration = (job.completed_at - job.started_at).total_seconds() if job.completed_at else 0
        if job.raw_file is None or job.log_file is None:
            return text_response(
                f"Job {job_id} completed but result files are missing.\n"
                f"raw_file: {job.raw_file}, log_file: {job.log_file}"
            )
        if not job.raw_file.exists() or not job.log_file.exists():
            return text_response(
                f"Job {job_id} completed but result files have been removed.\n"
                f"raw: {job.raw_file.exists()}, log: {job.log_file.exists()}"
            )
        summary = parse_success_summary(job.raw_file, job.log_file, duration)
        return _format_success_response(job_id, summary)
    elif job.status == "failed":
        # Return error with log excerpt
        duration = (job.completed_at - job.started_at).total_seconds() if job.completed_at else 0
        error_msg = job.error or "Unknown error"
        if job.log_file and job.log_file.exists():
            log_excerpt = extract_error_context(job.log_file, max_lines=20)
            error_msg = f"{error_msg}\n\nLog excerpt:\n{log_excerpt}"

        return text_response(
            f"Simulation failed\n"
            f"Job ID: {job_id}\n"
            f"Duration: {duration:.2f}s\n\n"
            f"{error_msg}"
        )
    elif job.status == "timeout":
        # Return timeout message with log excerpt
        duration = (job.completed_at - job.started_at).total_seconds() if job.completed_at else 0
        log_excerpt = ""
        if job.log_file and job.log_file.exists():
            log_excerpt = f"\n\nLog excerpt:\n{extract_error_context(job.log_file, max_lines=20)}"

        return text_response(
            f"Simulation timed out after {duration:.1f}s (killed by server)\n"
            f"Job ID: {job_id}\n"
            f"Netlist: {job.netlist}{log_excerpt}"
        )
    elif job.status == "cancelled":
        return text_response(f"Job {job_id} was cancelled\nNetlist: {job.netlist}")
    else:
        return text_response(f"Job {job_id} has unexpected status: {job.status}")


def _list_jobs(arguments: dict, state: SessionState) -> list[types.TextContent]:
    """List simulation jobs with optional status filter.

    Shows active jobs (running/queued) by default.
    """
    status_filter = arguments.get("status")

    # Determine which jobs to show
    if status_filter == "all":
        jobs_to_show = list(state.jobs.values())
    elif status_filter:
        jobs_to_show = [job for job in state.jobs.values() if job.status == status_filter]
    else:
        # Default: show active jobs only (running/queued)
        jobs_to_show = [
            job for job in state.jobs.values() if job.status in ("running", "queued")
        ]

    # Sort by started_at (most recent first)
    jobs_to_show.sort(key=lambda j: j.started_at, reverse=True)

    # Format response
    if not jobs_to_show:
        if status_filter == "all" or not status_filter:
            message = "No active jobs" if not status_filter else "No jobs found"
        else:
            message = f"No jobs with status '{status_filter}'"
        return text_response(message)

    # Build job table
    lines = [f"Simulation Jobs ({len(jobs_to_show)}):\n"]
    lines.append(
        f"{'ID':<28} | {'Status':<10} | {'Netlist':<20} | {'Started':<17} | Duration"
    )
    lines.append("-" * 100)

    for job in jobs_to_show:
        # Format duration/elapsed
        if job.completed_at:
            duration = (job.completed_at - job.started_at).total_seconds()
            duration_str = f"{duration:.1f}s"
        else:
            elapsed = (datetime.now() - job.started_at).total_seconds()
            duration_str = f"{elapsed:.1f}s (running)"

        started_str = job.started_at.strftime("%Y-%m-%d %H:%M")

        netlist_name = job.netlist.name
        if len(netlist_name) > 20:
            netlist_name = netlist_name[:17] + "..."

        lines.append(
            f"{job.job_id:<28} | {job.status:<10} | {netlist_name:<20} | {started_str:<17} | {duration_str}"
        )

    return text_response("\n".join(lines))


async def handle_cancel_job(arguments: dict, state: SessionState) -> list[types.TextContent]:
    """Cancel a running simulation job.

    Args:
        arguments: Tool arguments with job_id
        state: Current session state

    Returns:
        List containing TextContent with cancellation result
    """
    job_id = arguments["job_id"]

    # Look up job
    job = state.jobs.get(job_id)
    if not job:
        return text_response(f"Job not found: {job_id}\n\nUse ltspice_check_job() with no arguments to see all jobs")

    # Check if job is running
    if job.status not in ("running", "queued"):
        return text_response(f"Job {job_id} is not running (status: {job.status})")

    # Cancel the job
    require_simulator(state)
    runner = _get_or_create_runner(state)
    await runner.cancel(job)

    return text_response(f"Job {job_id} cancelled")


# Tool definitions
TOOL_DEFS: list[types.Tool] = [
    types.Tool(
        name="ltspice_run_simulation",
        description=(
            "Run a SPICE simulation on a netlist file. "
            "Automatically runs synchronously for short simulations (<=30s timeout) "
            "or asynchronously for longer ones. Use wait=true to force synchronous execution. "
            "Returns raw/log file paths and simulation summary on completion, "
            "or a job ID for async tracking."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "netlist": {
                    "type": "string",
                    "description": "Path to the netlist file (.cir, .net, .asc)",
                },
                "timeout": {
                    "type": "number",
                    "description": "Timeout in seconds. Default: 30. Simulations exceeding this are automatically run asynchronously unless wait=true.",
                },
                "wait": {
                    "type": "boolean",
                    "description": "Force synchronous execution. Blocks until completion or hard timeout (600s max). Default: false.",
                },
            },
            "required": ["netlist"],
        },
        annotations=types.ToolAnnotations(
            readOnlyHint=False,
            destructiveHint=False,
            idempotentHint=False,
            openWorldHint=False,
        ),
    ),
    types.Tool(
        name="ltspice_check_job",
        description=(
            "Check status of a simulation job by ID, or list all jobs. "
            "Without job_id: lists active jobs (filter with status param). "
            "With job_id: returns detailed status or completion results."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "job_id": {
                    "type": "string",
                    "description": "Job ID returned by ltspice_run_simulation. Omit to list all jobs.",
                },
                "status": {
                    "type": "string",
                    "description": "Filter by status when listing jobs (no job_id). Default: shows active jobs only.",
                    "enum": ["running", "queued", "completed", "failed", "timeout", "cancelled", "all"],
                },
            },
            "required": [],
        },
        annotations=types.ToolAnnotations(
            readOnlyHint=True,
            destructiveHint=False,
            idempotentHint=True,
            openWorldHint=False,
        ),
    ),
    types.Tool(
        name="ltspice_cancel_job",
        description="Cancel a running simulation job. Kills the simulator process and marks the job as cancelled.",
        inputSchema={
            "type": "object",
            "properties": {
                "job_id": {
                    "type": "string",
                    "description": "Job ID of the running simulation to cancel",
                }
            },
            "required": ["job_id"],
        },
        annotations=types.ToolAnnotations(
            readOnlyHint=False,
            destructiveHint=True,
            idempotentHint=True,
            openWorldHint=False,
        ),
    ),
]

TOOL_HANDLERS: dict[str, object] = {
    "ltspice_run_simulation": handle_run_simulation,
    "ltspice_check_job": handle_check_job,
    "ltspice_cancel_job": handle_cancel_job,
}
