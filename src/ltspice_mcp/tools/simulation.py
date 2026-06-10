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
from ltspice_mcp.state import (
    NON_TERMINAL_LIVE_STATUSES,
    BatchJob,
    SessionState,
    SimulationJob,
)
from ltspice_mcp.tools._base import (
    MEAS_ERRORS_SCHEMA,
    OBSERVATIONS_SCHEMA,
    ToolInput,
    format_meas_errors,
    format_observations,
    format_response,
    registry,
    require_simulator,
    resolve_output_folder,
    resolve_runnable_netlist,
    text_response,
)

# Constants for timeout behavior.
# 30s is a UX boundary, not a correctness one: short enough that a synchronous
# (blocking) call stays within a typical MCP client's tool-call patience, long
# enough that most .op/.ac/small-.tran runs finish inline without forcing the
# caller into the async check_job dance. Runs expected to exceed it return a
# job_id immediately; callers can override per-call with wait=true (bounded by
# HARD_MAX_TIMEOUT).
SYNC_TIMEOUT_THRESHOLD = 30.0
HARD_MAX_TIMEOUT = 600.0  # 10 minutes - max for wait=true mode


# Output-schema fragment shared by ``run_simulation`` and ``check_job`` —
# both surface the post-completion summary built by
# ``build_simulation_summary`` plus the job-tracking fields.
_SIM_RESULT_FIELDS_SCHEMA: dict[str, dict] = {
    "sim_type": {"type": "string"},
    "duration": {"type": "number"},
    "step_count": {"type": "integer"},
    "raw_file": {"type": "string"},
    "log_file": {"type": "string"},
    "signals": {"type": "array", "items": {"type": "string"}},
    "warnings": {"type": "array", "items": {"type": "string"}},
    "errors": {"type": "array", "items": {"type": "string"}},
    "meas_errors": MEAS_ERRORS_SCHEMA,
    "measurements": {"type": "object"},
    "fourier": {"type": "array", "items": {"type": "object"}},
    "range": {"type": "object"},
    "point_count": {"type": "integer"},
    "failed_measurements": {"type": "array", "items": {"type": "string"}},
    "observations": OBSERVATIONS_SCHEMA,
}


class RunSimulationInput(ToolInput):
    """Inputs for run_simulation."""

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
    """Inputs for check_job."""

    job_id: str | None = Field(
        default=None,
        description="Job ID returned by run_simulation. Omit to list jobs.",
    )
    status: (
        Literal["running", "queued", "completed", "failed", "timeout", "cancelled", "all"] | None
    ) = Field(
        default=None,
        description="Filter by status when listing jobs.",
    )
    format: Literal["json", "text"] | None = Field(
        default=None,
        description="Response format: 'json' for structured data, 'text' for human-readable",
    )


class CancelJobInput(ToolInput):
    """Inputs for cancel_job."""

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
    name="run_simulation",
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
            "netlist": {"type": "string"},
            "simulator": {"type": "string"},
            **_SIM_RESULT_FIELDS_SCHEMA,
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

    netlist_path = resolve_runnable_netlist(netlist_str, state)
    require_simulator(state)
    default_simulator = state.default_simulator
    assert default_simulator is not None  # guaranteed by require_simulator

    preflight_warnings = services.ngspice_preflight_warnings(netlist_path, default_simulator)

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
    await mcp_log(
        "info", f"Simulation started: {netlist_path.name} ({default_simulator.__name__})"
    )
    job.task = asyncio.create_task(runner.start_simulation(netlist_path, job, state))

    # Decide sync vs async
    # If wait=true: force sync with hard max timeout
    # Elif timeout <= threshold: sync
    # Else: async (return job ID immediately)
    if wait:
        effective_timeout = min(timeout, HARD_MAX_TIMEOUT)
        return await _wait_for_completion(
            job, effective_timeout, runner, state, fmt, preflight_warnings
        )
    elif timeout <= SYNC_TIMEOUT_THRESHOLD:
        return await _wait_for_completion(job, timeout, runner, state, fmt, preflight_warnings)
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
            f"Use check_job('{job_id}') to check status\n"
            f"Use check_job() to see all jobs\n"
            f"Use cancel_job('{job_id}') to cancel",
            data,
            fmt,
        )


async def _wait_for_completion(
    job: SimulationJob,
    timeout: float,  # noqa: ASYNC109
    runner: SimulationRunner,
    state: SessionState,
    fmt: str | None = None,
    preflight_warnings: list[str] | None = None,
):
    """Wait for simulation to complete (sync mode)."""
    # Monotonic clock for elapsed time: time.time() can run backwards under
    # WSL2 clock skew, producing a negative reported duration.
    start_time = time.monotonic()

    try:
        # Wait for completion with timeout
        await asyncio.wait_for(job.done_event.wait(), timeout=timeout)
    except TimeoutError:
        # Timeout - this is NOT a simulator error, it's a tool-level kill.
        # Record status=timeout BEFORE killing so that when the killed sim's
        # completion callback fires, _handle_completion sees a terminal status
        # and discards the partial raw instead of recording a false success.
        # NON_TERMINAL_LIVE_STATUSES covers a job still "queued" on the
        # concurrency gate as well as a "running" one: both must be marked
        # terminal so the pending start_simulation task self-heals (releases its
        # slot, doesn't launch) when a slot frees, instead of running orphaned.
        if job.status in NON_TERMINAL_LIVE_STATUSES:
            transition(job, "timeout", state=state)
        await runner.kill(job.job_id)
        # Use the post-kill elapsed (same source as check_job) so a
        # downstream consumer reading both endpoints sees a consistent
        # number rather than the user-set timeout limit.
        duration = (
            services.job_duration_seconds(
                job.started_at, job.completed_at, label=f"sim job {job.job_id}"
            )
            or time.monotonic() - start_time
        )

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
    duration = time.monotonic() - start_time

    if job.status == "completed":
        # Parse success summary
        if job.raw_file is None or job.log_file is None:
            raise ResultError(
                f"Job {job.job_id} completed but result files are missing.\n"
                f"raw_file: {job.raw_file}, log_file: {job.log_file}"
            )
        summary = parse_success_summary(
            job.raw_file, job.log_file, duration, dialect=state.raw_dialect, netlist=job.netlist
        )
        if preflight_warnings:
            existing = summary.get("warnings") or []
            summary["warnings"] = preflight_warnings + existing
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
    """Format simulation success response with structured data.

    Summary shape comes from ``parse_success_summary``, which now
    delegates to ``build_simulation_summary``. The new payload includes
    ``range``, ``measurements``, ``fourier``, and ``meas_errors`` on top
    of the legacy ``signals``/``step_count``/``sim_type`` fields.
    """
    # Format signal list (first 20 signals)
    signals = summary.get("signals", [])
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
    measurements = summary.get("measurements", {})
    fourier = summary.get("fourier", [])

    diagnostics_text = ""
    if errors:
        diagnostics_text += "\n\nErrors:\n" + "\n".join(f"  {e}" for e in errors)
    if warnings:
        diagnostics_text += "\n\nWarnings:\n" + "\n".join(f"  {w}" for w in warnings)
    meas_lines = format_meas_errors(meas_errors)
    if meas_lines:
        diagnostics_text += "\n\n" + "\n".join(meas_lines)
    if measurements:
        diagnostics_text += f"\n\nMeasurements: {len(measurements)} parsed"
    if fourier:
        diagnostics_text += f"\n\nFourier: {len(fourier)} signal(s)"

    # Surfaced observations. Relay observations already print above as Errors, so
    # the shared renderer shows only the new facts (unmet requests, extreme
    # values, skipped scans); the full list rides in structuredContent.
    observations = summary.get("observations", [])
    obs_lines = format_observations(observations)
    if obs_lines:
        diagnostics_text += "\n\n" + "\n".join(obs_lines)

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
        "observations": observations,
    }
    # Copy truthy summary fields through to the response. ``point_count``
    # is special-cased to allow 0 (truthy in the schema but falsy in
    # Python) — every other field is "omit when empty".
    for key in (
        "errors",
        "meas_errors",
        "measurements",
        "fourier",
        "range",
        "failed_measurements",
    ):
        if summary.get(key):
            data[key] = summary[key]
    if summary.get("point_count") is not None:
        data["point_count"] = summary["point_count"]
    return format_response(text, data, fmt)


def _resolve_any_job(job_id: str, state: SessionState) -> SimulationJob | BatchJob:
    """Union-store job lookup for check_job / cancel_job.

    These handlers predate ``resolve_job`` and raise ``SimulationError`` for
    unknown ids; preserve that error type (and its server-side hint) while
    sharing the union lookup.
    """
    try:
        return services.resolve_job(job_id, state)
    except ResultError:
        raise SimulationError(f"Job not found: {job_id}") from None


@registry.tool(
    name="check_job",
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
            **_SIM_RESULT_FIELDS_SCHEMA,
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

    # Single-sim and batch jobs share one store; route by type. Batch
    # (sweep/MC) jobs get a concise status here pointing at the richer
    # per-run view in batch_results.
    resolved = _resolve_any_job(job_id, state)
    if isinstance(resolved, BatchJob):
        return _check_batch_job(resolved, fmt)
    job = resolved

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
                f"Use cancel_job('{job_id}') to cancel"
            )
        else:
            text = (
                f"Job {job_id} is still running\n"
                f"Netlist: {job.netlist}\n"
                f"Simulator: {job.simulator}\n"
                f"Elapsed: {elapsed:.1f}s\n\n"
                f"Use cancel_job('{job_id}') to cancel"
            )
        return format_response(text, data, fmt)
    elif job.status == "completed":
        duration = (
            services.job_duration_seconds(
                job.started_at, job.completed_at, label=f"sim job {job.job_id}"
            )
            or 0
        )
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
        summary = parse_success_summary(
            job.raw_file, job.log_file, duration, dialect=state.raw_dialect, netlist=job.netlist
        )
        suggestions = services.suggestions_from_errors(summary.get("errors"), state.libraries)
        if suggestions:
            summary["suggestions"] = suggestions
        return _format_success_response(job_id, summary, fmt)
    elif job.status == "failed":
        duration = (
            services.job_duration_seconds(
                job.started_at, job.completed_at, label=f"sim job {job.job_id}"
            )
            or 0
        )
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
        duration = (
            services.job_duration_seconds(
                job.started_at, job.completed_at, label=f"sim job {job.job_id}"
            )
            or 0
        )
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
        data = _terminal_job_data(job, "cancelled")
        dur = data.get("duration")
        suffix = f" after {dur:.2f}s" if isinstance(dur, float) else ""
        return format_response(
            f"Job {job_id} was cancelled{suffix}\nNetlist: {job.netlist}", data, fmt
        )
    elif job.status == "interrupted":
        # Assigned on restart recovery when the server stopped mid-run and no
        # valid raw survived to promote the job to 'completed'. Surface that
        # plainly instead of falling through to "unexpected status" (the
        # single-sim analogue of the batch interrupted-formatter gap).
        return format_response(
            f"Job {job_id} was interrupted — the server stopped while it was running, "
            f"so results are incomplete; re-run if you need them.\nNetlist: {job.netlist}",
            _terminal_job_data(job, "interrupted"),
            fmt,
        )
    else:
        data = {"job_id": job_id, "status": job.status}
        return format_response(f"Job {job_id} has unexpected status: {job.status}", data, fmt)


def _terminal_job_data(job: SimulationJob, status: str) -> dict:
    """Response ``data`` for a file-less terminal single-sim job (cancelled /
    interrupted): job id, status, netlist, plus best-effort ``duration`` when it
    can be computed. Factors out the build the two branches shared verbatim.
    """
    data: dict = {"job_id": job.job_id, "status": status, "netlist": str(job.netlist)}
    duration = services.job_duration_seconds(
        job.started_at, job.completed_at, label=f"sim job {job.job_id}"
    )
    if duration is not None:
        data["duration"] = duration
    return data


def _check_batch_job(batch_job: BatchJob, fmt: str | None = None):
    """Concise status for a sweep/MC batch job, pointing at batch_results."""
    data = {
        "job_id": batch_job.job_id,
        "job_type": batch_job.job_type,
        "status": batch_job.status,
        "netlist": str(batch_job.netlist),
        "total_runs": batch_job.total_runs,
        "completed_runs": batch_job.completed_runs,
        "failed_runs": batch_job.failed_runs,
        "error": batch_job.error,
    }
    text = (
        f"Batch job {batch_job.job_id} ({batch_job.job_type}): {batch_job.status}\n"
        f"Netlist: {batch_job.netlist}\n"
        f"Runs: {batch_job.completed_runs}/{batch_job.total_runs} completed, "
        f"{batch_job.failed_runs} failed"
    )
    if batch_job.error:
        text += f"\nError: {batch_job.error}"
    text += (
        f"\n\nUse batch_results('{batch_job.job_id}') for per-run data, "
        "or measurement_stats for aggregated .MEAS statistics."
    )
    return format_response(text, data, fmt)


def _list_jobs(arguments: CheckJobInput, state: SessionState, fmt: str | None = None):
    """List simulation jobs (single + batch) with optional status filter."""
    status_filter = arguments.status

    # The union store holds every job (single-run and sweep/MC batch), so
    # check_job is a complete view of "what jobs exist".
    all_jobs: list[SimulationJob | BatchJob] = list(state.all_jobs.values())

    # Determine which jobs to show
    if status_filter == "all":
        jobs_to_show = all_jobs
    elif status_filter:
        jobs_to_show = [job for job in all_jobs if job.status == status_filter]
    else:
        jobs_to_show = [job for job in all_jobs if job.status in NON_TERMINAL_LIVE_STATUSES]

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
            duration = (
                services.job_duration_seconds(
                    job.started_at, job.completed_at, label=f"sim job {job.job_id}"
                )
                or 0.0
            )
            duration_str = f"{duration:.1f}s"
        else:
            duration = max(0.0, (now() - job.started_at).total_seconds())
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
                "job_type": getattr(job, "job_type", "single"),
                "status": job.status,
                "netlist": str(job.netlist),
                "started_at": job.started_at.isoformat(),
                "duration": duration,
            }
        )

    return format_response("\n".join(lines), {"jobs": jobs_data, "count": len(jobs_data)}, fmt)


@registry.tool(
    name="cancel_job",
    description="Cancel a running simulation job (single run, or a sweep/Monte-Carlo batch). Kills the simulator process(es) and marks the job as cancelled.",
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

    job = _resolve_any_job(job_id, state)

    # Check if job is running
    if job.status not in NON_TERMINAL_LIVE_STATUSES:
        raise SimulationError(f"Job {job_id} is not running (status: {job.status})")

    # Cancel via the runner that owns the job. A batch job's cancel event and
    # live-process map live on the SweepRunner/MonteCarloRunner instance that
    # launched it, so route by job type rather than assuming a single-sim runner.
    require_simulator(state)
    if isinstance(job, BatchJob):
        batch_runner = (
            state.runners.get_existing_mc_runner()
            if job.job_type == "montecarlo"
            else state.runners.get_existing_sweep_runner()
        )
        if batch_runner is None:
            raise SimulationError(
                f"Job {job_id} is marked running but its {job.job_type} runner is no "
                "longer live (server restarted?), so there is no process to cancel."
            )
        await batch_runner.cancel(job, state)
    else:
        await _get_or_create_runner(state).cancel(job, state)

    return text_response(f"Job {job_id} cancelled")
