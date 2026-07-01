"""Simulation execution tools. (Phase 3)"""

import asyncio
import logging
import time
from pathlib import Path
from typing import Literal

from mcp import types
from pydantic import Field

from ltspice_mcp.errors import ResultError, SimulationError
from ltspice_mcp.lib import now, services
from ltspice_mcp.lib.job_lifecycle import transition
from ltspice_mcp.lib.log_parser import extract_error_context, parse_success_summary
from ltspice_mcp.lib.mcp_logging import mcp_log
from ltspice_mcp.lib.runner_base import discard_logopinfo_netlist
from ltspice_mcp.lib.sim_runner import SimulationRunner, generate_job_id
from ltspice_mcp.lib.simulator import current_ngbehavior, is_ngspice, no_simulator_message
from ltspice_mcp.state import (
    NON_TERMINAL_LIVE_STATUSES,
    BatchJob,
    SessionState,
    SimulationJob,
)
from ltspice_mcp.tools._base import (
    MEAS_ERRORS_SCHEMA,
    MEASUREMENTS_SCHEMA,
    OBSERVATIONS_SCHEMA,
    ToolInput,
    format_meas_errors,
    format_observations,
    format_response,
    inject_logopinfo,
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

# Appended to both timeout messages (the sync wait path and the async check_job
# path). A timeout is a tool-set limit, not a simulator failure, so name the
# levers to raise it — otherwise the agent reads "timed out" as a dead end and
# loops. One constant so the two sites can't drift.
TIMEOUT_HINT = (
    "\n\nThis is the configured time limit, not a simulator error. To allow more "
    "time, pass run_simulation(timeout=<seconds>) for this run, or raise the "
    "default via [simulation] timeout in the config file or LTSPICE_MCP_TIMEOUT "
    "(restart required). server_status shows the current default."
)

logger = logging.getLogger(__name__)


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
    "measurements": MEASUREMENTS_SCHEMA,
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
            "Timeout in seconds (defaults to the server's configured default, 300s). "
            "Simulations exceeding 30s run asynchronously unless wait=true. With "
            "wait=true the effective limit is min(this timeout, 600s): 600s is a hard "
            "ceiling, not a floor — pass a larger timeout to use the full 600s."
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
        Literal[
            "running",
            "queued",
            "completed",
            "failed",
            "timeout",
            "cancelled",
            "interrupted",
            "all",
        ]
        | None
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


async def _get_or_create_runner(
    state: SessionState, netlist_path: Path | None = None
) -> SimulationRunner:
    """Get or create a SimulationRunner via the centralized RunnerManager."""
    default_simulator = state.default_simulator
    if default_simulator is None:
        raise SimulationError(no_simulator_message())
    return state.runners.get_sim_runner(
        loop=asyncio.get_running_loop(),
        simulator_class=default_simulator,
        output_folder=await resolve_output_folder(state, netlist_path),
        max_parallel=state.config.max_parallel_sims,
    )


@registry.tool(
    name="run_simulation",
    description=(
        "Run a SPICE simulation on a netlist file. Sets the right batch flags, "
        "handles the ngspice headerless-raw dialect, routes the raw/log "
        "artifacts, and parses the results — so you never hand-parse a rawfile. "
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

    # On LTspice .op runs, add '.options logopinfo' (in a per-job sibling file)
    # so the log carries each device's small-signal op point for operating_point
    # to read back by name. No-op for ngspice / non-.op decks. The job_id-stamped
    # name keeps concurrent/queued runs of the same netlist from clobbering each
    # other; start_simulation deletes the copy once spicelib has staged the run.
    # job.netlist stays the user's original path; only the simulator reads the copy.
    run_path = inject_logopinfo(netlist_path, default_simulator, job_id)

    job = SimulationJob(
        job_id=job_id,
        netlist=netlist_path,
        simulator=default_simulator.__name__,
        # "queued" until the runner accepts the work; then the
        # runner transitions to "running" and emits 'started'.
        status="queued",
        started_at=now(),
    )
    # Runner first, then register + create_task with no await between —
    # submit-ordering rule, see the concurrency contract in tools/_base.py.
    # If anything raises before start_simulation arms its own cleanup (e.g.
    # _get_or_create_runner failing on WSL cmd.exe interop or a read-only dir),
    # delete the generated logopinfo sibling so the error path leaves no orphan.
    started = False
    try:
        runner = await _get_or_create_runner(state, netlist_path)
        state.add_job(job)
        job.task = asyncio.create_task(runner.start_simulation(run_path, job, state))
        started = True
    finally:
        if not started:
            discard_logopinfo_netlist(run_path)
    await mcp_log(
        "info", f"Simulation started: {netlist_path.name} ({default_simulator.__name__})"
    )

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
        # Async path — return the job ID immediately, but arm a deadline
        # watchdog first: the sync branches enforce their deadline via
        # wait_for below, and without a watchdog an async job's timeout
        # (including the config default) was accepted and never enforced.
        _arm_timeout_watchdog(job, timeout, runner, state)
        # Let the submission task advance to its first suspension point so
        # the reported status reflects reality: "running" when a slot was
        # free, "queued" when the job is waiting on the concurrency cap.
        await asyncio.sleep(0)
        data = {
            "job_id": job_id,
            "status": job.status,
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


_timeout_watchdogs: set[asyncio.Task[None]] = set()
"""Strong refs to per-job deadline watchdogs — ``create_task`` results are
garbage-collectable while pending; each task discards itself when done."""


def _arm_timeout_watchdog(
    job: SimulationJob, timeout: float, runner: SimulationRunner, state: SessionState
) -> None:
    """Enforce ``timeout`` on an async job that no request is awaiting."""
    task = asyncio.create_task(_enforce_async_deadline(job, timeout, runner, state))
    _timeout_watchdogs.add(task)
    task.add_done_callback(_timeout_watchdogs.discard)


async def _timeout_job(job: SimulationJob, runner: SimulationRunner, state: SessionState) -> None:
    """Mark an overdue job timed out, then kill its simulator process.

    Shared by the sync wait and the async watchdog. Ordering matters: the
    job goes terminal FIRST so the killed sim's completion callback discards
    the partial raw instead of recording a false success.
    NON_TERMINAL_LIVE_STATUSES also covers a job still queued on the
    concurrency gate — marking it terminal makes the pending
    start_simulation task release its slot without launching.
    """
    if job.status in NON_TERMINAL_LIVE_STATUSES:
        transition(job, "timeout", state=state)
    await runner.kill(job.job_id)


async def _enforce_async_deadline(
    job: SimulationJob,
    timeout: float,  # noqa: ASYNC109
    runner: SimulationRunner,
    state: SessionState,
) -> None:
    try:
        await asyncio.wait_for(job.done_event.wait(), timeout=timeout)
    except TimeoutError:
        await _timeout_job(job, runner, state)
        logger.warning(
            "Async simulation %s exceeded its %.0fs timeout and was killed",
            job.job_id,
            timeout,
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
        # Timeout - this is NOT a simulator error, it's a tool-level kill
        # (see _timeout_job for the transition-before-kill ordering).
        await _timeout_job(job, runner, state)
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
        files_note = _attach_result_files(data, job)
        return format_response(
            f"Simulation timed out after {duration:.1f}s (killed by server)\n"
            f"Job ID: {job.job_id}\n"
            f"Netlist: {job.netlist}{log_excerpt}{files_note}{TIMEOUT_HINT}",
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
        await mcp_log("error", f"Simulation failed: {job.netlist.name} — {job.error or 'unknown'}")
        return _failed_response(job, duration, state, fmt)
    elif job.status == "cancelled":
        data = {"job_id": job.job_id, "status": "cancelled"}
        return format_response(f"Simulation cancelled\nJob ID: {job.job_id}", data, fmt)
    else:
        # Unexpected status
        data = {"job_id": job.job_id, "status": job.status}
        return format_response(f"Simulation ended with unexpected status: {job.status}", data, fmt)


def _failed_response(job, duration: float, state: SessionState, fmt: str | None):
    """Build the response for a failed job — shared by run_simulation and check_job.

    Surfaces the error with its log excerpt (appended only if ``job.error`` doesn't
    already carry one — sim_runner usually embeds it), adds the model-resolution
    recovery hint, mirrors the augmented message into the structured ``error``
    field so structured and text clients see the same guidance, and appends the
    result-file footer.
    """
    error_msg = job.error or "Unknown error"
    data = {"job_id": job.job_id, "status": "failed", "duration": duration, "error": error_msg}
    if job.log_file and job.log_file.exists():
        if "Log excerpt:" not in error_msg:
            excerpt = extract_error_context(job.log_file, max_lines=20)
            error_msg = f"{error_msg}\n\nLog excerpt:\n{excerpt}"
        error_msg = services.attach_suggestions_to_failure(
            error_msg, data, job.log_file, state.libraries
        )
        hint = services.ngbehavior_lib_hint(
            job.netlist,
            error_msg,
            is_ngspice=is_ngspice(state.default_simulator),
            current_mode=current_ngbehavior(),
        )
        if hint:
            error_msg = f"{error_msg}\n\n{hint}"
        data["error"] = error_msg
    files_note = _attach_result_files(data, job)
    return format_response(
        f"Simulation failed\nJob ID: {job.job_id}\nDuration: {duration:.2f}s\n\n{error_msg}{files_note}",
        data,
        fmt,
    )


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
            # Batch (sweep / Monte Carlo) status fields.
            "job_type": {"type": "string"},
            "total_runs": {"type": "integer"},
            "completed_runs": {"type": "integer"},
            "failed_runs": {"type": "integer"},
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
    resolved = services.resolve_job(job_id, state)
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
        return _failed_response(job, duration, state, fmt)
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
        files_note = _attach_result_files(data, job)
        return format_response(
            f"Simulation timed out after {duration:.1f}s (killed by server)\n"
            f"Job ID: {job_id}\n"
            f"Netlist: {job.netlist}{log_excerpt}{files_note}{TIMEOUT_HINT}",
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


def _attach_result_files(data: dict, job: SimulationJob) -> str:
    """Record the job's raw/log paths on a failed/timed-out response ``data``
    dict and return a matching text footer, so the caller can open the full
    .log/.raw instead of working from the truncated excerpt alone. Both schema
    keys are typed string, so a path is omitted when the job has none.
    """
    notes = []
    if job.log_file:
        data["log_file"] = str(job.log_file)
        notes.append(f"  log: {job.log_file}")
    if job.raw_file:
        data["raw_file"] = str(job.raw_file)
        notes.append(f"  raw: {job.raw_file}")
    return "\n\nResult files:\n" + "\n".join(notes) if notes else ""


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
    }
    # Omit-when-empty, like every other optional field: emitting "error":
    # null here violated the declared output schema (error is typed string),
    # which made schema-validating MCP clients reject every batch-job poll.
    if batch_job.error is not None:
        data["error"] = batch_job.error
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
        if status_filter == "all":
            message = "No jobs found"
        elif status_filter:
            message = (
                f"No jobs with status '{status_filter}'. Pass status=\"all\" to list every job."
            )
        elif all_jobs:
            # Default view shows only queued/running; terminal jobs are hidden.
            # Say so and how to widen, so a just-completed run isn't read as
            # "nothing exists".
            message = (
                f"No active jobs (queued/running). {len(all_jobs)} finished job(s) are "
                'hidden — pass status="all" to list them, or a specific status '
                "(completed, failed, timeout, cancelled, interrupted)."
            )
        else:
            message = "No active jobs"
        return format_response(message, {"jobs": [], "count": 0}, fmt)

    # Build structured data
    jobs_data = []
    lines = [f"Simulation Jobs ({len(jobs_to_show)}):\n"]
    lines.append(f"{'ID':<28} | {'Status':<10} | {'Netlist':<20} | {'Started':<17} | Duration")
    lines.append("-" * 100)

    for job in jobs_to_show:
        emit_duration = True
        if job.status in NON_TERMINAL_LIVE_STATUSES:
            duration = max(0.0, (now() - job.started_at).total_seconds())
            duration_str = f"{duration:.1f}s (running)"
        elif job.completed_at:
            duration = (
                services.job_duration_seconds(
                    job.started_at, job.completed_at, label=f"sim job {job.job_id}"
                )
                or 0.0
            )
            duration_str = f"{duration:.1f}s"
        else:
            # Terminal but no completed_at — an interrupted/recovered job. True
            # runtime is unknowable after a restart, so don't fabricate a
            # wall-clock-to-now number (which read as a multi-hour sim) or the
            # "(running)" label. Omit the duration key, matching the single-job
            # _terminal_job_data path.
            duration = None
            duration_str = "unknown"
            emit_duration = False

        started_str = job.started_at.strftime("%Y-%m-%d %H:%M")
        netlist_name = job.netlist.name
        if len(netlist_name) > 20:
            netlist_name = netlist_name[:17] + "..."

        lines.append(
            f"{job.job_id:<28} | {job.status:<10} | {netlist_name:<20} | {started_str:<17} | {duration_str}"
        )
        entry = {
            "job_id": job.job_id,
            "job_type": getattr(job, "job_type", "single"),
            "status": job.status,
            "netlist": str(job.netlist),
            "started_at": job.started_at.isoformat(),
        }
        if emit_duration:
            entry["duration"] = duration
        jobs_data.append(entry)

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

    job = services.resolve_job(job_id, state)

    # Check if job is running
    if job.status not in NON_TERMINAL_LIVE_STATUSES:
        # A terminal job has nothing to cancel — this is a job-state error, not
        # a simulator-availability one, so suppress the generic SimulationError
        # hint ("verify simulator availability") and point at check_job instead.
        raise SimulationError(
            f"Job {job_id} is not running (status: {job.status}) — it has already "
            f"finished, so there is nothing to cancel. Use check_job('{job_id}') to "
            "read its result.",
            show_hint=False,
        )

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
        # Resolve the runner via the JOB's netlist, so its output folder matches
        # the one the job launched with. Acquiring with no netlist resolves to a
        # different folder, which makes RunnerManager invalidate the live runner
        # (losing the spicelib process handle) — the simulator would keep running
        # while the job shows cancelled.
        sim_runner = await _get_or_create_runner(state, job.netlist)
        await sim_runner.cancel(job, state)

    return text_response(f"Job {job_id} cancelled")
