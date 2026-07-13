"""Simulation execution tools. (Phase 3)"""

import asyncio
import logging
import time
from pathlib import Path
from typing import Literal

from mcp import types
from pydantic import Field

from ltspice_mcp.config import ServerConfig
from ltspice_mcp.errors import ResultError, SimulationError
from ltspice_mcp.lib import now, services
from ltspice_mcp.lib.encoding import read_spice_text
from ltspice_mcp.lib.job_lifecycle import transition
from ltspice_mcp.lib.log_parser import extract_error_context, parse_success_summary
from ltspice_mcp.lib.mcp_logging import mcp_log
from ltspice_mcp.lib.runner_base import discard_logopinfo_netlist
from ltspice_mcp.lib.sim_runner import SimulationRunner, generate_job_id
from ltspice_mcp.lib.simulator import current_ngbehavior, is_ngspice, no_simulator_message
from ltspice_mcp.lib.spice_validator import estimate_analysis_points
from ltspice_mcp.state import (
    NON_TERMINAL_LIVE_STATUSES,
    BatchJob,
    SessionState,
    SimulationJob,
)
from ltspice_mcp.tools._base import (
    FORMAT_DESCRIPTION,
    HINT_SCHEMA,
    MEAS_ERRORS_SCHEMA,
    MEASUREMENTS_SCHEMA,
    OBSERVATIONS_SCHEMA,
    SUGGESTIONS_SCHEMA,
    WARNINGS_SCHEMA,
    ToolInput,
    format_meas_errors,
    format_observations,
    format_response,
    inject_logopinfo,
    registry,
    require_simulator,
    resolve_netlist_path,
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

# How long the async path waits inline for completion before returning a job
# handle. Most .op/.ac/small-.tran runs finish in well under a second, and
# returning their parsed results directly saves every caller a check_job
# round-trip; a run that outlives the grace window continues in the background
# under the deadline watchdog (nothing is killed at grace expiry).
SYNC_GRACE_WAIT = 10.0

# Appended to the timed-out response (built by _timeout_response, shared by the
# sync wait path and the async check_job path). A timeout is a tool-set limit,
# not a simulator failure, so name the levers to raise it — otherwise the agent
# reads "timed out" as a dead end and loops.
TIMEOUT_HINT = (
    "This is the configured time limit, not a simulator error. To allow more "
    "time, pass run_simulation(timeout=<seconds>) for this run, or raise the "
    "default via [simulation] timeout in the config file or LTSPICE_MCP_TIMEOUT "
    "(restart required). server_status shows the current default."
)

logger = logging.getLogger(__name__)

_BYTES_PER_POINT = 8  # one float64 per saved point — the single-trace lower bound


def _safe_stat_size(path: Path) -> int:
    """File size in bytes, or 0 if it doesn't exist / can't be statted."""
    try:
        return path.stat().st_size
    except OSError:
        return 0


def _preflight_size_guard(netlist_path: Path, config: ServerConfig) -> str | None:
    """Estimate the raw a run will produce and guard against a runaway.

    Refuses (raises ``SimulationError``) when the estimated raw exceeds
    ``max_raw_mb`` — sized as a single-trace lower bound (``8 bytes/point``), so a
    real multi-trace raw is only larger and this never false-refuses; it catches a
    directive typo (a fs step for a ns run) before it fills the disk. Warns (return
    value) when the estimated point count exceeds ``max_estimated_points``. Returns
    the warning string or ``None``. An unestimable directive (auto-timestep,
    parameterised) is left alone — the estimate is best-effort, not a gate on every
    run.
    """
    try:
        points = estimate_analysis_points(read_spice_text(netlist_path))
    except (OSError, OverflowError, ValueError):
        # Best-effort estimate: an unreadable deck or a pathological directive
        # (e.g. .tran 1e-200 1e200 → non-finite point count) leaves the run
        # ungated rather than crashing run_simulation.
        return None
    if points is None:
        return None
    est_mb = points * _BYTES_PER_POINT / (1024 * 1024)
    if est_mb > config.max_raw_mb:
        raise SimulationError(
            f"Refusing to run: the analysis directive estimates ~{points:,} points "
            f"(>={est_mb:,.0f} MB for even a single trace, over the {config.max_raw_mb} MB "
            "max_raw_mb cap — a real multi-trace raw is larger). Check the timestep / "
            "step size, or raise [simulation] max_raw_mb. (.tran uses adaptive stepping, "
            "so the real count may differ — this is an estimate from tstop/tstep.)",
            show_hint=False,
        )
    if points > config.max_estimated_points:
        return (
            f"Large run: the analysis directive estimates ~{points:,} points; the raw may "
            "be slow to produce and parse. check_job reports raw_bytes as it grows."
        )
    return None


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
    "warnings": WARNINGS_SCHEMA,
    "errors": {"type": "array", "items": {"type": "string"}},
    "meas_errors": MEAS_ERRORS_SCHEMA,
    "measurements": MEASUREMENTS_SCHEMA,
    "fourier": {"type": "array", "items": {"type": "object"}},
    "range": {"type": "object"},
    "point_count": {"type": "integer"},
    # Ambient / nominal temperature the simulator ran at (°C), when the log
    # records it — flows in from build_simulation_summary.
    "temp_c": {"type": "number"},
    "tnom_c": {"type": "number"},
    "failed_measurements": {"type": "array", "items": {"type": "string"}},
    "observations": OBSERVATIONS_SCHEMA,
    # Fuzzy library matches for unresolved model/subcircuit references, keyed
    # by the missing ref (attached on failure and on completed runs whose log
    # still reports an unresolved ref).
    "suggestions": SUGGESTIONS_SCHEMA,
    # Optional, path-dependent: caller guidance (async referral, timeout
    # levers, batch redirect, hidden-jobs note) and the timeout log excerpt.
    "hint": HINT_SCHEMA,
    "log_excerpt": {"type": "string"},
}


class RunSimulationInput(ToolInput):
    """Inputs for run_simulation."""

    netlist: str = Field(description="Path to the netlist file (.cir, .net, .asc)")
    simulator: str | None = Field(
        default=None,
        description=(
            "Simulator for this run, by detected name (e.g. 'ltspice', 'ngspice' — "
            "server_status lists them). Defaults to the server's default simulator. "
            "Lets one deck be cross-checked on a second engine without changing config."
        ),
    )
    timeout: float | None = Field(
        default=None,
        description=(
            "Timeout in seconds (defaults to the server's configured default, 300s). "
            "Runs that finish within a short grace window return results inline; "
            "longer runs return a job ID for check_job tracking. With wait=true the "
            "effective limit is min(this timeout, 600s): 600s is a hard ceiling, not "
            "a floor — pass a larger timeout to use the full 600s."
        ),
    )
    wait: bool = Field(
        default=False,
        description="Force synchronous execution. Blocks until completion or hard timeout.",
    )
    format: Literal["json", "text"] | None = Field(
        default=None,
        description=FORMAT_DESCRIPTION,
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
        description=FORMAT_DESCRIPTION,
    )


class CancelJobInput(ToolInput):
    """Inputs for cancel_job."""

    job_id: str = Field(description="Job ID of the running simulation to cancel")


async def _get_or_create_runner(
    state: SessionState,
    netlist_path: Path | None = None,
    simulator_class: type | None = None,
) -> SimulationRunner:
    """Get or create a SimulationRunner via the centralized RunnerManager."""
    sim_cls = simulator_class or state.default_simulator
    if sim_cls is None:
        raise SimulationError(no_simulator_message())
    return state.runners.get_sim_runner(
        loop=asyncio.get_running_loop(),
        simulator_class=sim_cls,
        output_folder=await resolve_output_folder(state, netlist_path, simulator=sim_cls),
        max_parallel=state.config.max_parallel_sims,
    )


@registry.tool(
    name="run_simulation",
    description=(
        "Run a SPICE simulation on a netlist file. Sets the right batch flags, "
        "handles the ngspice headerless-raw dialect, routes the raw/log "
        "artifacts, and parses the results — so you never hand-parse a rawfile. "
        "Fast runs return their parsed results inline (short grace wait); "
        "longer runs return a job ID for check_job tracking. Use wait=true to "
        "force synchronous execution up to the hard 600s ceiling. Pass "
        "simulator= to run on a non-default engine (e.g. cross-check a deck "
        "on ngspice)."
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

    # Cheap path validation first, so a bad path reports as such even when no
    # simulator is configured.
    resolve_netlist_path(netlist_str, state)

    if args.simulator is not None:
        sim_cls = state.available_simulators.get(args.simulator.lower())
        if sim_cls is None:
            raise SimulationError(
                f"Simulator '{args.simulator}' is not available on this server "
                f"(detected: {list(state.available_simulators)}). server_status "
                "lists the detected simulators.",
                show_hint=False,
            )
        default_simulator = sim_cls
    else:
        require_simulator(state)
        assert state.default_simulator is not None  # guaranteed by require_simulator
        default_simulator = state.default_simulator

    # Simulator resolved BEFORE the .asc export so an ngspice target gets the
    # sanitized export (LTspice's .backanno / µ / § would abort ngspice).
    netlist_path = await resolve_runnable_netlist(netlist_str, state, simulator=default_simulator)

    preflight_warnings = services.ngspice_preflight_warnings(netlist_path, default_simulator)

    # Preflight size guard: estimate the raw the analysis directive will produce
    # and refuse a runaway (e.g. a fs step for a ns run) before launching, or
    # warn on a merely-large one. Raises SimulationError on refuse.
    if size_warning := _preflight_size_guard(netlist_path, state.config):
        preflight_warnings = [*preflight_warnings, size_warning]

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
        runner = await _get_or_create_runner(
            state, netlist_path, simulator_class=default_simulator
        )
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
    # Elif timeout <= threshold: sync (the timeout IS the inline deadline)
    # Else: async with a short inline grace wait — fast runs return their
    # results directly instead of forcing a check_job round-trip.
    if wait:
        effective_timeout = min(timeout, HARD_MAX_TIMEOUT)
        return await _wait_for_completion(
            job, effective_timeout, runner, state, fmt, preflight_warnings
        )
    elif timeout <= SYNC_TIMEOUT_THRESHOLD:
        return await _wait_for_completion(job, timeout, runner, state, fmt, preflight_warnings)
    else:
        # Async path — arm the deadline watchdog first: the sync branches
        # enforce their deadline via wait_for, and without a watchdog an
        # async job's timeout (including the config default) was accepted
        # and never enforced.
        _arm_timeout_watchdog(job, timeout, runner, state)
        try:
            await asyncio.wait_for(job.done_event.wait(), timeout=min(SYNC_GRACE_WAIT, timeout))
        except TimeoutError:
            # Still running after the grace window — hand back the job id.
            # (The wait above also let the submission task advance, so the
            # reported status reflects reality: "running" when a slot was
            # free, "queued" when waiting on the concurrency cap.)
            data = {
                "job_id": job_id,
                "status": job.status,
                "netlist": str(netlist_path),
                "simulator": default_simulator.__name__,
                # Structured-content clients render only structuredContent, so
                # the follow-up referral must live in the data dict too.
                "hint": (
                    f"Use check_job('{job_id}') to check status, check_job() to see "
                    f"all jobs, or cancel_job('{job_id}') to cancel."
                ),
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
        duration = (
            services.job_duration_seconds(
                job.started_at, job.completed_at, label=f"sim job {job.job_id}"
            )
            or 0.0
        )
        return await _finished_job_response(job, duration, state, fmt, preflight_warnings)


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
        if job.log_file is None:
            # The completion callback (which normally records log_file) hasn't
            # fired yet, and once it does it early-returns on the terminal
            # status — so derive the path from the run-file naming contract
            # ({job_id}.log in the runner's output folder). Without this the
            # timeout response could never show a log excerpt.
            job.log_file = runner.output_folder / f"{job.job_id}.log"
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

        return await _timeout_response(job, duration, fmt)

    # Simulation completed (success or failure)
    duration = time.monotonic() - start_time
    return await _finished_job_response(job, duration, state, fmt, preflight_warnings)


async def _finished_job_response(
    job: SimulationJob,
    duration: float,
    state: SessionState,
    fmt: str | None,
    preflight_warnings: list[str] | None = None,
):
    """Build the response for a job that reached a terminal state.

    Shared by the sync wait path and the async grace-wait path — the latter
    can observe any terminal status, including a timeout that raced the
    deadline watchdog.
    """
    if job.status == "completed":
        if job.raw_file is None:
            # Log-only completion: a clean exit that wrote results (if any)
            # to the log rather than a raw file — see collect_run_outcome.
            await mcp_log(
                "info", f"Simulation completed (log-only): {job.netlist.name} ({duration:.1f}s)"
            )
            return await _log_only_response(job, duration, fmt, preflight_warnings)
        if job.log_file is None:
            raise ResultError(
                f"Job {job.job_id} completed but result files are missing.\n"
                f"raw_file: {job.raw_file}, log_file: {job.log_file}"
            )
        # Offload the raw parse off the event-loop thread (heavy, untrusted
        # I/O); dialect_for_job stays on the loop (cheap) before the hop, and
        # parse_success_summary returns a dict, not a CallToolResult.
        summary = await asyncio.to_thread(
            parse_success_summary,
            job.raw_file,
            job.log_file,
            duration,
            dialect=services.dialect_for_job(job, state),
            netlist=job.netlist,
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
    elif job.status == "timeout":
        return await _timeout_response(job, duration, fmt)
    else:
        # Unexpected status
        data = {"job_id": job.job_id, "status": job.status}
        return format_response(f"Simulation ended with unexpected status: {job.status}", data, fmt)


def _read_log_only_payload(log_file: Path) -> tuple[list[str], list[str], dict, list[str]]:
    """Read a log-only run's diagnostics and measurements (worker thread only —
    both calls below read the whole log, which can stall on a hung mount)."""
    from ltspice_mcp.lib.log_parser import extract_log_diagnostics, parse_measurements

    if not log_file.exists():
        return [], [], {}, []
    diagnostics = extract_log_diagnostics(log_file)
    measurements: dict = {}
    failed: list[str] = []
    try:
        meas = parse_measurements(log_file)
        measurements = meas["measurements"]
        failed = meas["failed_measurements"]
    except Exception as e:
        logger.debug("parse_measurements failed for %s: %s", log_file, e)
    return diagnostics["warnings"], diagnostics["errors"], measurements, failed


async def _log_only_response(
    job: SimulationJob,
    duration: float,
    fmt: str | None,
    preflight_warnings: list[str] | None = None,
):
    """Response for a completed run that produced no raw waveform data.

    Happens when a clean simulator exit writes results only to the log — an
    ngspice ``.control`` script driving its own analyses, or a deck with no
    analysis card. The log-parsed measurements are the payload; the no-raw
    fact rides as a coverage observation, and waveform tools will correctly
    refuse this job.
    """
    log_file = job.log_file
    warnings_list: list[str] = list(preflight_warnings or [])
    errors: list[str] = []
    measurements: dict = {}
    failed: list[str] = []
    if log_file is not None:
        log_warnings, errors, measurements, failed = await asyncio.to_thread(
            _read_log_only_payload, log_file
        )
        warnings_list.extend(log_warnings)

    hint = (
        "No raw waveform file was produced (common for .control-script decks "
        "whose results go to the log). Waveform tools cannot read this run; "
        "read the log file directly for printed output. Parsed .meas results, "
        "when present, are in 'measurements'."
    )
    data: dict = {
        "job_id": job.job_id,
        "status": "completed",
        "duration": duration,
        "observations": [
            {
                "code": "no_raw_output",
                "kind": "coverage",
                "detail": (
                    "The simulator exited cleanly but wrote no raw waveform "
                    "data; results, if any, are in the log."
                ),
            }
        ],
        "hint": hint,
    }
    if log_file is not None:
        data["log_file"] = str(log_file)
    if warnings_list:
        data["warnings"] = warnings_list
    if errors:
        data["errors"] = errors
    if measurements:
        data["measurements"] = measurements
    if failed:
        data["failed_measurements"] = failed

    meas_note = f"\nMeasurements parsed from log: {len(measurements)}" if measurements else ""
    text = (
        f"Simulation completed (log-only: no raw waveform data)\n"
        f"Job ID: {job.job_id}\n"
        f"Duration: {duration:.2f}s\n"
        f"Log file: {log_file}{meas_note}\n\n{hint}"
    )
    return format_response(text, data, fmt)


def _read_timeout_excerpt(log_file: Path) -> str | None:
    """Existence probe + capped excerpt read (worker thread only — file I/O)."""
    if not log_file.exists():
        return None
    return extract_error_context(log_file, max_lines=20)


async def _timeout_response(job, duration: float, fmt: str | None):
    """Build the timed-out response — shared by the sync wait path and check_job.

    Log excerpt and raise-the-limit guidance ride in the structured payload
    too (see format_response's self-sufficiency contract).
    """
    excerpt: str | None = None
    if job.log_file:
        excerpt = await asyncio.to_thread(_read_timeout_excerpt, job.log_file)
    log_excerpt = f"\n\nLog excerpt:\n{excerpt}" if excerpt else ""

    data = {
        "job_id": job.job_id,
        "status": "timeout",
        "duration": duration,
        "netlist": str(job.netlist),
        "hint": TIMEOUT_HINT,
    }
    if excerpt:
        data["log_excerpt"] = excerpt
    files_note = _attach_result_files(data, job)
    return format_response(
        f"Simulation timed out after {duration:.1f}s (killed by server)\n"
        f"Job ID: {job.job_id}\n"
        f"Netlist: {job.netlist}{log_excerpt}{files_note}\n\n{TIMEOUT_HINT}",
        data,
        fmt,
    )


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
            # The job's own simulator, not the session default: a per-run
            # simulator override (run_simulation(simulator=...)) makes them
            # differ, and the sectioned-.lib hint is ngspice-specific.
            is_ngspice=is_ngspice(services.simulator_class_for_job(job, state)),
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
        "suggestions",
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
            "raw_bytes": {"type": "integer"},
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
                        "job_type": {"type": "string"},
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
    resolved = await services.resolve_job_async(job_id, state)
    if isinstance(resolved, BatchJob):
        return _check_batch_job(resolved, fmt)
    job = resolved

    # Check status
    if job.status in NON_TERMINAL_LIVE_STATUSES:
        elapsed = (now() - job.started_at).total_seconds()
        # Progress signal: the raw grows on disk as the run writes it. One
        # on-demand stat (no poller), offloaded so a slow mount can't stall the
        # loop; 0 before the file appears.
        raw_bytes = (
            await asyncio.to_thread(_safe_stat_size, job.raw_file)
            if job.raw_file is not None
            else 0
        )
        data = {
            "job_id": job_id,
            "status": job.status,
            "netlist": str(job.netlist),
            "simulator": job.simulator,
            "elapsed": elapsed,
            "raw_bytes": raw_bytes,
            "hint": f"Use cancel_job('{job_id}') to cancel.",
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
        if job.raw_file is None:
            # Log-only completion (clean exit, results in the log, no raw).
            return await _log_only_response(job, duration, fmt)
        if job.log_file is None:
            raise ResultError(
                f"Job {job_id} completed but result files are missing.\n"
                f"raw_file: {job.raw_file}, log_file: {job.log_file}"
            )
        if not job.raw_file.exists() or not job.log_file.exists():
            raise ResultError(
                f"Job {job_id} completed but result files have been removed.\n"
                f"raw: {job.raw_file.exists()}, log: {job.log_file.exists()}"
            )
        # Offload the raw parse off the event-loop thread (heavy, untrusted I/O);
        # dialect_for_job stays on the loop (cheap) before the hop.
        summary = await asyncio.to_thread(
            parse_success_summary,
            job.raw_file,
            job.log_file,
            duration,
            dialect=services.dialect_for_job(job, state),
            netlist=job.netlist,
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
        return await _timeout_response(job, duration, fmt)
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
        data = _terminal_job_data(job, "interrupted")
        data["hint"] = (
            "The server stopped while this job was running, so results are "
            "incomplete; re-run if you need them."
        )
        return format_response(
            f"Job {job_id} was interrupted — the server stopped while it was running, "
            f"so results are incomplete; re-run if you need them.\nNetlist: {job.netlist}",
            data,
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
    redirect = (
        f"Use batch_results('{batch_job.job_id}') for per-run data, "
        "or measurement_stats for aggregated .MEAS statistics."
    )
    data["hint"] = redirect
    text = (
        f"Batch job {batch_job.job_id} ({batch_job.job_type}): {batch_job.status}\n"
        f"Netlist: {batch_job.netlist}\n"
        f"Runs: {batch_job.completed_runs}/{batch_job.total_runs} completed, "
        f"{batch_job.failed_runs} failed"
    )
    if batch_job.error:
        text += f"\nError: {batch_job.error}"
    text += f"\n\n{redirect}"
    return format_response(text, data, fmt)


def _list_jobs(arguments: CheckJobInput, state: SessionState, fmt: str | None = None):
    """List simulation jobs (single + batch) with optional status filter."""
    status_filter = arguments.status

    # The union store holds every job (single-run and sweep/MC batch), so
    # check_job is a complete view of "what jobs exist".
    all_jobs: list[SimulationJob | BatchJob] = state.job_registry.refreshed_jobs()

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
        empty_data: dict = {"jobs": [], "count": 0}
        if status_filter == "all":
            message = "No jobs found"
        elif status_filter:
            message = (
                f"No jobs with status '{status_filter}'. Pass status=\"all\" to list every job."
            )
            empty_data["hint"] = message
        elif all_jobs:
            # Default view shows only queued/running; terminal jobs are hidden.
            # Say so and how to widen, so a just-completed run isn't read as
            # "nothing exists". Mirrored into the data dict as a hint:
            # structured-content clients never see the text channel, and
            # {jobs: [], count: 0} alone reads as "nothing exists".
            message = (
                f"No active jobs (queued/running). {len(all_jobs)} finished job(s) are "
                'hidden — pass status="all" to list them, or a specific status '
                "(completed, failed, timeout, cancelled, interrupted)."
            )
            empty_data["hint"] = message
        else:
            message = "No active jobs"
        return format_response(message, empty_data, fmt)

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

    job = await services.resolve_job_async(job_id, state)

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
    # launched it, so route by ownership rather than assuming one runner per kind.
    require_simulator(state)
    if isinstance(job, BatchJob):
        batch_runner = state.runners.get_batch_runner_for(job)
        if batch_runner is None:
            raise SimulationError(
                f"Job {job_id} is marked running but its {job.job_type} runner is no "
                "longer live (server restarted?), so there is no process to cancel."
            )
        await batch_runner.cancel(job, state)
    else:
        # Resolve the runner via the JOB's netlist and recorded simulator, so
        # the cache key (class, output folder) matches the one the job
        # launched with — a mismatch would resolve to a different runner
        # whose kill scopes by the wrong executable names.
        sim_runner = await _get_or_create_runner(
            state, job.netlist, simulator_class=services.simulator_class_for_job(job, state)
        )
        await sim_runner.cancel(job, state)

    return text_response(f"Job {job_id} cancelled")
