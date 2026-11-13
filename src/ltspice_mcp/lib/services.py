"""Application-level services shared by tools and resources.

This module sits between the MCP adapters (tools/resources) and the pure
parsing helpers in ``raw_parser.py`` / ``log_parser.py``. It owns job
resolution, cached result loading, and reusable extraction/orchestration
logic. All functions raise domain exceptions rather than returning error text.
"""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import logging
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

import numpy as np

from ltspice_mcp.errors import (
    AnalysisDeadlineExceeded,
    JobNotFoundError,
    OtherRawNotWritten,
    ResultError,
)
from ltspice_mcp.lib import recent
from ltspice_mcp.lib.decoded_log import DecodedLog
from ltspice_mcp.lib.decoded_raw import DecodedPlot, DecodedRaw, RawData
from ltspice_mcp.lib.experiment_types import (
    ExperimentCase,
    ExperimentJob,
    failure_summary,
    partial_raw,
)
from ltspice_mcp.lib.job_lifecycle import runs_done
from ltspice_mcp.lib.log_parser import (
    LogDiagnostics,
    MeasurementsOutput,
    answered_measurements,  # noqa: TID251 - answer_measurements is the readers' door to it
    measurement_cards,
)
from ltspice_mcp.lib.parsed_artifacts import ParsedArtifacts
from ltspice_mcp.lib.parser_service import ParserCleanupError, select_raw
from ltspice_mcp.lib.parser_service import load_artifacts_sync as _parse_artifacts_sync
from ltspice_mcp.lib.parser_service import load_logs_sync as _parse_logs_sync
from ltspice_mcp.lib.parser_service import load_raw_sync as _parse_raw_sync
from ltspice_mcp.lib.raw_parser import (
    OtherRawKind,
    get_step_count,
    other_raw_path,
    points_reached,
)
from ltspice_mcp.lib.result_observations import (
    coarse_compressed_waveform,
    coarse_waveform_observation,
    compressed_transient,
    deck_tran_max_step,
)
from ltspice_mcp.lib.simulator import dialect_for_simulator_name
from ltspice_mcp.lib.simulator_build import (
    Producer,
    SimulatorExecutable,
    executable_identity,
    ltspice_before_24,
    same_executable,
)
from ltspice_mcp.lib.trace_names import (
    NODE_PAIR_RE,
    NameRead,
    read_in_plot,
    why_not,
)
from ltspice_mcp.state import SessionState

if TYPE_CHECKING:
    from ltspice_mcp.lib.hierarchy import SemanticProfile

logger = logging.getLogger(__name__)

T = TypeVar("T")


def resolve_job(job_id: str, state: SessionState) -> ExperimentJob:
    """Look up a job by id.

    Discovery belongs to the registry, which asks the store when it does not
    already hold the job; this function only turns the two ways of not having
    one into the exceptions callers up the stack expect, so they don't re-wrap
    them. Then a parallel session's live job is re-read from disk — only its
    owner updates it, so a status check here would otherwise stay frozen at
    "running" forever. No-op for this session's own jobs.
    """
    try:
        job = state.job_registry.get_or_load(job_id)
    except ValueError as exc:
        raise ResultError(str(exc)) from None
    if job is None:
        raise JobNotFoundError(f"Job not found: {job_id}")
    return state.job_registry.refresh_foreign_job(job)


async def resolve_job_async(job_id: str, state: SessionState) -> ExperimentJob:
    """Loop-safe ``resolve_job``: offload the store read and the foreign re-read.

    Use from async handlers so neither disk read (either can stall the loop on
    a wedged filesystem) runs on it. Same semantics otherwise.
    """
    try:
        job = await state.job_registry.get_or_load_async(job_id)
    except ValueError as exc:
        raise ResultError(str(exc)) from None
    if job is None:
        raise JobNotFoundError(f"Job not found: {job_id}")
    return await state.job_registry.refresh_foreign_job_async(job)


# TERMINAL SPICE solve-failure phrases. When the log carries one, the solve
# genuinely failed — it taints every value read, not one trace — so a read tool
# relays it regardless of which signal was asked for. Deliberately terminal-only:
# a bare "singular matrix" is NOT listed, because a transient can recover from it
# via gmin/source stepping and still write a valid raw (log_parser classifies it
# as non-terminal for exactly this reason). Flagging it would be a false
# accusation on a recovered run; a genuine non-recovery still trips one of the
# terminal phrases below (e.g. "gmin stepping failed"). ("no convergence", not
# bare "convergence", so a benign "convergence achieved" line doesn't match.)
#
# They live here rather than in either tool module because both profiles must
# classify a failed solve the same way: the full profile relays into its
# ``warnings`` channel and the consolidated one into ``observations``, and a
# rule kept in one of them is a rule the other can forget.
SOLVE_FAILURE_PHRASES = (
    "no convergence",
    "time step too small",
    "timestep too small",
    "gmin stepping failed",
    "source stepping failed",
    "iteration limit reached",
)


def solve_failure_lines(diagnostics: LogDiagnostics) -> list[str]:
    """The run-level solve-failure lines in an ``extract_log_diagnostics`` result.

    Reads both channels: LTspice prints these as errors, ngspice prints the
    same failures under a ``Warning:`` prefix.
    """
    return [
        line
        for line in (*diagnostics["warnings"], *diagnostics["errors"])
        if any(phrase in line.lower() for phrase in SOLVE_FAILURE_PHRASES)
    ]


@dataclass(frozen=True)
class RunContext:
    """Trusted, case-addressed experiment result and its provenance identity."""

    raw: Path | None
    log: Path | None
    netlist: Path
    #: The source schematic/deck the case was staged from — where per-circuit
    #: sidecars (plots, pointers) belong. ``netlist`` is the staged copy.
    circuit_path: Path
    dialect: str | None
    identity: dict[str, Any]
    console: Path | None = None
    #: The build the case's run named in its own output, as recorded.
    simulator_version: str | None = None
    #: What a read of this case's results has to say of them: a stopped case's
    #: kept raw ends where its run was stopped (``partial_raw_observation``).
    observations: tuple[dict[str, Any], ...] = ()
    #: Who wrote the case's log (``run_producer``).
    producer: Producer | None = None


@dataclass(frozen=True)
class AnalysisSource:
    """Resolved source injected into analysis adapters by the consolidated path."""

    raw: Path | None
    log: Path | None
    netlist: Path | None
    dialect: str | None
    identity: dict[str, Any] | None
    trusted_job_artifact: bool
    explicit_dialect: str | None = None
    plot_index: int = 0
    console: Path | None = None
    #: The build a job's run named in its own output, as its case recorded it;
    #: None for a bare path, whose build is read from the artifacts instead.
    simulator_version: str | None = None
    #: What a read of these results says of them, from the case that made them
    #: (``RunContext.observations``); none for a bare path.
    observations: tuple[dict[str, Any], ...] = ()
    #: The signals a read keeps the traces of (``trace_names.chosen_traces``),
    #: with the axis and any step parameters; None reads every trace.
    signals: tuple[str, ...] | None = None
    #: The span ``(start, end)`` of the read plot's axis whose points a read
    #: keeps (``raw_decode.AxisWindow``), an end None for open; None keeps all.
    window: tuple[float | None, float | None] | None = None
    #: Who wrote the log, for a job's case (``run_producer``); None for a bare
    #: path, whose producer is what ``source_producer`` says.
    producer: Producer | None = None
    # Resident facts belong to one evaluation, between fresh drift checks.
    captured: ParsedArtifacts | None = field(default=None, repr=False, compare=False)


def source_for_raw_path(
    raw: Path, state: SessionState, *, plot_index: int = 0, dialect: str | None = None
) -> AnalysisSource:
    """The source an already-validated caller-supplied ``.raw`` path resolves to.

    The path must already have passed ``safe_path``. Companion candidates are
    authorized independently on load, where a file the capture needs and does
    not find is refused (``parser_service``); only the contained capture worker
    reads writer evidence. Session simulator defaults are not provenance.
    """
    sibling = raw.with_suffix(".log")
    return AnalysisSource(
        raw=raw,
        log=sibling,
        netlist=None,
        dialect=None,
        identity=None,
        trusted_job_artifact=False,
        explicit_dialect=dialect,
        plot_index=plot_index,
        console=raw.with_suffix(".exe.log"),
    )


def source_for_run(
    run: RunContext, *, plot_index: int = 0, dialect: str | None = None
) -> AnalysisSource:
    """A resolved experiment case as the source its readers take.

    Trusted: the raw, log and staged deck are this server's own artifacts, so
    they are read where the record says they are rather than re-validated
    against ``allowed_paths`` — a reloaded job's raw legitimately lives outside
    it.
    """
    return AnalysisSource(
        raw=run.raw,
        log=run.log,
        netlist=run.netlist,
        dialect=run.dialect,
        identity=run.identity,
        trusted_job_artifact=True,
        explicit_dialect=dialect,
        plot_index=plot_index,
        console=run.console,
        simulator_version=run.simulator_version,
        observations=run.observations,
        producer=run.producer,
    )


def coarse_waveform(source: AnalysisSource, raw: DecodedRaw) -> dict[str, float] | None:
    """Whether waveform compression left ``raw``'s transient coarse
    (``coarse_compressed_waveform``), judged against the maximum step of the
    deck ``source`` names. The deck is read only for a plot already coarse by
    its run's share alone, since the deck's step can only raise the bar.
    Blocking: it may read the deck."""
    found = coarse_compressed_waveform(raw)
    if found is None or source.netlist is None:
        return found
    return coarse_compressed_waveform(raw, max_step=deck_tran_max_step(source.netlist))


def read_observations(source: AnalysisSource, raw: DecodedRaw) -> tuple[dict[str, Any], ...]:
    """What a read of ``raw`` from ``source`` says of the results it read: the
    source's own (a stopped case's ``partial_raw``), whether waveform
    compression left a transient coarse (:func:`coarse_waveform`), and which
    signals and window a selective read kept. Blocking: it may read the deck.

    Compression is judged over a run's every point, so a windowed read, which
    holds some of them, does not judge it, and says so.
    """
    observations = list(source.observations)
    found = None if source.window is not None else coarse_waveform(source, raw)
    if found is not None:
        case_id = (source.identity or {}).get("case_id")
        label = str(case_id) if case_id else (source.raw.name if source.raw else "this run")
        observations.append(dict(coarse_waveform_observation([(label, found)], run_cap=1)))
    if source.signals is not None:
        observations.append(
            {
                "code": "signals_chosen",
                "kind": "provenance",
                "detail": (
                    f"Read with signals={list(source.signals)}: this result holds the "
                    "traces they read, the plot's first variable (its axis, where it has "
                    "one) and any step parameters, and no other trace."
                ),
            }
        )
    if source.window is not None:
        start, end = ("None" if bound is None else f"{bound:g}" for bound in source.window)
        unjudged = (
            " Whether waveform compression left the run coarse "
            "(coarse_compressed_waveform) is judged over all of a run's points, so it "
            "was not judged here: read the run without window to see it."
            if compressed_transient(raw)
            else ""
        )
        observations.append(
            {
                "code": "window_chosen",
                "kind": "provenance",
                "detail": (
                    f"Read with window=({start}, {end}): this result holds the points of "
                    "its axis from the start to the end, in each step, and no other "
                    f"point.{unjudged}"
                ),
            }
        )
    return tuple(observations)


_analysis_deadline: contextvars.ContextVar[float | None] = contextvars.ContextVar(
    "ltspice-mcp.analysis-deadline",
    default=None,
)


@contextlib.contextmanager
def analysis_deadline(deadline: float | None):
    """Bound every result parse under this block to one monotonic deadline.

    Ambient on purpose, and the only thing about a read that is: a deadline is
    a property of the CALL, and every parse below it — the raw, the log, a
    digest — has to answer to the same one. Nothing takes a deadline argument
    that this could contradict. What a read is reading, by contrast, travels as
    an argument: an ambient source would let a call name one file and read
    another.
    """
    token = _analysis_deadline.set(deadline)
    try:
        yield
    finally:
        _analysis_deadline.reset(token)


def run_selectors_beside_path(
    *,
    run_index: int = 0,
    case_id: str | None = None,
    other_raw: str | None = None,
    case_ids: bool = False,
) -> str | None:
    """Why a read of the results file at a path refuses these selectors, or
    None when it takes them.

    The one answer for every reader that takes a path or a job_id:
    ``inspect(kind="results")``, ``plot_waveform``, ``analyze_results`` and
    the Python API's ``load_raw``, ``measurements`` and ``device_op_points``.
    A path names one file, while ``run_index``, ``case_id``, ``other_raw``
    and ``analyze_results``' ``runs={case_ids}`` (``case_ids``) choose within
    a job; beside a path the file would be read as though they had not been
    sent. ``run_index`` 0 is its default and chooses nothing.
    ``analyze_results``' list of runs is not refused: a run a path does not
    hold is reported missing, as for a job.
    """
    sent = [
        name
        for name, chosen in (
            ("run_index", run_index != 0),
            ("case_id", case_id is not None),
            ("other_raw", other_raw is not None),
            ("runs={case_ids}", case_ids),
        )
        if chosen
    ]
    if not sent:
        return None
    one = len(sent) == 1
    names = sent[0] if one else f"{', '.join(sent[:-1])} and {sent[-1]}"
    return (
        f"{names} {'chooses' if one else 'choose'} within a job and "
        f"{'needs' if one else 'need'} its job_id, but path names one results file: "
        f"leave {'it' if one else 'them'} out to read that file, or pass the job's "
        "job_id instead of path."
    )


def resolve_experiment_run(
    job_id: str,
    state: SessionState,
    *,
    run_index: int | None = None,
    case_id: str | None = None,
    require_raw: bool = True,
) -> RunContext:
    """Resolve a produced case from any experiment job whose runs are terminal."""
    job = resolve_job(job_id, state)
    return experiment_run_context(
        job, state, run_index=run_index, case_id=case_id, require_raw=require_raw
    )


def _no_raw_result(job_id: str, case: ExperimentCase) -> str:
    """Why ``case`` has no raw result to read (``failure_summary``), and the
    log it left when there is one."""
    text = (
        f"Experiment case {case.case_id!r} of job {job_id!r} did not produce a raw "
        f"result: {failure_summary(case)}"
    )
    if case.log_file is not None:
        text += f". Its log is {case.log_file}"
    return text


def partial_raw_observation(case: ExperimentCase, held: dict[str, Any]) -> dict[str, Any]:
    """What a read of a stopped case's kept raw says of it: every value ends
    at the last point the raw was cut to. ``held`` is ``partial_raw(case)``."""
    stop = case.failure_code or case.status
    reached = points_reached(held.get("points"), held.get("axis"), held.get("last_axis_value"))
    return {
        "code": "partial_raw",
        "kind": "coverage",
        "detail": (
            f"Case {case.case_id} was stopped ({stop}) before its run ended; its raw was "
            f"kept with its first {reached}, and every value read from it ends there."
        ),
        "evidence": {"case_id": case.case_id, "stop": stop, **held},
    }


def runs_unready(job: ExperimentJob) -> str | None:
    """Why ``job``'s runs cannot be read yet, or None once they can.

    The one readiness rule for everything that reads a job's runs by its id:
    ``analyze_results``, ``inspect``, ``plot_waveform``, the Python API's
    reads and ``jobs(place_beside_sheet)``. A job's runs are read once every
    run has finished (``job_lifecycle.runs_done``), which is the moment
    ``jobs(wait, wait_for="runs")`` returns, and not before: a case read while
    its siblings still run would be the answer to a different question than
    the job's. Which of them can be read is then the case's own answer
    (``case_unreadable``).
    """
    if runs_done(job):
        return None
    return f"Experiment job {job.job_id!r} has no readable runs yet (status={job.status!r})"


def case_unreadable(case: ExperimentCase) -> str | None:
    """Why a read of ``case``, of a job whose runs are done, takes no raw, or
    None when it does: the case produced one, or was stopped and kept what its
    run wrote (``partial_raw``). The answer is the case's failure summary."""
    readable = case.status == "produced" or partial_raw(case) is not None
    return None if readable else failure_summary(case)


def experiment_run_context(
    job: ExperimentJob,
    state: SessionState,
    *,
    run_index: int | None = None,
    case_id: str | None = None,
    require_raw: bool = True,
    other_raw: OtherRawKind | None = None,
) -> RunContext:
    """``resolve_experiment_run`` for a caller already holding the job.

    Carries the producing simulator and case provenance into the explicit
    source passed to each reader, independently of session defaults.
    Log readers can opt out of requiring RAW; case readiness is unchanged.
    ``other_raw`` reads the case's other raw of that kind in place of its raw,
    as its run row's ``other_raws`` lists it, and raises
    :class:`OtherRawNotWritten` when the run wrote none.
    """
    job_id = job.job_id
    # Per-case readiness is still gated case by case below.
    unready = runs_unready(job)
    if unready is not None:
        raise ResultError(unready)
    if case_id is None and run_index is None:
        run_index = 0
    matches = [
        case
        for case in job.cases
        if (case_id is not None and case.case_id == case_id)
        or (case_id is None and case.run_index == run_index)
    ]
    if not matches:
        selector = f"case_id={case_id!r}" if case_id is not None else f"run_index={run_index}"
        raise ResultError(f"Experiment job {job_id!r} has no case matching {selector}")
    return case_run_context(job, matches[0], state, require_raw=require_raw, other_raw=other_raw)


def case_run_context(
    job: ExperimentJob,
    case: ExperimentCase,
    state: SessionState,
    *,
    require_raw: bool = True,
    other_raw: OtherRawKind | None = None,
) -> RunContext:
    """``experiment_run_context`` for a caller already holding the case, of a
    job whose runs it has checked are done (``runs_unready``). A caller reading
    every case of a job takes each this way rather than finding it again by
    its id."""
    job_id = job.job_id
    if case_unreadable(case) is not None or (require_raw and case.raw_file is None):
        raise ResultError(_no_raw_result(job_id, case))
    raw = case.raw_file
    if other_raw is not None:
        if raw is None or other_raw not in case.other_raws:
            raise OtherRawNotWritten(
                f"Experiment case {case.case_id!r} of job {job_id!r} wrote no {other_raw} "
                f"raw; its run row's other_raws lists {sorted(case.other_raws) or 'none'}",
                show_hint=False,
            )
        raw = other_raw_path(raw, other_raw)
    held = partial_raw(case)
    identity: dict[str, Any] = {
        "case_id": case.case_id,
        "run_index": case.run_index,
        "assignments": dict(case.assignments),
        "circuit": case.circuit,
        "deck_sha256": case.deck_sha256,
        "step_index": case.step_index,
        "step_values": dict(case.step_values),
    }
    if case.native_statistics is not None:
        identity["native_statistics"] = case.native_statistics.public()
    if other_raw is not None:
        identity["other_raw"] = other_raw
    dialect = dialect_for_job(job, state)
    recorded = case.log_file or case.raw_file
    if recorded is not None:
        console = recorded.with_suffix(".exe.log")
    elif job.output_folder is not None and case.run_token:
        console = job.output_folder / f"{case.run_token}.exe.log"
    else:
        console = None
    return RunContext(
        raw=raw,
        log=case.log_file,
        netlist=case.staged_deck,
        circuit_path=case.circuit_path,
        dialect=dialect,
        identity=identity,
        console=console,
        simulator_version=case.simulator_version,
        observations=(partial_raw_observation(case, held),) if held is not None else (),
        producer=run_producer(job, case),
    )


def resolve_analysis_source(
    state: SessionState,
    *,
    raw_file: str | None = None,
    log_file: str | None = None,
    plot_index: int = 0,
    dialect: str | None = None,
) -> AnalysisSource:
    """Resolve the source a direct ``raw_file``/``log_file`` call reads.

    The caller-path route: every path here is untrusted input and goes through
    ``safe_path``. A caller that already resolved a run uses ``source_for_run``
    instead — this one deliberately cannot reach a job's artifacts. There is no
    job-addressed form: every job is an experiment and an experiment's runs are
    case-addressed, so a caller naming a job resolves the case first.
    """
    if raw_file:
        return source_for_raw_path(
            state.resolve_path(str(raw_file)),
            state,
            plot_index=plot_index,
            dialect=dialect,
        )
    if log_file:
        log = state.resolve_path(str(log_file))
        return AnalysisSource(
            raw=None,
            log=log,
            netlist=None,
            dialect=None,
            identity=None,
            trusted_job_artifact=False,
            explicit_dialect=dialect,
            plot_index=plot_index,
            console=log.with_suffix(".exe.log"),
        )
    raise ResultError("Provide one analysis source: a path or a job_id")


def dialect_for_job(job: ExperimentJob, state: SessionState) -> str | None:
    """Raw dialect for the simulator ``job`` actually ran on.

    A per-run simulator override can differ from the session default (and a
    persisted job may be read back under a different default), so the job's own
    recorded simulator wins. Resolved from the recorded name string, so a
    persisted ngspice job read back with only LTspice installed still parses
    with the ngspice dialect — the producing simulator need not remain
    configured. An unrecorded producer has no dialect evidence; the captured
    writer or an explicit caller selection must determine it instead.
    """
    simulator = getattr(job, "simulator", None)
    if simulator:
        return dialect_for_simulator_name(simulator)
    return None


def run_producer(job: ExperimentJob, case: ExperimentCase) -> Producer:
    """Who wrote a case's log: the simulator the job recorded, the build the
    case's run named in its own output, and the program the job launched.

    The one answer for a case, so the classification stored on its run row when
    the run ended and a later read of its log (the ``summary`` recipe) choose the
    same rules (``log_conditions.rules_for``).
    """
    executable = job.simulator_executable
    return Producer(
        family=dialect_for_simulator_name(job.simulator) if job.simulator else None,
        build=case.simulator_version,
        program=executable.path if executable is not None else None,
    )


def source_producer(
    source: AnalysisSource, *, read_dialect: str | None = None, reported: str | None = None
) -> Producer:
    """Who wrote ``source``'s log: a job case's recorded producer
    (``run_producer``). For a bare path, the dialect its caller named or, when
    it named none, the one its raw's header shows (``read_dialect``, only with
    evidence), and the build its files name (``reported``, from
    ``reported_build``); a bare path records no program."""
    if source.producer is not None:
        return source.producer
    return Producer(
        family=source.dialect or source.explicit_dialect or read_dialect,
        build=source.simulator_version or reported,
        program=None,
    )


async def answer_measurements(
    source: AnalysisSource, table: MeasurementsOutput, *, producer: Producer | None = None
) -> MeasurementsOutput:
    """``source``'s log measurements ``table`` with the field holding each
    one's result named (``log_parser.answered_measurements``), from who wrote
    the log and the run's deck where it is known. Every reader of a
    measurement reads it through here. ``producer`` is the one a caller has
    already resolved (the summary's, read with the build its files name);
    otherwise it is :func:`source_producer`'s, as a log-only reader opens no
    source file in this process: a bare path then names no build, and its log
    is read with every recorded build's rules (``log_conditions.rules_for``),
    which agree on every recorded form."""
    cards = await asyncio.to_thread(measurement_cards, source.netlist)
    return answered_measurements(table, producer or source_producer(source), cards)


def reported_version(
    state: SessionState,
    executable: SimulatorExecutable | None,
) -> tuple[str, dict[str, str]] | None:
    """The build the latest run on this same executable reported, and which run.

    Read from the jobs this session holds, its own and the recent ones loaded
    at startup, so it is a run's own output rather than a probe: asking the
    executable would launch the simulator. None until a run on this build has
    finished and named itself.
    """
    if executable is None:
        return None
    # A snapshot: a worker thread reads this too, while the loop adds jobs.
    jobs = tuple(state.all_jobs.values())
    latest = max(
        (
            (case.completed_at or job.started_at, job, case)
            for job in jobs
            if same_executable(job.simulator_executable, executable)
            for case in job.cases
            if case.simulator_version
        ),
        key=lambda run: run[0],
        default=None,
    )
    if latest is None:
        return None
    _, job, case = latest
    assert case.simulator_version is not None
    return case.simulator_version, {"job_id": job.job_id, "case_id": case.case_id}


def cp1252_ltspice(state: SessionState, executable: SimulatorExecutable | None) -> str | None:
    """The evidence that ``executable`` is an LTspice before 24, which decodes
    decks as cp1252, or None.

    LTspice XVII and earlier read a deck as cp1252, so a UTF-8 micro sign
    (C2 B5) reaches them as the two characters ``Âµ`` and loses its scale;
    LTspice 24 and later read the UTF-8 they write. The same builds refuse
    ``.options logopinfo`` and name subcircuit instances without their X, so
    every decision about them asks here, or passes the same two facts to
    ``simulator_build.ltspice_before_24``: the executable's own name
    (``XVIIx64.exe``), and the build the latest run on it reported, the one
    ``inspect(kind="capabilities")`` reports. Neither launches the simulator.
    Any other simulator answers None. The answer names the evidence, for a
    finding to cite.
    """
    if executable is None:
        return None
    # The name answers without the scan of every recorded run that the
    # reported build takes.
    named = ltspice_before_24(executable.path, None)
    if named is not None:
        return named
    return ltspice_before_24(executable.path, reported_build_of(state, executable))


def reported_build_of(state: SessionState, executable: SimulatorExecutable | None) -> str | None:
    """The build the latest run on ``executable`` reported (``reported_version``),
    without which run did."""
    reported = reported_version(state, executable)
    return reported[0] if reported is not None else None


def session_reader(state: SessionState) -> SemanticProfile:
    """Who reads a deck the session checks without running it
    (``verify_circuit``, ``inspect``'s net query): the family a
    ``run_experiments`` call with no override would run it on, that family's
    executable (``deck_producer``), and ngspice's mode. Blocking: call off
    the loop.

    A family with no reader profile (QSPICE, Xyce), or none detected, is read
    as LTspice reads a deck, as every deck was before the readers were told
    who reads it."""
    from ltspice_mcp.lib.hierarchy import SUPPORTED_FAMILIES, SemanticProfile
    from ltspice_mcp.lib.simulator import DEFAULT_FAMILY, family_for, run_ngbehavior

    family = family_for(None, state.default_simulator)
    if family not in SUPPORTED_FAMILIES:
        family = DEFAULT_FAMILY
    # The executable a run with no override uses: the default, when it is of
    # this family, else the session's executable of the family.
    simulator = (
        state.default_simulator
        if family_for(None, state.default_simulator) == family
        else state.available_simulators.get(family)
    )
    return SemanticProfile.of(
        deck_producer(state, family, simulator),
        run_ngbehavior(simulator) if family == "ngspice" else None,
    )


def deck_producer(state: SessionState, family: str | None, simulator: type | None) -> Producer:
    """Who will read a deck run on ``simulator`` of ``family``: the family,
    the build the latest run on the same executable reported
    (``reported_build_of``), and the program it launches. The one answer for
    a deck's reader before it runs, which ``hierarchy.SemanticProfile.of``
    takes, as ``run_producer`` is for a run that ended. With no simulator
    (an offline query) only the family is known. Blocking (the executable is
    identified): call off the loop."""
    if simulator is None:
        return Producer(family=family, build=None, program=None)
    return _producer_of(state, family, executable_identity(simulator))


def _producer_of(
    state: SessionState, family: str | None, identity: SimulatorExecutable | None
) -> Producer:
    program = identity.path if identity is not None else None
    # Only an LTspice's build changes how its names are read, and its file
    # name, where it says, answers without the scan of every recorded run.
    build = None
    if family == "ltspice" and ltspice_before_24(program, None) is None:
        build = reported_build_of(state, identity)
    return Producer(family=family, build=build, program=program)


# Absolute worker deadline; the supervisor confirms owned-tree cleanup before
# RAW/log loading returns. Path authorization metadata still happens off-loop.
RAW_PARSE_TIMEOUT_S = 120.0

# Paths identifying residual threaded work that timed out, with monotonic retry
# deadlines. A timeout stops waiting but cannot terminate the executor thread;
# cooldown limits repeated unfinished work in the shared thread pool. The path
# can identify resident computation, so it does not imply a corrupt file or
# report whether the timed-out thread is still running. Read/written only on the
# event loop with no await between check and store, so no lock is needed.
_wedged_raw_paths: dict[Path, float] = {}


async def bounded_parse(
    path: Path,
    thunk: Callable[[], T],
    *,
    timeout_s: float = RAW_PARSE_TIMEOUT_S,
) -> T:
    """Run residual result work off the loop with a deadline and path cooldown.

    Callers compute resident summaries, aggregate decoded measurements (with
    optional trusted staged-deck reads), or hash completed output artifacts.
    RAW/log capture and dependency decoding use the contained parser service.
    This deadline stops waiting; it cannot forcibly stop an executor thread.
    The cooldown limits retries of timed-out work in the shared thread pool.
    """
    loop = asyncio.get_running_loop()
    now_mono = loop.time()
    cooldown_s = timeout_s
    call_deadline = _analysis_deadline.get()
    if call_deadline is not None:
        timeout_s = min(timeout_s, max(0.0, call_deadline - now_mono))
    if timeout_s <= 0:
        raise AnalysisDeadlineExceeded(
            f"Analysis work for {path.name} exceeded the analysis item deadline"
        )
    wedged_until = _wedged_raw_paths.get(path)
    if wedged_until is not None:
        if now_mono < wedged_until:
            raise AnalysisDeadlineExceeded(
                f"Analysis work for {path.name} recently exceeded its deadline; "
                f"retries are paused for {wedged_until - now_mono:.0f}s more "
                "to limit unfinished work in the thread pool."
            )
        del _wedged_raw_paths[path]
    try:
        task = asyncio.create_task(asyncio.to_thread(thunk))
        try:
            return await asyncio.wait_for(asyncio.shield(task), timeout_s)
        except TimeoutError:
            # The worker cannot be killed.  Shielding keeps its completion
            # independent of the timeout and this callback consumes a late
            # exception so it cannot become an unhandled task warning.
            task.add_done_callback(lambda done: None if done.cancelled() else done.exception())
            raise
    except TimeoutError:
        _wedged_raw_paths[path] = loop.time() + cooldown_s
        raise AnalysisDeadlineExceeded(
            f"Analysis work for {path.name} exceeded {timeout_s:.3g}s. "
            "Its executor thread cannot be forcibly stopped; retries for "
            f"this path are paused for {cooldown_s:.0f}s."
        ) from None


def _raw_deadline() -> float:
    deadline = time.monotonic() + RAW_PARSE_TIMEOUT_S
    shared = _analysis_deadline.get()
    if shared is not None:
        deadline = min(deadline, shared)
    return deadline


def _resident_artifacts(
    source: AnalysisSource, *, require_raw: bool, deadline: float
) -> ParsedArtifacts | None:
    """Reuse explicit evaluation facts without extending their read deadline."""
    captured = source.captured
    if captured is None or (require_raw and captured.raw is None):
        return None
    if (source.identity or {}).get("snapshot_id") != captured.snapshot_id:
        raise ResultError("Resident artifacts do not match the analysis source snapshot")
    if time.monotonic() >= deadline:
        raise AnalysisDeadlineExceeded("Analysis exceeded its deadline")
    return captured


async def load_artifacts(
    source: AnalysisSource, state: SessionState, *, require_raw: bool
) -> ParsedArtifacts:
    """Return validated shared capture identity and resident RAW/log facts.

    Cancellation waits for the shared parser's owned-tree cleanup. Selection
    is left to the caller; snapshot_id binds all captured artifact roles.
    """
    deadline = _raw_deadline()
    captured = _resident_artifacts(source, require_raw=require_raw, deadline=deadline)
    if captured is not None:
        return captured
    cancel = threading.Event()
    context = contextvars.copy_context()

    def run() -> ParsedArtifacts:
        return context.run(
            _parse_artifacts_sync,
            source,
            state,
            deadline=deadline,
            cancel=cancel,
            require_raw=require_raw,
        )

    future = asyncio.get_running_loop().run_in_executor(None, run)
    cancelled = False
    while not future.done():
        try:
            await asyncio.shield(future)
        except asyncio.CancelledError:
            cancelled = True
            cancel.set()
        except Exception:
            break
    try:
        artifacts = future.result()
    except Exception as exc:
        if cancelled and not isinstance(exc, ParserCleanupError):
            raise asyncio.CancelledError from exc
        raise
    if cancelled:
        raise asyncio.CancelledError
    return artifacts


async def load_raw(source: AnalysisSource, state: SessionState) -> DecodedRaw:
    """Load a selected resident plot and its same-capture log facts."""
    return select_raw(await load_artifacts(source, state, require_raw=True), source.plot_index)


def load_raw_sync(source: AnalysisSource, state: SessionState) -> DecodedRaw:
    """Blocking source-based counterpart for callers already off the loop."""
    deadline = _raw_deadline()
    captured = _resident_artifacts(source, require_raw=True, deadline=deadline)
    if captured is not None:
        return select_raw(captured, source.plot_index)
    return _parse_raw_sync(source, state, deadline=deadline)


async def load_logs(source: AnalysisSource, state: SessionState) -> DecodedLog:
    """Load complete log facts, independent of RAW payload validity."""
    return (await load_artifacts(source, state, require_raw=False)).logs


def load_logs_sync(source: AnalysisSource, state: SessionState) -> DecodedLog:
    """Blocking counterpart using the same capture, cache and cleanup path."""
    deadline = _raw_deadline()
    captured = _resident_artifacts(source, require_raw=False, deadline=deadline)
    if captured is not None:
        return captured.logs
    return _parse_logs_sync(source, state, deadline=deadline)


@dataclass(frozen=True)
class Signal:
    """A requested signal resolved against one raw's trace list.

    ``name`` is what a reply reports: the trace as the simulator wrote it, or,
    for a node pair, ``V(a,b)`` with each node spelled as the raw spells it.
    The value is ``plus - minus``; either side is None when it is ground, so a
    plain trace is its own ``plus`` with no ``minus``.
    """

    name: str
    plus: str | None
    minus: str | None = None

    @property
    def trace(self) -> str:
        """A trace this signal is read from; both sides of a pair share its unit."""
        return self.plus or self.minus or self.name

    def wave(self, raw: RawData, step: int) -> np.ndarray:
        """One step of this signal. Both traces of a pair come from ``raw`` at
        ``step``, so they share that step's axis sample for sample."""
        if self.minus is None:
            assert self.plus is not None
            return np.asarray(raw.get_wave(self.plus, step=step))
        minus = np.asarray(raw.get_wave(self.minus, step=step))
        if self.plus is None:
            return -minus
        plus = np.asarray(raw.get_wave(self.plus, step=step))
        if plus.shape != minus.shape:
            # One step of one raw has one axis; a shape mismatch is a corrupt
            # file, and subtracting misaligned samples would hide it.
            raise ResultError(
                f"{self.name}: {self.plus} has {plus.size} samples at step {step} but "
                f"{self.minus} has {minus.size}; the raw file is likely corrupt."
            )
        return plus - minus


def _available_signals(raw: RawData) -> str:
    """The first traces of ``raw``, for an error that names what is there."""
    trace_names = raw.get_trace_names()
    available = ", ".join(trace_names[:10])
    if len(trace_names) > 10:
        available += f", ... ({len(trace_names)} total)"
    return available


def read_signal(raw: DecodedRaw | DecodedPlot, signal: str) -> NameRead:
    """What ``signal`` reads of ``raw``'s selected plot, or why it reads none
    (``trace_names.read_in_plot``, which the parser worker asks too)."""
    return read_in_plot(raw.header, signal)


def resolve_signal(raw: DecodedRaw | DecodedPlot, signal: str) -> Signal:
    """Resolve ``signal`` against ``raw``: one trace, or a node pair ``V(a,b)``.

    A trace the raw actually carries wins, including one literally named
    ``V(a,b)``; otherwise a node pair reads as ``V(a) - V(b)``, each side
    resolved like any single trace (``read_signal``). Read the data with
    :meth:`Signal.wave`: a pair's ``name`` is not a trace of the raw. A name
    that reads nothing is refused with why (``trace_names.why_not``).
    """
    read = read_signal(raw, signal)
    if read.refused is None:
        return Signal(read.name, read.plus, read.minus)
    raise signal_refused(raw, signal, read)


def signal_refused(raw: DecodedRaw | DecodedPlot, signal: str, read: NameRead) -> ResultError:
    """The error for a ``signal`` that reads no trace of ``raw`` (``read``)."""
    listed = (
        ""
        if read.refused in ("ground_twice", "noise_pair")
        else f" Available signals: {_available_signals(raw)}."
    )
    return ResultError(
        why_not(read, signal, raw.header, listed=listed),
        show_hint=read.refused not in ("expression", "noise_pair"),
        python_route=read.refused == "expression",
    )


def is_node_pair(signal: str) -> bool:
    """Whether ``signal`` is spelled as a node-pair voltage ``V(a,b)``."""
    return NODE_PAIR_RE.fullmatch(signal) is not None


def validate_step(raw: RawData, step: int) -> None:
    """Validate that a step index exists in a raw result."""
    step_count = get_step_count(raw)
    if step < 0 or step >= step_count:
        raise ResultError(f"Step {step} out of range. Valid range: 0 to {step_count - 1}")


@dataclass(frozen=True)
class CircuitJobSummary:
    """What one circuit's records add up to, for either circuit listing.

    ``spice://recent`` and ``jobs(action="list")`` are two views of the same
    join — the recent-circuits index against this store's experiment records —
    and everything below is derived from the job list alone, so the two cannot
    come to different numbers for the same records. What they do differ on is
    which jobs they hand in and how they name the circuit, and each of those
    three differences is deliberate:

    * **Pruning.** The resource asks ``recent.load(prune_missing=True)``: it
      is the index's own view and takes the chance to drop entries whose file
      is gone. ``jobs(list)`` does not prune, because listing must not rewrite
      a user-global index as a side effect, and a deleted circuit's recorded
      jobs are still addressable — ``exists`` is the fact it reports instead.
    * **Path identity.** ``jobs(list)`` resolves and de-duplicates paths,
      because it has to match a caller's ``circuit`` argument and two index
      entries can spell one file. The resource reports the entry as the index
      holds it, because that is what the index holds.
    * **Live jobs.** ``jobs(list)`` passes ``prefer=`` a registry snapshot
      taken on the event loop, so this process's running jobs are counted from
      memory rather than from a record whose last transitions may still be in
      flight. The resource read runs on a worker thread, where the registry
      may not be touched, so it reports what is on disk.
    """

    exists: bool
    status_counts: dict[str, int]
    interrupted_job_ids: list[str]
    total_jobs: int
    total_runs: int


def summarize_circuit_jobs(
    circuit_path: Path,
    jobs: Sequence[ExperimentJob],
) -> CircuitJobSummary:
    """Count one circuit's job records. See :class:`CircuitJobSummary`."""
    counts: dict[str, int] = {}
    interrupted: list[str] = []
    for job in jobs:
        counts[job.status] = counts.get(job.status, 0) + 1
        if job.status == "interrupted":
            interrupted.append(job.job_id)
    return CircuitJobSummary(
        exists=circuit_path.exists(),
        status_counts=counts,
        interrupted_job_ids=sorted(set(interrupted)),
        total_jobs=len(jobs),
        total_runs=sum(job.completeness.expanded for job in jobs),
    )


def collect_recent_circuits(working_dir: Path) -> list[dict[str, Any]]:
    """List recently-touched circuits with their persisted-job summaries.

    A circuit's jobs come from this working directory's store, through the
    per-circuit index, so a circuit last run by another session in the same
    directory still reports its jobs here — and one last run from a different
    working directory reports none.

    The counters are :func:`summarize_circuit_jobs`, shared with
    ``jobs(action="list")``; that class documents where the two views
    deliberately differ.

    Blocking — ``recent.load`` polls a cross-process file lock (up to 10 s)
    and each summary reads the store's JSON records; all reads (the prune
    rewrite is atomic), so safe under cancellation. Coroutine callers must
    run this via ``asyncio.to_thread``; the synchronous MCP resource router
    calls it directly (already off the loop).
    """
    from ltspice_mcp.lib import experiment_store

    entries = recent.load(prune_missing=True)
    circuits: list[dict[str, Any]] = []
    for entry in entries:
        raw_path = entry.get("path")
        if not isinstance(raw_path, str):
            continue
        jobs, _ = experiment_store.load_jobs_for_circuit(Path(raw_path), working_dir)
        summary = summarize_circuit_jobs(Path(raw_path), jobs)
        circuits.append(
            {
                "path": raw_path,
                "exists": summary.exists,
                "total_jobs": summary.total_jobs,
                "total_runs": summary.total_runs,
                "status_counts": summary.status_counts,
                "interrupted_job_ids": summary.interrupted_job_ids,
                "last_touched": entry.get("last_touched"),
            }
        )
    return circuits


# Substrings that indicate the per-run OP convergence didn't take the
# direct path. Presence doesn't prove the result is wrong, but it does
# mean the bias point may have landed on a degenerate solution that
# yields garbage AC results — worth surfacing alongside aggregate stats.
_CONVERGENCE_FLAG_SUBSTRINGS: tuple[str, ...] = (
    "direct newton iteration failed",
    "gmin stepping",
    "source stepping",
    "no convergence",
    "singular matrix",
    "time step too small",
)


# Reply cap for convergence_warnings. The full list stays cached on the
# BatchJob, but a reply re-sends it on EVERY terminal read of the job — a
# 400-run Monte Carlo where most runs trip a gmin/source-stepping marker would
# re-ship ~400 near-identical objects per poll.
_CONVERGENCE_STRUCTURED_CAP = 25


def extract_netlist_info(file_path: Path) -> dict[str, Any]:
    """Extract structured netlist data via spice_lex + ``lib.encoding``.

    Honours BOMs and UTF-16-no-BOM via ``read_spice_text``; walks
    truncated/hierarchical netlists via the spice_lex foundation and
    surfaces its warnings (e.g. unclosed ``.SUBCKT``) instead of raising.
    Component values come from ``InstanceLine`` views — uniform across
    behavioural sources, B-source ``V=expr`` forms, and active devices. The
    components are the cards the simulator reads (``spice_lex.lex_deck``):
    neither the title line nor a card after ``.END``.
    """
    from ltspice_mcp.errors import NetlistError
    from ltspice_mcp.lib.encoding import read_spice_text
    from ltspice_mcp.lib.spice_lex import lex_deck
    from ltspice_mcp.lib.spice_lex_views import (
        InstanceLine,
        body_has_stray_kv_remnant,
    )

    try:
        content = read_spice_text(file_path)
    except FileNotFoundError as e:
        raise NetlistError(f"File not found: {file_path}") from e
    result = lex_deck(content)
    comp_list: list[dict[str, Any]] = []
    for card in result.cards:
        if card.kind != "instance" or not card.name:
            continue
        if body_has_stray_kv_remnant(card.body):
            comp_list.append({"reference": card.name, "value": "<unparseable>"})
            continue
        try:
            inst = InstanceLine.from_card(card)
        except Exception as e:
            logger.debug(
                "extract_netlist_info: %s failed to parse via InstanceLine: %s",
                card.name,
                e,
            )
            comp_list.append({"reference": card.name, "value": "<unparseable>"})
            continue
        comp_list.append({"reference": card.name, "value": inst.display_value()})

    out: dict[str, Any] = {
        "file": str(file_path),
        "type": "netlist",
        "content": content,
        "components": comp_list,
    }
    if result.warnings:
        out["warnings"] = list(result.warnings)
    return out
