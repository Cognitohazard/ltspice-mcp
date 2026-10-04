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
import re
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar

import numpy as np
from spicelib import AscEditor, SpiceEditor

from ltspice_mcp.errors import AnalysisDeadlineExceeded, JobNotFoundError, ResultError
from ltspice_mcp.lib import recent
from ltspice_mcp.lib.decoded_log import DecodedLog
from ltspice_mcp.lib.decoded_raw import DecodedRaw, RawData
from ltspice_mcp.lib.experiment_types import ExperimentJob
from ltspice_mcp.lib.job_lifecycle import runs_terminal
from ltspice_mcp.lib.log_parser import LogDiagnostics
from ltspice_mcp.lib.netlist_graph import GROUND_ALIASES
from ltspice_mcp.lib.parsed_artifacts import ParsedArtifacts
from ltspice_mcp.lib.parser_service import ParserCleanupError, select_raw
from ltspice_mcp.lib.parser_service import load_artifacts_sync as _parse_artifacts_sync
from ltspice_mcp.lib.parser_service import load_logs_sync as _parse_logs_sync
from ltspice_mcp.lib.parser_service import load_raw_sync as _parse_raw_sync
from ltspice_mcp.lib.pathutil import resolve_safe_path
from ltspice_mcp.lib.raw_parser import (
    detect_sim_type,
    get_step_count,
    is_noise_analysis,
)
from ltspice_mcp.lib.simulator import dialect_for_simulator_name
from ltspice_mcp.lib.simulator_build import (
    SimulatorExecutable,
    is_cp1252_ltspice_build,
    is_cp1252_ltspice_executable,
    same_executable,
)
from ltspice_mcp.state import SessionState

logger = logging.getLogger(__name__)

Editor = AscEditor | SpiceEditor
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
    # Resident facts belong to one evaluation, between fresh drift checks.
    captured: ParsedArtifacts | None = field(default=None, repr=False, compare=False)

    @classmethod
    def for_raw(cls, raw_path: Path) -> AnalysisSource:
        """The candidate log and console companions of a bare ``.raw`` path.

        For a read that has only a path to go on. ``log`` is always a concrete
        path (existence not guaranteed); ``netlist``, ``dialect`` and
        ``identity`` are null because a bare path names no producing run. A read
        that DOES know the run gets its source from ``source_for_run`` or
        ``resolve_analysis_source`` instead — those carry the deck and identity
        this cannot.
        """
        return cls(
            raw=raw_path,
            log=raw_path.with_suffix(".log"),
            netlist=None,
            dialect=None,
            identity=None,
            trusted_job_artifact=False,
            console=raw_path.with_suffix(".exe.log"),
        )


def source_for_raw_path(
    raw: Path, state: SessionState, *, plot_index: int = 0, dialect: str | None = None
) -> AnalysisSource:
    """The source an already-validated caller-supplied ``.raw`` path resolves to.

    The path must already have passed ``safe_path``. Companion candidates are
    authorized independently on load; only the contained capture worker checks
    presence and writer evidence. Session simulator defaults are not provenance.
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
    )


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


def experiment_run_context(
    job: ExperimentJob,
    state: SessionState,
    *,
    run_index: int | None = None,
    case_id: str | None = None,
    require_raw: bool = True,
) -> RunContext:
    """``resolve_experiment_run`` for a caller already holding the job.

    Carries the producing simulator and case provenance into the explicit
    source passed to each reader, independently of session defaults.
    Log readers can opt out of requiring RAW; case readiness is unchanged.
    """
    job_id = job.job_id
    # Per-case readiness is still gated case by case below.
    if not runs_terminal(job.status):
        raise ResultError(
            f"Experiment job {job_id!r} has no readable runs yet (status={job.status!r})"
        )
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
    case = matches[0]
    if case.status != "produced" or (require_raw and case.raw_file is None):
        raise ResultError(
            f"Experiment case {case.case_id!r} did not produce a raw result "
            f"(status={case.status!r})"
        )
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
    dialect = dialect_for_job(job, state)
    recorded = case.log_file or case.raw_file
    if recorded is not None:
        console = recorded.with_suffix(".exe.log")
    elif job.output_folder is not None and case.run_token:
        console = job.output_folder / f"{case.run_token}.exe.log"
    else:
        console = None
    return RunContext(
        raw=case.raw_file,
        log=case.log_file,
        netlist=case.staged_deck,
        circuit_path=case.circuit_path,
        dialect=dialect,
        identity=identity,
        console=console,
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
            resolve_safe_path(str(raw_file), state.allowed_paths()),
            state,
            plot_index=plot_index,
            dialect=dialect,
        )
    if log_file:
        log = resolve_safe_path(str(log_file), state.allowed_paths())
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
    raise ResultError("Provide one analysis source: raw_file, log_file, or job_id")


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
    latest = max(
        (
            (case.completed_at or job.started_at, job, case)
            for job in state.all_jobs.values()
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
    """The evidence that ``executable`` is an LTspice that decodes decks as cp1252.

    LTspice XVII and earlier read a deck as cp1252, so a UTF-8 micro sign
    (C2 B5) reaches them as the two characters ``Âµ`` and loses its scale;
    LTspice 24 and later read the UTF-8 they write. Which one an executable is
    shows in its own name (``XVIIx64.exe``), or in the build the latest run on
    it reported, the one ``inspect(kind="capabilities")`` reports. Neither
    launches the simulator. Any other simulator answers None. The answer names
    the evidence, for a finding to cite.
    """
    if executable is None:
        return None
    if is_cp1252_ltspice_executable(executable.path):
        return executable.path
    reported = reported_version(state, executable)
    if reported is not None and is_cp1252_ltspice_build(reported[0]):
        return f"{reported[0]} ({executable.path})"
    return None


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


# A ``dev.param`` operating-point shorthand: everything before the LAST dot is
# the device, so a flattened subcircuit path ('m.x1.mn.gm') parses too.
DEV_PARAM_RE = re.compile(r"([a-z][\w.]*)\.([a-z]\w*)")


def device_param_forms(signal: str) -> list[str]:
    """The result names a ``dev.param`` operating-point shorthand can address.

    ngspice writes a device parameter bare (``@m1[gm]``), v-wrapped
    (``v(@m1[vth])``) or i-wrapped (``i(@m1[id])``) depending on the quantity,
    and LTspice's ``.log`` block is folded into ``device_op_points`` under the
    bare form. Empty when the name is not a shorthand.

    Public because two readers resolve the shorthand the docs promise —
    ``resolve_signal`` against a raw's trace list, ``analyze_results``
    against an operating-point result — and a second copy of the rule is how
    one of them ends up rejecting a name the other accepts.
    """
    match = DEV_PARAM_RE.fullmatch(signal.lower())
    if match is None:
        return []
    dev, param = match.group(1), match.group(2)
    return [f"@{dev}[{param}]", f"v(@{dev}[{param}])", f"i(@{dev}[{param}])"]


#: A node-pair voltage ``V(a,b)``. SPICE syntax names it, but no simulator
#: writes it as a trace, so it is read as ``V(a) - V(b)``. A node name holds
#: no comma, parenthesis or space, which keeps the split unambiguous.
_NODE_PAIR_RE = re.compile(r"\s*v\(\s*([^\s(),]+)\s*,\s*([^\s(),]+)\s*\)\s*", re.IGNORECASE)

_V_TRACE_RE = re.compile(r"v\((.+)\)", re.IGNORECASE)


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


def _find_trace(raw: RawData, signal: str) -> str | None:
    """The raw's own name for one trace ``signal`` addresses, or None.

    Lookup is case-insensitive: SPICE node names are case-insensitive per
    SPICE conventions, but spicelib preserves the case the simulator wrote.
    LTspice writes ``V(out)`` for transient/AC/DC sweep raws but ``v(onoise)``
    for ``.NOISE`` raws — case-sensitive match would reject the user's
    ``V(onoise)`` even though the data exists.
    """
    trace_names = raw.get_trace_names()
    if signal in trace_names:
        return signal
    sig_lower = signal.lower()
    for name in trace_names:
        if name.lower() == sig_lower:
            return name

    # Resolve cross-simulator / shorthand aliases transparently instead of
    # forcing a guaranteed retry on a deterministic rename:
    #   - noise: LTspice V(onoise)/V(inoise) <-> ngspice onoise_spectrum/
    #     inoise_spectrum, plus bare onoise/inoise shorthand
    #   - hierarchical separator: LTspice ':' (V(X1:mid)) <-> ngspice '.'
    by_lower = {t.lower(): t for t in trace_names}
    candidates: list[str] = []
    for kind in ("onoise", "inoise"):
        if sig_lower in (kind, f"v({kind})", f"{kind}_spectrum"):
            candidates += [f"v({kind})", f"{kind}_spectrum", kind]
    if ":" in sig_lower:
        candidates.append(sig_lower.replace(":", "."))
    if "." in sig_lower:
        candidates.append(sig_lower.replace(".", ":"))

    dev_param = DEV_PARAM_RE.fullmatch(sig_lower)
    candidates += device_param_forms(signal)

    for cand in candidates:
        if cand in by_lower:
            return by_lower[cand]

    # Hierarchical path that drops the device-type letter, e.g. 'x1.mn.gm' for
    # ngspice's '@m.x1.mn[gm]'. Match a unique @<path>[param] whose device path
    # ends with the requested segments; refuse if ambiguous — never guess.
    if dev_param:
        dev, param = dev_param.group(1), dev_param.group(2)
        suffix = "." + dev
        wrapped = re.compile(r"^[vi]?\(?@(.+)\[" + re.escape(param) + r"\]\)?$")
        hits = [
            orig
            for low, orig in by_lower.items()
            if (m := wrapped.match(low)) and (m.group(1) == dev or m.group(1).endswith(suffix))
        ]
        if len(hits) == 1:
            return hits[0]
    return None


def _looks_like_expression(signal: str) -> bool:
    """Whether ``signal`` combines traces rather than naming one.

    True for an operator outside every parenthesis (``V(a)-V(b)``,
    ``2*V(out)``) or a reference nested in another (``abs(V(out))``). A sign
    inside the parentheses is part of a node name — ``V(in-)`` is one trace.
    """
    depth = 0
    for char in signal:
        if char == "(":
            depth += 1
            if depth > 1:
                return True
        elif char == ")":
            depth -= 1
        elif depth == 0 and char in "+-*/^":
            return True
    return False


def _available_signals(raw: RawData) -> str:
    """The first traces of ``raw``, for an error that names what is there."""
    trace_names = raw.get_trace_names()
    available = ", ".join(trace_names[:10])
    if len(trace_names) > 10:
        available += f", ... ({len(trace_names)} total)"
    return available


def _not_found(raw: RawData, signal: str) -> ResultError:
    """The error for a ``signal`` no trace answers to."""
    trace_names = raw.get_trace_names()
    available = _available_signals(raw)
    if _looks_like_expression(signal):
        return ResultError(
            f"Signal '{signal}' not found: it reads as an expression, and a signal "
            "names one trace or a node-pair voltage V(a,b). Available signals: "
            f"{available}. For other trace math, combine the traces in Python: "
            "r = api.load_raw(job_id=..., case_id=...); t = r.axis(step=0); "
            "y = r.trace('V(out)', step=0) * r.trace('I(R1)', step=0); "
            "compute_signal_stats(t, y) gives its time-weighted statistics.",
            show_hint=False,
            python_route=True,
        )
    hint = ""
    sig_lo = signal.lower()
    dev_param = DEV_PARAM_RE.fullmatch(sig_lo)
    trace_lo = {t.lower() for t in trace_names}
    if sig_lo in ("v(onoise)", "v(inoise)") and (
        "onoise_spectrum" in trace_lo or "inoise_spectrum" in trace_lo
    ):
        hint = " (ngspice names noise signals 'onoise_spectrum'/'inoise_spectrum')"
    elif sig_lo in ("onoise_spectrum", "inoise_spectrum") and (
        "v(onoise)" in trace_lo or "v(inoise)" in trace_lo
    ):
        hint = " (LTspice names noise signals 'V(onoise)'/'V(inoise)')"
    elif dev_param or "@" in sig_lo:
        save_target = f"@{dev_param.group(1)}[{dev_param.group(2)}]" if dev_param else signal
        hint = (
            f" Per-device small-signal params (gm/gds/vth/...) come back from "
            f"operating_point(device=...). As a raw trace (for a sweep/plot), ngspice "
            f"needs '.save {save_target}' in the deck; LTspice writes them only to the "
            f".log under '.options logopinfo' (operating_point reads that), not the raw."
        )
    return ResultError(f"Signal '{signal}' not found.{hint} Available signals: {available}")


def _node_pair(raw: RawData, signal: str, plus_node: str, minus_node: str) -> Signal:
    """``V(plus_node, minus_node)`` read from the two node voltages of ``raw``.

    A ground alias (``0``, ``gnd``) that the raw carries no voltage for is
    ground and contributes nothing, so ``V(a,0)`` is ``V(a)``.
    """
    traces = {node: _find_trace(raw, f"V({node})") for node in (plus_node, minus_node)}
    grounded = {
        node for node, trace in traces.items() if trace is None and node.lower() in GROUND_ALIASES
    }
    if minus_node not in grounded and is_noise_analysis(detect_sim_type(raw)):
        raise ResultError(
            f"Signal '{signal}' is a node-pair difference, and a .noise result holds "
            "spectral densities: the difference of two densities is not the noise "
            "between the nodes. Put the pair in the .noise directive, "
            "e.g. .noise V(a,b) <source> ..., and read V(onoise).",
            show_hint=False,
        )
    for node, trace in traces.items():
        if trace is None and node not in grounded:
            raise ResultError(
                f"Signal '{signal}' reads as a node-pair difference, but node voltage "
                f"'V({node})' is not in this result. Available signals: "
                f"{_available_signals(raw)}"
            )
    plus, minus = traces[plus_node], traces[minus_node]
    if minus is None:
        if plus is None:
            raise ResultError(f"Signal '{signal}' names ground twice; it is zero by definition.")
        return Signal(plus, plus)
    name = f"V({_node_spelling(plus, plus_node)},{_node_spelling(minus, minus_node)})"
    return Signal(name, plus, minus)


def _node_spelling(trace: str | None, node: str) -> str:
    """``node`` as the raw spells it inside its ``V(...)`` trace, else as asked."""
    match = _V_TRACE_RE.fullmatch(trace or "")
    return match.group(1) if match else node


def resolve_signal(raw: RawData, signal: str) -> Signal:
    """Resolve ``signal`` against ``raw``: one trace, or a node pair ``V(a,b)``.

    A trace the raw actually carries wins, including one literally named
    ``V(a,b)``; otherwise a node pair reads as ``V(a) - V(b)``, each side
    resolved like any single trace. Read the data with :meth:`Signal.wave`:
    a pair's ``name`` is not a trace of the raw.
    """
    trace = _find_trace(raw, signal)
    if trace is not None:
        return Signal(trace, trace)
    pair = _NODE_PAIR_RE.fullmatch(signal)
    if pair is not None:
        return _node_pair(raw, signal, pair.group(1), pair.group(2))
    raise _not_found(raw, signal)


def is_node_pair(signal: str) -> bool:
    """Whether ``signal`` is spelled as a node-pair voltage ``V(a,b)``."""
    return _NODE_PAIR_RE.fullmatch(signal) is not None


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


# Structured-channel cap for convergence_warnings. The full list stays cached
# on the BatchJob and the text channel already caps its display at 10, but
# structuredContent is re-sent on EVERY terminal check_job/batch_results read
# — a 400-run Monte Carlo where most runs trip a gmin/source-stepping marker
# would re-ship ~400 near-identical objects per poll.
_CONVERGENCE_STRUCTURED_CAP = 25


def asc_component_value(editor: AscEditor, ref: str) -> str:
    """Return a component's primary value from its ``Value`` SYMATTR only.

    Spicelib's ``editor.get_component_value`` concatenates ``Value`` and
    ``Value2`` into a single space-separated string, which then collides
    with the ``attributes: {Value2: ...}`` map that downstream tools also
    surface. Read ``Value`` alone here and let ``Value2`` stay in
    the attributes map without duplication.
    """
    comp = editor.components.get(ref)
    if comp is None:
        # Fall back to spicelib's lookup so the "component not found"
        # error path is spicelib's own.
        return editor.get_component_value(ref)
    val = (comp.attributes or {}).get("Value", "")
    return str(val) if val is not None else ""


def asc_component_attributes(comp: Any) -> dict[str, str]:
    """Non-default SYMATTRs of an .asc component as a plain string dict.

    Filters ``Value``/``InstName`` (surfaced separately) and empty values.
    spicelib stores some attribute values as ``Text`` records rather than
    strings — those are not JSON-serializable and violate the string-typed
    output schemas, so coerce every value to its payload string.
    """
    return {
        k: str(getattr(v, "text", v))
        for k, v in (comp.attributes or {}).items()
        if k not in ("Value", "InstName") and v
    }


def extract_asc_info(editor: AscEditor, file_path: Path) -> dict[str, Any]:
    """Extract structured schematic data from an ``AscEditor``."""
    components = editor.get_components()
    comp_data = []
    for ref in components:
        value = asc_component_value(editor, ref)
        pos, rot = editor.get_component_position(ref)
        rot_str = f"R{rot.value}" if rot.value < 360 else f"M{rot.value - 360}"
        # Surface non-default SYMATTRs (e.g., SpiceLine, SpiceModel) so
        # callers don't need a per-component component_info round-trip.
        attrs = asc_component_attributes(editor.components[ref])
        entry = {"reference": ref, "value": value, "x": pos.X, "y": pos.Y, "rotation": rot_str}
        if attrs:
            entry["attributes"] = attrs
        comp_data.append(entry)

    label_data = [
        {"text": lbl.text, "x": int(lbl.coord.X), "y": int(lbl.coord.Y)} for lbl in editor.labels
    ]
    directive_data = [directive.text for directive in editor.directives]

    # Surface each wire segment's endpoints so callers can target a specific
    # wire for removal (the edit_schematic remove_wire op) without re-deriving it.
    wire_data = [
        {
            "x1": int(w.V1.X),
            "y1": int(w.V1.Y),
            "x2": int(w.V2.X),
            "y2": int(w.V2.Y),
        }
        for w in editor.wires
    ]

    return {
        "file": str(file_path),
        "type": "asc",
        "components": comp_data,
        "labels": label_data,
        "wires": wire_data,
        "wire_count": len(editor.wires),
        "directives": directive_data,
    }


def extract_netlist_info(file_path: Path) -> dict[str, Any]:
    """Extract structured netlist data via spice_lex + ``lib.encoding``.

    Honours BOMs and UTF-16-no-BOM via ``read_spice_text``; walks
    truncated/hierarchical netlists via the spice_lex foundation and
    surfaces its warnings (e.g. unclosed ``.SUBCKT``) instead of raising.
    Component values come from ``InstanceLine`` views — uniform across
    behavioural sources, B-source ``V=expr`` forms, and active devices.
    """
    from ltspice_mcp.errors import NetlistError
    from ltspice_mcp.lib.encoding import read_spice_text
    from ltspice_mcp.lib.spice_lex import lex
    from ltspice_mcp.lib.spice_lex_views import (
        InstanceLine,
        body_has_stray_kv_remnant,
    )

    try:
        content = read_spice_text(file_path)
    except FileNotFoundError as e:
        raise NetlistError(f"File not found: {file_path}") from e
    result = lex(content)
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
