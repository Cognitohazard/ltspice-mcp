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
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, TypeVar

from spicelib import AscEditor, SpiceEditor
from spicelib.raw.raw_read import RawRead

from ltspice_mcp.errors import AnalysisDeadlineExceeded, JobNotFoundError, ResultError
from ltspice_mcp.lib import experiment_store, job_store, recent
from ltspice_mcp.lib.experiment_types import ExperimentJob
from ltspice_mcp.lib.job_lifecycle import runs_terminal
from ltspice_mcp.lib.library_manager import LibraryManager
from ltspice_mcp.lib.log_parser import (
    LogDiagnostics,
    extract_missing_refs,
    missing_refs_from_text,
)
from ltspice_mcp.lib.pathutil import resolve_safe_path
from ltspice_mcp.lib.raw_parser import OffsetAwareRawRead, get_step_count
from ltspice_mcp.lib.simulator import dialect_for_simulator_name
from ltspice_mcp.state import (
    LegacyJobRecord,
    SessionState,
    legacy_record_message,
)

logger = logging.getLogger(__name__)

Editor = AscEditor | SpiceEditor
T = TypeVar("T")


def _suggestions_for_refs(
    refs: list[str], libraries: LibraryManager
) -> dict[str, list[dict]] | None:
    """Fuzzy-match each ref against loaded libraries only (never built-ins)."""
    if not refs:
        return None
    out: dict[str, list[dict]] = {}
    for ref in refs:
        matches = libraries.find_similar_models(ref, limit=3, cutoff=0.5)
        if matches:
            out[ref] = matches
    return out or None


def suggestions_from_errors(
    errors: list[str] | None, libraries: LibraryManager
) -> dict[str, list[dict]] | None:
    """Zero-cost when ``errors`` is falsy — skips the log re-read entirely."""
    if not errors:
        return None
    return _suggestions_for_refs(missing_refs_from_text("\n".join(errors)), libraries)


def extract_model_suggestions(
    log_path: Path | None, libraries: LibraryManager
) -> dict[str, list[dict]] | None:
    """Read ``log_path`` and fuzzy-match every missing ref against loaded libraries."""
    if log_path is None or not log_path.exists():
        return None
    return _suggestions_for_refs(extract_missing_refs(log_path), libraries)


def format_suggestion_block(
    suggestions: dict[str, list[dict]] | None,
    *,
    header: str = "Possible fixes (from loaded user libraries):",
) -> str:
    """Human-readable block for a suggestions dict; empty string if None/empty."""
    if not suggestions:
        return ""
    lines = ["", header]
    for ref, matches in suggestions.items():
        lines.append(f"  Missing '{ref}' — did you mean:")
        for m in matches:
            lines.append(f"    {m['name']} (score={m['score']}) - {m['source_path']}")
    return "\n".join(lines)


def attach_suggestions_to_failure(
    error_msg: str,
    data: dict,
    log_path: Path | None,
    libraries: LibraryManager,
) -> str:
    """Append model-resolution help to ``error_msg`` and mutate ``data``.

    Two complementary layers, both keyed off the unresolved model/subcircuit
    refs in the log: fuzzy matches against loaded user libraries (when any),
    and a recovery hint pointing at ``inspect``'s model search — which fires
    even with no library loaded, the common case stock parts fail in.
    Returns the (possibly-unchanged) error message. Called on
    simulation failure paths where the log already has the error context
    inline, so callers don't re-implement read-log / extract / format / attach.
    """
    if log_path is None or not log_path.exists():
        return error_msg
    refs = extract_missing_refs(log_path)
    if not refs:
        return error_msg
    block = ""
    suggestions = _suggestions_for_refs(refs, libraries)
    if suggestions:
        data["suggestions"] = suggestions
        block += "\n" + format_suggestion_block(suggestions)
    ref_list = ", ".join(refs)
    block += (
        f"\n\nUnresolved model/subcircuit(s): {ref_list}. Stock parts are not "
        "auto-included in the run. For each, call "
        'inspect(kind="model", mode="search", query="<ref>") to locate its '
        'definition in the loaded libraries — or mode="enumerate" with "libs" '
        "to read a specific stock library file — then add the returned .include "
        "directive to the netlist and rerun."
    )
    return f"{error_msg}{block}"


Job = LegacyJobRecord | ExperimentJob


def _experiment_was_reconciled(job: ExperimentJob) -> bool:
    return any(item.get("code") == "server_restarted" for item in job.observations)


def _load_experiment_direct(job_id: str, state: SessionState) -> ExperimentJob | None:
    try:
        experiment_store.validate_job_id(job_id)
    except ValueError as exc:
        raise ResultError(str(exc)) from None
    if not state.job_registry.persist_enabled:
        return None
    return experiment_store.load_job(job_id, state.working_dir, own_is_alive=True)


def resolve_job(job_id: str, state: SessionState) -> Job:
    """Look up any job by id in the union job store.

    Raises ``JobNotFoundError`` for an unknown id — the one place that
    translation happens, so callers up the stack don't re-wrap it.
    """
    job = state.all_jobs.get(job_id)
    if job is None:
        job = _load_experiment_direct(job_id, state)
        if job is None:
            raise JobNotFoundError(f"Job not found: {job_id}")
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            state.job_registry.jobs[job.job_id] = job
        if _experiment_was_reconciled(job):
            state.persist_job(job)
    # A parallel session's live job is only ever updated by its owner; pull
    # the owner's latest persisted state so status checks and result reads
    # here don't stay frozen at "running". No-op for this session's own jobs.
    return state.job_registry.refresh_foreign_job(job)


async def resolve_job_async(job_id: str, state: SessionState) -> Job:
    """Loop-safe ``resolve_job``: offload the foreign-job sidecar re-read.

    Use from async handlers so the parallel-session refresh (a sidecar read
    that stalls the loop on a wedged filesystem) runs in a worker thread. Same
    semantics otherwise — raises ``JobNotFoundError`` for an unknown id.
    """
    job = state.all_jobs.get(job_id)
    if job is None:
        try:
            experiment_store.validate_job_id(job_id)
        except ValueError as exc:
            raise ResultError(str(exc)) from None
        if state.job_registry.persist_enabled:
            job = await asyncio.to_thread(
                experiment_store.load_job,
                job_id,
                state.working_dir,
                own_is_alive=True,
            )
        if job is None:
            raise JobNotFoundError(f"Job not found: {job_id}")
        state.job_registry.jobs[job.job_id] = job
        if _experiment_was_reconciled(job):
            state.persist_job(job)
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

    raw: Path
    log: Path | None
    netlist: Path
    #: The source schematic/deck the case was staged from — where per-circuit
    #: sidecars (plots, pointers) belong. ``netlist`` is the staged copy.
    circuit_path: Path
    dialect: str | None
    identity: dict[str, Any]


@dataclass(frozen=True)
class AnalysisSource:
    """Resolved source injected into analysis adapters by the consolidated path."""

    raw: Path
    log: Path | None
    netlist: Path | None
    dialect: str | None
    identity: dict[str, Any] | None
    trusted_job_artifact: bool

    @classmethod
    def for_raw(cls, raw_path: Path) -> AnalysisSource:
        """The log/netlist companions of ``raw_path``, resolved through one seam.

        When the consolidated path injected a task-local source for this raw, its
        netlist and dialect win and its log is filled with the sibling ``.log``
        when it carries none. A direct read (no matching injected source) gets
        the sibling ``.log`` and no netlist — the uniform fallback every read
        shares. ``log`` is always a concrete path (existence not guaranteed);
        ``netlist`` is present only when a producing job supplied one.
        """
        injected = current_analysis_source()
        if injected is not None and injected.raw == raw_path:
            log = injected.log if injected.log is not None else raw_path.with_suffix(".log")
            return replace(injected, log=log)
        return cls(
            raw=raw_path,
            log=raw_path.with_suffix(".log"),
            netlist=None,
            dialect=None,
            identity=None,
            trusted_job_artifact=False,
        )


_analysis_source: contextvars.ContextVar[AnalysisSource | None] = contextvars.ContextVar(
    "ltspice-mcp.analysis-source",
    default=None,
)
_analysis_deadline: contextvars.ContextVar[float | None] = contextvars.ContextVar(
    "ltspice-mcp.analysis-deadline",
    default=None,
)


@contextlib.contextmanager
def analysis_source_context(
    source: AnalysisSource,
    *,
    deadline: float | None = None,
):
    """Inject a trusted source and optional monotonic deadline into adapters."""
    source_token = _analysis_source.set(source)
    deadline_token = _analysis_deadline.set(deadline)
    try:
        yield
    finally:
        _analysis_deadline.reset(deadline_token)
        _analysis_source.reset(source_token)


def current_analysis_source() -> AnalysisSource | None:
    """Return the task-local adapter source, if the consolidated path set one."""
    return _analysis_source.get()


def resolve_experiment_run(
    job_id: str,
    state: SessionState,
    *,
    run_index: int | None = None,
    case_id: str | None = None,
) -> RunContext:
    """Resolve a produced case from any experiment job whose runs are terminal."""
    job = resolve_job(job_id, state)
    if not isinstance(job, ExperimentJob):
        raise ResultError(f"Job {job_id!r} is not an experiment job")
    return experiment_run_context(job, state, run_index=run_index, case_id=case_id)


def experiment_run_context(
    job: ExperimentJob,
    state: SessionState,
    *,
    run_index: int | None = None,
    case_id: str | None = None,
) -> RunContext:
    """``resolve_experiment_run`` for a caller already holding the job.

    Records the case raw's dialect hint here, as the legacy resolvers do for
    their runs: resolution always precedes the load, so every reader parses a
    per-run simulator override with the right dialect without remembering to.
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
    if case.status != "produced" or case.raw_file is None:
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
    dialect = dialect_for_job(job, state)
    state.raw_dialect_hints[case.raw_file] = dialect
    return RunContext(
        raw=case.raw_file,
        log=case.log_file,
        netlist=case.staged_deck,
        circuit_path=case.circuit_path,
        dialect=dialect,
        identity=identity,
    )


def resolve_analysis_source(
    args: Any,
    state: SessionState,
    *,
    injected: AnalysisSource | RunContext | None = None,
) -> AnalysisSource:
    """Resolve an adapter source without weakening the legacy completion gate.

    Consolidated callers inject a pre-resolved source, so trusted artifacts
    never pass through ``safe_path`` and experiment cases never re-enter
    ``resolve_run``.  Direct callers use the same completed-only legacy
    resolver as before.
    """
    if isinstance(injected, AnalysisSource):
        return injected
    if isinstance(injected, RunContext):
        return AnalysisSource(
            raw=injected.raw,
            log=injected.log,
            netlist=injected.netlist,
            dialect=injected.dialect,
            identity=injected.identity,
            trusted_job_artifact=True,
        )
    contextual = current_analysis_source()
    if contextual is not None:
        return contextual

    raw_file = getattr(args, "raw_file", None)
    log_file = getattr(args, "log_file", None)
    job_id = getattr(args, "job_id", None)
    if hasattr(args, "raw_file") and bool(raw_file) == bool(job_id):
        raise ResultError(
            "Pass exactly one of 'raw_file' or 'job_id'. Analysis tools read "
            "an existing result — if you only have a netlist, run_experiments "
            "produces the job_id/raw to analyze.",
            show_hint=False,
        )
    if job_id:
        # Every job this version creates is an experiment, and an experiment's
        # runs are case-addressed — the consolidated callers inject a resolved
        # source above rather than arriving here. Anything that reaches this
        # point is a record an earlier release wrote.
        job = resolve_job(job_id, state)
        if isinstance(job, ExperimentJob):
            raise ResultError(
                f"Job {job_id!r} is an experiment; its runs are case-addressed. "
                "Read them with analyze_results (job_id plus run_index or case_id)."
            )
        raise ResultError(legacy_record_message(job_id))
    if raw_file:
        raw = resolve_safe_path(str(raw_file), state.config.allowed_paths)
        sibling = raw.with_suffix(".log")
        return AnalysisSource(
            raw=raw,
            log=sibling if sibling.is_file() else None,
            netlist=None,
            dialect=raw_dialect_for(raw, state),
            identity=None,
            trusted_job_artifact=False,
        )
    if log_file:
        log = resolve_safe_path(str(log_file), state.config.allowed_paths)
        return AnalysisSource(
            raw=log.with_suffix(".raw"),
            log=log,
            netlist=None,
            dialect=None,
            identity=None,
            trusted_job_artifact=False,
        )
    raise ResultError("Provide one analysis source: raw_file, log_file, or job_id")


def dialect_for_job(job: Job, state: SessionState) -> str | None:
    """Raw dialect for the simulator ``job`` actually ran on.

    A per-run simulator override can differ from the session default (and a
    persisted job may be read back under a different default), so the job's own
    recorded simulator wins. Resolved from the recorded name string, so a
    persisted ngspice job read back with only LTspice installed still parses
    with the ngspice dialect — the producing simulator need not remain
    configured. Falls back to the session default only when the job records no
    simulator at all — which is every legacy record, none of which is readable
    here anyway.
    """
    simulator = getattr(job, "simulator", None)
    if simulator:
        return dialect_for_simulator_name(simulator)
    return state.raw_dialect


def raw_dialect_for(raw_path: Path, state: SessionState) -> str | None:
    """Raw dialect for the simulator that produced ``raw_path``.

    Job-addressed reads record the producing job's dialect when the path is
    resolved (see ``_resolve_result_file``), so a run launched with a per-run
    simulator override parses with that simulator's dialect rather than the
    session default's. Paths with no recorded producer (a user-supplied
    ``raw_file``) use the default.
    """
    return state.raw_dialect_hints.get(raw_path, state.raw_dialect)


# Hard wall-clock bound on one raw parse. A raw is an untrusted simulator
# artifact parsed through a third-party library — a shape spicelib didn't
# anticipate can loop indefinitely (it has: the ngspice noise-raw multi-plot
# loop, since guarded in raw_parser). Offloading protects the loop from a
# SLOW parse; only a deadline protects the session from a RUNAWAY one — this
# fails the one call instead of wedging its worker thread forever. Generous:
# multi-GB DrvFs parses land in tens of seconds, not minutes.
RAW_PARSE_TIMEOUT_S = 120.0

# Paths whose last parse hit the deadline, by monotonic expiry time. Gates
# retries: an abandoned worker cannot be killed, so immediate retries could
# abandon more executor threads and eventually stall unrelated to_thread work.
# During cooldown the retry fails fast on the loop instead; after it, one fresh
# attempt is allowed (worst case the leak grows by one thread per cooldown
# period, not per call). Read/written only on the event loop with no await
# between check and store, so no lock is needed.
_wedged_raw_paths: dict[Path, float] = {}


async def bounded_parse(
    path: Path,
    thunk: Callable[[], T],
    *,
    timeout_s: float = RAW_PARSE_TIMEOUT_S,
) -> T:
    """Run one result parse off the event loop with a deadline and cooldown.

    The cooldown is shared by every parser using this helper, so a raw file
    whose abandoned worker may still be running cannot consume another worker
    through a different result-reading path until the cooldown expires.
    """
    loop = asyncio.get_running_loop()
    now_mono = loop.time()
    cooldown_s = timeout_s
    call_deadline = _analysis_deadline.get()
    if call_deadline is not None:
        timeout_s = min(timeout_s, max(0.0, call_deadline - now_mono))
    if timeout_s <= 0:
        raise AnalysisDeadlineExceeded(f"Parsing {path.name} exceeded the analysis item deadline")
    wedged_until = _wedged_raw_paths.get(path)
    if wedged_until is not None:
        if now_mono < wedged_until:
            raise AnalysisDeadlineExceeded(
                f"Parsing {path.name} recently exceeded its deadline and "
                "its worker is still abandoned; retries are paused for "
                f"{wedged_until - now_mono:.0f}s more so a wedged file can't "
                "drain the worker pool. Check the file (size, mtime, source "
                "simulator) before retrying."
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
            f"Parsing {path.name} exceeded {timeout_s:.3g}s and was "
            "abandoned — the file may be corrupt in a way that wedges the parser, "
            "or on a stalled mount. The file was not modified; retries are "
            f"paused for {cooldown_s:.0f}s, then one fresh attempt is "
            "allowed."
        ) from None


async def load_raw(raw_path: Path, state: SessionState) -> RawRead:
    """Load and cache a ``RawRead`` instance without blocking the event loop.

    Fast path: a fresh cache entry is returned inline — one ``os.stat`` on
    the loop per call, accepted as far cheaper than a thread hop for the
    guaranteed-hit patterns (``bode_metrics all_steps`` re-enters this once
    per step of the same raw). On miss or stale entry the probe and parse
    run in a worker thread via ``asyncio.to_thread``, bounded by
    ``RAW_PARSE_TIMEOUT_S``. The parsed value is immutable and the cache
    store is lock-guarded, so cancellation leaves the cache consistent: a
    worker that has started runs to completion (at worst storing a benign
    extra cache entry), and work cancelled before the executor picks it up
    never begins. On deadline expiry the worker thread is abandoned (Python
    can't kill it) and the path enters a retry cooldown (see
    ``_wedged_raw_paths``) so hammering the same file cannot leak a thread
    per call.
    """
    cached = state.results.peek(raw_path)
    if cached is not None:
        return cached
    return await bounded_parse(
        raw_path,
        lambda: load_raw_sync(raw_path, state),
        timeout_s=RAW_PARSE_TIMEOUT_S,
    )


def load_raw_sync(raw_path: Path, state: SessionState) -> RawRead:
    """Blocking implementation behind :func:`load_raw`.

    Call directly only from code already off the event loop, or from the
    synchronous resource router. Coroutine handlers must ``await load_raw``.
    """
    dialect = raw_dialect_for(raw_path, state)
    try:
        raw = state.results.get(
            raw_path,
            lambda p: OffsetAwareRawRead(str(p), traces_to_read="*", dialect=dialect),
        )
    except FileNotFoundError:
        raise ResultError(f"Result file not found: {raw_path}") from None
    except ResultError:
        raise
    except Exception as e:
        raise ResultError(
            f"Failed to parse result file: {e}. "
            "File may be corrupted or not a valid SPICE .raw file"
        ) from e
    # spicelib parses a header truncated mid-write into a "valid" raw with no
    # variables at all. A real SPICE raw always carries at least its axis
    # variable, so zero variables is a corruption signature — fail here with
    # the real cause instead of letting every consumer misreport it as
    # "signal not found" against an empty signal list.
    if not raw.get_trace_names():
        raise ResultError(
            f"Result file {raw_path} parsed with zero variables — the file is "
            "most likely truncated or corrupt (e.g. a simulation killed mid-write)."
        )
    return raw


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
    ``validate_signal`` against a raw's trace list, ``analyze_results``
    against an operating-point result — and a second copy of the rule is how
    one of them ends up rejecting a name the other accepts.
    """
    match = DEV_PARAM_RE.fullmatch(signal.lower())
    if match is None:
        return []
    dev, param = match.group(1), match.group(2)
    return [f"@{dev}[{param}]", f"v(@{dev}[{param}])", f"i(@{dev}[{param}])"]


def validate_signal(raw: RawRead, signal: str) -> str:
    """Validate that a signal exists in a raw result and return the canonical trace name.

    Lookup is case-insensitive: SPICE node names are case-insensitive per
    SPICE conventions, but spicelib preserves the case the simulator wrote.
    LTspice writes ``V(out)`` for transient/AC/DC sweep raws but ``v(onoise)``
    for ``.NOISE`` raws — case-sensitive match would reject the user's
    ``V(onoise)`` even though the data exists.

    The returned canonical name is what callers must pass to ``raw.get_wave``
    to actually read the trace.
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

    available = ", ".join(trace_names[:10])
    if len(trace_names) > 10:
        available += f", ... ({len(trace_names)} total)"
    hint = ""
    sig_lo = sig_lower
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
    raise ResultError(f"Signal '{signal}' not found.{hint} Available signals: {available}")


def validate_step(raw: RawRead, step: int) -> None:
    """Validate that a step index exists in a raw result."""
    step_count = get_step_count(raw)
    if step < 0 or step >= step_count:
        raise ResultError(f"Step {step} out of range. Valid range: 0 to {step_count - 1}")


def collect_recent_circuits() -> list[dict[str, Any]]:
    """List recently-touched circuits with their persisted-job summaries.

    Blocking — ``recent.load`` polls a cross-process file lock (up to 10 s)
    and each summary reads a circuit's job-sidecar JSON files; all reads
    (the prune rewrite is atomic), so safe under cancellation. Coroutine
    callers must run this via ``asyncio.to_thread``; the synchronous MCP
    resource router calls it directly (already off the loop).
    """
    entries = recent.load(prune_missing=True)
    circuits: list[dict[str, Any]] = []
    for entry in entries:
        raw_path = entry.get("path")
        if not isinstance(raw_path, str):
            continue
        summary = job_store.summarize_circuit(Path(raw_path))
        summary["last_touched"] = entry.get("last_touched")
        circuits.append(summary)
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
        # error path is identical to the legacy code.
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
