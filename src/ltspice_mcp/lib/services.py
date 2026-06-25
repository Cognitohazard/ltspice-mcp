"""Application-level services shared by tools and resources.

This module sits between the MCP adapters (tools/resources) and the pure
parsing helpers in ``raw_parser.py`` / ``log_parser.py``. It owns job
resolution, cached result loading, and reusable extraction/orchestration
logic. All functions raise domain exceptions rather than returning error text.
"""

from __future__ import annotations

import asyncio
import logging
import re
from pathlib import Path
from typing import Any

from spicelib import AscEditor, SpiceEditor
from spicelib.raw.raw_read import RawRead

from ltspice_mcp.errors import BatchJobError, JobNotFoundError, ResultError, SimulationError
from ltspice_mcp.lib import job_store, recent
from ltspice_mcp.lib.batch_results import (
    compute_batch_stats,
    filter_runs_by_params,
    get_progress_snapshot,
)
from ltspice_mcp.lib.library_manager import LibraryManager
from ltspice_mcp.lib.log_parser import (
    extract_missing_refs,
    missing_refs_from_text,
    parse_measurements,
    read_log_text,
)
from ltspice_mcp.lib.raw_parser import get_step_count
from ltspice_mcp.state import (
    TERMINAL_STATUSES,
    BatchJob,
    RunRef,
    SessionState,
    SimulationJob,
)

logger = logging.getLogger(__name__)

Editor = AscEditor | SpiceEditor


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
    and a library-independent recovery hint pointing at find_model's built-in
    search — which fires even with no library loaded, the common case stock
    parts fail in. Returns the (possibly-unchanged) error message. Called on
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
        'find_model(name="<ref>", include_builtin=true), add the returned '
        ".include directive to the netlist, and rerun."
    )
    return f"{error_msg}{block}"


def resolve_job(job_id: str, state: SessionState) -> SimulationJob | BatchJob:
    """Look up any job by id in the union job store.

    Raises ``JobNotFoundError`` for an unknown id — the one place that
    translation happens, so callers up the stack don't re-wrap it.
    """
    job = state.all_jobs.get(job_id)
    if job is None:
        raise JobNotFoundError(f"Job not found: {job_id}")
    return job


def resolve_simulation_job(job_id: str, state: SessionState) -> SimulationJob:
    """Look up a single-simulation job by id.

    An unknown id propagates ``JobNotFoundError`` from ``resolve_job``. A
    batch (sweep/MC) id gets an honest redirect instead of "not found" —
    the job exists, it just isn't readable through the single-sim surface.
    """
    job = resolve_job(job_id, state)
    if isinstance(job, BatchJob):
        raise SimulationError(
            f"Job '{job_id}' is a {job.job_type} batch job — "
            "use batch_results for its per-run results."
        )
    return job


def resolve_batch_job(job_id: str, state: SessionState) -> BatchJob:
    """Look up a batch (sweep/MC) job by id.

    A single-simulation id gets an honest redirect instead of "not found" —
    the job exists, it just has no per-run batch surface.
    """
    try:
        job = resolve_job(job_id, state)
    except JobNotFoundError:
        # Batch tools historically say "Batch job not found"; keep that
        # surface text — this is the one remaining not-found translation.
        raise BatchJobError(f"Batch job not found: {job_id}") from None
    if isinstance(job, SimulationJob):
        # Point only at tools that accept a job id: check_job (status +
        # completion results) and query_value (job_id + run_index for a
        # signal). simulation_summary takes a raw_file, not a job id, so it
        # is reached only after check_job hands back that path.
        raise BatchJobError(
            f"Job '{job_id}' is a single simulation job — read its results with "
            "check_job (status + completion summary) or query_value (job_id + "
            "run_index) for a signal value."
        )
    return job


def _as_path(p: object) -> Path | None:
    """Coerce a stored raw/log path to a real Path, treating ""/"." as absent.

    An empty Path coerces to "." which would silently point at the current
    directory, so those sentinels become None.
    """
    if p is None or str(p) in ("", "."):
        return None
    return p if isinstance(p, Path) else Path(str(p))


def runs_of(job: SimulationJob | BatchJob) -> list[RunRef]:
    """Project any job into a uniform list of result runs (the read-model seam).

    A single-run job is the degenerate batch-of-one: one ``RunRef`` at index 0.
    A batch job yields one ``RunRef`` per ``run_results`` entry, ordered by run
    index. This is the ONLY place that knows the two physical result layouts;
    extraction routines consume ``RunRef`` and stay job-agnostic.
    """
    if isinstance(job, SimulationJob):
        return [RunRef(0, _as_path(job.raw_file), _as_path(job.log_file), {})]
    return [
        RunRef(
            index=idx,
            raw_file=_as_path(run.get("raw_file")),
            log_file=_as_path(run.get("log_file")),
            params=dict(run.get("params") or {}),
        )
        for idx, run in sorted(job.run_results.items())
    ]


def resolve_run(job_id: str, state: SessionState, run_index: int = 0) -> RunRef:
    """Resolve one run of a COMPLETED job by id + run index (default 0).

    Gates on completion so every job_id-addressed read (resolve_raw_file,
    query_value/bode_metrics via the read-model) behaves identically — you can't
    read partial data from a running/failed job through one path while a sibling
    tool rejects it. A single-run job has exactly one run (index 0). Raises
    ``ResultError`` for an out-of-range index, listing the indices actually
    present (batch run indices can be non-contiguous after a mid-batch failure).
    """
    job = resolve_job(job_id, state)
    if job.status != "completed":
        raise ResultError(f"Job {job_id!r} is not completed (status={job.status!r})")
    runs = {r.index: r for r in runs_of(job)}
    if not runs:
        raise ResultError(f"Job {job_id!r} has no run results")
    if run_index not in runs:
        raise ResultError(
            f"Run index {run_index} out of range for job {job_id!r}; valid indices: {sorted(runs)}"
        )
    return runs[run_index]


def ngspice_preflight_warnings(netlist_path: Path, simulator_class: type) -> list[str]:
    """Pre-flight check for ngspice-incompatible directives in a base netlist.

    Returns warnings to surface in the response; raises ``SimulationError`` for
    the hard ``.step`` blocker. No-op for non-ngspice simulators. Shared by the
    single-run path (run_simulation) and the batch paths (configure_sweep /
    configure_montecarlo) so all three surface the same ".meas skipped in batch
    mode" warning instead of silently dropping measurements.
    """
    from spicelib.simulators.ngspice_simulator import NGspiceSimulator

    if not issubclass(simulator_class, NGspiceSimulator):
        return []
    from ltspice_mcp.lib.spice_lex import MEAS_ANALYSIS_TOKENS

    try:
        content = netlist_path.read_text(errors="replace")
    except OSError:
        return []
    warnings: list[str] = []
    meas_names: list[str] = []
    for line in content.splitlines():
        stripped = line.strip().lower()
        if stripped.startswith(".step"):
            raise SimulationError(
                "ngspice batch mode does not support .step directives. "
                "Use configure_sweep + run_sweep for parametric sweeps, "
                "or remove the .step line and set the parameter to a fixed value."
            )
        if stripped.startswith(".meas"):
            # .meas[ure] [analysis-type] <name> <FIND|PARAM|TRIG|WHEN|...> ...
            # The analysis-type token is OPTIONAL; when it is omitted the name is
            # parts[1], not parts[2] (e.g. ".meas vfoo FIND V(a)" — parts[2] would
            # wrongly grab "find"). ``stripped`` is already lowercased. Mirrors
            # MeasCard.from_card's optional-token skip.
            parts = stripped.split()
            idx = 1
            if len(parts) > idx and parts[idx] in MEAS_ANALYSIS_TOKENS:
                idx += 1
            if len(parts) > idx:
                meas_names.append(parts[idx])
    if meas_names:
        names = ", ".join(meas_names)
        warnings.append(
            "ngspice cannot evaluate .meas in batch mode. "
            f"The following measurements will be skipped: {names}. "
            "Use signal_stats or query_value to compute them from the raw data."
        )
    return warnings


def _resolve_result_file(
    job_id: str, state: SessionState, field: str, label: str, *, run_index: int = 0
) -> Path:
    """Resolve a result file (raw or log) from a completed job's run.

    ``run_index`` selects which run (default 0 — the only run for single-run
    jobs, the first run for a batch). Shares ``resolve_run``'s completion + bounds
    gate so single and batch jobs behave identically.
    """
    run = resolve_run(job_id, state, run_index)
    file_path = run.raw_file if field == "raw_file" else run.log_file
    if file_path is None:
        raise ResultError(f"Job {job_id!r} run {run_index} has no {label} file")
    return file_path


def resolve_raw_file(job_id: str, state: SessionState, run_index: int = 0) -> Path:
    """Get the raw result file for a completed simulation or batch job run."""
    return _resolve_result_file(job_id, state, "raw_file", "raw", run_index=run_index)


def resolve_log_file(job_id: str, state: SessionState, run_index: int = 0) -> Path:
    """Get the log file for a completed simulation or batch job run."""
    return _resolve_result_file(job_id, state, "log_file", "log", run_index=run_index)


async def load_raw(raw_path: Path, state: SessionState) -> RawRead:
    """Load and cache a ``RawRead`` instance without blocking the event loop.

    Fast path: a fresh cache entry is returned inline — one ``os.stat`` on
    the loop per call, accepted as far cheaper than a thread hop for the
    guaranteed-hit patterns (``bode_metrics all_steps`` re-enters this once
    per step of the same raw). On miss or stale entry the probe and parse
    run in a worker thread via ``asyncio.to_thread``. The parsed value is
    immutable and the cache store is lock-guarded, so cancellation leaves
    the cache consistent: a worker that has started runs to completion
    (at worst storing a benign extra cache entry), and work cancelled
    before the executor picks it up never begins.
    """
    cached = state.results.peek(raw_path)
    if cached is not None:
        return cached
    return await asyncio.to_thread(load_raw_sync, raw_path, state)


def load_raw_sync(raw_path: Path, state: SessionState) -> RawRead:
    """Blocking implementation behind :func:`load_raw`.

    Call directly only from code already off the event loop, or from the
    synchronous resource router. Coroutine handlers must ``await load_raw``.
    """
    dialect = state.raw_dialect
    try:
        raw = state.results.get(
            raw_path,
            lambda p: RawRead(str(p), traces_to_read="*", dialect=dialect),
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

    # Device operating-point small-signal / model parameters. ngspice writes these as
    # @dev[param] depending on the quantity: bare (@m1[gm]), v-wrapped
    # (v(@m1[vth])), or i-wrapped (i(@m1[id])). Accept a uniform 'dev.param'
    # shorthand (e.g. 'm1.gm') and resolve to whichever form the raw contains.
    # 'dev.param' shorthand (e.g. 'm1.gm'), including a flattened subcircuit-
    # hierarchical device path ('m.x1.mn.gm'): everything before the LAST dot is
    # the device, the last segment is the parameter.
    dev_param = re.fullmatch(r"([a-z][\w.]*)\.([a-z]\w*)", sig_lower)
    if dev_param:
        dev, param = dev_param.group(1), dev_param.group(2)
        candidates += [f"@{dev}[{param}]", f"v(@{dev}[{param}])", f"i(@{dev}[{param}])"]

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


def load_signal_names(job_id: str, state: SessionState) -> list[str]:
    """Load signal names from a completed job.

    Stays synchronous (via ``load_raw_sync``) because its only caller is the
    synchronous MCP resource router.
    """
    raw_path = resolve_raw_file(job_id, state)
    raw = load_raw_sync(raw_path, state)
    return raw.get_trace_names()


def load_measurements(
    job_id: str, state: SessionState, *, include_log_text: bool = False
) -> dict[str, Any]:
    """Load measurements from a completed job.

    Return type is ``dict[str, Any]`` (not ``MeasurementsOutput``) because
    this helper may add a ``log_text`` field beyond the parser's shape.
    """
    log_path = resolve_log_file(job_id, state)
    data: dict[str, Any] = dict(parse_measurements(log_path))
    if include_log_text and log_path.exists():
        data["log_text"] = log_path.read_text(encoding="utf-8", errors="replace")
    return data


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


async def get_batch_status(batch_job: BatchJob) -> dict[str, Any]:
    """Build structured status/progress data for a batch job.

    Owns its offload: the convergence scan walks every per-run log once
    the job is terminal, so it runs via ``asyncio.to_thread``; the status
    fields themselves are assembled inline.
    """
    base = {
        "job_id": batch_job.job_id,
        "job_type": batch_job.job_type,
        "status": batch_job.status,
        "netlist": batch_job.netlist.name,
        "total_runs": batch_job.total_runs,
        "completed_runs": batch_job.completed_runs,
        "failed_runs": batch_job.failed_runs,
    }

    if batch_job.status == "running":
        snap = get_progress_snapshot(batch_job, batch_job.started_at.timestamp())
        return {
            **base,
            "completed": snap["completed"],
            "total": snap["total"],
            "failed": snap["failed"],
            "elapsed_s": snap["elapsed_s"],
            "eta_s": snap["eta_s"],
        }

    duration = job_duration_seconds(
        batch_job.started_at, batch_job.completed_at, label=f"batch job {batch_job.job_id}"
    )

    out: dict[str, Any] = {
        **base,
        "duration": duration,
        "successful": batch_job.completed_runs - batch_job.failed_runs,
    }
    # Omit-when-empty: "error": null breaks clients that validate against
    # an output schema typing error as string (same class as check_job's
    # batch branch).
    if batch_job.error is not None:
        out["error"] = batch_job.error
    convergence = await asyncio.to_thread(scan_batch_convergence, batch_job)
    if convergence:
        out["convergence_warnings"] = convergence
    return out


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


def scan_batch_convergence(batch_job: BatchJob) -> list[dict[str, Any]]:
    """Walk every per-run log and surface convergence-fallback markers.

    Returns an empty list while the job is still running — the per-run
    logs are still being written and re-reading every poll loop is a
    waste. Once the job is terminal the result is cached on the
    ``BatchJob`` so the (status, signal-data) round-trip a typical poll
    issues doesn't pay for two full walks.
    """
    # Scan once the job is terminal (incl. ``interrupted`` — its completed
    # sub-runs are real results worth surfacing). A hardcoded tuple here used to
    # omit ``interrupted``, silently skipping the convergence scan for recovered
    # batches; use the canonical terminal set so the class can't recur.
    if batch_job.status not in TERMINAL_STATUSES:
        return []
    cached = batch_job.convergence_warnings
    if cached is not None:
        return cached
    flagged: list[dict[str, Any]] = []
    for run_index in sorted(batch_job.run_results.keys()):
        log_str = batch_job.run_results[run_index].get("log_file")
        if not log_str:
            continue
        text = read_log_text(Path(log_str)).lower()
        if not text:
            continue
        markers = [s for s in _CONVERGENCE_FLAG_SUBSTRINGS if s in text]
        if markers:
            flagged.append({"run_index": run_index, "markers": markers})
    batch_job.convergence_warnings = flagged
    return flagged


def job_duration_seconds(
    started_at: Any | None,
    completed_at: Any | None,
    *,
    label: str = "job",
) -> float | None:
    """Compute ``completed_at - started_at`` in seconds, clamped at 0.

    Guard: clock skew, persistence round-trips, or out-of-order
    timestamps occasionally produce negative durations. Clamping with a
    warning surfaces the anomaly without leaking garbage to clients.
    """
    if not started_at or not completed_at:
        return None
    delta = (completed_at - started_at).total_seconds()
    if delta < 0:
        logger.warning(
            "%s reports negative duration (%.3fs); started_at=%s completed_at=%s — clamping to 0.",
            label,
            delta,
            started_at.isoformat(),
            completed_at.isoformat(),
        )
        return 0.0
    return delta


def _copy_present(out: dict, src: dict, *keys: str) -> None:
    """Copy each key from src to out when it holds a truthy value (skip empty)."""
    for k in keys:
        if src.get(k):
            out[k] = src[k]


async def get_batch_signal_data(
    batch_job: BatchJob,
    signal: str,
    *,
    filters: dict[str, str] | None = None,
    raw: bool = False,
    offset: int = 0,
    limit: int = 50,
    at: float | None = None,
    dialect: str | None = None,
) -> dict[str, Any]:
    """Extract structured batch signal data for aggregated or raw mode.

    The ``run_results`` snapshot is taken on the event loop before the
    first await (a batch runner may still be appending runs); the per-run
    ``RawRead`` loop — one header+trace parse per matching run — and the
    per-run log walk behind the convergence scan are offloaded to worker
    threads, since each can take seconds for a large Monte Carlo batch.
    """
    if batch_job.completed_runs == 0:
        raise BatchJobError(f"No completed runs yet for job {batch_job.job_id}")

    if raw and (offset < 0 or limit < 1):
        raise BatchJobError(
            f"Invalid pagination for job {batch_job.job_id}: "
            f"offset must be >= 0 and limit must be >= 1 (got offset={offset}, limit={limit})"
        )

    if filters:
        matching_indices = filter_runs_by_params(batch_job.run_results, filters)
    else:
        matching_indices = sorted(batch_job.run_results.keys())

    total_matching = len(matching_indices)
    if total_matching == 0:
        raise BatchJobError(
            f"No runs match the specified filters for job {batch_job.job_id}: {filters}"
        )

    # Single pre-await snapshot — both modes draw their rows from it below.
    # Re-reading batch_job.run_results after an await could see rows a
    # still-appending batch runner added meanwhile, breaking the contract above.
    matching_run_results = {idx: batch_job.run_results[idx] for idx in matching_indices}
    total_available = len(batch_job.run_results)

    convergence = await asyncio.to_thread(scan_batch_convergence, batch_job)

    if raw:
        paginated_indices = matching_indices[offset : offset + limit]
        if not paginated_indices:
            raise BatchJobError(
                f"No runs in requested page range for job {batch_job.job_id}: "
                f"offset={offset}, limit={limit}"
            )
        paginated_run_results = {idx: matching_run_results[idx] for idx in paginated_indices}
        page_stats = await asyncio.to_thread(
            compute_batch_stats, paginated_run_results, signal, at=at, dialect=dialect
        )
        # Mirror the aggregate path's guard: if the signal could not be read
        # from ANY run in the page, raise instead of silently returning
        # runs:[] (which reads as "no data produced"). A typo, a .MEAS name,
        # or a derived expression all land here.
        if page_stats["run_count"] == 0 and paginated_indices:
            raise ResultError(
                f"Signal '{signal}' could not be read from any run of job "
                f"{batch_job.job_id}. If it is a .MEAS name use measurement_stats; "
                f"otherwise check the trace name against a run's raw signals.",
                show_hint=False,
            )
        out_raw: dict[str, Any] = {
            "mode": "raw",
            "job_id": batch_job.job_id,
            "job_type": batch_job.job_type,
            "signal": signal,
            "runs": page_stats["runs"],
            "filtered": filters is not None,
            "total_matching": total_matching,
            "total_available": total_available,
            "offset": offset,
            "limit": limit,
        }
        _copy_present(out_raw, page_stats, "step_collapsed_runs", "step_unknown_runs")
        if convergence:
            out_raw["convergence_warnings"] = convergence
        return out_raw

    batch_stats = await asyncio.to_thread(
        compute_batch_stats, matching_run_results, signal, at=at, dialect=dialect
    )
    if batch_stats["run_count"] == 0:
        raise ResultError(f"Signal '{signal}' not found in any completed run")

    out: dict[str, Any] = {
        "mode": "aggregate",
        "job_id": batch_job.job_id,
        "job_type": batch_job.job_type,
        "signal": signal,
        "at": at,
        "run_count": batch_stats["run_count"],
        "filtered": filters is not None,
        "total_matching": total_matching,
        "total_available": total_available,
        "stats": batch_stats["stats"],
        "max_case_run": batch_stats["max_case_run"],
        "min_case_run": batch_stats["min_case_run"],
    }
    _copy_present(out, batch_stats, "step_collapsed_runs", "step_unknown_runs")
    if convergence:
        out["convergence_warnings"] = convergence
    return out


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
    # wire for removal (apply_schematic_ops remove_wire) without re-deriving it.
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
