"""Application-level services shared by tools and resources.

This module sits between the MCP adapters (tools/resources) and the pure
parsing helpers in ``raw_parser.py`` / ``log_parser.py``. It owns job
resolution, cached result loading, and reusable extraction/orchestration
logic. All functions raise domain exceptions rather than returning error text.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from spicelib import AscEditor, SpiceEditor
from spicelib.raw.raw_read import RawRead

from ltspice_mcp.errors import BatchJobError, ResultError, SimulationError
from ltspice_mcp.lib.batch_results import (
    compute_batch_stats,
    filter_runs_by_params,
    get_progress_snapshot,
)
from ltspice_mcp.lib.log_parser import parse_measurements
from ltspice_mcp.lib.raw_parser import get_step_count, get_trace_names
from ltspice_mcp.state import BatchJob, SessionState, SimulationJob

Editor = AscEditor | SpiceEditor


def resolve_simulation_job(job_id: str, state: SessionState) -> SimulationJob:
    """Look up a simulation job by id."""
    job = state.jobs.get(job_id)
    if job is None:
        raise SimulationError(f"Job not found: {job_id}")
    return job


def resolve_batch_job(job_id: str, state: SessionState) -> BatchJob:
    """Look up a batch job by id."""
    batch_job = state.batch_jobs.get(job_id)
    if batch_job is None:
        raise BatchJobError(f"Batch job not found: {job_id}")
    return batch_job


def resolve_job(job_id: str, state: SessionState) -> SimulationJob | BatchJob:
    """Look up any job by id."""
    job = state.jobs.get(job_id)
    if job is not None:
        return job
    batch_job = state.batch_jobs.get(job_id)
    if batch_job is not None:
        return batch_job
    raise ResultError(f"Job not found: {job_id}")


def _resolve_result_file(
    job_id: str, state: SessionState, field: str, label: str
) -> Path:
    """Resolve a result file (raw or log) from a simulation or batch job."""
    job = resolve_job(job_id, state)

    if isinstance(job, SimulationJob):
        file_path = getattr(job, field)
        if job.status != "completed" or file_path is None:
            raise ResultError(
                f"Job is not completed (status={job.status!r}) or has no {label} file"
            )
        return file_path

    batch_job = job
    if batch_job.status != "completed":
        raise ResultError(f"Batch job is not completed (status={batch_job.status!r})")
    if not batch_job.run_results:
        raise ResultError(f"Batch job {job_id!r} has no run results")
    first_run = batch_job.run_results[min(batch_job.run_results)]
    result_file = first_run.get(field)
    if result_file is None:
        raise ResultError(f"Batch job {job_id!r} first run has no {label} file")
    return Path(result_file) if not isinstance(result_file, Path) else result_file


def resolve_raw_file(job_id: str, state: SessionState) -> Path:
    """Get the raw result file for a completed simulation or batch job."""
    return _resolve_result_file(job_id, state, "raw_file", "raw")


def resolve_log_file(job_id: str, state: SessionState) -> Path:
    """Get the log file for a completed simulation or batch job."""
    return _resolve_result_file(job_id, state, "log_file", "log")


def job_to_dict(job: SimulationJob) -> dict[str, Any]:
    """Convert a simulation job to structured status data."""
    duration = None
    if job.completed_at is not None:
        duration = (job.completed_at - job.started_at).total_seconds()

    return {
        "job_id": job.job_id,
        "status": job.status,
        "netlist": str(job.netlist),
        "simulator": job.simulator,
        "started_at": job.started_at.isoformat(),
        "completed_at": job.completed_at.isoformat() if job.completed_at else None,
        "duration": duration,
        "raw_file": str(job.raw_file) if job.raw_file else None,
        "log_file": str(job.log_file) if job.log_file else None,
        "error": job.error,
    }


def load_raw(raw_path: Path, state: SessionState) -> RawRead:
    """Load and cache a ``RawRead`` instance."""
    try:
        return state.results.get(
            raw_path,
            lambda p: RawRead(str(p), traces_to_read="*"),
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


def validate_signal(raw: RawRead, signal: str) -> None:
    """Validate that a signal exists in a raw result."""
    trace_names = get_trace_names(raw)
    if signal not in trace_names:
        available = ", ".join(trace_names[:10])
        if len(trace_names) > 10:
            available += f", ... ({len(trace_names)} total)"
        raise ResultError(f"Signal '{signal}' not found. Available signals: {available}")


def validate_step(raw: RawRead, step: int) -> None:
    """Validate that a step index exists in a raw result."""
    step_count = get_step_count(raw)
    if step < 0 or step >= step_count:
        raise ResultError(f"Step {step} out of range. Valid range: 0 to {step_count - 1}")


def load_signal_names(job_id: str, state: SessionState) -> list[str]:
    """Load signal names from a completed job."""
    raw_path = resolve_raw_file(job_id, state)
    raw = load_raw(raw_path, state)
    return get_trace_names(raw)


def load_measurements(
    job_id: str, state: SessionState, *, include_log_text: bool = False
) -> dict[str, Any]:
    """Load measurements from a completed job."""
    log_path = resolve_log_file(job_id, state)
    data = parse_measurements(log_path)
    if include_log_text and log_path.exists():
        data = dict(data)
        data["log_text"] = log_path.read_text(encoding="utf-8", errors="replace")
    return data


def get_batch_status(batch_job: BatchJob) -> dict[str, Any]:
    """Build structured status/progress data for a batch job."""
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

    duration = None
    if batch_job.completed_at and batch_job.started_at:
        duration = (batch_job.completed_at - batch_job.started_at).total_seconds()

    return {
        **base,
        "duration": duration,
        "successful": batch_job.completed_runs - batch_job.failed_runs,
        "error": batch_job.error,
    }


def get_batch_signal_data(
    batch_job: BatchJob,
    signal: str,
    *,
    filters: dict[str, str] | None = None,
    raw: bool = False,
    offset: int = 0,
    limit: int = 50,
) -> dict[str, Any]:
    """Extract structured batch signal data for aggregated or raw mode."""
    if batch_job.completed_runs == 0:
        raise BatchJobError(f"No completed runs yet for job {batch_job.job_id}")

    if filters:
        matching_indices = filter_runs_by_params(batch_job.run_results, filters)
    else:
        matching_indices = sorted(batch_job.run_results.keys())

    total_matching = len(matching_indices)
    if total_matching == 0:
        raise BatchJobError(
            f"No runs match the specified filters for job {batch_job.job_id}: {filters}"
        )

    matching_run_results = {idx: batch_job.run_results[idx] for idx in matching_indices}

    if raw:
        paginated_indices = matching_indices[offset : offset + limit]
        if not paginated_indices:
            raise BatchJobError(
                f"No runs in requested page range for job {batch_job.job_id}: "
                f"offset={offset}, limit={limit}"
            )
        paginated_run_results = {idx: batch_job.run_results[idx] for idx in paginated_indices}
        page_stats = compute_batch_stats(paginated_run_results, signal)
        return {
            "mode": "raw",
            "job_id": batch_job.job_id,
            "job_type": batch_job.job_type,
            "signal": signal,
            "runs": page_stats["runs"],
            "filtered": filters is not None,
            "total_matching": total_matching,
            "total_available": len(batch_job.run_results),
            "offset": offset,
            "limit": limit,
        }

    batch_stats = compute_batch_stats(matching_run_results, signal)
    if batch_stats["run_count"] == 0:
        raise ResultError(f"Signal '{signal}' not found in any completed run")

    return {
        "mode": "aggregate",
        "job_id": batch_job.job_id,
        "job_type": batch_job.job_type,
        "signal": signal,
        "run_count": batch_stats["run_count"],
        "filtered": filters is not None,
        "total_matching": total_matching,
        "total_available": len(batch_job.run_results),
        "stats": batch_stats["stats"],
        "worst_case_run": batch_stats["worst_case_run"],
        "best_case_run": batch_stats["best_case_run"],
    }


def extract_asc_info(editor: AscEditor, file_path: Path) -> dict[str, Any]:
    """Extract structured schematic data from an ``AscEditor``."""
    components = editor.get_components()
    comp_data = []
    for ref in components:
        value = editor.get_component_value(ref)
        pos, rot = editor.get_component_position(ref)
        rot_str = f"R{rot.value}" if rot.value < 360 else f"M{rot.value - 360}"
        comp_data.append(
            {"reference": ref, "value": value, "x": pos.X, "y": pos.Y, "rotation": rot_str}
        )

    label_data = [
        {"text": label.text, "x": label.coord.X, "y": label.coord.Y}
        for label in editor.labels
    ]
    directive_data = [directive.text for directive in editor.directives]

    return {
        "file": str(file_path),
        "type": "asc",
        "components": comp_data,
        "labels": label_data,
        "wire_count": len(editor.wires),
        "directives": directive_data,
    }


def extract_netlist_info(editor: Editor, file_path: Path) -> dict[str, Any]:
    """Extract structured netlist data from a ``SpiceEditor``-compatible editor."""
    content = file_path.read_text(encoding="utf-8", errors="replace")
    components = editor.get_components()
    comp_list = []
    for comp_ref in components:
        value = editor.get_component_value(comp_ref)
        comp_list.append({"reference": comp_ref, "value": value})

    return {
        "file": str(file_path),
        "type": "netlist",
        "content": content,
        "components": comp_list,
    }
