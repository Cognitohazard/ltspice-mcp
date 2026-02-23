"""MCP resource handlers for browsing netlists, results, models, and config."""

import json
import logging
from pathlib import Path
from typing import Any

from mcp import types
from pydantic import AnyUrl

from ltspice_mcp.state import SessionState
from ltspice_mcp.lib.pathutil import resolve_safe_path
from ltspice_mcp.tools._base import run_sync

logger = logging.getLogger(__name__)

NETLIST_EXTENSIONS = {".asc", ".net", ".sp", ".cir", ".spice"}


def get_static_resources() -> list[types.Resource]:
    """Return the 4 static resources always present on this server."""
    return [
        types.Resource(
            name="netlists",
            uri=AnyUrl("ltspice://netlists/"),
            description="List of netlist files in the working directory",
            mimeType="application/json",
        ),
        types.Resource(
            name="results",
            uri=AnyUrl("ltspice://results/"),
            description="List of all simulation jobs and their status",
            mimeType="application/json",
        ),
        types.Resource(
            name="models",
            uri=AnyUrl("ltspice://models/"),
            description="User-loaded SPICE model libraries and their models",
            mimeType="application/json",
        ),
        types.Resource(
            name="config",
            uri=AnyUrl("ltspice://config"),
            description="Server configuration and detected simulators",
            mimeType="application/json",
        ),
    ]


def get_resource_templates() -> list[types.ResourceTemplate]:
    """Return the 3 dynamic resource templates."""
    return [
        types.ResourceTemplate(
            name="netlist_content",
            uriTemplate="ltspice://netlists/{filename}",
            description="Full text content of a specific netlist file",
            mimeType="text/plain",
        ),
        types.ResourceTemplate(
            name="job_signals",
            uriTemplate="ltspice://results/{job_id}/signals",
            description="List of signal/trace names in a simulation result",
            mimeType="application/json",
        ),
        types.ResourceTemplate(
            name="job_measurements",
            uriTemplate="ltspice://results/{job_id}/measurements",
            description="SPICE .MEAS measurement results for a simulation",
            mimeType="application/json",
        ),
    ]


async def handle_read_resource(
    uri_str: str, state: SessionState
) -> types.ReadResourceResult:
    """Dispatch read request to the appropriate handler based on URI.

    Args:
        uri_str: The resource URI string to read
        state: Current session state

    Returns:
        ReadResourceResult with resource contents

    Raises:
        ValueError: If the URI is unknown or resource cannot be loaded
    """
    if uri_str == "ltspice://config":
        return _read_config(uri_str, state)
    elif uri_str == "ltspice://netlists/":
        return await _read_netlists_list(uri_str, state)
    elif uri_str.startswith("ltspice://netlists/"):
        filename = uri_str[len("ltspice://netlists/"):]
        return await _read_netlist_content(uri_str, filename, state)
    elif uri_str == "ltspice://results/":
        return _read_results_list(uri_str, state)
    elif uri_str == "ltspice://models/":
        return _read_models(uri_str, state)
    elif uri_str.startswith("ltspice://results/") and uri_str.endswith("/signals"):
        parts = uri_str[len("ltspice://results/"):].split("/")
        job_id = parts[0]
        return await _read_signals(uri_str, job_id, state)
    elif uri_str.startswith("ltspice://results/") and uri_str.endswith("/measurements"):
        parts = uri_str[len("ltspice://results/"):].split("/")
        job_id = parts[0]
        return await _read_measurements(uri_str, job_id, state)
    else:
        raise ValueError(f"Unknown resource URI: {uri_str}")


def _make_result(
    uri_str: str, text: str, mime: str = "application/json"
) -> types.ReadResourceResult:
    """Build a ReadResourceResult with a single TextResourceContents entry."""
    return types.ReadResourceResult(
        contents=[
            types.TextResourceContents(
                uri=AnyUrl(uri_str),
                text=text,
                mimeType=mime,
            )
        ]
    )


def _read_config(uri_str: str, state: SessionState) -> types.ReadResourceResult:
    """Return full server configuration and detected simulator info."""
    try:
        cfg = state.config
        data = {
            "working_dir": str(cfg.working_dir),
            "allowed_paths": [str(p) for p in cfg.allowed_paths],
            "simulator": cfg.simulator,
            "simulator_exe": str(cfg.simulator_exe) if cfg.simulator_exe else None,
            "detected_simulators": list(state.available_simulators.keys()),
            "default_simulator": (
                state.default_simulator.__name__
                if state.default_simulator is not None
                else None
            ),
            "max_parallel_sims": cfg.max_parallel_sims,
            "default_timeout": cfg.default_timeout,
            "max_points_returned": cfg.max_points_returned,
            "plot_dpi": cfg.plot_dpi,
            "plot_style": cfg.plot_style,
            "log_level": cfg.log_level,
        }
        return _make_result(uri_str, json.dumps(data, indent=2))
    except Exception as e:
        logger.error(f"Failed to read config resource: {e}")
        raise ValueError(f"Failed to read config: {e}") from e


async def _read_netlists_list(
    uri_str: str, state: SessionState
) -> types.ReadResourceResult:
    """List all netlist files in the working directory."""
    try:
        working_dir = state.working_dir

        def _scan() -> list[dict]:
            return [
                {"name": f.name, "uri": f"ltspice://netlists/{f.name}"}
                for f in working_dir.iterdir()
                if f.is_file() and f.suffix.lower() in NETLIST_EXTENSIONS
            ]

        netlists = await run_sync(_scan)
        netlists.sort(key=lambda x: x["name"])
        data = {"netlists": netlists, "count": len(netlists)}
        return _make_result(uri_str, json.dumps(data, indent=2))
    except Exception as e:
        logger.error(f"Failed to list netlists: {e}")
        raise ValueError(f"Failed to list netlists: {e}") from e


async def _read_netlist_content(
    uri_str: str, filename: str, state: SessionState
) -> types.ReadResourceResult:
    """Read the full text of a specific netlist file."""
    try:
        # Security: validate filename is safe and within working dir
        file_path = state.working_dir / filename
        resolved = file_path.resolve()
        working_resolved = state.working_dir.resolve()
        if not resolved.is_relative_to(working_resolved):
            raise ValueError(
                f"File {filename!r} is outside the working directory"
            )

        def _read() -> str:
            return resolved.read_text(encoding="utf-8", errors="replace")

        text = await run_sync(_read)
        return _make_result(uri_str, text, mime="text/plain")
    except ValueError:
        raise
    except Exception as e:
        logger.error(f"Failed to read netlist {filename!r}: {e}")
        raise ValueError(f"Failed to read netlist {filename!r}: {e}") from e


def _read_results_list(uri_str: str, state: SessionState) -> types.ReadResourceResult:
    """List all simulation and batch jobs with their status."""
    try:
        items: list[dict] = []

        for j in state.jobs.values():
            items.append(
                {
                    "job_id": j.job_id,
                    "type": "simulation",
                    "netlist": j.netlist.name,
                    "simulator": j.simulator,
                    "status": j.status,
                    "started_at": j.started_at.isoformat() if j.started_at else None,
                    "completed_at": (
                        j.completed_at.isoformat() if j.completed_at else None
                    ),
                }
            )

        for bj in state.batch_jobs.values():
            items.append(
                {
                    "job_id": bj.job_id,
                    "type": bj.job_type,
                    "netlist": bj.netlist.name,
                    "status": bj.status,
                    "total_runs": bj.total_runs,
                    "completed_runs": bj.completed_runs,
                    "failed_runs": bj.failed_runs,
                    "started_at": (
                        bj.started_at.isoformat() if bj.started_at else None
                    ),
                    "completed_at": (
                        bj.completed_at.isoformat() if bj.completed_at else None
                    ),
                }
            )

        # Sort by started_at descending (most recent first)
        items.sort(
            key=lambda x: x.get("started_at") or "",
            reverse=True,
        )

        data = {"jobs": items, "count": len(items)}
        return _make_result(uri_str, json.dumps(data, indent=2))
    except Exception as e:
        logger.error(f"Failed to list results: {e}")
        raise ValueError(f"Failed to list results: {e}") from e


async def _read_signals(
    uri_str: str, job_id: str, state: SessionState
) -> types.ReadResourceResult:
    """List signal/trace names from a completed simulation's .raw file."""
    try:
        job = state.jobs.get(job_id)
        if job is None:
            raise ValueError(f"Job not found: {job_id!r}")

        if job.status != "completed" or job.raw_file is None:
            data = {
                "job_id": job_id,
                "error": (
                    f"Job is not completed (status={job.status!r}) "
                    "or has no raw file"
                ),
            }
            return _make_result(uri_str, json.dumps(data, indent=2))

        raw_file = job.raw_file

        def _load_signals() -> list[str]:
            from spicelib.raw.raw_read import RawRead
            raw = RawRead(str(raw_file))
            return raw.get_trace_names()

        signal_names = await run_sync(_load_signals)
        data = {"job_id": job_id, "signals": signal_names}
        return _make_result(uri_str, json.dumps(data, indent=2))
    except ValueError:
        raise
    except Exception as e:
        logger.error(f"Failed to read signals for job {job_id!r}: {e}")
        raise ValueError(f"Failed to read signals for job {job_id!r}: {e}") from e


async def _read_measurements(
    uri_str: str, job_id: str, state: SessionState
) -> types.ReadResourceResult:
    """Return .MEAS measurement results from a completed simulation's log file."""
    try:
        job = state.jobs.get(job_id)
        if job is None:
            raise ValueError(f"Job not found: {job_id!r}")

        if job.status != "completed" or job.log_file is None:
            data = {
                "job_id": job_id,
                "error": (
                    f"Job is not completed (status={job.status!r}) "
                    "or has no log file"
                ),
            }
            return _make_result(uri_str, json.dumps(data, indent=2))

        log_file = job.log_file

        def _load_measurements() -> dict:
            from spicelib.log.ltsteps import LTSpiceLogReader
            reader = LTSpiceLogReader(str(log_file))
            measure_names = reader.get_measure_names()
            measurements: dict[str, Any] = {}
            for name in measure_names:
                values = reader.dataset.get(name.lower(), [])
                python_values = []
                for val in values:
                    if val is None or (
                        isinstance(val, str) and val.upper() == "FAILED"
                    ):
                        python_values.append(None)
                    else:
                        python_values.append(
                            float(val.item()) if hasattr(val, "item") else float(val)
                        )
                measurements[name] = python_values
            return measurements

        def _read_log_text() -> str:
            return log_file.read_text(encoding="utf-8", errors="replace")

        measurements = await run_sync(_load_measurements)

        log_text: str | None = None
        if log_file.exists():
            log_text = await run_sync(_read_log_text)

        data: dict[str, Any] = {
            "job_id": job_id,
            "measurements": measurements,
        }
        if log_text is not None:
            data["log_text"] = log_text

        return _make_result(uri_str, json.dumps(data, indent=2))
    except ValueError:
        raise
    except Exception as e:
        logger.error(f"Failed to read measurements for job {job_id!r}: {e}")
        raise ValueError(
            f"Failed to read measurements for job {job_id!r}: {e}"
        ) from e


def _read_models(uri_str: str, state: SessionState) -> types.ReadResourceResult:
    """List user-loaded libraries and their models (not built-ins)."""
    try:
        libraries: list[dict] = []

        for path, entry in state.libraries._user_libs._entries.items():
            # entry is (mtime, LibraryIndex)
            _mtime, index = entry
            models = [
                {
                    "name": m.name,
                    "type": m.model_type,
                    "parameters": m.parameters,
                }
                for m in index.models
            ]
            libraries.append({"path": str(path), "models": models})

        data = {
            "libraries": libraries,
            "note": (
                "Use the search_library tool to find models in built-in libraries."
            ),
        }
        return _make_result(uri_str, json.dumps(data, indent=2))
    except Exception as e:
        logger.error(f"Failed to read models resource: {e}")
        raise ValueError(f"Failed to read models: {e}") from e
