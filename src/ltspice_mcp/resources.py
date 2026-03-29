"""MCP resource handlers for browsing netlists, results, models, and config."""

import json
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from mcp import types
from pydantic import AnyUrl

from ltspice_mcp.lib import services
from ltspice_mcp.state import SessionState

logger = logging.getLogger(__name__)

NETLIST_EXTENSIONS = {".asc", ".net", ".sp", ".cir", ".spice"}
RouteHandler = Callable[[str, dict[str, str], SessionState], types.ReadResourceResult]


@dataclass(frozen=True)
class _Route:
    pattern: re.Pattern[str]
    handler: RouteHandler


class ResourceRouter:
    """Simple URI-template router for MCP resources."""

    def __init__(self) -> None:
        self._routes: list[_Route] = []

    def route(self, template: str) -> Callable[[Callable], Callable]:
        """Register a URI template and convert it to a regex route."""
        pattern = self._compile_template(template)

        def decorator(handler: Callable) -> Callable:
            self._routes.append(_Route(pattern=pattern, handler=handler))
            return handler

        return decorator

    def dispatch(self, uri_str: str, state: SessionState) -> types.ReadResourceResult:
        """Dispatch a URI to the first matching route."""
        for route in self._routes:
            match = route.pattern.fullmatch(uri_str)
            if match is None:
                continue
            params = match.groupdict()
            return route.handler(uri_str, params, state)
        raise ValueError(f"Unknown resource URI: {uri_str}")

    @staticmethod
    def _compile_template(template: str) -> re.Pattern[str]:
        parts = re.split(r"(\{[^}]+\})", template)
        pattern = ""
        for part in parts:
            if not part:
                continue
            if part.startswith("{") and part.endswith("}"):
                name = part[1:-1]
                pattern += f"(?P<{name}>[^/]+)"
            else:
                pattern += re.escape(part)
        return re.compile(pattern)


_router = ResourceRouter()


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


def handle_read_resource(uri_str: str, state: SessionState) -> types.ReadResourceResult:
    """Dispatch read request to the appropriate handler based on URI.

    Args:
        uri_str: The resource URI string to read
        state: Current session state

    Returns:
        ReadResourceResult with resource contents

    Raises:
        ValueError: If the URI is unknown or resource cannot be loaded
    """
    return _router.dispatch(uri_str, state)


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


@_router.route("ltspice://config")
def _read_config(
    uri_str: str, params: dict[str, str], state: SessionState
) -> types.ReadResourceResult:
    """Return full server configuration and detected simulator info."""
    del params
    cfg = state.config
    data = {
        "working_dir": str(cfg.working_dir),
        "allowed_paths": [str(p) for p in cfg.allowed_paths],
        "simulator": cfg.simulator,
        "simulator_exe": str(cfg.simulator_exe) if cfg.simulator_exe else None,
        "detected_simulators": list(state.available_simulators.keys()),
        "default_simulator": (
            state.default_simulator.__name__ if state.default_simulator is not None else None
        ),
        "max_parallel_sims": cfg.max_parallel_sims,
        "default_timeout": cfg.default_timeout,
        "max_points_returned": cfg.max_points_returned,
        "plot_dpi": cfg.plot_dpi,
        "plot_style": cfg.plot_style,
        "log_level": cfg.log_level,
    }
    return _make_result(uri_str, json.dumps(data, indent=2))


@_router.route("ltspice://netlists/")
def _read_netlists_list(
    uri_str: str, params: dict[str, str], state: SessionState
) -> types.ReadResourceResult:
    """List all netlist files in the working directory."""
    del params
    working_dir = state.working_dir
    netlists = [
        {"name": f.name, "uri": f"ltspice://netlists/{f.name}"}
        for f in working_dir.iterdir()
        if f.is_file() and f.suffix.lower() in NETLIST_EXTENSIONS
    ]
    netlists.sort(key=lambda x: x["name"])
    data = {"netlists": netlists, "count": len(netlists)}
    return _make_result(uri_str, json.dumps(data, indent=2))


@_router.route("ltspice://netlists/{filename}")
def _read_netlist_content(
    uri_str: str, params: dict[str, str], state: SessionState
) -> types.ReadResourceResult:
    """Read the full text of a specific netlist file."""
    filename = params["filename"]
    # Security: validate filename is safe and within working dir
    file_path = state.working_dir / filename
    resolved = file_path.resolve()
    working_resolved = state.working_dir.resolve()
    if not resolved.is_relative_to(working_resolved):
        raise ValueError(f"File {filename!r} is outside the working directory")
    text = resolved.read_text(encoding="utf-8", errors="replace")
    return _make_result(uri_str, text, mime="text/plain")


@_router.route("ltspice://results/")
def _read_results_list(
    uri_str: str, params: dict[str, str], state: SessionState
) -> types.ReadResourceResult:
    """List all simulation and batch jobs with their status."""
    del params
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
                "completed_at": (j.completed_at.isoformat() if j.completed_at else None),
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
                "started_at": (bj.started_at.isoformat() if bj.started_at else None),
                "completed_at": (bj.completed_at.isoformat() if bj.completed_at else None),
            }
        )

    items.sort(key=lambda x: x.get("started_at") or "", reverse=True)
    data = {"jobs": items, "count": len(items)}
    return _make_result(uri_str, json.dumps(data, indent=2))


@_router.route("ltspice://results/{job_id}/signals")
def _read_signals(
    uri_str: str, params: dict[str, str], state: SessionState
) -> types.ReadResourceResult:
    """List signal/trace names from a completed simulation's .raw file."""
    job_id = params["job_id"]
    signal_names = services.load_signal_names(job_id, state)
    data = {"job_id": job_id, "signals": signal_names}
    return _make_result(uri_str, json.dumps(data, indent=2))


@_router.route("ltspice://results/{job_id}/measurements")
def _read_measurements(
    uri_str: str, params: dict[str, str], state: SessionState
) -> types.ReadResourceResult:
    """Return .MEAS measurement results from a completed simulation's log file."""
    job_id = params["job_id"]
    meas_data = services.load_measurements(job_id, state, include_log_text=True)
    data: dict[str, Any] = {"job_id": job_id, "measurements": meas_data["measurements"]}
    if "log_text" in meas_data:
        data["log_text"] = meas_data["log_text"]
    return _make_result(uri_str, json.dumps(data, indent=2))


@_router.route("ltspice://models/")
def _read_models(
    uri_str: str, params: dict[str, str], state: SessionState
) -> types.ReadResourceResult:
    """List user-loaded libraries and their models (not built-ins)."""
    del params
    libraries: list[dict] = []

    for path, index in state.libraries.get_loaded_libraries():
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
        "note": ("Use the ltspice_search_library tool to find models in built-in libraries."),
    }
    return _make_result(uri_str, json.dumps(data, indent=2))
