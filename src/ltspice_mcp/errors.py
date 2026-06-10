"""Error hierarchy for ltspice-mcp server."""

from typing import Any


class LTSpiceMCPError(Exception):
    """Base exception for all ltspice-mcp errors.

    ``suggestions`` — optional ranked candidate dicts surfaced by the MCP
    dispatch layer as ``structuredContent`` on the error response.

    ``show_hint`` — when False, the dispatch layer does NOT append the generic
    per-error-type hint. Set it on errors that already carry precise, actionable
    guidance (e.g. "use operating_point for .OP raws"), so the generic
    "verify with check_job / simulation_summary" hint doesn't misdirect.
    """

    def __init__(
        self,
        *args: object,
        suggestions: list[dict[str, Any]] | None = None,
        show_hint: bool = True,
    ) -> None:
        super().__init__(*args)
        self.suggestions: list[dict[str, Any]] = suggestions or []
        self.show_hint: bool = show_hint


class PathSecurityError(LTSpiceMCPError):
    """Path resolves outside allowed directories."""


class NetlistError(LTSpiceMCPError):
    """Invalid netlist or component reference."""


class SimulationError(LTSpiceMCPError):
    """Simulation execution failed."""


class ConvergenceError(SimulationError):
    """Time step too small / failed to converge."""


class SingularMatrixError(SimulationError):
    """Singular matrix — floating node or short circuit."""


class MissingModelError(SimulationError):
    """Referenced subcircuit or model not found."""


class ResultError(LTSpiceMCPError):
    """Error reading simulation results."""


class JobNotFoundError(ResultError):
    """No job with the requested id exists in the job store."""


class LibraryError(LTSpiceMCPError):
    """Component library error (load, parse, or lookup failure)."""


class BatchJobError(LTSpiceMCPError):
    """Batch job error (config not found, job not found, invalid config, etc.)."""
