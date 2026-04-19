"""Error hierarchy for ltspice-mcp server."""

from typing import Any


class LTSpiceMCPError(Exception):
    """Base exception for all ltspice-mcp errors.

    ``suggestions`` — optional ranked candidate dicts surfaced by the MCP
    dispatch layer as ``structuredContent`` on the error response.
    """

    def __init__(
        self, *args: object, suggestions: list[dict[str, Any]] | None = None
    ) -> None:
        super().__init__(*args)
        self.suggestions: list[dict[str, Any]] = suggestions or []


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


class LibraryError(LTSpiceMCPError):
    """Component library error (load, parse, or lookup failure)."""


class BatchJobError(LTSpiceMCPError):
    """Batch job error (config not found, job not found, invalid config, etc.)."""
