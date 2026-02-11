"""Error hierarchy for ltspice-mcp server."""


class LTSpiceMCPError(Exception):
    """Base exception for all ltspice-mcp errors."""


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
