"""Tests for the error hierarchy."""

from ltspice_mcp.errors import (
    BatchJobError,
    ConvergenceError,
    LTSpiceMCPError,
    LibraryError,
    MissingModelError,
    NetlistError,
    PathSecurityError,
    ResultError,
    SimulationError,
    SingularMatrixError,
)


class TestErrorHierarchy:

    def test_all_inherit_base(self):
        """Every error class is a subtype of LTSpiceMCPError."""
        for cls in (
            PathSecurityError,
            NetlistError,
            SimulationError,
            ConvergenceError,
            SingularMatrixError,
            MissingModelError,
            ResultError,
            LibraryError,
            BatchJobError,
        ):
            assert issubclass(cls, LTSpiceMCPError), f"{cls.__name__} not subclass of base"

    def test_simulation_subtypes(self):
        for cls in (ConvergenceError, SingularMatrixError, MissingModelError):
            assert issubclass(cls, SimulationError), f"{cls.__name__} not SimulationError"

    def test_non_simulation_subtypes(self):
        for cls in (PathSecurityError, NetlistError, ResultError, LibraryError, BatchJobError):
            assert not issubclass(cls, SimulationError), (
                f"{cls.__name__} should not be SimulationError"
            )

    def test_message_preserved(self):
        msg = "timestep too small at t=1.234e-6"
        err = ConvergenceError(msg)
        assert msg in str(err)

    def test_catch_simulation_catches_subtypes(self):
        """try/except SimulationError catches ConvergenceError — the real handler pattern."""
        caught = False
        try:
            raise ConvergenceError("timestep too small")
        except SimulationError:
            caught = True
        assert caught
