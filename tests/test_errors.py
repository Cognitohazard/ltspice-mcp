"""Tests for the error hierarchy."""

from ltspice_mcp.errors import (
    BatchJobError,
    ConvergenceError,
    LibraryError,
    LTSpiceMCPError,
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


class TestErrorHints:
    def test_full_hints_reference_tools(self):
        """Full-profile hints should reference MCP tool names."""
        from ltspice_mcp.server import _get_error_hint

        hint = _get_error_hint(ConvergenceError, "full")
        assert hint is not None
        assert "ltspice_edit_directive" in hint

    def test_agentic_hints_no_filtered_tools(self):
        """Agentic hints should not reference tools excluded from the profile."""
        from ltspice_mcp.server import _get_error_hint
        from ltspice_mcp.tools import get_tools_for_profile

        filtered_tools = {
            "ltspice_edit_directive",
            "ltspice_read_circuit",
            "ltspice_load_library",
            "ltspice_unload_library",
            "ltspice_list_libraries",
        }
        agentic_defs, _ = get_tools_for_profile("agentic")
        agentic_tools = {tool_def.name for tool_def in agentic_defs}
        for err_type in (
            ConvergenceError,
            SingularMatrixError,
            NetlistError,
            LibraryError,
            MissingModelError,
        ):
            hint = _get_error_hint(err_type, "agentic")
            if hint is None:
                continue
            for tool_name in filtered_tools:
                if tool_name not in agentic_tools:
                    assert tool_name not in hint, (
                        f"Agentic hint for {err_type.__name__} references "
                        f"filtered tool {tool_name}"
                    )

    def test_all_error_types_have_both_hints(self):
        """Every entry in _ERROR_HINTS should have both full and agentic variants."""
        from ltspice_mcp.server import _ERROR_HINTS

        for err_type, pair in _ERROR_HINTS.items():
            assert isinstance(pair, tuple), f"{err_type.__name__}: hint is not a tuple"
            assert len(pair) == 2, f"{err_type.__name__}: expected 2-tuple"
            assert pair[0], f"{err_type.__name__}: full hint is empty"
            assert pair[1], f"{err_type.__name__}: agentic hint is empty"
