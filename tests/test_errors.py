"""Tests for the error hierarchy."""

import pytest

from ltspice_mcp.errors import (
    BatchJobError,
    JobNotFoundError,
    LibraryError,
    LTSpiceMCPError,
    NetlistError,
    PathSecurityError,
    ResultError,
    SimulationError,
)


class TestSuggestions:
    def test_default_empty_list(self):
        e = LibraryError("nope")
        assert e.suggestions == []

    def test_suggestions_kwarg_stored(self):
        s = [{"name": "SW", "score": 0.9}]
        e = LibraryError("model 'sx' not found", suggestions=s)
        assert e.suggestions is s
        assert str(e) == "model 'sx' not found"

    @pytest.mark.parametrize(
        "cls",
        [LibraryError, SimulationError, NetlistError, ResultError, BatchJobError],
    )
    def test_all_subclasses_accept_suggestions(self, cls):
        e = cls("msg", suggestions=[{"name": "X"}])
        assert e.suggestions == [{"name": "X"}]


class TestErrorHierarchy:
    def test_all_inherit_base(self):
        """Every error class is a subtype of LTSpiceMCPError."""
        for cls in (
            PathSecurityError,
            NetlistError,
            SimulationError,
            ResultError,
            JobNotFoundError,
            LibraryError,
            BatchJobError,
        ):
            assert issubclass(cls, LTSpiceMCPError), f"{cls.__name__} not subclass of base"

    def test_job_not_found_is_result_error(self):
        """except ResultError must keep catching unknown-job-id errors."""
        assert issubclass(JobNotFoundError, ResultError)

    def test_non_simulation_subtypes(self):
        for cls in (
            PathSecurityError,
            NetlistError,
            ResultError,
            JobNotFoundError,
            LibraryError,
            BatchJobError,
        ):
            assert not issubclass(cls, SimulationError), (
                f"{cls.__name__} should not be SimulationError"
            )

    def test_message_preserved(self):
        msg = "timestep too small at t=1.234e-6"
        err = SimulationError(msg)
        assert msg in str(err)


class TestErrorHints:
    def test_hints_reference_tools(self):
        """A hint's job is to name the tool that recovers from the failure."""
        from ltspice_mcp.server import _get_error_hint

        hint = _get_error_hint(NetlistError)
        assert hint is not None
        assert "verify_circuit" in hint

    def test_every_error_type_carries_one_hint_string(self):
        """One tool surface, one hint per error type — a hint that is not a
        non-empty string reaches the caller as an empty recovery step."""
        from ltspice_mcp.server import _ERROR_HINTS

        for err_type, hint in _ERROR_HINTS.items():
            assert isinstance(hint, str), f"{err_type.__name__}: hint is not a string"
            assert hint.strip(), f"{err_type.__name__}: hint is empty"

    def test_unhinted_error_type_returns_none(self):
        """PathSecurityError builds its hint dynamically from allowed_paths, so
        the table must not answer for it with a stale generic string."""
        from ltspice_mcp.server import _get_error_hint

        assert _get_error_hint(PathSecurityError) is None
