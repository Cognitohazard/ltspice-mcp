"""A failed call's recovery hint follows the error's family.

The hint table names a family once; a subclass that does not say otherwise is
a member of that family and gets its recovery, while one whose family's
recovery would mislead is listed as taking none.
"""

from dataclasses import replace
from typing import Any

import pytest

from ltspice_mcp import errors
from ltspice_mcp.server import _ERROR_HINTS, _UNHINTED, _get_error_hint, _invoke
from ltspice_mcp.state import SessionState


class _UnlistedNetlistError(errors.NetlistError):
    """A member of the netlist family the hint table never names."""


class _UnlistedResultError(errors.ResultError):
    code = "result_unreadable"


def _all_subclasses(base: type) -> set[type]:
    found: set[type] = set()
    pending = [base]
    while pending:
        for sub in pending.pop().__subclasses__():
            if sub not in found:
                found.add(sub)
                pending.append(sub)
    return found


class TestHintLookup:
    def test_a_subclass_without_its_own_entry_takes_its_familys_hint(self) -> None:
        assert _get_error_hint(_UnlistedNetlistError) == _ERROR_HINTS[errors.NetlistError]
        assert _get_error_hint(_UnlistedResultError) == _ERROR_HINTS[errors.ResultError]

    def test_an_own_entry_outranks_the_familys(self) -> None:
        assert _get_error_hint(errors.JobNotFoundError) == _ERROR_HINTS[errors.JobNotFoundError]
        assert _get_error_hint(errors.JobNotFoundError) != _ERROR_HINTS[errors.ResultError]

    @pytest.mark.parametrize("err_type", sorted(_UNHINTED, key=str))
    def test_a_type_listed_as_unhinted_takes_none(self, err_type: type) -> None:
        assert _get_error_hint(err_type) is None

    def test_a_subclass_of_an_unhinted_type_takes_none(self) -> None:
        class _Narrower(errors.NoAxisError):
            pass

        assert _get_error_hint(_Narrower) is None

    def test_every_unhinted_type_has_a_hinted_ancestor(self) -> None:
        """An entry in the opt-out set only means something where the walk
        would otherwise have found a hint."""
        for err_type in _UNHINTED:
            ancestors = err_type.__mro__[1:]
            assert any(cls in _ERROR_HINTS for cls in ancestors), err_type.__name__

    def test_every_package_subclass_is_hinted_or_listed(self) -> None:
        """Each error type the package raises either reaches a hint through its
        family or is listed as taking none: nothing is unhinted by accident of
        the lookup."""
        import ltspice_mcp.server  # noqa: F401  (imports every module that defines one)

        undecided = [
            sub.__name__
            for sub in _all_subclasses(errors.LTSpiceMCPError)
            if sub.__module__.startswith("ltspice_mcp.")
            and sub is not errors.PathSecurityError
            and not any(cls in _UNHINTED or cls in _ERROR_HINTS for cls in sub.__mro__)
        ]
        # Types with no hinted ancestor at all: the hint table is the place to
        # add one if a recovery applies.
        assert sorted(undecided) == ["BatchJobError", "RasterUnavailableError"]


class TestHintReachesTheCaller:
    async def test_a_family_member_raised_by_a_handler_carries_the_family_hint(
        self, state_no_sim: SessionState
    ) -> None:
        async def raises(arguments: dict[str, Any], state: SessionState) -> Any:
            raise _UnlistedNetlistError("R1 has no value")

        registered = replace(state_no_sim.tool_dispatch["inspect"], handler=raises)
        result = await _invoke(registered, "inspect", {}, state_no_sim)
        assert result.is_error
        assert result.structured_content is not None
        assert result.structured_content["code"] == "netlist_invalid"
        assert result.structured_content["hint"] == _ERROR_HINTS[errors.NetlistError]
