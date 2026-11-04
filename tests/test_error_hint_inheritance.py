"""A failed call's recovery hint follows the error's family.

The hint table names a family once; a subclass that does not say otherwise is
a member of that family and gets its recovery, while one whose family's
recovery would mislead is listed as taking none.
"""

from dataclasses import replace
from typing import Any

import pytest

from ltspice_mcp import errors
from ltspice_mcp.errors import ERROR_HINTS, hint_for_type
from ltspice_mcp.server import _invoke
from ltspice_mcp.state import SessionState
from tests.test_error_codes import _error_classes


@pytest.fixture
def family_member() -> type[errors.LTSpiceMCPError]:
    """A NetlistError subclass the hint table never names, made for one test."""
    return type("UnlistedNetlistError", (errors.NetlistError,), {})


def _unhinted() -> set[type]:
    """The package's error types that opt out of their family's hint."""
    return {err for err in _error_classes() if err.__dict__.get("takes_family_hint") is False}


_UNHINTED = _unhinted()


class TestHintLookup:
    def test_a_subclass_without_its_own_entry_takes_its_familys_hint(
        self, family_member: type[errors.LTSpiceMCPError]
    ) -> None:
        assert hint_for_type(family_member) == ERROR_HINTS[errors.NetlistError]

    def test_an_envelope_surface_gives_a_subclass_the_same_hint_as_a_raise(
        self, family_member: type[errors.LTSpiceMCPError]
    ) -> None:
        """A failure carried in an envelope (a coverage row, a batch item, a
        jobs error) is hinted by the same lookup as one a handler raises."""
        assert errors.error_hint(family_member("R1 has no value")) == hint_for_type(family_member)
        assert (
            errors.error_hint(family_member("R1 has no value"))
            == (ERROR_HINTS[errors.NetlistError])
        )

    def test_an_own_entry_outranks_the_familys(self) -> None:
        assert hint_for_type(errors.JobNotFoundError) == ERROR_HINTS[errors.JobNotFoundError]
        assert hint_for_type(errors.JobNotFoundError) != ERROR_HINTS[errors.ResultError]

    @pytest.mark.parametrize("err_type", sorted(_UNHINTED, key=str))
    def test_a_type_listed_as_unhinted_takes_none(self, err_type: type) -> None:
        assert hint_for_type(err_type) is None

    def test_every_unhinted_type_has_a_hinted_ancestor(self) -> None:
        """An entry in the opt-out set only means something where the walk
        would otherwise have found a hint."""
        for err_type in _UNHINTED:
            ancestors = err_type.__mro__[1:]
            assert any(cls in ERROR_HINTS for cls in ancestors), err_type.__name__

    def test_every_package_subclass_is_hinted_or_listed(self) -> None:
        """Each error type the package raises either reaches a hint through its
        family or is listed as taking none: nothing is unhinted by accident of
        the lookup."""
        undecided = [
            sub.__name__
            for sub in _error_classes()
            if sub not in (errors.LTSpiceMCPError, errors.PathSecurityError)
            and not any(cls in _UNHINTED or cls in ERROR_HINTS for cls in sub.__mro__)
        ]
        # Types with no hinted ancestor at all: the hint table is the place to
        # add one if a recovery applies.
        assert sorted(undecided) == ["BatchJobError", "RasterUnavailableError"]


class TestHintReachesTheCaller:
    async def test_a_family_member_raised_by_a_handler_carries_the_family_hint(
        self, state_no_sim: SessionState, family_member: type[errors.LTSpiceMCPError]
    ) -> None:
        async def raises(arguments: dict[str, Any], state: SessionState) -> Any:
            raise family_member("R1 has no value")

        registered = replace(state_no_sim.tool_dispatch["inspect"], handler=raises)
        result = await _invoke(registered, "inspect", {}, state_no_sim)
        assert result.is_error
        assert result.structured_content is not None
        assert result.structured_content["code"] == "netlist_invalid"
        assert result.structured_content["hint"] == ERROR_HINTS[errors.NetlistError]
