"""Tests for the workflow-starter prompts."""

import re

import pytest
from mcp import types

from ltspice_mcp import prompts
from ltspice_mcp.config import VALID_PROFILES
from ltspice_mcp.tools import get_tools_for_profile

_STARTERS = {"characterize_filter", "run_and_plot", "step_response"}
_SAMPLE = {"path": "c.cir", "node": "out", "signal": "out"}


def _text(result: types.GetPromptResult) -> str:
    content = result.messages[0].content
    assert isinstance(content, types.TextContent)
    return content.text


class TestEditionSelection:
    def test_every_valid_profile_chooses_an_edition(self):
        """Total, not a default with one exception: a profile added to the
        config without a line here must fail loudly rather than inherit an
        edition that names tools it cannot see."""
        assert set(prompts.EDITIONS) == VALID_PROFILES

    def test_an_unknown_profile_is_not_silently_given_an_edition(self):
        with pytest.raises(KeyError):
            prompts.edition_for("toolbox")


class TestListPrompts:
    @pytest.mark.parametrize("profile", sorted(VALID_PROFILES))
    def test_every_profile_lists_the_three_starters(self, profile: str):
        assert {p.name for p in prompts.list_prompts(profile)} == _STARTERS

    @pytest.mark.parametrize("profile", sorted(VALID_PROFILES))
    def test_each_declares_a_required_path(self, profile: str):
        for p in prompts.list_prompts(profile):
            args = {a.name: a for a in (p.arguments or [])}
            assert "path" in args and args["path"].required


class TestGetPrompt:
    def test_interpolates_path_and_names_tools(self):
        text = _text(prompts.get_prompt("characterize_filter", {"path": "rc.cir"}, "full"))
        assert "rc.cir" in text
        assert "bode_metrics" in text and "plot_waveform" in text

    def test_optional_node_appears_only_when_given(self):
        with_node = _text(
            prompts.get_prompt("characterize_filter", {"path": "x.cir", "node": "out"}, "full")
        )
        assert "node out" in with_node
        without = _text(prompts.get_prompt("characterize_filter", {"path": "x.cir"}, "full"))
        assert "at node" not in without

    def test_run_and_plot_and_step_response_interpolate(self):
        assert "sig.cir" in _text(prompts.get_prompt("run_and_plot", {"path": "sig.cir"}, "full"))
        assert "amp.cir" in _text(prompts.get_prompt("step_response", {"path": "amp.cir"}, "full"))

    @pytest.mark.parametrize("profile", sorted(VALID_PROFILES))
    def test_missing_required_path_errors(self, profile: str):
        with pytest.raises(ValueError, match="required"):
            prompts.get_prompt("run_and_plot", {}, profile)

    def test_blank_path_errors(self):
        with pytest.raises(ValueError, match="required"):
            prompts.get_prompt("characterize_filter", {"path": "   "}, "full")

    def test_unknown_prompt_errors(self):
        with pytest.raises(ValueError, match="Unknown prompt"):
            prompts.get_prompt("nope", {}, "full")


class TestConsolidatedEdition:
    """The consolidated profile shares no tool names with the others, so its
    prompts teach the same workflows through the six tools it does expose."""

    @pytest.mark.parametrize("name", sorted(_STARTERS))
    def test_routes_through_run_experiments_and_the_follow_up_tools(self, name: str):
        text = _text(prompts.get_prompt(name, _SAMPLE, "consolidated"))
        assert "run_experiments" in text
        assert "analyze_results" in text
        assert "jobs(" in text

    def test_differs_from_the_classic_edition(self):
        classic = _text(prompts.get_prompt("characterize_filter", _SAMPLE, "full"))
        consolidated = _text(prompts.get_prompt("characterize_filter", _SAMPLE, "consolidated"))
        assert classic != consolidated
        assert "bode_filter" in consolidated, "the AC recipe replaces the bode_metrics call"

    def test_agentic_reads_the_classic_edition(self):
        assert _text(prompts.get_prompt("run_and_plot", _SAMPLE, "agentic")) == _text(
            prompts.get_prompt("run_and_plot", _SAMPLE, "full")
        )


class TestPromptsRespectProfiles:
    """A prompt's text must never instruct a tool the profile it was listed
    under hides — the workflow would dead-end on the first call."""

    @pytest.mark.parametrize("profile", sorted(VALID_PROFILES))
    def test_no_prompt_names_a_tool_its_profile_hides(self, profile: str):
        visible = {t.name for t in get_tools_for_profile(profile)[0]}
        every_tool: set[str] = set()
        for other in VALID_PROFILES:
            every_tool |= {t.name for t in get_tools_for_profile(other)[0]}
        hidden = every_tool - visible
        assert hidden, f"expected {profile!r} to hide some tools"

        for p in prompts.list_prompts(profile):
            # Quoted spans are argument payloads (recipe metric names, signals,
            # paths), not calls — several recipe names mirror a tool name of
            # another profile on purpose.
            text = re.sub(r'"[^"]*"', "", _text(prompts.get_prompt(p.name, _SAMPLE, profile)))
            for tool in hidden:
                assert not re.search(rf"\b{re.escape(tool)}\b", text), (
                    f"prompt {p.name!r} names {tool!r}, which the {profile!r} "
                    "profile does not expose"
                )
