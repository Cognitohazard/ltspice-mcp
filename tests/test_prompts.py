"""Tests for the workflow-starter prompts."""

import re

import pytest
from mcp import types

from ltspice_mcp import prompts
from ltspice_mcp.config import VALID_PROFILES
from ltspice_mcp.tools import get_tools
from tests.test_consolidated_profile import TOOLS_REMOVED_IN_0_6

_STARTERS = {"characterize_filter", "run_and_plot", "step_response"}
_SAMPLE = {"path": "c.cir", "node": "out", "signal": "out"}
_PROFILE = "consolidated"


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
        text = _text(prompts.get_prompt("characterize_filter", {"path": "rc.cir"}, _PROFILE))
        assert "rc.cir" in text
        assert "run_experiments" in text and "analyze_results" in text

    def test_optional_node_selects_the_measured_signal(self):
        """The optional argument is read where it matters — it picks the signal
        the recipe measures — and its absence falls back to the documented
        default instead of dropping the signal from the recipe."""
        with_node = _text(
            prompts.get_prompt("characterize_filter", {"path": "x.cir", "node": "mid"}, _PROFILE)
        )
        assert "V(mid)" in with_node
        without = _text(prompts.get_prompt("characterize_filter", {"path": "x.cir"}, _PROFILE))
        assert "V(mid)" not in without
        assert "V(out)" in without

    def test_run_and_plot_and_step_response_interpolate(self):
        assert "sig.cir" in _text(
            prompts.get_prompt("run_and_plot", {"path": "sig.cir"}, _PROFILE)
        )
        assert "amp.cir" in _text(
            prompts.get_prompt("step_response", {"path": "amp.cir"}, _PROFILE)
        )

    @pytest.mark.parametrize("profile", sorted(VALID_PROFILES))
    def test_missing_required_path_errors(self, profile: str):
        with pytest.raises(ValueError, match="required"):
            prompts.get_prompt("run_and_plot", {}, profile)

    def test_blank_path_errors(self):
        with pytest.raises(ValueError, match="required"):
            prompts.get_prompt("characterize_filter", {"path": "   "}, _PROFILE)

    def test_unknown_prompt_errors(self):
        with pytest.raises(ValueError, match="Unknown prompt"):
            prompts.get_prompt("nope", {}, _PROFILE)


class TestConsolidatedEdition:
    """Every workflow is taught through the tools the surface actually exposes."""

    @pytest.mark.parametrize("name", sorted(_STARTERS))
    def test_routes_through_run_experiments_and_the_follow_up_tools(self, name: str):
        text = _text(prompts.get_prompt(name, _SAMPLE, _PROFILE))
        assert "run_experiments" in text
        assert "analyze_results" in text
        assert "jobs(" in text

    def test_the_ac_workflow_asks_for_the_bode_recipe(self):
        text = _text(prompts.get_prompt("characterize_filter", _SAMPLE, _PROFILE))
        assert "bode_filter" in text, "the AC recipe is what replaced the bode_metrics call"

    def test_every_prompt_has_a_builder_for_the_single_edition(self):
        """One surface, one edition. A prompt with no builder for it is dead
        weight: list_prompts filters it out, so it never reaches a client and
        nothing else reports it missing."""
        assert len(prompts.EDITIONS) == 1
        edition = prompts.edition_for(_PROFILE)
        missing = [
            entry.prompt.name for entry in prompts._PROMPTS if edition not in entry.builders
        ]
        assert not missing, f"prompts with no {edition!r} builder: {missing}"


class TestPromptsRespectProfiles:
    """A prompt's text must never instruct a tool the client cannot call — the
    workflow would dead-end on the first call. The tools deleted in 0.6.0 are
    the live risk: they read exactly like a real call, and several of them
    survive as recipe metric names that a rewrite could confuse for one."""

    @pytest.mark.parametrize("profile", sorted(VALID_PROFILES))
    def test_no_prompt_names_a_removed_tool(self, profile: str):
        visible = {t.name for t in get_tools()[0]}
        uncallable = TOOLS_REMOVED_IN_0_6 - visible
        assert uncallable, "the removed-tool list no longer names anything uncallable"

        for p in prompts.list_prompts(profile):
            # Quoted spans are argument payloads (recipe metric names, signals,
            # paths), not calls — several recipe names mirror a removed tool
            # name on purpose.
            text = re.sub(r'"[^"]*"', "", _text(prompts.get_prompt(p.name, _SAMPLE, profile)))
            for tool in uncallable:
                assert not re.search(rf"\b{re.escape(tool)}\b", text), (
                    f"prompt {p.name!r} names {tool!r}, which no longer exists"
                )

    @pytest.mark.parametrize("profile", sorted(VALID_PROFILES))
    def test_every_prompt_still_names_a_callable_tool(self, profile: str):
        """Guards the check above from passing vacuously: stripping the quoted
        spans must not strip the prompt's actual instructions with them."""
        visible = {t.name for t in get_tools()[0]}
        for p in prompts.list_prompts(profile):
            text = re.sub(r'"[^"]*"', "", _text(prompts.get_prompt(p.name, _SAMPLE, profile)))
            named = {tool for tool in visible if re.search(rf"\b{re.escape(tool)}\b", text)}
            assert named, f"prompt {p.name!r} names no callable tool once quotes are stripped"
