"""Tests for the workflow-starter prompts."""

import re

import pytest
from mcp import types

from ltspice_mcp import prompts
from ltspice_mcp.tools import get_tools
from tests.test_consolidated_profile import TOOLS_REMOVED_IN_0_6

_STARTERS = {"characterize_filter", "run_and_plot", "step_response"}
_SAMPLE = {"path": "c.cir", "node": "out", "signal": "out"}


def _text(result: types.GetPromptResult) -> str:
    content = result.messages[0].content
    assert isinstance(content, types.TextContent)
    return content.text


class TestListPrompts:
    def test_the_three_starters_are_listed(self):
        assert {p.name for p in prompts.list_prompts()} == _STARTERS

    def test_each_declares_a_required_path(self):
        for p in prompts.list_prompts():
            args = {a.name: a for a in (p.arguments or [])}
            assert "path" in args and args["path"].required


class TestGetPrompt:
    def test_interpolates_path_and_names_tools(self):
        text = _text(prompts.get_prompt("characterize_filter", {"path": "rc.cir"}))
        assert "rc.cir" in text
        assert "run_experiments" in text and "analyze_results" in text

    def test_optional_node_selects_the_measured_signal(self):
        """The optional argument is read where it matters — it picks the signal
        the recipe measures — and its absence falls back to the documented
        default instead of dropping the signal from the recipe."""
        with_node = _text(
            prompts.get_prompt("characterize_filter", {"path": "x.cir", "node": "mid"})
        )
        assert "V(mid)" in with_node
        without = _text(prompts.get_prompt("characterize_filter", {"path": "x.cir"}))
        assert "V(mid)" not in without
        assert "V(out)" in without

    def test_run_and_plot_and_step_response_interpolate(self):
        assert "sig.cir" in _text(prompts.get_prompt("run_and_plot", {"path": "sig.cir"}))
        assert "amp.cir" in _text(prompts.get_prompt("step_response", {"path": "amp.cir"}))

    def test_missing_required_path_errors(self):
        with pytest.raises(ValueError, match="required"):
            prompts.get_prompt("run_and_plot", {})

    def test_blank_path_errors(self):
        with pytest.raises(ValueError, match="required"):
            prompts.get_prompt("characterize_filter", {"path": "   "})

    def test_unknown_prompt_errors(self):
        with pytest.raises(ValueError, match="Unknown prompt"):
            prompts.get_prompt("nope", {})


class TestEveryWorkflowUsesTheLiveTools:
    """Every workflow is taught through the tools the surface actually exposes."""

    @pytest.mark.parametrize("name", sorted(_STARTERS))
    def test_routes_through_run_experiments_and_the_follow_up_tools(self, name: str):
        text = _text(prompts.get_prompt(name, _SAMPLE))
        assert "run_experiments" in text
        assert "analyze_results" in text
        assert "jobs(" in text

    def test_the_ac_workflow_asks_for_the_bode_recipe(self):
        text = _text(prompts.get_prompt("characterize_filter", _SAMPLE))
        assert "bode_filter" in text, "the AC recipe is what replaced the bode_metrics call"


class TestPromptsNameOnlyLiveTools:
    """A prompt's text must never instruct a tool the client cannot call — the
    workflow would dead-end on the first call. The tools deleted in 0.6.0 are
    the live risk: they read exactly like a real call, and several of them
    survive as recipe metric names that a rewrite could confuse for one."""

    def test_no_prompt_names_a_removed_tool(self):
        visible = {t.name for t in get_tools()[0]}
        uncallable = TOOLS_REMOVED_IN_0_6 - visible
        assert uncallable, "the removed-tool list no longer names anything uncallable"

        for p in prompts.list_prompts():
            # Quoted spans are argument payloads (recipe metric names, signals,
            # paths), not calls — several recipe names mirror a removed tool
            # name on purpose.
            text = re.sub(r'"[^"]*"', "", _text(prompts.get_prompt(p.name, _SAMPLE)))
            for tool in uncallable:
                assert not re.search(rf"\b{re.escape(tool)}\b", text), (
                    f"prompt {p.name!r} names {tool!r}, which no longer exists"
                )

    def test_every_prompt_still_names_a_callable_tool(self):
        """Guards the check above from passing vacuously: stripping the quoted
        spans must not strip the prompt's actual instructions with them."""
        visible = {t.name for t in get_tools()[0]}
        for p in prompts.list_prompts():
            text = re.sub(r'"[^"]*"', "", _text(prompts.get_prompt(p.name, _SAMPLE)))
            named = {tool for tool in visible if re.search(rf"\b{re.escape(tool)}\b", text)}
            assert named, f"prompt {p.name!r} names no callable tool once quotes are stripped"
