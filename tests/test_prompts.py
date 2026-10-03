"""Tests for the workflow-starter prompts."""

import pytest
from mcp import types

from ltspice_mcp import prompts
from ltspice_mcp.lib.recipes import DISCRIMINANTS
from ltspice_mcp.tools import get_tools
from tests._text import names
from tests.conftest import registered_tool_names, removed_tools_named_in

_STARTERS = {"characterize_filter", "run_and_plot", "step_response"}
_SAMPLE = {"path": "c.cir", "node": "out", "signal": "out"}


def _text(result: types.GetPromptResult) -> str:
    content = result.messages[0].content
    assert isinstance(content, types.TextContent)
    return content.text


def _routes_to_tool(text: str, tool: str) -> bool:
    """The prompt names ``tool``, and ``tool`` is one the registry serves."""
    assert tool in registered_tool_names(), f"{tool} is not a registered tool"
    return names(text, tool)


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
        assert _routes_to_tool(text, "run_experiments")
        assert _routes_to_tool(text, "analyze_results")

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
        for tool in ("run_experiments", "analyze_results", "jobs"):
            assert _routes_to_tool(text, tool), f"prompt {name!r} never names {tool}"

    def test_the_ac_workflow_asks_for_the_bode_recipe(self):
        text = _text(prompts.get_prompt("characterize_filter", _SAMPLE))
        assert "bode_filter" in DISCRIMINANTS
        assert names(text, "bode_filter"), "the AC workflow no longer asks for bode_filter"


class TestPromptsNameOnlyLiveTools:
    """A prompt's text must never instruct a tool the client cannot call — the
    workflow would dead-end on the first call. The removed tools are the live
    risk: they read exactly like a real call. Names that live on as a recipe or
    an op (``signal_stats``, ``thd``) are not in the removed set, so the recipe
    payloads a prompt carries are scanned along with its prose."""

    def test_no_prompt_names_a_removed_tool(self):
        for p in prompts.list_prompts():
            text = _text(prompts.get_prompt(p.name, _SAMPLE))
            named = removed_tools_named_in(text)
            assert not named, f"prompt {p.name!r} names removed tools {named}"

    def test_every_prompt_names_a_callable_tool(self):
        visible = {t.name for t in get_tools()[0]}
        for p in prompts.list_prompts():
            text = _text(prompts.get_prompt(p.name, _SAMPLE))
            assert any(names(text, tool) for tool in visible), (
                f"prompt {p.name!r} names no callable tool"
            )
