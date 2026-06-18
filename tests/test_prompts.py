"""Tests for the workflow-starter prompts."""

import re

import pytest
from mcp import types

from ltspice_mcp import prompts
from ltspice_mcp.tools import get_tools_for_profile


def _text(result: types.GetPromptResult) -> str:
    content = result.messages[0].content
    assert isinstance(content, types.TextContent)
    return content.text


class TestListPrompts:
    def test_lists_the_three_starters(self):
        names = {p.name for p in prompts.list_prompts()}
        assert names == {"characterize_filter", "run_and_plot", "step_response"}

    def test_each_declares_a_required_path(self):
        for p in prompts.list_prompts():
            args = {a.name: a for a in (p.arguments or [])}
            assert "path" in args and args["path"].required


class TestGetPrompt:
    def test_interpolates_path_and_names_tools(self):
        text = _text(prompts.get_prompt("characterize_filter", {"path": "rc.cir"}))
        assert "rc.cir" in text
        assert "bode_metrics" in text and "plot_waveform" in text

    def test_optional_node_appears_only_when_given(self):
        with_node = _text(
            prompts.get_prompt("characterize_filter", {"path": "x.cir", "node": "out"})
        )
        assert "node out" in with_node
        without = _text(prompts.get_prompt("characterize_filter", {"path": "x.cir"}))
        assert "at node" not in without

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


class TestPromptsRespectProfiles:
    """Prompts are listed unconditionally, so their text must never instruct a
    tool that some supported profile hides — or the workflow breaks there."""

    def test_no_prompt_names_a_profile_filtered_tool(self):
        full_defs, _ = get_tools_for_profile("full")
        agentic_defs, _ = get_tools_for_profile("agentic")
        full_only = {t.name for t in full_defs} - {t.name for t in agentic_defs}
        assert full_only, "expected the agentic profile to drop some full-only tools"

        sample = {"path": "c.cir", "node": "out", "signal": "out"}
        for p in prompts.list_prompts():
            text = _text(prompts.get_prompt(p.name, sample))
            for tool in full_only:
                assert not re.search(rf"\b{re.escape(tool)}\b", text), (
                    f"prompt {p.name!r} names full-only tool {tool!r}; it would be "
                    "unusable in the agentic profile"
                )
