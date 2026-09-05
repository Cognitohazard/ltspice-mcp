"""Profile wiring for the consolidated tool surface — the only one since 0.6.0.

Locks the exposed set, the design annotations table (mcp_v1_design.md section 3,
normative), env-var profile selection, and error hints that never name a tool
the surface no longer carries.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from ltspice_mcp.config import VALID_PROFILES, ServerConfig
from ltspice_mcp.server import _ERROR_HINTS, _get_error_hint
from ltspice_mcp.tools import get_tools
from tests.conftest import TOOLS_REMOVED_IN_0_6 as _TOOLS_REMOVED_TUPLE

# Single-homed in conftest; frozen view under the name this file always used.
TOOLS_REMOVED_IN_0_6 = frozenset(_TOOLS_REMOVED_TUPLE)


CONSOLIDATED_TOOLS = frozenset(
    {
        "run_experiments",
        "jobs",
        "analyze_results",
        "edit_schematic",
        "verify_circuit",
        "inspect",
        "plot_waveform",
    }
)

# The 0.5 tool surface, deleted with the "full"/"agentic" profiles in 0.6.0.
# Kept as data because it is what the text guards scan for: an error hint or a
# prompt that still names one of these sends the caller at a tool no client can
# call any more, and the name alone reads as if it were live. Also imported by
# test_prompts. Two entries ("parameter", "recent") are ordinary English words,
# so prose that happens to use them trips the scan — the fix is to reword the
# hint or prompt, not to drop the name from this set.

# mcp_v1_design.md section 3 — the normative annotations table.
# (readOnlyHint, destructiveHint, idempotentHint, openWorldHint)
ANNOTATIONS_TABLE: dict[str, tuple[bool, bool, bool, bool]] = {
    "run_experiments": (False, False, True, True),
    "jobs": (False, True, True, False),
    "analyze_results": (False, False, True, False),
    "edit_schematic": (False, True, False, False),
    "verify_circuit": (False, True, True, False),
    "inspect": (True, False, True, False),
    # The render tool that survived the profile removal: it writes an HTML file
    # and hands it to the local desktop, so it is neither read-only nor closed.
    "plot_waveform": (False, False, False, True),
}


def _names(profile: str) -> set[str]:
    defs, _ = get_tools()
    return {tool_def.name for tool_def in defs}


class TestExposureCounts:
    """Exact membership — a tool registered by accident trips one of these."""

    def test_consolidated_exposes_exactly_the_declared_surface(self):
        assert _names("consolidated") == set(CONSOLIDATED_TOOLS)

    def test_consolidated_is_the_only_profile(self):
        assert set(VALID_PROFILES) == {"consolidated"}

    def test_the_removed_surface_is_really_gone(self):
        # If a 0.5 tool is ever re-registered, the text guards below (and the
        # prompt guard) would start rejecting a name that is legitimate again.
        assert TOOLS_REMOVED_IN_0_6.isdisjoint(_names("consolidated"))


class TestAnnotationsTable:
    """Each exposed tool against its design-table row."""

    @pytest.mark.parametrize("name", sorted(CONSOLIDATED_TOOLS))
    def test_annotations_match_design_table(self, name: str):
        defs, _ = get_tools()
        by_name = {tool_def.name: tool_def for tool_def in defs}
        annotations = by_name[name].annotations
        assert annotations is not None
        actual = (
            annotations.readOnlyHint,
            annotations.destructiveHint,
            annotations.idempotentHint,
            annotations.openWorldHint,
        )
        assert actual == ANNOTATIONS_TABLE[name]

    def test_inspect_is_the_only_read_only_tool(self):
        defs, _ = get_tools()
        read_only = {
            tool_def.name
            for tool_def in defs
            if tool_def.annotations and tool_def.annotations.readOnlyHint
        }
        assert read_only == {"inspect"}

    def test_only_the_tools_that_leave_the_process_are_open_world(self):
        """run_experiments launches a simulator; plot_waveform opens a browser.
        Nothing else reaches outside, and a tool that claims to is telling the
        client to gate a call that never leaves the box."""
        defs, _ = get_tools()
        open_world = {
            tool_def.name
            for tool_def in defs
            if tool_def.annotations and tool_def.annotations.openWorldHint
        }
        assert open_world == {"run_experiments", "plot_waveform"}


class TestEnvVarProfileSelection:
    def test_consolidated_is_a_valid_profile(self):
        assert "consolidated" in VALID_PROFILES

    def test_consolidated_selected_via_env(self, work_dir: Path, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("LTSPICE_MCP_TOOL_PROFILE", "consolidated")
        config = ServerConfig.load(work_dir / "nonexistent.toml")
        assert config.tool_profile == "consolidated"

    def test_consolidated_selected_via_toml(self, work_dir: Path):
        toml_path = work_dir / "ltspice-mcp.toml"
        toml_path.write_text('[tools]\nprofile = "consolidated"\n')
        config = ServerConfig.load(toml_path)
        assert config.tool_profile == "consolidated"


class TestErrorHints:
    """Hints are recovery instructions, so they must name callable tools."""

    def test_every_hint_resolves(self):
        for err_type in _ERROR_HINTS:
            hint = _get_error_hint(err_type)
            assert isinstance(hint, str) and hint

    def test_no_hint_names_a_removed_tool(self):
        for err_type, hint in _ERROR_HINTS.items():
            for tool in TOOLS_REMOVED_IN_0_6:
                assert not re.search(rf"\b{re.escape(tool)}\b", hint), (
                    f"{err_type.__name__} hint names removed tool {tool!r}: {hint!r}"
                )

    def test_every_hint_names_a_tool_the_caller_can_actually_call(self):
        """A hint that names no tool is a dead end: the caller just failed, and
        the recovery step has to be something it can invoke."""
        live = _names("consolidated")
        for err_type, hint in _ERROR_HINTS.items():
            named = {tool for tool in live if re.search(rf"\b{re.escape(tool)}\b", hint)}
            assert named, f"{err_type.__name__} hint names no callable tool: {hint!r}"
