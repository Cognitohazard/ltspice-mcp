"""Profile wiring for the EXPERIMENTAL consolidated six-tool surface.

Locks the exposure counts (full 49 / agentic 41 / consolidated 6), the design
annotations table (mcp_v1_design.md section 3, normative), env-var profile
selection, and profile-aware error hints that never name a tool the
consolidated profile hides.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from ltspice_mcp.config import VALID_PROFILES, ServerConfig
from ltspice_mcp.server import _ERROR_HINTS, _get_error_hint
from ltspice_mcp.tools import get_tools_for_profile

CONSOLIDATED_TOOLS = frozenset(
    {
        "run_experiments",
        "jobs",
        "analyze_results",
        "edit_schematic",
        "verify_circuit",
        "inspect",
    }
)

# mcp_v1_design.md section 3 — the normative annotations table.
# (readOnlyHint, destructiveHint, idempotentHint, openWorldHint)
ANNOTATIONS_TABLE: dict[str, tuple[bool, bool, bool, bool]] = {
    "run_experiments": (False, False, True, True),
    "jobs": (False, True, True, False),
    "analyze_results": (False, False, True, False),
    "edit_schematic": (False, True, False, False),
    "verify_circuit": (False, True, True, False),
    "inspect": (True, False, True, False),
}


def _names(profile: str) -> set[str]:
    defs, _ = get_tools_for_profile(profile)
    return {tool_def.name for tool_def in defs}


class TestExposureCounts:
    """Exact counts — a tool added to the wrong profile trips one of these."""

    def test_full_exposes_exactly_49(self):
        assert len(_names("full")) == 49

    def test_agentic_exposes_exactly_41(self):
        assert len(_names("agentic")) == 41

    def test_consolidated_exposes_exactly_the_six(self):
        assert _names("consolidated") == set(CONSOLIDATED_TOOLS)

    def test_consolidated_count_is_six(self):
        assert len(_names("consolidated")) == 6

    def test_consolidated_tools_live_only_in_consolidated(self):
        # Clean break: the six are exposed by no other profile.
        assert not (set(CONSOLIDATED_TOOLS) & _names("full"))
        assert not (set(CONSOLIDATED_TOOLS) & _names("agentic"))


class TestAnnotationsTable:
    """Each of the six against its design-table row."""

    @pytest.mark.parametrize("name", sorted(CONSOLIDATED_TOOLS))
    def test_annotations_match_design_table(self, name: str):
        defs, _ = get_tools_for_profile("consolidated")
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
        defs, _ = get_tools_for_profile("consolidated")
        read_only = {
            tool_def.name
            for tool_def in defs
            if tool_def.annotations and tool_def.annotations.readOnlyHint
        }
        assert read_only == {"inspect"}

    def test_run_experiments_is_the_only_open_world_tool(self):
        defs, _ = get_tools_for_profile("consolidated")
        open_world = {
            tool_def.name
            for tool_def in defs
            if tool_def.annotations and tool_def.annotations.openWorldHint
        }
        assert open_world == {"run_experiments"}


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


class TestProfileAwareHints:
    """Consolidated hints resolve and never name a tool the profile hides."""

    def test_every_hint_resolves_for_consolidated(self):
        for err_type in _ERROR_HINTS:
            hint = _get_error_hint(err_type, "consolidated")
            assert isinstance(hint, str) and hint

    def test_consolidated_hints_name_no_hidden_tool(self):
        hidden = _names("full") - set(CONSOLIDATED_TOOLS)
        for err_type, hint in _ERROR_HINTS.items():
            text = hint.consolidated
            for tool in hidden:
                assert not re.search(rf"\b{re.escape(tool)}\b", text), (
                    f"{err_type.__name__} consolidated hint names hidden tool {tool!r}: {text!r}"
                )

    def test_unknown_profile_falls_back_to_full(self):
        for err_type in _ERROR_HINTS:
            assert _get_error_hint(err_type, "bogus") == _get_error_hint(err_type, "full")

    def test_consolidated_hint_differs_from_full_where_full_names_a_hidden_tool(self):
        # Sanity: at least some hints were genuinely rewritten for the profile,
        # not copied from a `full` text that points at hidden tools.
        rewritten = [
            err_type.__name__
            for err_type, hint in _ERROR_HINTS.items()
            if hint.consolidated != hint.full
        ]
        assert rewritten
