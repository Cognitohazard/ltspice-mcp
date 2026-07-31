"""Tests for the 3-channel schematic/authoring guidance delivery.

The guidance must reach the consuming LLM without relying on a client-side
skill being installed: an always-on floor (server instructions + tool
descriptions), a just-in-time checklist (create_schematic result), and the
single-sourced ``spice://guide`` resource.
"""

from importlib.resources import files
from pathlib import Path
from typing import cast

import pytest
from mcp import types

from ltspice_mcp.config import VALID_PROFILES, ServerConfig, ToolProfile
from ltspice_mcp.resources import (
    _PROFILE_MARKER_RE,
    _select_profile_blocks,
    handle_read_resource,
)
from ltspice_mcp.server import SERVER_INSTRUCTIONS
from ltspice_mcp.state import SessionState
from ltspice_mcp.tools.circuit import (
    CreateSchematicInput,
    handle_create_schematic,
)

_GUIDE_ASSET = files("ltspice_mcp") / "assets" / "spice_guide.md"


class TestServerInstructionsFloor:
    def test_mentions_key_schematic_guidance(self):
        # Always-on floor: the schematic build doctrine survives even when no
        # client-side skill is installed.
        assert "apply_schematic_ops" in SERVER_INSTRUCTIONS
        assert "spice://guide" in SERVER_INSTRUCTIONS
        assert "do NOT net-label" in SERVER_INSTRUCTIONS


class TestCreateSchematicChecklist:
    async def test_result_includes_layout_checklist(self, state_no_sim: SessionState):
        result = await handle_create_schematic(
            CreateSchematicInput(name="checklist_probe"), state_no_sim
        )
        text = result.content[0].text  # type: ignore[union-attr]
        assert "Layout checklist" in text
        assert "spice://guide" in text
        # Structured-aware clients show only structuredContent, so the same
        # checklist must ride in the data channel too.
        data = result.structuredContent
        assert data is not None
        assert "Layout checklist" in data["hint"]
        assert "spice://guide" in data["hint"]


class TestGuideIsEngineGeneral:
    """The packaged guide is the union of both engines (the per-engine skills
    stay engine-specific). These are coverage checks, not a byte-mirror — the
    guide is hand-authored, so its per-engine sections duplicate the skills'
    and can drift; the anchors below flag a section that went missing.
    """

    def test_covers_both_engines_and_fundamentals(self):
        guide = _GUIDE_ASSET.read_text("utf-8")
        assert "# SPICE Circuit Simulation Guide" in guide
        assert "## SPICE Fundamentals" in guide
        assert "## LTspice-Specific" in guide
        assert "## ngspice-Specific" in guide
        assert "LTspice vs ngspice" in guide  # the differences table

    def test_includes_each_engines_distinctive_sections(self):
        guide = _GUIDE_ASSET.read_text("utf-8")
        ltspice_anchors = ("### .asc Schematics", "### Other LTspice Quirks")
        ngspice_anchors = ("### .control / .endc Blocks", "### XSPICE", "### .save Directive")
        for anchor in ltspice_anchors + ngspice_anchors:
            assert anchor in guide, f"guide is missing section: {anchor}"


def _guide_for(profile: str, work_dir: Path) -> str:
    """The guide as a client on ``profile`` receives it, through the resource route."""
    config = ServerConfig(
        working_dir=work_dir,
        allowed_paths=[work_dir],
        tool_profile=cast("ToolProfile", profile),
    )
    state = SessionState.create(config, available={})
    contents = handle_read_resource("spice://guide", state).contents[0]
    assert isinstance(contents, types.TextResourceContents)
    return contents.text


class TestGuideIsProfileScoped:
    """One document, one set of simulator facts, and per-profile tool passages:
    no profile may be told to call a tool it cannot see."""

    @pytest.mark.parametrize("profile", sorted(VALID_PROFILES))
    def test_simulator_facts_are_shared_by_every_profile(self, profile: str, work_dir: Path):
        guide = _guide_for(profile, work_dir)
        for anchor in (
            "### Value Notation — CRITICAL",
            "ngspice skips `.meas` under the server's",
            "### .control / .endc Blocks",
            "### .asc Schematics",
            "LTspice vs ngspice",
        ):
            assert anchor in guide, f"{profile} guide is missing shared content: {anchor}"

    @pytest.mark.parametrize("profile", sorted(VALID_PROFILES))
    def test_no_fence_markers_reach_the_client(self, profile: str, work_dir: Path):
        # Matched on the marker pattern, not on the opening spelling: a stray
        # CLOSING marker is internal markup in front of the client too, and a
        # substring check for "<!-- profile" would walk straight past it.
        served = _guide_for(profile, work_dir)
        assert _PROFILE_MARKER_RE.search(served) is None
        assert "<!-- profile" not in served
        assert "<!-- /profile" not in served

    def test_full_and_agentic_keep_the_shipped_tool_text(self, work_dir: Path):
        full = _guide_for("full", work_dir)
        assert full == _guide_for("agentic", work_dir)
        assert "use the server's schematic tools (`create_schematic`" in full
        assert "`apply_schematic_ops` ops, so batch them in one transaction" in full
        assert "## Tool surface on this profile" not in full

    def test_consolidated_maps_the_six_tools_and_replaces_the_asc_entry(self, work_dir: Path):
        guide = _guide_for("consolidated", work_dir)
        assert "## Tool surface on this profile" in guide
        for tool in (
            "run_experiments",
            "jobs",
            "analyze_results",
            "inspect",
            "edit_schematic",
            "verify_circuit",
        ):
            assert tool in guide, f"consolidated guide never names {tool}"
        assert "use the server's schematic tools (`create_schematic`" not in guide
        assert '`edit_schematic(target=..., base="blank")` starts a new sheet' in guide

    def test_a_fence_naming_an_unknown_profile_is_rejected(self):
        """A misspelled fence matches nobody, so it would delete its block for
        every profile — a typo whose only symptom is guidance silently gone."""
        text = "before\n<!-- profile: full agentc -->\nbody\n<!-- /profile -->\nafter\n"
        with pytest.raises(ValueError, match="agentc"):
            _select_profile_blocks(text, "full")

    @pytest.mark.parametrize("profile", sorted(VALID_PROFILES))
    def test_the_shipped_guide_fences_only_real_profiles(self, profile: str, work_dir: Path):
        # The same check over the asset itself: reading it must not raise.
        assert _guide_for(profile, work_dir)

    def test_the_shipped_guide_balances_every_fence(self):
        """Every opening has its closing, and none nests — measured on the asset
        rather than assumed, because the failure is silent by nature."""
        markers = _PROFILE_MARKER_RE.findall(_GUIDE_ASSET.read_text("utf-8"))
        opens = [close for close in markers if not close]
        closes = [close for close in markers if close]
        assert opens and len(opens) == len(closes)

    @pytest.mark.parametrize(
        ("text", "problem"),
        [
            (
                "before\n<!-- profile: full -->\nbody\nafter\n",
                "opened but never closed",
            ),
            (
                "<!-- profile: full -->\na\n<!-- profile: agentic -->\nb\n<!-- /profile -->\n",
                "opened inside another fence",
            ),
            (
                "before\n<!-- /profile -->\nafter\n",
                "closed but never opened",
            ),
        ],
        ids=["unterminated", "nested", "stray-close"],
    )
    def test_a_broken_fence_structure_is_rejected(self, text: str, problem: str):
        """A fence that does not pair is not recoverable — it silently mis-scopes
        its block for every profile — so it fails at parse, loudly, like a fence
        naming a profile nobody has."""
        with pytest.raises(ValueError, match=problem):
            _select_profile_blocks(text, "full")
