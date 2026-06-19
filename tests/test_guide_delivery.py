"""Tests for the 3-channel schematic/authoring guidance delivery.

The guidance must reach the consuming LLM without relying on a client-side
skill being installed: an always-on floor (server instructions + tool
descriptions), a just-in-time checklist (create_schematic result), and the
single-sourced ``spice://guide`` resource.
"""

from importlib.resources import files

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
