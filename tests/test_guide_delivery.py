"""Tests for the 3-channel schematic/authoring guidance delivery.

The guidance must reach the consuming LLM without relying on a client-side
skill being installed: an always-on floor (server instructions + tool
descriptions), a just-in-time checklist (create_schematic result), and the
single-sourced ``ltspice://guide`` resource.
"""

import re
from importlib.resources import files
from pathlib import Path

from ltspice_mcp.server import SERVER_INSTRUCTIONS
from ltspice_mcp.state import SessionState
from ltspice_mcp.tools.circuit import (
    CreateSchematicInput,
    handle_create_schematic,
)

_SKILL_MD = Path(__file__).resolve().parents[1] / "skills" / "ltspice" / "SKILL.md"


class TestServerInstructionsFloor:
    def test_mentions_key_schematic_guidance(self):
        # Always-on floor: the schematic build doctrine survives even when no
        # client-side skill is installed.
        assert "apply_schematic_ops" in SERVER_INSTRUCTIONS
        assert "ltspice://guide" in SERVER_INSTRUCTIONS
        assert "do NOT net-label" in SERVER_INSTRUCTIONS


class TestCreateSchematicChecklist:
    async def test_result_includes_layout_checklist(self, state_no_sim: SessionState):
        result = await handle_create_schematic(
            CreateSchematicInput(name="checklist_probe"), state_no_sim
        )
        text = result.content[0].text  # type: ignore[union-attr]
        assert "Layout checklist" in text
        assert "ltspice://guide" in text


class TestGuideDriftGuard:
    def test_asset_mirrors_skill_body(self):
        # The packaged guide asset and skills/ltspice/SKILL.md are MIRRORED:
        # the asset is the SKILL.md body (everything after the YAML
        # frontmatter). Keep them in sync — edit SKILL.md and re-copy the body
        # into the asset whenever the guidance changes.
        skill_src = _SKILL_MD.read_text("utf-8")
        frontmatter = re.match(r"^---\n.*?\n---\n", skill_src, re.DOTALL)
        assert frontmatter is not None, "SKILL.md is missing its YAML frontmatter"
        skill_body = skill_src[frontmatter.end() :]

        asset = (files("ltspice_mcp") / "assets" / "ltspice_guide.md").read_text("utf-8")
        assert asset.strip() == skill_body.strip()
