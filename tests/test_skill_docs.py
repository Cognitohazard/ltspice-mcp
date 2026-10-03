"""Pins on the Claude Code plugin's skill, which only sends a session to the guide.

The guide is the one copy of the guidance (``lib/guide.py``); the plugin keeps a
single skill because Claude Code loads a skill by its description without the
model asking, which is how a session that never read the server instructions
still reaches the guide. These pins keep it that: a pointer, small, naming the
doors that exist and sections the guide has. Tool-name drift is pinned in
test_doc_drift.py, which covers this file too.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from ltspice_mcp.lib import guide

ROOT = Path(__file__).resolve().parents[1]
SKILL_PATH = ROOT / "skills" / "spice-guide" / "SKILL.md"

#: The skill is a pointer, not a second copy of the guide: once it holds
#: guidance of its own, the two drift, which is what folding the old skills into
#: the guide ended.
SKILL_BUDGET_CHARS = 1500


def _text() -> str:
    return SKILL_PATH.read_text(encoding="utf-8")


def test_the_plugin_ships_one_skill():
    """Every further skill is a second copy of something; new guidance is a
    guide section (a task playbook when it walks a job)."""
    assert [path.parent.name for path in (ROOT / "skills").glob("*/SKILL.md")] == ["spice-guide"]


def test_it_stays_a_pointer():
    text = _text()
    assert len(text) <= SKILL_BUDGET_CHARS, (
        f"the trigger skill grew to {len(text)} characters (limit {SKILL_BUDGET_CHARS})"
    )


def test_it_names_each_door_to_the_guide():
    text = _text()
    assert 'inspect(queries=[{"kind": "guide"}])' in text
    assert "Api.guide()" in text
    assert "python -m ltspice_mcp.api guide" in text


def test_every_section_it_names_exists():
    text = " ".join(_text().split())
    named = re.findall(r'"section": "([^"]+)"', text) + re.findall(r'guide\("([^"]+)"\)', text)
    assert named, "the skill no longer shows how to read a section"
    assert set(named) <= set(guide.names())


def test_its_description_triggers_on_the_circuit_domain():
    """Claude Code matches a request against the description, so it has to
    name the work (simulators, decks, schematics), not the server's tools."""
    head = _text().split("---", 2)[1].lower()
    for trigger in ("ltspice", "ngspice", "netlist", "schematic", "spice"):
        assert trigger in head


@pytest.mark.parametrize(
    "path", sorted((ROOT / "skills").glob("*/SKILL.md")), ids=lambda p: p.parent.name
)
def test_frontmatter_declares_name_and_description(path: Path):
    # Every shipped skill doc opens with frontmatter whose name matches its
    # directory — that is what the plugin loader keys on.
    text = path.read_text(encoding="utf-8")
    assert text.startswith("---\n")
    head = text.split("---", 2)[1]
    assert f"name: {path.parent.name}" in head
    assert "description:" in head
