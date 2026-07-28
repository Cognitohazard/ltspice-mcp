"""Pins on the experiment-workflow skill doc shipped in the plugin.

These are cheap string checks, not behavior tests: a skill file is read by an
agent before it ever calls a tool, so the failure mode is a doc that promises a
surface we do not have, or one that has quietly grown past what an agent will
read up front. Doc-vs-registry drift (tool names) is pinned in
test_doc_drift.py, which covers this doc too.
"""

from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SKILL_PATH = ROOT / "skills" / "spice-experiments" / "SKILL.md"

# ~800 tokens at 4 chars/token: the doc is size-budgeted on purpose — it is
# loaded before the work starts, so growth has to be a deliberate edit to this
# number rather than something that happens one paragraph at a time.
SKILL_BUDGET_CHARS = 3200

# Two rules share this denylist. (1) Absent behavior: "rerun", "columnar" and
# "case_axis" name things this six-tool surface does not have, and a doc that
# names them teaches calls that do not exist. (2) Positioning: the doc sells
# coordination and parsed numbers, never response size — "token" and "budget"
# are how that pitch would be made. Known collisions, deliberate until the
# features ship: "token" also bars control_token (so the doc's cancel teaching
# stays scoped to the submitting session) and "budget" also bars the live
# analysis_budget_s deferral knob (the doc does not teach deferrals).
FORBIDDEN_TERMS = ("rerun", "budget", "columnar", "case_axis", "token")


def _text() -> str:
    return SKILL_PATH.read_text(encoding="utf-8")


class TestSpiceExperimentsSkill:
    def test_size_is_pinned(self):
        text = _text()
        assert len(text) <= SKILL_BUDGET_CHARS, (
            f"skill doc grew to {len(text)} characters (limit {SKILL_BUDGET_CHARS})"
        )

    def test_names_no_forbidden_terms(self):
        low = _text().lower()
        for term in FORBIDDEN_TERMS:
            assert term not in low, f"skill doc names forbidden term {term!r}"

    def test_teaches_both_idioms(self):
        # Idiom 1: scalars come from .MEAS authored in the deck, read back
        # through the measurements recipe. Idiom 2: device small-signal params
        # come from a .op run with .options logopinfo, read back through the
        # operating_point recipe. Pin the mechanism words, not just the
        # headings, so gutting either idiom's teaching content fails.
        text = _text()
        assert ".meas" in text.lower()
        assert "measurements" in text
        assert "logopinfo" in text
        assert "operating_point" in text


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
