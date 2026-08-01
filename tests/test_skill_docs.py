"""Pins on the experiment-workflow skill doc shipped in the plugin.

These are cheap string checks, not behavior tests: a skill file is read by an
agent before it ever calls a tool, so the failure mode is a doc that promises a
surface we do not have, or one that has quietly grown past what an agent will
read up front. Doc-vs-registry drift (tool names) is pinned in
test_doc_drift.py, which covers this doc too.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SKILL_PATH = ROOT / "skills" / "spice-experiments" / "SKILL.md"

# ~900 tokens at 4 chars/token: the doc is size-budgeted on purpose — it is
# loaded before the work starts, so growth has to be a deliberate edit to this
# number rather than something that happens one paragraph at a time.
# Raised 3200 → 3600 when the doc was allowed to teach the caller-set
# 'budget' response cap (the benchmark that froze that pitch is over).
# Raised 3600 → 3700 when the trigger description was rewritten to fire on
# the circuit domain itself rather than on already-using-the-tools — a
# listing line that only matches agents already converted cannot convert one.
SKILL_BUDGET_CHARS = 3700

# Two rules share this denylist. (1) Absent behavior: "rerun" and "case_axis"
# name things this six-tool surface does not have, and a doc that names them
# teaches calls that do not exist; "columnar" is the ladder rung's internal
# name — the doc teaches the effect (rows as value arrays), not the jargon.
# (2) Scoped knobs the doc deliberately does not teach: "control_token" (its
# cancel teaching stays scoped to the submitting session) and the
# analysis_budget_s deferral knob. The blanket "budget"/"token" bans that
# once held those two were lifted when the doc was cleared to teach the
# caller-set 'budget' response cap in estimated tokens.
FORBIDDEN_TERMS = ("rerun", "columnar", "case_axis", "control_token", "analysis_budget_s")


def _text() -> str:
    return SKILL_PATH.read_text(encoding="utf-8")


class TestSpiceExperimentsSkill:
    def test_size_is_pinned(self):
        text = _text()
        assert len(text) <= SKILL_BUDGET_CHARS, (
            f"skill doc grew to {len(text)} characters (limit {SKILL_BUDGET_CHARS})"
        )

    def test_names_no_forbidden_terms(self):
        # Separator-tolerant: "control token" / "control-token" / "Control_Token"
        # all name the same knob the ban exists to keep out of the doc.
        text = _text()
        for term in FORBIDDEN_TERMS:
            pattern = re.compile(
                r"\b" + r"[\s_-]?".join(re.escape(p) for p in term.split("_")) + r"\b",
                re.IGNORECASE,
            )
            hit = pattern.search(text)
            assert hit is None, f"skill doc names forbidden term {term!r} as {hit.group(0)!r}"

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

    def test_teaches_response_budget(self):
        # The caller-set response cap: pin the parameter name, its unit
        # convention, and the floor, so the section can't be gutted silently.
        text = _text()
        assert "`budget`" in text
        assert "estimated tokens" in text
        assert "500" in text


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
