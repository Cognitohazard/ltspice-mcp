"""Pins on the SPICE workflow and bench-craft skill docs shipped in the plugin.

These are cheap string checks, not behavior tests: a skill file is read by an
agent before it ever calls a tool, so the failure mode is a doc that promises a
surface we do not have, or one that has quietly grown past what an agent will
read up front. Doc-vs-registry drift (tool names) is pinned in
test_doc_drift.py, which covers this doc too.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pytest

from ltspice_mcp.lib.recipes import DISCRIMINANTS
from ltspice_mcp.lib.response_budget import BUDGET_MIN_TOKENS
from ltspice_mcp.tools import get_tools
from tests._text import flat, names, section

ROOT = Path(__file__).resolve().parents[1]
SKILL_PATH = ROOT / "skills" / "spice-experiments" / "SKILL.md"
BENCH_SKILL_PATH = ROOT / "skills" / "spice-bench-craft" / "SKILL.md"
BENCH_NOTES_PATH = BENCH_SKILL_PATH.parent / "references" / "BENCH_NOTES.md"

# Skill docs are size-budgeted on purpose: they are loaded before the work
# starts, so growth has to be a deliberate edit to these rows rather than
# something that happens one paragraph at a time.
# Size caps on the shipped skill docs. Each sits just above what its doc needs
# today, so text that grows has to earn the room. What the bytes buy:
# spice-experiments must name every registered tool including plot_waveform
# (the doc-drift gate derives coverage from the registry), teach the caller-set
# 'budget' response cap, and describe its trigger in terms of the circuit domain
# rather than of already using the tools. spice-bench-craft must carry the
# payload-size routing clause, because the same result set costs roughly 5,200
# characters through a tool call against roughly 150 through a script that
# prints its own summary. Each body loads only when its skill fires, so these
# pins guard readability, not per-session context cost.
SKILL_BUDGETS = (
    # +230 for the routing sentence to the Python API: in thirteen sessions with
    # both doors available no agent ever mentioned the API, because nothing they
    # read named it as a route.
    pytest.param(SKILL_PATH, 4300, id="spice-experiments"),
    pytest.param(BENCH_SKILL_PATH, 8300, id="spice-bench-craft"),
)

# Two rules share this denylist. (1) Absent behavior: "rerun", "case_axis" and
# "columnar" name things this six-tool surface does not have, and a doc that
# names them teaches calls that do not exist — the budget ladder's columnar
# rung was removed before 0.6.0, so rows are objects at every budget.
# (2) Scoped knobs the doc deliberately does not teach: "control_token" (its
# cancel teaching stays scoped to the submitting session) and the
# analysis_budget_s deferral knob. The blanket "budget"/"token" bans that
# once held those two were lifted when the doc was cleared to teach the
# caller-set 'budget' response cap in estimated tokens.
FORBIDDEN_TERMS = ("rerun", "columnar", "case_axis", "control_token", "analysis_budget_s")


def _text() -> str:
    return SKILL_PATH.read_text(encoding="utf-8")


@pytest.mark.parametrize(("path", "budget"), SKILL_BUDGETS)
def test_skill_size_is_pinned(path: Path, budget: int):
    text = path.read_text(encoding="utf-8")
    assert len(text) <= budget, (
        f"{path.parent.name} skill doc grew to {len(text)} characters (limit {budget})"
    )


class TestSpiceExperimentsSkill:
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
        # operating_point recipe. Pin the mechanism words, not the headings,
        # and the recipes against the live union so a rename fails here.
        text = _text()
        assert ".meas" in text.lower()
        assert "logopinfo" in text.lower()
        for recipe in ("measurements", "operating_point"):
            assert recipe in DISCRIMINANTS, f"{recipe} is no longer a recipe"
            assert names(text, recipe), f"the skill never names {recipe}"

    def test_teaches_response_budget(self):
        # The caller-set response cap: the section that teaches it names every
        # tool that takes it, its unit, and its floor — each read off the code,
        # so the section can neither be gutted nor fall behind the surface.
        owners = {
            name
            for name, registered in get_tools()[1].items()
            if "budget" in registered.definition.input_schema.get("properties", {})
        }
        assert owners, "no tool takes a budget any more"
        budget = section(_text(), "budget")
        missing = sorted(tool for tool in owners if not names(budget, tool))
        assert not missing, f"the budget section never names {missing}"
        assert "token" in budget.lower()
        assert names(budget, str(BUDGET_MIN_TOKENS)), "the budget floor is not stated"


class TestSpiceBenchCraftSkill:
    def test_trigger_covers_each_bench_need(self):
        # The words a client matches a request against to load this skill.
        # Hyphens, case and line wrapping in the frontmatter are not the point.
        text = BENCH_SKILL_PATH.read_text(encoding="utf-8")
        description = flat(text.split("---", 2)[1].replace("-", " "))
        for trigger in ("authoring", "servo loop", "dc servo", "biasing", "template"):
            assert trigger in description, f"the skill's trigger never says {trigger!r}"

    def test_vendored_notes_are_pinned(self):
        digest = hashlib.sha256(BENCH_NOTES_PATH.read_bytes()).hexdigest()
        assert digest == "61917d4b411659b53a77d162af5f2e9c53405a1f3fb41c26a1abba89596fd4a1"

    def test_teaches_the_dc_servo_element(self):
        # The DC-only feedback inductor that closes the loop at DC.
        text = BENCH_SKILL_PATH.read_text(encoding="utf-8")
        assert re.search(r"^\s*LFB\s+out\s+inn\s+1T\b", text, re.IGNORECASE | re.MULTILINE)

    def test_ships_a_bench_template_per_analysis(self):
        # An operating-point, an open-loop AC and a closed-loop transient bench,
        # each as a complete deck an agent can render.
        decks = re.findall(
            r"```spice\n(.*?)```", BENCH_SKILL_PATH.read_text(encoding="utf-8"), re.S
        )
        for analysis in (".op", ".ac", ".tran"):
            assert any(
                re.search(rf"^{re.escape(analysis)}\b", deck, re.IGNORECASE | re.MULTILINE)
                and re.search(r"^\.end\s*$", deck, re.IGNORECASE | re.MULTILINE)
                for deck in decks
            ), f"no complete {analysis} bench template"

    def test_teaches_ngspice_batch_output(self):
        text = BENCH_SKILL_PATH.read_text(encoding="utf-8")
        # Under -b -r a measurement goes in .control as the dot-less command.
        controls = re.findall(r"^\.control\b(.*?)^\.endc\b", text, re.S | re.M | re.I)
        assert any(re.search(r"^meas\s", block, re.M | re.I) for block in controls)
        # wrdata repeats the scale column before every dumped vector.
        assert re.search(r"scale\s*,\s*v\(out\)\s*,\s*scale\s*,\s*i\(vdd\)", text, re.I)


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
