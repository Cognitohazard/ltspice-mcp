"""How many parallel copies ngspice builds from one card, against ngspice itself.

ngspice gives a subcircuit that declares no ``m`` an ``m`` parameter whenever an
X line calls it with ``m``, and appends ``m={m}`` to the cards of its body: the
appended ``m`` replaces a card's own outside the HSPICE modes and multiplies it
in them (``hierarchy_expr.card_copies``). Each deck here is run through ngspice
in both modes the server runs, and the copies it builds are compared with what
the server reads: hierarchy discovery (a deck with a semantic profile) and the
one-level mismatch plan (a deck without one). Every resistor hangs off the one
source at 1 V through 1 kOhm, so the source's current in mA counts the copies;
a MOS deck is read against the same deck with one plain device.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest
from spicelib.simulators.ngspice_simulator import NGspiceSimulator

from ltspice_mcp.errors import NetlistError
from ltspice_mcp.lib.hierarchy import SemanticProfile, load_hierarchy
from ltspice_mcp.lib.subckt_mismatch import ClosureFile, MismatchPlanError, build_plan
from tests.conftest import LIVENESS_S

pytestmark = pytest.mark.skipif(shutil.which("ngspice") is None, reason="ngspice not on PATH")

MODES = ("kiltpsa", "hsa")

#: name -> (definitions, X lines); every resistor is 1 kOhm from node a's side to ground.
RESISTOR_DECKS: dict[str, tuple[str, str]] = {
    "x line m, nothing declared": (".subckt r1 p\nR1 p 0 1k\n.ends", "X1 a r1 m=4"),
    "m declared on the .subckt line": (
        ".subckt r1 p params: m=1\nR1 p 0 1k\n.ends",
        "X1 a r1 m=4",
    ),
    "declared m read by the card": (
        ".subckt r1 p params: m=1\nR1 p 0 1k m={m}\n.ends",
        "X1 a r1 m=4",
    ),
    "m declared by a body .param": (
        ".subckt r1 p\n.param m=2\nR1 p 0 1k m=3\n.ends",
        "X1 a r1 m=4",
    ),
    "card's own m under an x line m": (".subckt r1 p\nR1 p 0 1k m=3\n.ends", "X1 a r1 m=4"),
    "own m and none in one body": (
        ".subckt r1 p\nR1 p 0 1k m=3\nR2 p 0 1k\n.ends",
        "X1 a r1 m=4",
    ),
    "m={mult} under an x line m": (
        ".subckt r1 p params: mult=3\nR1 p 0 1k m={mult}\n.ends",
        "X1 a r1 m=4",
    ),
    "a sibling call without m": (".subckt r1 p\nR1 p 0 1k m=3\n.ends", "X1 a r1 m=4\nX2 a r1"),
    "own m with no x line m": (".subckt r1 p\nR1 p 0 1k m=3\n.ends", "X1 a r1"),
    "nested calls, both with m": (
        ".subckt inner p\nR1 p 0 1k\n.ends\n.subckt outer p\nX2 p inner m=3\n.ends",
        "X1 a outer m=2",
    ),
    "nested, inner declares m": (
        ".subckt inner p params: m=1\nR1 p 0 1k\n.ends\n.subckt outer p\nX2 p inner m=3\n.ends",
        "X1 a outer m=2",
    ),
    "nested, outer declares m": (
        ".subckt inner p\nR1 p 0 1k\n.ends\n.subckt outer p params: m=1\nX2 p inner m=3\n.ends",
        "X1 a outer m=2",
    ),
    "nested, outer passes its m on": (
        ".subckt inner p\nR1 p 0 1k\n.ends\n.subckt outer p params: m=1\nX2 p inner m={m}\n.ends",
        "X1 a outer m=2",
    ),
    "inner call gets m from a body rewrite": (
        ".subckt leaf p params: k=1\nR1 p 0 1k\n.ends\n.subckt block p\nXleaf p leaf\n.ends",
        "X1 a block m=4",
    ),
    "inner subcircuit rewritten by another call": (
        ".subckt leaf p\nR1 p 0 1k\n.ends\n.subckt block p\nXleaf p leaf\n.ends",
        "X1 a block m=4\nX9 a leaf m=2",
    ),
    "an element that takes no m": (
        ".subckt r1 p\nV9 p q 0\nR1 q 0 1k\n.ends",
        "X1 a r1 m=4",
    ),
}

#: Decks ngspice stops on: an X line given m by a body rewrite calls a
#: subcircuit that takes no parameters.
REFUSED_DECKS: dict[str, tuple[str, str]] = {
    "rewrite reaches a call to a subcircuit with no parameters": (
        ".subckt leaf p\nR1 p 0 1k\n.ends\n.subckt block p\nXleaf p leaf\n.ends",
        "X1 a block m=4",
    ),
    "the same with an own m inside": (
        ".subckt inner p\nR1 p 0 1k m=5\n.ends\n.subckt outer p\nX2 p inner\n.ends",
        "X1 a outer m=2",
    ),
}


_MEASURE = ".control\nop\nprint -i(Va)*1000\n.endc\n"


def _deck(definitions: str, calls: str) -> str:
    return f"* multiplier\nVa a 0 1\n{definitions}\n{calls}\n.op\n.end\n"


def _ngspice(path: Path, mode: str) -> subprocess.CompletedProcess[str]:
    """Run ``path`` with a control block that prints the source's current,
    from a copy beside it: hierarchy discovery reads the deck without one."""
    measured = path.with_name(f"measured_{path.name}")
    measured.write_text(
        path.read_text(encoding="ascii").replace(".end\n", _MEASURE + ".end\n"), encoding="ascii"
    )
    return subprocess.run(
        ["ngspice", "-D", f"ngbehavior={mode}", "-b", str(measured)],
        capture_output=True,
        text=True,
        timeout=LIVENESS_S,
        check=False,
    )


def _measured(path: Path, mode: str) -> float:
    run = _ngspice(path, mode)
    found = re.search(r"-i\(va\)\*1000\s*=\s*(\S+)", run.stdout + run.stderr)
    assert found, run.stdout + run.stderr
    return float(found[1])


def _hierarchy_copies(path: Path, mode: str, element: str) -> dict[tuple[str, ...], float]:
    hierarchy = load_hierarchy(str(path), [path.parent], SemanticProfile("ngspice", mode))
    copies = {}
    for row in hierarchy.instances:
        if row.element == element:
            assert row.multiplier is not None and row.multiplier.value is not None, row
            copies[row.instance] = row.multiplier.value
    return copies


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("name", list(RESISTOR_DECKS))
def test_hierarchy_reads_the_copies_ngspice_builds(tmp_path: Path, name: str, mode: str):
    path = tmp_path / "deck.cir"
    path.write_text(_deck(*RESISTOR_DECKS[name]), encoding="ascii")

    copies = _hierarchy_copies(path, mode, "R")

    assert sum(copies.values()) == pytest.approx(_measured(path, mode), rel=1e-6), copies


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("name", list(REFUSED_DECKS))
def test_hierarchy_refuses_a_deck_ngspice_stops_on(tmp_path: Path, name: str, mode: str):
    path = tmp_path / "deck.cir"
    path.write_text(_deck(*REFUSED_DECKS[name]), encoding="ascii")

    run = _ngspice(path, mode)
    assert "0 formal but 1 actual params" in run.stdout + run.stderr

    with pytest.raises(NetlistError, match="ngspice stops on this deck"):
        load_hierarchy(str(path), [path.parent], SemanticProfile("ngspice", mode))


# --- one subcircuit level, the plan a deck with no semantic profile is drawn by ---

_MOS_MODEL = ".model n nmos level=54"
_MOS_SOURCES = "Va a 0 1\nVb b 0 1\nVg g 0 1.2"

#: name -> (definition of fet, X lines, the X line drawn); X1 sits on node a, X2 on b.
MOS_DECKS: dict[str, tuple[str, str]] = {
    "x line m, nothing declared": (
        ".subckt fet d g\nM0 d g 0 0 n w=1u l=1u\n.ends",
        "X1 a g fet m=4",
    ),
    "m declared on the .subckt line": (
        ".subckt fet d g params: m=1\nM0 d g 0 0 n w=1u l=1u\n.ends",
        "X1 a g fet m=4",
    ),
    "declared m read by the device": (
        ".subckt fet d g params: m=1\nM0 d g 0 0 n w=1u l=1u m={m}\n.ends",
        "X1 a g fet m=4",
    ),
    "m declared by a body .param": (
        ".subckt fet d g\n.param m=2\nM0 d g 0 0 n w=1u l=1u m=3\n.ends",
        "X1 a g fet m=4",
    ),
    "device's own m under an x line m": (
        ".subckt fet d g\nM0 d g 0 0 n w=1u l=1u m=3\n.ends",
        "X1 a g fet m=4",
    ),
    "m={mult} beside an x line m": (
        ".subckt fet d g params: mult=1\nM0 d g 0 0 n w=1u l=1u m={mult}\n.ends",
        "X1 a g fet mult=3 m=2",
    ),
    "both callers drawn": (
        ".subckt fet d g\nM0 d g 0 0 n w=1u l=1u m=3\n.ends",
        "X1 a g fet m=4\nX2 b g fet",
    ),
}


def _mos_deck(definition: str, calls: str) -> str:
    return f"* mos multiplier\n{_MOS_SOURCES}\n{_MOS_MODEL}\n{definition}\n{calls}\n.op\n.end\n"


def _one_device(tmp_path: Path, mode: str) -> float:
    path = tmp_path / "one.cir"
    path.write_text(
        _mos_deck(".subckt fet d g\nM0 d g 0 0 n w=1u l=1u\n.ends", "X1 a g fet"),
        encoding="ascii",
    )
    return _measured(path, mode)


def _plan_copies(text: str, selectors: list[str], mode: str, monkeypatch) -> dict[str, float]:
    monkeypatch.setattr(NGspiceSimulator, "_compatibility_mode", mode)
    plan = build_plan([ClosureFile(0, Path("/deck/deck.cir"), text)], selectors=selectors)
    copies = {}
    for target in plan.targets:
        assert target.multiplier.value is not None, target
        copies[target.x_ref] = target.multiplier.value
    return copies


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("name", list(MOS_DECKS))
def test_both_readers_count_the_copies_ngspice_builds_of_a_drawn_device(
    tmp_path: Path, name: str, mode: str, monkeypatch
):
    text = _mos_deck(*MOS_DECKS[name])
    path = tmp_path / "deck.cir"
    path.write_text(text, encoding="ascii")
    selectors = ["X1", "X2"] if "X2" in MOS_DECKS[name][1] else ["X1"]

    ngspice = _measured(path, mode) / _one_device(tmp_path, mode)
    hierarchy = _hierarchy_copies(path, mode, "M")[("X1", "M0")]
    plan = _plan_copies(text, selectors, mode, monkeypatch)["X1"]

    assert hierarchy == pytest.approx(ngspice, rel=1e-6)
    assert plan == pytest.approx(ngspice, rel=1e-6)


@pytest.mark.parametrize("mode", MODES)
def test_the_plan_refuses_a_copy_that_would_change_what_ngspice_builds(
    tmp_path: Path, mode: str, monkeypatch
):
    """Drawing X1 alone re-points it at a copy of fet, so fet keeps only X2,
    which writes no m. Outside the HSPICE modes ngspice then stops rewriting
    fet, and X2's device goes from the 1 copy the rewrite left it to its own 3:
    the plan refuses rather than run another circuit. In the HSPICE modes the
    rewrite multiplies, X2 builds 3 copies either way, and the plan draws."""
    text = _mos_deck(*MOS_DECKS["both callers drawn"])
    if mode == "hsa":
        assert _plan_copies(text, ["X1"], mode, monkeypatch) == {"X1": 12.0}
        return
    with pytest.raises(MismatchPlanError) as caught:
        _plan_copies(text, ["X1"], mode, monkeypatch)
    assert caught.value.code == "multiplier_rewrite_changed"
    assert "fet" in str(caught.value) and "X1" in str(caught.value)


# --- a nested target's private copies, drawn with a semantic profile ---

_SPLIT_DECK = (
    "* split\nVa a 0 1\nVb b 0 1\nVg g 0 1.2\n.model n nmos level=54\n"
    ".subckt cell d g\nR1 d 0 1k m=3\nM0 d g 0 0 n w=1u l=1u\n.ends\n"
    "XA a g cell m=4\nXB b g cell\n.op\n.end\n"
)


def _draw_on_xb(tmp_path: Path, mode: str):
    from pydantic import TypeAdapter

    from ltspice_mcp.lib.variations import (
        CircuitDeck,
        Variation,
        expand_variations,
        materialize_variants,
    )

    tmp_path.mkdir()
    path = tmp_path / "split.cir"
    path.write_text(_SPLIT_DECK, encoding="ascii")
    circuit = CircuitDeck(
        "split", path, _SPLIT_DECK, (), semantic_profile=SemanticProfile("ngspice", mode)
    )
    rule = {"rule": "mismatch", "instance": ["XB", "M0"], "AVT": 0.003}
    variations = TypeAdapter(list[Variation]).validate_python(
        [{"kind": "random", "runs": 1, "seed": 5, "rules": [rule]}]
    )
    return materialize_variants(circuit, expand_variations([circuit], variations), tmp_path)


@pytest.mark.parametrize("mode", MODES)
def test_a_nested_draw_refuses_a_copy_that_would_change_what_ngspice_builds(
    tmp_path: Path, mode: str
):
    """Drawing XB.M0 gives XB a private copy of cell, and no caller of that
    copy writes m, so ngspice no longer rewrites it. Outside the HSPICE modes
    XB's R1 then builds its own 3 copies instead of the 1 the rewrite gave it,
    as ngspice shows on the two decks; the draw is refused. In the HSPICE modes
    the rewrite multiplies, R1 builds 3 either way, and the draw goes ahead."""
    from ltspice_mcp.lib.variations import VariationError

    written = tmp_path / "written.cir"
    written.write_text(
        _SPLIT_DECK.replace("Va a 0 1", "Va b 0 1").replace("Vb b 0 1", "Vb a 0 1"),
        encoding="ascii",
    )
    copied = tmp_path / "copied.cir"
    copied.write_text(
        written.read_text(encoding="ascii").replace(
            "XB b g cell\n",
            ".subckt cellb d g\nR1 d 0 1k m=3\nM0 d g 0 0 n w=1u l=1u\n.ends\nXB b g cellb\n",
        ),
        encoding="ascii",
    )
    if mode == "hsa":
        assert _measured(written, mode) == pytest.approx(_measured(copied, mode), rel=1e-9)
        (case,) = _draw_on_xb(tmp_path / "case", mode)
        assert any("delvto" in key for key in case.assignments), case.assignments
        return
    assert _measured(written, mode) != pytest.approx(_measured(copied, mode), rel=1e-3)
    with pytest.raises(VariationError) as caught:
        _draw_on_xb(tmp_path / "case", mode)
    assert caught.value.code == "multiplier_rewrite_changed"
    assert "XB.R1" in str(caught.value)
