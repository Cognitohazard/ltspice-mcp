"""How many parallel copies ngspice builds from one card, against ngspice itself.

ngspice rewrites a subcircuit for an X line's ``m`` before it expands a deck
(``hierarchy_expr.card_copies``). Each deck here runs through ngspice in both
modes the server runs, with ``-n`` as the server's runs, and ngspice reports
each card's ``m`` (``@r.x1.r1[m]``); hierarchy discovery must give every card
the same copies, or, where ngspice stops on the deck, refuse it.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from ltspice_mcp.errors import MultiplierRewriteStop
from ltspice_mcp.lib.hierarchy import SemanticProfile, load_hierarchy
from ltspice_mcp.lib.simulator import NGSPICE_NO_USER_INIT
from tests.conftest import LIVENESS_S

pytestmark = pytest.mark.skipif(shutil.which("ngspice") is None, reason="ngspice not on PATH")

#: The modes a profile names; "" is ngspice's own default, run with no -D.
MODES = ("kiltpsa", "hsa", "")

#: name -> the deck's cards between its title and ``.op``, in deck order, and
#: the files it includes. Every deck hangs off ``Va`` on node ``a``.
DECKS: dict[str, tuple[str, dict[str, str]]] = {
    "x line m, nothing declared": (".subckt r1 p\nR1 p 0 1k\n.ends\nX1 a r1 m=4", {}),
    "m declared on the .subckt line": (
        ".subckt r1 p params: m=1\nR1 p 0 1k\n.ends\nX1 a r1 m=4",
        {},
    ),
    "declared m read by the card": (
        ".subckt r1 p params: m=1\nR1 p 0 1k m={m}\n.ends\nX1 a r1 m=4",
        {},
    ),
    "m declared by a body .param": (
        ".subckt r1 p\n.param m=2\nR1 p 0 1k m=3\n.ends\nX1 a r1 m=4",
        {},
    ),
    "a card's own m under an x line m": (".subckt r1 p\nR1 p 0 1k m=3\n.ends\nX1 a r1 m=4", {}),
    "m={mult} under an x line m": (
        ".subckt r1 p params: mult=3\nR1 p 0 1k m={mult}\n.ends\nX1 a r1 m=4",
        {},
    ),
    "a sibling call without m": (
        ".subckt r1 p\nR1 p 0 1k m=3\nR2 p 0 1k\n.ends\nX1 a r1 m=4\nX2 a r1",
        {},
    ),
    "own m and no x line m": (".subckt r1 p\nR1 p 0 1k m=3\n.ends\nX1 a r1", {}),
    "nested, both calls with m": (
        ".subckt inner p\nR1 p 0 1k\n.ends\n.subckt outer p\nX2 p inner m=3\n.ends\n"
        "X1 a outer m=2",
        {},
    ),
    "nested, inner declares m": (
        ".subckt inner p params: m=1\nR1 p 0 1k\n.ends\n.subckt outer p\nX2 p inner m=3\n.ends\n"
        "X1 a outer m=2",
        {},
    ),
    "nested, outer passes its declared m on": (
        ".subckt inner p\nR1 p 0 1k\n.ends\n.subckt outer p params: m=1\nX2 p inner m={m}\n.ends\n"
        "X1 a outer m=2",
        {},
    ),
    "inner call reaches a subcircuit with other parameters": (
        ".subckt leaf p params: k=1\nR1 p 0 1k\n.ends\n.subckt block p\nXleaf p leaf\n.ends\n"
        "X1 a block m=4",
        {},
    ),
    "inner subcircuit rewritten by another call": (
        ".subckt leaf p\nR1 p 0 1k\n.ends\n.subckt block p\nXleaf p leaf\n.ends\n"
        "X1 a block m=4\nX9 a leaf m=2",
        {},
    ),
    "definitions before the call, inner takes no parameters": (
        ".subckt leaf p\nR1 p 0 1k\n.ends\n.subckt block p\nXleaf p leaf\n.ends\nX1 a block m=4",
        {},
    ),
    "call before the definitions": (
        "X1 a block m=4\n.subckt block p\nXleaf p leaf\n.ends\n.subckt leaf p\nR1 p 0 1k\n.ends",
        {},
    ),
    "call before the definitions, inner with other parameters": (
        "X1 a g blk m=3\nVg g 0 1.2\n.subckt blk d g\nXf d g fet\n.ends\n"
        ".subckt fet d g params: k=1\nM0 d g 0 0 n w=1u l=1u\n.ends\n.model n nmos level=54",
        {},
    ),
    "top-down definitions, then the call": (
        ".subckt top p\nXb p block\n.ends\n.subckt block p\nXl p leaf\n.ends\n"
        ".subckt leaf p\nR1 p 0 1k\n.ends\nX1 a top m=2",
        {},
    ),
    "top-down definitions after the call": (
        "X1 a top m=2\n.subckt top p\nXb p block\n.ends\n.subckt block p\nXl p leaf\n.ends\n"
        ".subckt leaf p\nR1 p 0 1k\n.ends",
        {},
    ),
    "definitions in an include after the call": (
        'X1 a block m=4\n.include "defs.inc"',
        {"defs.inc": ".subckt block p\nXleaf p leaf\n.ends\n.subckt leaf p\nR1 p 0 1k\n.ends\n"},
    ),
    "a call with m inside an unused subcircuit": (
        ".subckt r1 p\nR1 p 0 1k m=3\n.ends\n.subckt unused p\nXu p r1 m=7\n.ends\nX1 a r1",
        {},
    ),
    "a call with m inside an unused subcircuit of a .lib section": (
        '.lib "defs.lib" sec\nX1 a r1',
        {
            "defs.lib": ".lib sec\n.subckt r1 p\nR1 p 0 1k m=3\n.ends\n"
            ".subckt unused p\nXu p r1 m=7\n.ends\n.endl sec\n"
        },
    ),
    "an unused call would make a used body take m": (
        ".subckt q p\nR1 p 0 1k\n.ends\n.subckt r1 p\nXq p q\n.ends\n"
        ".subckt unused p\nXu p r1 m=7\n.ends\nX1 a r1",
        {},
    ),
    "a leaf called with m only from an unused subcircuit": (
        ".subckt leaf p\nR1 p 0 1k\n.ends\n.subckt block p\nXleaf p leaf\n.ends\n"
        ".subckt unused p\nXu p leaf m=2\n.ends\nX1 a block m=4",
        {},
    ),
    "an unused subcircuit's call ngspice would refuse to read": (
        ".subckt r1 p\nR1 p 0 1k\n.ends\n.subckt unused p\nXu p r1 k=1 K=2\n.ends\nX1 a r1 m=2",
        {},
    ),
    "every element kind": (
        ".model dd d\n.model qq npn\n.subckt els p\nR1 p 0 1k\nC1 p 0 1p\nL1 p q 1n\n"
        "L2 q r 1n\nK1 L1 L2 0.5\nR2 r 0 1k\nR9 q 0 1meg\nI1 p 0 1m\nG1 p 0 p 0 1m\n"
        "V9 s 0 0\nR3 s 0 1k\nF1 p 0 V9 1\nE1 t 0 p 0 1\nR4 t 0 1k\nD1 p 0 dd\nQ1 p p 0 qq\n"
        "J1 p p 0 jj\nZ1 p p 0 zz\nA1 p u amp\nR5 u 0 1k\n"
        ".ends\n.model jj njf\n.model zz nmf\n.model amp gain(gain=1)\nX1 a els m=4",
        {},
    ),
}


def _write(tmp_path: Path, name: str) -> Path:
    cards, includes = DECKS[name]
    for file, text in includes.items():
        (tmp_path / file).write_text(text, encoding="ascii")
    path = tmp_path / "deck.cir"
    path.write_text(f"* multiplier\nVa a 0 1\n{cards}\n.op\n.end\n", encoding="ascii")
    return path


def _ngspice(path: Path, mode: str, names: list[str]) -> str:
    """Run ``path`` from a copy that prints each named device's ``m``:
    hierarchy discovery reads the deck without a control block."""
    prints = "".join(f"print @{name}[m]\n" for name in names)
    measured = path.with_name(f"measured_{path.name}")
    measured.write_text(
        path.read_text(encoding="ascii").replace(".end\n", f".control\nop\n{prints}.endc\n.end\n"),
        encoding="ascii",
    )
    home = path.parent / "home"
    home.mkdir(exist_ok=True)
    run = subprocess.run(
        # -n keeps a user's .spiceinit, which can set another mode, out of
        # the run, as the server's runs do; the installation's spinit, which
        # loads the code models, is still read.
        [
            "ngspice",
            NGSPICE_NO_USER_INIT,
            *(["-D", f"ngbehavior={mode}"] if mode else []),
            "-b",
            str(measured),
        ],
        capture_output=True,
        text=True,
        timeout=LIVENESS_S,
        check=False,
        cwd=home,
    )
    return run.stdout + run.stderr


def _ngspice_name(instance: tuple[str, ...]) -> str:
    """ngspice's name for a card: its letter, then the X path, then itself."""
    reference = instance[-1].lower()
    if len(instance) == 1:
        return reference
    return ".".join((reference[0], *(part.lower() for part in instance[:-1]), reference))


#: The deck with an XSPICE code model, which only the installation's spinit
#: loads.
_CODE_MODEL_DECK = "every element kind"


def _deck_params() -> list[object]:
    return [
        pytest.param(
            name,
            marks=pytest.mark.xfail(
                sys.platform == "win32",
                strict=True,
                reason=(
                    "ngspice-47 started by this test on Windows reports that it cannot find "
                    "its spinit, so the gain code model is unknown and the deck is not read; "
                    "the server's own runs and the CI install probe read the same spinit"
                ),
            ),
        )
        if name == _CODE_MODEL_DECK
        else name
        for name in DECKS
    ]


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("name", _deck_params())
def test_each_card_builds_the_copies_ngspice_builds(tmp_path: Path, name: str, mode: str):
    if ".lib" in DECKS[name][0] and ("lt" in mode or "ps" in mode):
        pytest.skip(
            "an lt or ps mode reads a .lib section as a plain include; discovery refuses it"
        )
    path = _write(tmp_path, name)
    profile = SemanticProfile("ngspice", mode)
    try:
        hierarchy = load_hierarchy(str(path), [tmp_path], profile)
    except MultiplierRewriteStop:
        assert "0 formal but 1 actual params" in _ngspice(path, mode, [])
        return

    copies = {
        _ngspice_name(row.instance): row.multiplier
        for row in hierarchy.instances
        if row.multiplier is not None and row.element != "X"
    }
    output = _ngspice(path, mode, list(copies))
    assert "0 formal but 1 actual params" not in output
    measured = {
        found[1].lower(): float(found[2])
        for found in re.finditer(r"@(\S+)\[m\]\s*=\s*(\S+)", output, re.IGNORECASE)
    }
    assert copies and set(measured) == set(copies), output
    assert {key: fact.value for key, fact in copies.items()} == pytest.approx(measured)


# --- edits and private copies on a deck with a semantic profile ---

_SPLIT_DECK = (
    "* split\nVa a 0 1\nVb b 0 1\nVg g 0 1.2\n.model n nmos level=54\n"
    ".subckt cell d g\nR1 d 0 1k m=3\nM0 d g 0 0 n w=1u l=1u\n.ends\n"
    "XA a g cell m=4\nXB b g cell\n.op\n.end\n"
)


def _materialize(tmp_path: Path, mode: str, text: str, entries: list[dict]):
    from pydantic import TypeAdapter

    from ltspice_mcp.lib.variations import (
        CircuitDeck,
        Variation,
        expand_variations,
        materialize_variants,
    )

    tmp_path.mkdir()
    path = tmp_path / "deck.cir"
    path.write_text(text, encoding="ascii")
    circuit = CircuitDeck(
        "deck", path, text, (), semantic_profile=SemanticProfile("ngspice", mode)
    )
    variations = TypeAdapter(list[Variation]).validate_python(entries)
    return materialize_variants(circuit, expand_variations([circuit], variations), tmp_path)


def _copies(tmp_path: Path, text: str, mode: str, device: str) -> float:
    path = tmp_path / f"{device.replace('.', '_')}_{mode}_{len(text)}.cir"
    path.write_text(text, encoding="ascii")
    found = re.search(rf"@{re.escape(device)}\[m\]\s*=\s*(\S+)", _ngspice(path, mode, [device]))
    assert found
    return float(found[1])


@pytest.mark.parametrize("mode", MODES)
def test_a_nested_draw_refuses_a_copy_that_would_change_what_ngspice_builds(
    tmp_path: Path, mode: str
):
    """Drawing XB.M0 gives XB a private copy of cell, and no caller of that
    copy writes m, so ngspice no longer rewrites it. Outside the HSPICE modes
    XB's R1 then builds its own 3 copies instead of the 1 the rewrite gave it,
    as ngspice shows on the deck and its hand-made copy; the draw is refused.
    In the HSPICE modes the rewrite multiplies, R1 builds 3 either way, and
    the draw goes ahead."""
    from ltspice_mcp.lib.variations import VariationError

    copied = _SPLIT_DECK.replace(
        "XB b g cell\n",
        ".subckt cellb d g\nR1 d 0 1k m=3\nM0 d g 0 0 n w=1u l=1u\n.ends\nXB b g cellb\n",
    )
    as_written = _copies(tmp_path, _SPLIT_DECK, mode, "r.xb.r1")
    with_copy = _copies(tmp_path, copied, mode, "r.xb.r1")
    rule = {"rule": "mismatch", "instance": ["XB", "M0"], "AVT": 0.003}
    entries = [{"kind": "random", "runs": 1, "seed": 5, "rules": [rule]}]
    if mode == "hsa":
        assert as_written == with_copy == 3
        (case,) = _materialize(tmp_path / "case", mode, _SPLIT_DECK, entries)
        assert any("delvto" in key for key in case.assignments), case.assignments
        return
    assert (as_written, with_copy) == (1, 3)
    with pytest.raises(VariationError) as caught:
        _materialize(tmp_path / "case", mode, _SPLIT_DECK, entries)
    assert caught.value.code == "multiplier_rewrite_changed"
    assert "XB.R1" in str(caught.value) and "Write m=1 on XB" in str(caught.value)

    # The fix it names keeps the circuit: with m=1 on XB the copy is rewritten
    # as cell is, and the case deck builds what the deck as written does.
    fixed = _SPLIT_DECK.replace("XB b g cell\n", "XB b g cell m=1\n")
    (case,) = _materialize(tmp_path / "fixed", mode, fixed, entries)
    for device in ("r.xa.r1", "r.xb.r1"):
        assert _copies(tmp_path, case.text, mode, device) == _copies(tmp_path, fixed, mode, device)


@pytest.mark.parametrize("mode", MODES)
def test_a_card_that_builds_one_copy_either_way_does_not_refuse_a_draw(tmp_path: Path, mode: str):
    """XB.J1 and XB.M0 build one copy whether or not XB's private copy of
    cell is rewritten, so drawing XB.M0 changes nothing and goes ahead; a JFET
    is a card ngspice appends m to like any other."""
    deck = _SPLIT_DECK.replace("R1 d 0 1k m=3\n", "J1 d d 0 jj\n").replace(
        ".model n nmos level=54\n", ".model n nmos level=54\n.model jj njf\n"
    )
    rule = {"rule": "mismatch", "instance": ["XB", "M0"], "AVT": 0.003}
    entries = [{"kind": "random", "runs": 1, "seed": 5, "rules": [rule]}]

    (case,) = _materialize(tmp_path / "case", mode, deck, entries)

    assert any("delvto" in key for key in case.assignments), case.assignments
    for device in ("j.xb.j1", "j.xa.j1"):
        assert _copies(tmp_path, case.text, mode, device) == _copies(tmp_path, deck, mode, device)


@pytest.mark.parametrize("mode", MODES)
def test_an_m_written_on_one_call_that_changes_another_call_is_refused(tmp_path: Path, mode: str):
    """No X line of this deck writes m, so XB's R1 builds its own 3 copies. An
    edit writing m=4 on XA makes ngspice rewrite cell for every caller, and
    outside the HSPICE modes XB's R1 then builds 1, as ngspice shows; an edit
    to XA must not change XB, so it is refused. In the HSPICE modes XB's R1
    stays at 3 and the edit is made."""
    from ltspice_mcp.lib.variations import VariationError

    unwritten = _SPLIT_DECK.replace("XA a g cell m=4\n", "XA a g cell\n")
    as_written = _copies(tmp_path, unwritten, mode, "r.xb.r1")
    edited = _copies(tmp_path, _SPLIT_DECK, mode, "r.xb.r1")
    entries = [
        {
            "kind": "assign",
            "instances": [
                {"instance": ["XA"], "attribute": "parameter", "parameter": "m", "values": ["4"]}
            ],
        }
    ]
    if mode == "hsa":
        assert as_written == edited == 3
        (case,) = _materialize(tmp_path / "case", mode, unwritten, entries)
        assert "XA a g cell m=4" in case.text
        return
    assert (as_written, edited) == (3, 1)
    with pytest.raises(VariationError) as caught:
        _materialize(tmp_path / "case", mode, unwritten, entries)
    assert caught.value.code == "multiplier_rewrite_changed"
    assert "XB.R1" in str(caught.value) and "cell" in str(caught.value)
