"""The hierarchy reader's values checked against ngspice's own reading.

``lib/hierarchy_check.py`` asks ngspice to read a deck the way a run does and
print each value the reader resolved, then compares them. Outside the modes
the reader's rules were established for ('', hsa, kiltpsa), ``inspect``
answers only when ngspice confirms the values, and code that edits a deck by
the reader's answers refuses.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import ClassVar

import pytest
from spicelib.simulators.ngspice_simulator import NGspiceSimulator

from ltspice_mcp.config import ServerConfig
from ltspice_mcp.lib.hierarchy import SemanticProfile, load_hierarchy
from ltspice_mcp.lib.hierarchy_check import check_hierarchy, compare, plan, printed_values
from ltspice_mcp.lib.ngspice_mode import run_batch
from ltspice_mcp.lib.simulator import detect_simulators
from ltspice_mcp.state import SessionState
from tests.conftest import needs_controlled_startup, terminal_experiment
from tests.test_inspect_tools import _run

needs_ngspice = pytest.mark.skipif(shutil.which("ngspice") is None, reason="ngspice not on PATH")

DECK = """* checked deck
.options scale=1u
.model n NMOS(level=1 vto=.5 kp=100u)
.param a=2
.subckt cell d g w=4 l=2 r=5
.param rb={r*2}
M0 d g 0 0 n w={w} l={l} m=2
R1 d 0 {rb}
.ends
.subckt pair p
R2 p 0 1k m=3
.ends
M1 d g 0 0 n w={a*5} l=2
XC d g cell w=6 r=7
XP d pair m=4
Rtop d 0 {a*100} m=2
C1 d 0 1p
V1 d 0 1
V2 g 0 1
.op
.end
"""
"""Twelve values ngspice reports: M1's W, L and copies; XC.M0's W, L and
copies; XC.R1's resistance and copies; XP.R2's resistance and copies, which
XP's m rewrites; Rtop's resistance and copies. M1 and XC.R1 write no m and no
call rewrites theirs, so ngspice prints 0, one copy. C1, V1 and V2 have no
ngspice name in the reader."""

AGREES = {
    "status": "agrees",
    "compared": 12,
    "not_compared": {"no_ngspice_name": 3},
}


def _session(work_dir: Path, mode: str, monkeypatch: pytest.MonkeyPatch) -> SessionState:
    monkeypatch.setattr(
        NGspiceSimulator, "_compatibility_mode", NGspiceSimulator._compatibility_mode
    )
    config = ServerConfig(
        simulator="ngspice", working_dir=work_dir, allowed_paths=[work_dir], ngbehavior=mode
    )
    available = detect_simulators(config)
    assert "ngspice" in available, available
    return SessionState.create(config, available)


def _deck(work_dir: Path, text: str = DECK) -> Path:
    path = work_dir / "checked.cir"
    path.write_text(text, encoding="utf-8")
    return path


async def _hierarchy(state: SessionState, path: Path, **query) -> dict:
    (result,) = await _run(
        state, [{"kind": "hierarchy", "path": str(path), "simulator": "ngspice", **query}]
    )
    return result


@needs_ngspice
@pytest.mark.parametrize("mode", ["", "hsa", "kiltpsa"])
async def test_ngspice_confirms_the_readers_values(
    mode: str, work_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    # The query sets the mode: a configured "" means spicelib's default.
    state = _session(work_dir, "hsa", monkeypatch)
    result = await _hierarchy(state, _deck(work_dir), ngbehavior=mode)
    assert result["ok"], result
    assert result["data"]["profile"]["ngbehavior"] == mode
    assert result["data"]["ngspice_check"] == AGREES


REWRITTEN = """* rewritten m
.subckt leaf p
R1 p 0 1k m=3
R2 p 0 2k
.ends
.subckt mid p
Xl p leaf m=2
R3 p 0 1k
.ends
XA a mid m=4
XE b leaf
XM c mid
R0 a 0 0
V1 a 0 1
.op
.end
"""
"""XA's m rewrites mid, and Xl's m, inside mid, rewrites leaf, so every call of
either is rewritten, XE and XM included, though neither sets m. A mode that
multiplies (``hs`` set) gives XA.Xl.R1 24 copies, one that replaces 8."""


@needs_ngspice
@pytest.mark.parametrize(
    "mode",
    ["", "hsa", "kiltpsa", "hs", "ps", "lt", "kia", "HSA", "Hsa", "hsps", "spehs", "spelt"],
)
async def test_the_copies_ngspice_builds_under_a_rewrite_are_the_readers(
    mode: str, work_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    """The reader's copies (``card_copies``) for every card, direct and nested
    rewrites alike, are the ones ngspice builds, in each way a mode's letters
    can read (``ngspice_compat``). A 0 ohm resistor, read as 1 milliohm, has
    only its copies compared."""
    state = _session(work_dir, "hsa", monkeypatch)
    result = await _hierarchy(state, _deck(work_dir, REWRITTEN), ngbehavior=mode)
    assert result["ok"], result
    assert result["data"]["ngspice_check"] == {
        "status": "agrees",
        "compared": 17,
        "not_compared": {"no_ngspice_name": 1, "zero_resistance": 1},
    }


@needs_ngspice
@pytest.mark.parametrize("mode", ["kia", "ps", "lt"])
async def test_a_mode_the_rules_were_not_established_for_is_answered_when_ngspice_confirms_it(
    mode: str, work_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    state = _session(work_dir, mode, monkeypatch)
    result = await _hierarchy(state, _deck(work_dir))
    assert result["ok"], result
    assert result["data"]["profile"]["ngbehavior"] == mode
    assert result["data"]["ngspice_check"] == AGREES


@pytest.mark.parametrize("mode", ["kia", "HSA"])
async def test_without_ngspice_another_mode_is_refused_and_an_established_one_says_unchecked(
    mode: str, state_no_sim: SessionState, work_dir: Path
):
    """``HSA`` is not ``hsa``: ngspice matches mode letters case-sensitively
    and reads it as native mode."""
    path = _deck(work_dir)
    refused = await _hierarchy(state_no_sim, path, ngbehavior=mode)
    assert not refused["ok"]
    message = refused["error"]["message"]
    assert f"unsupported hierarchy ngbehavior {mode!r}" in message
    assert "no ngspice executable" in message
    assert "Query with one of those modes" in message

    answered = await _hierarchy(state_no_sim, path, ngbehavior="hsa")
    assert answered["ok"], answered
    check = answered["data"]["ngspice_check"]
    assert check["status"] == "not_run"
    assert check["reason"] == "no_ngspice"


@needs_ngspice
async def test_a_spinit_that_sets_another_mode_keeps_the_values_unchecked(
    work_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    state = _session(work_dir, "kiltpsa", monkeypatch)
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "spinit").write_text("set ngbehavior=hsa\n", encoding="utf-8")
    monkeypatch.setenv("SPICE_SCRIPTS", str(scripts))
    result = await _hierarchy(state, _deck(work_dir))
    assert result["data"]["ngspice_check"]["reason"] == "mode_differs"


@needs_ngspice
def test_a_deck_that_changes_before_ngspice_reads_it_is_not_compared(tmp_path: Path):
    path = _deck(tmp_path)
    hierarchy = load_hierarchy(str(path), [tmp_path], SemanticProfile("ngspice", "hsa"))
    path.write_text(DECK.replace("{a*100}", "{a*150}"), encoding="utf-8")
    check = check_hierarchy(hierarchy, NGspiceSimulator)
    assert check.status == "not_run"
    assert check.reason == "inputs_changed"


@needs_ngspice
def test_a_value_ngspice_reads_differently_is_a_difference(tmp_path: Path):
    """The reader's values for one deck against what ngspice printed for a
    variant whose Rtop is 300 ohms, both real."""
    path = _deck(tmp_path)
    hierarchy = load_hierarchy(str(path), [tmp_path], SemanticProfile("ngspice", "hsa"))
    planned, skipped = plan(hierarchy)
    variant = tmp_path / "variant.cir"
    variant.write_text(DECK.replace("{a*100}", "{a*150}"), encoding="utf-8")
    prints = "".join(f"print @{item.row.device}[{item.parameter}]\n" for item in planned)
    driver = f"* variant\n.control\nsource '{variant.as_posix()}'\n{prints}quit\n.endc\n.end\n"
    log = run_batch(
        tuple(NGspiceSimulator.spice_exe),
        "hsa",
        driver.encode(),
        cwd=tmp_path,
        clean_startup=False,
        timeout=60,
    )

    check = compare(planned, printed_values(log), skipped)
    assert check.status == "differs"
    assert check.compared == 12
    payload = check.payload()
    assert payload["differences"] == [
        {"instance": ["Rtop"], "field": "value", "reader": 200.0, "ngspice": 300.0}
    ]
    assert "Rtop value: reader 200, ngspice 300" in check.explanation()


@needs_ngspice
@pytest.mark.parametrize(("mode", "copies"), [("hsa", 8.0), ("kiltpsa", 4.0)])
def test_a_copy_count_ngspice_builds_differently_is_a_difference(
    tmp_path: Path, mode: str, copies: float
):
    """The reader's copies for the deck against what ngspice printed for a
    variant whose XC call writes m=4, both real. The rewrite reaches XC.R1,
    which writes no m, as well as XC.M0, which writes m=2."""
    path = _deck(tmp_path)
    hierarchy = load_hierarchy(str(path), [tmp_path], SemanticProfile("ngspice", mode))
    planned, skipped = plan(hierarchy)
    variant = tmp_path / "variant.cir"
    variant.write_text(DECK.replace("XC d g cell w=6 r=7", "XC d g cell w=6 r=7 m=4"), "utf-8")
    prints = "".join(f"print @{item.row.device}[{item.parameter}]\n" for item in planned)
    driver = f"* variant\n.control\nsource '{variant.as_posix()}'\n{prints}quit\n.endc\n.end\n"
    log = run_batch(
        tuple(NGspiceSimulator.spice_exe),
        mode,
        driver.encode(),
        cwd=tmp_path,
        clean_startup=False,
        timeout=60,
    )

    check = compare(planned, printed_values(log), skipped)
    assert check.payload()["differences"] == [
        {"instance": ["XC", "M0"], "field": "multiplier", "reader": 2.0, "ngspice": copies},
        {"instance": ["XC", "R1"], "field": "multiplier", "reader": 1.0, "ngspice": 4.0},
    ]


@needs_ngspice
async def test_a_deck_runs_in_another_mode_but_is_not_edited_by_instance(
    work_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    """A plain run needs no hierarchy reading, so it runs in any mode. A
    variation that edits one instance reads the deck by the reader's rules,
    which are not established for the mode, so it is refused."""
    state = _session(work_dir, "kia", monkeypatch)
    path = _deck(work_dir)
    plain = await terminal_experiment(
        state,
        {
            "request_id": "plain-kia",
            "circuits": [{"path": str(path)}],
            "execution": {"simulator": "ngspice", "wait_s": 90},
        },
    )
    assert plain["status"] == "completed", plain

    edited = await terminal_experiment(
        state,
        {
            "request_id": "edited-kia",
            "circuits": [{"path": str(path)}],
            "execution": {"simulator": "ngspice", "wait_s": 90},
            "variations": [
                {
                    "kind": "assign",
                    "instances": [
                        {"instance": ["XP", "R2"], "attribute": "value", "values": ["2k"]}
                    ],
                }
            ],
        },
    )
    assert edited["status"] != "completed", edited
    assert "unsupported hierarchy ngbehavior 'kia'" in str(edited), edited


async def test_ground_named_gnd_is_refused_in_a_mode_whose_reading_of_it_is_not_recorded(
    state_no_sim: SessionState, work_dir: Path
):
    """Values can be checked against ngspice; which node ``gnd`` is cannot,
    so outside native and hsa mode the reader refuses it, as in kiltpsa."""
    path = _deck(work_dir, DECK.replace("C1 d 0 1p", "C1 d gnd 1p"))
    result = await _hierarchy(state_no_sim, path, ngbehavior="kia")
    assert not result["ok"]
    assert (
        "unrecorded reading of the ground alias 'gnd' in kia mode" in (result["error"]["message"])
    )


@needs_ngspice
async def test_a_deck_ngspice_cannot_load_is_not_read(
    work_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    """ngspice names the deck's mode before it fails on the missing model, so
    the deck is known unread by the values it never printed."""
    state = _session(work_dir, "hsa", monkeypatch)
    path = _deck(work_dir, DECK.replace("M1 d g 0 0 n ", "M1 d g 0 0 nosuch "))
    result = await _hierarchy(state, path)
    assert result["data"]["ngspice_check"]["reason"] == "deck_not_read"
    refused = await _hierarchy(state, path, ngbehavior="kia")
    assert "did not finish reading the deck" in refused["error"]["message"]


@needs_ngspice
async def test_a_path_the_control_language_would_change_is_not_checked(
    work_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    """ngspice drops the braces from a sourced path, so it would read
    another file."""
    state = _session(work_dir, "hsa", monkeypatch)
    folder = work_dir / "{guid}"
    folder.mkdir()
    result = await _hierarchy(state, _deck(folder))
    assert result["data"]["ngspice_check"]["reason"] == "path_unsupported"


def test_an_ngspice_that_cannot_start_is_tried_once(tmp_path: Path):
    """A failure is kept, so later pages of the query do not wait on it."""

    class Missing(NGspiceSimulator):
        spice_exe: ClassVar[list[str]] = [str(tmp_path / "no-such-ngspice")]

    hierarchy = load_hierarchy(str(_deck(tmp_path)), [tmp_path], SemanticProfile("ngspice", "hsa"))
    first = check_hierarchy(hierarchy, Missing)
    assert first.reason == "ngspice_failed"
    assert check_hierarchy(hierarchy, Missing) is first


@needs_ngspice
@needs_controlled_startup
async def test_a_recoverable_run_in_another_mode_is_refused(
    work_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    """A recoverable run records where its cards came from by the reader's
    rules, which are not established for the mode."""
    state = _session(work_dir, "kia", monkeypatch)
    receipt = await terminal_experiment(
        state,
        {
            "request_id": "recoverable-kia",
            "circuits": [{"path": str(_deck(work_dir))}],
            "execution": {"simulator": "ngspice", "recoverable": True, "wait_s": 90},
        },
    )
    assert receipt["error"]["code"] == "recovery_ngbehavior_unsupported", receipt
    assert "recording where the cards" in receipt["error"]["message"]
