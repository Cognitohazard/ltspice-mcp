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

import pytest
from spicelib.simulators.ngspice_simulator import NGspiceSimulator

from ltspice_mcp.config import ServerConfig
from ltspice_mcp.lib.hierarchy import SemanticProfile, load_hierarchy
from ltspice_mcp.lib.hierarchy_check import check_hierarchy, compare, plan, printed_values
from ltspice_mcp.lib.ngspice_mode import run_batch
from ltspice_mcp.lib.simulator import detect_simulators
from ltspice_mcp.state import SessionState
from tests.conftest import terminal_experiment
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
"""Nine values ngspice reports: M1's W and L; XC.M0's W, L and m; XC.R1's
and XP.R2's resistance; Rtop's resistance and m. XP.R2's m is set by the call
above it, and C1, V1 and V2 have no ngspice name in the reader."""

AGREES = {
    "status": "agrees",
    "compared": 9,
    "not_compared": {"m_set_by_subcircuit_call": 1, "no_ngspice_name": 3},
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
    state = _session(work_dir, mode, monkeypatch)
    result = await _hierarchy(state, _deck(work_dir))
    assert result["ok"], result
    assert result["data"]["ngspice_check"] == AGREES


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


async def test_without_ngspice_another_mode_is_refused_and_an_established_one_says_unchecked(
    state_no_sim: SessionState, work_dir: Path
):
    path = _deck(work_dir)
    refused = await _hierarchy(state_no_sim, path, ngbehavior="kia")
    assert not refused["ok"]
    assert "unsupported hierarchy ngbehavior 'kia'" in refused["error"]["message"]
    assert "no ngspice executable" in refused["error"]["message"]

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
    prints = "".join(f"print @{item.row.device}[{item.quantity}]\n" for item in planned)
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
    assert check.compared == 9
    payload = check.payload()
    assert payload["differences"] == [
        {"instance": ["Rtop"], "quantity": "value", "reader": 200.0, "ngspice": 300.0}
    ]
    assert "Rtop value: reader 200, ngspice 300" in check.explanation()


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


@needs_ngspice
async def test_ground_named_gnd_is_refused_in_a_mode_whose_reading_of_it_is_not_recorded(
    work_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    """Values can be checked against ngspice; which node ``gnd`` is cannot,
    so outside native and hsa mode the reader refuses it, as in kiltpsa."""
    state = _session(work_dir, "kia", monkeypatch)
    result = await _hierarchy(state, _deck(work_dir, DECK.replace("C1 d 0 1p", "C1 d gnd 1p")))
    assert not result["ok"]
    assert "ground alias 'gnd' in kia mode" in result["error"]["message"]
