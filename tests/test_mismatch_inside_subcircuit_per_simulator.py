"""On which simulators the server draws mismatch for a MOS device inside a
subcircuit.

A draw inside a subcircuit needs a reading of the subcircuit's body: which
``W`` and ``L`` the device takes, how the body's expressions evaluate and how
``m`` passes down. The hierarchy reader models that for LTspice and ngspice
only, so a QSPICE or Xyce deck, which gets no semantic profile, is refused
before any case runs, with the capability table's reason
(``capabilities.gap_reason(family, "hierarchy")``). Before, such a deck was
drawn on a plan that read the geometry off the ``X`` line and evaluated the
body as native ngspice, and reported a spread two to three times too large
with nothing to say so.

Each way a draw inside a subcircuit is asked for (a ``mismatch`` rule whose
prefix names an ``X`` instance, a nested ``instance`` rule, and the
``X1:delvto`` alias) runs through ``run_experiments`` and through the Python
API. LTspice and ngspice still draw: LTspice through the profile path, with a
stand-in program, and ngspice live.
"""

from __future__ import annotations

import asyncio
import shutil
from pathlib import Path
from typing import Any

import pytest
from spicelib.simulators.qspice_simulator import Qspice
from spicelib.simulators.xyce_simulator import XyceSimulator

from ltspice_mcp.api import ApiCallError
from ltspice_mcp.config import ServerConfig
from ltspice_mcp.lib import capabilities
from ltspice_mcp.lib.hierarchy import SemanticProfile
from ltspice_mcp.lib.simulator import detect_simulators
from ltspice_mcp.lib.variations import (
    CircuitDeck,
    RandomVariation,
    VariationError,
    expand_variations,
    materialize_variants,
)
from ltspice_mcp.state import SessionState
from tests.conftest import (
    FakeSim,
    SyncApi,
    stand_in_program,
    submit_experiment,
    terminal_experiment,
)
from tests.conftest import fake_host as _host
from tests.test_simulator_families import XYCE_LOG, _qspice_ac_qraw, _stand_in

#: A BSIM4 device wrapped in a subcircuit, as a foundry deck wraps every FET,
#: with its geometry set on the X line and read in the body as an expression.
DECK = """\
* a BSIM device inside a subcircuit
.subckt minifet d g s b w=1 l=0.15
m0 d g s b minifet_model w={w*1u} l={l*1u}
.model minifet_model nmos level=54 vth0=0.4 u0=0.06
.ends minifet
XN1 d g 0 0 minifet w=2 l=0.15
Vd d 0 0.9
Vg g 0 0.9
.op
.end
"""

#: Each way a draw inside the subcircuit is asked for.
VARIATIONS: dict[str, dict[str, Any]] = {
    "mismatch-rule": {
        "kind": "random",
        "runs": 2,
        "seed": 7,
        "rules": [
            {
                "rule": "mismatch",
                "prefix": "XN",
                "AVT": 3e-3,
                "vth_param": "VTH0",
                "k_param": "U0",
            }
        ],
    },
    "nested-instance-rule": {
        "kind": "random",
        "runs": 2,
        "seed": 7,
        "rules": [
            {
                "rule": "mismatch",
                "instance": ["XN1", "m0"],
                "AVT": 3e-3,
                "vth_param": "VTH0",
                "k_param": "U0",
            }
        ],
    },
    "delvto-alias": {"kind": "assign", "assign": {"XN1:delvto": [-0.02, 0.02]}},
}

#: The reason a deck with no semantic profile is refused, per family.
UNMODELLED = ("qspice", "xyce")


def _gap(family: str) -> str:
    reason = capabilities.gap_reason(family, "hierarchy")
    assert reason is not None, family
    return reason


@pytest.fixture
def qspice(work_dir: Path) -> type:
    program = stand_in_program(work_dir / "sim" / "QSPICE64.exe", b"a QSPICE build")
    return _stand_in(Qspice, program, _qspice_ac_qraw(), "Simulation completed.\n")


@pytest.fixture
def xyce(work_dir: Path) -> type:
    program = stand_in_program(work_dir / "sim" / "Xyce", b"a Xyce build")
    return _stand_in(XyceSimulator, program, b"", XYCE_LOG)


def _payload(deck: Path, request_id: str, simulator: str, variation: str) -> dict[str, Any]:
    return {
        "request_id": request_id,
        "circuits": [{"path": str(deck), "id": "dut"}],
        "variations": [VARIATIONS[variation]],
        "execution": {"simulator": simulator, "wait_s": 30},
    }


def _refusal(data: dict[str, Any]) -> tuple[str, str]:
    """The code and message a reply refused the circuit with."""
    if "error" in data:
        return data["error"].get("detail_code", data["error"]["code"]), data["error"]["message"]
    (failure,) = {(f["code"], f["message"]) for f in data["failures"]}
    return failure


@pytest.mark.parametrize("variation", sorted(VARIATIONS))
@pytest.mark.parametrize("family", UNMODELLED)
def test_a_draw_inside_a_subcircuit_is_refused_on_an_unmodelled_simulator(
    config: ServerConfig,
    work_dir: Path,
    qspice: type,
    xyce: type,
    monkeypatch: pytest.MonkeyPatch,
    family: str,
    variation: str,
):
    """Through ``run_experiments`` and the Python API: refused with the
    capability table's reason, which names what works, and no case runs."""
    _host(monkeypatch, "windows")  # the one host QSPICE runs on
    simulator = {"qspice": qspice, "xyce": xyce}[family]
    state = SessionState.create(config, available={"ltspice": FakeSim, family: simulator})
    deck = work_dir / "nested.cir"
    deck.write_text(DECK)

    _, data = asyncio.run(
        submit_experiment(state, _payload(deck, f"{family}-{variation}", family, variation))
    )
    code, message = _refusal(data)
    assert code == "mismatch_inside_subcircuit_unmodelled", data
    assert _gap(family) in message
    assert "LTspice or ngspice" in message

    state.guide_read = True
    try:
        reply = SyncApi(state).run_experiments(
            **_payload(deck, f"{family}-{variation}-api", family, variation)
        )
    except ApiCallError as refused:
        reply = refused.payload
    code, message = _refusal(reply)
    assert code == "mismatch_inside_subcircuit_unmodelled", reply
    assert _gap(family) in message

    assert simulator.launched == []


@pytest.mark.parametrize("variation", sorted(VARIATIONS))
@pytest.mark.parametrize("family", [*UNMODELLED, None])
def test_the_refusal_is_made_while_the_variations_are_read(
    tmp_path: Path, family: str | None, variation: str
):
    """The one place a deck with no profile is refused is the expansion, so
    nothing is written; a reader that is not known is refused too."""
    path = tmp_path / "nested.cir"
    path.write_text(DECK)
    circuit = CircuitDeck("dut", path, DECK, family=family)
    parsed = (
        RandomVariation.model_validate(VARIATIONS[variation])
        if VARIATIONS[variation]["kind"] == "random"
        else _assign(VARIATIONS[variation])
    )
    with pytest.raises(VariationError) as exc:
        expand_variations([circuit], [parsed])
    assert exc.value.code == "mismatch_inside_subcircuit_unmodelled"
    if family is not None:
        assert _gap(family) in str(exc.value)
    assert "LTspice or ngspice" in str(exc.value)


def test_a_mixed_deck_is_refused_rather_than_drawn_for_its_top_level_devices_alone(
    tmp_path: Path,
):
    """A prefix that names both a top-level M and an X instance would draw for
    the M alone and report a spread that leaves out what the prefix asked for."""
    text = DECK.replace(
        "Vd d 0 0.9",
        ".model nm nmos(vto=0.7 kp=100u)\nMX1 d g 0 0 nm W=10u L=1u\nVd d 0 0.9",
    ).replace("XN1 ", "MX2 d g 0 0 nm W=10u L=1u\nXX1 ")
    path = tmp_path / "mixed.cir"
    path.write_text(text)
    circuit = CircuitDeck("dut", path, text, family="xyce")
    rule = {"rule": "mismatch", "prefix": "", "AVT": 3e-3}
    with pytest.raises(VariationError) as exc:
        expand_variations(
            [circuit],
            [RandomVariation.model_validate({"kind": "random", "runs": 1, "rules": [rule]})],
        )
    assert exc.value.code == "mismatch_inside_subcircuit_unmodelled"
    # Narrowed to the top-level devices, the same deck draws.
    rule["prefix"] = "M"
    variation = RandomVariation.model_validate({"kind": "random", "runs": 1, "rules": [rule]})
    (case,) = materialize_variants(
        circuit, expand_variations([circuit], [variation]), tmp_path / "out"
    )
    assert {"random:mismatch:MX1.VTO", "random:mismatch:MX2.VTO"} <= set(case.assignments)


def _assign(entry: dict[str, Any]) -> Any:
    from ltspice_mcp.lib.variations import AssignVariation

    return AssignVariation.model_validate(entry)


@pytest.mark.parametrize("variation", sorted(VARIATIONS))
def test_ltspice_still_draws_through_its_profile(tmp_path: Path, variation: str):
    """The same deck on LTspice is drawn on the hierarchy LTspice reads."""
    path = tmp_path / "nested.cir"
    path.write_text(DECK)
    circuit = CircuitDeck(
        "dut", path, DECK, semantic_profile=SemanticProfile("ltspice"), family="ltspice"
    )
    entry = VARIATIONS[variation]
    parsed = RandomVariation.model_validate(entry) if entry["kind"] == "random" else _assign(entry)
    cases = materialize_variants(circuit, expand_variations([circuit], [parsed]), tmp_path / "out")
    assert len(cases) == 2
    drawn = [
        {k: v for k, v in case.assignments.items() if '"xn1","m0"' in k or k == "XN1:delvto"}
        for case in cases
    ]
    assert all(drawn) and drawn[0] != drawn[1]


@pytest.mark.asyncio
@pytest.mark.skipif(shutil.which("ngspice") is None, reason="ngspice not on PATH")
@pytest.mark.parametrize("variation", sorted(VARIATIONS))
async def test_ngspice_still_draws_live(work_dir: Path, variation: str):
    """The same deck on ngspice runs, and each case carries its own draw."""
    config = ServerConfig(simulator="ngspice", working_dir=work_dir, allowed_paths=[work_dir])
    available = detect_simulators(config)
    assert "ngspice" in available
    state = SessionState.create(config, available)
    deck = work_dir / "nested.cir"
    deck.write_text(DECK)
    payload = _payload(deck, f"ngspice-{variation}", "ngspice", variation)
    payload["execution"]["wait_s"] = 90
    receipt = await terminal_experiment(state, payload)
    assert receipt["status"] == "completed", receipt
    assert receipt["completeness"]["produced"] == 2, receipt
    drawn = [
        {k: v for k, v in run["assignments"].items() if '"xn1","m0"' in k or k == "XN1:delvto"}
        for run in receipt["runs"]["items"]
    ]
    assert all(drawn) and drawn[0] != drawn[1], drawn
