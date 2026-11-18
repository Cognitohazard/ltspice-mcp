"""A subcircuit that calls itself has no circuit to flatten, in either mode.

The netlist comparison flattens strictly first and, after a port-arity
mismatch, again leniently, keeping the mismatched instance as a black box. The
lenient pass used to black-box a recursive instance too, without a record, so
whether recursion was an error depended on whether an arity mismatch came
first, and the part of the circuit under it went uncompared with nothing to
say so. Now both passes raise, and each caller states the failure: the
equivalence comparison of ``verify_circuit`` and of ``edit_schematic``'s
reference stage, and the connections ``diff_vs_prior`` compares.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ltspice_mcp.lib.netlist_diff import connection_delta
from ltspice_mcp.lib.netlist_graph import (
    NetlistGraphError,
    compare_graphs,
    parse_netlist_graph,
)
from ltspice_mcp.state import SessionState
from ltspice_mcp.tools import schematic_edit as se
from ltspice_mcp.tools import verify as vc
from ltspice_mcp.tools.schematic_edit import handle_edit_schematic
from tests import _fake_netlister as fake_netlister
from tests.conftest import LTSPICE_READER
from tests.test_edit_schematic import _ADD_R2, _edit_input, _sha
from tests.test_edit_schematic import _assert_schema as _assert_edit_schema
from tests.test_verify_circuit import _run, _with_ltspice, _write

#: ``X0`` gives ``cell`` three nodes for its two ports, then ``X2`` calls
#: ``loop``, which calls itself.
ARITY_THEN_RECURSION = """\
* an arity mismatch before a recursive definition
.subckt cell a b
R1 a b 1k
.ends cell
.subckt loop a b
X1 a b loop
.ends loop
V1 in 0 1
X0 in out 0 cell
X2 out 0 loop
.end
"""

#: The same deck with no arity mismatch, which the strict pass already refused.
RECURSION_ONLY = ARITY_THEN_RECURSION.replace("X0 in out 0 cell", "X0 in out cell")

PLAIN = "* plain\nV1 in 0 1\nR1 in out 1k\nR2 out 0 1k\n.end\n"


@pytest.mark.parametrize(
    "deck", [ARITY_THEN_RECURSION, RECURSION_ONLY], ids=["after-arity", "alone"]
)
@pytest.mark.parametrize("side", ["reference", "candidate"])
def test_the_comparison_refuses_a_recursive_definition_in_either_mode(deck: str, side: str):
    reference, candidate = (deck, PLAIN) if side == "reference" else (PLAIN, deck)
    with pytest.raises(NetlistGraphError, match=r"recursive \.SUBCKT definition: loop -> loop"):
        compare_graphs(reference, candidate, reader=LTSPICE_READER)


def test_the_connections_since_the_prior_export_refuse_it():
    """``diff_vs_prior``'s wiring comparison asks the same flattening."""
    prior = parse_netlist_graph(ARITY_THEN_RECURSION, reader=LTSPICE_READER)
    fresh = parse_netlist_graph(ARITY_THEN_RECURSION.replace("1k", "2k"), reader=LTSPICE_READER)
    with pytest.raises(NetlistGraphError, match="recursive"):
        connection_delta(prior, fresh)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "deck", [ARITY_THEN_RECURSION, RECURSION_ONLY], ids=["after-arity", "alone"]
)
async def test_verify_circuit_states_the_failure_and_its_fix(
    state_no_sim: SessionState, work_dir: Path, deck: str
):
    path = _write(work_dir, "loop.cir", deck)
    data = await _run(state_no_sim, path=str(path), checks=["compare"], reference=PLAIN)
    assert "comparison" not in data
    (failure,) = [f for f in data["failures"] if f["stage"] == "compare"]
    assert "recursive .SUBCKT definition: loop -> loop" in failure["error"]
    assert "calls itself" in failure["remedy"]


@pytest.mark.asyncio
async def test_diff_vs_prior_says_the_connections_were_not_compared(
    config, work_dir: Path, asc_symbols, monkeypatch: pytest.MonkeyPatch
):
    exports = iter([ARITY_THEN_RECURSION, ARITY_THEN_RECURSION.replace("1k", "2k")])

    def exporter(_cls, asc_path, timeout=0):
        net = Path(asc_path).with_suffix(".net")
        net.write_text(next(exports))
        return net

    monkeypatch.setattr(vc, "_create_netlist", exporter)
    state = _with_ltspice(config)
    asc = _write(work_dir, "m.asc", fake_netlister.amp_asc())
    await _run(state, path=str(asc), checks=["export"])
    second = await _run(state, path=str(asc), checks=["export"])
    not_compared = second["export"]["diff_vs_prior"]["not_compared"]
    assert any("recursive .SUBCKT definition" in entry for entry in not_compared), not_compared


@pytest.mark.asyncio
@pytest.mark.edit_reply_per_platform(
    "the export's .lib line names a Windows install path, which Windows reads as "
    "absolute and outside the sandbox, so the comparison notes it denied; WSL reads "
    "it through its /mnt mount and other Linux as a relative name"
)
async def test_edit_schematic_reference_stage_states_the_failure(
    asc_state: SessionState, work_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(se, "_export_asc_to_netlist", fake_netlister.export_asc_to_netlist)
    sheet = work_dir / "amp.asc"
    sheet.write_text(fake_netlister.amp_asc(), newline="\n")
    data = _assert_edit_schema(
        await handle_edit_schematic(
            _edit_input(
                path="amp.asc",
                expected_sha256=_sha(sheet),
                ops=_ADD_R2,
                compare={"reference": ARITY_THEN_RECURSION, "mode": "equivalence"},
            ),
            asc_state,
        )
    )
    assert data["commit_state"] == "committed"
    verification = data["verification"]
    assert "recursive .SUBCKT definition: loop -> loop" in verification["compare_error"]
    assert "calls itself" in verification["compare_error"]
    assert verification["equivalent"] is None
