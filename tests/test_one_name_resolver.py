"""A name means the same thing to every reader of it.

Which net, trace or device a name stands for is answered in
``lib/deck_nodes.py``, ``lib/device_names.py`` and ``lib/trace_names.py``,
told who reads the deck or wrote the result. Each test here sends one input
through every path that takes a name and checks each against a recording or
a live ngspice run:

- a part LTspice exported as ``R§Load``, whose current both builds name
  ``I(Load)`` (``export/load_name_op``). That recording is an operating
  point, which plot_waveform does not chart; plot_waveform reads a name
  through the same lookup (``services.resolve_signal``), and a transient of
  such a part is listed for the next recording run;
- a subcircuit port named ``gnd`` in a deck for ngspice, which ngspice reads
  as ground (a live run);
- a name inside nested subcircuits, which each LTspice build's ``.save``
  takes only in its own spelling (``raw/save_subckt_names*``).
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest
from spicelib.simulators.ngspice_simulator import NGspiceSimulator

from ltspice_mcp.errors import ResultError
from ltspice_mcp.lib.hierarchy import load_hierarchy
from ltspice_mcp.lib.raw_header import preflight_raw
from ltspice_mcp.state import SessionState
from ltspice_mcp.tools import experiments
from ltspice_mcp.tools.experiments import RunExperimentsInput, handle_run_experiments
from tests import _ltspice_recorded as rec
from tests.conftest import LIVENESS_S, SyncApi, dispatched
from tests.ltspice_recorder import INPUTS

# --- A part named without its element letter ---------------------------------

#: ``R§Load`` in the netlist, ``I(Load)`` in the raw, on both builds; 2k
#: across 1 V.
LOAD_CURRENT = 0.5e-3


@pytest.mark.parametrize("build", rec.BUILDS)
def test_a_section_sign_name_reads_the_parts_own_trace(
    build: str, state_no_sim: SessionState, work_dir: Path
):
    """``I(R§Load)`` is the netlist's name for the current both builds write
    as ``I(Load)``: the Python API's trace and a read of that signal alone, the
    value recipe, and the operating point's device filter all read it."""
    raw = rec.stage_run(build, "export/load_name_op", work_dir)
    api = SyncApi(state_no_sim)
    loaded = api.load_raw(path=raw)
    assert loaded.trace("I(R§Load)") == pytest.approx([LOAD_CURRENT])
    assert "I(Load)" in api.load_raw(path=raw, signals=["I(R§Load)"]).signals

    analyzed = api.analyze_results(
        sources=[{"path": str(raw), "label": "run"}],
        recipes=[
            {"metric": "value", "key": "v", "expr": "I(R§Load)"},
            {"metric": "operating_point", "key": "op", "device": "R§Load"},
        ],
    )
    assert "failures" not in analyzed, analyzed.get("failures")
    (row,) = analyzed["results"]["v"]["per_run"]["items"]
    assert row["value"] == pytest.approx(LOAD_CURRENT)
    (op,) = analyzed["results"]["op"]["per_run"]["items"]
    assert {name.lower() for name in op["value"]["currents"]} == {"i(load)"}


@pytest.mark.parametrize("build", rec.BUILDS)
def test_a_name_that_misses_by_its_element_letter_is_refused_naming_the_trace(
    build: str, state_no_sim: SessionState, work_dir: Path
):
    """``I(RLoad)``, the name ``verify_circuit`` and ngspice's copy of the deck
    give the part, matches no trace of LTspice's raw, which names the part by
    its own name. Every reader refuses it and names ``I(Load)``."""
    raw = rec.stage_run(build, "export/load_name_op", work_dir)
    api = SyncApi(state_no_sim)
    texts = []
    with pytest.raises(ResultError) as refused:
        api.load_raw(path=raw).trace("I(RLoad)")
    texts.append(str(refused.value))
    with pytest.raises(ResultError) as refused:
        api.load_raw(path=raw, signals=["I(RLoad)"])
    texts.append(str(refused.value))
    analyzed = api.analyze_results(
        sources=[{"path": str(raw), "label": "run"}],
        recipes=[{"metric": "value", "key": "v", "expr": "I(RLoad)"}],
    )
    (failure,) = analyzed["failures"]
    texts.append(failure["message"])
    for text in texts:
        assert "I(Load) is in this result" in text, text


# --- A subcircuit port named gnd, in a deck for ngspice ----------------------

#: Inside ``ported``, ``gnd`` is the port to LTspice and ground to ngspice. The
#: ampere into ``c`` reaches ground only through the body's R1 to ``gnd``.
GND_PORT = (
    "* a port named gnd\n.subckt ported top gnd\nR1 top gnd 1k\n.ends\n"
    "I1 0 c 1m\nX2 c c ported\n.op\n.end\n"
)
#: The same circuit as ngspice reads it: the body's resistor to node 0.
GND_PORT_AS_GROUND = GND_PORT.replace("R1 top gnd 1k", "R1 top 0 1k")


def ngspice_voltage(tmp_path: Path, deck: str, node: str, mode: str) -> float:
    """The operating point ngspice gives ``node`` of ``deck`` in ``mode``."""
    executable = shutil.which("ngspice")
    if executable is None:
        pytest.skip("ngspice is not on PATH")
    control = f".control\nop\nprint v({node})\n.endc\n.end\n"
    path = tmp_path / f"deck_{mode or 'native'}.cir"
    path.write_text(deck.replace(".end\n", control), encoding="utf-8")
    command = [executable, "-n", "-b", *(["-D", f"ngbehavior={mode}"] if mode else []), str(path)]
    done = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=LIVENESS_S,
    )
    found = dict(re.findall(r"^v\((\S+)\) = (\S+)$", done.stdout, re.MULTILINE))
    return float(found[node])


def ngspice_session(config: Any, mode: str) -> SessionState:
    """A session whose simulator is ngspice, run in ``mode``."""
    ngspice = type("NGspiceSimulator", (NGspiceSimulator,), {"_compatibility_mode": mode})
    return SessionState.create(config, available={"ngspice": ngspice})


#: The modes in which ngspice's reading of ``gnd`` is established.
ESTABLISHED = ["", "hsa"]


@pytest.mark.parametrize("mode", ESTABLISHED)
async def test_a_port_named_gnd_is_ground_to_every_reader_of_an_ngspice_deck(
    mode: str, config: Any, work_dir: Path
):
    """ngspice runs the body's ``gnd`` as ground: the ampere crosses R1 to
    it, so ``c`` sits at 1 V. Told the deck is for ngspice, the connectivity
    comparison reads it as the deck that ties R1 to node 0, the floating-node
    check finds ``c`` reaches ground, and hierarchy discovery puts R1 on
    ground."""
    assert ngspice_voltage(work_dir, GND_PORT, "c", mode) == pytest.approx(1.0)
    state = ngspice_session(config, mode)
    deck = work_dir / "ported.cir"
    deck.write_text(GND_PORT, encoding="utf-8")
    reference = work_dir / "as_ground.cir"
    reference.write_text(GND_PORT_AS_GROUND, encoding="utf-8")

    data = await dispatched(
        state,
        "verify_circuit",
        path=str(deck),
        checks=["quality", "compare"],
        compare={"reference": str(reference)},
    )
    assert data["comparison"]["equivalent"] is True, data["comparison"]
    floating = [f for f in data.get("findings", []) if f["rule_id"] == "floating_net"]
    assert floating == [], floating

    hierarchy = await dispatched(
        state,
        "inspect",
        queries=[
            {"kind": "hierarchy", "path": str(deck), "simulator": "ngspice", "ngbehavior": mode}
        ],
    )
    (result,) = hierarchy["results"]
    rows = {tuple(row["instance"]): row for row in result["data"]["instances"]}
    assert [n["name"] for n in rows[("X2", "R1")]["nodes"]] == ["c", "0"]


# --- A name inside nested subcircuits, on each LTspice build ----------------

#: The recorded decks that .save a current and a node inside subcircuits, by
#: whether they spell each instance with its X.
SAVE_SPELLINGS = {True: "raw/save_subckt_names", False: "raw/save_subckt_names_no_x"}


def stored(build: str, case_id: str) -> set[str]:
    """The traces a recorded run stored beside its time axis, case-folded."""
    (plot,) = preflight_raw(rec.recorded(build, f"{case_id}.raw"), limits=rec.RAW_LIMITS).plots
    return {v.name.casefold() for v in plot.variables if v.name.casefold() != "time"}


def build_stores_its_own(build: str, saves: list[str]) -> None:
    """Each of ``saves`` names a trace of the run spelled for ``build``,
    which stores what its own spelling names and nothing the other's names."""
    own = SAVE_SPELLINGS[rec.generation(build) != "xvii"]
    traces = stored(build, own)
    for save in saves:
        assert save.casefold() in traces, (save, sorted(traces))


def hierarchy_saves(profile: Any) -> list[str]:
    """What the hierarchy read with ``profile`` asks a run to save: each
    resistor's ``.save`` current and the nested node's voltage."""
    deck = INPUTS / f"{SAVE_SPELLINGS[True]}.cir"
    hierarchy = load_hierarchy(str(deck), None, profile)
    rows = {"/".join(row.instance).casefold(): row for row in hierarchy.instances}
    saves = [rows[path].save for path in ("xa/xin/rin", "xa/rd")]
    assert all(save is not None and save.startswith(".save ") for save in saves), saves
    (mid,) = {
        node.voltage_trace
        for row in hierarchy.instances
        for node in row.nodes
        if node.voltage_trace is not None and node.voltage_trace.casefold().endswith("mid)")
    }
    return [*(str(save).removeprefix(".save ") for save in saves), mid]


@pytest.mark.parametrize("build", rec.BUILDS)
async def test_run_experiments_spells_nested_names_for_the_build_that_runs(
    build: str, config: Any, work_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    """The profile ``run_experiments`` reads a deck's hierarchy with, for a run
    on a build's own executable, spells a nested name as that build stores it.
    The run is not submitted: the profile is taken where the deck is handed to
    the variation expansion."""
    seen: list[Any] = []
    real = experiments.expand_variations

    def expand(decks, *args, **kwargs):
        seen.extend(deck.semantic_profile for deck in decks)
        return real(decks, *args, **kwargs)

    monkeypatch.setattr(experiments, "expand_variations", expand)
    submitted: list[str] = []

    def submit(self, netlist, run_filename, callback, **_kwargs):
        submitted.append(run_filename)
        return object()

    from ltspice_mcp.lib.experiment_runner import ExperimentRunner

    monkeypatch.setattr(ExperimentRunner, "submit_netlist", submit)
    deck = work_dir / "nested.cir"
    deck.write_bytes((INPUTS / f"{SAVE_SPELLINGS[True]}.cir").read_bytes())
    state = SessionState.create(config, available={"ltspice": rec.launcher(build)})
    args = RunExperimentsInput.model_validate(
        {
            "request_id": f"nested-{build}",
            "circuits": [{"path": str(deck)}],
            "execution": {"wait_s": 0},
            "lint": "off",
        }
    )
    await handle_run_experiments(args, state)
    (profile,) = seen
    assert profile is not None
    build_stores_its_own(build, hierarchy_saves(profile))
