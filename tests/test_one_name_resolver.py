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
from ltspice_mcp.lib.device_names import (
    device_address,
    naming_disagreement,
    naming_for,
    section_sign_spellings,
    spelled_name,
)
from ltspice_mcp.lib.hierarchy import load_hierarchy
from ltspice_mcp.lib.raw_header import preflight_raw
from ltspice_mcp.lib.raw_parser import raw_writer_command
from ltspice_mcp.lib.simulator_build import Producer
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
    (row,) = analyzed["results"]["v"]["values"]
    assert row["value"]["signal"] == "I(Load)"
    assert row["value"]["value"] == pytest.approx(LOAD_CURRENT)
    (op,) = analyzed["results"]["op"]["values"]
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


# --- One derivation of a build's spelling ------------------------------------


@pytest.mark.parametrize(
    ("producer", "xvii", "read_as", "noted"),
    [
        (
            Producer("ltspice", "LTspice 26.0.2 for Windows", "LTspice.exe"),
            False,
            "LTspice 26",
            False,
        ),
        (Producer("ltspice", None, "C:/LTC/XVIIx64.exe"), True, "LTspice XVII", False),
        (
            Producer("ltspice", "Linear Technology Corporation LTspice XVII", None),
            True,
            "LTspice XVII",
            False,
        ),
        (Producer("ltspice", None, "C:/LTC/scad3.exe"), True, "LTspice XVII", True),
        (
            Producer("ltspice", "LTspice 24.0.12 for Windows", "LTspice.exe"),
            False,
            "LTspice 26",
            True,
        ),
        (
            Producer("ltspice", "LTspice 27.0.1 for Windows", "LTspice.exe"),
            False,
            "LTspice 26",
            True,
        ),
        (Producer("ltspice", None, "LTspice.exe"), False, "LTspice 26", True),
        (Producer("ngspice", "ngspice-42", "/usr/bin/ngspice"), False, "ngspice", False),
    ],
)
def test_a_builds_spelling_comes_from_its_producer(
    producer: Producer, xvii: bool, read_as: str, noted: bool
):
    """XVII and IV write each instance without its X, every build from 24 on
    with it; a build with no recording is read as its nearest recorded one,
    and the answer says so."""
    naming = naming_for(producer)
    assert (naming.xvii, naming.read_as, naming.note is not None) == (xvii, read_as, noted)


@pytest.mark.parametrize("build", rec.BUILDS)
def test_each_builds_executable_is_read_with_the_spelling_its_raw_is_written_in(build: str):
    """Each recorded build's executable gives the spelling its raw is written
    in, so a run on it reports no disagreement; read as the other build, it
    does."""
    command = raw_writer_command(rec.recorded(build, f"{SAVE_SPELLINGS[True]}.raw"))
    program = f"C:/install/{rec.manifest(build)['executable']['name']}"
    reported = None if rec.generation(build) == "xvii" else "LTspice 26.1.1 for Windows"
    own = naming_for(Producer("ltspice", reported, program)).xvii
    assert naming_disagreement(own, command) is None
    said = naming_disagreement(not own, command)
    assert said is not None and "names each subcircuit instance" in said


@pytest.mark.parametrize(
    ("producer", "trace", "device"),
    [
        (Producer("ltspice", "LTspice 26.1.1", None), "xa:xin:mid", "xa:xin:mtail"),
        (Producer("ltspice", None, "XVIIx64.exe"), "a:in:mid", "xa:xin:mtail"),
        (Producer("ngspice", None, None), "xa.xin.mid", "m.xa.xin.mtail"),
    ],
)
def test_one_speller_spells_a_path_for_each_reader(producer: Producer, trace: str, device: str):
    naming = naming_for(producer)
    assert spelled_name(("Xa", "Xin"), "mid", naming) == trace
    assert device_address(("Xa", "Xin", "Mtail"), naming) == device


def test_a_section_sign_name_has_the_parts_own_name_and_the_bare_one():
    assert section_sign_spellings("I(R§Load)") == ("I(Load)", "I(RLoad)")
    assert section_sign_spellings("I(x1:R§Load)") == ("I(x1:Load)", "I(x1:RLoad)")
    assert section_sign_spellings("I(R1)") == ()


async def test_inspect_says_how_a_build_with_no_recording_is_spelled(config: Any, work_dir: Path):
    """An LTspice whose build no run has named yet is spelled as LTspice 26,
    and the hierarchy's profile says so."""
    deck = work_dir / "nested.cir"
    deck.write_bytes((INPUTS / f"{SAVE_SPELLINGS[True]}.cir").read_bytes())
    state = SessionState.create(config, available={"ltspice": rec.launcher("ltspice26")})
    data = await dispatched(
        state,
        "inspect",
        queries=[{"kind": "hierarchy", "path": str(deck), "simulator": "ltspice"}],
    )
    (result,) = data["results"]
    assert result["ok"], result
    profile = result["data"]["profile"]
    assert profile["names_read_as"] == "LTspice 26"
    assert "not known until a run on it names itself" in profile["names_note"]


async def test_a_run_whose_raw_is_spelled_otherwise_than_its_deck_was_read_says_so(
    config: Any, work_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    """An XVII under another name that has not run before is read as LTspice
    26 when its job is submitted (it names no build yet). The job records that
    spelling, the record keeps it, and an XVII raw from that job's run, which
    writes each instance without its X, is reported against it; a job on an
    executable named as XVII is not. The run is not submitted: the check is
    the runner's, given the job and a case holding XVII's recorded raw."""
    from types import SimpleNamespace

    from spicelib.simulators.ltspice_simulator import LTspice

    from ltspice_mcp.lib.experiment_runner import ExperimentRunner
    from ltspice_mcp.lib.experiment_store import deserialize_job, serialize_job
    from ltspice_mcp.lib.store import OwnerLiveness

    monkeypatch.setattr(ExperimentRunner, "submit_netlist", lambda *a, **k: object())
    deck = work_dir / "nested.cir"
    deck.write_bytes((INPUTS / f"{SAVE_SPELLINGS[True]}.cir").read_bytes())
    raw = work_dir / "run.raw"
    raw.write_bytes(rec.recorded("ltspice17", f"{SAVE_SPELLINGS[True]}.raw").read_bytes())
    for program, disagrees in (("C:/tools/lt.exe", True), ("C:/LTC/XVIIx64.exe", False)):
        launcher = type(LTspice.__name__, (LTspice,), {"spice_exe": [program]})
        state = SessionState.create(config, available={"ltspice": launcher})
        args = RunExperimentsInput.model_validate(
            {
                "request_id": f"spelled-{disagrees}",
                "circuits": [{"path": str(deck)}],
                "execution": {"wait_s": 0},
                "lint": "off",
            }
        )
        await handle_run_experiments(args, state)
        (job,) = state.all_jobs.values()
        assert job.deck_names_xvii is (not disagrees)
        reread = deserialize_job(serialize_job(job), job.store_path, liveness=OwnerLiveness.ALIVE)
        assert reread.deck_names_xvii is job.deck_names_xvii
        case = job.cases[0]
        case.raw_file = raw
        before = len(case.observations)
        await ExperimentRunner._check_raw_names(SimpleNamespace(job=job), case)  # type: ignore[arg-type]
        codes = [o["code"] for o in case.observations[before:]]
        assert codes == (["raw_names_disagree"] if disagrees else []), program


def test_a_section_sign_device_names_neither_part_where_both_names_are_there(
    state_no_sim: SessionState,
):
    """``device=R§Load`` reads LTspice's ``I(Load)``; where a result also holds
    ``I(RLoad)``, another part, it names neither, as a trace name does."""
    from ltspice_mcp.lib.metrics import filter_operating_point

    alone = {"currents": {"I(Load)": 1.0, "I(R1)": 2.0}}
    assert filter_operating_point(alone, "R§Load", xvii=False)
    assert alone["currents"] == {"I(Load)": 1.0}
    both = {"currents": {"I(Load)": 1.0, "I(RLoad)": 3.0}}
    assert not filter_operating_point(both, "R§Load", xvii=False)


def test_a_hierarchy_cursor_does_not_change_when_a_run_names_the_build():
    """The note on a build no run has named goes once a run names it; the
    profile a cursor and a native record are bound to stays the same."""
    from ltspice_mcp.lib.hierarchy import SemanticProfile

    before = SemanticProfile.of(Producer("ltspice", None, "C:/LT/LTspice.exe"), None)
    after = SemanticProfile.of(Producer("ltspice", "LTspice 26.1.1", "C:/LT/LTspice.exe"), None)
    assert before.reply() != after.reply()
    assert before.record() == after.record()
    assert set(before.record()) == {"simulator", "ngbehavior", "evaluation_context", "xvii_names"}


def test_the_lint_reads_an_ngspice_deck_in_the_runs_mode_when_told_none():
    from ltspice_mcp.lib.lint_rules import lint_deck

    deck = "* t\nV1 a 0 1\nR1 a b 1k\nR2 b 0 1k\n.save V(a)\n.meas op x find V(a,b)\n.op\n.end\n"
    assert isinstance(lint_deck(deck, Path("t.cir"), "ngspice", None), list)


def test_a_port_ngspice_reads_as_ground_binds_nothing_to_the_floating_check():
    """``foo``'s body ties its port ``gnd`` to node 0 through R1. To LTspice
    that grounds ``b``, the node wired to the port. To ngspice the body's
    ``gnd`` is ground itself and the port binds nothing, so ``b``, otherwise
    wired only to capacitors, floats, as hierarchy discovery leaves the port
    unbound."""
    from ltspice_mcp.lib.hierarchy import SemanticProfile
    from ltspice_mcp.lib.spice_lex import lex_deck
    from ltspice_mcp.lib.spice_validator import validate_netlist_bias_topology
    from tests.conftest import LTSPICE_READER

    deck = (
        "* t\n.subckt foo gnd\nR1 gnd 0 1k\n.ends\nV1 a 0 1\nC1 a b 1p\nX1 b foo\n"
        "C2 b 0 1p\n.op\n.end\n"
    )
    cards = lex_deck(deck).cards
    assert validate_netlist_bias_topology(cards, LTSPICE_READER) == []
    floating = validate_netlist_bias_topology(cards, SemanticProfile("ngspice", "hsa"))
    assert floating and "'b'" in str(floating[0]["message"]), floating


async def test_a_later_job_on_the_same_executable_does_not_change_an_earlier_jobs_check(
    config: Any, work_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    """Two jobs on a renamed XVII: the first is read as LTspice 26 (no run
    has named the build), and its run reports XVII. The second is then read
    as XVII. The first job's raw is still held to the spelling its own decks
    were read with, and is reported; the second's agrees."""
    from types import SimpleNamespace

    from spicelib.simulators.ltspice_simulator import LTspice

    from ltspice_mcp.lib.experiment_runner import ExperimentRunner

    monkeypatch.setattr(ExperimentRunner, "submit_netlist", lambda *a, **k: object())
    deck = work_dir / "nested.cir"
    deck.write_bytes((INPUTS / f"{SAVE_SPELLINGS[True]}.cir").read_bytes())
    raw = work_dir / "run.raw"
    raw.write_bytes(rec.recorded("ltspice17", f"{SAVE_SPELLINGS[True]}.raw").read_bytes())
    launcher = type(LTspice.__name__, (LTspice,), {"spice_exe": ["C:/tools/lt.exe"]})
    state = SessionState.create(config, available={"ltspice": launcher})

    async def submit(request_id: str) -> Any:
        args = RunExperimentsInput.model_validate(
            {
                "request_id": request_id,
                "circuits": [{"path": str(deck)}],
                "execution": {"wait_s": 0},
                "lint": "off",
            }
        )
        known = set(state.all_jobs)
        await handle_run_experiments(args, state)
        (job_id,) = set(state.all_jobs) - known
        return state.all_jobs[job_id]

    async def reported(job: Any) -> list[str]:
        case = job.cases[0]
        case.raw_file = raw
        before = len(case.observations)
        await ExperimentRunner._check_raw_names(SimpleNamespace(job=job), case)  # type: ignore[arg-type]
        return [o["code"] for o in case.observations[before:]]

    first = await submit("first")
    assert first.deck_names_xvii is False
    first.cases[0].simulator_version = "Linear Technology Corporation LTspice XVII"
    second = await submit("second")
    assert second.deck_names_xvii is True
    assert await reported(first) == ["raw_names_disagree"]
    assert await reported(second) == []
