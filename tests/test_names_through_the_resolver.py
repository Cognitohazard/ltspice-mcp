"""Names a run saves, measures and reports are read through the one resolver.

Each test sends one input through every path that takes the name and checks
it against a recording under ``tests/fixtures/ltspice_recorded`` or a live
ngspice run (skipped without ngspice on PATH):

- the ``.save`` and ``.meas`` coverage lint, which counts a ``.save`` name as
  stored only in the spelling the run's reader stores it in, and a ``.meas``
  probe as covered when it resolves to a stored name;
- a device inside a block, which ``api.device_op_points`` and the
  ``operating_point`` recipe give one name whichever build wrote the table;
- one fold for names, which folds ASCII letters only;
- the input-referred noise trace, named exactly, never by a substring.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from ltspice_mcp.lib import device_names, trace_names
from ltspice_mcp.lib.connectivity import fold_case
from ltspice_mcp.lib.decoded_raw import DecodedPlot, DecodedRaw
from ltspice_mcp.lib.device_names import trace_keys
from ltspice_mcp.lib.lint_rules import lint_deck
from ltspice_mcp.lib.metrics import noise_trace_unit
from ltspice_mcp.lib.raw_header import preflight_raw
from ltspice_mcp.lib.simulator_build import Producer
from ltspice_mcp.lib.trace_names import find_trace
from ltspice_mcp.state import SessionState
from ltspice_mcp.tools.analyze import AnalyzeResultsInput, handle_analyze_results
from tests import _ltspice_recorded as rec
from tests.conftest import LIVENESS_S, SyncApi, dispatched
from tests.ltspice_recorder import INPUTS
from tests.test_metric_plot_kinds import header


def stored_traces(raw: Path, dialect: str | None = None) -> list[str]:
    """The traces a run stored beside its axis."""
    (plot,) = preflight_raw(raw, limits=rec.RAW_LIMITS, producing_dialect=dialect).plots
    return [v.name for v in plot.variables if v.name.casefold() not in ("time", "frequency")]


def lint(text: str, path: Path, producer: Producer, **options: Any) -> dict[str, list[dict]]:
    """The linter's findings on ``text`` for ``producer``'s run, by rule."""
    path.write_text(text, encoding="utf-8")
    family = producer.family or "ltspice"
    simulator = "LTspice" if family == "ltspice" else "ngspice"
    findings = lint_deck(text, path, family, simulator, reader=producer, **options)
    by_rule: dict[str, list[dict]] = {}
    for finding in findings:
        by_rule.setdefault(finding["rule_id"], []).append(finding)
    return by_rule


# --- ngspice: a .save is stored in its dotted spelling ----------------------

DIVIDER = (
    ".subckt div top\nR1 top mid 1k\nR2 mid 0 1k\n.ends\n"
    "V1 in 0 PULSE(0 1 0 1u 1u 1 2)\nX1 in div\n.tran 10u 1m\n"
)


def ngspice_run(tmp_path: Path, name: str, deck: str) -> tuple[list[str], str]:
    """The traces ngspice-42 stores for ``deck`` and what it printed."""
    executable = shutil.which("ngspice")
    if executable is None:
        pytest.skip("ngspice is not on PATH")
    path = tmp_path / f"{name}.cir"
    path.write_text(deck, encoding="utf-8")
    raw = tmp_path / f"{name}.raw"
    done = subprocess.run(
        [executable, "-n", "-b", "-r", str(raw), str(path)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=LIVENESS_S,
    )
    return stored_traces(raw, "ngspice"), done.stdout + done.stderr


def ngspice_producer() -> Producer:
    return Producer("ngspice", "ngspice-42", shutil.which("ngspice"))


def test_ngspice_stores_a_dotted_save_the_server_reads_by_its_colon_name(tmp_path: Path):
    """``.save v(x1.mid)`` stores the node, and the server reads ``v(x1:mid)``
    from it: the lint accepts a ``.meas`` of that name. ngspice's own
    ``.meas`` does not parse ``x1:mid``, but a batch run with a raw
    evaluates no top-level ``.meas`` (``meas-ngspice-batch``)."""
    deck = f"* dotted save\n{DIVIDER}.save v(in) v(x1.mid)\n.meas tran vmid FIND v(x1:mid) AT=0.5m\n.end\n"
    traces, _ = ngspice_run(tmp_path, "dotted", deck)
    assert [t.casefold() for t in traces] == ["v(in)", "v(x1.mid)"]
    assert find_trace(traces, "v(x1:mid)", xvii=False) == "v(x1.mid)"
    found = lint(deck, tmp_path / "dotted_lint.cir", ngspice_producer(), ngbehavior="")
    assert "save-meas-coverage" not in found, found


def test_ngspice_ignores_a_colon_name_in_a_save(tmp_path: Path):
    """``.save v(in) v(x1:mid)``: ngspice-42 stores v(in) alone and says
    nothing of the other name, so a ``.meas`` of the node is refused and the
    ``.save`` name is reported as one ngspice ignores."""
    deck = f"* colon save\n{DIVIDER}.save v(in) v(x1:mid)\n.meas tran vmid FIND v(x1.mid) AT=0.5m\n.end\n"
    traces, _ = ngspice_run(tmp_path, "colon", deck)
    assert [t.casefold() for t in traces] == ["v(in)"]
    assert find_trace(traces, "v(x1.mid)", xvii=False) is None
    found = lint(deck, tmp_path / "colon_lint.cir", ngspice_producer(), ngbehavior="")
    (coverage,) = found["save-meas-coverage"]
    assert coverage["evidence"]["missing_signals"] == ["v(x1.mid)"]
    (ignored,) = found["save-name-not-stored"]
    assert ignored["subject"] == "v(x1:mid)"
    assert "ngspice" in ignored["evidence"]["reason"]


# --- LTspice: each build's .save and .meas take only its own spelling --------

SAVE_SPELLINGS = {True: "raw/save_subckt_names", False: "raw/save_subckt_names_no_x"}


@pytest.mark.parametrize("build", rec.BUILDS)
@pytest.mark.parametrize("with_x", [True, False])
def test_a_save_inside_subcircuits_counts_only_in_the_builds_spelling(
    build: str, with_x: bool, tmp_path: Path
):
    """A ``.meas`` of the node a deck saved in the other build's spelling is
    refused, as the recorded run stores no such node; in the build's own
    spelling it is accepted, and the run stores it."""
    case = SAVE_SPELLINGS[with_x]
    source = (INPUTS / f"{case}.cir").read_text(encoding="utf-8")
    node = "V(xa:xin:mid)" if with_x else "V(a:in:mid)"
    deck = source.replace(".tran 1u", f".tran 1u\n.meas tran vmid FIND {node} AT=0.5u")
    own = with_x == (rec.generation(build) != "xvii")
    stored = [t.casefold() for t in stored_traces(rec.recorded(build, f"{case}.raw"))]
    assert (node.casefold() in stored) == own, stored
    found = lint(deck, tmp_path / "nested.cir", rec.producer(build))
    if own:
        assert "save-meas-coverage" not in found, found
    else:
        (coverage,) = found["save-meas-coverage"]
        assert coverage["evidence"]["missing_signals"] == [node.casefold()]
        assert {f["subject"] for f in found["save-name-not-stored"]} >= {node}


#: The deck of ``meas-subckt-node`` each build measures, on the settings that
#: save subcircuit nodes; on the defaults neither build measures either.
OWN_MEAS_SPELLING = {"current": "deck/meas_subckt_node_x", "xvii": "deck/meas_subckt_node_no_x"}


@pytest.mark.parametrize("build", rec.BUILDS)
@pytest.mark.parametrize("stem", sorted(OWN_MEAS_SPELLING.values()))
def test_a_measurement_inside_a_subcircuit_is_checked_for_the_build(
    build: str, stem: str, tmp_path: Path
):
    """The other build's spelling fails on every setting, and the lint
    refuses it. The build's own spelling is measured only with subcircuit
    node voltages saved, which is off by default: the lint warns, and does
    not block."""
    from ltspice_mcp.lib.log_parser import parse_measurements

    own = stem == OWN_MEAS_SPELLING[rec.generation(build)]
    saved = parse_measurements(rec.recorded(build, f"{stem}_saved.log"))
    values = [entry["values"][0] for entry in saved["measurements"].values()]
    assert (values == [0.5]) == own
    text = (INPUTS / f"{stem}.cir").read_text(encoding="utf-8")
    found = lint(text, tmp_path / "meas.cir", rec.producer(build))
    if own:
        assert "save-meas-coverage" not in found, found
        assert "meas-subckt-node-unsaved-ltspice" in found, found
    else:
        assert "save-meas-coverage" in found, found


def test_an_instance_in_an_included_file_is_an_instance_of_the_deck(tmp_path: Path):
    """The instance a name's path starts at may be placed in an included
    file: the deck saves and measures the node in the build's own spelling,
    and the lint accepts it."""
    body = tmp_path / "body.inc"
    body.write_text("XAMP in div\n", encoding="utf-8")
    deck = (
        "* included\n.subckt div top\nR1 top mid 1k\nR2 mid 0 1k\n.ends\n"
        f'V1 in 0 1\n.include "{body.name}"\n.save V(in) V(xamp:mid)\n'
        ".tran 1m\n.meas tran vmid FIND V(xamp:mid) AT=0.5m\n.end\n"
    )
    for build in rec.BUILDS:
        own = rec.generation(build) == "current"
        text = deck if own else deck.replace("xamp:mid", "amp:mid")
        found = lint(
            text,
            tmp_path / "included.cir",
            rec.producer(build),
            includes=[(body, body.read_text(encoding="utf-8"), "utf-8")],
        )
        assert "save-meas-coverage" not in found, (build, found)
        assert "save-name-not-stored" not in found, (build, found)


def test_xvii_refuses_a_save_naming_another_spelling_and_saves_everything(tmp_path: Path):
    """XVII refuses the whole ``.save`` that names a path it does not store
    and saves every signal instead ("Invalid .save command. Just saving
    everything...", recorded): a ``.meas`` of a top-level node is measured,
    and one of a node inside the subcircuit is kept only on the setting."""
    (xvii,) = [build for build in rec.BUILDS if rec.generation(build) == "xvii"]
    log = rec.recorded(xvii, "raw/save_subckt_names.log").read_bytes()
    assert b"Invalid .save command.  Just saving everything..." in log.replace(b"\x00", b"")
    divider = (
        "* xvii\n.subckt div top\nR1 top mid 1k\nR2 mid 0 1k\n.ends\n"
        "V1 in 0 1\nR9 in out 1k\nR8 out 0 1k\nXamp in div\n"
    )
    top = divider + ".save V(xamp:mid) V(out)\n.tran 1m\n.meas tran v FIND V(in) AT=0.5m\n.end\n"
    found = lint(top, tmp_path / "top.cir", rec.producer(xvii))
    assert "save-meas-coverage" not in found, found
    (refused,) = found["save-name-not-stored"]
    assert "Just saving everything" in refused["evidence"]["reason"]
    inside = (
        divider + ".save V(xamp:mid) V(amp:mid)\n.tran 1m\n"
        ".meas tran v FIND V(amp:mid) AT=0.5m\n.end\n"
    )
    found = lint(inside, tmp_path / "inside.cir", rec.producer(xvii))
    assert "save-meas-coverage" not in found, found
    assert "meas-subckt-node-unsaved-ltspice" in found, found


@pytest.mark.parametrize("stem", sorted(OWN_MEAS_SPELLING.values()))
def test_a_spelling_inside_a_subcircuit_is_not_established_without_the_build(
    stem: str, tmp_path: Path
):
    """With no build known, which spelling the run takes is not established:
    the lint says so and does not block."""
    text = (INPUTS / f"{stem}.cir").read_text(encoding="utf-8")
    found = lint(text, tmp_path / "meas.cir", Producer("ltspice", None, None))
    assert "save-meas-coverage" not in found, found
    assert "save-meas-spelling-unestablished" in found, found


def test_a_dotted_name_in_an_ltspice_deck_is_not_established(tmp_path: Path):
    """Whether LTspice reads ``.`` as ``:`` in a deck is not recorded: the
    lint says the spelling is not established, and does not block."""
    deck = (
        "* dotted\n.subckt div top\nR1 top mid 1k\nR2 mid 0 1k\n.ends\n"
        "V1 in 0 1\nXamp in div\n.save V(in) V(xamp.mid)\n"
        ".tran 1m\n.meas tran vmid FIND V(xamp.mid) AT=0.5m\n.end\n"
    )
    for build in rec.BUILDS:
        found = lint(deck, tmp_path / "dotted.cir", rec.producer(build))
        assert "save-meas-coverage" not in found, found
        assert "save-meas-spelling-unestablished" in found, found


# --- One device, one name ---------------------------------------------------

BLOCK_DEVICE = "xu1.d1"
#: The table's own name for that device, which a caller may quote.
BLOCK_DEVICE_IN_THE_TABLE = {"current": "u1:d1", "xvii": "d:u1:1"}


def _block_case(build: str) -> str:
    if rec.generation(build) == "current":
        return "export/hier_device_logopinfo"
    return "export/hier_device_plain"


@pytest.mark.parametrize("build", rec.BUILDS)
def test_the_api_names_a_device_inside_a_block_alike_on_both_builds(
    build: str, state_no_sim: SessionState, work_dir: Path
):
    """D1 inside the block placed as U1 is ``xu1.d1`` to the Python API on
    both builds, with the table's own name beside it."""
    log = work_dir / "hier_device.log"
    log.write_bytes(rec.recorded(build, f"{_block_case(build)}.log").read_bytes())
    points = SyncApi(state_no_sim).device_op_points(log)
    assert points == {BLOCK_DEVICE: {"id": 2.26e-3, "vd": 0.676, "req": 11.5, "cap": 0.0}}
    assert points.table_names == {BLOCK_DEVICE: BLOCK_DEVICE_IN_THE_TABLE[rec.generation(build)]}


@pytest.mark.parametrize("build", rec.BUILDS)
async def test_the_recipe_names_a_device_inside_a_block_alike_on_both_builds(
    build: str, state_no_sim: SessionState, work_dir: Path
):
    """The operating-point recipe keys the device ``xu1.d1`` on both builds,
    and finds it by that name or by the table's own."""
    case = _block_case(build)
    table = BLOCK_DEVICE_IN_THE_TABLE[rec.generation(build)]
    raw = rec.stage_run(build, case, work_dir)
    request = AnalyzeResultsInput.model_validate(
        {
            "sources": [{"path": str(raw), "label": "dut"}],
            "recipes": [
                {"key": "all", "metric": "operating_point"},
                {"key": "d1", "metric": "operating_point", "device": BLOCK_DEVICE},
                {"key": "quoted", "metric": "operating_point", "device": table},
            ],
        }
    )
    data = (await handle_analyze_results(request, state_no_sim)).structured_content
    assert data is not None
    assert "failures" not in data, data["failures"]
    (bias,) = data["results"]["all"]["values"]
    assert bias["value"]["device_names"][BLOCK_DEVICE]["op_table"] == table
    for key in ("d1", "quoted"):
        (value,) = data["results"][key]["values"]
        assert value["value"]["device_op_points"][f"@{table}[id]"] == pytest.approx(2.26e-3)


# --- One fold for names ----------------------------------------------------


def test_one_fold_folds_ascii_letters_only():
    """``node-name-case`` and ``deck/case_insensitive`` establish that ASCII
    case does not tell nodes apart, and nothing beyond it."""
    assert device_names.fold_name("OUT") == "out"
    assert device_names.fold_name("Maß") == "maß"
    assert device_names.fold_name("MASS") != device_names.fold_name("maß")
    assert device_names.fold_name("É") == "É"
    assert fold_case("Maß") == device_names.fold_name("Maß")
    assert trace_keys("V(Maß)", xvii=False, schematic=False) != trace_keys(
        "V(mass)", xvii=False, schematic=False
    )


MASS_PORT = "* a port\n.subckt sub maß b\nR1 mass b 1k\n.ends\nX1 a b sub\nV1 a 0 1\nR2 b 0 1k\n.op\n.end\n"
MASS_SAME = MASS_PORT.replace("R1 mass b 1k", "R1 maß b 1k")


async def test_verify_circuit_reads_mass_and_eszett_alike_in_all_its_parts(
    state_no_sim: SessionState, work_dir: Path
):
    """``mass`` inside the body is not the port ``maß``: the quality check
    says the port is wired to nothing, and the comparison with the deck that
    wires it finds the two decks differ."""
    deck = work_dir / "port.cir"
    deck.write_text(MASS_PORT, encoding="utf-8")
    reference = work_dir / "same.cir"
    reference.write_text(MASS_SAME, encoding="utf-8")
    data = await dispatched(
        state_no_sim,
        "verify_circuit",
        path=str(deck),
        checks=["quality", "compare"],
        compare={"reference": str(reference)},
    )
    dangling = [f for f in data.get("findings", []) if f["rule_id"] == "dangling_node"]
    assert any("'maß'" in f["evidence"]["detail"] for f in dangling), dangling
    assert data["comparison"]["equivalent"] is False, data["comparison"]


def test_the_lint_reads_mass_and_eszett_as_the_trace_reader_does(tmp_path: Path):
    """``.save V(maß)`` stores no ``V(mass)`` that a reader of the run could
    find, so a ``.meas`` of it is refused."""
    deck = (
        "* eszett\nV1 maß 0 PULSE(0 1 0 1u 1u 1 2)\nR1 maß 0 1k\n.save V(maß)\n"
        ".tran 1m\n.meas tran x FIND V(mass) AT=0.5m\n.end\n"
    )
    assert find_trace(["V(maß)"], "V(mass)", xvii=False) is None
    for build in rec.BUILDS:
        found = lint(deck, tmp_path / "eszett.cir", rec.producer(build))
        (coverage,) = found["save-meas-coverage"]
        assert coverage["evidence"]["missing_signals"] == ["v(mass)"]


def test_ngspice_joins_non_ascii_names_and_the_lint_warns(tmp_path: Path):
    """ngspice-42 replaces each non-ASCII byte of a node name with ``_``, so
    ``aé`` and ``aè`` are one node ``a__``. The lint warns of each such name
    and models nothing of the replacement."""
    deck = "* non-ascii\nV1 aé 0 1\nR1 aé 0 1k\nI1 0 aè 1m\nR2 aè 0 1k\n.op\n.end\n"
    traces, _ = ngspice_run(tmp_path, "nonascii", deck)
    assert traces == ["v(a__)", "i(v1)"]
    found = lint(deck, tmp_path / "nonascii_lint.cir", ngspice_producer(), ngbehavior="")
    assert {f["subject"] for f in found["node-non-ascii-ngspice"]} == {"aé", "aè"}
    assert all(f["severity"] == "warning" for f in found["node-non-ascii-ngspice"])
    ltspice = lint(deck, tmp_path / "nonascii_lt.cir", Producer("ltspice", None, None))
    assert "node-non-ascii-ngspice" not in ltspice


# --- Input-referred noise --------------------------------------------------


def test_a_device_named_rinoise_is_not_input_referred_noise():
    """LTspice names a device's noise contribution after the device, so a
    resistor Rinoise contributes ``V(Rinoise)``, a voltage density; only the
    input-referred traces take the input source's unit."""
    raw = DecodedRaw(
        [
            DecodedPlot(
                header(
                    "Noise Spectral Density",
                    [
                        ("frequency", "frequency"),
                        ("V(inoise)", "voltage"),
                        ("V(Rinoise)", "voltage"),
                    ],
                    2,
                ),
                [np.array([1.0, 10.0]), np.array([2e-9, 2e-9]), np.array([1e-9, 1e-9])],
                snapshot_id="rinoise",
            )
        ]
    )
    assert trace_names.is_input_noise("V(inoise)") and trace_names.is_input_noise(
        "inoise_spectrum"
    )
    assert not trace_names.is_input_noise("V(Rinoise)")
    traces = {t.name: t for t in raw.descriptor.traces}
    assert traces["V(Rinoise)"].unit == "V/√Hz"
    assert traces["V(inoise)"].unit is None
    assert noise_trace_unit(raw, "V(Rinoise)", "A") == ("V", True)
    assert noise_trace_unit(raw, "V(inoise)", "A") == ("A", True)
