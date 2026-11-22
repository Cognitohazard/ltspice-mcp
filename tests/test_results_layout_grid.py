"""How a results file is laid out, and what its plot is, on every route that answers it.

The question is the results-layout grid's (``[grid.results-layout]`` in
``inputs/cases.toml``; ``docs/TESTING.md``, "Grids"). The raw header reader
names the writer (``raw_header._resolve_dialect``) and the bytes and order of
each value (``raw_header._layout``); the run row reads a binary record its own
way (``raw_parser._binary_layout``, behind ``read_partial_raw_progress``); a
plot's analysis is read off its name (``result_kinds.plot_analysis``), whether
it opens with a sampled axis or with the stepped values off its analysis and
first variable (``result_kinds.sampled_axis``, ``raw_header.leading_parameters``),
and its kind of stepped result off all of them (``result_kinds.stepped_kind``).

A cell is one results file of one case: the run's own raw, the ``.op.raw`` a
transient or an AC run leaves beside it, or the ``.fra_1.raw`` of a frequency
response analysis. What each cell is expected to give is derived here from
its form (its coordinates and its deck's ``.step``), never from the reader
under test, and from the recordings where the form alone does not say (the
cell's ``kind`` and ``steps``). Every recorded cell, on each build, is read
through the header reader, the run row's reader, the decoder, ``inspect``'s
plots, the Python API's ``load_raw``, ``analyze_results`` and, for a plot
with an axis, ``plot_waveform``. A route a cell gives nothing to read is said,
per cell.

The same layout questions have an ngspice answer, checked live where ngspice
is on PATH; nothing of ngspice is recorded.
"""

from __future__ import annotations

import asyncio
import re
import shutil
from pathlib import Path
from typing import Any

import pytest

from ltspice_mcp.lib.raw_header import preflight_raw
from ltspice_mcp.lib.raw_parser import read_partial_raw_progress
from ltspice_mcp.lib.result_kinds import STEPPED_KINDS, plot_analysis, stepped_kind
from ltspice_mcp.lib.spice_lex import directive_head
from ltspice_mcp.state import SessionState
from tests import _ltspice_recorded as rec
from tests._asc_ops import inspect_one
from tests._ngspice_decks import run_ngspice
from tests._recorded_grids import Cell, grid
from tests.conftest import SyncApi, dispatched

GRID = grid("results-layout")

#: How many routes answer otherwise than a cell, per build, each declared on
#: its cell (``differs``, with the route and the reason). A count may only
#: fall: a fix lowers it here in the same change.
KNOWN_DIFFERENCES: dict[str, int] = {"ltspice26": 1, "ltspice17": 1}

#: The analysis each coordinate names, as ``plot_analysis`` spells it.
ANALYSES = {
    "tran": "transient",
    "ac": "ac",
    "dc": "dc",
    "noise": "noise",
    "op": "op",
    "tf": "tf",
    "fra": "fra",
}
_DIRECTIVES = {f".{key}": key for key in ANALYSES}


def _stem(cell: Cell) -> str:
    """The recorded file of a cell, without ``.raw``: ``raw/fra.fra_1``."""
    return cell.case if cell.item == "raw" else f"{cell.case}.{cell.item.removesuffix('.raw')}"


def _per_build(value: Any, build: str) -> Any:
    return value.get(build) if isinstance(value, dict) else value


def _read_on(build: str, cell: Cell) -> bool:
    """Whether ``build`` wrote the cell's file: its case is recorded there and
    neither the deck nor the item was refused."""
    return build not in cell.refused and rec.has(build, f"{_stem(cell)}.raw")


# The index, held to the decks ------------------------------------------------


def _directives(source: str) -> list[str]:
    return [directive_head(card) for card in rec.deck_cards(source) if card.kind == "directive"]


def _card(source: str, head: str) -> list[str]:
    """The words of a deck's directive ``head``, or none."""
    for card in rec.deck_cards(source):
        if card.kind == "directive" and directive_head(card) == head:
            return card.body.split()
    return []


def _stepped_by(source: str) -> tuple[str, int]:
    """What a deck's ``.step`` steps and over how many values: read off its
    words, ``.step [dec|oct|lin] <what> list v1 v2 ...``, where ``<what>`` is
    ``param <name>``, ``temp``, a model's ``<type> <model>(<parameter>)`` or a
    source's name."""
    words = _card(source, ".step")
    if not words:
        return "no", 1
    rest = words[1:]
    if rest[0].lower() in ("dec", "oct", "lin"):
        rest = rest[1:]
    if rest[0].lower() == "param":
        stepped = "param"
    elif rest[0].lower() == "temp":
        stepped = "temp"
    elif len(rest) > 1 and "(" in rest[1]:
        stepped = "model"
    else:
        stepped = "source"
    lowered = [word.lower() for word in rest]
    count = len(rest) - lowered.index("list") - 1 if "list" in lowered else 0
    return stepped, count


def _one_point(source: str, analysis: str) -> bool:
    """Whether a deck's frequency or DC sweep is of one point."""
    words = [word.lower() for word in _card(source, f".{analysis}")]
    if analysis == "dc":
        return words[2] == words[3]
    sweep = words[1:] if analysis == "ac" else words[3:]
    if sweep[0] == "list":
        return len(sweep) == 2
    return sweep[1] == "1" and sweep[2] == sweep[3]


def _numdgt(source: str) -> int:
    for word in _card(source, ".options"):
        name, _, value = word.partition("=")
        if name.lower() == "numdgt":
            return int(value)
    return 6


def test_a_cell_is_the_form_its_case_runs():
    """Each cell's coordinates are what its case runs, read off the deck and
    the case's own settings: the analysis of the file, how the run is
    stepped and over how many values, a sweep of one point, the precision,
    the storage, the order and the compression."""
    for cell in GRID.cells:
        case = rec.CASES.case(cell.case)
        at = cell.coordinates
        heads = _directives(case.source)
        (analysis,) = {
            _DIRECTIVES[head] for head in heads if head in _DIRECTIVES and head != ".op"
        } or {"op"}
        if cell.item == "fra_1.raw":
            assert analysis == "fra" and at["analysis"] == "fra", cell.id
        elif cell.item == "op.raw":
            assert analysis in ("tran", "ac") and at["analysis"] == "op", cell.id
            assert at["file"] == "op_beside", cell.id
        else:
            want = "tran" if analysis == "fra" else analysis
            assert at["analysis"] == want, (cell.id, analysis)
            assert at["file"] == "own", cell.id
        stepped, count = _stepped_by(case.source)
        assert at["stepped"] == stepped, (cell.id, stepped)
        steps = cell.extra.get("steps", 1)
        if not isinstance(steps, dict):
            assert steps == count, (cell.id, count)
        if at["analysis"] in ("ac", "noise", "dc") and cell.item == "raw":
            one = _one_point(case.source, at["analysis"])
            assert at["points"] == ("one" if one else "swept"), cell.id
        else:
            assert at["points"] == "", cell.id
        assert (at["numdgt"] == "above_6") == (_numdgt(case.source) > 6), cell.id
        text = "-ascii" in case.switches or case.ini.get("AsciiRawFile") == "true"
        assert (at["storage"] == "text") == text, cell.id
        assert (at["order"] == "trace") == (case.kind == "fastaccess"), cell.id
        if at["analysis"] == "tran" and cell.item == "raw":
            off = any("plotwinsize=0" in word.lower() for word in _card(case.source, ".options"))
            assert at["compression"] == ("off" if off else "on"), cell.id
        else:
            assert at["compression"] == "", cell.id


def test_every_case_of_the_grid_is_waiting_or_has_no_prediction_left():
    """A recorded cell declares what the recording shows: a prediction
    (``not_established``) is settled against its recording, and the key
    removed, once the case is recorded."""
    for cell in GRID.cells:
        if "not_established" in cell.extra:
            recorded = [build for build in rec.BUILDS if cell.case in rec.manifest(build)["cases"]]
            assert not recorded, (
                f"{cell.id} is recorded on {recorded}: settle it against the recording "
                "and remove not_established"
            )


def test_the_count_of_known_differences_only_falls():
    for build in rec.BUILDS:
        count = sum(
            1
            for cell in GRID.cells
            for entry in cell.extra.get("differs", ())
            if entry["build"] == build
        )
        assert count <= KNOWN_DIFFERENCES[build], (build, count)
        assert count == KNOWN_DIFFERENCES[build], f"lower the pin for {build} to {count}"


# What a cell is expected to give, by its form ---------------------------------


def _expected_kind(cell: Cell, build: str) -> str | None:
    at = cell.at(build)
    if "kind" in cell.extra:
        return _per_build(cell.extra["kind"], build)
    if at["stepped"] == "no":
        return None
    analysis = at["analysis"]
    if analysis == "op":
        return "stepped_op"
    if analysis == "tf":
        return "stepped_tf"
    if at["points"] == "one" and analysis == "ac":
        return "stepped_ac_one_frequency"
    if at["points"] == "one" and analysis == "noise":
        return "stepped_noise_one_frequency"
    return "stepped_sweep"


def _expected_axis(cell: Cell, build: str) -> bool:
    """Whether the plot opens with a sampled axis: every sweep does, and a
    frequency sweep of one point does unless it is stepped, when it opens
    with the stepped values instead (raw/step_ac_one_freq)."""
    at = cell.at(build)
    if at["file"] == "op_beside" or at["analysis"] in ("op", "tf"):
        return False
    if at["analysis"] in ("ac", "noise") and at["points"] == "one":
        return at["stepped"] == "no"
    return True


def _expected_widths(cell: Cell, build: str, variables: int) -> tuple[int, ...]:
    """The bytes of each value of a binary record: complex values are 16
    bytes, axis included; under numdgt above 6 a real plot stores doubles
    (raw/tran_double); else the axis is a double and the rest 4-byte floats."""
    at = cell.at(build)
    if at["analysis"] in ("ac", "fra"):
        return (16,) * variables
    if at["numdgt"] == "above_6":
        return (8,) * variables
    return (8,) + (4,) * (variables - 1)


def _expected(cell: Cell, build: str, variables: int, points: int) -> dict[str, Any]:
    at = cell.at(build)
    kind = _expected_kind(cell, build)
    axis = _expected_axis(cell, build)
    steps = _per_build(cell.extra.get("steps", 1), build)
    want: dict[str, Any] = {
        "header dialect": "ltspice",
        "header analysis": ANALYSES[at["analysis"]],
        "header storage": at["storage"],
        "header order": at["order"],
        "header stepped": at["stepped"] != "no",
        # The .fra_1.raw of a run under numdgt above 6 holds no double flag
        # (raw/fra_numdgt7.fra_1); every other plot under it does.
        "header double": at["numdgt"] == "above_6" and at["analysis"] != "fra",
        "header stepped kind": kind,
        "run row plot": True,
        "run row points": points,
        "run row axis": axis,
        "run row stepped": at["stepped"] != "no",
        "decoded analysis": ANALYSES[at["analysis"]],
        "decoded stepped kind": kind,
        "decoded axis": axis,
        "decoded steps": steps,
        "inspect analysis": ANALYSES[at["analysis"]],
        "inspect stepped kind": kind,
        "api analysis": ANALYSES[at["analysis"]],
        "api stepped kind": kind,
        "analyze_results rows": steps if steps else "refused",
        "plot_waveform charted": True,
    }
    if at["storage"] == "binary":
        want["header widths"] = _expected_widths(cell, build, variables)
        want["header payload ends the file"] = True
    return want


# Every route --------------------------------------------------------------------


def _signal(names: list[str], axis: str | None) -> str:
    """A signal of the plot to read: its first node voltage, else its first
    trace that is not its axis."""
    others = [name for name in names if name != axis]
    return next((name for name in others if name.lower().startswith("v(")), others[0])


def _rows_or_refusal(reply: dict[str, Any]) -> int | str:
    if reply.get("failures"):
        return "refused"
    return len(reply["results"]["k"]["per_run"]["items"])


async def _every_route(
    build: str, cell: Cell, state: SessionState, directory: Path, not_run: dict[str, str]
) -> dict[str, Any]:
    stem = _stem(cell)
    staged = rec.stage_run(build, stem, directory / "run")
    # Beside the run's own log, as the run left them: an .op.raw or a
    # .fra_1.raw has no log of its own name.
    log = f"{cell.case}.log"
    if cell.item != "raw" and rec.has(build, log):
        shutil.copyfile(rec.recorded(build, log), directory / "run" / Path(log).name)
    routes: dict[str, Any] = {}
    header = preflight_raw(staged, limits=rec.RAW_LIMITS)
    (plot,) = header.plots
    routes["header dialect"] = plot.dialect
    routes["header analysis"] = plot_analysis(plot.plot_name, plot.first_type)
    # The header names a text plot by its marker, "Values:".
    routes["header storage"] = {"values": "text"}.get(plot.storage, plot.storage)
    routes["header order"] = plot.storage_order
    routes["header stepped"] = plot.stepped
    routes["header double"] = "double" in (flag.lower() for flag in plot.flags)
    kind = stepped_kind(plot)
    routes["header stepped kind"] = kind.name if kind is not None else None
    if plot.storage != "values":
        routes["header widths"] = plot.value_bytes
        routes["header payload ends the file"] = plot.payload_end == staged.stat().st_size
    else:
        not_run["header widths"] = "a text plot has no binary record"
    progress = read_partial_raw_progress(staged)
    assert progress is not None, cell.id
    routes["run row plot"] = progress.plot == plot.plot_name
    routes["run row points"] = progress.points
    routes["run row axis"] = progress.axis is not None
    routes["run row stepped"] = progress.stepped
    decoded = rec.decode(build, stem, directory / "decoded").raw
    descriptor = decoded.descriptor
    routes["decoded analysis"] = descriptor.analysis
    routes["decoded stepped kind"] = descriptor.stepped_kind
    routes["decoded axis"] = descriptor.axis is not None
    routes["decoded steps"] = len(descriptor.steps)
    listed = await inspect_one(state, {"kind": "results", "view": "plots", "path": str(staged)})
    (row,) = listed["plots"]
    routes["inspect analysis"] = row["analysis"]
    routes["inspect stepped kind"] = row.get("stepped_kind")
    api = SyncApi(state)
    loaded = await asyncio.to_thread(lambda: api.load_raw(staged))
    routes["api analysis"] = loaded.analysis_type
    routes["api stepped kind"] = loaded.descriptor["stepped_kind"]
    axis = descriptor.axis.name if descriptor.axis is not None else None
    signal = _signal([trace.name for trace in descriptor.signals()], axis)
    # Statistics need a sweep; a plot of one point a step is read by value.
    swept = axis is not None and plot.point_count > max(len(descriptor.steps), 1)
    recipe = (
        {"metric": "signal_stats", "signal": signal}
        if swept
        else {"metric": "value", "expr": signal}
    )
    if descriptor.analysis == "tf":
        # A transfer function is a table, which analyze_results refers to
        # inspect's table view; that view is its route.
        table = await inspect_one(state, {"kind": "results", "view": "table", "path": str(staged)})
        # One row a quantity a step.
        routes["analyze_results rows"] = len(table["table"]) // len(descriptor.signals())
        not_run["analyze_results"] = "a .tf plot is a table, read by inspect's table view"
    else:
        reply = await dispatched(
            state,
            "analyze_results",
            sources=[{"path": str(staged), "label": "dut"}],
            recipes=[{"key": "k", **recipe}],
            all_steps=True,
            include={"per_run": True},
        )
        routes["analyze_results rows"] = _rows_or_refusal(reply)
    if axis is not None:
        plotted = await dispatched(
            state, "plot_waveform", path=str(staged), signals=[signal], open=False
        )
        routes["plot_waveform charted"] = bool(plotted.get("traces"))
    else:
        not_run["plot_waveform charted"] = "a plot with no sampled axis has nothing to chart"
    return routes


def _runs() -> list[Any]:
    runs = []
    for case_id in sorted({cell.case for cell in GRID.cells}):
        pending = rec.CASES.behaviours[rec.CASES.case(case_id).behaviour].pending
        for build in rec.BUILDS:
            param = pytest.param(build, case_id, id=f"{build}/{case_id}")
            if case_id in rec.manifest(build)["cases"]:
                runs.append(param)
            elif pending:
                runs.append(
                    pytest.param(
                        build, case_id, id=f"{build}/{case_id}", marks=pytest.mark.skip(pending)
                    )
                )
    return runs


@pytest.mark.asyncio
@pytest.mark.parametrize(("build", "case_id"), _runs())
async def test_every_recorded_cell_reads_as_its_form_on_every_route(
    build: str, case_id: str, state_no_sim: SessionState, work_dir: Path
):
    """Each recorded file is read on every route as its form says: its
    writer, analysis, storage, order, value widths, precision, whether it is
    stepped, of which kind and over how many steps, and whether it has an
    axis. A refused file is not read; a route that answers otherwise is
    declared on its cell (``differs``)."""
    for index, cell in enumerate(GRID.cells_of(case_id)):
        if build in cell.refused:
            assert not rec.has(build, f"{_stem(cell)}.raw"), (cell.id, "refused, yet written")
            continue
        assert _read_on(build, cell), f"{cell.id}: {build} recorded the case but not its file"
        not_run: dict[str, str] = {}
        routes = await _every_route(build, cell, state_no_sim, work_dir / f"cell{index}", not_run)
        print(f"{build}/{cell.id}: routes not run: {not_run}")
        fields = rec.header_fields(build, _stem(cell))
        want = _expected(
            cell, build, int(fields["No. Variables"][0]), int(fields["No. Points"][0])
        )
        declared = {
            entry["route"]: entry["reason"]
            for entry in cell.extra.get("differs", ())
            if entry["build"] == build
        }
        differ = {route: (got, want[route]) for route, got in routes.items() if got != want[route]}
        assert set(differ) == set(declared), (cell.id, build, differ, sorted(declared))


# Every kind the code declares, against the recordings ---------------------------

#: The plots LTspice's binary layout is validated for (``raw_header``), by
#: the analysis each holds.
_LTSPICE_ANALYSES = ("tran", "ac", "dc", "noise", "op", "tf", "fra")

#: The kinds, by build, that the code claims for an LTspice build with no
#: recorded cell of the grid on that build to show them. Each is reported. A
#: set may only shrink: a recording that shows a kind takes it out.
#:
#: stepped_run_op: the .op.raw a stepped transient leaves is that kind on
#: LTspice 26 and a stepped_op on XVII (raw/step_tran_op_raw), and XVII
#: leaves none beside a stepped AC sweep (raw/step_ac_op_raw). Nothing in
#: ``stepped_kind`` reads the build, so it claims the kind for XVII too.
#:
#: fra: XVII 17.0.37 refuses .fra (raw/fra), so neither the analysis nor a
#: stepped sweep of it is shown on XVII; ``_layout`` and ``stepped_kind``
#: take an LTspice frequency response plot whatever its build.
KINDS_WITHOUT_EVIDENCE: dict[str, frozenset[str]] = {
    "ltspice26": frozenset(),
    "ltspice17": frozenset({"stepped_run_op", "analysis:fra"}),
}


def _shown(build: str, scratch: Path) -> set[str]:
    """What the recorded cells of the grid show on ``build``: each stepped
    kind and each analysis a recorded file decodes as."""
    shown: set[str] = set()
    for index, cell in enumerate(GRID.cells):
        if not _read_on(build, cell):
            continue
        descriptor = rec.decode(build, _stem(cell), scratch / f"{index}").raw.descriptor
        shown.add(f"analysis:{cell.at(build)['analysis']}")
        if descriptor.stepped_kind is not None:
            shown.add(descriptor.stepped_kind)
    return shown


@pytest.mark.parametrize("build", rec.BUILDS)
def test_every_kind_the_code_declares_is_shown_by_a_recording_or_counted(
    build: str, tmp_path: Path
):
    """Every stepped kind and every LTspice analysis the code declares is
    shown by a recorded cell on each build, or is listed in
    ``KINDS_WITHOUT_EVIDENCE`` as claimed on no evidence."""
    declared = {kind.name for kind in STEPPED_KINDS} | {
        f"analysis:{analysis}" for analysis in _LTSPICE_ANALYSES
    }
    without = declared - _shown(build, tmp_path)
    assert without <= KINDS_WITHOUT_EVIDENCE[build], (
        f"{sorted(without - KINDS_WITHOUT_EVIDENCE[build])} claimed for {build} on no "
        "recorded evidence: record a cell that shows it, or list it with the reason"
    )
    assert without == KINDS_WITHOUT_EVIDENCE[build], (
        f"{sorted(KINDS_WITHOUT_EVIDENCE[build] - without)} is now shown on {build}: "
        "take it out of KINDS_WITHOUT_EVIDENCE"
    )


def test_a_kind_naming_a_recording_on_one_build_names_its_cell():
    """A kind's own list of the recordings that establish it names only
    files the grid indexes, so the grid's count covers them."""
    indexed = {_stem(cell) for cell in GRID.cells}
    for kind in STEPPED_KINDS:
        for name in kind.recorded:
            assert name.rpartition(":")[2] in indexed, (kind.name, name)


# ngspice ----------------------------------------------------------------------

#: Decks of the grid ngspice runs: one analysis each, with no .step (which
#: ngspice does not run) and no .fra (which it does not have).
_NGSPICE_DECKS = {
    "raw/tran.cir": "tran",
    "raw/ac.cir": "ac",
    "raw/dc.cir": "dc",
    "raw/noise.cir": "noise",
    "raw/op.cir": "op",
    "raw/tf.cir": "tf",
    "raw/layout_grid_dc_one.cir": "dc",
    "raw/noise_one_freq.cir": "noise",
}


def stored_as(raw: Path) -> list[str]:
    """How each plot of an ngspice raw stores its values, read off the file's
    own data markers (``Binary:`` or ``Values:`` at the start of a line of its
    ASCII header), without the reader under test."""
    data = raw.read_bytes()
    return [
        marker.decode("ascii")[:-1].lower()
        for marker in re.findall(rb"(?m)^(Binary:|Values:)", data)
    ]


@pytest.mark.parametrize("text", [False, True], ids=["default", "text"])
@pytest.mark.parametrize(("source", "analysis"), list(_NGSPICE_DECKS.items()))
def test_ngspice_lays_out_a_plot_as_every_route_reads_it(
    source: str, analysis: str, text: bool, tmp_path: Path
):
    """An ngspice raw of each analysis, as the installation writes it by
    default and asked for text: the header reader names its writer ngspice
    and each plot's storage as the file's own markers state it, one of its
    plots holds the analysis, and the decoder and the run row read the plot
    the file ends in as the header does.

    Which storage ngspice writes by default is the installation's: its
    ``spinit`` may set ``filetype`` (the official Windows build sets it to
    ascii, which the CI workflow notes), and ``-n`` reads no user file but
    still reads ``spinit``. So the storage is asserted as the file states it,
    not as asked. What ngspice names its plots beyond that, and how many it
    writes, differs by version, so only these are asserted."""
    raw = run_ngspice(rec.INPUTS / source, tmp_path / "run", text=text)
    header = preflight_raw(raw, limits=rec.RAW_LIMITS, producing_dialect="ngspice")
    assert {plot.dialect for plot in header.plots} == {"ngspice"}
    stored = stored_as(raw)
    print(f"{source}: ngspice stored {stored} ({'text asked' if text else 'default'})")
    assert [plot.storage for plot in header.plots] == stored
    if text:
        assert set(stored) == {"values"}, stored
    analyses = [plot_analysis(plot.plot_name, plot.first_type) for plot in header.plots]
    assert ANALYSES[analysis] in analyses, analyses
    last = header.plots[-1]
    progress = read_partial_raw_progress(raw, "ngspice")
    assert progress is not None
    assert progress.plot == last.plot_name
    assert progress.points == last.point_count
    decoded = rec.decode_file(raw, tmp_path, producing="ngspice").raw
    assert decoded.descriptor.dialect == "ngspice"
    assert decoded.descriptor.stepped_kind is None
