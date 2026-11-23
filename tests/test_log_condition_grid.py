"""Which condition a line of an LTspice log states, on every route that answers it.

The question is the log-condition grid's (``[grid.log-condition]`` in
``inputs/cases.toml``; ``docs/TESTING.md``, "Grids"). It is decided by the
rules of ``log_conditions`` (``_RULES``, read line by line over
``log_lines``), picked for the run's producer (``rules_for``) and reported by
``run_diagnostics``; ``log_parser.classify_line`` reads the same lines as
errors and warnings.

A cell is one line of a recorded log, taken by one rule on each build. This
module holds the index to the code (the rule axis is the rules that claim an
LTspice build, named from the code; each cell's line is taken by the rule it
names and states the code it declares; every line of an indexed log that
states a condition is a cell; every rule a recording matches on a build is a
cell's), and reads every recorded case through every route: the rules of the
build, the reader over the parser worker's decode, the decoded log read
through the parser process, the ``summary`` recipe by path and by job, the
run rows of ``run_experiments`` and ``jobs`` and the Python API's, the rules
of every recorded LTspice build (a build the server has no recording of) and
of every simulator (a producer it does not know), and ``classify_line``. The
cells whose condition an ngspice rule also names run through ngspice live,
where it is on PATH.
"""

from __future__ import annotations

import asyncio
import collections
import functools
import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import Any, ClassVar

import pytest
from spicelib.simulators.ltspice_simulator import LTspice

from ltspice_mcp.config import ServerConfig
from ltspice_mcp.lib import log_conditions, wsl
from ltspice_mcp.lib.encoding import read_spice_text
from ltspice_mcp.lib.log_parser import classify_line, extract_log_diagnostics
from ltspice_mcp.lib.services import AnalysisSource, load_logs_sync
from ltspice_mcp.lib.simulator import detect_simulators
from ltspice_mcp.lib.simulator_build import Producer, log_banner_build
from ltspice_mcp.lib.spice_lex import directive_head, lex_deck
from ltspice_mcp.state import SessionState
from tests import _ltspice_recorded as rec
from tests._recorded_grids import Cell, grid
from tests._stand_in_pe import stand_in_windows_program
from tests.conftest import FIXTURES_DIR, SyncApi, dispatched, terminal_experiment

GRID = grid("log-condition")

_LTSPICE = frozenset({"ltspice26", "ltspice17"})
_TAG = {
    frozenset({"ltspice26"}): "lt26",
    frozenset({"ltspice17"}): "xvii",
    frozenset({"ltspice26", "ltspice17"}): "lt",
    frozenset({"ltspice26", "ltspice17", "ngspice"}): "lt-ng",
    frozenset({"ngspice"}): "ng",
}

#: The build each build's log names: XVII is known by its program, 26 by the
#: banner its log opens with.
REPORTED = {
    "ltspice26": "LTspice 26.1.1 for Windows",
    "ltspice17": "Linear Technology Corporation LTspice XVII",
}

#: How many routes and functions answer otherwise than a cell, per build,
#: each declared on its cell (``differs``) with the reason. A count may only
#: fall: a fix lowers it here in the same change.
KNOWN_DIFFERENCES: dict[str, int] = {"ltspice26": 0, "ltspice17": 0}


#: The rules whose lines carry no word of the simulator's own (LTspice 26's
#: ``<deck>(<line>): <message>`` blocks, and its bare refusals of a deck) and
#: that ``classify_line`` reads as an error. Kept apart on purpose:
#: ``classify_line`` decides whether a run failed and what the summary's
#: ``errors`` list (``log_parser.extract_log_diagnostics``), and
#: ``log_conditions.said_as`` what the simulator called a line, which every
#: reply that shows a line's severity reports. Each is a rule and a build.
CLASSIFY_LINE_READS_AN_UNWORDED_LINE_AS_AN_ERROR: frozenset[tuple[str, str]] = frozenset(
    {
        ("unknown_model_parameter.lt26", "ltspice26"),
        ("unknown_instance_parameter.lt26", "ltspice26"),
        ("missing_include.lt26", "ltspice26"),
        ("missing_model.lt26", "ltspice26"),
        ("undefined_parameter.lt26", "ltspice26"),
        ("measurement_error.lt26", "ltspice26"),
        ("deck_error.lt26.1", "ltspice26"),
        ("deck_error.lt26.2", "ltspice26"),
        ("convergence_failed.lt", "ltspice26"),
        ("singular_matrix.lt", "ltspice26"),
    }
)


# The rules, named -----------------------------------------------------------


@functools.cache
def rule_names() -> tuple[str, ...]:
    """Each rule of ``_RULES`` by the name the grid's rule axis gives it: its
    code (``routine`` for none), the builds it claims, and its place among the
    rules of that code and those builds, where there is more than one."""
    keys = [(rule.code or "routine", _TAG[rule.builds]) for rule in log_conditions._RULES]
    total = collections.Counter(keys)
    seen: collections.Counter[tuple[str, str]] = collections.Counter()
    names = []
    for key in keys:
        seen[key] += 1
        names.append(f"{key[0]}.{key[1]}" + (f".{seen[key]}" if total[key] > 1 else ""))
    return tuple(names)


def _rule_name(rule: Any) -> str:
    return rule_names()[log_conditions._RULES.index(rule)]


#: An occurrence: the rule's name, the index of its first line, and its lines.
Occurrence = tuple[str | None, int, list[str]]


def occurrences(text: str, build: str) -> list[Occurrence]:
    """Each run of lines the build's rules take, with the rule that took it:
    the walk of ``log_conditions._occurrences`` over the same rule set, which
    this test checks gives its codes and lines (``_walk_is_the_classifiers``),
    so the rule it names is the one the classifier used."""
    rules = log_conditions._CLASSIFIERS[build]
    lines = log_conditions.log_lines(text)
    found: list[Occurrence] = []
    index = 0
    while index < len(lines):
        if not lines[index].strip():
            index += 1
            continue
        after = 0
        while (position := rules.first(lines, index, after)) is not None:
            rule = rules.rules[position]
            taken = rule.follow(lines, index + 1)
            if rule.needs_follow and not taken:
                after = position + 1
                continue
            found.append((_rule_name(rule), index, lines[index : index + 1 + taken]))
            index += 1 + taken
            break
        else:
            found.append((None, index, [lines[index]]))
            index += 1
    return found


def _walk_is_the_classifiers(text: str, build: str, found: Sequence[Occurrence]) -> None:
    code_of = dict(zip(rule_names(), (r.code for r in log_conditions._RULES), strict=True))
    walked = [
        (log_conditions.UNCLASSIFIED if name is None else code_of[name], block)
        for name, _, block in found
        if name is None or code_of[name] is not None
    ]
    assert walked == log_conditions._occurrences(text, log_conditions._CLASSIFIERS[build])


def _log_text(build: str, case_id: str) -> str:
    return read_spice_text(rec.recorded(build, f"{case_id}.log"))


def _recorded(build: str, case_id: str) -> bool:
    return case_id in rec.manifest(build)["cases"] and rec.has(build, f"{case_id}.log")


def _states(cell: Cell, build: str) -> str:
    states = cell.extra["states"]
    return states[build] if isinstance(states, dict) else states


def _text(cell: Cell, build: str) -> str | None:
    return cell.extra.get("text", {}).get(build)


def _cases() -> list[str]:
    return sorted({cell.case for cell in GRID.cells})


def _recorded_runs() -> list[Any]:
    runs = []
    for case_id in _cases():
        pending = rec.CASES.behaviours[rec.CASES.case(case_id).behaviour].pending
        for build in rec.BUILDS:
            if _recorded(build, case_id):
                runs.append(pytest.param(build, case_id, id=f"{build}/{case_id}"))
            elif pending:
                runs.append(
                    pytest.param(
                        build, case_id, id=f"{build}/{case_id}", marks=pytest.mark.skip(pending)
                    )
                )
    return runs


# The index, held to the code -------------------------------------------------


def test_the_rule_axis_is_every_rule_that_claims_an_ltspice_build():
    claimed = [
        name
        for name, rule in zip(rule_names(), log_conditions._RULES, strict=True)
        if rule.builds & _LTSPICE
    ]
    assert list(GRID.axes["rule"].values) == claimed
    assert len(set(rule_names())) == len(rule_names())


def test_a_rule_is_excluded_on_exactly_the_builds_it_does_not_claim():
    for name, rule in zip(rule_names(), log_conditions._RULES, strict=True):
        if not rule.builds & _LTSPICE:
            continue
        for build in rec.BUILDS:
            excluded = any(e.covers({"rule": name}, build) for e in GRID.exclusions)
            assert excluded == (build not in rule.builds), (name, build)


def test_the_form_axis_is_every_named_condition_an_ltspice_rule_states():
    generic = {"simulator_error", "simulator_warning"}
    named = {
        rule.code
        for rule in log_conditions._RULES
        if rule.code and rule.builds & _LTSPICE and rule.code not in generic
    }
    assert set(GRID.axes["form"].values) == named
    assert generic <= log_conditions.diagnostic_codes()


def test_every_cell_states_a_code_the_rules_can_give():
    codes = log_conditions.diagnostic_codes() | {"none"}
    for cell in GRID.cells:
        for build in rec.BUILDS:
            if build in cell.refused:
                assert _text(cell, build) is None, (cell.id, build)
                assert not cell.at(build).get("rule"), (cell.id, build)
                continue
            assert _states(cell, build) in codes, (cell.id, build)
            assert _text(cell, build), f"{cell.id}: no text for {build}"
            rule = cell.at(build).get("rule")
            assert rule, f"{cell.id}: no rule on {build}"
            state = _states(cell, build)
            assert rule.partition(".")[0] == ("routine" if state == "none" else state), (
                cell.id,
                build,
                rule,
                state,
            )


@functools.cache
def _has_step(source: str) -> bool:
    deck = rec.INPUTS / source
    cards = lex_deck(read_spice_text(deck)).cards
    return any(card.kind == "directive" and directive_head(card) == ".step" for card in cards)


def test_a_cell_in_a_stepped_run_is_one_whose_deck_steps():
    for cell in GRID.cells:
        phase = cell.coordinates.get("phase", "")
        if not phase:
            continue
        source = rec.CASES.case(cell.case).source
        assert (phase == "stepped") == _has_step(source), (cell.id, phase)


def test_every_case_of_the_grid_is_waiting_or_has_no_prediction_left():
    """A recorded cell declares what the recording shows: a cell whose
    expectation was a prediction (``not_established``) is settled against its
    recording, and the key removed, once the case is recorded."""
    for cell in GRID.cells:
        if "not_established" not in cell.extra:
            continue
        recorded = [build for build in rec.BUILDS if _recorded(build, cell.case)]
        assert not recorded, (
            f"{cell.id} is recorded on {recorded}: settle what it declares against the "
            "recording and remove not_established"
        )


@pytest.mark.parametrize(("build", "case_id"), _recorded_runs())
def test_each_cell_is_the_line_its_rule_takes_and_every_condition_is_a_cell(
    build: str, case_id: str
):
    """On this build, the case's log has a run of lines for each cell, found by
    the cell's text, taken by the rule the cell names, of the code the cell
    states; and every run of lines that states a condition is a cell's."""
    text = _log_text(build, case_id)
    found = occurrences(text, build)
    _walk_is_the_classifiers(text, build, found)
    code_of = dict(zip(rule_names(), (r.code for r in log_conditions._RULES), strict=True))
    claimed: set[int] = set()
    for cell in GRID.cells_of(case_id):
        if build in cell.refused:
            continue
        wanted = _text(cell, build)
        assert wanted is not None
        mine = [(name, index) for name, index, block in found if wanted in block[0]]
        assert mine, f"{cell.id}: no line of {build}'s log holds {wanted!r}"
        for name, index in mine:
            assert name is not None and name == cell.at(build)["rule"], (cell.id, build, name)
            assert (code_of[name] or "none") == _states(cell, build), (cell.id, build)
            claimed.add(index)
    # A condition the log states again, by the same rule (the rungs of one
    # bias-point search, a warning for each device), is the cell's that names
    # its first line; a condition no cell states is not indexed.
    stated = {_states(cell, build) for cell in GRID.cells_of(case_id) if build not in cell.refused}
    unindexed = [
        (name, block[0])
        for name, index, block in found
        if name is not None
        and code_of[name] is not None
        and index not in claimed
        and code_of[name] not in stated
    ]
    assert not unindexed, f"{build}/{case_id}: lines that state a condition and are no cell"
    assert all(name is not None for name, _, _ in found), "a line no rule takes"


def test_every_rule_a_recording_matches_on_a_build_is_a_cells():
    """The grid's rule axis is filled from cells, so a rule a recorded log
    matches must be a cell's on that build; otherwise the grid would count a
    rule as not recorded that a recording shows."""
    filled = {
        (cell.at(build).get("rule"), build)
        for cell in GRID.cells
        for build in rec.BUILDS
        if _recorded(build, cell.case) and build not in cell.refused
    }
    matched: set[tuple[str, str]] = set()
    for build in rec.BUILDS:
        for path in sorted((rec.FIXTURES / build).rglob("*.log")):
            for name, _, _ in occurrences(read_spice_text(path), build):
                if name is not None:
                    matched.add((name, build))
    assert matched - filled == set()


# Every route ---------------------------------------------------------------


def _staged(build: str, case_id: str, directory: Path) -> tuple[Path, Path, Path]:
    """The case's deck and log in ``directory``, and a raw beside the log:
    the one the build wrote for the case, or for ``deck/after_end``, a run of
    the same build, where it wrote none. The summary reads what the log says,
    and finds a path's producer from the raw's header and the log's banner."""
    directory.mkdir(parents=True, exist_ok=True)
    name = Path(case_id).name
    deck = directory / f"{name}{Path(rec.CASES.case(case_id).source).suffix}"
    shutil.copyfile(rec.INPUTS / rec.CASES.case(case_id).source, deck)
    log = directory / f"{name}.log"
    shutil.copyfile(rec.recorded(build, f"{case_id}.log"), log)
    raw = directory / f"{name}.raw"
    own = f"{case_id}.raw"
    shutil.copyfile(rec.recorded(build, own if rec.has(build, own) else "deck/after_end.raw"), raw)
    return deck, raw, log


def _replaying(work_dir: Path, build: str, case_id: str) -> type:
    """A stand-in for ``build`` that writes the files it recorded for the case
    and exits as it did."""
    log = rec.recorded(build, f"{case_id}.log")
    raw = f"{case_id}.raw"
    raw_path = rec.recorded(build, raw) if rec.has(build, raw) else None
    exit_code = rec.entry(build, case_id)["exit_code"]

    class _Recorded(LTspice):
        spice_exe: ClassVar[list[str]] = []

        @classmethod
        def run(
            cls,
            netlist_file,
            cmd_line_switches=None,
            timeout=None,
            stdout=None,
            stderr=None,
            cwd=None,
            exe_log=False,
        ) -> int:
            netlist = Path(netlist_file)
            shutil.copy(log, netlist.with_suffix(".log"))
            if raw_path is not None:
                shutil.copy(raw_path, netlist.with_suffix(".raw"))
            return exit_code

        @classmethod
        def valid_switch(cls, switch: str, path: str = "") -> list:
            return []

    # The executable carries the file version the recording read from the
    # build's own, which is how the server tells the builds apart before a run
    # has named its build (simulator_build.which_ltspice).
    executable = rec.manifest(build)["executable"]
    program = stand_in_windows_program(
        work_dir / "sim" / executable["name"], executable["file_version"]
    )
    return type("LTspice", (_Recorded,), {"spice_exe": [str(program)]})


def _fields(row: dict[str, Any]) -> dict[str, Any]:
    found: dict[str, Any] = {}
    if "diagnostics" in row:
        found["diagnostics"] = row["diagnostics"]
    if "diagnostics_coverage" in row:
        found["coverage"] = row["diagnostics_coverage"]
    return found


def _summary_value(reply: dict[str, Any]) -> dict[str, Any]:
    assert "items" in reply["results"]["s"].get("per_run", {}), reply.get("failures")
    (row,) = reply["results"]["s"]["per_run"]["items"]
    return row["value"]


_SUMMARY: dict[str, Any] = {
    "recipes": [{"key": "s", "metric": "summary"}],
    "include": {"per_run": {"limit": 10}, "fields": ["value"]},
}


async def _every_route(
    build: str, case_id: str, work_dir: Path, not_run: dict[str, str]
) -> dict[str, dict[str, Any]]:
    """What each route that answers the question says of the case's log, as
    its diagnostics fields; ``not_run`` gets each route that could not read
    it, with the reason."""
    producer = Producer("ltspice", REPORTED[build], rec.manifest(build)["executable"]["name"])
    deck, raw, log = _staged(build, case_id, work_dir / "path")
    routes: dict[str, dict[str, Any]] = {}
    conditions = extract_log_diagnostics(log)["conditions"]
    routes["reader"] = log_conditions.run_diagnostics(conditions, producer)
    state = SessionState.create(
        ServerConfig(working_dir=work_dir, allowed_paths=[work_dir], simulator="ltspice"),
        available={"ltspice": _replaying(work_dir, build, case_id)},
    )
    decoded = await asyncio.to_thread(
        load_logs_sync,
        AnalysisSource(
            raw=None,
            log=log,
            netlist=None,
            dialect="ltspice",
            identity=None,
            trusted_job_artifact=True,
        ),
        state,
    )
    routes["decoded log"] = log_conditions.decoded_run_diagnostics(decoded, producer)
    by_path = await dispatched(
        state, "analyze_results", sources=[{"path": str(raw), "label": "p"}], **_SUMMARY
    )
    api = SyncApi(state)
    routes["summary, by path"] = _fields(_summary_value(by_path))
    api_by_path = await asyncio.to_thread(
        lambda: api.analyze_results(sources=[{"path": str(raw), "label": "p"}], **_SUMMARY)
    )
    routes["api summary, by path"] = _fields(_summary_value(api_by_path))
    # A producer the server has no recording of reads with every recorded
    # LTspice build's wording, and one it does not know with every simulator's.
    routes["every recorded LTspice build's rules"] = log_conditions.run_diagnostics(
        conditions, Producer("ltspice", None, None)
    )
    routes["every simulator's rules"] = log_conditions.run_diagnostics(
        conditions, Producer(None, None, None)
    )

    run_deck = work_dir / "run" / deck.name
    run_deck.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(deck, run_deck)
    receipt = await terminal_experiment(
        state,
        {
            "request_id": f"conditions-{build}-{case_id.replace('/', '-')}",
            "circuits": [{"path": str(run_deck), "id": "dut"}],
            "execution": {"wait_s": 30},
            "lint": "off",
        },
    )
    if not receipt["completeness"]["submitted"]:
        # The server refused the deck before any simulator ran it (an include
        # it could not stage): no run row has a log to report. Said, not
        # skipped silently.
        (failure,) = receipt["failures"]
        not_run["run rows and the summary by job"] = failure["code"]
        return routes
    (row,) = receipt["runs"]["items"]
    assert row.get("simulator_version") in (REPORTED[build], None), row
    routes["run_experiments row"] = _fields(row)
    job_id = receipt["job_id"]
    runs = await dispatched(state, "jobs", action="runs", job_id=job_id)
    (jobs_row,) = runs["items"]
    routes["jobs row"] = _fields(jobs_row)
    api_runs = await asyncio.to_thread(api.jobs, action="runs", job_id=job_id)
    (api_row,) = api_runs["items"]
    routes["api jobs row"] = _fields(api_row)
    if row["status"] == "produced":
        by_job = await dispatched(
            state, "analyze_results", sources=[{"job_id": job_id, "label": "j"}], **_SUMMARY
        )
        routes["summary, by job"] = _fields(_summary_value(by_job))
        api_by_job = await asyncio.to_thread(
            lambda: api.analyze_results(sources=[{"job_id": job_id, "label": "j"}], **_SUMMARY)
        )
        routes["api summary, by job"] = _fields(_summary_value(api_by_job))
    return routes


#: Routes given the run's producer, rather than finding it.
_GIVEN_THE_PRODUCER = frozenset({"reader", "decoded log"})

#: A run of LTspice 26 whose log names no build (it refused the deck, and
#: opens its log with the refusal) is read with every recorded LTspice build's
#: rules and says ``build_unrecorded``, though the server tells the build
#: from the executable's file version before any run (``rules_for`` reads the
#: version resource only to tell XVII). The codes are the same here; the run
#: row says it was not read with its own build's rules.
RUN_OF_LTSPICE_26_NAMING_NO_BUILD_IS_READ_AS_UNRECORDED = True

#: Routes that read with the rules of another producer than the run's, and
#: say so in a coverage key.
_OTHER_RULES = {
    "every recorded LTspice build's rules": "build_unrecorded",
    "every simulator's rules": "producer_unknown",
}


def _declared(cells: Sequence[Cell], build: str) -> dict[tuple[str, str], str]:
    return {
        (route, cell.id): reason
        for cell in cells
        for entry in cell.extra.get("differs", ())
        if entry["build"] == build
        for route, reason in [(entry["route"], entry["reason"])]
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(("build", "case_id"), _recorded_runs())
async def test_every_recorded_cell_states_its_condition_on_every_route(
    build: str, case_id: str, work_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    """The conditions each route reports for the case's log are the codes its
    cells state on this build, none unclassified; the routes that read with
    the run's own rules report the same entries as the reader; and
    A route that answers otherwise is declared on its cell (``differs``), and
    counted in :data:`KNOWN_DIFFERENCES`."""
    monkeypatch.setattr(wsl, "is_wsl", lambda: False)
    cells = [cell for cell in GRID.cells_of(case_id) if build not in cell.refused]
    expected = {_states(cell, build) for cell in cells} - {"none"}
    not_run: dict[str, str] = {}
    routes = await _every_route(build, case_id, work_dir, not_run)
    assert set(not_run.values()) <= {"include_unstaged"}, not_run
    print(f"routes not run for {build}/{case_id}: {not_run}")
    text = _log_text(build, case_id)
    named = log_banner_build(text) is not None or rec.generation(build) == "xvii"
    declared = _declared(GRID.cells_of(case_id), build)
    found: set[tuple[str, str]] = set()

    def differs(route: str, ok: bool, detail: Any) -> None:
        # A route's whole answer for the case is owed to the cell whose code
        # it gets wrong; one that gets no cell wrong is owed to none.
        if ok:
            return
        owner = [key for key in declared if key[0] == route]
        assert owner, (route, detail)
        found.update(owner)

    reader = routes["reader"]
    for route, said in routes.items():
        codes = {entry["code"] for entry in said.get("diagnostics", ())}
        coverage = _OTHER_RULES.get(route)
        if coverage is None and route not in _GIVEN_THE_PRODUCER and not named:
            # LTspice 26 opens a log it refuses the deck in with the refusal,
            # not its banner, and writes no results file: a path, and a run of
            # it, name no build, and are read with every recorded LTspice
            # build's rules, and say so, though a run's executable is 26.1.1.
            assert RUN_OF_LTSPICE_26_NAMING_NO_BUILD_IS_READ_AS_UNRECORDED
            coverage = "build_unrecorded"
        assert said.get("coverage") == coverage, (route, said)
        differs(route, codes == expected, (codes, expected))
        if coverage is None:
            assert said.get("diagnostics") == reader.get("diagnostics"), route

    assert found == set(declared), set(declared) - found


def test_classify_line_reads_each_cells_lines_as_the_word_the_simulator_wrote():
    """``classify_line`` reads each recorded cell's lines as the error or the
    warning the simulator wrote on them, except where a build writes no word
    and it reads an error: those are the rules and builds of
    :data:`CLASSIFY_LINE_READS_AN_UNWORDED_LINE_AS_AN_ERROR`, held to exactly
    the ones the recordings show."""
    read_otherwise: set[tuple[str, str]] = set()
    for cell in GRID.cells:
        for build in rec.BUILDS:
            wanted = _text(cell, build)
            if wanted is None or not _recorded(build, cell.case):
                continue
            text = _log_text(build, cell.case)
            lines = log_conditions.log_lines(text)
            for _, index, block in occurrences(text, build):
                if wanted not in block[0]:
                    continue
                entries = log_conditions.classify({"log": "\n".join(block)}, build)
                said = log_conditions.said_as(entries[0]) if entries else None
                kinds = {classify_line(lines, i).kind for i in range(index, index + len(block))}
                read = "error" if "error" in kinds else "warning" if "warning" in kinds else None
                if read != said:
                    assert (read, said) == ("error", None), (cell.id, build, read, said, block)
                    read_otherwise.add((cell.at(build)["rule"], build))
    assert read_otherwise == CLASSIFY_LINE_READS_AN_UNWORDED_LINE_AS_AN_ERROR


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


# ngspice --------------------------------------------------------------------

#: Decks of cells whose condition an ngspice rule also names, run through
#: ngspice live. The LTspice deck of the unknown model parameter steps it,
#: which ngspice does not run, so its ngspice copy leaves the .step out.
_NGSPICE_DECKS = (
    FIXTURES_DIR / "ngspice_runs" / "unknown_model_param.cir",
    rec.INPUTS / "log/err_source_loop.cir",
    rec.INPUTS / "log/err_floating_node.cir",
    rec.INPUTS / "log/err_unknown_model.cir",
)


@pytest.mark.asyncio
@pytest.mark.parametrize("source", _NGSPICE_DECKS, ids=lambda path: path.stem)
async def test_ngspice_names_a_conditions_line_the_same_on_every_route(
    source: Path, work_dir: Path
):
    """Each route that reports an ngspice run's conditions (its row in
    ``run_experiments``, ``jobs`` and the Python API, and the summary of a
    run that produced) reports the same entries, read with ngspice's own
    rules. What ngspice writes differs by version (CI runs 45 and 47), so
    which conditions it states is printed, not asserted."""
    if shutil.which("ngspice") is None:
        pytest.skip("ngspice is not on PATH")
    deck = work_dir / source.name
    shutil.copyfile(source, deck)
    config = ServerConfig(working_dir=work_dir, allowed_paths=[work_dir], simulator="ngspice")
    state = SessionState.create(config, available=detect_simulators(config))
    receipt = await terminal_experiment(
        state,
        {
            "request_id": f"conditions-ngspice-{source.stem}",
            "circuits": [{"path": str(deck), "id": "dut"}],
            "execution": {"wait_s": 60},
            "lint": "off",
        },
    )
    (row,) = receipt["runs"]["items"]
    routes = {"run_experiments row": _fields(row)}
    job_id = receipt["job_id"]
    (jobs_row,) = (await dispatched(state, "jobs", action="runs", job_id=job_id))["items"]
    routes["jobs row"] = _fields(jobs_row)
    api = SyncApi(state)
    (api_row,) = (await asyncio.to_thread(api.jobs, action="runs", job_id=job_id))["items"]
    routes["api jobs row"] = _fields(api_row)
    if row["status"] == "produced":
        by_job = await dispatched(
            state, "analyze_results", sources=[{"job_id": job_id, "label": "j"}], **_SUMMARY
        )
        routes["summary, by job"] = _fields(_summary_value(by_job))
    print(f"ngspice on {source.stem}: {routes['run_experiments row']}")
    for route, said in routes.items():
        assert "coverage" not in said, (route, said)
        assert said == routes["run_experiments row"], route
