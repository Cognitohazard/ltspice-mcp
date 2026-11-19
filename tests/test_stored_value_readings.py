"""Every reply that gives a number read from a stored deck says when a
recorded LTspice build reads that value's stored bytes as another number.

The question is "which recorded builds read this stored value as another
number than the server does", and ``lib/stored_values.py`` answers it. The
recorded decks of ``deck-encoding`` hold one micro sign or Greek mu each, in
every encoding a build was recorded reading (``inputs/cases.toml``); each
build's ``.op`` raw shows the number it took. Every route that reads a value
from a stored deck and sends a number or a comparison runs each deck, and its
first reply states exactly the builds whose recorded number differs from the
server's, with that number, and no other.
"""

from __future__ import annotations

import asyncio
import math
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import pytest

from ltspice_mcp.lib.format import SAME_VALUE_RTOL
from ltspice_mcp.lib.hierarchy import SemanticProfile, load_hierarchy
from ltspice_mcp.state import SessionState
from tests import _ltspice_recorded as rec
from tests.conftest import FakeSim, SyncApi, dispatched, fake_simulator
from tests.ltspice_recorder import INPUTS

#: The code a statement carries (``stored_values.CODE``).
CODE = "value_read_otherwise"

#: The recorded decks whose one value is a micro sign or a Greek mu, on node a.
DECKS = (
    "deck/micro_utf8",
    "deck/micro_utf8_bom",
    "deck/greek_mu_utf8",
    "deck/micro_cp1252",
    "deck/micro_utf16le",
    "deck/micro_utf16le_bom",
)

#: The name a statement gives each recorded build.
BUILD_NAMES = {"ltspice26": "LTspice 26", "ltspice17": "LTspice XVII"}

#: What the server reads each deck's value as: its run spells the sign u.
SERVER_NUMBER = 1e-6

#: The same circuit with the value spelled in ASCII, which every build reads
#: alike: the other side of a comparison.
ASCII_DECK = "* the same value, spelled u\nV1 a 0 1u\n.op\n.end\n"


def _source(case_id: str) -> Path:
    return INPUTS / rec.CASES.case(case_id).source


def _recorded_numbers(case_id: str, scratch: Path) -> dict[str, float]:
    """The number each recorded build took for the deck's value."""
    return {
        BUILD_NAMES[build]: rec.operating_point(build, case_id, scratch)["v(a)"]
        for build in rec.BUILDS
        if case_id in rec.manifest(build)["cases"]
    }


def _expected(case_id: str, scratch: Path) -> dict[str, float]:
    """The builds whose recorded number is not the server's, with that number."""
    return {
        build: number
        for build, number in _recorded_numbers(case_id, scratch).items()
        if not math.isclose(number, SERVER_NUMBER, rel_tol=SAME_VALUE_RTOL)
    }


# The routes ---------------------------------------------------------------------

Route = Callable[[SessionState, Path, Path], Awaitable[list[dict[str, Any]]]]


async def _inspect(state: SessionState, deck: Path, _ascii: Path) -> list[dict[str, Any]]:
    query = {"kind": "hierarchy", "path": str(deck), "simulator": "ltspice"}
    (result,) = (await dispatched(state, "inspect", queries=[query]))["results"]
    assert result["ok"], result
    return result["data"].get("values_read_otherwise", [])


async def _api_inspect(state: SessionState, deck: Path, _ascii: Path) -> list[dict[str, Any]]:
    query = {"kind": "hierarchy", "path": str(deck), "simulator": "ltspice"}
    reply = await asyncio.to_thread(lambda: SyncApi(state).inspect(queries=[query]))
    (result,) = reply["results"]
    assert result["ok"], result
    return result["data"].get("values_read_otherwise", [])


async def _load_hierarchy(_state: SessionState, deck: Path, _ascii: Path) -> list[dict[str, Any]]:
    hierarchy = load_hierarchy(str(deck), None, SemanticProfile("ltspice"))
    return [reading.statement() for reading in getattr(hierarchy, "readings", ())]


def _verify_compare(mode: str, *, deck_is_reference: bool) -> Route:
    async def route(state: SessionState, deck: Path, ascii_deck: Path) -> list[dict[str, Any]]:
        path, reference = (ascii_deck, deck) if deck_is_reference else (deck, ascii_deck)
        data = await dispatched(
            state,
            "verify_circuit",
            path=str(path),
            checks=["compare"],
            compare={"reference": str(reference), "mode": mode},
        )
        comparison = data["comparison"]
        assert comparison is not None, data
        return comparison.get("values_read_otherwise", [])

    return route


ROUTES: dict[str, Route] = {
    "inspect": _inspect,
    "api inspect": _api_inspect,
    "load_hierarchy": _load_hierarchy,
    "verify_circuit compare equivalence": _verify_compare("equivalence", deck_is_reference=False),
    "verify_circuit compare equivalence, the deck as reference": _verify_compare(
        "equivalence", deck_is_reference=True
    ),
    "verify_circuit compare structural_diff": _verify_compare(
        "structural_diff", deck_is_reference=False
    ),
    "verify_circuit compare structural_diff, the deck as reference": _verify_compare(
        "structural_diff", deck_is_reference=True
    ),
}


@pytest.mark.parametrize("route", sorted(ROUTES))
@pytest.mark.parametrize("case_id", DECKS)
async def test_every_route_states_each_build_that_reads_the_stored_value_otherwise(
    case_id: str, route: str, tmp_path: Path, work_dir: Path, state_no_sim: SessionState
):
    expected = _expected(case_id, tmp_path)
    deck = work_dir / _source(case_id).name
    deck.write_bytes(_source(case_id).read_bytes())
    ascii_deck = work_dir / "ascii.cir"
    ascii_deck.write_text(ASCII_DECK, encoding="ascii")

    stated = await ROUTES[route](state_no_sim, deck, ascii_deck)

    assert all(statement["code"] == CODE for statement in stated), stated
    assert all(statement["file"] == str(deck) for statement in stated), stated
    by_build = {statement["build"]: statement for statement in stated}
    assert len(by_build) == len(stated), stated
    assert set(by_build) == set(expected), (case_id, route, stated)
    for build, number in expected.items():
        statement = by_build[build]
        assert statement["established"], statement
        assert math.isclose(statement["build_number"], number, rel_tol=SAME_VALUE_RTOL)
        assert math.isclose(statement["server_number"], SERVER_NUMBER, rel_tol=SAME_VALUE_RTOL)
        assert statement["value"] in ("1µ", "1μ")
        assert statement["line"] == 3


# A run: the receipt states the source's readings ---------------------------------

#: A run of the deck as it is, and a Monte Carlo draw around its value's
#: nominal, which the server reads from the stored deck.
RUNS: dict[str, dict[str, Any]] = {
    "run_experiments": {},
    "run_experiments Monte Carlo nominal": {
        "variations": [
            {
                "kind": "random",
                "runs": 2,
                "seed": 7,
                "rules": [{"rule": "component", "target": "R1", "tolerance": 0.1}],
            }
        ]
    },
}


def _run_statements(reply: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        statement
        for item in reply.get("observations", [])
        if item["code"] == CODE
        for statement in item["evidence"].get("values", ())
    ]


@pytest.mark.parametrize("run", sorted(RUNS))
@pytest.mark.parametrize("case_id", DECKS)
async def test_a_run_states_each_build_that_reads_the_source_otherwise(
    case_id: str,
    run: str,
    tmp_path: Path,
    work_dir: Path,
    config,
    monkeypatch: pytest.MonkeyPatch,
):
    """The run reads the value as the server does, whatever its simulator; its
    receipt, and the job's status, state the builds that read the source
    otherwise, as every other route does."""
    expected = _expected(case_id, tmp_path)
    state = SessionState.create(config, available={"fake": FakeSim})
    fake_simulator(monkeypatch)
    deck = work_dir / _source(case_id).name
    deck.write_bytes(_source(case_id).read_bytes())

    reply = await dispatched(
        state,
        "run_experiments",
        request_id=f"{case_id}-{run}".replace("/", "-").replace(" ", "-"),
        circuits=[{"path": str(deck), "id": "dut"}],
        lint="off",
        execution={"wait_s": 60},
        **RUNS[run],
    )
    status = await dispatched(state, "jobs", action="status", job_id=reply["job_id"])

    for stated in (_run_statements(reply), _run_statements(status)):
        by_build = {statement["build"]: statement for statement in stated}
        assert set(by_build) == set(expected), (case_id, run, stated)
        for build, number in expected.items():
            assert by_build[build]["file"] == str(deck.resolve())
            assert math.isclose(by_build[build]["build_number"], number, rel_tol=SAME_VALUE_RTOL)


# What no recording shows is said not to be established ---------------------------


def test_a_reading_no_recording_shows_is_stated_as_not_established():
    """A micro sign in braces or in a directive, read by XVII as letters,
    takes a number no recording shows; a Greek mu stored as UTF-16 is
    recorded on no build. Each is stated, with established false, never
    assumed alike or otherwise."""
    from ltspice_mcp.lib.stored_values import readings_of_bytes

    braces = readings_of_bytes(
        "* t\nR1 a 0 {1µ}\n.param p=1µ\n.end\n".encode(), root=True, file="d.cir"
    )
    assert [(r.build, r.line, r.build_reads, r.established) for r in braces] == [
        ("LTspice XVII", 2, "1Âµ", False),
        ("LTspice XVII", 3, "1Âµ", False),
    ]
    utf16 = readings_of_bytes("* t\nR1 a 0 1μ\n.end\n".encode("utf-16"), root=True)
    assert [(r.build, r.build_reads, r.established) for r in utf16] == [
        ("LTspice 26", None, False),
        ("LTspice XVII", None, False),
    ]
    assert all(not r.statement()["established"] for r in (*braces, *utf16))


def test_a_sign_outside_an_instances_own_value_is_not_established():
    """The bare number is recorded for an instance's own value; a parameter or
    a source function's argument is a position no recording shows."""
    from ltspice_mcp.lib.stored_values import readings_of_bytes

    deck = "* t\nC1 a 0 1µ Rser=1µ\nV1 a 0 SIN(0 1 1µ)\nQ1 c b 0 QN 1µ\n.end\n".encode()
    assert [(r.line, r.offset, r.build_number) for r in readings_of_bytes(deck, root=True)] == [
        (2, 8, 1.0),
        (2, 16, None),
        (3, 16, None),
        (4, 13, None),
    ]


@pytest.mark.parametrize("mode", ["equivalence", "structural_diff"])
def test_a_comparison_states_no_reading_of_the_servers_export_of_a_sheet(
    mode: str, work_dir: Path, state_no_sim: SessionState
):
    """A side compared as the export of a sheet is the server's file, not one
    the caller stored, so how a build reads its bytes is not stated; the
    stored reference beside it is."""
    from ltspice_mcp.tools.verify import VerifyCompareSpec, compare_netlists

    export = work_dir / "sheet.net"
    export.write_bytes(_source("deck/micro_utf8").read_bytes())
    reference = work_dir / "reference.cir"
    reference.write_bytes(_source("deck/micro_utf8").read_bytes())
    sheet = work_dir / "sheet.asc"

    spec = VerifyCompareSpec(reference=str(reference), mode=mode)  # type: ignore[arg-type]
    comparison, _findings, failure, _warnings = compare_netlists(
        spec, reference, export, reference, sheet, state_no_sim, ref_base_dir=work_dir
    )

    assert failure is None and comparison is not None
    assert [(s["file"], s["build"]) for s in comparison["values_read_otherwise"]] == [
        (str(reference), "LTspice XVII")
    ]
    # Not checking the sheet is said, since an absent statement reads as checked.
    assert [entry["file"] for entry in comparison["values_read_otherwise_not_checked"]] == [
        str(sheet)
    ]


def test_a_comparison_states_only_the_sections_of_a_library_the_deck_reads(work_dir: Path):
    """The comparison's graph and the hierarchy state the same values: those
    of the cards the reader reads, not those of a library section it does not
    (ngspice reads sections; LTspice has none)."""
    from ltspice_mcp.lib.netlist_graph import parse_netlist_graph

    (work_dir / "parts.lib").write_bytes(
        ".lib used\n.subckt cap a b\nC1 a b 2µ\n.ends\n.endl used\n"
        ".lib unused\n.subckt big a b\nC1 a b 3µ\n.ends\n.endl unused\n".encode()
    )
    deck = work_dir / "deck.cir"
    deck.write_bytes(b"* t\n.lib parts.lib used\nX1 a 0 cap\nR1 a 0 1k\n.op\n.end\n")
    reader = SemanticProfile("ngspice", "hsa")

    graph = parse_netlist_graph(deck, reader=reader)
    hierarchy = load_hierarchy(str(deck), None, reader)

    compared = {(r.value, r.build) for r in graph.readings}
    assert compared == {(r.value, r.build) for r in hierarchy.readings} == {("2µ", "LTspice XVII")}


def test_staging_says_a_sheet_run_as_its_export_was_not_checked(tmp_path: Path):
    """A sheet is run as the server's export of it, which is not a file the
    caller stored: staging states no reading of the export's bytes, and says
    the sheet's own values were not checked; an include the export names is
    the caller's file and is stated."""
    from ltspice_mcp.lib.deck_staging import stage_deck

    root = tmp_path / "project"
    root.mkdir()
    sheet = root / "sheet.asc"
    sheet.write_text("Version 4\nSHEET 1 880 680\n", encoding="ascii")
    (root / "part.inc").write_bytes("C2 a 0 2µ\n".encode())
    export = root / "sheet.net"
    export.write_bytes("* sheet\nR1 a 0 1µ\n.include part.inc\n.op\n.end\n".encode())

    staged = stage_deck(export, tmp_path / "stage", [root], origin=sheet, dialect=None)

    observed = {
        item["evidence"]["file"]: item["evidence"]
        for item in staged.observations
        if item["code"] == CODE
    }
    assert set(observed) == {str(sheet.resolve()), str((root / "part.inc").resolve())}
    assert set(observed[str(sheet.resolve())]) == {"file", "not_checked"}
    assert [s["value"] for s in observed[str((root / "part.inc").resolve())]["values"]] == ["2µ"]


def test_a_sheets_export_handed_over_as_text_is_said_not_checked(
    work_dir: Path, state_no_sim: SessionState
):
    """edit_schematic compares the edited sheet's export as text, which has no
    stored bytes: where it holds a micro sign the sheet is listed as not
    checked, never left out as though it were read alike."""
    from ltspice_mcp.tools.verify import VerifyCompareSpec, compare_netlists

    reference = work_dir / "reference.cir"
    reference.write_text(ASCII_DECK, encoding="ascii")
    sheet = work_dir / "sheet.asc"
    spec = VerifyCompareSpec(reference=str(reference))

    for text, expected in (("* s\nV1 a 0 1µ\n.op\n.end\n", [str(sheet)]), (ASCII_DECK, [])):
        comparison, _findings, failure, _warnings = compare_netlists(
            spec, reference, text, reference, sheet, state_no_sim, ref_base_dir=work_dir
        )
        assert failure is None and comparison is not None
        listed = comparison["values_read_otherwise_not_checked"]
        assert [entry["file"] for entry in listed] == expected
        # No bytes, so no claim that a build reads them otherwise.
        assert all("reads otherwise" not in entry["reason"] for entry in listed)
