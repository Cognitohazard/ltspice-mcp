"""Every reply that gives a number read from a stored file says when an
LTspice build the session runs reads that value's stored bytes as another
number.

The question is "which recorded builds read this stored value as another
number than the server does", and ``lib/stored_values.py`` answers it for
every recorded build. A reply states the answer for the LTspice builds the
session runs (``services.session_ltspice_builds``), one summary per file,
build and kind of reading, whatever the number of values.

The recorded decks of ``deck-encoding`` hold one micro sign or Greek mu each,
in every encoding a build was recorded reading (``inputs/cases.toml``); each
build's ``.op`` raw shows the number it took. Every route that reads a value
from a stored deck and sends a number or a comparison runs each deck, in a
session that runs LTspice XVII, LTspice 26, both, or ngspice alone, and its
first reply states exactly the builds the session runs whose recorded number
differs from the server's, with that number. The sheets' recordings
(``export-encoding``) hold each build's export of a sheet of each encoding,
which shows the text that build read from the sheet.
"""

from __future__ import annotations

import asyncio
import json
import math
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any, ClassVar

import pytest

from ltspice_mcp.config import ServerConfig
from ltspice_mcp.lib.encoding import read_spice_text_with_encoding
from ltspice_mcp.lib.format import SAME_VALUE_RTOL, fold_micro_sign, parse_spice_value
from ltspice_mcp.lib.hierarchy import SemanticProfile, load_hierarchy
from ltspice_mcp.lib.services import session_ltspice_builds
from ltspice_mcp.lib.spice_lex import lex_deck
from ltspice_mcp.lib.spice_lex_ops import value_suffix_sites
from ltspice_mcp.lib.stored_values import (
    CODE,
    EXAMPLES,
    readings_of_bytes,
    readings_of_sheet,
    summaries,
)
from ltspice_mcp.state import SessionState
from tests import _ltspice_recorded as rec
from tests.conftest import FakeSim, SyncApi, dispatched, fake_simulator
from tests.ltspice_recorder import INPUTS

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


class FakeXVII(FakeSim):
    """An LTspice XVII install: the executable name is the build's identity."""

    spice_exe: ClassVar[list[str]] = ["C:\\Program Files\\LTC\\LTspiceXVII\\XVIIx64.exe"]


class Fake26(FakeSim):
    """An LTspice install of 24 or later, which reads decks as LTspice 26 does."""

    spice_exe: ClassVar[list[str]] = ["C:\\Program Files\\ADI\\LTspice\\LTspice.exe"]


#: The sessions, by the LTspice builds each runs.
SESSIONS: dict[str, tuple[dict[str, type], dict[str, type], frozenset[str]]] = {
    "xvii": ({"ltspice": FakeXVII}, {}, frozenset({"LTspice XVII"})),
    "26": ({"ltspice": Fake26}, {}, frozenset({"LTspice 26"})),
    "both": (
        {"ltspice": FakeXVII},
        {"ltspice:26": Fake26},
        frozenset({"LTspice XVII", "LTspice 26"}),
    ),
    "ngspice": ({"ngspice": FakeSim}, {}, frozenset()),
}


def _session(config: ServerConfig, name: str) -> SessionState:
    available, named, _builds = SESSIONS[name]
    return SessionState.create(config, available=available, named=named)


def _source(case_id: str) -> Path:
    return INPUTS / rec.CASES.case(case_id).source


def _expected(case_id: str, scratch: Path) -> dict[str, float]:
    """The builds whose recorded number is not the server's, with that number."""
    taken = {
        BUILD_NAMES[build]: rec.operating_point(build, case_id, scratch)["v(a)"]
        for build in rec.BUILDS
        if case_id in rec.manifest(build)["cases"]
    }
    return {
        build: number
        for build, number in taken.items()
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


async def _load_hierarchy(state: SessionState, deck: Path, _ascii: Path) -> list[dict[str, Any]]:
    hierarchy = load_hierarchy(str(deck), None, SemanticProfile("ltspice"))
    return summaries(hierarchy.readings, session_ltspice_builds(state))


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
        assert "values_read_otherwise_not_checked" not in comparison, comparison
        return comparison.get("values_read_otherwise", [])

    return route


def _observed(reply: dict[str, Any]) -> list[dict[str, Any]]:
    """A run's summaries, from its observations (the evidence is the summary)."""
    return [
        {"code": item["code"], **item["evidence"]}
        for item in reply.get("observations", [])
        if item["code"] == CODE
    ]


def _run(variations: list[dict[str, Any]], *, status: bool) -> Route:
    async def route(state: SessionState, deck: Path, _ascii: Path) -> list[dict[str, Any]]:
        reply = await dispatched(
            state,
            "run_experiments",
            request_id=f"run-{deck.parent.name}-{deck.stem}-{len(variations)}-{status}",
            circuits=[{"path": str(deck), "id": "dut"}],
            lint="off",
            execution={"wait_s": 60},
            variations=variations,
        )
        if status:
            reply = await dispatched(state, "jobs", action="status", job_id=reply["job_id"])
        return _observed(reply)

    return route


#: A Monte Carlo draw around the value's nominal, which the server reads from
#: the stored deck.
_MONTE_CARLO = [
    {
        "kind": "random",
        "runs": 2,
        "seed": 7,
        "rules": [{"rule": "component", "target": "R1", "tolerance": 0.1}],
    }
]

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
    "run_experiments": _run([], status=False),
    "run_experiments Monte Carlo nominal": _run(_MONTE_CARLO, status=False),
    "jobs status of a Monte Carlo run": _run(_MONTE_CARLO, status=True),
}


@pytest.mark.parametrize("session", sorted(SESSIONS))
@pytest.mark.parametrize("route", sorted(ROUTES))
@pytest.mark.parametrize("case_id", DECKS)
async def test_every_route_states_each_build_the_session_runs_that_reads_the_value_otherwise(
    case_id: str,
    route: str,
    session: str,
    tmp_path: Path,
    work_dir: Path,
    config: ServerConfig,
    monkeypatch: pytest.MonkeyPatch,
):
    """A session that runs LTspice XVII is told the number XVII takes for a
    micro sign stored as UTF-8; a session that runs LTspice 26 or ngspice
    alone is told nothing, since no build it runs reads the value otherwise."""
    state = _session(config, session)
    fake_simulator(monkeypatch)
    expected = {
        build: number
        for build, number in _expected(case_id, tmp_path).items()
        if build in SESSIONS[session][2]
    }
    deck = work_dir / _source(case_id).name
    deck.write_bytes(_source(case_id).read_bytes())
    ascii_deck = work_dir / "ascii.cir"
    ascii_deck.write_text(ASCII_DECK, encoding="ascii")

    stated = await ROUTES[route](state, deck, ascii_deck)
    # A run names the file it staged from, resolved.
    names = {str(deck), str(await asyncio.to_thread(deck.resolve))}

    assert all(summary["code"] == CODE for summary in stated), stated
    by_build = {summary["build"]: summary for summary in stated}
    assert len(by_build) == len(stated), stated
    assert set(by_build) == set(expected), (case_id, route, session, stated)
    for build, number in expected.items():
        summary = by_build[build]
        assert summary["file"] in names, summary
        assert summary["reading"] == "established" and summary["count"] == 1, summary
        (example,) = summary["examples"]
        assert math.isclose(example["build_number"], number, rel_tol=SAME_VALUE_RTOL)
        assert math.isclose(example["server_number"], SERVER_NUMBER, rel_tol=SAME_VALUE_RTOL)
        assert example["value"] in ("1µ", "1μ")
        assert example["line"] == 3
        assert summary["listed_by"]["tool"] == "verify_circuit"


def test_the_session_runs_the_builds_its_ltspice_executables_stand_for(config: ServerConfig):
    """The one answer to which builds a session runs: each LTspice executable,
    the detected one and every named one, read as XVII or as 26."""
    for name, (_available, _named, builds) in SESSIONS.items():
        assert session_ltspice_builds(_session(config, name)) == builds, name


# The size of a statement ----------------------------------------------------------


def _many(count: int) -> bytes:
    return "".join(f"C{n} n{n} 0 1µ\n" for n in range(1, count + 1)).encode()


@pytest.mark.parametrize("where", ["inline", "included"])
@pytest.mark.parametrize("route", sorted(ROUTES))
async def test_a_statement_does_not_grow_with_the_number_of_values(
    route: str,
    where: str,
    work_dir: Path,
    config: ServerConfig,
    monkeypatch: pytest.MonkeyPatch,
):
    """The same deck with 3 and with 300 micro-sign values: the statement is
    one summary per file and build either way, naming the same examples, and
    only its count differs."""
    if route.startswith(("run_", "jobs")) and "Monte Carlo" in route:
        pytest.skip("the Monte Carlo draw targets R1, which these decks do not hold")
    if "structural_diff" in route and where == "included":
        pytest.skip("a structural diff reads no included file, and its not_compared says so")
    state = _session(config, "xvii")
    fake_simulator(monkeypatch)
    ascii_deck = work_dir / "ascii.cir"
    ascii_deck.write_text(ASCII_DECK, encoding="ascii")
    sizes: dict[int, list[dict[str, Any]]] = {}
    for count in (3, 300):
        folder = work_dir / f"n{count}"
        folder.mkdir()
        deck = folder / "deck.cir"
        if where == "inline":
            deck.write_bytes(b"* many\n" + _many(count) + b".op\n.end\n")
        else:
            (folder / "parts.inc").write_bytes(_many(count))
            deck.write_bytes(b"* many\n.include parts.inc\n.op\n.end\n")
        sizes[count] = await ROUTES[route](state, deck, ascii_deck)
    small, large = sizes[3], sizes[300]
    assert [s["count"] for s in small] == [3] and [s["count"] for s in large] == [300]
    assert len(large[0]["examples"]) == EXAMPLES

    def shape(stated: list[dict[str, Any]]) -> str:
        return json.dumps(
            [
                {k: v for k, v in s.items() if k not in ("count", "file", "detail", "listed_by")}
                for s in stated
            ],
            sort_keys=True,
            ensure_ascii=False,
        )

    assert shape(large) == shape(small)


# Sheets: the sheet's own bytes ----------------------------------------------------


def _sheets() -> list[Any]:
    """Every recorded sheet, with each build's recorded export of it."""
    found = []
    for build in rec.BUILDS:
        for net in sorted((rec.recorded(build, "export")).glob("*.net")):
            sheet = INPUTS / "export" / f"{net.stem}.asc"
            if sheet.exists():
                found.append(pytest.param(build, net.stem, id=f"{build}/{net.stem}"))
    return found


def _token_as_read(export_text: str) -> dict[str, str]:
    """Each micro-sign or mis-decoded value of an export, by its element."""
    values = {}
    for card in lex_deck(export_text).cards:
        if card.kind == "instance" and card.name and card.body.split()[-1:]:
            values[card.name] = card.body.split()[-1]
    return values


@pytest.mark.parametrize(("exporter", "name"), _sheets())
def test_a_sheet_is_stated_as_each_build_reads_its_own_bytes(exporter: str, name: str):
    """A sheet read as the server's export of it is stated from the sheet's
    own bytes: a value counts where the server read the export's value as
    micro and a build's export of the same sheet shows that build reading
    other text there. A sheet with no micro sign, or one every build reads
    alike, is stated nothing."""
    sheet = INPUTS / "export" / f"{name}.asc"
    export = rec.recorded(exporter, f"export/{name}.net")
    export_text, _codec = read_spice_text_with_encoding(export)
    cards = lex_deck(export_text).cards
    stated = readings_of_sheet(cards, sheet)
    assert stated is not None

    server_micro = {
        site.card.name for site in value_suffix_sites(cards) if site.micro and site.card.name
    }
    # What each build read from the sheet: its own export, as it reads its
    # export (LTspice 26's is UTF-8, XVII's is 8-bit text).
    read_by = {}
    for build in rec.BUILDS:
        if not rec.has(build, f"export/{name}.net"):
            continue
        raw = rec.recorded(build, f"export/{name}.net").read_bytes()
        text = raw.decode("utf-8") if build == "ltspice26" else raw.decode("cp1252")
        read_by[BUILD_NAMES[build]] = _token_as_read(text.replace("\r\n", "\n"))
    sheet_text, _ = read_spice_text_with_encoding(sheet)
    expected = {
        (element, build)
        for element in server_micro
        for build, values in read_by.items()
        if values.get(element) is not None
        and values[element] != _token_as_read(export_text)[element]
        and _token_as_read(export_text)[element] in sheet_text
    }
    found = {
        (next(s.card.name for s in value_suffix_sites(cards) if s.token == r.value), r.build)
        for r in stated
    }
    assert found == expected, (exporter, name, stated)
    for reading in stated:
        assert reading.server_number is not None
        assert math.isclose(
            reading.server_number,
            parse_spice_value(fold_micro_sign(reading.value)),
            rel_tol=SAME_VALUE_RTOL,
        )


def _sheet_with_export(folder: Path, name: str, exporter: str) -> tuple[Path, Path]:
    folder.mkdir(parents=True, exist_ok=True)
    sheet = folder / f"{name}.asc"
    sheet.write_bytes((INPUTS / "export" / f"{name}.asc").read_bytes())
    export = folder / f"{name}.net"
    export.write_bytes(rec.recorded(exporter, f"export/{name}.net").read_bytes())
    return sheet, export


#: The sheets a build's export makes the server read as micro, and the
#: sheets it does not: the UTF-8 sheet LTspice XVII exported, against the
#: same sheet exported by LTspice 26 and a cp1252 sheet XVII exported.
SHEET_CASES = {
    ("ltspice17", "micro_utf8"): {"LTspice XVII", "LTspice 26"},
    ("ltspice26", "micro_utf8"): set(),
    ("ltspice17", "micro_cp1252"): set(),
    ("ltspice26", "micro_cp1252"): set(),
}


@pytest.mark.parametrize(("exporter", "name"), sorted(SHEET_CASES))
async def test_every_route_that_reads_a_sheet_as_its_export_states_the_sheet(
    exporter: str, name: str, work_dir: Path, config: ServerConfig
):
    """A run, a comparison with the sheet on either side, and edit_schematic's
    comparison of the export's text: each states the sheet's own readings, for
    the builds the session runs, and never the export's."""
    from ltspice_mcp.lib.deck_staging import stage_deck
    from ltspice_mcp.tools.verify import VerifyCompareSpec, compare_netlists

    state = _session(config, "both")
    expected = SHEET_CASES[(exporter, name)]
    sheet, export = _sheet_with_export(work_dir / "sheet", name, exporter)
    reference = work_dir / "reference.cir"
    reference.write_text(ASCII_DECK, encoding="ascii")
    spec = VerifyCompareSpec(reference=str(reference))

    answers: dict[str, list[dict[str, Any]]] = {}
    staged = await asyncio.to_thread(
        stage_deck,
        export,
        work_dir / "stage",
        None,
        origin=sheet,
        dialect=None,
        stated_builds=session_ltspice_builds(state),
    )
    answers["run"] = _observed({"observations": staged.observations})
    for label, args in {
        "candidate": (reference, export, reference, sheet),
        "reference": (export, reference, sheet, reference),
        "edit_schematic text": (
            reference,
            read_spice_text_with_encoding(export)[0],
            reference,
            sheet,
        ),
    }.items():
        ref, cand, ref_source, cand_source = args
        comparison, _findings, failure, _warnings = await asyncio.to_thread(
            lambda ref=ref, cand=cand, ref_source=ref_source, cand_source=cand_source: (
                compare_netlists(
                    spec, ref, cand, ref_source, cand_source, state, ref_base_dir=work_dir
                )
            )
        )
        assert failure is None and comparison is not None
        assert "values_read_otherwise_not_checked" not in comparison or not comparison[
            "values_read_otherwise_not_checked"
        ]
        answers[label] = comparison.get("values_read_otherwise", [])

    for label, stated in answers.items():
        assert {s["build"] for s in stated} == expected, (label, stated)
        assert all(s["file"] == str(sheet) for s in stated), (label, stated)
        assert all(s["listed_by"]["tool"] == "inspect" for s in stated)


@pytest.mark.parametrize("exporter", rec.BUILDS)
async def test_a_sheet_with_no_micro_sign_is_stated_nothing_on_any_route(
    exporter: str, work_dir: Path, config: ServerConfig
):
    """Every recorded sheet whose own text holds no micro sign: LTspice 26
    exports its 'u' as a micro sign, which the server reads as micro, but
    the sheet says 'u', which every build reads alike."""
    from ltspice_mcp.lib.deck_staging import stage_deck

    state = _session(config, "both")
    builds = session_ltspice_builds(state)
    checked = 0
    for net in sorted(rec.recorded(exporter, "export").glob("*.net")):
        source = INPUTS / "export" / f"{net.stem}.asc"
        if not source.exists():
            continue
        text, _ = read_spice_text_with_encoding(source)
        if "µ" in text or "μ" in text:
            continue
        sheet, export = _sheet_with_export(work_dir / net.stem, net.stem, exporter)
        try:
            staged = await asyncio.to_thread(
                stage_deck,
                export,
                work_dir / net.stem / "stage",
                None,
                origin=sheet,
                dialect=None,
                stated_builds=builds,
            )
        except Exception:  # a recorded export that does not stage alone
            continue
        assert not _observed({"observations": staged.observations}), net.stem
        export_cards = lex_deck(read_spice_text_with_encoding(export)[0]).cards
        assert readings_of_sheet(export_cards, sheet) == (), net.stem
        checked += 1
    assert checked >= 20


# What no recording shows is said not to be established ---------------------------


def test_a_reading_no_recording_shows_is_stated_as_not_established():
    """A micro sign in braces or in a directive, read by XVII as letters,
    takes a number no recording shows; a Greek mu stored as UTF-16 is
    recorded on no build. Each is stated, with established false, never
    assumed alike or otherwise."""
    braces = readings_of_bytes(
        "* t\nR1 a 0 {1µ}\n.param p=1µ\n.end\n".encode(), root=True, file="d.cir"
    )
    assert [(r.build, r.line, r.build_reads, r.established) for r in braces] == [
        ("LTspice XVII", 2, "1Âµ", False),
        ("LTspice XVII", 3, "1Âµ", False),
    ]
    utf16 = readings_of_bytes("* t\nR1 a 0 1μ\n.end\n".encode("utf-16"), root=True, file="u.cir")
    assert [(r.build, r.build_reads, r.established) for r in utf16] == [
        ("LTspice 26", None, False),
        ("LTspice XVII", None, False),
    ]
    stated = summaries((*braces, *utf16), {"LTspice 26", "LTspice XVII"})
    assert [(s["file"], s["build"], s["reading"], s["count"]) for s in stated] == [
        ("d.cir", "LTspice XVII", "not_established", 2),
        ("u.cir", "LTspice 26", "not_established", 1),
        ("u.cir", "LTspice XVII", "not_established", 1),
    ]


def test_a_sign_outside_an_instances_own_value_is_not_established():
    """The bare number is recorded for a two-terminal part's own value; a
    parameter, a source function's argument or a transistor's area is a
    position no recording shows."""
    deck = "* t\nC1 a 0 1µ Rser=1µ\nV1 a 0 SIN(0 1 1µ)\nQ1 c b 0 QN 1µ\n.end\n".encode()
    assert [(r.line, r.offset, r.build_number) for r in readings_of_bytes(deck, root=True)] == [
        (2, 8, 1.0),
        (2, 16, None),
        (3, 16, None),
        (4, 13, None),
    ]


async def test_two_equal_values_on_one_line_each_carry_their_own_reading(
    work_dir: Path, config: ServerConfig
):
    """verify_circuit's finding for each of two equal values on one line
    carries that value's reading: the capacitor's own value is established,
    its Rser is not."""
    state = _session(config, "xvii")
    deck = work_dir / "deck.cir"
    deck.write_bytes("* t\nC1 a 0 1µ Rser=1µ\n.op\n.end\n".encode())

    data = await dispatched(state, "verify_circuit", path=str(deck), checks=["syntax"])

    read = [
        (finding["evidence"]["read_otherwise"]["reading"], finding["evidence"]["card"])
        for finding in data.get("findings", [])
        if finding["rule_id"] == "value_suffix_micro_sign"
    ]
    assert read == [
        ("established", "C1 a 0 1µ Rser=1µ"),
        ("not_established", "C1 a 0 1µ Rser=1µ"),
    ]


def test_a_comparison_states_only_the_sections_of_a_library_the_deck_reads(work_dir: Path):
    """The comparison's graph and the hierarchy hold the same readings: those
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
