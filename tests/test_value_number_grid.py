"""What number a value is, on every route that answers it.

The question is the value-number grid's (``[grid.value-number]`` in
``inputs/cases.toml``; ``docs/TESTING.md``, "Grids"). LTspice reads a value
as a mantissa, a scale suffix and whatever follows; the server answers with
``format.parse_spice_value`` and, where a caller first asks whether a word is
a bare number, ``format.is_scaled_number``. Inside braces an expression's
number tokens are ``hierarchy_expr._NUMBER``'s. A few readers answer the same
question on their own: the plain reader a variation's assignment is held to
(``parse_plain_value``), the plain-number test (``plain_number``), the lint's
reading of a capital M (``lint_rules._MILLI_SUFFIX_RE``) and a frequency's
reader (``metrics.parse_freq``).

A cell is one value as a deck writes it, on one source or resistor whose node
voltage in the ``.op`` raw is the number LTspice read. Every recorded cell, on
each build, is read through every route that turns a value's text into a
number: the value readers above, the Monte Carlo nominal
(``montecarlo.parse_value``), the netlist comparison (``values_equal``), a
``.step`` value (``step_values.step_number``), a time and a frequency an
``analyze_results`` recipe takes (``metrics.parse_time``,
``metrics.parse_freq``), a window bound of the Python API, the leading numbers
of a directive (``spice_validator.leading_numeric_tokens``), the evaluator
(``hierarchy_expr.evaluate``), and ``inspect(kind="hierarchy")`` through the
server's dispatch and the Python API. A route either takes LTspice's number
(within ``SAME_VALUE_RTOL``: a raw stores a node voltage in four bytes, which
round it by less than a part in ten million) or declines the value; a route
that takes another number, or takes one where LTspice refused the deck, is a
difference, declared on its cell (``differs``) and counted in
``KNOWN_DIFFERENCES``. A build that reads other text than the server from
the stored file (a micro sign stored as UTF-8, which LTspice XVII decodes as
cp1252) is declared on its cell (``decoded``, with the text it reads): that
difference is the decoding's, so no route that reads the server's text is
held to the build's number, and every route that starts from the stored file
states it, with that number (``stored_values``); a route states nothing for a
build no cell declares so. The same forms are read live by ngspice in each
compatibility mode and compared with the server's reading for ngspice.
"""

from __future__ import annotations

import asyncio
import math
import re
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any, ClassVar

import pytest

from ltspice_mcp.config import ServerConfig
from ltspice_mcp.errors import LTSpiceMCPError
from ltspice_mcp.lib import montecarlo
from ltspice_mcp.lib.encoding import read_spice_text, read_spice_text_with_encoding, rewrite_codec
from ltspice_mcp.lib.format import (
    SAME_VALUE_RTOL,
    is_scaled_number,
    parse_plain_value,
    parse_spice_value,
    plain_number,
)
from ltspice_mcp.lib.hierarchy import SemanticProfile, load_hierarchy
from ltspice_mcp.lib.hierarchy_expr import evaluate
from ltspice_mcp.lib.lint_rules import lint_deck
from ltspice_mcp.lib.metrics import parse_freq, parse_time
from ltspice_mcp.lib.netlist_graph import values_equal
from ltspice_mcp.lib.ngspice_mode import run_batch
from ltspice_mcp.lib.services import session_ltspice_builds
from ltspice_mcp.lib.spice_lex import lex_deck
from ltspice_mcp.lib.spice_lex_views import read_instance
from ltspice_mcp.lib.spice_validator import leading_numeric_tokens
from ltspice_mcp.lib.step_values import step_number
from ltspice_mcp.lib.stored_values import summaries
from ltspice_mcp.state import SessionState
from tests import _ltspice_recorded as rec
from tests._recorded_grids import Cell, grid
from tests.conftest import LIVENESS_S, FakeSim, SyncApi, dispatched
from tests.ltspice_recorder import INPUTS

GRID = grid("value-number")

#: How many routes answer otherwise than a recorded cell, per build, each
#: declared on its cell (``differs``, with the route and the reason). A count
#: may only fall: a fix lowers it here in the same change.
KNOWN_DIFFERENCES: dict[str, int] = {"ltspice26": 3, "ltspice17": 4}


def _recorded(build: str, case_id: str) -> bool:
    return case_id in rec.manifest(build)["cases"]


def _source(cell: Cell) -> Path:
    return INPUTS / rec.CASES.case(cell.case).source


def _cases() -> list[str]:
    return sorted({cell.case for cell in GRID.cells})


def _runs() -> list[Any]:
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


def _number(cell: Cell, build: str) -> float | None:
    """The number a cell declares for ``build``, None where it declares none."""
    number = cell.extra.get("number")
    if isinstance(number, dict):
        return number.get(build)
    return number


# The index, held to the decks -------------------------------------------------

#: The scale suffixes, longest first, as a deck writes them in lower case.
_SUFFIX_WORDS = ("meg", "mil", "t", "g", "k", "m", "u", "n", "p", "f", "r")
_WHOLE_TEXTS = {"1_000": "underscore", "inf": "inf", "nan": "nan", "1e999": "overflow"}
_SIGNS = {"µ": "micro_sign", "μ": "greek_mu", "Ω": "ohm_sign"}


def _mantissa(body: str) -> tuple[str, str]:
    """How a value's mantissa is written, and the text after it."""
    i, sign = 0, ""
    if body[:1] in "+-":
        sign, i = body[0], 1
    if not body[i].isascii() and body[i].isdigit():
        return "nonascii_digits", body[i + 1 :]
    start = i
    while i < len(body) and body[i] in "0123456789":
        i += 1
    lead, dot, frac, exponent = i > start, False, False, ""
    if body[i : i + 1] == ".":
        dot, i = True, i + 1
        digits_from = i
        while i < len(body) and body[i] in "0123456789":
            i += 1
        frac = i > digits_from
    if body[i : i + 1] in ("e", "E"):
        j = i + 1 + (body[i + 1 : i + 2] in ("+", "-"))
        if body[j : j + 1].isdigit():
            signed = j > i + 1
            exponent = (
                "exponent_signed"
                if signed
                else ("exponent_upper" if body[i] == "E" else "exponent")
            )
            while j < len(body) and body[j] in "0123456789":
                j += 1
            i = j
    if sign:
        kind = "plus" if sign == "+" else "minus"
    elif exponent:
        kind = exponent
    elif dot:
        kind = "decimal" if frac and lead else "leading_dot" if frac else "trailing_dot"
    else:
        kind = "whole"
    return kind, body[i:]


def _form(text: str, encoding: str) -> dict[str, str]:
    """A value's coordinates, read from how it is written."""
    at: dict[str, str] = {}
    body = text
    if body.startswith("{") and body.endswith("}"):
        at["where"], body = "braces", body[1:-1]
    if body in _WHOLE_TEXTS:
        return {**at, "mantissa": _WHOLE_TEXTS[body], "after": "nothing"}
    at["mantissa"], rest = _mantissa(body)
    if rest.startswith("%"):
        return {**at, "symbol": "percent"}
    suffix = next((word for word in _SUFFIX_WORDS if rest.lower().startswith(word)), "none")
    if rest[:1] in ("µ", "μ"):
        suffix = _SIGNS[rest[0]]
    width = {"none": 0, "micro_sign": 1, "greek_mu": 1}.get(suffix, len(suffix))
    written, after = rest[:width], rest[width:]
    at["suffix"] = suffix
    at["case"] = (
        "lower"
        if written == written.lower()
        else "upper"
        if written == written.upper()
        else "mixed"
    )
    if not after:
        at["tail"] = "none"
        if suffix == "none":
            at["after"] = "nothing"
        elif suffix != "r":
            at["after"] = "suffix"
    elif after.isdigit():
        at["after"] = "r_digits" if suffix == "r" else "suffix_digits"
    elif after.startswith("Ω"):
        at["tail"] = "ohm_sign"
    elif suffix == "none" and after.rstrip("0123456789") != after:
        del at["suffix"], at["case"]
        at["after"] = "letter_digits"
    else:
        at["tail"] = "letters"
    for char, sign in _SIGNS.items():
        if char in text:
            at["sign"], at["encoding"] = sign, encoding
    return at


def _encoding(case_id: str) -> str:
    """The encoding a deck is stored in, from its bytes; an 8-bit deck is in
    the code page its name says (cp1253), else cp1252."""
    data = _source_of(case_id).read_bytes()
    if data.startswith(b"\xff\xfe"):
        return "utf16le_bom"
    if b"\x00" in data:
        return "utf16le"
    if data.startswith(b"\xef\xbb\xbf"):
        return "utf8_bom"
    try:
        data.decode("utf-8")
    except UnicodeDecodeError:
        return "cp1253" if case_id.endswith("cp1253") else "cp1252"
    return "utf8"


def _source_of(case_id: str) -> Path:
    return INPUTS / rec.CASES.case(case_id).source


def _values_by_node(case_id: str, *, as_written: bool = True) -> dict[str, str]:
    """Each probe node of a deck and the value written on its source or
    resistor: as written, in the code page an 8-bit deck is named for, or as
    the server decodes the deck (``read_spice_text``, which names every 8-bit
    deck cp1252)."""
    path = _source_of(case_id)
    if as_written and _encoding(case_id) == "cp1253":
        text = path.read_bytes().decode("cp1253")
    else:
        text = read_spice_text(path)
    values: dict[str, str] = {}
    for card in lex_deck(text).cards:
        line = read_instance(card) if card.kind == "instance" else None
        if line is not None and line.ref[:1].upper() in ("R", "V"):
            values[line.nodes[0].lower()] = line.value or ""
    return values


@pytest.mark.parametrize("cell", GRID.cells, ids=lambda cell: cell.id)
def test_a_cell_is_the_value_its_deck_writes_on_its_node_and_has_its_form(cell: Cell):
    values = _values_by_node(cell.case)
    assert values.get(cell.extra["node"]) == cell.item, (cell.id, values.get(cell.extra["node"]))
    declared = {axis: value for axis, value in cell.coordinates.items() if value != ""}
    form = {"where": "value", **_form(cell.item, _encoding(cell.case))}
    assert declared == form, cell.id


def test_every_suffix_the_reader_knows_is_a_value_of_the_suffix_axis():
    """Each scale suffix ``parse_spice_value`` reads, the R it reads as a
    decimal point and each micro sign it folds is a value of the grid's
    suffix axis, so a suffix the reader learns needs cells, recorded on each
    build or counted as not recorded."""
    from ltspice_mcp.lib.format import _SCALE_FACTORS, MICRO_SIGNS

    known = {suffix for suffix, _ in _SCALE_FACTORS} | {"r"}
    known |= {_SIGNS[sign] for sign in MICRO_SIGNS}
    assert known == set(GRID.axes["suffix"].values) - {"none"}
    assert set(_SUFFIX_WORDS) == {suffix for suffix, _ in _SCALE_FACTORS} | {"r"}


def test_every_value_of_a_grid_deck_is_a_cell():
    """Every source and resistor value of a deck of the grid's cases is a cell."""
    indexed = {(cell.case, cell.item) for cell in GRID.cells}
    missing = [
        f"{case_id}:{value}"
        for case_id in _cases()
        for value in _values_by_node(case_id).values()
        if (case_id, value) not in indexed
    ]
    assert not missing, missing


def test_a_recorded_cell_has_no_prediction_left():
    """A prediction (``not_established``) is settled against its recording,
    and the key removed, once the case is recorded."""
    for cell in GRID.cells:
        if "not_established" in cell.extra:
            recorded = [build for build in rec.BUILDS if _recorded(build, cell.case)]
            assert not recorded, (
                f"{cell.id} is recorded on {recorded}: settle its number against the "
                "recording and remove not_established"
            )


def test_the_count_of_known_differences_only_falls():
    for build in rec.BUILDS:
        count = sum(
            len(entry["routes"])
            for cell in GRID.cells
            for entry in cell.extra.get("differs", ())
            if entry["build"] == build
        )
        assert count <= KNOWN_DIFFERENCES[build], (build, count)
        assert count == KNOWN_DIFFERENCES[build], f"lower the pin for {build} to {count}"


# Every route -------------------------------------------------------------------


def _taken(read: Callable[[str], float | None], text: str) -> float | None:
    """The number a route takes from ``text``, None where it declines it."""
    try:
        value = read(text)
    except (ValueError, ArithmeticError, LTSpiceMCPError):
        # ValueError for the value readers and the API's bound (ApiValidationError);
        # ResultError for an analyze_results time or frequency.
        return None
    return None if value is None else float(value)


def _api_window_bound(text: str) -> float | None:
    from ltspice_mcp.api._methods import _window_bound

    return _window_bound(text, "start")


def _no_names(name: str) -> float:
    raise ValueError(f"unknown parameter {name}")


#: Every route that turns a value's text into a number, by the name a
#: ``differs`` entry gives it. The hierarchy's are read per deck below.
TEXT_ROUTES: dict[str, Callable[[str], float | None]] = {
    "parse_spice_value": parse_spice_value,
    "plain_number": plain_number,
    "is_scaled_number": lambda text: parse_spice_value(text) if is_scaled_number(text) else None,
    "parse_plain_value": parse_plain_value,
    "montecarlo.parse_value": montecarlo.parse_value,
    "step_number": step_number,
    "parse_time": lambda text: parse_time(text, "at"),
    "parse_freq": parse_freq,
    "api window bound": _api_window_bound,
    "leading_numeric_tokens": lambda text: next(iter(leading_numeric_tokens([text])), None),
    "evaluate": lambda text: evaluate(
        text, _no_names, simulator="ltspice", ngbehavior=None, element_value=True
    ),
}


def _same(taken: float | None, ltspice: float | None) -> bool:
    if taken is None:
        return True
    if ltspice is None:
        return False
    return math.isclose(taken, ltspice, rel_tol=SAME_VALUE_RTOL)


async def _hierarchy_values(
    state: SessionState, deck: Path
) -> tuple[dict[str, dict[str, float | None]], dict[str, list[dict[str, Any]]]]:
    """Each probe node's value as ``inspect(kind="hierarchy")`` resolves it,
    through the server's dispatch and through the Python API, and what each
    route states of a value a recorded build reads otherwise
    (``stored_values``). These are the routes that start from the stored file."""
    query = {"kind": "hierarchy", "path": str(deck), "simulator": "ltspice"}
    reply = await dispatched(state, "inspect", queries=[query])
    api_reply = await asyncio.to_thread(lambda: SyncApi(state).inspect(queries=[query]))
    found: dict[str, dict[str, float | None]] = {}
    stated: dict[str, list[dict[str, Any]]] = {}
    for route, data in (("inspect", reply), ("api inspect", api_reply)):
        (result,) = data["results"]
        assert result["ok"], result
        stated[route] = result["data"].get("values_read_otherwise", [])
        for row in result["data"]["instances"]:
            if row["instance"][0][:1].upper() in ("R", "V"):
                node = row["nodes"][0]["name"].lower()
                found.setdefault(route, {})[node] = row["value"].get("value")
    in_process = load_hierarchy(str(deck), None, SemanticProfile("ltspice"))
    stated["load_hierarchy"] = summaries(in_process.readings, session_ltspice_builds(state))
    for row in in_process.instances:
        if row.reference[:1].upper() in ("R", "V"):
            found.setdefault("load_hierarchy", {})[row.nodes[0].name.lower()] = row.value.value
    return found, stated


#: The name a statement gives each recorded build.
_BUILD_NAMES = {"ltspice26": "LTspice 26", "ltspice17": "LTspice XVII"}


class _FakeXVII(FakeSim):
    """An LTspice XVII install: the executable name is the build's identity."""

    spice_exe: ClassVar[list[str]] = ["C:\\Program Files\\LTC\\LTspiceXVII\\XVIIx64.exe"]


class _Fake26(FakeSim):
    """An LTspice install of 24 or later, read as LTspice 26."""

    spice_exe: ClassVar[list[str]] = ["C:\\Program Files\\ADI\\LTspice\\LTspice.exe"]


def _decoded(cell: Cell, build: str) -> dict[str, Any] | None:
    """The cell's ``decoded`` entry for ``build``: that build reads other text
    than the server from the stored file."""
    return next((entry for entry in cell.extra.get("decoded", ()) if entry["build"] == build), None)


@pytest.mark.parametrize(("build", "case_id"), _runs())
async def test_every_recorded_cell_reads_as_ltspice_read_it_on_every_route(
    build: str, case_id: str, tmp_path: Path, config: ServerConfig, work_dir: Path
):
    """The declared number is the number LTspice took, and every route takes
    it or declines; a route that takes another is declared on its cell."""
    cells = GRID.cells_of(case_id)
    entry = rec.entry(build, case_id)
    voltages = (
        rec.operating_point(build, case_id, tmp_path)
        if entry["exit_code"] == 0 and rec.has(build, f"{case_id}.raw")
        else None
    )
    deck = work_dir / _source_of(case_id).name
    deck.write_bytes(_source_of(case_id).read_bytes())
    # A session that runs both recorded builds, so each build's reading of the
    # stored file is stated (a reply states only the builds a session runs).
    session = SessionState.create(
        config, available={"ltspice": _FakeXVII}, named={"ltspice:26": _Fake26}
    )
    hierarchy, stated = await _hierarchy_values(session, deck)
    read = _values_by_node(case_id, as_written=False)
    found: set[tuple[str, str]] = set()
    declared: set[tuple[str, str]] = set()
    for cell in cells:
        node = cell.extra["node"]
        # The text the server reads: the value as it decodes the deck.
        text = read[node]
        if voltages is None:
            assert build in cell.refused, f"{cell.id}: {build} took no number"
            ltspice = None
        else:
            assert build not in cell.refused, f"{cell.id}: {build} ran the deck"
            ltspice = voltages[f"v({node})"]
            declared_number = _number(cell, build)
            assert declared_number is not None, f"{cell.id} declares no number on {build}"
            assert math.isclose(ltspice, declared_number, rel_tol=SAME_VALUE_RTOL), (
                f"{cell.id}: LTspice took {ltspice}, the cell declares {declared_number}"
            )
        decoded = _decoded(cell, build)
        # What each route that starts from the stored file states of this value.
        on_build = {
            route: [
                example
                for summary in said
                if summary["build"] == _BUILD_NAMES[build]
                for example in summary["examples"]
                if example["value"] == cell.item
            ]
            for route, said in stated.items()
        }
        if decoded is not None:
            # The build reads other text from the file: no route that reads
            # the server's text is held to its number, and every route that
            # starts from the file states the number it takes.
            assert ltspice is not None, f"{cell.id}: {build} took no number to state"
            for route, said in on_build.items():
                assert len(said) == 1, (cell.id, route, stated[route])
                (statement,) = said
                assert statement["build_reads"] == decoded["reads"], (cell.id, route)
                assert math.isclose(statement["build_number"], ltspice, rel_tol=SAME_VALUE_RTOL)
            continue
        assert not any(on_build.values()), (cell.id, build, on_build)
        answers = {route: _taken(reader, text) for route, reader in TEXT_ROUTES.items()}
        for route, values in hierarchy.items():
            answers[route] = values[node]
        # The comparison reads the value against LTspice's number written out.
        if ltspice is not None and not values_equal(text, repr(ltspice), SAME_VALUE_RTOL):
            found.add((cell.item, "values_equal"))
        for route, taken in answers.items():
            if not _same(taken, ltspice):
                found.add((cell.item, route))
        declared |= {
            (cell.item, route)
            for entry in cell.extra.get("differs", ())
            if entry["build"] == build
            for route in entry["routes"]
        }
    assert found == declared, (
        f"answered otherwise, not declared: {sorted(found - declared)}; "
        f"declared, not found: {sorted(declared - found)}"
    )


# The lint's reading of a capital M ---------------------------------------------


@pytest.mark.parametrize(("build", "case_id"), _runs())
def test_the_mega_or_milli_lint_flags_each_capital_m_ltspice_read_as_milli(
    build: str, case_id: str
):
    """``suffix-mega-milli`` flags a value whose capital M LTspice read as
    milli (the cell's suffix is m, in upper case, and its recorded number
    says so), and no other."""
    path = _source_of(case_id)
    text, encoding = read_spice_text_with_encoding(path)
    findings = lint_deck(text, path, "ltspice", "LTspice", codec=rewrite_codec(encoding))
    flagged = {
        token
        for finding in findings
        if finding["rule_id"] == "suffix-mega-milli"
        for token in finding["evidence"]["tokens"]
    }
    milli = {
        cell.item.strip("{}")
        for cell in GRID.cells_of(case_id)
        if cell.at(build).get("suffix") == "m" and cell.at(build).get("case") == "upper"
    }
    assert flagged == milli


# ngspice, live ----------------------------------------------------------------

needs_ngspice = pytest.mark.skipif(shutil.which("ngspice") is None, reason="ngspice not on PATH")

#: The compatibility modes the server models (``lib/ngspice_compat.py``): no
#: mode, the HSPICE and the LTspice/PSPICE modes spicelib and a user pick, and
#: each flag alone that changes a reading the server models.
NGSPICE_MODES = ("", "hsa", "kiltpsa", "lt", "ps", "spe")

#: Forms where ngspice takes another number than the server's reading for
#: ngspice, with the reason. Not asserted to differ: another ngspice version
#: may agree.
NGSPICE_DIFFERS: dict[str, str] = {
    "2A": (
        "ngspice-42 reads A as atto (2e-18); the server's reading for ngspice has no "
        "atto suffix (format._SCALE_FACTORS) and takes 2, as LTspice does"
    ),
    "1μ": (
        "ngspice-42 reads the Greek mu as no suffix (1); the server folds it to u for "
        "every reader (format.fold_micro_sign) and takes 1e-6"
    ),
    "1μV": (
        "ngspice-42 reads the Greek mu as no suffix (1); the server folds it to u for "
        "every reader (format.fold_micro_sign) and takes 1e-6"
    ),
    "{1μ}": (
        "ngspice-42 reads the Greek mu as no suffix (1); the server folds it to u for "
        "every reader (format.fold_micro_sign) and takes 1e-6"
    ),
    "{1mil}": (
        "inside braces ngspice-42 reads mil as m followed by letters (1e-3); the "
        "server's _NUMBER reads mil (2.54e-5), as ngspice does in a value"
    ),
    "{1MIL}": (
        "inside braces ngspice-42 reads mil as m followed by letters (1e-3); the "
        "server's _NUMBER reads mil (2.54e-5), as ngspice does in a value"
    ),
    "{1Mil}": (
        "inside braces ngspice-42 reads mil as m followed by letters (1e-3); the "
        "server's _NUMBER reads mil (2.54e-5), as ngspice does in a value"
    ),
    "٣": (
        "ngspice-42 stops on a digit that is not ASCII; Python's float and the \\d "
        "of _MANTISSA and _NUMBER read it as 3"
    ),
    "٣k": (
        "ngspice-42 stops on a digit that is not ASCII; Python's float and the \\d "
        "of _MANTISSA and _NUMBER read it as 3"
    ),
    "{٣}": (
        "ngspice-42 stops on a digit that is not ASCII; Python's float and the \\d "
        "of _MANTISSA and _NUMBER read it as 3"
    ),
}

_PRINTED = re.compile(r"^(n\d+)\s*=\s*(\S+)", re.M)


def _ngspice_reads(mode: str, forms: list[str], folder: Path) -> dict[str, float | None]:
    """What ngspice takes from each form, as a source's value: its number, or
    None where it stops on the form. All forms in one deck, and each alone
    where the deck stops."""

    def run(chosen: list[str]) -> dict[str, float]:
        cards = [f"V{i} n{i} 0 {form}" for i, form in enumerate(chosen, 1)]
        deck = "\n".join(
            [
                "* forms",
                *cards,
                ".control",
                "set numdgt=12",
                "op",
                "print all",
                ".endc",
                ".end",
                "",
            ]
        )
        log = run_batch(
            ("ngspice",),
            mode,
            deck.encode("utf-8"),
            cwd=folder,
            clean_startup=True,
            timeout=LIVENESS_S,
        )
        printed = {name: value for name, value in _PRINTED.findall(log)}
        return {
            form: float(printed[f"n{i}"]) for i, form in enumerate(chosen, 1) if f"n{i}" in printed
        }

    together = run(forms)
    if len(together) == len(forms):
        return dict(together)
    return {form: run([form]).get(form) for form in forms}


def _ngspice_server(mode: str, form: str) -> float | None:
    return _taken(
        lambda text: evaluate(
            text, _no_names, simulator="ngspice", ngbehavior=mode, element_value=True
        ),
        form,
    )


@needs_ngspice
@pytest.mark.parametrize("mode", NGSPICE_MODES)
def test_the_server_reads_each_form_for_ngspice_as_ngspice_does(mode: str, tmp_path: Path):
    """Each form of the grid, as a source's value, read by the ngspice on
    PATH in ``mode``: where the server's reading for ngspice takes a number,
    ngspice takes the same one. The hierarchy reads a value for ngspice
    through the evaluator, which this asks; a form that is not ASCII is
    written in UTF-8, as ngspice reads a deck."""
    forms = sorted({cell.item for cell in GRID.cells})
    taken = _ngspice_reads(mode, forms, tmp_path)
    wrong = []
    for form in forms:
        server = _ngspice_server(mode, form)
        if server is None or form in NGSPICE_DIFFERS:
            continue
        if not _same(server, taken[form]):
            wrong.append((form, server, taken[form]))
    assert not wrong, wrong
    assert set(NGSPICE_DIFFERS) <= set(forms)
    assert all(NGSPICE_DIFFERS.values())
