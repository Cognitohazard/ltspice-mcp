"""Reading the committed LTspice recordings the way the server reads live output.

Shared by the tests that hold the server's model against
``tests/fixtures/ltspice_recorded``. Everything here goes through the code a
real run goes through: the contained raw/log decoder (run in process), the
schematic editor, the netlist lexer. Nothing is substituted for them.
"""

from __future__ import annotations

import functools
import json
import os
import shutil
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ltspice_mcp.lib.decoded_log import DecodedLog
from ltspice_mcp.lib.decoded_raw import DecodedRaw
from ltspice_mcp.lib.encoding import read_spice_text
from ltspice_mcp.lib.log_types import LogLimits
from ltspice_mcp.lib.netlist_graph import canon_ref
from ltspice_mcp.lib.parsed_artifacts import ParsedArtifacts
from ltspice_mcp.lib.parser_protocol import read_parsed_artifacts
from ltspice_mcp.lib.parser_worker import parse_request
from ltspice_mcp.lib.raw_header import RawLimits
from ltspice_mcp.lib.spice_lex import SpiceCard, cards_from_path
from ltspice_mcp.lib.spice_lex_views import InstanceLine, read_instance
from tests.ltspice_recorder import (
    FIXTURES,
    INPUTS,
    Build,
    discover_builds,
    load_cases,
    load_manifest,
    recorded_builds,
    unavailable_reason,
)

BUILDS = recorded_builds()
CASES = load_cases()

RAW_LIMITS = RawLimits(4_000_000, 128_000, 32_000, 16, 512, 100_000, 4_000_000)
LOG_LIMITS = LogLimits(1024**2, 65536, 20000, 50000, 2 * 1024**2)


@functools.cache
def manifest(build: str) -> dict[str, Any]:
    return load_manifest(FIXTURES / build)


def generation(build: str) -> str:
    """``current`` or ``xvii``: the server's own split of LTspice builds."""
    return manifest(build)["generation"]


def recorded(build: str, name: str) -> Path:
    """One recorded file: ``recorded("ltspice26", "raw/tran.log")``."""
    return FIXTURES / build / name


def has(build: str, name: str) -> bool:
    return name in {
        output for entry in manifest(build)["cases"].values() for output in entry["outputs"]
    }


def entry(build: str, case_id: str) -> dict[str, Any]:
    return manifest(build)["cases"][case_id]


def installed_counterpart(label: str) -> Build | str:
    """The installed build to hold against the recording ``label``, or why there is none.

    The same major version when it is installed. Otherwise a newer build of
    the same generation stands in for the newest recording of that generation,
    which is how a new release gets compared with the last one recorded.
    """
    if os.environ.get("LTSPICE_MCP_RUN_LTSPICE_INTEGRATION") != "1":
        return "LTspice integration tests are opt-in; set LTSPICE_MCP_RUN_LTSPICE_INTEGRATION=1"
    installed = discover_builds()
    for build in installed:
        if build.label == label:
            return unavailable_reason(build) or build
    newest = max(
        (other for other in BUILDS if generation(other) == generation(label)),
        key=lambda other: int(other.removeprefix("ltspice")),
    )
    if label == newest:
        for build in installed:
            newer = int(build.label.removeprefix("ltspice")) > int(label.removeprefix("ltspice"))
            if build.generation == generation(label) and newer:
                return unavailable_reason(build) or build
    return f"no LTspice build to compare with the {label} recording is installed here"


def cases_of(behaviour: str, *, kind: str | None = None) -> list[str]:
    return [case.case_id for case in CASES.of(behaviour) if kind is None or case.kind == kind]


def per_build(case_ids: list[str]) -> Iterator[tuple[str, str]]:
    """Every (build, case) pair, for ``pytest.mark.parametrize``."""
    for build in BUILDS:
        for case_id in case_ids:
            if case_id in manifest(build)["cases"]:
                yield build, case_id


# --------------------------------------------------------------------------
# Results: the production decode, in process
# --------------------------------------------------------------------------


def parser_request(raw: Path | None, log: Path | None) -> dict[str, Any]:
    """The request the server sends its parser process for these sources."""
    return {
        "version": 1,
        "op": "load_raw" if raw is not None else "load_logs",
        "sources": {
            "raw": str(raw.resolve()) if raw is not None else None,
            "log": str(log.resolve()) if log is not None else None,
            "console": None,
        },
        "dialect": None,
        "producing_dialect": "ltspice",
        "limits": {
            "raw": asdict(RAW_LIMITS),
            "log": asdict(LOG_LIMITS),
            "step_rows": 1000,
            "metadata_bytes": 2 * 1024**2,
        },
        "existing_cache_keys": [],
    }


@dataclass(frozen=True)
class DecodedRun:
    """A recorded run as the server holds one: its raw and what its log says."""

    raw: DecodedRaw
    logs: DecodedLog


def _decode(build: str, case_id: str, scratch: Path, *, raw: bool) -> ParsedArtifacts:
    raw_path = recorded(build, f"{case_id}.raw") if raw else None
    log_path = recorded(build, f"{case_id}.log")
    request = parser_request(raw_path, log_path if log_path.is_file() else None)
    directory = scratch / f"decode-{build}-{case_id.replace('/', '-')}"
    directory.mkdir(parents=True)
    parse_request(request, directory)
    metadata = json.loads((directory / "result.json").read_bytes())
    return read_parsed_artifacts(
        metadata, directory, limits=RAW_LIMITS, require_raw=raw, request=request
    )


def decode(build: str, case_id: str, scratch: Path) -> DecodedRun:
    """Decode a recorded run with the decoder the server runs, raising what it raises."""
    parsed = _decode(build, case_id, scratch, raw=True)
    assert parsed.raw is not None
    return DecodedRun(raw=parsed.raw, logs=parsed.logs)


def decode_log(build: str, case_id: str, scratch: Path) -> DecodedLog:
    """Decode the log of a recorded run that left no raw worth reading."""
    return _decode(build, case_id, scratch, raw=False).logs


def operating_point(build: str, case_id: str, scratch: Path) -> dict[str, float]:
    """Every trace of a recorded ``.op`` run, by lower-cased name."""
    raw = decode(build, case_id, scratch).raw
    return {name.lower(): float(raw.get_wave(name, 0)[0]) for name in raw.get_trace_names()}


# --------------------------------------------------------------------------
# Decks and exports: the foundation lexer
# --------------------------------------------------------------------------


def deck_cards(name: str) -> list[SpiceCard]:
    """The cards of an input deck."""
    return cards_from_path(INPUTS / name).cards


def export_text(build: str, case_id: str) -> str:
    return read_spice_text(recorded(build, f"{case_id}.net"))


def export_cards(build: str, case_id: str) -> list[SpiceCard]:
    return cards_from_path(recorded(build, f"{case_id}.net")).cards


def export_instances(build: str, case_id: str) -> dict[str, InstanceLine]:
    """Every component card of a recorded export, by canonical reference."""
    found: dict[str, InstanceLine] = {}
    for card in export_cards(build, case_id):
        if card.kind == "instance" and card.name:
            line = read_instance(card)
            assert line is not None, card.body
            found[canon_ref(card.name)] = line
    return found


# --------------------------------------------------------------------------
# Sheets: the schematic editor, with the symbols LTspice used
# --------------------------------------------------------------------------


def _stock_symbol(name: str, facts: dict[str, Any]) -> str:
    """A symbol file carrying the pins the build's own library gives ``name``."""
    lines = ["Version 4", "SymbolType CELL", f"SYMATTR Prefix {facts['prefix']}"]
    for pin in facts["pins"]:
        lines += [
            f"PIN {pin['x']} {pin['y']} NONE 0",
            f"PINATTR PinName {pin['name']}",
            f"PINATTR SpiceOrder {pin['order']}",
        ]
    return "\n".join(lines) + "\n"


def stage_sheet(build: str, case_id: str, directory: Path) -> Path:
    """Copy a recorded sheet into ``directory`` with the symbols LTspice resolved.

    A symbol the recorder placed beside the sheet is copied beside it here. A
    symbol that came from the build's library is written from the pins the
    manifest recorded for it, so the server places the sheet's pins from the
    same symbol LTspice did.
    """
    case = CASES.case(case_id)
    directory.mkdir(parents=True, exist_ok=True)
    sheet = directory / Path(case.source).name
    shutil.copyfile(INPUTS / case.source, sheet)
    for extra in case.extra:
        beside = directory / case.copies[extra]
        beside.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(INPUTS / extra, beside)
    stock = manifest(build)["library"]["symbols"]
    for name, facts in stock.items():
        target = directory / f"{name}.asy"
        if not target.exists():
            target.write_text(_stock_symbol(name, facts), encoding="utf-8", newline="\n")
    return sheet
