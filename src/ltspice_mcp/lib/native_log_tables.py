"""Bounded native print facts; physical blocks do not imply plot identities."""

from __future__ import annotations

import io
import re
from typing import Any, NoReturn, get_args

from ltspice_mcp.lib.log_types import (
    LogDecodeError,
    LogLimitError,
    LogLimits,
    NativeAnalysisLabel,
    log_section,
)

_ATOM = re.compile(
    r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?|[+-]?(?:nan|inf|infinity)", re.I
)
_ASSIGN = re.compile(r"([^\s=]+)[ \t]*=[ \t]*(.*)")
_CURRENT = re.compile(r"Current[ \t]+(\S+)[ \t]+.*\(([^()]+)\)")
_LISTING = re.compile(r"\S+[ \t]+.*\([^()]+\)")
_RULE = re.compile(r"-{3,}")
_TITLE = re.compile(
    r"(.+?)[ \t]+(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)[ \t]+[A-Z][a-z]{2}[ \t]+[0-9]{1,2}[ \t]+[0-9]{2}:[0-9]{2}:[0-9]{2}[ \t]+[0-9]{4}"
)
_DONE = re.compile(r"ngspice-[0-9]+ done")
_INDEXED_ROW = re.compile(r"([0-9]+)[ \t]+(\S+)[ \t]+(.+)")
_LABELS = get_args(NativeAnalysisLabel)
_SCALAR_LABELS = {"Transfer Function", "Pole-Zero Analysis", "Sensitivity Analysis"}
# The common section budget counts each row container and its scalar fields.
_SCALAR_ENTRY_NODES = 7
_FREQUENCY_ROW_NODES = 6


class NativeTableError(LogDecodeError):
    """A recognized print candidate is unsupported, malformed or unclosed."""


def _number(token: str) -> float:
    if _ATOM.fullmatch(token.strip()) is None:
        raise ValueError("Expected a whole decimal/exponent or nonfinite numeric token")
    return float(token)


def _value(text: str) -> tuple[str, float, float | None]:
    parts = text.split(",")
    if len(parts) == 1:
        return "real", _number(parts[0]), None
    if len(parts) == 2:
        return "complex", _number(parts[0]), _number(parts[1])
    raise ValueError("A complex value must have exactly two components")


def _nodes(value: Any) -> int:
    if isinstance(value, dict):
        return 1 + sum(_nodes(item) for item in value.values())
    if isinstance(value, list):
        return 1 + sum(_nodes(item) for item in value)
    return 1


def _parse(text: str, limits: LogLimits) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    used = 1
    current = None
    listing = False
    title = None
    title_rule = False
    block = None
    needs_rule = False
    last_line = 0
    circuit_title = None
    repeated_title_line = None

    def reserve(count: int) -> None:
        nonlocal used
        if used + count > limits.section_entries:
            raise LogLimitError(
                "native_tables section_entries limit exceeded before row allocation"
            )
        used += count

    def fail(line: int, message: str) -> NoReturn:
        raise NativeTableError(f"Native block {len(blocks)} at line {line}: {message}")

    def base(line: int) -> dict[str, Any]:
        return {
            "ordinal": len(blocks),
            "line_start": line,
            "line_end": line,
            "printed_analysis_label": None,
            "printed_title_line": None,
            "listing_current": dict(current) if current else None,
            "closed_by": "",
            "analysis_extent": "unknown",
        }

    def close(reason: str, line: int) -> None:
        nonlocal block
        if block is None:
            return
        if needs_rule or not block.get("entries", block.get("rows")):
            fail(line, "Recognized print block is empty or missing its header separator")
        block["closed_by"] = reason
        blocks.append(block)
        block = None

    for last_line, physical in enumerate(io.StringIO(text), 1):
        raw = physical.rstrip("\r\n")
        stripped = raw.strip()
        if not stripped and "\f" not in raw:
            continue
        native_title = stripped.startswith(("Sensitivity Analysis", "DISTORTION"))
        if repeated_title_line is not None:
            if not native_title:
                fail(repeated_title_line, "Repeated circuit title has no native print header")
            repeated_title_line = None
        if "\f" in raw:
            if raw.strip(" \t") != "\f":
                fail(last_line, "Embedded form feed cannot delimit a numeric row")
            if title:
                fail(last_line, "Printed title has no table header")
            close("form_feed", last_line)
            continue
        if _DONE.fullmatch(stripped):
            if title:
                fail(last_line, "Printed title has no table header")
            close("ngspice_done", last_line)
            listing = False
            continue
        if native_title:
            match = _TITLE.fullmatch(stripped)
            if match is None or match[1] not in _LABELS:
                fail(last_line, "Unsupported native printed analysis title")
            if title:
                fail(last_line, "Printed title has no table header")
            close("next_print_header", last_line)
            title = (last_line, raw, match[1])
            title_rule = False
            listing = False
            continue
        if title:
            if _RULE.fullmatch(stripped):
                title_rule = True
                continue
            if not stripped.startswith("Index"):
                fail(last_line, "Expected an Index frequency header after printed title")
        if stripped.startswith("Index"):
            header = stripped.split()
            if (
                not title
                or not title_rule
                or len(header) != 3
                or header[:2] != ["Index", "frequency"]
            ):
                fail(
                    last_line, "Unsupported native header; expected Index frequency and one column"
                )
            if block:
                fail(last_line, "A repeated header needs its own printed title")
            block = base(title[0])
            block.update(
                layout="frequency_table",
                printed_analysis_label=title[2],
                printed_title_line=title[1],
                coordinate={"label": "frequency", "unit": None, "convention": "printed"},
                column={"label": header[2], "representation": "real", "unit": None},
                rows=[],
            )
            reserve(_nodes(block))
            title = None
            needs_rule = True
            continue
        if block and block["layout"] == "frequency_table":
            if needs_rule:
                if not _RULE.fullmatch(stripped):
                    fail(last_line, "Missing separator after table header")
                needs_rule = False
                continue
            if circuit_title and stripped == circuit_title:
                repeated_title_line = last_line
                continue
            reserve(_FREQUENCY_ROW_NODES)
            match = _INDEXED_ROW.fullmatch(stripped)
            if match is None:
                fail(last_line, "Expected a complete indexed numeric row")
            try:
                index = int(match[1])
                frequency = _number(match[2])
                representation, real, imag = _value(match[3])
            except ValueError as exc:
                fail(last_line, str(exc))
            if index != len(block["rows"]):
                fail(last_line, "Row indices must begin at zero and advance without gaps")
            if block["rows"] and representation != block["column"]["representation"]:
                fail(last_line, "Real/complex syntax changed within a printed block")
            block["column"]["representation"] = representation
            block["rows"].append(
                {
                    "line": last_line,
                    "index": index,
                    "frequency": frequency,
                    "real": real,
                    "imag": imag,
                }
            )
            block["line_end"] = last_line
            continue
        if stripped.startswith("Circuit:"):
            circuit_title = stripped.removeprefix("Circuit:").strip()
        if stripped == "List of plots available:":
            if block:
                fail(last_line, "Scalar print block has no recognized closure")
            listing = True
            current = None
            continue
        if listing:
            if match := _CURRENT.fullmatch(stripped):
                current = {"id": match[1], "analysis_label": match[2], "line": last_line}
                continue
            if _LISTING.fullmatch(stripped):
                continue
            if (
                not current
                or current["analysis_label"] not in _SCALAR_LABELS
                or not _ASSIGN.fullmatch(stripped)
            ):
                listing = False
        if block or listing:
            match = _ASSIGN.fullmatch(stripped)
            if match is None:
                fail(last_line, "Expected a complete scalar assignment or print closure")
            if block is None:
                block = base(last_line)
                block.update(layout="scalar_print", axis=None, entries=[])
                reserve(_nodes(block))
                listing = False
            reserve(_SCALAR_ENTRY_NODES)
            try:
                representation, real, imag = _value(match[2])
            except ValueError as exc:
                fail(last_line, str(exc))
            block["entries"].append(
                {
                    "line": last_line,
                    "label": match[1],
                    "representation": representation,
                    "real": real,
                    "imag": imag,
                    "unit": None,
                }
            )
            block["line_end"] = last_line
    if block or title or repeated_title_line is not None:
        fail(last_line + 1, "EOF before a recognized print-block closure")
    return blocks


def decode_native_tables(text: str, *, limits: LogLimits) -> dict[str, Any]:
    """Decode prevalidated captured text through the common section contract."""
    section = log_section(lambda: _parse(text, limits), limits)
    if section["status"] == "parsed" and not section["value"]:
        section["status"] = "absent"
    return section
