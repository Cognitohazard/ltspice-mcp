"""Graders. Each judges the workspace the agent left, never its transcript,
and none runs the server's code, so no variant can move its own grade.

The ``answer`` grader compares ``answer.json`` with the task's reference:

- an object must have every key the reference has (others are ignored);
- a list must have the reference's length, compared item by item;
- a number passes within the tolerance; the agent may write it as a number or
  as a string with an SI or SPICE suffix and a unit (``"1.59kHz"``,
  ``"1.59 k"``, ``"1590"``). ``meg`` and ``M`` are mega, ``m`` is milli;
- a boolean also accepts ``"true"``/``"false"``/``"yes"``/``"no"``;
- a string compares without regard to case or surrounding space;
- ``null`` must be ``null``: the form a task gives for "this cannot be read".
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ltspice_eval.tasks import Task, Tolerance

ANSWER_FILE = "answer.json"

#: SI and SPICE multipliers. ``M`` is mega and ``m`` milli, as a person
#: writes them; ``meg`` (any case) is SPICE's mega. The others ignore case.
_SUFFIXES = {
    "f": 1e-15,
    "p": 1e-12,
    "n": 1e-9,
    "u": 1e-6,
    "\u00b5": 1e-6,
    "\u03bc": 1e-6,
    "m": 1e-3,
    "k": 1e3,
    "g": 1e9,
    "t": 1e12,
}
_NUMBER = re.compile(
    r"^\s*(?P<value>[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)\s*(?P<rest>.*?)\s*$"
)
_UNITS = frozenset(
    {
        "",
        "hz",
        "s",
        "sec",
        "v",
        "a",
        "ohm",
        "ohms",
        "\u03c9",
        "f",
        "h",
        "db",
        "deg",
        "\u00b0",
        "w",
        "%",
    }
)


@dataclass
class Grade:
    passed: bool
    #: ``ok``, or ``unparseable`` when there was no answer to compare.
    status: str
    mismatches: list[str] = field(default_factory=list)
    answer: Any = None

    def as_row(self) -> dict[str, Any]:
        return {"pass": self.passed, "status": self.status, "mismatches": self.mismatches}


def grade(task: Task, workspace: Path) -> Grade:
    if task.grader.kind != "answer":
        raise ValueError(f"no grader for {task.grader.kind!r}")
    reference = json.loads(task.grader.reference.read_text(encoding="utf-8"))
    path = workspace / ANSWER_FILE
    try:
        answer = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return Grade(passed=False, status="unparseable", mismatches=[f"{ANSWER_FILE} is missing"])
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        return Grade(passed=False, status="unparseable", mismatches=[f"{ANSWER_FILE}: {exc}"])
    mismatches = compare(answer, reference, task.grader.tolerance, task.grader.fields)
    return Grade(passed=not mismatches, status="ok", mismatches=mismatches, answer=answer)


def compare(
    answer: Any,
    reference: Any,
    tolerance: Tolerance,
    fields: dict[str, Tolerance] | None = None,
    path: str = "$",
) -> list[str]:
    """Every way ``answer`` differs from ``reference``, as ``path: reason``."""
    if isinstance(reference, dict):
        if not isinstance(answer, dict):
            return [f"{path}: expected an object"]
        out: list[str] = []
        for key, expected in reference.items():
            where = f"{path}.{key}"
            if key not in answer:
                out.append(f"{where}: missing")
                continue
            local = (fields or {}).get(key, tolerance) if path == "$" else tolerance
            out += compare(answer[key], expected, local, None, where)
        return out
    if isinstance(reference, list):
        if not isinstance(answer, list):
            return [f"{path}: expected a list"]
        if len(answer) != len(reference):
            return [f"{path}: expected {len(reference)} items, got {len(answer)}"]
        out = []
        for index, (got, expected) in enumerate(zip(answer, reference, strict=True)):
            out += compare(got, expected, tolerance, None, f"{path}[{index}]")
        return out
    if reference is None:
        return [] if answer is None else [f"{path}: expected null, got {answer!r}"]
    if isinstance(reference, bool):
        got = as_bool(answer)
        return [] if got is reference else [f"{path}: expected {reference}, got {answer!r}"]
    if isinstance(reference, (int, float)):
        got = as_number(answer)
        if got is None:
            return [f"{path}: expected a number, got {answer!r}"]
        if within(got, float(reference), tolerance):
            return []
        return [f"{path}: {got!r} is not within {_describe(tolerance)} of {reference!r}"]
    if isinstance(reference, str):
        if isinstance(answer, str) and _normal(answer) == _normal(reference):
            return []
        return [f"{path}: expected {reference!r}, got {answer!r}"]
    raise TypeError(f"{path}: a reference holds JSON values only")


def within(value: float, reference: float, tolerance: Tolerance) -> bool:
    if not math.isfinite(value):
        return False
    allowed = max(tolerance.abs, tolerance.rel * abs(reference))
    return abs(value - reference) <= allowed


def as_number(value: Any) -> float | None:
    """``value`` as a number, if it is one or a string spelling one.

    The text after the digits is read as a unit alone (``"5 F"`` is 5), or
    else as a multiplier and a unit (``"5 fF"``, ``"2ms"``, ``"10 MHz"``).
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str) or (match := _NUMBER.match(value)) is None:
        return None
    number, rest = float(match["value"]), match["rest"]
    if rest.lower() in _UNITS:
        return number
    if rest.lower().startswith("meg") and rest[3:].lower() in _UNITS:
        return number * 1e6
    head, unit = rest[:1], rest[1:].lower()
    if unit not in _UNITS:
        return None
    if head == "M":
        return number * 1e6
    multiplier = _SUFFIXES.get(head if head == "m" else head.lower())
    return None if multiplier is None else number * multiplier


def as_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return {"true": True, "yes": True, "false": False, "no": False}.get(value.strip().lower())
    return None


def _normal(text: str) -> str:
    return " ".join(text.split()).casefold()


def _describe(tolerance: Tolerance) -> str:
    parts = []
    if tolerance.rel:
        parts.append(f"{tolerance.rel:g} relative")
    if tolerance.abs:
        parts.append(f"{tolerance.abs:g} absolute")
    return " or ".join(parts) or "exact equality"
