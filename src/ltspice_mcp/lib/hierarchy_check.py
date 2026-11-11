"""Check the hierarchy reader's values against ngspice's own reading of a deck.

The hierarchy reader (``hierarchy.py``) re-creates how ngspice reads a deck,
so its values are an assumption until ngspice gives the same ones. When an
ngspice executable is at hand, ``check_hierarchy`` asks it. A driver deck
``source``s the deck in the mode the reader read it in, as a run would, and
``print``s each device quantity the reader resolved. Then the two are
compared.

What is compared, by the reader's field and ngspice's instance parameter
(recorded with ngspice-42):

=================  ==========================  ===========================
Element            Reader                      ngspice
=================  ==========================  ===========================
``R``              ``value``                   ``resistance``
``M``              ``geometry.w``/``.l``       ``w``/``l``, both scaled by
                                               ``.options scale``, as the
                                               reader's geometry is
``R``, ``M``       ``parameters.m``            ``m``, unless a subcircuit
                                               call above sets ``m``, which
                                               ngspice replaces or
                                               multiplies by design
=================  ==========================  ===========================

Nothing else is compared, and the result counts what was not: elements the
reader names no ngspice device for (``address.device``), values it left
unresolved, and values ngspice printed none for. Nodes, models and sources are
not compared at all.

ngspice runs a deck's ``.control`` block when it reads the deck. The reader
refuses a deck that has one, so no hierarchy this module is given has one.
"""

from __future__ import annotations

import logging
import math
import re
import sys
import threading
from collections import Counter, OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from ltspice_mcp.lib.hierarchy import Hierarchy, ResolvedInstance
from ltspice_mcp.lib.ngspice_mode import deck_flags, flags_for, mode_notes, run_batch
from ltspice_mcp.lib.simulator import is_ngspice

logger = logging.getLogger(__name__)

_TIMEOUT_S = 60.0
_REL_TOL = 1e-6
"""ngspice prints seven significant digits."""

_PRINTED = re.compile(
    r"^@(?P<device>[^\s\[]+)\[(?P<quantity>[^\]\s]+)\]\s*=\s*(?P<value>\S+)", re.MULTILINE
)
_UNSAFE_PATH = re.compile(r"['\"$`\\\r\n]")
"""Characters ngspice's control language would read in a single-quoted path.
A double-quoted path keeps its quotes as part of the name (ngspice-42)."""

_CACHE_SIZE = 32
_cache: OrderedDict[tuple[Any, ...], HierarchyCheck] = OrderedDict()
_cache_lock = threading.Lock()

Reason = Literal[
    "no_ngspice",
    "path_unsupported",
    "nothing_to_compare",
    "ngspice_failed",
    "inputs_changed",
    "deck_not_read",
    "mode_unchecked",
    "mode_differs",
]

_REASONS: dict[str, str] = {
    "no_ngspice": "no ngspice executable is available to this session",
    "path_unsupported": (
        "the deck's path holds a character ngspice's control language would read "
        "(a quote, $, `, a backslash or a line break), or is not ASCII on Windows"
    ),
    "nothing_to_compare": "the reader resolved no value that ngspice reports",
    "ngspice_failed": "ngspice could not be run, or wrote no log",
    "inputs_changed": "a file the reader read changed before ngspice read it",
    "deck_not_read": "ngspice did not finish reading the deck",
    "mode_unchecked": "ngspice's log does not show which compatibility mode it read the deck in",
    "mode_differs": (
        "ngspice read the deck in another compatibility mode than the reader; a startup "
        "file it reads, most likely the installation's spinit, sets ngbehavior"
    ),
}


@dataclass(frozen=True)
class Difference:
    instance: tuple[str, ...]
    quantity: str
    reader: float
    ngspice: float


@dataclass(frozen=True)
class HierarchyCheck:
    """How the reader's values compare with ngspice's."""

    status: Literal["agrees", "differs", "not_run"]
    compared: int = 0
    differences: tuple[Difference, ...] = ()
    not_compared: tuple[tuple[str, int], ...] = ()
    reason: Reason | None = None

    def explanation(self) -> str:
        if self.reason is not None:
            return _REASONS[self.reason]
        if self.differences:
            first = self.differences[0]
            return (
                f"ngspice gives other values for {len(self.differences)} of "
                f"{self.compared} compared, first {'.'.join(first.instance)} "
                f"{first.quantity}: reader {first.reader:g}, ngspice {first.ngspice:g}"
            )
        return f"ngspice gives the same {self.compared} values"

    def payload(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "compared": self.compared,
            "differences": [
                {
                    "instance": list(item.instance),
                    "quantity": item.quantity,
                    "reader": item.reader,
                    "ngspice": item.ngspice,
                }
                for item in self.differences
            ],
            "not_compared": dict(self.not_compared),
            "reason": self.reason,
            "detail": _REASONS[self.reason] if self.reason is not None else None,
        }


@dataclass(frozen=True)
class Planned:
    row: ResolvedInstance
    field: str
    quantity: str
    value: float


def plan(hierarchy: Hierarchy) -> tuple[list[Planned], Counter[str]]:
    """The reader's values ngspice can be asked for, and a count of the rest."""
    by_path = {tuple(p.casefold() for p in row.instance): row for row in hierarchy.instances}
    skipped: Counter[str] = Counter()
    planned: list[Planned] = []
    for row in hierarchy.instances:
        if row.element == "X":
            continue
        if row.device is None:
            skipped["no_ngspice_name"] += 1
            continue
        facts = []
        if row.element == "R":
            facts.append(("value", "resistance", row.value))
        if row.element == "M":
            geometry = dict(row.geometry)
            facts += [
                ("geometry.w", "w", geometry.get("w")),
                ("geometry.l", "l", geometry.get("l")),
            ]
        parameters = {key.casefold(): fact for key, fact in row.parameters}
        if "m" in parameters:
            path = tuple(p.casefold() for p in row.instance)
            callers = (by_path.get(path[:i]) for i in range(1, len(path)))
            if any(
                caller is not None and any(key.casefold() == "m" for key, _ in caller.parameters)
                for caller in callers
            ):
                skipped["m_set_by_subcircuit_call"] += 1
            else:
                facts.append(("parameters.m", "m", parameters["m"]))
        for field, quantity, fact in facts:
            if fact is None or fact.status != "resolved" or fact.value is None:
                skipped["unresolved"] += 1
            else:
                planned.append(Planned(row, field, quantity, fact.value))
    return planned, skipped


def _driver(root: Path, planned: list[Planned]) -> bytes | None:
    path = root.as_posix()
    if _UNSAFE_PATH.search(path) or (sys.platform == "win32" and not path.isascii()):
        return None
    prints = dict.fromkeys(f"print @{item.row.device}[{item.quantity}]" for item in planned)
    lines = ["* hierarchy check", ".control", f"source '{path}'", *prints, "quit", ".endc", ".end"]
    return ("\n".join(lines) + "\n").encode("utf-8")


def check_hierarchy(hierarchy: Hierarchy, simulator: type | None) -> HierarchyCheck:
    """Compare the reader's values for ``hierarchy`` with the ones the ngspice
    ``simulator`` starts gives, reading the deck in the hierarchy's mode.

    Never raises: what keeps the check from running is the result's reason.
    A result is kept per deck content, mode and executable for the life of
    the process, so a paged query asks once.
    """
    planned, skipped = plan(hierarchy)
    not_compared = tuple(sorted(skipped.items()))
    command = tuple(str(part) for part in getattr(simulator, "spice_exe", ()) or ())
    if simulator is None or not is_ngspice(simulator) or not command:
        return HierarchyCheck("not_run", not_compared=not_compared, reason="no_ngspice")
    if not planned:
        return HierarchyCheck("not_run", not_compared=not_compared, reason="nothing_to_compare")
    if hierarchy.root is None:
        return HierarchyCheck("not_run", not_compared=not_compared, reason="path_unsupported")
    driver = _driver(hierarchy.root, planned)
    if driver is None:
        return HierarchyCheck("not_run", not_compared=not_compared, reason="path_unsupported")
    mode = hierarchy.profile.ngbehavior or ""
    key = (
        command,
        mode,
        str(hierarchy.root),
        tuple((str(captured.path), captured.digest) for captured in hierarchy.inputs),
    )
    with _cache_lock:
        if key in _cache:
            _cache.move_to_end(key)
            return _cache[key]
    result = _ask(hierarchy, hierarchy.root, command, mode, driver, planned, skipped)
    if result.reason not in ("ngspice_failed", "inputs_changed"):
        with _cache_lock:
            _cache[key] = result
            while len(_cache) > _CACHE_SIZE:
                _cache.popitem(last=False)
    return result


def _ask(
    hierarchy: Hierarchy,
    root: Path,
    command: tuple[str, ...],
    mode: str,
    driver: bytes,
    planned: list[Planned],
    skipped: Counter[str],
) -> HierarchyCheck:
    def not_run(reason: Reason) -> HierarchyCheck:
        return HierarchyCheck(
            "not_run", not_compared=tuple(sorted(skipped.items())), reason=reason
        )

    # Any failure: a check that cannot run must not fail the query it checks.
    try:
        log = run_batch(
            command, mode, driver, cwd=root.parent, clean_startup=False, timeout=_TIMEOUT_S
        )
    except Exception as exc:
        logger.warning("Could not ask %s to read %s: %s", command, root, exc)
        return not_run("ngspice_failed")
    if _changed(hierarchy):
        return not_run("inputs_changed")
    if not mode_notes(log):
        return not_run("mode_unchecked")
    logged = deck_flags(log, driver=True)
    if logged is None:
        return not_run("deck_not_read")
    expected = flags_for(command, mode)
    if expected is None:
        return not_run("mode_unchecked")
    if logged != expected:
        return not_run("mode_differs")
    return compare(planned, printed_values(log), skipped)


def _changed(hierarchy: Hierarchy) -> bool:
    """Whether a file the reader read holds other bytes now, so ngspice may
    have read something else."""
    for captured in hierarchy.inputs:
        try:
            if captured.path.read_bytes() != captured.content:
                return True
        except OSError:
            return True
    return False


def printed_values(log: str) -> dict[tuple[str, str], float]:
    """Each ``@device[quantity] = value`` line a log holds, by device and
    quantity, casefolded."""
    printed: dict[tuple[str, str], float] = {}
    for match in _PRINTED.finditer(log):
        try:
            value = float(match["value"])
        except ValueError:
            continue
        printed[(match["device"].casefold(), match["quantity"].casefold())] = value
    return printed


def compare(
    planned: list[Planned], printed: dict[tuple[str, str], float], skipped: Counter[str]
) -> HierarchyCheck:
    """Compare each planned reader value with the value ngspice printed."""
    skipped = Counter(skipped)
    differences: list[Difference] = []
    compared = 0
    for item in planned:
        value = printed.get(((item.row.device or "").casefold(), item.quantity))
        if value is None:
            skipped["no_ngspice_value"] += 1
            continue
        compared += 1
        if not math.isclose(item.value, value, rel_tol=_REL_TOL, abs_tol=0.0):
            differences.append(Difference(item.row.instance, item.field, item.value, value))
    not_compared = tuple(sorted(skipped.items()))
    if differences:
        return HierarchyCheck("differs", compared, tuple(differences), not_compared)
    if not compared:
        return HierarchyCheck("not_run", not_compared=not_compared, reason="nothing_to_compare")
    return HierarchyCheck("agrees", compared, (), not_compared)
