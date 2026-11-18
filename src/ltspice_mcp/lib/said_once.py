"""Standing notes about a job, said in full once per session and by code after.

A standing note describes a job or the request that made it and reads the same
on every reply about that job: a micro sign folded at staging, a run whose
ngspice mode could not be checked, results left in the job folder, the
request's lint findings, its source block, and the route to the
``measurements`` recipe. A caller that waits on a job sees each one on every
reply, so after the first time it is sent as its code and how many replies
carried it before (``said_before``).

One table says which deliveries of a job's answer shorten a note
(:data:`SHORTENS`): ``jobs(action="status")`` always sends every note in full,
which is how a caller that lost its context reads them again; a wait and a
run receipt shorten. :func:`shortened` is the one decision, and takes the
delivery as its argument; :func:`plan` asks it for each note of an answer,
:func:`apply` sends the shortened ones by code, and :func:`commit` remembers
what the reply sent carried.

The memory (:class:`NotesSent`) is the session's: one stdio session is one
caller, so it is keyed by the job, the note's code and a digest of its text,
and needs no caller identity. A note whose text changed is a new note and is
sent in full. The memory is bounded and never written to disk, so a new
session starts with nothing remembered.

Never shortened, because they are about this reply's outcome: failures, run
rows and their diagnostics and measurement fields, progress, and every
observation whose code is not in :data:`STANDING_CODES`.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Final, Literal

from ltspice_mcp.lib.continuation import mcp_call
from ltspice_mcp.lib.cursor_codec import canonical_hash

#: How a job's answer is delivered.
Delivery = Literal["status", "wait", "run", "resume"]

#: Which deliveries send a note already sent in full by its code. A status
#: call sends every note in full, so a caller that lost its context reads
#: them there.
SHORTENS: Final[Mapping[Delivery, bool]] = {
    "status": False,
    "wait": True,
    "run": True,
    "resume": True,
}

#: The observations that describe the job or its request and do not change
#: between replies about it.
STANDING_CODES: Final = frozenset(
    {"micro_sign_folded", "ngspice_mode_unverified", "results_not_beside_sheet"}
)

#: The code the request's lint findings for one circuit are remembered by.
LINT_CODE: Final = "lint"

#: The code a source entry is remembered by.
SOURCE_CODE: Final = "source"

#: The code the route to the ``measurements`` recipe is remembered by.
MEASUREMENTS_ROUTE_CODE: Final = "measurements_route"

#: How many notes a session remembers; the oldest is forgotten first, and is
#: then sent in full again.
MEMORY_LIMIT: Final = 4096


def digest(text: Any) -> str:
    """The digest a note is remembered by: its canonical JSON's hash."""
    return canonical_hash(text)[:16]


#: A note as the memory names it: the job, the note's code and its digest.
_Key = tuple[str, str, str]


class NotesSent:
    """What this session has sent: per (job, code, digest), how many replies
    carried the note. Bounded at :data:`MEMORY_LIMIT` notes."""

    def __init__(self, limit: int = MEMORY_LIMIT) -> None:
        self._limit = limit
        self._sent: OrderedDict[_Key, int] = OrderedDict()

    def before(self, key: _Key) -> int:
        """How many replies carried the note before; 0 for a new one."""
        return self._sent.get(key, 0)

    def carried(self, key: _Key) -> None:
        """Record one more reply carrying the note."""
        count = self._sent.pop(key, 0)
        self._sent[key] = count + 1
        while len(self._sent) > self._limit:
            self._sent.popitem(last=False)

    def __len__(self) -> int:
        return len(self._sent)


def shortened(memory: NotesSent | None, delivery: Delivery, key: _Key) -> int | None:
    """How many replies carried this note before, when this delivery sends it
    by its code; None when it goes out in full: a new note, a delivery that
    sends every note in full, or no memory (the Python API, which returns
    everything in full)."""
    if memory is None:
        return None
    before = memory.before(key)
    return before if before and SHORTENS[delivery] else None


@dataclass(frozen=True)
class Plan:
    """The standing notes of one job's answer, and which of them this
    delivery sends by code. Made once per reply, applied to each candidate the
    reply is built from, and remembered (:func:`commit`) for the reply sent."""

    job_id: str
    seen: tuple[_Key, ...]
    short: Mapping[_Key, int]
    route: str | None = None
    short_route: str = ""

    @property
    def shortens(self) -> bool:
        return bool(self.short)


def _notes(answer: Mapping[str, Any], job_id: str, route: str | None) -> list[_Key]:
    keys: list[_Key] = []
    for item in answer.get("observations") or ():
        if isinstance(item, dict) and item.get("code") in STANDING_CODES:
            keys.append((job_id, str(item["code"]), digest(item)))
    for field, code in (("lint", LINT_CODE), ("source", SOURCE_CODE)):
        for entry in answer.get(field) or ():
            if isinstance(entry, dict) and "circuit" in entry:
                keys.append((job_id, code, digest(entry)))
    hint = answer.get("hint")
    if route and isinstance(hint, str) and route in hint:
        keys.append((job_id, MEASUREMENTS_ROUTE_CODE, digest(route)))
    return keys


def plan(
    answer: Mapping[str, Any],
    *,
    memory: NotesSent | None,
    delivery: Delivery,
    route: str | None = None,
    short_route: str = "",
) -> Plan | None:
    """Which standing notes of ``answer`` this delivery sends by code. None
    for an answer that names no job, or with no memory.

    ``route`` is the sentence of the answer's hint that routes to the
    ``measurements`` recipe, replaced by ``short_route`` when shortened.
    """
    job_id = answer.get("job_id")
    if not isinstance(job_id, str) or memory is None:
        return None
    seen = tuple(dict.fromkeys(_notes(answer, job_id, route)))
    short = {
        key: before
        for key in seen
        if (before := shortened(memory, delivery, key)) is not None
    }
    return Plan(job_id, seen, short, route, short_route)


def apply(answer: dict[str, Any], planned: Plan | None) -> int:
    """Send the notes ``planned`` shortens by code, in place; how many."""
    if planned is None or not planned.short:
        return 0
    job_id, short = planned.job_id, planned.short
    count = 0

    def said(code: str, text: Any) -> int | None:
        nonlocal count
        before = short.get((job_id, code, digest(text)))
        if before is not None:
            count += 1
        return before

    observations = answer.get("observations")
    if isinstance(observations, list):
        kept: list[Any] = []
        for item in observations:
            if isinstance(item, dict) and item.get("code") in STANDING_CODES:
                code = str(item["code"])
                before = said(code, item)
                if before is not None:
                    item = {"code": code, "kind": item.get("kind", ""), "said_before": before}
            kept.append(item)
        answer["observations"] = kept
    for field, code in (("lint", LINT_CODE), ("source", SOURCE_CODE)):
        entries = answer.get(field)
        if not isinstance(entries, list):
            continue
        rendered: list[Any] = []
        for entry in entries:
            if isinstance(entry, dict) and "circuit" in entry:
                before = said(code, entry)
                if before is not None:
                    entry = {"circuit": entry["circuit"], "said_before": before}
            rendered.append(entry)
        answer[field] = rendered
    hint = answer.get("hint")
    route = planned.route
    if (
        route
        and isinstance(hint, str)
        and route in hint
        and said(MEASUREMENTS_ROUTE_CODE, route) is not None
    ):
        hint = hint.replace(route, planned.short_route)
    if count:
        repeat = mcp_call("jobs", {"action": "status", "job_id": job_id})
        sentence = (
            f"{count} note(s) sent in full earlier in this session carry said_before in "
            f"place of their text; {repeat} sends every note in full."
        )
        hint = f"{hint} {sentence}" if isinstance(hint, str) and hint else sentence
        answer["hint"] = hint
    return count


def commit(planned: Plan | None, memory: NotesSent | None) -> None:
    """Remember that the reply sent carried each of ``planned``'s notes."""
    if planned is None or memory is None:
        return
    for key in planned.seen:
        memory.carried(key)
