"""A run's results beside its circuit file, where LTspice keeps them.

LTspice writes a circuit's results beside it under its name (``<name>.raw``,
``<name>.log``, and ``<name>.op.raw`` or ``<name>.fra_1.raw`` where the run
wrote one), a sheet's and a netlist's alike: the batch runs recorded under
``raw/`` on both builds leave each netlist's there. LTspice's window opens a
sheet's from there, so a click on a net in the sheet plots it (recorded on
LTspice 26); what the window does with results beside a netlist is not
recorded. The server's runs write theirs to the job folder instead. This
module copies one run's beside its circuit on request, from
``jobs(action="place_beside_sheet")``, from ``plot_waveform`` with
``in_ltspice`` before it opens the run from its sheet, or, when a job asked
for it with ``execution.results_beside_sheet``, as the job's runs finish.
Each places through ``place_checked``, which asks ``place_case`` under
``placement_lock``, so each gives the same answer; ``pyproject.toml`` bans the
copy, the check-and-copy and the lock outside this module. The results in the
job folder stay where they are, and remain what every read of the job takes.

Which files beside a circuit its run goes to is decided once, by
``files_beside``; whose run a results file there holds, by ``results_owner``.
"""

from __future__ import annotations

import asyncio
import contextlib
import functools
import os
import shutil
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ltspice_mcp.errors import NetlistError
from ltspice_mcp.lib import atomic_write
from ltspice_mcp.lib import filelock as filelock_mod
from ltspice_mcp.lib.asc_document import is_asc
from ltspice_mcp.lib.capabilities import entry
from ltspice_mcp.lib.deck_staging import sha256_file
from ltspice_mcp.lib.experiment_types import OWNER_STOPPED, ExperimentCase, ExperimentJob
from ltspice_mcp.lib.filelock import async_file_lock, circuit_file_lock
from ltspice_mcp.lib.job_lifecycle import runs_done  # noqa: TID251 - a placement waits for the job's own copy
from ltspice_mcp.lib.ltspice_window import results_beside  # noqa: TID251 - the one caller
from ltspice_mcp.lib.raw_parser import OTHER_RAW_KINDS, other_raw_path
from ltspice_mcp.lib.simulator import dialect_for_simulator_name, simulator_family
from ltspice_mcp.lib.store import Store, file_key

#: The capability the request check and the case check ask the table about.
CAPABILITY = "results_beside_sheet"

#: What has not been recorded of LTspice's window and a netlist, said in every
#: reply that put results beside a netlist (``tests/fixtures/ltspice_recorded/
#: inputs/cases.toml``, behaviour ``netlist-results-in-window``).
NETLIST_WINDOW_UNCHECKED = (
    "What LTspice's window does with results put beside a netlist has not been "
    "checked: which of its commands open them, and whether a plot opened from the "
    "netlist is tied to it."
)


class NotPlaced(Exception):
    """Why a run's results cannot stand as its circuit's own; ``code`` is the
    wire code the caller reports."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class ResultFiles:
    """The files beside ``circuit`` that a run of it goes to, under its name,
    as LTspice writes them for a sheet and for a netlist."""

    circuit: Path
    #: The results file (``ltspice_window.results_beside``).
    raw: Path
    log: Path

    def other_raw(self, kind: str) -> Path:
        """Where the run's raw of ``kind`` (``op``, ``fra_1``) goes."""
        return other_raw_path(self.raw, kind)

    def every(self) -> list[Path]:
        """Each file beside the circuit that a run of it can go to."""
        return [self.raw, self.log, *(self.other_raw(kind) for kind in OTHER_RAW_KINDS)]


def files_beside(circuit: Path) -> ResultFiles:
    """Which files beside ``circuit`` its run goes to: the results, the log, and
    (``ResultFiles.other_raw``) the operating-point and frequency-response
    files where a run wrote one. The request check, the placement and its lock
    ask this; nothing else names those files."""
    return ResultFiles(circuit, results_beside(circuit), circuit.with_suffix(".log"))


def results_owner(circuit: Path) -> Path:
    """The circuit whose run the results file beside ``circuit`` holds as its
    own: a sheet beside it whose results file it is, else ``circuit`` itself.

    A sheet's results file is what LTspice's window shows as that sheet's run,
    and the server cannot check that a netlist is that sheet as it stands, so
    the sheet owns the file even when a netlist of its name ran. This is the
    one place that decides it. Blocking: it lists ``circuit``'s folder.
    """
    if is_asc(circuit):
        return circuit
    wanted = files_beside(circuit).raw.name.casefold()
    try:
        names = sorted(os.listdir(circuit.parent))
    except OSError:
        return circuit
    for name in names:
        candidate = circuit.parent / name
        if (
            is_asc(candidate)
            and files_beside(candidate).raw.name.casefold() == wanted
            and candidate.is_file()
        ):
            return candidate
    return circuit


def places_results(simulator: str | None) -> bool:
    """Whether a run on ``simulator`` (a job's recorded class name)
    has its results put beside its circuit: the capability table's answer.

    A record that names no simulator passes, as it did before the table was
    asked: such a record predates the field, when every job that could ask for
    this ran on LTspice, and refusing it would refuse the user's own run.
    """
    if simulator is None or simulator == "":
        return True
    # The family maps know this platform's class names; a record written on
    # WSL (LTspiceWSL) and read elsewhere is known to the dialect map.
    family = simulator_family(simulator) or dialect_for_simulator_name(simulator)
    found = entry(family, CAPABILITY) if family is not None else None
    return found is not None and found.status == "supported"


@contextlib.asynccontextmanager
async def placement_lock(circuit: Path) -> AsyncIterator[None]:
    """The locks a copy beside ``circuit`` holds: the circuit's own
    (``circuit_file_lock``), then its results file's (``Store.results_file_lock``).

    Two circuits whose runs go to one results file hold different circuit
    locks, and without the second their copies could interleave. The order is
    always circuit, then results file. A wait that runs out raises
    ``NetlistError`` in the caller's terms, as the circuit's lock does.
    """
    raw = files_beside(circuit).raw
    async with circuit_file_lock(circuit):
        timeout = filelock_mod.DEFAULT_TIMEOUT
        target = await asyncio.to_thread(Store.results_file_lock, raw)
        async with contextlib.AsyncExitStack() as stack:
            try:
                await stack.enter_async_context(async_file_lock(target, acquire_timeout=timeout))
            except TimeoutError as exc:
                raise NetlistError(
                    f"{raw.name} is being written by another ltspice-mcp process "
                    f"(waited {timeout:.0f}s). Retry once it finishes."
                ) from exc
            yield


async def place_checked(job: ExperimentJob, case: ExperimentCase, circuit: Path) -> Placement:
    """Put ``case``'s results beside ``circuit`` (``place_case``) under
    ``placement_lock``: the one way any caller places them, so each gives the
    same answer. Raises ``NotPlaced`` with the refusal, and ``NetlistError``
    when another process held a lock past its wait.

    The check and the copy are one unit in a worker thread, which a cancel
    cannot stop. So a cancel does not end the copy, and the locks are held
    until it has ended, whatever ends the call: a second placement cannot
    write the same files beside it meanwhile. The call then raises the
    ``CancelledError``.
    """
    async with placement_lock(circuit):
        loop = asyncio.get_running_loop()
        copy = loop.run_in_executor(None, functools.partial(place_case, job, case, circuit))
        try:
            return await asyncio.shield(copy)
        except asyncio.CancelledError:
            while not copy.done():
                # A second cancel is the same request; the copy still ends first.
                with contextlib.suppress(asyncio.CancelledError):
                    await asyncio.wait({copy})
            if not copy.cancelled():
                copy.exception()  # retrieved: the cancel is what the caller hears
            raise


def place_beside_sheet(
    circuit: Path,
    raw: Path,
    log: Path | None,
    other_raws: Mapping[str, Path] | None = None,
) -> tuple[list[dict[str, Any]], list[Path]]:
    """Copy a run's results beside ``circuit`` under its name (``files_beside``),
    as a run in LTspice leaves them; what is there is replaced, each file
    whole or not at all. Returns one ``{from, to, replaced}`` per file copied,
    and each file of the circuit's beside it that this run did not write.

    A file of the circuit's that this run has no counterpart for (an
    ``.op.raw`` an earlier run left, say) is left where it is, and a file that
    already is the one beside the circuit is neither copied nor listed.
    Only ``place_case`` calls this, under the checks it makes.
    """
    beside = files_beside(circuit)
    pairs = [(raw, beside.raw), (log, beside.log)]
    pairs += [(path, beside.other_raw(kind)) for kind, path in (other_raws or {}).items()]
    placed: list[dict[str, Any]] = []
    written: set[Path] = set()
    for source, target in pairs:
        if source is None or not source.is_file():
            continue
        written.add(target)
        replaced = target.exists()
        if replaced and os.path.samefile(source, target):
            continue
        with source.open("rb") as read, atomic_write(target, mode="wb", durable=False) as write:
            shutil.copyfileobj(read, write)
        placed.append({"from": str(source), "to": str(target), "replaced": replaced})
    left = [path for path in beside.every() if path not in written and path.is_file()]
    return placed, left


def _names(paths: Sequence[Path]) -> str:
    """Each spelling in ``paths`` once, in order, as ``a`` or ``a and b``: by
    file name, or by full path where two spellings share a name."""
    spellings = list(dict.fromkeys(str(path) for path in paths))
    shown = [Path(spelling).name for spelling in spellings]
    if len(set(shown)) < len(shown):
        shown = spellings
    return shown[0] if len(shown) == 1 else f"{', '.join(shown[:-1])} and {shown[-1]}"


def _belongs_to_sheet(circuit: Path, sheet: Path) -> NotPlaced:
    raw = files_beside(circuit).raw.name
    return NotPlaced(
        "results_belong_to_sheet",
        f"{raw} beside {circuit.name} is the results file of the sheet {sheet.name}: "
        "LTspice's window shows what it holds as that sheet's run, and the server cannot "
        f"check that {circuit.name} is that sheet as it stands. Run the sheet, or copy "
        f"{circuit.name} under another name and run that.",
    )


def check_request(circuits: Sequence[tuple[Path, int]], simulator: str) -> None:
    """Refuse, before a job exists, a request to put its runs' results beside
    their circuits (``execution.results_beside_sheet``) that could not end
    with each results file holding one run of its own circuit: ``circuits``
    pairs each circuit's path with the number of cases it runs, and
    ``simulator`` is the class name the job runs on and will record, judged
    as ``check_case`` judges the record, so the two cannot disagree.

    A results file holds one run's results, so a file that would receive
    several cases (a sweep, one circuit named twice by any spelling, or
    ``foo.asc`` and ``foo.net``, both ``foo.raw``) is refused, with the action
    that copies a chosen case afterwards; so is a netlist whose results file
    is a sheet's (``results_owner``). Blocking: run it off the loop.
    """
    if not places_results(simulator):
        raise NotPlaced(
            "not_ltspice",
            "execution.results_beside_sheet puts a run's results beside its circuit "
            f"where LTspice keeps them, and this job runs on {simulator}, whose "
            "results the server does not put there. Leave the option off.",
        )
    by_file: dict[str, tuple[list[Path], int]] = {}
    for path, count in circuits:
        key = file_key(files_beside(path).raw)
        paths, total = by_file.get(key, ([], 0))
        by_file[key] = ([*paths, path], total + count)
    for paths, count in by_file.values():
        if count > 1:
            raw = files_beside(paths[0]).raw.name
            names = _names(paths)
            spellings = len({str(path) for path in paths})
            said = (
                f"{names} runs {count} cases in this job, and its results go to {raw}"
                if spellings == 1
                else f"{names} {'both' if spellings == 2 else 'all'} put their results in {raw}"
            )
            raise NotPlaced(
                "several_cases_per_sheet",
                f"{said}, and one results file holds one run. Leave "
                "execution.results_beside_sheet off, and copy the case you want afterwards "
                "with jobs(action='place_beside_sheet', job_id=..., case_id=...).",
            )
    for path, _count in circuits:
        owner = results_owner(path)
        if owner != path:
            raise _belongs_to_sheet(path, owner)


def check_case(job: ExperimentJob, case: ExperimentCase) -> None:
    """Refuse, from the job's record alone, a case whose results cannot be put
    beside its circuit: a case with no results, and a run on a simulator whose
    results the server does not put there (``places_results``), and any case
    of a job whose runs have not all ended.

    The last is the job's, not the case's: a job run with
    ``execution.results_beside_sheet`` copies each produced case beside its
    circuit once its runs have all ended (``place_job_beside_sheets``), and
    its record does not say whether it was asked to. A placement made while
    its runs go on would be copied over then, and on Windows that copy fails
    while LTspice holds the first one open. Reading a case does not wait for
    the job (``services.case_unreadable``); placing it does.
    """
    if not runs_done(job):
        raise NotPlaced(
            "job_not_terminal",
            f"Case {case.case_id} of job {job.job_id} is placed beside its circuit once "
            "the job's runs have all ended, since a job that copies its results there "
            "itself does so then; wait for them with jobs(action='wait', "
            f"job_id='{job.job_id}', wait_for='runs').",
        )
    if case.status != "produced" or case.raw_file is None:
        raise NotPlaced(
            "case_not_produced",
            f"Case {case.case_id} of job {job.job_id} has no results to place (status "
            f"{case.status}).",
        )
    if not places_results(job.simulator):
        raise NotPlaced(
            "not_ltspice",
            f"Job {job.job_id} ran on {job.simulator}; the server puts only an LTspice "
            "run's results beside its circuit, where LTspice keeps them.",
        )


@dataclass
class Placement:
    """What ``place_case`` copied, and what it did not check."""

    files: list[dict[str, Any]]
    #: Each thing the placement did not check, said in full.
    not_checked: list[str] = field(default_factory=list)
    #: Each file beside the circuit that a run of it goes to and that this
    #: run did not write: another run's, left where it was.
    left_in_place: list[str] = field(default_factory=list)


def check_placement(job: ExperimentJob, case: ExperimentCase, circuit: Path) -> list[str]:
    """Whether ``case``'s results may stand beside ``circuit``, the circuit it
    ran, as its own: raises ``NotPlaced``, or returns what the check did not
    look at, each said in full. Besides ``check_case``, refused when the
    results or the circuit are gone, when the circuit has changed since the
    job read it (by the digest taken at submission: the sheet's, or the
    netlist's own deck), since the results would not match it, and when the
    results file is a sheet's and the circuit is not that sheet
    (``results_owner``).

    ``place_case`` asks this under the locks, just before it copies. A caller
    that has something to do before the copy that a refusal would make
    pointless (``plot_waveform`` asks LTspice's window to close a plot) may
    ask it first; the answer under the locks is the one that holds.
    Blocking: it reads the circuit.
    """
    check_case(job, case)
    raw = case.raw_file
    assert raw is not None  # check_case refuses a case without one
    if not raw.is_file():
        raise NotPlaced(
            "results_missing", f"Case {case.case_id}'s results are not there any more ({raw})."
        )
    if not circuit.is_file():
        raise NotPlaced("sheet_missing", f"{circuit} is not there any more.")
    owner = results_owner(circuit)
    if owner != circuit:
        raise _belongs_to_sheet(circuit, owner)
    not_checked: list[str] = []
    ran = next((source.sha256 for source in job.sources if source.circuit == case.circuit), None)
    if ran is None:
        not_checked.append(
            f"{circuit.name} was not compared with the file job {job.job_id} ran, nor were "
            "the files it includes: the job holds no record of that file's digest."
        )
    elif sha256_file(circuit) != ran:
        raise NotPlaced(
            "sheet_changed",
            f"{circuit.name} has changed since job {job.job_id} ran it, so these results "
            f"would not match it. Run {circuit.name} again and place that run's.",
        )
    else:
        # The digest is of the circuit file alone (``SourceRecord.sha256``).
        not_checked.append(
            f"The files {circuit.name} includes or uses (libraries, models, symbols) were "
            f"not compared with the ones the run read: only {circuit.name} itself was."
        )
    if not is_asc(circuit):
        not_checked.append(NETLIST_WINDOW_UNCHECKED)
        if "." in circuit.stem:
            not_checked.append(
                f"Where LTspice writes the results of a netlist with more than one dot in its "
                f"name has not been recorded; they were copied to "
                f"{files_beside(circuit).raw.name}."
            )
    return not_checked


def place_case(job: ExperimentJob, case: ExperimentCase, circuit: Path) -> Placement:
    """Copy ``case``'s results beside ``circuit``, the circuit it ran
    (``place_beside_sheet``), once ``check_placement`` lets them stand there,
    or raise ``NotPlaced``. A file of the circuit's that this run wrote no
    counterpart for is left where it is, and named (``left_in_place``).

    Blocking, and to be run under ``placement_lock(circuit)``: only
    ``place_checked`` calls it.
    """
    not_checked = check_placement(job, case, circuit)
    raw = case.raw_file
    assert raw is not None  # check_case refuses a case without one
    others = {kind: other_raw_path(raw, kind) for kind in case.other_raws}
    files, left = place_beside_sheet(circuit, raw, case.log_file, others)
    if left:
        not_checked.append(_left_said(circuit, left))
    return Placement(files, not_checked, [str(path) for path in left])


def _left_said(circuit: Path, left: Sequence[Path]) -> str:
    """What a reply says of the circuit's files a placement left where they were.

    What LTspice's window does with such a file beside a newer run's results
    is not recorded (``tests/fixtures/ltspice_recorded/inputs/cases.toml``,
    behaviour ``results-of-another-run-beside-a-circuit``).
    """
    one = len(left) == 1
    return (
        f"{_names([Path(path.name) for path in left])} beside {circuit.name} "
        f"{'is' if one else 'are'} from another run and {'was' if one else 'were'} left "
        f"in place: this run wrote no such file. What LTspice's window does with "
        f"{'it' if one else 'them'} beside this run's results has not been recorded."
    )


async def place_job_beside_sheets(job: ExperimentJob) -> list[dict[str, Any]]:
    """Copy each produced case's results beside the circuit it ran, as
    ``execution.results_beside_sheet`` asks once the job's runs are done, and
    return one observation per circuit: what was copied, or why not.

    ``check_request`` has already held the job to one case per results file
    and to runs the table puts beside their circuits. A case that produced
    nothing leaves its circuit alone, as its own row already says. A refusal
    or a failed copy costs only its own circuit.
    """
    observations: list[dict[str, Any]] = []
    for case in job.cases:
        if case.status != "produced":
            continue
        circuit = case.circuit_path
        try:
            # A job stopped now cannot leave a circuit with some of a run's
            # files, nor its locks taken by another copy while this one runs.
            placed = await place_checked(job, case, circuit)
        except NotPlaced as exc:
            observations.append(_not_placed(job, case, str(exc), code=exc.code))
            continue
        except NetlistError as exc:
            # Another server process held the circuit or its results file
            # past the lock's wait.
            reason = f"{exc} {_place_later(job, case)}"
            observations.append(_not_placed(job, case, reason, code="sheet_locked"))
            continue
        except OSError as exc:
            reason = f"The copy failed ({exc}). {_place_later(job, case)}"
            observations.append(_not_placed(job, case, reason, code="copy_failed"))
            continue
        observations.append(_placed(case, circuit, placed))
    return observations


def not_placed_for_owner_stop(job: ExperimentJob) -> list[dict[str, Any]]:
    """One ``results_not_placed`` per produced case, for a job whose process
    began to stop before the copy ``execution.results_beside_sheet`` asked for.

    The copy is not made then: it waits on each circuit's locks and copies whole
    raws, which can outlast the bound a stopping process gives its jobs
    (``JobRegistry.cancel_running``), and the results stay in the job folder,
    where the jobs action copies them from at any time.
    """
    return [
        _not_placed(
            job,
            case,
            "The process running the job stopped before it copied them; "
            f"{_place_call(job, case)} copies them.",
            code=OWNER_STOPPED,
        )
        for case in job.cases
        if case.status == "produced"
    ]


def _place_call(job: ExperimentJob, case: ExperimentCase) -> str:
    """The call that copies ``case``'s results beside its circuit."""
    return f"jobs(action='place_beside_sheet', job_id='{job.job_id}', case_id='{case.case_id}')"


def _place_later(job: ExperimentJob, case: ExperimentCase) -> str:
    """The call that copies ``case``'s results once what stopped it is fixed."""
    return f"{_place_call(job, case)} copies them once that is fixed."


def sheets_reported(observations: Sequence[Mapping[str, Any]]) -> set[str]:
    """The circuits, as paths, that ``place_job_beside_sheets`` already
    reported on in a job's ``observations``, placed or not."""
    return {
        str(observation["evidence"]["sheet"])
        for observation in observations
        if observation.get("code") in ("results_placed_beside_sheet", "results_not_placed")
    }


def where_placed(circuit: Path) -> str:
    """Why the files beside ``circuit`` are where a run's results belong: for a
    sheet, LTspice's window opens them there (recorded on LTspice 26); for a
    netlist, LTspice writes them there (the batch runs recorded on both
    builds), and the window has not been checked (``NETLIST_WINDOW_UNCHECKED``,
    which a reply carries in ``not_checked``)."""
    if is_asc(circuit):
        return "where LTspice's window opens them"
    return "where LTspice writes a netlist's own results and log"


def _placed(case: ExperimentCase, circuit: Path, placement: Placement) -> dict[str, Any]:
    """The observation that ``case``'s results now sit beside ``circuit``."""
    names = ", ".join(Path(placed["to"]).name for placed in placement.files)
    detail = (
        f"Case {case.case_id}'s results were copied beside {circuit.name}, under its name, "
        f"{where_placed(circuit)}: {names}."
    )
    evidence: dict[str, Any] = {
        "sheet": str(circuit),
        "case_id": case.case_id,
        "files": placement.files,
    }
    if case.assignments:
        values = ", ".join(f"{name}={value}" for name, value in case.assignments.items())
        detail += f" The run used values the {_kind(circuit)} does not show: {values}."
        evidence["assignments"] = dict(case.assignments)
    if placement.not_checked:
        detail += " " + " ".join(placement.not_checked)
        evidence["not_checked"] = list(placement.not_checked)
    if placement.left_in_place:
        evidence["left_in_place"] = list(placement.left_in_place)
    return {
        "code": "results_placed_beside_sheet",
        "kind": "provenance",
        "detail": detail,
        "evidence": evidence,
    }


def _kind(circuit: Path) -> str:
    return "sheet" if is_asc(circuit) else "netlist"


def _not_placed(
    job: ExperimentJob, case: ExperimentCase, reason: str, *, code: str
) -> dict[str, Any]:
    """The observation that ``case``'s results stayed in the job folder only;
    ``code`` says why, as the jobs action would refuse it."""
    return {
        "code": "results_not_placed",
        "kind": "provenance",
        "detail": (
            f"Case {case.case_id}'s results were not copied beside "
            f"{case.circuit_path.name}; they are in job {job.job_id}'s folder. {reason}"
        ),
        "evidence": {
            "sheet": str(case.circuit_path),
            "case_id": case.case_id,
            "reason_code": code,
        },
    }
