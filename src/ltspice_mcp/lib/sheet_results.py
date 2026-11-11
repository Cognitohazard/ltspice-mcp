"""A run's results beside its sheet, where LTspice keeps a sheet's own.

LTspice writes a sheet's results beside it under its name (``<sheet>.raw``,
``<sheet>.log``, and ``<sheet>.op.raw`` or ``<sheet>.fra_1.raw`` where the run
wrote one) and opens them from there, so a click on a net in the sheet plots
it. The server's runs write theirs to the job folder instead, which LTspice
never looks in. This module copies one run's there on request, from
``jobs(action="place_beside_sheet")`` or, when a job asked for it with
``execution.results_beside_sheet``, as the job's runs finish. Both ask the
same questions here (``check_case``, ``place_case``): the results in the job
folder stay where they are, and remain what every read of the job takes.
"""

from __future__ import annotations

import asyncio
import os
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ltspice_mcp.errors import NetlistError
from ltspice_mcp.lib import atomic_write
from ltspice_mcp.lib.asc_document import is_asc
from ltspice_mcp.lib.deck_staging import sha256_file
from ltspice_mcp.lib.experiment_types import OWNER_STOPPED, ExperimentCase, ExperimentJob
from ltspice_mcp.lib.filelock import circuit_file_lock
from ltspice_mcp.lib.ltspice_window import results_beside
from ltspice_mcp.lib.raw_parser import other_raw_path
from ltspice_mcp.lib.simulator import dialect_for_simulator_name


class NotPlaced(Exception):
    """Why a run's results cannot stand as its sheet's own; ``code`` is the
    wire code the caller reports."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def place_beside_sheet(
    sheet: Path,
    raw: Path,
    log: Path | None,
    other_raws: Mapping[str, Path] | None = None,
) -> list[dict[str, Any]]:
    """Copy a run's results beside ``sheet`` under its name, as a run in
    LTspice leaves them; what is there is replaced, each file whole or not at
    all. Returns one ``{from, to, replaced}`` per file copied.

    A file of the sheet's that this run has no counterpart for (an
    ``.op.raw`` an earlier run left, say) is left where it is, and a file that
    already is the one beside the sheet is neither copied nor listed.
    """
    beside = results_beside(sheet)
    pairs = [(raw, beside), (log, sheet.with_suffix(".log"))]
    pairs += [(path, other_raw_path(beside, kind)) for kind, path in (other_raws or {}).items()]
    placed: list[dict[str, Any]] = []
    for source, target in pairs:
        if source is None or not source.is_file():
            continue
        replaced = target.exists()
        if replaced and os.path.samefile(source, target):
            continue
        with source.open("rb") as read, atomic_write(target, mode="wb", durable=False) as write:
            shutil.copyfileobj(read, write)
        placed.append({"from": str(source), "to": str(target), "replaced": replaced})
    return placed


def run_by_ltspice(simulator: str | None) -> bool:
    """Whether a run recorded as made by ``simulator`` can stand as a sheet's
    own in LTspice's window: an LTspice's, or one whose simulator is unknown.
    An ``.asc`` run on another simulator was only exported by LTspice."""
    return dialect_for_simulator_name(simulator) in (None, "ltspice")


def check_request(sheets: Sequence[tuple[Path, int]], simulator: str) -> None:
    """Refuse, before a job exists, a request to put its runs' results beside
    their sheets (``execution.results_beside_sheet``) that could not end with
    each sheet holding its own run: ``sheets`` pairs each circuit's path with
    the number of cases it runs, and ``simulator`` is the class the job runs on.

    A sheet holds one run's results, so a sheet that would run several cases
    (a sweep, or the same sheet named twice) is refused, with the action that
    copies a chosen case afterwards.
    """
    for path, _count in sheets:
        if not is_asc(path):
            raise NotPlaced(
                "not_a_sheet",
                f"execution.results_beside_sheet copies a sheet's results beside it, and "
                f"{path.name} is a netlist. Run it in a job of its own, or leave the option off.",
            )
    if not run_by_ltspice(simulator):
        raise NotPlaced(
            "not_ltspice",
            "execution.results_beside_sheet needs an LTspice run: LTspice's window opens a "
            f"sheet's results only as LTspice writes them, and this job runs on {simulator}. "
            "Leave the option off.",
        )
    counts: dict[Path, int] = {}
    for path, count in sheets:
        counts[path] = counts.get(path, 0) + count
    for path, count in counts.items():
        if count > 1:
            raise NotPlaced(
                "several_cases_per_sheet",
                f"A sheet holds one run's results, and {path.name} runs {count} cases in this "
                "job. Leave execution.results_beside_sheet off, and copy the case you want "
                "afterwards with jobs(action='place_beside_sheet', job_id=..., case_id=...).",
            )


def check_case(job: ExperimentJob, case: ExperimentCase) -> None:
    """Refuse, from the job's record alone, a case whose results LTspice's
    window could not take as its sheet's own: a case with no results, a
    netlist's run, and a run of another simulator."""
    if case.status != "produced" or case.raw_file is None:
        raise NotPlaced(
            "case_not_produced",
            f"Case {case.case_id} of job {job.job_id} has no results to place (status "
            f"{case.status}).",
        )
    if not is_asc(case.circuit_path):
        raise NotPlaced(
            "not_a_sheet",
            f"Case {case.case_id} ran {case.circuit_path.name}, a netlist. LTspice keeps "
            "results beside a sheet; read a netlist's from the job.",
        )
    if not run_by_ltspice(job.simulator):
        raise NotPlaced(
            "not_ltspice",
            f"Job {job.job_id} ran on {job.simulator}; LTspice's window opens a sheet's "
            "results only as LTspice writes them.",
        )


def place_case(job: ExperimentJob, case: ExperimentCase, sheet: Path) -> list[dict[str, Any]]:
    """Copy ``case``'s results beside ``sheet``, the sheet it ran
    (``place_beside_sheet``), or raise ``NotPlaced``. Besides ``check_case``,
    refused when the results or the sheet are gone, and when the sheet has
    changed since the job read it, since the results would not match it.

    Blocking: run it off the loop, under the sheet's ``circuit_file_lock``.
    """
    check_case(job, case)
    raw = case.raw_file
    assert raw is not None  # check_case refuses a case without one
    if not raw.is_file():
        raise NotPlaced(
            "results_missing", f"Case {case.case_id}'s results are not there any more ({raw})."
        )
    if not sheet.is_file():
        raise NotPlaced("sheet_missing", f"{sheet} is not there any more.")
    ran = next((source.sha256 for source in job.sources if source.circuit == case.circuit), None)
    if ran is not None and sha256_file(sheet) != ran:
        raise NotPlaced(
            "sheet_changed",
            f"{sheet.name} has changed since job {job.job_id} ran it, so these results "
            "would not match it. Run the sheet again and place that run's.",
        )
    others = {kind: other_raw_path(raw, kind) for kind in case.other_raws}
    return place_beside_sheet(sheet, raw, case.log_file, others)


async def place_job_beside_sheets(job: ExperimentJob) -> list[dict[str, Any]]:
    """Copy each produced case's results beside the sheet it ran, as
    ``execution.results_beside_sheet`` asks once the job's runs are done, and
    return one observation per sheet: what was copied, or why not.

    ``check_request`` has already held the job to one case per sheet and to
    sheets run in LTspice. A case that produced nothing leaves its sheet alone,
    as its own row already says. A refusal or a failed copy costs only its own
    sheet.
    """
    observations: list[dict[str, Any]] = []
    for case in job.cases:
        if case.status != "produced":
            continue
        sheet = case.circuit_path
        try:
            async with circuit_file_lock(sheet):
                # Shielded as the jobs action's copy is: a job stopped now
                # cannot leave a sheet with some of a run's files.
                files = await asyncio.shield(asyncio.to_thread(place_case, job, case, sheet))
        except NotPlaced as exc:
            observations.append(_not_placed(job, case, str(exc), code=exc.code))
            continue
        except NetlistError as exc:
            # Another server process held the sheet past the lock's wait.
            reason = f"{exc} {_place_later(job, case)}"
            observations.append(_not_placed(job, case, reason, code="sheet_locked"))
            continue
        except OSError as exc:
            reason = f"The copy failed ({exc}). {_place_later(job, case)}"
            observations.append(_not_placed(job, case, reason, code="copy_failed"))
            continue
        observations.append(_placed(case, sheet, files))
    return observations


def not_placed_for_owner_stop(job: ExperimentJob) -> list[dict[str, Any]]:
    """One ``results_not_placed`` per produced case, for a job whose process
    began to stop before the copy ``execution.results_beside_sheet`` asked for.

    The copy is not made then: it waits on each sheet's lock and copies whole
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
    """The call that copies ``case``'s results beside its sheet."""
    return f"jobs(action='place_beside_sheet', job_id='{job.job_id}', case_id='{case.case_id}')"


def _place_later(job: ExperimentJob, case: ExperimentCase) -> str:
    """The call that copies ``case``'s results once what stopped it is fixed."""
    return f"{_place_call(job, case)} copies them once that is fixed."


def sheets_reported(observations: Sequence[Mapping[str, Any]]) -> set[str]:
    """The sheets, as paths, that ``place_job_beside_sheets`` already reported
    on in a job's ``observations``, placed or not."""
    return {
        str(observation["evidence"]["sheet"])
        for observation in observations
        if observation.get("code") in ("results_placed_beside_sheet", "results_not_placed")
    }


def _placed(case: ExperimentCase, sheet: Path, files: list[dict[str, Any]]) -> dict[str, Any]:
    """The observation that ``case``'s results now sit beside ``sheet``."""
    names = ", ".join(Path(placed["to"]).name for placed in files)
    detail = (
        f"Case {case.case_id}'s results were copied beside {sheet.name}, under its name, "
        f"where LTspice's window opens them: {names}."
    )
    evidence: dict[str, Any] = {"sheet": str(sheet), "case_id": case.case_id, "files": files}
    if case.assignments:
        values = ", ".join(f"{name}={value}" for name, value in case.assignments.items())
        detail += f" The run used values the sheet does not show: {values}."
        evidence["assignments"] = dict(case.assignments)
    return {
        "code": "results_placed_beside_sheet",
        "kind": "provenance",
        "detail": detail,
        "evidence": evidence,
    }


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
