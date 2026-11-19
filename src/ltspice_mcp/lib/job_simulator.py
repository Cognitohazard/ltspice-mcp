"""Which simulator a job ran with, in the words a caller can pass back.

A job's replies name its simulator once, as ``simulator``: the selector
``run_experiments``' ``execution.simulator`` takes (``ltspice``, or
``ltspice:xvii`` for an entry of ``[simulator.executables]``). A job records
the selector its run resolved to (``ExperimentJob.simulator_selector``,
written from ``simulator.simulator_selector``), so the reply reads it back.

A record written before the selector was recorded holds only the class name
and the executable's identity. Its family follows from the class name; the
name of a named build follows only when the recorded executable's path is the
path of one of this session's simulators of that family. Otherwise the reply
gives the family and says the build's name was not recorded, with the path.

A selector is a name in the configuration of the session that ran the job.
When the session answering has no simulator under it, the reply says so. Both
are lookups in the session's tables: no reply reads or hashes an executable
here (``provenance`` carries the executable's identity for that).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final

from ltspice_mcp.lib.simulator import simulator_family
from ltspice_mcp.lib.simulator_build import executable_path

if TYPE_CHECKING:
    from ltspice_mcp.lib.experiment_types import ExperimentJob
    from ltspice_mcp.state import SessionState

#: The record names no build: written before a job recorded its selector, and
#: its executable is none of this session's simulators of its family.
BUILD_NOT_RECORDED: Final = "simulator_build_not_recorded"

#: The record names no simulator this server knows.
SIMULATOR_NOT_RECORDED: Final = "simulator_not_recorded"

#: The session answering has no simulator under the job's selector.
NOT_IN_SESSION: Final = "simulator_not_in_session"

#: Every code a ``simulator_notes`` entry carries.
NOTE_CODES: Final = (BUILD_NOT_RECORDED, SIMULATOR_NOT_RECORDED, NOT_IN_SESSION)


#: The program each launch command resolves to, by command: a bare command
#: name is searched for on PATH once per process, not once per reply.
_LAUNCHED: dict[tuple[str, ...], str | None] = {}


def _launched(simulator: type) -> str | None:
    """The program ``simulator`` launches (``executable_path``), remembered
    by its launch command, which a session's classes keep from startup."""
    command = tuple(str(part) for part in getattr(simulator, "spice_exe", None) or ())
    if command not in _LAUNCHED:
        _LAUNCHED[command] = executable_path(simulator)
    return _LAUNCHED[command]


def _same_path(current: str | None, recorded: str) -> bool:
    """Whether two paths name one file as the platform compares them: on
    Windows, without regard to case."""
    return current is not None and os.path.normcase(current) == os.path.normcase(recorded)


@dataclass(frozen=True)
class JobSimulator:
    """A job's simulator as its replies name it."""

    selector: str | None
    notes: tuple[dict[str, str], ...] = ()

    def fields(self) -> dict[str, Any]:
        """The reply fields: ``simulator`` and, when there is something to
        say about it, ``simulator_notes``."""
        return {"simulator": self.selector, "simulator_notes": [dict(n) for n in self.notes]}


def _legacy_selector(job: ExperimentJob, state: SessionState | None) -> JobSimulator:
    """The selector a record written before selectors were recorded
    establishes, and what it does not establish."""
    # A class name this server's families are made of, else that of one of
    # the session's own classes (a subclass keeps its family's lineage).
    family = simulator_family(job.simulator) or next(
        (
            simulator_family(cls)
            for cls in (state.offered_simulators.values() if state is not None else ())
            if cls.__name__ == job.simulator
        ),
        None,
    )
    if family is None:
        return JobSimulator(
            None,
            (
                {
                    "code": SIMULATOR_NOT_RECORDED,
                    "detail": (
                        f"This job's record names its simulator {job.simulator!r}, which is "
                        "none of the simulator families this server runs."
                        if job.simulator
                        else "This job's record names no simulator."
                    ),
                },
            ),
        )
    recorded = job.simulator_executable.path if job.simulator_executable else None
    if recorded is not None and state is not None:
        offered = state.offered_simulators
        # The family's own simulator first: a record that ran it was named by
        # the family, whatever named entries share its executable.
        ordered = [family] if family in offered else []
        ordered += sorted(key for key in offered if key.startswith(f"{family}:"))
        for key in ordered:
            if _same_path(_launched(offered[key]), recorded):
                return JobSimulator(key)
    note = {
        "code": BUILD_NOT_RECORDED,
        "detail": (
            f"This job's record was written before a job recorded the selector it ran "
            f"with, and its executable is none of this session's {family} simulators, "
            f"so which {family} build it named is not recorded; '{family}' is its "
            "family."
        ),
    }
    if recorded:
        note["executable"] = recorded
    return JobSimulator(family, (note,))


def job_simulator(job: ExperimentJob, state: SessionState | None) -> JobSimulator:
    """Which simulator ``job`` ran with, as the selector a caller passes back
    as ``execution.simulator``, with what the session answering says of it.

    The one answer for a reply that names a job's simulator: the receipt, the
    ``jobs`` replies and the rows of ``jobs(action="list")`` all ask it.
    """
    found = (
        JobSimulator(job.simulator_selector)
        if job.simulator_selector
        else _legacy_selector(job, state)
    )
    selector = found.selector
    if selector is None or state is None or selector in state.offered_simulators:
        return found
    return JobSimulator(
        selector,
        (
            *found.notes,
            {
                "code": NOT_IN_SESSION,
                "detail": (
                    f"This session has no simulator named '{selector}', so "
                    f"execution.simulator='{selector}' is refused here; "
                    "inspect(kind='capabilities') lists the selectors it offers."
                ),
            },
        ),
    )
