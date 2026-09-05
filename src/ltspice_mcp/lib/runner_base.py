"""Shared scaffolding for simulation/sweep/Monte-Carlo runners.

The three runners all wrap spicelib's SimRunner with the same asyncio
integration pattern: a blocking submit runs in ``asyncio.to_thread``;
per-run callbacks fire in worker threads and bridge back to the event
loop via ``call_soon_threadsafe``; cancel sets an event + kills
spice processes. This module factors that shared machinery out so each
subclass only implements what's genuinely different — stepper setup
for sweeps, tolerance configuration for Monte Carlo, single-job
tracking for sim.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import re
import threading
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

from spicelib.sim.sim_runner import SimRunner

from ltspice_mcp.lib.encoding import read_spice_text
from ltspice_mcp.lib.log_parser import (
    classify_failure_code,
    extract_error_context,
    extract_log_diagnostics,
    is_op_stepping_failure,
    op_ladder_exhausted,
)
from ltspice_mcp.lib.proc_kill import kill_simulator_by_token, simulator_executable_names
from ltspice_mcp.lib.spice_validator import ANALYSIS_KINDS
from ltspice_mcp.lib.wsl import kill_windows_ltspice_by_token

if TYPE_CHECKING:
    pass

# Trailing `_<digits>` in spicelib-generated raw/log filenames. spicelib's
# SimRunner._run_file_name produces "<stem>_<runno><suffix>" (1-based runno).
_RUNNO_RE = re.compile(r"_(\d+)$")

# Marker embedded in the name of a generated '.options logopinfo' netlist copy
# (built by tools._base.inject_logopinfo). All cleanup keys on it, so the
# producer and every consumer share one constant — change the name scheme here
# and both sides move together.
LOGOPINFO_MARKER = ".logopinfo"

# Same idea for a generated ngspice ".control" write-injection copy (built by
# tools._base.inject_ngspice_control_write).
NGSPICE_CONTROL_WRITE_MARKER = ".ctrlwrite"

_GENERATED_NETLIST_MARKERS = (LOGOPINFO_MARKER, NGSPICE_CONTROL_WRITE_MARKER)

# Fallback concurrency cap used when a caller doesn't pass ``max_parallel``.
# The real cap comes from ``config.max_parallel_sims``; this default only
# applies to direct runner construction (mostly tests). Every runner
# constructor and the RunnerManager factory methods share this one value.
DEFAULT_MAX_PARALLEL = 4


class RunOutcome(NamedTuple):
    """Filesystem-derived facts about a finished run, collected off the loop."""

    raw_file: str
    log_file: str
    raw_size: int
    error: str | None
    observations: tuple[dict, ...] = ()
    failure_code: str | None = None
    failure_evidence: dict[str, Any] | None = None


_RAW_PRODUCING_ANALYSES: frozenset[str] = frozenset(f".{kind}" for kind in ANALYSIS_KINDS)
_INCLUDE_DIRECTIVES: frozenset[str] = frozenset({".include", ".inc", ".lib"})
_MAX_INCLUDE_DEPTH = 3


def _include_target(rest: str) -> str | None:
    """Return the file token from an include or library directive."""
    rest = rest.strip()
    if not rest:
        return None
    if rest[0] in "\"'":
        end = rest.find(rest[0], 1)
        return rest[1:end] if end != -1 else None
    return rest.split(None, 1)[0]


def deck_requests_raw(netlist: Path | None) -> tuple[list[str], bool]:
    """Snapshot a deck's raw-producing analyses and ``.save`` presence."""
    if netlist is None:
        return [], False
    analyses: list[str] = []
    has_save = False
    has_control = False
    seen: set[Path] = set()

    def scan(path: Path, depth: int) -> None:
        nonlocal has_save, has_control
        if depth > _MAX_INCLUDE_DEPTH:
            return
        try:
            key = path.resolve()
        except OSError:
            return
        if key in seen:
            return
        seen.add(key)
        try:
            text = read_spice_text(path)
        except OSError:
            return
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped.startswith("."):
                continue
            parts = stripped.split(None, 1)
            head = parts[0].lower()
            if head == ".end":
                break
            if head == ".control":
                has_control = True
            elif head in _RAW_PRODUCING_ANALYSES:
                if head not in analyses:
                    analyses.append(head)
            elif head == ".save":
                has_save = True
            elif head in _INCLUDE_DIRECTIVES and len(parts) > 1:
                target = _include_target(parts[1])
                if target is not None:
                    scan(path.parent / target, depth + 1)

    scan(netlist, 0)
    if has_control:
        return [], has_save
    return analyses, has_save


def _missing_required_raw_outcome(
    log_file: str,
    log_path: Path,
    analyses: list[str],
    has_save: bool,
) -> RunOutcome:
    """Build the failure facts for an expected but absent raw artifact."""
    analysis_str = "/".join(analyses)
    excerpt = extract_error_context(log_path, max_lines=20)
    if has_save:
        workaround = (
            " The deck sets a '.save' list; if it omits nodes the analysis "
            "probes, LTspice 26.0.2 has been observed to exit 0 without writing "
            "a .raw. List every probed node in the .save (or remove the .save "
            "directive) and re-run — a full .save list is the known workaround."
        )
    else:
        workaround = (
            " The simulator reported no error, so a re-run may succeed; if it "
            "recurs, check the analysis directive and any .save list."
        )
    excerpt_block = f"\n\nLog excerpt:\n{excerpt}" if excerpt else ""
    error = (
        "Simulation exited cleanly but produced no .raw waveform file, which the "
        f"deck's {analysis_str} analysis requires — the waveform results are "
        f"absent.{workaround}{excerpt_block}"
    )
    observation = {
        "code": "missing_required_raw",
        "kind": "reconciliation",
        "detail": (
            f"The deck requested a {analysis_str} analysis but the simulator "
            "exited without writing a .raw file; waveform results are absent."
        ),
        "evidence": {
            "expected_artifact": "raw",
            "analyses": analyses,
            "has_save_list": has_save,
        },
    }
    return RunOutcome("", log_file, 0, error, observations=(observation,))


def collect_run_outcome(
    raw_file: str,
    log_file: str,
    requirements: tuple[list[str], bool] | None = None,
    exit_code: int | None = None,
) -> RunOutcome:
    """Collect and classify completion artifacts on a worker thread.

    ``exit_code`` is the simulator process's own exit status, relayed as a
    fact when the run failed — the one signal that separates a process killed
    from outside from a deck the simulator declined.
    """
    log_path = Path(log_file)
    sim_failed = raw_file in ("", ".") or log_path.suffix == ".fail"
    raw_size = 0
    if not sim_failed:
        try:
            raw_size = Path(raw_file).stat().st_size
        except FileNotFoundError:
            raw_size = 0
        except OSError as exc:
            return RunOutcome(
                raw_file,
                log_file,
                0,
                f"Simulation finished but its raw file is unreadable: {exc}",
            )
    if raw_size > 0:
        return RunOutcome(raw_file, log_file, raw_size, None)

    try:
        log_exists = bool(log_file) and log_path.exists()
    except OSError:
        log_exists = False
    errors = extract_log_diagnostics(log_path)["errors"] if log_exists else []
    # The clean-exit branch below asserts "exited cleanly" from log-error
    # ABSENCE, so a killed process with a quiet log would be reported as a
    # clean exit — gate it on the one direct fact about exit status.
    if exit_code in (None, 0) and not sim_failed and log_exists:
        non_rung = [error for error in errors if not is_op_stepping_failure(error)]
        if not non_rung and not op_ladder_exhausted(errors):
            analyses, has_save = requirements if requirements is not None else ([], False)
            if not analyses:
                return RunOutcome("", log_file, 0, None)
            return _missing_required_raw_outcome(log_file, log_path, analyses, has_save)

    if log_exists:
        context = extract_error_context(log_path, max_lines=20)
        error = f"Simulation failed (no output generated)\n\nLog excerpt:\n{context}"
    else:
        error = "Simulation failed (no output generated, log file missing)"
    # The diagnostics above already name the cause; classifying here is what
    # turns it into a code a caller can branch on instead of prose it must read.
    code, evidence = classify_failure_code(errors)
    if exit_code not in (None, 0):
        error += f"\nSimulator exit code: {exit_code}"
        # Copy rather than mutate: classify_failure_code's return is typed
        # narrower than the relayed shape, and this is the cold path.
        evidence = {**(evidence or {}), "exit_code": exit_code}
    return RunOutcome(
        "" if sim_failed else raw_file,
        log_file,
        0,
        error,
        failure_code=code,
        failure_evidence=evidence,
    )


def inject_logopinfo(netlist_path: Path, simulator: type, job_id: str) -> Path:
    """Return a runnable netlist with ``.options logopinfo`` added, for LTspice ``.op`` runs.

    LTspice writes each semiconductor's small-signal operating point (gm, gds,
    vth, vdsat, junction caps) to the ``.log`` only under ``.options logopinfo``,
    and only for ``.op`` analyses — so adding it lets ``operating_point`` read
    those params back by name. ngspice uses ``@dev[param]`` raw traces instead
    and needs nothing here.

    Append-only into a per-job sibling file (a leading-dot, ``job_id``-stamped
    name) so the simulator sees the caller's deck byte-for-byte plus the one
    directive; the original is never touched and relative ``.include``/``.lib``
    paths still resolve from the same directory. That untouched original is what
    lets the experiments path inject at submit time without moving the staged
    deck's recorded digest. The ``job_id`` stamp keeps two concurrent or queued
    runs of the same netlist from clobbering each other's augmented copy; the
    submitting caller deletes it once spicelib has staged the run. Returns the
    original path unchanged when injection doesn't apply (non-LTspice, non-text
    netlist, no ``.op``, or ``logopinfo`` already present) or the sibling can't
    be written.
    """
    from spicelib.simulators.ltspice_simulator import LTspice

    if not (isinstance(simulator, type) and issubclass(simulator, LTspice)):
        return netlist_path
    if netlist_path.suffix.lower() not in (".cir", ".net", ".sp"):
        return netlist_path
    try:
        data = netlist_path.read_bytes()
    except OSError:
        return netlist_path

    # Detect on the raw bytes (the directives are ASCII) — same plane the .end
    # splice below works on, so no decode round-trip is needed.
    if b"logopinfo" in data.lower():
        return netlist_path
    # ``.op\b`` excludes ``.options`` (the 't' blocks the word boundary); only a
    # real .op analysis emits the operating-point block. ``.dc`` does not.
    if not re.search(rb"(?im)^[ \t]*\.op\b", data):
        return netlist_path

    # Byte-level insertion before the final ``.end`` keeps the original encoding
    # intact (the added line is pure ASCII). ``.end\b`` skips ``.ends``.
    line = b".options logopinfo\n"
    ends = list(re.finditer(rb"(?im)^[ \t]*\.end\b.*$", data))
    if ends:
        at = ends[-1].start()
        augmented = data[:at] + line + data[at:]
    else:
        augmented = data + (b"" if not data or data.endswith(b"\n") else b"\n") + line

    run_path = netlist_path.with_name(
        f".{netlist_path.stem}.{job_id}{LOGOPINFO_MARKER}{netlist_path.suffix}"
    )
    try:
        run_path.write_bytes(augmented)
    except OSError:
        return netlist_path
    return run_path


def discard_generated_netlist(path: Path | None) -> None:
    """Delete a generated per-job netlist copy (an ``.options logopinfo``
    injection or an ngspice ``.control`` write injection). No-op when ``path``
    is None or carries neither marker, so this can only ever remove a
    generated copy, never the user's own deck."""
    if path is not None and any(marker in path.name for marker in _GENERATED_NETLIST_MARKERS):
        with contextlib.suppress(OSError):
            path.unlink()


logger = logging.getLogger(__name__)

_SIMRUNNER_TIMEOUT = 600
"""Generous spicelib-level fallback timeout; real timeout is enforced at
the tool layer via ``asyncio.wait_for``."""

_CANCEL_KILL_MAX_PASSES = 5
"""Upper bound on cancel's kill/re-scan passes (see ``BatchRunnerBase.cancel``)."""

_CANCEL_KILL_RESCAN_DELAY = 0.5
"""Seconds between cancel kill passes — long enough for a resumed submission's
process to become visible to the next scan."""


class RunnerBase:
    """Shared constructor + thread-safe callback bridging."""

    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        simulator_class: type,
        output_folder: Path,
        max_parallel: int = DEFAULT_MAX_PARALLEL,
    ):
        self.loop = loop
        self.simulator_class = simulator_class
        self.output_folder = output_folder
        self.max_parallel = max_parallel
        # Submitted SimRunners, held until their simulation thread is done.
        # See _retire_finished_runners for why letting one go early is a trap.
        self._inflight_runners: dict[str, SimRunner] = {}

    def _build_sim_runner(self) -> SimRunner:
        """Construct a spicelib SimRunner with this runner's settings."""
        return SimRunner(
            simulator=self.simulator_class,
            output_folder=str(self.output_folder),
            parallel_sims=self.max_parallel,
            timeout=_SIMRUNNER_TIMEOUT,
        )

    def _kill_by_token(self, token: str, context_label: str = "") -> None:
        """Best-effort blocking termination scoped to a command-line token."""
        subject = f"{context_label} {token}".strip()
        try:
            killed = kill_windows_ltspice_by_token(token)
            if killed:
                logger.info("Killed %d Windows sim process(es) for %s", killed, subject)
        except Exception as exc:
            logger.warning("WSL process kill for %s failed: %s", subject, exc)
        try:
            killed = kill_simulator_by_token(
                token,
                simulator_executable_names(self.simulator_class),
            )
            if killed:
                logger.info("Killed %d local sim process(es) for %s", killed, subject)
        except Exception as exc:
            logger.warning("Scoped process kill for %s failed: %s", subject, exc)

    def submit_netlist(
        self,
        netlist: Path,
        run_filename: str,
        callback: Callable[[Any], Any],
    ) -> SimRunner:
        """Submit one deck and bridge its filesystem-derived outcome to the loop.

        This is the job-agnostic single-run primitive shared by legacy
        ``SimulationRunner`` jobs and experiment cases. It knows only the deck,
        the simulator-facing filename, and an event-loop callback; registration,
        lifecycle, persistence, and concurrency remain with its callers.

        Call from a worker thread. The requirements snapshot and completion
        artifact reads intentionally happen on spicelib's worker threads.

        The SimRunner is returned AND retained here. spicelib's
        ``SimRunner.__del__`` calls ``wait_completion()``, so a caller that
        discards the return value has its thread pinned inside the destructor
        for the entire simulation — the calling coroutine never resumes, so it
        never reaches the code that watches for a cancellation, and the run
        becomes unstoppable. Retaining at this choke point means no caller can
        re-arm that by forgetting to keep it.
        """
        requirements = deck_requests_raw(netlist)

        def completion_callback(raw_file: Path | None, log_file: Path | None) -> None:
            # This runner is fresh per submission, so active_tasks holds
            # exactly this run's task — appended before its thread starts,
            # so it is present whenever the callback can fire.
            try:
                outcome = collect_run_outcome(
                    str(raw_file) if raw_file else "",
                    str(log_file) if log_file else "",
                    requirements,
                    # spicelib invokes the callback from the RunTask's own
                    # thread, and the task IS a Thread subclass carrying its
                    # retcode — so the current thread is the exact task,
                    # race-free. Any other calling thread reads None.
                    exit_code=getattr(threading.current_thread(), "retcode", None),
                )
            except Exception as exc:
                outcome = RunOutcome(
                    "",
                    "",
                    0,
                    f"Simulation failed (outcome collection: {exc})",
                )
            self._bridge(callback, outcome, context=f"run {run_filename}")

        self._retire_finished_runners()
        runner = self._build_sim_runner()
        runner.run(
            str(netlist),
            run_filename=run_filename,
            callback=completion_callback,
            callback_on_error=True,
            exe_log=True,
        )
        self._inflight_runners[run_filename] = runner
        return runner

    def _retire_finished_runners(self) -> None:
        """Release SimRunners whose simulation threads have all exited.

        Pruned on the way into the next submission rather than from a
        completion callback: dropping the last reference runs
        ``SimRunner.__del__`` -> ``wait_completion()``, which waits on
        ``active_tasks`` — from inside a task's own callback that would be the
        task waiting for itself. Liveness is read off the RunTask threads
        instead of spicelib's bookkeeping, which only updates when something
        calls into it.
        """
        for key, runner in list(self._inflight_runners.items()):
            if not any(task.is_alive() for task in runner.active_tasks):
                self._inflight_runners.pop(key, None)

    def _bridge(self, handler: Callable[..., Any], *args: Any, context: str = "") -> bool:
        """Schedule ``handler`` on the event loop from a worker thread.

        Returns True on success, False if the loop is closed (graceful
        shutdown in progress). The ``context`` string appears in the
        warning message when the bridge fails.
        """
        try:
            self.loop.call_soon_threadsafe(handler, *args)
        except RuntimeError as e:
            logger.warning(
                "Event loop closed, %s not recorded: %s",
                context or "callback",
                e,
            )
            return False
        return True
