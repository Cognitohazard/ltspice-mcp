"""The on-disk store: where everything lives, and what shape a record is in.

One object owns the layout. Every path this server writes is a method on
:class:`Store`, so the set of directories the server can create is the set of
methods here — a new root is a visible addition to this file, not a path
literal somewhere in a tool module. ``tests/test_store_layout.py`` snapshots
the result, so adding one is a deliberate change.

The layout, rooted at the working directory::

    {working_dir}/.ltspice-mcp/
    |-- store.json                          this store's version stamp
    |-- experiments/
    |   |-- {job_id}.json                   the job record
    |   |-- by-request/{digest}.json        request_id -> job_id (idempotency)
    |   |-- by-circuit/{digest}/{job_id}     which jobs used a circuit
    |   `-- cancellations/{job_id}.json     durable cancellation marker
    |-- runs/{job_id}/                      everything one job produced
    |   `-- staged/{circuit_id}/            the decks it actually ran
    |-- results/
    |   |-- {result_set_id}.json            an immutable analyze_results set
    |   `-- artifacts/{result_set_id}/      files those results point at
    |-- renders/                            schematic images
    |-- verify/                             verify_circuit exports
    |-- edit-exports/{build_id}/            edit_schematic exports
    `-- locks/                              cross-process store locks

Three things live outside that root, each for a reason:

* **Run artifacts may not be there at all.** On WSL with LTspice the whole
  ``runs/`` tree moves to a Windows-native temp directory: LTspice is a Windows
  process reaching the Linux filesystem over a ``wsl.localhost`` UNC share, and
  SQLite — which is what ``.MEAS`` writes through — cannot write over UNC.
  :meth:`Store.artifact_base` is the single place that rule is applied; every
  writer asks it rather than re-deciding.
* **Circuit-scoped sidecars** sit next to the user's file, because they belong
  to that file and not to whichever directory a session was started in: the
  export snapshots a receipt's provenance names, the per-circuit locks parallel
  sessions coordinate on, and the read-only job sidecars a pre-0.6 release
  wrote. See the ``circuit_*`` helpers.
* **The recent-circuits index is user-global** (``lib/recent.py``), so a session
  started anywhere can surface prior work.

Record shape: one schema name and one version for the whole store. Every
durable record this build writes carries ``{"schema": "ltspice-mcp/store",
"store_version": N, "kind": ...}``, where ``kind`` says which record it is and
``store_version`` moves as one number for all of them. Before this there were
six independently-versioned schemas for a subsystem that had never shipped, and
the version of a request index said nothing about the job record it pointed at.
Bump :data:`STORE_VERSION` when any record's shape changes; a record from a
version this build does not read is skipped with a warning, never guessed at.

Records written by a pre-0.6 release are a different story and are read, never
written, by ``lib/job_store.py``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import psutil

from ltspice_mcp.lib import atomic_write_json as _atomic_write_json
from ltspice_mcp.lib import now

logger = logging.getLogger(__name__)

#: The directory name every ltspice-mcp sidecar uses, working-dir or circuit.
SIDECAR_DIRNAME = ".ltspice-mcp"

#: The one schema name every 0.6 store record carries.
STORE_SCHEMA = "ltspice-mcp/store"

#: The one version for the whole store. Bump it when ANY record's shape
#: changes; the manifest at the store root records which version wrote it.
STORE_VERSION = 1

#: Versions this build can read. Older entries appear here only once this build
#: can actually decode them.
SUPPORTED_STORE_VERSIONS: frozenset[int] = frozenset({STORE_VERSION})

MANIFEST_FILENAME = "store.json"

# Record kinds. One constant per durable record so a typo is an import error.
KIND_MANIFEST = "store-manifest"
KIND_EXPERIMENT = "experiment"
KIND_REQUEST_INDEX = "request-index"
KIND_CIRCUIT_INDEX = "circuit-index"
KIND_CANCELLATION = "cancellation"
KIND_RESULT_SET = "result-set"
KIND_ANALYSIS_SNAPSHOT = "attached-analysis"
KIND_RECENT = "recent-circuits"

_JOB_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class StoreError(ValueError):
    """A store path or record could not be formed as asked."""


def validate_job_id(job_id: str) -> str:
    """Validate a server-generated job id before it reaches path construction."""
    if not isinstance(job_id, str) or _JOB_ID_RE.fullmatch(job_id) is None:
        raise ValueError(
            "Invalid job id: expected 1-64 letters, digits, underscores, or hyphens, "
            "starting with a letter or digit"
        )
    return job_id


def _validate_name(name: str, what: str) -> str:
    """Validate any other caller-supplied path segment."""
    if not isinstance(name, str) or _NAME_RE.fullmatch(name) is None:
        raise StoreError(f"Invalid {what}: {name!r}")
    return name


def path_digest(value: str) -> str:
    """Filesystem-safe digest for a value that must not become a path segment.

    Used for request ids (caller-supplied text) and circuit paths (absolute,
    and far longer than a filename may be).
    """
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Record envelopes
# ---------------------------------------------------------------------------


def json_default(obj: Any) -> Any:
    """Serialize the paths and datetimes store records carry."""
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Not JSON-serializable: {type(obj).__name__}")


def atomic_write_json(path: Path, data: Any) -> None:
    """Atomically and durably write a store record."""
    _atomic_write_json(path, data, default=json_default)


def envelope(kind: str, **payload: Any) -> dict[str, Any]:
    """Wrap one record in the store's single schema/version/kind envelope."""
    return {
        "schema": STORE_SCHEMA,
        "store_version": STORE_VERSION,
        "kind": kind,
        **payload,
    }


def accept(
    data: Any,
    source: Path | str,
    *,
    kind: str,
    log: logging.Logger | None = None,
) -> bool:
    """Whether a loaded document is a store record of ``kind`` this build reads.

    Three separate answers used to be three separate mechanisms; they are one
    check now. A document that is not this store's at all (a pre-0.6 sidecar, a
    neighbouring project's file) is rejected quietly by the caller that knows
    which directory it was reading — this function warns, because everything it
    is handed is supposed to be ours.
    """
    warn = (log or logger).warning
    if not isinstance(data, dict):
        warn("Skipping store record %s: not a JSON object", source)
        return False
    if data.get("schema") != STORE_SCHEMA:
        warn(
            "Skipping store record %s: unexpected schema %r (expected %s)",
            source,
            data.get("schema"),
            STORE_SCHEMA,
        )
        return False
    version = data.get("store_version")
    if not isinstance(version, int):
        warn("Skipping store record %s: store_version must be an integer", source)
        return False
    if version not in SUPPORTED_STORE_VERSIONS:
        warn(
            "Skipping store record %s: store_version %d (this build reads %s)",
            source,
            version,
            sorted(SUPPORTED_STORE_VERSIONS),
        )
        return False
    if data.get("kind") != kind:
        warn(
            "Skipping store record %s: expected a %r record, found %r",
            source,
            kind,
            data.get("kind"),
        )
        return False
    return True


def is_store_record(data: Any) -> bool:
    """Whether a document claims to be a store record at all, of any kind.

    Lets a directory scan tell "someone else's file" from "our record, wrong
    kind" without warning about the former.
    """
    return isinstance(data, dict) and data.get("schema") == STORE_SCHEMA


# ---------------------------------------------------------------------------
# Owning-process liveness
# ---------------------------------------------------------------------------


def pid_of(data: Mapping[str, Any]) -> int | None:
    """Owning-server pid from a stored record, or None if absent or invalid."""
    pid = data.get("pid")
    return pid if isinstance(pid, int) and pid > 0 else None


class OwnerLiveness(Enum):
    """What the liveness probe learned about a record's owning server process.

    Three answers, not two. Sessions share a working directory and read each
    other's job records, and the answer "the owner is gone" is what licenses a
    reader to rewrite a peer's running job as interrupted. A probe that could
    not reach an answer must therefore say so instead of reporting the process
    dead: it is the reading that takes a live run away from the session that
    owns it, and a transient probe error is not evidence of anything.
    """

    ALIVE = "alive"
    DEAD = "dead"
    UNKNOWN = "unknown"

    @property
    def is_dead(self) -> bool:
        """True only for a positive "the owner is gone" answer.

        Read the probe through this rather than negating ALIVE — ``not alive``
        folds UNKNOWN into dead, which is the whole defect.
        """
        return self is OwnerLiveness.DEAD


def owner_liveness(pid: int | None, *, own_is_alive: bool = False) -> OwnerLiveness:
    """Whether the record's owning server process is still running.

    ``own_is_alive`` decides how a record carrying this process's pid reads:
    registry loading treats it as a recycled pid, while disk-level summaries
    treat the common own-pid case as a genuinely running job.

    A record with no usable pid answers DEAD, not UNKNOWN: that is a record
    written before pids were stored, and the recovery of those interrupted
    jobs is the behaviour that predates this probe.
    """
    if not pid:
        return OwnerLiveness.DEAD
    if pid == os.getpid():
        return OwnerLiveness.ALIVE if own_is_alive else OwnerLiveness.DEAD
    try:
        return OwnerLiveness.ALIVE if psutil.pid_exists(pid) else OwnerLiveness.DEAD
    except Exception:
        # Deliberately broad, and deliberately NOT an answer: whatever went
        # wrong reaching the process table, the one thing this call must never
        # do is report a peer's live job dead because the probe itself failed.
        return OwnerLiveness.UNKNOWN


def owner_unknown_observation(pid: int | None) -> dict[str, str]:
    """The fact to surface when the liveness probe could not reach an answer."""
    return {
        "code": "owner_liveness_unknown",
        "kind": "lifecycle",
        "detail": (
            "Could not determine whether the owning server process "
            f"(pid {pid if pid else 'unrecorded'}) is still running; the status "
            "recorded by that server is kept as written."
        ),
    }


# ---------------------------------------------------------------------------
# Artifact routing
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ArtifactRoot:
    """Where one simulator's run artifacts may be written, and how paths read.

    ``windows_native`` is True when the tree sits on the Windows filesystem for
    a Windows simulator driven across the WSL boundary — that simulator needs
    any absolute path written into a deck spelled the Windows way.
    """

    runs: Path
    windows_native: bool = False


class WindowsNativeStorageUnavailable(StoreError):
    """WSL LTspice needs a Windows-native directory and none could be found."""


# ---------------------------------------------------------------------------
# The store
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Store:
    """Every path this server writes, rooted at one working directory.

    Cheap to construct (it holds a path and nothing else), so a caller that has
    a working directory can make one rather than thread an instance through.
    ``SessionState.store`` is the session's own.
    """

    working_dir: Path

    # -- root ---------------------------------------------------------------

    @property
    def root(self) -> Path:
        """The working directory's store root."""
        return self.working_dir / SIDECAR_DIRNAME

    @property
    def manifest(self) -> Path:
        """The version stamp that says which build's layout is on disk."""
        return self.root / MANIFEST_FILENAME

    def ensure_root(self) -> None:
        """Create the store root and stamp its version if it has none yet.

        Called from the durable write paths rather than at startup, so merely
        booting a server in a directory does not litter it with a store.
        """
        self.root.mkdir(parents=True, exist_ok=True)
        manifest = self.manifest
        if manifest.exists():
            return
        atomic_write_json(
            manifest,
            envelope(KIND_MANIFEST, created_at=now().isoformat()),
        )

    def store_version_on_disk(self) -> int | None:
        """The version stamped in the manifest, or None if there is no store."""
        try:
            data = json.loads(self.manifest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        version = data.get("store_version") if isinstance(data, dict) else None
        return version if isinstance(version, int) else None

    # -- experiment records -------------------------------------------------

    @property
    def experiments_dir(self) -> Path:
        """Where the job records live. One directory, one record kind."""
        return (self.root / "experiments").resolve()

    def job_record(self, job_id: str) -> Path:
        """The durable record for one experiment job."""
        validate_job_id(job_id)
        root = self.experiments_dir
        candidate = (root / f"{job_id}.json").resolve()
        if candidate.parent != root:
            raise StoreError(f"Experiment job path escapes the store: {job_id!r}")
        return candidate

    def request_index(self, request_id: str) -> Path:
        """Where a ``request_id`` records which job it already submitted.

        The raw request id is caller text and never becomes a path segment.
        """
        return self.experiments_dir / "by-request" / f"{path_digest(request_id)}.json"

    def circuit_index_dir(self, circuit_path: Path) -> Path:
        """The directory listing every job that ran one circuit.

        A directory of one file per job rather than one list file per circuit:
        registering a job is a single write and forgetting one a single unlink,
        so two sessions recording jobs for the same circuit cannot lose each
        other's entry to a read-modify-write race.
        """
        try:
            resolved = str(circuit_path.resolve())
        except OSError:
            resolved = str(circuit_path)
        return self.experiments_dir / "by-circuit" / path_digest(resolved)

    def circuit_index(self, circuit_path: Path, job_id: str) -> Path:
        """One job's entry in a circuit's index."""
        validate_job_id(job_id)
        return self.circuit_index_dir(circuit_path) / f"{job_id}.json"

    def cancellation(self, job_id: str) -> Path:
        """The durable "someone asked for this to stop" marker."""
        validate_job_id(job_id)
        return self.experiments_dir / "cancellations" / f"{job_id}.json"

    # -- locks --------------------------------------------------------------

    def lock(self, name: str) -> Path:
        """Anchor for a cross-process store lock (``file_lock`` appends .lock).

        Store locks live in one directory rather than beside the thing they
        guard, so a scan of the records never has to skip lock files.
        """
        return self.root / "locks" / _validate_name(name, "lock name")

    def request_lock(self, request_id: str) -> Path:
        """The gate that makes a ``request_id`` submit exactly one job."""
        return self.lock(f"request-{path_digest(request_id)}")

    def cancellation_lock(self, job_id: str) -> Path:
        """The gate shared by cancellation and case submission."""
        return self.lock(f"cancel-{validate_job_id(job_id)}")

    # -- run artifacts ------------------------------------------------------

    def artifact_base(self, simulator: type | None = None) -> ArtifactRoot:
        """Decide, once, where this simulator's run artifacts may be written.

        The rule, in one place because it is one rule: LTspice under WSL is a
        Windows process reaching the Linux filesystem over a ``wsl.localhost``
        UNC share, and the SQLite ``.db`` behind ``.MEAS`` cannot be written
        over UNC — so its whole ``runs/`` tree, staged decks included, moves to
        a Windows-native temp directory. Everything else stays in the store.
        """
        from spicelib.simulators.ltspice_simulator import LTspice

        from ltspice_mcp.lib import wsl

        is_ltspice = isinstance(simulator, type) and issubclass(simulator, LTspice)
        if wsl.is_wsl() and is_ltspice:
            windows_root = wsl.get_windows_output_dir()
            if windows_root is None:
                raise WindowsNativeStorageUnavailable(
                    "WSL LTspice experiments require a Windows-native directory for both "
                    "staged decks and simulator output, but no Windows temp directory "
                    "is available"
                )
            return ArtifactRoot(runs=windows_root / "experiments" / "runs", windows_native=True)
        return ArtifactRoot(runs=self.root / "runs")

    def runs_root(self, simulator: type | None = None) -> Path:
        """The runner's output folder — ONE per box, deliberately.

        ``RunnerManager`` caches a runner per (kind, simulator, output folder)
        and a runner owns the concurrency semaphore that enforces
        ``max_parallel_sims``, so a folder that varied per job would hand every
        job its own runner and its own full quota. Per-job grouping happens a
        level down, through :func:`run_filename_in`, which the simulator layer
        joins onto this folder without the runner ever changing.
        """
        return self.artifact_base(simulator).runs

    def run_dir(self, job_id: str, simulator: type | None = None) -> Path:
        """Everything one job produced, in its own directory.

        Sessions share this box, and before this grouping every session's raws,
        logs and staged decks sat together in one folder under job-id-prefixed
        names — which is how an agent asked to recover a crashed session
        inventories the folder, finds a peer's job tokens, and analyses another
        run believing it is its own. A directory per job does not make the
        files private, but it does mean a job's artifacts are enumerable as a
        set instead of by guessing at a filename prefix.
        """
        return run_dir_in(self.runs_root(simulator), job_id)

    def staged_deck_root(
        self,
        job_id: str,
        circuit_id: str,
        simulator: type | None = None,
    ) -> Path:
        """Where one circuit's staged deck closure lands for one job."""
        return self.run_dir(job_id, simulator) / "staged" / circuit_id

    # -- analysis results ---------------------------------------------------

    @property
    def results_dir(self) -> Path:
        return (self.root / "results").resolve()

    def result_set(self, result_set_id: str) -> Path:
        """One immutable ``analyze_results`` set."""
        root = self.results_dir
        path = (root / f"{_validate_name(result_set_id, 'result_set_id')}.json").resolve()
        if path.parent != root:
            raise StoreError(f"Invalid result_set_id: {result_set_id!r}")
        return path

    def result_artifacts(self, result_set_id: str) -> Path:
        """Files one result set points at, deleted with it."""
        return self.results_dir / "artifacts" / _validate_name(result_set_id, "result_set_id")

    # -- authoring outputs --------------------------------------------------

    @property
    def renders_dir(self) -> Path:
        """Schematic images an edit or a verify rendered."""
        return self.root / "renders"

    def verify_artifact(self, name: str) -> Path:
        """One ``verify_circuit`` export."""
        return self.root / "verify" / _validate_name(name, "verify artifact name")

    def edit_export(self, build_id: str) -> Path:
        """One ``edit_schematic`` export directory."""
        return self.root / "edit-exports" / _validate_name(build_id, "build id")

    # -- circuit-scoped sidecars -------------------------------------------
    #
    # These belong to the user's file rather than to a session's working
    # directory, so they are static: any Store can name them, and a session
    # whose working directory is elsewhere still finds them.

    @staticmethod
    def circuit_sidecar(circuit_path: Path) -> Path:
        """The sidecar directory beside one circuit file."""
        return circuit_path.parent / SIDECAR_DIRNAME

    @staticmethod
    def circuit_exports(circuit_path: Path) -> Path:
        """Content-addressed deck snapshots an experiment's provenance names.

        Beside the circuit, not in the working-dir store: a receipt's replay
        identity names one of these files, so it has to outlive the session,
        and it belongs to the schematic it was exported from.
        """
        return Store.circuit_sidecar(circuit_path) / "exports"

    @staticmethod
    def legacy_jobs_dir(circuit_path: Path) -> Path:
        """Where a pre-0.6 release wrote its job sidecars. Read, never written."""
        return Store.circuit_sidecar(circuit_path) / "jobs"


def run_dir_in(runs_root: Path, job_id: str) -> Path:
    """One job's artifact directory inside a runner's stable output folder.

    Free function as well as a :class:`Store` method because the runner holds
    only its output folder — which may be the Windows-native one — and must be
    able to name a job's directory without re-deciding where the tree lives.
    """
    return runs_root / validate_job_id(job_id)


def run_filename_in(job_id: str, name: str) -> str:
    """The ``run_filename`` that lands one run's artifacts in its job directory.

    spicelib copies the deck to ``output_folder / run_filename`` and derives the
    raw and log from that copy's own path, so a relative sub-path here groups a
    job's artifacts without the runner's output folder — and therefore the
    runner cache and its shared concurrency semaphore — ever changing. The
    directory must exist first; ``shutil.copy`` will not create it.
    """
    return f"{validate_job_id(job_id)}/{name}"
