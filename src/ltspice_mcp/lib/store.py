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
    |   |-- recovery/{digest}.json          authoritative lineage admission
    |   |-- by-circuit/{digest}/{job_id}     which jobs used a circuit
    |   `-- cancellations/{job_id}.json     durable cancellation marker
    |-- runs/{job_id}/                      everything one job produced
    |   |-- startup/spinit                  controlled recoverable ngspice startup
    |   |-- startup/template.ini            immutable established LTspice settings
    |   |-- startup/{run_token}/LTspice.ini  one attempt's writable settings
    |   `-- staged/{circuit_id}/            the decks it actually ran
    |-- results/
    |   |-- {result_set_id}.json            an immutable analyze_results set
    |   `-- artifacts/{result_set_id}/      files those results point at
    |-- detached/                           per-job detached owner hand-off
    |   |-- {digest}.request.json           the request one owner was spawned for
    |   |-- {digest}.receipt.json           the receipt that owner reported back
    |   `-- {digest}.log                    that owner's stdout and stderr
    |-- verify/                             verify_circuit exports
    |-- edit-exports/{build_id}/            edit_schematic exports
    |-- exports/{name}.run-{hash}.net       schematic exports experiments ran
    |-- plots/                              plot_waveform charts with no out_dir
    |-- parsing/{parse_id}/                 temporary captured inputs and decoded arrays
    `-- locks/                              cross-process store locks

``$LTSPICE_MCP_STORE_DIR`` moves that root, whole, to one directory per working
directory under it (:func:`relocated_store_root`); the layout inside is the same.

Nothing the server keeps is written beside a user's circuit. Two things live
outside the root, each for a reason:

* **Run artifacts may not be there at all.** On WSL with LTspice the whole
  ``runs/`` tree moves to a Windows-native temp directory: LTspice is a Windows
  process reaching the Linux filesystem over a ``wsl.localhost`` UNC share, and
  SQLite — which is what ``.MEAS`` writes through — cannot write over UNC.
  :meth:`Store.artifact_base` is the single place that rule is applied; every
  writer asks it rather than re-deciding.
* **Per-user state lives in** :func:`user_home`, shared by every session this
  user runs wherever it was started: the recent-circuits index
  (``lib/recent.py``), so a session can surface prior work, and one lock file
  per circuit (:meth:`Store.circuit_lock`), so two sessions editing one file
  contend on one lock whatever their working directories.

Record shape: one schema name and one version for the whole store. Every
durable record this build writes carries ``{"schema": "ltspice-mcp/store",
"store_version": N, "kind": ...}``, where ``kind`` says which record it is and
``store_version`` moves as one number for all of them. Before this there were
six independently-versioned schemas for a subsystem that had never shipped, and
the version of a request index said nothing about the job record it pointed at.
Bump :data:`STORE_VERSION` when any record's shape changes; a record from a
version this build does not read is skipped with a warning, never guessed at.
"""

from __future__ import annotations

import functools
import hashlib
import json
import logging
import os
import re
import sys
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

#: The store's directory name inside a working directory.
STORE_DIRNAME = ".ltspice-mcp"

#: The one schema name every 0.6 store record carries.
STORE_SCHEMA = "ltspice-mcp/store"

#: The one version for the whole store. Bump it when ANY record's shape
#: changes; the manifest at the store root records which version wrote it.
STORE_VERSION = 4

#: Versions this build can read. Older entries appear here only once this build
#: can actually decode them.
#:
#: 2 differs from 3 only by what the experiment record lacks: the job's
#: ``simulator_executable`` and each case's ``simulator_version``. Both read as
#: unknown, which is what they are, and a replay of such a job is refused
#: because nothing shows it ran on the executable in use now. The bump is what
#: stops a version-2 build reading a version-3 record and dropping both fields
#: when it writes the record back.
#: 4 adds grouped frozen-input, lineage and execution facts. Older records
#: remain inspectable, with no recovery claim added while reading them.
SUPPORTED_STORE_VERSIONS: frozenset[int] = frozenset({2, 3, STORE_VERSION})

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
KIND_RECOVERY_JOURNAL = "recovery-journal"
# The two sides of the detached-owner hand-off. Transient rather than durable —
# both are consumed by the call that created them — but they live in the store
# tree and carry the same envelope, so a file that is not one of ours is
# refused rather than interpreted.
KIND_DETACHED_REQUEST = "detached-request"
KIND_DETACHED_RECEIPT = "detached-receipt"

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


def _record(directory: Path, name: str, error: str) -> Path:
    """One record file under an already-resolved store directory.

    An existing record comes back resolved and must resolve inside its
    directory (a planted symlink is refused with ``error``). One that does not
    exist yet is returned as named and NOT resolved: the name is already one
    validated segment, and on Windows resolving a path whose directory a peer
    is creating at that moment can come back with the ``\\?\\`` prefix still
    on, which no comparison survives.
    """
    candidate = directory / name
    try:
        resolved = candidate.resolve(strict=True)
    except OSError:
        return candidate
    if resolved.parent != directory:
        raise StoreError(error)
    return resolved


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


#: Overrides :func:`user_home` wherever the platform default would be.
USER_HOME_ENV = "LTSPICE_MCP_HOME"

#: Keeps each working directory's store under this directory instead of in it.
STORE_DIR_ENV = "LTSPICE_MCP_STORE_DIR"

_APP_DIRNAME = "ltspice-mcp"


def user_home() -> Path:
    """The per-user directory: what every session this user runs shares.

    Resolution order, first match wins:

    1. ``$LTSPICE_MCP_HOME`` (tests, and anyone who wants it elsewhere).
    2. On Windows, ``%LOCALAPPDATA%\\ltspice-mcp`` — local rather than roaming,
       because nothing here should follow the user to another machine.
    3. ``$XDG_STATE_HOME/ltspice-mcp``, else ``~/.local/state/ltspice-mcp``.

    Read on every call rather than once, so a test's environment override
    applies to the call it wraps.
    """
    override = os.getenv(USER_HOME_ENV)
    if override:
        return Path(override)
    if sys.platform == "win32":
        local = os.getenv("LOCALAPPDATA")
        base = Path(local) if local else Path.home() / "AppData" / "Local"
        return base / _APP_DIRNAME
    xdg = os.getenv("XDG_STATE_HOME")
    base = Path(xdg) if xdg else Path.home() / ".local" / "state"
    return base / _APP_DIRNAME


def _resolved(path: Path) -> Path:
    """``path`` resolved, or made absolute when it cannot be (a symlink loop)."""
    try:
        return path.resolve()
    except (OSError, RuntimeError):
        return path.absolute()


@functools.lru_cache(maxsize=64)
def relocated_store_root(store_dir: Path, working_dir: Path) -> Path:
    """One working directory's store, kept under ``store_dir`` instead of inside it.

    One store per working directory, named for the directory and keyed by the
    digest of its resolved path, so everything the working directory scopes
    stays scoped to it: the request index behind idempotency, the job listing,
    the result sets. What changes is who finds them. A process looks here only
    if it carries the same ``$LTSPICE_MCP_STORE_DIR``; one that does not — a
    script started without it, a server started from another client config —
    looks in ``{working_dir}/.ltspice-mcp`` and sees none of these records. The
    detached owners and ``run_code`` workers a session starts inherit its
    environment, so they follow it.

    A relative ``store_dir`` is taken from the working directory, so every
    process sharing that directory reads it the same way. Cached, because the
    working directory is resolved to name the store and this is asked for on
    every store path.
    """
    from ltspice_mcp.lib.sweep_utils import sanitize_stem

    base = store_dir if store_dir.is_absolute() else working_dir / store_dir
    resolved = _resolved(working_dir)
    readable = sanitize_stem(resolved.name) or "root"
    return base / f"{readable}-{path_digest(os.path.normcase(str(resolved)))[:16]}"


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

    An exited process whose parent has not collected it yet still holds its pid
    in the process table, and answers DEAD: it has stopped running, so it is no
    longer supervising anything. This is not a corner case since jobs can be
    detached — the process that spawned a detached owner is exactly the parent
    that has not collected it, and without this a job whose owner died would
    read as running for as long as that process lived.
    """
    if not pid:
        return OwnerLiveness.DEAD
    if pid == os.getpid():
        return OwnerLiveness.ALIVE if own_is_alive else OwnerLiveness.DEAD
    try:
        # One lookup answers both questions: constructing the Process raises
        # for a pid that is not in the table at all, and its status tells a
        # zombie from a running process.
        if psutil.Process(pid).status() == psutil.STATUS_ZOMBIE:
            return OwnerLiveness.DEAD
        return OwnerLiveness.ALIVE
    except psutil.NoSuchProcess:
        return OwnerLiveness.DEAD
    except psutil.Error:
        # The process exists but would not say what it is doing. Existing is
        # the answer this probe has always given on that evidence.
        return OwnerLiveness.ALIVE
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
        """The working directory's store root.

        ``{working_dir}/.ltspice-mcp`` unless ``$LTSPICE_MCP_STORE_DIR`` names a
        directory to keep stores in (see :func:`relocated_store_root`).
        """
        store_dir = os.getenv(STORE_DIR_ENV)
        if store_dir:
            return relocated_store_root(Path(store_dir).expanduser(), self.working_dir)
        return self.working_dir / STORE_DIRNAME

    @property
    def manifest(self) -> Path:
        """The version stamp that says which build's layout is on disk."""
        return self.root / MANIFEST_FILENAME

    def ensure_root(self) -> None:
        """Create the store root and stamp its version if it has none yet.

        Called from the write paths rather than at startup, so merely booting a
        server in a directory does not litter it with a store. Every durable
        submission reaches one of them before its job record lands — the
        request index is written under the idempotency gate first — so a store
        holding records always carries the stamp that says which build's layout
        it is.
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
        return _record(
            self.experiments_dir,
            f"{job_id}.json",
            f"Experiment job path escapes the store: {job_id!r}",
        )

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

    # -- detached owners ----------------------------------------------------

    @property
    def detached_dir(self) -> Path:
        """Hand-off files and console logs for per-job detached owner processes.

        A ``run_experiments(detach=True)`` call writes its request here, the
        owner it spawns writes the receipt back here, and the owner's stdout
        and stderr go to a log here. All three carry the ``request_id``'s
        digest so a person can find one call's files, and a per-call ``nonce``
        so two callers detaching the same id — the advertised idempotent
        replay — never read each other's report or share a log.
        """
        return self.root / "detached"

    def detached_request(self, request_id: str, nonce: str) -> Path:
        """The request a detached owner is spawned to run. Deleted once read."""
        return self.detached_dir / f"{self._handoff_stem(request_id, nonce)}.request.json"

    def detached_receipt(self, request_id: str, nonce: str) -> Path:
        """Where a detached owner reports its durable receipt, or its failure."""
        return self.detached_dir / f"{self._handoff_stem(request_id, nonce)}.receipt.json"

    def detached_log(self, request_id: str, nonce: str) -> Path:
        """One detached owner's console output."""
        return self.detached_dir / f"{self._handoff_stem(request_id, nonce)}.log"

    @staticmethod
    def _handoff_stem(request_id: str, nonce: str) -> str:
        return f"{path_digest(request_id)}.{_validate_name(nonce, 'handoff nonce')}"

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

    def recovery_journal(self, root_request_id: str) -> Path:
        """Authoritative lineage admission, locatable before derived indexes."""
        return _record(
            self.experiments_dir / "recovery",
            f"{path_digest(root_request_id)}.json",
            "Invalid recovery journal path",
        )

    def recovery_lock(self, root_request_id: str) -> Path:
        """The gate shared by lineage admission, reconstruction and deletion."""
        return self.lock(f"recovery-{path_digest(root_request_id)}")

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

        The output folder is part of ``RunnerManager``'s cache key, so a folder
        that varied per job would hand every job a different runner instance —
        and three things live on the instance: the handles of the simulator
        processes it has in flight, the per-job cancel state that stops a
        running batch, and the launch permits that enforce
        ``max_parallel_sims`` across every job in the process. Splitting the
        runner splits all three: cancel loses the job it was meant to stop, and
        each new instance admits a full fresh quota. Per-job grouping happens a
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

    def lineage_run_dir(
        self, root_job_id: str, recorded: Path, simulator: type | None = None
    ) -> Path:
        """Validate the retained initial artifact directory before reuse."""
        runs = self.runs_root(simulator).resolve()
        expected = run_dir_in(runs, root_job_id)
        resolved = expected.resolve()
        if not recorded.is_absolute() or recorded.resolve() != resolved or resolved != expected:
            raise StoreError("Recorded lineage directory is outside its Store run root")
        return resolved

    def recovery_spinit(self, root_job_id: str, simulator: type | None = None) -> Path:
        """Inert system startup input retained inside the initial run root."""
        return self.run_dir(root_job_id, simulator) / "startup" / "spinit"

    def recovery_ini_template(self, root_job_id: str, simulator: type | None = None) -> Path:
        """Immutable established settings retained for every LTspice attempt."""
        return self.run_dir(root_job_id, simulator) / "startup" / "template.ini"

    def recovery_ini(
        self, root_job_id: str, run_token: str, simulator: type | None = None
    ) -> Path:
        """Writable LTspice settings owned by exactly one case attempt."""
        return (
            self.run_dir(root_job_id, simulator)
            / "startup"
            / _validate_name(run_token, "run_token")
            / "LTspice.ini"
        )

    def staged_deck_root(
        self,
        job_id: str,
        circuit_id: str,
        simulator: type | None = None,
    ) -> Path:
        """Where one circuit's staged deck closure lands for one job."""
        return (
            self.run_dir(job_id, simulator) / "staged" / _validate_name(circuit_id, "circuit_id")
        )

    def native_input(self, job_id: str, run_token: str, simulator: type | None = None) -> Path:
        """Electrical input sourced by a native statistical setup."""
        return (
            self.run_dir(job_id, simulator) / f"{_validate_name(run_token, 'run_token')}.input.cir"
        )

    def native_driver(self, job_id: str, run_token: str, simulator: type | None = None) -> Path:
        """Prepared setup retained separately from the simulator's deck copy."""
        return (
            self.run_dir(job_id, simulator) / f"{_validate_name(run_token, 'run_token')}.setup.cir"
        )

    # -- analysis results ---------------------------------------------------

    @property
    def results_dir(self) -> Path:
        return (self.root / "results").resolve()

    def result_set(self, result_set_id: str) -> Path:
        """One immutable ``analyze_results`` set."""
        return _record(
            self.results_dir,
            f"{_validate_name(result_set_id, 'result_set_id')}.json",
            f"Invalid result_set_id: {result_set_id!r}",
        )

    def result_artifacts(self, result_set_id: str) -> Path:
        """Files one result set points at, deleted with it."""
        return self.results_dir / "artifacts" / _validate_name(result_set_id, "result_set_id")

    # -- authoring outputs --------------------------------------------------

    def verify_artifact(self, name: str) -> Path:
        """One ``verify_circuit`` export."""
        return self.root / "verify" / _validate_name(name, "verify artifact name")

    def edit_export(self, build_id: str) -> Path:
        """One ``edit_schematic`` export directory."""
        return self.root / "edit-exports" / _validate_name(build_id, "build id")

    @property
    def exports_dir(self) -> Path:
        """The netlists exported from schematics that experiments ran.

        In the store, beside the job records that name them: a receipt's replay
        identity names one of these files, so it lives exactly as long as the
        records do, and no schematic's folder collects one per distinct edit.
        Staging resolves a snapshot's relative includes beside its schematic
        (``deck_staging.stage_deck``).
        """
        return self.root / "exports"

    def export_snapshot(self, name: str) -> Path:
        """One content-addressed export snapshot."""
        return self.exports_dir / _validate_name(name, "export snapshot name")

    @property
    def plots_dir(self) -> Path:
        """Where ``plot_waveform`` writes a chart when the caller names no ``out_dir``.

        The response carries the path, and ``out_dir`` puts a chart beside the
        circuit for a caller who wants it there.
        """
        return self.root / "plots"

    def parser_dir(self, parse_id: str) -> Path:
        """One parser process's temporary capture and numeric output directory."""
        name = _validate_name(parse_id, "parser id")
        base = (self.root / "parsing").resolve()
        candidate = base / name
        if candidate.resolve() != candidate:
            raise StoreError("Parser directory must not redirect to another location")
        return candidate

    def parser_file(self, parse_id: str, name: str) -> Path:
        """One bounded parser input, output or control file."""
        return parser_file_in(self.parser_dir(parse_id), name)

    # -- per-user ----------------------------------------------------------
    #
    # Shared by every session this user runs, whatever its working directory,
    # so these are static: any Store names the same path.

    @staticmethod
    def circuit_lock(circuit_path: Path) -> Path:
        """Anchor for the cross-process lock on one circuit file (``file_lock`` appends .lock).

        Per user, so sessions started in different working directories contend
        on one lock per file, and nothing is written beside the circuit.

        Keyed by the digest of the resolved path, case-folded so two spellings
        of one file on a case-insensitive filesystem (Windows, macOS, a WSL
        ``/mnt/c`` mount) share their lock. On a case-sensitive one, two files
        differing only in case share one too: a wait, and past the timeout the
        "locked, retry" error, never a missed exclusion. Resolving is filesystem
        work, so an event-loop caller runs this off-loop.
        """
        resolved = _resolved(circuit_path)
        return user_home() / "locks" / "circuits" / path_digest(str(resolved).casefold())


def run_dir_in(runs_root: Path, job_id: str) -> Path:
    """One job's artifact directory inside a runner's stable output folder.

    Free function as well as a :class:`Store` method because the runner holds
    only its output folder — which may be the Windows-native one — and must be
    able to name a job's directory without re-deciding where the tree lives.
    """
    return runs_root / validate_job_id(job_id)


def parser_file_in(directory: Path, name: str) -> Path:
    """Name a file in the already admitted directory handed to a parser worker."""
    return _record(
        directory,
        _validate_name(name, "parser artifact name"),
        "Parser artifact must remain inside its temporary directory",
    )


def run_filename_in(job_id: str, name: str) -> str:
    """The ``run_filename`` that lands one run's artifacts in its job directory.

    spicelib copies the deck to ``output_folder / run_filename`` and derives the
    raw and log from that copy's own path, so a relative sub-path here groups a
    job's artifacts without the runner's output folder — and therefore the
    runner cache and its shared concurrency semaphore — ever changing. The
    directory must exist first; ``shutil.copy`` will not create it.
    """
    return f"{validate_job_id(job_id)}/{name}"
