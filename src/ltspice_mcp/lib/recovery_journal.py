"""Authoritative recovery admission, with snapshots only until launch.

The caller holds the Store recovery lock across each read/modify/write transaction.
These blocking helpers neither acquire locks nor claim, adopt or launch jobs. The
request digest locates the root before its derived job record or index exists.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, ClassVar, Literal, Self

from pydantic import ConfigDict, TypeAdapter, ValidationError

from ltspice_mcp.lib import experiment_store
from ltspice_mcp.lib.experiment_types import ExperimentJob
from ltspice_mcp.lib.recovery_records import RecoveryError
from ltspice_mcp.lib.store import (
    KIND_EXPERIMENT,
    KIND_RECOVERY_JOURNAL,
    STORE_SCHEMA,
    STORE_VERSION,
    OwnerLiveness,
    Store,
    atomic_write_json,
    envelope,
    json_default,
    path_digest,
    validate_job_id,
)

_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_ENVELOPE_FIELDS = frozenset({"schema", "store_version", "kind"})


def _invalid(message: str) -> RecoveryError:
    return RecoveryError("recovery_journal_invalid", message)


def _request_id(value: str) -> None:
    if type(value) is not str or not value:
        raise _invalid("Journal request IDs must be nonempty strings")
    try:
        value.encode("utf-8")
    except UnicodeError as exc:
        raise _invalid("Journal request IDs must encode as UTF-8") from exc


def _job_id(value: str) -> None:
    try:
        validate_job_id(value)
    except ValueError as exc:
        raise _invalid("Invalid journal job ID") from exc


class _Record:
    __pydantic_config__: ClassVar[ConfigDict] = ConfigDict(extra="forbid", strict=True)

    def to_record(self) -> dict[str, Any]:
        """Return an independent JSON snapshot of the journal payload."""
        return TypeAdapter(type(self)).dump_python(self, mode="json")

    @classmethod
    def from_record(cls, data: Any) -> Self:
        """Reject unknown fields and type coercion in journal metadata."""
        try:
            return TypeAdapter(cls).validate_json(
                json.dumps(data, default=json_default, allow_nan=False), strict=True
            )
        except (ValidationError, ValueError, TypeError, RecursionError) as exc:
            raise _invalid("Invalid recovery journal payload") from exc


@dataclass(frozen=True)
class JournalEntry(_Record):
    request_id: str
    fingerprint: str
    parent_job_id: str | None
    job_id: str | None
    attempt_index: int
    phase: Literal["prepared", "launched", "noop"]
    candidate: dict[str, Any] | None

    def __post_init__(self) -> None:
        _request_id(self.request_id)
        if type(self.fingerprint) is not str or not _DIGEST.fullmatch(self.fingerprint):
            raise _invalid("Journal fingerprints must be lowercase SHA256 digests")
        if type(self.attempt_index) is not int or self.attempt_index < 0:
            raise _invalid("Journal attempt indices must be nonnegative integers")
        if self.parent_job_id is not None:
            _job_id(self.parent_job_id)
        if self.phase == "noop":
            if self.job_id is not None or self.candidate is not None or self.parent_job_id is None:
                raise _invalid("No-op entries require a parent and cannot contain a child")
            return
        if self.phase not in {"prepared", "launched"} or self.job_id is None:
            raise _invalid("Child entries require a job ID and a valid launch phase")
        _job_id(self.job_id)
        if self.phase == "prepared" and not isinstance(self.candidate, dict):
            raise _invalid("Prepared entries require the complete prelaunch candidate")
        if self.candidate is not None:
            _decode_candidate(self, Path("candidate.json"))


@dataclass(frozen=True)
class RecoveryJournal(_Record):
    root_request_id: str
    root_job_id: str
    head_job_id: str
    root: JournalEntry
    resumes: dict[str, JournalEntry]

    def __post_init__(self) -> None:
        _validate_journal(self)


def _check_envelope(data: Any, kind: str) -> None:
    if (
        not isinstance(data, dict)
        or data.get("schema") != STORE_SCHEMA
        or type(data.get("store_version")) is not int
        or data["store_version"] != STORE_VERSION
        or data.get("kind") != kind
    ):
        raise _invalid("Recovery evidence requires the current Store envelope")


def _decode_candidate(entry: JournalEntry, store_path: Path) -> ExperimentJob:
    data = entry.candidate
    if data is None:
        raise _invalid("This entry has no reconstructable candidate")
    _check_envelope(data, KIND_EXPERIMENT)
    if (
        data.get("job_id") != entry.job_id
        or data.get("request_id") != entry.request_id
        or data.get("fingerprint") != entry.fingerprint
    ):
        raise _invalid("Candidate identity disagrees with its journal entry")
    try:
        job = experiment_store.deserialize_job(data, store_path, liveness=OwnerLiveness.ALIVE)
        # The legacy decoder permits defaults and coercions. Compare against the
        # shared serializer to require a complete, lossless candidate instead.
        serialized = _snapshot(job)
        if not _same_json(data, serialized):
            raise _invalid("Candidate must decode without dropped, defaulted or coerced fields")
    except (ValueError, TypeError, KeyError, AttributeError, OverflowError) as exc:
        raise _invalid("Invalid serialized recovery candidate") from exc
    recovery = job.recovery
    if (
        recovery is None
        or recovery.parent_job_id != entry.parent_job_id
        or recovery.attempt_index != entry.attempt_index
    ):
        raise _invalid("Candidate parent or attempt disagrees with its journal entry")
    if type(data["pid"]) is not int or data["pid"] != recovery.owner.pid:
        raise _invalid("Candidate owner PID disagrees with its recorded process identity")
    if (
        job.simulator_executable is not None
        and job.simulator_executable != recovery.execution.executable
    ):
        raise _invalid("Candidate executable disagrees with its recorded execution")
    case_ids: set[str] = set()
    run_tokens: set[str] = set()
    for case in job.cases:
        if case.recovery is None:
            raise _invalid("Candidate case is missing frozen recovery facts")
        attempt = case.recovery.attempt
        if (
            case.case_id in case_ids
            or case.run_token in run_tokens
            or case.run_token != attempt.run_token
        ):
            raise _invalid("Candidate case identities and run tokens must be distinct and bound")
        case_ids.add(case.case_id)
        run_tokens.add(case.run_token)
        if (
            entry.phase == "prepared"
            and attempt.execution_job_id == job.job_id
            and (attempt.launch is not None or attempt.outputs is not None or attempt.reused)
        ):
            raise _invalid("Prepared candidates cannot contain current-attempt execution facts")
    return job


def _same_json(left: Any, right: Any) -> bool:
    """Compare JSON types without conflating booleans with numeric values."""
    if type(left) in {int, float} and type(right) in {int, float}:
        return left == right
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(
            _same_json(value, right[key]) for key, value in left.items()
        )
    if isinstance(left, list):
        return len(left) == len(right) and all(
            _same_json(a, b) for a, b in zip(left, right, strict=True)
        )
    return left == right


def _snapshot(job: ExperimentJob) -> dict[str, Any]:
    """Freeze the shared serializer's mutable dictionaries and path values."""
    return json.loads(
        json.dumps(experiment_store.serialize_job(job), default=json_default, allow_nan=False)
    )


def _validate_journal(journal: RecoveryJournal, store: Store | None = None) -> None:
    _request_id(journal.root_request_id)
    _job_id(journal.root_job_id)
    _job_id(journal.head_job_id)
    root = journal.root
    if not isinstance(root, JournalEntry) or not isinstance(journal.resumes, dict):
        raise _invalid("Journal requires typed root and resume entries")
    if (
        root.phase == "noop"
        or root.request_id != journal.root_request_id
        or root.job_id != journal.root_job_id
        or root.parent_job_id is not None
        or root.attempt_index != 0
    ):
        raise _invalid("Journal root identity disagrees")
    children: list[JournalEntry] = []
    for digest, entry in journal.resumes.items():
        if (
            type(digest) is not str
            or not _DIGEST.fullmatch(digest)
            or not isinstance(entry, JournalEntry)
            or digest != path_digest(entry.request_id)
        ):
            raise _invalid("Resume entry does not match its request digest")
        if entry.phase != "noop":
            children.append(entry)
    attempts = {journal.root_job_id: 0}
    parent_id = journal.root_job_id
    for index, entry in enumerate(sorted(children, key=lambda item: item.attempt_index), start=1):
        if (
            entry.job_id is None
            or entry.job_id in attempts
            or entry.attempt_index != index
            or entry.parent_job_id != parent_id
        ):
            raise _invalid("Journal children must form one distinct, contiguous parent chain")
        attempts[entry.job_id] = index
        parent_id = entry.job_id
    if journal.head_job_id != parent_id:
        raise _invalid("Journal head does not identify the last child")
    for entry in (root, *journal.resumes.values()):
        if entry.phase == "noop":
            if (
                entry.parent_job_id is None
                or attempts.get(entry.parent_job_id) != entry.attempt_index
            ):
                raise _invalid("No-op entry does not address a recorded parent attempt")
            continue
        if entry.job_id is None:
            raise _invalid("Child entry has no job ID")
        if entry.candidate is None:
            if entry.phase == "prepared":
                raise _invalid("Prepared candidate is missing")
            continue
        path = store.job_record(entry.job_id) if store is not None else Path("candidate.json")
        job = _decode_candidate(entry, path)
        assert job.recovery is not None
        if (
            job.recovery.root_job_id != journal.root_job_id
            or job.recovery.root_request_id != journal.root_request_id
        ):
            raise _invalid("Candidate belongs to a different root lineage")
        for case in job.cases:
            assert case.recovery is not None
            attempt = case.recovery.attempt
            if (
                attempts.get(attempt.execution_job_id) != attempt.attempt_index
                or attempt.attempt_index > entry.attempt_index
                or (attempt.reused and attempt.attempt_index >= entry.attempt_index)
            ):
                raise _invalid("Candidate case attempt is outside its committed ancestry")


def _unique_members(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _invalid("Duplicate JSON members in recovery evidence")
        result[key] = value
    return result


def load_journal(store: Store, root_request_id: str) -> RecoveryJournal | None:
    """Return None only for absent evidence; reject every present invalid record."""
    _request_id(root_request_id)
    try:
        path = store.recovery_journal(root_request_id)
        try:
            data = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_members)
        except FileNotFoundError:
            # A broken symlink is present authoritative evidence, not a free ID.
            if os.path.lexists(path):
                raise _invalid("Recovery journal points to missing evidence") from None
            return None
        _check_envelope(data, KIND_RECOVERY_JOURNAL)
        journal = RecoveryJournal.from_record(
            {key: value for key, value in data.items() if key not in _ENVELOPE_FIELDS}
        )
        if journal.root_request_id != root_request_id:
            raise _invalid("Recovery journal belongs to a different root request")
        _validate_journal(journal, store)
        return journal
    except (OSError, ValueError, TypeError, RecursionError) as exc:
        raise _invalid("Recovery journal cannot be read or validated") from exc


def save_journal(store: Store, journal: RecoveryJournal) -> None:
    """Replace authoritative evidence durably; caller holds the recovery lock."""
    _validate_journal(journal, store)
    try:
        atomic_write_json(
            store.recovery_journal(journal.root_request_id),
            envelope(KIND_RECOVERY_JOURNAL, **journal.to_record()),
        )
    except (OSError, ValueError, TypeError) as exc:
        raise RecoveryError(
            "recovery_journal_write_failed", "Recovery journal was not saved"
        ) from exc


def prepared_entry(job: ExperimentJob) -> JournalEntry:
    """Freeze a complete prelaunch candidate for admission or dead-owner adoption."""
    if job.recovery is None:
        raise _invalid("Recovery admission requires a recorded lineage")
    try:
        snapshot = _snapshot(job)
    except (ValueError, TypeError, RecursionError) as exc:
        raise _invalid("Recovery candidate cannot be serialized") from exc
    return JournalEntry(
        job.request_id,
        job.fingerprint,
        job.recovery.parent_job_id,
        job.job_id,
        job.recovery.attempt_index,
        "prepared",
        snapshot,
    )


def new_root_journal(job: ExperimentJob) -> RecoveryJournal:
    """Build a prepared root snapshot before writing any derived records."""
    entry = prepared_entry(job)
    return RecoveryJournal(job.request_id, job.job_id, job.job_id, entry, {})


def _resume_key(journal: RecoveryJournal, request_id: str, parent_id: str | None) -> str:
    _validate_journal(journal)
    _request_id(request_id)
    key = path_digest(request_id)
    if key in journal.resumes:
        raise _invalid("Resume request is already committed; caller must replay it")
    if parent_id != journal.head_job_id:
        raise _invalid("Resume must address the current lineage head")
    return key


def add_child(
    journal: RecoveryJournal, resume_request_id: str, fingerprint: str, job: ExperimentJob
) -> RecoveryJournal:
    """Append a prepared child and move the head, without mutating the input."""
    entry = prepared_entry(job)
    key = _resume_key(journal, resume_request_id, entry.parent_job_id)
    if entry.request_id != resume_request_id or entry.fingerprint != fingerprint:
        raise _invalid("Child does not match its normalized resume identity")
    return replace(journal, head_job_id=job.job_id, resumes={**journal.resumes, key: entry})


def add_noop(
    journal: RecoveryJournal, resume_request_id: str, fingerprint: str, parentjob: ExperimentJob
) -> RecoveryJournal:
    """Commit a no-op at the addressed head without creating a child."""
    key = _resume_key(journal, resume_request_id, parentjob.job_id)
    recovery = parentjob.recovery
    if (
        recovery is None
        or recovery.root_job_id != journal.root_job_id
        or recovery.root_request_id != journal.root_request_id
    ):
        raise _invalid("No-op parent belongs to a different root lineage")
    entry = JournalEntry(
        resume_request_id,
        fingerprint,
        parentjob.job_id,
        None,
        recovery.attempt_index,
        "noop",
        None,
    )
    return replace(journal, resumes={**journal.resumes, key: entry})


def mark_launched(journal: RecoveryJournal, job_id: str) -> RecoveryJournal:
    """Drop the candidate before any launch intent can execute; never reconstruct it."""
    _validate_journal(journal)
    if job_id == journal.root_job_id:
        return replace(journal, root=replace(journal.root, phase="launched", candidate=None))
    for key, entry in journal.resumes.items():
        if entry.job_id == job_id:
            return replace(
                journal,
                resumes={**journal.resumes, key: replace(entry, phase="launched", candidate=None)},
            )
    raise _invalid("Cannot advance launch phase for an uncommitted job")


def candidate_job(entry: JournalEntry, store: Store) -> ExperimentJob:
    """Decode a retained candidate; launched entries without one fail closed."""
    if entry.job_id is None or entry.candidate is None:
        raise _invalid("This entry has no reconstructable candidate")
    return _decode_candidate(entry, store.job_record(entry.job_id))
