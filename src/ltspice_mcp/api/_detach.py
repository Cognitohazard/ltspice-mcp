"""Spawning and talking to a per-job detached owner.

The caller's half of `run_experiments(wait=False, detach=True)`. It writes the
request, starts `python -m ltspice_mcp.detached_owner` in its own session, and
blocks until that process reports a durable receipt — or dies, or runs out of
time trying.

Nothing here writes a job record. The owner does, so the record's `owner_pid`
names a live supervising process from its first byte; see
`docs/design/python_api.md` §11 for why that ordering and not the other one.
"""

from __future__ import annotations

import contextlib
import json
import os
import secrets
import subprocess
import sys
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ltspice_mcp.api._exceptions import ApiCallError, ApiInternalError
from ltspice_mcp.lib import atomic_write_json
from ltspice_mcp.lib.store import (
    KIND_DETACHED_RECEIPT,
    KIND_DETACHED_REQUEST,
    accept,
    envelope,
)
from ltspice_mcp.state import SessionState

#: How long the owner gets to boot, stage its decks and submit before the
#: caller stops waiting. Generous because staging a large include closure on a
#: slow filesystem is legitimately slow, and because the cost of being wrong is
#: a spawned process the caller then has to reason about.
HANDSHAKE_TIMEOUT_S = 300.0

#: How often the receipt file is looked for. Small: the whole point of
#: detaching is to get the handle back and move on.
HANDSHAKE_POLL_S = 0.02

#: How much of the owner's log an error message carries back.
_LOG_TAIL_BYTES = 2000

#: How many finished calls' hand-off logs the directory keeps. Every call now
#: writes its own, so without a bound a script detaching in a loop leaves one
#: file per run behind for good. Pruning is by modification time, newest kept,
#: so a live owner's log — always among the newest — is not the one dropped.
_KEPT_HANDOFF_LOGS = 50


@dataclass(frozen=True)
class DetachedBoot:
    """The `Api` constructor arguments an owner has to reproduce.

    Recorded as the caller wrote them, plus the directory they were interpreted
    in, so `working_dir=None` — which means "this process's cwd" — resolves to
    the same place in the owner as it did here.
    """

    working_dir: str | None = None
    config_path: str | None = None
    overrides: Mapping[str, Any] = field(default_factory=dict)
    cwd: str = field(default_factory=os.getcwd)


@dataclass(frozen=True)
class DetachedHandoff:
    """What an owner reported back once its submission was durable."""

    receipt: dict[str, Any]
    supervisor_pid: int
    log_file: Path


def boot_spec(
    working_dir: str | os.PathLike[str] | None,
    config_path: str | os.PathLike[str] | None,
    overrides: Mapping[str, object],
) -> DetachedBoot:
    """Record one `Api`'s constructor arguments in a form an owner can be given."""
    return DetachedBoot(
        working_dir=None if working_dir is None else str(working_dir),
        config_path=None if config_path is None else str(config_path),
        # Kept as given and converted at spawn time: an override this module
        # cannot serialize must fail the detached call, not the constructor of
        # every session that never detaches anything.
        overrides=dict(overrides),
        cwd=os.getcwd(),
    )


def _jsonable(value: Any) -> Any:
    """One configuration value in a form JSON can carry.

    Deliberately narrow: paths become strings and containers are walked, and
    anything else is refused by name. A silently dropped or stringified
    override would give the owner a different engine than the caller has.
    """
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    raise TypeError(
        f"A detached owner cannot be given a {type(value).__name__} configuration value"
    )


def request_arguments(arguments: Mapping[str, Any]) -> dict[str, Any]:
    """The call's arguments as the owner will receive them.

    Round-tripped through JSON here rather than at spawn time so the caller
    validates exactly the payload the owner will validate — the fingerprint
    behind idempotent replay is computed from it, and a value that only
    survives in this process would fork the two.
    """
    return json.loads(json.dumps(arguments, default=_argument_default))


def _argument_default(value: Any) -> Any:
    from pydantic import BaseModel

    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    raise TypeError(f"A detached owner cannot be given a {type(value).__name__} argument")


def _log_tail(path: Path) -> str:
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            handle.seek(max(0, handle.tell() - _LOG_TAIL_BYTES))
            return handle.read().decode("utf-8", errors="replace").strip()
    except OSError:
        return ""


def _failure(request_id: str, code: str, message: str, log_file: Path) -> ApiCallError:
    """A pre-submission failure, shaped like every other one this tool returns."""
    from ltspice_mcp.tools import experiments

    detail = f"{message} The detached owner's log is at {log_file}."
    tail = _log_tail(log_file)
    if tail:
        detail = f"{detail} Its last output was:\n{tail}"
    payload = experiments.submission_error_payload(
        request_id,
        code=code,
        message=detail,
        stage="submission",
        retryable=True,
        # The owner is the only process that knows whether it got as far as
        # submitting, and it died without saying. Claiming not_started here
        # would tell the caller a job that may exist does not.
        commit_state="unknown",
    )
    return ApiCallError(detail, payload=payload)


def _read_report(path: Path) -> dict[str, Any] | None:
    """The owner's report, or None while it has not written one yet."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not accept(data, path, kind=KIND_DETACHED_RECEIPT):
        return None
    return data


def prune(children: list[subprocess.Popen[bytes]]) -> None:
    """Reap the owners that have already exited, and keep the rest.

    A caller that detaches repeatedly would otherwise leave one zombie per
    finished owner behind it. Owners still running are deliberately not waited
    on — outliving this process is what they are for.
    """
    children[:] = [child for child in children if child.poll() is None]


def submit(
    state: SessionState,
    boot: DetachedBoot,
    arguments: Mapping[str, Any],
    request_id: str,
    children: list[subprocess.Popen[bytes]],
) -> DetachedHandoff:
    """Spawn one owner for this request and return the receipt it reports."""
    store = state.store
    store.ensure_root()
    store.detached_dir.mkdir(parents=True, exist_ok=True)
    # This call's own hand-off, not this request id's. Two scripts in one
    # working directory replaying the same request_id — the advertised use —
    # would otherwise unlink each other's report, read each other's receipt,
    # and append to one log; the report path is what the owner is told to
    # write, so a name only this call knows is what makes "a report exists"
    # mean "the owner this call spawned wrote one".
    nonce = secrets.token_hex(8)
    request_path = store.detached_request(request_id, nonce)
    report_path = store.detached_receipt(request_id, nonce)
    log_path = store.detached_log(request_id, nonce)
    _prune_handoff_logs(store.detached_dir)

    atomic_write_json(
        request_path,
        envelope(
            KIND_DETACHED_REQUEST,
            receipt_file=str(report_path),
            working_dir=boot.working_dir,
            config_path=boot.config_path,
            overrides=_jsonable(dict(boot.overrides)),
            arguments=dict(arguments),
        ),
    )

    prune(children)
    try:
        with log_path.open("ab") as log:
            # Fixed argv, no shell: the only caller-derived element is the
            # request path this module just wrote.
            process = subprocess.Popen(
                [sys.executable, "-m", "ltspice_mcp.detached_owner", str(request_path)],
                cwd=boot.cwd,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                # Its own session, so the caller's terminal signals — the Ctrl-C
                # that ends the script — do not reach the process the script
                # detached precisely so it would survive.
                start_new_session=True,
            )
    except OSError as exc:
        request_path.unlink(missing_ok=True)
        raise _failure(
            request_id,
            "detached_owner_not_started",
            f"The detached owner process could not be started: {exc}",
            log_path,
        ) from exc
    children.append(process)

    try:
        report = _await_report(process, report_path, request_id, log_path)
    finally:
        # The caller is the only reader, and it has read. Leaving the file
        # behind would accumulate one per call for the life of the directory.
        report_path.unlink(missing_ok=True)
    if not report.get("ok"):
        error = report.get("error")
        message = error.get("message") if isinstance(error, Mapping) else None
        code = error.get("code") if isinstance(error, Mapping) else None
        raise _failure(
            request_id,
            str(code or "detached_submission_failed"),
            str(message or "The detached owner reported a failure with no message."),
            log_path,
        )

    receipt = report.get("receipt")
    if not isinstance(receipt, dict):
        raise ApiInternalError("The detached owner reported a submission with no receipt")
    supervisor_pid = report.get("supervisor_pid")
    return DetachedHandoff(
        receipt=receipt,
        supervisor_pid=supervisor_pid if isinstance(supervisor_pid, int) else process.pid,
        log_file=log_path,
    )


def _prune_handoff_logs(detached_dir: Path) -> None:
    """Keep the newest hand-off logs and drop the rest.

    Best-effort housekeeping, never a reason to fail a call: a directory that
    cannot be listed or a file another process removed first is ignored.
    """
    try:
        logs = sorted(
            detached_dir.glob("*.log"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
    except OSError:
        return
    for stale in logs[_KEPT_HANDOFF_LOGS:]:
        with contextlib.suppress(OSError):
            stale.unlink()


def _await_report(
    process: subprocess.Popen[bytes],
    report_path: Path,
    request_id: str,
    log_path: Path,
) -> dict[str, Any]:
    deadline = time.monotonic() + HANDSHAKE_TIMEOUT_S
    while True:
        report = _read_report(report_path)
        if report is not None:
            return report
        exited = process.poll()
        if exited is not None:
            # Looked again after the exit, not instead of it: the owner writes
            # its report and then exits, so a poll that wins that race would
            # call a completed submission a dead process.
            report = _read_report(report_path)
            if report is not None:
                return report
            raise _failure(
                request_id,
                "detached_owner_exited",
                (f"The detached owner exited with status {exited} before reporting a submission."),
                log_path,
            )
        if time.monotonic() >= deadline:
            process.terminate()
            raise _failure(
                request_id,
                "detached_owner_timeout",
                (
                    f"The detached owner did not report a submission within "
                    f"{HANDSHAKE_TIMEOUT_S:.0f}s and was stopped. If it had already "
                    f"submitted, request_id {request_id!r} still names that job: ask "
                    "for it again to replay it."
                ),
                log_path,
            )
        time.sleep(HANDSHAKE_POLL_S)
