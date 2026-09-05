"""One experiment's detached owner: submit it, supervise it, exit.

``Api.run_experiments(wait=False, detach=True)`` spawns this module as

    sys.executable -m ltspice_mcp.detached_owner <request-file>

so a short-lived script can submit work that outlives it. The process that
spawned this one never owns the job: everything on disk about it — the request
index entry and the coordinator record, with its ``owner_pid`` — is written
here, by the process that will supervise it. There is therefore no interval in
which the record names a process that is not its owner.

This is a program, not library mode: it configures logging on its own root and
writes to the stdout and stderr the parent redirected into a log file.

The hand-off is two files under ``.ltspice-mcp/detached/``, both named from the
``request_id`` and both owned by :class:`~ltspice_mcp.lib.store.Store`:

* the **request file** this process is given as its one argument. It carries
  the ``Api`` constructor arguments to reproduce and the ``run_experiments``
  arguments to submit. It is deleted as soon as it has been read.
* the **receipt file**, written exactly once, before supervision begins:
  ``{"ok": true, "receipt": {...}}`` once the submission is durable, or
  ``{"ok": false, "error": {...}}`` if it never got that far. The parent is
  blocked reading for it, so nothing may overwrite it afterwards — a job that
  fails after submission is reported through its own record, like any other.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import traceback
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class _Handshake:
    """The one report this process owes the process that spawned it.

    Write-once by construction rather than by discipline: the success report
    and every failure path go through the same object, and the first one to
    arrive is the one the parent reads.
    """

    def __init__(self, path: Path | None) -> None:
        self.path = path
        self.reported = False

    def _write(self, **payload: Any) -> None:
        if self.reported or self.path is None:
            return
        from ltspice_mcp.lib import atomic_write_json
        from ltspice_mcp.lib.store import KIND_DETACHED_RECEIPT, envelope

        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_json(self.path, envelope(KIND_DETACHED_RECEIPT, **payload))
            self.reported = True
        except OSError:
            logger.exception("could not write the detached owner's receipt file")

    def receipt(self, receipt: dict[str, Any]) -> None:
        # This process, which is the job's owner on a fresh submission and is
        # not on a replay. The caller reads the owner from the record itself;
        # this is only the fallback for a receipt naming no job.
        self._write(ok=True, supervisor_pid=os.getpid(), receipt=receipt)

    def failure(self, code: str, message: str) -> int:
        logger.error("%s: %s", code, message)
        self._write(ok=False, error={"code": code, "message": message})
        return 1


def _read_request(path: Path) -> dict[str, Any]:
    from ltspice_mcp.lib.store import KIND_DETACHED_REQUEST, accept

    data = json.loads(path.read_text(encoding="utf-8"))
    if not accept(data, path, kind=KIND_DETACHED_REQUEST, log=logger):
        raise ValueError(f"{path} is not a detached owner request this build reads")
    return data


def run(request: dict[str, Any], handshake: _Handshake) -> int:
    """Submit the request, report the receipt, then supervise to terminality."""
    from ltspice_mcp.api import Api

    api = Api(
        working_dir=request.get("working_dir"),
        config_path=request.get("config_path"),
        **request.get("overrides", {}),
    )
    try:
        try:
            receipt = api.run_experiments(wait=False, **request["arguments"])
        except Exception as exc:
            return handshake.failure("detached_submission_failed", str(exc))

        job_id = receipt.get("job_id")
        handshake.receipt(receipt)
        if not isinstance(job_id, str):
            # A receipt naming no job names nothing to supervise. The parent
            # has the receipt already and can read whatever it does name.
            logger.error("the submission receipt named no job to supervise")
            return 1

        logger.info("submitted %s; supervising until it is terminal", job_id)
        final = api.wait(job_id)
        logger.info("job %s finished with status %s", job_id, final.get("status"))
        return 0
    finally:
        try:
            # Cancels whatever this process still owns, which after a completed
            # wait is nothing. An owner exiting mid-job — because the wait
            # raised — is the crashed-owner case, and leaving its cases running
            # with nobody watching them would be the worse end.
            api.close()
        except Exception:
            logger.exception("the detached owner's engine did not close cleanly")


def _configure_logging() -> None:
    """Log to the stderr the parent pointed at the owner's log file."""
    level = os.getenv("LTSPICE_MCP_LOG_LEVEL", "INFO").upper()
    if level not in {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"}:
        level = "INFO"
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stderr,
    )


def main(argv: list[str]) -> int:
    _configure_logging()
    if len(argv) != 1:
        print("usage: python -m ltspice_mcp.detached_owner <request-file>", file=sys.stderr)
        return 2

    request_path = Path(argv[0])
    try:
        request = _read_request(request_path)
    except (OSError, ValueError) as exc:
        return _Handshake(None).failure("detached_request_unreadable", str(exc))

    handshake = _Handshake(Path(request["receipt_file"]))
    # Read once, then gone: the request has served its purpose, and a leftover
    # copy would outlive the run it describes.
    request_path.unlink(missing_ok=True)

    try:
        return run(request, handshake)
    except BaseException as exc:
        # The last chance to tell the caller anything at all: a failure that
        # escapes here would leave it polling for a report that never comes.
        traceback.print_exc()
        return handshake.failure("detached_owner_failed", f"{type(exc).__name__}: {exc}")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
