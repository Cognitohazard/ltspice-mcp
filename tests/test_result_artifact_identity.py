"""Stored analysis identity distinguishes every captured artifact's absence."""

import hashlib
import json

import pytest

from ltspice_mcp.errors import ResultError
from ltspice_mcp.lib import result_store


def test_console_changes_and_empty_files_are_distinct_from_absence():
    log_digest = hashlib.sha256(b"vout = 1\n").hexdigest()
    empty_digest = hashlib.sha256(b"").hexdigest()
    console_digest = hashlib.sha256(b"Error: analysis not run\n").hexdigest()
    snapshots = [
        result_store.composite_digest(
            raw_sha256=None,
            log_sha256=log_digest,
            console_sha256=console,
        )
        for console in (None, empty_digest, console_digest)
    ]
    assert len(set(snapshots)) == 3
    assert (
        result_store.composite_digest(
            raw_sha256=empty_digest,
            log_sha256=log_digest,
            console_sha256=None,
        )
        not in snapshots
    )


def test_artifact_roles_cannot_exchange_the_same_bytes():
    digest = hashlib.sha256(b"recorded artifact").hexdigest()
    snapshots = {
        result_store.composite_digest(
            raw_sha256=digest if role == "raw" else None,
            log_sha256=digest if role == "log" else None,
            console_sha256=digest if role == "console" else None,
        )
        for role in ("raw", "log", "console")
    }
    assert len(snapshots) == 3


def test_result_sets_with_older_artifact_contract_are_not_reinterpreted(tmp_path):
    item = result_store.create(
        working_dir=tmp_path,
        inputs={},
        work=[],
        source_manifests=[],
        source_jobs={},
        ttl_hours=1,
    )
    assert result_store.load(item.result_set_id, tmp_path) == item
    path = result_store.result_path(item.result_set_id, tmp_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    data["store_version"] = 3
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ResultError, match="unsupported storage schema"):
        result_store.load(item.result_set_id, tmp_path)
