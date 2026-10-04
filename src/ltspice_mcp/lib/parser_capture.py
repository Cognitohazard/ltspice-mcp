"""Bounded snapshots of the files one parser process may read.

Call from the owned parser process: even copying a regular file can block on
its filesystem. Captured bytes and companion presence define the revision;
source paths and modification times are not cache identities.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from ltspice_mcp.lib.store import parser_file_in

_CHUNK_BYTES = 1024 * 1024
_FILENAMES = {"raw": "input.raw", "log": "input.log", "console": "input.exe.log"}
PARSER_REVISION = "artifacts-materialize-3"


class CaptureError(ValueError):
    """The requested files could not form a complete, bounded snapshot."""


@dataclass(frozen=True)
class SourceFiles:
    raw: Path | None = None
    log: Path | None = None
    console: Path | None = None


@dataclass(frozen=True)
class CapturedFile:
    role: str
    name: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True)
class CapturedInputs:
    files: tuple[CapturedFile, ...]
    absent: tuple[str, ...]

    def cache_key(
        self, *, dialect: str | None, producing_dialect: str | None, revision: str
    ) -> str:
        """Bind parser behavior and dialect evidence as well as every captured byte."""
        value = {
            "files": [asdict(item) for item in self.files],
            "absent": self.absent,
            "dialect": dialect,
            "producing_dialect": producing_dialect,
            "revision": revision,
        }
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


def parser_cache_key(
    captured: CapturedInputs,
    *,
    dialect: str | None,
    producing_dialect: str | None,
    limits: dict,
) -> str:
    """One identity for raw and log facts captured under the same parser policy."""
    return captured.cache_key(
        dialect=dialect,
        producing_dialect=producing_dialect,
        revision=json.dumps({"decoder": PARSER_REVISION, "limits": limits}, sort_keys=True),
    )


def _stamp(info: os.stat_result) -> tuple[int, int, int, int, int]:
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns


def _source_stat(path: Path | None) -> os.stat_result | None:
    if path is None:
        return None
    try:
        info = path.stat()
    except FileNotFoundError:
        return None
    if not stat.S_ISREG(info.st_mode):
        raise CaptureError("Parser inputs must be regular files")
    return info


def _copy_file(source: Path, target: Path, before: os.stat_result, budget: int) -> tuple[int, str]:
    digest = hashlib.sha256()
    copied = 0
    with source.open("rb") as reader:
        opened = _stamp(os.fstat(reader.fileno()))
        # Windows stat and fstat can disagree on creation/change time for an
        # unchanged file. Compare that field only within the same view; retain
        # identity, size and modification time across the path/handle boundary.
        if opened[:4] != _stamp(before)[:4]:
            raise CaptureError("Parser input changed before capture")
        # Each attempt owns a new directory. Never overwrite another capture.
        with target.open("xb") as writer:
            while block := reader.read(min(_CHUNK_BYTES, budget - copied + 1)):
                copied += len(block)
                if copied > budget:
                    raise CaptureError("Parser input byte limit exceeded")
                writer.write(block)
                digest.update(block)
        if _stamp(os.fstat(reader.fileno())) != opened or copied != before.st_size:
            raise CaptureError("Parser input changed during capture")
    return copied, digest.hexdigest()


def capture_inputs(
    sources: SourceFiles,
    directory: Path,
    *,
    input_bytes: int,
    log_bytes: int,
    require_raw: bool = True,
) -> CapturedInputs:
    """Copy complete inputs or refuse; the owner cleans up after process exit.

    Missing companions are explicit revision facts. A named RAW must exist
    unless this is a log-only parse with require_raw=False.
    Both limits are positive integers; input_bytes bounds all input roles
    together, while log_bytes bounds the two logs together. No partial log is
    presented as a complete capture.
    """
    if any(type(limit) is not int or limit <= 0 for limit in (input_bytes, log_bytes)):
        raise ValueError("Capture limits must be finite positive integers")
    paths = {field.name: getattr(sources, field.name) for field in fields(sources)}
    initial = {role: _source_stat(path) for role, path in paths.items()}
    if require_raw and sources.raw is not None and initial["raw"] is None:
        raise FileNotFoundError("Requested RAW file is unavailable")
    if not any(initial.values()):
        raise FileNotFoundError("No parser input files are available")
    if sum(info.st_size for info in initial.values() if info is not None) > input_bytes:
        raise CaptureError("Parser input byte limit exceeded")
    if sum(info.st_size for role, info in initial.items() if role != "raw" and info) > log_bytes:
        raise CaptureError("Parser companion log byte limit exceeded")
    captured = []
    used = 0
    used_logs = 0
    for role, path in paths.items():
        info = initial[role]
        if path is None or info is None:
            continue
        budget = input_bytes - used
        if role != "raw":
            budget = min(budget, log_bytes - used_logs)
        name = _FILENAMES[role]
        size, digest = _copy_file(path, parser_file_in(directory, name), info, budget)
        captured.append(CapturedFile(role, name, size, digest))
        used += size
        if role != "raw":
            used_logs += size
    for role, path in paths.items():
        after = _source_stat(path)
        before = initial[role]
        if (after is None) != (before is None) or (
            after is not None and before is not None and _stamp(after) != _stamp(before)
        ):
            raise CaptureError("Parser inputs or companion presence changed during capture")
    return CapturedInputs(
        tuple(captured), tuple(role for role, info in initial.items() if info is None)
    )
