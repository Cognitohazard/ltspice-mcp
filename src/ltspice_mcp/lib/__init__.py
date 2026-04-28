"""Library utilities for ltspice-mcp."""

from __future__ import annotations

import contextlib
import json
import logging
import os
import sys
import tempfile
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import IO, Any, Literal

logger = logging.getLogger(__name__)

_EST = timezone(timedelta(hours=-5), name="EST")

# Recognised netlist / schematic file extensions. Shared by the recent-
# circuits tracker, sidecar loaders, and the netlist resource listing.
CIRCUIT_EXTENSIONS: frozenset[str] = frozenset({".asc", ".net", ".sp", ".cir", ".spice"})

# macOS-only: fcntl.F_FULLFSYNC forces a platter-level flush. Plain fsync on
# APFS/HFS+ stops at the drive's write cache, so durability-critical writes
# need this. SQLite, Postgres, and git all use it on Darwin.
if sys.platform == "darwin":
    import fcntl as _fcntl

    _F_FULLFSYNC: int | None = getattr(_fcntl, "F_FULLFSYNC", None)
else:
    _fcntl = None  # type: ignore[assignment]
    _F_FULLFSYNC = None


def now() -> datetime:
    """Return the current time in US Eastern (EST, UTC-5)."""
    return datetime.now(tz=_EST)


def parse_iso_datetime(value: str | None) -> datetime | None:
    """Parse an ISO timestamp defensively; returns None on failure or non-string input."""
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _fsync_fd(fd: int) -> None:
    """Durable flush of ``fd`` to persistent storage.

    On macOS, plain ``fsync`` only reaches the drive's write cache;
    ``F_FULLFSYNC`` forces a true platter-level flush. On Linux and Windows,
    ``fsync``/``FlushFileBuffers`` already provides that guarantee.
    Falls back to plain ``fsync`` if ``F_FULLFSYNC`` fails — e.g., on some
    network mounts that don't implement it.
    """
    if _F_FULLFSYNC is not None and _fcntl is not None:
        try:
            _fcntl.fcntl(fd, _F_FULLFSYNC)
            return
        except OSError:
            pass  # fall through to regular fsync
    os.fsync(fd)


def _fsync_dir(path: Path) -> None:
    """Best-effort fsync of a directory, persisting rename metadata.

    No-op on Windows (the API has no equivalent). OSError is swallowed:
    by the time we call this the rename has already succeeded and the data
    file is already fsync'd, so a dir-fsync failure only affects durability
    of the rename itself — not correctness. We log at debug for diagnosis.
    """
    if sys.platform == "win32":
        return
    try:
        dir_fd = os.open(str(path), os.O_RDONLY)
    except OSError as exc:
        logger.debug("dir fsync: open(%s) failed: %s", path, exc)
        return
    try:
        os.fsync(dir_fd)
    except OSError as exc:
        logger.debug("dir fsync: fsync(%s) failed: %s", path, exc)
    finally:
        os.close(dir_fd)


def _commit(tmp_path: Path, dst: Path, *, overwrite: bool) -> None:
    """Atomically move ``tmp_path`` over ``dst``.

    ``overwrite=True`` uses ``os.replace`` (POSIX rename with force-overwrite;
    ``MOVEFILE_REPLACE_EXISTING`` on Windows). ``overwrite=False`` uses
    ``os.link`` + ``os.unlink`` on POSIX (race-free, fails cleanly with
    ``FileExistsError`` if ``dst`` exists) and ``os.rename`` on Windows
    (which fails if ``dst`` exists, unlike POSIX ``rename``).
    """
    if overwrite:
        os.replace(tmp_path, dst)
        return
    if sys.platform == "win32":
        # On Windows, os.rename raises if dst exists — that's the behavior
        # we want. On POSIX, os.rename would silently overwrite, so we
        # take the os.link branch instead.
        os.rename(tmp_path, dst)
    else:
        # Hard-link succeeds only if dst does not exist; then unlink the
        # source path. Both paths remain pointing at the same inode
        # momentarily — fine, readers see a consistent file either way.
        os.link(tmp_path, dst)
        os.unlink(tmp_path)


@contextmanager
def atomic_write(
    path: Path,
    *,
    mode: Literal["w", "wb"] = "w",
    encoding: str | None = "utf-8",
    durable: bool = True,
    overwrite: bool = True,
) -> Iterator[IO[Any]]:
    """Open ``path`` for atomic, durable writing via tempfile-then-rename.

    Yields a writable file handle (text or binary per ``mode``). On clean
    exit the buffer is flushed, ``fsync``'d (when ``durable``), closed, and
    atomically moved over ``path``; the parent directory is then ``fsync``'d
    so the rename itself is durable. On any exception — including
    ``KeyboardInterrupt`` — the tempfile is unlinked and ``path`` is left
    untouched.

    Parameters
    ----------
    path
        Destination path. Parent directories are created if missing.
    mode
        ``"w"`` for text (default) or ``"wb"`` for binary.
    encoding
        Text encoding. Ignored in binary mode.
    durable
        When ``True`` (default), ``fsync`` the file before rename and the
        parent directory after. Use ``F_FULLFSYNC`` on macOS for true
        platter-level durability. Set ``False`` for hot-path writes where
        crash-safety-sans-durability is acceptable (e.g., caches).
    overwrite
        When ``True`` (default), replace ``path`` if it exists. When
        ``False``, raise ``FileExistsError`` instead (race-free on POSIX
        via ``os.link``; best-effort on Windows via ``os.rename``).

    Guarantees
    ----------
    * **Atomicity**: concurrent readers see either the old contents or the
      complete new contents, never a partial write.
    * **Durability** (``durable=True``): data is flushed to stable storage
      before the rename, and the rename is flushed after. Survives power
      loss on common filesystems (ext4, XFS, APFS, NTFS). On macOS the
      flush is ``F_FULLFSYNC`` (platter-level), not plain ``fsync``.
    * **Permission preservation**: if ``path`` exists, its mode is copied
      onto the replacement before rename. Fresh files inherit the
      tempfile's default mode (``0600``).
    * **Cleanup**: the tempfile is unlinked on any exception, including
      ``KeyboardInterrupt`` and ``SystemExit``.

    Caveats
    -------
    * If ``path`` is a symlink, the symlink itself is replaced — not its
      target. Pass ``path.resolve()`` if you want the target replaced.
    * On Windows, ``os.replace`` may briefly fail with ``PermissionError``
      if another process (e.g., antivirus) has the file open; no retry
      is attempted.
    * ``overwrite=False`` on Windows has a TOCTOU window between the
      existence check (implicit in ``os.rename``) and the rename itself.
      POSIX uses ``os.link``, which is race-free.
    """
    binary = "b" in mode
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    tmp_path = Path(tmp_name)

    # Permissions only need preserving when we might replace an existing file.
    existing_mode: int | None = None
    if overwrite:
        with contextlib.suppress(FileNotFoundError):
            existing_mode = path.stat().st_mode & 0o7777

    try:
        open_kwargs: dict[str, Any] = {}
        if not binary:
            open_kwargs["encoding"] = encoding
        with os.fdopen(fd, mode, **open_kwargs) as f:
            yield f
            f.flush()
            if durable:
                _fsync_fd(f.fileno())

        if existing_mode is not None:
            with contextlib.suppress(OSError):
                os.chmod(tmp_path, existing_mode)

        _commit(tmp_path, path, overwrite=overwrite)
    except BaseException:
        with contextlib.suppress(OSError):
            tmp_path.unlink()
        raise

    if durable:
        _fsync_dir(path.parent)


def atomic_write_text(
    path: Path,
    text: str,
    *,
    encoding: str = "utf-8",
    durable: bool = True,
    overwrite: bool = True,
) -> None:
    """Write ``text`` to ``path`` atomically (and durably, by default)."""
    with atomic_write(path, encoding=encoding, durable=durable, overwrite=overwrite) as f:
        f.write(text)


def atomic_write_bytes(
    path: Path,
    data: bytes,
    *,
    durable: bool = True,
    overwrite: bool = True,
) -> None:
    """Write ``data`` to ``path`` atomically (and durably, by default)."""
    with atomic_write(path, mode="wb", durable=durable, overwrite=overwrite) as f:
        f.write(data)


def atomic_write_json(
    path: Path,
    data: Any,
    *,
    default: Callable[[Any], Any] | None = None,
    indent: int = 2,
    durable: bool = True,
    overwrite: bool = True,
) -> None:
    """Write ``data`` as JSON to ``path`` atomically (and durably, by default)."""
    with atomic_write(path, durable=durable, overwrite=overwrite) as f:
        json.dump(data, f, indent=indent, default=default)


__all__ = [
    "CIRCUIT_EXTENSIONS",
    "atomic_write",
    "atomic_write_bytes",
    "atomic_write_json",
    "atomic_write_text",
    "now",
    "parse_iso_datetime",
]
