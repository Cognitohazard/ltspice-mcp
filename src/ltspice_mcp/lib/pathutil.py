"""Resolving the paths callers name, and the optional sandbox they must lie in."""

from collections.abc import Iterable, Sequence
from pathlib import Path

from ltspice_mcp.errors import PathSecurityError


def relative_base(allowed_dirs: Sequence[Path] | None, working_dir: Path | None) -> Path | None:
    """The directory a relative path is taken from: the working directory,
    unless a sandbox leaves it out, and then the first allowed directory.

    The one rule for it, in every host: a tool call, the Python API (and so a
    ``run_code`` snippet and a detached job's owner) and a resource read take
    a relative name from the same folder, and resolving a relative path,
    listing the files a relative name can reach and telling a caller where
    relatives go all name it. ``ServerConfig`` makes the working directory and
    every sandbox entry absolute when it is loaded, so the folder does not
    depend on the folder a process was started in. With neither, ``None``:
    the process's folder.
    """
    if not allowed_dirs:
        return working_dir
    if working_dir is not None:
        try:
            if _inside(working_dir.resolve(), allowed_dirs):
                return working_dir
        except (OSError, RuntimeError):
            pass
    return allowed_dirs[0]


def _inside(resolved: Path, allowed_dirs: Sequence[Path]) -> bool:
    """Whether the resolved ``resolved`` lies in one of ``allowed_dirs``; an
    entry that cannot be resolved holds nothing."""
    for allowed_dir in allowed_dirs:
        try:
            if resolved.is_relative_to(allowed_dir.resolve()):
                return True
        except (OSError, RuntimeError):
            continue
    return False


def with_roots(allowed_dirs: Sequence[Path] | None, extra: Iterable[Path]) -> list[Path] | None:
    """``allowed_dirs`` and ``extra`` together. ``None``, no sandbox, stays
    ``None``: a directory added to a sandbox must never create one."""
    return None if allowed_dirs is None else [*allowed_dirs, *extra]


def resolve_safe_path(
    user_path: str,
    allowed_dirs: Sequence[Path] | None,
    *,
    relative_to: Path | None = None,
) -> Path:
    """Resolve a user-provided path, and refuse it outside ``allowed_dirs``.

    ``allowed_dirs`` is the sandbox: ``None`` when there is none, and then the
    path is resolved and returned wherever it lands. With a sandbox the path is
    judged by where it lands, never by how it is spelled:
    1. Rejects an embedded NUL byte (with or without a sandbox)
    2. Resolves symlinks and ``..`` segments
    3. Checks that the resolved path is within an allowed directory

    A ``..`` is therefore accepted when the path it forms stays inside the
    sandbox (``sub/../deck.cir``, or a deck's ``.include ../models/x.lib``
    joined onto its folder) and refused when it climbs out. The resolution
    is what makes that safe: on POSIX a symlink is followed before the ``..``
    after it, as the open would follow it, and on Windows ``..`` collapses
    before any link is followed, as the open collapses it.

    A relative path is taken from ``relative_base(allowed_dirs, relative_to)``.

    Args:
        user_path: Path string from user (relative or absolute)
        allowed_dirs: The allowed base directories, or None for no sandbox
        relative_to: The session's working directory

    Returns:
        The resolved absolute path

    Raises:
        PathSecurityError: If the path cannot be resolved, or there is a
                          sandbox and the path resolves outside it
    """
    # Refused up front: on Windows with Python 3.13, resolve() returns a path
    # holding the NUL instead of raising, and the sandbox check would pass it.
    if "\x00" in user_path:
        raise PathSecurityError(f"Failed to resolve path {user_path!r}: embedded null byte")

    path = Path(user_path)
    base = relative_base(allowed_dirs, relative_to)
    if base is not None and not path.is_absolute():
        path = base / path

    # Resolve symlinks and normalize (strict=False allows non-existent files).
    # ValueError covers an embedded NUL byte ("embedded null byte"), which
    # path.resolve raises rather than OSError — without it the raw ValueError
    # would escape the path-security boundary.
    try:
        resolved = path.resolve(strict=False)
    except (OSError, RuntimeError, ValueError) as e:
        raise PathSecurityError(f"Failed to resolve path {user_path}: {e}") from e
    if allowed_dirs is None:
        return resolved

    # A configured entry is already absolute (``ServerConfig`` anchors it on
    # the working directory); a relative one handed in directly is taken from
    # the cwd.
    if _inside(resolved, allowed_dirs):
        return resolved

    # Path is outside all allowed directories
    raise PathSecurityError(
        f"Path {resolved} is outside allowed directories [{describe_roots(allowed_dirs)}]"
    )


def describe_roots(allowed_dirs: Sequence[Path]) -> str:
    """The allowed directories as a refusal and its guidance name them: each
    as written, and a relative one with the directory it resolves to, since
    "." alone does not say where the sandbox is."""

    def named(allowed_dir: Path) -> str:
        if allowed_dir.is_absolute():
            return str(allowed_dir)
        try:
            return f"{allowed_dir} ({allowed_dir.resolve()})"
        except (OSError, RuntimeError):
            return str(allowed_dir)

    return ", ".join(named(allowed_dir) for allowed_dir in allowed_dirs)
