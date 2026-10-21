"""Resolving the paths callers name, and the optional sandbox they must lie in."""

from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

from ltspice_mcp.errors import PathSecurityError

#: Directory a relative path is taken from, when a host declares one
#: (``resolve_safe_path`` gives the order). The in-process ``Api`` sets it to its
#: ``working_dir`` around every call: ``Api(working_dir=D)`` plus a bare
#: ``"opamp2.asc"`` names ``D/opamp2.asc``. A context variable rather than an
#: argument: every relative path on every op has to be rebased, so the rule
#: belongs at the one chokepoint they all pass through, where no new path field
#: can be added without inheriting it.
_relative_base: ContextVar[Path | None] = ContextVar("relative_path_base", default=None)


@contextmanager
def relative_paths_from(base: Path | None) -> Iterator[None]:
    """Resolve relative paths against ``base`` inside this block (and its tasks)."""
    token = _relative_base.set(base)
    try:
        yield
    finally:
        _relative_base.reset(token)


def _anchor(path: Path, base: Path | None) -> Path:
    return path if base is None or path.is_absolute() else base / path


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

    A relative path is taken from the host's declared base
    (``relative_paths_from``) when there is one, else from the first allowed
    directory, else from ``relative_to``.

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

    base = _relative_base.get()
    path = _anchor(
        Path(user_path),
        base if base is not None else allowed_dirs[0] if allowed_dirs else relative_to,
    )

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

    # Check if resolved path is within any allowed directory. A relative entry
    # in that list (``allowed_paths = ["."]`` in a config file, say) is anchored
    # the same way the user path was, so a declared base moves the sandbox with
    # it instead of leaving it pinned to the process cwd.
    for allowed_dir in allowed_dirs:
        try:
            allowed_resolved = _anchor(allowed_dir, base).resolve()
            if resolved.is_relative_to(allowed_resolved):
                return resolved
        except (OSError, RuntimeError):
            # Skip this allowed_dir if it can't be resolved
            continue

    # Path is outside all allowed directories. A relative entry is named with
    # the directory it stands for: "." alone does not say where the sandbox is.
    allowed_list = ", ".join(_as_named(d, base) for d in allowed_dirs)
    raise PathSecurityError(f"Path {resolved} is outside allowed directories [{allowed_list}]")


def _as_named(allowed_dir: Path, base: Path | None) -> str:
    """An allowed directory as a refusal names it: as written, and when that is
    relative, with the directory it resolves to."""
    if allowed_dir.is_absolute():
        return str(allowed_dir)
    try:
        return f"{allowed_dir} ({_anchor(allowed_dir, base).resolve()})"
    except (OSError, RuntimeError):
        return str(allowed_dir)
