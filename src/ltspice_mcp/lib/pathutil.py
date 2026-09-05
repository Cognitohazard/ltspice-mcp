"""Path security utilities with sandboxing."""

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

from ltspice_mcp.errors import PathSecurityError

#: Directory a relative path is taken from, when a host declares one.
#:
#: Unset (the MCP server, and any caller that does not opt in), a relative path
#: is resolved the way it always was: against the first allowed directory, and
#: through ``Path.resolve()`` for anything relative in that list — which is the
#: process working directory. The in-process ``Api`` sets it to its
#: ``working_dir`` for the duration of each call, because that interface lets the
#: caller name a working directory that is not their cwd; without this, the
#: contract's own documented idiom (``Api(working_dir=D)`` plus a bare
#: ``"opamp2.asc"``) looked in the wrong place and reported a path the caller
#: never wrote. A context variable rather than an argument: every relative path
#: on every op has to be rebased, so the rule belongs at the one chokepoint they
#: all pass through, where no new path field can be added without inheriting it.
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


def resolve_safe_path(user_path: str, allowed_dirs: list[Path]) -> Path:
    """Resolve a user-provided path within security sandbox.

    This function implements strict sandboxing:
    1. Explicitly rejects path traversal attempts (../)
    2. Resolves symlinks before validation
    3. Checks that resolved path is within allowed directories
    4. Returns specific error messages for security violations

    Args:
        user_path: Path string from user (relative or absolute)
        allowed_dirs: List of allowed base directories (sandbox)

    Returns:
        Resolved absolute path within sandbox

    Raises:
        PathSecurityError: If path contains traversal attempts or resolves
                          outside allowed directories
    """
    if not allowed_dirs:
        raise PathSecurityError("No allowed directories configured")

    # Convert to Path object
    path = Path(user_path)

    # Check for explicit path traversal attempts
    # This catches patterns like "../../etc/passwd"
    if ".." in path.parts:
        raise PathSecurityError(
            f"Path traversal attempts (..) are not allowed: {user_path}. "
            "This is rejected before resolution even when the target would land "
            "inside the sandbox — pass the equivalent absolute path instead."
        )

    # Resolve relative paths against the host's declared base when there is one,
    # else against the first allowed_dir (working directory). Absolute paths are
    # used as-is.
    base = _relative_base.get()
    if not path.is_absolute():
        path = (allowed_dirs[0] if base is None else base) / path

    # Resolve symlinks and normalize (strict=False allows non-existent files).
    # ValueError covers an embedded NUL byte ("embedded null byte"), which
    # path.resolve raises rather than OSError — without it the raw ValueError
    # would escape the path-security boundary.
    try:
        resolved = path.resolve(strict=False)
    except (OSError, RuntimeError, ValueError) as e:
        raise PathSecurityError(f"Failed to resolve path {user_path}: {e}") from e

    # Check if resolved path is within any allowed directory. A relative entry
    # in that list (``allowed_paths = ["."]``, what the generated TOML ships) is
    # anchored the same way the user path was, so a declared base moves the
    # sandbox with it instead of leaving it pinned to the process cwd.
    for allowed_dir in allowed_dirs:
        try:
            allowed_resolved = _anchor(allowed_dir, base).resolve()
            if resolved.is_relative_to(allowed_resolved):
                return resolved
        except (OSError, RuntimeError):
            # Skip this allowed_dir if it can't be resolved
            continue

    # Path is outside all allowed directories
    allowed_list = ", ".join(str(d) for d in allowed_dirs)
    raise PathSecurityError(f"Path {resolved} is outside allowed directories [{allowed_list}]")
