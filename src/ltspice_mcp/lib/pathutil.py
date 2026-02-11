"""Path security utilities with sandboxing."""

from pathlib import Path

from ltspice_mcp.errors import PathSecurityError


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
            f"Path traversal attempts (..) are not allowed: {user_path}"
        )

    # Resolve relative paths against first allowed_dir (working directory)
    # Absolute paths are used as-is
    if not path.is_absolute():
        base_dir = allowed_dirs[0]
        path = base_dir / path

    # Resolve symlinks and normalize (strict=False allows non-existent files)
    try:
        resolved = path.resolve(strict=False)
    except (OSError, RuntimeError) as e:
        raise PathSecurityError(f"Failed to resolve path {user_path}: {e}") from e

    # Check if resolved path is within any allowed directory
    for allowed_dir in allowed_dirs:
        try:
            allowed_resolved = allowed_dir.resolve()
            if resolved.is_relative_to(allowed_resolved):
                return resolved
        except (OSError, RuntimeError):
            # Skip this allowed_dir if it can't be resolved
            continue

    # Path is outside all allowed directories
    allowed_list = ", ".join(str(d) for d in allowed_dirs)
    raise PathSecurityError(
        f"Path {resolved} is outside allowed directories [{allowed_list}]"
    )
