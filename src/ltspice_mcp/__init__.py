"""LTSpice MCP Server - Circuit simulation through Model Context Protocol."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    __version__: str


def __getattr__(name: str) -> str:
    """Read the installed version on first access, not at import.

    ``importlib.metadata`` walks the installed distributions, and that walk is
    most of the cost of a bare ``spice-mcp --help`` — a command with no use for
    the version. Everything that does need it asks by name and pays there.
    """
    if name == "__version__":
        from importlib.metadata import version

        return version("ltspice-mcp")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
