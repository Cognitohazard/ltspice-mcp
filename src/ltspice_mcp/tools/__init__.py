"""Tool registration entrypoint for ltspice-mcp."""

from ltspice_mcp.tools._base import registry

# Importing these modules triggers @registry.tool registrations. The
# consolidated modules (analyze, experiments, inspect_tools, schematic_edit,
# verify) import internals from the base modules at module scope; the sorted
# order below already loads each dependency first — analysis precedes analyze.
from . import (  # noqa: F401
    analysis,
    analyze,
    experiments,
    inspect_tools,
    schematic_edit,
    verify,
)


def get_tools():
    """Return the advertised tool definitions and their dispatch metadata."""
    return registry.get_tools()
