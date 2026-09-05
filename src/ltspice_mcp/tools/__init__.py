"""Tool registration entrypoint for ltspice-mcp."""

from ltspice_mcp.tools._base import registry

# Importing these modules triggers @registry.tool registrations, and a tool is
# advertised in the order it registered — so this list IS the advertised order,
# and is kept off the import sorter for that reason. Each module's dependencies
# also load ahead of it (analysis before analyze; experiments before jobs, which
# reads the wait cap from it).
# isort: off
from . import (  # noqa: F401
    analysis,
    analyze,
    experiments,
    jobs,
    inspect_tools,
    schematic_edit,
    verify,
)

# isort: on


def get_tools():
    """Return the advertised tool definitions and their dispatch metadata."""
    return registry.get_tools()
