"""Tool registration entrypoint for ltspice-mcp."""

from ltspice_mcp.tools._base import registry

# Importing these modules triggers @registry.tool registrations.
from . import advanced, analysis, circuit, library, simulation, status  # noqa: F401


def get_tools_for_profile(profile: str):
    """Return tool definitions and dispatch metadata for a profile."""
    return registry.get_for_profile(profile)
