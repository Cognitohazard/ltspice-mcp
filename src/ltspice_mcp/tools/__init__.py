"""Tool registration entrypoint for ltspice-mcp."""

from mcp import types

from ltspice_mcp.config import ToolListing
from ltspice_mcp.tools._base import RegisteredTool, registry
from ltspice_mcp.tools._schema import strip_argument_descriptions

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


#: The order tools/list advertises. Declared, not inherited from the import
#: graph: clients cache the list and prompt caching keys on its exact bytes, so
#: a tool module importing a sibling must not be able to reorder it. A
#: registered tool missing from this list is a startup error.
ADVERTISED_ORDER: tuple[str, ...] = (
    "plot_waveform",
    "analyze_results",
    "run_experiments",
    "jobs",
    "inspect",
    "edit_schematic",
    "verify_circuit",
)


def get_tools(
    listing: ToolListing = "full",
) -> tuple[list[types.Tool], dict[str, RegisteredTool]]:
    """Return the advertised tool definitions and their dispatch metadata.

    ``listing`` selects how the same capabilities are put on the wire
    (``[tools] listing`` in the config):

    * ``full`` — the registered definitions, unchanged. The default, and the
      only mode whose output is pinned byte for byte.
    * ``compact`` — the same seven tools and the same schemas with every
      per-argument description removed from the published copy. Dispatch is
      untouched, so each tool still accepts exactly what it did.
    """
    tool_defs, tool_dispatch = registry.get_tools()
    unlisted = sorted(set(tool_dispatch) - set(ADVERTISED_ORDER))
    if unlisted:
        raise RuntimeError(f"registered tools missing from ADVERTISED_ORDER: {unlisted}")
    tool_defs.sort(key=lambda definition: ADVERTISED_ORDER.index(definition.name))
    if listing == "compact":
        tool_defs = [
            definition.model_copy(
                update={"input_schema": strip_argument_descriptions(definition.input_schema)}
            )
            for definition in tool_defs
        ]
    return tool_defs, tool_dispatch
