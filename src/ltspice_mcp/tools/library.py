"""Component library management tools. (Phase 5)"""

from typing import Literal

from mcp import types
from pydantic import Field

from ltspice_mcp.errors import LibraryError
from ltspice_mcp.state import SessionState
from ltspice_mcp.tools._base import (
    PAGINATION_SCHEMA,
    RO_ANNOTATIONS,
    ToolInput,
    format_response,
    paginate,
    pagination_metadata,
    registry,
    safe_path,
    text_response,
)


class SearchLibraryInput(ToolInput):
    query: str = Field(description="Search term (case-insensitive substring match)")
    source: Literal["user", "builtin"] = Field(default="user", description="Search user-loaded or built-in libraries")
    offset: int = Field(default=0, description="Pagination offset")
    limit: int = Field(default=50, description="Max results to return")
    format: Literal["json", "text"] | None = Field(default=None)


class GetModelInfoInput(ToolInput):
    name: str = Field(description="Model or subcircuit name (case-insensitive)")
    full: bool = Field(default=False, description="Include full SPICE definition text")
    format: Literal["json", "text"] | None = Field(default=None)


class LoadLibraryInput(ToolInput):
    path: str = Field(description="Path to library file or directory")


class UnloadLibraryInput(ToolInput):
    path: str = Field(description="Path to library file or directory to unload")


class ListLibrariesInput(ToolInput):
    detail: bool = Field(default=False, description="Include model names from each library")
    path: str | None = Field(default=None, description="Filter to a specific library path")
    offset: int = Field(default=0, description="Pagination offset")
    limit: int = Field(default=50, description="Max results to return")
    format: Literal["json", "text"] | None = Field(default=None)


@registry.tool(
    name="ltspice_search_library",
    description=(
        "Search component libraries for models and subcircuits by name "
        "(case-insensitive substring match). Search user-loaded or built-in libraries."
    ),
    input_model=SearchLibraryInput,
    annotations=RO_ANNOTATIONS,
    profiles=("full", "agentic"),
    output_schema={
        "type": "object",
        "properties": {
            "results": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "type": {"type": "string"},
                        "source_path": {"type": "string"},
                    },
                },
            },
            "pagination": PAGINATION_SCHEMA,
        },
    },
)
async def handle_search_library(arguments: SearchLibraryInput, state: SessionState):
    """Search component libraries by name."""
    query = arguments.query
    source = arguments.source
    offset = arguments.offset
    limit = min(arguments.limit, 50)
    fmt = arguments.format

    try:
        if source == "user":
            result = state.libraries.search_user_libraries(query, offset, limit)
        elif source == "builtin":
            result = state.libraries.search_builtin_libraries(query, offset, limit)
        else:
            raise LibraryError(f"Invalid source: {source}. Must be 'user' or 'builtin'")
    except LibraryError:
        raise
    except Exception as e:
        raise LibraryError(f"Search failed: {e}") from e

    results = result["results"]
    total = result["total"]

    if not results:
        return format_response(
            f"No models found matching '{query}'",
            {"results": [], "pagination": pagination_metadata(0, offset, limit)},
            fmt,
        )

    lines = [f"Found {total} model(s) matching '{query}'"]
    lines.append(f"Showing {offset + 1}-{offset + len(results)} of {total}")
    lines.append("")

    for r in results:
        lines.append(f"  {r['name']} ({r['type']}) - {r['source_path']}")

    data = {
        "results": results,
        "pagination": pagination_metadata(total, offset, limit),
    }
    return format_response("\n".join(lines), data, fmt)


@registry.tool(
    name="ltspice_get_model_info",
    description=(
        "Get SPICE model/subcircuit details including parameters and ready-to-use "
        ".include directive. Set full=true to get the complete SPICE definition text."
    ),
    input_model=GetModelInfoInput,
    annotations=RO_ANNOTATIONS,
    profiles=("full", "agentic"),
)
async def handle_get_model_info(arguments: GetModelInfoInput, state: SessionState):
    """Get SPICE model/subcircuit details."""
    name = arguments.name
    full = arguments.full
    fmt = arguments.format

    try:
        info = state.libraries.get_model_info(name, full)
    except Exception as e:
        raise LibraryError(f"Failed to get model info: {e}") from e

    if info is None:
        raise LibraryError(
            f"Model '{name}' not found in loaded or built-in libraries. "
            "Use ltspice_search_library to find models."
        )

    lines = [
        f"Model: {info['name']}",
        f"Type: {info['type']}",
        f"Source: {info['source_path']}",
        "",
        "Include directive:",
        f"  {info['include_directive']}",
        "",
    ]

    if info["parameters"]:
        lines.append("Parameters:")
        for param in info["parameters"]:
            lines.append(f"  {param}")
        lines.append("")

    if full and "raw_text" in info:
        lines.append("Full SPICE definition:")
        lines.append(info["raw_text"])

    return format_response("\n".join(lines), info, fmt)


@registry.tool(
    name="ltspice_load_library",
    description=(
        "Load a SPICE library file (.lib, .mod) or directory of library files into the session."
    ),
    input_model=LoadLibraryInput,
    annotations=types.ToolAnnotations(
        readOnlyHint=False,
        destructiveHint=False,
        idempotentHint=False,
        openWorldHint=False,
    ),
    profiles=("full",),
)
async def handle_load_library(arguments: LoadLibraryInput, state: SessionState) -> types.CallToolResult:
    """Load a SPICE library file or directory.

    Args:
        arguments: Contains 'path' (string)
        state: Session state with library manager

    Returns:
        List with single TextContent containing load summary

    Raises:
        PathSecurityError: Path outside sandbox
        LibraryError: Load failed
    """
    path = safe_path(arguments.path, state)

    try:
        summary = state.libraries.load_library(path)
    except LibraryError:
        raise
    except Exception as e:
        raise LibraryError(f"Failed to load library: {e}") from e

    result = (
        f"Loaded {summary['path']}: "
        f"{summary['models']} models, {summary['subcircuits']} subcircuits "
        f"from {summary['files_loaded']} file(s)"
    )

    return text_response(result)


@registry.tool(
    name="ltspice_unload_library",
    description="Unload a previously loaded library from the session.",
    input_model=UnloadLibraryInput,
    annotations=types.ToolAnnotations(
        readOnlyHint=False,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    ),
    profiles=("full",),
)
async def handle_unload_library(arguments: UnloadLibraryInput, state: SessionState) -> types.CallToolResult:
    """Unload a library from the session.

    Args:
        arguments: Contains 'path' (string)
        state: Session state with library manager

    Returns:
        List with single TextContent confirming unload

    Raises:
        PathSecurityError: Path outside sandbox
        LibraryError: Library not loaded
    """
    path = safe_path(arguments.path, state)

    try:
        result = state.libraries.unload_library(path)
    except Exception as e:
        raise LibraryError(f"Failed to unload library: {e}") from e

    if not result["removed"]:
        raise LibraryError(f"Library not loaded: {path}")

    return text_response(f"Unloaded library: {path}")


@registry.tool(
    name="ltspice_list_libraries",
    description="List loaded libraries. With detail=true, also shows subcircuit models from each library.",
    input_model=ListLibrariesInput,
    annotations=RO_ANNOTATIONS,
    profiles=("full",),
)
async def handle_list_libraries(arguments: ListLibrariesInput, state: SessionState):
    """List loaded libraries, optionally with subcircuit detail."""
    detail = arguments.detail
    fmt = arguments.format
    filter_path = None
    if arguments.path is not None:
        filter_path = safe_path(arguments.path, state)

    libs = state.libraries.list_libraries()

    if not libs:
        return format_response("No libraries loaded", {"libraries": []}, fmt)

    # Apply path filter
    if filter_path:
        libs = [lp for lp in libs if str(filter_path) in str(lp)]

    if not libs:
        return format_response(f"No libraries matching {filter_path}", {"libraries": []}, fmt)

    libs_page, total, offset, limit = paginate(libs, arguments)
    has_more = offset + len(libs_page) < total
    header = f"Loaded libraries: showing {offset + 1}-{offset + len(libs_page)} of {total}"

    if not detail:
        lines = [header]
        lib_data = []
        for lib_path in libs_page:
            lines.append(f"  {lib_path}")
            lib_data.append({"path": str(lib_path)})
        if has_more:
            lines.append(f"\nNext page: ltspice_list_libraries(offset={offset + limit})")
        data = {"libraries": lib_data, "pagination": pagination_metadata(total, offset, limit)}
        return format_response("\n".join(lines), data, fmt)

    # Detail mode: include subcircuit names per library
    try:
        result = state.libraries.search_user_libraries("", 0, 999999)
    except Exception as e:
        raise LibraryError(f"Failed to list subcircuits: {e}") from e

    subcircuits = [r for r in result["results"] if r["type"] == ".SUBCKT"]

    subcircuits_by_path: dict[str, list[str]] = {}
    for sc in subcircuits:
        src = sc["source_path"]
        subcircuits_by_path.setdefault(src, []).append(sc["name"])

    lines = [header]
    lib_data = []
    for lib_path in libs_page:
        lib_str = str(lib_path)
        matching_subs: list[str] = []
        for src, names in subcircuits_by_path.items():
            if lib_str in src:
                matching_subs.extend(names)
        lines.append(f"  {lib_path}")
        if matching_subs:
            for name in sorted(matching_subs):
                lines.append(f"    .SUBCKT {name}")
        else:
            lines.append("    (no subcircuits)")
        lib_data.append({"path": str(lib_path), "subcircuits": sorted(matching_subs)})

    if has_more:
        lines.append(f"\nNext page: ltspice_list_libraries(detail=true, offset={offset + limit})")

    data = {"libraries": lib_data, "pagination": pagination_metadata(total, offset, limit)}
    return format_response("\n".join(lines), data, fmt)

