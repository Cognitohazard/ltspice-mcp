"""Component library management tools. (Phase 5)"""

from typing import Literal

from mcp import types
from pydantic import Field

from ltspice_mcp.errors import LibraryError
from ltspice_mcp.lib.mcp_logging import mcp_log
from ltspice_mcp.lib.services import format_suggestion_block
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


class GetModelInfoInput(ToolInput):
    name: str = Field(description="Model or subcircuit name (case-insensitive)")
    full: bool = Field(default=False, description="Include full SPICE definition text")
    format: Literal["json", "text"] | None = Field(default=None, description="Response format: 'json' for structured data, 'text' for human-readable")


class FindModelInput(ToolInput):
    name: str = Field(description="Model/subcircuit name to match (case-insensitive)")
    exact: bool = Field(default=False, description="Only return the exact case-insensitive match (score=1.0) if any; skips fuzzy scoring.")
    limit: int = Field(default=5, description="Max suggestions to return (1-25). Ignored when exact=true.")
    cutoff: float = Field(default=0.6, description="Minimum fuzzy similarity ratio (0.0-1.0). Lower = more matches, noisier. Ignored when exact=true.")
    include_builtin: bool = Field(default=False, description="Also walk built-in simulator libraries (slower; lazy-parses all built-ins on first call).")
    format: Literal["json", "text"] | None = Field(default=None, description="Response format: 'json' for structured data, 'text' for human-readable")


class LoadLibraryInput(ToolInput):
    path: str = Field(description="Path to library file or directory")


class UnloadLibraryInput(ToolInput):
    path: str = Field(description="Path to library file or directory to unload")


class ListLibrariesInput(ToolInput):
    detail: bool = Field(default=False, description="Include model names from each library")
    path: str | None = Field(default=None, description="Filter to a specific library path")
    offset: int = Field(default=0, description="Pagination offset")
    limit: int = Field(default=50, description="Max results to return")
    format: Literal["json", "text"] | None = Field(default=None, description="Response format: 'json' for structured data, 'text' for human-readable")


@registry.tool(
    name="ltspice_get_model_info",
    description=(
        "Get SPICE model/subcircuit details including parameters and ready-to-use "
        ".include directive. Set full=true to get the complete SPICE definition text."
    ),
    input_model=GetModelInfoInput,
    annotations=RO_ANNOTATIONS,
    profiles=("full", "agentic"),
    output_schema={
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "type": {"type": "string"},
            "source_path": {"type": "string"},
            "include_directive": {"type": "string"},
            "parameters": {"type": "array", "items": {"type": "string"}},
            "raw_text": {"type": "string"},
        },
    },
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
        suggestions = state.libraries.find_similar_models(
            name, limit=3, cutoff=0.6, include_builtin=False
        )
        msg = f"Model '{name}' not found in loaded libraries."
        if suggestions:
            block = format_suggestion_block({name: suggestions}, header="Did you mean:")
            raise LibraryError(
                f"{msg}{block}\n\nUse ltspice_find_model to browse more candidates.",
                suggestions=suggestions,
            )
        raise LibraryError(
            f"{msg} Load a library containing it with ltspice_load_library, "
            "or lower the cutoff via ltspice_find_model(cutoff=...) for a wider search."
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
    name="ltspice_find_model",
    description=(
        "Find model/subcircuit candidates across loaded (and optionally built-in) "
        "libraries. Default is fuzzy matching — finds typos, case variants, and "
        "near-neighbour part numbers (e.g., '2N3905' → '2N3904'); pass exact=true "
        "to only return the exact case-insensitive match. Returns ranked candidates "
        "with similarity score and ready-to-paste .include directive."
    ),
    input_model=FindModelInput,
    annotations=RO_ANNOTATIONS,
    profiles=("full", "agentic"),
    output_schema={
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "results": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "type": {"type": "string"},
                        "source_path": {"type": "string"},
                        "include_directive": {"type": "string"},
                        "score": {"type": "number"},
                        "parameters": {"type": "object"},
                    },
                },
            },
            "include_builtin": {"type": "boolean"},
            "exact": {"type": "boolean"},
            "cutoff": {"type": "number"},
        },
    },
)
async def handle_find_model(arguments: FindModelInput, state: SessionState):
    name = arguments.name
    exact = arguments.exact
    limit = max(1, min(arguments.limit, 25))
    cutoff = max(0.0, min(arguments.cutoff, 1.0))
    include_builtin = arguments.include_builtin
    fmt = arguments.format

    try:
        results = state.libraries.find_similar_models(
            name,
            exact=exact,
            limit=limit,
            cutoff=cutoff,
            include_builtin=include_builtin,
        )
    except Exception as e:
        raise LibraryError(f"Model search failed: {e}") from e

    data = {
        "query": name,
        "results": results,
        "include_builtin": include_builtin,
        "exact": exact,
        "cutoff": cutoff,
    }
    scope = "loaded + built-in" if include_builtin else "loaded"

    if not results:
        if exact:
            hint = " Retry ltspice_find_model with exact=false for fuzzy matches."
        elif not include_builtin:
            hint = " Try lowering cutoff or set include_builtin=true."
        else:
            hint = " Try lowering cutoff or ltspice_load_library to add more sources."
        reason = "No exact match" if exact else f"No fuzzy matches (cutoff={cutoff})"
        return format_response(
            f"{reason} for '{name}' in {scope} libraries.{hint}", data, fmt
        )

    mode = "Exact match" if exact else f"Fuzzy matches (cutoff={cutoff})"
    lines = [f"{mode} for '{name}' in {scope} libraries:", ""]
    for r in results:
        lines.append(f"  {r['name']} ({r['type']}, score={r['score']}) - {r['source_path']}")
        lines.append(f"    {r['include_directive']}")
    return format_response("\n".join(lines), data, fmt)


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
    await mcp_log("info", f"Library loaded: {result}")

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
    output_schema={
        "type": "object",
        "properties": {
            "libraries": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "subcircuits": {"type": "array", "items": {"type": "string"}},
                    },
                },
            },
            "pagination": PAGINATION_SCHEMA,
        },
    },
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

