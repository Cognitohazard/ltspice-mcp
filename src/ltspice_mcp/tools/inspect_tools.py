"""inspect — the consolidated, honestly read-only UNDERSTAND surface.

One tool answers a batch of independent read-only ``queries`` about the server
and the circuits it can reach. Each query is one of six kinds:

* ``capabilities`` — detected simulators + dialects, exporter presence, job
  persistence, allowed roots, active profile, the configured limits, and the
  linter version. Pulled from ``state``/``config``/``lint_rules``; nothing is
  probed.
* ``symbols`` — the legal ``.asy`` symbol names and the resolution-order
  precedence they resolve through. A ``path`` adds that schematic's own
  directory to the front of the reported precedence.
* ``symbol`` — one symbol's pin positions per rotation (``R0``…``M270``),
  bounding box, and origin (the ``symbol_info`` geometry internals).
* ``net`` — everything on a net. On a ``.asc`` this is a geometric trace
  (``trace_net`` internals: pins, wire vertices, labels, shorts). On a
  ``.cir``/``.net``/``.sp`` netlist it is card-membership — which element cards
  reference the node — carrying **no geometry** at all.
* ``components`` — the component list (``detail:"list"``) or full per-component
  detail (``detail:"full"``) of any circuit file.
* ``model`` — model/subcircuit lookup: ``search`` fuzzy-matches a ``query``;
  ``enumerate`` lists every model defined in the given ``libs``.

Per-item isolation is the contract: a denied path, a tampered/stale cursor, an
unknown ``kind``, or a malformed query fails **only that item** and carries a
structured ``error``; every other query in the batch still returns. The
paginated kinds (``symbols``/``net``/``components``/``model``) resume through an
opaque, checksummed ``cursor``; a ``.asc`` ``net`` pages two collections — pins
and wire-vertex coordinates — under that one cursor, each advancing by its own
offset, so paging until ``next_cursor`` is null reaches every row of both.

Concurrency discipline (see the contract in ``tools/_base.py``): cached
``AscEditor`` access (the ``.asc`` net and component paths) stays on the event
loop; only immutable parsing (netlist lexing, ``.asy`` and ``.lib`` parses,
symbol-directory walks) is offloaded to a worker thread. Every response is built
in the handler coroutine.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Annotated, Any, Literal, TypeAlias, get_args

from mcp import types
from pydantic import Field, SkipValidation, TypeAdapter, ValidationError, model_validator

from ltspice_mcp.errors import LTSpiceMCPError, PathSecurityError
from ltspice_mcp.lib import services
from ltspice_mcp.lib.cursor_codec import canonical_hash
from ltspice_mcp.lib.encoding import read_spice_text
from ltspice_mcp.lib.library_manager import _part_aware_score, parse_library_file_cached
from ltspice_mcp.lib.lint_rules import linter_version
from ltspice_mcp.lib.pin_legend import PageCursorError, paginate_pair, paginate_view
from ltspice_mcp.lib.schematic_scene import SymbolResolver, default_stock_paths
from ltspice_mcp.lib.simulator import current_ngbehavior, dialect_for_simulator_name
from ltspice_mcp.lib.spice_lex import SpiceLexError, lex
from ltspice_mcp.lib.spice_lex_views import InstanceLine, instances_by_ref
from ltspice_mcp.lib.symbol_geometry import compute_placed_geometry, parse_asy_file
from ltspice_mcp.state import SessionState
from ltspice_mcp.tools._base import (
    HINT_SCHEMA,
    RO_ANNOTATIONS,
    StrictModel,
    ToolInput,
    format_response,
    registry,
    safe_path,
    symbol_resolver_for,
)
from ltspice_mcp.tools.circuit import (
    TraceNetInput,
    _get_asc_editor,
    handle_trace_net,
    netlist_card_value,
)

# The circuit module's private editor helpers are reused verbatim rather than
# duplicated; the .asc net/component paths depend on the SAME cached editor.
# pyright: reportPrivateUsage=false

# Fixed server-side page size for the paginated kinds. The A.5 query specs list
# a cursor but no caller-facing limit knob, so the page is a server constant.
_PAGE_SIZE = 100

# Page size for the .asc net trace's wire-vertex coordinates — the second
# collection of the net item, paged alongside its pins under the same cursor.
# Larger than the pin page because a coordinate is two integers.
_COORD_PAGE_SIZE = 500

NETLIST_SUFFIXES = frozenset({".cir", ".net", ".sp"})

# Every rotation LTspice can place a symbol at, in a stable reported order.
ROTATIONS: tuple[str, ...] = ("R0", "R90", "R180", "R270", "M0", "M90", "M180", "M270")

# Dwell caps mirrored for capabilities disclosure. Source of truth: the field
# constraints on experiments.ExecutionInput.wait_s (run_experiments) and
# experiments.JobsInput.timeout_s (jobs), themselves the A5 timing defaults.
# Duplicated as documented constants to keep inspect free of experiments'
# heavy import side effects; the capabilities test pins key presence, not value.
_RUN_EXPERIMENTS_WAIT_DEFAULT_S = 60.0
_RUN_EXPERIMENTS_WAIT_MAX_S = 120.0
_JOBS_WAIT_DEFAULT_S = 60.0
_JOBS_WAIT_MAX_S = 300.0


# ---------------------------------------------------------------------------
# Query models — a sealed discriminated union on ``kind``
# ---------------------------------------------------------------------------


_CURSOR_DESCRIPTION = (
    "Opaque page token taken verbatim from a previous page's 'next_cursor' — echo "
    "it back unmodified. It is bound to this exact query, so it will not resume a "
    "different one, and an edited or stale token is rejected."
)


class CapabilitiesQuery(StrictModel):
    """What this server can do: the detected simulators and the raw dialect each
    parses, whether the .asc netlist exporter is available, job persistence, the
    allowed path roots, the active tool profile, the configured limits and dwell
    caps, and the linter version. Takes no arguments and probes nothing."""

    kind: Literal["capabilities"]


class SymbolsQuery(StrictModel):
    """The .asy symbol names that resolve, and the directory precedence they
    resolve through. Ask this before placing a component — a name absent here
    will not place."""

    kind: Literal["symbols"]
    path: str | None = Field(
        default=None,
        description=(
            "Optional schematic whose OWN directory is put at the front of the "
            "reported precedence — pass it to see what that sheet would resolve."
        ),
    )
    filter: str | None = Field(
        default=None,
        description="Case-insensitive substring; only symbol names containing it are returned.",
    )
    cursor: str | None = Field(default=None, description=_CURSOR_DESCRIPTION)


class SymbolQuery(StrictModel):
    """One symbol's geometry: pin positions at every rotation, bounding box, and
    origin. This is the non-destructive pre-placement view — the pins reported
    for a rotation are where they land when the part is placed at it."""

    kind: Literal["symbol"]
    name: str = Field(
        min_length=1,
        description="Symbol name without the .asy extension, e.g. 'res', 'nmos4'.",
    )
    path: str | None = Field(
        default=None,
        description="Optional schematic whose directory is searched first, for a sheet-local symbol.",
    )


class NetQuery(StrictModel):
    """Everything on one net. On a .asc this is a geometric trace — pins, wire
    vertices, net labels, and whether two labels short the net. On a netlist it
    is card membership — which element cards reference the node — with no
    geometry at all."""

    kind: Literal["net"]
    path: str = Field(description="The .asc schematic, or .cir/.net/.sp netlist, to read.")
    at: str | list[int] = Field(
        description=(
            "Where the net is: 'REF.PIN' (e.g. 'M1.D'), 'net:NAME', or [x, y]. A "
            "netlist carries no geometry, so there it takes 'net:NAME', a bare node "
            "name, or 'REF.<terminal-number>', and rejects a coordinate."
        )
    )
    cursor: str | None = Field(default=None, description=_CURSOR_DESCRIPTION)

    @model_validator(mode="after")
    def _valid_at(self) -> NetQuery:
        if isinstance(self.at, list):
            if len(self.at) != 2:
                raise ValueError("'at' as coordinates must be exactly [x, y]")
        elif not self.at.strip():
            raise ValueError("'at' must be 'REF.PIN', 'net:NAME', or [x, y]")
        return self


class ComponentsQuery(StrictModel):
    """The components of a .asc schematic or a .cir/.net/.sp netlist."""

    kind: Literal["components"]
    path: str = Field(description="The .asc schematic, or .cir/.net/.sp netlist, to read.")
    prefix: str | None = Field(
        default=None,
        description=(
            "Keep only components whose reference starts with this element letter "
            "('R', 'C', 'M', …). A single letter; anything longer is rejected."
        ),
    )
    detail: Literal["list", "full"] = Field(
        default="list",
        description=(
            "'list' returns reference and value only. 'full' adds nodes, model and "
            "params on a netlist, and symbol, position, rotation, pins and bounding "
            "box on a schematic."
        ),
    )
    cursor: str | None = Field(default=None, description=_CURSOR_DESCRIPTION)


class ModelQuery(StrictModel):
    """Find a .model or .subckt definition — by fuzzy name match, or by listing
    everything the given libraries define."""

    kind: Literal["model"]
    mode: Literal["search", "enumerate"] = Field(
        description=(
            "'search' fuzzy-matches 'query' and requires it. 'enumerate' lists every "
            "model in 'libs' and REJECTS a 'query' rather than echoing back a filter "
            "it never applied."
        )
    )
    query: str | None = Field(
        default=None,
        description="Part name or fragment to match. Required by 'search', refused by 'enumerate'.",
    )
    libs: list[str] | None = Field(
        default=None,
        description=(
            "Library files to read. Required by 'enumerate'; optional for 'search', "
            "which searches the session's loaded libraries when it is omitted."
        ),
    )
    cursor: str | None = Field(default=None, description=_CURSOR_DESCRIPTION)

    @model_validator(mode="after")
    def _mode_requirements(self) -> ModelQuery:
        if self.mode == "search" and not self.query:
            raise ValueError("model search requires 'query'")
        if self.mode == "enumerate":
            if not self.libs:
                raise ValueError("model enumerate requires 'libs'")
            # Enumerate lists every model in 'libs' unfiltered. Accepting a
            # 'query' here would echo the caller's filter back on a response
            # that never applied it — reject instead of silently ignoring.
            if self.query is not None:
                raise ValueError("model enumerate does not filter; use mode 'search' with 'query'")
        return self


Query: TypeAlias = Annotated[
    CapabilitiesQuery | SymbolsQuery | SymbolQuery | NetQuery | ComponentsQuery | ModelQuery,
    Field(discriminator="kind"),
]

_QUERY_ADAPTER = TypeAdapter(Query)
_QUERY_MODELS: tuple[type[StrictModel], ...] = get_args(get_args(Query)[0])
SUPPORTED_KINDS: tuple[str, ...] = tuple(
    get_args(model.model_fields["kind"].annotation)[0] for model in _QUERY_MODELS
)
_SUPPORTED_KIND_SET = frozenset(SUPPORTED_KINDS)


class _QueryError(Exception):
    """A per-item failure carrying a structured error code (isolated to one query)."""

    def __init__(self, code: str, message: str, *, supported: list[str] | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.supported = supported


def _validate_query(data: Any) -> Query:
    """Validate one query; an unknown ``kind`` raises with the supported list."""
    if isinstance(data, dict):
        kind = data.get("kind")
        if kind not in _SUPPORTED_KIND_SET:
            raise _QueryError(
                "unsupported_variant",
                f"unsupported query kind {kind!r}; supported kinds: {', '.join(SUPPORTED_KINDS)}",
                supported=list(SUPPORTED_KINDS),
            )
    return _QUERY_ADAPTER.validate_python(data)


class InspectInput(ToolInput):
    queries: list[SkipValidation[Query]] = Field(
        min_length=1,
        max_length=64,
        description=(
            "Independent read-only lookups, 1-64 per call, each tagged by its 'kind'. "
            "Batch freely: they share one round trip and are isolated from each other, "
            "so a denied path, a stale cursor, an unknown kind, or a malformed query "
            "fails only its own item and every other query still returns its data."
        ),
    )

    # SkipValidation keeps the strict discriminated union in the published JSON
    # Schema while letting the handler validate each query independently, so a
    # single malformed item fails only itself (per-item isolation).


# ---------------------------------------------------------------------------
# Pagination — the shared paginator, bound to each query's identity
# ---------------------------------------------------------------------------


def _binding(kind: str, identity: dict[str, Any]) -> str:
    """The cursor's view binding: the paginated ``kind`` plus this query's identity.

    Folding a hash of ``identity`` into the binding means a cursor minted for one
    query cannot resume a different one — a changed path/filter/prefix yields a
    different binding, and the shared codec rejects the mismatch.
    """
    return f"{kind}:{canonical_hash(identity)}"


def _invalid_cursor(exc: PageCursorError) -> _QueryError:
    """A tampered, stale, or cross-query cursor, isolated to the one query."""
    return _QueryError("invalid_cursor", f"cursor is invalid or stale: {exc}")


def _paginate(
    items: list[Any], kind: str, identity: dict[str, Any], cursor: str | None
) -> dict[str, Any]:
    """Page ``items`` through the shared paginator, bound to this query's identity."""
    try:
        return paginate_view(items, _binding(kind, identity), cursor=cursor, limit=_PAGE_SIZE)
    except PageCursorError as exc:
        raise _invalid_cursor(exc) from exc


def _paginate_pair(
    primary: list[Any],
    secondary: list[Any],
    kind: str,
    identity: dict[str, Any],
    cursor: str | None,
) -> dict[str, Any]:
    """Page an item's two collections under its single cursor (both offsets ride in it)."""
    try:
        return paginate_pair(
            primary,
            secondary,
            _binding(kind, identity),
            cursor=cursor,
            limit=_PAGE_SIZE,
            secondary_limit=_COORD_PAGE_SIZE,
        )
    except PageCursorError as exc:
        raise _invalid_cursor(exc) from exc


def _page_meta(page: dict[str, Any]) -> dict[str, int | bool]:
    """The paged-collection facts surfaced in each item's ``page`` field.

    ``total``/``returned`` describe the item's primary collection; ``truncated``
    means "this item has more" and always equals ``next_cursor`` being present,
    so paging until the cursor is null never stops short of a second collection.
    """
    return {
        "total": page["total"],
        "returned": page["returned"],
        "truncated": page["truncated"],
    }


# ---------------------------------------------------------------------------
# capabilities
# ---------------------------------------------------------------------------


def _do_capabilities(state: SessionState) -> dict[str, Any]:
    simulators: dict[str, Any] = {}
    for name, cls in state.available_simulators.items():
        info: dict[str, Any] = {
            "available": True,
            "default": cls is state.default_simulator,
            # Version is not probed (that would run the executable); the raw
            # dialect spicelib parses results with is the cheap, honest fact.
            "version": None,
            "dialect": dialect_for_simulator_name(cls.__name__),
        }
        exe = getattr(cls, "spice_exe", None)
        if exe is not None:
            info["executable"] = str(exe[0] if isinstance(exe, list) else exe)
        simulators[name] = info

    return {
        "simulators": simulators,
        "default_simulator": (
            state.default_simulator.__name__ if state.default_simulator else None
        ),
        # The .asc → LTspice netlist exporter needs LTspice itself.
        "exporter_available": "ltspice" in state.available_simulators,
        "dialects": {
            name: dialect_for_simulator_name(cls.__name__)
            for name, cls in state.available_simulators.items()
        },
        "ngbehavior": (current_ngbehavior() if "ngspice" in state.available_simulators else None),
        "persist_jobs": state.config.persist_jobs,
        "allowed_paths": [str(p) for p in state.config.allowed_paths],
        "tool_profile": state.config.tool_profile,
        "limits": {
            "max_experiment_cases": state.config.max_experiment_cases,
            "analysis_budget_s": state.config.analysis_budget_s,
            "result_set_ttl_hours": state.config.result_set_ttl_hours,
            "max_points_returned": state.config.max_points_returned,
            "max_parallel_sims": state.config.max_parallel_sims,
            "default_timeout_s": state.config.default_timeout,
            "inspect_page_size": _PAGE_SIZE,
            "inspect_coordinate_page_size": _COORD_PAGE_SIZE,
            "dwell": {
                "run_experiments_wait_default_s": _RUN_EXPERIMENTS_WAIT_DEFAULT_S,
                "run_experiments_wait_max_s": _RUN_EXPERIMENTS_WAIT_MAX_S,
                "jobs_wait_default_s": _JOBS_WAIT_DEFAULT_S,
                "jobs_wait_max_s": _JOBS_WAIT_MAX_S,
            },
        },
        "linter_version": linter_version,
    }


# ---------------------------------------------------------------------------
# symbols — legal names + resolution-order precedence
# ---------------------------------------------------------------------------


def _symbol_precedence(asc_dir: Path | None, state: SessionState) -> list[tuple[Path, str]]:
    """The ordered (dir, tier) search precedence, mirroring ``symbol_resolver_for``.

    A schematic-local dir (when a ``path`` is given) leads, then configured
    project libraries, then the stock LTspice/symbol libraries. Duplicates —
    including a symlink pointing at an already-listed directory — collapse to
    their highest-precedence occurrence (keyed by resolved real path).
    """
    from spicelib import AscEditor

    ordered: list[tuple[Path, str]] = []
    if asc_dir is not None:
        ordered.append((asc_dir, "schematic-local"))
    for p in state.config.symbol_paths:
        ordered.append((Path(p), "configured"))
    for p in AscEditor.custom_lib_paths or []:
        ordered.append((Path(p), "configured"))
    for p in getattr(AscEditor, "simulator_lib_paths", None) or []:
        ordered.append((Path(p), "stock"))
    for p in default_stock_paths():
        ordered.append((Path(p), "stock"))

    seen: set[str] = set()
    deduped: list[tuple[Path, str]] = []
    for path, tier in ordered:
        try:
            key = str(path.resolve())
        except OSError:
            key = str(path)
        if key in seen:
            continue
        seen.add(key)
        deduped.append((path, tier))
    return deduped


def _symbols_payload(
    precedence: list[tuple[Path, str]], name_filter: str | None
) -> tuple[list[dict[str, Any]], list[str]]:
    """Walk the precedence dirs, collecting legal ``.asy`` names (immutable I/O)."""
    prec_report: list[dict[str, Any]] = []
    names: list[str] = []
    seen_names: set[str] = set()
    filt = name_filter.lower() if name_filter else None
    for path, tier in precedence:
        exists = path.is_dir()
        prec_report.append({"dir": str(path), "tier": tier, "exists": exists})
        if not exists:
            continue
        for asy in path.rglob("*.asy"):
            stem = asy.stem
            if stem in seen_names:
                continue
            seen_names.add(stem)
            if filt is not None and filt not in stem.lower():
                continue
            names.append(stem)
    names.sort(key=str.lower)
    return prec_report, names


async def _do_symbols(q: SymbolsQuery, state: SessionState) -> dict[str, Any]:
    asc_dir: Path | None = None
    if q.path is not None:
        asc_dir = _resolve_path(q, q.path, state).parent

    precedence = _symbol_precedence(asc_dir, state)
    prec_report, names = await asyncio.to_thread(_symbols_payload, precedence, q.filter)

    page = _paginate(
        names, "symbols", {"path": str(asc_dir) if asc_dir else None, "filter": q.filter}, q.cursor
    )
    return {
        "data": {
            "precedence": prec_report,
            "symbols": page["items"],
            "total": page["total"],
            "returned": page["returned"],
            "filter": q.filter,
        },
        "next_cursor": page["next_cursor"],
        "page": _page_meta(page),
    }


# ---------------------------------------------------------------------------
# symbol — pins per rotation, bbox, origin
# ---------------------------------------------------------------------------


def _symbol_geometry(resolver: SymbolResolver, name: str) -> dict[str, Any] | None:
    """Resolve + parse a symbol and compute its geometry at every rotation.

    Returns ``None`` when the symbol does not resolve. Pure/immutable — safe to
    run off the event loop.
    """
    asy = resolver.resolve(name)
    if asy is None:
        return None
    sym_info = parse_asy_file(asy)
    pins_by_rotation: dict[str, list[dict[str, Any]]] = {}
    bbox_by_rotation: dict[str, dict[str, int]] = {}
    for rot in ROTATIONS:
        geom = compute_placed_geometry(sym_info, 0, 0, rot)
        pins_by_rotation[rot] = geom["pins"]
        bbox_by_rotation[rot] = geom["bounding_box"]
    return {
        "symbol": sym_info.name,
        "description": sym_info.description,
        "source_path": str(asy),
        "origin": {"x": 0, "y": 0},
        "bounding_box": sym_info.bbox.to_origin_size_dict(),
        "pins_by_rotation": pins_by_rotation,
        "bbox_by_rotation": bbox_by_rotation,
    }


async def _do_symbol(q: SymbolQuery, state: SessionState) -> dict[str, Any]:
    asc_path: Path | None = None
    if q.path is not None:
        asc_path = _resolve_path(q, q.path, state)
    resolver = symbol_resolver_for(asc_path, state)
    payload = await asyncio.to_thread(_symbol_geometry, resolver, q.name)
    if payload is None:
        raise _QueryError(
            "symbol_not_found",
            f"symbol '{q.name}' did not resolve against the active symbol libraries; "
            "inspect(symbols) lists the legal names and their resolution precedence",
        )
    return {"data": payload}


# ---------------------------------------------------------------------------
# net — .asc geometric trace vs netlist card-membership
# ---------------------------------------------------------------------------


def _netlist_node_from_at(at: str | list[int], nodes_of: dict[str, list[str]]) -> str:
    """Resolve a netlist ``at`` to a node name. Netlists carry no geometry, so a
    coordinate is rejected; ``REF.<terminal>`` maps to the node at that 1-based
    terminal; ``net:NAME`` and a bare name are node names directly."""
    if isinstance(at, list):
        raise _QueryError(
            "invalid_at",
            "coordinates address geometry; a netlist net is addressed by node name "
            "(at='net:NAME' or at='REF.<terminal>')",
        )
    if at.startswith("net:"):
        return at[4:]
    if "." in at:
        ref, _, term = at.rpartition(".")
        if term.isdigit():
            nodes = nodes_of.get(ref.lower())
            if nodes is None:
                raise _QueryError("not_found", f"component '{ref}' not found in netlist")
            index = int(term)
            if not 1 <= index <= len(nodes):
                raise _QueryError(
                    "invalid_at",
                    f"terminal {index} out of range for '{ref}' ({len(nodes)} terminals)",
                )
            return nodes[index - 1]
        # A dotted name whose tail is not a terminal index is a hierarchical node.
        return at
    return at


def _route_circuit_kind(path: Path, query: str) -> Literal["asc", "netlist"]:
    """Classify a resolved circuit ``path`` as an ``.asc`` schematic or a netlist.

    Anything else fails only this item with an ``unsupported_file`` error naming
    what the ``query`` kind accepts. Shared by the ``net`` and ``components``
    paths so their file routing and error text stay identical.
    """
    suffix = path.suffix.lower()
    if suffix == ".asc":
        return "asc"
    if suffix in NETLIST_SUFFIXES:
        return "netlist"
    raise _QueryError(
        "unsupported_file",
        f"'{suffix}' is not a circuit file; {query} queries take a .asc "
        "schematic or a .cir / .net / .sp netlist",
    )


def _net_netlist_payload(text: str, at: str | list[int]) -> dict[str, Any]:
    """Card-membership for a node in a netlist — NO geometry keys (by contract)."""
    cards = lex(text).cards
    by_ref = instances_by_ref(cards)
    parsed: dict[str, InstanceLine] = {}
    nodes_of: dict[str, list[str]] = {}
    unparseable = 0
    for ref_lower, card in by_ref.items():
        try:
            line = InstanceLine.from_card(card)
        except Exception:
            unparseable += 1
            continue
        parsed[ref_lower] = line
        nodes_of[ref_lower] = line.nodes

    node = _netlist_node_from_at(at, nodes_of)
    node_lower = node.lower()
    members: list[dict[str, Any]] = []
    for line in parsed.values():
        for i, n in enumerate(line.nodes, start=1):
            if n.lower() == node_lower:
                members.append({"reference": line.ref, "terminal": i})
    members.sort(key=lambda m: (m["reference"], m["terminal"]))
    return {"node": node, "members": members, "unparseable_cards": unparseable}


async def _do_net(q: NetQuery, state: SessionState) -> dict[str, Any]:
    path = _resolve_path(q, q.path, state)

    if _route_circuit_kind(path, "net") == "netlist":
        try:
            text = await asyncio.to_thread(read_spice_text, path)
        except OSError as exc:
            raise _QueryError("read_error", str(exc)) from exc
        try:
            payload = await asyncio.to_thread(_net_netlist_payload, text, q.at)
        except SpiceLexError as exc:
            raise _QueryError("parse_error", str(exc)) from exc
        members = payload.pop("members")
        page = _paginate(members, "net", {"path": str(path), "at": q.at}, q.cursor)
        return {
            "data": {
                "source": "netlist",
                "node": payload["node"],
                "members": page["items"],
                "total_members": page["total"],
                "returned": page["returned"],
                "unparseable_cards": payload["unparseable_cards"],
            },
            "next_cursor": page["next_cursor"],
            "page": _page_meta(page),
        }

    # .asc: geometric trace via trace_net internals. The cached AscEditor is
    # touched only on the event loop, so this runs inline (never offloaded).
    trace_input = _trace_input_for(q.path, q.at)
    trace = await handle_trace_net(trace_input, state)
    tdata = trace.structuredContent or {}
    pins = list(tdata.get("pins", []))
    coords = list(tdata.get("coordinates", []))

    # Pins and wire vertices are two independently long collections of one net,
    # and the item carries one cursor — so both offsets ride in it and both
    # advance. A coordinate window that restarted every page would re-serve the
    # same vertices forever while claiming more existed.
    page = _paginate_pair(pins, coords, "net.asc", {"path": str(path), "at": q.at}, q.cursor)
    data: dict[str, Any] = {
        "source": "schematic",
        "start": tdata.get("start"),
        "labels": tdata.get("labels", []),
        "pins": page["items"],
        "total_pins": page["total"],
        "returned": page["returned"],
        "is_shorted": tdata.get("is_shorted", False),
        "coordinates": page["secondary_items"],
        "total_coordinates": page["secondary_total"],
        "returned_coordinates": page["secondary_returned"],
        "coordinates_truncated": page["secondary_truncated"],
    }
    if page["secondary_truncated"]:
        data["hint"] = (
            f"{page['secondary_returned']} of {page['secondary_total']} wire-vertex "
            "coordinates returned; repeat this query with next_cursor for the rest "
            "(one cursor advances pins and coordinates independently)."
        )
    if tdata.get("warnings"):
        data["warnings"] = tdata["warnings"]
    return {"data": data, "next_cursor": page["next_cursor"], "page": _page_meta(page)}


def _trace_input_for(path: str, at: str | list[int]) -> TraceNetInput:
    if isinstance(at, list):
        return TraceNetInput(path=path, x=at[0], y=at[1])
    if at.startswith("net:") or "." in at:
        return TraceNetInput(path=path, pin=at)
    raise _QueryError(
        "invalid_at",
        "'at' on a schematic must be 'REF.PIN', 'net:NAME', or [x, y]",
    )


# ---------------------------------------------------------------------------
# components — list vs full, over .asc or netlist
# ---------------------------------------------------------------------------


def _check_prefix(prefix: str | None) -> None:
    if prefix is not None and (len(prefix) != 1 or not prefix.isalpha()):
        raise _QueryError(
            "invalid_prefix",
            f"component prefix must be a single letter (e.g. 'R', 'C'), got {prefix!r}",
        )


def _components_netlist_payload(
    text: str, prefix: str | None, detail: str
) -> list[dict[str, Any]]:
    from ltspice_mcp.lib.spice_lex_views import body_has_stray_kv_remnant

    cards = lex(text).cards
    by_ref = instances_by_ref(cards)
    upper = prefix.upper() if prefix else None
    rows: list[dict[str, Any]] = []
    for card in by_ref.values():
        ref = card.name
        if not ref:
            continue
        if upper is not None and ref[:1].upper() != upper:
            continue
        entry: dict[str, Any] = {"reference": ref, "value": netlist_card_value(card)}
        if detail == "full" and not body_has_stray_kv_remnant(card.body):
            try:
                line: InstanceLine | None = InstanceLine.from_card(card)
            except Exception:
                line = None
            if line is not None:
                entry["nodes"] = list(line.nodes)
                entry["model"] = line.model
                entry["params"] = dict(line.params)
        rows.append(entry)
    rows.sort(key=lambda r: r["reference"])
    return rows


def _components_asc_page(editor: Any, refs: list[str], detail: str) -> list[dict[str, Any]]:
    """Build per-component detail for a page of .asc references (editor on loop)."""
    from ltspice_mcp.lib.symbol_geometry import get_symbol_info

    rows: list[dict[str, Any]] = []
    for ref in refs:
        try:
            value = services.asc_component_value(editor, ref)
        except Exception:
            value = "<unparseable>"
        entry: dict[str, Any] = {"reference": ref, "value": value}
        comp = editor.components.get(ref)
        if comp is not None:
            attrs = services.asc_component_attributes(comp)
            if attrs:
                entry["attributes"] = attrs
            if detail == "full":
                pos, erot = editor.get_component_position(ref)
                rot_str = erot.name if erot else "R0"
                entry["symbol"] = comp.symbol
                entry["position"] = {"x": pos.X, "y": pos.Y}
                entry["rotation"] = rot_str
                sym_info = get_symbol_info(comp.symbol) if comp.symbol else None
                if sym_info is not None:
                    geom = compute_placed_geometry(sym_info, int(pos.X), int(pos.Y), rot_str)
                    entry["pins"] = geom["pins"]
                    entry["bounding_box"] = geom["bounding_box"]
        rows.append(entry)
    return rows


async def _do_components(q: ComponentsQuery, state: SessionState) -> dict[str, Any]:
    _check_prefix(q.prefix)
    path = _resolve_path(q, q.path, state)
    identity = {"path": str(path), "prefix": q.prefix, "detail": q.detail}

    if _route_circuit_kind(path, "components") == "asc":
        # Cached editor + component reads stay on the event loop.
        editor = _get_asc_editor(path, state)
        try:
            refs = sorted(editor.get_components(q.prefix) if q.prefix else editor.get_components())
        except Exception as exc:
            raise _QueryError("parse_error", f"failed to list components: {exc}") from exc
        page = _paginate(refs, "components", identity, q.cursor)
        rows = _components_asc_page(editor, page["items"], q.detail)
    else:
        try:
            text = await asyncio.to_thread(read_spice_text, path)
        except OSError as exc:
            raise _QueryError("read_error", str(exc)) from exc
        try:
            all_rows = await asyncio.to_thread(
                _components_netlist_payload, text, q.prefix, q.detail
            )
        except SpiceLexError as exc:
            raise _QueryError("parse_error", str(exc)) from exc
        page = _paginate(all_rows, "components", identity, q.cursor)
        rows = page["items"]

    return {
        "data": {
            "components": rows,
            "detail": q.detail,
            "prefix": q.prefix,
            "total": page["total"],
            "returned": page["returned"],
        },
        "next_cursor": page["next_cursor"],
        "page": _page_meta(page),
    }


# ---------------------------------------------------------------------------
# model — search (fuzzy) vs enumerate (source-driven)
# ---------------------------------------------------------------------------


def _model_entry(entry: Any) -> dict[str, Any]:
    return {
        "name": entry.name,
        "type": entry.model_type,
        "source_path": str(entry.source_path),
        "ports": list(entry.ports),
        "params": dict(entry.params),
    }


def _enumerate_libs(lib_paths: list[Path]) -> list[dict[str, Any]]:
    """Every model/subcircuit defined across the given library files (immutable parse)."""
    rows: list[dict[str, Any]] = []
    for lib in lib_paths:
        index = parse_library_file_cached(lib)
        for entry in index.models:
            rows.append(_model_entry(entry))
    rows.sort(key=lambda r: (r["name"].lower(), r["source_path"]))
    return rows


def _search_libs(lib_paths: list[Path], query: str, cutoff: float = 0.6) -> list[dict[str, Any]]:
    """Fuzzy-match ``query`` against the models defined in the given library files."""
    query_lower = query.lower()
    scored: list[tuple[float, dict[str, Any]]] = []
    seen: set[str] = set()
    for lib in lib_paths:
        index = parse_library_file_cached(lib)
        for entry in index.models:
            score = _part_aware_score(query_lower, entry.name_lower)
            if score < cutoff or entry.name_lower in seen:
                continue
            seen.add(entry.name_lower)
            row = _model_entry(entry)
            row["score"] = round(score, 3)
            scored.append((score, row))
    scored.sort(key=lambda pair: (-pair[0], pair[1]["name"].lower()))
    return [row for _, row in scored]


async def _do_model(q: ModelQuery, state: SessionState) -> dict[str, Any]:
    if q.mode == "enumerate":
        lib_paths = [_resolve_path(q, lib, state) for lib in (q.libs or [])]
        try:
            rows = await asyncio.to_thread(_enumerate_libs, lib_paths)
        except OSError as exc:
            raise _QueryError("read_error", str(exc)) from exc
        identity: dict[str, Any] = {"mode": "enumerate", "libs": [str(p) for p in lib_paths]}
    else:
        assert q.query is not None  # guaranteed by the model validator
        if q.libs:
            lib_paths = [_resolve_path(q, lib, state) for lib in q.libs]
            try:
                rows = await asyncio.to_thread(_search_libs, lib_paths, q.query)
            except OSError as exc:
                raise _QueryError("read_error", str(exc)) from exc
        else:
            # No libs given: fall back to the session's loaded libraries (loop-owned
            # mutable state — read inline, never offloaded).
            try:
                rows = state.libraries.find_similar_models(
                    q.query, exact=False, limit=10_000, cutoff=0.6
                )
            except Exception as exc:
                raise _QueryError("search_error", str(exc)) from exc
        identity = {"mode": "search", "query": q.query, "libs": q.libs}

    page = _paginate(rows, "model", identity, q.cursor)
    return {
        "data": {
            "mode": q.mode,
            "query": q.query,
            "results": page["items"],
            "total": page["total"],
            "returned": page["returned"],
        },
        "next_cursor": page["next_cursor"],
        "page": _page_meta(page),
    }


# ---------------------------------------------------------------------------
# Shared per-item helpers + dispatch
# ---------------------------------------------------------------------------


def _resolve_path(q: Any, user_path: str, state: SessionState) -> Path:
    """safe_path with the denial mapped to a per-item ``path_denied`` failure."""
    try:
        return safe_path(user_path, state)
    except PathSecurityError as exc:
        raise _QueryError("path_denied", str(exc)) from exc


async def _dispatch(query: Query, state: SessionState) -> dict[str, Any]:
    if isinstance(query, CapabilitiesQuery):
        return {"data": _do_capabilities(state)}
    if isinstance(query, SymbolsQuery):
        return await _do_symbols(query, state)
    if isinstance(query, SymbolQuery):
        return await _do_symbol(query, state)
    if isinstance(query, NetQuery):
        return await _do_net(query, state)
    if isinstance(query, ComponentsQuery):
        return await _do_components(query, state)
    # Exhaustive over the sealed union: ModelQuery is the only remaining member.
    return await _do_model(query, state)


_ERROR_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "code": {"type": "string"},
        "message": {"type": "string"},
        "supported": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["code", "message"],
}

_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        # Call-level outcome over the per-item batch: complete when every query
        # succeeded, partial when any query failed. The shared envelope's other
        # values are absent because inspect cannot reach them: per-item
        # isolation turns every query fault into that item's error, and a
        # call-level fault raises — the SDK then answers with isError and no
        # structuredContent, so it is never carried by this envelope.
        "outcome": {"type": "string", "enum": ["complete", "partial"]},
        "results": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "index": {"type": "integer"},
                    "kind": {"type": ["string", "null"]},
                    "ok": {"type": "boolean"},
                    "error": _ERROR_SCHEMA,
                    # Kind-specific payload; its shape is documented per kind in
                    # the module docstring and stays open here by design.
                    "data": {"type": "object"},
                    "next_cursor": {"type": ["string", "null"]},
                    "page": {
                        "type": "object",
                        "properties": {
                            "total": {"type": "integer"},
                            "returned": {"type": "integer"},
                            "truncated": {"type": "boolean"},
                        },
                    },
                },
                "required": ["index", "kind", "ok"],
                "additionalProperties": False,
            },
        },
        "count": {"type": "integer"},
        "ok_count": {"type": "integer"},
        "error_count": {"type": "integer"},
        "hint": HINT_SCHEMA,
    },
    "required": ["outcome", "results", "count"],
}

INSPECT_DESCRIPTION = (
    "Read-only lookups over the server and the circuits it can reach, batched as "
    "independent 'queries'. Kinds: 'capabilities' (simulators, dialects, exporter, "
    "job persistence, allowed roots, profile, limits, linter version); 'symbols' "
    "(legal .asy names + resolution-order precedence; a 'path' adds that "
    "schematic's own directory to the front); 'symbol' (one symbol's pins per "
    "rotation R0..M270, bounding box, origin); 'net' (on a .asc: a geometric trace "
    "of pins, wire vertices, labels, and shorts; on a .cir/.net/.sp: which element "
    "cards reference the node, with no geometry); 'components' (the component list, "
    "or full per-component detail with detail='full'); 'model' (search fuzzy-matches "
    "a 'query'; enumerate lists every model in the given 'libs'). A denied path, a "
    "stale or tampered cursor, an unknown kind, or a malformed query fails only that "
    "item — every other query still returns. Paginated kinds resume via 'cursor'. "
    "This is the surface's only honestly read-only tool."
)


@registry.tool(
    name="inspect",
    description=INSPECT_DESCRIPTION,
    input_model=InspectInput,
    annotations=RO_ANNOTATIONS,
    profiles=("consolidated",),
    output_schema=_OUTPUT_SCHEMA,
)
async def handle_inspect(args: InspectInput, state: SessionState) -> types.CallToolResult:
    """Answer a batch of read-only queries with per-item success/failure isolation."""
    results: list[dict[str, Any]] = []
    ok_count = 0
    error_count = 0

    for index, raw in enumerate(args.queries):
        try:
            query = _validate_query(raw)
            outcome = await _dispatch(query, state)
        except _QueryError as exc:
            error = {"code": exc.code, "message": exc.message}
            if exc.supported is not None:
                error["supported"] = exc.supported  # type: ignore[assignment]
            results.append(_failure_item(index, raw, error))
            error_count += 1
            continue
        except ValidationError as exc:
            results.append(
                _failure_item(
                    index, raw, {"code": "invalid_query", "message": _compact_error(exc)}
                )
            )
            error_count += 1
            continue
        except LTSpiceMCPError as exc:
            results.append(_failure_item(index, raw, {"code": "error", "message": str(exc)}))
            error_count += 1
            continue
        except Exception as exc:
            # Last-resort isolation: an unexpected fault in one query must not
            # sink the batch — the per-item contract is that others still return.
            results.append(
                _failure_item(index, raw, {"code": "internal_error", "message": str(exc)})
            )
            error_count += 1
            continue

        item: dict[str, Any] = {"index": index, "kind": query.kind, "ok": True}
        item.update(outcome)
        results.append(item)
        ok_count += 1

    data: dict[str, Any] = {
        # Per-item failures isolate to their result and never fail the call, so
        # the batch is "partial" when any query failed and "complete" otherwise.
        "outcome": "partial" if error_count else "complete",
        "results": results,
        "count": len(results),
        "ok_count": ok_count,
        "error_count": error_count,
    }
    if error_count:
        data["hint"] = (
            f"{error_count} of {len(results)} queries failed; see each result's "
            "'error.code'. Other queries returned normally."
        )
    return format_response(_summary_text(results), data)


def _failure_item(index: int, raw: Any, error: dict[str, Any]) -> dict[str, Any]:
    return {"index": index, "kind": _kind_of(raw), "ok": False, "error": error}


def _kind_of(raw: Any) -> str | None:
    kind = raw.get("kind") if isinstance(raw, dict) else getattr(raw, "kind", None)
    return kind if isinstance(kind, str) else None


def _compact_error(exc: ValidationError) -> str:
    return "; ".join(err["msg"] for err in exc.errors(include_url=False)) or str(exc)


def _summary_text(results: list[dict[str, Any]]) -> str:
    lines = []
    for r in results:
        if r["ok"]:
            lines.append(f"[{r['index']}] {r['kind']}: ok")
        else:
            lines.append(
                f"[{r['index']}] {r.get('kind')}: {r['error']['code']} — {r['error']['message']}"
            )
    return "\n".join(lines) if lines else "(no queries)"
