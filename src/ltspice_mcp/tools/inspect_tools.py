"""inspect — the consolidated read-only UNDERSTAND surface.

One tool answers a batch of independent read-only ``queries`` about the server
and the circuits it can reach. Each query is one of seven kinds:

* ``capabilities`` — detected simulators + dialects, exporter presence, job
  persistence, allowed roots, the active profile and which of the two tool
  listings this session was served, the configured limits, the
  linter version, and ``diagnostics``: the startup notes (bad configured
  simulator path, a requested engine that fell back, WSL auto-detection) that
  say whether this server started degraded. Pulled from
  ``state``/``config``/``lint_rules``; nothing is probed.
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

On a netlist, ``net`` and ``components`` add ``warnings`` when the lexer had to
guess about the deck (an unclosed ``.SUBCKT``, an ``.ENDS`` matching nothing, a
stray continuation) — the answer was read from cards that mean something other
than the file says, and the key is absent when it read cleanly.

Both circuit kinds report the sheet's ``sha256`` when the target is a ``.asc``
— the token ``edit_schematic`` requires as ``expected_sha256``. Reading it here
is what lets a first edit commit in one call; an edit attempted without it is
refused with the current digest attached, so that path costs one retry rather
than a hunt.
* ``model`` — model/subcircuit lookup: ``search`` fuzzy-matches a ``query``;
  ``enumerate`` lists every model defined in the given ``libs``.
* ``reference`` — the tools' own branch vocabulary (``tools/reference_index.py``):
  a plain-words ``query`` returns the closest analysis recipes, schematic ops,
  variation kinds, checks and job actions with their full field tables, and no
  ``query`` returns the table of contents. It is the answer to "which recipe
  gives me phase margin" and to "what does this branch take" on the ``compact``
  tool listing, where per-argument descriptions are not on the wire at all. It
  reads no file and touches no session state.

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
import copy
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any, Literal, TypeAlias

from mcp import types
from pydantic import (
    BaseModel,
    Field,
    SkipValidation,
    TypeAdapter,
    ValidationError,
    model_validator,
)

from ltspice_mcp.errors import (
    LTSpiceMCPError,
    NetlistError,
    PathSecurityError,
    compact_validation_error,
)
from ltspice_mcp.lib import response_budget, services
from ltspice_mcp.lib.cache import file_stamp
from ltspice_mcp.lib.cursor_codec import canonical_hash
from ltspice_mcp.lib.deck_staging import sha256_file
from ltspice_mcp.lib.encoding import read_spice_text
from ltspice_mcp.lib.library_manager import parse_library_file_cached, part_aware_score
from ltspice_mcp.lib.lint_rules import linter_version
from ltspice_mcp.lib.model_fields import literal_values, model_union
from ltspice_mcp.lib.pin_legend import PageCursorError, paginate_pair, paginate_view
from ltspice_mcp.lib.schematic_ops import (
    get_asc_editor,
    named_labels,
    net_partition,
    netlist_card_value,
    require_asc,
    resolve_pin,
    same_instance_dropped_segments,
    wire_segments_of,
)
from ltspice_mcp.lib.schematic_scene import SymbolResolver, default_stock_paths
from ltspice_mcp.lib.simulator import (
    SIMULATORS,
    current_ngbehavior,
    dialect_for_simulator_name,
    simulator_remediation,
)
from ltspice_mcp.lib.spice_lex import LexResult, SpiceLexError, lex
from ltspice_mcp.lib.spice_lex_views import InstanceLine, instances_by_ref
from ltspice_mcp.lib.symbol_geometry import compute_placed_geometry, parse_asy_file
from ltspice_mcp.state import SessionState
from ltspice_mcp.tools._base import (
    FORMAT_DESCRIPTION,
    HINT_SCHEMA,
    RO_ANNOTATIONS,
    WARNINGS_SCHEMA,
    ResponseBudget,
    StrictModel,
    ToolInput,
    declare_output_schema,
    format_response,
    outcome_of,
    outcome_schema,
    registry,
    resolve_response_budget,
    safe_path,
    symbol_resolver_for,
)
from ltspice_mcp.tools.reference_index import build_index, search_branches, table_of_contents


class TraceNetInput(ToolInput):
    path: str = Field(description="Path to an .asc schematic")
    pin: str | None = Field(
        default=None,
        description=(
            "Pin or net reference to start from: 'Ref.Pin' (e.g. 'M1.D'), "
            "'net:NAME' (e.g. 'net:VDD'), or omit and pass x/y."
        ),
    )
    x: int | None = Field(default=None, description="X coordinate (with y) to trace from")
    y: int | None = Field(default=None, description="Y coordinate (with x) to trace from")
    format: Literal["json", "text"] | None = Field(
        default=None,
        description=FORMAT_DESCRIPTION,
    )


@declare_output_schema(
    {
        "type": "object",
        "properties": {
            "start": {
                "type": "object",
                "properties": {"x": {"type": "integer"}, "y": {"type": "integer"}},
            },
            "labels": {"type": "array", "items": {"type": "string"}},
            "pins": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "reference": {"type": "string"},
                        "pin": {"type": "string"},
                        "x": {"type": "integer"},
                        "y": {"type": "integer"},
                    },
                },
            },
            "coordinates": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"x": {"type": "integer"}, "y": {"type": "integer"}},
                },
            },
            "is_shorted": {"type": "boolean"},
            "warnings": WARNINGS_SCHEMA,
        },
    }
)
async def handle_trace_net(args: TraceNetInput, state: SessionState) -> types.CallToolResult:
    """Trace every pin/label/wire vertex on the net at a pin, label, or (x,y)."""
    asc_path = safe_path(args.path, state)
    require_asc(asc_path)
    editor = get_asc_editor(asc_path, state)

    if args.pin is not None and args.pin.startswith("net:"):
        # A net: reference legitimately matches many same-name FLAGs — the
        # normal case on label-wired schematics (one FLAG per pin).
        # resolve_pin refuses ambiguous net labels, but trace_net's own
        # name-merge step below absorbs duplicates, so just seed from any
        # matching label coordinate (lowest, for determinism).
        net_name = args.pin[4:]
        matches = sorted(
            (int(lbl.coord.X), int(lbl.coord.Y)) for lbl in editor.labels if lbl.text == net_name
        )
        if not matches:
            raise NetlistError(
                f"Net label '{net_name}' not found in schematic. Add it with the "
                "add_net_label op of edit_schematic first, or trace a "
                "component pin / coordinate."
            )
        x, y = matches[0]
    elif args.pin is not None:
        x, y = resolve_pin(args.pin, editor)
    elif args.x is not None and args.y is not None:
        x, y = args.x, args.y
    else:
        raise NetlistError("inspect(kind='net') needs either 'pin' or both 'x' and 'y'.")

    part = net_partition(editor)
    start = (x, y)
    physical_members = part.members.get(part.root(start), set())
    if start not in physical_members and start not in part.pin_owners:
        # The coordinate isn't on any pin/label/wire endpoint — an empty point
        # (or a bare mid-wire span carrying nothing).
        raise NetlistError(
            f"Nothing found at ({x},{y}): no component pin, net label, or wire "
            "vertex sits there. Use inspect(kind='components') to inspect the layout."
        )

    # The physical partition connects by wire only; LTspice also makes FLAGs
    # with the same NAME electrically common. Fold physical nets that share a
    # label name together (a second union-find over physical roots) so
    # trace_net answers "what's on net X" on label-wired schematics,
    # not just wire-routed ones.
    root_parent: dict[tuple[int, int], tuple[int, int]] = {}

    def _rfind(r: tuple[int, int]) -> tuple[int, int]:
        root_parent.setdefault(r, r)
        while root_parent[r] != r:
            root_parent[r] = root_parent[root_parent[r]]
            r = root_parent[r]
        return r

    label_first: dict[str, tuple[int, int]] = {}
    for root, coords in part.members.items():
        for coord in coords:
            for lbl in part.label_texts.get(coord, ()):
                if lbl in label_first:
                    ra, rb = _rfind(label_first[lbl]), _rfind(root)
                    if ra != rb:
                        root_parent[ra] = rb
                else:
                    label_first[lbl] = root

    target_root = _rfind(part.root(start))
    member_coords: set[tuple[int, int]] = set()
    for root, coords in part.members.items():
        if _rfind(root) == target_root:
            member_coords |= coords
    if not member_coords:
        member_coords = {start}

    labels: set[str] = set()
    pins: list[dict] = []
    for coord in member_coords:
        labels.update(part.label_texts.get(coord, set()))
        for ref, pin_name in part.pin_owners.get(coord, []):
            pins.append({"reference": ref, "pin": pin_name, "x": coord[0], "y": coord[1]})

    named = sorted(named_labels(frozenset(labels)))
    is_shorted = len(named) > 1
    pins.sort(key=lambda p: (p["reference"], p["pin"]))
    coords = sorted(member_coords)

    # LTspice drops a wire segment joining two pins of one component instance,
    # so a net whose shape here depends on such a segment over-reports what
    # LTspice will actually netlist. Surface each dropped segment on this net as
    # a fact (not a verdict) — the model decides whether the tie was intended.
    net_segments = [
        s
        for s in wire_segments_of(editor)
        if (s[0], s[1]) in member_coords and (s[2], s[3]) in member_coords
    ]
    warnings = [
        f"{d['ref']}.{d['pins'][0]} and {d['ref']}.{d['pins'][1]} are joined by a "
        f"wire between two pins of the same component; LTspice drops that wire from "
        f"the netlist, so this connection may not exist in simulation. Reroute the "
        f"wire to bend out of line with the two pins, or label both pins with the "
        f"same net name."
        for d in same_instance_dropped_segments(part.pin_owners, net_segments)
    ]

    data: dict = {
        "start": {"x": x, "y": y},
        "labels": sorted(labels),
        "pins": pins,
        "coordinates": [{"x": cx, "y": cy} for cx, cy in coords],
        "is_shorted": is_shorted,
    }
    if warnings:
        data["warnings"] = warnings

    net_name = ", ".join(sorted(labels)) if labels else "<unnamed>"
    lines = [f"Net at ({x},{y}): {net_name}"]
    if pins:
        lines.append("  Pins:")
        for p in pins:
            lines.append(f"    {p['reference']}.{p['pin']} at ({p['x']},{p['y']})")
    else:
        lines.append("  (no component pins on this net)")
    if is_shorted:
        lines.append(f"  WARNING: net carries multiple labels {named} — likely a short.")
    for w in warnings:
        lines.append(f"  WARNING: {w}")
    return format_response("\n".join(lines), data, args.format)


# The .asc net and component paths share lib/schematic_ops.py's cached editor —
# the same object an edit mutates, so a read never sees a stale sheet.

# Fixed server-side page size for the paginated kinds. The A.5 query specs list
# a cursor but no caller-facing limit knob, so the page is a server constant.
_PAGE_SIZE = 100

# Page size for the .asc net trace's wire-vertex coordinates — the second
# collection of the net item, paged alongside its pins under the same cursor.
# Larger than the pin page because a coordinate is two integers.
_COORD_PAGE_SIZE = 500

NETLIST_SUFFIXES = frozenset({".cir", ".net", ".sp"})


@dataclass(frozen=True)
class _View:
    """How much of an answer a query pass renders — the budget ladder's inputs.

    Frozen and hashable so a pass can be reused: the ladder walks three rungs
    but only two of them change what a query has to read, and re-reading a
    library tree per rung would pay for the budget twice.
    """

    # Read at construction, not at class definition, so the module constant
    # stays the live source of the default page size rather than a value copied
    # into this class once at import. test_response_budget's shrink-ladder test
    # rests on that: it raises _PAGE_SIZE to page a wider set in one go.
    limit: int = field(default_factory=lambda: _PAGE_SIZE)
    coord_limit: int = field(default_factory=lambda: _COORD_PAGE_SIZE)
    #: Revoke the caller's detail opt-in (``detail='full'``) — the answer rung.
    lean: bool = False
    #: Whether the budget's shrink rung sized the limits above. Set from the
    #: rung rather than inferred from a limit, so a kind that names the lever a
    #: caller should reach for cannot mistake a small page size for a budget
    #: cut. Out of the comparison because it changes nothing a query READS, so
    #: two rungs that ask for the same rows still share one pass.
    shrunk: bool = field(default=False, compare=False)


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
    "Page token from a previous 'next_cursor' — echo it back unmodified. It "
    "binds this query and carries a row offset, not a snapshot."
)

#: Most matches one reference lookup will return. Each match carries a full
#: field table, so the cap is what stops a vague query from answering with the
#: whole vocabulary at full detail.
REFERENCE_LIMIT_CAP = 20

# The file-backed kinds bind the file itself, so the stronger claim holds there.
_CURSOR_DESCRIPTION_FILE = (
    "Page token from a previous 'next_cursor' — echo it back unmodified. It "
    "binds this query and the file's size and modification time, so a token "
    "minted before an edit is rejected."
)


class CapabilitiesQuery(StrictModel):
    """What this server can do: detected simulators and their raw dialects,
    whether the .asc exporter is available, job persistence, allowed roots, the
    configured limits, and the linter version. Takes no arguments."""

    kind: Literal["capabilities"]


class SymbolsQuery(StrictModel):
    """The .asy symbol names that resolve, and the directory precedence they
    resolve through. Ask it before an add_component op to learn the name to
    place, and after a symbol_unresolved failure to see what this box has."""

    kind: Literal["symbols"]
    path: str | None = Field(
        default=None,
        description=(
            "Optional schematic whose own directory is put at the front of the "
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
    origin. The pins reported for a rotation are where they land when the part
    is placed at it."""

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
    is card membership, with no geometry."""

    kind: Literal["net"]
    path: str = Field(description="The .asc schematic, or .cir/.net/.sp netlist, to read.")
    at: str | list[int] = Field(
        description=(
            "Where the net is: 'REF.PIN' (e.g. 'M1.D'), 'net:NAME', or [x, y]; "
            "on a netlist, which has no geometry, it takes 'net:NAME', a node "
            "name, or 'REF.<terminal-number>' and rejects a coordinate."
        )
    )
    cursor: str | None = Field(default=None, description=_CURSOR_DESCRIPTION_FILE)

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
            "'list' returns reference and value only; 'full' adds nodes, model "
            "and params on a netlist, and symbol, position, rotation, pins and "
            "bounding box on a schematic."
        ),
    )
    cursor: str | None = Field(default=None, description=_CURSOR_DESCRIPTION_FILE)


class ModelQuery(StrictModel):
    """Find a .model or .subckt definition — by fuzzy name match, or by listing
    everything the given libraries define. Ask it when a run failed on a model
    the deck names, or to get a part's exact spelling before writing it in."""

    kind: Literal["model"]
    mode: Literal["search", "enumerate"] = Field(
        description=(
            "'search' fuzzy-matches 'query' and requires it; 'enumerate' lists "
            "every model in 'libs' and rejects a 'query' rather than echoing "
            "back a filter it never applied."
        )
    )
    query: str | None = Field(
        default=None,
        description="Part name or fragment to match; required by 'search', refused by 'enumerate'.",
    )
    libs: list[str] | None = Field(
        default=None,
        description=(
            "Library files to read: required by 'enumerate', optional for "
            "'search', which searches the session's loaded libraries when omitted."
        ),
    )
    cursor: str | None = Field(
        default=None,
        description=(
            _CURSOR_DESCRIPTION_FILE + " The files are those named in 'libs'; "
            "with 'libs' omitted it binds the query alone."
        ),
    )

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


class ReferenceQuery(StrictModel):
    """Look up this server's tool vocabulary — each tool's own arguments, plus
    analysis recipes, schematic ops, variation kinds, query kinds, checks and
    job actions — with every field's type, default and units."""

    kind: Literal["reference"]
    query: str | None = Field(
        default=None,
        description=(
            "What you are trying to measure or do, in plain words: 'phase "
            "margin', 'connect two pins'. Omit it for the table of contents: "
            "every branch name and one line, no fields."
        ),
    )
    limit: int = Field(
        default=5,
        ge=1,
        le=REFERENCE_LIMIT_CAP,
        description="How many matches to return with their full field tables.",
    )


Query: TypeAlias = Annotated[
    CapabilitiesQuery
    | SymbolsQuery
    | SymbolQuery
    | NetQuery
    | ComponentsQuery
    | ModelQuery
    | ReferenceQuery,
    Field(discriminator="kind"),
]

_QUERY_ADAPTER = TypeAdapter(Query)
QUERY_MODELS: tuple[type[BaseModel], ...] = model_union(Query)
SUPPORTED_KINDS: tuple[str, ...] = tuple(
    values[0] for model in QUERY_MODELS if (values := literal_values(model, "kind"))
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
            "Independent read-only lookups, 1-64 per call, each tagged by its "
            "'kind'. Batch freely: they share one round trip and are isolated, "
            "so a bad path, cursor or kind fails only its own item."
        ),
    )
    budget: int | None = Field(
        default=None,
        ge=response_budget.BUDGET_MIN_TOKENS,
        description=response_budget.BUDGET_DESCRIPTION,
    )

    # SkipValidation keeps the strict discriminated union in the published JSON
    # Schema while letting the handler validate each query independently, so a
    # single malformed item fails only itself (per-item isolation).


# ---------------------------------------------------------------------------
# Pagination — the shared paginator, bound to each query's identity
# ---------------------------------------------------------------------------


def _binding(kind: str, identity: dict[str, Any], sources: Sequence[Path]) -> str:
    """The cursor's view binding: the paginated ``kind``, this query's identity,
    and the revision of every file the rows were derived from.

    Folding a hash of ``identity`` into the binding means a cursor minted for one
    query cannot resume a different one — a changed path/filter/prefix yields a
    different binding, and the shared codec rejects the mismatch. ``sources``
    extends that to the files themselves: a cursor carries a row offset into a
    list the server re-derives on every page, so without their stamps a token
    minted before an edit still validates and seeks into the NEW list, silently
    skipping or repeating rows. With them the binding changes and the stale token
    is rejected — a clean error instead of a wrong page.

    ``sources`` is required rather than opt-in so a new file-backed kind cannot
    forget it; a kind that pages something the server does not read off named
    files passes ``()`` on purpose. Stat granularity bounds the guarantee: a
    rewrite of identical size within one filesystem clock tick still reads as
    unchanged.
    """
    bound = dict(identity)
    if sources:
        bound["sources"] = [[str(path), _file_stamp(path)] for path in sources]
    return f"{kind}:{canonical_hash(bound)}"


def _file_stamp(path: Path) -> list[int] | None:
    """This file's revision marker, or ``None`` when it cannot be stat'd."""
    try:
        return list(file_stamp(path))
    except OSError:
        return None


def _invalid_cursor(exc: PageCursorError) -> _QueryError:
    """A tampered, stale, or cross-query cursor, isolated to the one query."""
    return _QueryError("invalid_cursor", f"cursor is invalid or stale: {exc}")


def _paginate(
    items: list[Any],
    kind: str,
    identity: dict[str, Any],
    cursor: str | None,
    sources: Sequence[Path],
    view: _View,
) -> dict[str, Any]:
    """Page ``items`` through the shared paginator, bound to this query's identity
    and to the revision of the ``sources`` the rows came from.

    The limit comes from ``view``, so a budget that shrinks the page shrinks it
    HERE — before the cursor is minted — and the token the caller gets back
    always resumes at the row after the last one it was shown.
    """
    try:
        return paginate_view(
            items, _binding(kind, identity, sources), cursor=cursor, limit=view.limit
        )
    except PageCursorError as exc:
        raise _invalid_cursor(exc) from exc


def _paginate_pair(
    primary: list[Any],
    secondary: list[Any],
    kind: str,
    identity: dict[str, Any],
    cursor: str | None,
    sources: Sequence[Path],
    view: _View,
) -> dict[str, Any]:
    """Page an item's two collections under its single cursor (both offsets ride in it)."""
    try:
        return paginate_pair(
            primary,
            secondary,
            _binding(kind, identity, sources),
            cursor=cursor,
            limit=view.limit,
            secondary_limit=view.coord_limit,
        )
    except PageCursorError as exc:
        raise _invalid_cursor(exc) from exc


#: Per-collection restatement keys: collection name → the ``data`` keys that
#: repeat its total, its returned count, and (where one exists) its truncation
#: flag. Declared beside the handlers that emit them so a collector recomputing
#: those counters cannot drift from what a page actually carries.
COLLECTION_COUNTERS: dict[str, tuple[str, str, str | None]] = {
    "symbols": ("total", "returned", None),
    "members": ("total_members", "returned", None),
    "pins": ("total_pins", "returned", None),
    "coordinates": ("total_coordinates", "returned_coordinates", "coordinates_truncated"),
    "components": ("total", "returned", None),
    "results": ("total", "returned", None),
}


def _page_meta(page: dict[str, Any], primary: str, secondary: str | None = None) -> dict[str, Any]:
    """The paged-collection facts surfaced in each item's ``page`` field.

    ``primary`` and ``secondary`` are the data keys this page carries, named
    rather than positional so the two cannot be swapped into each other's
    counters. The top-level counters describe BOTH: a ``.asc`` net pages pins and
    wire-vertex coordinates under one cursor, and pins-only counters read
    ``returned == total`` while hundreds of coordinates are still unfetched.
    Summed instead, ``truncated`` implies ``returned < total`` on every page, and
    ``collections`` says which collection is the one that continues. It is
    omitted for an item that pages exactly one collection: with nothing to
    disambiguate it restates the three counters directly above it.
    """
    collections: dict[str, dict[str, Any]] = {
        primary: {
            "total": page["total"],
            "returned": page["returned"],
            "truncated": page["primary_truncated"],
        }
    }
    if secondary is not None:
        collections[secondary] = {
            "total": page["secondary_total"],
            "returned": page["secondary_returned"],
            "truncated": page["secondary_truncated"],
        }
    meta: dict[str, Any] = {
        "total": sum(c["total"] for c in collections.values()),
        "returned": sum(c["returned"] for c in collections.values()),
        "truncated": page["truncated"],
    }
    if secondary is not None:
        meta["collections"] = collections
    return meta


# ---------------------------------------------------------------------------
# capabilities
# ---------------------------------------------------------------------------


def _gated_tool_report(name: str, state: SessionState) -> dict[str, Any]:
    """Whether this session serves a gated tool, and the config line that
    decides it — read off the registration's own gate."""
    from ltspice_mcp.config import config_key

    gate = registry.gate_of(name)
    return {
        "enabled": name in state.tool_dispatch,
        "config_key": config_key(gate) if gate is not None else None,
        "restart_required": True,
    }


def _python_runtime_facts() -> dict[str, Any]:
    """The interpreter this engine runs in, and whether it will still exist.

    An agent that wants the Python API (``from ltspice_mcp.api import
    Api``) must pick an interpreter that has the package — this one. The
    install kind is the durability fact: a uvx cache environment is rebuilt
    per invocation and may vanish, while pipx/venv/system interpreters are
    stable paths worth writing into a script.
    """
    import ltspice_mcp

    executable = sys.executable
    normalized = executable.replace("\\", "/")
    if "/pipx/venvs/" in normalized:
        kind = "pipx"
    elif "/.cache/uv/" in normalized or "/uv/cache/" in normalized or "/Caches/uv/" in normalized:
        kind = "uvx-cache"
    elif sys.prefix != sys.base_prefix:
        kind = "venv"
    else:
        kind = "system"
    return {
        "executable": executable,
        "install_kind": kind,
        "ephemeral": kind == "uvx-cache",
        "package_location": str(Path(ltspice_mcp.__file__).resolve().parent),
    }


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
    # Every known-but-undetected simulator appears with the exact keys that
    # would turn it on — the config self-diagnosis surface. Detection runs at
    # startup, so a fix always ends in a server restart; the remediation says
    # so rather than leaving the agent to loop on the same absence.
    for name in SIMULATORS:
        if name not in simulators:
            simulators[name] = {
                "available": False,
                "remediation": simulator_remediation(name, state.config),
            }

    return {
        "config_path": str(state.config.config_path),
        "python": _python_runtime_facts(),
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
        # What went wrong at startup, verbatim: a configured simulator path
        # that does not exist, a requested engine that fell back to another,
        # a WSL auto-detection. Otherwise these live only in the server's own
        # stderr log, which no client reads — so a session running degraded
        # looks identical to a healthy one from the outside.
        "diagnostics": list(state.diagnostics),
        "ngbehavior": (current_ngbehavior() if "ngspice" in state.available_simulators else None),
        "persist_jobs": state.config.persist_jobs,
        "allowed_paths": [str(p) for p in state.config.allowed_paths],
        # One surface, and no setting selects it; the key stays because a
        # client reads it to know which one it is talking to.
        "tool_profile": "consolidated",
        # The same six ops as a library, for loops over runs and numpy on
        # samples. The interpreter that has the package is "python" above;
        # this names the import, the session on this working directory, and
        # where an op's arguments are read before they are guessed.
        "python_api": {
            "import": "from ltspice_mcp.api import Api",
            "open": f"Api(working_dir={str(state.working_dir)!r})",
            "reference": (
                "api.reference('run_experiments') lists an op's arguments; "
                "help(Api.<op>) and inspect.signature(Api.<op>) answer too"
            ),
            # The same engine behind a tool call, when the operator turned it
            # on; the key and the restart are what an agent relays to them.
            "run_code": _gated_tool_report("run_code", state),
        },
        # Which of the two tool listings this session was served. The guide
        # tells a caller to reach for inspect(kind="reference") whenever the
        # listing is compact, because the per-argument descriptions are then
        # not on the wire — and this is the only place that fact is readable.
        "tool_listing": state.config.tool_listing,
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


async def _do_symbols(q: SymbolsQuery, state: SessionState, view: _View) -> dict[str, Any]:
    asc_dir: Path | None = None
    if q.path is not None:
        asc_dir = _resolve_path(q, q.path, state).parent

    precedence = _symbol_precedence(asc_dir, state)
    prec_report, names = await asyncio.to_thread(_symbols_payload, precedence, q.filter)

    # No sources: the rows are directory listings, not the content of named
    # files, so there is nothing here a stamp could bind (see _CURSOR_DESCRIPTION).
    page = _paginate(
        names,
        "symbols",
        {"path": str(asc_dir) if asc_dir else None, "filter": q.filter},
        q.cursor,
        (),
        view,
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
        "page": _page_meta(page, "symbols"),
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


async def _asc_digest(path: Path) -> str:
    """The sheet's SHA-256 — the edit token ``edit_schematic`` takes as
    ``expected_sha256``.

    Read tools are where a caller gets it alongside the content it describes;
    ``edit_schematic``'s own refusals also report the target's current digest,
    so a caller who edits without reading first still recovers in one retry.
    Taken BEFORE the rows are read, so the digest can never be
    newer than the content reported alongside it — a token from the future
    would let an edit made against stale rows commit, while a stale token only
    conflicts, which is the safe direction.
    """
    return await asyncio.to_thread(sha256_file, path)


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


def _lex_warnings(lexed: LexResult) -> list[str]:
    """The lexer's own notes about what it had to guess, for a payload's
    ``warnings``.

    An unclosed ``.SUBCKT``, an ``.ENDS`` matching nothing, a continuation with
    no card to continue: each means the cards this answer was read from say
    something other than the file does (an unclosed subcircuit swallows every
    card after it into its scope). The lexer assigns no severity, so these are
    relayed as written. Reading only ``.cards`` dropped them and made the
    answer look authoritative.
    """
    return [f"netlist lexer: {note}" for note in lexed.warnings]


def _net_netlist_payload(text: str, at: str | list[int]) -> dict[str, Any]:
    """Card-membership for a node in a netlist — no geometry keys (by contract)."""
    lexed = lex(text)
    cards = lexed.cards
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
    payload: dict[str, Any] = {
        "node": node,
        "members": members,
        "unparseable_cards": unparseable,
    }
    if lexed.warnings:
        payload["warnings"] = _lex_warnings(lexed)
    return payload


async def _do_net(q: NetQuery, state: SessionState, view: _View) -> dict[str, Any]:
    path = _resolve_path(q, q.path, state)
    identity = {"path": str(path), "at": q.at}

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
        page = _paginate(members, "net", identity, q.cursor, [path], view)
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
            "page": _page_meta(page, "members"),
        }

    # .asc: geometric trace via trace_net internals. The cached AscEditor is
    # touched only on the event loop, so this runs inline (never offloaded).
    digest = await _asc_digest(path)
    trace_input = _trace_input_for(q.path, q.at)
    trace = await handle_trace_net(trace_input, state)
    tdata = trace.structured_content or {}
    pins = list(tdata.get("pins", []))
    coords = list(tdata.get("coordinates", []))

    # Pins and wire vertices are two independently long collections of one net,
    # and the item carries one cursor — so both offsets ride in it and both
    # advance. A coordinate window that restarted every page would re-serve the
    # same vertices forever while claiming more existed.
    page = _paginate_pair(pins, coords, "net.asc", identity, q.cursor, [path], view)
    data: dict[str, Any] = {
        "source": "schematic",
        "sha256": digest,
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
    return {
        "data": data,
        "next_cursor": page["next_cursor"],
        "page": _page_meta(page, "pins", "coordinates"),
    }


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
) -> tuple[list[dict[str, Any]], list[str]]:
    """The component rows, plus the lexer's notes about how the deck read."""
    from ltspice_mcp.lib.spice_lex_views import body_has_stray_kv_remnant

    lexed = lex(text)
    cards = lexed.cards
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
    return rows, _lex_warnings(lexed)


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


async def _do_components(q: ComponentsQuery, state: SessionState, view: _View) -> dict[str, Any]:
    _check_prefix(q.prefix)
    path = _resolve_path(q, q.path, state)
    # The answer rung revokes detail='full' — the one payload-growing opt-in
    # inspect has. The cursor binds the detail it actually rendered, so a page
    # taken under a budget cannot resume as an unbudgeted one at the same offset.
    detail = "list" if view.lean else q.detail
    identity = {"path": str(path), "prefix": q.prefix, "detail": detail}
    digest: str | None = None
    lex_notes: list[str] = []

    if _route_circuit_kind(path, "components") == "asc":
        digest = await _asc_digest(path)
        # Cached editor + component reads stay on the event loop.
        editor = get_asc_editor(path, state)
        try:
            refs = sorted(editor.get_components(q.prefix) if q.prefix else editor.get_components())
        except Exception as exc:
            raise _QueryError("parse_error", f"failed to list components: {exc}") from exc
        page = _paginate(refs, "components", identity, q.cursor, [path], view)
        rows = _components_asc_page(editor, page["items"], detail)
    else:
        try:
            text = await asyncio.to_thread(read_spice_text, path)
        except OSError as exc:
            raise _QueryError("read_error", str(exc)) from exc
        try:
            all_rows, lex_notes = await asyncio.to_thread(
                _components_netlist_payload, text, q.prefix, detail
            )
        except SpiceLexError as exc:
            raise _QueryError("parse_error", str(exc)) from exc
        page = _paginate(all_rows, "components", identity, q.cursor, [path], view)
        rows = page["items"]

    data: dict[str, Any] = {
        "components": rows,
        "detail": detail,
        "prefix": q.prefix,
        "total": page["total"],
        "returned": page["returned"],
    }
    if digest is not None:
        data["sha256"] = digest
    if lex_notes:
        # Every page of this deck carries them: they describe the read the rows
        # came out of, and a caller who pages past the first one is reading the
        # same suspect scoping.
        data["warnings"] = lex_notes
    return {
        "data": data,
        "next_cursor": page["next_cursor"],
        "page": _page_meta(page, "components"),
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
            score = part_aware_score(query_lower, entry.name_lower)
            if score < cutoff or entry.name_lower in seen:
                continue
            seen.add(entry.name_lower)
            row = _model_entry(entry)
            row["score"] = round(score, 3)
            scored.append((score, row))
    scored.sort(key=lambda pair: (-pair[0], pair[1]["name"].lower()))
    return [row for _, row in scored]


async def _do_model(q: ModelQuery, state: SessionState, view: _View) -> dict[str, Any]:
    # The library files these rows were read from, so the cursor binds their
    # revision: an edited library must reject a stale token, not page into the
    # re-parsed list at the old offset.
    sources: list[Path] = []
    if q.mode == "enumerate":
        sources = [_resolve_path(q, lib, state) for lib in (q.libs or [])]
        try:
            rows = await asyncio.to_thread(_enumerate_libs, sources)
        except OSError as exc:
            raise _QueryError("read_error", str(exc)) from exc
        identity: dict[str, Any] = {"mode": "enumerate", "libs": [str(p) for p in sources]}
    else:
        assert q.query is not None  # guaranteed by the model validator
        if q.libs:
            sources = [_resolve_path(q, lib, state) for lib in q.libs]
            try:
                rows = await asyncio.to_thread(_search_libs, sources, q.query)
            except OSError as exc:
                raise _QueryError("read_error", str(exc)) from exc
        else:
            # No libs given: fall back to the session's loaded libraries (loop-owned
            # mutable state — read inline, never offloaded). Those rows come from
            # the session's own load/unload state and the stock library tree, not
            # from files this query names, so there is nothing to stamp.
            try:
                rows = state.libraries.find_similar_models(
                    q.query, exact=False, limit=10_000, cutoff=0.6
                )
            except Exception as exc:
                raise _QueryError("search_error", str(exc)) from exc
        identity = {"mode": "search", "query": q.query, "libs": q.libs}

    page = _paginate(rows, "model", identity, q.cursor, sources, view)
    return {
        "data": {
            "mode": q.mode,
            "query": q.query,
            "results": page["items"],
            "total": page["total"],
            "returned": page["returned"],
        },
        "next_cursor": page["next_cursor"],
        "page": _page_meta(page, "results"),
    }


# ---------------------------------------------------------------------------
# reference
# ---------------------------------------------------------------------------

_REFERENCE_CONTENTS_HINT = (
    "Ask again with a 'query' in plain words — inspect(kind='reference', "
    "query='phase margin') — for a branch's fields, types, defaults and units."
)


def _do_reference(q: ReferenceQuery, view: _View, served: frozenset[str]) -> dict[str, Any]:
    """Search the tools' branch vocabulary, or list it when no query is given.

    Nothing here reads a file, so there is no path to resolve, no cursor to
    bind and nothing to offload: the index is derived from the input models
    once per process and searched in memory. ``served`` is the one thing read
    off the session — the tools it dispatches — so a tool the operator did not
    turn on is not in the table a session that would refuse it hands out.

    ``view`` is how a caller's ``budget`` reaches this kind. The shrink rung
    lowers the page size, and taking the smaller of it and the caller's own
    ``limit`` is what lets a tight budget return fewer branches instead of
    reporting that it could not be met. There is no cursor to leave pointing
    past rows nobody saw, because the caller's own ``limit`` is the handle —
    and when the budget rather than that limit is what cut the page (the view
    says whether the budget acted, and the page says whether it cut), the hint
    names the budget instead of a knob the caller already set.
    """
    if q.query is None:
        return {
            "data": {
                "contents": table_of_contents(served),
                "total_branches": sum(1 for entry in build_index() if entry.tool in served),
                "hint": _REFERENCE_CONTENTS_HINT,
            }
        }

    page_limit = min(q.limit, view.limit)
    matches, total = search_branches(q.query, limit=page_limit, tools=served)
    data: dict[str, Any] = {
        "query": q.query,
        "matches": [entry.as_dict() for entry in matches],
        "total_matches": total,
        "returned": len(matches),
    }
    if not matches:
        data["hint"] = (
            f"Nothing matched {q.query!r}. Call inspect(kind='reference') with no "
            "query for the whole vocabulary, or read spice://guide."
        )
    elif total > len(matches):
        # Name the lever that is actually free. Under a response budget the
        # page was cut below what the caller asked for, so 'limit' is not the
        # handle; at the cap, raising 'limit' is not possible at all.
        if view.shrunk and page_limit < q.limit:
            lever = (
                f"The response 'budget' cut this page to {page_limit}; raise it, "
                "or narrow the query."
            )
        elif q.limit < REFERENCE_LIMIT_CAP:
            lever = f"Raise 'limit' (up to {REFERENCE_LIMIT_CAP}) or narrow the query."
        else:
            lever = f"'limit' is already at its cap of {REFERENCE_LIMIT_CAP}; narrow the query."
        data["hint"] = f"{total} branches matched; the {len(matches)} closest are shown. {lever}"
    return {"data": data}


# ---------------------------------------------------------------------------
# Shared per-item helpers + dispatch
# ---------------------------------------------------------------------------


def _resolve_path(q: Any, user_path: str, state: SessionState) -> Path:
    """safe_path with the denial mapped to a per-item ``path_denied`` failure."""
    try:
        return safe_path(user_path, state)
    except PathSecurityError as exc:
        raise _QueryError("path_denied", str(exc)) from exc


async def _dispatch(query: Query, state: SessionState, view: _View) -> dict[str, Any]:
    if isinstance(query, CapabilitiesQuery):
        return {"data": _do_capabilities(state)}
    if isinstance(query, SymbolsQuery):
        return await _do_symbols(query, state, view)
    if isinstance(query, SymbolQuery):
        return await _do_symbol(query, state)
    if isinstance(query, NetQuery):
        return await _do_net(query, state, view)
    if isinstance(query, ComponentsQuery):
        return await _do_components(query, state, view)
    if isinstance(query, ReferenceQuery):
        return _do_reference(query, view, frozenset(state.tool_dispatch))
    # Exhaustive over the sealed union: ModelQuery is the only remaining member.
    return await _do_model(query, state, view)


_ERROR_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "code": {"type": "string"},
        "message": {"type": "string"},
        "supported": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["code", "message"],
}

#: The ``reference`` kind's payload, declared inside the otherwise-open per-item
#: ``data``. ``matches`` and ``contents`` are names no other kind returns, so
#: declaring them here says what a reference answer looks like without saying
#: anything about a net trace or a component list.
_REFERENCE_FIELD_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "name": {"type": "string", "description": "Argument name; dotted inside a nested object."},
        "type": {
            "type": "string",
            "description": "The type, with enum members written out and any bounds appended.",
        },
        "required": {"type": "boolean", "description": "Present only when the field is required."},
        "default": {
            "type": "string",
            "description": "The default as JSON, or a word for what a factory produces.",
        },
        "description": {
            "type": "string",
            "description": "Units and conventions, where it has any.",
        },
    },
    "required": ["name", "type"],
}

_REFERENCE_BRANCH_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "tool": {"type": "string"},
        "family": {
            "type": "string",
            "description": "What this branch is to its tool: recipe, op, check, action, ...",
        },
        "name": {"type": "string", "description": "The discriminant value to pass."},
        "summary": {"type": "string"},
        "call": {"type": "string", "description": "How the call carrying this branch is written."},
        "fields": {"type": "array", "items": _REFERENCE_FIELD_SCHEMA},
    },
    "required": ["tool", "family", "name", "summary"],
}

_REFERENCE_DATA_PROPERTIES: dict[str, Any] = {
    # Echoed by the 'reference' and 'model' kinds alike, and null on a 'model'
    # enumerate, which asks for everything rather than for a match.
    "query": {"type": ["string", "null"]},
    "matches": {
        "type": "array",
        "items": _REFERENCE_BRANCH_SCHEMA,
        "description": "Best matches first, each with its full field table.",
    },
    "total_matches": {"type": "integer"},
    "contents": {
        "type": "array",
        "description": "The table of contents, returned when no query was given.",
        "items": {
            "type": "object",
            "properties": {
                "tool": {"type": "string"},
                "family": {"type": "string"},
                "branches": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {"name": {"type": "string"}, "summary": {"type": "string"}},
                        "required": ["name", "summary"],
                    },
                },
            },
            "required": ["tool", "family", "branches"],
        },
    },
    "total_branches": {"type": "integer"},
    "hint": {
        "type": "string",
        "description": "What to ask next: how to narrow, widen, or reach the full guide.",
    },
}

_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        # Call-level outcome over the per-item batch: complete when every query
        # succeeded, partial when any query failed. The shared envelope's other
        # values are absent because inspect cannot reach them: per-item
        # isolation turns every query fault into that item's error, and a
        # call-level fault raises — server.call_tool's catch-all then answers
        # with isError and no structuredContent, so it is never carried by
        # this envelope.
        "outcome": outcome_schema("complete", "partial"),
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
                    # the module docstring and stays open here by design. The
                    # reference lookup's two collections are named because
                    # nothing else on this tool returns them, so declaring them
                    # constrains that kind without constraining any other.
                    "data": {"type": "object", "properties": _REFERENCE_DATA_PROPERTIES},
                    "next_cursor": {"type": ["string", "null"]},
                    "page": {
                        "type": "object",
                        "description": (
                            "Paging counters for this item, summed over every collection "
                            "it pages (a .asc net pages pins and coordinates under one "
                            "cursor)."
                        ),
                        "properties": {
                            "total": {
                                "type": "integer",
                                "description": "Rows this item has in all, across all pages.",
                            },
                            "returned": {
                                "type": "integer",
                                "description": "Rows in this page.",
                            },
                            "truncated": {
                                "type": "boolean",
                                "description": (
                                    "More rows remain somewhere in this item; always equals "
                                    "next_cursor being non-null. Stop on next_cursor, not on "
                                    "returned == total."
                                ),
                            },
                            "collections": {
                                "type": "object",
                                "description": (
                                    "Per-collection counters, keyed by the data key they page "
                                    "— which collection still has rows. Present only for an "
                                    "item paging more than one (a .asc net's pins and "
                                    "coordinates); otherwise the counters above describe it."
                                ),
                                "additionalProperties": {
                                    "type": "object",
                                    "properties": {
                                        "total": {"type": "integer"},
                                        "returned": {"type": "integer"},
                                        "truncated": {"type": "boolean"},
                                    },
                                },
                            },
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
        # Emitted only when a caller-set 'budget' degraded the response — the
        # one call-level fact inspect can reach. Per-query faults stay on their
        # own item's 'error' and never surface here.
        "observations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "code": {"type": "string"},
                    "kind": {"type": "string"},
                    "detail": {"type": "string"},
                },
                "required": ["code", "kind", "detail"],
                "additionalProperties": True,
            },
        },
        "hint": HINT_SCHEMA,
    },
    "required": ["outcome", "results", "count"],
}

INSPECT_DESCRIPTION = (
    "Read-only lookups over the server and the circuits it can reach, batched as "
    "independent 'queries'. Kinds: 'capabilities', 'symbols', 'symbol', 'net', "
    "'components', 'model', 'reference' — each with its own arguments, described "
    "on its branch of the query schema. 'reference' searches every tool's "
    "recipes, ops, checks and their fields in plain words ('phase margin'). A "
    "denied path, a stale cursor, an unknown kind, or a malformed query fails "
    "only that item; every other query still returns, and paginated kinds resume "
    "via 'cursor'. It is the only tool on this surface that never writes."
)


@registry.tool(
    name="inspect",
    title="Inspect Circuit",
    description=INSPECT_DESCRIPTION,
    input_model=InspectInput,
    annotations=RO_ANNOTATIONS,
    output_schema=_OUTPUT_SCHEMA,
)
async def handle_inspect(args: InspectInput, state: SessionState) -> types.CallToolResult:
    """Answer a batch of read-only queries with per-item success/failure isolation."""
    budget = resolve_response_budget(args.budget, state)
    if budget.tokens is None:
        results = await _run_queries(args, state, _View())
        return format_response(_summary_text(results), inspect_envelope(results))
    return await _negotiate_inspect(args, state, budget)


async def _run_queries(
    args: InspectInput, state: SessionState, view: _View
) -> list[dict[str, Any]]:
    """Every query in the batch, answered at ``view``, isolated from each other."""
    results: list[dict[str, Any]] = []

    for index, raw in enumerate(args.queries):
        try:
            query = _validate_query(raw)
            outcome = await _dispatch(query, state, view)
        except _QueryError as exc:
            error = {"code": exc.code, "message": exc.message}
            if exc.supported is not None:
                error["supported"] = exc.supported  # type: ignore[assignment]
            results.append(_failure_item(index, raw, error))
            continue
        except ValidationError as exc:
            results.append(
                _failure_item(
                    index,
                    raw,
                    {"code": "invalid_query", "message": compact_validation_error(exc)},
                )
            )
            continue
        except LTSpiceMCPError as exc:
            results.append(_failure_item(index, raw, {"code": "error", "message": str(exc)}))
            continue
        except Exception as exc:
            # Last-resort isolation: an unexpected fault in one query must not
            # sink the batch — the per-item contract is that others still return.
            results.append(
                _failure_item(index, raw, {"code": "internal_error", "message": str(exc)})
            )
            continue

        item: dict[str, Any] = {"index": index, "kind": query.kind, "ok": True}
        item.update(outcome)
        results.append(item)
    return results


def inspect_envelope(results: list[dict[str, Any]]) -> dict[str, Any]:
    """The shared envelope over an answered batch."""
    error_count = sum(1 for item in results if not item["ok"])
    data: dict[str, Any] = {
        # Per-item failures isolate to their result and never fail the call, so
        # the batch is "partial" when any query failed and "complete" otherwise.
        "outcome": outcome_of(error_count),
        "results": results,
        "count": len(results),
        "ok_count": len(results) - error_count,
        "error_count": error_count,
    }
    if error_count:
        data["hint"] = (
            f"{error_count} of {len(results)} queries failed; see each result's "
            "'error.code'. Other queries returned normally."
        )
    return data


# Rung 0's allowlist, declared as data rather than spelled inside the ``if``
# that applies it: a rung that exempts content is the one place a checker can
# silently lose coverage, so it has to be a list a test can read. Both keys are
# optional on an item schema — pinned by tests/test_response_budget.py. They go
# only from an item with nothing left to fetch, for which the paging block
# restates counts the rows carry and a null cursor that leads nowhere.
_TRIM_REMOVE_EXHAUSTED: tuple[str, ...] = ("page", "next_cursor")


def _degrade_inspect(data: dict[str, Any], rung: response_budget.Rung) -> None:
    """Apply the budget ladder's in-place trim rung to an inspect envelope.

    The answer rung and the shrink rung are not here: both change what a query
    reads, so they are inputs to the query pass (``_View``) rather than edits to
    its answer — a page shrunk after the fact would leave its cursor pointing
    past rows the caller never saw. Nothing below touches an item's ``error``.

    Idempotent, so the ladder may re-apply it to an envelope it already degraded
    on the way down.
    """
    if rung.trim:
        for item in data["results"]:
            page = item.get("page")
            if item.get("next_cursor") is None and (
                not isinstance(page, dict) or not page.get("truncated")
            ):
                for key in _TRIM_REMOVE_EXHAUSTED:
                    item.pop(key, None)


#: This tool's budget epilogue. The hint mirror is why it is a value: the note's
#: detail is written twice under a hint key, and the reserve has to know that.
#: Structured-aware clients render only structuredContent, and 'hint' is where
#: this tool puts guidance, so the mirror is not optional.
_BUDGET_NOTES = response_budget.Notes(
    cut="presentation was reduced; no query was dropped and no error was hidden.",
    route=(
        "Ask again with a larger 'budget' for the full presentation, or page on "
        "with each item's next_cursor."
    ),
    hint_key="hint",
)


def _paged_rows(data: dict[str, Any]) -> list[Any]:
    """Every row the answered batch is currently showing, across all items."""
    rows: list[Any] = []
    for item in data["results"]:
        payload = item.get("data")
        if not isinstance(payload, dict):
            continue
        for value in payload.values():
            if isinstance(value, list):
                rows.extend(value)
    return rows


async def _negotiate_inspect(
    args: InspectInput, state: SessionState, budget: ResponseBudget
) -> types.CallToolResult:
    """Answer the batch at the mildest ladder rung that fits ``budget``."""
    # One pass per distinct view, not one per rung: the trim rung re-renders an
    # answered batch, only the answer and shrink rungs re-ask it.
    passes: dict[_View, list[dict[str, Any]]] = {}
    rendered: dict[str, Any] = {"results": []}
    # The view the standing envelope was built from. A rung that does not change
    # the view is the previous rung degraded a step further, so it edits that
    # envelope rather than copying the pass and rebuilding over it.
    built_from: _View | None = None

    async def render(rung: response_budget.Rung) -> dict[str, Any]:
        nonlocal rendered, built_from
        view = _View(lean=rung.answer_channel)
        if rung.shrink:
            measure = response_budget.RowMeasure.of(_paged_rows(rendered))
            view = _View(
                limit=measure.fit_limit(_PAGE_SIZE, rung),
                coord_limit=measure.fit_limit(_COORD_PAGE_SIZE, rung),
                lean=rung.answer_channel,
                shrunk=rung.shrink,
            )
        if built_from != view:
            if view not in passes:
                passes[view] = await _run_queries(args, state, view)
            # A copy per envelope, because a pass is cached and reused across
            # rungs while the envelope built from it is degraded in place.
            rendered = inspect_envelope(copy.deepcopy(passes[view]))
            built_from = view
        _degrade_inspect(rendered, rung)
        return rendered

    assert budget.tokens is not None  # the undegraded path never reaches here
    result = await response_budget.negotiate(
        budget.tokens, render, _BUDGET_NOTES, max_rung=budget.max_rung
    )
    response_budget.attach_notes(result, _BUDGET_NOTES)
    data = result.data
    return format_response(_summary_text(data["results"]), data)


def _failure_item(index: int, raw: Any, error: dict[str, Any]) -> dict[str, Any]:
    return {"index": index, "kind": _kind_of(raw), "ok": False, "error": error}


def _kind_of(raw: Any) -> str | None:
    kind = raw.get("kind") if isinstance(raw, dict) else getattr(raw, "kind", None)
    return kind if isinstance(kind, str) else None


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
