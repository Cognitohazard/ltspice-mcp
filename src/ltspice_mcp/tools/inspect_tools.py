"""inspect — the consolidated read-only UNDERSTAND surface.

One tool answers a batch of independent read-only ``queries`` about the server
and the circuits it can reach. Each query names one supported kind:

* ``capabilities`` — detected simulators + dialects and whether a run can
  select each (``selectable``, with the ``refusal`` when this host cannot run
  that family), exporter presence, job persistence, allowed roots, the active
  profile and which of the two tool listings this session was served, the
  configured limits, the linter version, and ``diagnostics``: the startup
  notes (bad configured simulator path, a requested engine that fell back, WSL
  auto-detection) that say whether this server started degraded. Pulled from
  ``state``/``config``/``lint_rules``; nothing is probed.
* ``symbols`` — the legal ``.asy`` symbol names and the resolution-order
  precedence they resolve through. A ``path`` adds that schematic's own
  directory to the front of the reported precedence.
* ``symbol`` — one symbol's pin positions per rotation (``R0``…``M270``),
  bounding box, and origin (the ``symbol_info`` geometry internals), with its
  ``SymbolType`` and every ``SYMATTR`` it carries (``Prefix``, ``SpiceModel``,
  ``Value``, ``SpiceLine``, ``ModelFile``...): the model and parameters an
  instance of it is netlisted with.
* ``net`` — everything on a net. On a ``.asc`` this is a geometric trace
  (``trace_net`` internals: pins, wire vertices, labels, shorts); an ``[x, y]``
  on a wire's interior traces that wire and reports it as ``snapped_to_wire``. On a
  ``.cir``/``.net``/``.sp`` netlist it is card-membership — which element cards
  reference the node — carrying **no geometry** at all.
* ``components`` — the component list (``detail:"list"``) or full per-component
  detail (``detail:"full"``) of any circuit file.
* ``hierarchy`` — bounded netlist instance expansion, scoped ports, numeric
  facts and backend addresses from captured active dependencies. See
  ``lib/hierarchy.py`` for supported grammar and explicit refusal boundaries.
* ``results`` — RAW plot inventory, selected signal descriptors and axisless
  quantities, or detached log measurements/native print rows. Pages bind the
  captured snapshot; RAW reads accept explicit plot/dialect selection.

On a netlist, ``net`` and ``components`` add ``warnings`` when the lexer had to
guess about the deck (an unclosed ``.SUBCKT``, an ``.ENDS`` matching nothing, a
stray continuation) — the answer was read from cards that mean something other
than the file says, and the key is absent when it read cleanly.

Both circuit kinds report the sheet's ``sha256`` when the target is a ``.asc``
— the token ``edit_schematic`` requires as ``expected_sha256``. Reading it here
is what lets a first edit commit in one call; an edit attempted without it is
refused with the current digest attached, so that path costs one retry rather
than a hunt.
* ``model`` — model/subcircuit lookup: ``search`` fuzzy-matches a ``query``
  in the given ``libs``, or in the detected simulators' own model libraries
  when ``libs`` is omitted; ``enumerate`` lists every model defined in the
  given ``libs``, narrowed to the names containing ``query`` when one is
  given. A ``libs`` file may sit inside the sandbox or inside one of those
  simulator libraries, so every ``source_path`` a search returns reads back.
* ``reference`` — the tools' own branch vocabulary (``tools/reference_index.py``):
  a plain-words ``query`` returns the closest analysis recipes, schematic ops,
  variation kinds, checks and job actions with their full field tables, and no
  ``query`` returns the table of contents. It is the answer to "which recipe
  gives me phase margin" and to "what does this branch take" on the ``compact``
  tool listing, where per-argument descriptions are not on the wire at all. It
  reads no file and touches no session state.
* ``guide`` — the packaged guide (``lib/guide.py``): with no ``section``, the
  core a session reads first and the index of topic sections and task
  playbooks; with one, that section. It records on the session that the guide
  was read, which retires the one read-the-guide reminder.
* ``simulator_docs`` — the reference documents the simulator's vendor installs
  with it (``lib/simulator_docs.py``): for LTspice 26.1 and later, its own
  account of the keyboard shortcuts, menus, schematic file format, ``.MEAS``
  and waveform viewer. With no ``name`` it lists them; with one it returns
  that document in sections, paged at its headings. They are read from the
  install, never packaged, and are the authority on the program itself where
  the guide is about this server.
* ``open_in_ltspice`` — the documents open in the LTspice windows on this
  machine, and which is in front in its window: where a request about "this
  circuit" starts when no path was given. A sheet inside the sandbox comes
  with its ``sha256`` and with whether the window's copy differs from the file
  (unsaved changes, or a file that changed after it was opened), which is what
  ``edit_schematic`` refuses on. A document outside the sandbox is listed and
  not read. It is asked of LTspice itself, through the bridge it ships from
  26.1; where that cannot be reached the query fails, saying why, since
  "nothing is open" would be a different answer.

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
import hashlib
import sys
from bisect import bisect_right
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Annotated, Any, Literal, TypeAlias, overload

from mcp import types
from numpy.typing import NDArray
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
from ltspice_mcp.lib import (
    NETLIST_SUFFIX_TEXT,
    NETLIST_SUFFIXES,
    guide,
    response_budget,
    services,
    simulator_docs,
)
from ltspice_mcp.lib.cache import file_stamp
from ltspice_mcp.lib.cursor_codec import canonical_hash
from ltspice_mcp.lib.deck_staging import sha256_file
from ltspice_mcp.lib.decoded_raw import DecodedRaw, TraceDescriptor
from ltspice_mcp.lib.encoding import read_spice_text
from ltspice_mcp.lib.hierarchy import SemanticProfile, load_hierarchy
from ltspice_mcp.lib.library_manager import (
    LibraryManager,
    model_row,
    parse_library_file_cached,
    rank_models,
)
from ltspice_mcp.lib.lint_rules import linter_version
from ltspice_mcp.lib.ltspice_bridge import BridgeError
from ltspice_mcp.lib.ltspice_window import OpenDesign, WindowsUnavailable
from ltspice_mcp.lib.model_fields import literal_values, model_union
from ltspice_mcp.lib.montecarlo import matches_prefix
from ltspice_mcp.lib.pin_legend import PageCursorError, paginate_pair, paginate_view
from ltspice_mcp.lib.raster import RasterSupport, raster_support
from ltspice_mcp.lib.schematic_ops import (
    get_asc_editor,
    label_folded_nets,
    named_labels,
    net_members,
    net_partition,
    netlist_card_value,
    placed_geometry,
    require_asc,
    resolve_pin,
    same_instance_dropped_segments,
    segment_json,
    segment_text,
    wire_segments_of,
    wires_of_one_net,
)
from ltspice_mcp.lib.schematic_scene import SymbolResolver, default_stock_paths
from ltspice_mcp.lib.simulator import (
    SIMULATORS,
    current_ngbehavior,
    dialect_for_simulator_name,
    family_refusal,
    simulator_family,
    simulator_library_roots,
    simulator_remediation,
)
from ltspice_mcp.lib.simulator_build import (
    SimulatorExecutable,
    executable_identity,
)
from ltspice_mcp.lib.spice_lex import LexResult, SpiceLexError, lex
from ltspice_mcp.lib.spice_lex_views import InstanceLine, instances_by_ref
from ltspice_mcp.lib.symbol_geometry import compute_placed_geometry, parse_asy_file
from ltspice_mcp.state import SessionState
from ltspice_mcp.tools._base import (
    FORMAT_DESCRIPTION,
    HINT_SCHEMA,
    LTSPICE_DIFFERENCE_ENTRIES,
    LTSPICE_WINDOW_PROPERTIES,
    RO_ANNOTATIONS,
    WARNINGS_SCHEMA,
    OptionalRawSelectionFields,
    ResponseBudget,
    StrictModel,
    ToolInput,
    declare_output_schema,
    format_response,
    outcome_of,
    outcome_schema,
    registry,
    resolve_response_budget,
    safe_library_path,
    safe_path,
    sandboxed,
    symbol_resolver_for,
    window_difference,
)
from ltspice_mcp.tools.reference_index import build_index, search_branches, table_of_contents


class TraceNetInput(ToolInput):
    path: str = Field(description="Path to an .asc schematic")
    pin: str | None = Field(
        default=None,
        description=(
            "Pin or net reference to start from: 'Ref.Pin', the pin by name or "
            "1-based SpiceOrder (e.g. 'M1.D', 'X1.2'), 'net:NAME' (e.g. "
            "'net:VDD'), or omit and pass x/y."
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
            "snapped_to_wire": {
                "type": "object",
                "description": (
                    "Present when the start point lies on a wire's interior: the "
                    "wire whose net was traced."
                ),
                "properties": {
                    "from": {
                        "type": "object",
                        "properties": {"x": {"type": "integer"}, "y": {"type": "integer"}},
                    },
                    "to": {
                        "type": "object",
                        "properties": {"x": {"type": "integer"}, "y": {"type": "integer"}},
                    },
                },
            },
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

    # The physical partition connects by wire only; LTspice also makes FLAGs
    # with the same NAME electrically common, so trace_net answers "what's on
    # net X" on label-wired schematics, not just wire-routed ones.
    net_of = label_folded_nets(part)

    trace_from = start
    snapped: tuple[int, int, int, int] | None = None
    if start not in part.members.get(part.root(start), set()):
        # No pin, label or wire end sits here. A point on a wire's interior is
        # still on that wire's net, since LTspice joins anything placed there,
        # so trace from the wire's end and say so.
        through = wires_of_one_net(
            start, wire_segments_of(editor), net_of, "Trace from an end of the wire you mean."
        )
        if not through:
            raise NetlistError(
                f"Nothing found at ({x},{y}): no component pin, net label or wire "
                "touches it. Use inspect(kind='components') to inspect the layout."
            )
        snapped = through[0]
        trace_from = snapped[:2]

    member_coords = net_members(part, net_of, net_of(trace_from))

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
    if snapped is not None:
        data["snapped_to_wire"] = segment_json(snapped)
    if warnings:
        data["warnings"] = warnings

    net_name = ", ".join(sorted(labels)) if labels else "<unnamed>"
    lines = [f"Net at ({x},{y}): {net_name}"]
    if snapped is not None:
        lines.append(
            f"  ({x},{y}) lies on the wire {segment_text(snapped)}; traced that wire's net."
        )
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


#: The capabilities report's top-level keys, which are what ``fields`` selects.
#: ``tests/test_inspect_tools.py`` holds it equal to what ``_do_capabilities``
#: returns, so a key added to one and not the other fails there.
CapabilityField: TypeAlias = Literal[
    "config_path",
    "python",
    "simulators",
    "named_executables",
    "default_simulator",
    "exporter_available",
    "render",
    "open_window_sync",
    "dialects",
    "diagnostics",
    "ngbehavior",
    "persist_jobs",
    "allowed_paths",
    "tool_profile",
    "python_api",
    "tool_listing",
    "limits",
    "linter_version",
]


class CapabilitiesQuery(StrictModel):
    """What this server can do: detected simulators with their executables, last
    reported builds and raw dialects, and which a run can select; the named
    executables, whether the .asc exporter is available, job persistence,
    allowed roots, the configured
    limits, and the linter version."""

    kind: Literal["capabilities"]
    fields: list[CapabilityField] | None = Field(
        default=None,
        min_length=1,
        description=(
            "Return only these keys, e.g. ['allowed_paths', 'config_path'] after "
            "a config edit. Omit for the whole report."
        ),
    )


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


#: The circuit file a net or components query reads.
_CIRCUIT_PATH_DESCRIPTION = f"The .asc schematic, or {NETLIST_SUFFIX_TEXT} netlist, to read."


class NetQuery(StrictModel):
    """Everything on one net. On a .asc this is a geometric trace — pins, wire
    vertices, net labels, and whether two labels short the net. On a netlist it
    is card membership, with no geometry."""

    kind: Literal["net"]
    path: str = Field(description=_CIRCUIT_PATH_DESCRIPTION)
    at: str | list[int] = Field(
        description=(
            "Where the net is: 'REF.PIN', PIN a pin name or 1-based SpiceOrder "
            "(e.g. 'M1.D', 'X1.2'), 'net:NAME' or a bare net name, or [x, y]; "
            "on a netlist, which has no geometry, PIN is a terminal number and "
            "a coordinate is rejected."
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


#: One filter rule for both kinds that list references. On ``hierarchy`` it
#: reads each instance's own reference, the last segment of its path.
_PREFIX_DESCRIPTION = (
    "Keep only references starting with this, case-insensitively: 'M' for "
    "every MOSFET, 'LX' for LX1, LX2…. Plain text, not a glob."
)


class ComponentsQuery(StrictModel):
    """The components of a .asc schematic or a netlist."""

    kind: Literal["components"]
    path: str = Field(description=_CIRCUIT_PATH_DESCRIPTION)
    prefix: str | None = Field(default=None, description=_PREFIX_DESCRIPTION)
    detail: Literal["list", "full"] = Field(
        default="list",
        description=(
            "'list' returns reference and value only; 'full' adds nodes, model "
            "and params on a netlist, and symbol, position, rotation, pins and "
            "bounding box on a schematic."
        ),
    )
    cursor: str | None = Field(default=None, description=_CURSOR_DESCRIPTION_FILE)


class HierarchyQuery(StrictModel):
    """Resolve repeated netlist instances, ports, parameters and backend device addresses."""

    kind: Literal["hierarchy"]
    path: str = Field(description=f"Netlist {NETLIST_SUFFIX_TEXT}; export a schematic first.")
    simulator: Literal["ltspice", "ngspice"] = Field(
        description="Offline semantic backend; installation is not required."
    )
    ngbehavior: str | None = Field(
        default=None,
        max_length=32,
        description="ngspice compatibility mode; defaults to the configured effective mode.",
    )
    instance: list[Annotated[str, Field(min_length=1, max_length=256)]] | None = Field(
        default=None,
        min_length=1,
        max_length=33,
        description="Exact reference segments selecting a subtree, matched case-insensitively.",
    )
    prefix: str | None = Field(default=None, description=_PREFIX_DESCRIPTION)
    cursor: str | None = Field(
        default=None,
        description="Resume token bound to captured dependency content, profile and filters.",
    )

    @model_validator(mode="after")
    def _profile(self) -> HierarchyQuery:
        if self.simulator == "ltspice" and self.ngbehavior is not None:
            raise ValueError("ngbehavior is only valid for ngspice")
        return self


class ModelQuery(StrictModel):
    """Find a .model or .subckt definition — by fuzzy name match, or by listing
    everything the given libraries define. Ask it when a run failed on a model
    the deck names, or to get a part's exact spelling before writing it in."""

    kind: Literal["model"]
    mode: Literal["search", "enumerate"] = Field(
        description=(
            "'search' fuzzy-matches 'query' and requires it; 'enumerate' lists "
            "every model in 'libs', or with 'query' those whose name contains it."
        )
    )
    query: str | None = Field(
        default=None,
        description="Part name or fragment; required by 'search', a name filter for 'enumerate'.",
    )
    libs: list[str] | None = Field(
        default=None,
        description=(
            "Library files to read: required by 'enumerate', optional for "
            "'search', which searches the simulator's own libraries when omitted."
        ),
    )
    cursor: str | None = Field(
        default=None,
        description=(
            _CURSOR_DESCRIPTION_FILE + " The files are those named in 'libs', "
            "or the simulator's own when it is omitted."
        ),
    )

    @model_validator(mode="after")
    def _mode_requirements(self) -> ModelQuery:
        if self.mode == "search" and not self.query:
            raise ValueError("model search requires 'query'")
        if self.mode == "enumerate" and not self.libs:
            raise ValueError("model enumerate requires 'libs'")
        return self


class ResultsQuery(OptionalRawSelectionFields):
    """RAW inventory or detached log facts with snapshot-bound pages."""

    kind: Literal["results"]
    view: Literal["plots", "signals", "table", "measurements", "native_tables"] = "plots"
    path: str | None = Field(
        default=None,
        description="RAW path for plots/signals/table, log path for measurements/native_tables; pass this or job_id.",
    )
    job_id: str | None = None
    run_index: int = Field(default=0, strict=True, ge=0)
    case_id: str | None = None
    prefix: str | None = Field(
        default=None,
        description="Literal signal, measurement or native quantity prefix; omitted lists all.",
    )
    cursor: str | None = Field(default=None, description=_CURSOR_DESCRIPTION)
    limit: int = Field(default=100, ge=1, le=100)

    @model_validator(mode="after")
    def _source(self) -> ResultsQuery:
        if bool(self.path) == bool(self.job_id):
            raise ValueError("provide exactly one of path or job_id")
        if self.path is not None and (self.case_id is not None or self.run_index != 0):
            raise ValueError("run_index and case_id require job_id")
        if self.view == "plots" and self.prefix is not None:
            raise ValueError("prefix does not apply to plot inventory")
        if self.view in ("measurements", "native_tables") and self.plot_index is not None:
            raise ValueError("log views do not select a RAW plot")
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


class GuideQuery(StrictModel):
    """Read the guide: its core and index, or one topic section or task playbook."""

    kind: Literal["guide"]
    section: str | None = Field(
        default=None,
        description=(
            "A name from the core's index ('ltspice', 'bench-craft'). "
            "Omit it for the core and the index."
        ),
    )


class SimulatorDocsQuery(StrictModel):
    """The simulator vendor's own reference documents (LTspice 26.1 and later):
    keyboard shortcuts, menus, the .asc format, .MEAS, the waveform viewer."""

    kind: Literal["simulator_docs"]
    name: str | None = Field(
        default=None,
        description="A listed document, e.g. 'MEAS-REFERENCE.md'; omit it for the list.",
    )
    cursor: str | None = Field(default=None, description=_CURSOR_DESCRIPTION)


class OpenInLtspiceQuery(StrictModel):
    """List the sheets and netlists open in LTspice windows, and which one is in front."""

    kind: Literal["open_in_ltspice"]


Query: TypeAlias = Annotated[
    CapabilitiesQuery
    | SymbolsQuery
    | SymbolQuery
    | NetQuery
    | ComponentsQuery
    | HierarchyQuery
    | ModelQuery
    | ResultsQuery
    | ReferenceQuery
    | GuideQuery
    | SimulatorDocsQuery
    | OpenInLtspiceQuery,
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


def _binding(
    kind: str,
    identity: dict[str, Any],
    sources: Sequence[Path],
    revision: str | None = None,
) -> str:
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
    files passes ``()`` on purpose. ``revision`` stands in for the stamps of a
    file set too large to stat on the event loop: a digest of them, taken off
    the loop where the files were read. Stat granularity bounds the guarantee:
    a rewrite of identical size within one filesystem clock tick still reads as
    unchanged.
    """
    bound = dict(identity)
    if revision is not None:
        bound["sources"] = revision
    elif sources:
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
    items: Sequence[Any],
    kind: str,
    identity: dict[str, Any],
    cursor: str | None,
    sources: Sequence[Path],
    view: _View,
    revision: str | None = None,
) -> dict[str, Any]:
    """Page ``items`` through the shared paginator, bound to this query's identity
    and to the revision of the ``sources`` the rows came from (or ``revision``,
    see ``_binding``).

    The limit comes from ``view``, so a budget that shrinks the page shrinks it
    HERE — before the cursor is minted — and the token the caller gets back
    always resumes at the row after the last one it was shown.
    """
    try:
        return paginate_view(
            items, _binding(kind, identity, sources, revision), cursor=cursor, limit=view.limit
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
    "instances": ("total", "returned", None),
    "results": ("total", "returned", None),
    "docs": ("total", "returned", None),
    "sections": ("total", "returned", None),
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


def _build_facts(
    state: SessionState, cls: type, executable: SimulatorExecutable | None
) -> dict[str, Any]:
    """What one simulator class runs: its last reported build, its dialect, and
    the program it launches."""
    reported = services.reported_version(state, executable)
    info: dict[str, Any] = {
        # What a run on this same executable said about itself; nothing
        # is launched to ask. Null until one has run.
        "version": reported[0] if reported else None,
        "version_source": reported[1] if reported else None,
        "dialect": dialect_for_simulator_name(cls.__name__),
    }
    if executable is not None:
        # The simulator itself, not its launcher: under Wine the command
        # starts with "wine".
        info["executable"] = executable.path
        info["executable_sha256"] = executable.sha256
    return info


def _selection_facts(cls: type) -> dict[str, Any]:
    """Whether run_experiments' execution.simulator may name this simulator:
    being detected is not enough when this host cannot run its family at all,
    and then ``refusal`` says why."""
    refusal = family_refusal(simulator_family(cls))
    if refusal is None:
        return {"selectable": True}
    return {"selectable": False, "refusal": refusal}


def _do_capabilities(
    state: SessionState,
    raster: RasterSupport,
    executables: Mapping[str, SimulatorExecutable | None],
) -> dict[str, Any]:
    """The capabilities report. ``raster`` and ``executables`` (the
    ``executable_identity`` of each available simulator and named executable,
    by family name or selector) are computed off the loop by the caller."""
    simulators: dict[str, Any] = {}
    for name, cls in state.available_simulators.items():
        simulators[name] = {
            "available": True,
            **_selection_facts(cls),
            "default": cls is state.default_simulator,
            **_build_facts(state, cls, executables.get(name)),
        }
    # Every known-but-undetected simulator appears with the exact keys that
    # would turn it on — the config self-diagnosis surface. Detection runs at
    # startup, so a fix always ends in a server restart; the remediation says
    # so rather than leaving the agent to loop on the same absence.
    for name in SIMULATORS:
        if name not in simulators:
            simulators[name] = {
                "available": False,
                "selectable": False,
                "remediation": simulator_remediation(name, state.config),
            }
            refusal = family_refusal(name)
            if refusal is not None:
                simulators[name]["refusal"] = refusal

    # The other builds a run can be put on, by the selector that names one in
    # execution.simulator. A configured executable that could not be bound is
    # not here; the diagnostics below say why.
    named_executables = {
        selector: {
            "family": selector.partition(":")[0],
            **_selection_facts(cls),
            **_build_facts(state, cls, executables.get(selector)),
        }
        for selector, cls in sorted(state.named_simulators.items())
    }

    return {
        "config_path": str(state.config.config_path),
        "python": _python_runtime_facts(),
        "simulators": simulators,
        "named_executables": named_executables,
        "default_simulator": (
            state.default_simulator.__name__ if state.default_simulator else None
        ),
        # The .asc → LTspice netlist exporter needs LTspice itself.
        "exporter_available": "ltspice" in state.available_simulators,
        # Whether verify_circuit can draw a PNG, the only format it returns
        # inline. Asked here so an agent that cannot read files knows before it
        # renders whether it will see the picture, and what to install if not.
        "render": asdict(raster),
        # Whether edit_schematic can keep a sheet that is open in an LTspice
        # window in step with the file, and if not, what is missing.
        "open_window_sync": {
            "available": state.open_windows.available,
            "reason": state.open_windows.unavailable,
            # Whether in_ltspice starts LTspice when no window is open.
            "starts_ltspice": state.open_windows.starts_ltspice,
        },
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
        "allowed_paths": [str(p) for p in state.allowed_paths()],
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
            # null: a case with no execution.run_timeout_s runs until it ends.
            "run_timeout_s": state.config.run_timeout,
            "export_timeout_s": state.config.default_timeout,
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
        asc_dir = safe_path(q.path, state).parent

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
        "symbol_type": sym_info.symbol_type or None,
        "attributes": dict(sym_info.attributes),
        "source_path": str(asy),
        "origin": {"x": 0, "y": 0},
        "bounding_box": sym_info.bbox.to_origin_size_dict(),
        "pins_by_rotation": pins_by_rotation,
        "bbox_by_rotation": bbox_by_rotation,
    }


async def _do_symbol(q: SymbolQuery, state: SessionState) -> dict[str, Any]:
    asc_path: Path | None = None
    if q.path is not None:
        asc_path = safe_path(q.path, state)
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
        f"schematic or a {NETLIST_SUFFIX_TEXT} netlist",
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
    path = safe_path(q.path, state)
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
    if "snapped_to_wire" in tdata:
        data["snapped_to_wire"] = tdata["snapped_to_wire"]
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
    """The trace a schematic ``at`` names. A bare name is a net label, as
    ``net:NAME`` spells it and as a netlist reads the same bare node name."""
    if isinstance(at, list):
        return TraceNetInput(path=path, x=at[0], y=at[1])
    if at.startswith("net:") or "." in at:
        return TraceNetInput(path=path, pin=at)
    return TraceNetInput(path=path, pin=f"net:{at}")


# ---------------------------------------------------------------------------
# components — list vs full, over .asc or netlist
# ---------------------------------------------------------------------------


def _check_prefix(prefix: str | None) -> str | None:
    """The validated prefix, upper-cased: references match it without regard to case.

    Refuses a prefix no reference could start with, rather than answer it with
    an empty list that reads as "no such components".
    """
    if prefix is None:
        return None
    if not prefix or any(ch.isspace() for ch in prefix):
        raise _QueryError(
            "invalid_prefix",
            f"prefix must be the start of a reference, without spaces (e.g. 'R', "
            f"'LX'), got {prefix!r}",
        )
    wildcard = next((i for i, ch in enumerate(prefix) if ch in "*?["), None)
    if wildcard is not None:
        stem = prefix[:wildcard]
        remedy = f"use prefix='{stem}'" if stem else "omit it to list every reference"
        raise _QueryError(
            "invalid_prefix",
            f"prefix matches the start of a reference as plain text and takes no "
            f"wildcards; for {prefix!r}, {remedy}",
        )
    return prefix.upper()


def _components_netlist_payload(
    text: str, prefix: str | None, detail: str
) -> tuple[list[dict[str, Any]], list[str]]:
    """The component rows, plus the lexer's notes about how the deck read.

    ``prefix`` is the upper-cased prefix :func:`_check_prefix` returns.
    """
    from ltspice_mcp.lib.spice_lex_views import body_has_stray_kv_remnant

    lexed = lex(text)
    cards = lexed.cards
    by_ref = instances_by_ref(cards)
    rows: list[dict[str, Any]] = []
    for card in by_ref.values():
        ref = card.name
        if not ref or (prefix is not None and not matches_prefix(ref, prefix)):
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
                entry["symbol"] = comp.symbol
                entry["position"] = {"x": pos.X, "y": pos.Y}
                entry["rotation"] = erot.name if erot else "R0"
                geom = placed_geometry(editor, ref)
                if geom is not None:
                    entry["pins"] = geom["pins"]
                    entry["bounding_box"] = geom["bounding_box"]
        rows.append(entry)
    return rows


async def _do_components(q: ComponentsQuery, state: SessionState, view: _View) -> dict[str, Any]:
    prefix = _check_prefix(q.prefix)
    path = safe_path(q.path, state)
    # The answer rung revokes detail='full' — the one payload-growing opt-in
    # inspect has. The cursor binds the detail it actually rendered, so a page
    # taken under a budget cannot resume as an unbudgeted one at the same offset.
    detail = "list" if view.lean else q.detail
    identity = {"path": str(path), "prefix": prefix, "detail": detail}
    digest: str | None = None
    lex_notes: list[str] = []

    if _route_circuit_kind(path, "components") == "asc":
        digest = await _asc_digest(path)
        # Cached editor + component reads stay on the event loop.
        editor = get_asc_editor(path, state)
        try:
            refs = sorted(editor.get_components())
        except Exception as exc:
            raise _QueryError("parse_error", f"failed to list components: {exc}") from exc
        # Filtered here as in the netlist branch: spicelib's prefix filter reads
        # its argument as a set of case-sensitive first letters (docs/spicelib_bugs.md).
        if prefix is not None:
            refs = [ref for ref in refs if matches_prefix(ref, prefix)]
        page = _paginate(refs, "components", identity, q.cursor, [path], view)
        rows = _components_asc_page(editor, page["items"], detail)
    else:
        try:
            text = await asyncio.to_thread(read_spice_text, path)
        except OSError as exc:
            raise _QueryError("read_error", str(exc)) from exc
        try:
            all_rows, lex_notes = await asyncio.to_thread(
                _components_netlist_payload, text, prefix, detail
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


def _enumerate_libs(lib_paths: list[Path]) -> list[dict[str, Any]]:
    """Every model/subcircuit defined across the given library files (immutable parse)."""
    rows = [
        model_row(entry) for lib in lib_paths for entry in parse_library_file_cached(lib).models
    ]
    rows.sort(key=lambda r: (r["name"].lower(), r["source_path"]))
    return rows


def _search_libs(lib_paths: list[Path], query: str) -> list[dict[str, Any]]:
    """Fuzzy-match ``query`` against the models defined in the given library files.

    The ranking and the row are the ones a search of the simulator's own
    libraries returns, so the two routes differ only in which files they read.
    """
    return rank_models((parse_library_file_cached(lib) for lib in lib_paths), query)


def _search_simulator_libraries(libraries: LibraryManager, query: str) -> tuple[list[dict], str]:
    """Fuzzy-match ``query`` across the detected simulators' own libraries.

    Returns the rows and a digest of the revisions of the files searched,
    hashed here in the worker because a full install is thousands of files.
    """
    rows, revisions = libraries.search(query)
    return rows, canonical_hash(revisions)


async def _admit_libs(libs: list[str], state: SessionState) -> list[Path]:
    """Resolve the named library files through ``safe_library_path``, off the
    loop: a path outside the sandbox is checked against the simulators'
    library directories, which may sit on a slow WSL mount."""
    return await asyncio.to_thread(lambda: [safe_library_path(lib, state) for lib in libs])


async def _do_model(q: ModelQuery, state: SessionState, view: _View) -> dict[str, Any]:
    # The library files these rows were read from, so the cursor binds their
    # revision: an edited library must reject a stale token, not page into the
    # re-parsed list at the old offset.
    sources: list[Path] = []
    revision: str | None = None
    if q.mode == "enumerate":
        sources = await _admit_libs(q.libs or [], state)
        try:
            rows = await asyncio.to_thread(_enumerate_libs, sources)
        except OSError as exc:
            raise _QueryError("read_error", str(exc)) from exc
        if q.query:
            # A listing narrowed by name, case-insensitively: the filter the
            # response echoes is the one applied, unlike search's fuzzy score.
            needle = q.query.casefold()
            rows = [row for row in rows if needle in row["name"].casefold()]
        identity: dict[str, Any] = {
            "mode": "enumerate",
            "libs": [str(p) for p in sources],
            "query": q.query,
        }
    else:
        assert q.query is not None  # guaranteed by the model validator
        identity = {"mode": "search", "query": q.query, "libs": q.libs}
        if q.libs:
            sources = await _admit_libs(q.libs, state)
            try:
                rows = await asyncio.to_thread(_search_libs, sources, q.query)
            except OSError as exc:
                raise _QueryError("read_error", str(exc)) from exc
        else:
            # No libs given: search the detected simulators' own libraries,
            # the directories _admit_libs accepts, so every source_path in the
            # rows reads back through 'libs'. Offloaded because the first search
            # parses the whole install.
            try:
                rows, revision = await asyncio.to_thread(
                    _search_simulator_libraries, state.libraries, q.query
                )
            except Exception as exc:
                raise _QueryError("search_error", str(exc)) from exc

    page = _paginate(rows, "model", identity, q.cursor, sources, view, revision)
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
            "query for the whole vocabulary, or read the guide: inspect(kind='guide')."
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
# simulator_docs
# ---------------------------------------------------------------------------


def _reference_directory(state: SessionState) -> Path:
    """Where the detected LTspice keeps its reference documents, the default
    simulator's first. Reads the filesystem: call off the loop."""
    candidates = [
        state.default_simulator,
        *state.available_simulators.values(),
        *state.named_simulators.values(),
    ]
    for simulator in candidates:
        directory = simulator_docs.reference_directory(simulator_library_roots(simulator))
        if directory is not None:
            return directory
    raise _QueryError(
        "simulator_docs_unavailable",
        "No reference documents are installed here: LTspice 26.1 and later installs "
        "them in a 'reference' directory beside its library, and no detected simulator "
        "has one. The guide (inspect kind 'guide') is this server's own.",
    )


def _simulator_docs_page(
    q: SimulatorDocsQuery, state: SessionState, view: _View
) -> dict[str, Any]:
    """The list of reference documents, or one of them in sections. Blocking."""
    directory = _reference_directory(state)
    if q.name is None:
        rows = [
            {"name": doc.name, "title": doc.title, "description": doc.description}
            for doc in simulator_docs.documents(directory)
        ]
        page = _paginate(rows, "simulator_docs", {"name": None}, q.cursor, (), view)
        return {
            "data": {
                "source": str(directory),
                "docs": page["items"],
                "total": page["total"],
                "returned": page["returned"],
                "hint": (
                    "Read one with name. These are the vendor's own reference, the "
                    "authority on the program itself; the guide is about this server."
                ),
            },
            "next_cursor": page["next_cursor"],
            "page": _page_meta(page, "docs"),
        }
    found = simulator_docs.read(directory, q.name)
    if found is None:
        raise _QueryError(
            "unknown_document",
            f"No reference document is called {q.name!r}.",
            supported=simulator_docs.names(directory),
        )
    document, sections = found
    rows = [{"heading": section.heading, "text": section.text} for section in sections]
    page = _paginate(
        rows, "simulator_docs", {"name": document.name}, q.cursor, (document.path,), view
    )
    return {
        "data": {
            "name": document.name,
            "title": document.title,
            "sections": page["items"],
            "total": page["total"],
            "returned": page["returned"],
        },
        "next_cursor": page["next_cursor"],
        "page": _page_meta(page, "sections"),
    }


# ---------------------------------------------------------------------------
# open_in_ltspice
# ---------------------------------------------------------------------------

#: The most documents one reply lists; the rest are counted.
_OPEN_DESIGNS_LIMIT = 100
#: The most differing entries listed from each side of one sheet; the rest are counted.
_DIFFERENCE_ENTRIES_LIMIT = 25


def _design_kind(spelled: str) -> str:
    suffix = Path(spelled).suffix.lower()
    if suffix == ".asc":
        return "schematic"
    return "netlist" if suffix in NETLIST_SUFFIXES else "other"


def _design_row(design: OpenDesign, resolved: Path | None) -> dict[str, Any]:
    """One open document as the reply lists it; ``resolved`` is its path where
    the sandbox admits it. Reads the file: call off the loop."""
    row: dict[str, Any] = {
        "path": design.path,
        "kind": _design_kind(design.path),
        "active": design.active,
        "pid": design.pid,
        "version": design.version,
        "in_sandbox": resolved is not None,
    }
    if resolved is None or design.text is None or not resolved.is_file():
        return row
    on_disk = resolved.read_bytes()
    row["sha256"] = hashlib.sha256(on_disk).hexdigest()
    row.update(window_difference(on_disk, design.text, entries=_DIFFERENCE_ENTRIES_LIMIT))
    return row


async def _do_open_in_ltspice(state: SessionState) -> dict[str, Any]:
    """What LTspice has open, asked of LTspice (``OpenWindows.designs``).

    A sheet's copy in the window is read only for a file the sandbox admits,
    and compared with that file by content. Both the bridge and the files are
    read off the loop.
    """

    def read() -> tuple[int, int, list[dict[str, Any]]]:
        # Where the sandbox admits each document, decided once: it says which
        # copies a window is asked for and which files are then read.
        admitted: dict[str, Path | None] = {}

        def is_readable_sheet(spelled: str) -> bool:
            admitted[spelled] = sandboxed(spelled, state)
            return admitted[spelled] is not None and _design_kind(spelled) == "schematic"

        count, designs = state.open_windows.designs(is_readable_sheet)
        listed = designs[:_OPEN_DESIGNS_LIMIT]
        return count, len(designs), [_design_row(d, admitted[d.path]) for d in listed]

    try:
        count, total, rows = await asyncio.to_thread(read)
    except WindowsUnavailable as exc:
        raise _QueryError(
            "open_windows_unavailable",
            f"{exc}. inspect(kind='capabilities') reports this under open_window_sync.",
        ) from exc
    except BridgeError as exc:
        raise _QueryError(
            "open_windows_unreachable", f"LTspice could not be asked what it has open: {exc}."
        ) from exc
    data: dict[str, Any] = {"windows": count, "designs": rows, "total": total}
    hints: list[str] = []
    if any(row.get("differs_from_file") for row in rows):
        hints.append(
            "A sheet whose window differs from its file is refused by edit_schematic until "
            "it is saved or closed in LTspice; a run or a check reads the file, not the "
            "window. only_in_window and only_in_file list where the two differ."
        )
    if any(not row["in_sandbox"] for row in rows):
        hints.append(
            "A document outside the sandbox is listed and not read. " + state.sandbox_guidance()
        )
    if hints:
        data["hint"] = " ".join(hints)
    return {"data": data}


# ---------------------------------------------------------------------------
# guide
# ---------------------------------------------------------------------------


def _do_guide(q: GuideQuery, state: SessionState) -> dict[str, Any]:
    """Serve the guide's core and index, or one section.

    The text is packaged and read once per process (``lib/guide.py``), so
    nothing is offloaded. A read here is what the session's read-the-guide
    reminder waits for, whichever section it names.
    """
    try:
        text = guide.read(q.section)
    except guide.UnknownGuideSection as exc:
        raise _QueryError("unknown_section", str(exc), supported=list(exc.known)) from exc
    state.guide_read = True
    return {"data": {"section": q.section, "title": guide.title_of(q.section), "text": text}}


# ---------------------------------------------------------------------------
# Shared per-item helpers + dispatch
# ---------------------------------------------------------------------------


def _hierarchy_page(q: HierarchyQuery, state: SessionState, view: _View) -> dict[str, Any]:
    prefix = _check_prefix(q.prefix)
    profile = SemanticProfile(
        q.simulator,
        (q.ngbehavior if q.ngbehavior is not None else current_ngbehavior())
        if q.simulator == "ngspice"
        else None,
    )
    hierarchy = load_hierarchy(
        q.path,
        state.allowed_paths(),
        profile,
        simulator_roots=simulator_library_roots(state.available_simulators.get(q.simulator)),
    )
    selected = tuple(p.casefold() for p in q.instance) if q.instance else ()
    if selected and not any(
        tuple(p.casefold() for p in row.instance) == selected for row in hierarchy.instances
    ):
        raise NetlistError(f"instance segments not found: {q.instance}")
    rows = [
        row
        for row in hierarchy.instances
        if tuple(p.casefold() for p in row.instance[: len(selected)]) == selected
        and (prefix is None or matches_prefix(row.reference, prefix))
    ]
    identity = {
        **hierarchy.binding(),
        "instance": selected,
        "prefix": prefix,
    }
    page = _paginate(rows, "hierarchy", identity, q.cursor, (), view)
    metadata = _page_meta(page, "instances")
    # The complete Python collector identifies rows through named collections.
    metadata["collections"] = {"instances": dict(metadata)}
    return {
        "data": {
            "instances": [row.row() for row in page["items"]],
            "profile": identity["profile"],
            "inputs": identity["inputs"],
            "total": page["total"],
            "returned": page["returned"],
        },
        "next_cursor": page["next_cursor"],
        "page": metadata,
    }


class _TableRows(Sequence[dict[str, Any]]):
    """Index resident native quantities; materialize only the requested page."""

    def __init__(self, raw: DecodedRaw, traces: Sequence[TraceDescriptor]) -> None:
        self._blocks: list[tuple[int, TraceDescriptor, NDArray]] = []
        self._ends: list[int] = []
        total = 0
        for step in raw.get_steps():
            for trace in traces:
                wave = raw.get_wave(trace.name, step=step)
                if len(wave):
                    total += len(wave)
                    self._blocks.append((step, trace, wave))
                    self._ends.append(total)

    def __len__(self) -> int:
        return self._ends[-1] if self._ends else 0

    @overload
    def __getitem__(self, index: int) -> dict[str, Any]: ...

    @overload
    def __getitem__(self, index: slice) -> list[dict[str, Any]]: ...

    def __getitem__(self, index: int | slice) -> dict[str, Any] | list[dict[str, Any]]:
        if isinstance(index, slice):
            return [self[row] for row in range(*index.indices(len(self)))]
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        block = bisect_right(self._ends, index)
        step, trace, wave = self._blocks[block]
        sample_index = index - (self._ends[block - 1] if block else 0)
        sample = wave[sample_index]
        number = (
            {"real": float(sample.real), "imag": float(sample.imag)}
            if wave.dtype.kind == "c"
            else float(sample)
        )
        return {
            "signal": trace.name,
            "step_index": step,
            "sample_index": sample_index,
            "value": number,
            "unit": trace.unit,
        }


class _LogRows(Sequence[dict[str, Any]]):
    """Page detached measurements or physical native print blocks lazily."""

    def __init__(self, facts: Any, view: str, prefix: str | None) -> None:
        self._view = view
        self._blocks: list[tuple[dict[str, Any], list[Any]]] = []
        self._ends: list[int] = []
        if view == "measurements":
            for name, entry in (facts or {}).get("measurements", {}).items():
                if prefix is None or matches_prefix(name, prefix):
                    self._append({"measurement": name, **entry}, entry["values"])
        else:
            for block in facts or []:
                if block["layout"] == "scalar_print":
                    entries = [
                        entry
                        for entry in block["entries"]
                        if prefix is None or matches_prefix(entry["label"], prefix)
                    ]
                    self._append(
                        {key: value for key, value in block.items() if key != "entries"}, entries
                    )
                elif prefix is None or matches_prefix(block["column"]["label"], prefix):
                    self._append(
                        {key: value for key, value in block.items() if key != "rows"},
                        block["rows"],
                    )

    def _append(self, metadata: dict[str, Any], rows: list[Any]) -> None:
        if rows:
            self._blocks.append((metadata, rows))
            self._ends.append((self._ends[-1] if self._ends else 0) + len(rows))

    def __len__(self) -> int:
        return self._ends[-1] if self._ends else 0

    @overload
    def __getitem__(self, index: int) -> dict[str, Any]: ...

    @overload
    def __getitem__(self, index: slice) -> list[dict[str, Any]]: ...

    def __getitem__(self, index: int | slice) -> dict[str, Any] | list[dict[str, Any]]:
        if isinstance(index, slice):
            return [self[row] for row in range(*index.indices(len(self)))]
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        block = bisect_right(self._ends, index)
        metadata, rows = self._blocks[block]
        ordinal = index - (self._ends[block - 1] if block else 0)
        if self._view == "measurements":
            row = {
                "measurement": metadata["measurement"],
                "value_index": ordinal,
                "value": rows[ordinal],
            }
            for key in ("range_from", "range_to", "at"):
                value = metadata.get(key)
                row[key] = value[ordinal] if isinstance(value, list) else value
            return row
        key = "entry" if metadata["layout"] == "scalar_print" else "row"
        return {**metadata, key: rows[ordinal]}


async def _log_results_page(
    query: ResultsQuery, source: services.AnalysisSource, state: SessionState, view: _View
) -> dict[str, Any]:
    artifacts = await services.load_artifacts(source, state, require_raw=False)
    logs = artifacts.logs
    section_name = "measurements" if query.view == "measurements" else "native_tables"
    section = logs.section(section_name)
    facts = logs.value(section_name)
    rows = _LogRows(facts, query.view, _check_prefix(query.prefix))
    identity = {
        **query.model_dump(exclude={"cursor", "limit"}),
        "snapshot_id": artifacts.snapshot_id,
    }
    page = _paginate(
        rows,
        "results",
        identity,
        query.cursor,
        (),
        _View(limit=min(query.limit, view.limit), lean=view.lean, shrunk=view.shrunk),
    )
    data = {
        "snapshot_id": artifacts.snapshot_id,
        "capture_facts": logs.capture_facts,
        "scan": logs.scan,
        "section": {key: value for key, value in section.items() if key != "value"},
        query.view: page["items"],
    }
    if query.view == "measurements" and facts is not None:
        data.update({key: value for key, value in facts.items() if key != "measurements"})
    metadata = _page_meta(page, "results")
    metadata["collections"] = {query.view: dict(metadata)}
    return {"data": data, "next_cursor": page["next_cursor"], "page": metadata}


async def _results_page(query: ResultsQuery, state: SessionState, view: _View) -> dict[str, Any]:
    log_view = query.view in ("measurements", "native_tables")
    if query.path is not None:
        source = services.resolve_analysis_source(
            state,
            raw_file=None if log_view else query.path,
            log_file=query.path if log_view else None,
            plot_index=query.plot_index or 0,
            dialect=query.dialect,
        )
    else:
        assert query.job_id is not None
        job = await services.resolve_job_async(query.job_id, state)
        run = services.experiment_run_context(
            job, state, run_index=query.run_index, case_id=query.case_id, require_raw=not log_view
        )
        source = services.source_for_run(
            run, plot_index=query.plot_index or 0, dialect=query.dialect
        )
    if log_view:
        return await _log_results_page(query, source, state, view)
    raw = await services.load_raw(source, state)
    descriptor = raw.descriptor
    prefix = _check_prefix(query.prefix)
    traces = [
        trace
        for trace in descriptor.traces
        if prefix is None or matches_prefix(trace.name, prefix)
    ]
    rows: Sequence[dict[str, Any]]
    if query.view == "plots":
        rows = [asdict(plot.descriptor) for plot in raw.plots]
    elif query.view == "signals":
        rows = [
            {
                **asdict(trace),
                "axis": descriptor.axis is not None and trace.name == descriptor.axis.name,
            }
            for trace in traces
        ]
    else:
        if descriptor.axis is not None:
            raise _QueryError("no_table", "table view requires a plot with no sampled axis")

        rows = await asyncio.to_thread(_TableRows, raw, traces)
    identity = {
        **query.model_dump(exclude={"cursor", "limit"}),
        "snapshot_id": descriptor.snapshot_id,
        "resolved_dialect": descriptor.dialect,
    }
    page = _paginate(
        rows,
        "results",
        identity,
        query.cursor,
        (),
        _View(limit=min(query.limit, view.limit), lean=view.lean, shrunk=view.shrunk),
    )
    data = {
        "snapshot_id": descriptor.snapshot_id,
        "plot_index": raw.plot_index,
        "dialect": descriptor.dialect,
        "analysis": descriptor.analysis,
        "axis": asdict(descriptor.axis) if descriptor.axis else None,
        query.view: page["items"],
    }
    metadata = _page_meta(page, "results")
    metadata["collections"] = {query.view: dict(metadata)}
    return {"data": data, "next_cursor": page["next_cursor"], "page": metadata}


async def _dispatch(query: Query, state: SessionState, view: _View) -> dict[str, Any]:
    if isinstance(query, ResultsQuery):
        return await _results_page(query, state, view)
    if isinstance(query, CapabilitiesQuery):
        wanted = set(query.fields) if query.fields is not None else None
        # Executables are identified only for a report that shows them. A
        # selector carries a ':' and a family name never does, so the two
        # tables share one mapping.
        simulators = {
            **(state.available_simulators if wanted is None or "simulators" in wanted else {}),
            **(state.named_simulators if wanted is None or "named_executables" in wanted else {}),
        }

        def probe() -> tuple[RasterSupport, dict[str, SimulatorExecutable | None]]:
            # Off the loop: the first successful raster probe loads the native
            # Cairo library, and the first identification of an executable
            # digests it.
            identities = {name: executable_identity(cls) for name, cls in simulators.items()}
            return raster_support(), identities

        raster, executables = await asyncio.to_thread(probe)
        report = _do_capabilities(state, raster, executables)
        if wanted is not None:
            report = {key: value for key, value in report.items() if key in wanted}
        return {"data": report}
    if isinstance(query, SymbolsQuery):
        return await _do_symbols(query, state, view)
    if isinstance(query, SymbolQuery):
        return await _do_symbol(query, state)
    if isinstance(query, NetQuery):
        return await _do_net(query, state, view)
    if isinstance(query, ComponentsQuery):
        return await _do_components(query, state, view)
    if isinstance(query, HierarchyQuery):
        return await asyncio.to_thread(_hierarchy_page, query, state, view)
    if isinstance(query, ReferenceQuery):
        return _do_reference(query, view, frozenset(state.tool_dispatch))
    if isinstance(query, GuideQuery):
        return _do_guide(query, state)
    if isinstance(query, SimulatorDocsQuery):
        return await asyncio.to_thread(_simulator_docs_page, query, state, view)
    if isinstance(query, OpenInLtspiceQuery):
        return await _do_open_in_ltspice(state)
    # Exhaustive over the sealed union: ModelQuery is the only remaining member.
    return await _do_model(query, state, view)


_ERROR_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "code": {"type": "string"},
        "message": {"type": "string"},
        "supported": {"type": "array", "items": {"type": "string"}},
        "hint": HINT_SCHEMA,
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
    # enumerate that lists everything rather than the names containing it.
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

#: The ``guide`` kind's payload: the text itself, and which part of the guide
#: it is. Result queries also return ``section``, as log-section metadata.
_GUIDE_DATA_PROPERTIES: dict[str, Any] = {
    "section": {
        "type": ["string", "null"],
        "description": "The section read; null for the core and its index.",
    },
    "title": {"type": "string"},
    "text": {"type": "string", "description": "Markdown."},
}

#: The ``simulator_docs`` kind's payload: the list, or one document's sections.
_SIMULATOR_DOCS_DATA_PROPERTIES: dict[str, Any] = {
    "source": {"type": "string", "description": "The directory the documents are read from."},
    "docs": {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "What to pass as name."},
                "title": {"type": "string"},
                "description": {"type": "string"},
            },
            "required": ["name", "title", "description"],
        },
    },
    "sections": {
        "type": "array",
        "description": "The document in order, cut at its second-level headings.",
        "items": {
            "type": "object",
            "properties": {
                "heading": {"type": "string"},
                "text": {"type": "string", "description": "Markdown, heading line included."},
            },
            "required": ["heading", "text"],
        },
    },
}

#: The ``open_in_ltspice`` kind's payload.
_OPEN_DESIGNS_DATA_PROPERTIES: dict[str, Any] = {
    "windows": {"type": "integer", "description": "How many LTspice windows are running."},
    "total": {"type": "integer", "description": "How many documents they have open in all."},
    "designs": {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "As LTspice spells it."},
                "kind": {"enum": ["schematic", "netlist", "other"]},
                "active": {
                    "type": "boolean",
                    "description": "The document in front in its window.",
                },
                **LTSPICE_WINDOW_PROPERTIES,
                "in_sandbox": {
                    "type": "boolean",
                    "description": "Whether this server may read and edit the file.",
                },
                "sha256": {
                    "type": "string",
                    "description": "Of the file: what edit_schematic takes as expected_sha256.",
                },
                "differs_from_file": {
                    "type": "boolean",
                    "description": (
                        "The window's copy is not the file's: unsaved changes, or a "
                        "file that changed after it was opened. A sheet in the sandbox only."
                    ),
                },
                "difference": {"type": "string"},
                **LTSPICE_DIFFERENCE_ENTRIES,
            },
            "required": ["path", "kind", "active", "pid", "version", "in_sandbox"],
        },
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
                    # the module docstring and stays open here by design.
                    "data": {
                        "type": "object",
                        "properties": {
                            **_REFERENCE_DATA_PROPERTIES,
                            **_GUIDE_DATA_PROPERTIES,
                            **_SIMULATOR_DOCS_DATA_PROPERTIES,
                            **_OPEN_DESIGNS_DATA_PROPERTIES,
                            "plot_index": {"type": "integer", "minimum": 0},
                            "snapshot_id": {"type": "string"},
                            "dialect": {"type": "string"},
                            "analysis": {"type": "string"},
                            "axis": {"type": ["object", "null"]},
                            "plots": {"type": "array", "items": {"type": "object"}},
                            "signals": {"type": "array", "items": {"type": "object"}},
                            "table": {"type": "array", "items": {"type": "object"}},
                            "measurements": {"type": "array", "items": {"type": "object"}},
                            "native_tables": {"type": "array", "items": {"type": "object"}},
                            "capture_facts": {"type": "object"},
                            "scan": {"type": "object"},
                            "section": {
                                "anyOf": [
                                    _GUIDE_DATA_PROPERTIES["section"],
                                    {"type": "object"},
                                ]
                            },
                        },
                    },
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
    "'components', 'hierarchy', 'model', 'results', 'reference', 'guide', 'simulator_docs', "
    "'open_in_ltspice' "
    "— each with its own arguments, "
    "described on its branch of the query schema. 'reference' searches every tool's "
    "recipes, ops, checks and their fields in plain words ('phase margin'). 'guide' "
    "returns the guide's core, or one 'section' its index names. A "
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
        except PathSecurityError as exc:
            # Every kind's refusal lands here, whichever resolver raised it (the
            # hierarchy loader checks each file it opens), so all report one code
            # and one remedy.
            error = {"code": exc.code, "message": str(exc), "hint": state.sandbox_guidance()}
            results.append(_failure_item(index, raw, error))
            continue
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


#: This tool's budget epilogue, on ``observations``. Its trim rung drops only
#: an exhausted item's page metadata, which the rows it returned restate, so the
#: server's default budget never has anything to report here.
_BUDGET_NOTES = response_budget.Notes(
    cut="presentation was reduced; no query was dropped and no error was hidden.",
    route=(
        "Ask again with a larger 'budget' for the full presentation, or page on "
        "with each item's next_cursor."
    ),
)


def _paged_surfaces(data: dict[str, Any]) -> list[list[Any]]:
    """Every row list the answered batch is showing, one per item's list: the
    shrunk limit caps each of them on its own."""
    surfaces: list[list[Any]] = []
    for item in data["results"]:
        payload = item.get("data")
        if not isinstance(payload, dict):
            continue
        surfaces.extend(value for value in payload.values() if isinstance(value, list))
    return surfaces


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

    def shrunk(rung: response_budget.Rung, page: int, limit: int, coord_limit: int) -> _View:
        """The view the standing envelope's rows, on a page of ``page`` estimated
        tokens, leave room for under ``rung``."""
        measure = response_budget.RowMeasure.of(_paged_surfaces(rendered), page=page)
        return _View(
            limit=measure.fit_limit(limit, rung),
            coord_limit=measure.fit_limit(coord_limit, rung),
            lean=rung.answer_channel,
            shrunk=True,
        )

    async def build(view: _View, rung: response_budget.Rung) -> None:
        nonlocal rendered, built_from
        if built_from != view:
            if view not in passes:
                passes[view] = await _run_queries(args, state, view)
            # A copy per envelope, because a pass is cached and reused across
            # rungs while the envelope built from it is degraded in place.
            rendered = inspect_envelope(copy.deepcopy(passes[view]))
            built_from = view
        _degrade_inspect(rendered, rung)

    async def render(rung: response_budget.Rung) -> dict[str, Any]:
        if not rung.shrink:
            await build(_View(lean=rung.answer_channel), rung)
            return rendered
        view = shrunk(rung, rung.measured, _PAGE_SIZE, _COORD_PAGE_SIZE)
        await build(view, rung)
        measured = response_budget.estimate_tokens(rendered)
        if measured > rung.body_budget:
            # A cut adds what the uncut envelope never showed — a cursor for
            # every query it cut — so the estimate is taken once more from this
            # envelope, which carries them. Once, not a search: each limit
            # re-asks every query in the batch.
            await build(shrunk(rung, measured, view.limit, view.coord_limit), rung)
        return rendered

    assert budget.tokens is not None  # the undegraded path never reaches here
    result = await response_budget.negotiate(budget.tokens, render, max_rung=budget.max_rung)
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
            if r["error"].get("hint"):
                lines.append(r["error"]["hint"])
    return "\n".join(lines) if lines else "(no queries)"
