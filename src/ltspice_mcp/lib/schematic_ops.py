"""The .asc schematic edit engine: op models, appliers, geometry and net tracing.

One implementation, three consumers — ``edit_schematic`` applies the ops,
``inspect`` reads the same nets and components through the same cached editor,
and ``verify_circuit`` reuses the wiring geometry so a refusal the editor
enforces and a finding the checker reports cannot drift apart.

What lives here:

- the typed op union (``OpAddComponent`` … ``OpSetPlotPanes``), the in-place
  applier ``apply_op_inplace``, the facts its results report
  (``OP_RESULT_FACTS``), the ops that leave wiring alone (``KEEPS_WIRING``),
  and the batch runner ``run_op_batch``;
- ``SheetPlotSettings``, the plot settings file beside a sheet as one batch
  changes it: the one op that does not edit the sheet itself writes there;
- ``edit_guard``, which serializes one file's mutation in-process and across
  parallel server sessions (with the files beside it a batch writes,
  ``files_written_beside``), and the cached-editor accessors it wraps;
- the placement and net-partition geometry (``placed_geometry``,
  ``resolve_pin``, ``net_partition``, ``trace_nets``), which reads each
  symbol once per request through ``symbol_info_for``, and the route
  planner behind ``wire_pins``, which resolves the ends of a route and asks
  ``lib/routing.py`` what drawing it would do;
- the wiring profile. What is said of a whole sheet after an edit is
  ``lib/sheet_findings.py``, read by ``tools/schematic_edit.py``.

Names imported by another module are public.

Extension-based dispatch: the file extension picks the spicelib editor
(SpiceEditor for .cir/.net, AscEditor for .asc). Schematic-only operations
validate the extension and raise NetlistError for a non-.asc file.
"""

import asyncio
import importlib
import math
import re
import traceback
from collections import defaultdict
from collections.abc import AsyncIterator, Callable, Container, Iterable, Mapping, Sequence
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from types import CodeType
from typing import Any, Literal, NamedTuple
from weakref import WeakKeyDictionary

from pydantic import Field
from spicelib import AscEditor, SpiceEditor
from spicelib.editor.asc_editor import LTSPICE_ATTRIBUTES, LTSPICE_PARAMETERS
from spicelib.editor.base_schematic import (
    ERotation,
    Line,
    Point,
    SchematicComponent,
    Text,
    TextTypeEnum,
)
from spicelib.utils.detect_encoding import EncodingDetectError
from spicelib.utils.file_search import search_file_in_containers

# The concrete class to instantiate for a from-scratch .asc component.
# spicelib 1.6 introduced ``AscComponent`` (the type its own .asc parser
# builds) and turned ``Component.attributes`` into a lazy property that raises
# ``NotImplementedError`` for a bare ``SchematicComponent`` whose attribute
# store is still empty — so building one from scratch detonates on first
# attribute access. ``AscComponent`` implements the contract; on spicelib < 1.6
# (no ``AscComponent``) the plain ``SchematicComponent`` already works.
# Resolved dynamically because the symbol does not exist on the pinned (<1.6)
# spicelib, so a static import would not type-check.
try:
    _SchematicComponentClass: type = importlib.import_module(
        "spicelib.editor.asc_editor"
    ).AscComponent
except (ImportError, AttributeError):  # spicelib < 1.6 (the currently pinned range)
    _SchematicComponentClass = SchematicComponent

from ltspice_mcp.errors import NetlistError, SymbolResolutionError
from ltspice_mcp.lib.component_value import POSITIONAL_KINDS
from ltspice_mcp.lib.connectivity import (
    NamedNets,
    NetPartition,
    build_on_wire_predicate,
    is_ground,
    label_folded_nets,
    name_key,
    name_keys,
    named_labels,
    partition,
    same_instance_dropped_segments,
)
from ltspice_mcp.lib.encoding import refused_sheet_mark, refused_sheet_mark_note
from ltspice_mcp.lib.filelock import circuit_file_lock, path_lock
from ltspice_mcp.lib.format import is_scaled_number, parse_spice_value
from ltspice_mcp.lib.geometry import BBox
from ltspice_mcp.lib.models import StrictModel
from ltspice_mcp.lib.plot_settings import (
    SECTION_NAMES,
    PlotAnalysis,
    PlotPane,
    PlotSettings,
    XScale,
    YScale,
    inherit_grid,
    pane_specs,
    plot_settings_path,
    read_plot_settings,
    scales_of,
    with_panes,
    write_plot_settings,
)
from ltspice_mcp.lib.routing import (
    Route,
    judge,
    segment_json,
    segment_text,
    wires_through,
)
from ltspice_mcp.lib.sheet import (
    INSTANCE_PREFIX,
    PARAMETER_ATTRIBUTES,
    PlacedPin,
    element_letter,
    find_pin,
)
from ltspice_mcp.lib.sheet_findings import Part, SheetView
from ltspice_mcp.lib.spice_lex import (
    SpiceCard,
    SpiceLexError,
    TokenKind,
    key_value_tokens,
    tokenize_body,
)
from ltspice_mcp.lib.spice_validator import (
    validate_directive,
)
from ltspice_mcp.lib.symbol_geometry import SymbolInfo, compute_placed_geometry, get_symbol_info
from ltspice_mcp.state import SessionState


def _reject_empty_attr_value(reference: str, attribute: str, value: str) -> None:
    """Refuse an empty SYMATTR value on any .asc write path.

    LTspice writes ``SYMATTR <attr>`` (keyword + name, no value) for an empty
    value; spicelib's parser then can't unpack that 2-token line on the NEXT
    read, leaving the .asc permanently unreadable until hand-edited. Reject up
    front rather than emit an .asc the server's own reader rejects.
    """
    if not value.strip():
        raise NetlistError(
            f"Attribute {attribute!r} on {reference!r}: an empty value writes a "
            "2-token SYMATTR line the schematic parser cannot read back, "
            "corrupting the .asc. Provide a non-empty value or omit the attribute."
        )


def _attribute_value(reference: str, attribute: str, value: str) -> str:
    """The value an attribute is written with.

    A ``Prefix`` must start with the element letter it gives the part, and is
    written in upper case, the way spicelib reads it back; LTspice reads either
    case the same.
    """
    if attribute != INSTANCE_PREFIX:
        return value
    letter = value.strip()[:1]
    if not (letter.isascii() and letter.isalpha()):
        raise NetlistError(
            f"Prefix {value!r} on {reference!r} does not start with an element "
            "letter: give the letter the part netlists as, such as 'X' for a "
            "subcircuit."
        )
    return value.strip().upper()


def _require_clearable_attr(reference: str, attribute: str) -> None:
    """Refuse clearing an attribute the component cannot live without.

    An empty value on set_component_attribute means "remove the SYMATTR line"
    (LTspice's format has no empty-value representation — a 2-token SYMATTR
    line is unreadable on the next parse). InstName is the instance's
    identity and must never be removed.
    """
    if attribute == "InstName":
        raise NetlistError(
            f"InstName on {reference!r} cannot be cleared — it is the "
            "component's identity. Use remove_component to delete the "
            "instance, or set a new non-empty name."
        )


def _reject_unknown_attr(attribute: str) -> None:
    """Refuse a SYMATTR name outside LTspice's fixed slot set.

    Any name outside the known slots is dropped at netlist-export time, so a
    typo (``Val`` for ``Value``) would silently no-op. Reject up-front with a
    'did you mean' hint. Shared by set_component_attribute (handler + op) and
    the add_component create path so all three refuse identically.
    """
    if attribute in _LTSPICE_ATTR_NAMES:
        return
    canonical = _LTSPICE_ATTR_CANONICAL.get(attribute.lower())
    if canonical:
        raise NetlistError(
            f"Unknown attribute {attribute!r}. Did you mean {canonical!r}? "
            "LTspice attribute names are case-sensitive."
        )
    raise NetlistError(
        f"Unknown attribute {attribute!r}. LTspice silently ignores "
        f"unrecognised SYMATTR keys at netlist time. Valid attributes: "
        f"{', '.join(sorted(_LTSPICE_ATTR_NAMES))}. For arbitrary KEY=val "
        "pairs, set them through 'SpiceLine' instead."
    )


def create_component(
    editor: AscEditor,
    reference: str,
    symbol: str,
    x: int,
    y: int,
    rotation: ERotation,
    *,
    value: str | None = None,
    attributes: dict[str, str] | None = None,
) -> None:
    """Create and add a SchematicComponent to an AscEditor.

    Wraps the fragile pattern of constructing a blank SchematicComponent
    then manually setting .reference, .symbol, .position, .rotation.
    """
    comp = _SchematicComponentClass(editor, "")
    comp.reference = reference
    comp.symbol = symbol  # pyright: ignore[reportAttributeAccessIssue]
    comp.position = Point(x, y)
    comp.rotation = rotation
    if value is not None:
        # An empty Value writes the same 2-token SYMATTR line that corrupts the
        # .asc on the next read, so guard this path too (not just attributes).
        _reject_empty_attr_value(reference, "Value", value)
        comp.attributes["Value"] = value
    if attributes:
        for attr_name, attr_val in attributes.items():
            _reject_unknown_attr(attr_name)
            _reject_empty_attr_value(reference, attr_name, attr_val)
            comp.attributes[attr_name] = _attribute_value(reference, attr_name, attr_val)
    editor.add_component(comp)


# Per-file locks to prevent concurrent edits to the same circuit file.
# Bounded to avoid unbounded growth; only evicts *unheld* locks.
_MAX_EDIT_LOCKS = 64
_edit_locks: dict[Path, asyncio.Lock] = {}


def _get_edit_lock(path: Path) -> asyncio.Lock:
    """Get or create a per-file edit lock (shared LRU mechanism: ``path_lock``)."""
    return path_lock(_edit_locks, path, _MAX_EDIT_LOCKS)


@asynccontextmanager
async def edit_guard(path: Path, *beside: Path) -> AsyncIterator[None]:
    """Serialize a mutation of one circuit file, in-process and cross-process.

    Layering: the per-path asyncio lock first (tasks in this session), then
    the cross-process file lock (parallel server sessions in the same
    directory). Every mutation is a whole-file read-modify-write, so an
    unserialized concurrent edit is last-writer-wins; this guard plus the
    editor cache's stat-on-fetch — which must happen INSIDE the guard —
    turn that into edit-on-latest.

    ``beside`` names the other files the same mutation writes
    (``files_written_beside``). Their file locks are taken after the
    circuit's own, the order a netlist export takes a sheet's and its
    netlist's in, so no two guards wait on each other in a cycle.
    """
    async with _get_edit_lock(path), circuit_file_lock(path), AsyncExitStack() as stack:
        for other in beside:
            await stack.enter_async_context(circuit_file_lock(other))
        yield


# Standard LTspice SYMATTR slot names. Anything outside this set is
# silently ignored at netlist-export time, so we reject up-front rather
# than letting a typo no-op silently. Sourced from spicelib so a future
# release that adds a slot is picked up without a code change here.
_LTSPICE_ATTR_NAMES: frozenset[str] = frozenset(
    (*LTSPICE_PARAMETERS, *LTSPICE_ATTRIBUTES, INSTANCE_PREFIX)
)
_LTSPICE_ATTR_CANONICAL: dict[str, str] = {n.lower(): n for n in _LTSPICE_ATTR_NAMES}

# Rotation string -> ERotation enum mapping (shared by move/add handlers)
_ROTATION_MAP: dict[str, ERotation] = {
    "R0": ERotation.R0,
    "R90": ERotation.R90,
    "R180": ERotation.R180,
    "R270": ERotation.R270,
    "M0": ERotation.M0,
    "M90": ERotation.M90,
    "M180": ERotation.M180,
    "M270": ERotation.M270,
}


def _parse_rotation(rotation: str) -> ERotation:
    """Parse a rotation string to ERotation enum. Raises NetlistError if invalid."""
    erot = _ROTATION_MAP.get(rotation)
    if erot is None:
        raise NetlistError(
            f"Invalid rotation '{rotation}'. Valid: {', '.join(_ROTATION_MAP.keys())}"
        )
    return erot


# Matches one ``KEY=VALUE`` token (value may have braces, parens, sign, etc.).
# Validation uses it to accept a value shaped ``"NMOS1 W=10u L=1u"``: one head
# token, then only parameters.
_PARAM_TOKEN_RE = re.compile(r"(\w+)\s*=\s*([^\s=]+)")


# Element classes with exactly two terminals before a value that runs to the
# end of the line: an independent source's spec (``DC 5 AC 1``). LTspice writes
# the nodes from the symbol's pins, so nothing in the value can be read as a
# node. A behavioural source's expression is the same case, taken first.
_SPEC_VALUE_CLASSES = frozenset("VI")

# Element classes whose model name may be followed by an area factor and
# ``off``: ``Q1 c b e 2N3904 2 off``, ``J1 d g s NJF 2``, ``D1 a k 1N4148 2``.
# A MOSFET takes ``off`` but no positional area.
_AREA_CLASSES = frozenset("QJD")
_OFF_CLASSES = frozenset("QJDM")


def _is_spice_number(text: str) -> bool:
    """Whether ``text`` is one finite SPICE number (``2``, ``0.5``, ``10u``).

    Strict: a model name such as ``2N2222`` is a value to LTspice, but here it
    is the name it looks like.
    """
    return is_scaled_number(text) and math.isfinite(parse_spice_value(text))


def _device_tail_ok(element: str, tokens: list) -> bool:
    """Whether ``tokens`` read as ``MODEL [area] [off] [KEY=VALUE ...]`` for ``element``,
    one of the classes in ``_OFF_CLASSES``.

    The model name comes first; after it come at most one area factor (a number
    or a braced expression, for the classes that take one), at most one
    ``off``, and any ``KEY=VALUE`` instance parameters.
    """
    if not tokens or tokens[0].kind not in (TokenKind.BARE, TokenKind.QUOTED):
        return False
    area = off = 0
    for tok in tokens[1:]:
        if tok.kind == TokenKind.KEY_VALUE:
            continue
        if tok.kind == TokenKind.BARE and tok.text.lower() == "off":
            off += 1
        elif element in _AREA_CLASSES and (
            tok.kind == TokenKind.BRACED
            or (tok.kind == TokenKind.BARE and _is_spice_number(tok.text))
        ):
            area += 1
        else:
            return False
    return area <= 1 and off <= 1


def _validate_component_value(reference: str, value: str, element: str) -> None:
    """Reject values that would corrupt the netlist line on write.

    LTspice writes the Value verbatim after the symbol's pins. A space in a
    single-token value (a resistor's ``1 k``, a subcircuit's ``opamp 2``)
    splits it into two tokens, and the netlist reader then takes the first as
    another node. ``element`` is the element class letter the part netlists
    as (``element_class``); the shapes that cannot do that are accepted:

    - SPICE expressions in braces (``{1/(2*pi*RC)}``) and quoted strings;
    - ``[MODEL] KEY=VALUE ...`` parameter lists (split by ``_apply_component_value``);
    - waveform functions (``PULSE(...)``, ``SIN(...) AC 1``), whose parentheses
      protect their spaces;
    - for an independent or behavioural source, any well-formed run of tokens
      (``AC 1``, ``DC 5 AC 1``, ``V=V(a) + V(b)``): the value follows exactly
      two nodes and runs to the end of the line;
    - for a BJT, JFET or diode, a model name followed by an area factor and/or
      ``off`` (``2N3904 2``, ``NPN 8 off``); for a MOSFET, a model name and ``off``.
    """
    if not isinstance(value, str):  # type: ignore[reportUnnecessaryIsInstance]
        # Pydantic should have rejected non-strings already, but guard
        # anyway since this writes to disk verbatim.
        raise NetlistError(
            f"Component '{reference}' value must be a string, got {type(value).__name__}"
        )
    stripped = value.strip()
    if not stripped:
        raise NetlistError(f"Component '{reference}' value must not be empty")
    if "\n" in stripped or "\r" in stripped:
        raise NetlistError(
            f"Component '{reference}' value must be a single line; "
            f"got embedded newline in {value!r}"
        )
    if not any(c.isspace() for c in stripped):
        return
    # Brace-balanced expression or quoted literal — spaces are safe.
    if (stripped.startswith("{") and stripped.endswith("}")) or (
        stripped.startswith('"') and stripped.endswith('"')
    ):
        return
    # A behavioural source's expression may carry anything its own grammar
    # allows (comparison operators, a ternary); it cannot reach a node slot.
    if element == "B":
        return
    try:
        toks = [t for t in tokenize_body(stripped) if t.kind != TokenKind.COMMENT_TRAIL]
    except SpiceLexError:
        toks = []
    well_formed = bool(toks) and all(
        t.kind in (*POSITIONAL_KINDS, TokenKind.KEY_VALUE) for t in toks
    )
    if well_formed and element in _SPEC_VALUE_CLASSES:
        return
    if well_formed and element in _OFF_CLASSES and _device_tail_ok(element, toks):
        return
    # Independent-source waveform spec: ``PULSE(...)``, ``SIN(...)``,
    # ``EXP(...)``, ``PWL(...)``, ``SFFM(...)``, ``TABLE(...)``, ``AM(...)``,
    # ``NOISE(...)``. The keyword is followed by a balanced parenthetical
    # group whose parens protect the embedded whitespace. Optionally
    # preceded by a DC magnitude (``"1 PULSE(...)"``) and followed by an
    # ``AC <mag>`` annotation (``"PULSE(...) AC 1"``).
    # If the body is a sequence of positional tokens (no stray equals signs,
    # no unbalanced quotes), the parens protect their internal whitespace from
    # corrupting the netlist line.
    if (
        toks
        and any(t.kind == TokenKind.PARENED for t in toks)
        and all(t.kind in POSITIONAL_KINDS for t in toks)
    ):
        return
    # ``[MODEL_NAME] KEY=VALUE [KEY=VALUE ...]`` is valid: at most one bare
    # head token (the model name) followed by a non-empty list of KEY=VALUE
    # tokens. The pure-params and head+params forms collapse into one rule.
    if "=" in stripped:
        tokens = stripped.split()
        head_tokens: list[str] = []
        for tok in tokens:
            if "=" in tok:
                break
            head_tokens.append(tok)
        rest = tokens[len(head_tokens) :]
        if (
            len(head_tokens) <= 1
            and rest
            and all(bool(_PARAM_TOKEN_RE.fullmatch(tok)) for tok in rest)
        ):
            return
    if element in _AREA_CLASSES:
        shape = "a model name, then an optional area factor and 'off' (e.g. '2N3904 2 off')"
    elif element == "M":
        shape = "a model name, then optional 'off' and KEY=VALUE parameters"
    else:
        shape = "a single token"
    raise NetlistError(
        f"Component '{reference}' value {value!r} contains whitespace where a "
        f"{element}-class value is {shape}; LTspice would read the extra token "
        "as another node. Wrap SPICE expressions in braces ({...}) or use the "
        "parameter form (e.g. 'NMOS1 W=10u L=1u')."
    )


def _asc_component_value(editor, reference: str) -> str | None:
    """Current Value of an .asc component, or ``None`` if it has no Value slot.

    ``AscEditor.get_component_value`` raises for a component added without a
    Value (e.g. ``add_component`` with no ``value=``); callers distinguish that
    from a missing component via ``editor.components`` membership.
    """
    try:
        return str(editor.get_component_value(reference))
    except Exception:
        return None


def _attributes_of(editor, reference: str) -> dict[str, str]:
    """A part's attributes as its SYMATTR lines hold them, by name. The editor's
    own bookkeeping, under names starting ``_``, is left out."""
    return {
        name: str(text)
        for name, text in editor.get_component(reference).attributes.items()
        if not name.startswith("_")
    }


def _attributes_changed(
    before: Mapping[str, str], after: Mapping[str, str]
) -> dict[str, dict[str, str | None]]:
    """Each attribute that differs between ``before`` and ``after``, with what
    it was and what it is now; None where it was not there or is gone."""
    return {
        name: {"was": before.get(name), "now": after.get(name)}
        for name in sorted(before.keys() | after.keys())
        if before.get(name) != after.get(name)
    }


def _set_or_create_value(editor, reference: str, value: str) -> None:
    """Set the Value slot, creating the ``SYMATTR Value`` line if it has none.

    ``set_component_value`` only updates an EXISTING Value slot; a component
    added without one (``add_component`` with no ``value=``) needs the line
    written directly via ``set_component_attribute`` — symmetric with
    ``add_component(value=)``. Callers pre-validate the value non-empty.
    """
    if _asc_component_value(editor, reference) is None:
        editor.set_component_attribute(reference, "Value", value)
    else:
        editor.set_component_value(reference, value)


def _apply_component_value(editor, reference: str, value: str, element: str) -> None:
    """Set a component's value, splitting trailing ``KEY=VALUE`` tokens off.

    spicelib's ``set_component_value`` writes only the model/value field of
    the element line — it does NOT touch the trailing parameter section.
    Calling it with ``"NMOS1 W=10u L=1u"`` against an existing ``M1 ... NMOS1 W=20u L=1u``
    leaves both sets in place (``... NMOS1 W=10u L=1u W=20u L=1u``), which
    LTspice may parse either way. To DWIM, we split off any ``KEY=VALUE``
    tokens and set each where the part already gives it
    (:func:`_set_instance_params`), keeping the model/value field for
    ``set_component_value``.

    Token-based split via ``spice_lex.tokenize_body``: params are the
    ``KEY_VALUE`` tokens and the head is the rest of the value as written
    (``2N3904 2 off``, ``PULSE(0 1 0 1n 1n 5n 10n)``). The classified-token
    layer knows model-name vs param-name by construction, so adversarial
    cases like ``M1 d g s b "NMOS_lvt" W=10u`` and
    ``R1 n1 n2 {1/(2*pi*RC)}`` route correctly.
    """
    _validate_component_value(reference, value, element)
    # Behavioral sources: the whole value IS an equation whose first token is
    # V=/I=/R=... — not a model name with trailing parameters. The KEY=VALUE
    # split below would route it to set_component_parameters, which the .asc
    # editor writes into SpiceLine while the stale expression stays in Value:
    # the netlisted B-line then carries two expressions ("No such node")
    # behind a success message.
    if element == "B" or "=" not in value:
        _set_or_create_value(editor, reference, value)
        return
    try:
        tokens = tokenize_body(value)
    except SpiceLexError as e:
        raise NetlistError(f"Component '{reference}' value {value!r} failed to parse: {e}") from e
    params: dict[str, str] = {}
    head_parts: list[str] = []
    # The head is the value with its KEY=VALUE spans cut out, each remaining
    # run kept as written: a source spec's ``PULSE(0 1 0 1n 1n 5n 10n)`` is a
    # BARE name and a PARENED group that must stay joined.
    cursor, end = 0, len(value)
    for tok in tokens:
        if tok.kind == TokenKind.COMMENT_TRAIL:
            end = tok.body_offset
            break
        if tok.kind == TokenKind.KEY_VALUE:
            assert tok.key is not None
            assert tok.value is not None
            params[tok.key] = tok.value
            head_parts.append(value[cursor : tok.body_offset].strip())
            cursor = tok.body_end
    head_parts.append(value[cursor:end].strip())
    head = " ".join(part for part in head_parts if part)
    if head:
        _set_or_create_value(editor, reference, head)
    if params:
        _set_instance_params(editor, reference, params)


def _replace_param(text: str, key: str, value: str) -> tuple[str, int]:
    """``text`` with every ``KEY=VALUE`` copy of ``key`` (in any case) set to
    ``value``, keeping its spelling of the key, and how many copies there were."""
    copies = [
        tok for tok in key_value_tokens(text) if (tok.key or "").casefold() == key.casefold()
    ]
    for tok in reversed(copies):
        text = f"{text[: tok.body_offset]}{tok.key}={value}{text[tok.body_end :]}"
    return text, len(copies)


def _set_instance_params(editor, reference: str, params: dict[str, str]) -> None:
    """Set each parameter where the part already gives it, else add it to
    SpiceLine.

    spicelib's ``set_component_parameters`` looks for a key only in SpiceLine
    and SpiceLine2, spelled exactly, so a key the part gives in Value or Value2
    (a Schmitt trigger's ``Vt``, a MOSFET's ``w``) got a second copy in
    SpiceLine. It also rebuilds SpiceLine from its own parse when it adds a
    key, rewriting the rest of the line. So nothing here goes through it: SPICE
    reads a name in any case, and every copy found is set, so copies already on
    the part come to agree. That the part gives a key more than once is the
    sheet's ``duplicate_parameter`` finding to say.
    """
    attributes = editor.get_component(reference).attributes
    for key, value in params.items():
        found = False
        for slot in PARAMETER_ATTRIBUTES:
            text, copies = _replace_param(attributes.get(slot, ""), key, value)
            if copies:
                editor.set_component_attribute(reference, slot, text)
                found = True
        if not found:
            line = attributes.get("SpiceLine", "").strip()
            editor.set_component_attribute(
                reference, "SpiceLine", f"{line} {key}={value}" if line else f"{key}={value}"
            )


# Each editor's symbols, resolved once per request: get_asc_editor hands every
# editor out on a fresh memo, so a symbol redrawn between requests is read
# again, while the many geometry passes one request makes over a sheet cost one
# lookup per distinct symbol rather than a file check per part per pass.
_symbols_by_editor: WeakKeyDictionary[AscEditor, dict[str, SymbolInfo | None]] = (
    WeakKeyDictionary()
)


def symbol_info_for(editor: AscEditor, symbol: str) -> SymbolInfo | None:
    """``symbol`` as ``editor``'s sheet resolves it: beside the sheet, then the libraries."""
    memo = _symbols_by_editor.setdefault(editor, {})
    if symbol not in memo:
        memo[symbol] = get_symbol_info(symbol, editor.asc_file_path)
    return memo[symbol]


def placed_geometry(editor: AscEditor, reference: str) -> dict | None:
    """A placed component's absolute pins and bounding box (see
    ``compute_placed_geometry``); ``None`` when its symbol does not resolve."""
    symbol = editor.components[reference].symbol
    info = symbol_info_for(editor, symbol) if symbol else None
    if info is None:
        return None
    pos, erot = editor.get_component_position(reference)
    return compute_placed_geometry(info, int(pos.X), int(pos.Y), erot.name if erot else "R0")


def element_class(editor: AscEditor, reference: str) -> str:
    """The element letter a placed part netlists as (``sheet.element_letter``),
    read from the editor's part and its symbol."""
    comp = editor.components.get(reference)
    symbol = getattr(comp, "symbol", None)
    info = symbol_info_for(editor, symbol) if symbol else None
    return element_letter(
        (getattr(comp, "attributes", None) or {}).get(INSTANCE_PREFIX),
        info.prefix if info is not None else None,
        reference,
    )


def export_parts(editor: AscEditor, geometry: Iterable[dict]) -> list[tuple[str, str, list[str]]]:
    """Each placed part's reference, element letter and pin names in
    SpiceOrder, from ``collect_component_geometry``'s rows: what
    ``export_nets.exported_pin_nodes`` finds its card and nodes by. A symbol's
    pins are already in SpiceOrder; a part with a pin of none (order 0) is
    left out."""
    return [
        (row["ref"], element_class(editor, row["ref"]), [pin["name"] for pin in row["pins"]])
        for row in geometry
        if all(pin["order"] for pin in row["pins"])
    ]


def collect_component_geometry(editor: AscEditor) -> list[dict]:
    """Collect bounding boxes and pin positions for all components."""
    result: list[dict] = []
    for ref in editor.get_components():
        geo = placed_geometry(editor, ref)
        if geo is not None:
            result.append({"ref": ref, **geo["bounding_box"], "pins": geo["pins"]})
    return result


def _component_pin_coords(editor: AscEditor, reference: str) -> set[tuple[int, int]]:
    """Pin coordinates for a single component, ``set()`` if symbol unknown."""
    if reference not in editor.components:
        return set()
    geo = placed_geometry(editor, reference)
    return set() if geo is None else {(p["x"], p["y"]) for p in geo["pins"]}


def _other_components_pin_coords(editor: AscEditor, exclude_ref: str) -> set[tuple[int, int]]:
    """Union of pin coordinates for every component except ``exclude_ref``.

    Used by remove_component's wire cleanup: a wire ending on another
    component's pin is that component's wire, and stays.
    """
    coords: set[tuple[int, int]] = set()
    for ref in editor.get_components():
        if ref == exclude_ref:
            continue
        coords.update(_component_pin_coords(editor, ref))
    return coords


def net_partition(
    editor: AscEditor,
    extra_segments: list[tuple[int, int, int, int]] | None = None,
    without: str | None = None,
    geometry: Iterable[dict] | None = None,
) -> NetPartition:
    """The sheet's nets, as ``connectivity.partition`` groups its pins, labels and wires.

    The joining rules, and the recordings each comes from, are in
    ``lib/connectivity.py``; this collects what ``editor`` holds and hands it
    over.

    ``extra_segments`` lets the caller include not-yet-committed wire
    segments (e.g. the route ``wire_pins`` is about to add) so checks operate
    on the post-route net layout. Shared by ``trace_nets`` (labels-per-net)
    and ``trace_net`` (full net membership). ``without`` leaves one part's
    pins out: the sheet as it is apart from a part about to be placed or moved.
    ``geometry`` is the parts' geometry when the caller already holds it.
    """
    pins = [
        ((pin["x"], pin["y"]), (entry["ref"], pin["name"]))
        for entry in (collect_component_geometry(editor) if geometry is None else geometry)
        if entry["ref"] != without
        for pin in entry["pins"]
    ]
    labels = [((int(lbl.coord.X), int(lbl.coord.Y)), lbl.text) for lbl in editor.labels]
    segments = [((int(w.V1.X), int(w.V1.Y)), (int(w.V2.X), int(w.V2.Y))) for w in editor.wires]
    segments += [((x1, y1), (x2, y2)) for x1, y1, x2, y2 in extra_segments or ()]
    return partition(pins, labels, segments)


def trace_nets(
    editor: AscEditor,
    extra_segments: list[tuple[int, int, int, int]] | None = None,
) -> dict[tuple[int, int], frozenset[str]]:
    """Map each pin/label/wire coordinate to the labels on its net.

    Thin labels-per-coordinate view over :func:`net_partition`. See it for
    the segment-aware semantics and ``extra_segments`` contract.
    """
    return labels_per_coord(net_partition(editor, extra_segments))


def labels_per_coord(part: NetPartition) -> dict[tuple[int, int], frozenset[str]]:
    """:func:`trace_nets` over a partition the caller already holds."""
    labels_by_root: dict[tuple[int, int], set[str]] = {}
    for coord, texts in part.label_texts.items():
        labels_by_root.setdefault(part.root(coord), set()).update(texts)

    return {
        p: frozenset(labels_by_root.get(part.root(p), set()))
        for members in part.members.values()
        for p in members
    }


def _net_label_at(
    nets: dict[tuple[int, int], frozenset[str]], coord: tuple[int, int]
) -> frozenset[str]:
    """Labels on the net at ``coord``; empty when net is unnamed."""
    return nets.get(coord, frozenset())


def _segment_key(
    a: tuple[int, int], b: tuple[int, int]
) -> tuple[tuple[int, int], tuple[int, int]]:
    """One segment's identity, orientation-independent."""
    return (a, b) if a <= b else (b, a)


def _wire_segment_keys(editor: AscEditor) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    return [
        _segment_key((int(w.V1.X), int(w.V1.Y)), (int(w.V2.X), int(w.V2.Y))) for w in editor.wires
    ]


def _append_wire_segments(
    editor: AscEditor,
    segments: Sequence[tuple[int, int, int, int]],
) -> list[tuple[int, int, int, int]]:
    """Draw the planned segments, skipping any the sheet already carries.

    Returns the ones that were already there. A second copy of a segment draws
    nothing and connects nothing, but a caller cannot tell it from a real second
    wire — and asking to delete "the duplicate" deletes every copy, so the
    request that meant "tidy up" cut the connection instead. Not creating the
    duplicate is what makes that sequence impossible; the alternative, warning
    about it afterwards, is what led the caller into it.
    """
    present = set(_wire_segment_keys(editor))
    already: list[tuple[int, int, int, int]] = []
    for sx1, sy1, sx2, sy2 in segments:
        key = _segment_key((sx1, sy1), (sx2, sy2))
        if key in present:
            already.append((sx1, sy1, sx2, sy2))
            continue
        present.add(key)
        editor.wires.append(Line(Point(sx1, sy1), Point(sx2, sy2)))
    return already


def wiring_profile(editor: AscEditor) -> dict[str, int]:
    """Neutral whole-schematic connectivity counts: are connections drawn as
    wires or carried by net-labels?

    Lets a caller tell a routed schematic from net-label soup (every pin
    tagged with a flag, no wires drawn — simulates identically but reads as a
    wiring list, not a schematic). Facts only, no verdict (see
    ``lib/result_observations.py``): a high ``pins_label_only`` with
    ``wire_segments`` near zero is the soup signature, but net-labels are also
    the right tool for ground/power/genuinely distant nets — the model judges,
    this only counts. A pin counts as wired if a wire passes through it, else
    label-only if a flag sits on it; pins that are floating or directly
    abutting another pin fall into neither. ``pins_total`` is every pin over
    components with resolvable symbol geometry (unknown symbols are skipped) —
    the denominator that makes the two classified counts interpretable.
    """
    segments = [((int(w.V1.X), int(w.V1.Y)), (int(w.V2.X), int(w.V2.Y))) for w in editor.wires]
    on_wire = build_on_wire_predicate(segments)
    label_coords = {(int(lbl.coord.X), int(lbl.coord.Y)) for lbl in editor.labels}

    pins_total = 0
    pins_wired = 0
    pins_label_only = 0
    for entry in collect_component_geometry(editor):
        for p in entry["pins"]:
            pins_total += 1
            coord = (p["x"], p["y"])
            if on_wire(coord):
                pins_wired += 1
            elif coord in label_coords:
                pins_label_only += 1

    return {
        "wire_segments": len(segments),
        "pins_total": pins_total,
        "pins_wired": pins_wired,
        "pins_label_only": pins_label_only,
    }


# Type alias for the union returned by make_editor / _get_editor.
# Schematic-only handlers narrow this to AscEditor after require_asc.
Editor = AscEditor | SpiceEditor


# A sheet coordinate: a route waypoint, or a wire_pins endpoint on a wire or pin.
# No docstring, since the class docstring would be published with every use.
class GridPoint(StrictModel):
    x: int
    y: int


# ---------------------------------------------------------------------------
# Editor factory — extension-based dispatch
# ---------------------------------------------------------------------------


def _refusing_read(exc: BaseException, code: CodeType) -> dict[str, Any]:
    """The locals of the innermost ``code`` frame ``exc`` passed through, or none.

    A block symbol's sheet is read while its parent loads, so the sheet
    spicelib refused may not be the one being opened, and its errors do not
    say which it was. The innermost frame of the method that raised does: its
    ``self`` is the editor reading that sheet.
    """
    names: dict[str, Any] = {}
    for frame, _ in traceback.walk_tb(exc.__traceback__):
        if frame.f_code is code and isinstance(frame.f_locals.get("self"), AscEditor):
            names = dict(frame.f_locals)
    return names


def _unreadable_record(path: Path, exc: NotImplementedError) -> NetlistError:
    """The refusal for a sheet holding a line spicelib's reader has no branch for.

    spicelib raises ``NotImplementedError`` naming the line but neither its
    file nor its number (docs/spicelib_bugs.md, Bug 26). The read that refused
    names the sheet and its codec, and its ``line`` the record.

    Both LTspice builds read a sheet holding a bus tap or an empty line
    (recorded as ``export/bus_tap`` and ``export/blank_line``), so this refuses
    sheets that LTspice opens.
    """
    names = _refusing_read(exc, AscEditor.reset_netlist.__code__)
    record = names.get("line")
    if not isinstance(record, str):
        return NetlistError(
            f"Cannot open {path}: the schematic editor cannot read it: {exc}", show_hint=False
        )
    sheet, encoding = Path(names["self"].asc_file_path), names["self"].encoding
    number = None
    try:
        with open(sheet, encoding=encoding) as text:
            # The reader stops at the first line it cannot read, so the first
            # line equal to the record is that line.
            number = next((n for n, line in enumerate(text, 1) if line == record), None)
    except (OSError, UnicodeError, LookupError):
        pass
    shown = record.removesuffix("\n")
    if not shown.strip():
        what = "is empty"
    elif shown[0].isspace():
        what = f"starts with whitespace ({shown!r})"
    else:
        what = f"is a {shown.split()[0]} record ({shown!r})"
    where = f"line {number}" if number is not None else "a line"
    if sheet != path:
        where = f"{sheet}, a sheet it loads: {where}"
    return NetlistError(
        f"Cannot open {path}: {where} {what}, which the schematic editor does not "
        "read, so the sheet cannot be opened for reading or editing here.",
        show_hint=False,
    )


def _unrecognised_sheet(path: Path, exc: EncodingDetectError) -> NetlistError:
    """The refusal for a sheet spicelib finds no codec for.

    spicelib looks for the ``Version`` line in each codec it tries, and none of
    them reads past a UTF-8 byte order mark, so such a sheet is refused as if
    it had no ``Version`` line (docs/spicelib_bugs.md, Bug 27). Both LTspice
    builds refuse it too, recorded as ``export/micro_utf8_bom``.
    """
    names = _refusing_read(exc, AscEditor.__init__.__code__)
    sheet = Path(names["self"].asc_file_path) if names else path
    try:
        with open(sheet, "rb") as stream:
            mark = refused_sheet_mark(stream.read(4))
    except OSError:
        mark = None
    where = "it" if sheet == path else f"{sheet}, a sheet it loads,"
    if mark is not None:
        return NetlistError(
            f"Cannot open {path}: {where} {refused_sheet_mark_note(mark)}", show_hint=False
        )
    return NetlistError(
        f"Cannot open {path}: {where} does not start with a Version line in any "
        "encoding the schematic editor reads.",
        show_hint=False,
    )


class _AscEditor(AscEditor):
    """spicelib's editor, opening a sheet whose block symbol has no sheet.

    LTspice netlists an instance of a block symbol as a call to a subcircuit
    of the symbol's name, defined by its own sheet, by a library on the sheet,
    or not yet at all. spicelib's loader requires the sheet and refuses to open
    the parent without it (``docs/spicelib_bugs.md``, Bug 22). Here such an
    instance loads with no resolved subcircuit, as spicelib already loads a
    cell symbol with no library, and a sheet that is there opens as one of
    these, so a block nested further down is read the same way.
    """

    def __init__(
        self,
        asc_file: str | Path,
        encoding: str = "autodetect",
        *,
        searched: dict[tuple[str, str], str | None] | None = None,
    ) -> None:
        # Where each sheet not beside its symbol was found, shared with the
        # sheets this one opens: a search walks every folder it is given, and
        # a sheet may place the same block many times.
        self._searched = {} if searched is None else searched
        super().__init__(asc_file, encoding)

    def _get_subcircuit(self, symbol: Any) -> Any:
        if symbol.symbol_type != "BLOCK" or symbol.get_library() is not None:
            return super()._get_subcircuit(symbol)
        sheet = symbol.get_schematic_file()
        if not sheet.exists():
            folder = str(self.asc_file_path.parent)
            if (sheet.name, folder) not in self._searched:
                self._searched[sheet.name, folder] = search_file_in_containers(
                    sheet.name, folder, ".", *self.custom_lib_paths
                )
            sheet = self._searched[sheet.name, folder]
        return None if sheet is None else type(self)(sheet, searched=self._searched)


def make_editor(path: Path) -> Editor:
    """Create an AscEditor or SpiceEditor based on file extension.

    Raises NetlistError if the file is not found, has no Version line the
    schematic editor can find, or holds a line it cannot read, and
    SymbolResolutionError if a file the schematic refers to (a symbol, a
    sub-sheet) is missing.
    """
    try:
        if path.suffix.lower() != ".asc":
            return SpiceEditor(str(path))
        try:
            return _AscEditor(str(path))
        except NotImplementedError as e:
            raise _unreadable_record(path, e) from e
        except EncodingDetectError as e:
            raise _unrecognised_sheet(path, e) from e
    except FileNotFoundError as e:
        if not path.is_file():
            raise NetlistError(f"File not found: {path}") from e
        # The schematic itself opened, so what is missing is something it
        # refers to: a symbol or a model library.
        # Which one it is comes from the file that is there, not from whether
        # the editor's message happened to spell ".asy".
        raise SymbolResolutionError(
            f"Cannot open .asc schematic: {e}\n\n"
            "A file the schematic refers to was not found; for a symbol, "
            "LTspice symbol libraries (.asy files) are required. "
            "Set [schematic] symbol_paths in ltspice-mcp.toml or "
            "LTSPICE_MCP_SYMBOL_PATHS environment variable."
        ) from e


def _get_editor(path: Path, state: SessionState) -> Editor:
    """Get a cached editor instance, creating via make_editor if needed."""
    return state.editors.get(path, lambda p: make_editor(p))


def get_asc_editor(path: Path, state: SessionState) -> AscEditor:
    """Get a cached AscEditor. Caller must have validated require_asc first."""
    editor = _get_editor(path, state)
    if not isinstance(editor, AscEditor):
        raise NetlistError(f"This operation requires an .asc schematic, got '{path.suffix}'. ")
    _symbols_by_editor.pop(editor, None)  # this request resolves symbols afresh
    return editor


def is_asc(path: Path) -> bool:
    return path.suffix.lower() == ".asc"


def require_asc(path: Path) -> None:
    """Raise if path is not an .asc file (for schematic-only operations)."""
    if not is_asc(path):
        raise NetlistError(f"This operation requires an .asc schematic, got '{path.suffix}'. ")


# ---------------------------------------------------------------------------
# Handlers — shared operations (work on .cir/.net and .asc)
# ---------------------------------------------------------------------------


def netlist_card_value(card: SpiceCard) -> str:
    """Display value for one netlist instance card, or ``"<unparseable>"``.

    Rejects a card whose body survived lexing but carries a broken ``k=v``
    remnant; otherwise the typed ``InstanceLine`` view yields the element-class
    display value. Shared by the netlist ``list_components`` path and the
    ``inspect`` component queries so the two agree on what a value is.
    """
    from ltspice_mcp.lib.spice_lex_views import read_instance

    inst = read_instance(card)
    return "<unparseable>" if inst is None else inst.display_value()


_ASC_TEXT_DEFAULT_SIZE = 2
"""LTspice's normal font size — the fallback when a caller doesn't pick one."""

_ASC_TEXT_LINE_PITCH = 16
"""One grid cell — the downward step used to declutter exactly-stacked text.

Multiple directives added without explicit coordinates all default to the same
anchor (16,16) and render on top of each other. Nudging each new one down by a
grid cell until its anchor is free keeps them readable. Only exact-anchor
collisions move — no font-metric guesswork, so no false shifts."""


def _text_record_body(text: str) -> str:
    """``text`` as a TEXT record holds it: each line break as a ``\\n`` escape.

    LTspice's on-disk TEXT record is one physical line; embedded newlines are
    stored as literal "\\n" escapes (LTspice's own multi-line text convention).
    A raw newline would split the record and corrupt the file — the downstream
    symptom is an unrelated-looking "Primitive not supported" parse error. Text
    already in that form is returned unchanged, so a removal matches whether the
    caller gives the line breaks or the escapes the file holds.
    """
    return text.replace("\r\n", "\n").replace("\r", "\n").replace("\n", "\\n")


def _append_asc_text(
    editor: AscEditor,
    text: str,
    text_type: TextTypeEnum,
    x: int | None,
    y: int | None,
    size: int | None,
    *,
    default_x: int,
    default_y: int,
) -> None:
    """Append one TEXT record (directive or comment) to an ``.asc``.

    The single producer of placed schematic text — ``edit_directive``'s two
    kinds and ``edit_schematic``'s ``add_directive`` op all come through
    here, so placement defaulting can't drift between them. The per-site
    ``default_x``/``default_y`` differ deliberately (comments default to the
    sheet origin; directives to (16,16)).

    Auto-declutter: if the resolved anchor is already occupied by another text
    object (the common case when several directives take the (16,16) default),
    step it down one grid cell at a time until it's free, so stacked directives
    don't render on top of each other.
    """
    text = _text_record_body(text)
    px = x if x is not None else default_x
    py = y if y is not None else default_y
    occupied = {(int(d.coord.X), int(d.coord.Y)) for d in editor.directives}
    while (px, py) in occupied:
        py += _ASC_TEXT_LINE_PITCH
    editor.directives.append(
        Text(
            coord=Point(px, py),
            text=text,
            type=text_type,
            size=size if size is not None else _ASC_TEXT_DEFAULT_SIZE,
        )
    )


def _remove_exact_directive(editor, instruction: str) -> bool:
    """Remove one DIRECTIVE-type record matching ``instruction`` by FULL-TEXT
    equality (not substring), returning whether one was removed.

    spicelib's ``remove_instruction`` matches by substring and removes the first
    hit, so an undo of ``.tran 1m`` could instead delete ``.tran 10m`` (or a
    pre-existing duplicate) and silently corrupt the simulation setup. On an
    AscEditor (``directives`` is a list of typed Text records) we filter by exact
    text and drop a single record. Netlist editors (SpiceEditor — no
    ``directives`` list) keep spicelib's substring matcher; the ``.cir``/``.net``
    edit path is out of scope here.
    """
    directives = getattr(editor, "directives", None)
    if directives is None:
        return bool(editor.remove_instruction(instruction))
    body = _text_record_body(instruction)
    for idx, d in enumerate(directives):
        dtype = getattr(d, "type", None)
        dtext = getattr(d, "text", None)
        if dtype == TextTypeEnum.DIRECTIVE and dtext == body:
            del directives[idx]
            return True
    return False


def _remove_directive_or_comment(editor, instruction: str) -> str:
    """Remove a directive or comment matching ``instruction``.

    The default path is an exact, literal match: a DIRECTIVE is removed only when
    its full text equals ``instruction`` (see ``_remove_exact_directive``), and a
    comment only when its body equals it. ``instruction`` is never treated as a
    regex by default — common SPICE directives contain ``(`` and ``)`` (every
    ``.meas``/``.four``/``.print`` referencing ``V(node)``) which would silently
    turn into regex capture groups under any "metachar means regex" heuristic.
    Pass an explicit ``regex:`` prefix to opt in to regex matching.

    Returns a label describing what was removed. Raises NetlistError when
    nothing matched, so the user can't think they cleaned a directive
    that's still in the file.
    """
    matcher = _text_matcher(instruction)
    if isinstance(matcher, re.Pattern):
        directive_hit = bool(editor.remove_Xinstruction(matcher.pattern))
        comment_hit = _strip_matching_comments(editor, matcher)
        if not (directive_hit or comment_hit):
            raise NetlistError(
                f"No directive or comment matched regex {matcher.pattern!r}. It is matched "
                "against each TEXT line's text, where a line break is a \\n escape."
            )
        return "directive(s)/comment(s)"

    directive_hit = _remove_exact_directive(editor, instruction)
    comment_hit = _strip_matching_comments(editor, instruction)
    if not (directive_hit or comment_hit):
        raise NetlistError(
            f"No directive or comment matched {instruction!r} exactly. "
            "Match is literal by default — pass 'regex:<pattern>' for regex "
            "matching, or copy the text of the sheet's TEXT line, after its '!' "
            "(directive) or ';' (comment); a line break may be given as a newline "
            "or as the \\n escape the file holds."
        )
    return "directive"


def _strip_matching_comments(editor, matcher) -> bool:
    """Best-effort removal of TEXT-COMMENT entries whose body matches.

    ``matcher`` is either a literal string (exact match) or a compiled
    regex. ``editor.directives`` only exists on AscEditor — silently
    no-op for netlist-mode editors. Returns True when at least one
    comment was removed so the caller can decide whether the overall
    remove operation hit anything.
    """
    directives = getattr(editor, "directives", None)
    if directives is None:
        return False
    if isinstance(matcher, str):
        matcher = _text_record_body(matcher.strip())
    keep = []
    for entry in directives:
        body = getattr(entry, "text", None)
        entry_kind = getattr(entry, "type", None)
        if entry_kind == TextTypeEnum.COMMENT and isinstance(body, str):
            if isinstance(matcher, str):
                if body.strip() == matcher:
                    continue
            else:
                if matcher.search(body):
                    continue
        keep.append(entry)
    if len(keep) != len(directives):
        directives[:] = keep
        return True
    return False


# ---------------------------------------------------------------------------
# Handlers — schematic-only operations (.asc only)
# ---------------------------------------------------------------------------


def _xy(point: Point) -> tuple[int, int]:
    return int(point.X), int(point.Y)


def _text_matcher(instruction: str) -> "str | re.Pattern[str]":
    """How a directive or comment is named: ``regex:<pattern>`` compiled, to be
    searched in each text, or else the literal text itself."""
    if not instruction.startswith("regex:"):
        return instruction
    pattern = instruction[6:]
    if not pattern.strip():
        raise NetlistError(
            "Empty regex pattern would match every directive; "
            "provide an explicit regex after 'regex:'."
        )
    try:
        return re.compile(pattern)
    except re.error as e:
        raise NetlistError(f"Invalid regex {pattern!r}: {e}") from e


def _texts_matching(editor: AscEditor, wanted: str) -> list[int]:
    """Indices of the directives and comments ``wanted`` names, for ``move``.

    Exact text, compared as a TEXT record holds it (line breaks as ``\\n``,
    surrounding space ignored), or ``regex:<pattern>`` searched in each text.
    """
    matcher = _text_matcher(wanted)
    if isinstance(matcher, re.Pattern):
        return [i for i, d in enumerate(editor.directives) if matcher.search(d.text)]
    body = _text_record_body(matcher.strip())
    return [i for i, d in enumerate(editor.directives) if d.text.strip() == body]


class _MovePlan(NamedTuple):
    """What a ``move`` selects, by index into the editor's record lists."""

    parts: list[str]
    labels: set[int]
    wires: set[int]
    texts: set[int]
    # Each unselected wire with an end to drag: whether its first and its
    # second end moves.
    dragged: dict[int, tuple[bool, bool]]


def _plan_move(editor: AscEditor, op: "OpMove", geometry: dict[str, dict]) -> _MovePlan:
    """Resolve a ``move``'s selection, refusing anything it names that is not there."""
    for ref in op.references:
        if ref not in editor.components:
            raise NetlistError(f"Component '{ref}' not found.")
    labels: set[int] = set()
    for point in op.labels:
        at = [i for i, lbl in enumerate(editor.labels) if _xy(lbl.coord) == (point.x, point.y)]
        if not at:
            raise NetlistError(f"No net label at ({point.x},{point.y}).")
        labels.update(at)
    keys = _wire_segment_keys(editor)
    wires: set[int] = set()
    for seg in op.wires:
        key = _segment_key((seg.x1, seg.y1), (seg.x2, seg.y2))
        found = [i for i, k in enumerate(keys) if k == key]
        if not found:
            raise NetlistError(f"No wire from ({seg.x1},{seg.y1}) to ({seg.x2},{seg.y2}).")
        wires.update(found)
    texts: set[int] = set()
    for wanted in op.texts:
        found = _texts_matching(editor, wanted)
        if not found:
            raise NetlistError(f"No directive or comment matched {wanted!r}.")
        texts.update(found)

    parts = list(dict.fromkeys(op.references))
    region = op.region
    if region is not None:

        def inside(ref: str) -> bool:
            geo = geometry.get(ref)
            if geo is None:
                return region.holds(_xy(editor.get_component_position(ref)[0]))
            return region.holds((geo["x"], geo["y"])) and region.holds(
                (geo["x"] + geo["width"], geo["y"] + geo["height"])
            )

        parts += [ref for ref in editor.get_components() if ref not in parts and inside(ref)]
        labels.update(i for i, lbl in enumerate(editor.labels) if region.holds(_xy(lbl.coord)))
        wires.update(i for i, (a, b) in enumerate(keys) if region.holds(a) and region.holds(b))
        texts.update(i for i, d in enumerate(editor.directives) if region.holds(_xy(d.coord)))
    if not (parts or labels or wires or texts):
        raise NetlistError(
            "move selects nothing: name parts, labels, wires or texts, or give a "
            "region that holds some."
        )

    dragged: dict[int, tuple[bool, bool]] = {}
    if op.drag:
        moving = {
            (pin["x"], pin["y"]) for ref in parts for pin in geometry.get(ref, {}).get("pins", ())
        }
        moving |= {_xy(editor.labels[i].coord) for i in labels}
        moving |= {end for i in wires for end in keys[i]}
        # LTspice joins a point anywhere along a wire, so what lies along one
        # that moves whole is attached to it as what sits on its ends is.
        along = build_on_wire_predicate([keys[i] for i in wires])

        def moves(end: tuple[int, int]) -> bool:
            return end in moving or along(end) or (region is not None and region.holds(end))

        # A label or ground on what moves goes with it rather than being left on nothing.
        labels.update(i for i, lbl in enumerate(editor.labels) if moves(_xy(lbl.coord)))
        for i, w in enumerate(editor.wires):
            ends = (moves(_xy(w.V1)), moves(_xy(w.V2)))
            if i not in wires and any(ends):
                dragged[i] = ends
    return _MovePlan(parts, labels, wires, texts, dragged)


def _apply_move(editor: AscEditor, op: "OpMove") -> dict[str, object]:
    """Apply a ``move``: plan it, refuse a move that would short two named nets,
    then shift every selected record and dragged wire end by ``(dx, dy)``.

    Nothing is changed before the check, so a refused move leaves the sheet as
    it was even in a batch that goes on past a failed op. What the moved sheet
    then shows (a wire left on nothing, a part over another) is the finished
    sheet's findings to say.
    """
    if op.dx == 0 and op.dy == 0:
        raise NetlistError("move needs a non-zero dx or dy.")
    geometry = {entry["ref"]: entry for entry in collect_component_geometry(editor)}
    plan = _plan_move(editor, op, geometry)
    moved_parts = set(plan.parts)

    def shifted(point: tuple[int, int], moves: bool = True) -> tuple[int, int]:
        return (point[0] + op.dx, point[1] + op.dy) if moves else point

    def wire_ends(i: int, w: Any) -> tuple[tuple[int, int], tuple[int, int]]:
        first, second = (True, True) if i in plan.wires else plan.dragged.get(i, (False, False))
        return shifted(_xy(w.V1), first), shifted(_xy(w.V2), second)

    ends = [wire_ends(i, w) for i, w in enumerate(editor.wires)]
    after = partition(
        (
            (
                shifted((pin["x"], pin["y"]), entry["ref"] in moved_parts),
                (entry["ref"], pin["name"]),
            )
            for entry in geometry.values()
            for pin in entry["pins"]
        ),
        (
            (shifted(_xy(lbl.coord), i in plan.labels), lbl.text)
            for i, lbl in enumerate(editor.labels)
        ),
        ends,
    )
    # A net of the moved sheet with two names on it joins them; that is a short
    # unless the sheet as it is already made them one node (names compared as
    # LTspice compares them, nets that share a name or a ground already one).
    names_on: dict[tuple[int, int], set[str]] = defaultdict(set)
    for coord, texts in after.label_texts.items():
        names_on[after.root(coord)].update(named_labels(texts))
    before: NamedNets | None = None
    for names in names_on.values():
        if len(name_keys(names)) < 2:
            continue
        before = before or NamedNets(net_partition(editor, geometry=geometry.values()))
        if len({before.net_named(name) for name in names}) > 1:
            raise NetlistError(
                f"Refused to move: it would join nets that each carry a name {sorted(names)}, "
                "shorting them together. Move it clear of the other net's wiring, or "
                "rename one side if they are meant to be one net."
            )

    for ref in plan.parts:
        position, rotation = editor.get_component_position(ref)
        editor.set_component_position(ref, Point(*shifted(_xy(position))), rotation)
    for i in plan.labels:
        editor.labels[i].coord = Point(*shifted(_xy(editor.labels[i].coord)))
    for w, (a, b) in zip(editor.wires, ends, strict=True):
        w.V1, w.V2 = Point(*a), Point(*b)
    for i in plan.texts:
        editor.directives[i].coord = Point(*shifted(_xy(editor.directives[i].coord)))

    return {
        "moved": {
            "parts": plan.parts,
            "labels": len(plan.labels),
            "wires": len(plan.wires),
            "texts": len(plan.texts),
            "dragged_wire_ends": sum(a + b for a, b in plan.dragged.values()),
        }
    }


def _drop_wires_at(editor: AscEditor, coords: set[tuple[int, int]]) -> int:
    """Remove every wire with an endpoint in ``coords``; return how many were
    dropped. Shared by remove_component's wire cleanup and the remove_wire op."""
    if not coords:
        return 0
    kept = [
        w
        for w in editor.wires
        if (int(w.V1.X), int(w.V1.Y)) not in coords and (int(w.V2.X), int(w.V2.Y)) not in coords
    ]
    dropped = len(editor.wires) - len(kept)
    editor.wires = kept
    return dropped


def _pins_on_crossings(
    editor: AscEditor,
    reference: str,
    pins: Sequence[Mapping[str, Any]],
    doing: str,
    already: Container[tuple[int, int]] = (),
) -> None:
    """Refuse putting ``reference``'s pins at ``pins`` where one would short two named nets.

    A pin on a point where wires cross joins them (the ``pin_at_crossing``
    recording), as a label there does. Anywhere else a pin lands, on a wire's
    interior or end, a label or another part's pin, there is one net already,
    so a crossing is the one way placing or moving a part makes one net of
    two. Two that each carry a name are a short, refused as a wire or a label
    that joined them is. Any other join is taken as meant, and the finished
    sheet says it (the ``joined_at_crossing`` rule).

    The sheet is read without the part itself, and ``already`` is where its
    pins are now: what a pin joins from where it already is, is not this
    op's doing. Called before the part is placed or moved, so a refusal
    leaves the sheet as it was.
    """
    wires = wire_segments_of(editor)
    # Nearly every pin is on one wire or none, and the sheet's nets need not
    # be worked out for those.
    on_wires = [
        (pin, through)
        for pin in pins
        if (pin["x"], pin["y"]) not in already
        and len(through := wires_through((pin["x"], pin["y"]), wires)) > 1
    ]
    if not on_wires:
        return
    nets = NamedNets(net_partition(editor, without=reference))
    for pin, through in on_wires:
        joined = nets.joined_at(seg[:2] for seg in through)
        if len(joined.named) > 1:
            listed = "; ".join(str(sorted(names)) for names in joined.named)
            raise NetlistError(
                f"Refused to {doing}: {reference}.{pin['name']} would land at "
                f"({pin['x']},{pin['y']}), where "
                f"wires of nets that each carry a name cross ({listed}). A pin on a "
                "crossing joins the wires, shorting those nets together. Put the part "
                "where the pin is on the one wire it is meant for."
            )


def _placed_component_data(
    editor: AscEditor,
    reference: str,
    symbol: str,
    x: int,
    y: int,
    rotation: str,
    symbol_info: SymbolInfo,
) -> dict[str, object]:
    """Return the placed symbol's geometry: its pins and its bounding box."""
    geometry = compute_placed_geometry(symbol_info, x, y, rotation)
    return {
        "reference": reference,
        "symbol": symbol,
        "position": {"x": x, "y": y},
        "rotation": rotation,
        "pins": geometry["pins"],
        "bounding_box": geometry["bounding_box"],
    }


def resolve_pin(pin_ref: str, editor: AscEditor) -> tuple[int, int]:
    """Resolve a pin reference ('M1.D', 'X1.2' or 'net:VDD') to absolute (x, y) coordinates.

    The part after the last dot is matched against the symbol's pin names first,
    case-insensitively. Only when no pin carries that name, and it is all
    digits, is it read as the pin's 1-based SpiceOrder — the terminal number a
    netlist uses. Names win because some symbols name their pins '1', '2' in an
    order that need not be their SpiceOrder. The match is ``sheet.find_pin``,
    which ``inspect``'s net trace uses too.

    Raises NetlistError if the reference cannot be resolved.
    """
    if pin_ref.startswith("net:"):
        # Look up a FLAG/net label position in the .asc. A name finds its labels
        # in any case, and ground finds every ground flag, as LTspice joins them.
        net_name = pin_ref[4:]
        key = name_key(net_name)
        matches = [
            (int(lbl.coord.X), int(lbl.coord.Y))
            for lbl in editor.labels
            if name_key(lbl.text) == key
        ]
        if not matches:
            raise NetlistError(
                f"Net label '{net_name}' not found in schematic. Add it with the "
                "add_net_label op of edit_schematic first."
            )
        if len(matches) > 1:
            coords = ", ".join(f"({x},{y})" for x, y in matches)
            x, y = matches[0]
            raise NetlistError(
                f"Multiple '{net_name}' labels found at: {coords}. Give the one you "
                f"mean by its point, e.g. {{'x': {x}, 'y': {y}}}, or a component pin "
                "(Ref.Pin) on its net."
            )
        return matches[0]

    # Component.Pin format
    if "." not in pin_ref:
        raise NetlistError(
            f"Invalid pin reference '{pin_ref}'. Use 'Reference.Pin', the pin by "
            "name or 1-based SpiceOrder (e.g., 'M1.D', 'X1.2'), or 'net:name' "
            "(e.g., 'net:VDD')."
        )

    ref, pin_name = pin_ref.rsplit(".", 1)
    component_refs = editor.get_components()
    if ref not in component_refs:
        raise NetlistError(
            f"Component '{ref}' not found. Available: {', '.join(sorted(component_refs))}"
        )

    symbol = editor.components[ref].symbol
    geometry = placed_geometry(editor, ref)
    pins = [
        PlacedPin(pin["name"], pin["order"], pin["x"], pin["y"], pin["dir"])
        for pin in (geometry["pins"] if geometry is not None else ())
    ]
    return find_pin(pins, ref, symbol or "", pin_name, "not found" if geometry is None else "")


def _add_net_label_checks(editor: AscEditor, net: str, x: int, y: int) -> None:
    """Refuse placing net label ``net`` at ``(x, y)`` where it would short two named nets.

    That happens when the network here is named and ``net`` already names a
    different one, or when the label, ground included, lands where two named
    networks cross and joins both. Anything else it makes is the finished
    sheet's to say, once the batch is done: a second name for a net
    (``net_with_two_names``), two nets joined at a crossing
    (``joined_at_crossing``), a label still on nothing (``dangling_label``).
    """
    # A FLAG anywhere along a wire, its interior included, joins that wire's
    # net, and one at a crossing joins both (see connectivity.partition for the
    # export record). It is one node with whatever sits on its point, and with
    # every net already of its name; ground is no name (NamedNets.joined_at).
    through = wires_through((x, y), wire_segments_of(editor))
    grounding = is_ground(net)
    joined = NamedNets(net_partition(editor)).joined_at(
        (seg[:2] for seg in through), at=(x, y), name=None if grounding else net
    )
    if len(joined.named) > 1:
        listed = "; ".join(f"{sorted(names)}" for names in joined.named)
        if grounding:
            raise NetlistError(
                f"Refused to add ground at ({x},{y}): wires of nets that each carry a "
                f"name cross there ({listed}). A ground flag on a crossing joins the "
                "wires, shorting those nets together and to ground. Put the flag on a "
                "point of the one wire it is meant for."
            )
        raise NetlistError(
            f"Refused to add net '{net}' at ({x},{y}): the label would join nets "
            f"that each carry a name ({listed}), shorting them together. To join "
            "them on purpose, give both the same name: remove one side's labels "
            "with remove_net_label and label it with the other's name."
        )


class _ConnectPlan(NamedTuple):
    """Validated connect route ready to commit to the editor."""

    x1: int
    y1: int
    x2: int
    y2: int
    points: list[tuple[int, int]]
    segments: list[tuple[int, int, int, int]]
    # Every point other than the two endpoints where the route joins existing
    # wiring, and each endpoint that ends on a wire's interior (a T-junction).
    junctions: list[dict[str, object]]


def wire_segments_of(editor: AscEditor) -> list[tuple[int, int, int, int]]:
    """Every wire as a flat ``(x1, y1, x2, y2)`` integer tuple."""
    return [(int(w.V1.X), int(w.V1.Y), int(w.V2.X), int(w.V2.Y)) for w in editor.wires]


def wires_of_one_net(
    coord: tuple[int, int],
    segments: Sequence[tuple[int, int, int, int]],
    net_of: "Callable[[tuple[int, int]], tuple[int, int]]",
    remedy: str,
) -> list[tuple[int, int, int, int]]:
    """:func:`wires_through` for a point that must name one net.

    Refuses, naming the wires and ending with ``remedy``, when the point is
    where wires of separate nets cross: LTspice leaves a plain crossing unjoined
    (``tests/fixtures/t_junctions/crossing_wires``), so it is on neither net
    alone. An empty list means the point touches no wire.
    """
    through = wires_through(coord, segments)
    if len({net_of(s[:2]) for s in through}) > 1:
        listed = "; ".join(segment_text(s) for s in through)
        raise NetlistError(
            f"({coord[0]},{coord[1]}) is where wires on separate nets cross ({listed}); "
            f"LTspice leaves a plain crossing unjoined. {remedy}"
        )
    return through


def _endpoint_name(endpoint: "str | GridPoint") -> str:
    """A route endpoint as a message names it: as given, or as ``(x,y)``."""
    return endpoint if isinstance(endpoint, str) else f"({endpoint.x},{endpoint.y})"


def _resolve_route_endpoint(
    endpoint: "str | GridPoint",
    editor: AscEditor,
    existing_wires: list[tuple[int, int, int, int]],
    node_coords: "Container[tuple[int, int]]",
    net_of: "Callable[[tuple[int, int]], tuple[int, int]]",
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Resolve one ``wire_pins`` endpoint to its coordinate and an anchor.

    The anchor is a pin, label or wire end on the endpoint's net before the
    route is drawn, for looking the net up. A pin or label reference resolves
    through :func:`resolve_pin` and is its own anchor. A coordinate must already
    touch the circuit: at a pin or a net label (``node_coords``), its own
    anchor, or anywhere along a wire. An end there connects to it, and one on a
    wire's interior is a T-junction LTspice joins without the wire being split.
    A label's point is how a route reaches a label that stands on nothing, one
    of several labels of one name, or a ground, none of which ``net:NAME`` can
    name. A coordinate where wires of separate nets cross is refused, because a
    wire ending there would join all of them.
    """
    if isinstance(endpoint, str):
        coord = resolve_pin(endpoint, editor)
        return coord, coord
    coord = (endpoint.x, endpoint.y)
    if coord in node_coords:
        return coord, coord
    through = wires_of_one_net(
        coord,
        existing_wires,
        net_of,
        "A wire ending there would join them all; end the route on a point of "
        "just the wire you mean.",
    )
    if not through:
        raise NetlistError(
            f"{_endpoint_name(endpoint)} touches no wire, component pin or net "
            "label. A coordinate endpoint must land on one; to end a wire at a free "
            "point, place a net label there first, and to turn a corner there, give "
            "it as a waypoint instead."
        )
    return coord, through[0][:2]


def _plan_connect_route(
    editor: AscEditor,
    from_pin: "str | GridPoint",
    to_pin: "str | GridPoint",
    waypoints: list[GridPoint],
) -> _ConnectPlan:
    """Resolve, route, and validate a wire path between two endpoints.

    Returns a :class:`_ConnectPlan` whose ``segments`` are ready to append
    to ``editor.wires`` directly. Raises ``NetlistError`` for any
    validation failure. What was asked is checked here: an end that does not
    resolve, a route of no length, two named nets joined, a wire LTspice
    would drop. What drawing the route would do to the rest of the sheet (a
    diagonal segment, a pin passed over, a wire run along, another net's
    wiring touched) is ``routing.judge``'s, and so are the junctions it makes.

    Backs the ``wire_pins`` op, so a route the planner refuses is never
    written by any caller.
    """
    component_geo = collect_component_geometry(editor)
    existing_wires = wire_segments_of(editor)
    part_before = net_partition(editor)
    net_of = label_folded_nets(part_before)
    node_coords = part_before.pin_owners.keys() | part_before.label_texts.keys()
    from_name, to_name = _endpoint_name(from_pin), _endpoint_name(to_pin)

    (x1, y1), from_anchor = _resolve_route_endpoint(
        from_pin, editor, existing_wires, node_coords, net_of
    )
    (x2, y2), to_anchor = _resolve_route_endpoint(
        to_pin, editor, existing_wires, node_coords, net_of
    )

    if (x1, y1) == (x2, y2) and not waypoints:
        raise NetlistError(
            f"Cannot connect {from_name} to {to_name}: "
            f"both endpoints resolve to the same coordinate ({x1},{y1})."
        )

    raw_points: list[tuple[int, int]] = [(x1, y1)]
    raw_points.extend((wp.x, wp.y) for wp in waypoints)
    raw_points.append((x2, y2))
    points: list[tuple[int, int]] = [raw_points[0]]
    for pt in raw_points[1:]:
        if pt != points[-1]:
            points.append(pt)

    segments: list[tuple[int, int, int, int]] = []
    for i in range(len(points) - 1):
        px1, py1 = points[i]
        px2, py2 = points[i + 1]
        if px1 != px2 or py1 != py2:
            segments.append((px1, py1, px2, py2))

    if not segments:
        raise NetlistError(
            f"Cannot connect {from_name} to {to_name}: "
            "the requested route has zero length after deduplicating waypoints."
        )

    # Net-label conflict — checked first because it's a "wrong intent"
    # error: rejecting it gives the user a clearer signal than a route
    # geometry complaint. An end given as ``net:NAME`` is on the net of that
    # label and is checked like a pin: naming the net one end is on does not
    # make joining it to another named net any less a short. A coordinate
    # endpoint on a wire's interior reads the labels of that wire's net.
    # Two-phase check:
    #   1) BEFORE state — endpoints resolve to two different already-named
    #      nets (the standard short).
    #   2) AFTER state — proposed route drags a mid-segment label into
    #      the union, merging an additional named net.
    # Names are compared as LTspice compares them (name_key): a net named
    # VOUT and one named Vout are one node already.
    nets_before = labels_per_coord(part_before)
    from_labels_before = named_labels(_net_label_at(nets_before, from_anchor))
    to_labels_before = named_labels(_net_label_at(nets_before, to_anchor))
    from_keys, to_keys = name_keys(from_labels_before), name_keys(to_labels_before)
    if from_keys and to_keys and from_keys.isdisjoint(to_keys):
        raise NetlistError(
            f"Refused to connect {from_name} to {to_name}: "
            f"Net-label conflict — {from_name} is on net "
            f"{sorted(from_labels_before)} and {to_name} is on net "
            f"{sorted(to_labels_before)}. Connecting them would short "
            f"the two named nets. To join them on purpose, give both the "
            f"same name: remove one side's labels with remove_net_label "
            f"and label it with the other's name; no wire is needed."
        )
    nets_after = trace_nets(editor, extra_segments=segments)
    from_labels_after = named_labels(_net_label_at(nets_after, (x1, y1)))
    to_labels_after = named_labels(_net_label_at(nets_after, (x2, y2)))
    unioned = from_labels_after | to_labels_after
    if len(name_keys(unioned)) >= 2:
        # Some labels seen post-route weren't there pre-route on
        # either endpoint — that's the mid-segment case.
        known = from_keys | to_keys
        new_labels = {label for label in unioned if name_key(label) not in known}
        if new_labels:
            raise NetlistError(
                f"Refused to connect {from_name} to {to_name}: "
                f"Net-label conflict — the proposed route would "
                f"merge named nets {sorted(unioned)} (a label on a "
                f"mid-segment of the wire path adds "
                f"{sorted(new_labels)} to the merged net). Reroute "
                "to avoid the labelled wire."
            )

    # Same-instance self-loop — refused first because, like the net-label
    # conflict above, it's a "wrong intent" error: a wire tying two pins of one
    # component (directly, or through a collinear waypoint that merges back into
    # a straight wire) with nothing on it is dropped by LTspice at netlist time
    # (see same_instance_dropped_segments), so reporting the tie as connected
    # would be a lie. The fix is a route change, so surface it before route
    # geometry.
    dropped = same_instance_dropped_segments(
        part_before.pin_owners, segments, part_before.label_texts.keys()
    )
    if dropped:
        ref = dropped[0]["ref"]
        pin_a, pin_b = dropped[0]["pins"]
        raise NetlistError(
            f"Refused to connect {from_name} to {to_name}: this is a same-instance "
            f"wire — it ties two pins of the same component {ref} ({ref}.{pin_a} "
            f"and {ref}.{pin_b}). LTspice drops such a wire from the exported "
            f"netlist, so the tie would have no electrical effect. To tie two pins "
            f"of one component: route the wire so it bends out of line with the two "
            f"pins (a waypoint that stays collinear with them is merged back into a "
            f"straight wire and still dropped; the bend must leave that line), or "
            f"drop the wire and give both pins the same net label via the "
            f"edit_schematic add_net_label op (same-name labels merge into one "
            f"net)."
        )

    # Everything else said of a route is routing.judge's, which reads the sheet
    # as plain data: the parts with their pins, the wires, and the partition
    # worked out above. A route across a part is not judged at all here: the
    # finished sheet's own rule says it.
    view = SheetView(
        parts=tuple(
            Part(
                ref=part["ref"],
                pins=tuple((pin["name"], pin["x"], pin["y"]) for pin in part["pins"]),
            )
            for part in component_geo
        ),
        wires=tuple(existing_wires),
    )
    route = Route(
        points=tuple(points),
        segments=tuple(segments),
        ends=((x1, y1), (x2, y2)),
        anchors=(from_anchor, to_anchor),
        end_names=(from_name, to_name),
        ends_by_coordinate=(not isinstance(from_pin, str), not isinstance(to_pin, str)),
    )
    judged = judge(view, part_before, route)
    if judged.refusals:
        error_lines = [f"Refused to connect {from_name} to {to_name}:"]
        error_lines.extend(f"  {found.detail}" for found in judged.refusals)
        error_lines.append("\nFix the route with different waypoints to avoid these issues.")
        raise NetlistError("\n".join(error_lines))

    return _ConnectPlan(x1, y1, x2, y2, points, segments, judged.junctions)


# ---------------------------------------------------------------------------
# New tools: schematic seeding, netlist validation, .step querying, diff
# ---------------------------------------------------------------------------


def blank_sheet(width: int = 880, height: int = 680) -> str:
    """The .asc body of an empty sheet (880x680 = LTspice's default extent)."""
    return f"Version 4\nSHEET 1 {width} {height}\n"


# ---------------------------------------------------------------------------
# Batch-transaction op — apply many edits to one .asc atomically.
# ---------------------------------------------------------------------------


_RotationLiteral = Literal["R0", "R90", "R180", "R270", "M0", "M90", "M180", "M270"]

# Shared op-field descriptions. The op models are published in three profiles
# (full, agentic and the consolidated edit_schematic), so each string is paid
# for three times over — keep them to the fact the caller cannot infer. The
# coordinate convention is stated once on the ``ops`` field instead of on the
# dozen x/y pairs below.
_ROTATION_DESCRIPTION = (
    "'R<deg>' rotates clockwise by that many degrees, 'M<deg>' rotates the same "
    "way and then mirrors horizontally; pins move with the body."
)
_REFERENCE_DESCRIPTION = "Reference designator of an existing component, e.g. 'R1', 'M3'."
COORDINATE_DESCRIPTION = (
    "All x/y are LTspice grid units, with x increasing to the right and y increasing downward."
)


class OpAddComponent(StrictModel):
    """Place a new component from its symbol at a coordinate."""

    op: Literal["add_component"]
    reference: str = Field(
        description="Reference designator to give the new part, e.g. 'R1', 'M3'."
    )
    symbol: str = Field(
        description=(
            "Symbol name without the .asy extension, e.g. 'res', 'nmos4'; it "
            "must be one the active symbol libraries resolve."
        )
    )
    x: int
    y: int
    rotation: _RotationLiteral = Field(default="R0", description=_ROTATION_DESCRIPTION)
    value: str | None = Field(
        default=None, description="Value or model name, e.g. '10k', '1u', 'BSS123'."
    )
    attributes: dict[str, str] | None = Field(
        default=None,
        description="Further symbol attributes by name, e.g. SpiceLine, SpiceModel.",
    )


class OpSetComponentValue(StrictModel):
    """Set an existing component's primary value."""

    op: Literal["set_component_value"]
    reference: str = Field(description=_REFERENCE_DESCRIPTION)
    value: str = Field(
        description=(
            "New value or model name, e.g. '10k', '1u', 'BSS123'. A KEY=VALUE is set "
            "where the part already gives that key, else in SpiceLine."
        )
    )


class OpSetComponentAttribute(StrictModel):
    """Set one named symbol attribute on an existing component."""

    op: Literal["set_component_attribute"]
    reference: str = Field(description=_REFERENCE_DESCRIPTION)
    attribute: str = Field(
        description="Attribute name, e.g. Value, SpiceLine, SpiceModel, Prefix."
    )
    value: str = Field(description="New value for that attribute.")


class OpRemoveComponent(StrictModel):
    """Delete a component, optionally taking its dangling wires with it."""

    op: Literal["remove_component"]
    reference: str = Field(description=_REFERENCE_DESCRIPTION)
    cleanup_wires: bool = Field(
        default=False,
        description=(
            "Also delete wires left dangling at the removed component's pins; a "
            "later add_component does not restore them, and there is no undo."
        ),
    )


class OpMoveComponent(StrictModel):
    """Move an existing component, optionally re-rotating it."""

    op: Literal["move_component"]
    reference: str = Field(description=_REFERENCE_DESCRIPTION)
    x: int
    y: int
    rotation: _RotationLiteral | None = Field(
        default=None, description=f"Omit to keep the current rotation; {_ROTATION_DESCRIPTION}"
    )


class SheetRegion(StrictModel):
    """Two opposite corners; the edges are inside."""

    x1: int
    y1: int
    x2: int
    y2: int

    def holds(self, point: tuple[int, int]) -> bool:
        return BBox(self.x1, self.y1, self.x2, self.y2).contains_point(*point)


class WireEnds(StrictModel):
    """A wire's two ends, in either order."""

    x1: int
    y1: int
    x2: int
    y2: int


class OpMove(StrictModel):
    """Shift parts, wires, labels and texts by one offset."""

    op: Literal["move"]
    dx: int = Field(description="Sheet units (16 is one grid step); positive is right.")
    dy: int = Field(description="Positive is down.")
    references: list[str] = Field(default_factory=list, description="Parts, by reference.")
    labels: list[GridPoint] = Field(
        default_factory=list, description="Net labels and grounds, by point."
    )
    wires: list[WireEnds] = Field(default_factory=list)
    texts: list[str] = Field(
        default_factory=list,
        description="Directives and comments: exact text or 'regex:'; every match moves.",
    )
    region: SheetRegion | None = Field(
        default=None,
        description="Also what lies inside: a part's box, a wire's ends, a label, a text's anchor.",
    )
    drag: bool = Field(
        default=False,
        description="Also move what is attached to what moves (or inside region): other wires' ends and labels on it.",
    )


class OpAddNetLabel(StrictModel):
    """Name a net by placing a label, either at a pin or at a coordinate."""

    op: Literal["add_net_label"]
    net: str = Field(description="Net name the label declares, e.g. 'VDD', 'out'.")
    pin: str | None = Field(
        default=None,
        description="Place at this pin, e.g. 'M1.D' or 'X1.2'; give this or x/y, not both.",
    )
    x: int | None = None
    y: int | None = None


class OpWirePins(StrictModel):
    """Draw an orthogonal wire between two endpoints, refusing a diagonal run, a
    pin collision, or an overlapping wire junction."""

    op: Literal["wire_pins"]
    from_pin: str | GridPoint = Field(
        description=(
            "'REF.PIN' by name, else 1-based SpiceOrder ('M1.D', 'X1.2'), "
            "'net:NAME' for a label, or {x, y} on a wire, pin or label; a wire's "
            "interior makes a T-junction."
        )
    )
    to_pin: str | GridPoint = Field(description="Same forms as from_pin.")
    waypoints: list[GridPoint] = Field(
        default_factory=list,
        description=(
            "Corners, in order; with none, the ends must share an x or a y. "
            "One on another net's wire is refused; end there to join it."
        ),
    )


class OpRemoveNetLabel(StrictModel):
    """Delete a net label, addressed by its pin or its coordinate."""

    op: Literal["remove_net_label"]
    pin: str | None = Field(
        default=None, description="The pin the label sits on; give this or x/y, not both."
    )
    x: int | None = None
    y: int | None = None


class OpRemoveWire(StrictModel):
    """Delete wires, addressed either as one exact segment or as every segment
    incident on a point. Prefer the segment form to undo one wire_pins call."""

    op: Literal["remove_wire"]
    x1: int | None = Field(
        default=None,
        description=(
            "Segment form: with y1/x2/y2, removes only this segment, in either "
            "direction; byte-identical duplicates all go at once and 'removed' "
            "counts them."
        ),
    )
    y1: int | None = None
    x2: int | None = None
    y2: int | None = None
    pin: str | None = Field(
        default=None,
        description=(
            "Incident-point form: removes every segment touching this pin, including "
            "wires belonging to other connections at a shared node."
        ),
    )
    x: int | None = Field(default=None, description="Incident-point form, as a coordinate.")
    y: int | None = None


class OpAddDirective(StrictModel):
    """Add a SPICE directive or a comment to the sheet."""

    op: Literal["add_directive"]
    instruction: str = Field(
        description="Directive or comment text, e.g. '.tran 1m' or '.model NMOS ...'."
    )
    kind: Literal["directive", "comment"] = Field(
        default="directive",
        description="'directive' is simulated; 'comment' is annotation LTspice ignores.",
    )
    x: int | None = Field(
        default=None,
        description=(
            "Omit x/y to use the default anchor, stepped down past any text already "
            "there so directives do not stack on top of each other."
        ),
    )
    y: int | None = None
    size: int = Field(default=2, description="LTspice text size index.")


class OpRemoveDirective(StrictModel):
    """Delete a directive or comment by its text."""

    op: Literal["remove_directive"]
    instruction: str = Field(
        description=(
            "Directive or comment text to remove, matched literally by default; "
            "prefix with 'regex:' to match by pattern."
        )
    )


class PlotPaneSpec(StrictModel):
    """One waveform pane."""

    traces: list[str] = Field(
        min_length=1,
        description="Expressions as typed in Add Traces, no spaces: 'V(out)', 'V(in)-V(out)'.",
    )
    x_scale: XScale | None = Field(
        default=None, description="Default: log for ac and noise, else linear."
    )
    y_scale: YScale | None = Field(
        default=None, description="Left Y axis. Default: db for ac, else linear."
    )


class OpSetPlotPanes(StrictModel):
    """Set one analysis's waveform panes in the .plt beside the sheet."""

    op: Literal["set_plot_panes"]
    analysis: PlotAnalysis
    panes: list[PlotPaneSpec] = Field(
        description="Top to bottom; replaces the analysis's panes, and [] removes them."
    )


SchematicOp = (
    OpAddComponent
    | OpSetComponentValue
    | OpSetComponentAttribute
    | OpRemoveComponent
    | OpMoveComponent
    | OpMove
    | OpAddNetLabel
    | OpRemoveNetLabel
    | OpRemoveWire
    | OpWirePins
    | OpAddDirective
    | OpRemoveDirective
    | OpSetPlotPanes
)


def files_written_beside(sheet: Path, ops: Sequence[object]) -> tuple[Path, ...]:
    """The files other than ``sheet`` a batch of ``ops`` may write, for ``edit_guard``."""
    if any(isinstance(op, OpSetPlotPanes) for op in ops):
        return (plot_settings_path(sheet),)
    return ()


@dataclass
class SheetPlotSettings:
    """The plot settings file beside a sheet, as one op batch changes it.

    Nothing is read until a ``set_plot_panes`` loads it, so a batch without one
    neither reads nor writes the file. Once loaded, ``original`` is the file's
    bytes (None when there was none), so a commit that fails after replacing
    the file can put them back, and ``settings`` takes each ``set_plot_panes``
    in turn.
    """

    path: Path
    loaded: bool = False
    original: bytes | None = None
    settings: PlotSettings = field(default_factory=PlotSettings)

    @classmethod
    def beside(cls, sheet: Path) -> "SheetPlotSettings":
        return cls(path=plot_settings_path(sheet))

    def load(self) -> None:
        """Read the file, once; the batch holds the edit guard of both files."""
        if self.loaded:
            return
        try:
            original: bytes | None = self.path.read_bytes()
        except FileNotFoundError:
            original = None
        except OSError as exc:
            raise NetlistError(f"cannot read {self.path.name}: {exc}") from exc
        if original is not None:
            self.settings = read_plot_settings(original)
        self.original = original
        self.loaded = True

    def contents(self) -> bytes | None:
        """The bytes to commit, or None when no section is left and the file goes."""
        return write_plot_settings(self.settings) if self.settings.sections else None

    @property
    def changed(self) -> bool:
        """Whether committing the batch rewrites or removes the file."""
        return self.loaded and self.contents() != self.original


def _resolve_op_xy(
    op: "OpAddNetLabel | OpRemoveNetLabel | OpRemoveWire", editor: AscEditor
) -> tuple[int, int]:
    """Resolve an op's ``pin`` | ``x,y`` point locator to a coordinate."""
    if op.pin is not None:
        return resolve_pin(op.pin, editor)
    if op.x is not None and op.y is not None:
        return op.x, op.y
    raise NetlistError(f"{op.op} needs either pin or both x and y.")


#: The ops that leave every pin where it was and joined as it was: a part's
#: value or attributes, directives and comments, the plot settings beside the
#: sheet. A reply lists pins for the parts the other ops name, and none for
#: these; an op not listed is taken to rewire what it names.
KEEPS_WIRING = frozenset(
    {
        "set_component_value",
        "set_component_attribute",
        "add_directive",
        "remove_directive",
        "set_plot_panes",
    }
)

# The keys of an op's result that report what applying it found on the sheet:
# the segments a route found already drawn, the junctions it made, how much a
# removal took. A value or attribute change reports what it changed. The rest
# of a result restates the op or carries geometry a surface reports through
# views of its own. A route's
# ``wire_count`` is not one: ``already_present`` names the segments it did not
# draw, and relaying the count would put a line per route on every build. Nor
# is ``remove_directive``'s ``removed``, which names a kind, not an amount. An
# op missing here reports nothing beyond its arguments.
OP_RESULT_FACTS: dict[str, tuple[str, ...]] = {
    "set_component_value": ("changed",),
    "set_component_attribute": ("changed",),
    "wire_pins": ("already_present", "junctions"),
    "remove_wire": ("removed",),
    "remove_net_label": ("removed",),
    "remove_component": ("deleted_wires",),
    "move": ("moved",),
    "set_plot_panes": ("plot_settings", "replaced_panes"),
}


def _set_plot_panes(op: OpSetPlotPanes, plot: SheetPlotSettings) -> dict[str, object]:
    """Replace one analysis's panes in ``plot``; report the panes it had.

    ``replaced_panes`` is in this op's own form, so passing it back as
    ``panes`` restores them. The op has no grid argument; new panes keep the
    replaced panes' grid (``inherit_grid``).
    """
    plot.load()
    panes = [
        PlotPane(
            traces=tuple(spec.traces), scales=scales_of(op.analysis, spec.x_scale, spec.y_scale)
        )
        for spec in op.panes
    ]
    before = plot.settings.section(SECTION_NAMES[op.analysis])
    plot.settings = with_panes(plot.settings, op.analysis, inherit_grid(before, panes))
    replaced = pane_specs(before.panes if before is not None else ())
    return {"op": op.op, "plot_settings": str(plot.path), "replaced_panes": replaced}


def apply_op_inplace(
    editor: AscEditor,
    op: SchematicOp,
    asc_path: Path,
    plot: SheetPlotSettings | None = None,
) -> dict[str, object]:
    """Apply one schematic op against ``editor`` in place, return its result.

    Skips the load / save / lock dance: the batch runner's caller holds the
    edit guard and saves once, after every op in the batch has applied.
    ``set_plot_panes`` changes ``plot`` instead, which the caller commits
    beside the sheet; a batch holding one must pass it.

    Raises ``NetlistError`` on any per-op validation failure; the caller
    decides whether to abort or continue based on ``stop_on_error``.
    """
    if isinstance(op, OpSetPlotPanes):
        if plot is None:
            raise NetlistError("set_plot_panes needs the sheet's plot settings to write into.")
        return _set_plot_panes(op, plot)
    if isinstance(op, OpAddComponent):
        symbol_info = symbol_info_for(editor, op.symbol)
        if symbol_info is None:
            raise NetlistError(
                f"Symbol '{op.symbol}' not found beside the schematic or in any "
                "configured symbol library."
            )
        if op.reference in editor.components:
            raise NetlistError(f"Component '{op.reference}' already exists in {asc_path.name}.")
        erot = _parse_rotation(op.rotation)
        _pins_on_crossings(
            editor,
            op.reference,
            compute_placed_geometry(symbol_info, op.x, op.y, op.rotation)["pins"],
            f"place {op.reference}",
        )
        create_component(
            editor,
            op.reference,
            op.symbol,
            op.x,
            op.y,
            erot,
            value=op.value,
            attributes=op.attributes,
        )
        placed = _placed_component_data(
            editor,
            op.reference,
            op.symbol,
            op.x,
            op.y,
            op.rotation,
            symbol_info,
        )
        return {"op": "add_component", **placed}

    if isinstance(op, OpSetComponentValue):
        if op.reference not in editor.components:
            raise NetlistError(f"Component '{op.reference}' not found.")
        before = _attributes_of(editor, op.reference)
        _apply_component_value(editor, op.reference, op.value, element_class(editor, op.reference))
        result: dict[str, object] = {
            "op": "set_component_value",
            "reference": op.reference,
            "value": op.value,
        }
        if changed := _attributes_changed(before, _attributes_of(editor, op.reference)):
            result["changed"] = changed
        return result

    if isinstance(op, OpSetComponentAttribute):
        _reject_unknown_attr(op.attribute)
        if op.reference not in editor.components:
            raise NetlistError(f"Component '{op.reference}' not found.")
        before = _attributes_of(editor, op.reference)
        if not op.value.strip():
            # Empty value = clear. LTspice's format has no "empty value"
            # representation (a 2-token SYMATTR line is unreadable), so the
            # line is removed instead.
            _require_clearable_attr(op.reference, op.attribute)
            editor.get_component(op.reference).attributes.pop(op.attribute, None)
        else:
            editor.set_component_attribute(
                op.reference,
                op.attribute,
                _attribute_value(op.reference, op.attribute, op.value),
            )
        attr_result: dict[str, object] = {
            "op": "set_component_attribute",
            "reference": op.reference,
            "attribute": op.attribute,
        }
        if changed := _attributes_changed(before, _attributes_of(editor, op.reference)):
            attr_result["changed"] = changed
        return attr_result

    if isinstance(op, OpRemoveComponent):
        if op.reference not in editor.components:
            raise NetlistError(f"Component '{op.reference}' not found.")
        target_only = _component_pin_coords(editor, op.reference) - _other_components_pin_coords(
            editor, op.reference
        )
        editor.remove_component(op.reference)
        rm_result: dict[str, object] = {"op": "remove_component", "reference": op.reference}
        # A wire it leaves on nothing is the finished sheet's to say
        # (dangling_wire_end), so a later op of the batch that removes or
        # reconnects it leaves nothing said.
        if op.cleanup_wires and target_only:
            rm_result["deleted_wires"] = _drop_wires_at(editor, target_only)
        return rm_result

    if isinstance(op, OpMoveComponent):
        if op.reference not in editor.components:
            raise NetlistError(f"Component '{op.reference}' not found.")
        new_rot = (
            _parse_rotation(op.rotation)
            if op.rotation is not None
            else editor.get_component_position(op.reference)[1]
        )
        old_pins = _component_pin_coords(editor, op.reference)
        moved_symbol = editor.components[op.reference].symbol
        moved_info = symbol_info_for(editor, moved_symbol) if moved_symbol else None
        if moved_info is not None:
            _pins_on_crossings(
                editor,
                op.reference,
                compute_placed_geometry(moved_info, op.x, op.y, new_rot.name)["pins"],
                f"move {op.reference}",
                already=old_pins,
            )
        # Wires left at its old pins are the finished sheet's to say
        # (dangling_wire_end), as for remove_component.
        editor.set_component_position(op.reference, Point(op.x, op.y), new_rot)
        return {"op": "move_component", "reference": op.reference}

    if isinstance(op, OpMove):
        return {"op": "move", **_apply_move(editor, op)}

    if isinstance(op, OpAddNetLabel):
        x, y = _resolve_op_xy(op, editor)
        _add_net_label_checks(editor, op.net, x, y)
        editor.labels.append(Text(coord=Point(x, y), text=op.net, type=TextTypeEnum.LABEL))
        return {"op": "add_net_label", "net": op.net, "x": x, "y": y}

    if isinstance(op, OpRemoveNetLabel):
        x, y = _resolve_op_xy(op, editor)
        before = len(editor.labels)
        editor.labels = [
            lbl for lbl in editor.labels if not (int(lbl.coord.X) == x and int(lbl.coord.Y) == y)
        ]
        removed = before - len(editor.labels)
        if removed == 0:
            raise NetlistError(f"No net label found at ({x},{y}).")
        return {"op": "remove_net_label", "x": x, "y": y, "removed": removed}

    if isinstance(op, OpRemoveWire):
        before = len(editor.wires)
        if all(v is not None for v in (op.x1, op.y1, op.x2, op.y2)):
            # Exact-segment form: drop the matching segment in either direction.
            seg = ((op.x1, op.y1), (op.x2, op.y2))
            rev = ((op.x2, op.y2), (op.x1, op.y1))
            kept = [
                w
                for w in editor.wires
                if ((int(w.V1.X), int(w.V1.Y)), (int(w.V2.X), int(w.V2.Y))) not in (seg, rev)
            ]
            copies = len(editor.wires) - len(kept)
            if copies > 1:
                # A duplicated segment is one connection drawn twice, so "remove
                # the duplicate" and "remove the connection" are the same
                # request at this interface. Removing every copy is right only
                # when the connection was redundant — and redundancy is a NET
                # fact, not a pin fact: cutting a duplicated bridge between two
                # wired stubs splits the net while every pin still touches some
                # wire, so a floating-pin scan blesses exactly the cut this
                # guard exists to refuse. The oracle is the net partition.
                part_before = net_partition(editor)
                nets_before: dict[tuple[int, int], list[tuple[int, int]]] = {}
                for coord in part_before.pin_owners:
                    nets_before.setdefault(part_before.root(coord), []).append(coord)
                original, editor.wires = editor.wires, kept
                part_after = net_partition(editor)
                for coords in nets_before.values():
                    sides: dict[tuple[int, int], list[tuple[int, int]]] = {}
                    for coord in coords:
                        sides.setdefault(part_after.root(coord), []).append(coord)
                    if len(sides) < 2:
                        continue
                    editor.wires = original

                    def _side_names(side: list[tuple[int, int]]) -> str:
                        names = [
                            f"{ref}.{pin}" if pin else ref
                            for c in sorted(side)
                            for ref, pin in part_before.pin_owners[c]
                        ]
                        return ", ".join(names[:3]) + (", ..." if len(names) > 3 else "")

                    first, second, *_ = sorted(sides.values(), key=len, reverse=True)
                    raise NetlistError(
                        f"Refusing to remove the {copies} copies of "
                        f"({op.x1},{op.y1})->({op.x2},{op.y2}): they are one connection "
                        f"drawn {copies} times, and removing it would split the net, "
                        f"leaving {_side_names(second)} disconnected from "
                        f"{_side_names(first)}. Remove the pin's other segments first "
                        "if the disconnection is what you want."
                    )
            else:
                editor.wires = kept
        elif op.pin is not None or (op.x is not None and op.y is not None):
            # Incident-point form: drop every segment touching the coordinate.
            _drop_wires_at(editor, {_resolve_op_xy(op, editor)})
        else:
            raise NetlistError(
                "remove_wire needs either all of x1,y1,x2,y2 (a segment) or a "
                "pin / both x and y (a point)."
            )
        removed = before - len(editor.wires)
        if removed == 0:
            raise NetlistError("No matching wire segment found to remove.")
        return {"op": "remove_wire", "removed": removed}

    if isinstance(op, OpWirePins):
        plan = _plan_connect_route(editor, op.from_pin, op.to_pin, op.waypoints)
        already = _append_wire_segments(editor, plan.segments)
        result = {
            "op": op.op,
            "from_pin": _endpoint_name(op.from_pin),
            "to_pin": _endpoint_name(op.to_pin),
            "wire_count": len(plan.segments) - len(already),
        }
        if already:
            result["already_present"] = [segment_json(seg) for seg in already]
        if plan.junctions:
            result["junctions"] = plan.junctions
        return result

    if isinstance(op, OpAddDirective):
        # Comments allow any text; SPICE directives go through validate_directive.
        if op.kind == "directive":
            err = validate_directive(op.instruction, simulator="LTspice")
            if err is not None:
                raise NetlistError(
                    f"Refusing directive {op.instruction!r}: {err.message} ({err.suggestion})"
                )
        text_type = TextTypeEnum.DIRECTIVE if op.kind == "directive" else TextTypeEnum.COMMENT
        _append_asc_text(
            editor, op.instruction, text_type, op.x, op.y, op.size, default_x=16, default_y=16
        )
        return {"op": "add_directive", "instruction": op.instruction}

    if isinstance(op, OpRemoveDirective):
        # Literal-by-default, 'regex:' opt-in, raises if nothing matched.
        removed = _remove_directive_or_comment(editor, op.instruction)
        return {"op": "remove_directive", "instruction": op.instruction, "removed": removed}

    raise NetlistError(f"Unknown op type: {type(op).__name__}")


def run_op_batch(
    editor: AscEditor,
    ops: Sequence[SchematicOp],
    asc_path: Path,
    *,
    stop_on_error: bool,
    plot: SheetPlotSettings | None = None,
) -> tuple[list[dict[str, object]], str | None]:
    """Apply ``ops`` in order via ``apply_op_inplace``; return (results, abort_reason).

    One unified entry per attempted op — ``{index, op, ok, error, **op_result}``.
    A ``NetlistError``/``ValueError`` marks that op ``ok=False`` with its message;
    when ``stop_on_error`` is set the first failure aborts (``abort_reason`` set,
    loop stops).
    """
    results: list[dict[str, object]] = []
    abort_reason: str | None = None
    for i, op in enumerate(ops):
        entry: dict[str, object] = {"index": i, "op": op.op, "ok": True, "error": None}
        try:
            op_result = apply_op_inplace(editor, op, asc_path, plot)
            entry.update({k: v for k, v in op_result.items() if k != "op"})
        except (NetlistError, ValueError) as e:
            entry["ok"] = False
            entry["error"] = str(e)
            results.append(entry)
            if stop_on_error:
                abort_reason = f"op #{i} ({op.op}) failed: {e}"
                break
            continue
        results.append(entry)
    return results, abort_reason
