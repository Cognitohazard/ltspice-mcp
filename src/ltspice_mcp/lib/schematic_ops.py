"""The .asc schematic edit engine: op models, appliers, geometry and net tracing.

One implementation, three consumers — ``edit_schematic`` applies the ops,
``inspect`` reads the same nets and components through the same cached editor,
and ``verify_circuit`` reuses the wiring geometry so a refusal the editor
enforces and a finding the checker reports cannot drift apart.

What lives here:

- the typed op union (``OpAddComponent`` … ``OpSetPlotPanes``), the in-place
  applier ``apply_op_inplace``, the facts its results report
  (``OP_RESULT_FACTS``), and the batch runner ``run_op_batch``;
- ``SheetPlotSettings``, the plot settings file beside a sheet as one batch
  changes it: the one op that does not edit the sheet itself writes there;
- ``edit_guard``, which serializes one file's mutation in-process and across
  parallel server sessions (with the files beside it a batch writes,
  ``files_written_beside``), and the cached-editor accessors it wraps;
- the placement, routing and net-partition geometry (``placed_geometry``,
  ``resolve_pin``, ``plan_connect_route``, ``net_partition``, ``trace_nets``),
  which reads each symbol once per request through ``symbol_info_for``;
- the post-op validation pass (``post_op_warnings``) and the wiring profile.

Names imported by another module are public.

Extension-based dispatch: the file extension picks the spicelib editor
(SpiceEditor for .cir/.net, AscEditor for .asc). Schematic-only operations
validate the extension and raise NetlistError for a non-.asc file.
"""

import asyncio
import importlib
import itertools
import math
import re
import traceback
from collections import Counter, defaultdict
from collections.abc import AsyncIterator, Callable, Container, Sequence
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
    plot_settings_path,
    read_plot_settings,
    scale_names,
    scales_of,
    with_panes,
    write_plot_settings,
)
from ltspice_mcp.lib.spice_lex import SpiceCard, SpiceLexError, TokenKind, tokenize_body
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
    if attribute.lower() == "prefix":
        # A SpiceLine referral here misdirects: Prefix is a SYMBOL property,
        # not an instance attribute, and SpiceLine can't change it.
        raise NetlistError(
            "Attribute 'Prefix' is a symbol property, not an instance "
            "attribute — the element prefix comes from the symbol's .asy "
            "file ('SYMATTR Prefix ...'). To place a subcircuit (X-prefix) "
            "device, use the edit_schematic add_component op with a symbol whose .asy declares "
            "'SYMATTR Prefix X', then bind the subckt name via Value or "
            "SpiceModel."
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
            comp.attributes[attr_name] = attr_val
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
_LTSPICE_ATTR_NAMES: frozenset[str] = frozenset(LTSPICE_PARAMETERS + LTSPICE_ATTRIBUTES)
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
# Used to peel trailing parameters off a multi-token component value like
# ``"NMOS1 W=10u L=1u"`` so we can route each piece through the right
# spicelib API (model name → ``set_component_value``; W/L → ``set_component_parameters``).
_PARAM_TOKEN_RE = re.compile(r"(\w+)\s*=\s*([^\s=]+)")

# A GUI opamp complexity-level label (e.g. ``Level.2``) that belongs in the
# SpiceModel/level selector, NOT the Value slot. On a subcircuit (X) symbol
# LTspice emits the Value as an extra positional token on the instance line,
# so netlisting fails downstream with "sub-circuit name is not defined".
_LEVEL_LABEL_RE = re.compile(r"^\s*level\.\d+\s*$", re.IGNORECASE)


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


def level_label_lint(editor, reference: str, value: str) -> str | None:
    """Warn when a subcircuit (X) symbol's Value slot will corrupt the netlist.

    Two shapes, both ending in "sub-circuit name is not defined" at netlist
    time because LTspice appends the Value as a stray positional token on the X
    instance line:

    - a GUI opamp complexity label (``Level.2``) written to Value; or
    - any Value set on a symbol whose model is selected via the ``SpiceModel``
      attribute (e.g. UniversalOpamp2), whose Value slot must stay empty.

    The subcircuit gate is an OR of signals — the InstName may be ``U1`` while
    the netlist prefix is X (from the .asy Prefix), so the X-prefix test alone
    is not enough. A symbol that carries its subckt name IN Value (the common
    library part, no SpiceModel) is left alone. Returns the warning text, or
    ``None`` when neither shape applies.
    """
    if not value.strip():
        return None
    comp = editor.components.get(reference)
    attrs = getattr(comp, "attributes", None) or {}
    has_spicemodel = "SpiceModel" in attrs
    is_subckt = reference[:1].upper() == "X" or has_spicemodel or "_SUBCKT" in attrs
    if not is_subckt:
        return None
    if _LEVEL_LABEL_RE.match(value):
        return (
            f"{reference}: Value {value!r} is a GUI opamp complexity-level label, not a "
            "subcircuit value. LTspice emits it as an extra positional token on the X "
            "instance line, so netlisting fails with 'sub-circuit name is not defined'. "
            "Select the level via the SpiceModel attribute (or the symbol's default), "
            "not the Value slot."
        )
    if has_spicemodel:
        return (
            f"{reference}: this symbol's model is selected via its SpiceModel attribute, "
            f"so its Value slot must stay empty. LTspice emits Value {value!r} as an "
            "extra positional token on the X instance line, and netlisting fails with "
            "'sub-circuit name is not defined'. Clear the Value; pick the model through "
            "SpiceModel instead."
        )
    return None


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
    tokens and route them through ``set_component_parameters``, keeping
    the model/value field for ``set_component_value``.

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
        editor.set_component_parameters(reference, **params)


def _bboxes_overlap(a: dict, b: dict) -> bool:
    """AABB overlap test between two bounding boxes with {x, y, width, height}."""
    return BBox.from_origin_size(a["x"], a["y"], a["width"], a["height"]).overlaps(
        BBox.from_origin_size(b["x"], b["y"], b["width"], b["height"])
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
    """The element letter a placed part netlists as.

    LTspice takes it from the symbol's ``Prefix`` (``QN`` → ``Q``) and prepends
    that letter to an instance name that does not already start with it, so a
    part named ``Vin`` on a resistor symbol is a resistor. Falls back to the
    reference's own first letter when the symbol does not resolve.
    """
    comp = editor.components.get(reference)
    symbol = getattr(comp, "symbol", None)
    info = symbol_info_for(editor, symbol) if symbol else None
    prefix = info.prefix if info is not None and info.prefix else reference
    return prefix[:1].upper()


def collect_component_geometry(editor: AscEditor) -> list[dict]:
    """Collect bounding boxes and pin positions for all components."""
    result: list[dict] = []
    for ref in editor.get_components():
        geo = placed_geometry(editor, ref)
        if geo is not None:
            result.append({"ref": ref, **geo["bounding_box"], "pins": geo["pins"]})
    return result


def _overlap_warnings(editor: AscEditor, reference: str, bbox: dict[str, int]) -> list[str]:
    """Warn for each other component whose bounding box overlaps ``bbox``.

    Shared by add_component placement and move_component reposition — both flag
    where a just-placed or just-moved part lands on top of another.
    """
    return [
        f"Overlaps {existing['ref']} bounding box"
        for existing in collect_component_geometry(editor)
        if existing["ref"] != reference and _bboxes_overlap(bbox, existing)
    ]


def _component_pin_coords(editor: AscEditor, reference: str) -> set[tuple[int, int]]:
    """Pin coordinates for a single component, ``set()`` if symbol unknown."""
    if reference not in editor.components:
        return set()
    geo = placed_geometry(editor, reference)
    return set() if geo is None else {(p["x"], p["y"]) for p in geo["pins"]}


def _other_components_pin_coords(editor: AscEditor, exclude_ref: str) -> set[tuple[int, int]]:
    """Union of pin coordinates for every component except ``exclude_ref``.

    Used by remove/move handlers to filter orphaned-wire warnings: a wire
    endpoint that coincides with another component's pin isn't actually
    orphaned, it's that component's wire.
    """
    coords: set[tuple[int, int]] = set()
    for ref in editor.get_components():
        if ref == exclude_ref:
            continue
        coords.update(_component_pin_coords(editor, ref))
    return coords


def point_on_segment(point: tuple[int, int], v1: tuple[int, int], v2: tuple[int, int]) -> bool:
    """True iff ``point`` lies on the wire segment ``v1 → v2``, ends included.

    A wire need not be horizontal or vertical: LTspice draws diagonal ones and
    connects a pin or label that sits on one anywhere along its length, as it
    does on any other wire.
    """
    px, py = point
    x1, y1 = v1
    x2, y2 = v2
    if (x2 - x1) * (py - y1) != (y2 - y1) * (px - x1):
        return False
    return min(x1, x2) <= px <= max(x1, x2) and min(y1, y2) <= py <= max(y1, y2)


def build_on_wire_predicate(
    segments: list[tuple[tuple[int, int], tuple[int, int]]],
) -> "Callable[[tuple[int, int]], bool]":
    """Return an ``on_wire(coord)`` predicate with the same semantics as
    ``point_on_segment`` but O(1)-amortised per query.

    The naive ``any(point_on_segment(coord, *seg) for seg in segments)``
    scan is O(segments) per coord; calling it once per pin makes
    ``post_op_warnings`` O(pins × segments), which becomes the dominant
    cost during a long ``add_component`` build. Bucketing
    horizontal segments by row and vertical by column collapses each query
    to the handful of segments sharing that row/column.
    """
    endpoints: set[tuple[int, int]] = set()
    horiz: dict[int, list[tuple[int, int]]] = {}
    vert: dict[int, list[tuple[int, int]]] = {}
    for (x1, y1), (x2, y2) in segments:
        endpoints.add((x1, y1))
        endpoints.add((x2, y2))
        if y1 == y2 and x1 != x2:
            horiz.setdefault(y1, []).append((min(x1, x2), max(x1, x2)))
        elif x1 == x2 and y1 != y2:
            vert.setdefault(x1, []).append((min(y1, y2), max(y1, y2)))
        # Diagonal / zero-length segments contribute via endpoints only,
        # matching point_on_segment's diagonal fallback.

    def on_wire(coord: tuple[int, int]) -> bool:
        if coord in endpoints:
            return True
        px, py = coord
        if any(xmin <= px <= xmax for xmin, xmax in horiz.get(py, ())):
            return True
        return any(ymin <= py <= ymax for ymin, ymax in vert.get(px, ()))

    return on_wire


class NetPartition(NamedTuple):
    """Connected-component view of a schematic's nets.

    ``root`` maps any interest coordinate to its net's canonical
    representative; ``members`` maps a root to every coordinate on that net;
    ``pin_owners`` maps a coordinate to the ``(ref, pin_name)`` pairs sitting
    there; ``label_texts`` maps a coordinate to the FLAG texts placed there.
    """

    root: "Callable[[tuple[int, int]], tuple[int, int]]"
    members: dict[tuple[int, int], set[tuple[int, int]]]
    pin_owners: dict[tuple[int, int], list[tuple[str, str]]]
    label_texts: dict[tuple[int, int], set[str]]


def net_partition(
    editor: AscEditor,
    extra_segments: list[tuple[int, int, int, int]] | None = None,
) -> NetPartition:
    """Union-find over pins, labels, and wires → a connected-net partition.

    Segment-aware: a pin, a label or another wire's end lying anywhere ON a
    wire, its interior included, is unioned with that wire. Two wires that
    merely cross, neither ending at the crossing, stay separate. This is how
    LTspice's own netlister connects a sheet: checked against LTspice 26.1.1
    ``-netlist`` exports of a label, a pin and a wire end on a wire's interior
    (connected, with the wire left whole — no split needed), a plain crossing
    (not connected), and a label at a crossing (joins both wires).
    The sheets and their exports are ``tests/fixtures/t_junctions/``. LTspice
    26 and LTspice XVII agree on each of those, and on a pin at a crossing,
    collinear wires that overlap, two pins that only touch, and a point on a
    diagonal wire, in the connectivity sheets recorded from both
    (``docs/TESTING.md``, "Recorded LTspice behaviour").

    ``extra_segments`` lets the caller include not-yet-committed wire
    segments (e.g. the route ``wire_pins`` is about to add) so checks operate
    on the post-route net layout. Shared by ``trace_nets`` (labels-per-net)
    and ``trace_net`` (full net membership).
    """
    parent: dict[tuple[int, int], tuple[int, int]] = {}

    def find(p: tuple[int, int]) -> tuple[int, int]:
        if p not in parent:
            parent[p] = p
            return p
        while parent[p] != p:
            parent[p] = parent[parent[p]]
            p = parent[p]
        return p

    def union(a: tuple[int, int], b: tuple[int, int]) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    # Collect every "interest point": pin coords + label coords + wire
    # endpoints. A wire that touches one of these in its interior pulls
    # it into the same connected component as its endpoints.
    interest_points: set[tuple[int, int]] = set()
    pin_owners: dict[tuple[int, int], list[tuple[str, str]]] = {}
    for entry in collect_component_geometry(editor):
        ref = entry["ref"]
        for pin in entry["pins"]:
            coord = (pin["x"], pin["y"])
            interest_points.add(coord)
            find(coord)
            pin_owners.setdefault(coord, []).append((ref, pin["name"]))
    label_texts: dict[tuple[int, int], set[str]] = {}
    for lbl in editor.labels:
        coord = (int(lbl.coord.X), int(lbl.coord.Y))
        interest_points.add(coord)
        find(coord)
        label_texts.setdefault(coord, set()).add(lbl.text)

    segments: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for w in editor.wires:
        segments.append(((int(w.V1.X), int(w.V1.Y)), (int(w.V2.X), int(w.V2.Y))))
    if extra_segments:
        for sx1, sy1, sx2, sy2 in extra_segments:
            segments.append(((sx1, sy1), (sx2, sy2)))

    # Wire endpoints are interest points themselves.
    for v1, v2 in segments:
        interest_points.add(v1)
        interest_points.add(v2)
        union(v1, v2)

    # For each segment, union every interest point lying on it with the
    # segment's endpoints. This is O(segments * interest_points) — fine
    # for typical schematics (a few hundred of each).
    for v1, v2 in segments:
        for pt in interest_points:
            if pt in (v1, v2):
                continue
            if point_on_segment(pt, v1, v2):
                union(pt, v1)

    members: dict[tuple[int, int], set[tuple[int, int]]] = {}
    for p in parent:
        members.setdefault(find(p), set()).add(p)

    return NetPartition(root=find, members=members, pin_owners=pin_owners, label_texts=label_texts)


def label_folded_nets(part: NetPartition) -> "Callable[[tuple[int, int]], tuple[int, int]]":
    """Map a coordinate to its electrical net's representative.

    The partition connects by wire only; LTspice also makes every FLAG with the
    same name one node, so wired nets that share a label name fold into one
    here. Two coordinates are on the same netlist node iff this returns the same
    representative for both.
    """
    parent: dict[tuple[int, int], tuple[int, int]] = {}

    def find(r: tuple[int, int]) -> tuple[int, int]:
        parent.setdefault(r, r)
        while parent[r] != r:
            parent[r] = parent[parent[r]]
            r = parent[r]
        return r

    first_root: dict[str, tuple[int, int]] = {}
    for root, coords in part.members.items():
        for coord in coords:
            for text in part.label_texts.get(coord, ()):
                if text not in first_root:
                    first_root[text] = root
                    continue
                ra, rb = find(first_root[text]), find(root)
                if ra != rb:
                    parent[ra] = rb

    return lambda coord: find(part.root(coord))


def net_members(
    part: NetPartition,
    net_of: "Callable[[tuple[int, int]], tuple[int, int]]",
    net: tuple[int, int],
) -> set[tuple[int, int]]:
    """Every pin, label and wire-end coordinate on ``net``, a ``net_of`` value."""
    return {c for root, coords in part.members.items() if net_of(root) == net for c in coords}


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


def named_labels(labels: frozenset[str]) -> set[str]:
    """Strip ground ('0') from a label set so 'short to ground' isn't
    flagged as a conflict by detect-multi-label net checks."""
    return {lbl for lbl in labels if lbl != "0"}


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


def post_op_warnings(editor: AscEditor) -> list[dict]:
    """Schematic-state advisories surfaced after a mutating op succeeds.

    Returns structured warnings the agent can act on without a follow-up
    inspection turn:

    - ``floating_pin`` — a component pin with no wire passing through,
      no net label sitting on it, and no other component pin sharing
      the coordinate.
    - ``duplicate_wire`` — two wire segments sharing the same endpoints
      (in either order). Pure noise, costs nothing to drop.
    - ``dangling_label`` — a net label whose coordinate is neither on a
      wire nor at any component pin.
    - ``label_over_component`` — a net label whose coordinate falls strictly
      inside a component's bounding box while sitting on no component's pin.
      Surfaces the anchor-in-box fact only: the axis-aligned box also spans
      leads and empty corners, so this is not a guarantee the rendered glyph
      overlaps the drawn symbol. A label on a pin (any component's) — the normal
      ground-flag pattern — is on a box boundary and is excluded.
    - ``stacked_directive`` — two or more directive/comment text objects at
      the exact same anchor, rendering on top of each other. Exact-coordinate
      match only (no font-metric guessing), so this never fires on a
      deliberately tight-but-offset directive block.

    Read-only on the editor. Cheap to compute during an existing edit
    session; intended for callers to surface in their response payload.
    """
    pins: list[tuple[str, str, int, int]] = []
    comp_boxes: list[tuple[str, BBox]] = []
    for entry in collect_component_geometry(editor):
        ref = entry["ref"]
        comp_boxes.append(
            (ref, BBox.from_origin_size(entry["x"], entry["y"], entry["width"], entry["height"]))
        )
        for p in entry["pins"]:
            pins.append((ref, p["name"], p["x"], p["y"]))

    pin_count_at: dict[tuple[int, int], int] = {}
    for _, _, x, y in pins:
        pin_count_at[(x, y)] = pin_count_at.get((x, y), 0) + 1

    segments = [((int(w.V1.X), int(w.V1.Y)), (int(w.V2.X), int(w.V2.Y))) for w in editor.wires]
    label_coords = {(int(lbl.coord.X), int(lbl.coord.Y)) for lbl in editor.labels}

    _on_any_wire = build_on_wire_predicate(segments)

    warnings: list[dict] = []

    for ref, name, x, y in pins:
        coord = (x, y)
        if pin_count_at[coord] > 1:
            continue
        if coord in label_coords:
            continue
        if _on_any_wire(coord):
            continue
        pin_label = f"{ref}.{name}" if name else ref
        warnings.append(
            {
                "kind": "floating_pin",
                "ref": ref,
                "pin": name,
                "x": x,
                "y": y,
                "message": f"Floating pin: {pin_label} at ({x},{y})",
            }
        )

    seen_segments: dict[tuple[tuple[int, int], tuple[int, int]], int] = {}
    for v1, v2 in segments:
        if v1 == v2:
            continue
        key = (v1, v2) if v1 <= v2 else (v2, v1)
        seen_segments[key] = seen_segments.get(key, 0) + 1
    for (a, b), count in seen_segments.items():
        if count > 1:
            warnings.append(
                {
                    "kind": "duplicate_wire",
                    "from": {"x": a[0], "y": a[1]},
                    "to": {"x": b[0], "y": b[1]},
                    "count": count,
                    "message": (f"Duplicate wire ({count}×): ({a[0]},{a[1]})->({b[0]},{b[1]})"),
                }
            )

    pin_coords = pin_count_at.keys()
    for lbl in editor.labels:
        coord = (int(lbl.coord.X), int(lbl.coord.Y))
        if coord in pin_coords:
            continue
        if _on_any_wire(coord):
            continue
        warnings.append(
            {
                "kind": "dangling_label",
                "label": lbl.text,
                "x": coord[0],
                "y": coord[1],
                "message": f"Dangling label '{lbl.text}' at ({coord[0]},{coord[1]})",
            }
        )

    for lbl in editor.labels:
        coord = (int(lbl.coord.X), int(lbl.coord.Y))
        # A label on ANY component's pin is the normal flag pattern (pins sit on
        # symbol outlines) — never report it, even when it also lands inside a
        # different, overlapping component's box.
        if coord in pin_count_at:
            continue
        for ref, box in comp_boxes:
            # Strict interior only: a coordinate on the box boundary — where pins
            # and leads sit — is not "inside". No break: with overlapping boxes a
            # label can be inside more than one, and each is a distinct fact.
            if box.x1 < coord[0] < box.x2 and box.y1 < coord[1] < box.y2:
                warnings.append(
                    {
                        "kind": "label_over_component",
                        "label": lbl.text,
                        "ref": ref,
                        "x": coord[0],
                        "y": coord[1],
                        "message": (
                            f"Label '{lbl.text}' at ({coord[0]},{coord[1]}) is inside "
                            f"{ref}'s bounding box"
                        ),
                    }
                )

    directive_anchor_count: dict[tuple[int, int], int] = {}
    for d in editor.directives:
        anchor = (int(d.coord.X), int(d.coord.Y))
        directive_anchor_count[anchor] = directive_anchor_count.get(anchor, 0) + 1
    for (dx, dy), count in directive_anchor_count.items():
        if count > 1:
            warnings.append(
                {
                    "kind": "stacked_directive",
                    "x": dx,
                    "y": dy,
                    "count": count,
                    "message": (
                        f"{count} directives/comments share anchor ({dx},{dy}) — "
                        "they render on top of each other"
                    ),
                }
            )

    return warnings


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
    # LTspice's on-disk TEXT record is one physical line; embedded newlines
    # are stored as literal "\n" escapes (LTspice's own multi-line text
    # convention). A raw newline would split the record and corrupt the file
    # — the downstream symptom is an unrelated-looking "Primitive not
    # supported" parse error.
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\n", "\\n")
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
    for idx, d in enumerate(directives):
        dtype = getattr(d, "type", None)
        dtext = getattr(d, "text", None)
        if dtype == TextTypeEnum.DIRECTIVE and dtext == instruction:
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
    if instruction.startswith("regex:"):
        pattern = instruction[6:]
        if not pattern.strip():
            raise NetlistError(
                "Empty regex pattern would match every directive; "
                "provide an explicit regex after 'regex:'."
            )
        try:
            compiled = re.compile(pattern)
        except re.error as e:
            raise NetlistError(f"Invalid regex {pattern!r}: {e}") from e
        directive_hit = bool(editor.remove_Xinstruction(pattern))
        comment_hit = _strip_matching_comments(editor, compiled)
        if not (directive_hit or comment_hit):
            raise NetlistError(
                f"No directive or comment matched regex {pattern!r}. "
                "Use inspect(kind='components') to see what's actually in the file."
            )
        return "directive(s)/comment(s)"

    directive_hit = _remove_exact_directive(editor, instruction)
    comment_hit = _strip_matching_comments(editor, instruction)
    if not (directive_hit or comment_hit):
        raise NetlistError(
            f"No directive or comment matched {instruction!r} exactly. "
            "Match is literal by default — pass 'regex:<pattern>' for regex "
            "matching, or copy the line verbatim from inspect(kind='components')."
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
    keep = []
    for entry in directives:
        body = getattr(entry, "text", None)
        entry_kind = getattr(entry, "type", None)
        if entry_kind == TextTypeEnum.COMMENT and isinstance(body, str):
            if isinstance(matcher, str):
                if body.strip() == matcher.strip():
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


def _orphaned_wire_coords(editor: AscEditor, target_only_pins: set[tuple[int, int]]) -> list[str]:
    """``(x,y)`` strings for wires still ending on a removed/moved component's
    former pins (its pins minus other components' pins). Shared by the standalone
    handlers and the edit_schematic ops; call AFTER the mutation."""
    orphaned: list[str] = []
    for w in editor.wires:
        for coord in ((int(w.V1.X), int(w.V1.Y)), (int(w.V2.X), int(w.V2.Y))):
            label = f"({coord[0]},{coord[1]})"
            if coord in target_only_pins and label not in orphaned:
                orphaned.append(label)
    return orphaned


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


def _move_component_warnings(
    editor: AscEditor,
    reference: str,
    rot_name: str,
    x: int,
    y: int,
    old_pin_coords: set[tuple[int, int]],
    other_pins: set[tuple[int, int]],
) -> list[str]:
    """Bounding-box-overlap + orphaned-wire warnings for a just-moved component.

    ``old_pin_coords`` and ``other_pins`` must be captured BEFORE the move;
    call this AFTER ``set_component_position``.
    """
    warnings: list[str] = []
    new_pin_coords = _component_pin_coords(editor, reference)

    comp = editor.components[reference]
    moved_bb: dict[str, int] | None = None
    if comp.symbol:
        moved_sym = symbol_info_for(editor, comp.symbol)
        if moved_sym is not None:
            moved_bb = compute_placed_geometry(moved_sym, x, y, rot_name)["bounding_box"]
    if moved_bb is not None:
        warnings.extend(_overlap_warnings(editor, reference, moved_bb))

    # Pins that the move abandoned (no longer this component's, not a neighbour's)
    # but that still have a wire ending on them are orphaned by the move.
    abandoned_pins = old_pin_coords - new_pin_coords - other_pins
    orphaned = _orphaned_wire_coords(editor, abandoned_pins) if abandoned_pins else []
    if orphaned:
        warnings.append(
            f"wires left at old pin coordinates with no connection: {', '.join(orphaned)}. "
            "Re-route or delete these wires."
        )
    return warnings


def _placed_component_data(
    editor: AscEditor,
    reference: str,
    symbol: str,
    x: int,
    y: int,
    rotation: str,
    symbol_info: SymbolInfo,
) -> dict[str, object]:
    """Return the placed symbol geometry and any component-overlap warnings."""
    geometry = compute_placed_geometry(symbol_info, x, y, rotation)
    bounding_box = geometry["bounding_box"]
    warnings = _overlap_warnings(editor, reference, bounding_box)
    return {
        "reference": reference,
        "symbol": symbol,
        "position": {"x": x, "y": y},
        "rotation": rotation,
        "pins": geometry["pins"],
        "bounding_box": bounding_box,
        "warnings": warnings,
    }


def resolve_pin(pin_ref: str, editor: AscEditor) -> tuple[int, int]:
    """Resolve a pin reference ('M1.D', 'X1.2' or 'net:VDD') to absolute (x, y) coordinates.

    The part after the last dot is matched against the symbol's pin names first,
    case-insensitively. Only when no pin carries that name, and it is all
    digits, is it read as the pin's 1-based SpiceOrder — the terminal number a
    netlist uses. Names win because some symbols name their pins '1', '2' in an
    order that need not be their SpiceOrder.

    Raises NetlistError if the reference cannot be resolved.
    """
    if pin_ref.startswith("net:"):
        # Look up a FLAG/net label position in the .asc
        net_name = pin_ref[4:]
        matches = [
            (int(lbl.coord.X), int(lbl.coord.Y)) for lbl in editor.labels if lbl.text == net_name
        ]
        if not matches:
            raise NetlistError(
                f"Net label '{net_name}' not found in schematic. Add it with the "
                "add_net_label op of edit_schematic first."
            )
        if len(matches) > 1:
            coords = ", ".join(f"({x},{y})" for x, y in matches)
            raise NetlistError(
                f"Multiple '{net_name}' labels found at: {coords}. "
                "Connect to a component pin (Ref.Pin) instead, or place the label "
                "at a specific pin with the add_net_label op of edit_schematic "
                f"(net='{net_name}', pin='<Ref.Pin>')."
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
    if geometry is None:
        raise NetlistError(f"Cannot resolve pins for '{ref}': symbol '{symbol}' not found.")

    pins = geometry["pins"]
    for pin in pins:
        if pin["name"].upper() == pin_name.upper():
            return pin["x"], pin["y"]
    # A pin with no SpiceOrder line parses as order 0, which is no ordinal.
    ordinal = int(pin_name) if pin_name.isascii() and pin_name.isdigit() else 0
    if ordinal:
        for pin in pins:
            if pin["order"] == ordinal:
                return pin["x"], pin["y"]

    available = [f"{p['name']} ({p['order']})" if p["order"] else p["name"] for p in pins]
    raise NetlistError(
        f"Pin '{pin_name}' not found on {ref} ({symbol}). Available: {', '.join(available)} "
        "(pin name, with its SpiceOrder in parentheses)."
    )


def _add_net_label_checks(editor: AscEditor, net: str, x: int, y: int) -> list[str]:
    """Validate placing net label ``net`` at ``(x, y)``.

    Raises ``NetlistError`` if a non-ground label would join two nets that each
    carry a name (a short at netlist time) — a structural error, refused
    outright. That happens when the network here is named and ``net`` already
    names a different one, or when the label lands where two named networks
    cross and joins both. Returns advisory warnings (duplicate name, a second
    name for an already-named net, floating placement) as plain facts for the
    caller to surface; placing labels before wiring them is a legitimate
    workflow, so those are warnings, not refusals.
    """
    warnings: list[str] = []
    part = net_partition(editor)
    # A FLAG anywhere along a wire, its interior included, joins that wire's
    # net, and one at a crossing joins both (see net_partition for the export
    # record).
    through = wires_through((x, y), wire_segments_of(editor))
    if net != "0":
        # Duplicate non-ground label name. This is NOT a short: the netlist merges
        # same-name labels into one net, which is a valid (often simpler) way to
        # tie distant nets. The only downstream cost is that wire_pins(net=...) can't
        # disambiguate which label to route to — so surface that, not a scare.
        for lbl in editor.labels:
            if lbl.text == net:
                warnings.append(
                    f"'{net}' already labels a net at ({int(lbl.coord.X)},"
                    f"{int(lbl.coord.Y)}); the netlist merges the two into one net "
                    "(this is correct — a valid way to tie distant nets). Only a "
                    f"later wire_pins(net='{net}') is ambiguous with duplicate labels — "
                    "connect to a component pin (Ref.Pin) instead."
                )
                break
        joined = _label_join(part, through, net, (x, y))
        if len(joined) > 1:
            listed = "; ".join(f"{sorted(names)}" for names in joined)
            raise NetlistError(
                f"Refused to add net '{net}' at ({x},{y}): the label would join nets "
                f"that each carry a name ({listed}), shorting them together. To join "
                "them on purpose, give both the same name: remove one side's labels "
                "with remove_net_label and label it with the other's name."
            )
        if joined and net not in joined[0]:
            warnings.append(
                f"({x},{y}) is on the net already labelled {sorted(joined[0])}; '{net}' "
                "becomes a second name for it. LTspice netlists the node under one name, "
                "so a directive or probe that uses another may not find it."
            )
    # Floating label: a FLAG touching no wire and no component pin names
    # nothing at netlist time.
    if not through and (x, y) not in part.pin_owners:
        warnings.append(
            f"({x},{y}) touches no wire and no component pin — LTspice will ignore "
            "this floating label until you wire it up."
        )
    return warnings


def _label_join(
    part: NetPartition,
    through: Sequence[tuple[int, int, int, int]],
    net: str,
    at: tuple[int, int],
) -> list[frozenset[str]]:
    """The named nets a label ``net`` placed at ``at`` would make one node.

    Read off the sheet's partition without the label: the label joins every
    wire it touches (``through``, both wires at a crossing), whatever already
    sits at ``at`` (a pin, a label, a wire end), and, by name, every net
    already called ``net``. Each joined net comes back as its set of names,
    ground excluded; two or more is a short.
    """
    node_of = label_folded_nets(part)
    names_by_node: dict[tuple[int, int], set[str]] = defaultdict(set)
    for coord, texts in part.label_texts.items():
        names_by_node[node_of(coord)].update(named_labels(frozenset(texts)))
    # A point the sheet does not have reads as its own unnamed node.
    nodes = {node_of(at), *(node_of(seg[:2]) for seg in through)}
    nodes.update(node_of(coord) for coord, texts in part.label_texts.items() if net in texts)
    return [frozenset(names_by_node[n]) for n in sorted(nodes) if names_by_node[n]]


class _ConnectPlan(NamedTuple):
    """Validated connect route ready to commit to the editor."""

    x1: int
    y1: int
    x2: int
    y2: int
    points: list[tuple[int, int]]
    segments: list[tuple[int, int, int, int]]
    warnings: list[str]
    # Every point other than the two endpoints where the route joins existing
    # wiring, and each endpoint that ends on a wire's interior (a T-junction).
    junctions: list[dict[str, object]]


def _merge_collinear_runs(
    segments: list[tuple[int, int, int, int]],
    node_coords: set[tuple[int, int]],
) -> list[tuple[int, int, int, int]]:
    """Collapse straight runs of collinear wire segments into single segments,
    mirroring LTspice's netlist-time wire merge.

    A vertex breaks a run — stays its own node — when it is a pin coordinate
    (``node_coords``) or a corner/junction (its incident segment ends are not
    exactly two ends of one orientation). Only pure pass-through vertices
    (degree-2, both ends collinear, not a pin) are merged across. This is what
    makes a *collinear* waypoint disappear: an in-line bend leaves only bare
    pass-through vertices, so the run collapses back to one segment; a bend that
    turns a corner leaves the corner vertices as breaks, so its segments stay
    split. Verticals (``x1==x2``) and horizontals (``y1==y2``) are merged
    per-line by interval union split at breaks; any diagonal passes through
    unchanged.
    """
    ends: dict[tuple[int, int], list[str]] = defaultdict(list)
    verticals: dict[int, list[tuple[int, int]]] = defaultdict(list)
    horizontals: dict[int, list[tuple[int, int]]] = defaultdict(list)
    merged: list[tuple[int, int, int, int]] = []
    for x1, y1, x2, y2 in segments:
        if (x1, y1) == (x2, y2):
            continue  # zero-length record (hand-corrupted WIRE) — nothing to merge
        if x1 == x2:
            verticals[x1].append((min(y1, y2), max(y1, y2)))
            ends[(x1, y1)].append("V")
            ends[(x2, y2)].append("V")
        elif y1 == y2:
            horizontals[y1].append((min(x1, x2), max(x1, x2)))
            ends[(x1, y1)].append("H")
            ends[(x2, y2)].append("H")
        else:
            merged.append((x1, y1, x2, y2))  # diagonal — passed through as-is

    def _breaks(coord: tuple[int, int]) -> bool:
        es = ends.get(coord, [])
        return coord in node_coords or len(es) != 2 or len(set(es)) != 1

    def _emit(intervals: list[tuple[int, int]], cuts: set[int], vertical: bool, line: int) -> None:
        intervals.sort()
        runs: list[list[int]] = []
        for lo, hi in intervals:
            if runs and lo <= runs[-1][1]:
                runs[-1][1] = max(runs[-1][1], hi)
            else:
                runs.append([lo, hi])
        for lo, hi in runs:
            pts = sorted({lo, hi} | {c for c in cuts if lo < c < hi})
            for a, b in itertools.pairwise(pts):
                merged.append((line, a, line, b) if vertical else (a, line, b, line))

    # A component pin on the interior of a run is a junction too — LTspice splits
    # the wire there — but it is not a wire endpoint, so it never appears in
    # ``ends`` and a break test over ``ends`` alone would miss it. Add every pin
    # sitting on the line as a cut candidate; ``_emit`` keeps only those strictly
    # inside a run's span, so a pin at a run end (already a natural node) or off
    # any run adds nothing.
    for x, iv in verticals.items():
        cuts = {y for (cx, y) in ends if cx == x and _breaks((cx, y))}
        cuts |= {py for (px, py) in node_coords if px == x}
        _emit(iv, cuts, vertical=True, line=x)
    for y, iv in horizontals.items():
        cuts = {x for (x, cy) in ends if cy == y and _breaks((x, cy))}
        cuts |= {px for (px, py) in node_coords if py == y}
        _emit(iv, cuts, vertical=False, line=y)
    return merged


def wire_segments_of(editor: AscEditor) -> list[tuple[int, int, int, int]]:
    """Every wire as a flat ``(x1, y1, x2, y2)`` integer tuple."""
    return [(int(w.V1.X), int(w.V1.Y), int(w.V2.X), int(w.V2.Y)) for w in editor.wires]


def wires_through(
    coord: tuple[int, int], segments: Sequence[tuple[int, int, int, int]]
) -> list[tuple[int, int, int, int]]:
    """The segments ``coord`` lies on, ends included, in the order given.

    A point on a wire's interior touches that wire the way its end would: LTspice
    joins anything placed there (see :func:`net_partition`). More than one
    segment comes back where wires meet, overlap or cross.
    """
    return [s for s in segments if point_on_segment(coord, (s[0], s[1]), (s[2], s[3]))]


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


def segment_text(seg: tuple[int, int, int, int]) -> str:
    """A wire as a message names it: ``(x1,y1)->(x2,y2)``."""
    return f"({seg[0]},{seg[1]})->({seg[2]},{seg[3]})"


def segment_json(seg: tuple[int, int, int, int]) -> dict[str, dict[str, int]]:
    """A wire as a response carries it: ``{from: {x, y}, to: {x, y}}``."""
    return {"from": {"x": seg[0], "y": seg[1]}, "to": {"x": seg[2], "y": seg[3]}}


def same_instance_dropped_segments(
    pin_owners: dict[tuple[int, int], list[tuple[str, str]]],
    segments: list[tuple[int, int, int, int]],
) -> list[dict]:
    """Wire segments LTspice discards from the exported netlist.

    LTspice drops a wire run whose two ends both land exactly on pins of the
    SAME single component instance (recorded from the ``-netlist`` export of
    LTspice 26 and of LTspice XVII: such a run never reaches the netlist, so the
    two pins stay on separate nodes and the drawn tie has no electrical
    effect). Two routes still get kept, and both are in the same recordings: a
    run spanning two *different* instances, and a same-instance tie that turns
    a corner OUT OF LINE with the two pins. A waypoint that stays *collinear*
    with the pins does NOT survive —
    LTspice merges the in-line segments back into one and drops it — so the
    segments are collinear-merged (:func:`_merge_collinear_runs`) before this
    check, which is what catches an all-in-line waypoint route as well as the
    bare direct wire. A net label on one pin does not rescue the run either.

    Returns one dict per dropped run with ``segment`` (the merged ``(x1, y1, x2,
    y2)`` tuple), ``ref`` (the shared instance), and ``pins`` (the two pin
    names on that instance), ordered deterministically. ``pin_owners`` maps each
    pin coordinate to its ``(ref, pin_name)`` owners, as :func:`net_partition`
    builds it.
    """
    dropped: list[dict] = []
    for seg in _merge_collinear_runs(list(segments), set(pin_owners)):
        sx1, sy1, sx2, sy2 = seg
        if (sx1, sy1) == (sx2, sy2):
            continue  # defensive: a collapsed/zero-length run is not a tie
        owners_a = pin_owners.get((sx1, sy1), [])
        owners_b = pin_owners.get((sx2, sy2), [])
        if not owners_a or not owners_b:
            continue  # an end is a bare vertex/waypoint, not a pin — kept
        refs_a = {r for r, _ in owners_a}
        refs_b = {r for r, _ in owners_b}
        if len(refs_a | refs_b) != 1:
            continue  # the run bridges two distinct instances — kept
        ref = next(iter(refs_a))
        pin_a, pin_b = owners_a[0][1], owners_b[0][1]
        dropped.append({"segment": seg, "ref": ref, "pins": (pin_a, pin_b)})
    dropped.sort(key=lambda d: (d["ref"], d["segment"]))
    return dropped


def _endpoint_name(endpoint: "str | GridPoint") -> str:
    """A route endpoint as a message names it: as given, or as ``(x,y)``."""
    return endpoint if isinstance(endpoint, str) else f"({endpoint.x},{endpoint.y})"


def _crossing_warning(x: int, y: int, wire: tuple[int, int, int, int]) -> str:
    """The advisory for a route that crosses a wire where neither one ends.

    LTspice leaves such a crossing unjoined (``tests/fixtures/t_junctions/``
    ``crossing_wires``), so the route creates no connection there and the nets
    stay apart. It is reported, not refused: the only cost is a reader taking
    the crossing for a junction.
    """
    return (
        f"The route crosses the wire {segment_text(wire)} at ({x},{y}) where neither "
        "ends; LTspice leaves a plain crossing unjoined, so the two stay separate "
        "nets. To join them, end the route on that wire instead."
    )


def _collinear_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    """True iff two orthogonal segments lie on one line and share a stretch of it."""
    if a[0] == a[2] == b[0] == b[2]:
        lo, hi = max(min(a[1], a[3]), min(b[1], b[3])), min(max(a[1], a[3]), max(b[1], b[3]))
        return hi > lo
    if a[1] == a[3] == b[1] == b[3]:
        lo, hi = max(min(a[0], a[2]), min(b[0], b[2])), min(max(a[0], a[2]), max(b[0], b[2]))
        return hi > lo
    return False


def _resolve_route_endpoint(
    endpoint: "str | GridPoint",
    editor: AscEditor,
    existing_wires: list[tuple[int, int, int, int]],
    pin_coords: "Container[tuple[int, int]]",
    net_of: "Callable[[tuple[int, int]], tuple[int, int]]",
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Resolve one ``wire_pins`` endpoint to its coordinate and an anchor.

    The anchor is a pin, label or wire end on the endpoint's net before the
    route is drawn, for looking the net up. A pin or label reference resolves
    through :func:`resolve_pin` and is its own anchor. A coordinate must already
    touch the circuit, at a pin or anywhere along a wire: an end there connects
    to it, and one on a wire's interior is a T-junction LTspice joins without
    the wire being split. A coordinate where wires of separate nets cross is
    refused, because a wire ending there would join all of them.
    """
    if isinstance(endpoint, str):
        coord = resolve_pin(endpoint, editor)
        return coord, coord
    coord = (endpoint.x, endpoint.y)
    if coord in pin_coords:
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
            f"{_endpoint_name(endpoint)} touches no wire and no component pin. A "
            "coordinate endpoint must land on existing wiring; to turn a corner at "
            "a free point, give it as a waypoint instead."
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
    validation failure (zero-length route, diagonal segment, pin
    collision, wire-junction overlap, named-net short, or a waypoint or
    route segment touching another net's wiring).

    Backs the ``wire_pins`` op, so a route the planner refuses is never
    written by any caller.
    """
    component_geo = collect_component_geometry(editor)
    existing_wires = wire_segments_of(editor)
    part_before = net_partition(editor)
    net_of = label_folded_nets(part_before)
    pin_coords = part_before.pin_owners.keys()
    from_name, to_name = _endpoint_name(from_pin), _endpoint_name(to_pin)

    (x1, y1), from_anchor = _resolve_route_endpoint(
        from_pin, editor, existing_wires, pin_coords, net_of
    )
    (x2, y2), to_anchor = _resolve_route_endpoint(
        to_pin, editor, existing_wires, pin_coords, net_of
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

    endpoints = {(x1, y1), (x2, y2)}
    skip_refs = {
        ref.rsplit(".", 1)[0]
        for ref in (from_pin, to_pin)
        if isinstance(ref, str) and "." in ref and not ref.startswith("net:")
    }

    errors: list[str] = []
    warnings: list[str] = []

    # Net-label conflict — checked first because it's a "wrong intent"
    # error: rejecting it gives the user a clearer signal than a route
    # geometry complaint. Skip when either side uses ``net:`` form (those
    # are already named explicitly). A coordinate endpoint on a wire's interior
    # reads the labels of that wire's net. Two-phase check:
    #   1) BEFORE state — endpoints resolve to two different already-named
    #      nets (the standard short).
    #   2) AFTER state — proposed route drags a mid-segment label into
    #      the union, merging an additional named net.
    if not any(isinstance(ep, str) and ep.startswith("net:") for ep in (from_pin, to_pin)):
        nets_before = labels_per_coord(part_before)
        from_labels_before = named_labels(_net_label_at(nets_before, from_anchor))
        to_labels_before = named_labels(_net_label_at(nets_before, to_anchor))
        if (
            from_labels_before
            and to_labels_before
            and from_labels_before.isdisjoint(to_labels_before)
        ):
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
        if len(unioned) >= 2:
            # Some labels seen post-route weren't there pre-route on
            # either endpoint — that's the mid-segment case.
            unioned_before = from_labels_before | to_labels_before
            new_labels = unioned - unioned_before
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
    # a straight wire) is dropped by LTspice at netlist time (see
    # same_instance_dropped_segments), so reporting the tie as connected would
    # be a lie. The fix is a route change, so surface it before route geometry.
    dropped = same_instance_dropped_segments(part_before.pin_owners, segments)
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

    for sx1, sy1, sx2, sy2 in segments:
        if sx1 != sx2 and sy1 != sy2:
            errors.append(f"Diagonal wire ({sx1},{sy1})->({sx2},{sy2}): not orthogonal")

    # Pin-collision check: a pin is safe if it's already wired to the
    # same net as our target (an existing wire reaches both that pin and
    # one of our endpoints), e.g. T-junction onto a power rail.
    def _pin_on_target_net(px: int, py: int) -> bool:
        for ex1, ey1, ex2, ey2 in existing_wires:
            wire_pts = {(ex1, ey1), (ex2, ey2)}
            if (px, py) in wire_pts and wire_pts & endpoints:
                return True
        return False

    # Pin-collision exemption is by exact endpoint *coordinate* (the
    # ``(px, py) in endpoints`` check below), NOT by whole component: the
    # OTHER pin of an endpoint component still lies on the route and must be
    # flagged — otherwise a waypoint landing on it silently shorts the
    # component while wire_pins reports success. ``skip_refs`` stays in the
    # bbox-crossing *warning* loop, where exempting an endpoint component is
    # reasonable.
    junctions: list[dict[str, object]] = []
    for cg in component_geo:
        for pin in cg["pins"]:
            px, py = pin["x"], pin["y"]
            if (px, py) in endpoints:
                continue
            # A pin at the shared corner of two consecutive segments satisfies
            # point_on_segment for both, so test the route once per pin.
            if not wires_through((px, py), segments):
                continue
            pin_label = f"{cg['ref']}.{pin['name']}"
            if _pin_on_target_net(px, py):
                junctions.append({"x": px, "y": py, "via": "pin", "pin": pin_label})
                warnings.append(
                    f"The route passes over {pin_label} at ({px},{py}), already wired "
                    "to this net; LTspice joins it there."
                )
                continue
            errors.append(
                f"Wire passes through {pin_label} at ({px},{py}): "
                "will create unintended connection"
            )

    # Wire-junction check: forbid overlaps with existing wires unless the
    # existing wire already terminates at one of our endpoints (intended
    # T-junction). A plain crossing, where neither wire ends, is only
    # reported: LTspice leaves it unjoined.
    flagged: set[int] = set()
    for sx1, sy1, sx2, sy2 in segments:
        for ext_index, (ex1, ey1, ex2, ey2) in enumerate(existing_wires):
            ext_endpoints = {(ex1, ey1), (ex2, ey2)}
            if ext_endpoints & endpoints:
                continue
            if sx1 == sx2 and ex1 == ex2 and sx1 == ex1:
                new_min, new_max = min(sy1, sy2), max(sy1, sy2)
                ext_min, ext_max = min(ey1, ey2), max(ey1, ey2)
                if new_min < ext_max and new_max > ext_min:
                    overlap_y = max(new_min, ext_min)
                    if (sx1, overlap_y) not in endpoints:
                        flagged.add(ext_index)
                        errors.append(
                            f"Wire overlap at x={sx1} between y={max(new_min, ext_min)} "
                            f"and y={min(new_max, ext_max)}: will create unintended junction"
                        )
                        break
            elif sy1 == sy2 and ey1 == ey2 and sy1 == ey1:
                new_min, new_max = min(sx1, sx2), max(sx1, sx2)
                ext_min, ext_max = min(ex1, ex2), max(ex1, ex2)
                if new_min < ext_max and new_max > ext_min:
                    overlap_x = max(new_min, ext_min)
                    if (overlap_x, sy1) not in endpoints:
                        flagged.add(ext_index)
                        errors.append(
                            f"Wire overlap at y={sy1} between x={max(new_min, ext_min)} "
                            f"and x={min(new_max, ext_max)}: will create unintended junction"
                        )
                        break
            elif sx1 == sx2 and ey1 == ey2:
                cross_x, cross_y = sx1, ey1
                new_min, new_max = min(sy1, sy2), max(sy1, sy2)
                ext_min, ext_max = min(ex1, ex2), max(ex1, ex2)
                if (
                    new_min < cross_y < new_max
                    and ext_min < cross_x < ext_max
                    and (cross_x, cross_y) not in endpoints
                ):
                    warnings.append(_crossing_warning(cross_x, cross_y, existing_wires[ext_index]))
            elif sy1 == sy2 and ex1 == ex2:
                cross_x, cross_y = ex1, sy1
                new_min, new_max = min(sx1, sx2), max(sx1, sx2)
                ext_min, ext_max = min(ey1, ey2), max(ey1, ey2)
                if (
                    ext_min < cross_y < ext_max
                    and new_min < cross_x < new_max
                    and (cross_x, cross_y) not in endpoints
                ):
                    warnings.append(_crossing_warning(cross_x, cross_y, existing_wires[ext_index]))

    # An endpoint on a wire's interior is a T-junction onto that wire. Its leg
    # must leave the wire: one running along it overlaps the wire it joins,
    # which the overlap check above exempts because it starts at an endpoint.
    for endpoint, coord, leg in (
        (from_pin, (x1, y1), segments[0]),
        (to_pin, (x2, y2), segments[-1]),
    ):
        if isinstance(endpoint, str) or coord in pin_coords:
            continue
        for ext_index, wire in enumerate(existing_wires):
            if ext_index in flagged or not point_on_segment(coord, wire[:2], wire[2:]):
                continue
            if _collinear_overlap(leg, wire):
                flagged.add(ext_index)
                errors.append(
                    f"Wire overlap: the leg from {_endpoint_name(endpoint)} runs along "
                    f"the wire {segment_text(wire)} it ends on; leave that wire at a "
                    "right angle"
                )
            elif coord not in {wire[:2], wire[2:]}:
                junctions.append(
                    {"x": coord[0], "y": coord[1], "via": "endpoint", "wire": segment_json(wire)}
                )

    # Contact check. LTspice joins a wire wherever another wire's end, a pin or
    # a label touches it (see net_partition), so a waypoint on existing wiring,
    # or a route passing through an existing wire's end or a lone label, joins
    # the route there as an endpoint would. Onto a net the route already joins
    # that is a redundant junction, reported; onto any other net it would merge
    # a net nobody named, so it is refused, pointing at the coordinate endpoint
    # that makes the same T on purpose. Pins are the pin-collision check's, and
    # a wire the overlap check already refused is not reported twice.
    endpoint_nets = {net_of(from_anchor), net_of(to_anchor)}
    vertices = [v for v in points[1:-1] if v not in endpoints and v not in pin_coords]
    # (point, net it touches) -> (kind, the wire touched, or None for a label).
    # One entry per point and net, the first found, in the order found.
    contacts: dict[
        tuple[tuple[int, int], tuple[int, int]], tuple[str, tuple[int, int, int, int] | None]
    ] = {}
    for ext_index, wire in enumerate(existing_wires):
        if ext_index in flagged:
            continue
        touched = net_of(wire[:2])
        for v in vertices:
            if point_on_segment(v, wire[:2], wire[2:]):
                contacts.setdefault((v, touched), ("waypoint", wire))
        for end in (wire[:2], wire[2:]):
            if end in endpoints or end in vertices or end in pin_coords:
                continue
            if wires_through(end, segments):
                contacts.setdefault((end, touched), ("wire_end", wire))
    for coord in sorted(part_before.label_texts):
        if coord in endpoints or coord in pin_coords or not wires_through(coord, segments):
            continue
        if not wires_through(coord, existing_wires):
            contacts.setdefault((coord, net_of(coord)), ("label", None))

    def _describe_net(net: tuple[int, int]) -> str:
        on_net = net_members(part_before, net_of, net)
        labels = sorted({t for c in on_net for t in part_before.label_texts.get(c, ())})
        if labels:
            return "net " + ", ".join(f"'{t}'" for t in labels)
        pins = sorted(
            f"{ref}.{name}" for c in on_net for ref, name in part_before.pin_owners.get(c, ())
        )
        if pins:
            more = " and others" if len(pins) > 3 else ""
            return "the net of " + ", ".join(pins[:3]) + more
        return "a wire no pin or label is on"

    for ((cx, cy), touched), (via, wire) in contacts.items():
        entry: dict[str, object] = {"x": cx, "y": cy, "via": via}
        if wire is None:
            texts = sorted(part_before.label_texts[(cx, cy)])
            what = "the net label " + ", ".join(f"'{t}'" for t in texts) + f" at ({cx},{cy})"
            entry["label"] = texts[0]
        else:
            entry["wire"] = segment_json(wire)
            what = (
                f"the wire {segment_text(wire)} at waypoint ({cx},{cy})"
                if via == "waypoint"
                else f"the end of the wire {segment_text(wire)} at ({cx},{cy})"
            )
        if touched in endpoint_nets:
            junctions.append(entry)
            warnings.append(
                f"The route touches {what}, already on {_describe_net(touched)}; "
                "LTspice joins them there."
            )
            continue
        remedy = (
            f"To join it on purpose, end a route there with "
            f'{{"x": {cx}, "y": {cy}}} as from_pin or to_pin; otherwise move the '
            "waypoint off it."
            if via == "waypoint"
            else "Reroute around it."
        )
        errors.append(
            f"Route touches {what}, on {_describe_net(touched)}: LTspice joins "
            f"wiring wherever it touches, so this would merge that net. {remedy}"
        )

    if errors:
        error_lines = [f"Refused to connect {from_name} to {to_name}:"]
        for e in errors:
            error_lines.append(f"  {e}")
        error_lines.append("\nFix the route with different waypoints to avoid these issues.")
        raise NetlistError("\n".join(error_lines))

    total_length = sum(abs(sx2 - sx1) + abs(sy2 - sy1) for sx1, sy1, sx2, sy2 in segments)
    if total_length > 400:
        warnings.append(
            f"Long wire run ({total_length} units): consider placing components closer "
            "or adding a local net label"
        )

    for sx1, sy1, sx2, sy2 in segments:
        for bb in component_geo:
            if bb["ref"] in skip_refs:
                continue
            bx, by, bw, bh = bb["x"], bb["y"], bb["width"], bb["height"]
            if sy1 == sy2:
                wy = sy1
                wx_min, wx_max = min(sx1, sx2), max(sx1, sx2)
                if by < wy < by + bh and wx_min < bx + bw and wx_max > bx:
                    warnings.append(
                        f"Wire at y={wy} crosses {bb['ref']} bounding box "
                        f"({bx},{by})-({bx + bw},{by + bh})"
                    )
            elif sx1 == sx2:
                wx = sx1
                wy_min, wy_max = min(sy1, sy2), max(sy1, sy2)
                if bx < wx < bx + bw and wy_min < by + bh and wy_max > by:
                    warnings.append(
                        f"Wire at x={wx} crosses {bb['ref']} bounding box "
                        f"({bx},{by})-({bx + bw},{by + bh})"
                    )

    return _ConnectPlan(x1, y1, x2, y2, points, segments, warnings, junctions)


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
    value: str = Field(description="New value or model name, e.g. '10k', '1u', 'BSS123'.")


class OpSetComponentAttribute(StrictModel):
    """Set one named symbol attribute on an existing component."""

    op: Literal["set_component_attribute"]
    reference: str = Field(description=_REFERENCE_DESCRIPTION)
    attribute: str = Field(
        description="Attribute name, e.g. Value, Value2, SpiceLine, SpiceModel."
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
            "'net:NAME' for a label, or {x, y} on a wire or pin; a wire's "
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
        default=None, description="Default: linear for tran, log for ac."
    )
    y_scale: YScale | None = Field(
        default=None, description="Left Y axis. Default: linear for tran, db for ac."
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


# The keys of an op's result that report what applying it found on the sheet:
# the segments a route found already drawn, the junctions it made, how much a
# removal took. The rest of a result restates the op or carries geometry and
# advisories a surface reports through views and warnings of its own. A route's
# ``wire_count`` is not one: ``already_present`` names the segments it did not
# draw, and relaying the count would put a line per route on every build. Nor
# is ``remove_directive``'s ``removed``, which names a kind, not an amount. An
# op missing here reports nothing beyond its arguments.
OP_RESULT_FACTS: dict[str, tuple[str, ...]] = {
    "wire_pins": ("already_present", "junctions"),
    "remove_wire": ("removed",),
    "remove_net_label": ("removed",),
    "remove_component": ("deleted_wires",),
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
    replaced = [
        {"traces": list(pane.traces), **scale_names(pane.scales)}
        for pane in (before.panes if before is not None else ())
    ]
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
        return {
            "op": "add_component",
            **_placed_component_data(
                editor,
                op.reference,
                op.symbol,
                op.x,
                op.y,
                op.rotation,
                symbol_info,
            ),
        }

    if isinstance(op, OpSetComponentValue):
        if op.reference not in editor.components:
            raise NetlistError(f"Component '{op.reference}' not found.")
        lint = level_label_lint(editor, op.reference, op.value)
        _apply_component_value(editor, op.reference, op.value, element_class(editor, op.reference))
        result = {"op": "set_component_value", "reference": op.reference, "value": op.value}
        if lint:
            result["warnings"] = [lint]
        return result

    if isinstance(op, OpSetComponentAttribute):
        _reject_unknown_attr(op.attribute)
        if op.reference not in editor.components:
            raise NetlistError(f"Component '{op.reference}' not found.")
        if not op.value.strip():
            # Empty value = clear. LTspice's format has no "empty value"
            # representation (a 2-token SYMATTR line is unreadable), so the
            # line is removed instead.
            _require_clearable_attr(op.reference, op.attribute)
            editor.get_component(op.reference).attributes.pop(op.attribute, None)
        else:
            editor.set_component_attribute(op.reference, op.attribute, op.value)
        return {
            "op": "set_component_attribute",
            "reference": op.reference,
            "attribute": op.attribute,
        }

    if isinstance(op, OpRemoveComponent):
        if op.reference not in editor.components:
            raise NetlistError(f"Component '{op.reference}' not found.")
        target_only = _component_pin_coords(editor, op.reference) - _other_components_pin_coords(
            editor, op.reference
        )
        editor.remove_component(op.reference)
        rm_result: dict[str, object] = {"op": "remove_component", "reference": op.reference}
        if op.cleanup_wires and target_only:
            rm_result["deleted_wires"] = _drop_wires_at(editor, target_only)
        elif target_only:
            # Same orphaned-wire warning the standalone handler surfaces.
            orphaned = _orphaned_wire_coords(editor, target_only)
            if orphaned:
                rm_result["warnings"] = [
                    f"orphaned wires remain at: {', '.join(orphaned)}. "
                    "Re-run with cleanup_wires=true to delete them."
                ]
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
        other_pins = _other_components_pin_coords(editor, op.reference)
        editor.set_component_position(op.reference, Point(op.x, op.y), new_rot)
        # Same bbox-overlap + orphaned-wire warnings as the standalone handler.
        mv_warnings = _move_component_warnings(
            editor, op.reference, new_rot.name, op.x, op.y, old_pins, other_pins
        )
        mv_result: dict[str, object] = {"op": "move_component", "reference": op.reference}
        if mv_warnings:
            mv_result["warnings"] = mv_warnings
        return mv_result

    if isinstance(op, OpAddNetLabel):
        x, y = _resolve_op_xy(op, editor)
        # Same short-refusal + duplicate/floating warnings as the standalone
        # handler — the op is the public path, so it enforces the same rules.
        warnings = _add_net_label_checks(editor, op.net, x, y)
        editor.labels.append(Text(coord=Point(x, y), text=op.net, type=TextTypeEnum.LABEL))
        result: dict[str, object] = {"op": "add_net_label", "net": op.net, "x": x, "y": y}
        if warnings:
            result["warnings"] = warnings
        return result

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
        if plan.warnings:
            result["warnings"] = plan.warnings
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


def collapse_result_warnings(results: list[dict[str, object]]) -> None:
    """Collapse identical per-op warnings across one ops batch, in place.

    The documented per-pin-label style repeats the same duplicate-label
    advisory on every add_net_label op of that net — hundreds of identical
    lines per converter-scale batch. Keep the first occurrence (annotated
    with the repeat count) and drop the copies.
    """
    counts: Counter[str] = Counter()
    for entry in results:
        warnings = entry.get("warnings")
        if isinstance(warnings, list):
            counts.update(warnings)
    if not counts or max(counts.values()) < 2:
        return
    emitted: set[str] = set()
    for entry in results:
        warnings = entry.get("warnings")
        if not isinstance(warnings, list):
            continue
        kept: list[str] = []
        for w in warnings:
            if w in emitted:
                continue
            emitted.add(w)
            n = counts[w]
            kept.append(
                w if n == 1 else f"{w} (identical warning on {n} ops in this batch; collapsed)"
            )
        if kept:
            entry["warnings"] = kept
        else:
            del entry["warnings"]


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
