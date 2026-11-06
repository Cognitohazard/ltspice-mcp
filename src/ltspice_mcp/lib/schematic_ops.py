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
- the placement geometry and the sheet's nets (``placed_geometry``,
  ``resolve_pin``, ``sheet_nets``), which reads each
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
from collections import Counter, defaultdict
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
from spicelib.editor.asy_reader import AsyReader
from spicelib.editor.base_schematic import (
    ERotation,
    Line,
    Point,
    SchematicComponent,
    Text,
    TextTypeEnum,
)
from spicelib.editor.ltspice_utils import TEXT_REGEX, asc_text_align_set
from spicelib.utils.detect_encoding import EncodingDetectError

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
from ltspice_mcp.lib.asc_document import is_asc, parse_asc
from ltspice_mcp.lib.component_value import POSITIONAL_KINDS
from ltspice_mcp.lib.connectivity import (
    NamedNets,
    build_on_wire_predicate,
    is_ground,
    name_key,
    name_keys,
    named_labels,
    named_nets,
    same_instance_dropped_segments,
)
from ltspice_mcp.lib.encoding import refused_sheet_mark, refused_sheet_mark_note
from ltspice_mcp.lib.filelock import circuit_file_lock, path_lock
from ltspice_mcp.lib.format import is_scaled_number, parse_spice_value
from ltspice_mcp.lib.geometry import BBox
from ltspice_mcp.lib.models import StrictModel
from ltspice_mcp.lib.nets import PinAt
from ltspice_mcp.lib.pin_legend import pin_ties
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
from ltspice_mcp.lib.schematic_scene import decode_text_lines
from ltspice_mcp.lib.sheet import (
    INSTANCE_PREFIX,
    PARAMETER_ATTRIBUTES,
    PlacedPin,
    element_letter,
    missing_part,
    placed_pin,
    references_named,
)
from ltspice_mcp.lib.sheet_findings import Part, SheetView
from ltspice_mcp.lib.spice_lex import (
    SpiceCard,
    SpiceLexError,
    Token,
    TokenKind,
    key_value_tokens,
    tokenize_body,
)
from ltspice_mcp.lib.spice_validator import (
    validate_directive,
)
from ltspice_mcp.lib.symbol_geometry import SymbolInfo, compute_placed_geometry, usable_symbol_info
from ltspice_mcp.lib.symbol_library import SymbolLibrary, SymbolResolver
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


def _prefix_text(reference: str, value: str) -> str:
    """The text a ``Prefix`` is written with: it must start with the element
    letter it gives the part, and is written in upper case, the way spicelib
    reads it back; LTspice reads either case the same."""
    letter = value.strip()[:1]
    if not (letter.isascii() and letter.isalpha()):
        raise NetlistError(
            f"Prefix {value!r} on {reference!r} does not start with an element "
            "letter: give the letter the part netlists as, such as 'X' for a "
            "subcircuit."
        )
    return value.strip().upper()


def _attribute_texts(
    editor: AscEditor,
    part: SchematicComponent,
    given: Mapping[str, str],
    *,
    cleared: Container[str] = (),
) -> dict[str, str]:
    """The text each attribute in ``given`` is written with on ``part``, a part
    on ``editor``'s sheet or one about to be placed on it, in ``given``'s order.
    ``cleared`` names the attributes the same op removes from the part.

    Every op that writes or clears a part's attributes takes their text from
    here (``add_component``, ``set_component_value``,
    ``set_component_attribute``), so a text one op refuses no other op writes:

    - a ``Prefix`` is written as ``_prefix_text`` gives it;
    - the part's ``Value`` once the op is done, the one in ``given`` or else
      its own, must be one LTspice reads whole as the value of the element
      class the part then netlists as (``_validate_component_value``): the
      class given by the ``Prefix`` in ``given``, else by the part's own
      unless ``cleared`` removes it, else by its symbol's. A ``Prefix``
      written or cleared changes that class, so it holds the part's own Value
      to the new class.

    Raises NetlistError for a text the part cannot be given.
    """
    texts = {
        name: _prefix_text(part.reference, text) if name == INSTANCE_PREFIX else text
        for name, text in given.items()
    }
    touched = {*texts, *(name for name in ("Value", INSTANCE_PREFIX) if name in cleared)}
    if not touched & {"Value", INSTANCE_PREFIX}:
        return texts
    own = getattr(part, "attributes", None) or {}
    after = {
        name: str(own[name])
        for name in ("Value", INSTANCE_PREFIX)
        if name in own and name not in cleared
    }
    after.update(texts)
    if "Value" not in after:
        return texts
    element = _element_of(editor, part, part.reference, after.get(INSTANCE_PREFIX))
    try:
        _validate_component_value(part.reference, after["Value"], element)
    except NetlistError as exc:
        if "Value" in texts:
            raise
        change = (
            f"set Prefix {texts[INSTANCE_PREFIX]!r} on"
            if INSTANCE_PREFIX in texts
            else "clear the Prefix of"
        )
        raise NetlistError(
            f"Refused to {change} {part.reference!r}: it would netlist as a "
            f"{element}-class part, and its Value does not read as one. {exc} "
            "Give it a Value that class reads (set_component_value) in an op "
            "ahead of this one."
        ) from exc
    return texts


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
    # ``value`` is the Value attribute; a Value in ``attributes`` as well
    # replaces it, in its place. An empty value writes the 2-token SYMATTR
    # line that corrupts the .asc on the next read.
    given: dict[str, str] = {}
    if value is not None:
        _reject_empty_attr_value(reference, "Value", value)
        given["Value"] = value
    for attr_name, attr_val in (attributes or {}).items():
        _reject_unknown_attr(attr_name)
        _reject_empty_attr_value(reference, attr_name, attr_val)
        given[attr_name] = attr_val
    # Every text is checked before the part reaches the sheet.
    comp.attributes.update(_attribute_texts(editor, comp, given))
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


def _head_then_parameters(tokens: Sequence[Token]) -> bool:
    """Whether ``tokens`` read as ``[MODEL] KEY=VALUE ...``: at most one token,
    then one or more ``KEY=VALUE`` tokens and nothing else."""
    first = next(
        (index for index, tok in enumerate(tokens) if tok.kind == TokenKind.KEY_VALUE),
        len(tokens),
    )
    return (
        first <= 1
        and first < len(tokens)
        and all(tok.kind == TokenKind.KEY_VALUE for tok in tokens[first:])
    )


def _validate_component_value(reference: str, value: str, element: str) -> None:
    """Reject values that would corrupt the netlist line on write.

    LTspice writes the Value verbatim after the symbol's pins. A space in a
    single-token value (a resistor's ``1 k``, a subcircuit's ``opamp 2``)
    splits it into two tokens, and the netlist reader then takes the first as
    another node. ``element`` is the element class letter the part netlists
    as (``element_class``); the shapes that cannot do that are accepted:

    - SPICE expressions in braces (``{1/(2*pi*RC)}``) and quoted strings;
    - ``[MODEL] KEY=VALUE ...`` parameter lists, split by the lexer as
      ``_apply_component_value`` splits them;
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
    # ``[MODEL_NAME] KEY=VALUE [KEY=VALUE ...]`` is valid: at most one head
    # token (the model name) followed by a non-empty list of KEY=VALUE tokens,
    # as the lexer splits them, which is the split ``_apply_component_value``
    # writes: a braced parameter value with spaces in it is one token.
    if well_formed and _head_then_parameters(toks):
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


def _reference_taken(
    editor: AscEditor, reference: str, *, other_than: str | None = None
) -> str | None:
    """The reference of a part, other than ``other_than``, that ``reference``
    names to LTspice (``sheet.references_named``); None when no part has it.

    LTspice compares references without case: neither build netlists a sheet
    with parts named ``R1`` and ``r1`` (recorded as
    ``connectivity/shared_reference_case``).
    """
    others = [name for name in editor.components if name != other_than]
    return next(iter(references_named(reference, others)), None)


def part_reference(editor: AscEditor, reference: str) -> str:
    """The reference of the part ``reference`` names on the editor's sheet
    (``sheet.references_named``): the one spelled so, else the one whose
    reference differs from it only in case.

    Raises NetlistError naming the parts there are when it names none, and
    naming each when it names several, which differ from it only in case.
    """
    named = references_named(reference, editor.components)
    if not named:
        raise missing_part(reference, editor.components)
    if len(named) > 1:
        raise NetlistError(
            f"'{reference}' names {len(named)} parts whose references differ only in case "
            f"({', '.join(sorted(named))}); give the reference as one of them spells it."
        )
    return named[0]


def _same_reference_note(asked: str, taken: str) -> str:
    """What to add when ``asked`` differs from ``taken`` only in case."""
    if asked == taken:
        return ""
    return f" LTspice compares references without case, so '{asked}' names it too."


def _rename_component(editor: AscEditor, reference: str, new: str) -> dict[str, object]:
    """Give the part ``reference`` the name ``new``, in place.

    spicelib keeps a part under its reference and writes that as its
    ``InstName``, so setting the attribute alone would leave the part under
    its old name and write both names (docs/spicelib_bugs.md, Bug 32).
    """
    result: dict[str, object] = {
        "op": "set_component_attribute",
        "reference": reference,
        "attribute": "InstName",
    }
    if new == reference:
        return result
    if (taken := _reference_taken(editor, new, other_than=reference)) is not None:
        raise NetlistError(
            f"Refused to rename {reference} to {new}: '{taken}' already names a part."
            + _same_reference_note(new, taken)
        )
    parts = list(editor.components.items())
    editor.components.clear()
    for name, part in parts:
        if name == reference:
            part.reference = new
            part.attributes.pop("InstName", None)
            name = new
        editor.components[name] = part
    result["changed"] = {"InstName": {"was": reference, "now": new}}
    return result


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


def _apply_component_value(editor, reference: str, value: str) -> None:
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
    _attribute_texts(editor, editor.get_component(reference), {"Value": value})
    element = element_class(editor, reference)
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


# The resolver each editor's sheet finds its symbols with, for one request:
# get_asc_editor attaches a fresh resolver from the session's library to every
# editor it hands out, so a symbol redrawn or added between requests is read
# again, while the many geometry passes one request makes over a sheet cost one
# lookup per distinct symbol (the resolver keeps what it found) rather than a
# file check per part per pass.
_resolver_of_editor: WeakKeyDictionary[AscEditor, SymbolResolver] = WeakKeyDictionary()


def attach_symbols(editor: AscEditor, resolver: SymbolResolver) -> AscEditor:
    """Have ``editor``'s pin geometry find its symbols with ``resolver``."""
    _resolver_of_editor[editor] = resolver
    return editor


def symbols_of(editor: AscEditor) -> SymbolResolver:
    """The resolver ``editor``'s sheet finds its symbols with (``attach_symbols``)."""
    resolver = _resolver_of_editor.get(editor)
    if resolver is None:
        raise RuntimeError(
            "This schematic editor has no symbol library attached; open it with "
            "get_asc_editor, or attach one with attach_symbols."
        )
    return resolver


def symbol_info_for(editor: AscEditor, symbol: str) -> SymbolInfo | None:
    """``symbol`` as ``editor``'s sheet resolves it: beside the sheet, then the
    session's libraries (``attach_symbols``).

    ``None`` for a symbol no file is found for, and for one that cannot be used
    (a pin line that does not read leaves a terminal no wire could be checked
    against).
    """
    return usable_symbol_info(symbols_of(editor).load(symbol))


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
    part = editor.components.get(reference)
    return _element_of(
        editor, part, reference, (getattr(part, "attributes", None) or {}).get(INSTANCE_PREFIX)
    )


def _element_of(editor: AscEditor, part: object, reference: str, own_prefix: str | None) -> str:
    """The element letter ``part`` netlists as (``sheet.element_letter``), on
    ``editor``'s sheet or about to be placed on it, with ``own_prefix`` as its
    own ``Prefix``, the one it has or is about to have: from that, else from
    its symbol's, else from ``reference``."""
    symbol = getattr(part, "symbol", None)
    info = symbol_info_for(editor, symbol) if symbol else None
    return element_letter(own_prefix, info.prefix if info is not None else None, reference)


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


def sheet_nets(
    editor: AscEditor,
    extra_segments: Sequence[tuple[int, int, int, int]] = (),
    without: str | None = None,
    geometry: Iterable[dict] | None = None,
) -> NamedNets:
    """The sheet's nets as LTspice netlists them (``connectivity.named_nets``).

    The joining rules, and the recordings each comes from, are in
    ``lib/connectivity.py``; this collects what ``editor`` holds and hands it
    over, each part's pins in SpiceOrder and each part told by its place among
    the parts.

    ``extra_segments`` lets the caller include not-yet-committed wire
    segments (e.g. the route ``wire_pins`` is about to add) so checks operate
    on the post-route net layout. ``without`` leaves one part's pins out: the
    sheet as it is apart from a part about to be placed or moved.
    ``geometry`` is the parts' geometry when the caller already holds it.
    """
    parts = [
        (entry["ref"], [(pin["name"], (pin["x"], pin["y"])) for pin in entry["pins"]])
        for entry in (collect_component_geometry(editor) if geometry is None else geometry)
        if entry["ref"] != without
    ]
    labels = [((int(lbl.coord.X), int(lbl.coord.Y)), lbl.text) for lbl in editor.labels]
    segments = [*wire_segments_of(editor), *extra_segments]
    return named_nets(parts, labels, [((x1, y1), (x2, y2)) for x1, y1, x2, y2 in segments])


def _segment_key(
    a: tuple[int, int], b: tuple[int, int]
) -> tuple[tuple[int, int], tuple[int, int]]:
    """One segment's identity, orientation-independent."""
    return (a, b) if a <= b else (b, a)


def _wire_segment_keys(editor: AscEditor) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    return [_segment_key(seg[:2], seg[2:]) for seg in wire_segments_of(editor)]


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
    segments = wire_segments_of(editor)
    label_coords = {(int(lbl.coord.X), int(lbl.coord.Y)) for lbl in editor.labels}
    ties = [
        tie
        for _part, _index, tie in pin_ties(
            collect_component_geometry(editor),
            is_wired=build_on_wire_predicate(((x1, y1), (x2, y2)) for x1, y1, x2, y2 in segments),
            is_labeled=label_coords.__contains__,
        )
    ]
    return {
        "wire_segments": len(segments),
        "pins_total": len(ties),
        "pins_wired": ties.count("wired"),
        "pins_label_only": ties.count("label"),
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


class _PlacedSymbol(AsyReader):
    """A part's symbol as the editor read it, with the name the sheet gives it."""

    def __init__(self, path: Path, placed_as: str) -> None:
        super().__init__(path)
        self.placed_as = placed_as


class _AscEditor(AscEditor):
    """spicelib's editor, finding a sheet's files as the session does and opening
    a sheet whose block symbol has no sheet.

    A part's symbol is the file ``symbols`` resolves its name to
    (``symbol_library.find_symbol``), the one ``inspect``, ``verify_circuit``
    and the drawing read, and a block's own sheet the one it finds for the
    block (``SymbolResolver.block_sheet``). spicelib's own searches walked
    every folder under the sheet's and the folder the server runs in, so they
    found a file where neither LTspice build looks; kept the folder a name
    gives, so they missed a symbol XVII finds beside the sheet; and cached
    each symbol's file for the whole process (``docs/spicelib_bugs.md``, Bugs
    17 and 37).

    LTspice netlists an instance of a block symbol as a call to a subcircuit
    of the symbol's name, defined by its own sheet, by a library on the sheet,
    or not yet at all. spicelib's loader requires the sheet and refuses to open
    the parent without it (``docs/spicelib_bugs.md``, Bug 22). Here such an
    instance loads with no resolved subcircuit, as spicelib already loads a
    cell symbol with no library, and a sheet that is there opens as one of
    these, so a block nested further down is read the same way. A block symbol
    that gives a Prefix is the element it names, as a cell is, and LTspice
    reads no sheet for it even beside it (``export-hierarchical-block``;
    ``docs/spicelib_bugs.md``, Bug 33). A comment or directive drawn on its
    side stays on its side, which spicelib's reader drops (Bug 34).

    A symbol that names a model library, a cell's or a block's, loads with no
    resolved subcircuit either. Nothing that reads or edits a sheet here needs
    the library, which a deck reads when it runs. spicelib looked for it as it
    looked for symbols, refused the sheet where it found none, and where it
    found one could not save the sheet (Bug 39).
    """

    def __init__(
        self, asc_file: str | Path, encoding: str = "autodetect", *, symbols: SymbolResolver
    ) -> None:
        self._symbols = symbols
        super().__init__(asc_file, encoding)

    def _get_symbol(self, symbol: str) -> AsyReader:
        found = self._symbols.resolve(symbol)
        if found is None:
            raise FileNotFoundError(
                f"Symbol '{symbol}' not found beside the sheet or in any symbol library"
            )
        return _PlacedSymbol(found, symbol)

    def reset_netlist(self, create_blank: bool = False) -> None:
        super().reset_netlist(create_blank)
        # spicelib keeps a text for each TEXT line its pattern matches, in
        # order, but its pattern leaves the V of VLeft, VRight and the rest out
        # of the justification it reads, so every one is read upright. Each is
        # given its whole justification again.
        with open(self.asc_file_path, encoding=self.encoding) as asc_file:
            justifications = [
                line.split()[3]
                for line in asc_file
                if line.startswith("TEXT") and TEXT_REGEX.match(line)
            ]
        for text, justification in zip(self.directives, justifications, strict=True):
            asc_text_align_set(text, justification)

    def _get_subcircuit(self, symbol: Any) -> Any:
        """What an instance of ``symbol`` (a ``_PlacedSymbol``) calls: a block's
        own sheet, opened as one of these, or nothing resolved."""
        if symbol.symbol_type not in ("CELL", "BLOCK"):
            # spicelib's refusal of such a part, which it makes before it looks
            # for any file, kept as it was.
            raise ValueError(f"Symbol type {symbol.symbol_type} not supported")
        if (
            symbol.symbol_type == "CELL"
            or symbol.get_library() is not None
            or INSTANCE_PREFIX in symbol.attributes
        ):
            return None
        sheet = self._symbols.block_sheet(symbol.placed_as)
        if sheet is None:
            return None
        return type(self)(sheet, symbols=self._symbols.for_sheet(sheet))


def _refuse_lost_records(path: Path, editor: AscEditor) -> None:
    """Refuse a sheet the editor holds fewer records of than the file has.

    An edit is committed from what the editor read, so a record it did not
    read would be deleted by any edit. spicelib keeps one part per reference
    (docs/spicelib_bugs.md, Bug 31), so a sheet on which two parts share one
    opens with only the last of them. Neither LTspice build netlists such a
    sheet either (recorded as ``connectivity/shared_reference``). spicelib also
    skips, without a word, a ``TEXT`` line its pattern does not match, such as
    one with two spaces after the keyword (Bug 36). How LTspice reads that line
    is not recorded; the sheet's own reader (``asc_document``) reads it, so
    ``inspect`` and ``verify_circuit`` show it, and it is refused here rather
    than lost.
    """
    document = parse_asc(path.read_bytes())
    placed = document.symbols
    if len(placed) != len(editor.components):
        by_reference: dict[str, list[tuple[int, int]]] = {}
        for symbol in placed:
            by_reference.setdefault(symbol.reference, []).append((symbol.x, symbol.y))
        shared = "; ".join(
            f"'{reference}' at " + ", ".join(f"({x},{y})" for x, y in places)
            for reference, places in by_reference.items()
            if len(places) > 1
        )
        raise NetlistError(
            f"Cannot open {path.name} for editing: parts share a reference ({shared}). "
            "The schematic editor keeps one part per reference and would delete the "
            "others on saving, and neither LTspice build netlists such a sheet. "
            "Rename the repeated parts in LTspice, then edit the sheet here."
        )
    held = {"TEXT": editor.directives, "WIRE": editor.wires, "FLAG": editor.labels}
    read = {"TEXT": document.texts, "WIRE": document.wires, "FLAG": document.flags}
    lost = [keyword for keyword in held if len(held[keyword]) != len(read[keyword])]
    if not lost:
        return
    first, number = lost[0], 1
    for record in document.records:
        line = (record.source or ("",))[0]
        if first == "TEXT" and line.startswith("TEXT") and not TEXT_REGEX.match(line):
            raise NetlistError(
                f"Cannot open {path.name} for editing: line {number} "
                f"({line.rstrip()!r}) is a TEXT record the schematic editor skips, "
                "and saving any edit would delete it. Write it as LTspice saves one, "
                "with one space between its fields, then edit the sheet here.",
                show_hint=False,
            )
        number += len(record.source or ())
    raise NetlistError(
        f"Cannot open {path.name} for editing: the schematic editor skips "
        f"{len(read[first]) - len(held[first])} of its {len(read[first])} {first} records, "
        "and saving any edit would delete them.",
        show_hint=False,
    )


def make_editor(path: Path, symbols: SymbolResolver | None = None) -> Editor:
    """Create an AscEditor or SpiceEditor based on file extension.

    A sheet's symbols are found with ``symbols``, the session's resolver for
    it (``SymbolLibrary.for_sheet``); without one, with the libraries spicelib
    is configured with (``engine.py`` sets them from the session's), searched
    as the session searches them.

    Raises NetlistError if the file is not found, has no Version line the
    schematic editor can find, holds a line it cannot read or would skip, or
    places two parts under one reference, and
    SymbolResolutionError if a symbol it or a block's own sheet places is
    missing.
    """
    try:
        if not is_asc(path):
            return SpiceEditor(str(path))
        try:
            if symbols is None:
                library = SymbolLibrary(
                    [Path(folder) for folder in AscEditor.custom_lib_paths],
                    [Path(folder) for folder in AscEditor.simulator_lib_paths],
                )
                symbols = library.for_sheet(path)
            editor = _AscEditor(str(path), symbols=symbols)
        except NotImplementedError as e:
            raise _unreadable_record(path, e) from e
        except EncodingDetectError as e:
            raise _unrecognised_sheet(path, e) from e
        _refuse_lost_records(path, editor)
        return editor
    except FileNotFoundError as e:
        if not path.is_file():
            raise NetlistError(f"File not found: {path}") from e
        # The schematic itself opened, so what is missing is something it
        # refers to: a symbol it or a block's own sheet places.
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
    return state.editors.get(path, lambda p: make_editor(p, state.symbols.for_sheet(p)))


def get_asc_editor(path: Path, state: SessionState) -> AscEditor:
    """Get a cached AscEditor. Caller must have validated require_asc first."""
    editor = _get_editor(path, state)
    if not isinstance(editor, AscEditor):
        raise NetlistError(f"This operation requires an .asc schematic, got '{path.suffix}'. ")
    # This request resolves symbols afresh, from the session's library.
    return attach_symbols(editor, state.symbols.for_sheet(path))


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


def _remove_directive_or_comment(editor: AscEditor, instruction: str) -> None:
    """Remove every directive and comment ``instruction`` names (``_text_matcher``).

    Raises NetlistError when nothing matched, so the user can't think they
    cleaned a directive that's still in the file.
    """
    named = _text_matcher(instruction)
    kept = [text for text in editor.directives if not named(text)]
    if len(kept) < len(editor.directives):
        editor.directives[:] = kept
        return
    raise _no_text_matched(instruction)


def _no_text_matched(instruction: str) -> NetlistError:
    """The refusal of a text op whose ``instruction`` names no directive or comment."""
    if instruction.startswith("regex:"):
        return NetlistError(
            f"No directive or comment matched regex {instruction[6:]!r}. It is searched "
            "for in each TEXT line's text, in the case it is given, where a line break "
            "is a \\n escape."
        )
    return NetlistError(
        f"No directive or comment matched {instruction!r} exactly. "
        "Match is literal by default — pass 'regex:<pattern>' for regex "
        "matching, or copy the text of the sheet's TEXT line, after its '!' "
        "(directive) or ';' (comment); a line break may be given as a newline "
        "or as the \\n escape the file holds."
    )


def _shown_text(body: str) -> str:
    """A TEXT record's text as it reads, a line break as a line break: decoded
    as the drawing and the sheet findings decode it (``decode_text_lines``)."""
    return "\n".join(decode_text_lines(body))


def _text_kind(text: Text) -> str:
    return "comment" if text.type == TextTypeEnum.COMMENT else "directive"


def _refuse_unrunnable_directive(text: str) -> None:
    """Refuse a directive the pre-flight rules refuse, as every op that writes
    one does; a comment is not checked."""
    err = validate_directive(text, simulator="LTspice")
    if err is not None:
        raise NetlistError(f"Refusing directive {text!r}: {err.message} ({err.suggestion})")


def _listed_texts(texts: Sequence[Text], shown: int = 10) -> str:
    """The texts, each with its anchor, for a refusal; past ``shown`` it says how
    many more there are."""
    listed = "; ".join(
        f"{_shown_text(text.text)!r} at ({int(text.coord.X)}, {int(text.coord.Y)})"
        for text in texts[:shown]
    )
    more = len(texts) - shown
    return f"{listed}; and {more} more" if more > 0 else listed


# ---------------------------------------------------------------------------
# Handlers — schematic-only operations (.asc only)
# ---------------------------------------------------------------------------


def _xy(point: Point) -> tuple[int, int]:
    return int(point.X), int(point.Y)


def _text_matcher(instruction: str) -> Callable[[Text], bool]:
    """Whether a directive or comment is one ``instruction`` names: the one rule
    ``remove_directive`` and ``move`` select texts by, for both kinds of text.

    A literal names every text whose words equal it, space around either
    ignored, a line break given as a newline or as the ``\\n`` escape a TEXT
    record holds. ``regex:<pattern>`` names every text the pattern is found in
    (``re.search``), in the case it is given: ``(?i)`` ignores case and ``^``
    anchors it at the start of the text.
    """
    if not instruction.startswith("regex:"):
        body = _text_record_body(instruction.strip())
        return lambda text: text.text.strip() == body
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
    return lambda text: compiled.search(text.text) is not None


def _texts_matching(editor: AscEditor, wanted: str) -> list[int]:
    """Indices of the directives and comments ``wanted`` names, for ``move``."""
    named = _text_matcher(wanted)
    return [i for i, text in enumerate(editor.directives) if named(text)]


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
    references = [part_reference(editor, ref) for ref in op.references]
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

    parts = list(dict.fromkeys(references))
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
    after = named_nets(
        [
            (
                entry["ref"],
                [
                    (pin["name"], shifted((pin["x"], pin["y"]), entry["ref"] in moved_parts))
                    for pin in entry["pins"]
                ],
            )
            for entry in geometry.values()
        ],
        [
            (shifted(_xy(lbl.coord), i in plan.labels), lbl.text)
            for i, lbl in enumerate(editor.labels)
        ],
        ends,
    )
    # A net of the moved sheet, by its wiring, with two names on it joins them;
    # that is a short unless the sheet as it is already made them one node
    # (names compared as LTspice compares them, nets that share a name or a
    # ground already one).
    names_on: dict[int, set[str]] = defaultdict(set)
    for label in after.labels:
        names_on[after.nets.of_label(label)].update(named_labels([label.text]))
    before: NamedNets | None = None
    for names in names_on.values():
        if len(name_keys(names)) < 2:
            continue
        before = before or sheet_nets(editor, geometry=geometry.values())
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
    recording), as a label there does, and a pin on a wire LTspice drops,
    straight between two pins of one part, keeps the wire, joining those two
    (``same_instance_pin_inside``). Anywhere else a pin lands, on a wire's
    interior or end, a label or another part's pin, there is one net already,
    so those are the ways placing or moving a part makes one net of two. Two
    that each carry a name are a short, refused as a wire or a label that
    joined them is. Any other join is taken as meant, and the finished sheet
    says it (the ``joined_at_crossing`` rule).

    The sheet is read without the part itself, and ``already`` is where its
    pins are now: what a pin joins from where it already is, is not this
    op's doing. What each pin joins is read off the sheet with the part in
    its new place, since the part's own pins can leave a wire straight
    between two of them, which LTspice drops: a pin on a crossing then joins
    only the piece of wire on its side. Called before the part is placed or
    moved, so a refusal leaves the sheet as it was.
    """
    wires = wire_segments_of(editor)
    # Nearly every pin is on one wire's end or on none, and the sheet's nets
    # need not be worked out for those.
    on_wires = [
        (order, pin, through)
        for order, pin in enumerate(pins)
        if (point := (pin["x"], pin["y"])) not in already
        and (through := wires_through(point, wires))
        and (len(through) > 1 or point not in (through[0][:2], through[0][2:]))
    ]
    if not on_wires:
        return
    others = [entry for entry in collect_component_geometry(editor) if entry["ref"] != reference]
    nets = sheet_nets(editor, geometry=others)
    after = sheet_nets(editor, geometry=[*others, {"ref": reference, "pins": list(pins)}])
    for order, pin, through in on_wires:
        landed = PinAt((pin["x"], pin["y"]), len(others), order, reference, pin["name"])
        # Each net it lands on is known by the names that net carries itself
        # (NamedNets.own_names), as for a label.
        joined = sorted(
            {
                net
                for name in after.own_names_of(landed)
                if (net := nets.net_named(name)) is not None
            }
        )
        if len(joined) < 2:
            continue
        listed = "; ".join(str(sorted(nets.names(net))) for net in joined)
        if len(through) == 1:
            raise NetlistError(
                f"Refused to {doing}: {reference}.{pin['name']} would land at "
                f"({pin['x']},{pin['y']}) on the wire {segment_text(through[0])}, which "
                "runs straight between two pins of one part and which LTspice leaves out "
                "of the netlist. A pin on it keeps the wire, joining nets that each carry "
                f"a name ({listed}) and shorting them together. Put the part where the "
                "pin is on the one net it is meant for."
            )
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
    order that need not be their SpiceOrder. The match is ``sheet.placed_pin``,
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
    ref = part_reference(editor, ref)
    symbol = editor.components[ref].symbol
    geometry = placed_geometry(editor, ref)
    pins = [
        PlacedPin(pin["name"], pin["order"], pin["x"], pin["y"], pin["dir"])
        for pin in (geometry["pins"] if geometry is not None else ())
    ]
    pin = placed_pin(pins, ref, symbol or "", pin_name, "not found" if geometry is None else "")
    return pin.x, pin.y


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
    # net, and one at a crossing joins both (see lib/connectivity.py for the
    # export records). It is one node with whatever sits on its point, and with
    # every net already of its name; ground is no name (NamedNets.joined_at).
    grounding = is_ground(net)
    nets = sheet_nets(editor)
    joined = nets.joined_at((x, y), landing=True, name=None if grounding else net)
    if len(joined.named) > 1:
        listed = "; ".join(f"{sorted(names)}" for names in joined.named)
        if grounding and len(set(nets.wired_at((x, y)))) < 2:
            # Not a crossing: a wire LTspice drops between two pins of one
            # part, which a flag on it keeps.
            raise NetlistError(
                f"Refused to add ground at ({x},{y}): the flag would join nets that each "
                f"carry a name ({listed}), shorting them together and to ground. Put the "
                "flag on a point of the one net it is meant for."
            )
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
    nets: NamedNets,
    remedy: str,
) -> list[tuple[int, int, int, int]]:
    """:func:`wires_through` for a point that must name one net of ``nets``.

    Refuses, naming the wires and ending with ``remedy``, when the point is
    where wires of separate nets cross: LTspice leaves a plain crossing unjoined
    (``tests/fixtures/t_junctions/crossing_wires``), so it is on neither net
    alone. An empty list means the point touches no wire.
    """
    through = wires_through(coord, segments)
    if len(set(nets.wired_at(coord))) > 1:
        raise crossing_refused(coord, through, remedy)
    return through


def crossing_refused(
    coord: tuple[int, int], through: Sequence[tuple[int, int, int, int]], remedy: str
) -> NetlistError:
    """The refusal of a point where the wires ``through`` it, on separate nets,
    cross, naming them and ending with ``remedy``."""
    listed = "; ".join(segment_text(s) for s in through)
    return NetlistError(
        f"({coord[0]},{coord[1]}) is where wires on separate nets cross ({listed}); "
        f"LTspice leaves a plain crossing unjoined. {remedy}"
    )


def _endpoint_name(endpoint: "str | GridPoint") -> str:
    """A route endpoint as a message names it: as given, or as ``(x,y)``."""
    return endpoint if isinstance(endpoint, str) else f"({endpoint.x},{endpoint.y})"


def _resolve_route_endpoint(
    endpoint: "str | GridPoint",
    editor: AscEditor,
    existing_wires: list[tuple[int, int, int, int]],
    nets: NamedNets,
) -> tuple[int, int]:
    """Resolve one ``wire_pins`` endpoint, on the sheet whose nets are ``nets``,
    to its coordinate.

    A pin or label reference resolves through :func:`resolve_pin`. A coordinate
    must already touch the circuit: at a pin or a net label, or anywhere along
    a wire. An end there connects to it, and one on a wire's interior is a
    T-junction LTspice joins without the wire being split.
    A label's point is how a route reaches a label that stands on nothing, one
    of several labels of one name, or a ground, none of which ``net:NAME`` can
    name. A coordinate where wires of separate nets cross is refused, because a
    wire ending there would join all of them.
    """
    if isinstance(endpoint, str):
        return resolve_pin(endpoint, editor)
    coord = (endpoint.x, endpoint.y)
    if any(pin.point == coord for pin in nets.pins) or any(
        label.point == coord for label in nets.labels
    ):
        return coord
    through = wires_of_one_net(
        coord,
        existing_wires,
        nets,
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
    return coord


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
    before = sheet_nets(editor, geometry=component_geo)
    from_name, to_name = _endpoint_name(from_pin), _endpoint_name(to_pin)

    x1, y1 = _resolve_route_endpoint(from_pin, editor, existing_wires, before)
    x2, y2 = _resolve_route_endpoint(to_pin, editor, existing_wires, before)

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
    # endpoint on a wire's interior reads the names of that wire's net.
    # Two-phase check:
    #   1) BEFORE state — the endpoints are on two different already-named
    #      nets (the standard short).
    #   2) AFTER state — proposed route drags a mid-segment label into
    #      the union, merging an additional named net.
    # Nets are the ones LTspice makes, names compared as LTspice compares them
    # (name_key): a net named VOUT and one named Vout, or two names a wire
    # already joins, are one node already. An end's nets are known by the
    # names they carry themselves (NamedNets.named_at), so a wire grounded on
    # its own joins no name: it grounds what it is routed to.
    from_nets, to_nets = before.named_at((x1, y1)), before.named_at((x2, y2))
    from_labels_before = set().union(*(before.names(net) for net in from_nets))
    to_labels_before = set().union(*(before.names(net) for net in to_nets))
    named = sorted(from_nets | to_nets)
    if len(named) > 1:
        if from_labels_before and to_labels_before:
            joined = (
                f"{from_name} is on net {sorted(from_labels_before)} and {to_name} is on "
                f"net {sorted(to_labels_before)}. Connecting them would short the two "
                "named nets."
            )
        else:
            listed = "; ".join(str(sorted(before.names(net))) for net in named)
            joined = (
                f"the route would join nets that each carry a name ({listed}), shorting "
                "them together."
            )
        raise NetlistError(
            f"Refused to connect {from_name} to {to_name}: "
            f"Net-label conflict — {joined} To join them on purpose, give both the "
            f"same name: remove one side's labels with remove_net_label "
            f"and label it with the other's name; no wire is needed."
        )
    from_keys, to_keys = name_keys(from_labels_before), name_keys(to_labels_before)
    after = sheet_nets(editor, extra_segments=segments, geometry=component_geo)
    unioned = after.own_names((x1, y1)) | after.own_names((x2, y2))
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
    pin_owners: dict[tuple[int, int], list[tuple[str, str]]] = {}
    for pin in before.pins:
        pin_owners.setdefault(pin.point, []).append((pin.reference, pin.name))
    dropped = same_instance_dropped_segments(
        pin_owners, segments, [label.point for label in before.labels]
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
    # as plain data: the parts with their pins, the wires, and the nets worked
    # out above. A route across a part is not judged at all here: the
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
        end_names=(from_name, to_name),
        ends_by_coordinate=(not isinstance(from_pin, str), not isinstance(to_pin, str)),
    )
    judged = judge(view, before, route)
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
    """Move or turn one part alone; move with drag takes its labels."""

    op: Literal["move_component"]
    reference: str = Field(description=_REFERENCE_DESCRIPTION)
    x: int
    y: int
    rotation: _RotationLiteral | None = Field(
        default=None, description=f"Omit to keep the current rotation; {_ROTATION_DESCRIPTION}"
    )


class OpReplaceComponent(StrictModel):
    """Swap a part's symbol in place; each moved pin is wired back to its net."""

    op: Literal["replace_component"]
    reference: str = Field(description=_REFERENCE_DESCRIPTION)
    symbol: str = Field(description="The new symbol, without .asy.")
    pins: dict[str, str] | None = Field(
        default=None,
        description="Old pin to new pin (name or SpiceOrder) where names differ; others pair by name.",
    )
    value: str | None = Field(default=None, description="New value; omit to keep the old one.")


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
        description="Directives and comments, named as for remove_directive; each moves.",
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
            "Incident-point form: removes each wire ending on this pin, other "
            "connections' too; one passing through stays."
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
            "Removes each directive and comment equal to this, outer space ignored; "
            "with 'regex:', each a case-sensitive search finds."
        )
    )


class OpReplaceDirective(StrictModel):
    """Change one text; its place, size, justification and kind stay."""

    op: Literal["replace_directive"]
    instruction: str = Field(description="Names one text, as remove_directive names them.")
    text: str = Field(description="The new text.")
    x: int | None = Field(default=None, description="With y, picks the named text anchored there.")
    y: int | None = None


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
    | OpReplaceComponent
    | OpMove
    | OpAddNetLabel
    | OpRemoveNetLabel
    | OpRemoveWire
    | OpWirePins
    | OpAddDirective
    | OpRemoveDirective
    | OpReplaceDirective
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
        "replace_directive",
        "set_plot_panes",
    }
)

# The keys of an op's result that report what applying it found on the sheet:
# the segments a route found already drawn, the junctions it made, how much a
# removal took. A value or attribute change reports what it changed. The rest
# of a result restates the op or carries geometry a surface reports through
# views of its own. A route's ``wire_count`` is not one: ``already_present``
# names the segments it did not draw, and relaying the count would put a line
# per route on every build. ``remove_directive``'s ``removed_texts`` names each
# text it took, since a regex leaves that unknown. An op missing here reports nothing
# beyond its arguments, and the reply's ``applied`` count says it ran.
OP_RESULT_FACTS: dict[str, tuple[str, ...]] = {
    "set_component_value": ("changed",),
    "set_component_attribute": ("changed",),
    "wire_pins": ("already_present", "junctions"),
    "remove_wire": ("removed",),
    "remove_net_label": ("removed",),
    "remove_directive": ("removed_texts",),
    "replace_directive": ("text",),
    "remove_component": ("deleted_wires",),
    "replace_component": ("was", "wires", "left_at", "unpaired_pins"),
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


def _pin_pairs(
    op: OpReplaceComponent,
    reference: str,
    old_symbol: str,
    old_pins: Sequence[PlacedPin],
    new_pins: Sequence[PlacedPin],
) -> list[tuple[PlacedPin, PlacedPin]]:
    """Each old pin with the new pin it becomes: as ``op.pins`` names them
    (by name or SpiceOrder, as ``placed_pin`` reads a pin), then by a name the
    two symbols share, in any case, where each has one pin of it."""
    pairs: list[tuple[PlacedPin, PlacedPin]] = []
    for old_name, new_name in (op.pins or {}).items():
        old = placed_pin(old_pins, reference, old_symbol, old_name)
        new = placed_pin(new_pins, reference, op.symbol, new_name)
        if any(old is o for o, _ in pairs) or any(new is n for _, n in pairs):
            raise NetlistError(
                f"replace_component pairs each pin once, and pins names {old_name!r} or "
                f"{new_name!r} twice."
            )
        pairs.append((old, new))

    def by_name(pins: Sequence[PlacedPin]) -> dict[str, PlacedPin]:
        names = Counter(pin.name.casefold() for pin in pins if pin.name)
        return {
            pin.name.casefold(): pin
            for pin in pins
            if pin.name and names[pin.name.casefold()] == 1
        }

    new_named = by_name(new_pins)
    for name, old in by_name(old_pins).items():
        new = new_named.get(name)
        if new is None or any(old is o for o, _ in pairs) or any(new is n for _, n in pairs):
            continue
        pairs.append((old, new))
    return pairs


def _joins(nets: NamedNets, reference: str) -> tuple[dict[str, frozenset], frozenset]:
    """What each pin of ``reference`` is one net with apart from the part
    itself (other parts' pins and labels), by pin, and how everything else on
    the sheet is grouped into nets. The two are what ``replace_component``
    must leave as they were."""
    others: dict[int, set[tuple]] = defaultdict(set)
    for pin in nets.pins:
        if pin.reference != reference:
            others[nets.of_pin(pin)].add(("pin", pin.qualified_name, pin.point))
    for label in nets.labels:
        others[nets.of_label(label)].add(("label", label.text, label.point))
    own = {
        _pin_key(pin.name, pin.order): frozenset(others.get(nets.of_pin(pin), ()))
        for pin in nets.pins
        if pin.reference == reference
    }
    return own, frozenset(frozenset(group) for group in others.values())


def _pin_key(name: str, order: int) -> str:
    """A pin as ``_joins`` keys it: its name, or its SpiceOrder for a pin with none."""
    return name or str(order)


def _joined_text(joined: frozenset) -> str:
    """What a pin is one net with, as a reader names it."""
    if not joined:
        return "nothing"
    names = sorted(
        f"{name}" if kind == "pin" else f"label '{name}'" for kind, name, _point in joined
    )
    return ", ".join(names[:6]) + (f" and {len(names) - 6} more" if len(names) > 6 else "")


def _replace_component(editor: AscEditor, op: OpReplaceComponent) -> dict[str, object]:
    """Swap ``op.reference``'s symbol where it stands and reconnect its pins.

    The part keeps its name, place, turn and attributes (its value too, unless
    ``op.value`` gives one); the old symbol's text placements (``WINDOW``) go,
    so the new symbol places its own. Each paired pin that moved gets a wire
    from where it is now to where the old one was, planned and checked as
    ``wire_pins`` plans one. Then the sheet must join each paired pin to what
    its old pin joined, an unpaired new pin to nothing, and everything else as
    before (``_joins``); otherwise the op is refused and the sheet put back.
    """
    reference = part_reference(editor, op.reference)
    component = editor.components[reference]
    old_symbol = component.symbol or ""
    old_geometry = placed_geometry(editor, reference)
    if old_geometry is None:
        raise NetlistError(
            f"Cannot replace {reference}: its symbol '{old_symbol}' is not found, so its "
            "pins, which the new symbol's take over, are unknown."
        )
    new_info = symbol_info_for(editor, op.symbol)
    if new_info is None:
        raise NetlistError(
            f"Symbol '{op.symbol}' not found beside the schematic or in any configured "
            "symbol library."
        )
    position, rotation = editor.get_component_position(reference)
    x, y = int(position.X), int(position.Y)
    turn = rotation.name if rotation else "R0"
    new_geometry = compute_placed_geometry(new_info, x, y, turn)

    def placed(pins: Sequence[Mapping[str, Any]]) -> list[PlacedPin]:
        return [PlacedPin(p["name"], p["order"], p["x"], p["y"], p["dir"]) for p in pins]

    old_pins, new_pins = placed(old_geometry["pins"]), placed(new_geometry["pins"])
    pairs = _pin_pairs(op, reference, old_symbol, old_pins, new_pins)
    _pins_on_crossings(
        editor,
        reference,
        new_geometry["pins"],
        f"replace {reference}",
        already={(pin.x, pin.y) for pin in old_pins},
    )
    before = sheet_nets(editor)
    joined_before, groups_before = _joins(before, reference)

    kept_attributes = dict(component.attributes)
    wire_count = len(editor.wires)

    def put_back() -> None:
        component.symbol = old_symbol  # pyright: ignore[reportAttributeAccessIssue]
        component.attributes.clear()
        component.attributes.update(kept_attributes)
        del editor.wires[wire_count:]

    component.symbol = op.symbol  # pyright: ignore[reportAttributeAccessIssue]
    # The old symbol's text placements and hierarchical block, not texts.
    for name in [n for n in component.attributes if n.startswith(("_WINDOW", "_SUBCKT"))]:
        del component.attributes[name]
    try:
        if op.value is not None:
            _apply_component_value(editor, reference, op.value)
        drawn: list[tuple[int, int, int, int]] = []
        for old, new in pairs:
            old_key = _pin_key(old.name, old.order)
            if (old.x, old.y) == (new.x, new.y) or not joined_before.get(old_key):
                continue
            drawn += _bridge(editor, reference, old, new)
        after = sheet_nets(editor)
        joined_after, groups_after = _joins(after, reference)
        paired = {_pin_key(new.name, new.order): old for old, new in pairs}
        for new in new_pins:
            key = _pin_key(new.name, new.order)
            old = paired.get(key)
            want = (
                joined_before.get(_pin_key(old.name, old.order), frozenset())
                if old
                else frozenset()
            )
            if joined_after.get(key, frozenset()) != want:
                said = (
                    f"its old pin {old.name or old.order} was joined to"
                    if old
                    else "no old pin pairs with it, so it should join"
                )
                raise NetlistError(
                    f"Refused to replace {reference} with {op.symbol}: its pin "
                    f"{new.name or new.order} at ({new.x},{new.y}) would be joined to "
                    f"{_joined_text(joined_after.get(key, frozenset()))}, where {said} "
                    f"{_joined_text(want)}. Pair the pins with pins (old pins: "
                    f"{', '.join(p.name or str(p.order) for p in old_pins)}; new pins: "
                    f"{', '.join(p.name or str(p.order) for p in new_pins)}), or move the "
                    "part first so the new pins land clear of other wiring."
                )
        if groups_after != groups_before:
            raise NetlistError(
                f"Refused to replace {reference} with {op.symbol}: the new symbol or the "
                "wires that reconnect its pins would join or split other nets of the "
                "sheet. Move the part first, or reconnect its pins with wire_pins."
            )
    except (NetlistError, ValueError):
        put_back()
        raise
    paired_old = {id(old) for old, _ in pairs}
    left = [
        {"pin": old.name or str(old.order), "x": old.x, "y": old.y}
        for old in old_pins
        if id(old) not in paired_old and joined_before.get(_pin_key(old.name, old.order))
    ]
    paired_new = {id(new) for _, new in pairs}
    unpaired = [new.name or str(new.order) for new in new_pins if id(new) not in paired_new]
    result: dict[str, object] = {
        "op": "replace_component",
        "reference": reference,
        "symbol": op.symbol,
        "was": old_symbol,
    }
    if drawn:
        result["wires"] = [segment_json(segment) for segment in drawn]
    if left:
        result["left_at"] = left
    if unpaired:
        result["unpaired_pins"] = unpaired
    return result


def _bridge(
    editor: AscEditor, reference: str, old: PlacedPin, new: PlacedPin
) -> list[tuple[int, int, int, int]]:
    """Draw a wire from ``new``, where the pin is now, to ``old``, where the
    wiring of the pin it replaces ends, by the first route the route planner
    accepts (``_plan_connect_route``, as ``wire_pins`` plans a wire): straight
    where the two share a line, else round one corner, either way, else across
    between them with two (``_bridge_routes``)."""
    start, end = GridPoint(x=new.x, y=new.y), GridPoint(x=old.x, y=old.y)
    refusal: NetlistError | None = None
    for waypoints in _bridge_routes((new.x, new.y), (old.x, old.y)):
        try:
            plan = _plan_connect_route(
                editor, start, end, [GridPoint(x=x, y=y) for x, y in waypoints]
            )
        except NetlistError as exc:
            refusal = refusal or exc
            continue
        already = _append_wire_segments(editor, plan.segments)
        return [segment for segment in plan.segments if segment not in already]
    raise NetlistError(
        f"Refused to replace {reference}: its pin {new.name or new.order} moves to "
        f"({new.x},{new.y}), and no wire back to ({old.x},{old.y}), where its wiring "
        f"ends, can be drawn: {refusal}"
    )


#: LTspice's grid step, in sheet units: where the pins of its symbols land.
_GRID = 16


def _bridge_routes(start: tuple[int, int], end: tuple[int, int]) -> list[list[tuple[int, int]]]:
    """The corners of each route ``_bridge`` offers the planner, in order: none
    when the ends share a line; else one corner, either way round; else two,
    crossing over halfway on the 16-unit grid, either way round. A route of one
    corner can run along the old wiring it ends on, which the planner refuses;
    one that crosses over halfway meets it at a right angle."""
    (x1, y1), (x2, y2) = start, end
    if x1 == x2 or y1 == y2:
        return [[]]
    routes: list[list[tuple[int, int]]] = [[(x1, y2)], [(x2, y1)]]
    mid_y = round((y1 + y2) / 2 / _GRID) * _GRID
    if mid_y not in (y1, y2):
        routes.append([(x1, mid_y), (x2, mid_y)])
    mid_x = round((x1 + x2) / 2 / _GRID) * _GRID
    if mid_x not in (x1, x2):
        routes.append([(mid_x, y1), (mid_x, y2)])
    return routes


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
        if (taken := _reference_taken(editor, op.reference)) is not None:
            raise NetlistError(
                f"Component '{taken}' already exists in {asc_path.name}."
                + _same_reference_note(op.reference, taken)
            )
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
        reference = part_reference(editor, op.reference)
        before = _attributes_of(editor, reference)
        _apply_component_value(editor, reference, op.value)
        result: dict[str, object] = {
            "op": "set_component_value",
            "reference": reference,
            "value": op.value,
        }
        if changed := _attributes_changed(before, _attributes_of(editor, reference)):
            result["changed"] = changed
        return result

    if isinstance(op, OpSetComponentAttribute):
        _reject_unknown_attr(op.attribute)
        reference = part_reference(editor, op.reference)
        before = _attributes_of(editor, reference)
        if not op.value.strip():
            # Empty value = clear. LTspice's format has no "empty value"
            # representation (a 2-token SYMATTR line is unreadable), so the
            # line is removed instead.
            _require_clearable_attr(reference, op.attribute)
            part = editor.get_component(reference)
            _attribute_texts(editor, part, {}, cleared={op.attribute})
            part.attributes.pop(op.attribute, None)
        elif op.attribute == "InstName":
            return _rename_component(editor, reference, op.value.strip())
        else:
            text = _attribute_texts(
                editor, editor.get_component(reference), {op.attribute: op.value}
            )[op.attribute]
            editor.set_component_attribute(reference, op.attribute, text)
        attr_result: dict[str, object] = {
            "op": "set_component_attribute",
            "reference": reference,
            "attribute": op.attribute,
        }
        if changed := _attributes_changed(before, _attributes_of(editor, reference)):
            attr_result["changed"] = changed
        return attr_result

    if isinstance(op, OpRemoveComponent):
        reference = part_reference(editor, op.reference)
        target_only = _component_pin_coords(editor, reference) - _other_components_pin_coords(
            editor, reference
        )
        editor.remove_component(reference)
        rm_result: dict[str, object] = {"op": "remove_component", "reference": reference}
        # A wire it leaves on nothing is the finished sheet's to say
        # (dangling_wire_end), so a later op of the batch that removes or
        # reconnects it leaves nothing said.
        if op.cleanup_wires and target_only:
            rm_result["deleted_wires"] = _drop_wires_at(editor, target_only)
        return rm_result

    if isinstance(op, OpMoveComponent):
        reference = part_reference(editor, op.reference)
        new_rot = (
            _parse_rotation(op.rotation)
            if op.rotation is not None
            else editor.get_component_position(reference)[1]
        )
        old_pins = _component_pin_coords(editor, reference)
        moved_symbol = editor.components[reference].symbol
        moved_info = symbol_info_for(editor, moved_symbol) if moved_symbol else None
        if moved_info is not None:
            _pins_on_crossings(
                editor,
                reference,
                compute_placed_geometry(moved_info, op.x, op.y, new_rot.name)["pins"],
                f"move {reference}",
                already=old_pins,
            )
        # Wires left at its old pins are the finished sheet's to say
        # (dangling_wire_end), as for remove_component.
        editor.set_component_position(reference, Point(op.x, op.y), new_rot)
        return {"op": "move_component", "reference": reference}

    if isinstance(op, OpReplaceComponent):
        return _replace_component(editor, op)

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
                # guard exists to refuse. The oracle is the sheet's wired nets.
                nets_before = sheet_nets(editor).nets
                on_net: dict[int, list[PinAt]] = {}
                for pin in nets_before.pins:
                    on_net.setdefault(nets_before.of_pin(pin), []).append(pin)
                original, editor.wires = editor.wires, kept
                nets_after = sheet_nets(editor).nets
                for pins in on_net.values():
                    sides: dict[int, list[PinAt]] = {}
                    for pin in pins:
                        sides.setdefault(nets_after.of_pin(pin), []).append(pin)
                    if len(sides) < 2:
                        continue
                    editor.wires = original

                    def _side_names(side: list[PinAt]) -> str:
                        names = [
                            f"{pin.reference}.{pin.name}" if pin.name else pin.reference
                            for pin in sorted(
                                side, key=lambda pin: (pin.point, pin.part, pin.order)
                            )
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
        if op.kind == "directive":
            _refuse_unrunnable_directive(op.instruction)
        text_type = TextTypeEnum.DIRECTIVE if op.kind == "directive" else TextTypeEnum.COMMENT
        _append_asc_text(
            editor, op.instruction, text_type, op.x, op.y, op.size, default_x=16, default_y=16
        )
        return {"op": "add_directive", "instruction": op.instruction}

    if isinstance(op, OpRemoveDirective):
        # Literal-by-default, 'regex:' opt-in, raises if nothing matched. What
        # it removed is read off the sheet's texts, not the matcher's say-so.
        had = list(editor.directives)
        _remove_directive_or_comment(editor, op.instruction)
        kept = {id(text) for text in editor.directives}
        removed_texts = [
            {
                "kind": _text_kind(text),
                "text": _shown_text(text.text),
                "x": int(text.coord.X),
                "y": int(text.coord.Y),
            }
            for text in had
            if id(text) not in kept
        ]
        return {
            "op": "remove_directive",
            "instruction": op.instruction,
            "removed_texts": removed_texts,
        }

    if isinstance(op, OpReplaceDirective):
        # One text, named as remove_directive names it, keeps its record (place,
        # size, justification, kind) and takes the new text.
        # Everything is checked before the text changes, so a refused op leaves
        # the sheet as it was, and a dry run's later ops see it so.
        if (op.x is None) != (op.y is None):
            raise NetlistError("replace_directive takes x and y together, or neither.")
        named = _text_matcher(op.instruction)
        matched = [text for text in editor.directives if named(text)]
        if not matched:
            raise _no_text_matched(op.instruction)
        if op.x is not None:
            there = [
                text for text in matched if (int(text.coord.X), int(text.coord.Y)) == (op.x, op.y)
            ]
            if not there:
                raise NetlistError(
                    f"None of the texts {op.instruction!r} names is anchored at "
                    f"({op.x}, {op.y}); they are {_listed_texts(matched)}."
                )
            matched = there
        if len(matched) > 1:
            place = " at that anchor" if op.x is not None else ""
            fix = (
                "Move one of them first: texts on one anchor cannot be told apart."
                if op.x is not None
                else "Give the x and y of the one to change."
            )
            raise NetlistError(
                f"replace_directive changes one text, and {op.instruction!r} names "
                f"{len(matched)}{place}: {_listed_texts(matched)}. {fix}"
            )
        (text,) = matched
        kind = _text_kind(text)
        if kind == "directive":
            _refuse_unrunnable_directive(op.text)
        was = _shown_text(text.text)
        text.text = _text_record_body(op.text)
        return {
            "op": "replace_directive",
            "text": {
                "kind": kind,
                "x": int(text.coord.X),
                "y": int(text.coord.Y),
                "was": was,
                "now": _shown_text(text.text),
            },
        }

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
