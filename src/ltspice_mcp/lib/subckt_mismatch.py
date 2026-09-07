"""Per-instance mismatch for MOS devices reached through one subcircuit level.

Foundry model decks wrap every FET in an ``X`` subcircuit, so a mismatch rule
written against top-level ``M`` cards matches nothing at all. ngspice accepts
two instance parameters that express exactly what a mismatch draw needs —
``delvto`` (an additive shift to the *signed* ``vth0``) and ``mulu0`` (a
mobility multiplier) — but only on the ``M`` card itself, which sits inside the
subcircuit body where no caller can address one instance of it.

This module bridges that gap the way a foundry preprocessor does. Per device
type it clones the device subcircuit once, gives the clone a forwarded
parameter pair for each targeted inner device, threads that pair down onto the
inner ``M`` card, and re-points each targeted ``X`` line at the clone carrying
its own exact values::

    XN1 d g s b nfet__mcpatch W=1 L=0.15 mc_delvto__m0=-0.02 mc_mulu0__m0=1.05

Every instance then carries its own shift on its own line, the library files
are never written to, and the clone lands in the root deck where a top-level
subcircuit definition is visible to every caller.

Fixed at one subcircuit level: ``X`` → ``M``. A wrapper that instantiates
another wrapper (``X`` → ``X`` → ``M``) is refused by name rather than
partially handled.

ngspice only. A LEVEL 1-3 device rejects both parameters — measured, the
simulator stops with ``unknown parameter (delvto)`` — so an inner device that
is not BSIM is refused while the deck is being built instead of dying opaquely
mid-simulation.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

from ltspice_mcp.lib.deck_staging import (
    DEFAULT_INCLUDE_DEPTH,
    card_sections,
    closure_depth,
    resolve_existing,
    resolve_reference,
    scan_include_references,
)
from ltspice_mcp.lib.encoding import read_spice_text
from ltspice_mcp.lib.montecarlo import (
    InstanceGeometry,
    MCSampler,
    MismatchRule,
    find_mismatch_rule,
    matches_prefix,
    parse_value,
    sample_instance_mismatch,
)
from ltspice_mcp.lib.spice_lex import (
    SpiceCard,
    SpiceLexError,
    TokenKind,
    emit,
    find_matching_ends,
    lex,
    tokenize_body,
)
from ltspice_mcp.lib.spice_lex_ops import inject_cards_before_end, rename_subckt
from ltspice_mcp.lib.spice_lex_views import InstanceLine, SubcktCard

logger = logging.getLogger(__name__)

# The two ngspice instance parameters this engine writes. ``delvto`` adds to
# the signed vth0, so a pFET (negative vth0) shifted by +30 mV shows a SMALLER
# threshold magnitude; callers and receipts stay in the signed domain.
VTH_PARAM = "delvto"
MOBILITY_PARAM = "mulu0"
MISMATCH_PARAMS: tuple[str, str] = (VTH_PARAM, MOBILITY_PARAM)

CLONE_SUFFIX = "__mcpatch"

# Model levels that carry delvto/mulu0: BSIM3 and BSIM4 under both ngspice's
# native numbering (8, 14) and its HSPICE-compatible numbering (49, 54). The
# 49/54 pair is the measured one; 8/14 name the same two models.
BSIM_LEVELS: frozenset[int] = frozenset({8, 14, 49, 54})

# Levels measured to reject the parameters outright.
_REFUSING_LEVELS: frozenset[int] = frozenset({1, 2, 3})

# Verbatim ngspice diagnostic for a delvto on a non-BSIM device, quoted back so
# a refusal here reads as the failure it is standing in for.
NGSPICE_UNKNOWN_PARAMETER = "unknown parameter (delvto)"


class MismatchPlanError(ValueError):
    """A per-instance mismatch plan cannot be built or applied as asked."""

    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(message)


@dataclass(frozen=True)
class ClosureFile:
    """One file of a deck's include closure, as this engine reads it.

    ``index`` is both the file's identity and its position in the closure a
    caller passes: every producer numbers its files in list order, and a plan
    records ``index`` and then indexes straight back into the list rather than
    searching it. ``require_positional`` is where that contract is checked.
    """

    index: int
    path: Path
    text: str

    @property
    def depth(self) -> int:
        return closure_depth(self.index)


def require_positional(files: Sequence[ClosureFile]) -> None:
    """Refuse a closure whose file indexes are not its list positions.

    A plan is built against one closure and rendered against another (the
    deck's own text changes between runs), and it addresses both by ``index``.
    A caller that hands over a subset, or a reordering, would have every
    recorded index point at a different file.
    """
    for position, file in enumerate(files):
        if file.index != position:
            raise ValueError(
                f"Closure file {file.path.name!r} is numbered {file.index} at position "
                f"{position}; a mismatch plan addresses closure files by index, so the "
                "closure has to be passed whole and in order"
            )


@dataclass(frozen=True)
class XFetTarget:
    """One inner MOS device addressable through exactly one ``X`` level."""

    x_ref: str
    inner_ref: str
    subckt: str
    clone: str
    x_file: int
    def_file: int
    model_name: str
    width: float | None
    length: float | None

    @property
    def ref(self) -> str:
        return f"{self.x_ref}.{self.inner_ref}"


@dataclass(frozen=True)
class MismatchSkip:
    """One selected instance the plan deliberately carries no target for."""

    x_ref: str
    code: str
    detail: str


@dataclass(frozen=True)
class MismatchValues:
    """The values one inner device receives on its own ``X`` line."""

    delvto: float | None = None
    mulu0: float | None = None


@dataclass(frozen=True)
class MismatchClone:
    """Where a patched copy of a device subcircuit comes from, and what to call it.

    ``text`` is the patched body, emitted once at plan time — but only for a
    device defined OUTSIDE the root deck. Per-run perturbation rewrites the
    root deck alone; every other closure file is fed back to ``render`` as the
    snapshot planning read, so those bodies are byte-stable across runs and
    copying them once is copying them right.

    A device defined IN the root deck carries no text and is re-emitted per
    run, because that is exactly where a run's own edits land — a model card
    the deck declares inside the subcircuit body is where process variation
    goes — and a copy taken at plan time would carry the nominal version while
    the instances naming it report the perturbed one.
    """

    name: str
    subckt: str
    def_file: int
    section: str | None
    inner_refs: tuple[str, ...]
    text: str | None = None


@dataclass(frozen=True)
class MismatchPlan:
    """An immutable per-job plan: which inner devices are reachable, and how.

    Built once from the active include closure, then rendered as many times as
    there are runs. Planning is what resolves a subcircuit name to one
    definition site; rendering only writes the values.
    """

    targets: tuple[XFetTarget, ...]
    skips: tuple[MismatchSkip, ...]
    clones: tuple[MismatchClone, ...]

    @cached_property
    def by_ref(self) -> dict[str, XFetTarget]:
        """Case-folded hierarchical reference (``x1.m0``) → target."""
        return {t.ref.casefold(): t for t in self.targets}


def encode_ref(inner_ref: str) -> str:
    """Encode a device reference into a legal parameter identifier, injectively.

    A reference may hold characters a parameter name cannot (``$``, ``.``, a
    hyphen), and pasting one in unchanged emits a netlist the simulator cannot
    parse. Letters and digits pass through; ``_`` doubles; anything else becomes
    ``_`` plus its two-digit hex code. Since a hex digit is never ``_``, the two
    escapes cannot be confused, so distinct references always encode to distinct
    names — which is what keeps two devices in one body from sharing a knob.

    The reference is case-folded first: SPICE does not distinguish ``M0`` from
    ``m0``, so folding them together is what the simulator already does.
    """
    out: list[str] = []
    for char in inner_ref.casefold():
        if char.isascii() and char.isalnum():
            out.append(char)
        elif char == "_":
            out.append("__")
        elif ord(char) <= 0xFF:
            out.append(f"_{ord(char):02x}")
        else:
            raise MismatchPlanError(
                "unencodable_device_ref",
                f"Device reference {inner_ref!r} contains {char!r}, which has no "
                "representation in a SPICE parameter name; rename the device",
            )
    return "".join(out)


def forwarded_param(param: str, inner_ref: str) -> str:
    """Name the reserved subcircuit parameter that carries one device's value.

    Keyed by the inner card's real reference so two devices in one body stay
    independently addressable, and namespaced so a collision with a parameter
    the library already declares is a refusal rather than a silent override.
    """
    return f"mc_{param}__{encode_ref(inner_ref)}"


def require_geometry(target: XFetTarget) -> tuple[float, float]:
    """Return literal W/L, or refuse a rule whose sigma cannot be computed.

    A Pelgrom-scaled rule divides by ``√(W·L)``; a deck that writes ``W={wn}``
    has no number to divide by at plan time, and guessing one would report a
    sigma nothing was drawn from.
    """
    if target.width is None or target.length is None:
        raise MismatchPlanError(
            "geometry_not_literal",
            f"Instance {target.x_ref!r} does not give W and L as literal numbers on "
            f"its X line, so a geometry-scaled sigma cannot be computed for "
            f"{target.ref!r}; supply exact per-instance values instead, or write "
            "numeric W/L on the instance",
        )
    return target.width, target.length


def closure_from_deck(
    path: Path,
    text: str,
    *,
    max_depth: int = DEFAULT_INCLUDE_DEPTH,
) -> list[ClosureFile]:
    """Read a deck's include closure without writing anything.

    For the callers that hand a simulator one file rather than a staged bundle:
    the device library still has to be READ to find where a subcircuit is
    defined, even though only the deck is ever rewritten. A reference that does
    not resolve is left out rather than raised on — the simulator is the
    authority on whether a deck is runnable, and a plan that cannot see a file
    simply finds no device there.
    """
    files = [ClosureFile(index=0, path=path, text=text)]
    seen = {path.resolve()}
    frontier = [(path, text, 0)]
    while frontier:
        source, source_text, depth = frontier.pop(0)
        if depth >= max_depth:
            continue
        for reference in scan_include_references(
            lex(source_text).cards, source, depth=closure_depth(depth)
        ):
            resolved = resolve_existing(resolve_reference(source.parent, reference.raw_path))
            if resolved is None or resolved in seen:
                continue
            try:
                included = read_spice_text(resolved)
            except OSError:
                continue
            seen.add(resolved)
            files.append(ClosureFile(index=len(files), path=resolved, text=included))
            frontier.append((resolved, included, depth + 1))
    return files


# ---------------------------------------------------------------------------
# Closure indexing
# ---------------------------------------------------------------------------


@dataclass
class _FileIndex:
    file: ClosureFile
    cards: list[SpiceCard]
    sections: list[str | None]
    # Sections a caller selected for this file, or None when every card is
    # active (a plain .include, or a library nobody entered by section).
    active: frozenset[str] | None = None

    def is_active(self, card_index: int) -> bool:
        if self.active is None:
            return True
        section = self.sections[card_index]
        return section is None or section.casefold() in self.active


def _index_files(files: Sequence[ClosureFile]) -> list[_FileIndex]:
    indexes = [
        _FileIndex(
            file=f,
            cards=(cards := lex(f.text).cards),
            sections=card_sections(cards, f.path, f.depth),
        )
        for f in files
    ]
    _mark_active_sections(indexes)
    return indexes


def _mark_active_sections(indexes: list[_FileIndex]) -> None:
    """Record which ``.lib`` sections the closure actually enters.

    A library file holds one definition of a device per corner section, and
    only the section a caller selected is in the circuit. Without this the same
    subcircuit reads as defined many times over and every plan refuses as
    ambiguous.
    """
    by_path = {index.file.path.resolve(): index.file.index for index in indexes}
    selected: dict[int, set[str]] = {}
    for index in indexes:
        for reference in scan_include_references(
            index.cards, index.file.path, depth=index.file.depth
        ):
            if reference.section is None:
                continue
            target = resolve_reference(index.file.path.parent, reference.raw_path).resolve()
            hit = by_path.get(target)
            if hit is not None:
                selected.setdefault(hit, set()).add(reference.section.casefold())
    for index in indexes:
        chosen = selected.get(index.file.index)
        index.active = frozenset(chosen) if chosen else None


_OPTION_HEADS = frozenset({".option", ".options", ".opt"})


def _deck_scale(indexes: list[_FileIndex]) -> float:
    """The length multiplier ``.option scale`` puts on every device dimension.

    A foundry deck sets ``scale=1.0u`` and then writes ``W=1 L=0.15`` meaning
    one micron by 0.15 micron. Geometry read without it is a million times too
    large, which drives a Pelgrom sigma a million times too small — every draw
    collapses to noise and the run reads as a design with no mismatch at all.
    Absent, the multiplier is 1 and the numbers are already metres.
    """
    found: dict[float, str] = {}
    for index in indexes:
        for card_index, card in enumerate(index.cards):
            if card.kind != "directive" or not index.is_active(card_index):
                continue
            tokens = tokenize_body(card.body)
            if not tokens or tokens[0].text.casefold() not in _OPTION_HEADS:
                continue
            for token in tokens[1:]:
                if token.kind != TokenKind.KEY_VALUE or (token.key or "").casefold() != "scale":
                    continue
                value = parse_value(token.value or "")
                if value is not None and value > 0.0:
                    found.setdefault(value, index.file.path.name)
    if len(found) > 1:
        detail = ", ".join(f"{value:g} in {where}" for value, where in sorted(found.items()))
        raise MismatchPlanError(
            "ambiguous_scale",
            f"The deck sets .option scale to {len(found)} different values ({detail}); "
            "device geometry means something different under each, so which one a "
            "mismatch sigma is computed from cannot be decided here",
        )
    return next(iter(found), 1.0)


def _subckt_definitions(indexes: list[_FileIndex]) -> dict[str, list[tuple[int, int]]]:
    """Map each subcircuit name to every active ``(file index, card index)``."""
    definitions: dict[str, list[tuple[int, int]]] = {}
    for index in indexes:
        for card_index, card in enumerate(index.cards):
            if card.kind != "subckt" or not card.name:
                continue
            if not index.is_active(card_index):
                continue
            definitions.setdefault(card.name.casefold(), []).append((index.file.index, card_index))
    return definitions


def _closer_index(cards: list[SpiceCard], opener_index: int) -> int:
    closer = find_matching_ends(cards, opener_index)
    return len(cards) if closer is None else closer


def _direct_children(index: _FileIndex, opener_index: int) -> list[tuple[int, SpiceCard]]:
    """Cards one level inside a subcircuit body, nested bodies excluded.

    Addressed by span rather than by scope: two definitions of one name in a
    file — the ordinary shape of a library entered under two corner sections —
    give their bodies the same scope tuple, so a scope query would return both
    bodies' cards for whichever definition was resolved.
    """
    depth = len(index.cards[opener_index].scope) + 1
    stop = _closer_index(index.cards, opener_index)
    return [
        (i, index.cards[i])
        for i in range(opener_index + 1, stop)
        if len(index.cards[i].scope) == depth
    ]


def _instances(children: Iterable[tuple[int, SpiceCard]], prefix: str) -> list[SpiceCard]:
    return [
        card
        for _, card in children
        if card.kind == "instance" and card.name and card.name[:1].upper() == prefix
    ]


def _model_level(indexes: list[_FileIndex], model_name: str) -> float | None:
    """Read LEVEL off the model card an inner device names, binned or not.

    A binned foundry model declares ``.model <name>.0``, ``.model <name>.1``,
    … and the instance names the unsuffixed base, so the lookup accepts both
    spellings. Every bin of one device shares a level.
    """
    folded = model_name.casefold()
    for index in indexes:
        for card_index, card in enumerate(index.cards):
            if card.kind != "model" or not card.name:
                continue
            name = card.name.casefold()
            if name != folded and not name.startswith(f"{folded}."):
                continue
            if not index.is_active(card_index):
                continue
            match = _LEVEL_RE.search(_model_card_text(index, card_index))
            if match is not None:
                return float(match.group(1))
    return None


_LEVEL_RE = re.compile(r"\blevel\s*=\s*([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)


def _model_card_text(index: _FileIndex, card_index: int) -> str:
    """A model card's own text, including continuations a comment interrupted.

    Foundry model decks section their parameter blocks with full-line comments
    between the ``+`` continuations. A simulator reads straight through those,
    but the lexer ends a card at the comment and hands the rest back as loose
    continuation lines, so the card alone holds only the model's name and type.
    Re-joining the continuation run is what lets the level be read at all.
    """
    parts = list(index.cards[card_index].raw_lines)
    for card in index.cards[card_index + 1 :]:
        if card.kind not in ("comment", "blank"):
            break
        if card.raw_lines and card.raw_lines[0].lstrip().startswith("+"):
            parts.extend(card.raw_lines)
    return "".join(parts)


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------


def build_plan(
    files: Sequence[ClosureFile],
    *,
    prefix: str | Sequence[str] | None = None,
    selectors: Sequence[str] | None = None,
    exact: bool = False,
) -> MismatchPlan:
    """Resolve which inner MOS devices a selection reaches, and how to reach them.

    Exactly one of ``prefix`` (a set query over top-level ``X`` cards; several
    prefixes may be given together so one plan owns every clone name) or
    ``selectors`` (``X1`` or ``X1.M0``, addressable at any scope) is given.
    ``exact`` requires each selector to name a single device, which is what a
    caller-supplied value needs when a body holds more than one ``M``.

    A prefix deliberately does not descend into subcircuit bodies: a foundry
    library is full of internal ``X`` cards, and a set query that swept them
    would perturb the library's own wiring rather than the caller's devices.
    """
    if (prefix is None) == (selectors is None):
        raise ValueError("build_plan takes exactly one of prefix or selectors")
    require_positional(files)
    indexes = _index_files(files)
    definitions = _subckt_definitions(indexes)
    all_subckt_names = set(definitions)

    if prefix is not None:
        picked = _select_by_prefix(indexes, (prefix,) if isinstance(prefix, str) else prefix)
        wanted_inner: dict[str, set[str] | None] = {ref: None for ref, _, _, _ in picked}
    else:
        picked, wanted_inner = _select_by_ref(indexes, selectors or ())

    # Only decks this selection actually reaches into pay for the scale lookup,
    # and only they can be refused by it.
    scale = _deck_scale(indexes) if picked else 1.0
    targets: list[XFetTarget] = []
    skips: list[MismatchSkip] = []
    # Keyed by the definition SITE, not just the name: one library entered
    # under two corner sections carries two devices under one name, and a key
    # that cannot tell them apart patches one corner and points both at it.
    drafts: dict[tuple[int, str, str | None], _CloneDraft] = {}

    for x_ref, file_index, card, x_section in picked:
        view = InstanceLine.from_card(card)
        subckt = view.model
        if subckt is None:
            skips.append(MismatchSkip(x_ref, "subckt_unresolved", f"{x_ref} names no subcircuit"))
            continue
        sites = definitions.get(subckt.casefold(), [])
        if not sites:
            if selectors is not None:
                raise MismatchPlanError(
                    "subckt_unresolved",
                    f"Instance {x_ref!r} instantiates {subckt!r}, which no file in the "
                    "include closure defines",
                )
            skips.append(
                MismatchSkip(
                    x_ref,
                    "subckt_unresolved",
                    f"{x_ref} instantiates {subckt!r}, which the closure does not define",
                )
            )
            continue
        sites = _narrow_to_section(sites, indexes, file_index, x_section)
        if len(sites) > 1:
            where = ", ".join(
                f"{indexes[f].file.path.name} line {indexes[f].cards[c].line_start}"
                for f, c in sites
            )
            raise MismatchPlanError(
                "ambiguous_subckt",
                f"Subcircuit {subckt!r} instantiated by {x_ref!r} is defined at "
                f"{len(sites)} active sites ({where}); a mismatch plan has to patch one "
                "definition, so remove or section-select all but the one this deck means",
            )
        def_file, opener_index = sites[0]
        def_index = indexes[def_file]
        children = _direct_children(def_index, opener_index)
        inner_cards = _instances(children, "M")
        if not inner_cards:
            _refuse_or_skip_without_mos(x_ref, subckt, children, indexes, definitions, skips)
            continue

        _refuse_preexisting(x_ref, view, MISMATCH_PARAMS, "X line")
        wanted = wanted_inner.get(x_ref)
        chosen = _choose_inner(x_ref, subckt, inner_cards, wanted, exact=exact)
        def_section = def_index.sections[opener_index]
        clone_key = (def_file, subckt.casefold(), def_section)
        draft = drafts.setdefault(
            clone_key,
            _CloneDraft(
                name=_allocate_clone_name(subckt, all_subckt_names),
                def_file=def_file,
                opener_index=opener_index,
            ),
        )
        raw_width = parse_value(view.get_param("W") or "")
        raw_length = parse_value(view.get_param("L") or "")
        width = None if raw_width is None else raw_width * scale
        length = None if raw_length is None else raw_length * scale
        for inner in chosen:
            inner_view = InstanceLine.from_card(inner)
            _refuse_preexisting(
                x_ref, inner_view, MISMATCH_PARAMS, f"inner device {inner_view.ref}"
            )
            if inner_view.model is None:
                raise MismatchPlanError(
                    "inner_model_unresolved",
                    f"Inner device {inner_view.ref!r} of {subckt!r} names no model",
                )
            _require_bsim(x_ref, inner_view, indexes)
            draft.inner_refs.add(inner_view.ref)
            targets.append(
                XFetTarget(
                    x_ref=x_ref,
                    inner_ref=inner_view.ref,
                    subckt=subckt,
                    clone=draft.name,
                    x_file=file_index,
                    def_file=def_file,
                    model_name=inner_view.model,
                    width=width,
                    length=length,
                )
            )

    clones: list[MismatchClone] = []
    for key in sorted(drafts, key=lambda k: (k[0], k[1], k[2] or "")):
        draft = drafts[key]
        index = indexes[draft.def_file]
        inner_refs = sorted(draft.inner_refs)
        # Emitted here whatever its home, because every refusal the emission
        # can raise belongs at plan time, where it stops the job instead of
        # stopping one run. Only a clone from a file no run rewrites is KEPT;
        # see MismatchClone for why the root deck's are re-emitted per run.
        clone_cards = _render_clone(
            index.file,
            index.cards,
            draft.opener_index,
            draft.name,
            inner_refs,
        )
        clones.append(
            MismatchClone(
                name=draft.name,
                subckt=index.cards[draft.opener_index].name or "",
                def_file=draft.def_file,
                section=index.sections[draft.opener_index],
                inner_refs=tuple(inner_refs),
                text=None if draft.def_file == 0 else emit(clone_cards),
            )
        )
    for skip in skips:
        # The receipt carries the code; the sentence that says which subcircuit
        # it was and why has nowhere else to go.
        logger.debug("mismatch plan skipped %s (%s): %s", skip.x_ref, skip.code, skip.detail)
    return MismatchPlan(targets=tuple(targets), skips=tuple(skips), clones=tuple(clones))


@dataclass
class _CloneDraft:
    """One patched copy being accumulated while the selection is walked."""

    name: str
    def_file: int
    opener_index: int
    inner_refs: set[str] = field(default_factory=set)


def _narrow_to_section(
    sites: list[tuple[int, int]],
    indexes: list[_FileIndex],
    x_file: int,
    x_section: str | None,
) -> list[tuple[int, int]]:
    """Prefer a definition declared alongside the instance that names it.

    One physical library is routinely entered under more than one section, and
    each section carries its own copy of a device. A definition sitting in the
    same file and the same section as the instance is the one that instance
    means; without that, selecting two corners of one library reads as two
    definitions of the same name and refuses a deck the simulator accepts.
    """
    if len(sites) < 2:
        return sites
    local = [
        (file_index, card_index)
        for file_index, card_index in sites
        if file_index == x_file and indexes[file_index].sections[card_index] == x_section
    ]
    return local if len(local) == 1 else sites


def _select_by_prefix(
    indexes: list[_FileIndex],
    prefixes: Sequence[str],
) -> list[tuple[str, int, SpiceCard, str | None]]:
    folded = tuple(p.casefold() for p in prefixes)
    picked: list[tuple[str, int, SpiceCard, str | None]] = []
    for index in indexes:
        for card_index, card in enumerate(index.cards):
            if card.kind != "instance" or not card.name or card.scope != ():
                continue
            if card.name[:1].upper() != "X" or not card.name.casefold().startswith(folded):
                continue
            if not index.is_active(card_index):
                continue
            picked.append((card.name, index.file.index, card, index.sections[card_index]))
    _refuse_repeated_instance(indexes, picked)
    return picked


def _select_by_ref(
    indexes: list[_FileIndex],
    selectors: Sequence[str],
) -> tuple[list[tuple[str, int, SpiceCard, str | None]], dict[str, set[str] | None]]:
    wanted: dict[str, set[str] | None] = {}
    order: list[str] = []
    for selector in selectors:
        x_ref, _, inner = selector.partition(".")
        if not x_ref:
            raise MismatchPlanError(
                "invalid_instance_target", f"Instance target {selector!r} names no instance"
            )
        if x_ref not in wanted:
            wanted[x_ref] = None
            order.append(x_ref)
        if inner:
            existing = wanted[x_ref]
            wanted[x_ref] = {inner} if existing is None else existing | {inner}
    picked: list[tuple[str, int, SpiceCard, str | None]] = []
    for x_ref in order:
        hits = [
            (x_ref, index.file.index, card, index.sections[card_index])
            for index in indexes
            for card_index, card in enumerate(index.cards)
            if card.kind == "instance"
            and card.name
            and card.name.casefold() == x_ref.casefold()
            and index.is_active(card_index)
        ]
        if not hits:
            raise MismatchPlanError(
                "instance_not_found",
                f"Instance {x_ref!r} is not declared in the deck or its include closure",
            )
        picked.extend(hits)
    _refuse_repeated_instance(indexes, picked)
    return picked, wanted


def _refuse_repeated_instance(
    indexes: list[_FileIndex],
    picked: list[tuple[str, int, SpiceCard, str | None]],
) -> None:
    """Refuse a selection holding one reference twice.

    Two declarations of ``X1`` — one per subcircuit body, or one per file —
    would each be patched with their own values while the receipt records one
    of them, so the deck and the receipt would describe different circuits.
    """
    seen: dict[str, list[str]] = {}
    for x_ref, file_index, card, _section in picked:
        where = indexes[file_index].file.path.name
        scope = ".".join(card.scope) if card.scope else "top level"
        seen.setdefault(x_ref.casefold(), []).append(f"{where} ({scope})")
    for folded, places in seen.items():
        if len(places) > 1:
            raise MismatchPlanError(
                "ambiguous_instance_ref",
                f"Instance reference {folded!r} is declared at {len(places)} sites "
                f"({', '.join(sorted(places))}); each would take its own values while "
                "one is reported, so give the declarations distinct references",
            )


def _refuse_or_skip_without_mos(
    x_ref: str,
    subckt: str,
    children: list[tuple[int, SpiceCard]],
    indexes: list[_FileIndex],
    definitions: dict[str, list[tuple[int, int]]],
    skips: list[MismatchSkip],
) -> None:
    """Name what a body without a first-level ``M`` actually is."""
    for card in _instances(children, "X"):
        inner_view = InstanceLine.from_card(card)
        if inner_view.model is None:
            continue
        sites = definitions.get(inner_view.model.casefold(), [])
        if not sites:
            continue
        def_file, opener_index = sites[0]
        nested = _direct_children(indexes[def_file], opener_index)
        if _instances(nested, "M"):
            raise MismatchPlanError(
                "nested_fet_unsupported",
                f"Instance {x_ref!r} reaches its MOS device through two subcircuit "
                f"levels ({subckt} -> {inner_view.model}); this engine patches exactly "
                "one level, so instantiate the inner device wrapper directly",
            )
    skips.append(
        MismatchSkip(
            x_ref,
            "no_mos_at_depth",
            f"{x_ref} instantiates {subckt!r}, whose body holds no MOS device",
        )
    )


def _choose_inner(
    x_ref: str,
    subckt: str,
    inner_cards: list[SpiceCard],
    wanted: set[str] | None,
    *,
    exact: bool,
) -> list[SpiceCard]:
    by_ref = {card.name.casefold(): card for card in inner_cards if card.name}
    if wanted is None:
        if exact and len(inner_cards) > 1:
            names = ", ".join(sorted(card.name or "" for card in inner_cards))
            raise MismatchPlanError(
                "ambiguous_inner_device",
                f"Instance {x_ref!r} instantiates {subckt!r}, whose body holds "
                f"{len(inner_cards)} MOS devices ({names}); name the one this value is "
                f"for as {x_ref}.<device>",
            )
        return list(inner_cards)
    chosen: list[SpiceCard] = []
    for name in sorted(wanted):
        card = by_ref.get(name.casefold())
        if card is None:
            names = ", ".join(sorted(card.name or "" for card in inner_cards))
            raise MismatchPlanError(
                "inner_device_not_found",
                f"Subcircuit {subckt!r} instantiated by {x_ref!r} holds no MOS device "
                f"{name!r} (it holds: {names})",
            )
        chosen.append(card)
    return chosen


def _refuse_preexisting(
    x_ref: str,
    view: InstanceLine,
    params: Sequence[str],
    where: str,
) -> None:
    for param in params:
        if view.get_param(param) is not None:
            raise MismatchPlanError(
                "preexisting_mismatch_param",
                f"The {where} of {x_ref!r} already sets {param!r}; applying a mismatch "
                "value on top of it would silently double-apply the shift, so this deck "
                f"is refused — remove the existing {param} to let the engine own it",
            )


def _require_bsim(x_ref: str, inner_view: InstanceLine, indexes: list[_FileIndex]) -> None:
    model = inner_view.model or ""
    level = _model_level(indexes, model)
    if level is None:
        raise MismatchPlanError(
            "inner_model_unresolved",
            f"Model {model!r} of inner device {inner_view.ref!r} (reached through "
            f"{x_ref!r}) declares no LEVEL in the include closure, so it cannot be "
            f"confirmed to accept {VTH_PARAM}/{MOBILITY_PARAM}; ngspice would stop with "
            f"'{NGSPICE_UNKNOWN_PARAMETER}' at run time if it does not",
        )
    as_int = int(level)
    if as_int in BSIM_LEVELS:
        return
    if as_int in _REFUSING_LEVELS:
        detail = f"measured: ngspice stops with '{NGSPICE_UNKNOWN_PARAMETER}'"
    else:
        detail = f"{VTH_PARAM}/{MOBILITY_PARAM} are BSIM instance parameters"
    raise MismatchPlanError(
        "non_bsim_inner_device",
        f"Inner device {inner_view.ref!r} (reached through {x_ref!r}) uses model "
        f"{model!r} at LEVEL {as_int}, which does not accept {VTH_PARAM}/"
        f"{MOBILITY_PARAM} ({detail}); per-instance mismatch needs a BSIM device "
        f"(LEVEL {', '.join(str(level) for level in sorted(BSIM_LEVELS))})",
    )


def _allocate_clone_name(subckt: str, taken: set[str]) -> str:
    candidate = f"{subckt}{CLONE_SUFFIX}"
    suffix = 2
    while candidate.casefold() in taken:
        candidate = f"{subckt}{CLONE_SUFFIX}_{suffix}"
        suffix += 1
    taken.add(candidate.casefold())
    return candidate


def _render_clone(
    file: ClosureFile,
    cards: list[SpiceCard],
    opener_index: int,
    clone_name: str,
    inner_refs: Sequence[str],
) -> list[SpiceCard]:
    """Build one patched copy of a device subcircuit, from the cards given.

    The copy is renamed, gains a reserved parameter pair per targeted inner
    device, and forwards each pair onto that device's ``M`` card. Nothing else
    about the body changes, so a caller reading the clone sees the library's own
    device with two knobs added — including whatever a run has already
    perturbed inside it, since the body is read from the cards handed in rather
    than from a copy taken earlier.

    Cards, not text: they splice straight into the deck being written, and a
    caller that wants text has ``emit``.
    """
    stop = _closer_index(cards, opener_index)
    raw = "".join(line for card in cards[opener_index : stop + 1] for line in card.raw_lines)
    clone_cards = lex(raw).cards
    if scan_include_references(clone_cards, file.path, depth=file.depth):
        raise MismatchPlanError(
            "clone_include_unsupported",
            f"Subcircuit {cards[opener_index].name!r} pulls in another file from "
            "inside its own body; copying it would duplicate that include, so this "
            "device cannot be patched per instance",
        )
    original = cards[opener_index].name or ""
    rename_subckt(clone_cards, original, clone_name)

    declared = _declared_params(clone_cards)
    for param in MISMATCH_PARAMS:
        if param in declared:
            raise MismatchPlanError(
                "preexisting_mismatch_param",
                f"Subcircuit {original!r} already declares {param!r} as a parameter of "
                "its own; forwarding a second one would double-apply the shift, so this "
                f"device is refused — drive the library's own {param} instead",
            )
    forwarded: dict[str, str] = {}
    for inner_ref in inner_refs:
        for param in MISMATCH_PARAMS:
            name = forwarded_param(param, inner_ref)
            if name in declared:
                raise MismatchPlanError(
                    "param_namespace_collision",
                    f"Subcircuit {original!r} already declares a parameter named "
                    f"{name!r}, which is the reserved name this engine forwards "
                    f"{param} through; rename the library parameter",
                )
            forwarded[name] = "0" if param == VTH_PARAM else "1"

    _declare_defaults(clone_cards, forwarded)
    folded_inner = {ref.casefold() for ref in inner_refs}
    for card in clone_cards:
        if card.kind != "instance" or not card.name:
            continue
        if card.name.casefold() not in folded_inner:
            continue
        view = InstanceLine.from_card(card)
        for param in MISMATCH_PARAMS:
            view.set_param(param, f"{{{forwarded_param(param, view.ref)}}}")
    return clone_cards


def _declared_params(clone_cards: list[SpiceCard]) -> set[str]:
    """Every parameter name the cloned subcircuit already declares."""
    opener = clone_cards[0]
    declared = {key.casefold() for key in SubcktCard.from_card(opener).param_defaults}
    depth = len(opener.scope) + 1
    for card in clone_cards:
        if card.kind == "param" and len(card.scope) == depth:
            for token in tokenize_body(card.body)[1:]:
                if token.kind == TokenKind.KEY_VALUE and token.key:
                    declared.add(token.key.casefold())
    return declared


def _declare_defaults(clone_cards: list[SpiceCard], forwarded: dict[str, str]) -> None:
    """Give the clone neutral defaults for every forwarded parameter.

    A foundry cell declares its parameters on a ``.param`` card inside the body
    rather than on the ``.subckt`` line, and an instance parameter overrides
    either spelling; the defaults follow whichever the library already uses so
    the patched cell keeps the shape its author wrote.
    """
    if not forwarded:
        return
    opener = clone_cards[0]
    depth = len(opener.scope) + 1
    body_param = next(
        (card for card in clone_cards if card.kind == "param" and len(card.scope) == depth),
        None,
    )
    if body_param is not None:
        trailing = "".join(f" {key} = {value}" for key, value in forwarded.items())
        _append_to_code(body_param, trailing)
        return
    view = SubcktCard.from_card(opener)
    for key, value in forwarded.items():
        view.set_param_default(key, value)


def _append_to_code(card: SpiceCard, text: str) -> None:
    """Append to a card's code, ahead of anything trailing it.

    A library writes ``.param w = 1 ; channel width`` as readily as not. The
    lexer keeps that comment out of ``body`` but not out of the line, so an
    insertion addressed by body offset lands before it — where the parameter is
    live code — while rewriting the whole body would drop it. Either way the
    declaration must end up outside the comment: inside it the parameter never
    exists and the ``{...}`` the inner device references is undefined at
    simulation time.
    """
    comment = next(
        (token for token in tokenize_body(card.body) if token.kind == TokenKind.COMMENT_TRAIL),
        None,
    )
    at = comment.body_offset if comment is not None else len(card.body.rstrip())
    try:
        card.replace_span(at, at, text)
    except (SpiceLexError, ValueError):
        card.replace_body(card.body.rstrip() + text)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def render(
    files: Sequence[ClosureFile],
    plan: MismatchPlan,
    values: dict[str, MismatchValues],
    *,
    cards: dict[int, list[SpiceCard]] | None = None,
) -> dict[int, str]:
    """Apply one run's per-instance values; return the text of changed files.

    Keys of ``values`` are hierarchical device references (``X1.M0``) as the
    plan reports them. Files not named in the result are unchanged, so a caller
    only stages what actually moved.

    ``cards`` hands over files the caller already holds lexed — a run that has
    just perturbed the deck card by card would otherwise emit it and have it
    parsed straight back. A file supplied this way is written from those cards
    and its ``ClosureFile.text`` is never read.
    """
    require_positional(files)
    by_ref = plan.by_ref
    supplied = {ref.casefold(): value for ref, value in values.items()}
    unknown = sorted(ref for ref in values if ref.casefold() not in by_ref)
    if unknown:
        raise MismatchPlanError(
            "unplanned_instance",
            f"No mismatch plan target for {', '.join(unknown)}; the plan covers "
            f"{', '.join(sorted(t.ref for t in plan.targets)) or 'no device'}",
        )
    applied = [by_ref[ref] for ref in supplied]
    if not applied:
        return {}

    cards_by_file: dict[int, list[SpiceCard]] = dict(cards or {})
    instances_by_file: dict[int, dict[str, SpiceCard]] = {}
    # A file read to copy a subcircuit out of is not a file that changed, so
    # rewriting is tracked separately from lexing: only what actually moved is
    # handed back for the caller to stage.
    rewritten: set[int] = set()

    def cards_for(index: int) -> list[SpiceCard]:
        if index not in cards_by_file:
            cards_by_file[index] = lex(files[index].text).cards
        return cards_by_file[index]

    def instance_in(index: int, folded: str) -> SpiceCard | None:
        """The first card declaring ``folded`` in one file, or None.

        First rather than last: a reference declared both at top level and
        inside some body is ambiguous, and planning already refused every
        selection where that could matter, so the tiebreak only has to stay
        the one the plan was built under.
        """
        lookup = instances_by_file.get(index)
        if lookup is None:
            lookup = {}
            for card in cards_for(index):
                if card.kind == "instance" and card.name:
                    lookup.setdefault(card.name.casefold(), card)
            instances_by_file[index] = lookup
        return lookup.get(folded)

    by_instance: dict[tuple[int, str], list[XFetTarget]] = {}
    for target in applied:
        by_instance.setdefault((target.x_file, target.x_ref.casefold()), []).append(target)

    for (file_index, x_folded), group in by_instance.items():
        card = instance_in(file_index, x_folded)
        if card is None:
            raise MismatchPlanError(
                "instance_not_found",
                f"Instance {group[0].x_ref!r} is no longer present in "
                f"{files[file_index].path.name}",
            )
        rewritten.add(file_index)
        view = InstanceLine.from_card(card)
        view.set_model(group[0].clone)
        for target in group:
            value = supplied[target.ref.casefold()]
            if value.delvto is not None:
                view.set_param(forwarded_param(VTH_PARAM, target.inner_ref), value.delvto)
            if value.mulu0 is not None:
                view.set_param(forwarded_param(MOBILITY_PARAM, target.inner_ref), value.mulu0)

    used = {target.clone.casefold() for target in applied}
    root = cards_for(0)
    for clone in plan.clones:
        if clone.name.casefold() not in used:
            continue
        if clone.text is not None:
            block = lex(clone.text).cards
        else:
            source = files[clone.def_file]
            def_cards = cards_for(clone.def_file)
            opener = _locate_subckt(def_cards, source, clone)
            block = _render_clone(source, def_cards, opener, clone.name, clone.inner_refs)
        inject_cards_before_end(root, block)
        rewritten.add(0)

    return {index: emit(cards_by_file[index]) for index in sorted(rewritten)}


def _locate_subckt(cards: list[SpiceCard], source: ClosureFile, clone: MismatchClone) -> int:
    """Find the definition the plan chose, in the file as it stands now.

    By name and section rather than by the position planning recorded: this
    runs only for a subcircuit defined in the root deck, which is the one file
    a run rewrites, so cards can have been inserted ahead of it and a position
    that has drifted would silently copy whatever now sits there.
    """
    folded = clone.subckt.casefold()
    sections = card_sections(cards, source.path, source.depth)
    matches = [
        i
        for i, card in enumerate(cards)
        if card.kind == "subckt"
        and card.name
        and card.name.casefold() == folded
        and sections[i] == clone.section
    ]
    if len(matches) != 1:
        raise MismatchPlanError(
            "subckt_unresolved",
            f"Subcircuit {clone.subckt!r} now has {len(matches)} definitions in "
            f"{source.path.name} where the plan resolved one",
        )
    return matches[0]


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MismatchDraws:
    """One run's draws over a plan: what to write, and what was drawn.

    ``values`` is what ``render`` writes. ``delvto``/``mulu0`` are the same
    numbers keyed by device reference for the receipt, in the units they were
    written in — an additive shift of the signed threshold, and a mobility
    multiplier — so a caller can check the receipt against the deck. A device
    whose rule drew nothing appears in neither.
    """

    values: dict[str, MismatchValues]
    delvto: dict[str, float]
    mulu0: dict[str, float]


def draw_mismatch(
    plan: MismatchPlan,
    rules: list[MismatchRule],
    sampler: MCSampler,
) -> MismatchDraws:
    """Draw one run's per-instance values for every device the plan reaches.

    A rule aimed at a whole ``X`` instance draws independently for every MOS
    device inside it, so a body holding a matched pair gets two uncorrelated
    shifts rather than one shared one.

    The draw order is the plan's target order, which is what makes a seeded run
    reproducible: the sampler is keyed per device, but a caller that reordered
    the targets between two runs of one seed would still get two different
    circuits out of them.
    """
    values: dict[str, MismatchValues] = {}
    delvto: dict[str, float] = {}
    mulu0: dict[str, float] = {}
    for target in plan.targets:
        rule = find_mismatch_rule(target.x_ref, rules)
        if rule is None or (rule.avt <= 0.0 and rule.ak <= 0.0):
            continue
        width, length = require_geometry(target)
        deltas = sample_instance_mismatch(
            sampler,
            InstanceGeometry(
                ref=target.ref,
                model_name=target.model_name,
                width_m=width,
                length_m=length,
            ),
            rule,
        )
        if deltas["dvth"] == 0.0 and deltas["dk_over_k"] == 0.0:
            continue
        values[target.ref] = MismatchValues(
            delvto=deltas["dvth"] if deltas["dvth"] != 0.0 else None,
            mulu0=1.0 + deltas["dk_over_k"] if deltas["dk_over_k"] != 0.0 else None,
        )
        delvto[target.ref] = deltas["dvth"]
        mulu0[target.ref] = 1.0 + deltas["dk_over_k"]
    return MismatchDraws(values=values, delvto=delvto, mulu0=mulu0)


def overlapping_claims(
    plan: MismatchPlan,
    prefixes: Sequence[str],
) -> tuple[str, tuple[str, ...]] | None:
    """The first instance more than one prefix claims, with those prefixes.

    Composing two mismatch rules on one device is not defined — the
    coefficients are per-device sigmas, not contributions to add — and picking
    one by declaration order would draw from a rule the caller has no reason to
    expect. What to do about it is the caller's (each has its own error class);
    naming the collision is all this decides.
    """
    for target in plan.targets:
        claimed = tuple(p for p in prefixes if matches_prefix(target.x_ref, p))
        if len(claimed) > 1:
            return target.x_ref, claimed
    return None
