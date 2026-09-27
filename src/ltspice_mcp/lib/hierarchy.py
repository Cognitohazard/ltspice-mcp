"""Read-only, bounded elaboration of an explicitly supported initial SPICE deck.

Runtime identity is a tuple of reference segments; source occurrences retain
file, physical line and section. Captured inputs are per-file revisions, not an
atomic filesystem snapshot. The pure resolver never consults the filesystem.
Local definitions, conditional structure and opaque control programs are refused.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Literal

from ltspice_mcp.errors import NetlistError
from ltspice_mcp.lib.deck_staging import (
    DEFAULT_INCLUDE_DEPTH,
    card_sections,
    resolve_existing,
    resolve_reference,
    scan_include_references,
    unquote,
)
from ltspice_mcp.lib.encoding import decode_spice_bytes
from ltspice_mcp.lib.format import parse_spice_value
from ltspice_mcp.lib.hierarchy_expr import (
    Environment,
    NumericFact,
    references_sibling,
    scaled_geometry,
)
from ltspice_mcp.lib.pathutil import resolve_safe_path
from ltspice_mcp.lib.spice_lex import (
    SpiceCard,
    Token,
    TokenKind,
    find_matching_ends,
    lex,
    tokenize_body,
)
from ltspice_mcp.lib.spice_lex_views import InstanceLine, ModelCard, SubcktCard

MAX_BYTES = 16 * 1024 * 1024
MAX_CARDS = 100_000
MAX_INSTANCES = 20_000
MAX_INSTANCE_DEPTH = 32
_SAFE_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
_SAFE_NODE = re.compile(r"[A-Za-z0-9_]+\Z")


@dataclass(frozen=True)
class SemanticProfile:
    simulator: Literal["ltspice", "ngspice"]
    ngbehavior: str | None = None
    evaluation_context: str = "initial_deck"

    def __post_init__(self) -> None:
        if self.simulator not in {"ltspice", "ngspice"}:
            raise NetlistError("hierarchy supports ltspice or ngspice")
        if self.simulator == "ltspice" and self.ngbehavior is not None:
            raise NetlistError("ngbehavior is only valid for ngspice")
        if self.simulator == "ngspice" and self.ngbehavior is None:
            raise NetlistError("ngspice hierarchy requires its effective ngbehavior")
        if self.simulator == "ngspice" and (self.ngbehavior or "").casefold() not in {
            "",
            "hsa",
            "kiltpsa",
        }:
            raise NetlistError(
                "unsupported hierarchy ngbehavior; supported profiles are '', hsa, kiltpsa"
            )
        if self.evaluation_context != "initial_deck":
            raise NetlistError("hierarchy supports only initial_deck evaluation context")


@dataclass(frozen=True)
class CapturedFile:
    path: Path
    content: bytes
    existence: tuple[tuple[Path, bool], ...]
    targets: tuple[tuple[str, Path | None], ...]

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.content).hexdigest()


@dataclass(frozen=True)
class Source:
    path: str
    line: int
    section: str | None
    definition: str | None = None


@dataclass(frozen=True)
class Node:
    scope: tuple[str, ...]
    name: str
    voltage_trace: str | None
    reason: str | None = None


@dataclass(frozen=True)
class ModelDefinition:
    name: str
    source: Source
    raw: str
    level: float | None


@dataclass(frozen=True)
class ResolvedInstance:
    instance: tuple[str, ...]
    reference: str
    element: str
    source: Source
    raw: str
    nodes: tuple[Node, ...]
    connectivity_reason: str | None
    model_name: str | None
    model_source: Source | None
    model_reason: str | None
    value: NumericFact
    parameters: tuple[tuple[str, NumericFact], ...]
    environment: tuple[tuple[str, NumericFact], ...]
    geometry: tuple[tuple[str, NumericFact], ...]
    ports: tuple[tuple[str, Node], ...]
    device: str | None
    save: str | None
    address_reason: str | None
    model_family: tuple[ModelDefinition, ...] = ()

    def row(self) -> dict[str, Any]:
        return {
            "instance": list(self.instance),
            "reference": self.reference,
            "element": self.element,
            "source": asdict(self.source),
            "raw": self.raw,
            "nodes": [{**asdict(n), "scope": list(n.scope)} for n in self.nodes],
            "connectivity_reason": self.connectivity_reason,
            "model": {
                "name": self.model_name,
                "source": asdict(self.model_source) if self.model_source else None,
                "reason": self.model_reason,
                "family": [
                    {"name": model.name, "source": asdict(model.source), "level": model.level}
                    for model in self.model_family
                ],
            },
            "value": asdict(self.value),
            "parameters": {k: asdict(v) for k, v in self.parameters},
            "environment": {k: asdict(v) for k, v in self.environment},
            "geometry": {k: asdict(v) for k, v in self.geometry},
            "ports": {k: {**asdict(v), "scope": list(v.scope)} for k, v in self.ports},
            "address": {"device": self.device, "save": self.save, "reason": self.address_reason},
        }


@dataclass(frozen=True)
class Hierarchy:
    profile: SemanticProfile
    instances: tuple[ResolvedInstance, ...]
    inputs: tuple[CapturedFile, ...]
    definitions: tuple[_Definition, ...] = ()
    occurrences: tuple[ActiveOccurrence, ...] = ()
    library_bindings: tuple[LibraryBinding, ...] = ()
    scale: NumericFact = field(default_factory=lambda: NumericFact("1", 1.0, status="resolved"))
    include_bindings: tuple[IncludeBinding, ...] = ()

    def binding(self) -> dict[str, Any]:
        return {
            "profile": asdict(self.profile),
            "inputs": [
                {
                    "path": str(f.path),
                    "sha256": f.digest,
                    "existence": [(str(p), exists) for p, exists in f.existence],
                    "targets": [(raw, str(p) if p else None) for raw, p in f.targets],
                }
                for f in self.inputs
            ],
        }


@dataclass(frozen=True)
class ActiveOccurrence:
    card: SpiceCard
    source: Source


_Occurrence = ActiveOccurrence


@dataclass(frozen=True)
class LibraryBinding:
    source: Source
    target: Path
    section: str


@dataclass(frozen=True)
class IncludeBinding:
    source: Source
    target: Path
    section: str | None


@dataclass(frozen=True)
class _Definition:
    view: SubcktCard
    source: Source
    body: tuple[_Occurrence, ...]
    closer: Source


def _tokens(card: SpiceCard) -> list[Token]:
    return [t for t in tokenize_body(card.body) if t.kind != TokenKind.COMMENT_TRAIL]


def _assignments(card: SpiceCard, *, only: bool = False) -> dict[str, str]:
    result: dict[str, str] = {}
    tokens = _tokens(card)[1:]
    if card.kind == "model":
        tokens = [
            inner
            for token in tokens
            for inner in (
                tokenize_body(token.text[1:-1]) if token.kind == TokenKind.PARENED else [token]
            )
        ]
    for token in tokens:
        if token.kind == TokenKind.KEY_VALUE:
            assert token.key is not None and token.value is not None
            key = token.key.casefold()
            if key in result:
                raise NetlistError(f"duplicate assignment {token.key} at line {card.line_start}")
            result[key] = token.value
        elif only:
            if not result or token.kind not in {TokenKind.BARE, TokenKind.PARENED}:
                raise NetlistError(f"unsupported parameter assignment at line {card.line_start}")
            # Foundry .param expressions can be unbraced and spaced. Keep the
            # whole expression as a fact; unsupported functions remain unresolved.
            last_key = next(reversed(result))
            result[last_key] += " " + token.text
    return result


def parse_hierarchy_cards(content: bytes, *, root: bool) -> list[SpiceCard]:
    text = decode_spice_bytes(content)
    if root:
        # Root decks begin with a title; included fragments begin with a card.
        # Replace before lexing so a title cannot change scope or terminate input.
        lines = text.splitlines(keepends=True)
        text = "* hierarchy root title\n" + "".join(lines[1:])
    parsed = lex(text)
    # Explicit validation below handles active boundaries, including definitions
    # whose body includes another file. Inactive sections cannot poison a read.
    return [card for card in parsed.cards if not card.trailing]


def walk_active(
    root: Path,
    get_file: Callable[[Path, int], CapturedFile],
    profile: SemanticProfile,
    library_bindings: list[LibraryBinding] | None = None,
    include_bindings: list[IncludeBinding] | None = None,
) -> tuple[_Occurrence, ...]:
    result: list[_Occurrence] = []
    count = 0
    lexical_scope: list[str] = []

    def visit(path: Path, section: str | None, stack: tuple[tuple[Path, str | None], ...]) -> None:
        nonlocal count
        key = (path, section.casefold() if section else None)
        if key in stack:
            raise NetlistError(f"active include cycle at {path}")
        if len(stack) > DEFAULT_INCLUDE_DEPTH:
            raise NetlistError(f"include depth exceeds {DEFAULT_INCLUDE_DEPTH}")
        captured = get_file(path, len(stack))
        cards = parse_hierarchy_cards(captured.content, root=not stack)
        count += len(cards)
        if count > MAX_CARDS:
            raise NetlistError(f"hierarchy exceeds {MAX_CARDS} cards")
        existence = dict(captured.existence)
        sections = card_sections(cards, path, len(stack), exists=existence.__getitem__)
        refs = {
            id(ref.card): ref
            for ref in scan_include_references(
                cards, path, depth=len(stack), exists=existence.__getitem__
            )
        }
        targets = dict(captured.targets)
        opened: str | None = None
        declared: set[str] = set()
        for card, owner in zip(cards, sections, strict=True):
            tokens = _tokens(card)
            head = tokens[0].text.casefold() if tokens else ""
            if head == ".lib" and id(card) not in refs:
                if len(tokens) != 2 or opened is not None:
                    raise NetlistError(f"malformed library section at {path}:{card.line_start}")
                opened = unquote(tokens[1].text).casefold()
                if opened in declared:
                    raise NetlistError(f"duplicate library section {opened} in {path}")
                declared.add(opened)
                continue
            if head == ".endl":
                if (
                    opened is None
                    or len(tokens) > 2
                    or (len(tokens) == 2 and unquote(tokens[1].text).casefold() != opened)
                ):
                    raise NetlistError(f"malformed .endl at {path}:{card.line_start}")
                opened = None
                continue
            active = (
                owner is None
                if section is None
                else owner is not None and owner.casefold() == section.casefold()
            )
            if not active:
                continue
            if card.kind == "comment" and any(
                line.lstrip().startswith("+") for line in card.raw_lines
            ):
                raise NetlistError(f"orphan continuation at {path}:{card.line_start}")
            if card.kind == "end":
                break
            ref = refs.get(id(card))
            if ref is not None:
                if (
                    ref.section
                    and profile.simulator == "ngspice"
                    and any(mode in (profile.ngbehavior or "").casefold() for mode in ("lt", "ps"))
                ):
                    raise NetlistError(
                        "sectioned libraries require ngbehavior without lt/ps reinterpretation"
                    )
                target = targets.get(ref.raw_path)
                if target is None:
                    raise NetlistError(f"missing include {ref.raw_path!r} from {path}")
                source = Source(
                    str(path), card.line_start, owner, lexical_scope[-1] if lexical_scope else None
                )
                result.append(_Occurrence(card, source))
                if include_bindings is not None:
                    include_bindings.append(IncludeBinding(source, target, ref.section))
                if ref.section is not None and library_bindings is not None:
                    library_bindings.append(LibraryBinding(source, target, ref.section))
                visit(target, ref.section, (*stack, key))
            else:
                if card.kind == "ends" and lexical_scope:
                    lexical_scope.pop()
                result.append(
                    _Occurrence(
                        card,
                        Source(
                            str(path),
                            card.line_start,
                            owner,
                            lexical_scope[-1] if lexical_scope else None,
                        ),
                    )
                )
                if card.kind == "subckt":
                    lexical_scope.append(card.name or "")
        if opened is not None:
            raise NetlistError(f"unclosed library section {opened} in {path}")
        if section is not None and section.casefold() not in declared:
            raise NetlistError(f"missing library section {section!r} in {path}")

    visit(root, None, ())
    return tuple(result)


def load_hierarchy(
    path: str,
    allowed_roots: list[Path],
    profile: SemanticProfile,
    *,
    simulator_roots: Sequence[Path] = (),
) -> Hierarchy:
    """Capture bounded bytes and path/classification facts, then resolve purely."""
    root = resolve_safe_path(path, allowed_roots)
    if root.suffix.casefold() not in {".cir", ".net", ".sp"}:
        raise NetlistError(
            "hierarchy requires a .cir/.net/.sp netlist; explicitly export .asc first"
        )
    files: dict[Path, CapturedFile] = {}
    total = 0

    def capture(source: Path, depth: int) -> CapturedFile:
        nonlocal total
        if source in files:
            return files[source]
        checked = resolve_safe_path(str(source), allowed_roots + list(simulator_roots))
        with checked.open("rb") as stream:
            content = stream.read(MAX_BYTES - total + 1)
        total += len(content)
        if total > MAX_BYTES:
            raise NetlistError(f"hierarchy exceeds aggregate input limit of {MAX_BYTES} bytes")
        cards = parse_hierarchy_cards(content, root=depth == 0)
        if len(cards) > MAX_CARDS:
            raise NetlistError(f"hierarchy exceeds {MAX_CARDS} cards")
        existence: dict[Path, bool] = {}

        def exists(candidate: Path) -> bool:
            if candidate not in existence:
                existence[candidate] = candidate.exists()
            return existence[candidate]

        refs = scan_include_references(cards, source, depth=depth, exists=exists)
        card_sections(cards, source, depth, exists=exists)
        targets = tuple(
            (ref.raw_path, resolve_existing(resolve_reference(source.parent, ref.raw_path)))
            for ref in refs
        )
        captured = CapturedFile(source, content, tuple(existence.items()), targets)
        files[source] = captured
        return captured

    walk_active(root, capture, profile)
    return resolve_hierarchy(root, files, profile)


def _parameters(body: Sequence[_Occurrence], profile: SemanticProfile) -> dict[str, str]:
    params: dict[str, str] = {}
    for item in body:
        if item.card.kind == "param":
            values = _assignments(item.card, only=True)
            if params.keys() & values.keys() and not (
                profile.simulator == "ngspice" and profile.ngbehavior == "hsa"
            ):
                raise NetlistError(f"duplicate parameter declaration at {item.source}")
            params.update(values)
    return params


def _models(body: Sequence[_Occurrence], enclosing: str | None) -> dict[str, ModelDefinition]:
    models: dict[str, ModelDefinition] = {}
    for item in body:
        if item.card.kind != "model":
            continue
        params = _assignments(item.card)
        model = ModelCard.from_card(item.card)
        key = model.name.casefold()
        if key in models:
            raise NetlistError(f"duplicate active model {model.name}")
        try:
            level = parse_spice_value(params["level"]) if "level" in params else None
        except ValueError:
            level = None
        models[key] = ModelDefinition(
            model.name, replace(item.source, definition=enclosing), item.card.body, level
        )
    return models


def _shape(card: SpiceCard) -> tuple[InstanceLine, bool]:
    view = InstanceLine.from_card(card)
    _assignments(card)
    tokens = _tokens(card)[1:]
    first = next((i for i, t in enumerate(tokens) if t.kind == TokenKind.KEY_VALUE), len(tokens))
    if any(t.kind != TokenKind.KEY_VALUE for t in tokens[first:]):
        raise NetlistError(
            f"unsupported tokens after instance parameters at line {card.line_start}"
        )
    pos = tokens[:first]
    kind = view.ref[0].upper()
    if kind == "X":
        if pos and pos[-1].text.casefold() == "params:":
            pos = pos[:-1]
        if not pos or any(t.kind != TokenKind.BARE for t in pos):
            raise NetlistError(f"unsupported subcircuit call at line {card.line_start}")
        view.nodes = [t.text for t in pos[:-1]]
        view.model = pos[-1].text
        return view, True
    if kind == "M":
        return view, len(pos) == 5 and all(t.kind == TokenKind.BARE for t in pos)
    if kind in "RCL":
        keyed = any(t.key and t.key.casefold() == kind.casefold() for t in tokens[first:])
        return view, (
            len(pos) == (2 if keyed else 3) and all(t.kind == TokenKind.BARE for t in pos[:2])
        )
    if kind in "VI":
        return view, len(pos) >= 3 and all(t.kind == TokenKind.BARE for t in pos[:2])
    return view, False


def _address(
    path: tuple[str, ...], kind: str, profile: SemanticProfile
) -> tuple[str | None, str | None, str | None]:
    if kind not in "RM" or not all(len(p) > 1 and _SAFE_NAME.fullmatch(p) for p in path):
        return None, None, "backend address unavailable for this element or spelling"
    if profile.simulator == "ngspice":
        device = ".".join(p.lower() for p in path)
        if len(path) > 1:
            device = kind.lower() + "." + device
        parameter = "gm" if kind == "M" else "i"
        return device, f".save @{device}[{parameter}]", None
    device = ":".join(p.lower() for p in path)
    if kind == "M":
        return device, ".options logopinfo", None
    return device, f".save I({device})", None


def resolve_hierarchy(
    root: Path, files: Mapping[Path, CapturedFile], profile: SemanticProfile
) -> Hierarchy:
    """Resolve immutable captured inputs without live path, content or stat reads."""
    active: dict[Path, CapturedFile] = {}
    active_bytes = 0

    def active_file(path: Path, _depth: int) -> CapturedFile:
        nonlocal active_bytes
        captured = files[path]
        if path not in active:
            active_bytes += len(captured.content)
            if active_bytes > MAX_BYTES:
                raise NetlistError(f"hierarchy exceeds aggregate input limit of {MAX_BYTES} bytes")
            active[path] = captured
        return captured

    library_bindings: list[LibraryBinding] = []
    include_bindings: list[IncludeBinding] = []
    occurrences = walk_active(root, active_file, profile, library_bindings, include_bindings)
    definitions: dict[str, _Definition] = {}
    models: dict[str | None, dict[str, ModelDefinition]] = {None: {}}
    top: list[_Occurrence] = []
    cards = [o.card for o in occurrences]
    i = 0
    while i < len(occurrences):
        item = occurrences[i]
        card = item.card
        if card.kind == "subckt":
            _assignments(card)
            view = SubcktCard.from_card(card)
            parameters_started = False
            for token in _tokens(card)[2:]:
                if token.kind == TokenKind.KEY_VALUE or token.text.casefold() == "params:":
                    parameters_started = True
                elif parameters_started or token.kind != TokenKind.BARE:
                    raise NetlistError(f"unsupported subcircuit header at {item.source}")
            end = find_matching_ends(cards, i)
            if end is None:
                raise NetlistError(f"unclosed subcircuit {view.name}")
            closer = cards[end]
            if len(_tokens(closer)) > 2:
                raise NetlistError(f"malformed .ends for {view.name}")
            if closer.name and closer.name.casefold() != view.name.casefold():
                raise NetlistError(f"mismatched .ends for {view.name}")
            if view.name.casefold() in definitions:
                raise NetlistError(f"duplicate active subcircuit {view.name}")
            if len({p.casefold() for p in view.ports}) != len(view.ports):
                raise NetlistError(f"duplicate ports in {view.name}")
            body = occurrences[i + 1 : end]
            if any(o.card.kind in {"subckt", "ends"} for o in body):
                raise NetlistError(
                    "local/nested definitions are unsupported by hierarchy discovery"
                )
            definitions[view.name.casefold()] = _Definition(
                view, item.source, body, occurrences[end].source
            )
            models[view.name.casefold()] = _models(body, view.name)
            i = end + 1
            continue
        if card.kind == "ends":
            raise NetlistError(f"unmatched .ends at {item.source}")
        if card.kind == "model":
            entry = _models((item,), None)
            if models[None].keys() & entry.keys():
                raise NetlistError(f"duplicate active model {card.name}")
            models[None].update(entry)
        top.append(item)
        i += 1

    globals_: dict[str, str] = {"0": "0"}
    if profile.simulator == "ngspice":
        globals_["gnd"] = "0"
    scale_expression = "1"
    dynamic_reason = None
    for item in occurrences:
        tokens = _tokens(item.card)
        head = tokens[0].text.casefold() if tokens else ""
        if item.card.kind == "control":
            raise NetlistError(
                "opaque control programs may alter hierarchy; discovery is unsupported"
            )
        if head in {".if", ".elseif", ".else", ".endif"}:
            raise NetlistError("conditional hierarchy is unsupported")
        if head in {".alter", ".altermod", ".del", ".delete"}:
            raise NetlistError("runtime circuit alterations are unsupported")
        if head == ".step":
            dynamic_reason = "context-dependent .step: no single effective numeric value"
        if head == ".global":
            for token in tokens[1:]:
                globals_.setdefault(token.text.casefold(), token.text)
        if head in {".option", ".options"}:
            values = _assignments(item.card)
            if "scale" in values:
                if profile.simulator == "ltspice":
                    raise NetlistError("explicit scale is unsupported by LTspice")
                if item not in top:
                    raise NetlistError("local .option scale is unsupported")
                scale_expression = values["scale"]
    global_env = Environment(
        _parameters(top, profile), simulator=profile.simulator, dynamic_reason=dynamic_reason
    )
    scale = global_env.fact(scale_expression)
    rows: list[ResolvedInstance] = []
    node_cache: dict[tuple[tuple[str, ...], str], Node] = {}

    def node(name: str, scope: tuple[str, ...], ports: Mapping[str, Node]) -> Node:
        folded = name.casefold()
        if folded == "gnd" and profile.simulator == "ngspice" and profile.ngbehavior == "kiltpsa":
            raise NetlistError(
                "version-dependent ground alias 'gnd' in kiltpsa mode; "
                "use explicit node 0 or inspect a supported native/hsa deck"
            )
        if folded in globals_:
            scope, name = (), globals_[folded]
        elif folded in ports:
            return ports[folded]
        key = (tuple(segment.casefold() for segment in scope), name.casefold())
        if key in node_cache:
            return node_cache[key]
        trace = None
        reason = None
        if all(_SAFE_NAME.fullmatch(p) for p in scope) and _SAFE_NODE.fullmatch(name):
            sep = ":" if profile.simulator == "ltspice" else "."
            trace = "V(" + sep.join((*[p.lower() for p in scope], name.lower())) + ")"
        else:
            reason = "backend node spelling unavailable"
        resolved = Node(scope, name, trace, reason)
        node_cache[key] = resolved
        return resolved

    def expand(
        body: Sequence[_Occurrence],
        scope: tuple[str, ...],
        env: Environment,
        ports: Mapping[str, Node],
        stack: tuple[str, ...],
        enclosing: str | None,
    ) -> None:
        if len(scope) > MAX_INSTANCE_DEPTH:
            raise NetlistError(f"runtime hierarchy depth exceeds {MAX_INSTANCE_DEPTH}")
        refs: set[str] = set()
        for item in body:
            if item.card.kind != "instance":
                continue
            view, supported = _shape(item.card)
            folded = view.ref.casefold()
            if folded in refs:
                raise NetlistError(f"duplicate active reference {view.ref} in {scope}")
            refs.add(folded)
            path = (*scope, view.ref)
            kind = view.ref[0].upper()
            nodes = tuple(node(n, scope, ports) for n in view.nodes) if supported else ()
            params = _assignments(item.card)
            param_facts = tuple(
                (
                    k,
                    NumericFact(v, reason="sibling override dependency is unsupported")
                    if kind == "X"
                    and profile.simulator == "ngspice"
                    and references_sibling(v, params.keys() - {k})
                    else env.fact(v),
                )
                for k, v in params.items()
            )
            child = None
            child_env = env
            mapped: tuple[tuple[str, Node], ...] = ()
            model_source = None
            model_reason = None
            model_family: tuple[ModelDefinition, ...] = ()
            if kind == "X":
                child = definitions.get((view.model or "").casefold())
                if child is None:
                    raise NetlistError(f"unresolved child definition {view.model} at {path}")
                if child.view.name.casefold() in stack:
                    raise NetlistError(f"recursive subcircuit instantiation at {path}")
                if len(nodes) != len(child.view.ports):
                    raise NetlistError(f"port arity mismatch at {path}")
                model_source = child.source
                defaults = {k.casefold(): v for k, v in child.view.param_defaults.items()}
                local = _parameters(child.body, profile)
                expressions = (
                    {**defaults, **local}
                    if profile.simulator == "ngspice"
                    else {**local, **defaults}
                )
                child_env = Environment(
                    expressions,
                    simulator=profile.simulator,
                    parent=env,
                    overrides=dict(param_facts),
                    dynamic_reason=dynamic_reason,
                )
                mapped = tuple(zip(child.view.ports, nodes, strict=True))
            elif kind == "M" and supported:
                name = (view.model or "").casefold()
                for owner in (*reversed(stack), None):
                    candidates = models.get(owner, {})
                    model_family = tuple(
                        m
                        for key, m in candidates.items()
                        if key == name or key.startswith(name + ".")
                    )
                    if model_family:
                        break
                if any(m.name.casefold() != name for m in model_family):
                    model_reason = "binned model selection is unresolved"
                elif model_family:
                    model_source = model_family[0].source
                else:
                    model_reason = "model definition is unavailable in active inputs"
            elif view.model is not None or not supported:
                model_reason = "model binding unsupported for this element form"
            geometry = (
                tuple(
                    (key, scaled_geometry(env.fact(params.get(key)), scale)) for key in ("w", "l")
                )
                if kind == "M" and supported
                else ()
            )
            unit = {"R": "ohm", "C": "F", "L": "H", "V": "V", "I": "A"}.get(kind)
            value = (
                env.fact(view.value, unit)
                if supported
                else NumericFact(
                    view.value, reason="value semantics unsupported for this element form"
                )
            )
            device, save, reason = _address(path, kind, profile)
            if not supported:
                device, save, reason = None, None, "unsupported element form"
            source = Source(item.source.path, item.source.line, item.source.section, enclosing)
            rows.append(
                ResolvedInstance(
                    path,
                    view.ref,
                    kind,
                    source,
                    item.card.body,
                    nodes,
                    None if supported else "connectivity unsupported for this element form",
                    view.model if supported else None,
                    model_source,
                    model_reason,
                    value,
                    param_facts,
                    child_env.facts(),
                    geometry,
                    mapped,
                    device,
                    save,
                    reason,
                    model_family,
                )
            )
            if len(rows) > MAX_INSTANCES:
                raise NetlistError(f"runtime hierarchy exceeds {MAX_INSTANCES} instances")
            if child is not None:
                expand(
                    child.body,
                    path,
                    child_env,
                    {k.casefold(): v for k, v in mapped},
                    (*stack, child.view.name.casefold()),
                    child.view.name,
                )

    expand(top, (), global_env, {}, (), None)
    if profile.simulator == "ltspice":
        # Existing operating-point selectors match ancestral suffixes. A suffix
        # that also names another runtime device cannot promise exact selection.
        paths = {tuple(p.casefold() for p in row.instance) for row in rows}
        ambiguous = {path[i:] for path in paths for i in range(1, len(path)) if path[i:] in paths}
        rows = [
            replace(
                row,
                device=None,
                save=None,
                address_reason="operating_point selector also matches another instance",
            )
            if tuple(p.casefold() for p in row.instance) in ambiguous
            else row
            for row in rows
        ]
    rows.sort(key=lambda row: tuple(p.casefold() for p in row.instance))
    return Hierarchy(
        profile,
        tuple(rows),
        tuple(active[p] for p in sorted(active, key=str)),
        tuple(definitions.values()),
        occurrences,
        tuple(library_bindings),
        scale,
        tuple(include_bindings),
    )
