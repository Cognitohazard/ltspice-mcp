"""Schema generation and shrinking for the advertised tool surface.

Three jobs, all of them about the JSON Schema a tool publishes rather than
about what a tool *does*:

* ``ToolInput`` and ``build_input_schema`` — turn a Pydantic input model into
  the ``inputSchema`` the registry advertises, through the shrinking passes
  (title strip, type-keyword compaction, shared-fragment ``$defs`` hoist).
* ``strip_wire_prose`` — the advertised copy of a schema, carrying only the
  load-bearing field prose (see ``WIRE_PROSE_KEEP``).
* ``schema_from_typeddict`` — the output-schema generator, so a tool's
  ``structuredContent`` contract is derived from the TypedDict the lib already
  returns instead of being hand-written twice.

Split out of ``tools/_base`` so a change to how schemas are shrunk stops being
a change to the module every tool imports. ``tools/_base`` re-exports what the
tool modules and the tests reach for.
"""

import json
import re
import types as _stdlib_types
import typing
from functools import cache
from typing import Any, Literal, Union, get_args, get_origin, get_type_hints

from ltspice_mcp.lib.models import StrictModel


class ToolInput(StrictModel):
    """Base for top-level tool input models registered via @registry.tool(input_model=...)."""

    @classmethod
    def wire_input_schema(cls) -> dict[str, Any]:
        """The JSON Schema this tool advertises, before the shrinking passes.

        The model's own schema, except for a tool whose arguments are a
        top-level union: pydantic emits a bare ``oneOf`` for one of those, and
        MCP requires an object schema at the top level. Such a model overrides
        this to wrap its branches; ``build_input_schema`` calls it either way.
        """
        return cls.model_json_schema()


# JSON Schema keywords whose value is a map from caller-visible NAMES to
# schemas. Inside one of these the keys are the argument's own vocabulary, so
# nothing in it may be filtered as a schema keyword.
_SCHEMA_NAME_MAPS = frozenset(
    {"properties", "$defs", "definitions", "patternProperties", "dependentSchemas"}
)


def _strip_titles(node: Any, *, in_name_map: bool = False) -> Any:
    """Remove pydantic's ``title`` annotations, keeping a field named ``title``.

    Filtering the key at every level also deleted the entry for a *property*
    called ``title`` — which the plot recipe has and the handler reads — so a
    real, accepted argument was absent from every published schema and no client
    could discover it. Descend structurally instead: inside a ``properties`` or
    ``$defs`` map the keys are argument names, not schema keywords.
    """
    if isinstance(node, dict):
        if in_name_map:
            return {key: _strip_titles(value) for key, value in node.items()}
        return {
            key: _strip_titles(value, in_name_map=key in _SCHEMA_NAME_MAPS)
            for key, value in node.items()
            if key != "title"
        }
    if isinstance(node, list):
        return [_strip_titles(item) for item in node]
    return node


# Keywords that make a schema branch more than a plain type constraint, so it
# cannot be merged into a multi-type ``type`` array without changing meaning
# (``const``/``enum`` would also constrain the ``null`` member; the composition
# and reference keywords have no per-type semantics to inherit).
_UNFOLDABLE_KEYWORDS = frozenset({"$ref", "allOf", "anyOf", "const", "enum", "not", "oneOf"})

# Cheap pre-filter: a fragment shorter than the ``$ref`` that would replace it
# can never pay, so it is not worth hashing. The gain test below is the real
# gate.
_HOIST_MIN_CHARS = 28

# A shared definition has to earn its indirection: a reader now has to look the
# name up. Roughly twenty-five tokens of net saving is the line.
_HOIST_MIN_GAIN = 100


_JSON_TYPE_OF: dict[type, str] = {
    bool: "boolean",
    str: "string",
    int: "integer",
    float: "number",
}


def _compact_type_keywords(node: Any) -> Any:
    """Collapse type keywords that say something the schema already said.

    Pydantic spells every ``X | None`` as a two-branch ``anyOf``. Where the
    branches are plain type constraints, the equivalent ``{"type": [...]}``
    array says the same thing in far fewer characters: type-specific keywords
    (``items``, ``minLength``, ``minimum``) are no-ops for the other members,
    so ``{"items": ..., "type": ["array", "null"]}`` accepts exactly what the
    two-branch form did. Branches carrying ``const``/``enum``/``$ref`` are left
    alone — those constrain the *value*, not just its type.

    A ``type`` sitting beside a ``const`` of that same type is dropped for the
    same reason: the literal already pins the value, so the keyword narrows
    nothing. The union discriminators (``metric``, ``op``, ``kind``) are all
    of that shape, one per member of each tagged union.
    """
    if isinstance(node, list):
        return [_compact_type_keywords(item) for item in node]
    if not isinstance(node, dict):
        return node

    folded = {key: _compact_type_keywords(value) for key, value in node.items()}
    if "const" in folded and _JSON_TYPE_OF.get(type(folded["const"])) == folded.get("type"):
        folded.pop("type")
    branches = folded.get("anyOf")
    if not isinstance(branches, list) or len(branches) < 2:
        return folded
    if not all(
        isinstance(b, dict)
        and isinstance(b.get("type"), str)
        and not (_UNFOLDABLE_KEYWORDS & b.keys())
        for b in branches
    ):
        return folded

    # Only one branch may carry type-specific keywords; merging two constrained
    # branches would apply each one's keywords to the other's type.
    constrained = [b for b in branches if b.keys() != {"type"}]
    if len(constrained) > 1:
        return folded

    types = list(dict.fromkeys(b["type"] for b in branches))
    if len(types) != len(branches):
        return folded

    siblings = {k: v for k, v in folded.items() if k != "anyOf"}
    inner = dict(constrained[0]) if constrained else {}
    if inner.keys() & (siblings.keys() - {"type"}):
        return folded
    return {**inner, **siblings, "type": types}


def _shape_name(node: dict[str, Any]) -> str | None:
    """Name a fragment after the shape it describes, e.g. ``StringListOrNull``.

    Used when one fragment is shared by properties with different names, where
    naming it after any one of them would misdescribe the others. Only a plain
    type constraint gets one: a shape name that hid a real default (a
    ``Boolean`` that is secretly false unless set) would cost a reader more
    than the characters it saved.
    """
    if node.keys() - {"type", "items", "default"} or node.get("default") is not None:
        return None
    raw = node.get("type")
    types = [raw] if isinstance(raw, str) else list(raw or [])
    core = [t for t in types if t != "null"]
    if len(core) != 1:
        return None
    if core[0] == "array":
        items = node.get("items")
        if not isinstance(items, dict) or not isinstance(items.get("type"), str):
            return None
        base = f"{items['type'].capitalize()}List"
    else:
        base = core[0].capitalize()
    return base + ("OrNull" if "null" in types else "")


def _defs_name(hint: str, taken: set[str]) -> str:
    """Turn a naming hint into a ``$defs`` key that is free to use."""
    base = "".join(part[:1].upper() + part[1:] for part in hint.split("_") if part)
    base = re.sub(r"[^0-9A-Za-z]", "", base) or "Shared"
    name = base
    suffix = 0
    while name in taken:
        suffix += 1
        name = f"{base}Arg" if suffix == 1 else f"{base}Arg{suffix}"
    return name


def _hoist_shared_fragments(schema: dict[str, Any]) -> dict[str, Any]:
    """Move sub-schemas repeated across the document into shared ``$defs``.

    Pydantic re-emits a field's schema at every model that declares it, so a
    mixin field (``sources``, ``step``, ``spec`` on the analysis recipes) is
    serialized once per recipe. A single ``$defs`` entry with ``$ref`` use
    sites says the same thing once.

    Only nested property values are hoisted, and only when every use site sits
    under the same property name — the name it lends the ``$defs`` entry has to
    describe every reference, or the indirection costs a reader more than the
    characters it saves. The tools' own top-level properties are never hoisted:
    their inline ``description`` is the only documentation a caller gets.
    """
    counts: dict[str, int] = {}
    hints: dict[str, set[str]] = {}

    def survey(node: Any, key_hint: str | None, depth: int) -> None:
        if isinstance(node, list):
            for item in node:
                survey(item, key_hint, depth)
            return
        if not isinstance(node, dict):
            return
        if key_hint is not None:
            blob = json.dumps(node, separators=(",", ":"), sort_keys=True)
            if len(blob) >= _HOIST_MIN_CHARS:
                counts[blob] = counts.get(blob, 0) + 1
                hints.setdefault(blob, set()).add(key_hint)
        for name, value in node.items():
            if name == "properties" and isinstance(value, dict):
                # Depth 0 is the tool's own argument list — leave it inline.
                for prop, sub in value.items():
                    survey(sub, prop if depth else None, depth + 1)
            elif name == "$defs" and isinstance(value, dict):
                for sub in value.values():
                    survey(sub, None, max(depth, 1))
            else:
                survey(value, None, depth)

    survey(schema, None, 0)

    defs: dict[str, Any] = dict(schema.get("$defs") or {})
    taken = set(defs)
    replacements: dict[str, str] = {}
    for blob, count in sorted(counts.items()):
        if count < 2:
            continue
        fragment = json.loads(blob)
        names = hints[blob]
        hint = next(iter(names)) if len(names) == 1 else _shape_name(fragment)
        if hint is None:
            continue
        name = _defs_name(hint, taken)
        ref_cost = len(f'{{"$ref":"#/$defs/{name}"}}')
        # Net saving: every use site shrinks, minus the one def entry we add.
        gain = count * (len(blob) - ref_cost) - (len(name) + 3 + len(blob))
        if gain < _HOIST_MIN_GAIN:
            continue
        taken.add(name)
        defs[name] = fragment
        replacements[blob] = name

    if not replacements:
        return schema

    def rewrite(node: Any, key_hint: str | None, depth: int) -> Any:
        if isinstance(node, list):
            return [rewrite(item, key_hint, depth) for item in node]
        if not isinstance(node, dict):
            return node
        if key_hint is not None:
            blob = json.dumps(node, separators=(",", ":"), sort_keys=True)
            target = replacements.get(blob)
            if target is not None:
                return {"$ref": f"#/$defs/{target}"}
        out: dict[str, Any] = {}
        for name, value in node.items():
            if name == "properties" and isinstance(value, dict):
                out[name] = {
                    prop: rewrite(sub, prop if depth else None, depth + 1)
                    for prop, sub in value.items()
                }
            elif name == "$defs" and isinstance(value, dict):
                out[name] = {
                    def_name: rewrite(sub, None, max(depth, 1)) for def_name, sub in value.items()
                }
            else:
                out[name] = rewrite(value, None, depth)
        return out

    rewritten = rewrite({k: v for k, v in schema.items() if k != "$defs"}, None, 0)
    # Def bodies are rewritten with no key hint at their own root, so a hoisted
    # fragment can never be rewritten into a reference to itself; a fragment
    # nested inside another one is strictly shorter, so the refs cannot cycle.
    rewritten["$defs"] = {name: rewrite(body, None, 1) for name, body in defs.items()}
    return rewritten


def _referenced_defs(node: Any) -> set[str]:
    """Every ``#/$defs/<name>`` this node names, at any depth."""
    if isinstance(node, list):
        return {name for item in node for name in _referenced_defs(item)}
    if not isinstance(node, dict):
        return set()
    found: set[str] = set()
    ref = node.get("$ref")
    if isinstance(ref, str) and ref.startswith("#/$defs/"):
        found.add(ref.split("/")[-1])
    for value in node.values():
        found |= _referenced_defs(value)
    return found


def prune_unreferenced_defs(schema: dict[str, Any]) -> dict[str, Any]:
    """Drop the ``$defs`` entries nothing in the schema body reaches.

    A tool that advertises a compact stand-in for one sub-schema orphans
    whatever only the replaced shape referenced. An orphan is pure weight on
    the wire — every client downloads it and no ``$ref`` leads to it — so it
    goes. Reachability is transitive: a definition kept alive only by another
    orphan is an orphan too.
    """
    defs = schema.get("$defs")
    if not isinstance(defs, dict):
        return schema
    body = {key: value for key, value in schema.items() if key != "$defs"}
    reachable: set[str] = set()
    frontier = _referenced_defs(body)
    while frontier:
        name = frontier.pop()
        if name in reachable or name not in defs:
            continue
        reachable.add(name)
        frontier |= _referenced_defs(defs[name])
    kept = {name: body_ for name, body_ in defs.items() if name in reachable}
    return {**body, "$defs": kept} if kept else body


def build_input_schema(input_model: type[ToolInput]) -> dict[str, Any]:
    """Generate a cleaned MCP-ready JSON schema from a Pydantic model.

    ``$defs`` are kept as Pydantic emits them, not inlined: a shared submodel
    appears once and every use site is a ``$ref``, which measured 21% smaller
    on the consolidated surface. Every ref is internal to
    the one schema document, so any conformant client resolves it locally.

    Two further passes shrink the *advertised* shape only — the Pydantic model
    stays the validator and accepts exactly what it did before.
    ``_compact_type_keywords`` drops the type keywords a schema already implies
    (a nullable branch becomes a multi-type ``type`` array; the ``type`` beside
    a ``const`` goes), and ``_hoist_shared_fragments`` gives a sub-schema
    repeated across models one ``$defs`` entry instead of a copy per use site.
    Together they measured a tenth off the consolidated surface, most of it on
    ``analyze_results``, whose twenty-odd recipe models each restated the same
    seven shared fields. ``tests/test_consolidated_contracts.py`` pins the
    resulting size per tool.
    """
    schema = _strip_titles(input_model.wire_input_schema())
    return _hoist_shared_fragments(_compact_type_keywords(schema))


# What earns a description a place on the advertised wire: unit, convention,
# inversion, and protocol-contract markers — the sentence class measured as
# load-bearing (agents who lost it silently guessed field units wrong by
# orders of magnitude), against routing/derivable prose measured as inert.
# Substring semantics are deliberate and fail-open: a marker inside a longer
# token (the 'hz' in 'from_hz', the 'db' in 'level_db') KEEPS the text — an
# over-match ships a sentence it could have cut, never the reverse — and the
# surface-size pins ratchet what over-matching may cost.
# A live A/B over the full 11-request bench then licensed serving ONLY this
# class: the lean wire lost nothing and cost 15% less. Names, structure,
# enums, and defaults always stay; the full text remains on the registered
# definition and the models, so api.reference() and spice://guide carry the
# depth. The benchmark harness's schema-prune tooling mirrors this pattern —
# keep them in step if either changes.
WIRE_PROSE_KEEP = re.compile(
    r"(dB|degrees?|unwrapp?ed|percent|fraction|volts?|seconds?|hertz|Hz|µm|"
    r"V·µm|mV|sigma|√|sqrt|·|0 disables|echo it back|verbatim|clockwise|"
    # A pointer to the depth channels is protocol-contract prose: dropping it
    # would orphan the very branch stubs that rely on it (the dormant-recipe
    # stubs advertise nothing BUT their pointer).
    r"mirrors|api\.reference|spice://guide|"
    # Context cost is a unit statement too: an argument that makes every later
    # turn more expensive (an inline image) names its price in tokens. Word-
    # bounded, unlike the rest: the bare stem would also ship every sentence
    # that mentions a control_token.
    r"\btokens\b)",
    re.I,
)


def _keep_wire_prose(description: str | None) -> str | None:
    """The advertised copy of one description: itself, or nothing."""
    if description is not None and WIRE_PROSE_KEEP.search(description):
        return description
    return None


def strip_wire_prose(node: Any) -> Any:
    """Advertised-schema copy with every non-load-bearing description dropped.

    Unlike ``_strip_titles`` this walker needs no name-map awareness: it only
    ever touches a ``description`` key whose VALUE is a string, so a property
    that happens to be named ``description`` keeps its (dict) schema intact.
    """
    if isinstance(node, dict):
        out = {}
        for key, value in node.items():
            if key == "description" and isinstance(value, str):
                kept = _keep_wire_prose(value)
                if kept is not None:
                    out[key] = kept
                continue
            out[key] = strip_wire_prose(value)
        return out
    if isinstance(node, list):
        return [strip_wire_prose(value) for value in node]
    return node


# ---------------------------------------------------------------------------
# TypedDict → JSON Schema generator
# ---------------------------------------------------------------------------


_PRIMITIVE_MAP: dict[type, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
}


def _is_typeddict(tp: Any) -> bool:
    return isinstance(tp, type) and typing.is_typeddict(tp)


def _jsontype_from_union(args: tuple[Any, ...]) -> dict[str, Any]:
    """Handle ``X | None`` and ``X | Y | None`` unions.

    ``X | None`` becomes ``{"type": ["X", "null"]}`` when X is a single
    primitive — the common case for ``float | None`` fields. Mixed unions
    with complex members fall back to ``anyOf``.
    """
    non_none = [a for a in args if a is not type(None)]
    has_none = len(non_none) != len(args)
    if len(non_none) == 1:
        inner = schema_for_type(non_none[0])
        if has_none and "type" in inner and isinstance(inner["type"], str):
            type_val = inner["type"]
            return {**inner, "type": [type_val, "null"]}
        if has_none:
            # Complex inner (nested object/array) — use anyOf with null.
            return {"anyOf": [inner, {"type": "null"}]}
        return inner
    variants = [schema_for_type(a) for a in non_none]
    if has_none:
        variants.append({"type": "null"})
    return {"anyOf": variants}


def _is_union(tp: Any) -> bool:
    """True for both ``typing.Union[X, Y]`` and ``X | Y`` syntax."""
    if get_origin(tp) is Union:
        return True
    # Python 3.10+: `X | Y` has origin == types.UnionType (the class).
    return get_origin(tp) is _stdlib_types.UnionType


def schema_for_type(tp: Any) -> dict[str, Any]:
    """Return a JSON Schema fragment for a type annotation."""
    if tp is Any:
        return {}
    if tp is type(None):
        return {"type": "null"}
    if tp in _PRIMITIVE_MAP:
        return {"type": _PRIMITIVE_MAP[tp]}
    if _is_typeddict(tp):
        return schema_from_typeddict(tp)

    origin = get_origin(tp)
    args = get_args(tp)

    if origin is Literal:
        return {"enum": list(args)}
    if _is_union(tp):
        return _jsontype_from_union(args)
    if origin in (list, tuple):
        # A fixed, heterogeneous tuple (``tuple[int, str]``) has no single
        # ``items`` schema; rendering ``args[0]`` only would SILENTLY drop the
        # rest, so refuse it loudly (use a TypedDict or list[...], or add
        # prefixItems support) rather than emit a schema that lies about the
        # shape. A list, a ``tuple[X, ...]``, and a single-type/empty tuple all
        # map faithfully to an array of one item type.
        if origin is tuple and len(args) > 1 and args[1] is not Ellipsis:
            raise TypeError(
                f"Fixed heterogeneous tuple {tp!r} has no faithful single-`items` "
                "JSON Schema. Use a TypedDict (named fields) or list[...] for the "
                "output model, or add prefixItems support to schema_for_type."
            )
        item_type = args[0] if args else Any
        return {"type": "array", "items": schema_for_type(item_type)}
    if origin is dict:
        value_type = args[1] if len(args) == 2 else Any
        return {
            "type": "object",
            "additionalProperties": schema_for_type(value_type),
        }

    raise TypeError(
        f"Unsupported type annotation for schema generation: {tp!r}. "
        "Extend schema_for_type in tools/_schema.py if this construct is "
        "now used in the repo."
    )


@cache
def schema_from_typeddict(td: type) -> dict[str, Any]:
    """Generate a JSON Schema (``{"type": "object", ...}``) from a TypedDict.

    Every field is emitted under ``properties``. ``required`` reflects the
    two DISTINCT ways a field can be optional, both of which exist in the
    wire format: a ``NotRequired``/``total=False`` field may be ABSENT
    (omit-when-empty convention; e.g. ``GainAtPoint.phase_deg_unwrapped``),
    while an ``X | None`` field is always present but may be null. Marking
    an omitted key as required makes schema-validating MCP clients reject
    responses that follow the documented omit-when-empty behavior.
    """
    if not _is_typeddict(td):
        raise TypeError(f"Expected TypedDict, got {td!r}")

    hints = get_type_hints(td)
    # ``__required_keys__`` is computed at class-creation time and is UNRELIABLE
    # for ``NotRequired`` fields when the defining module uses ``from __future__
    # import annotations``: the wrapper is stringized, so the TypedDict metaclass
    # can't see it and wrongly counts the field as required (verified on 3.13).
    # ``get_type_hints(..., include_extras=True)`` EVALUATES the annotation,
    # recovering the ``NotRequired`` wrapper, so it detects optionality regardless
    # of stringization. (``Required`` in a ``total=False`` class is the symmetric
    # case but isn't used in this repo, so it's not special-cased.)
    extra_hints = get_type_hints(td, include_extras=True)
    structurally_required = getattr(td, "__required_keys__", frozenset(hints))
    properties: dict[str, Any] = {}
    required: list[str] = []
    for field_name, field_type in hints.items():
        properties[field_name] = schema_for_type(field_type)
        # A field is required unless the key may be absent entirely
        # (NotRequired / total=False) or its type admits None.
        admits_none = _is_union(field_type) and type(None) in get_args(field_type)
        not_required = get_origin(extra_hints.get(field_name)) is typing.NotRequired
        if field_name in structurally_required and not admits_none and not not_required:
            required.append(field_name)

    schema: dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        schema["required"] = required
    return schema
