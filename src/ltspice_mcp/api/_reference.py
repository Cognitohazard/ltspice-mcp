"""Readable argument catalogue for the six operations, rendered from the models.

The in-process door carries no tool schemas, so a caller arriving cold has to
discover the argument vocabulary somehow. Measured, the ways they find are
expensive: firing a deliberately bogus value to read the valid set out of a
validation error, and importing private modules to reflect over their fields.
The alternative that was tried — publishing the generated JSON Schema — cost
just as much, because navigating ``$ref`` chains through a 96 KB document is
its own kind of search.

So the catalogue is a *resolved* tree: every field of every operation, nested
models flattened onto dotted paths, enum members and union branches written out
where they are used, and one worked example per op. No ``$ref``, no ``anyOf``,
nothing to follow. It is generated from the same pydantic models the wire
validates against, so it cannot drift from what a call will accept.

Two deliveries, one renderer: :func:`reference` for the whole catalogue on
demand, and the six methods' ``__doc__``, set at class-definition time, so
``help(api.edit_schematic)`` answers the reflex a Python caller already has.
"""

from __future__ import annotations

import textwrap
import types as pytypes
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Annotated, Any, Literal, Union, get_args, get_origin

from pydantic import BaseModel
from pydantic.fields import FieldInfo
from pydantic_core import PydanticUndefined

_INDENT = "  "
_WRAP = 88
#: Nesting depth at which flattening stops and says so. Nothing in the six ops
#: goes this deep; the cap is what makes a future self-referential model a
#: truncated tree instead of a hang.
_MAX_DEPTH = 5


@dataclass(frozen=True)
class _Operation:
    name: str
    summary: str
    model: type[BaseModel]
    example: str


# ---------------------------------------------------------------------------
# Type rendering
# ---------------------------------------------------------------------------


def _strip_annotated(annotation: Any) -> Any:
    while get_origin(annotation) is Annotated:
        annotation = get_args(annotation)[0]
    return annotation


def _union_members(annotation: Any) -> tuple[Any, ...] | None:
    origin = get_origin(annotation)
    if origin is Union or origin is pytypes.UnionType:
        return get_args(annotation)
    return None


def _model_of(annotation: Any) -> type[BaseModel] | None:
    annotation = _strip_annotated(annotation)
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return annotation
    return None


def _scalar_name(annotation: Any) -> str:
    simple = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        type(None): "null",
        Any: "any",
    }
    if annotation in simple:
        return simple[annotation]
    if isinstance(annotation, type):
        if issubclass(annotation, BaseModel):
            return "object"
        return annotation.__name__
    return str(annotation).replace("typing.", "")


def _type_label(annotation: Any) -> str:
    """A one-line type, with enum members and union branches written out."""
    annotation = _strip_annotated(annotation)

    if get_origin(annotation) is Literal:
        return "one of: " + " | ".join(repr(value) for value in get_args(annotation))

    members = _union_members(annotation)
    if members is not None:
        rendered = [_type_label(member) for member in members if member is not type(None)]
        label = " or ".join(dict.fromkeys(rendered))
        return f"{label} or null" if type(None) in members else label

    origin = get_origin(annotation)
    if origin in (list, Sequence):
        args = get_args(annotation)
        return f"list of {_type_label(args[0])}" if args else "list"
    if origin is dict:
        args = get_args(annotation)
        return f"map of {_type_label(args[1])}" if len(args) == 2 else "object"
    if origin is tuple:
        return "list"
    return _scalar_name(annotation)


def _accepted_annotation(field: FieldInfo) -> Any:
    """What the field accepts, which a coercing validator can widen.

    A ``BeforeValidator`` that takes ``True`` for a default policy declares that
    wider input for the JSON Schema; the catalogue reads the same declaration,
    so the two doors advertise one answer.
    """
    for meta in field.metadata:
        declared = getattr(meta, "json_schema_input_type", PydanticUndefined)
        if declared is not PydanticUndefined:
            return declared
    return field.annotation


def _default_label(field: FieldInfo) -> str:
    if field.default_factory is not None:
        try:
            produced = field.default_factory()  # pyright: ignore[reportCallIssue]
        except TypeError:  # pragma: no cover - validated-data factories take an argument
            return "computed"
        if isinstance(produced, BaseModel):
            return "all defaults"
        if isinstance(produced, (list, dict, set)) and not produced:
            return "empty"
        return repr(produced)
    if field.default is PydanticUndefined:
        return "REQUIRED"
    return repr(field.default)


def _field_name(model: type[BaseModel], name: str, field: FieldInfo) -> str:
    """The name a caller writes, which an alias may make different from the attribute."""
    alias = field.validation_alias
    if isinstance(alias, str):
        return alias
    choices = getattr(alias, "choices", None)
    if choices:
        first = choices[0]
        if isinstance(first, str):
            return first
    return field.alias if isinstance(field.alias, str) else name


# ---------------------------------------------------------------------------
# Model rendering
# ---------------------------------------------------------------------------


def _describe(field: FieldInfo) -> str:
    if field.description:
        return field.description
    # A leaf model documents itself in its docstring more often than in a
    # per-field description; use it rather than emit a bare line.
    nested = _model_of(_non_null(field.annotation))
    if nested is not None and nested.__doc__:
        return " ".join(nested.__doc__.split())
    return ""


#: Abbreviations that end in a period without ending a sentence. Without them a
#: description gets cut at "e.g." and the example — the useful half — is lost.
_ABBREVIATIONS = ("e.g", "i.e", "etc", "cf", "vs", "approx", "Fig")


def _first_sentence(description: str, *, limit: int = 220) -> str:
    """Enough of a description to act on, inside a union's per-branch listing."""
    text = " ".join(description.split())
    if not text:
        return ""
    sentence = text
    start = 0
    while True:
        index = text.find(". ", start)
        if index == -1:
            break
        head = text[:index]
        if any(head.endswith(abbreviation) for abbreviation in _ABBREVIATIONS):
            start = index + 2
            continue
        sentence = head + "."
        break
    return sentence if len(sentence) <= limit else sentence[: limit - 1].rstrip() + "…"


def _non_null(annotation: Any) -> Any:
    """The annotation with its ``| None`` branch dropped."""
    annotation = _strip_annotated(annotation)
    members = _union_members(annotation)
    if members is None:
        return annotation
    remaining = [member for member in members if member is not type(None)]
    if len(remaining) == 1:
        return _strip_annotated(remaining[0])
    return annotation


def _item_model(annotation: Any) -> Any:
    """What a ``list[...]`` field holds, or None when it is not a list."""
    annotation = _non_null(annotation)
    if get_origin(annotation) in (list, Sequence):
        args = get_args(annotation)
        return _strip_annotated(args[0]) if args else None
    return None


def _model_union(annotation: Any) -> tuple[type[BaseModel], ...] | None:
    """The model branches of a union, when every branch is a model."""
    annotation = _strip_annotated(annotation)
    members = _union_members(annotation)
    if members is None:
        return None
    models = [_model_of(member) for member in members if member is not type(None)]
    if len(models) < 2 or any(model is None for model in models):
        return None
    return tuple(model for model in models if model is not None)


def _lines_for_field(
    path: str,
    label: str,
    default: str,
    description: str,
    indent: int,
) -> list[str]:
    head = f"{_INDENT * indent}{path}: {label}"
    head += "  [required]" if default == "REQUIRED" else f"  [default {default}]"
    lines = [head]
    if description:
        lines.extend(
            textwrap.wrap(
                description,
                width=_WRAP,
                initial_indent=_INDENT * (indent + 2),
                subsequent_indent=_INDENT * (indent + 2),
            )
        )
    return lines


def _render_model(
    model: type[BaseModel],
    *,
    prefix: str = "",
    seen: frozenset[type[BaseModel]] = frozenset(),
    depth: int = 0,
    indent: int = 1,
) -> list[str]:
    if depth >= _MAX_DEPTH or model in seen:
        return [f"{_INDENT * indent}{prefix.rstrip('.')}: object (described above)"]

    seen = seen | {model}
    lines: list[str] = []
    for name, field in model.model_fields.items():
        spelled = _field_name(model, name, field)
        path = f"{prefix}{spelled}"
        annotation = field.annotation
        lines.extend(
            _lines_for_field(
                path,
                _type_label(_accepted_annotation(field)),
                _default_label(field),
                _describe(field),
                indent,
            )
        )

        deeper = {"seen": seen, "depth": depth + 1, "indent": indent}
        nested = _model_of(_non_null(annotation))
        if nested is not None:
            lines.extend(_render_model(nested, prefix=f"{path}.", **deeper))
            continue

        branches = _model_union(_non_null(annotation))
        if branches is not None:
            lines.extend(_render_union(branches, path, seen=seen, depth=depth + 1, indent=indent))
            continue

        item = _item_model(annotation)
        item_model = _model_of(item) if item is not None else None
        if item_model is not None:
            lines.extend(_render_model(item_model, prefix=f"{path}[].", **deeper))
            continue
        item_branches = _model_union(item) if item is not None else None
        if item_branches is not None:
            lines.extend(
                _render_union(
                    item_branches, f"{path}[]", seen=seen, depth=depth + 1, indent=indent
                )
            )
    return lines


def _literal_values(model: type[BaseModel], name: str) -> tuple[str, ...] | None:
    field = model.model_fields.get(name)
    if field is None:
        return None
    annotation = _strip_annotated(field.annotation)
    if get_origin(annotation) is not Literal:
        return None
    values = get_args(annotation)
    if not values or not all(isinstance(value, str) for value in values):
        return None
    return tuple(str(value) for value in values)


def _tag_field(branches: tuple[type[BaseModel], ...]) -> str | None:
    """The shared literal field whose values tell the branches apart.

    Distinctness is the test, not "the first literal": several branches can
    share a literal (a mismatch rule and a tolerance rule both take a `scale`),
    and picking that one would label three different kinds identically.
    """
    common = set.intersection(*(set(model.model_fields) for model in branches))
    for name in sorted(common):
        per_branch = [_literal_values(model, name) for model in branches]
        if any(values is None for values in per_branch):
            continue
        flat = [value for values in per_branch if values for value in values]
        if len(flat) == len(set(flat)):
            return name
    return None


def _render_union(
    branches: tuple[type[BaseModel], ...],
    path: str,
    *,
    seen: frozenset[type[BaseModel]],
    depth: int,
    indent: int = 1,
) -> list[str]:
    """Every branch by name, with the fields they share stated once."""
    shared: set[str] = set.intersection(*(set(model.model_fields) for model in branches))
    tag_name = _tag_field(branches)
    shared.discard(tag_name or "")
    head = indent + 1

    lines = [
        f"{_INDENT * head}{path} is one of {len(branches)} kinds"
        + (f", chosen by '{tag_name}':" if tag_name else ":")
    ]
    if shared:
        lines.extend(
            textwrap.wrap(
                "every kind also takes: " + ", ".join(sorted(shared)),
                width=_WRAP,
                initial_indent=_INDENT * (head + 1),
                subsequent_indent=_INDENT * (head + 2),
            )
        )
    # A leaf shape shared by many branches (a recipe's `spec`, a window) is
    # spelled out the first time it appears and referred back to after: twenty
    # identical four-line blocks are what makes a reference unreadable.
    expanded: set[type[BaseModel]] = set()
    # A field that several branches share (`signal`, `reduce`) carries its
    # description once, on its first appearance. Repeating it per branch is what
    # a schema dump does and is why nobody reads one.
    described: set[tuple[str, str]] = set()
    for model in branches:
        values = _literal_values(model, tag_name) if tag_name else None
        title = (
            f"{tag_name}={' | '.join(repr(value) for value in values)}"
            if values
            else model.__name__.lstrip("_")
        )
        lines.append(f"{_INDENT * (head + 1)}{title}")
        own = {
            name: field
            for name, field in model.model_fields.items()
            if name not in shared and name != tag_name
        }
        if not own:
            lines.append(f"{_INDENT * (head + 2)}(no further fields)")
            continue
        for name, field in own.items():
            spelled = _field_name(model, name, field)
            default = _default_label(field)
            bracket = "required" if default == "REQUIRED" else f"default {default}"
            label = _type_label(_accepted_annotation(field))
            lines.append(f"{_INDENT * (head + 2)}{spelled}: {label}  [{bracket}]")
            summary = _first_sentence(_describe(field))
            if summary and (spelled, summary) not in described:
                described.add((spelled, summary))
                lines.extend(
                    textwrap.wrap(
                        summary,
                        width=_WRAP,
                        initial_indent=_INDENT * (head + 4),
                        subsequent_indent=_INDENT * (head + 4),
                    )
                )
            if depth >= _MAX_DEPTH:
                continue
            nested = _model_of(_non_null(field.annotation)) or _model_of(
                _item_model(field.annotation)
            )
            if nested is not None and nested in expanded:
                lines[-1] = f"{_INDENT * (head + 2)}{spelled}: {label}  [{bracket}] (as above)"
                continue
            if nested is not None and nested not in seen:
                expanded.add(nested)
                lines.extend(
                    _render_model(
                        nested,
                        prefix=f"{spelled}.",
                        seen=seen | {model},
                        depth=depth + 2,
                        indent=head + 3,
                    )
                )
                continue
            inner = _model_union(_non_null(field.annotation)) or _model_union(
                _item_model(field.annotation)
            )
            if inner is not None:
                lines.extend(
                    _render_union(
                        inner,
                        f"{spelled}[]",
                        seen=seen | {model},
                        depth=depth + 2,
                        indent=head + 2,
                    )
                )
    return lines


# ---------------------------------------------------------------------------
# The catalogue
# ---------------------------------------------------------------------------


def _operations() -> tuple[_Operation, ...]:
    # Imported here rather than at module import: this module is also what sets
    # the methods' docstrings, and the tool modules import back into the api
    # package.
    from ltspice_mcp.tools import analyze, experiments, inspect_tools, schematic_edit, verify

    return (
        _Operation(
            name="run_experiments",
            summary="EXECUTE — run one or many decks over a variation grid; returns a receipt.",
            model=experiments.RunExperimentsInput,
            example=(
                "api.run_experiments(\n"
                '    circuits=[{"path": "opamp.asc"}],\n'
                '    variations=[{"kind": "assign", "assign": {"Cc": ["2p", "4p", "8p"]}}],\n'
                ")"
            ),
        ),
        _Operation(
            name="jobs",
            summary="EXECUTE — status, wait, cancel, list circuits, or page one job's runs.",
            model=experiments.JobsInput,
            example='api.jobs(action="status", job_id="exp_opamp_1785...")',
        ),
        _Operation(
            name="analyze_results",
            summary="UNDERSTAND — measure finished runs with typed recipes.",
            model=analyze.AnalyzeResultsInput,
            example=(
                "api.analyze_results(\n"
                '    sources=[{"label": "dut", "job_id": "exp_opamp_1785..."}],\n'
                '    recipes=[{"key": "pm", "metric": "stability", "signal": "V(out)"}],\n'
                ")"
            ),
        ),
        _Operation(
            name="inspect",
            summary="UNDERSTAND — batched read queries over sheets, symbols, nets and models.",
            model=inspect_tools.InspectInput,
            example='api.inspect(queries=[{"kind": "net", "path": "opamp.asc", "at": "M6.G"}])',
        ),
        _Operation(
            name="edit_schematic",
            summary="AUTHOR — one transactional batch of typed ops against an .asc sheet.",
            model=schematic_edit.EditSchematicInput,
            example=(
                "api.edit_schematic(\n"
                '    target="opamp.asc",\n'
                '    expected_sha256="<from inspect or a previous edit>",\n'
                '    ops=[{"op": "set_component_value", "reference": "I1", "value": "30u"}],\n'
                ")"
            ),
        ),
        _Operation(
            name="verify_circuit",
            summary="AUTHOR — check a circuit without changing it, and optionally draw it.",
            model=verify.VerifyCircuitInput,
            example='api.verify_circuit(path="opamp.asc", render=True)',
        ),
    )


def _find(name: str) -> _Operation:
    for operation in _operations():
        if operation.name == name:
            return operation
    known = ", ".join(operation.name for operation in _operations())
    raise ValueError(f"unknown operation {name!r}; the six operations are: {known}")


def op_names() -> tuple[str, ...]:
    """The six operation names, in the order the index lists them."""
    return tuple(operation.name for operation in _operations())


def index() -> str:
    """One line per operation, plus how to drill into one."""
    width = max(len(operation.name) for operation in _operations())
    lines = ["ltspice_mcp.api — six operations", ""]
    lines.extend(
        f"{_INDENT}{operation.name.ljust(width)}  {operation.summary}"
        for operation in _operations()
    )
    lines.extend(
        [
            "",
            "api.reference('<name>') prints one operation's full argument tree with an",
            "example. Every argument is a plain dict or scalar; the typed models are",
            "importable from ltspice_mcp.api.types if you want them.",
            "",
            "Also on the object: wait(job_id), load_raw(...) for numpy arrays, and",
            "measurements(job_id=...) for parsed .meas data.",
        ]
    )
    return "\n".join(lines)


def op_reference(name: str) -> str:
    """One operation's resolved argument tree and a worked example."""
    operation = _find(name)
    lines = [f"{operation.name} — {operation.summary}", "", "arguments"]
    lines.extend(_render_model(operation.model))
    lines.extend(["", "example", *(f"{_INDENT}{line}" for line in operation.example.splitlines())])
    return "\n".join(lines)


def reference(op: str | None = None) -> str:
    """The catalogue: the six-operation index, or one operation's argument tree."""
    if op is None:
        return index()
    if not isinstance(op, str):
        raise TypeError("op must be an operation name or None")
    return op_reference(op)


def method_doc(name: str) -> str:
    """The ``__doc__`` one operation's method carries: summary, tree, example."""
    operation = _find(name)
    return "\n".join(
        [
            operation.summary,
            "",
            "Arguments are passed as keywords; nested ones are plain dicts. The tree",
            "below is generated from the models this call validates against, and is",
            f"the same text api.reference({operation.name!r}) prints.",
            "",
            "arguments",
            *_render_model(operation.model),
            "",
            "example",
            *(f"{_INDENT}{line}" for line in operation.example.splitlines()),
        ]
    )


def install_method_docs(namespace: type) -> None:
    """Give each operation's method the catalogue entry as its docstring."""
    for name in op_names():
        method = getattr(namespace, name, None)
        function = getattr(method, "__func__", method)
        if function is not None:
            function.__doc__ = method_doc(name)
