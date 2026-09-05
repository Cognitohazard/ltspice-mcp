"""Readable argument catalogue for the six operations, rendered from the models.

The Python API carries no tool schemas, so a caller arriving cold has to
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

import functools
import textwrap
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

from ltspice_mcp.lib.model_fields import (
    accepted_annotation,
    default_label,
    describe_field,
    field_name,
    first_sentence,
    item_model,
    literal_values,
    model_of,
    non_null,
    strip_annotated,
    type_label,
    union_members,
)

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
    # Rendered after the example on both catalogue surfaces. For contract
    # facts a caller must know BEFORE submitting — the receipt's own
    # observations arrive only after.
    note: str = ""


def _model_union(annotation: Any) -> tuple[type[BaseModel], ...] | None:
    """The model branches of a union, when every branch is a model."""
    annotation = strip_annotated(annotation)
    members = union_members(annotation)
    if members is None:
        return None
    models = [model_of(member) for member in members if member is not type(None)]
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
        spelled = field_name(model, name, field)
        path = f"{prefix}{spelled}"
        annotation = field.annotation
        lines.extend(
            _lines_for_field(
                path,
                type_label(accepted_annotation(field)),
                default_label(field),
                describe_field(field),
                indent,
            )
        )

        deeper = {"seen": seen, "depth": depth + 1, "indent": indent}
        nested = model_of(non_null(annotation))
        if nested is not None:
            lines.extend(_render_model(nested, prefix=f"{path}.", **deeper))
            continue

        branches = _model_union(non_null(annotation))
        if branches is not None:
            lines.extend(_render_union(branches, path, seen=seen, depth=depth + 1, indent=indent))
            continue

        item = item_model(annotation)
        nested_item = model_of(item) if item is not None else None
        if nested_item is not None:
            lines.extend(_render_model(nested_item, prefix=f"{path}[].", **deeper))
            continue
        item_branches = _model_union(item) if item is not None else None
        if item_branches is not None:
            lines.extend(
                _render_union(
                    item_branches, f"{path}[]", seen=seen, depth=depth + 1, indent=indent
                )
            )
    return lines


def _tag_field(branches: tuple[type[BaseModel], ...]) -> str | None:
    """The shared literal field whose values tell the branches apart.

    Distinctness is the test, not "the first literal": several branches can
    share a literal (a mismatch rule and a tolerance rule both take a `scale`),
    and picking that one would label three different kinds identically.
    """
    common = set.intersection(*(set(model.model_fields) for model in branches))
    for name in sorted(common):
        per_branch = [literal_values(model, name) for model in branches]
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
        values = literal_values(model, tag_name) if tag_name else None
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
            spelled = field_name(model, name, field)
            default = default_label(field)
            bracket = "required" if default == "REQUIRED" else f"default {default}"
            label = type_label(accepted_annotation(field))
            lines.append(f"{_INDENT * (head + 2)}{spelled}: {label}  [{bracket}]")
            summary = first_sentence(describe_field(field))
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
            nested = model_of(non_null(field.annotation)) or model_of(item_model(field.annotation))
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
            inner = _model_union(non_null(field.annotation)) or _model_union(
                item_model(field.annotation)
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


def _render_arguments(model: type[BaseModel]) -> list[str]:
    """One operation's whole argument tree.

    An operation whose arguments ARE a union (``jobs``, keyed by ``action``)
    declares its branches as ``VARIANTS``; the model itself then carries only
    what every branch shares, and the branches are rendered after it — the
    fields the caller actually writes live there.
    """
    lines = _render_model(model)
    branches = tuple(getattr(model, "VARIANTS", ()))
    if branches:
        lines.extend(_render_union(branches, "the call", seen=frozenset(), depth=1, indent=0))
    return lines


# ---------------------------------------------------------------------------
# The catalogue
# ---------------------------------------------------------------------------


@functools.cache
def _operations() -> tuple[_Operation, ...]:
    # Imported here rather than at module import: this module is also what sets
    # the methods' docstrings, and the tool modules import back into the api
    # package.
    from ltspice_mcp.tools import (
        analyze,
        experiments,
        inspect_tools,
        jobs,
        schematic_edit,
        verify,
    )

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
            note=(
                "wait=True (the default) blocks until the complete receipt. "
                "wait=False returns the submission receipt immediately, but the "
                "job is owned by this process and is cancelled when it exits — "
                "Api.close(), the end of a 'with' block, or the interpreter "
                "exiting. Keep the process alive until the job finishes, or run "
                "work that must outlive it through a long-lived server."
            ),
        ),
        _Operation(
            name="jobs",
            summary="EXECUTE — status, wait, cancel, list circuits, or page one job's runs.",
            model=jobs.JobsInput,
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


@functools.cache
def op_reference(name: str) -> str:
    """One operation's resolved argument tree and a worked example."""
    operation = _find(name)
    lines = [f"{operation.name} — {operation.summary}", "", "arguments"]
    lines.extend(_render_arguments(operation.model))
    lines.extend(["", "example", *(f"{_INDENT}{line}" for line in operation.example.splitlines())])
    lines.extend(_note_lines(operation))
    return "\n".join(lines)


def _note_lines(operation: _Operation) -> list[str]:
    if not operation.note:
        return []
    return [
        "",
        "note",
        *textwrap.wrap(
            operation.note, width=_WRAP, initial_indent=_INDENT, subsequent_indent=_INDENT
        ),
    ]


def reference(op: str | None = None) -> str:
    """The catalogue: the six-operation index, or one operation's argument tree."""
    # A catalogue read pays the tool-model import; install the method
    # docstrings on the same event (lazy — see _methods.ensure_method_docs).
    from ltspice_mcp.api import _methods

    _methods.ensure_method_docs()
    if op is None:
        return index()
    if not isinstance(op, str):
        raise TypeError("op must be an operation name or None")
    return op_reference(op)


@functools.cache
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
            *_render_arguments(operation.model),
            "",
            "example",
            *(f"{_INDENT}{line}" for line in operation.example.splitlines()),
            *_note_lines(operation),
        ]
    )


def install_method_docs(namespace: type) -> None:
    """Give each operation's method the catalogue entry as its docstring."""
    for name in op_names():
        method = getattr(namespace, name, None)
        function = getattr(method, "__func__", method)
        if function is not None:
            function.__doc__ = method_doc(name)
