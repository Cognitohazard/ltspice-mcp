"""Every referral to a tool names a tool, branch and argument that exist.

A hint, note, description or guide line that says "call this with that" is a
recovery step. One that names an argument the tool does not take sends the
caller into a validation error at exactly the moment it was trying to
recover. The checks here read every string literal in the package and every
guide section, find the referrals in them, and hold each one to the live
input models:

- ``tool(arg=value, ...)``, including ``jobs(runs)`` and ``inspect(symbols)``
  for the action or kind, and JSON-style ``queries=[{"kind": ...}]``: the
  tool is registered, each argument is a field of the model the call selects
  (the jobs action's, the inspect query kind's, the recipe metric's...), and
  a value given for a field with a fixed set of values is one of them;
- ``tool (words ...)`` in prose: every snake_case name in the parentheses is
  a field or a fixed value somewhere in that tool's arguments;
- ``kind=``, ``action=`` and ``view=`` anywhere: the value is one some tool
  argument of that name takes;
- ``api.method(...)``: the method exists and takes those keyword arguments,
  and, where a caller reads it, every argument the signature requires.

A call is also validated whole, as the tool would validate it: against the
model of the action or query kind it names, with every placeholder (``...``,
``"<your sheet>"``, a format field) filled by a value of its field's type.
So a call that leaves out a required argument fails as one with a misspelt
argument does. The calls the code builds with ``mcp_call``, ``python_call``
and ``call_to_continue`` are held to the same rule. The files under
``fixtures/referrals`` hold referrals the tools refuse, and the checker must
report each one.
"""

from __future__ import annotations

import ast
import functools
import inspect
import itertools
import json
import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, get_args, get_origin

import pytest
from mcp.types import TextResourceContents
from pydantic import BaseModel, TypeAdapter, ValidationError

from ltspice_mcp.api import Api
from ltspice_mcp.lib.model_fields import (
    field_name,
    models_in,
    strip_annotated,
    union_members,
)
from ltspice_mcp.resources import handle_read_resource
from ltspice_mcp.state import SessionState
from ltspice_mcp.tools._base import registry
from ltspice_mcp.tools.inspect_tools import InspectInput, Query, handle_inspect
from tests._source_scan import PACKAGE, parsed
from tests.conftest import make_experiment_job, stage_recorded_fixture

GUIDE = PACKAGE / "assets" / "guide"

#: What a format field or f-string interpolation becomes: a value nothing checks.
WILD = "__wild__"

#: The positional shorthand a tool's referral may use for its branch name.
_SHORTHAND = {"jobs": "action", "inspect": "kind"}

#: Referral-shaped text that is not a referral to the tool of that name.
_NOT_REFERRALS: dict[str, str] = {
    "inspect(self, *, raw_page: bool = False, **arguments: Any)": (
        "a rendered Python signature of api.inspect"
    ),
}

#: Modules whose ``kind=``, ``action=`` or ``view=`` text names something of
#: their own, not a tool argument.
_OWN_VOCABULARY: dict[tuple[str, str], str] = {
    ("lib/spice_lex.py", "kind"): "the lexer's card kinds",
}


# ---------------------------------------------------------------------------
# What the package says
# ---------------------------------------------------------------------------

_FORMAT_FIELD = re.compile(r"(?<!\{)\{[A-Za-z_][A-Za-z0-9_.\[\]]*(?:![rsa])?(?::[^{}]*)?\}(?!\})")


def _template_text(value: str) -> str:
    """A ``str.format`` template as its output reads, fields left open."""
    text = _FORMAT_FIELD.sub(WILD, value)
    return text.replace("{{", "{").replace("}}", "}")


def _fstring_text(node: ast.JoinedStr) -> str:
    parts: list[str] = []
    for value in node.values:
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            parts.append(value.value)
        else:
            parts.append(WILD)
    return "".join(parts)


def _contributor_docstrings(tree: ast.Module, where: str) -> set[int]:
    """The docstrings of a module and its functions, which only a contributor
    reads; the Python API's own (``help(api.jobs)``) and a class's, which the
    reference table quotes, reach a caller."""
    if where.startswith("api/"):
        return set()
    found: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef)) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant):
                found.add(id(first.value))
    return found


def _file_texts(path: Path, where: str) -> list[tuple[str, str, bool]]:
    """Every string one file holds: ``(where, text, served)``, where ``served``
    says whether a caller can be shown it. A Markdown file is one text."""
    if path.suffix == ".md":
        return [(where, path.read_text(encoding="utf-8"), True)]
    tree = parsed(path)
    inside_fstrings = {
        id(child)
        for node in ast.walk(tree)
        if isinstance(node, ast.JoinedStr)
        for child in ast.walk(node)
        if child is not node
    }
    unserved = _contributor_docstrings(tree, where)
    found: list[tuple[str, str, bool]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.JoinedStr) and id(node) not in inside_fstrings:
            found.append((f"{where}:{node.lineno}", _fstring_text(node), True))
        elif (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and id(node) not in inside_fstrings
        ):
            served = id(node) not in unserved
            found.append((f"{where}:{node.lineno}", _template_text(node.value), served))
    return found


def _package_files() -> list[tuple[Path, str]]:
    files = [*sorted(PACKAGE.rglob("*.py")), *sorted(GUIDE.glob("*.md"))]
    return [(path, path.relative_to(PACKAGE).as_posix()) for path in files]


@functools.cache
def _texts() -> tuple[tuple[str, str, bool], ...]:
    """Every string the package holds, and every guide section."""
    return tuple(text for path, where in _package_files() for text in _file_texts(path, where))


#: The helpers that write a call from a tool name and its arguments.
_CALL_WRITERS = frozenset({"mcp_call", "python_call", "call_to_continue"})


def _built_calls(path: Path, where: str) -> list[tuple[str, _Call]]:
    """The calls a module writes with a :data:`_CALL_WRITERS` helper from a
    literal tool name and a literal mapping; a value the code computes is the
    caller's to fill."""
    if path.suffix != ".py":
        return []
    found: list[tuple[str, _Call]] = []
    for node in ast.walk(parsed(path)):
        if not isinstance(node, ast.Call) or len(node.args) != 2:
            continue
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
        tool, arguments = node.args
        if (
            name in _CALL_WRITERS
            and isinstance(tool, ast.Constant)
            and isinstance(tool.value, str)
            and isinstance(arguments, ast.Dict)
            and all(isinstance(k, ast.Constant) and isinstance(k.value, str) for k in arguments.keys)
        ):
            keywords: dict[str, ast.AST] = {
                k.value: v
                for k, v in zip(arguments.keys, arguments.values, strict=True)
                if isinstance(k, ast.Constant) and isinstance(k.value, str)
            }
            found.append((f"{where}:{node.lineno}", _Call(tool.value, [], keywords)))
    return found


# ---------------------------------------------------------------------------
# What the models take
# ---------------------------------------------------------------------------


def _tool_models() -> dict[str, type[BaseModel]]:
    import ltspice_mcp.tools  # noqa: F401  (registers every tool)

    return {
        registered.definition.name: registered.input_model
        for registered in registry._registered  # pyright: ignore[reportPrivateUsage]
        if registered.input_model is not None
    }


def _variants(model: type[BaseModel]) -> tuple[type[BaseModel], ...]:
    """A routing model's branches (the jobs actions), or the model itself."""
    return tuple(model.__dict__.get("VARIANTS", ())) or (model,)


def _literals(annotation: Any) -> set[str] | None:
    """The fixed string values an annotation takes, or None if it takes others."""
    annotation = strip_annotated(annotation)
    if get_origin(annotation) is Literal:
        values = get_args(annotation)
        return (
            {v for v in values if isinstance(v, str)}
            if all(isinstance(v, str) for v in values)
            else None
        )
    members = union_members(annotation)
    if members is None:
        return None
    found: set[str] = set()
    for member in members:
        if member is type(None):
            continue
        inner = _literals(member)
        if inner is None:
            return None
        found |= inner
    return found


def _fields(model: type[BaseModel]) -> dict[str, Any]:
    """A model's fields by every name a caller may write, with their annotations."""
    names: dict[str, Any] = {}
    for raw, field in model.model_fields.items():
        names[raw] = field.annotation
        names[field_name(model, raw, field)] = field.annotation
    return names


def _vocabulary(model: type[BaseModel]) -> set[str]:
    """Every field name and fixed value anywhere in a tool's arguments."""
    words: set[str] = set()
    seen: set[type[BaseModel]] = set()
    pending = list(_variants(model))
    while pending:
        current = pending.pop()
        if current in seen:
            continue
        seen.add(current)
        pending += list(_variants(current))
        for name, annotation in _fields(current).items():
            words.add(name)
            words |= _literals(annotation) or set()
            pending += models_in(annotation)
    return words


@functools.cache
def _named_values(name: str) -> frozenset[str]:
    """Every fixed value some tool argument called ``name`` takes, at any depth."""
    values: set[str] = set()
    seen: set[type[BaseModel]] = set()
    pending = [m for model in _tool_models().values() for m in _variants(model)]
    while pending:
        current = pending.pop()
        if current in seen:
            continue
        seen.add(current)
        pending += list(_variants(current))
        for field, annotation in _fields(current).items():
            if field == name:
                values |= _literals(annotation) or set()
            pending += models_in(annotation)
    return frozenset(values)


# ---------------------------------------------------------------------------
# Checking one call against a model
# ---------------------------------------------------------------------------


def _strings(node: ast.AST) -> list[str] | None:
    """The string values a node spells, ``'a'|'b'`` included; None if not strings."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return [node.value]
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        left, right = _strings(node.left), _strings(node.right)
        return None if left is None or right is None else left + right
    return None


def _checked(value: str) -> bool:
    return WILD not in value and "..." not in value and "<" not in value


def _select(candidates: list[type[BaseModel]], keys: dict[str, ast.AST]) -> list[type[BaseModel]]:
    """The union branches a dict names through its discriminator values."""
    chosen = list(candidates)
    for key, node in keys.items():
        values = _strings(node)
        if not values or not all(_checked(v) for v in values):
            continue
        narrowed = [
            model
            for model in chosen
            if key in _fields(model)
            and (literal := _literals(_fields(model)[key])) is not None
            and set(values) <= literal
        ]
        if narrowed:
            chosen = narrowed
    return chosen


def _check_keys(
    candidates: list[type[BaseModel]], keys: dict[str, ast.AST], where: str
) -> list[str]:
    problems: list[str] = []
    chosen = _select(candidates, keys)
    for key, node in keys.items():
        if not _checked(key):
            continue
        owners = [model for model in chosen if key in _fields(model)]
        if not owners:
            names = sorted({m.__name__ for m in chosen})
            problems.append(f"{where}: {key!r} is not an argument of {', '.join(names)}")
            continue
        problems += _check_value([_fields(m)[key] for m in owners], node, f"{where}.{key}")
    return problems


def _check_value(annotations: list[Any], node: ast.AST, where: str) -> list[str]:
    strings = _strings(node)
    if strings is not None:
        allowed: set[str] = set()
        for annotation in annotations:
            literal = _literals(annotation)
            if literal is None:
                return []
            allowed |= literal
        bad = [v for v in strings if _checked(v) and v not in allowed]
        return [f"{where}: {v!r} is not one of {sorted(allowed)}" for v in bad]
    if isinstance(node, ast.Dict):
        models = [m for annotation in annotations for m in models_in(annotation)]
        if not models:
            return []
        keys: dict[str, ast.AST] = {}
        for key, value in zip(node.keys, node.values, strict=True):
            if isinstance(key, ast.Constant) and isinstance(key.value, str):
                keys[key.value] = value
        return _check_keys(models, keys, where)
    if isinstance(node, (ast.List, ast.Tuple)):
        items = [
            get_args(strip_annotated(a))[0]
            for annotation in annotations
            for a in [annotation, *(union_members(strip_annotated(annotation)) or ())]
            if get_origin(strip_annotated(a)) is list
        ]
        if not items:
            return []
        problems: list[str] = []
        for element in node.elts:
            problems += _check_value(items, element, where)
        return problems
    return []


@dataclass(frozen=True)
class _Call:
    tool: str
    positional: list[ast.AST]
    keywords: dict[str, ast.AST]


def _parse_call(name: str, inner: str) -> _Call | None:
    # Elisions (``...`` or ``…``) and a Markdown table's escaped ``\|``.
    text = inner.replace("…", "...").replace("\\|", "|")
    text = re.sub(r",\s*\.\.\.\s*([}\]]|$)", r"\1", text)
    # An unquoted ``<your sheet>`` is a placeholder too.
    holed = re.sub(r"<[^<>]*>", "...", text)
    colons = r"(?<![\w\"'])(\w+)\s*:\s*(?=[\"'\[{\w.])"
    for attempt in (text, re.sub(colons, r"\1=", text), holed, re.sub(colons, r"\1=", holed)):
        try:
            node = ast.parse(f"_({attempt})", mode="eval").body
        except SyntaxError:
            continue
        assert isinstance(node, ast.Call)
        if any(k.arg is None for k in node.keywords) or any(
            isinstance(a, ast.Starred) for a in node.args
        ):
            return None
        # A bare ``...`` stands for arguments the text leaves out.
        positional: list[ast.AST] = [
            a for a in node.args if not (isinstance(a, ast.Constant) and a.value is Ellipsis)
        ]
        return _Call(name, positional, {k.arg: k.value for k in node.keywords if k.arg})
    return None


def _check_tool_call(call: _Call, where: str, served: bool = True) -> list[str]:
    """The call's names checked against the tool's models; a call a caller can
    be shown is also validated whole (:func:`_validation_problems`)."""
    models = _tool_models()
    model = models[call.tool]
    keywords: dict[str, ast.AST] = dict(call.keywords)
    if call.positional:
        branch = _SHORTHAND.get(call.tool)
        if branch is None or len(call.positional) != 1:
            return [f"{where}: {call.tool} takes no positional arguments"]
        shorthand = call.positional[0]
        if not isinstance(shorthand, ast.Name):
            return [f"{where}: {call.tool}'s shorthand is not a name"]
        keywords = {branch: ast.Constant(shorthand.id), **keywords}
    flat = call.tool == "inspect" and "queries" not in keywords and bool(keywords)
    if flat:
        # ``inspect(kind=...)`` names one query's fields; hold them to the query.
        values = [v for v in keywords.values() if isinstance(v, ast.expr)]
        query = ast.Dict([ast.Constant(k) for k in keywords], values)
        keywords = {"queries": ast.List([query])}
    problems = _check_keys(list(_variants(model)), keywords, where)
    if problems or call.positional or not served:
        # The positional shorthand names a branch; it is not a call to make.
        return problems
    if flat:
        return [f"{where}: inspect takes its lookups as queries=[{{'kind': ...}}]"]
    return _validation_problems(call.tool, keywords, where)


# ---------------------------------------------------------------------------
# Validating one call, its placeholders filled
# ---------------------------------------------------------------------------


class _Hole:
    """A value the text leaves to the caller: ``<your sheet>``, ``...``, ``x``."""


_HOLE = _Hole()


def _value(node: ast.AST) -> Any:
    """What a node spells, with a hole wherever the caller supplies the value.
    A call's alternatives (``action='status'|'wait'``) are validated one by one
    by :func:`_validation_problems`; one nested deeper, with its first."""
    if isinstance(node, ast.Constant):
        if node.value is Ellipsis:
            return _HOLE
        if isinstance(node.value, str) and not _checked(node.value):
            return _HOLE
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        operand = _value(node.operand)
        return _HOLE if operand is _HOLE else -operand
    if (strings := _strings(node)) is not None:
        return _value(ast.Constant(strings[0]))
    if isinstance(node, (ast.List, ast.Tuple)):
        return [_value(element) for element in node.elts]
    if isinstance(node, ast.Dict):
        return {
            key.value: _value(value)
            for key, value in zip(node.keys, node.values, strict=True)
            if isinstance(key, ast.Constant) and isinstance(key.value, str)
        }
    # A name (``x``, ``job_id``), a call, an attribute: the caller's value.
    return _HOLE


def _sample(annotation: Any) -> Any:
    """A value of ``annotation``'s type; a hole where it holds a model."""
    annotation = strip_annotated(annotation)
    if get_origin(annotation) is Literal:
        return get_args(annotation)[0]
    members = union_members(annotation)
    if members is not None:
        return _sample(next(m for m in members if m is not type(None)))
    if get_origin(annotation) in (list, tuple, set, frozenset) or annotation in (list, tuple):
        args = get_args(annotation)
        return [_sample(args[0])] if args else []
    if get_origin(annotation) is dict or annotation is dict:
        return {}
    for kind, sample in ((bool, True), (int, 1), (float, 1.0), (str, "circuit.asc")):
        if annotation is kind:
            return sample
    return _HOLE


def _branch(models: list[type[BaseModel]], value: dict[str, Any]) -> type[BaseModel] | None:
    """The one model of a union a dict names through its fixed values."""
    chosen = [
        model
        for model in models
        if all(
            (literal := _literals(_fields(model).get(key))) is None or item in literal
            for key, item in value.items()
            if key in _fields(model) and isinstance(item, str)
        )
    ]
    return chosen[0] if len(chosen) == 1 else None


def _fill(value: Any, annotation: Any, loc: tuple[Any, ...], holes: list[tuple[Any, ...]]) -> Any:
    """``value`` with each hole filled by its argument's type; ``holes`` gets
    the location of each, where any value of that type may stand."""
    if value is _HOLE:
        holes.append(loc)
        sample = _sample(annotation)
        return {} if sample is _HOLE else sample
    annotation = strip_annotated(annotation)
    if isinstance(value, dict):
        models = models_in(annotation)
        model = _branch(models, value) if models else None
        if models and model is None:
            # A placeholder names the branch: any branch's arguments may follow.
            holes.append(loc)
        fields = _fields(model) if model is not None else {}
        return {k: _fill(v, fields.get(k, Any), (*loc, k), holes) for k, v in value.items()}
    if isinstance(value, list):
        items = [
            get_args(a)[0]
            for a in [annotation, *(union_members(annotation) or ())]
            if get_origin(strip_annotated(a)) is list and get_args(a)
        ]
        item = items[0] if items else Any
        return [_fill(v, item, (*loc, i), holes) for i, v in enumerate(value)]
    return value


def _rejections(model: Any, value: Any, holes: list[tuple[Any, ...]]) -> list[str]:
    """What ``model`` (a model or a type adapter) says of ``value``, less what a
    hole's own value would settle: a hole stands for a right value of its type.
    A union adds its branch's tag to an error's location, so a hole covers an
    error whose location holds the hole's steps in order."""
    try:
        model.validate_python(value) if isinstance(model, TypeAdapter) else model.model_validate(
            value
        )
    except ValidationError as error:
        return [
            f"{'.'.join(map(str, problem['loc'])) or '(call)'}: {problem['msg']}"
            for problem in error.errors()
            if not any(_within(hole, problem["loc"]) for hole in holes)
        ]
    return []


def _within(hole: tuple[Any, ...], loc: tuple[Any, ...]) -> bool:
    steps = iter(loc)
    return all(any(step == part for step in steps) for part in hole)


@functools.cache
def _query_adapter() -> TypeAdapter[Any]:
    """The queries as ``inspect`` validates them, one by one."""
    return TypeAdapter(list[Query])


def _validation_problems(tool: str, keywords: dict[str, ast.AST], where: str) -> list[str]:
    """The call, its placeholders filled by type, as the tool validates it.

    ``inspect`` takes its queries unvalidated and validates each one apart
    (``SkipValidation``), so each query is held to the query union too."""
    options = [
        [(key, ast.Constant(value)) for value in strings]
        if (strings := _strings(node)) is not None
        else [(key, node)]
        for key, node in keywords.items()
    ]
    found: list[str] = []
    for chosen in itertools.product(*options):
        found += _validated(tool, dict(chosen))
    return [f"{where}: {tool} refuses it: {problem}" for problem in dict.fromkeys(found)]


def _validated(tool: str, keywords: dict[str, ast.AST]) -> list[str]:
    """What the tool says of one call, each argument a single value."""
    model = _tool_models()[tool]
    branches = _select(list(_variants(model)), keywords)
    if len(branches) != 1:
        return []  # a placeholder names the action: no one call to validate
    fields = _fields(branches[0])
    holes: list[tuple[Any, ...]] = []
    arguments = {
        key: _fill(_value(node), fields.get(key, Any), (key,), holes)
        for key, node in keywords.items()
    }
    found = _rejections(model, arguments, holes)
    if tool == "inspect" and isinstance(arguments.get("queries"), list):
        query_holes = [hole[1:] for hole in holes if hole[:1] == ("queries",)]
        found += _rejections(_query_adapter(), arguments["queries"], query_holes)
    return found


def _check_api_call(method: str, inner: str, where: str, served: bool = True) -> list[str]:
    target = getattr(Api, method, None)
    if target is None or method.startswith("_"):
        return [f"{where}: api.{method} does not exist"]
    call = _parse_call(method, inner)
    if call is None:
        return []
    models = _tool_models()
    if method in models:
        own = set(inspect.signature(getattr(target, "__func__", target)).parameters) - {"self"}
        rest = {k: v for k, v in call.keywords.items() if k not in own}
        if call.positional:
            return [f"{where}: api.{method} takes keyword arguments only"]
        return _check_tool_call(_Call(method, [], rest), where, served) if rest else []
    signature = inspect.signature(target)
    parameters = [p for p in signature.parameters.values() if p.name != "self"]
    names = {p.name for p in parameters}
    takes_more = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters)
    problems = [
        f"{where}: api.{method} takes no argument {k!r}"
        for k in call.keywords
        if k not in names and not takes_more
    ]
    positional = [
        p
        for p in parameters
        if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    if len(call.positional) > len(positional):
        problems.append(f"{where}: api.{method} takes {len(positional)} positional argument(s)")
    elif served and not problems:
        missing = [
            p.name
            for p in parameters[len(call.positional) :]
            if p.default is inspect.Parameter.empty
            and p.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
            and p.name not in call.keywords
        ]
        problems += [f"{where}: api.{method} needs {name!r}" for name in missing]
    return problems


# ---------------------------------------------------------------------------
# Finding referrals
# ---------------------------------------------------------------------------


def _closing(text: str, start: int) -> int | None:
    """The index of the parenthesis that closes the one before ``start``."""
    depth = 1
    quote: str | None = None
    index = start
    while index < len(text):
        char = text[index]
        if quote is not None:
            if char == quote:
                quote = None
        elif char in "\"'":
            quote = char
        elif char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
            if depth == 0:
                return index
        elif char == "\n" and depth == 1 and text[start:index].count("\n") > 3:
            return None
        index += 1
    return None


@dataclass(frozen=True)
class _Referral:
    where: str
    form: Literal["call", "prose", "api"]
    name: str
    inner: str
    served: bool


def _referrals(
    texts: tuple[tuple[str, str, bool], ...] | list[tuple[str, str, bool]] | None = None,
) -> Iterator[_Referral]:
    tools = sorted(_tool_models(), key=len, reverse=True)
    call = re.compile(rf"(?<![\w.])({'|'.join(tools)})(\s?)\(")
    api_call = re.compile(r"\bapi\.(\w+)\(")
    for where, text, served in _texts() if texts is None else texts:
        for match in call.finditer(text):
            end = _closing(text, match.end())
            if end is None:
                continue
            inner = text[match.end() : end]
            whole = text[match.start() : end + 1]
            if whole in _NOT_REFERRALS:
                continue
            form: Literal["call", "prose"] = "prose" if match.group(2) else "call"
            yield _Referral(where, form, match.group(1), inner, served)
        for match in api_call.finditer(text):
            end = _closing(text, match.end())
            if end is not None:
                yield _Referral(where, "api", match.group(1), text[match.end() : end], served)


_SNAKE = re.compile(r"\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+\b")


def _problems(referral: _Referral) -> list[str]:
    where = f"{referral.where} {referral.name}({referral.inner[:60]})"
    if referral.form == "api":
        return _check_api_call(referral.name, referral.inner, where, referral.served)
    call = _parse_call(referral.name, referral.inner)
    # Prose that spells arguments (``jobs (action:"status")``) is a call too.
    if call is not None and (
        referral.form == "call" or (call.keywords and not call.positional)
    ):
        return _check_tool_call(call, where, referral.served)
    # Prose, or a call written loosely: every snake_case name is the tool's.
    vocabulary = _vocabulary(_tool_models()[referral.name]) | set(_tool_models())
    return [
        f"{where}: {word!r} is no argument or value of {referral.name}"
        for word in _SNAKE.findall(referral.inner.replace(WILD, ""))
        if word not in vocabulary
    ]


# ---------------------------------------------------------------------------
# The tests
# ---------------------------------------------------------------------------


def test_the_scan_finds_the_referrals() -> None:
    """A scan that finds nothing passes everything: hold it to finding some."""
    found = list(_referrals())
    forms = {referral.form for referral in found}
    assert forms == {"call", "prose", "api"}
    assert len(found) > 100


def _all_problems(files: list[tuple[Path, str]]) -> list[str]:
    """Every problem in ``files``: their written referrals and the calls their
    code builds."""
    texts = [text for path, where in files for text in _file_texts(path, where)]
    problems = [problem for referral in _referrals(texts) for problem in _problems(referral)]
    for path, where in files:
        for at, call in _built_calls(path, where):
            if call.tool not in _tool_models():
                problems.append(f"{at}: {call.tool} is no tool")
            else:
                problems += _check_tool_call(call, f"{at} {call.tool}(...)")
    return problems


def test_every_tool_referral_names_arguments_the_tool_takes() -> None:
    problems = _all_problems(_package_files())
    assert not problems, "\n".join(problems)


def test_the_scan_finds_the_calls_the_code_builds() -> None:
    built = [call for path, where in _package_files() for call in _built_calls(path, where)]
    assert {call.tool for _, call in built} >= {"inspect", "jobs"}


_NAMED_VALUE = re.compile(r"""\b(kind|action|view)["']?\s*[=:]\s*["']([A-Za-z_][\w-]*)["']""")


@pytest.mark.parametrize("name", ["kind", "action", "view"])
def test_every_named_branch_exists(name: str) -> None:
    known = _named_values(name)
    assert known, f"no tool argument called {name} takes fixed values"
    problems = [
        f"{where}: {match.group(0)!r}"
        for where, text, _ in _texts()
        if (where.split(":")[0], name) not in _OWN_VOCABULARY
        for match in _NAMED_VALUE.finditer(text)
        if match.group(1) == name and match.group(2) not in known and _checked(match.group(2))
    ]
    assert not problems, f"{name} values no tool takes:\n" + "\n".join(problems)


class TestTheCheckerCatchesWhatItIsFor:
    """The checker is held to known-bad referrals, so a pass means something."""

    @pytest.mark.parametrize(
        ("name", "inner"),
        [
            ("analyze_results", "job_id plus run_index or case_id"),
            ("jobs", "action='stauts'"),
            ("jobs", "runs, run_feilds=['x']"),
            ("inspect", "kind='results', view='tabel'"),
            ("inspect", 'queries=[{"kind": "results", "veiw": "table"}]'),
            ("verify_circuit", "path='a.asc', render_png=True"),
        ],
    )
    def test_a_broken_referral_is_reported(self, name: str, inner: str) -> None:
        form: Literal["call", "prose"] = "prose" if " " in inner and "=" not in inner else "call"
        assert _problems(_Referral("test", form, name, inner, True))

    @pytest.mark.parametrize(
        ("name", "inner"),
        [
            ("inspect", 'queries=[{"kind": "results", "view": "signals", "job_id": "x"}]'),
            ("jobs", "runs, run_fields=['assignments.R']"),
            ("jobs", "action='status'|'wait', job_id=..."),
        ],
    )
    def test_a_sound_referral_passes(self, name: str, inner: str) -> None:
        assert not _problems(_Referral("test", "call", name, inner, True))

    def test_an_unknown_api_keyword_is_reported(self) -> None:
        assert _problems(_Referral("test", "api", "wait", "job_id, timeout_seconds=5", True))
        assert not _problems(_Referral("test", "api", "wait", "job_id, timeout_s=5", True))


FIXTURES = Path(__file__).parent / "fixtures" / "referrals"


class TestTheHardCases:
    """The referral forms the checker must validate whole, each against the
    model of the branch it names."""

    @staticmethod
    def _call(name: str, inner: str) -> list[str]:
        return _problems(_Referral("test", "call", name, inner, True))

    def test_a_flat_inspect_is_refused_and_its_queries_form_passes(self) -> None:
        assert self._call("inspect", "kind='capabilities'")
        assert self._call("inspect", "kind='reference', query='phase margin'")
        assert not self._call("inspect", "queries=[{'kind': 'capabilities'}]")
        assert not self._call("inspect", "queries=[{'kind': 'reference', 'query': 'phase margin'}]")

    def test_a_placeholder_is_filled_by_the_type_of_its_field(self) -> None:
        sound = 'queries=[{"kind": "net", "path": "<your sheet>", "at": [x, y]}]'
        assert not self._call("inspect", sound)
        # The placeholder stands for a path, so the call is checked whole: it
        # still lacks the point.
        problems = self._call("inspect", 'queries=[{"kind": "net", "path": "<your sheet>"}]')
        assert any("at" in problem for problem in problems), problems

    def test_a_jobs_call_is_validated_against_the_action_it_names(self) -> None:
        assert not self._call("jobs", "action='status', job_id=...")
        assert self._call("jobs", "action='cancel'")
        problems = self._call("jobs", "action='resume', job_id=...")
        assert any("resume_request_id" in problem for problem in problems), problems
        assert not self._call("jobs", "action='resume', job_id=..., resume_request_id=...")

    def test_an_api_call_must_fit_the_method_signature(self) -> None:
        assert _problems(_Referral("test", "api", "wait", "", True))
        assert not _problems(_Referral("test", "api", "wait", "job_id", True))
        # A contributor's docstring is not shown to a caller: only its names count.
        assert not _problems(_Referral("test", "api", "wait", "", False))

    def test_a_guide_code_span_and_code_block_are_checked(self) -> None:
        problems = _all_problems([(FIXTURES / "wrong_referrals.md", "wrong_referrals.md")])
        assert any('inspect(kind="symbol")' in problem for problem in problems), problems
        assert any("resume_request_id" in problem for problem in problems), problems

    def test_every_wrong_referral_in_a_module_is_reported(self) -> None:
        problems = _all_problems([(FIXTURES / "wrong_referrals.py", "wrong_referrals.py")])
        expected = {
            ":9 ": "flat inspect",
            ":10 ": "jobs without job_id",
            ":11 ": "net query without its point",
            ":12 ": "api.wait without job_id",
            ":13 ": "built flat inspect",
            ":14 ": "either of two jobs actions without job_id",
            ":15 ": "an unquoted placeholder, the call still lacking ops",
            ":16 ": "prose spelling arguments without job_id",
        }
        missed = [
            what
            for line, what in expected.items()
            if not any(problem.startswith(f"wrong_referrals.py{line}") for problem in problems)
        ]
        assert not missed, f"not reported: {missed}\n" + "\n".join(problems)


class TestTheResultsResourceNote:
    """Following the note a per-job results resource gives reads the run."""

    @pytest.mark.parametrize("view", ["signals", "measurements"])
    async def test_the_note_is_a_query_that_reads_the_case(
        self, state_no_sim: SessionState, work_dir: Path, view: str
    ) -> None:
        raw = stage_recorded_fixture(work_dir, "ltspice_tran_rc")
        job = make_experiment_job(state_no_sim, job_id="exp_note_0001", case_id="c0", raw=raw)
        contents = handle_read_resource(f"spice://results/{job.job_id}/{view}", state_no_sim)
        first = contents.contents[0]
        assert isinstance(first, TextResourceContents)
        note = json.loads(first.text)["note"]

        start = note.index("inspect(") + len("inspect(")
        end = _closing(note, start)
        assert end is not None
        call = _parse_call("inspect", note[start:end])
        assert call is not None and set(call.keywords) == {"queries"}
        (query,) = ast.literal_eval(call.keywords["queries"])
        query["case_id"] = "c0"

        args = InspectInput.model_validate({"queries": [query]})
        data = (await handle_inspect(args, state_no_sim)).structured_content
        assert data is not None
        (item,) = data["results"]
        assert item["ok"], item
