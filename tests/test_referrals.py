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
- ``api.method(...)``: the method exists and takes those keyword arguments.

A value spelled through a format field (``kind="{name}"``) is not checked.
"""

from __future__ import annotations

import ast
import functools
import inspect
import json
import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, get_args, get_origin

import pytest
from mcp.types import TextResourceContents
from pydantic import BaseModel

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
from ltspice_mcp.tools.inspect_tools import InspectInput, handle_inspect
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


@functools.cache
def _texts() -> tuple[tuple[str, str], ...]:
    """Every string the package can show a caller: ``(where, text)``."""
    found: list[tuple[str, str]] = []
    for path in sorted(PACKAGE.rglob("*.py")):
        tree = parsed(path)
        inside_fstrings = {
            id(child)
            for node in ast.walk(tree)
            if isinstance(node, ast.JoinedStr)
            for child in ast.walk(node)
            if child is not node
        }
        where = path.relative_to(PACKAGE).as_posix()
        for node in ast.walk(tree):
            if isinstance(node, ast.JoinedStr) and id(node) not in inside_fstrings:
                found.append((f"{where}:{node.lineno}", _fstring_text(node)))
            elif (
                isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and id(node) not in inside_fstrings
            ):
                found.append((f"{where}:{node.lineno}", _template_text(node.value)))
    for path in sorted(GUIDE.glob("*.md")):
        found.append((path.relative_to(PACKAGE).as_posix(), path.read_text(encoding="utf-8")))
    return tuple(found)


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
    for attempt in (text, re.sub(r"(?<![\w\"'])(\w+)\s*:\s*(?=[\"'\[{\w])", r"\1=", text)):
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


def _check_tool_call(call: _Call, where: str) -> list[str]:
    models = _tool_models()
    model = models[call.tool]
    keywords = dict(call.keywords)
    if call.positional:
        branch = _SHORTHAND.get(call.tool)
        if branch is None or len(call.positional) != 1:
            return [f"{where}: {call.tool} takes no positional arguments"]
        shorthand = call.positional[0]
        if not isinstance(shorthand, ast.Name):
            return [f"{where}: {call.tool}'s shorthand is not a name"]
        keywords = {branch: ast.Constant(shorthand.id), **keywords}
    if call.tool == "inspect" and "queries" not in keywords:
        # inspect(kind=..., ...) names one query, with the call's own fields.
        queries_annotation = _fields(model)["queries"]
        query_models = models_in(get_args(strip_annotated(queries_annotation))[0])
        own = {k: v for k, v in keywords.items() if k in _fields(model)}
        rest = {k: v for k, v in keywords.items() if k not in own}
        return _check_keys([model], own, where) + _check_keys(query_models, rest, where)
    return _check_keys(list(_variants(model)), keywords, where)


def _check_api_call(method: str, inner: str, where: str) -> list[str]:
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
        return _check_tool_call(_Call(method, [], rest), where) if rest else []
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


def _referrals() -> Iterator[_Referral]:
    tools = sorted(_tool_models(), key=len, reverse=True)
    call = re.compile(rf"(?<![\w.])({'|'.join(tools)})(\s?)\(")
    api_call = re.compile(r"\bapi\.(\w+)\(")
    for where, text in _texts():
        for match in call.finditer(text):
            end = _closing(text, match.end())
            if end is None:
                continue
            inner = text[match.end() : end]
            whole = text[match.start() : end + 1]
            if whole in _NOT_REFERRALS:
                continue
            form: Literal["call", "prose"] = "prose" if match.group(2) else "call"
            yield _Referral(where, form, match.group(1), inner)
        for match in api_call.finditer(text):
            end = _closing(text, match.end())
            if end is not None:
                yield _Referral(where, "api", match.group(1), text[match.end() : end])


_SNAKE = re.compile(r"\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+\b")


def _problems(referral: _Referral) -> list[str]:
    where = f"{referral.where} {referral.name}({referral.inner[:60]})"
    if referral.form == "api":
        return _check_api_call(referral.name, referral.inner, where)
    if referral.form == "call":
        call = _parse_call(referral.name, referral.inner)
        if call is not None:
            return _check_tool_call(call, where)
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


def test_every_tool_referral_names_arguments_the_tool_takes() -> None:
    problems = [problem for referral in _referrals() for problem in _problems(referral)]
    assert not problems, "\n".join(problems)


_NAMED_VALUE = re.compile(r"""\b(kind|action|view)["']?\s*[=:]\s*["']([A-Za-z_][\w-]*)["']""")


@pytest.mark.parametrize("name", ["kind", "action", "view"])
def test_every_named_branch_exists(name: str) -> None:
    known = _named_values(name)
    assert known, f"no tool argument called {name} takes fixed values"
    problems = [
        f"{where}: {match.group(0)!r}"
        for where, text in _texts()
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
        assert _problems(_Referral("test", form, name, inner))

    @pytest.mark.parametrize(
        ("name", "inner"),
        [
            ("inspect", 'queries=[{"kind": "results", "view": "signals", "job_id": "x"}]'),
            ("jobs", "runs, run_fields=['assignments.R']"),
            ("jobs", "action='status'|'wait'"),
        ],
    )
    def test_a_sound_referral_passes(self, name: str, inner: str) -> None:
        assert not _problems(_Referral("test", "call", name, inner))

    def test_an_unknown_api_keyword_is_reported(self) -> None:
        assert _problems(_Referral("test", "api", "wait", "job_id, timeout_seconds=5"))
        assert not _problems(_Referral("test", "api", "wait", "job_id, timeout_s=5"))


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
