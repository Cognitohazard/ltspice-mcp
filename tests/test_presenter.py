"""The presenter sends a complete answer whole when it fits, and otherwise
stores it, cuts the lists that may be cut, and says how to read the rest.

Design principle "A short reply, the data behind an id" (``docs/DESIGN.md``).
These are properties of ``lib/presenter.py`` over every declared output
schema, each filled with a made-up answer larger than a target:

- a list declared ``never`` is untouched, and so is everything else outside
  the lists the presenter cut;
- a cut list keeps whole entries from its start;
- reading every page in order rebuilds each cut list of the stored answer
  exactly;
- an answer that fits is, byte for byte, what ``reply_payload`` sends.
"""

from __future__ import annotations

import itertools
import json
import re
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import pytest

from ltspice_mcp.errors import InvalidCursorError
from ltspice_mcp.lib.presenter import (
    ListCut,
    Presented,
    cuttable_lists,
    open_page,
    present,
    present_page,
    resolve_pointer,
)
from ltspice_mcp.lib.reply_cuts import schema_lists
from ltspice_mcp.lib.response_budget import estimate_tokens
from ltspice_mcp.lib.store import Store
from ltspice_mcp.tools._base import reply_payload
from tests.conftest import declared_output_schemas

# ---------------------------------------------------------------------------
# A made-up answer for a schema
# ---------------------------------------------------------------------------

#: Entries a made-up list holds, by how many lists it sits inside.
_ENTRIES = (12, 3, 1)


class _Maker:
    """Fills a schema with values, every leaf distinct so that order shows."""

    def __init__(self, schema: Mapping[str, Any]) -> None:
        self.definitions = schema.get("$defs", {})
        self.counter = itertools.count()

    def make(self, node: Any, depth: int = 0, lists: int = 0) -> Any:
        if not isinstance(node, Mapping) or depth > 12:
            return f"leaf-{next(self.counter)}"
        ref = node.get("$ref")
        if isinstance(ref, str) and ref.startswith("#/$defs/"):
            return self.make(self.definitions[ref.rsplit("/", 1)[-1]], depth + 1, lists)
        if "const" in node:
            return node["const"]
        if node.get("enum"):
            return node["enum"][0]
        merged: dict[str, Any] = dict(node)
        for key in ("allOf", "oneOf", "anyOf"):
            branches = [b for b in merged.pop(key, ()) if isinstance(b, Mapping)]
            chosen = branches if key == "allOf" else branches[:1]
            for branch in chosen:
                if "$ref" in branch:
                    branch = self.definitions[branch["$ref"].rsplit("/", 1)[-1]]
                for name, value in branch.items():
                    if name == "properties":
                        merged["properties"] = {**merged.get("properties", {}), **value}
                    else:
                        merged.setdefault(name, value)
        declared = merged.get("type")
        if isinstance(declared, list):
            declared = next((t for t in declared if t != "null"), "null")
        if declared is None:
            declared = (
                "object"
                if "properties" in merged or "additionalProperties" in merged
                else "array"
                if "items" in merged
                else "string"
            )
        if declared == "object":
            made: dict[str, Any] = {}
            for name, child in (merged.get("properties") or {}).items():
                made[name] = self.make(child, depth + 1, lists)
            extra = merged.get("additionalProperties")
            if isinstance(extra, Mapping) and not made:
                for index in range(2):
                    made[f"key{index}-{next(self.counter)}"] = self.make(extra, depth + 1, lists)
            return made
        if declared == "array":
            count = _ENTRIES[min(lists, len(_ENTRIES) - 1)]
            return [self.make(merged.get("items"), depth + 1, lists + 1) for _ in range(count)]
        if declared == "integer":
            return next(self.counter) + 1
        if declared == "number":
            return next(self.counter) + 0.5
        if declared == "boolean":
            return True
        if declared == "null":
            return None
        return f"text-{next(self.counter):06d}-padding-to-weigh-something"


def _schemas() -> dict[str, dict[str, Any]]:
    return {
        f"{module.__name__.rsplit('.', 1)[-1]}.{handler.__name__}": schema
        for module, handler, schema in declared_output_schemas()
    }


_SCHEMAS = _schemas()


def _answer(schema: Mapping[str, Any]) -> dict[str, Any]:
    made = _Maker(schema).make(schema)
    assert isinstance(made, dict)
    return reply_payload(made)


def _store(tmp_path: Path) -> Store:
    store = Store(tmp_path)
    store.ensure_root()
    return store


def _presented(
    schema: Mapping[str, Any], answer: dict[str, Any], store: Store, target: int | None
) -> Presented:
    return present(
        answer,
        schema,
        target=target,
        tool="tool",
        snapshot={"path": "made-up", "sha256": "0" * 64},
        store=store,
        ttl_hours=1.0,
    )


def _without(reply: Mapping[str, Any], cut: Mapping[str, ListCut], stored: Any) -> Any:
    """``reply`` with each cut list put back whole from ``stored`` and the two
    presenter fields taken out."""
    rebuilt = json.loads(json.dumps(reply))
    rebuilt.pop("cut", None)
    rebuilt.pop("stored_answer", None)
    for pointer in cut:
        parent, key = _parent(rebuilt, pointer)
        parent[key] = resolve_pointer(stored, pointer)
    return rebuilt


def _parent(document: Any, pointer: str) -> tuple[Any, Any]:
    parts = [p.replace("~1", "/").replace("~0", "~") for p in pointer.split("/")[1:]]
    node = document
    for part in parts[:-1]:
        node = node[int(part)] if isinstance(node, list) else node[part]
    return node, parts[-1]


def _concrete(value: Any, path: str, pointer: str = "") -> Iterator[tuple[str, Any]]:
    """Each value of ``value`` at the schema path ``path`` (``a.b``, ``[]``,
    ``.*``, union branches), with its JSON Pointer."""
    segments = re.findall(r"\[\]|[^.\[\]]+", re.sub(r"<(?:anyOf|oneOf|allOf)\d+>", "", path))

    def walk(node: Any, rest: list[str], at: str) -> Iterator[tuple[str, Any]]:
        if not rest:
            yield at, node
            return
        head, tail = rest[0], rest[1:]
        if head == "[]" and isinstance(node, list):
            for index, entry in enumerate(node):
                yield from walk(entry, tail, f"{at}/{index}")
        elif head == "*" and isinstance(node, dict):
            for key, child in node.items():
                yield from walk(child, tail, f"{at}/{_escape(key)}")
        elif isinstance(node, dict) and head in node:
            yield from walk(node[head], tail, f"{at}/{_escape(head)}")

    yield from walk(value, segments, pointer)


def _escape(key: str) -> str:
    return str(key).replace("~", "~0").replace("/", "~1")


# ---------------------------------------------------------------------------
# The properties, over every declared schema
# ---------------------------------------------------------------------------


@pytest.fixture(params=sorted(_SCHEMAS), ids=sorted(_SCHEMAS))
def made_up(request: pytest.FixtureRequest) -> tuple[str, dict[str, Any], dict[str, Any]]:
    schema = _SCHEMAS[request.param]
    return request.param, schema, _answer(schema)


def test_a_made_up_answer_is_larger_than_its_target(made_up):
    _, _, answer = made_up
    assert estimate_tokens(answer, as_sent=False) > 60


def test_an_answer_that_fits_is_what_reply_payload_sends(made_up, tmp_path):
    _, schema, answer = made_up
    store = _store(tmp_path)

    shown = _presented(schema, answer, store, estimate_tokens(answer, as_sent=False))

    assert json.dumps(shown.reply) == json.dumps(answer)
    assert not shown.cut
    assert shown.stored is None
    assert not list(store.results_dir.glob("*.json"))


def test_no_target_sends_the_answer_whole(made_up, tmp_path):
    _, schema, answer = made_up

    shown = _presented(schema, answer, _store(tmp_path), None)

    assert json.dumps(shown.reply) == json.dumps(answer)


def test_only_whole_entries_of_cuttable_lists_are_cut(made_up, tmp_path):
    name, schema, answer = made_up
    target = estimate_tokens(answer, as_sent=False) // 3

    shown = _presented(schema, answer, _store(tmp_path), target)

    cuttable = {pointer: kind for pointer, kind in cuttable_lists(answer, schema)}
    for pointer, entry in shown.cut.items():
        assert cuttable.get(pointer) in ("items", "rows"), (name, pointer)
        whole = resolve_pointer(answer, pointer)
        sent = resolve_pointer(shown.reply, pointer, default=[])
        assert sent == whole[: entry.returned], (name, pointer)
        assert (entry.total, entry.offset) == (len(whole), 0)
        assert entry.returned >= 1
        assert entry.next is not None
    if not shown.cut:
        return
    assert shown.stored is not None
    assert shown.stored.answer == answer
    assert shown.reply["stored_answer"] == shown.stored.as_reply()
    assert shown.reply["cut"] == {
        pointer: entry.as_reply() for pointer, entry in shown.cut.items()
    }
    # It fits, or every list it could cut holds one entry.
    if estimate_tokens(shown.reply, as_sent=False) > target:
        assert all(len(resolve_pointer(shown.reply, p, default=[])) <= 1 for p in cuttable)


def test_nothing_outside_the_cut_lists_is_touched(made_up, tmp_path):
    name, schema, answer = made_up

    shown = _presented(
        schema, answer, _store(tmp_path), estimate_tokens(answer, as_sent=False) // 3
    )

    assert _without(shown.reply, shown.cut, answer) == answer, name


def test_a_list_sent_whole_is_never_cut(made_up, tmp_path):
    name, schema, answer = made_up

    shown = _presented(schema, answer, _store(tmp_path), 1)

    for item in schema_lists(schema):
        if item.kind in ("items", "rows"):
            continue
        for pointer, whole in _concrete(answer, item.path):
            assert pointer not in shown.cut, (name, pointer)
            assert resolve_pointer(shown.reply, pointer) == whole, (name, pointer)


def test_every_page_in_order_rebuilds_each_list(made_up, tmp_path):
    name, schema, answer = made_up
    store = _store(tmp_path)
    target = estimate_tokens(answer, as_sent=False) // 3

    shown = _presented(schema, answer, store, target)

    for pointer, entry in shown.cut.items():
        read = list(resolve_pointer(shown.reply, pointer, default=[]))
        following: str | None = entry.next
        pages = 0
        while following is not None:
            stored, at, offset = open_page(following, tool="tool", store=store, gone="gone")
            assert (at, offset) == (pointer, len(read))
            page = present_page(stored, at, offset, schema, target=target)
            assert page.reply["stored_answer"] == stored.as_reply()
            more = page.cut[pointer]
            assert more.offset == offset
            assert more.total == len(resolve_pointer(answer, pointer))
            read.extend(resolve_pointer(page.reply, pointer, default=[]))
            assert len(read) == offset + more.returned
            following = more.next
            pages += 1
            assert pages <= len(resolve_pointer(answer, pointer))
        assert read == resolve_pointer(answer, pointer), (name, pointer)


# ---------------------------------------------------------------------------
# The stored answer and its cursor
# ---------------------------------------------------------------------------

_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "rows": {"type": "array", "items": {"type": "string"}, "x-cut": "items"},
        "notes": {"type": "array", "items": {"type": "string"}, "x-cut": "never"},
        "keyed": {
            "type": "object",
            "additionalProperties": {
                "type": "array",
                "items": {"type": "integer"},
                "x-cut": "items",
            },
        },
    },
}


def _big() -> dict[str, Any]:
    return {
        "rows": [f"row {index:04d} " + "x" * 40 for index in range(200)],
        "notes": [f"note {index}" for index in range(30)],
        "keyed": {"a/b": list(range(300)), "c~d": [1]},
    }


def test_pointers_escape_map_keys(tmp_path):
    shown = _presented(_SCHEMA, _big(), _store(tmp_path), 400)

    assert set(shown.cut) <= {"/rows", "/keyed/a~1b"}
    assert "/keyed/c~0d" not in shown.cut  # one entry: nothing to cut
    assert shown.reply["notes"] == _big()["notes"]


def test_the_stored_answer_is_one_file_named_by_its_content_and_snapshot(tmp_path):
    store = _store(tmp_path)

    first = _presented(_SCHEMA, _big(), store, 400)
    again = _presented(_SCHEMA, _big(), store, 400)
    other = present(
        _big(),
        _SCHEMA,
        target=400,
        tool="tool",
        snapshot={"path": "made-up", "sha256": "1" * 64},
        store=store,
        ttl_hours=1.0,
    )

    assert first.stored is not None and again.stored is not None and other.stored is not None
    assert first.stored.id == again.stored.id != other.stored.id
    assert first.stored.path == store.stored_answer(first.stored.id)
    assert json.loads(first.stored.path.read_text(encoding="utf-8"))["answer"] == _big()
    assert sorted(p.name for p in store.results_dir.glob("*.json")) == sorted(
        {f"{first.stored.id}.json", f"{other.stored.id}.json"}
    )


def test_a_missing_stored_answer_is_refused_with_the_fix(tmp_path):
    store = _store(tmp_path)
    shown = _presented(_SCHEMA, _big(), store, 400)
    assert shown.stored is not None
    shown.stored.path.unlink()

    with pytest.raises(InvalidCursorError, match="run it again"):
        open_page(shown.cut["/rows"].next or "", tool="tool", store=store, gone="run it again")


def test_an_expired_stored_answer_is_refused_and_removed(tmp_path):
    store = _store(tmp_path)
    shown = present(
        _big(),
        _SCHEMA,
        target=400,
        tool="tool",
        snapshot={},
        store=store,
        ttl_hours=-1.0,
    )
    assert shown.stored is not None

    with pytest.raises(InvalidCursorError, match="gone"):
        open_page(shown.cut["/rows"].next or "", tool="tool", store=store, gone="run it again")
    assert not shown.stored.path.exists()


def test_a_cursor_for_another_tool_is_refused(tmp_path):
    store = _store(tmp_path)
    shown = _presented(_SCHEMA, _big(), store, 400)

    with pytest.raises(InvalidCursorError):
        open_page(shown.cut["/rows"].next or "", tool="other", store=store, gone="again")


@pytest.mark.parametrize("text", ["", "o:25", "not base64 !"])
def test_a_cursor_that_is_not_one_is_refused(tmp_path, text):
    with pytest.raises(InvalidCursorError):
        open_page(text, tool="tool", store=_store(tmp_path), gone="again")


def test_an_annotation_is_measured_with_the_reply(tmp_path):
    seen: list[tuple[str | None, int]] = []

    def annotate(
        reply: dict[str, Any], cut: Mapping[str, ListCut], stored: Any, page: Any
    ) -> None:
        seen.append((page, len(cut)))
        if cut:
            reply["hint"] = "read the rest " + "y" * 400

    shown = present(
        _big(),
        _SCHEMA,
        target=500,
        tool="tool",
        snapshot={},
        store=_store(tmp_path),
        ttl_hours=1.0,
        annotate=annotate,
    )

    assert shown.reply["hint"].startswith("read the rest")
    assert estimate_tokens(shown.reply, as_sent=False) <= 500
    assert seen and all(page is None for page, _ in seen)
