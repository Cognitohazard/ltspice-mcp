"""The one writer of the sentences that tell a caller how to read the rest.

A reply cut short, and a refusal of a continuation that cannot go on, name the
call that reads the rest. :func:`call_to_continue` writes that call the same
way for every tool: the MCP call with its arguments as JSON, and beside it the
same call through the Python API, where paging arguments need
``raw_page=True``. A value the writer does not know, such as the cursor a
caller's own previous page returned, is an :class:`Unfilled` placeholder.

Every such sentence is written here, so a caller meets one form in every tool;
``tests/test_reply_cuts.py`` refuses one written by hand elsewhere.
"""

from __future__ import annotations

import json
import keyword
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Unfilled:
    """A value the caller fills in, written ``<name>``."""

    name: str


def _mcp_value(value: Any) -> str:
    if isinstance(value, Unfilled):
        return f"<{value.name}>"
    if isinstance(value, Mapping):
        inner = ", ".join(f"{json.dumps(str(k))}: {_mcp_value(v)}" for k, v in value.items())
        return "{" + inner + "}"
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_mcp_value(v) for v in value) + "]"
    return json.dumps(value, ensure_ascii=False)


def _python_value(value: Any) -> str:
    if isinstance(value, Unfilled):
        return f"<{value.name}>"
    if isinstance(value, Mapping):
        inner = ", ".join(f"{json.dumps(str(k))}: {_python_value(v)}" for k, v in value.items())
        return "{" + inner + "}"
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_python_value(v) for v in value) + "]"
    if value is None or isinstance(value, bool):
        return repr(value)
    return json.dumps(value, ensure_ascii=False)


def mcp_call(tool: str, arguments: Mapping[str, Any]) -> str:
    """The MCP call: ``tool(name=value, ...)``, each value as JSON."""
    return f"{tool}(" + ", ".join(f"{k}={_mcp_value(v)}" for k, v in arguments.items()) + ")"


def python_call(tool: str, arguments: Mapping[str, Any]) -> str:
    """The same call through the Python API, which takes a paging argument
    only with ``raw_page=True``. An argument whose name is a Python keyword
    (``continue``) goes in a ``**`` mapping."""
    plain = [
        f"{k}={_python_value(v)}"
        for k, v in arguments.items()
        if k.isidentifier() and not keyword.iskeyword(k)
    ]
    spelled = {k: v for k, v in arguments.items() if not k.isidentifier() or keyword.iskeyword(k)}
    parts = [*plain, "raw_page=True"]
    if spelled:
        parts.append("**" + _python_value(spelled))
    return f"api.{tool}(" + ", ".join(parts) + ")"


def call_to_continue(tool: str, arguments: Mapping[str, Any]) -> str:
    """The call that reads the rest, written one way: the MCP call, then its
    Python form in parentheses."""
    return f"{mcp_call(tool, arguments)} (Python: {python_call(tool, arguments)})"
