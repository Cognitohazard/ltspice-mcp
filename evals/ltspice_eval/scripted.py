"""A model that plays a script, for testing the harness without API calls.

A script is a generator function. It receives a ``ScriptContext`` and yields
what the model does next: a ``Call`` (or a list of them, made in one turn),
or a ``Final`` text that ends the agent's turn. Each yield receives back the
results of those calls, as the model saw them::

    def script(ctx):
        [reply] = yield Call("mcp__spice__inspect", {"kind": "guide"})
        yield Call("write_file", {"path": "answer.json", "content": "{}"})

Returning ends the trial with the text ``Done.``. The scripted model goes
through the real loop and a real server process; only the API is replaced.
It records every request it is sent, so a test can check what a model would
have read.
"""

from __future__ import annotations

import importlib.util
import json
from collections.abc import Callable, Generator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ltspice_eval.models import ModelReply, Price, Usage

SCRIPTED_MODEL = "scripted"

#: A price for the scripted model, so the dollar ceiling can be tested: each
#: turn's fixed usage below costs one dollar.
SCRIPTED_PRICE = Price(input=1000.0, output=0.0, cache_read=0.0)
SCRIPTED_USAGE = {"input_tokens": 1000, "output_tokens": 0}


@dataclass(frozen=True)
class Call:
    name: str
    input: dict[str, Any]


@dataclass(frozen=True)
class Final:
    text: str = "Done."


@dataclass(frozen=True)
class Seen:
    """One tool result as the model received it."""

    text: str
    is_error: bool

    @property
    def json(self) -> Any:
        """The result parsed as JSON: the structured rendering's payload."""
        return json.loads(self.text)


@dataclass(frozen=True)
class ScriptContext:
    task_id: str
    workspace: Path
    #: The trial's private ``LTSPICE_MCP_HOME``, which names its server's processes.
    home: Path


Step = Call | list[Call] | Final
Script = Callable[[ScriptContext], Generator[Step, list[Seen], None]]


@dataclass
class ScriptedModel:
    script: Script
    context: ScriptContext
    #: The model id the replies claim; a test of the served-model check sets
    #: another.
    served_model: str = SCRIPTED_MODEL
    requests: list[dict[str, Any]] = field(default_factory=list)
    _steps: Generator[Step, list[Seen], None] | None = None
    _calls: int = 0

    async def create(self, request: dict[str, Any]) -> ModelReply:
        self.requests.append(json.loads(json.dumps(request)))
        seen = _last_results(request["messages"])
        try:
            if self._steps is None:
                self._steps = self.script(self.context)
                step = next(self._steps)
            else:
                step = self._steps.send(seen)
        except StopIteration:
            step = Final()
        return self._reply(step)

    def _reply(self, step: Step) -> ModelReply:
        usage = Usage.of(SCRIPTED_USAGE)
        if isinstance(step, Final):
            return ModelReply(
                content=[{"type": "text", "text": step.text}],
                stop_reason="end_turn",
                usage=usage,
                model=self.served_model,
            )
        calls = step if isinstance(step, list) else [step]
        blocks = []
        for call in calls:
            self._calls += 1
            blocks.append(
                {
                    "type": "tool_use",
                    "id": f"toolu_{self._calls:04d}",
                    "name": call.name,
                    "input": call.input,
                }
            )
        return ModelReply(
            content=blocks, stop_reason="tool_use", usage=usage, model=self.served_model
        )


def _last_results(messages: list[dict[str, Any]]) -> list[Seen]:
    if not messages or messages[-1]["role"] != "user":
        return []
    content = messages[-1]["content"]
    if isinstance(content, str):
        return []
    seen = []
    for block in content:
        if block.get("type") != "tool_result":
            continue
        parts = block.get("content", [])
        if isinstance(parts, str):
            text = parts
        else:
            text = "\n".join(part.get("text", "") for part in parts if part.get("type") == "text")
        seen.append(Seen(text=text, is_error=bool(block.get("is_error"))))
    return seen


def load_script(path: Path) -> Script:
    """The ``script`` function a task's ``oracle.py`` defines."""
    spec = importlib.util.spec_from_file_location(
        f"oracle_{path.parent.name.replace('-', '_')}", path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    script = getattr(module, "script", None)
    if script is None:
        raise ImportError(f"{path} defines no script()")
    return script


def null_script(ctx: ScriptContext) -> Generator[Step, list[Seen], None]:
    """Answers at once with a constant, without looking: must fail every task."""
    yield Call("write_file", {"path": "answer.json", "content": '{"answer": 0}'})
