"""Tasks: a prompt, the files the agent starts with, and how the result is graded.

A task is a directory under ``evals/tasks/``::

    task.toml      id, category, requirements, prompt, grader
    workspace/     copied into each trial's workspace
    grader/        reference data; never copied into a workspace
    oracle.py      a scripted solution through the tools (see ``scripted``)

A task sets no limits of its own: every trial runs under the suite's ceilings.
In a prompt, wrapped lines are joined and a blank line separates paragraphs.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from ltspice_eval import TASKS_DIR

Category = Literal["analyze", "measure", "design", "debug", "schematic", "ltspice"]
CATEGORIES: frozenset[str] = frozenset(
    {"analyze", "measure", "design", "debug", "schematic", "ltspice"}
)
REQUIREMENTS: frozenset[str] = frozenset({"ngspice", "ltspice"})
GRADER_KINDS: frozenset[str] = frozenset({"answer"})


class TaskError(ValueError):
    """A task directory that does not describe a task."""


@dataclass(frozen=True)
class Tolerance:
    """How close a number must be: within ``abs`` or within ``rel`` of the
    reference, whichever is larger. Both zero means exactly equal."""

    rel: float = 0.0
    abs: float = 0.0

    @classmethod
    def parse(cls, raw: Any, where: str) -> Tolerance:
        if raw is None:
            return cls()
        if not isinstance(raw, dict) or set(raw) - {"rel", "abs"}:
            raise TaskError(f"{where}: a tolerance is a table of 'rel' and 'abs'")
        values = {key: float(value) for key, value in raw.items()}
        if any(value < 0 for value in values.values()):
            raise TaskError(f"{where}: a tolerance is not negative")
        return cls(**values)


@dataclass(frozen=True)
class Grader:
    kind: str
    reference: Path
    tolerance: Tolerance
    #: Per-key tolerances, by the top-level key of the answer they apply to.
    fields: dict[str, Tolerance] = field(default_factory=dict)


@dataclass(frozen=True)
class Task:
    id: str
    category: str
    requires: tuple[str, ...]
    prompt: str
    grader: Grader
    directory: Path
    #: Workspace files the agent may change; every other input is read-only,
    #: and a change to one is reported as a guardrail violation.
    editable: tuple[str, ...] = ()
    #: What makes the task hard, in the author's words.
    note: str = ""

    @property
    def workspace(self) -> Path:
        return self.directory / "workspace"

    @property
    def oracle(self) -> Path:
        return self.directory / "oracle.py"


def load_task(directory: Path) -> Task:
    path = directory / "task.toml"
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise TaskError(f"{path}: {exc}") from exc
    where = str(path)
    task_id = raw.get("id")
    if not isinstance(task_id, str) or task_id != directory.name:
        raise TaskError(f"{where}: id {task_id!r} is not the directory's name {directory.name!r}")
    category = raw.get("category")
    if category not in CATEGORIES:
        raise TaskError(f"{where}: category must be one of {sorted(CATEGORIES)}")
    requires = tuple(raw.get("requires", []))
    if unknown := set(requires) - REQUIREMENTS:
        raise TaskError(f"{where}: unknown requirements {sorted(unknown)}")
    prompt = raw.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise TaskError(f"{where}: a task needs a prompt")
    if "limits" in raw:
        raise TaskError(f"{where}: a task sets no limits; the suite's ceilings apply")
    grader_raw = raw.get("grader")
    if not isinstance(grader_raw, dict) or grader_raw.get("kind") not in GRADER_KINDS:
        raise TaskError(f"{where}: [grader] kind must be one of {sorted(GRADER_KINDS)}")
    reference = directory / str(grader_raw.get("reference", "grader/reference.json"))
    if directory / "grader" not in reference.parents:
        raise TaskError(f"{where}: the reference must live under grader/")
    fields = {
        name: Tolerance.parse(spec.get("tolerance"), f"{where} [grader.fields.{name}]")
        for name, spec in grader_raw.get("fields", {}).items()
    }
    grader = Grader(
        kind=grader_raw["kind"],
        reference=reference,
        tolerance=Tolerance.parse(grader_raw.get("tolerance"), f"{where} [grader]"),
        fields=fields,
    )
    workspace_raw = raw.get("workspace", {})
    return Task(
        id=task_id,
        category=category,
        requires=requires,
        prompt=_paragraphs(prompt),
        grader=grader,
        directory=directory,
        editable=tuple(workspace_raw.get("editable", [])),
        note=str(raw.get("note", "")),
    )


def _paragraphs(text: str) -> str:
    """A prompt as the model reads it: the TOML's wrapped lines joined, and
    paragraphs, separated by a blank line, kept."""
    blocks = [" ".join(block.split()) for block in text.strip().split("\n\n")]
    return "\n\n".join(block for block in blocks if block)


def load_tasks(root: Path = TASKS_DIR, ids: list[str] | None = None) -> list[Task]:
    """Every task under ``root``, by id, or the ones ``ids`` names."""
    tasks = [
        load_task(path)
        for path in sorted(root.iterdir())
        if path.is_dir() and (path / "task.toml").is_file()
    ]
    if ids:
        known = {task.id: task for task in tasks}
        if missing := [task_id for task_id in ids if task_id not in known]:
            raise TaskError(f"no such tasks: {missing}")
        tasks = [known[task_id] for task_id in ids]
    return tasks
