"""A suite's output: one row per trial, written as each trial ends.

``runs/<suite>/``::

    run.json                         the comparison's fixed parameters
    results.jsonl                    one row per graded trial
    errors.jsonl                     one row per trial that ended outside the agent's control
    traces/<variant>/<task>_rep<k>.json

A run resumes at the (task, variant, rep) key: a key with a row in
``results.jsonl`` is done; one that only errored runs again.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from ltspice_eval import RUNS_DIR

Key = tuple[str, str, int]


class RunMismatch(ValueError):
    """A resumed suite asked for different parameters than it was started with."""


class RunFiles:
    def __init__(self, suite: str, root: Path = RUNS_DIR) -> None:
        self.dir = root / suite
        self.run = self.dir / "run.json"
        self.results = self.dir / "results.jsonl"
        self.errors = self.dir / "errors.jsonl"
        self.traces = self.dir / "traces"

    def start(self, parameters: dict[str, Any]) -> None:
        """Record the parameters, or check that a resumed run repeats them."""
        self.dir.mkdir(parents=True, exist_ok=True)
        if self.run.is_file():
            recorded = json.loads(self.run.read_text(encoding="utf-8"))
            if recorded != parameters:
                changed = sorted(
                    key
                    for key in set(recorded) | set(parameters)
                    if recorded.get(key) != parameters.get(key)
                )
                raise RunMismatch(f"{self.dir.name} was started with other {', '.join(changed)}")
            return
        _write_json(self.run, parameters)

    def done(self) -> set[Key]:
        return {_key(row) for row in read_rows(self.results)}

    def append_result(self, row: dict[str, Any]) -> None:
        _append(self.results, row)

    def append_error(self, row: dict[str, Any]) -> None:
        _append(self.errors, row)

    def write_trace(self, variant: str, task: str, rep: int, trace: dict[str, Any]) -> Path:
        path = self.traces / variant / f"{task}_rep{rep}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        _write_json(path, trace)
        return path


def read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _key(row: dict[str, Any]) -> Key:
    return (row["task"], row["variant"], int(row["rep"]))


def _append(path: Path, row: dict[str, Any]) -> None:
    """One line, flushed and synced, so a killed run loses at most the
    trial in flight."""
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(line)
        handle.flush()
        os.fsync(handle.fileno())


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=1), encoding="utf-8", newline="\n"
    )
    os.replace(temporary, path)


def keys(rows: Iterable[dict[str, Any]]) -> set[Key]:
    return {_key(row) for row in rows}
