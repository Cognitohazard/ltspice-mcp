"""Computing a task's reference from its recorded run, with spicelib and numpy
alone: a reference must not come from the server code a variant changes.

Each task's ``grader/make_reference.py`` defines ``compute(workspace) ->
dict`` with these helpers; ``tests/test_eval_harness.py`` checks that the
committed ``reference.json`` is what it computes. To write one::

    cd evals
    uv run python -m ltspice_eval.references analyze-step-ac-f3db
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
from spicelib import RawRead

from ltspice_eval import TASKS_DIR

#: Significant digits a reference keeps, so a recomputation on another
#: platform's floating point compares equal.
DIGITS = 9


def open_raw(path: Path) -> RawRead:
    return RawRead(str(path), verbose=False)


def steps(raw: RawRead) -> list[int]:
    return list(raw.get_steps())


def axis(raw: RawRead, step: int = 0) -> np.ndarray:
    """The sweep axis of one step, as real numbers.

    Read as trace 0 rather than with ``get_axis``, which on a binary raw
    raises "This RAW file does not have an axis" until some trace has been
    read (``docs/spicelib_bugs.md``, Bug 35).
    """
    names = raw.get_trace_names()
    return np.abs(np.real(raw.get_trace(names[0]).get_wave(step)))


def wave(raw: RawRead, name: str, step: int = 0) -> np.ndarray:
    return np.asarray(raw.get_trace(name).get_wave(step))


def crossing(x: np.ndarray, y: np.ndarray, level: float, *, log_x: bool = False) -> float | None:
    """The first ``x`` where ``y`` crosses ``level``, linearly interpolated
    (in ``log10(x)`` when ``log_x``)."""
    above = y >= level
    for i in range(1, len(y)):
        if above[i] != above[i - 1]:
            x0, x1 = (np.log10(x[i - 1]), np.log10(x[i])) if log_x else (x[i - 1], x[i])
            fraction = (level - y[i - 1]) / (y[i] - y[i - 1])
            value = x0 + fraction * (x1 - x0)
            return float(10**value if log_x else value)
    return None


def rounded(value: Any) -> Any:
    """``value`` with every float kept to ``DIGITS`` significant digits."""
    if isinstance(value, float):
        if value == 0 or not math.isfinite(value):
            return value
        return float(f"{value:.{DIGITS}g}")
    if isinstance(value, dict):
        return {key: rounded(item) for key, item in value.items()}
    if isinstance(value, list):
        return [rounded(item) for item in value]
    return value


def compute(task_dir: Path) -> dict[str, Any]:
    """The reference ``task_dir/grader/make_reference.py`` computes."""
    path = task_dir / "grader" / "make_reference.py"
    spec = importlib.util.spec_from_file_location(
        f"reference_{task_dir.name.replace('-', '_')}", path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return rounded(module.compute(task_dir / "workspace"))


def main(argv: list[str]) -> int:
    for task_id in argv:
        task_dir = TASKS_DIR / task_id
        reference = compute(task_dir)
        target = task_dir / "grader" / "reference.json"
        target.write_text(json.dumps(reference, indent=1) + "\n", encoding="utf-8", newline="\n")
        print(f"{target}: {json.dumps(reference)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
