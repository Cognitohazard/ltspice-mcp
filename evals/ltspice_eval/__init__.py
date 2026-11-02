"""An evaluation harness for the ltspice-mcp server.

It runs a model against the server on tasks with known answers, grades what
the agent leaves in its workspace, and compares server variants on the same
tasks, paired. ``docs/design/agent_eval.md`` is the design.
"""

from __future__ import annotations

from pathlib import Path

EVALS_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = EVALS_DIR.parent
TASKS_DIR = EVALS_DIR / "tasks"
VARIANTS_DIR = EVALS_DIR / "variants"
RUNS_DIR = EVALS_DIR / "runs"
LAUNCHER = EVALS_DIR / "launch_server.py"

#: Bumped when a row's meaning changes, so rows of different versions are not
#: compared by accident.
ROW_SCHEMA = 1
