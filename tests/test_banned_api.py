"""Every source file allowed a banned name still uses one.

A name under ``[tool.ruff.lint.flake8-tidy-imports.banned-api]`` in
``pyproject.toml`` is one a replacement now answers for, kept only until its
callers move. Ruff refuses a new use, and the source files that still use one
are listed under ``per-file-ignores`` with the rule. Without this check a file
that has moved would keep its exception, so the list would only grow and never
show that a move was finished.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RULE = "TID251"


def _excepted_sources() -> set[str]:
    with (ROOT / "pyproject.toml").open("rb") as file:
        lint = tomllib.load(file)["tool"]["ruff"]["lint"]
    return {
        pattern
        for pattern, rules in lint.get("per-file-ignores", {}).items()
        if RULE in rules and pattern.startswith("src/")
    }


def _sources_using_a_banned_name(paths: set[str]) -> set[str]:
    """The files of ``paths`` in which Ruff finds a banned name, exceptions aside."""
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "ruff",
            "check",
            "--no-cache",
            "--exit-zero",
            "--select",
            RULE,
            "--output-format",
            "json",
            "--config",
            "lint.per-file-ignores = {}",
            *sorted(paths),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=120,
        check=True,
    )
    return {
        Path(finding["filename"]).resolve().relative_to(ROOT).as_posix()
        for finding in json.loads(result.stdout)
    }


def test_every_excepted_source_file_still_uses_a_banned_name():
    excepted = _excepted_sources()
    present = {path for path in excepted if (ROOT / path).is_file()}
    using = _sources_using_a_banned_name(present) if present else set()
    assert excepted == using, (
        "remove these from [tool.ruff.lint.per-file-ignores] in pyproject.toml, "
        f"since they no longer use a banned name: {sorted(excepted - using)}"
    )
