"""Variants: the server under test, as a checkout and a configuration overlay.

A variant is ``evals/variants/<name>.toml``::

    description = "the server's defaults"
    checkout = "."          # optional, relative to the repository root
    [config.tools]          # merged over the base configuration below
    listing = "full"

A variant changes the server only. The model, the client and the tasks are
the environment it is measured in, and are the same for every variant in a
comparison.
"""

from __future__ import annotations

import copy
import subprocess
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tomlkit

from ltspice_eval import REPO_ROOT, VARIANTS_DIR


class VariantError(ValueError):
    """A variant file that does not describe a variant."""


@dataclass(frozen=True)
class Variant:
    name: str
    description: str
    checkout: Path
    config: dict[str, Any]

    @property
    def python(self) -> Path:
        """The checkout's venv Python, which runs its server."""
        venv = self.checkout / ".venv"
        if sys.platform == "win32":
            return venv / "Scripts" / "python.exe"
        return venv / "bin" / "python"

    @property
    def runs_code(self) -> bool:
        return bool(self.config.get("tools", {}).get("run_code", False))


def load_variant(name: str, root: Path = VARIANTS_DIR) -> Variant:
    path = root / f"{name}.toml"
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise VariantError(f"{path}: {exc}") from exc
    if unknown := set(raw) - {"description", "checkout", "config"}:
        raise VariantError(f"{path}: unknown keys {sorted(unknown)}")
    checkout = (REPO_ROOT / raw.get("checkout", ".")).resolve()
    config = raw.get("config", {})
    if not isinstance(config, dict):
        raise VariantError(f"{path}: [config] is a table")
    variant = Variant(
        name=name, description=str(raw.get("description", "")), checkout=checkout, config=config
    )
    if not variant.python.is_file():
        raise VariantError(
            f"{path}: {variant.python} does not exist; run `uv sync` in the checkout"
        )
    return variant


def trial_config(variant: Variant, workspace: Path) -> dict[str, Any]:
    """The ``ltspice-mcp.toml`` one trial's server reads.

    The workspace is the sandbox, and the job store lives inside it (the
    server keeps it in its working directory). ``run_code`` is off unless the
    variant is about it, because a snippet has the server's own authority,
    which the sandbox does not confine.
    """
    base: dict[str, Any] = {
        "security": {"allowed_paths": [str(workspace)]},
        "tools": {"run_code": False},
    }
    return _merged(base, variant.config)


def write_trial_config(variant: Variant, workspace: Path) -> Path:
    path = workspace / "ltspice-mcp.toml"
    text = tomlkit.dumps(trial_config(variant, workspace))
    path.write_text(text, encoding="utf-8", newline="\n")
    return path


def revision(checkout: Path) -> dict[str, Any]:
    """The checkout's commit, and whether its tree differs from it."""

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(checkout), *args],
            capture_output=True,
            text=True,
            check=True,
            encoding="utf-8",
        ).stdout.strip()

    try:
        return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain"))}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def _merged(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merged(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out
