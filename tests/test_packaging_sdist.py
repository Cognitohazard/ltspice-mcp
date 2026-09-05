"""The published source distribution must build an installable wheel that
carries its runtime resources.

An explicit sdist include list can silently drop a file the package reads at
runtime while ``twine check`` stays green, so this builds the sdist, builds a
wheel FROM that sdist, installs it into a fresh interpreter environment, and
reads both packaged resources through the installed package.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

pytestmark = pytest.mark.skipif(shutil.which("uv") is None, reason="needs uv on PATH")


def _run(*cmd: str, cwd: Path) -> None:
    subprocess.run(cmd, cwd=cwd, check=True, capture_output=True, text=True, timeout=600)


def test_sdist_builds_a_wheel_that_installs_with_its_resources(tmp_path: Path) -> None:
    dist = tmp_path / "dist"
    # A bare `uv build` builds the sdist and then the wheel FROM that sdist.
    _run("uv", "build", "-o", str(dist), cwd=ROOT)
    wheel = next(dist.glob("*.whl"))
    venv = tmp_path / "venv"
    _run("uv", "venv", "-q", str(venv), cwd=tmp_path)
    python = venv / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    _run("uv", "pip", "install", "-q", "--python", str(python), str(wheel), cwd=tmp_path)
    probe = (
        "from importlib.resources import files\n"
        "import ltspice_mcp, ltspice_mcp.api\n"
        "assets = files('ltspice_mcp') / 'assets'\n"
        "guide = (assets / 'spice_guide.md').read_text(encoding='utf-8')\n"
        "assert '### .asc Schematics' in guide\n"
        "assert (assets / 'uplot' / 'uPlot.iife.min.js').is_file()\n"
        "assert (assets / 'ext-apps' / 'app-with-deps.js').is_file()\n"
        "print('ok')\n"
    )
    out = subprocess.run(
        [str(python), "-c", probe], cwd=tmp_path, check=True, capture_output=True, text=True
    )
    assert out.stdout.strip() == "ok"
