"""Running an input deck of the recorded LTspice behaviour through ngspice.

The grids whose question also has an ngspice answer run some of their decks
through ngspice live. ngspice refuses a few forms LTspice takes, so each deck
is run as a copy with those forms rewritten (``ngspice_copy``). Nothing of
ngspice is recorded.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

from tests.conftest import LIVENESS_S

_TRAN = re.compile(r"^\.tran\s+(.*)$", re.I | re.M)


def ngspice_copy(text: str) -> str:
    """A deck as ngspice runs it: a ``.tran`` with a print step of 0 or none,
    which LTspice takes and ngspice refuses, given one of a hundredth of its
    stop time."""

    def tran(match: re.Match[str]) -> str:
        words = match.group(1).split()
        stop = words[1] if len(words) > 1 else words[0]
        rest = words[2:] if len(words) > 1 else []
        seconds = float(stop.lower().replace("m", "e-3").replace("u", "e-6"))
        return " ".join([".tran", f"{seconds / 100:g}", stop, *rest])

    return _TRAN.sub(tran, text)


def run_ngspice(source: Path, directory: Path, *, text: bool) -> Path:
    """The raw ngspice writes for a copy of ``source`` run in ``directory``,
    binary or, with ``text``, as text; the test skips without ngspice."""
    executable = shutil.which("ngspice")
    if executable is None:
        pytest.skip("ngspice is not on PATH")
    directory.mkdir(parents=True, exist_ok=True)
    deck = directory / source.name
    deck.write_text(ngspice_copy(source.read_text(encoding="utf-8")), encoding="utf-8")
    raw = deck.with_suffix(".raw")
    # -n: no user startup file, which could set another mode or file type.
    command = [executable, "-b", "-n", "-r", raw.name, deck.name]
    if text:
        command[2:2] = ["-D", "filetype=ascii"]
    subprocess.run(command, cwd=directory, capture_output=True, timeout=LIVENESS_S, check=False)
    assert raw.is_file(), f"ngspice wrote no raw for {source.name}"
    return raw
