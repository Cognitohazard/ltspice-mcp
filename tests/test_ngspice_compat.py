"""How ngspice reads a compatibility mode string, against ngspice itself.

``lib/ngspice_compat.py`` models which flags ngspice sets for a mode. Every
reader of a mode asks it: the ``lib-section-ngspice`` lint, the failure
classifier, the hierarchy reader's refusal of a sectioned ``.lib``, and the
reader's ``m`` rule. Each live test here measures what ngspice does in a mode
and holds every one of those readers to it.
"""

from __future__ import annotations

import ast
import re
import shutil
from pathlib import Path

import pytest
from spicelib.simulators.ngspice_simulator import NGspiceSimulator

from ltspice_mcp.errors import NetlistError
from ltspice_mcp.lib.hierarchy import SemanticProfile, load_hierarchy
from ltspice_mcp.lib.lint_rules import lint_deck
from ltspice_mcp.lib.ngspice_compat import (
    FLAGS,
    compat_flags,
    drops_lib_sections,
    multiplies_m,
    reads_lib_as_includes,
)
from ltspice_mcp.lib.ngspice_mode import flags_for, run_batch
from ltspice_mcp.lib.runner_base import _is_ngspice_lib_section_failure
from tests._source_scan import modules
from tests.conftest import LIVENESS_S

needs_ngspice = pytest.mark.skipif(shutil.which("ngspice") is None, reason="ngspice not on PATH")

#: Modes whose letters ngspice reads in each way the model distinguishes:
#: case (``LTPS``, ``HSA``, ``Hsa``), ``spe`` clearing ``ps`` and ``lt``
#: (``spelt``), ``ps`` clearing ``hs`` (``hsps``) only after ``spe`` has
#: run (``spehsps``), ``mc`` turning the others off (``ltmc``, ``hsmc``,
#: ``psmc``) but ``s3`` (``s3mc``), ``s3`` beside others (``hsas3``,
#: ``lts3``).
MODES = (
    "",
    "hsa",
    "kiltpsa",
    "hs",
    "ps",
    "lt",
    "kia",
    "LTPS",
    "HSA",
    "Hsa",
    "spelt",
    "spehs",
    "hsps",
    "spehsps",
    "ltmc",
    "hsmc",
    "psmc",
    "s3",
    "s3mc",
    "hsas3",
    "lts3",
    "xs",
    "xsspe",
)

#: The flags ngspice-42 names for each mode in its log, measured with
#: ``flags_for``; ``mc`` makes it name none.
MEASURED_FLAGS = {
    "": set(),
    "hsa": {"hs", "a"},
    "kiltpsa": {"ps", "lt", "ki", "a"},
    "hs": {"hs"},
    "ps": {"ps"},
    "lt": {"lt"},
    "kia": {"ki", "a"},
    "LTPS": set(),
    "HSA": set(),
    "Hsa": {"a"},
    "spelt": {"spe"},
    "spehs": {"hs", "spe"},
    "hsps": {"ps"},
    "spehsps": {"hs", "spe"},
    "s3": {"s3"},
    "hsas3": {"hs", "a", "s3"},
    "lts3": {"lt", "s3"},
    "xsspe": {"xs", "spe"},
}

SECTIONED = "* sectioned\n.lib models.lib tt\nR1 1 0 {rv}\nV1 1 0 1\n.op\n.end\n"
MODELS = ".lib tt\n.param rv=2k\n.endl tt\n"
MULTIPLIED = "* m\n.subckt leaf p\nR1 p 0 1k m=3\n.ends\nXA a leaf m=4\nVa a 0 1\n.op\n.end\n"


@pytest.mark.parametrize(("mode", "flags"), list(MEASURED_FLAGS.items()))
def test_the_model_gives_the_flags_ngspice_42_named(mode: str, flags: set[str]):
    assert compat_flags(mode) == flags


@needs_ngspice
@pytest.mark.parametrize("mode", [mode for mode in MODES if "mc" not in mode])
def test_the_model_gives_the_flags_the_ngspice_on_path_names(mode: str):
    named = flags_for(("ngspice",), mode)
    assert named is not None
    assert compat_flags(mode) == set(named)


@needs_ngspice
@pytest.mark.parametrize("mode", MODES)
def test_every_reader_of_a_mode_reads_a_lib_section_as_ngspice_does(tmp_path: Path, mode: str):
    (tmp_path / "models.lib").write_text(MODELS, encoding="ascii")
    deck = tmp_path / "deck.cir"
    deck.write_text(SECTIONED, encoding="ascii")
    log = run_batch(
        ("ngspice",),
        mode,
        SECTIONED.encode("ascii"),
        cwd=tmp_path,
        clean_startup=True,
        timeout=LIVENESS_S,
    )
    # lt or ps read the line as two includes, so its section's name is a
    # missing include; s3 does not read .lib. Either way rv, which only the
    # section defines, is never set.
    as_includes = "Could not find include file tt" in log
    drops = as_includes or "Undefined parameter [rv]" in log
    assert drops or "error" not in log.casefold(), log

    assert drops_lib_sections(mode) is drops
    assert reads_lib_as_includes(mode) is as_includes
    findings = lint_deck(SECTIONED, deck, "ngspice", "NGspiceSimulator", ngbehavior=mode)
    assert ("lib-section-ngspice" in {row["rule_id"] for row in findings}) is drops
    # The classifier names the mode only for the failure it causes: the
    # section's name as a missing include, which s3 without lt or ps never
    # reports.
    classified = _is_ngspice_lib_section_failure(deck, NGspiceSimulator, mode)
    assert (classified is not None) is as_includes
    if drops:
        with pytest.raises(NetlistError, match="sectioned libraries"):
            load_hierarchy(str(deck), [tmp_path], SemanticProfile("ngspice", mode))
    else:
        load_hierarchy(str(deck), [tmp_path], SemanticProfile("ngspice", mode))


@needs_ngspice
@pytest.mark.parametrize("mode", MODES)
def test_every_reader_of_a_mode_multiplies_m_as_ngspice_does(tmp_path: Path, mode: str):
    log = run_batch(
        ("ngspice",),
        mode,
        MULTIPLIED.replace(".end\n", ".control\nop\nprint @r.xa.r1[m]\n.endc\n.end\n").encode(),
        cwd=tmp_path,
        clean_startup=True,
        timeout=LIVENESS_S,
    )
    printed = re.search(r"@r\.xa\.r1\[m\]\s*=\s*(\S+)", log)
    assert printed is not None, log
    built = float(printed[1])
    assert built in (4.0, 12.0), log

    assert multiplies_m(mode) is (built == 12.0)
    deck = tmp_path / "deck.cir"
    deck.write_text(MULTIPLIED, encoding="ascii")
    hierarchy = load_hierarchy(str(deck), [tmp_path], SemanticProfile("ngspice", mode))
    (row,) = [row for row in hierarchy.instances if row.instance == ("XA", "R1")]
    assert row.multiplier is not None
    assert row.multiplier.value == built


#: The flags a scan can tell in source: ``a`` is too common a string.
_SCANNED_FLAGS = frozenset(FLAGS) - {"a"}


def _flag(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant) and node.value in _SCANNED_FLAGS


def _mode_letter_readers(tree: ast.Module) -> list[int]:
    """Lines that test a flag's letters against a string (``"lt" in mode``)
    or loop over flag literals (``for flag in ("lt", "ps")``)."""
    lines = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Compare)
            and _flag(node.left)
            and any(isinstance(op, (ast.In, ast.NotIn)) for op in node.ops)
        ):
            lines.append(node.lineno)
        if (
            isinstance(node, (ast.comprehension, ast.For))
            and isinstance(node.iter, (ast.Tuple, ast.List, ast.Set))
            and node.iter.elts
            and all(_flag(item) for item in node.iter.elts)
        ):
            lines.append(node.iter.lineno)
    return lines


def test_only_ngspice_compat_reads_a_modes_letters():
    """A reader that tests a mode's letters itself reads ``LTPS`` or ``hsps``
    as ngspice does not; it asks ``ngspice_compat`` instead."""
    readers = {
        name: lines
        for name, tree in modules().items()
        if name != "lib/ngspice_compat.py" and (lines := _mode_letter_readers(tree))
    }
    assert readers == {}


def test_the_scan_finds_a_reader_of_a_modes_letters():
    for source in (
        'mode = ""\nif "lt" not in mode: pass\n',
        'mode = ""\nany(flag in mode for flag in ("lt", "ps"))\n',
    ):
        assert _mode_letter_readers(ast.parse(source))
