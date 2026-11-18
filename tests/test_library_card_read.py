"""What a ``.lib`` card brings in, for the simulator and the ngspice mode
that read the deck (``deck_staging.library_read``).

Every reader of a deck's library cards asks that one function: hierarchy
discovery (``inspect``), the linter, the analyses a run expects
(``runner_base.deck_requirements``), the comparison of ``verify_circuit``
(``netlist_graph``) and deck staging. Each test here sends its deck through
every one of them.

Measured on ngspice-42 (``test_ngspice_reads_a_lib_card_as_the_table_says``):

=========================================  ===============  =====================
Card                                       native and hsa   kiltpsa
=========================================  ===============  =====================
``.lib file tt``                           section tt only  two includes (refused)
``.lib file``, file with no sections       fails            the whole file
``.lib file``, file with sections          fails            nothing, the run goes on
=========================================  ===============  =====================

LTspice reads the whole rest of a ``.lib`` line as one path, quoted or not
(recorded: ``deck/lib_path_space_unquoted``, ``deck/lib_path_space_quoted``,
``deck/lib_section``).
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

from ltspice_mcp.errors import NetlistError
from ltspice_mcp.lib.deck_staging import (
    LibraryRead,
    LibraryReader,
    card_sections,
    library_read,
    resolve_existing,
    resolve_reference,
    scan_include_references,
    stage_deck,
)
from ltspice_mcp.lib.encoding import read_spice_text
from ltspice_mcp.lib.hierarchy import SemanticProfile, load_hierarchy
from ltspice_mcp.lib.lint_rules import lint_deck
from ltspice_mcp.lib.netlist_graph import compare_graphs, parse_netlist_graph
from ltspice_mcp.lib.runner_base import deck_requirements
from ltspice_mcp.lib.simulator_build import Producer
from ltspice_mcp.lib.spice_lex import lex_deck
from tests.ltspice_recorder import INPUTS

needs_ngspice = pytest.mark.skipif(shutil.which("ngspice") is None, reason="ngspice not on PATH")

MODES = ("", "hsa", "kiltpsa")

#: A library whose cards outside every section and inside section tt set
#: different values, so which ones were read shows in the answer.
SECTIONED = (
    "* a sectioned library\n"
    ".param rv=1\n"
    ".model nout NMOS(level=1)\n"
    ".save v(a)\n"
    ".tran 1n 10n\n"
    ".subckt amp p\nR1 p 0 1k\n.ends amp\n"
    ".lib tt\n"
    ".param rv=2\n"
    ".model ntt NMOS(level=1)\n"
    ".subckt amp p\nR1 p 0 2k\n.ends amp\n"
    ".endl tt\n"
)
PLAIN = "* a library with no sections\n.param rv=1\n.model nplain NMOS(level=1)\n"


def _write(folder: Path, name: str, text: str) -> Path:
    path = folder / name
    path.write_text(text, encoding="utf-8")
    return path


def _deck(folder: Path, lib_line: str, body: str = "V1 a 0 1\nR1 a 0 1k\n") -> Path:
    _write(folder, "sect.lib", SECTIONED)
    _write(folder, "plain.lib", PLAIN)
    return _write(folder, "d.cir", f"* deck\n{lib_line}\n{body}.op\n.end\n")


def _lint(deck: Path, mode: str, family: str = "ngspice") -> dict[str, list[dict]]:
    findings = lint_deck(
        deck.read_text(encoding="utf-8"),
        deck,
        family,
        family,
        ngbehavior=mode if family == "ngspice" else None,
    )
    by_rule: dict[str, list[dict]] = {}
    for finding in findings:
        if finding["severity"] == "error":
            by_rule.setdefault(finding["rule_id"], []).append(finding)
    return by_rule


# --------------------------------------------------------------------------
# A bare .lib on a file that declares sections, in kiltpsa: nothing is read
# --------------------------------------------------------------------------


def test_kiltpsa_reads_nothing_of_a_sectioned_file_named_by_a_bare_lib(tmp_path: Path):
    """kiltpsa reads ``.lib sect.lib`` as an include, and the file's own
    ``.lib tt`` as an include of a file ``tt`` that does not exist: ngspice
    drops every card of the file and runs on. Every reader says so."""
    deck = _deck(tmp_path, ".lib sect.lib", "V1 a 0 1\nM1 a a 0 0 nout\nX1 a amp\n")
    findings = _lint(deck, "kiltpsa")
    (refusal,) = findings["lib-sections-dropped-ngspice"]
    assert refusal["at"]["line"] == 2
    reason = refusal["evidence"]["reason"]
    assert "drops every card" in reason and "Name the section" in reason
    # The models the file declares are not there for ngspice.
    assert [f["subject"] for f in findings["model-missing"]] == ["M1", "X1"]
    # The run expects only the deck's own analysis.
    requirements = deck_requirements(deck, reader=LibraryReader("ngspice", "kiltpsa"))
    assert requirements.analyses == (".op",)
    assert not requirements.has_save
    # Hierarchy discovery and the comparison say what they did not read.
    with pytest.raises(NetlistError, match="drops every card"):
        load_hierarchy(str(deck), [tmp_path], SemanticProfile("ngspice", "kiltpsa"))
    graph = parse_netlist_graph(deck, reader=SemanticProfile("ngspice", "kiltpsa"))
    assert "amp" not in graph.subckts
    (missing,) = graph.missing_includes
    assert "drops every card" in missing.reason


def test_kiltpsa_reads_a_file_with_no_sections_whole(tmp_path: Path):
    deck = _deck(tmp_path, ".lib plain.lib", "V1 a 0 1\nM1 a a 0 0 nplain\n")
    assert _lint(deck, "kiltpsa") == {}
    hierarchy = load_hierarchy(str(deck), [tmp_path], SemanticProfile("ngspice", "kiltpsa"))
    (fet,) = [row for row in hierarchy.instances if row.reference == "M1"]
    assert [model.name for model in fet.model_family] == ["nplain"]


@pytest.mark.parametrize("mode", ["", "hsa"])
@pytest.mark.parametrize("library", ["sect.lib", "plain.lib"])
def test_native_and_hsa_fail_on_a_lib_card_with_no_section(
    mode: str, library: str, tmp_path: Path
):
    """ngspice in native and hsa mode stops on ``.lib <file>`` with no
    section ("unimplemented dot command '.lib'"), whatever the file holds."""
    deck = _deck(tmp_path, f".lib {library}")
    (refusal,) = _lint(deck, mode)["lib-without-section-ngspice"]
    reason = refusal["evidence"]["reason"]
    assert "unimplemented" in reason
    # The fix: name a section the file declares, or include a file with none.
    assert ("Name the section" if library == "sect.lib" else ".include plain.lib") in reason
    requirements = deck_requirements(deck, reader=LibraryReader("ngspice", mode))
    assert requirements.analyses == (".op",)
    with pytest.raises(NetlistError, match="unimplemented"):
        load_hierarchy(str(deck), [tmp_path], SemanticProfile("ngspice", mode))
    graph = parse_netlist_graph(deck, reader=SemanticProfile("ngspice", mode))
    (missing,) = graph.missing_includes
    assert "unimplemented" in missing.reason


@pytest.mark.parametrize(
    ("mode", "rule"),
    [
        ("kiltpsa", "lib-sections-dropped-ngspice"),
        ("hsa", "lib-without-section-ngspice"),
        ("", "lib-without-section-ngspice"),
    ],
)
def test_an_include_of_a_file_that_declares_sections_reads_nothing(
    mode: str, rule: str, tmp_path: Path
):
    """ngspice reads the included file's own ``.lib tt`` as it reads one in a
    deck: kiltpsa drops every card of the file, native and hsa stop on it."""
    deck = _deck(tmp_path, ".include sect.lib", "V1 a 0 1\nX1 a amp\n")
    (refusal,) = _lint(deck, mode)[rule]
    assert "section line of sect.lib" in refusal["evidence"]["reason"]
    assert ".lib sect.lib tt" in refusal["evidence"]["reason"]
    with pytest.raises(NetlistError, match="section line"):
        load_hierarchy(str(deck), [tmp_path], SemanticProfile("ngspice", mode))
    graph = parse_netlist_graph(deck, reader=SemanticProfile("ngspice", mode))
    assert "amp" not in graph.subckts
    assert deck_requirements(deck, reader=LibraryReader("ngspice", mode)).analyses == (".op",)
    # A file with no sections is still read whole.
    plain = _deck(tmp_path, ".include plain.lib", "V1 a 0 1\nM1 a a 0 0 nplain\n")
    assert _lint(plain, mode) == {}


def test_a_section_selected_in_an_included_file_is_refused_in_kiltpsa(tmp_path: Path):
    """kiltpsa reads ``.lib sect.lib tt`` as two includes wherever it sits;
    one in a file the deck includes was not refused."""
    _deck(tmp_path, ".op")
    _write(tmp_path, "wrap.inc", ".lib sect.lib tt\n")
    deck = _write(tmp_path, "w.cir", "* deck\n.include wrap.inc\nV1 a 0 1\n.op\n.end\n")
    (refusal,) = _lint(deck, "kiltpsa")["lib-section-ngspice"]
    assert refusal["at"]["file"] == str(tmp_path / "wrap.inc")
    assert refusal["subject"] == "tt"
    assert _lint(deck, "hsa") == {}


def test_a_profile_standing_in_for_another_family_reads_a_library_as_before(tmp_path: Path):
    """A session whose simulator has no profile of its own (QSPICE, Xyce)
    reads a deck through an LTspice profile; whose reading of a library is
    not decided, so the comparison reads it as before: the section named."""
    _write(tmp_path, "corners.lib", ".lib tt\n.subckt load p\nR1 p 0 2k\n.ends load\n.endl tt\n")
    deck = _write(tmp_path, "s.cir", "* deck\n.lib corners.lib tt\nX1 a load\n.op\n.end\n")
    reader = SemanticProfile.of(
        Producer("ltspice", None, None), None, definitions_of=Producer("qspice", None, None)
    )
    graph = parse_netlist_graph(deck, reader=reader)
    assert graph.missing_includes == ()
    assert "load" in graph.subckts


# --------------------------------------------------------------------------
# .lib file tt: section tt only, not the cards outside every section
# --------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["", "hsa"])
def test_a_model_outside_every_section_is_missing(mode: str, tmp_path: Path):
    deck = _deck(tmp_path, ".lib sect.lib tt", "V1 a 0 1\nM1 a a 0 0 nout\nM2 a a 0 0 ntt\n")
    assert [f["subject"] for f in _lint(deck, mode)["model-missing"]] == ["M1"]
    hierarchy = load_hierarchy(str(deck), [tmp_path], SemanticProfile("ngspice", mode))
    families = {row.reference: row.model_family for row in hierarchy.instances}
    assert families["M1"] == ()
    assert [model.name for model in families["M2"]] == ["ntt"]


@pytest.mark.parametrize("mode", ["", "hsa"])
def test_the_save_and_analysis_outside_every_section_are_not_expected(mode: str, tmp_path: Path):
    deck = _deck(tmp_path, ".lib sect.lib tt")
    requirements = deck_requirements(deck, reader=LibraryReader("ngspice", mode))
    assert requirements.analyses == (".op",)
    assert not requirements.has_save


@pytest.mark.parametrize("mode", ["", "hsa"])
def test_the_comparison_reads_the_definition_in_the_selected_section(mode: str, tmp_path: Path):
    """ngspice reads only section tt's ``amp`` (2k), not the one outside
    every section (1k). The comparison held the deck equal to the 1k
    reference and different from the 2k one."""
    deck = _deck(tmp_path, ".lib sect.lib tt", "V1 a 0 1\nX1 a amp\n")
    reader = SemanticProfile("ngspice", mode)

    def reference(value: str) -> Path:
        return _write(
            tmp_path,
            f"ref_{value}.cir",
            f"* reference\n.subckt amp p\nR1 p 0 {value}\n.ends amp\n"
            "V1 a 0 1\nX1 a amp\n.op\n.end\n",
        )

    assert compare_graphs(reference("2k"), deck, reader=reader).equivalent
    assert not compare_graphs(reference("1k"), deck, reader=reader).equivalent
    hierarchy = load_hierarchy(str(deck), [tmp_path], reader)
    (inner,) = [row for row in hierarchy.instances if row.instance == ("X1", "R1")]
    assert inner.value.value == 2000.0


# --------------------------------------------------------------------------
# LTspice: the rest of the line is the path
# --------------------------------------------------------------------------


@pytest.mark.parametrize("stem", ["lib_path_space_unquoted", "lib_path_space_quoted"])
def test_ltspice_reads_a_lib_path_holding_a_space_as_one_path(stem: str, tmp_path: Path):
    """Both builds run either deck (recorded); every reader finds ``load``."""
    deck = INPUTS / "deck" / f"{stem}.cir"
    assert _lint(deck, "", family="ltspice") == {}
    requirements = deck_requirements(deck, reader=LibraryReader("ltspice"))
    assert requirements.analyses == (".op",)
    hierarchy = load_hierarchy(str(deck), [INPUTS], SemanticProfile("ltspice"))
    assert {row.instance for row in hierarchy.instances} >= {("X1", "R1")}
    graph = parse_netlist_graph(deck, reader=SemanticProfile("ltspice"))
    assert "load" in graph.subckts
    assert graph.missing_includes == ()
    staged = stage_deck(deck, tmp_path / "staged", [INPUTS], origin=deck, dialect="ltspice")
    assert [entry.path.name for entry in staged.manifest if entry.path != deck] == ["load.lib"]
    assert [entry.section for entry in staged.manifest] == [None] * len(staged.manifest)


def test_ltspice_reads_a_section_name_as_part_of_the_path(tmp_path: Path):
    """``deck/lib_section``: both builds look for ``corners.lib tt`` and
    stop. Every reader says that nothing was read, and why."""
    deck = INPUTS / "deck" / "lib_section.cir"
    (refusal,) = _lint(deck, "", family="ltspice")["lib-section-ltspice"]
    assert "no library sections" in refusal["evidence"]["reason"]
    assert "'corners.lib' alone names a file" in refusal["evidence"]["reason"]
    graph = parse_netlist_graph(deck, reader=SemanticProfile("ltspice"))
    (missing,) = graph.missing_includes
    assert missing.target == "corners.lib tt"
    assert "no library sections" in missing.reason
    with pytest.raises(NetlistError, match="no library sections"):
        load_hierarchy(str(deck), [INPUTS], SemanticProfile("ltspice"))


# --------------------------------------------------------------------------
# Live: ngspice reads each card as the one answer says
# --------------------------------------------------------------------------


def _library_reads_of(deck: Path, reader: LibraryReader) -> list[LibraryRead]:
    """What each include and library card of the deck brings in for
    ``reader``, as every reader of a library asks it (``library_read``)."""
    cards = lex_deck(read_spice_text(deck)).cards
    reads = []
    for reference in scan_include_references(cards, deck, family=reader.family):
        target = resolve_existing(resolve_reference(deck.parent, reference.raw_path))
        sections = (
            None
            if target is None
            else card_sections(lex_deck(read_spice_text(target), title=False).cards, target, 1)
        )
        reads.append(library_read(reference, sections, reader))
    return reads


def _ngspice_version() -> str:
    out = subprocess.run(
        ["ngspice", "-v"], capture_output=True, text=True, timeout=60, check=False
    ).stdout
    match = re.search(r"ngspice-\S+", out)
    return match.group(0) if match else out.strip()


def _ngspice_value(deck: Path, mode: str) -> float | str:
    """``v(b)`` as ngspice prints it in ``mode``, or its first error line."""
    command = ["ngspice", "-n", "-b"]
    if mode:
        command += ["-D", f"ngbehavior={mode}"]
    result = subprocess.run(
        [*command, deck.name],
        capture_output=True,
        text=True,
        timeout=60,
        cwd=deck.parent,
        check=False,
    )
    text = result.stdout + result.stderr
    match = re.search(r"v\(b\) = (\S+)", text)
    if match:
        return float(match.group(1))
    if "unimplemented dot command" in text:
        return "unimplemented"
    return "failed"


#: v(b) of a 1 V source over rv and a 1 ohm resistor: 1/(1+rv). The deck
#: sets rv=1000 ahead of the card; the library's cards outside every section
#: set rv=1, section tt's 2 and a card after it 3.
_LIVE_LIBRARY = ".param rv=1\n.lib tt\n.param rv=2\n.endl tt\n.param rv=3\n"
_VALUE = {1000: 1 / 1001, 2: 1 / 3, 3: 1 / 4}

#: What ngspice-42 does with each card in each mode (measured).
LIVE_TABLE = {
    (".lib sect.lib tt", ""): 2,
    (".lib sect.lib tt", "hsa"): 2,
    # kiltpsa: two includes; "tt" is not found and the file's cards are dropped.
    (".lib sect.lib tt", "kiltpsa"): 1000,
    (".lib sect.lib", ""): "unimplemented",
    (".lib sect.lib", "hsa"): "unimplemented",
    (".lib sect.lib", "kiltpsa"): 1000,
    (".lib plain.lib", ""): "unimplemented",
    (".lib plain.lib", "hsa"): "unimplemented",
    (".lib plain.lib", "kiltpsa"): 3,
    # An include reads the file's own section lines as the deck's.
    (".include sect.lib", ""): "unimplemented",
    (".include sect.lib", "hsa"): "unimplemented",
    (".include sect.lib", "kiltpsa"): 1000,
    (".include plain.lib", ""): 3,
    (".include plain.lib", "hsa"): 3,
    (".include plain.lib", "kiltpsa"): 3,
    # Modes outside the three the reader models, read by their flags
    # (``ngspice_compat``): lt reads .lib as an include as kiltpsa does, and
    # s3 stops on every .lib card.
    (".lib sect.lib tt", "lt"): 1000,
    (".lib sect.lib", "lt"): 1000,
    (".lib plain.lib", "lt"): 3,
    (".lib sect.lib tt", "s3"): "unimplemented",
    (".lib sect.lib", "s3"): "unimplemented",
    (".lib plain.lib", "s3"): "unimplemented",
}


@needs_ngspice
@pytest.mark.parametrize(("card", "mode"), list(LIVE_TABLE))
def test_ngspice_reads_a_lib_card_as_the_table_says(card: str, mode: str, tmp_path: Path):
    print(f"ngspice that ran: {_ngspice_version()}")
    _write(tmp_path, "sect.lib", _LIVE_LIBRARY)
    _write(tmp_path, "plain.lib", ".param rv=3\n")
    deck = _write(
        tmp_path,
        "d.cir",
        f"* live\n.param rv=1000\n{card}\nV1 a 0 1\nR1 a b {{rv}}\nR2 b 0 1\n"
        ".op\n.control\nop\nprint v(b)\n.endc\n.end\n",
    )
    measured = _ngspice_value(deck, mode)
    expected = LIVE_TABLE[card, mode]
    if isinstance(expected, str):
        assert measured == expected
    else:
        assert measured == pytest.approx(_VALUE[expected], rel=1e-5)
    # The one answer says the same: what it reads, or that it reads nothing.
    reads = _library_reads_of(deck, LibraryReader("ngspice", mode))
    (read,) = reads
    if expected == 2:
        assert read.selected == "tt" and not read.outside and read.unread is None
    elif expected == 3:
        assert read.selected is None and read.outside and read.unread is None
    elif expected == 1000 and card.endswith(" tt"):
        # Refused by the lint already (lib-section-ngspice).
        assert read.unread is not None and "two plain includes" in read.unread
    else:
        assert read.unread is not None
