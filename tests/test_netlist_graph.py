"""Connectivity comparison of exported netlists (``lib/netlist_graph.py``).

Hand-built netlists shaped like real LTspice ``.net`` exports: node names in the
``N001`` style, ``§``-marked subcircuit instances, ``µ`` value suffixes, and the
``.backanno`` / ``.end`` tail. Covers the graph model, hierarchy flattening, and
the structural / value / anchor / arity dimensions of ``compare_graphs``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ltspice_mcp.lib.netlist_graph import (
    NetlistGraphError,
    compare_graphs,
    flatten_graph,
    parse_netlist_graph,
)

# ---------------------------------------------------------------------------
# Fixtures: netlists as text
# ---------------------------------------------------------------------------

# A resistor divider with a buffered-in source. `mid` is the internal tap.
DIVIDER = """\
* divider.asc
Vin in 0 5
R1 in mid 1k
R2 mid 0 2k
.op
.backanno
.end
"""

# Identical divider, internal net `mid` renamed to `n001` (LTspice-export style).
DIVIDER_RENAMED = """\
* divider.asc
Vin in 0 5
R1 in N001 1k
R2 N001 0 2k
.op
.backanno
.end
"""

# Same nets and refs as DIVIDER, but the resistors trade places: R1 now shorts
# in->gnd and R2 bridges in->mid. Same component set, different topology.
DIVIDER_SWAPPED = """\
* divider.asc
Vin in 0 5
R1 in 0 1k
R2 in mid 2k
.op
.backanno
.end
"""


def test_parse_reuses_lexer_and_projects_components() -> None:
    graph = parse_netlist_graph(DIVIDER)
    refs = {c.ref for c in graph.components}
    assert refs == {"Vin", "R1", "R2"}
    r1 = next(c for c in graph.components if c.ref == "R1")
    assert r1.type_letter == "R"
    assert r1.nodes == ("in", "mid")
    assert r1.value == "1k"
    vin = next(c for c in graph.components if c.ref == "Vin")
    assert vin.type_letter == "V"
    assert vin.nodes == ("in", "0")


def test_renamed_internal_nets_are_equivalent() -> None:
    result = compare_graphs(DIVIDER, DIVIDER_RENAMED, anchors=["in"])
    assert result.equivalent is True
    assert result.structurally_equivalent is True
    assert result.node_partition_mismatches == []
    assert result.value_mismatches == []


def test_swapped_resistors_report_named_partition_mismatch() -> None:
    result = compare_graphs(DIVIDER, DIVIDER_SWAPPED, anchors=["in", "mid"])
    assert result.equivalent is False
    # A reference net is forced onto more than one candidate net...
    mismatches = result.node_partition_mismatches
    assert mismatches, "expected a named node-partition mismatch"
    # ...naming the resistor pins and the nets that cannot coexist under the swap.
    involved = {pin for d in mismatches for pin in d.involved}
    named_nets = {n for d in mismatches for n in d.reference_nets} | {
        n for d in mismatches for n in d.candidate_nets
    }
    assert any(pin.startswith("R") for pin in involved)
    assert "mid" in named_nets and "0" in named_nets


def test_shorted_nets_report_names_both_reference_nets() -> None:
    # Reference keeps `a` and `b` as distinct nets; candidate merges both onto
    # `m` (a short). The mismatch must name BOTH reference nets and the shared
    # candidate net, so the caller sees which two nets were shorted together.
    ref = "Vin p 0 5\nR1 p a 1k\nR2 p b 1k\n.end\n"
    cand = "Vin p 0 5\nR1 p m 1k\nR2 p m 1k\n.end\n"
    result = compare_graphs(ref, cand, anchors=["p"])
    mismatches = result.node_partition_mismatches
    assert mismatches, "expected a short to be reported"
    short = next(d for d in mismatches if len(d.reference_nets) > 1)
    assert set(short.reference_nets) == {"a", "b"}
    assert short.candidate_nets == ("m",)
    assert result.equivalent is False


def test_swapped_resistors_are_not_structurally_equivalent() -> None:
    result = compare_graphs(DIVIDER, DIVIDER_SWAPPED, anchors=["in", "mid"])
    assert result.structurally_equivalent is False


# ---------------------------------------------------------------------------
# Anchors
# ---------------------------------------------------------------------------

# Two isomorphic three-resistor strings. Reference labels the middle tap `out`;
# candidate puts the `out` label one node over (on the wrong net).
STRING_REF = """\
Vin in 0 5
R1 in a 1k
R2 a out 1k
R3 out 0 1k
.end
"""
STRING_MISANCHORED = """\
Vin in 0 5
R1 in out 1k
R2 out b 1k
R3 b 0 1k
.end
"""


def test_misanchored_output_reports_anchor_violation() -> None:
    result = compare_graphs(STRING_REF, STRING_MISANCHORED, anchors=["in", "out"])
    # Wiring is the same three-resistor string...
    assert result.structurally_equivalent is True
    # ...but `out` sits on a structurally different net -> anchor violation.
    assert any(v.anchor == "out" for v in result.anchor_violations)
    assert result.equivalent is False


def test_anchor_absent_on_one_side_is_a_violation() -> None:
    # Net `a` exists in STRING_REF but not in STRING_MISANCHORED.
    result = compare_graphs(STRING_REF, STRING_MISANCHORED, anchors=["a"])
    violation = next(v for v in result.anchor_violations if v.anchor == "a")
    assert "reference" in violation.detail


# ---------------------------------------------------------------------------
# Value tolerance
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("ref_val", "cand_val", "equal"),
    [
        ("1k", "1000", True),
        ("1k", "1.0k", True),
        ("4.7u", "4.7µ", True),
        ("2.2µ", "2.2e-6", True),
        ("1k", "1.2k", False),
        ("1k", "1k1", False),
    ],
)
def test_value_tolerance(ref_val: str, cand_val: str, equal: bool) -> None:
    ref = f"Vin in 0 5\nR1 in 0 {ref_val}\n.end\n"
    cand = f"Vin in 0 5\nR1 in 0 {cand_val}\n.end\n"
    result = compare_graphs(ref, cand, anchors=["in"])
    if equal:
        assert result.value_mismatches == []
        assert result.equivalent is True
    else:
        assert len(result.value_mismatches) == 1
        vm = result.value_mismatches[0]
        assert vm.ref == "R1"
        assert vm.reference_value == ref_val
        assert vm.candidate_value == cand_val
        assert result.equivalent is False


def test_retype_and_param_mismatch_are_distinct_channels() -> None:
    ref = "Vin in 0 5\nM1 d g 0 0 nmos W=1u L=1u\n.end\n"
    # Same ref M1: model nmos -> pmos (retype) AND W widened (param mismatch).
    cand = "Vin in 0 5\nM1 d g 0 0 pmos W=2u L=1u\n.end\n"
    result = compare_graphs(ref, cand, anchors=["in", "d", "g"])
    assert [r.ref for r in result.retyped] == ["M1"]
    assert result.retyped[0].reference_type == "M:nmos"
    assert result.retyped[0].candidate_type == "M:pmos"
    param = next(p for p in result.param_mismatches if p.ref == "M1" and p.key == "w")
    assert param.reference_value == "1u"
    assert param.candidate_value == "2u"


def test_type_letter_change_is_add_remove_not_retype() -> None:
    # A resistor replaced by a capacitor changes the reference (R1 -> C1): the
    # component identity changed, so it reads as removed + added, not a retype.
    ref = "Vin in 0 5\nR1 in 0 1k\n.end\n"
    cand = "Vin in 0 5\nC1 in 0 1k\n.end\n"
    result = compare_graphs(ref, cand, anchors=["in"])
    assert [d.ref for d in result.removed] == ["R1"]
    assert [d.ref for d in result.added] == ["C1"]
    assert result.retyped == []


def test_added_and_removed_components() -> None:
    ref = "Vin in 0 5\nR1 in 0 1k\n.end\n"
    cand = "Vin in 0 5\nR1 in 0 1k\nC1 in 0 1p\n.end\n"
    result = compare_graphs(ref, cand, anchors=["in"])
    assert [d.ref for d in result.added] == ["C1"]
    assert result.removed == []
    assert result.equivalent is False


# ---------------------------------------------------------------------------
# Ground aliasing
# ---------------------------------------------------------------------------


def test_ground_alias_gnd_equals_zero() -> None:
    ref = "Vin in 0 5\nR1 in 0 1k\n.end\n"
    cand = "Vin in GND 5\nR1 in GND 1k\n.end\n"
    result = compare_graphs(ref, cand, anchors=["in"])
    assert result.equivalent is True
    assert result.structurally_equivalent is True


# ---------------------------------------------------------------------------
# Hierarchy flattening
# ---------------------------------------------------------------------------

# A nested subcircuit: OUTER instantiates INNER; INNER wraps a single resistor.
NESTED = """\
Vin in 0 5
XU1 in out OUTER
.subckt OUTER a b
XI1 a mid INNER
RL mid b 2k
.ends OUTER
.subckt INNER p q
R1 p q 1k
.ends INNER
.end
"""


def test_flatten_nested_subckt_hierarchical_refs_and_unique_nets() -> None:
    graph = parse_netlist_graph(NESTED)
    flat = flatten_graph(graph)
    refs = {c.ref for c in flat.components}
    # Top-level Vin stays flat; the nested resistors get hierarchical paths.
    assert "Vin" in refs
    assert "XU1.RL" in refs
    assert "XU1.XI1.R1" in refs
    # The deep resistor's terminals: p bound through OUTER's `a` -> top `in`;
    # q binds through to OUTER's internal `mid`, named at the level that defines
    # it (XU1's body), so it is `xu1/mid` -- shared, not duplicated per level.
    deep = next(c for c in flat.components if c.ref == "XU1.XI1.R1")
    assert deep.nodes[0] == "in"
    assert deep.nodes[1] == "xu1/mid"
    # `mid` inside OUTER is internal to XU1 and shared by XI1 and RL.
    rl = next(c for c in flat.components if c.ref == "XU1.RL")
    assert rl.nodes[0] == deep.nodes[1]  # INNER.q == OUTER.mid


def test_two_instances_of_a_subckt_do_not_share_internal_nets() -> None:
    text = """\
Vin in 0 5
XA in a HALF
XB a 0 HALF
.subckt HALF p q
R1 p m 1k
R2 m q 1k
.ends HALF
.end
"""
    flat = flatten_graph(parse_netlist_graph(text))
    a_mid = next(c for c in flat.components if c.ref == "XA.R1").nodes[1]
    b_mid = next(c for c in flat.components if c.ref == "XB.R1").nodes[1]
    assert a_mid != b_mid


# A PDK-style export: a FET wrapped in an X-subcircuit that is NOT defined in the
# file (its definition lives in a .lib include). It must stay a black-box leaf.
PDK_WRAPPED = """\
* pdk.asc
* Generated by LTspice 26.0.2 for Windows.
VIN VDD5 0 5
X§MB1 NB1 NB1 0 0 nmos_6p0 W=10u L=2u ;§pnba D)G)S)B
X§RB VDD5 NB1 0 ppolyf_u_1k r_length=820u r_width=2u ;§pnba A)B)BODY
.lib "sm141064.lib" typical
.backanno
.end
"""


def test_undefined_pdk_subckt_stays_a_black_box_leaf() -> None:
    graph = parse_netlist_graph(PDK_WRAPPED)
    flat = flatten_graph(graph)
    refs = {c.ref for c in flat.components}
    # § marker stripped; no expansion (subckt undefined) -> instance kept whole.
    assert refs == {"VIN", "XMB1", "XRB"}
    fet = next(c for c in flat.components if c.ref == "XMB1")
    assert fet.type_letter == "X"
    assert fet.model == "nmos_6p0"
    assert fet.nodes == ("nb1", "nb1", "0", "0")
    assert dict(fet.params) == {"W": "10u", "L": "2u"}


def test_pdk_export_compares_equal_to_itself() -> None:
    result = compare_graphs(PDK_WRAPPED, PDK_WRAPPED, anchors=["VDD5"])
    assert result.equivalent is True


def test_pdk_fet_widening_is_a_param_mismatch() -> None:
    widened = PDK_WRAPPED.replace("W=10u L=2u", "W=20u L=2u")
    result = compare_graphs(PDK_WRAPPED, widened, anchors=["VDD5"])
    assert any(p.ref == "XMB1" and p.key == "w" for p in result.param_mismatches)
    assert result.equivalent is False


# ---------------------------------------------------------------------------
# Arity and parse errors
# ---------------------------------------------------------------------------

ARITY_BAD = """\
Vin in 0 5
XU1 in out extra THREEPORT
.subckt THREEPORT a b
R1 a b 1k
.ends THREEPORT
.end
"""


def test_flatten_raises_on_port_arity_mismatch() -> None:
    graph = parse_netlist_graph(ARITY_BAD)
    with pytest.raises(NetlistGraphError) as exc:
        flatten_graph(graph)
    assert "arity" in str(exc.value).lower()
    assert "XU1" in str(exc.value)


def test_compare_surfaces_arity_error_without_aborting() -> None:
    good = """\
Vin in 0 5
XU1 in out TWOPORT
.subckt TWOPORT a b
R1 a b 1k
.ends TWOPORT
.end
"""
    result = compare_graphs(good, ARITY_BAD, anchors=["in"])
    assert result.arity_errors, "arity mismatch should be surfaced, not raised"
    assert any("XU1" in e.ref for e in result.arity_errors)
    assert result.equivalent is False


def test_unbalanced_brace_is_a_parse_failure_naming_the_line() -> None:
    bad = "Vin in 0 5\nBad n1 0 V={1+\n.end\n"
    with pytest.raises(NetlistGraphError) as exc:
        parse_netlist_graph(bad)
    assert exc.value.line == 2


def test_unclosed_subckt_is_a_parse_failure() -> None:
    bad = "Vin in 0 5\n.subckt LEAK a b\nR1 a b 1k\n.end\n"
    with pytest.raises(NetlistGraphError):
        parse_netlist_graph(bad)


# ---------------------------------------------------------------------------
# Backwards-authored passives and result payload
# ---------------------------------------------------------------------------


def test_symmetric_resistor_authored_backwards_still_matches() -> None:
    ref = "Vin in 0 5\nR1 in mid 1k\nR2 mid 0 2k\n.end\n"
    # R2 written 0->mid instead of mid->0: same connection, terminals swapped.
    cand = "Vin in 0 5\nR1 in mid 1k\nR2 0 mid 2k\n.end\n"
    result = compare_graphs(ref, cand, anchors=["in"])
    assert result.equivalent is True


def test_result_as_dict_is_json_shaped() -> None:
    result = compare_graphs(DIVIDER, DIVIDER_SWAPPED, anchors=["in", "mid"])
    payload = result.as_dict()
    assert payload["equivalent"] is False
    assert isinstance(payload["node_partition_mismatches"], list)
    assert set(payload).issuperset(
        {"equivalent", "structurally_equivalent", "added", "removed", "retyped"}
    )


# ---------------------------------------------------------------------------
# External .lib / .include resolution
# ---------------------------------------------------------------------------

# The reference states the block inline; the compiled candidate externalizes the
# same block to a project-local .lib, which is what a block-based compiler emits.
INLINE_BLOCK = """\
Vin in 0 5
XSTAGE in out DIVBLOCK
.subckt DIVBLOCK a b
R1 a mid 1k
R2 mid b 2k
.ends DIVBLOCK
.end
"""

BLOCK_LIB = """\
* project-local block implementation
.subckt DIVBLOCK a b
R1 a mid 1k
R2 mid b 2k
.ends DIVBLOCK
"""

EXTERNAL_BLOCK = """\
Vin in 0 5
XSTAGE in out DIVBLOCK
.include divblock.lib
.end
"""


def test_block_in_lib_matches_inline_reference(tmp_path: Path) -> None:
    (tmp_path / "divblock.lib").write_text(BLOCK_LIB)
    inline = tmp_path / "inline.cir"
    inline.write_text(INLINE_BLOCK)
    external = tmp_path / "external.cir"
    external.write_text(EXTERNAL_BLOCK)

    result = compare_graphs(inline, external, anchors=["in", "out"])
    assert result.equivalent is True, result.as_dict()
    assert result.unresolved_subckts == []
    # The block really was expanded on both sides, not black-boxed on both.
    flat = flatten_graph(parse_netlist_graph(external))
    assert {c.ref for c in flat.components} == {"Vin", "XSTAGE.R1", "XSTAGE.R2"}


def test_lib_with_section_token_still_resolves(tmp_path: Path) -> None:
    # `.lib <file> <section>` — the section name must not be read as the file.
    (tmp_path / "divblock.lib").write_text(BLOCK_LIB)
    deck = tmp_path / "sectioned.cir"
    deck.write_text(EXTERNAL_BLOCK.replace(".include divblock.lib", ".lib divblock.lib typical"))
    flat = flatten_graph(parse_netlist_graph(deck))
    assert "XSTAGE.R1" in {c.ref for c in flat.components}


def test_quoted_include_path_resolves(tmp_path: Path) -> None:
    (tmp_path / "div block.lib").write_text(BLOCK_LIB)
    deck = tmp_path / "quoted.cir"
    deck.write_text(EXTERNAL_BLOCK.replace(".include divblock.lib", '.include "div block.lib"'))
    flat = flatten_graph(parse_netlist_graph(deck))
    assert "XSTAGE.R1" in {c.ref for c in flat.components}


def test_missing_lib_names_the_unresolved_subckt(tmp_path: Path) -> None:
    deck = tmp_path / "external.cir"
    deck.write_text(EXTERNAL_BLOCK)  # divblock.lib deliberately not written

    graph = parse_netlist_graph(deck)
    assert [m.target for m in graph.missing_includes] == ["divblock.lib"]
    flat = flatten_graph(graph)
    assert flat.unresolved_subckts == ("DIVBLOCK",)

    result = compare_graphs(tmp_path / "external.cir", tmp_path / "external.cir")
    unresolved = result.unresolved_subckts[0]
    assert unresolved.name == "DIVBLOCK"
    # The missing include is what separates this from a plain PDK black box.
    assert "divblock.lib" in unresolved.missing_includes


def test_include_resolver_denial_blocks_the_read(tmp_path: Path) -> None:
    """A denying include resolver records a missing include and never opens the
    file — the sandbox seam the tool layer wires to ``safe_path`` (ID-28)."""
    lib = tmp_path / "divblock.lib"
    lib.write_text(BLOCK_LIB)  # present on disk, but the resolver denies it
    deck = tmp_path / "external.cir"
    deck.write_text(EXTERNAL_BLOCK)

    seen: list[Path] = []

    def deny(candidate: Path) -> Path | None:
        seen.append(candidate)
        return None

    graph = parse_netlist_graph(deck, include_resolver=deny)
    # The resolver was consulted with the joined candidate path.
    assert seen and seen[0].name == "divblock.lib"
    # Denied → recorded as missing with the denial reason, definition never loaded.
    assert [m.target for m in graph.missing_includes] == ["divblock.lib"]
    assert all("denied" in m.reason for m in graph.missing_includes)
    assert flatten_graph(graph).unresolved_subckts == ("DIVBLOCK",)


def test_include_resolver_allows_by_returning_path(tmp_path: Path) -> None:
    """An allowing resolver (identity) parses exactly as no resolver would."""
    (tmp_path / "divblock.lib").write_text(BLOCK_LIB)
    deck = tmp_path / "external.cir"
    deck.write_text(EXTERNAL_BLOCK)

    graph = parse_netlist_graph(deck, include_resolver=lambda p: p)
    assert graph.missing_includes == ()
    assert "XSTAGE.R1" in {c.ref for c in flatten_graph(graph).components}


def test_pdk_black_box_reports_no_missing_include() -> None:
    # No include directive at all: the model is simply not supplied, which must
    # read differently from "the definition file is missing".
    deck = "Vin in 0 5\nX1 in out nmos_6p0\n.end\n"
    result = compare_graphs(deck, deck, anchors=["in"])
    unresolved = result.unresolved_subckts[0]
    assert unresolved.name == "nmos_6p0"
    assert unresolved.missing_includes == ()
    assert result.equivalent is True  # a shared black box is not a difference


@pytest.mark.parametrize(
    ("inline_first", "expected_ref"),
    [(True, "1k"), (False, "9k")],
)
def test_duplicate_definition_first_in_textual_order_wins(
    tmp_path: Path, inline_first: bool, expected_ref: str
) -> None:
    """Pins the measured simulator rule: first definition wins, later ones are
    ignored (ngspice 42: "redefinition of .subckt DUP, ignored"). Inline does not
    win by virtue of being inline -- only by coming first."""
    (tmp_path / "dup.lib").write_text(".subckt DUP a b\nR1 a b 9k\n.ends DUP\n")
    inline_def = ".subckt DUP a b\nR1 a b 1k\n.ends DUP\n"
    include_line = ".include dup.lib\n"
    body = (inline_def + include_line) if inline_first else (include_line + inline_def)
    deck = tmp_path / "dup.cir"
    deck.write_text(f"Vin in 0 5\nXD in out DUP\n{body}.end\n")

    graph = parse_netlist_graph(deck)
    winner = graph.subckts["dup"]
    assert winner.components[0].value == expected_ref

    # The losing definition is surfaced, never a silent pick.
    dup = graph.duplicate_subckts[0]
    assert dup.name == "DUP"
    assert len(dup.ignored) == 1
    assert ("<inline>" in dup.used) is inline_first


def test_include_cycle_terminates(tmp_path: Path) -> None:
    # a.lib includes b.lib includes a.lib — must terminate and still register.
    (tmp_path / "a.lib").write_text(".include b.lib\n.subckt DIVBLOCK a b\nR1 a b 1k\n.ends\n")
    (tmp_path / "b.lib").write_text(".include a.lib\n.subckt OTHER p q\nR9 p q 5k\n.ends\n")
    deck = tmp_path / "cyc.cir"
    deck.write_text("Vin in 0 5\nXSTAGE in out DIVBLOCK\n.include a.lib\n.end\n")

    graph = parse_netlist_graph(deck)  # terminates rather than recursing forever
    assert "divblock" in graph.subckts
    assert "other" in graph.subckts
    flat = flatten_graph(graph)
    assert flat.unresolved_subckts == ()


def test_include_depth_is_bounded(tmp_path: Path) -> None:
    # Chain deeper than the 3-level bound: the deepest definition is not reached
    # and the give-up point is recorded rather than silently dropped.
    for level in range(1, 5):
        (tmp_path / f"L{level}.lib").write_text(f".include L{level + 1}.lib\n")
    (tmp_path / "L5.lib").write_text(".subckt DEEP a b\nR1 a b 1k\n.ends DEEP\n")
    deck = tmp_path / "deep.cir"
    deck.write_text("Vin in 0 5\nXD in out DEEP\n.include L1.lib\n.end\n")

    graph = parse_netlist_graph(deck)
    assert "deep" not in graph.subckts
    assert any("depth limit" in m.reason for m in graph.missing_includes)


def test_text_parse_records_unresolvable_include() -> None:
    # Bare text has no directory context, so an include cannot be followed --
    # a fact worth surfacing rather than a silent black box.
    graph = parse_netlist_graph(EXTERNAL_BLOCK)
    assert [m.target for m in graph.missing_includes] == ["divblock.lib"]
    assert "no base directory" in graph.missing_includes[0].reason


def test_base_dir_lets_text_resolve_includes(tmp_path: Path) -> None:
    (tmp_path / "divblock.lib").write_text(BLOCK_LIB)
    graph = parse_netlist_graph(EXTERNAL_BLOCK, base_dir=tmp_path)
    assert graph.missing_includes == ()
    flat = flatten_graph(graph)
    assert "XSTAGE.R1" in {c.ref for c in flat.components}


def test_unreadable_included_file_degrades_to_a_recorded_fact(tmp_path: Path) -> None:
    # A malformed third-party library must not fail the whole parse.
    (tmp_path / "divblock.lib").write_text(".subckt DIVBLOCK a b\nR1 a b {1+\n.ends\n")
    deck = tmp_path / "external.cir"
    deck.write_text(EXTERNAL_BLOCK)
    graph = parse_netlist_graph(deck)
    assert graph.missing_includes, "a broken include should be recorded, not swallowed"
    assert flatten_graph(graph).unresolved_subckts == ("DIVBLOCK",)


def test_windows_spelled_include_reaches_the_resolver_as_a_real_path(tmp_path: Path) -> None:
    """The include walk resolves references the way staging does.

    LTspice's own ``.asc`` netlister writes ``.lib C:\\...\\standard.mos``. Read
    as a relative name and hung off the deck's directory, that becomes a path
    that exists nowhere — so the sandbox seam is asked about the wrong file and
    the deck's own library is reported unusable.
    """
    deck = tmp_path / "winref.cir"
    deck.write_text(".lib C:\\Users\\dev\\LTspice\\lib\\cmp\\standard.mos\nR1 a 0 1k\n.end\n")
    seen: list[Path] = []

    def recording_resolver(candidate: Path) -> Path | None:
        seen.append(candidate)
        return None

    parse_netlist_graph(deck, include_resolver=recording_resolver)

    assert seen == [Path("/mnt/c/Users/dev/LTspice/lib/cmp/standard.mos")]
