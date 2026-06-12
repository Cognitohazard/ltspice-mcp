"""Pre-flight validation of SPICE directives — Layer A blocklist."""

import pytest

from ltspice_mcp.errors import NetlistError
from ltspice_mcp.lib.component_value import apply_value_to_instance
from ltspice_mcp.lib.spice_lex import lex
from ltspice_mcp.lib.spice_validator import (
    list_rules,
    validate_directive,
    validate_netlist_arity,
    validate_netlist_dangling_nodes,
)


class TestVdbInMeas:
    def test_vdb_when_form_blocked(self):
        err = validate_directive(".meas AC fc WHEN vdb(out)=-3")
        assert err is not None
        assert err.rule_name == "vdb_in_meas"
        assert "vdb()" in err.message
        assert "mag" in err.suggestion.lower()

    def test_vdb_find_form_blocked(self):
        err = validate_directive(".meas AC gain FIND vdb(V(out)) AT 1k")
        assert err is not None
        assert err.rule_name == "vdb_in_meas"

    def test_vdb_max_form_blocked(self):
        err = validate_directive(".meas AC peak MAX vdb(out)")
        assert err is not None
        assert err.rule_name == "vdb_in_meas"

    def test_vdb_case_insensitive(self):
        assert validate_directive(".MEAS AC peak MAX VDB(out)") is not None
        assert validate_directive(".Meas AC peak MAX VdB(out)") is not None

    def test_mag_form_allowed(self):
        # The recommended replacement should pass.
        assert validate_directive(".meas AC peak MAX mag(V(out))") is None

    def test_unrelated_directive_unaffected(self):
        assert validate_directive(".tran 0 10m 0 10u") is None
        assert validate_directive(".ac dec 100 1 1Meg") is None
        assert validate_directive(".param Rd=1k") is None


class TestPhaseInMeas:
    def test_phase_blocked(self):
        err = validate_directive(".meas AC pm FIND phase(V(out)) AT 1k")
        assert err is not None
        assert err.rule_name == "phase_in_meas"
        assert "ph()" in err.suggestion

    def test_ph_form_allowed(self):
        assert validate_directive(".meas AC pm FIND ph(V(out)) AT 1k") is None


class TestGroupDelayInMeas:
    def test_group_delay_blocked(self):
        err = validate_directive(".meas AC gd FIND group_delay(V(out)) AT 1k")
        assert err is not None
        assert err.rule_name == "group_delay_in_meas"


class TestEdgeCases:
    def test_empty_directive_is_no_op(self):
        assert validate_directive("") is None
        assert validate_directive("   ") is None

    def test_simulator_filter(self):
        # vdb_in_meas is LTspice-only; ngspice path doesn't trigger it.
        assert validate_directive(".meas AC peak MAX vdb(out)", simulator="LTspice") is not None
        assert validate_directive(".meas AC peak MAX vdb(out)", simulator="ngspice") is None

    def test_vdb_outside_meas_unaffected(self):
        # .PLOT and waveform-viewer expressions are fine; only .MEAS is restricted.
        assert validate_directive(".plot AC vdb(V(out))") is None


class TestRuleListing:
    def test_list_rules_shape(self):
        rules = list_rules()
        assert len(rules) >= 3
        names = {r["name"] for r in rules}
        assert "vdb_in_meas" in names
        assert "phase_in_meas" in names
        assert all("message" in r and "suggestion" in r for r in rules)


class TestElementArity:
    """validate_netlist must flag instance lines whose positional-
    node count is below the per-element minimum, and B-sources missing
    the V=/I= prefix. These manifest at simulation time as 'Expected 2
    node names here' or 'Unknown parameter'."""

    def _arity(self, text: str):
        return validate_netlist_arity(lex(text + "\n").cards)

    def test_three_node_e_source_flagged(self):
        # E1 buf 0 20  — 3 nodes (incl. ref consumed), needs 4 + gain.
        issues = self._arity("E1 buf 0 20\n.end")
        assert any("E1" in str(i["message"]) and "at least 4" in str(i["message"]) for i in issues)

    def test_well_formed_e_source_passes(self):
        # E1 buf 0 in 0 10  — 4 nodes + gain.
        issues = self._arity("E1 buf 0 in 0 10\n.end")
        assert all("E1" not in str(i["message"]) for i in issues)

    def test_b_source_without_v_or_i_prefix_flagged(self):
        # B1 fb 0 {V(in)*0.5+1}  — missing V= or I= prefix.
        issues = self._arity("B1 fb 0 {V(in)*0.5+1}\n.end")
        assert any("B1" in str(i["message"]) and "V= or I=" in str(i["message"]) for i in issues)

    def test_b_source_with_v_prefix_passes(self):
        issues = self._arity("B1 fb 0 V={V(in)*0.5+1}\n.end")
        assert all("B1" not in str(i["message"]) for i in issues)

    def test_passive_with_one_node_flagged(self):
        # R1 a 1k  — only one positional node, needs 2.
        issues = self._arity("R1 a 1k\n.end")
        assert any("R1" in str(i["message"]) and "at least 2" in str(i["message"]) for i in issues)

    def test_well_formed_passives_pass(self):
        issues = self._arity("R1 a b 1k\nC1 a b 1n\nL1 a b 1u\n.end")
        assert issues == []

    def test_e_source_value_keyed_form_passes(self):
        # Codex H2: E1 out 0 VALUE={V(in)*2} is a legal LTspice keyed
        # behavioral form. ``InstanceLine`` parses it as params_only with
        # 2 positional nodes — the validator must NOT require 4.
        issues = self._arity("E1 out 0 VALUE={V(in)*2}\n.end")
        assert all("E1" not in str(i["message"]) for i in issues)

    def test_g_source_value_keyed_form_passes(self):
        issues = self._arity("G1 out 0 VALUE={V(in)/1k}\n.end")
        assert all("G1" not in str(i["message"]) for i in issues)

    def test_e_source_keyed_with_too_few_output_nodes_flagged(self):
        # Even in keyed form, 1 node is below the 2-node output-pair floor.
        issues = self._arity("E1 out VALUE={V(in)}\n.end")
        assert any("E1" in str(i["message"]) and "at least 2" in str(i["message"]) for i in issues)

    def test_r_keyed_primary_value_form_passes(self):
        # Real LTspice accepts ``R1 a b R=1k``. InstanceLine treats this
        # keyed primary value as a value slot, not as a third node.
        issues = self._arity("R1 a b R=1k\n.end")
        assert all("R1" not in str(i["message"]) for i in issues)

    def test_c_l_keyed_primary_value_forms_are_ltspice_errors(self):
        # Real LTspice 26 rejects C=<value> / L=<value> as unknown
        # parameters even though ngspice accepts them.
        issues = self._arity("C1 c d C=10n\nL1 e f L=1u\n.end")
        assert any("C1" in str(i["message"]) and "LTspice" in str(i["message"]) for i in issues)
        assert any("L1" in str(i["message"]) and "LTspice" in str(i["message"]) for i in issues)

    def test_rcl_keyed_form_with_only_one_node_still_flagged(self):
        # Single positional node with the keyed form is still short.
        issues = self._arity("R1 a R=1k\n.end")
        assert any("R1" in str(i["message"]) and "at least 2" in str(i["message"]) for i in issues)

    def test_rcl_with_side_effect_kv_no_false_positive(self):
        # ``R1 a b 1k TC=0.001`` has positional value + TC= side-effect
        # KV (not the primary-value KV). The R= rule shouldn't trigger.
        issues = self._arity("R1 a b 1k TC=0.001\n.end")
        assert all("R1" not in str(i["message"]) for i in issues)


class TestDanglingNodes:
    """Single-connection nodes are statically detectable: a node touched by
    exactly one element terminal in its scope gets a warning-level issue.
    Legal in deliberately unterminated fragments, hence never an error."""

    def _dangling(self, text: str):
        return validate_netlist_dangling_nodes(lex(text + "\n").cards)

    def test_single_connection_node_warned_with_element_named(self):
        issues = self._dangling("V1 in 0 1\nR1 in out 1k\n.tran 1m\n.end")
        assert len(issues) == 1
        assert "'out'" in str(issues[0]["message"])
        assert "R1" in str(issues[0]["message"])

    def test_fully_connected_deck_yields_no_issues(self):
        issues = self._dangling("V1 in 0 1\nR1 in out 1k\nC1 out 0 1n\n.tran 1m\n.end")
        assert issues == []

    def test_ground_aliases_excluded(self):
        # "0" and case-insensitive "gnd" never count as dangling.
        issues = self._dangling("V1 a GND 1\nR1 a 0 1k\nR2 gnd 0 1k\n.end")
        assert issues == []

    def test_global_nodes_excluded(self):
        issues = self._dangling(".global VDD\nV1 vdd 0 5\nR1 in 0 1k\nR2 in 0 1k\n.end")
        assert issues == []

    def test_subckt_port_used_once_in_body_not_flagged(self):
        # The port name on the .SUBCKT header counts as one occurrence in
        # the body, so a port wired to a single element is fully connected.
        issues = self._dangling(".subckt div in out\nR1 in out 1k\nR2 out 0 1k\n.ends div")
        assert issues == []

    def test_subckt_body_counted_separately_from_top_level(self):
        # "mid" is dangling inside the body (R1 only) even though the
        # top level reuses the same name with two connections.
        issues = self._dangling(
            ".subckt amp in out\nR1 in mid 1k\nR2 in out 1k\n.ends amp\nV1 mid 0 1\nR3 mid 0 1k"
        )
        assert len(issues) == 1
        assert "'mid'" in str(issues[0]["message"])
        assert "R1" in str(issues[0]["message"])
        assert "amp" in str(issues[0]["message"])

    def test_unused_subckt_port_flagged(self):
        # A port with no body connection is its own fact: the header
        # declaration is not an element terminal, so the message states
        # the port case directly instead of pretending the header is one.
        issues = self._dangling(".subckt buf in out nc\nR1 in out 1k\nR2 out 0 1k\n.ends buf")
        assert len(issues) == 1
        msg = str(issues[0]["message"])
        assert "'nc'" in msg
        assert "declared as a port of .SUBCKT buf" in msg
        assert "connected to no element terminal in its body" in msg
        assert "only one element terminal" not in msg

    def test_x_card_subckt_name_and_params_not_counted(self):
        # Positional tokens between the refdes and the subckt name are
        # nodes; the subckt name and trailing k=v params are not.
        issues = self._dangling("X1 a b myamp gain=2\nR1 a b 1k\n.end")
        assert issues == []

    def test_f_source_controlling_ref_not_counted(self):
        # F1's third positional is the controlling V-source name, not a node.
        issues = self._dangling("V1 in 0 1\nR1 in 0 1k\nF1 out 0 V1 2\nR2 out 0 1k\n.end")
        assert issues == []

    def test_v_source_multi_token_value_not_counted(self):
        # DC/AC value tokens after the two node positions are not nodes.
        issues = self._dangling("V1 in 0 DC 5 AC 1\nR1 in 0 1k\n.end")
        assert issues == []

    def test_k_card_inductor_refs_not_counted(self):
        # K positional tokens are inductor refs, not nodes.
        issues = self._dangling("L1 a 0 1u\nL2 b 0 1u\nK1 L1 L2 0.9\nR1 a 0 1k\nR2 b 0 1k\n.end")
        assert issues == []

    def test_directive_only_cards_no_op(self):
        # No instance cards means no node counting at all.
        issues = self._dangling(".tran 1m\n.meas tran v_max MAX V(out)\n.end")
        assert issues == []

    def test_diode_rectifier_deck_no_false_positives(self):
        # D has no ELEMENT_SPECS entry, but its anode/cathode are still
        # terminals — and the model name must not become a phantom node.
        issues = self._dangling("V1 in 0 1\nD1 in out DMOD\nR1 out 0 1k\n.end")
        assert issues == []

    def test_diode_deck_genuinely_dangling_node_still_warned(self):
        # Counting D's terminals cuts both ways: a cathode wired to
        # nothing else is still a single-connection node.
        issues = self._dangling("V1 in 0 1\nD1 in out DMOD\n.end")
        assert len(issues) == 1
        assert "'out'" in str(issues[0]["message"])
        assert "D1" in str(issues[0]["message"])

    def test_switch_deck_no_false_positives(self):
        # S: two switch nodes + two control nodes, then the model name.
        issues = self._dangling("V1 c 0 1\nVin in 0 1\nS1 in out c 0 SW1\nR1 out 0 1k\n.end")
        assert issues == []

    def test_transmission_line_deck_no_false_positives(self):
        # T: four port nodes; the line parameters are KEY=VALUE tokens.
        issues = self._dangling("V1 in 0 1\nT1 in 0 out 0 Td=10n Z0=50\nR1 out 0 50\n.end")
        assert issues == []

    def test_w_switch_controlling_ref_and_model_not_counted(self):
        # W's third positional is the controlling V-source name, then the
        # model — only n+ and n- are terminals.
        issues = self._dangling("V1 in 0 1\nW1 in out V1 WMOD\nR1 out 0 1k\n.end")
        assert issues == []

    def test_b_source_sensed_node_suppresses_warning(self):
        # A node probed via V(...) in a behavioural expression is a
        # connection: it must veto the single-terminal warning.
        issues = self._dangling("V1 in 0 1\nB1 out 0 V=V(in)\nR1 out 0 1k\n.end")
        assert issues == []

    def test_differential_probe_suppresses_both_nodes(self):
        # V(a,b) references both names of the differential pair.
        issues = self._dangling("V1 in 0 1\nR2 fb 0 1k\nB1 out 0 V=V(in,fb)\nR1 out 0 1k\n.end")
        assert issues == []

    def test_xspice_a_card_positionals_suppress_not_warn(self):
        # A-cards (XSPICE) have port shapes this lint does not model: no
        # token is counted, and every token vetoes a warning on its name.
        issues = self._dangling("A1 in out xgate\nV1 in 0 1\nR1 out 0 1k\n.end")
        assert issues == []


class TestApplyValueRefusesMalformedBodies:
    """``apply_value_to_instance`` must refuse value edits that would
    write a malformed element body.

    These cases raise ``NetlistError`` at the input layer (rather
    than producing a corrupt write that ``validate_netlist`` would
    later catch), so the regression here is that the typed dispatcher
    refuses each shape *before* the file is touched.
    """

    def test_b_source_brace_with_no_existing_prefix_refused(self) -> None:
        cards = lex("B1 fb 0\n").cards
        b1 = next(c for c in cards if c.kind == "instance" and c.name == "B1")
        with pytest.raises(NetlistError):
            apply_value_to_instance(b1, "{V(in)}")

    def test_e_source_multi_positional_refused(self) -> None:
        cards = lex("E1 buf 0 in 0 10\n").cards
        e1 = next(c for c in cards if c.kind == "instance" and c.name == "E1")
        with pytest.raises(NetlistError):
            apply_value_to_instance(e1, "in 0 20")  # would clobber control nodes
