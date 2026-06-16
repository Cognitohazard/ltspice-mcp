"""Pre-flight validation of SPICE directives — Layer A blocklist."""

import pytest

from ltspice_mcp.errors import NetlistError
from ltspice_mcp.lib.component_value import apply_value_to_instance
from ltspice_mcp.lib.spice_lex import lex
from ltspice_mcp.lib.spice_validator import (
    list_rules,
    validate_directive,
    validate_netlist_arity,
    validate_netlist_bias_topology,
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


class TestTranTstepZero:
    """``.tran 0 <tstop>`` (zero step time, auto-timestep) runs on LTspice but
    ngspice rejects it. Flagged only for the ngspice target so a clean LTspice
    deck stays clean."""

    def test_tstep_zero_flagged_for_ngspice(self):
        err = validate_directive(".tran 0 5m", simulator="ngspice")
        assert err is not None
        assert err.rule_name == "tran_tstep_zero_ngspice"
        assert "TSTEP" in err.message

    def test_tstep_zero_allowed_for_ltspice(self):
        assert validate_directive(".tran 0 5m") is None  # default is LTspice
        assert validate_directive(".tran 0 5m", simulator="LTspice") is None

    def test_zero_tstep_with_trailing_args_still_flagged(self):
        # .tran <Tstep> <Tstop> <Tstart> <Tmax> [uic] — Tstep is still arg 1.
        assert validate_directive(".tran 0 5m 0 1u", simulator="ngspice") is not None
        assert validate_directive(".tran 0 5m uic", simulator="ngspice") is not None

    def test_nonzero_tstep_passes_ngspice(self):
        assert validate_directive(".tran 1u 5m", simulator="ngspice") is None
        assert validate_directive(".tran 1u 5m 0 1u", simulator="ngspice") is None

    def test_bare_tstop_not_flagged(self):
        # ".tran 5m" is a lone Tstop (LTspice shorthand), not Tstep=0.
        assert validate_directive(".tran 5m", simulator="ngspice") is None

    def test_parameterised_tstep_left_alone(self):
        # A braced/param step time is resolved at run time, not pre-flighted.
        assert validate_directive(".tran {ts} 5m", simulator="ngspice") is None


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

    def test_spaced_operator_expression_flagged_as_unparseable(self):
        # ``V = V(a) + V(b)`` (operators surrounded by whitespace) leaves orphan
        # tokens the lexer can't re-join; previously this passed validation and
        # both reads and edits silently dropped the tail. It must be flagged.
        issues = self._arity("B1 out 0 V = V(a) + V(b)\n.end")
        assert any(
            "B1" in str(i["message"]) and "not fully parsed" in str(i["message"]) for i in issues
        )

    def test_glued_operator_expression_passes(self):
        # The glued forms now tokenize as one value, so they are valid.
        issues = self._arity("B1 out 0 V=V(in)*2\n.end")
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

    def test_cl_keyed_primary_value_accepted_for_ngspice(self):
        # The C=/L= primary-value rejection is LTspice-only; ngspice accepts
        # those forms, so the ngspice target must not flag them.
        cards = lex("C1 c d C=10n\nL1 e f L=1u\n.end\n").cards
        issues = validate_netlist_arity(cards, simulator="ngspice")
        assert all(
            "C1" not in str(i["message"]) and "L1" not in str(i["message"]) for i in issues
        ), issues


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


class TestBiasTopology:
    """Nets with no DC path to ground are statically detectable: a node
    touched by two or more element terminals that still cannot reach ground
    through any DC-conductive element has undefined operating-point bias.

    The graph is built by conservative over-connection — only capacitor
    dielectrics, MOSFET gate oxides, and current-source branches are DC
    opens; bias-dependent diodes / transistor channels / switches all
    conduct — so a flag is provable, never a guess. Warning severity;
    degree-1 nodes belong to the dangling pass, not this one."""

    def _bias(self, text: str):
        return validate_netlist_bias_topology(lex(text + "\n").cards)

    # --- must FIRE: true degeneracies ---

    def test_mosfet_gate_on_coupling_cap_only_flagged(self):
        # Gate reached only through a coupling cap, no bias resistor: the
        # gate oxide is a DC open, so the gate net floats at DC.
        issues = self._bias(
            "V1 vdd 0 5\nVin in 0 AC 1\nC1 in g 1u\nM1 d g s 0 NMOS\nRd vdd d 1k\nRs s 0 1k\n.op"
        )
        assert len(issues) == 1
        msg = str(issues[0]["message"])
        assert "'g'" in msg
        assert "M1" in msg
        assert "no DC path to ground" in msg

    def test_capacitive_divider_midnode_flagged(self):
        # C-C divider to ground with no resistive tie: the mid node floats.
        issues = self._bias("V1 in 0 1\nC1 in mid 1n\nC2 mid 0 1n\n.op")
        assert len(issues) == 1
        msg = str(issues[0]["message"])
        assert "'mid'" in msg
        assert "capacitor" in msg.lower()

    def test_current_source_only_node_flagged(self):
        # A node driven by a current source (plus a cap) has no path that
        # pins its voltage — DC voltage is undefined.
        issues = self._bias("I1 0 n 1m\nC1 n 0 1u\n.op")
        assert len(issues) == 1
        msg = str(issues[0]["message"])
        assert "'n'" in msg
        assert "I1" in msg

    def test_isolated_transformer_secondary_collapses_to_one_issue(self):
        # Galvanically isolated secondary with no reference to node 0:
        # the whole domain floats and reports a SINGLE grouped issue.
        issues = self._bias(
            "V1 p1 0 1\nL1 p1 0 1m\nL2 s1 s2 1m\nK1 L1 L2 0.99\nRload s1 s2 1k\n.op"
        )
        assert len(issues) == 1
        msg = str(issues[0]["message"])
        assert "group" in msg.lower()
        assert "s1" in msg and "s2" in msg

    # --- must STAY CLEAN: no false positive ---

    def test_rc_lowpass_clean(self):
        assert self._bias("V1 in 0 1\nR1 in out 1k\nC1 out 0 1n\n.op") == []

    def test_biased_mosfet_gate_clean(self):
        # Gate tied to a rail through a bias resistor: it has a DC path.
        assert (
            self._bias(
                "V1 vdd 0 5\nVin in 0 AC 1\nC1 in g 1u\nRg vdd g 1meg\n"
                "M1 d g s 0 NMOS\nRd vdd d 1k\nRs s 0 1k\n.op"
            )
            == []
        )

    def test_common_source_amp_clean(self):
        assert (
            self._bias(
                "V1 vdd 0 5\nVin in 0 AC 1\nRg1 vdd g 1meg\nRg2 g 0 1meg\n"
                "Cin in g 1u\nM1 d g s 0 NMOS\nRd vdd d 1k\nRs s 0 1k\nCs s 0 10u\n.op"
            )
            == []
        )

    def test_switched_cap_clean(self):
        # A switch can be closed: conservative edge, so the sampling cap
        # node is not flagged floating.
        assert (
            self._bias(
                "V1 vdd 0 5\nVc clk 0 PULSE(0 5 0 1n 1n 1u 2u)\nS1 vdd n clk 0 SW\nC1 n 0 1p\n.op"
            )
            == []
        )

    def test_subckt_grounded_through_port_clean(self):
        # An internal node reaches "ground" only through a port — the port
        # is a sink, so no false positive inside the body.
        assert (
            self._bias(
                ".subckt rcfilter in out\nR1 in out 1k\nC1 out ref 1n\n"
                "R2 out ref 1meg\n.ends rcfilter"
            )
            == []
        )

    def test_x_instance_clique_reaches_ground_clean(self):
        # An X instance is a clique over its terminals: a node reaching
        # ground only through the subckt body approximation is not flagged.
        assert self._bias("V1 a 0 1\nR1 a 0 1k\nX1 a b SUB\nC1 b 0 1n\n.op") == []

    def test_diode_coupled_node_clean(self):
        # Reaches ground only through a diode (bias-dependent -> conservative
        # edge), so not flagged.
        assert self._bias("V1 in 0 1\nD1 in out DMOD\nR1 out 0 1k\n.op") == []

    def test_degree_one_stub_owned_by_dangling_not_bias(self):
        # A genuinely open degree-1 node is the dangling pass's job; the
        # bias-topology pass must not double-flag it.
        assert self._bias("V1 in 0 1\nR1 in out 1k\n.op") == []

    def test_fully_grounded_deck_clean(self):
        assert self._bias("V1 in 0 1\nR1 in mid 1k\nR2 mid 0 1k\n.op") == []

    # --- adversarial-verify regressions ---

    def test_subckt_grounded_via_internal_node_0_clean(self):
        # A subckt that references ground through node 0 INSIDE its body
        # (not via a passed port) still biases the external node — node 0 is
        # global. Common in op-amp / regulator / sensor macromodels.
        assert (
            self._bias(
                ".subckt LOAD sig\nR1 sig 0 1k\n.ends\nV1 vin 0 AC 1\nC1 vin a 1u\nX1 a LOAD\n.op"
            )
            == []
        )

    def test_single_port_subckt_grounded_internally_clean(self):
        # A one-port X instance has no clique peer, so internal-node-0
        # grounding is the only way its port can pass a DC reference up.
        assert (
            self._bias(
                ".subckt PULLDN p\nR1 p 0 1k\n.ends\n"
                "V1 vin 0 AC 1\nC1 vin n1 1u\nX1 n1 PULLDN\n.op"
            )
            == []
        )

    def test_capacitive_ladder_collapses_to_one_issue(self):
        # A physically-contiguous capacitive region is one floating domain,
        # not one issue per conductive fragment.
        issues = self._bias("C1 a b 1u\nC2 b c 1u\nC3 c d 1u\nV1 s 0 1\nR9 s 0 1k\n.op")
        assert len(issues) == 1, issues
        assert "group" in str(issues[0]["message"]).lower()

    def test_ac_coupled_chain_collapses_to_one_issue(self):
        # A floating multi-stage chain that spans capacitor boundaries
        # reports once, not once per R-bridged fragment.
        issues = self._bias(
            "C1 n1 n2 1u\nR1 n2 n3 1k\nC2 n3 n4 1u\nR2 n4 n5 1k\nC3 n5 n6 1u\n"
            "V1 s 0 1\nR9 s 0 1k\n.op"
        )
        assert len(issues) == 1, issues
        msg = str(issues[0]["message"])
        for node in ("n2", "n3", "n4", "n5"):
            assert node in msg

    def test_vccs_only_node_named_as_current_source(self):
        # A node reached only through a VCCS (G) output is a current-source
        # degeneracy, named as such — not the generic message.
        issues = self._bias("V1 in 0 1\nR1 in 0 1k\nG1 0 n3 in 0 1m\nC1 n3 0 1u\n.op")
        assert len(issues) == 1, issues
        msg = str(issues[0]["message"])
        assert "current source G1" in msg

    def test_degree_one_member_not_named_in_bias_group(self):
        # A degree-1 node belongs to the dangling pass; even when it hangs
        # off a floating domain, the bias group message must not name it.
        issues = self._bias("V1 in 0 1\nR1 in 0 1k\nC1 m1 m2 1u\nC2 m2 0 1u\nR2 m1 m3 1k\n.op")
        assert len(issues) == 1, issues
        assert "m3" not in str(issues[0]["message"])

    # --- behavioral B-source: I= is a current source (no DC voltage path) ---

    def test_behavioral_current_source_only_node_flagged(self):
        # B... I= is a current source (ngspice ASRC: I given => current
        # source); a node reached only through it and a cap is DC-undefined,
        # the same degeneracy as the independent I element.
        issues = self._bias("B1 0 n I=1m\nC1 n 0 1u\n.op")
        assert len(issues) == 1, issues
        assert "'n'" in str(issues[0]["message"])
        assert "no DC path to ground" in str(issues[0]["message"])

    def test_behavioral_voltage_source_node_clean(self):
        # B... V= is a voltage source — it pins the node-pair voltage, so it
        # conducts a DC reference like any voltage source.
        assert self._bias("B1 n 0 V=5\nC1 n 0 1u\n.op") == []

    def test_behavioral_current_source_with_rpar_clean(self):
        # LTspice's Rpar adds a parallel resistor — the explicit DC path
        # workaround — so an I= source with Rpar is not floating.
        assert self._bias("B1 0 n I=1m Rpar=1k\nC1 n 0 1u\n.op") == []

    def test_behavioral_current_source_with_cpar_still_flagged(self):
        # Cpar is a parallel capacitor — open at DC — so it does NOT rescue
        # the node; only Rpar does.
        issues = self._bias("B1 0 n I=1m Cpar=1p\nC1 n 0 1u\n.op")
        assert len(issues) == 1, issues

    # --- .GLOBAL is name-scoping, not a ground reference ---

    def test_floating_global_rail_flagged(self):
        # .global only shares the name across scopes; an undriven global rail
        # still has no DC path to ground (node 0 is the only inherent ref).
        issues = self._bias(".global vdd\nI1 0 vdd 1m\nC1 vdd 0 1u\n.op")
        assert len(issues) == 1, issues
        assert "vdd" in str(issues[0]["message"])

    def test_driven_global_rail_clean(self):
        # A global rail driven to a defined potential at top level reaches
        # ground and must not be flagged.
        assert self._bias(".global vdd\nVdd vdd 0 5\nR1 vdd out 1k\nR2 out 0 1k\n.op") == []

    def test_global_driven_inside_subckt_not_false_positive(self):
        # A global's driver may live in a different scope than where it is
        # used. The deck-wide reachability must see the subckt-internal drive
        # (here through a one-port subckt) so the top-level net is not
        # falsely flagged.
        assert (
            self._bias(
                ".global vdd\n.subckt REG out\nV1 out 0 5\n.ends\n"
                "X1 vdd REG\nR1 vdd a 1k\nC1 a 0 1u\n.op"
            )
            == []
        )


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
