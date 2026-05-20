"""Pre-flight validation of SPICE directives — Layer A blocklist."""

from ltspice_mcp.lib.spice_lex import lex
from ltspice_mcp.lib.spice_validator import (
    list_rules,
    validate_directive,
    validate_netlist_arity,
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
    """C-N4: validate_netlist must flag instance lines whose positional-
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
        assert any(
            "E1" in str(i["message"]) and "at least 2" in str(i["message"]) for i in issues
        )

    def test_rcl_keyed_primary_value_form_passes(self):
        # Codex round-2 M2: ``R1 a b R=1k`` is the keyed primary-value
        # form. InstanceLine eats "b" as the value slot, leaving
        # inst.nodes=["a"] — but the SPICE intent is two nodes (a, b).
        issues = self._arity("R1 a b R=1k\nC1 c d C=10n\nL1 e f L=1u\n.end")
        assert all(
            r not in str(i["message"]) for i in issues for r in ("R1", "C1", "L1")
        )

    def test_rcl_keyed_form_with_only_one_node_still_flagged(self):
        # Single positional node with the keyed form is still short.
        issues = self._arity("R1 a R=1k\n.end")
        assert any(
            "R1" in str(i["message"]) and "at least 2" in str(i["message"]) for i in issues
        )

    def test_rcl_with_side_effect_kv_no_false_positive(self):
        # ``R1 a b 1k TC=0.001`` has positional value + TC= side-effect
        # KV (not the primary-value KV). The R= rule shouldn't trigger.
        issues = self._arity("R1 a b 1k TC=0.001\n.end")
        assert all("R1" not in str(i["message"]) for i in issues)
