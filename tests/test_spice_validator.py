"""Pre-flight validation of SPICE directives — Layer A blocklist."""

from ltspice_mcp.lib.spice_validator import list_rules, validate_directive


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
