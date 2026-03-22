"""Unit tests for advanced tool helpers."""

from ltspice_mcp.tools.advanced import _resolve_mc_ref


class TestResolveMcRef:
    def test_single_letter_is_type(self):
        assert _resolve_mc_ref("R") == ("R", True)
        assert _resolve_mc_ref("C") == ("C", True)
        assert _resolve_mc_ref("L") == ("L", True)

    def test_single_letter_case_insensitive(self):
        assert _resolve_mc_ref("r") == ("R", True)
        assert _resolve_mc_ref("c") == ("C", True)

    def test_type_name_maps_to_prefix(self):
        assert _resolve_mc_ref("resistors") == ("R", True)
        assert _resolve_mc_ref("capacitors") == ("C", True)
        assert _resolve_mc_ref("inductors") == ("L", True)
        assert _resolve_mc_ref("resistor") == ("R", True)

    def test_type_name_case_insensitive(self):
        assert _resolve_mc_ref("Resistors") == ("R", True)
        assert _resolve_mc_ref("CAPACITORS") == ("C", True)

    def test_component_ref(self):
        ref, is_type = _resolve_mc_ref("R1")
        assert ref == "R1"
        assert is_type is False

    def test_component_ref_preserved(self):
        ref, is_type = _resolve_mc_ref("C3")
        assert ref == "C3"
        assert is_type is False

    def test_multichar_component(self):
        ref, is_type = _resolve_mc_ref("XU1")
        assert ref == "XU1"
        assert is_type is False
