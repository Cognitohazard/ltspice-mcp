"""Value-editing behaviour on netlist element cards.

Covers the element-class dispatcher behind the ``set_component_value`` op:
source waveform specs, behavioural-source prefixes, controlled-source gains,
model-name elements, and the level-label lint.
"""

import pytest

from ltspice_mcp.errors import NetlistError
from ltspice_mcp.lib.component_value import apply_value_to_instance
from ltspice_mcp.lib.spice_lex import emit, lex


class TestSourceWaveformValuesAccepted:
    """``set_component_value(V1, "PULSE(...)")`` was rejected as
    whitespace-bearing despite being a legal source spec."""

    def _run(self, body: str, ref: str, value: str) -> str:
        cards = lex(body).cards
        instance = next(c for c in cards if c.kind == "instance" and c.name == ref)
        apply_value_to_instance(instance, value)
        return emit(cards)

    def test_pulse_replaces_value_field(self) -> None:
        out = self._run(
            "V1 in 0 1\n",
            "V1",
            "PULSE(0 1 0 2n 2n 100n 200n)",
        )
        assert out.strip() == "V1 in 0 PULSE(0 1 0 2n 2n 100n 200n)"

    def test_sin_replaces_existing_pulse(self) -> None:
        out = self._run(
            "V1 in 0 PULSE(0 1 0 1n 1n 50n 100n)\n",
            "V1",
            "SIN(0 1 1k) AC 1",
        )
        assert out.strip() == "V1 in 0 SIN(0 1 1k) AC 1"

    def test_ac_magnitude_only(self) -> None:
        out = self._run("V1 in 0 1\n", "V1", "AC 1")
        assert out.strip() == "V1 in 0 AC 1"

    def test_pwl_with_internal_whitespace(self) -> None:
        out = self._run("I1 a 0 0\n", "I1", "PWL(0 0 1m 1 2m 0)")
        assert out.strip() == "I1 a 0 PWL(0 0 1m 1 2m 0)"


class TestBSourcePrefixPreserved:
    """A brace-only value used to drop ``V=``/``I=``."""

    def _run(self, body: str, ref: str, value: str) -> str:
        cards = lex(body).cards
        instance = next(c for c in cards if c.kind == "instance" and c.name == ref)
        apply_value_to_instance(instance, value)
        return emit(cards)

    def test_brace_keeps_v_prefix(self) -> None:
        out = self._run(
            "B1 fb 0 V={V(out)*0.5+1}\n",
            "B1",
            "{V(in)*0.5+1}",
        )
        assert out.strip() == "B1 fb 0 V={V(in)*0.5+1}"

    def test_explicit_kv_overrides_existing_type(self) -> None:
        # Switching from V= to I= drops the old V= rather than leaving
        # a stale slot behind.
        out = self._run(
            "B1 fb 0 V={V(out)*0.5+1}\n",
            "B1",
            "I=1m",
        )
        assert "V=" not in out
        assert "I=1m" in out

    def test_bare_value_with_no_existing_prefix_refuses(self) -> None:
        cards = lex("B1 fb 0 V=0\n").cards
        b1 = next(c for c in cards if c.kind == "instance" and c.name == "B1")
        # Strip V= manually so the body has no prefix to preserve.
        b1.replace_body("B1 fb 0")
        with pytest.raises(NetlistError, match="V=expr"):
            apply_value_to_instance(b1, "10")

    def test_operator_after_call_value_round_trips_without_orphan(self) -> None:
        # Editing ``V=V(in)*2`` used to rewrite only the ``V=V(in)`` span and
        # leave the ``*2`` behind, silently corrupting the card. The whole
        # expression must be replaced.
        out = self._run("B1 out 0 V=V(in)*2\n", "B1", "{V(in)*3}")
        assert out.strip() == "B1 out 0 V={V(in)*3}"

    def test_spaced_operator_expression_refused_not_corrupted(self) -> None:
        # The whitespace-around-operators form can't be re-joined
        # unambiguously, so editing it must refuse rather than leave orphans.
        cards = lex("B1 out 0 V = V(a) + V(b)\n").cards
        b1 = next(c for c in cards if c.kind == "instance" and c.name == "B1")
        with pytest.raises(NetlistError, match="not fully parseable"):
            apply_value_to_instance(b1, "{V(a)+V(b)}")


class TestControlledSourceGainReplacement:
    """``set_component_value(E1, "20")`` used to overwrite the
    controlling-node pair AND the gain. Should replace only the gain."""

    def _run(self, body: str, ref: str, value: str) -> str:
        cards = lex(body).cards
        instance = next(c for c in cards if c.kind == "instance" and c.name == ref)
        apply_value_to_instance(instance, value)
        return emit(cards)

    def test_e_source_gain_only(self) -> None:
        out = self._run("E1 buf 0 in 0 10\n", "E1", "20")
        assert out.strip() == "E1 buf 0 in 0 20"

    def test_g_source_gain_only(self) -> None:
        out = self._run("G1 out 0 in 0 5\n", "G1", "12")
        assert out.strip() == "G1 out 0 in 0 12"

    def test_f_source_gain_only(self) -> None:
        out = self._run("F1 out 0 V_sense 2\n", "F1", "5")
        assert out.strip() == "F1 out 0 V_sense 5"

    def test_f_source_with_control_ref_change(self) -> None:
        out = self._run("F1 out 0 V_sense 2\n", "F1", "V_new 5")
        assert out.strip() == "F1 out 0 V_new 5"


class TestModelNameElementsEditable:
    """A diode and the controlled switches carry a trailing model name, the
    same card shape as M/Q/J. ``set_component_value`` used to raise
    ``Unsupported element prefix`` for them; it must swap the model name."""

    def _run(self, body: str, ref: str, value: str) -> str:
        cards = lex(body).cards
        instance = next(c for c in cards if c.kind == "instance" and c.name == ref)
        apply_value_to_instance(instance, value)
        return emit(cards)

    def test_diode_model_swapped(self) -> None:
        out = self._run("D1 a k 1N4148\n", "D1", "1N5817")
        assert out.strip() == "D1 a k 1N5817"

    def test_voltage_switch_model_swapped(self) -> None:
        out = self._run("S1 n+ n- nc+ nc- SW1\n", "S1", "SW2")
        assert out.strip() == "S1 n+ n- nc+ nc- SW2"

    def test_current_switch_model_swapped(self) -> None:
        out = self._run("W1 n+ n- Vsense ISW1\n", "W1", "ISW2")
        assert out.strip() == "W1 n+ n- Vsense ISW2"

    def test_diode_with_area_factor_model_swapped(self) -> None:
        # The diode carries a trailing area factor. The swap must replace the
        # model and leave the area intact — it used to clobber the area "2" and
        # leave the real model 1N4148 in place.
        out = self._run("D1 a k 1N4148 2\n", "D1", "1N5817")
        assert out.strip() == "D1 a k 1N5817 2"

    def test_voltage_switch_with_state_model_swapped(self) -> None:
        # The trailing ON state must survive; only the model name changes.
        out = self._run("S1 n1 n2 nc1 nc2 MYSW ON\n", "S1", "NEWSW")
        assert out.strip() == "S1 n1 n2 nc1 nc2 NEWSW ON"

    def test_unsupported_prefix_offers_escape_hatch(self) -> None:
        # A still-unsupported prefix should point the user at editing the card
        # directly, not raise a bare "Unsupported element prefix".
        cards = lex("O1 a b c d TLINE\n").cards
        o1 = next(c for c in cards if c.kind == "instance" and c.name == "O1")
        with pytest.raises(NetlistError, match="Edit the card directly"):
            apply_value_to_instance(o1, "TLINE2")


class _FakeComp:
    def __init__(self, attributes: dict):
        self.attributes = attributes


class _FakeEditor:
    """Minimal stand-in exposing the ``.components`` mapping the lint reads."""

    def __init__(self, components: dict):
        self.components = components


class TestLevelLabelLint:
    """A GUI opamp complexity label (Level.N) written to a subcircuit's Value
    becomes a stray positional token → 'sub-circuit name is not defined'. Warn
    at edit time; fire only on the dotted-level pattern AND a subcircuit signal."""

    def test_x_prefix_reference_warns(self):
        from ltspice_mcp.tools.circuit import _level_label_lint

        ed = _FakeEditor({"X1": _FakeComp({})})
        assert _level_label_lint(ed, "X1", "Level.2") is not None

    def test_spicemodel_attr_on_u_prefix_warns(self):
        # InstName is U1 but the .asy Prefix makes it an X device → SpiceModel
        # attribute is the tell.
        from ltspice_mcp.tools.circuit import _level_label_lint

        ed = _FakeEditor({"U1": _FakeComp({"SpiceModel": "UniversalOpamp2"})})
        assert _level_label_lint(ed, "U1", "level.1") is not None

    def test_plain_resistor_value_not_flagged(self):
        from ltspice_mcp.tools.circuit import _level_label_lint

        ed = _FakeEditor({"R1": _FakeComp({})})
        assert _level_label_lint(ed, "R1", "10k") is None

    def test_level_label_on_non_subckt_not_flagged(self):
        # The label pattern alone isn't enough — no subcircuit signal, no warning.
        from ltspice_mcp.tools.circuit import _level_label_lint

        ed = _FakeEditor({"R1": _FakeComp({})})
        assert _level_label_lint(ed, "R1", "Level.2") is None

    def test_any_value_on_spicemodel_symbol_warns(self):
        # The general case: ANY value on a SpiceModel-selected symbol (not just
        # Level.N) becomes a stray positional token and corrupts the netlist.
        from ltspice_mcp.tools.circuit import _level_label_lint

        ed = _FakeEditor({"U1": _FakeComp({"SpiceModel": "UniversalOpamp2"})})
        assert _level_label_lint(ed, "U1", "10k") is not None

    def test_subckt_by_value_without_spicemodel_not_flagged(self):
        # A library part that carries its subckt name IN Value (no SpiceModel)
        # is the normal case — it must stay quiet.
        from ltspice_mcp.tools.circuit import _level_label_lint

        ed = _FakeEditor({"X1": _FakeComp({})})
        assert _level_label_lint(ed, "X1", "LT1013") is None

    def test_empty_value_not_flagged(self):
        from ltspice_mcp.tools.circuit import _level_label_lint

        ed = _FakeEditor({"U1": _FakeComp({"SpiceModel": "UniversalOpamp2"})})
        assert _level_label_lint(ed, "U1", "") is None
