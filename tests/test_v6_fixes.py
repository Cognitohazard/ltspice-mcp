"""Regressions for the v6 stress-test findings.

Each test pins behaviour to the fix delivered in lib/component_value.py,
the MC engine's spice_lex migration, the read_circuit pipeline migration,
and the segment-aware net-trace check. See workspace/stress_test_v6/
FINDINGS.md for the originating bug report.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ltspice_mcp.errors import NetlistError
from ltspice_mcp.lib.component_value import apply_value_to_instance
from ltspice_mcp.lib.spice_lex import emit, lex

# ---------------------------------------------------------------------------
# C-N1/2/3 — element-class-typed dispatch on set_component_value
# ---------------------------------------------------------------------------


class TestCN1PulseAcceptedOnVI:
    """C-N1: ``set_component_value(V1, "PULSE(...)")`` was rejected as
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


class TestCN2BSourcePrefixPreserved:
    """C-N2: a brace-only value used to drop ``V=``/``I=``."""

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


class TestCN3EgPositionalGain:
    """C-N3: ``set_component_value(E1, "20")`` used to overwrite the
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


class TestCN4ValidatorCatchesCorruption:
    """C-N4: ``validate_netlist`` should flag the malformed bodies that
    used to slip through C-N2/N3 before the typed dispatch landed.

    These cases now raise ``NetlistError`` at the input layer (rather
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


# ---------------------------------------------------------------------------
# A-N1 — MC engine on hierarchical netlists
# ---------------------------------------------------------------------------


class TestAN1HierarchicalMcDoesNotJoinSpiceCircuits:
    """A-N1: the MC runner used to do ``"".join(editor.netlist)`` which
    crashed on hierarchical netlists where ``editor.netlist`` contains
    ``SpiceCircuit`` objects for ``.subckt`` blocks. The fix reads the
    netlist via ``read_spice_text`` and lexes once instead.
    """

    def test_hierarchical_netlist_lexes_via_read_spice_text(self, tmp_path: Path) -> None:
        from ltspice_mcp.lib.encoding import read_spice_text
        from ltspice_mcp.lib.spice_lex import lex
        from ltspice_mcp.lib.spice_lex_views import InstanceLine, ModelCard

        cir = tmp_path / "hier.cir"
        cir.write_text(
            "* hierarchical\n"
            ".subckt stage in out vss\n"
            "M1 out in vss vss NM W=10u L=0.5u\n"
            ".model NM NMOS(VTO=0.4 KP=200u)\n"
            ".ends stage\n"
            "X1 in1 out1 0 stage\n"
            "V1 in1 0 1\n"
            ".tran 1u\n"
            ".end\n"
        )

        baseline_text = read_spice_text(cir)
        cards = lex(baseline_text).cards

        # The model inside the subckt is reachable.
        model_cards = [c for c in cards if c.kind == "model"]
        assert any(c.name == "NM" for c in model_cards)
        nm = next(c for c in model_cards if c.name == "NM")
        view = ModelCard.from_card(nm)
        view.set_param("VTO", 0.5)
        assert view.params["VTO"] == "0.5"

        # The X-instance is also reachable as an instance card with model "stage".
        x_cards = [c for c in cards if c.kind == "instance" and c.name == "X1"]
        assert len(x_cards) == 1
        x_view = InstanceLine.from_card(x_cards[0])
        assert x_view.model == "stage"


# ---------------------------------------------------------------------------
# J-N1 — read_circuit / list_components on encoding-edge files
# ---------------------------------------------------------------------------


class TestJN1ReadCircuitEncodingZoo:
    """J-N1: ``read_circuit`` used to crash on UTF-8-BOM, UTF-16-BE-no-BOM,
    and unclosed-``.SUBCKT`` files because spicelib's ``SpiceEditor`` was
    in the read path. The fix routes ``.cir/.net`` reads through
    ``services.extract_netlist_info`` which uses ``read_spice_text`` +
    ``cards_from_path``.
    """

    def _write_extra(self, path: Path, prefix: bytes, encoding: str) -> None:
        body = "* probe\nR1 in out 1k\n.tran 1u\n.end\n"
        path.write_bytes(prefix + body.encode(encoding))

    def test_utf8_bom_does_not_crash(self, tmp_path: Path) -> None:
        from ltspice_mcp.lib.services import extract_netlist_info

        cir = tmp_path / "utf8bom.cir"
        self._write_extra(cir, b"\xef\xbb\xbf", "utf-8")
        info = extract_netlist_info(cir)
        assert info["type"] == "netlist"
        refs = [c["reference"] for c in info["components"]]
        assert "R1" in refs

    def test_utf16le_no_bom(self, tmp_path: Path) -> None:
        from ltspice_mcp.lib.services import extract_netlist_info

        cir = tmp_path / "utf16le.cir"
        body = "* probe\nR1 in out 1k\n.tran 1u\n.end\n"
        cir.write_bytes(body.encode("utf-16-le"))
        info = extract_netlist_info(cir)
        # content must be properly decoded — no NUL interleavings
        assert "\x00" not in info["content"]
        refs = [c["reference"] for c in info["components"]]
        assert "R1" in refs

    def test_unclosed_subckt_warns_not_crashes(self, tmp_path: Path) -> None:
        from ltspice_mcp.lib.services import extract_netlist_info

        cir = tmp_path / "trunc.cir"
        cir.write_text(
            ".subckt amp in out\nR1 in mid 1k\nR2 mid out 1k\n* missing .ENDS\nV1 vdd 0 5\n.end\n"
        )
        info = extract_netlist_info(cir)
        assert "warnings" in info
        assert any("unclosed .subckt" in w.lower() for w in info["warnings"])


# ---------------------------------------------------------------------------
# E-N1 — segment-aware net trace for connect short detection
# ---------------------------------------------------------------------------


class TestEN1MidSegmentLabelDetected:
    """E-N1: a label sitting mid-segment on a wire used to be invisible
    to ``connect``'s endpoint-only label compare. The fix is segment-
    aware: the trace dragon-swallows interest points that lie on a wire
    even if they're not at an endpoint.
    """

    def test_point_on_segment_horizontal(self) -> None:
        from ltspice_mcp.tools.circuit import _point_on_segment

        # Mid-x point on a horizontal wire.
        assert _point_on_segment((150, 100), (100, 100), (200, 100))
        # Same y but outside x-range.
        assert not _point_on_segment((300, 100), (100, 100), (200, 100))
        # Different y.
        assert not _point_on_segment((150, 101), (100, 100), (200, 100))

    def test_point_on_segment_vertical(self) -> None:
        from ltspice_mcp.tools.circuit import _point_on_segment

        assert _point_on_segment((100, 150), (100, 100), (100, 200))
        assert not _point_on_segment((100, 250), (100, 100), (100, 200))
        assert not _point_on_segment((101, 150), (100, 100), (100, 200))

    def test_named_labels_strips_ground(self) -> None:
        from ltspice_mcp.tools.circuit import _named_labels

        assert _named_labels(frozenset({"OUTP", "0"})) == {"OUTP"}
        assert _named_labels(frozenset({"0"})) == set()
        assert _named_labels(frozenset()) == set()
