"""Regression tests for v5 stress-test bugs and frictions."""

from pathlib import Path

import numpy as np
import pytest

from ltspice_mcp.errors import NetlistError
from ltspice_mcp.lib.ac_analysis import compute_stability_metrics, gain_at_frequencies
from ltspice_mcp.lib.log_parser import parse_step_iterations
from ltspice_mcp.lib.signal_analysis import analyze_pulse_response
from ltspice_mcp.state import SessionState
from ltspice_mcp.tools.circuit import (
    AddComponentInput,
    ApplySchematicOpsInput,
    ConnectInput,
    NetLabelInput,
    RemoveComponentInput,
    SetComponentAttributeInput,
    handle_add_component,
    handle_add_net_label,
    handle_apply_schematic_ops,
    handle_connect,
    handle_remove_component,
    handle_set_component_attribute,
)


class TestN2StepTempDegreeStripping:
    """N2: log step parser strips trailing degree symbol."""

    def test_temp_axis_strips_degree(self, tmp_path: Path):
        log = tmp_path / "step_temp.log"
        log.write_text("Circuit: foo\n.step temp=-40°\n.step temp=27°\n.step temp=125°\n")
        steps = parse_step_iterations(log)
        # Without the ° fix, the value capture would include '°' and
        # parse_value would fail downstream — the row would be dropped.
        assert len(steps) == 3
        assert steps[0]["temp"] == -40.0
        assert steps[1]["temp"] == 27.0
        assert steps[2]["temp"] == 125.0


class TestN5StabilityPhaseWarning:
    """N5: stability_metrics warns when DC phase is near ±180°."""

    def test_warns_on_inverting_output(self):
        # Synthesize a CS-amp-style transfer: gain 1000 at DC, single pole at
        # 1 kHz, BUT with a sign inversion (phase starts at 180°).
        freqs = np.logspace(0, 7, 200)
        omega = 2 * np.pi * freqs
        H_lp = 1000 / (1 + 1j * omega / (2 * np.pi * 1e3))
        H = -H_lp  # inversion → phase starts at +180°
        out = compute_stability_metrics(freqs, H)
        assert any("doesn't look like a loop-gain probe" in w for w in out["warnings"])

    def test_silent_on_loop_probe(self):
        # Standard loop probe: phase starts at 0°.
        freqs = np.logspace(0, 7, 200)
        omega = 2 * np.pi * freqs
        H = 1000 / (1 + 1j * omega / (2 * np.pi * 1e3))
        out = compute_stability_metrics(freqs, H)
        assert not any("doesn't look like a loop-gain probe" in w for w in out["warnings"])


class TestFr4PulseResponseDoubleTransition:
    """Fr4: pulse_response refuses windows with both rising AND falling edges."""

    def test_refuses_full_pulse_window(self):
        # Window contains a 0→1→0 pulse: peak-to-peak is 1, but start and
        # end levels are within rounding of each other. Mimics the v5
        # finding where steady_state ≈ 2e-4 vs peak ≈ 1. abs_delta is
        # above _LEVEL_EPSILON but tiny relative to the swing.
        t = np.linspace(0, 1, 200)
        # Start at exactly 0; pulse to 1.0 from t=0.2..0.6; return to 1e-3
        # so |final - initial| = 1e-3 (above EPSILON) but pk_pk = 1.0.
        y = np.where(t < 0.2, 0.0, np.where(t < 0.6, 1.0, 1e-3))
        with pytest.raises(ValueError, match="full pulse"):
            analyze_pulse_response(t, y)

    def test_accepts_clean_step(self):
        # Single rising edge with stable start AND stable end — no double-
        # transition, should compute successfully.
        t = np.linspace(0, 1, 200)
        y = np.where(t < 0.3, 0.0, 1.0)
        out = analyze_pulse_response(t, y)
        assert out["direction"] == "rising"
        assert out["overshoot_pct"] == 0


class TestFr5PoleOrderTolerance:
    """Fr5: pole-order estimate accepts ±3 dB/dec around an integer."""

    def test_accepts_minus_18(self):
        # A real-world miller_ota slope was -17.97 dB/dec; the v4 ±2 cutoff
        # rejected it. ±3 should accept "1" as the order.
        from ltspice_mcp.lib.ac_analysis import _estimate_order_from_slope

        assert _estimate_order_from_slope(-17.97) == 1

    def test_rejects_far_from_integer(self):
        from ltspice_mcp.lib.ac_analysis import _estimate_order_from_slope

        # -10 dB/dec is half-way between order 0 and 1 — neither is a
        # confident answer, so return None.
        assert _estimate_order_from_slope(-10.0) is None


class TestFr6GainAtPhaseUnwrappedOmitted:
    """Fr6: phase_deg_unwrapped is absent when not requested."""

    def test_omitted_by_default(self):
        freqs = np.logspace(0, 6, 100)
        omega = 2 * np.pi * freqs
        H = 1.0 / (1 + 1j * omega / (2 * np.pi * 1e3))
        points, _ = gain_at_frequencies(freqs, H, [100.0, 1e4])
        for p in points:
            assert "phase_deg_unwrapped" not in p

    def test_present_when_requested(self):
        freqs = np.logspace(0, 6, 100)
        omega = 2 * np.pi * freqs
        H = 1.0 / (1 + 1j * omega / (2 * np.pi * 1e3))
        points, _ = gain_at_frequencies(freqs, H, [100.0, 1e4], include_unwrapped_phase=True)
        for p in points:
            assert "phase_deg_unwrapped" in p


@pytest.mark.asyncio
class TestN9SetAttributeAllowlist:
    """N9: set_component_attribute rejects unknown attribute names."""

    async def test_rejects_typo(self, asc_state: SessionState, asc_file: Path):
        with pytest.raises(NetlistError, match="Unknown attribute"):
            await handle_set_component_attribute(
                SetComponentAttributeInput(
                    path=asc_file.name,
                    reference="R1",
                    attribute="NotARealAttr",
                    value="x",
                ),
                asc_state,
            )

    async def test_suggests_canonical_for_case_typo(self, asc_state: SessionState, asc_file: Path):
        with pytest.raises(NetlistError, match="Did you mean 'SpiceLine'"):
            await handle_set_component_attribute(
                SetComponentAttributeInput(
                    path=asc_file.name, reference="R1", attribute="spiceline", value="x"
                ),
                asc_state,
            )

    async def test_accepts_spiceline(self, asc_state: SessionState, asc_file: Path):
        # Sanity: the canonical name still works.
        result = await handle_set_component_attribute(
            SetComponentAttributeInput(
                path=asc_file.name, reference="R1", attribute="SpiceLine", value="tc=10ppm"
            ),
            asc_state,
        )
        assert "SpiceLine" in result.content[0].text


@pytest.mark.asyncio
class TestN12FloatingLabelWarning:
    """N12: add_net_label warns on labels placed away from any wire/pin."""

    async def test_warns_on_floating(self, asc_state: SessionState, asc_file: Path):
        result = await handle_add_net_label(
            NetLabelInput(path=asc_file.name, net="VCC_floating", x=10, y=10),
            asc_state,
        )
        assert "floating" in result.content[0].text.lower()


@pytest.mark.asyncio
class TestN8NetConflictInConnect:
    """N8: connect detects shorts between two named nets."""

    async def test_refuses_named_net_short(self, asc_state: SessionState):
        # Build a clean schematic with two resistors on disjoint named nets,
        # then try to connect them.
        from ltspice_mcp.tools.circuit import (
            CreateSchematicInput,
            handle_create_schematic,
        )

        await handle_create_schematic(CreateSchematicInput(name="net_conflict_test"), asc_state)
        await handle_add_component(
            AddComponentInput(
                path="net_conflict_test.asc",
                reference="R1",
                symbol="res",
                x=100,
                y=100,
            ),
            asc_state,
        )
        await handle_add_component(
            AddComponentInput(
                path="net_conflict_test.asc",
                reference="R2",
                symbol="res",
                x=300,
                y=100,
            ),
            asc_state,
        )
        # The test fixture's stripped 'res' symbol uses numeric pin names.
        await handle_add_net_label(
            NetLabelInput(path="net_conflict_test.asc", net="LEFT", pin="R1.1"),
            asc_state,
        )
        await handle_add_net_label(
            NetLabelInput(path="net_conflict_test.asc", net="RIGHT", pin="R2.1"),
            asc_state,
        )
        with pytest.raises(NetlistError, match="Net-label conflict"):
            await handle_connect(
                ConnectInput(
                    path="net_conflict_test.asc",
                    from_pin="R1.1",
                    to_pin="R2.1",
                ),
                asc_state,
            )


@pytest.mark.asyncio
class TestN11RemoveComponentNoFalseOrphans:
    """N11: remove_component doesn't flag wires belonging to other components."""

    async def test_other_component_pin_not_flagged(self, asc_state: SessionState, asc_file: Path):
        # Add a second resistor whose pin coincides with R1's existing wire.
        # When we remove R2, the wire connecting R1 stays — and our orphan
        # detector should NOT flag it.
        await handle_add_component(
            AddComponentInput(
                path=asc_file.name,
                reference="R2",
                symbol="res",
                x=128,
                y=112,  # same coords as R1 — pins overlap
                value="2k",
                rotation="R90",
            ),
            asc_state,
        )
        result = await handle_remove_component(
            RemoveComponentInput(path=asc_file.name, reference="R2"),
            asc_state,
        )
        # The remaining R1's wires shouldn't be flagged as orphans.
        assert "orphaned" not in result.content[0].text


@pytest.mark.asyncio
class TestFr1ApplySchematicOps:
    """Fr1: apply_schematic_ops batches add/connect/label/directive."""

    async def test_basic_transaction(self, asc_state: SessionState, work_dir: Path):
        from ltspice_mcp.tools.circuit import (
            CreateSchematicInput,
            handle_create_schematic,
        )

        await handle_create_schematic(CreateSchematicInput(name="batch_demo"), asc_state)

        result = await handle_apply_schematic_ops(
            ApplySchematicOpsInput(
                path="batch_demo.asc",
                ops=[  # type: ignore[arg-type]  # pydantic validates dicts
                    {
                        "op": "add_component",
                        "reference": "R1",
                        "symbol": "res",
                        "x": 100,
                        "y": 100,
                        "value": "1k",
                    },
                    {
                        "op": "add_component",
                        "reference": "C1",
                        "symbol": "cap",
                        "x": 200,
                        "y": 100,
                        "value": "1u",
                    },
                    {
                        "op": "add_directive",
                        "instruction": ".tran 1m",
                    },
                ],
            ),
            asc_state,
        )
        text = result.content[0].text
        data = result.structuredContent
        assert data["applied_count"] == 3
        assert data["failed_count"] == 0
        assert data["saved"] is True
        assert "All changes saved." in text

    async def test_stop_on_error_aborts(self, asc_state: SessionState, work_dir: Path):
        from ltspice_mcp.tools.circuit import (
            CreateSchematicInput,
            handle_create_schematic,
        )

        await handle_create_schematic(CreateSchematicInput(name="batch_abort"), asc_state)
        # Op #1 succeeds, op #2 fails (unknown symbol). The R1 add must NOT
        # be persisted because stop_on_error defaults to True.
        result = await handle_apply_schematic_ops(
            ApplySchematicOpsInput(
                path="batch_abort.asc",
                ops=[  # type: ignore[arg-type]  # pydantic validates dicts
                    {
                        "op": "add_component",
                        "reference": "R1",
                        "symbol": "res",
                        "x": 100,
                        "y": 100,
                    },
                    {
                        "op": "add_component",
                        "reference": "X1",
                        "symbol": "definitely_not_a_symbol",
                        "x": 200,
                        "y": 100,
                    },
                ],
            ),
            asc_state,
        )
        data = result.structuredContent
        assert data["saved"] is False
        assert data["failed_count"] == 1
        assert "Transaction aborted" in result.content[0].text

        # Verify the file actually doesn't have R1 — load and check.
        from ltspice_mcp.tools.circuit import (
            CircuitReadInput,
            handle_read_circuit,
        )

        read = await handle_read_circuit(CircuitReadInput(path="batch_abort.asc"), asc_state)
        refs = {c["reference"] for c in read.structuredContent.get("components", [])}
        assert "R1" not in refs

    async def test_continue_on_error_persists_partial(
        self, asc_state: SessionState, work_dir: Path
    ):
        from ltspice_mcp.tools.circuit import (
            CreateSchematicInput,
            handle_create_schematic,
        )

        await handle_create_schematic(CreateSchematicInput(name="batch_partial"), asc_state)
        result = await handle_apply_schematic_ops(
            ApplySchematicOpsInput(
                path="batch_partial.asc",
                ops=[  # type: ignore[arg-type]  # pydantic validates dicts
                    {
                        "op": "add_component",
                        "reference": "R1",
                        "symbol": "res",
                        "x": 100,
                        "y": 100,
                    },
                    {
                        "op": "add_component",
                        "reference": "X1",
                        "symbol": "definitely_not_a_symbol",
                        "x": 200,
                        "y": 100,
                    },
                    {
                        "op": "add_component",
                        "reference": "C1",
                        "symbol": "cap",
                        "x": 300,
                        "y": 100,
                    },
                ],
                stop_on_error=False,
            ),
            asc_state,
        )
        data = result.structuredContent
        assert data["applied_count"] == 2
        assert data["failed_count"] == 1
        assert data["saved"] is True
