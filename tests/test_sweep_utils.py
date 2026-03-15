"""Unit tests for sweep range generation."""

import pytest

from ltspice_mcp.lib.sweep_utils import generate_sweep_range


class TestLinearSweep:

    def test_linear_with_points(self):
        values = generate_sweep_range(0, 10, None, 5, "linear")
        assert len(values) == 5
        assert values[0] == pytest.approx(0.0)
        assert values[-1] == pytest.approx(10.0)

    def test_linear_with_step(self):
        values = generate_sweep_range(0, 10, 2.5, None, "linear")
        assert len(values) == 5
        assert values == pytest.approx([0, 2.5, 5.0, 7.5, 10.0])

    def test_linear_single_point(self):
        values = generate_sweep_range(5, 5, None, 1, "linear")
        assert len(values) == 1
        assert values[0] == pytest.approx(5.0)

    def test_returns_python_floats(self):
        values = generate_sweep_range(0, 1, None, 3, "linear")
        for v in values:
            assert type(v) is float


class TestLogSweep:

    def test_log_with_points(self):
        values = generate_sweep_range(1, 1000, None, 4, "log")
        assert len(values) == 4
        assert values[0] == pytest.approx(1.0)
        assert values[1] == pytest.approx(10.0)
        assert values[2] == pytest.approx(100.0)
        assert values[-1] == pytest.approx(1000.0)

    def test_log_with_step(self):
        # step=10 means multiply by 10 each step: 1, 10, 100, 1000
        values = generate_sweep_range(1, 1000, 10, None, "log")
        assert len(values) == 4
        assert values[0] == pytest.approx(1.0)
        assert values[-1] == pytest.approx(1000.0)

    def test_log_rejects_non_positive(self):
        with pytest.raises(ValueError, match="positive"):
            generate_sweep_range(0, 100, None, 5, "log")
        with pytest.raises(ValueError, match="positive"):
            generate_sweep_range(-1, 100, None, 5, "log")


class TestValidation:

    def test_neither_step_nor_points(self):
        with pytest.raises(ValueError, match="neither"):
            generate_sweep_range(0, 10, None, None, "linear")

    def test_both_step_and_points(self):
        with pytest.raises(ValueError, match="mutually exclusive"):
            generate_sweep_range(0, 10, 1, 10, "linear")

    def test_unknown_scale(self):
        with pytest.raises(ValueError, match="Unknown scale"):
            generate_sweep_range(0, 10, None, 5, "quadratic")
