"""Unit tests for sweep range generation and id minting."""

import re

import pytest

from ltspice_mcp.lib.store import validate_job_id
from ltspice_mcp.lib.sweep_utils import (
    generate_id,
    generate_sweep_range,
    sanitize_stem,
    sweep_range_count,
)


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


class TestSweepRangeCount:
    """sweep_range_count must equal len(generate_sweep_range) — without building it."""

    @pytest.mark.parametrize(
        ("start", "stop", "step", "points", "scale"),
        [
            (0, 10, None, 5, "linear"),
            (0, 10, 2.5, None, "linear"),
            (5, 5, None, 1, "linear"),
            (0, 1, None, 3, "linear"),
            (0, 100, 7, None, "linear"),
            (1, 1000, None, 4, "log"),
            (1, 1000, 10, None, "log"),
            (1, 1e6, 2, None, "log"),
        ],
    )
    def test_count_matches_generated_length(self, start, stop, step, points, scale):
        assert sweep_range_count(start, stop, step, points, scale) == len(
            generate_sweep_range(start, stop, step, points, scale)
        )

    def test_count_is_cheap_for_huge_ranges(self):
        # The whole point: a billion-point range is counted, never allocated.
        assert sweep_range_count(0, 1, None, 10**9, "linear") == 10**9
        assert sweep_range_count(0, 1e9, 1, None, "linear") >= 10**9

    def test_count_validates_like_generate(self):
        with pytest.raises(ValueError, match="neither"):
            sweep_range_count(0, 10, None, None, "linear")
        with pytest.raises(ValueError, match="mutually exclusive"):
            sweep_range_count(0, 10, 1, 10, "linear")
        with pytest.raises(ValueError, match="Unknown scale"):
            sweep_range_count(0, 10, None, 5, "quadratic")
        with pytest.raises(ValueError, match="positive"):
            sweep_range_count(0, 100, None, 5, "log")


class TestGenerateId:
    """Ids name their deck, and stay usable as filenames and kill tokens."""

    def test_stemless_form_is_prefix_timestamp_random(self):
        assert re.fullmatch(r"exp_\d+_[0-9a-f]{8}", generate_id("exp"))

    def test_stem_is_embedded_between_prefix_and_timestamp(self):
        assert re.fullmatch(r"exp_rc-filter_\d+_[0-9a-f]{8}", generate_id("exp", "rc-filter"))

    @pytest.mark.parametrize(
        ("stem", "expected"),
        [
            ("RC Filter", "rc-filter"),
            ("rc.filter.v2", "rc-filter-v2"),
            ("two__scores", "two-scores"),
            ("--edges--", "edges"),
            ("ldo_µA_bias", "ldo-a-bias"),
            ("x" * 200, "x" * 24),
        ],
    )
    def test_sanitization(self, stem: str, expected: str):
        assert sanitize_stem(stem) == expected

    @pytest.mark.parametrize("stem", ["", "   ", "___", "回路", "..."])
    def test_unusable_stem_falls_back_to_the_stemless_form(self, stem: str):
        assert sanitize_stem(stem) == ""
        assert re.fullmatch(r"exp_\d+_[0-9a-f]{8}", generate_id("exp", stem))

    def test_a_long_stem_cannot_push_the_id_past_the_job_id_limit(self):
        job_id = generate_id("exp", "a-very-long-deck-name-" + "z" * 200)
        assert validate_job_id(job_id) == job_id

    @pytest.mark.parametrize("stem", [None, "amp", "RC Filter.v2", "x" * 200])
    def test_ids_stay_filename_safe(self, stem: str | None):
        assert re.fullmatch(r"[a-z0-9_-]+", generate_id("exp", stem))

    @pytest.mark.parametrize("stem", [None, "amp", "amp_v2", "RC Filter.v2", "回路"])
    def test_underscore_count_is_fixed_whatever_the_stem(self, stem: str | None):
        # proc_kill matches a job id at a filename boundary; that match can only
        # be scoped to one job because no id is a prefix of another, which rests
        # on the stem contributing no underscores of its own.
        job_id = generate_id("exp", stem)
        assert job_id.count("_") == (3 if sanitize_stem(stem or "") else 2)
