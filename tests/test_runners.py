"""Direct unit tests for SimulationRunner / SweepRunner / MonteCarloRunner internals.

These tests bypass the spicelib SimRunner machinery and exercise the
event-loop callback handlers (_handle_completion, _handle_run_completion,
_handle_sweep_completion, etc.) and cancel() methods, all of which are
pure logic operating on BatchJob/SimulationJob state.
"""

import asyncio
from pathlib import Path

import pytest

from ltspice_mcp.lib.montecarlo import MCSampler, MismatchRule
from ltspice_mcp.lib.runner_base import discard_generated_netlist
from ltspice_mcp.lib.spice_lex import lex


async def _wait_for(cond, *, timeout_s: float = 5.0, interval: float = 0.01) -> None:
    """Poll ``cond`` until it holds, failing at ``timeout_s``.

    Deadline-based stand-in for a fixed ``asyncio.sleep`` before an assertion:
    the awaited effect (a job admitted through the concurrency gate, a bridged
    completion callback draining onto the loop) can take longer than any single
    fixed sleep on a saturated runner, yet a real result still lands well within
    the deadline. Modeled on ``_poll_batch_done`` in test_ngspice_e2e.py.
    """
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_s
    while not cond():
        if loop.time() > deadline:
            pytest.fail(f"condition not met within {timeout_s}s")
        await asyncio.sleep(interval)


class FakeSim:
    """Minimal simulator stub."""


class FakeStepper:
    """Minimal SimStepper stand-in; ``run_all`` behavior injected per test."""

    def __init__(self, run_all=None):
        self._run_all = run_all

    def add_value_sweep(self, *a, **k):
        pass

    def add_param_sweep(self, *a, **k):
        pass

    def total_number_of_simulations(self):
        return 0

    def run_all(self, **kwargs):
        if self._run_all is not None:
            self._run_all(**kwargs)


@pytest.fixture
def loop():
    return asyncio.new_event_loop()


class TestMCSampler:
    """Our own MC perturbation engine. Replaces spicelib's Montecarlo class."""

    def test_normal_distribution_is_multiplicative(self):
        import statistics

        from ltspice_mcp.lib.montecarlo import ToleranceSpec

        sampler = MCSampler(seed=42)
        spec = ToleranceSpec(tolerance=0.05, distribution="normal")
        nominal = 1e-3  # 1 mH

        samples = [sampler.sample(nominal, spec) for _ in range(2000)]
        mean = statistics.fmean(samples)
        stdev = statistics.stdev(samples)
        # Mean within 3σ/√n of nominal.
        assert abs(mean - nominal) < 3 * (nominal * 0.05 / 3) / (len(samples) ** 0.5)
        # Stddev within 20% of theoretical σ = value * tol / 3.
        expected_sigma = nominal * 0.05 / 3
        assert 0.8 * expected_sigma < stdev < 1.2 * expected_sigma
        # No nonsense negatives or off-by-orders-of-magnitude values.
        assert all(s > 0 for s in samples)
        assert all(0.7 * nominal < s < 1.3 * nominal for s in samples)

    def test_uniform_distribution_within_tolerance(self):
        from ltspice_mcp.lib.montecarlo import ToleranceSpec

        sampler = MCSampler(seed=1)
        spec = ToleranceSpec(tolerance=0.10, distribution="uniform")
        nominal = 25e-6  # 25 µF
        samples = [sampler.sample(nominal, spec) for _ in range(500)]
        # Every sample within ±10% of nominal.
        assert all(nominal * 0.9 <= s <= nominal * 1.1 for s in samples)

    def test_seed_reproducibility(self):
        from ltspice_mcp.lib.montecarlo import ToleranceSpec

        spec = ToleranceSpec(tolerance=0.05, distribution="normal")
        s1 = MCSampler(seed=12345)
        s2 = MCSampler(seed=12345)
        seq1 = [s1.sample(1e-3, spec) for _ in range(20)]
        seq2 = [s2.sample(1e-3, spec) for _ in range(20)]
        assert seq1 == seq2

    def test_different_seeds_diverge(self):
        from ltspice_mcp.lib.montecarlo import ToleranceSpec

        spec = ToleranceSpec(tolerance=0.05, distribution="normal")
        s1 = MCSampler(seed=1).sample(1e-3, spec)
        s2 = MCSampler(seed=2).sample(1e-3, spec)
        assert s1 != s2

    def test_unknown_distribution_raises(self):
        import pytest

        from ltspice_mcp.lib.montecarlo import ToleranceSpec

        sampler = MCSampler(seed=0)
        # Deliberately bypass the Literal type to exercise the runtime
        # error path — the engine validates the distribution name even
        # though the static type system already constrains it.
        bad_spec = ToleranceSpec(tolerance=0.1, distribution="weibull")  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="Unknown distribution"):
            sampler.sample(1.0, bad_spec)


class TestExpandTolerances:
    def test_per_ref_override_wins_over_type(self):
        from ltspice_mcp.lib.montecarlo import expand_tolerances

        out = expand_tolerances(
            ["R1", "R2", "C1"],
            type_tolerances={"R": (0.05, "normal")},
            component_overrides={"R1": (0.01, "uniform")},
        )
        assert out["R1"].tolerance == 0.01
        assert out["R1"].distribution == "uniform"
        # R2 falls back to the type rule.
        assert out["R2"].tolerance == 0.05
        # C1 has no rule, so it's not in the map.
        assert "C1" not in out

    def test_unperturbable_prefixes_skipped(self):
        from ltspice_mcp.lib.montecarlo import expand_tolerances

        # Voltage sources, switches, etc. are excluded even if rules try.
        out = expand_tolerances(
            ["V1", "S1", "R1"],
            type_tolerances={"V": (0.05, "normal"), "S": (0.05, "normal"), "R": (0.05, "normal")},
            component_overrides={},
        )
        assert "R1" in out
        assert "V1" not in out
        assert "S1" not in out


class TestParseValue:
    def test_engineering_suffixes(self):
        import pytest

        from ltspice_mcp.lib.montecarlo import parse_value

        assert parse_value("1k") == pytest.approx(1e3)
        assert parse_value("100u") == pytest.approx(1e-4)
        assert parse_value("2.2n") == pytest.approx(2.2e-9)
        assert parse_value("10Meg") == pytest.approx(10e6)
        assert parse_value("1m") == pytest.approx(1e-3)
        assert parse_value("1") == pytest.approx(1.0)
        assert parse_value("1.5e-6") == pytest.approx(1.5e-6)

    def test_parametric_returns_none(self):
        from ltspice_mcp.lib.montecarlo import parse_value

        assert parse_value("{Rd}") is None
        assert parse_value("R*2") is None  # operator
        assert parse_value("table(...)") is None

    def test_invalid_returns_none(self):
        from ltspice_mcp.lib.montecarlo import parse_value

        assert parse_value("") is None
        assert parse_value("abc") is None


class TestSampleOffset:
    """``sample_offset`` returns additive deltas — the call site composes
    them with the nominal. Relative kind scales by |nominal|; absolute
    kind uses the raw tolerance as σ (or half-range)."""

    def test_relative_zero_nominal_yields_zero_delta(self):
        from ltspice_mcp.lib.montecarlo import ToleranceSpec

        sampler = MCSampler(seed=1)
        spec = ToleranceSpec(tolerance=0.10, kind="relative")
        # σ = |nominal| * tol / 3 = 0 → all samples are 0.
        samples = [sampler.sample_offset(0.0, spec) for _ in range(20)]
        assert all(s == 0.0 for s in samples)

    def test_absolute_kind_uses_raw_tolerance_as_3sigma(self):
        import statistics

        from ltspice_mcp.lib.montecarlo import ToleranceSpec

        sampler = MCSampler(seed=42)
        # 30 mV ± 3σ → σ = 10 mV. Sample many; check std ≈ 10 mV.
        spec = ToleranceSpec(tolerance=0.030, kind="absolute")
        samples = [sampler.sample_offset(0.7, spec) for _ in range(5000)]
        sigma_estimate = statistics.stdev(samples)
        assert 0.0085 < sigma_estimate < 0.0115  # within ~15% of σ=10mV

    def test_relative_kind_scales_by_nominal(self):
        import statistics

        from ltspice_mcp.lib.montecarlo import ToleranceSpec

        sampler = MCSampler(seed=7)
        spec = ToleranceSpec(tolerance=0.10, kind="relative")
        # σ = |1k| * 0.10 / 3 ≈ 33.3
        samples = [sampler.sample_offset(1000.0, spec) for _ in range(5000)]
        sigma = statistics.stdev(samples)
        assert 28.0 < sigma < 38.0


class TestModelPerturbationMath:
    def test_sample_model_perturbation_skips_missing_nominals(self):
        from ltspice_mcp.lib.montecarlo import (
            ToleranceSpec,
            sample_model_perturbation,
        )

        sampler = MCSampler(seed=0)
        out = sample_model_perturbation(
            sampler,
            "NMOS1",
            nominals={"VTO": 0.7},  # KP is missing
            tolerances={
                "VTO": ToleranceSpec(tolerance=0.05, kind="relative"),
                "KP": ToleranceSpec(tolerance=0.10, kind="relative"),  # not in nominals
            },
        )
        # Only VTO comes back; KP is skipped without raising.
        assert "VTO" in out
        assert "KP" not in out

    def test_sample_model_perturbation_accepts_lowercase_spec(self):
        from ltspice_mcp.lib.montecarlo import (
            ToleranceSpec,
            sample_model_perturbation,
        )

        # parse_model_params upper-cases nominal keys; a user spelling the
        # parameter lowercase must still get a perturbation (set_param is
        # case-insensitive on the write side, so the round trip holds).
        sampler = MCSampler(seed=0)
        out = sample_model_perturbation(
            sampler,
            "NMOS1",
            nominals={"VTO": 0.7},
            tolerances={"vto": ToleranceSpec(tolerance=0.05, kind="relative")},
        )
        # Presence is the regression (a case-sensitive lookup skipped it);
        # the value stays a plausible perturbation of the 0.7 nominal.
        assert out["vto"] == pytest.approx(0.7, abs=0.7 * 0.05 * 2)

    def test_sample_model_perturbation_adds_delta(self):
        from ltspice_mcp.lib.montecarlo import (
            ToleranceSpec,
            sample_model_perturbation,
        )

        sampler = MCSampler(seed=42)
        # Absolute σ_VTH = 12 mV → ±36 mV at 3σ tolerance
        out = sample_model_perturbation(
            sampler,
            "NMOS1",
            nominals={"VTO": 0.7, "KP": 100e-6},
            tolerances={
                "VTO": ToleranceSpec(tolerance=0.036, kind="absolute"),
                "KP": ToleranceSpec(tolerance=0.10, kind="relative"),
            },
        )
        # VTO perturbation should be in vicinity of 0.7 ± several σ
        assert 0.6 < out["VTO"] < 0.8
        assert 80e-6 < out["KP"] < 120e-6


class TestPerturbModelInText:
    def test_replaces_existing_param(self):
        from ltspice_mcp.lib.montecarlo import perturb_model_in_text

        text = (
            "* test\n"
            ".MODEL NMOS1 NMOS(VTO=0.7 KP=100u LAMBDA=0.02)\n"
            "M1 d g 0 0 NMOS1 W=10u L=1u\n"
            ".END\n"
        )
        out = perturb_model_in_text(text, "NMOS1", {"VTO": 0.715, "KP": 0.000105})
        # Old values must be gone
        assert "VTO=0.7\b" not in out
        assert "KP=100u" not in out
        # New values present
        assert "VTO=0.715" in out
        assert "KP=0.000105" in out

    def test_appends_missing_param_inside_paren(self):
        from ltspice_mcp.lib.montecarlo import perturb_model_in_text

        text = ".MODEL NMOS1 NMOS(VTO=0.7 KP=100u)\n"
        out = perturb_model_in_text(text, "NMOS1", {"LAMBDA": 0.025})
        assert "LAMBDA=0.025" in out
        # Closing paren still present and balanced.
        assert out.count("(") == out.count(")")

    def test_case_insensitive_match(self):
        from ltspice_mcp.lib.montecarlo import perturb_model_in_text

        text = ".model nmos1 nmos(vto=0.7)\n"
        out = perturb_model_in_text(text, "NMOS1", {"VTO": 0.65})
        assert "0.65" in out

    def test_continuation_lines_merged(self):
        from ltspice_mcp.lib.montecarlo import perturb_model_in_text

        text = ".MODEL NMOS1 NMOS(VTO=0.7\n+ KP=100u LAMBDA=0.02)\n.END\n"
        out = perturb_model_in_text(text, "NMOS1", {"VTO": 0.715})
        assert "0.715" in out

    def test_missing_model_raises(self):
        from ltspice_mcp.lib.montecarlo import perturb_model_in_text

        with pytest.raises(ValueError, match="not found"):
            perturb_model_in_text(".MODEL OTHER NPN(BF=200)\n", "NMOS1", {"VTO": 0.7})


class TestPelgromMismatch:
    def test_smaller_devices_have_larger_sigma(self):
        import statistics

        from ltspice_mcp.lib.montecarlo import (
            InstanceGeometry,
            sample_instance_mismatch,
        )

        rule = MismatchRule(prefix="M", avt=3e-3, ak=0.0)
        # Big device: W=L=10 µm → W·L=100 µm² → σ_VTH = 3mV/√100 = 300 µV
        big = InstanceGeometry("M1", "NMOS1", width_m=10e-6, length_m=10e-6)
        # Small device: W=L=0.5 µm → W·L=0.25 µm² → σ_VTH = 6 mV
        small = InstanceGeometry("M2", "NMOS1", width_m=0.5e-6, length_m=0.5e-6)

        sampler_big = MCSampler(seed=1)
        sampler_small = MCSampler(seed=2)
        big_samples = [
            sample_instance_mismatch(sampler_big, big, rule)["dvth"] for _ in range(2000)
        ]
        small_samples = [
            sample_instance_mismatch(sampler_small, small, rule)["dvth"] for _ in range(2000)
        ]
        sigma_big = statistics.stdev(big_samples)
        sigma_small = statistics.stdev(small_samples)
        # Theoretical ratio: σ_small/σ_big = √(WL_big / WL_small) = √(100/0.25) = 20
        ratio = sigma_small / sigma_big
        assert 15 < ratio < 25  # within ~25% of the analytical 20

    def test_disabled_when_coefficients_zero(self):
        from ltspice_mcp.lib.montecarlo import (
            InstanceGeometry,
            sample_instance_mismatch,
        )

        rule = MismatchRule(prefix="M", avt=0.0, ak=0.0)
        inst = InstanceGeometry("M1", "NMOS1", width_m=1e-6, length_m=1e-6)
        sampler = MCSampler(seed=0)
        out = sample_instance_mismatch(sampler, inst, rule)
        assert out["dvth"] == 0.0
        assert out["dk_over_k"] == 0.0


class TestVariantModelGeneration:
    def test_render_variant_renames_and_overrides(self):
        from ltspice_mcp.lib.montecarlo import render_variant_model_card

        base = ".MODEL NMOS1 NMOS(VTO=0.7 KP=100u LAMBDA=0.02)\n"
        variant = render_variant_model_card(base, "NMOS1__M1", {"VTO": 0.714, "KP": 0.000098})
        assert ".MODEL NMOS1__M1" in variant
        # Make sure the original NMOS1 token isn't left behind in the card
        assert ".MODEL NMOS1 " not in variant
        assert "VTO=0.714" in variant
        assert "KP=9.8e-05" in variant or "KP=0.0000980" in variant or "KP=9.8e-5" in variant

    def test_inject_card_before_end(self):
        from ltspice_mcp.lib.montecarlo import inject_card_before_end

        text = ".MODEL NMOS1 NMOS(VTO=0.7)\nM1 d g 0 0 NMOS1\n.END\n"
        out = inject_card_before_end(text, ".MODEL NMOS1__M1 NMOS(VTO=0.715)\n")
        assert ".MODEL NMOS1__M1" in out
        # Variant card must appear before .END
        end_idx = out.lower().rindex(".end")
        variant_idx = out.index("NMOS1__M1")
        assert variant_idx < end_idx

    def test_rewrite_instance_model_preserves_params(self):
        from ltspice_mcp.lib.montecarlo import rewrite_instance_model

        text = "M1 d g 0 0 NMOS1 W=10u L=1u m=2\n"
        out = rewrite_instance_model(text, "M1", "NMOS1__M1")
        assert "NMOS1__M1" in out
        # W= and L= preserved; the original model token is replaced not
        # duplicated.
        assert "W=10u" in out
        assert "L=1u" in out
        assert " NMOS1 " not in out

    def test_rewrite_instance_model_no_params(self):
        from ltspice_mcp.lib.montecarlo import rewrite_instance_model

        text = "Q1 c b e MYNPN\n"
        out = rewrite_instance_model(text, "Q1", "MYNPN__Q1")
        assert "Q1 c b e MYNPN__Q1" in out


class TestExtractMosfetInstances:
    def test_finds_W_L_geometry(self):
        from ltspice_mcp.lib.montecarlo import extract_mosfet_instances

        text = (
            "* test\n"
            ".MODEL NMOS1 NMOS(VTO=0.7)\n"
            "M1 d g 0 0 NMOS1 W=10u L=180n\n"
            "M2 d g 0 0 NMOS1 W=2u L=180n\n"
            ".END\n"
        )
        instances = extract_mosfet_instances(text)
        refs = {i.ref: i for i in instances}
        assert "M1" in refs and "M2" in refs
        assert refs["M1"].width_m == pytest.approx(10e-6)
        assert refs["M1"].length_m == pytest.approx(180e-9)
        assert refs["M2"].width_m == pytest.approx(2e-6)
        assert refs["M1"].model_name == "NMOS1"

    def test_skips_instances_without_geometry(self):
        from ltspice_mcp.lib.montecarlo import extract_mosfet_instances

        # No W= / L= → can't compute Pelgrom σ; skipped.
        text = "M1 d g 0 0 NMOS1\n"
        instances = extract_mosfet_instances(text)
        assert instances == []

    def test_finds_lowercase_w_l_geometry(self):
        from ltspice_mcp.lib.montecarlo import extract_mosfet_instances

        # ngspice decks conventionally write lowercase "w="/"l="; a
        # case-sensitive lookup silently skipped every instance, yielding a
        # zero-mismatch Monte Carlo that looked like a clean run.
        text = "* test\n.model nmos1 nmos(vto=0.7)\nm1 d g 0 0 nmos1 w=10u l=180n\n.end\n"
        instances = extract_mosfet_instances(text)
        assert len(instances) == 1
        assert instances[0].width_m == pytest.approx(10e-6)
        assert instances[0].length_m == pytest.approx(180e-9)
        assert instances[0].model_name == "nmos1"


class TestParamPerturbation:
    def test_perturb_param_replaces_value(self):
        from ltspice_mcp.lib.montecarlo import perturb_param_in_text

        text = "* test\n.PARAM vto_n=0.7\n.PARAM kp_n=100u\n.END\n"
        out = perturb_param_in_text(text, "vto_n", 0.715)
        assert ".PARAM vto_n=0.715" in out
        assert ".PARAM kp_n=100u" in out  # untouched

    def test_perturb_param_case_insensitive(self):
        from ltspice_mcp.lib.montecarlo import perturb_param_in_text

        text = ".param vto_n = 0.7\n"
        out = perturb_param_in_text(text, "VTO_N", 0.715)
        assert "0.715" in out

    def test_perturb_param_missing_raises(self):
        from ltspice_mcp.lib.montecarlo import perturb_param_in_text

        with pytest.raises(ValueError, match="not found"):
            perturb_param_in_text(".PARAM rd=1k\n", "vto_n", 0.7)

    def test_parse_param_nominal(self):
        from ltspice_mcp.lib.montecarlo import parse_param_nominal

        text = ".PARAM vto_n=0.7\n.PARAM kp_n=100u\n"
        assert parse_param_nominal(text, "vto_n") == pytest.approx(0.7)
        assert parse_param_nominal(text, "kp_n") == pytest.approx(100e-6)
        assert parse_param_nominal(text, "missing") is None


class TestMismatchRuleMatching:
    def test_finds_first_matching_prefix(self):
        from ltspice_mcp.lib.montecarlo import find_mismatch_rule

        rules = [
            MismatchRule(prefix="M", avt=3e-3, ak=0.02),
            MismatchRule(prefix="Q", avt=2e-3),
        ]
        m_rule = find_mismatch_rule("M1", rules)
        assert m_rule is not None
        assert m_rule.prefix == "M"

        q_rule = find_mismatch_rule("Q5", rules)
        assert q_rule is not None
        assert q_rule.prefix == "Q"

        assert find_mismatch_rule("R7", rules) is None


class TestStreamIsolation:
    """Per-stream RNGs in MCSampler — adding/removing a perturbation
    source mustn't shift other sources' samples. This is the property
    that makes regression-fixed-seed tests stable as the engine evolves."""

    def test_stream_keys_independent(self):

        sampler = MCSampler(seed=42)
        a1 = sampler.stream("A").gauss(0.0, 1.0)
        b1 = sampler.stream("B").gauss(0.0, 1.0)

        # Re-create with same seed, draw from B first then A — order doesn't
        # matter because each stream is a self-contained RNG keyed by name.
        sampler2 = MCSampler(seed=42)
        b2 = sampler2.stream("B").gauss(0.0, 1.0)
        a2 = sampler2.stream("A").gauss(0.0, 1.0)

        assert a1 == a2
        assert b1 == b2

    def test_default_stream_compat(self):
        """The default stream still works for legacy single-stream callers."""
        from ltspice_mcp.lib.montecarlo import ToleranceSpec

        s1 = MCSampler(seed=7)
        s2 = MCSampler(seed=7)
        spec = ToleranceSpec(tolerance=0.05)
        assert s1.sample(100.0, spec) == s2.sample(100.0, spec)

    def test_derive_yields_independent_child(self):
        """``derive(namespace)`` produces a child sampler whose streams
        are independent of the parent's, but reproducible from the parent
        seed + namespace."""

        parent = MCSampler(seed=99)
        child_a = parent.derive("run1")
        child_b = parent.derive("run1")  # same namespace → same samples
        assert child_a.stream("rcl:R1").gauss(0, 1) == child_b.stream("rcl:R1").gauss(0, 1)

        child_c = parent.derive("run2")
        # Different namespace → different stream output (>99% probability;
        # we just check inequality on a single draw, sufficient given seed).
        assert child_a.stream("rcl:R1").gauss(0, 1) != child_c.stream("rcl:R1").gauss(0, 1)

    def test_adding_stream_doesnt_shift_existing(self):
        """If a future engine version adds a new perturbation source, the
        existing sources' sample sequences must be unchanged."""

        # Old engine: only one stream "rcl:R1"
        old = MCSampler(seed=123)
        old_samples = [old.stream("rcl:R1").gauss(0, 1) for _ in range(5)]

        # New engine: adds a "model:NMOS1.VTO" stream. Sampling from the
        # new stream first must not shift "rcl:R1"'s subsequent draws.
        new = MCSampler(seed=123)
        _ = [new.stream("model:NMOS1.VTO").gauss(0, 1) for _ in range(3)]
        new_samples = [new.stream("rcl:R1").gauss(0, 1) for _ in range(5)]

        assert old_samples == new_samples


class TestTruncatedGaussian:
    """The ±tolerance bound is the user-promised ±3σ truncation. Without
    truncation, rare-but-real outliers produce nonsensical perturbed
    values (e.g. negative VTO) that don't reflect real silicon."""

    def test_normal_samples_stay_within_bound(self):
        from ltspice_mcp.lib.montecarlo import ToleranceSpec

        sampler = MCSampler(seed=1)
        spec = ToleranceSpec(tolerance=0.10, distribution="normal")
        # 10000 samples — at least one would fall outside ±3σ in the
        # untruncated distribution (~27 expected). With truncation, all
        # must satisfy |delta/value - 1| <= 0.10.
        for _ in range(10000):
            perturbed = sampler.sample(1.0, spec)
            assert abs(perturbed - 1.0) <= 0.10 + 1e-12  # within ±10% bound

    def test_offset_samples_stay_within_bound_absolute(self):
        from ltspice_mcp.lib.montecarlo import ToleranceSpec

        sampler = MCSampler(seed=2)
        spec = ToleranceSpec(tolerance=0.030, distribution="normal", kind="absolute")
        for _ in range(10000):
            delta = sampler.sample_offset(0.7, spec)
            assert abs(delta) <= 0.030 + 1e-12

    def test_offset_samples_stay_within_bound_relative(self):
        from ltspice_mcp.lib.montecarlo import ToleranceSpec

        sampler = MCSampler(seed=3)
        spec = ToleranceSpec(tolerance=0.10, distribution="normal", kind="relative")
        for _ in range(10000):
            delta = sampler.sample_offset(1000.0, spec)
            assert abs(delta) <= 100.0 + 1e-9  # |nominal| * tolerance

    def test_truncation_preserves_distribution_shape(self):
        """Truncation at ±3σ should leave the central distribution
        approximately Gaussian — std should still be close to the
        nominal σ (a tiny shrinkage from rejection at the tails)."""
        import statistics

        from ltspice_mcp.lib.montecarlo import ToleranceSpec

        sampler = MCSampler(seed=4)
        spec = ToleranceSpec(tolerance=0.10, distribution="normal")
        samples = [sampler.sample(100.0, spec) - 100.0 for _ in range(20000)]
        sigma_est = statistics.stdev(samples)
        # σ = nominal * tol / 3 = 100 * 0.10 / 3 = 3.333
        # Truncation at ±3σ shrinks σ by ~2-3% (analytical) — well within
        # the ±10% bound below.
        assert 3.0 < sigma_est < 3.5


class TestMCRunnerCardFlowIntegration:
    """Integration coverage for the per-run card-mutation hot path.

    ``execute_montecarlo`` is an async closure inside ``MonteCarloRunner``
    that's hard to unit-test directly. These tests exercise the same
    composition (lex → build lookup dicts → Phase 1/2/3 mutations →
    emit) the runner uses, so a regression in any of:

    - lookup-dict construction
    - per-key shifts after sequential setters
    - variant-card injection updating the model dict
    - emit pushing back to the right shape

    is caught here rather than only at simulation time.
    """

    def _baseline_netlist(self) -> str:
        return (
            "* MC integration test\n"
            ".PARAM Vdd=5\n"
            ".MODEL NMOS1 NMOS(VTO=0.7 KP=100u)\n"
            "M1 out gate 0 0 NMOS1 W=10u L=1u\n"
            "M2 out gate 0 0 NMOS1 W=20u L=1u\n"
            "R1 vdd out 1k\n"
            ".TRAN 1m\n"
            ".END\n"
        )

    def test_phase1_model_perturbation_mutates_card_in_place(self):
        from ltspice_mcp.lib.spice_lex import SpiceCard
        from ltspice_mcp.lib.spice_lex_views import ModelCard

        cards = lex(self._baseline_netlist()).cards
        model_by_name: dict[str, SpiceCard] = {
            c.name.lower(): c for c in cards if c.kind == "model" and c.name
        }
        # Phase 1: perturb VTO and KP on NMOS1.
        view = ModelCard.from_card(model_by_name["nmos1"])
        view.set_param("VTO", 0.715)
        view.set_param("KP", 95e-6)
        # The cached model card now reflects both edits — the second
        # set_param relied on _shift_cached_param_tokens to keep KP's
        # body_offset aligned after the VTO length change.
        from ltspice_mcp.lib.spice_lex import emit

        text = emit(cards)
        assert "VTO=0.715" in text
        assert "KP=9.5e-05" in text
        # The corruption signature (KP glued to VTO's value) must be absent.
        assert "VTO=0.715KP" not in text

    def test_phase2_variant_injection_updates_lookup(self):
        from ltspice_mcp.lib.montecarlo import (
            render_variant_model_card,
            variant_model_name,
        )
        from ltspice_mcp.lib.spice_lex import SpiceCard, emit
        from ltspice_mcp.lib.spice_lex_ops import inject_card_before_end
        from ltspice_mcp.lib.spice_lex_views import InstanceLine

        cards = lex(self._baseline_netlist()).cards
        model_by_name: dict[str, SpiceCard] = {
            c.name.lower(): c for c in cards if c.kind == "model" and c.name
        }
        instance_by_ref: dict[str, SpiceCard] = {
            c.name.lower(): c for c in cards if c.kind == "instance" and c.name
        }
        base = model_by_name["nmos1"]
        variant = variant_model_name("NMOS1", "M1")
        variant_text = render_variant_model_card("".join(base.raw_lines), variant, {"VTO": 0.715})
        new_card = inject_card_before_end(cards, variant_text)
        # The runner registers the new model in the lookup dict so a
        # subsequent Phase-2 instance referencing it could resolve.
        if new_card.name:
            model_by_name[new_card.name.lower()] = new_card
        assert variant.lower() in model_by_name
        # Rewrite M1's model token through the cached instance card.
        InstanceLine.from_card(instance_by_ref["m1"]).set_model(variant)

        out = emit(cards)
        assert variant in out
        # Variant card must land before the .END.
        assert out.index(variant) < out.lower().rindex(".end")
        # M1 line uses the variant; M2 still references the base model.
        for line in out.splitlines():
            if line.startswith("M1 "):
                assert variant in line
            elif line.startswith("M2 "):
                assert "NMOS1" in line and variant not in line

    def test_phase3_param_perturbation_mutates_param_card(self):
        from ltspice_mcp.lib.spice_lex import SpiceCard, emit
        from ltspice_mcp.lib.spice_lex_views import ParamCard

        cards = lex(self._baseline_netlist()).cards
        param_by_name: dict[str, SpiceCard] = {
            c.name.lower(): c for c in cards if c.kind == "param" and c.name
        }
        ParamCard.from_card(param_by_name["vdd"]).set_value(3.3)
        out = emit(cards)
        assert "Vdd=3.3" in out
        assert "Vdd=5" not in out

    def test_full_run_compose_phases_in_order(self):
        # Replicates execute_montecarlo's per-run flow: build lookup
        # dicts once after lex, apply all three phases, emit. Verifies
        # the composition produces a self-consistent netlist with all
        # mutations present.
        from ltspice_mcp.lib.montecarlo import (
            render_variant_model_card,
            variant_model_name,
        )
        from ltspice_mcp.lib.spice_lex import SpiceCard, emit
        from ltspice_mcp.lib.spice_lex_ops import inject_card_before_end
        from ltspice_mcp.lib.spice_lex_views import (
            InstanceLine,
            ModelCard,
            ParamCard,
        )

        cards = lex(self._baseline_netlist()).cards
        model_by_name: dict[str, SpiceCard] = {
            c.name.lower(): c for c in cards if c.kind == "model" and c.name
        }
        instance_by_ref: dict[str, SpiceCard] = {
            c.name.lower(): c for c in cards if c.kind == "instance" and c.name
        }
        param_by_name: dict[str, SpiceCard] = {
            c.name.lower(): c for c in cards if c.kind == "param" and c.name
        }

        # Phase 1
        ModelCard.from_card(model_by_name["nmos1"]).set_param("VTO", 0.71)

        # Phase 2 — variant for M1 only
        base = model_by_name["nmos1"]
        variant = variant_model_name("NMOS1", "M1")
        variant_text = render_variant_model_card("".join(base.raw_lines), variant, {"VTO": 0.72})
        new_card = inject_card_before_end(cards, variant_text)
        if new_card.name:
            model_by_name[new_card.name.lower()] = new_card
        InstanceLine.from_card(instance_by_ref["m1"]).set_model(variant)

        # Phase 3
        ParamCard.from_card(param_by_name["vdd"]).set_value(3.3)

        out = emit(cards)
        # All three phases visible.
        assert "VTO=0.71" in out  # Phase 1
        assert variant in out  # Phase 2 variant card
        assert "Vdd=3.3" in out  # Phase 3
        # Re-parse to confirm the result is structurally valid.
        re_cards = lex(out).cards
        models = [c.name for c in re_cards if c.kind == "model"]
        assert "NMOS1" in models
        assert variant in models


class TestHierarchicalMcDoesNotJoinSpiceCircuits:
    """The MC runner used to do ``"".join(editor.netlist)`` which
    crashed on hierarchical netlists where ``editor.netlist`` contains
    ``SpiceCircuit`` objects for ``.subckt`` blocks. The fix reads the
    netlist via ``read_spice_text`` and lexes once instead.
    """

    def test_hierarchical_netlist_lexes_via_read_spice_text(self, tmp_path: Path) -> None:
        from ltspice_mcp.lib.encoding import read_spice_text
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


class TestDiscardGeneratedNetlist:
    """``discard_generated_netlist`` deletes a generated per-job netlist copy
    (an '.options logopinfo' injection or an ngspice '.control' write
    injection — run in each batch runner's finally + the single-sim cleanup),
    and only a marked copy — never the user's own deck (no marker) or a None
    path."""

    def test_discards_logopinfo_marked_copy(self, work_dir: Path):
        copy = work_dir / ".n.sweep_x.logopinfo.cir"
        copy.write_text("* aug\n.op\n.options logopinfo\n.end\n")
        discard_generated_netlist(copy)
        assert not copy.exists()

    def test_discards_ngspice_control_write_marked_copy(self, work_dir: Path):
        copy = work_dir / ".n.sim_x.ctrlwrite.cir"
        copy.write_text("* aug\n.control\nrun\nwrite\n.endc\n.end\n")
        discard_generated_netlist(copy)
        assert not copy.exists()

    def test_keeps_user_netlist_without_marker(self, work_dir: Path):
        user = work_dir / "n.cir"
        user.write_text("* user\n.op\n.end\n")
        discard_generated_netlist(user)
        assert user.exists()

    def test_none_is_noop(self):
        discard_generated_netlist(None)  # must not raise
