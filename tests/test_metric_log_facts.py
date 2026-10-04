"""Metrics consume captured log facts without dependency decoding in the host."""

import pytest

from ltspice_mcp.lib import log_parser, metrics, services
from ltspice_mcp.lib.recipes import MeasurementsRecipe, OperatingPointRecipe, SummaryRecipe
from tests.conftest import stage_recorded_fixture
from tests.test_device_op_points import _LOG_WITH_BLOCK


@pytest.fixture
def refuse_parent_log_decoders(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("A metric invoked a dependency log decoder in the parent")

    monkeypatch.setattr(log_parser, "make_log_reader", forbidden)
    monkeypatch.setattr(log_parser, "read_device_op_points", forbidden)


@pytest.mark.asyncio
@pytest.mark.usefixtures("refuse_parent_log_decoders")
@pytest.mark.parametrize("recipe", ["summary", "measurements"])
async def test_recorded_measurements_reach_metrics_as_captured_facts(
    state_no_sim, work_dir, recipe
):
    path = stage_recorded_fixture(work_dir, "ltspice_tran_rc")
    source = services.source_for_raw_path(path, state_no_sim)
    if recipe == "summary":
        result = await metrics.summary(
            source, SummaryRecipe(key="summary", metric="summary"), 0, state_no_sim
        )
        assert result["measurements"]["vfinal"]["values"] == [0.999876166042]
        assert result["temp_c"] == 27.0
    else:
        result = await metrics.measurements(
            source,
            MeasurementsRecipe(key="measurements", metric="measurements"),
            0,
            state_no_sim,
        )
        assert result["stats"]["vfinal"]["mean"] == 0.999876166042


@pytest.mark.asyncio
@pytest.mark.usefixtures("refuse_parent_log_decoders")
async def test_device_bias_points_come_from_the_contained_log_decoder(state_no_sim, work_dir):
    # Synthetic OP payload; the log block carries independently asserted device facts.
    path = work_dir / "bias.raw"
    path.write_bytes(
        b"Title: synthetic bias\nPlotname: Operating Point\nFlags: real\n"
        b"No. Variables: 2\nNo. Points: 1\nCommand: LTspice\nVariables:\n"
        b"\t0\tV(out)\tvoltage\n\t1\tI(M1)\tdevice_current\nValues:\n0\t1.8\n\t9.6e-5\n"
    )
    path.with_suffix(".log").write_text(_LOG_WITH_BLOCK, encoding="utf-8")
    result = await metrics.operating_point(
        services.source_for_raw_path(path, state_no_sim),
        OperatingPointRecipe(key="bias", metric="operating_point"),
        0,
        state_no_sim,
    )
    assert result["device_op_points"]["@m1[gm]"] == 4.8e-4
    assert result["device_op_points"]["@m1[vth]"] == 0.5


@pytest.mark.asyncio
async def test_console_only_solve_failure_is_not_silently_dropped(state_no_sim, work_dir):
    path = work_dir / "partial.raw"
    path.write_bytes(b"Incomplete RAW")
    path.with_suffix(".exe.log").write_text("Error: timestep too small\n", encoding="utf-8")
    failures = await metrics.solve_failures(
        services.source_for_raw_path(path, state_no_sim), state_no_sim
    )
    assert failures == ["Error: timestep too small"]
