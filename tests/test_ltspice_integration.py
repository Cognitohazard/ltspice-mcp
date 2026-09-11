"""Integration tests requiring a real LTspice binary.

These tests are skipped unless LTspice integration is explicitly enabled and
LTspice is available on the system.
They exercise the full simulation pipeline: create netlist → run sim → parse results.
"""

import asyncio
import math
import os
import shutil
from pathlib import Path

import pytest
from mcp.types import TextContent, TextResourceContents

from ltspice_mcp.config import ServerConfig
from ltspice_mcp.lib.simulator import detect_simulators
from ltspice_mcp.state import SessionState
from tests.conftest import terminal_experiment

# Path to the test fixture .asc schematic
_FIXTURE_DIR = Path(__file__).parent / "fixtures"


def _text(contents) -> str:
    """Extract text from a resource contents entry, asserting it is text."""
    assert isinstance(contents, TextResourceContents)
    return contents.text


def _result_text(result) -> str:
    """Extract text from a tool result's first content block, asserting it is text."""
    item = result.content[0]
    assert isinstance(item, TextContent)
    return item.text


def _make_ltspice_state(work_dir: Path) -> SessionState | None:
    """Create a SessionState with LTspice detected, or return None."""
    config = ServerConfig(
        working_dir=work_dir,
        allowed_paths=[work_dir],
        log_level="DEBUG",
    )
    # Load real config to get simulator_exe for WSL
    real_config = ServerConfig.load()
    if real_config.simulator_exe:
        config = ServerConfig(
            simulator="ltspice",
            simulator_exe=real_config.simulator_exe,
            working_dir=work_dir,
            allowed_paths=[work_dir],
            log_level="DEBUG",
        )
    available = detect_simulators(config)
    if "ltspice" not in available:
        return None

    # Configure AscEditor library paths (same as server_lifespan does)
    from ltspice_mcp.engine import configure_asc_editor

    configure_asc_editor(config, available)

    return SessionState.create(config, available)


def _ltspice_available() -> bool:
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        return _make_ltspice_state(Path(td)) is not None


def _ltspice_integration_skip_reason() -> str | None:
    if os.environ.get("LTSPICE_MCP_RUN_LTSPICE_INTEGRATION") != "1":
        return "LTspice integration tests are opt-in; set LTSPICE_MCP_RUN_LTSPICE_INTEGRATION=1"
    if not _ltspice_available():
        return "LTspice not available on this system"
    return None


# Skip entire module unless real LTspice tests are explicitly requested.
_skip_reason = _ltspice_integration_skip_reason()
pytestmark = pytest.mark.skipif(_skip_reason is not None, reason=_skip_reason or "")


@pytest.fixture
def ltspice_state(work_dir: Path) -> SessionState:
    state = _make_ltspice_state(work_dir)
    assert state is not None
    return state


@pytest.fixture
def rc_netlist(work_dir: Path) -> Path:
    """Simple RC low-pass filter for AC analysis."""
    p = work_dir / "rc_filter.cir"
    p.write_text(
        "* RC Low-Pass Filter\n"
        "R1 in out 1k\n"
        "C1 out 0 100n\n"
        "V1 in 0 AC 1\n"
        ".ac dec 100 1 1Meg\n"
        ".meas AC fc WHEN mag(V(out))=0.707\n"
        ".END\n"
    )
    return p


@pytest.fixture
def tran_netlist(work_dir: Path) -> Path:
    """Simple RC circuit for transient analysis."""
    p = work_dir / "rc_tran.cir"
    p.write_text(
        "* RC Transient\n"
        "R1 in out 1k\n"
        "C1 out 0 100n\n"
        "V1 in 0 PULSE(0 1 0 1n 1n 0.5m 1m)\n"
        ".tran 0 5m 0 1u\n"
        ".meas TRAN vout_max MAX V(out)\n"
        ".meas TRAN rise_time TRIG V(out) VAL=0.1 RISE=1 TARG V(out) VAL=0.9 RISE=1\n"
        ".END\n"
    )
    return p


async def _run_deck(state: SessionState, request_id: str, netlist: str) -> dict:
    """Run one deck on the real LTspice binary and return its terminal receipt."""
    receipt = await terminal_experiment(
        state,
        {
            "request_id": request_id,
            "circuits": [{"path": netlist, "id": "dut"}],
            "execution": {"wait_s": 120, "simulator": "ltspice"},
        },
        wait_timeout_s=240,
    )
    assert receipt["status"] == "completed", receipt.get("failures")
    # The point of this tier: a real LTspice binary produced this run.
    assert receipt["source"][0]["simulator"].startswith("LTspice")
    return receipt


async def _recipe(state: SessionState, job_id: str, recipe: dict, **kw) -> dict:
    """One recipe over a completed job's runs."""
    from ltspice_mcp.tools.analyze import AnalyzeResultsInput, handle_analyze_results

    result = await handle_analyze_results(
        AnalyzeResultsInput.model_validate(
            {
                "sources": [{"job_id": job_id, "runs": "all", "label": "dut"}],
                "recipes": [recipe],
                **kw,
            }
        ),
        state,
    )
    data = result.structured_content
    assert data is not None, result.content[0].text
    return data


@pytest.mark.asyncio
class TestEndToEndSimulation:
    """Full pipeline: deck -> real LTspice -> artifacts on disk."""

    async def test_ac_simulation_completes(self, ltspice_state: SessionState, rc_netlist: Path):
        receipt = await _run_deck(ltspice_state, "lt-ac-completes", rc_netlist.name)
        assert receipt["completeness"]["produced"] == 1

    async def test_raw_and_log_files_produced(self, ltspice_state: SessionState, rc_netlist: Path):
        receipt = await _run_deck(ltspice_state, "lt-artifacts", rc_netlist.name)
        data = await _recipe(
            ltspice_state,
            receipt["job_id"],
            {"key": "sum", "metric": "summary"},
            include={"provenance": True},
        )
        (hashes,) = data["source_hashes"]
        assert Path(hashes["raw_path"]).exists()  # noqa: ASYNC240
        assert Path(hashes["log_path"]).exists()  # noqa: ASYNC240


@pytest.mark.asyncio
class TestJobTracking:
    """The job control plane over a real run."""

    async def test_status_reports_a_completed_job(
        self, ltspice_state: SessionState, rc_netlist: Path
    ):
        from ltspice_mcp.tools.jobs import (
            JobsInput,
            handle_jobs,
        )

        receipt = await _run_deck(ltspice_state, "lt-job-status", rc_netlist.name)
        result = await handle_jobs(
            JobsInput.model_validate({"action": "status", "job_id": receipt["job_id"]}),
            ltspice_state,
        )
        data = result.structured_content
        assert data is not None
        assert data["job_id"] == receipt["job_id"]
        assert data["status"] == "completed"

    async def test_request_id_resolves_the_same_job(
        self, ltspice_state: SessionState, rc_netlist: Path
    ):
        # A durable request_id is the client's re-entry point after a dropped
        # connection: it must resolve to the job the first call created.
        from ltspice_mcp.tools.jobs import (
            JobsInput,
            handle_jobs,
        )

        receipt = await _run_deck(ltspice_state, "lt-request-id", rc_netlist.name)
        result = await handle_jobs(
            JobsInput.model_validate({"action": "status", "request_id": "lt-request-id"}),
            ltspice_state,
        )
        data = result.structured_content
        assert data is not None
        assert data["job_id"] == receipt["job_id"]


@pytest.mark.asyncio
class TestMeasExtraction:
    """.MEAS results — the reason a WSL run is routed to a Windows-native
    output dir: LTspice cannot write the SQLite .db behind .MEAS over UNC."""

    async def test_ac_measurement_extracted(self, ltspice_state: SessionState, rc_netlist: Path):
        receipt = await _run_deck(ltspice_state, "lt-meas-ac", rc_netlist.name)
        data = await _recipe(
            ltspice_state,
            receipt["job_id"],
            {"key": "meas", "metric": "measurements"},
            include={"per_run": {"limit": 1}},
        )
        stats = data["results"]["meas"]["per_run"]["items"][0]["value"]["stats"]
        fc = stats["fc"]
        assert fc["valid_count"] == 1
        assert fc.get("at", fc["mean"]) == pytest.approx(
            1 / (2 * math.pi * 1e3 * 100e-9), rel=0.005
        )

    async def test_transient_measurement_extracted(
        self, ltspice_state: SessionState, tran_netlist: Path
    ):
        receipt = await _run_deck(ltspice_state, "lt-meas-tran", tran_netlist.name)
        data = await _recipe(
            ltspice_state,
            receipt["job_id"],
            {"key": "meas", "metric": "measurements"},
            include={"per_run": {"limit": 1}},
        )
        stats = data["results"]["meas"]["per_run"]["items"][0]["value"]["stats"]
        assert stats["vout_max"]["valid_count"] == 1
        assert stats["vout_max"]["mean"] == pytest.approx(1 - math.exp(-5), abs=0.002)


@pytest.mark.asyncio
class TestACAnalysis:
    """AC signal listing and summary off a real LTspice raw."""

    async def test_signals_listed(self, ltspice_state: SessionState, rc_netlist: Path):
        receipt = await _run_deck(ltspice_state, "lt-ac-signals", rc_netlist.name)
        data = await _recipe(
            ltspice_state,
            receipt["job_id"],
            {"key": "sum", "metric": "summary"},
            include={"signals_available": True},
        )
        names = [n.lower() for names in data["signals_available"].values() for n in names]
        assert any(n == "v(out)" for n in names), names

    async def test_summary_reports_the_ac_sim_type(
        self, ltspice_state: SessionState, rc_netlist: Path
    ):
        receipt = await _run_deck(ltspice_state, "lt-ac-summary", rc_netlist.name)
        data = await _recipe(ltspice_state, receipt["job_id"], {"key": "sum", "metric": "summary"})
        summary = data["results"]["sum"]["values"][0]["value"]
        assert "ac" in summary["sim_type"].lower()


@pytest.mark.asyncio
class TestWSLPathConversion:
    """On WSL the deck path crosses into a Windows process; a completed run is
    the proof the conversion worked."""

    async def test_wsl_path_in_simulation(self, ltspice_state: SessionState, rc_netlist: Path):
        from ltspice_mcp.lib.wsl import is_wsl

        if not is_wsl():
            pytest.skip("Not running in WSL")

        receipt = await _run_deck(ltspice_state, "lt-wsl-path", rc_netlist.name)
        assert receipt["completeness"]["produced"] == 1


@pytest.mark.asyncio
class TestResourcesWithResults:
    """Resource browsing after a real simulation produced results."""

    async def test_netlists_resource_lists_files(
        self, ltspice_state: SessionState, rc_netlist: Path
    ):
        from ltspice_mcp.resources import handle_read_resource

        result = handle_read_resource("spice://netlists/", ltspice_state)
        assert "rc_filter.cir" in _text(result.contents[0])


@pytest.mark.asyncio
class TestManagedExport:
    """The real LTspice exporter turning a .asc into a netlist."""

    @pytest.fixture
    def asc_in_workdir(self, work_dir: Path) -> Path:
        src = _FIXTURE_DIR / "Draft1.asc"
        dst = work_dir / "Draft1.asc"
        shutil.copy2(src, dst)
        return dst

    async def _export(self, state: SessionState, path: Path, **kw) -> dict:
        from ltspice_mcp.tools.verify import VerifyCircuitInput, handle_verify_circuit

        result = await handle_verify_circuit(
            VerifyCircuitInput.model_validate({"path": path.name, "checks": ["export"], **kw}),
            state,
        )
        data = result.structured_content
        assert data is not None, result.content[0].text
        return data

    async def test_export_produces_a_netlist(
        self, ltspice_state: SessionState, asc_in_workdir: Path
    ):
        data = await self._export(ltspice_state, asc_in_workdir)
        export = data["export"]
        assert export["ok"] is True, export
        netlist = await asyncio.to_thread(Path(export["netlist"]).read_text, encoding="utf-8")
        assert "R1" in netlist

    async def test_sidecar_export_writes_the_net_file(
        self, ltspice_state: SessionState, asc_in_workdir: Path
    ):
        await self._export(ltspice_state, asc_in_workdir, export_to="sidecar")
        net_file = asc_in_workdir.with_suffix(".net")
        assert net_file.exists(), f"Expected {net_file} to exist after export"


# The live-LTspice wait: WSL interop and .asc export make these runs slower
# than the ngspice tier, so the shared helper gets a longer jobs(wait).
async def _run_experiment(state: SessionState, payload: dict) -> dict:
    return await terminal_experiment(state, payload, wait_timeout_s=240)


@pytest.mark.asyncio
class TestSweepIntegration:
    """Full pipeline: run_experiments fan-out → attached analysis → results."""

    async def test_sweep_runs_every_case(self, ltspice_state: SessionState, rc_netlist: Path):
        receipt = await _run_experiment(
            ltspice_state,
            {
                "request_id": "lt-sweep-cases",
                "circuits": [{"path": rc_netlist.name, "id": "rc"}],
                "variations": [{"kind": "assign", "assign": {"R1": ["500", "1k", "1.5k"]}}],
                "execution": {"wait_s": 120, "simulator": "ltspice"},
            },
        )
        assert receipt["status"] == "completed", receipt.get("failures")
        # The point of this tier: a real LTspice binary produced these runs.
        assert receipt["source"][0]["simulator"].startswith("LTspice")
        counts = receipt["completeness"]
        assert counts["expanded"] == counts["produced"] == 3
        assert counts["failed"] == counts["cancelled"] == counts["skipped"] == 0
        assert {run["assignments"]["R1"] for run in receipt["runs"]["items"]} == {
            "500",
            "1k",
            "1.5k",
        }

    async def test_sweep_results_carry_the_moving_corner(
        self, ltspice_state: SessionState, rc_netlist: Path
    ):
        """The attached analysis reads every run's real corner frequency, and
        the extremes are attributed to the case that produced them: a bigger R1
        gives a lower fc for a fixed C."""
        receipt = await _run_experiment(
            ltspice_state,
            {
                "request_id": "lt-sweep-corner",
                "circuits": [{"path": rc_netlist.name, "id": "rc"}],
                "variations": [{"kind": "assign", "assign": {"R1": ["800", "1.2k"]}}],
                "execution": {"wait_s": 120, "simulator": "ltspice"},
                "analyze": {
                    "recipes": [
                        {
                            "key": "corner",
                            "metric": "bode_filter",
                            "signal": "V(out)",
                            "reduce": ["min", "max"],
                            "field": "cutoff_high_hz",
                        }
                    ],
                    "include": {"per_run": {"limit": 10}},
                },
            },
        )
        assert receipt["status"] == "completed", receipt.get("failures")
        stage = receipt["analysis"]
        assert stage["error"] is None, stage["error"]
        entry = stage["result"]["results"]["corner"]

        reduced = {item["stat"]: item for item in entry["reduced"]}
        # 1/(2*pi*R*100n): 800 ohm -> ~1990 Hz, 1.2k -> ~1326 Hz.
        assert reduced["max"]["assignments"]["R1"] == "800"
        assert reduced["min"]["assignments"]["R1"] == "1.2k"
        assert reduced["max"]["value"] == pytest.approx(1990, rel=0.1)
        assert reduced["min"]["value"] == pytest.approx(1326, rel=0.1)
        assert len(entry["per_run"]["items"]) == 2


@pytest.mark.asyncio
class TestMonteCarloIntegration:
    """Full pipeline: random variation → runs → aggregated analysis."""

    async def test_montecarlo_runs_stay_inside_the_tolerance_band(
        self, ltspice_state: SessionState, rc_netlist: Path
    ):
        receipt = await _run_experiment(
            ltspice_state,
            {
                "request_id": "lt-mc-rc",
                "circuits": [{"path": rc_netlist.name, "id": "rc"}],
                "variations": [
                    {
                        "kind": "random",
                        "runs": 3,
                        "seed": 7,
                        "rules": [
                            {
                                "rule": "component",
                                "target": "R1",
                                "tolerance": 0.05,
                                "distribution": "uniform",
                            }
                        ],
                    }
                ],
                "execution": {"wait_s": 120, "simulator": "ltspice"},
                "analyze": {
                    "recipes": [
                        {
                            "key": "corner",
                            "metric": "bode_filter",
                            "signal": "V(out)",
                            "reduce": ["min", "max"],
                            "field": "cutoff_high_hz",
                        }
                    ]
                },
            },
        )
        assert receipt["status"] == "completed", receipt.get("failures")
        counts = receipt["completeness"]
        assert counts["expanded"] == counts["produced"] == 3
        assert counts["failed"] == 0

        stage = receipt["analysis"]
        assert stage["error"] is None, stage["error"]
        reduced = {item["stat"]: item for item in stage["result"]["results"]["corner"]["reduced"]}
        # R1 perturbed uniformly by +/-5% around 1k, C1 fixed at 100n: every
        # sampled resistance and every corner frequency stays in its band.
        for item in reduced.values():
            assert 950.0 <= item["assignments"]["random:component:R1"] <= 1050.0
            assert 1516 <= item["value"] <= 1676  # 1/(2*pi*R*100n) over that band
