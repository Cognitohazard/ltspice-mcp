"""Integration tests requiring a real LTspice binary.

These tests are skipped unless LTspice integration is explicitly enabled and
LTspice is available on the system.
They exercise the full simulation pipeline: create netlist → run sim → parse results.
"""

import asyncio
import math
import os
import shutil
import sys
import tempfile
import threading
from pathlib import Path

import psutil
import pytest
from mcp.types import TextContent, TextResourceContents

from ltspice_mcp.config import ServerConfig
from ltspice_mcp.lib import hidden_desktop
from ltspice_mcp.lib.hidden_desktop import DialogError
from ltspice_mcp.lib.simulator import SIMULATORS, bind_named_executable, detect_simulators
from ltspice_mcp.state import SessionState
from tests.conftest import LIVENESS_S, terminal_experiment
from tests.test_hidden_desktop import windows_only

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
        value = data["results"]["meas"]["per_run"]["items"][0]["value"]
        assert "failed_measurements" not in value
        assert value.get("at", {}).get("fc", value["measured"]["fc"]) == pytest.approx(
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
        value = data["results"]["meas"]["per_run"]["items"][0]["value"]
        assert "failed_measurements" not in value
        assert value["measured"]["vout_max"] == pytest.approx(1 - math.exp(-5), abs=0.002)


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

    @pytest.mark.parametrize(
        "stem", sorted(p.stem for p in (_FIXTURE_DIR / "t_junctions").glob("*.asc"))
    )
    async def test_t_junction_sheets_still_export_as_recorded(
        self, ltspice_state: SessionState, work_dir: Path, stem: str
    ):
        """The installed LTspice connects each T-junction sheet the way the
        recorded export says, so the connectivity rule the editor and the net
        trace rely on still holds for this LTspice version."""
        cases = _FIXTURE_DIR / "t_junctions"
        shutil.copy2(cases / f"{stem}.asc", work_dir / f"{stem}.asc")
        # The recorded exports were made with the fixture resistor beside them.
        shutil.copy2(_FIXTURE_DIR / "symbols" / "res.asy", work_dir / "res.asy")
        data = await self._export(ltspice_state, work_dir / f"{stem}.asc")
        assert data["export"]["ok"] is True, data["export"]

        def cards(text: str) -> list[str]:
            return [line for line in text.splitlines() if line and not line.startswith("*")]

        live = await asyncio.to_thread(Path(data["export"]["netlist"]).read_text, encoding="utf-8")
        recorded = (cases / f"{stem}.net").read_text(encoding="utf-8")
        assert cards(live) == cards(recorded)


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


# --------------------------------------------------------------------------
# LTspice's window stays off the desktop someone is working at
# --------------------------------------------------------------------------


class _WindowWatch:
    """Where the windows of this test's LTspice are, sampled while it runs.

    LTspice is told apart from any other on the machine by ``marker``, a
    directory only this test's command lines name. ``here`` is set when one of
    its windows was on the desktop the test runs on, ``foreground`` when it was
    the foreground window there, and ``hidden`` when one was on the server's
    own desktop: the last shows the watch saw the windows at all.
    """

    def __init__(self, marker: Path) -> None:
        self._marker = marker.as_posix().casefold()
        self._known: dict[int, bool] = {}
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._watch, name="window-watch")
        self.here = self.foreground = self.hidden = False

    def __enter__(self) -> "_WindowWatch":
        self._thread.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self._stop.set()
        self._thread.join()

    def _ours(self, pid: int) -> bool:
        if pid not in self._known:
            try:
                command = " ".join(psutil.Process(pid).cmdline())
            except psutil.Error:
                return False  # gone, or not ours to read: ask again next time
            self._known[pid] = self._marker in command.replace("\\", "/").casefold()
        return self._known[pid]

    def _watch(self) -> None:
        if sys.platform != "win32":
            return
        import ctypes
        from ctypes import wintypes

        user = ctypes.WinDLL("user32")
        listed = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
        user.EnumWindows.argtypes = [listed, wintypes.LPARAM]
        user.GetForegroundWindow.restype = wintypes.HWND
        user.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]

        def owner(window: int | None) -> int:
            pid = wintypes.DWORD(0)
            if window:
                user.GetWindowThreadProcessId(window, ctypes.byref(pid))
            return int(pid.value)

        while True:
            on_this_desktop: set[int] = set()

            def note(window: int, _unused: int, found: set[int] = on_this_desktop) -> bool:
                found.add(owner(window))
                return True

            user.EnumWindows(listed(note), 0)
            ours_here = {pid for pid in on_this_desktop if self._ours(pid)}
            desktop = hidden_desktop.shared()
            ours_hidden = {
                pid for pid in (desktop.window_owners() if desktop else ()) if self._ours(pid)
            }
            self.here = self.here or bool(ours_here)
            self.hidden = self.hidden or bool(ours_hidden)
            front = owner(user.GetForegroundWindow())
            self.foreground = self.foreground or (front != 0 and self._ours(front))
            # timing: a sampler; nothing signals that a window opened or closed
            if self._stop.wait(0.002):
                return


def _recovery_run(simulator: type, work_dir: Path) -> _WindowWatch:
    """One run through the audited recovery launch, watched; skips where that
    launch cannot run here."""
    from ltspice_mcp.lib import controlled_ltspice as controlled
    from ltspice_mcp.lib.recovery_records import ExecutionRecord, RecoveryError, StartupPolicy
    from ltspice_mcp.lib.simulator_build import executable_identity

    identity = executable_identity(simulator)
    if identity is None or identity.sha256 != controlled.AUDITED_EXECUTABLE_SHA256:
        pytest.skip("the installed LTspice is not the build the recovery launch was audited on")
    root = work_dir.resolve() / "lineage"
    try:
        template = controlled.capture_ini_template(
            Path(os.environ["APPDATA"]) / "LTspice.ini", root / "startup" / "template.ini", root
        )
    except (KeyError, RecoveryError) as exc:
        pytest.skip(f"no established LTspice profile to recover with: {exc}")
    execution = ExecutionRecord(
        run_timeout_s=None,
        timeout_source="test",
        max_parallel=1,
        job_deadline_s=None,
        kill_grace_s=LIVENESS_S,
        simulator_argv=(identity.path,),
        executable=identity,
        ngbehavior=None,
        platform="win32",
        startup=StartupPolicy(controlled.STARTUP_VERSION, False, ini_template=template),
    )
    ini = root / "startup" / "attempt" / "LTspice.ini"
    controlled.prepare_attempt_ini(template, ini, root)
    adapter = controlled.controlled_ltspice(execution, ini, lambda: None)
    deck = work_dir.resolve() / "recovery_run.cir"
    deck.write_text(
        "* long enough to be seen\n"
        "R1 in out 1k\n"
        "C1 out 0 100n\n"
        "V1 in 0 PULSE(0 1 0 1n 1n 0.5m 1m)\n"
        ".tran 0 300m 0 1u\n"
        ".END\n"
    )
    with _WindowWatch(work_dir) as watch:
        assert adapter.run(deck, timeout=LIVENESS_S, cwd=deck.parent) == 0
    assert deck.with_suffix(".raw").is_file()
    return watch


@windows_only
@pytest.mark.asyncio
class TestWindowStaysOffTheDesktop:
    """LTspice opens a window for every run and export and takes the keyboard
    focus for as long as it lasts. Started on the server's own desktop it has
    no window where the person is, so it cannot."""

    def _assert_kept_away(self, watch: _WindowWatch) -> None:
        assert watch.here or watch.hidden, "the watch saw no LTspice window: it proves nothing"
        assert watch.hidden, "LTspice had no window on the server's desktop"
        assert not watch.here, "LTspice had a window on the desktop the test runs on"
        assert not watch.foreground, "LTspice was the foreground window"

    async def test_a_run_has_no_window_here(self, ltspice_state: SessionState, work_dir: Path):
        deck = work_dir / "long_tran.cir"
        deck.write_text(
            "* long enough to be seen\n"
            "R1 in out 1k\n"
            "C1 out 0 100n\n"
            "V1 in 0 PULSE(0 1 0 1n 1n 0.5m 1m)\n"
            ".tran 0 300m 0 1u\n"
            ".END\n"
        )
        with _WindowWatch(work_dir) as watch:
            await _run_deck(ltspice_state, "off-desktop-run", deck.name)
        self._assert_kept_away(watch)

    async def test_a_recovery_run_has_no_window_here(
        self, ltspice_state: SessionState, work_dir: Path
    ):
        """The audited recovery launch (``lib/controlled_ltspice.py``) starts
        the audited build with a copy of an established profile. It too runs
        on the server's desktop. Runs only where the installed LTspice is that
        build and the profile in ``%APPDATA%`` is an established one."""
        watch = await asyncio.to_thread(
            _recovery_run, ltspice_state.available_simulators["ltspice"], work_dir
        )
        self._assert_kept_away(watch)

    async def test_an_export_has_no_window_here(self, ltspice_state: SessionState, work_dir: Path):
        from ltspice_mcp.tools.verify import VerifyCircuitInput, handle_verify_circuit

        sheet = work_dir / "Draft1.asc"
        shutil.copy2(_FIXTURE_DIR / "Draft1.asc", sheet)
        with _WindowWatch(work_dir) as watch:
            result = await handle_verify_circuit(
                VerifyCircuitInput.model_validate(
                    {"path": sheet.name, "checks": ["export"], "export_to": "sidecar"}
                ),
                ltspice_state,
            )
        data = result.structured_content
        assert data is not None
        assert data["export"]["ok"] is True, data["export"]
        self._assert_kept_away(watch)


@windows_only
@pytest.mark.asyncio
class TestMessageBoxWhereNobodyCanAnswer:
    """On a desktop nobody sees, a message box would hold LTspice until the
    timeout and say nothing. The launch ends it and reports the box."""

    async def test_xvii_refusing_a_sheet_ends_the_export_with_what_it_said(self, work_dir: Path):
        """LTspice XVII answers a sheet that starts with a byte order mark with
        a box and waits for OK, which is recorded as ``export/micro_utf8_bom``.
        The export raises what that box said, as it was recorded."""
        from tests import _ltspice_recorded as recorded

        case_id = "export/micro_utf8_bom"
        xvii = recorded.installed_counterpart("ltspice17")
        if isinstance(xvii, str):
            pytest.skip(xvii)
        source = recorded.CASES.case(case_id).source
        sheet = work_dir / Path(source).name
        shutil.copy2(recorded.INPUTS / source, sheet)
        simulator = bind_named_executable(SIMULATORS["ltspice"], "ltspice:xvii", xvii.exe)

        with pytest.raises(DialogError) as stopped:
            await asyncio.to_thread(simulator.create_netlist, sheet, timeout=120)

        assert stopped.value.text == recorded.entry("ltspice17", case_id)["dialog"]
        assert not sheet.with_suffix(".net").exists()


def _open_window_skip_reason() -> str | None:
    if sys.platform != "win32":
        return "an LTspice window is reached through a Windows program"
    from ltspice_mcp.lib.ltspice_bridge import bridge_command
    from ltspice_mcp.lib.simulator_build import executable_path

    with tempfile.TemporaryDirectory() as scratch:
        state = _make_ltspice_state(Path(scratch))
    exe = executable_path(state.default_simulator) if state is not None else None
    if exe is None or bridge_command(exe) is None:
        return "the detected LTspice has no ltspice-mcp-bridge.exe beside it (26.1 or later)"
    return None


# Asked only where the tier runs at all: the module's own mark skips the rest.
_window_skip_reason = _open_window_skip_reason() if _skip_reason is None else None


def _bridge_of(state: SessionState) -> list[str]:
    """The command of the bridge beside the LTspice ``state`` runs."""
    from ltspice_mcp.lib.ltspice_bridge import bridge_command
    from ltspice_mcp.lib.simulator_build import executable_path

    exe = executable_path(state.default_simulator)
    assert exe is not None
    command = bridge_command(exe)
    assert command is not None
    return command


@pytest.mark.skipif(_window_skip_reason is not None, reason=_window_skip_reason or "")
class TestSheetOpenInAWindow:
    """The stand-in bridge the suite runs is held to a recording; this is the
    recording's subject itself: a real window, the real bridge, a real edit."""

    @pytest.fixture
    def open_sheet(self, ltspice_state: SessionState, work_dir: Path):
        """A sheet in the sandbox, open in an LTspice window nobody can see."""
        from ltspice_mcp.lib.ltspice_bridge import bridge_command
        from ltspice_mcp.lib.simulator_build import executable_path
        from tests.ltspice_bridge_recorder import wait_for_window
        from tests.ltspice_recorder import identify_build, neutral_settings

        exe = executable_path(ltspice_state.default_simulator)
        assert exe is not None
        settings = identify_build(Path(exe)).settings_file
        if settings is None:
            pytest.skip("this LTspice has no settings file yet; start it once")
        sheet = work_dir / "Draft1.asc"
        shutil.copyfile(_FIXTURE_DIR / "Draft1.asc", sheet)
        ini = work_dir / settings.name
        ini.write_bytes(neutral_settings(settings.read_bytes(), {}))
        command = bridge_command(exe)
        assert command is not None
        # A desktop of the window's own, not the server's: the bridge then
        # reaches a window on a desktop it is not on, as it does a person's.
        with hidden_desktop.HiddenDesktop(f"ltspice-mcp-window-{os.getpid()}") as desktop:
            if not desktop.available:
                pytest.skip("Windows gave no desktop to keep the window off this one")
            with desktop.start([exe, str(sheet), "-ini", str(ini)]) as window:
                wait_for_window(command, window.pid, sheet)
                yield sheet, window.pid

    @staticmethod
    async def _edit(state: SessionState, sheet: Path, value: str) -> dict:
        import hashlib

        from ltspice_mcp.tools.schematic_edit import EditSchematicInput, handle_edit_schematic

        digest = await asyncio.to_thread(lambda: hashlib.sha256(sheet.read_bytes()).hexdigest())
        result = await handle_edit_schematic(
            EditSchematicInput.model_validate(
                {
                    "target": str(sheet),
                    "expected_sha256": digest,
                    "ops": [{"op": "set_component_value", "reference": "R1", "value": value}],
                }
            ),
            state,
        )
        assert result.structured_content is not None
        return result.structured_content

    async def test_an_edit_is_shown_in_the_window_and_a_changed_window_stops_the_next(
        self, ltspice_state: SessionState, open_sheet
    ):
        from ltspice_mcp.lib.ltspice_window import content_difference

        sheet, pid = open_sheet
        windows = ltspice_state.open_windows
        assert windows.available, windows.unavailable

        data = await self._edit(ltspice_state, sheet, "2.2k")

        assert data["commit_state"] == "committed"
        assert data["observations"] == []
        assert [(row["pid"], row["shown"]) for row in data["open_in_ltspice"]] == [(pid, True)]
        (held,) = await asyncio.to_thread(windows.holding, sheet)
        assert "SYMATTR Value 2.2k" in held.text
        committed = await asyncio.to_thread(sheet.read_bytes)
        assert content_difference(committed.decode("cp1252"), held.text) is None

        # Someone changes the sheet in the window and does not save it.
        changed = held.text.replace("SYMATTR Value 2.2k", "SYMATTR Value 9k")
        await asyncio.to_thread(windows.show, held, changed)

        refused = await self._edit(ltspice_state, sheet, "4.7k")

        assert refused["commit_state"] == "not_committed"
        assert refused["error"]["code"] == "open_window_differs"
        assert await asyncio.to_thread(sheet.read_bytes) == committed
        (still,) = await asyncio.to_thread(windows.holding, sheet)
        assert "SYMATTR Value 9k" in still.text

    async def test_inspect_lists_the_sheet_the_window_has_in_front(
        self, ltspice_state: SessionState, open_sheet
    ):
        import hashlib

        from ltspice_mcp.tools.inspect_tools import InspectInput, handle_inspect

        sheet, pid = open_sheet

        async def ask() -> dict:
            result = await handle_inspect(
                InspectInput.model_validate({"queries": [{"kind": "open_in_ltspice"}]}),
                ltspice_state,
            )
            assert result.structured_content is not None
            (item,) = result.structured_content["results"]
            assert item["ok"] is True, item
            (design,) = [row for row in item["data"]["designs"] if row["pid"] == pid]
            return design

        design = await ask()
        on_disk = await asyncio.to_thread(sheet.read_bytes)
        assert Path(design["path"]) == sheet
        assert (design["kind"], design["active"], design["in_sandbox"]) == (
            "schematic",
            True,
            True,
        )
        assert design["sha256"] == hashlib.sha256(on_disk).hexdigest()
        assert design["differs_from_file"] is False

        # Someone changes the sheet in the window and does not save it.
        windows = ltspice_state.open_windows
        (held,) = await asyncio.to_thread(windows.holding, sheet)
        await asyncio.to_thread(
            windows.show, held, held.text.replace("SYMATTR Value 1k", "SYMATTR Value 9k")
        )

        design = await ask()
        assert design["differs_from_file"] is True
        assert "SYMBOL res" in design["difference"]
        assert design["sha256"] == hashlib.sha256(on_disk).hexdigest()

    async def test_the_reference_documents_are_the_ones_ltspice_itself_lists(
        self, ltspice_state: SessionState, open_sheet
    ):
        """LTspice's own server lists its reference documents. Read from the
        install, they are the same documents. The summaries differ: LTspice
        takes its one from the index page's table, and each document's own
        front matter, which is what is read here, says more."""
        from ltspice_mcp.lib.ltspice_bridge import BridgeSession
        from ltspice_mcp.tools.inspect_tools import InspectInput, handle_inspect

        _sheet, pid = open_sheet
        command = _bridge_of(ltspice_state)

        def as_ltspice_lists_them() -> dict[str, str]:
            with BridgeSession(command) as session:
                session.attach(pid)
                listed = session.call("list_ltspice_reference_docs")["docs"]
            return {doc["docname"]: doc["summary"] for doc in listed}

        theirs = await asyncio.to_thread(as_ltspice_lists_them)
        result = await handle_inspect(
            InspectInput.model_validate({"queries": [{"kind": "simulator_docs"}]}), ltspice_state
        )
        assert result.structured_content is not None
        (item,) = result.structured_content["results"]
        assert item["ok"] is True, item
        ours = {doc["name"]: doc["description"] for doc in item["data"]["docs"]}

        assert theirs, "LTspice listed no reference documents"
        assert set(theirs) <= set(ours)
        # What the install holds and LTspice does not list is its index page.
        assert set(ours) - set(theirs) <= {"README.md"}
        assert all(ours[name] for name in theirs), "a listed document has no description"

        (document,) = [
            (
                await handle_inspect(
                    InspectInput.model_validate(
                        {"queries": [{"kind": "simulator_docs", "name": "MEAS-REFERENCE.md"}]}
                    ),
                    ltspice_state,
                )
            ).structured_content["results"][0]
        ]
        assert document["ok"] is True, document
        assert document["data"]["sections"], "the document came back with no sections"

    async def test_a_finished_run_is_opened_in_the_window_with_its_traces_named(
        self, ltspice_state: SessionState, open_sheet, tran_netlist: Path
    ):
        """A job this server ran, shown in the window: the plot settings are
        written beside the run's results, and LTspice then has those results
        in front. That it draws the traces was looked at and cannot be asked."""
        from ltspice_mcp.lib.ltspice_bridge import BridgeSession
        from ltspice_mcp.lib.plot_settings import read_plot_settings
        from ltspice_mcp.tools.analysis import PlotWaveformInput, handle_plot_waveform

        _sheet, pid = open_sheet
        receipt = await _run_deck(ltspice_state, "shown-in-ltspice", str(tran_netlist))
        result = await handle_plot_waveform(
            PlotWaveformInput(job_id=receipt["job_id"], signals=["V(out)"], in_ltspice=True),
            ltspice_state,
        )
        assert result.structured_content is not None
        shown = result.structured_content["ltspice"]

        assert shown["shown"] is True, shown
        assert shown["pid"] == pid
        assert shown["panes"] == [["V(out)"]]
        settings = await asyncio.to_thread(Path(shown["plot_settings"]).read_bytes)
        section = read_plot_settings(settings).section("Transient Analysis")
        assert section is not None
        assert [pane.traces for pane in section.panes] == [("V(out)",)]
        assert Path(shown["plot_settings"]).with_suffix(".raw") == Path(shown["results"])

        command = _bridge_of(ltspice_state)

        def results_in_front() -> str:
            with BridgeSession(command) as session:
                session.attach(pid)
                return str(session.call("get_raw_info")["path"])

        assert Path(await asyncio.to_thread(results_in_front)) == Path(shown["results"])
        # Nothing was opened in a browser beside it.
        assert result.structured_content["opened"] is False

    async def test_a_checked_sheet_is_opened_in_the_window_and_put_in_front(
        self, ltspice_state: SessionState, open_sheet, work_dir: Path
    ):
        from ltspice_mcp.lib.ltspice_bridge import BridgeSession
        from ltspice_mcp.tools.verify import VerifyCircuitInput, handle_verify_circuit

        already_open, pid = open_sheet
        another = work_dir / "another.asc"
        await asyncio.to_thread(shutil.copyfile, already_open, another)
        command = _bridge_of(ltspice_state)

        def in_front() -> tuple[str, list[str]]:
            with BridgeSession(command) as session:
                session.attach(pid)
                front = session.active_design()
                return str(front), session.open_designs()

        async def show(sheet: Path) -> dict:
            result = await handle_verify_circuit(
                VerifyCircuitInput(path=str(sheet), checks=["layout"], in_ltspice=True),
                ltspice_state,
            )
            assert result.structured_content is not None
            return result.structured_content["ltspice"]

        shown = await show(another)
        assert (shown["shown"], shown["pid"], shown["already_open"]) == (True, pid, False), shown
        front, designs = await asyncio.to_thread(in_front)
        assert Path(front) == another
        assert {Path(design) for design in designs} == {already_open, another}

        shown = await show(already_open)
        assert (shown["shown"], shown["already_open"]) == (True, True), shown
        assert shown["differs_from_file"] is False
        front, _designs = await asyncio.to_thread(in_front)
        assert Path(front) == already_open

    def test_an_ltspice_the_bridge_starts_is_out_of_sight_and_ends_with_the_session(
        self, ltspice_state: SessionState
    ):
        """What holds whatever the bridge does. Here nothing stops it starting
        an LTspice (the command leaves out the path that makes the launch
        fail): the one it starts has its windows on the server's desktop and
        none on this one, and is gone when the session is closed."""
        assert sys.platform == "win32"
        import ctypes
        from ctypes import wintypes

        from ltspice_mcp.lib.ltspice_bridge import BRIDGE_NAME, BridgeSession
        from ltspice_mcp.lib.simulator_build import executable_path
        from tests.conftest import LIVENESS_S

        exe = executable_path(ltspice_state.default_simulator)
        assert exe is not None
        desktop = hidden_desktop.shared()
        if desktop is None:
            pytest.skip("Windows gave no desktop to start the bridge on")

        user = ctypes.WinDLL("user32")
        listed = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
        user.EnumWindows.argtypes = [listed, wintypes.LPARAM]
        user.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]

        def owners_on_this_desktop() -> set[int]:
            found: set[int] = set()

            def note(window: int, _unused: int) -> bool:
                pid = wintypes.DWORD(0)
                user.GetWindowThreadProcessId(window, ctypes.byref(pid))
                found.add(int(pid.value))
                return True

            user.EnumWindows(listed(note), 0)
            return found

        with BridgeSession([str(Path(exe).with_name(BRIDGE_NAME))], timeout=LIVENESS_S) as session:
            if session.instances():
                pytest.skip("an LTspice is already running: the bridge would attach to it")
            session.call("list_open_designs")  # a read is enough for it to start one
            (started,) = session.instances()
            assert started.mode == "headless"
            launched = psutil.Process(started.pid)
            assert launched.name().casefold() == Path(exe).name.casefold()
            assert started.pid in desktop.window_owners()
            assert started.pid not in owners_on_this_desktop()
        launched.wait(timeout=LIVENESS_S)
        assert not launched.is_running()
