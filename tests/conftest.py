"""Shared fixtures and helpers for ltspice-mcp tests."""

import shutil
import typing
from collections.abc import Iterator
from datetime import timedelta
from pathlib import Path

import pytest
from spicelib import AscEditor

from ltspice_mcp.config import ServerConfig
from ltspice_mcp.lib import now
from ltspice_mcp.state import BatchJob, SessionState, SimulationJob

FIXTURES_DIR = Path(__file__).parent / "fixtures"
_FIXTURE_SYMBOLS = FIXTURES_DIR / "symbols"
_FIXTURE_DRAFT = FIXTURES_DIR / "Draft1.asc"

# Recorded real-LTspice fixture values shared across test modules.
# Single transient run of an RC low-pass (R=1k, C=100n, 1 V step input); its
# log holds the one .MEAS line ``vfinal: V(out)=0.999876166042 at 0.0009``.
LTSPICE_TRAN_RC_LOG = FIXTURES_DIR / "ltspice_tran_rc.log"
LTSPICE_TRAN_RC_VFINAL = 0.999876166042
# 3-run LTspice parameter sweep of the same RC low-pass (R1 = 1k / 2.2k /
# 4.7k), one .MEAS log per run as the sweep/MC runners record them.
LTSPICE_SWEEP_RUN_LOGS = [FIXTURES_DIR / f"ltspice_sweep_meas_run{i}.log" for i in range(3)]


class FakeSim:
    """Stub simulator class for tests that need a default simulator."""

    spice_exe: typing.ClassVar[list[str]] = ["/fake/path/sim.exe"]


def stage_recorded_fixture(work_dir: Path, name: str) -> Path:
    """Copy a recorded fixture's .raw (and .log when recorded) into work_dir.

    Returns the staged .raw path. The .log lands next to it so a handler's
    automatic ``raw_file`` -> ``.log`` derivation is exercised for real.
    """
    raw = work_dir / f"{name}.raw"
    shutil.copy(FIXTURES_DIR / f"{name}.raw", raw)
    log = FIXTURES_DIR / f"{name}.log"
    if log.exists():
        shutil.copy(log, work_dir / f"{name}.log")
    return raw


def make_sim_job(job_id: str = "j1", *, status: str = "completed", **overrides) -> SimulationJob:
    """SimulationJob with test defaults; any dataclass field is overridable.

    A ``completed`` job gets a ``completed_at`` one second after
    ``started_at`` unless the caller overrides it.
    """
    started_at = overrides.pop("started_at", None) or now()
    fields: dict = {
        "netlist": Path("/tmp/test.cir"),
        "simulator": "ltspice",
        "completed_at": started_at + timedelta(seconds=1) if status == "completed" else None,
    }
    fields.update(overrides)
    return SimulationJob(
        job_id=job_id,
        status=status,  # type: ignore[arg-type]
        started_at=started_at,
        **fields,
    )


def make_batch_job(job_id: str = "b1", *, status: str = "completed", **overrides) -> BatchJob:
    """BatchJob with test defaults; any dataclass field is overridable.

    A ``completed`` job gets a ``completed_at`` one second after
    ``started_at`` unless the caller overrides it.
    """
    fields: dict = {
        "job_type": "sweep",
        "netlist": Path("/tmp/test.cir"),
        "total_runs": 2,
    }
    fields.update(overrides)
    bj = BatchJob(
        job_id=job_id,
        status=status,  # type: ignore[arg-type]
        **fields,
    )
    if status == "completed" and bj.completed_at is None:
        bj.completed_at = bj.started_at + timedelta(seconds=1)
    return bj


@pytest.fixture
def work_dir(tmp_path: Path) -> Path:
    """Temporary working directory for tests."""
    return tmp_path


@pytest.fixture
def config(work_dir: Path) -> ServerConfig:
    """ServerConfig pointing at tmp working directory."""
    return ServerConfig(
        working_dir=work_dir,
        allowed_paths=[work_dir],
        log_level="DEBUG",
    )


@pytest.fixture
def state_no_sim(config: ServerConfig) -> SessionState:
    """SessionState with no simulators available (degraded mode)."""
    return SessionState.create(config, available={})


@pytest.fixture
def state_with_sim(config: ServerConfig) -> SessionState:
    """SessionState with a (fake) default simulator, so handlers that guard on
    simulator availability (e.g. cancel_job's require_simulator) get past it."""
    return SessionState.create(config, available={"fake": FakeSim})


@pytest.fixture(autouse=True)
def _reset_runners():
    """No-op: runners now live on SessionState instances (RunnerManager).

    Each test creates a fresh SessionState with a fresh RunnerManager,
    so there are no module-level globals to reset. This fixture is kept
    as a guard — if runner management ever regresses to globals, tests
    will still pass.
    """
    return


@pytest.fixture(scope="session")
def asc_symbols() -> Iterator[Path]:
    """Register tiny .asy fixture symbols with AscEditor (class-level).

    Session-scoped so the class-level ``symbol_cache`` is populated once and
    reused across all tests. ``AscEditor._asy_file_find`` otherwise walks
    ``os.path.curdir`` (the project root, with ``.venv`` and ``.git``) on every
    cold load — ~1s per symbol lookup. Keeping the cache warm across the
    session eliminates that walk for every test after the first.
    """
    AscEditor.set_custom_library_paths(str(_FIXTURE_SYMBOLS))
    for asy in _FIXTURE_SYMBOLS.glob("*.asy"):
        AscEditor.symbol_cache[asy.name] = str(asy)
    yield _FIXTURE_SYMBOLS
    AscEditor.custom_lib_paths = []
    AscEditor.symbol_cache = {}


@pytest.fixture
def asc_state(state_no_sim: SessionState, work_dir: Path, asc_symbols: Path) -> SessionState:
    """SessionState with .asc editor available and a Draft1.asc copied into work_dir."""
    dest = work_dir / "Draft1.asc"
    shutil.copy(_FIXTURE_DRAFT, dest)
    return state_no_sim


@pytest.fixture
def asc_file(asc_state: SessionState, work_dir: Path) -> Path:
    """Path to Draft1.asc within the test work_dir."""
    return work_dir / "Draft1.asc"


@pytest.fixture
def sample_netlist(work_dir: Path) -> Path:
    """Create a simple RC filter netlist and return its path."""
    p = work_dir / "rc_filter.cir"
    p.write_text(
        "* RC Low-Pass Filter\n"
        "R1 in out 1k\n"
        "C1 out 0 100n\n"
        "V1 in 0 AC 1\n"
        ".ac dec 100 1 1Meg\n"
        ".meas AC fc WHEN mag(V(out))=0.707\n"
        ".param Rval=1k\n"
        ".END\n"
    )
    return p
