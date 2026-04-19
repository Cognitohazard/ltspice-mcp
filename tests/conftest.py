"""Shared fixtures for ltspice-mcp tests."""

import shutil
from collections.abc import Iterator
from pathlib import Path

import pytest
from spicelib import AscEditor

from ltspice_mcp.config import ServerConfig
from ltspice_mcp.state import SessionState

_FIXTURE_SYMBOLS = Path(__file__).parent / "fixtures" / "symbols"
_FIXTURE_DRAFT = Path(__file__).parent / "fixtures" / "Draft1.asc"


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
