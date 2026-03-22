"""Shared fixtures for ltspice-mcp tests."""

from pathlib import Path

import pytest

from ltspice_mcp.config import ServerConfig
from ltspice_mcp.state import SessionState



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
    yield


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
