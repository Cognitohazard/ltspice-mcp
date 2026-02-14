"""Session state management and simulation job tracking."""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Type

from ltspice_mcp.config import ServerConfig
from ltspice_mcp.lib.cache import FileCache


@dataclass
class SimulationJob:
    """Track state of a running or completed simulation.

    Attributes:
        job_id: Unique identifier for this job
        netlist: Path to the netlist file being simulated
        simulator: Name of simulator used (ltspice, ngspice, etc.)
        status: Current job status
        started_at: When simulation started
        completed_at: When simulation finished (None if still running)
        raw_file: Path to generated .raw file (None until simulation completes)
        log_file: Path to simulation log file (None until available)
        error: Error message if simulation failed
        task: RunTask from spicelib (internal state, type: Any to avoid Phase 1 import)
        done_event: Event signaled when simulation completes
    """

    job_id: str
    netlist: Path
    simulator: str
    status: Literal["queued", "running", "completed", "failed", "timeout", "cancelled"]
    started_at: datetime
    completed_at: datetime | None = None
    raw_file: Path | None = None
    log_file: Path | None = None
    error: str | None = None
    task: Any | None = None  # RunTask from spicelib - typed as Any to defer Phase 3 import
    done_event: asyncio.Event = field(default_factory=asyncio.Event)


@dataclass
class SessionState:
    """Global server state for the current session.

    Holds configuration, detected simulators, file caches, and active jobs.
    Created at server startup and persists for the server lifetime.

    Attributes:
        config: Server configuration loaded from TOML/env vars
        available_simulators: Simulators detected at startup
        default_simulator: Simulator to use when not specified by user
        editors: Cache of parsed SpiceEditor instances (FileCache[SpiceEditor])
        results: Cache of parsed RawRead instances (FileCache[RawRead])
        jobs: Active and completed simulation jobs by job_id
        libraries: Loaded component libraries (deferred to Phase 6)
        working_dir: Base directory for relative paths
    """

    config: ServerConfig
    available_simulators: dict[str, Type]
    default_simulator: Type | None
    editors: FileCache  # FileCache[SpiceEditor] - type parameter for documentation
    results: FileCache  # FileCache[RawRead]
    jobs: dict[str, SimulationJob]
    libraries: dict[Path, Any]  # Will be dict[Path, LoadedLibrary] in Phase 6
    working_dir: Path

    @classmethod
    def create(cls, config: ServerConfig, available: dict[str, Type]) -> "SessionState":
        """Factory method to create session state at server startup.

        This is called by the server lifespan context manager to initialize
        the session state with detected simulators and empty caches.

        Args:
            config: Server configuration from ServerConfig.load()
            available: Available simulators from detect_simulators()

        Returns:
            Initialized SessionState instance
        """
        from ltspice_mcp.lib.simulator import select_default_simulator

        default = select_default_simulator(available, config)

        return cls(
            config=config,
            available_simulators=available,
            default_simulator=default,
            editors=FileCache(),
            results=FileCache(),
            jobs={},
            libraries={},
            working_dir=config.working_dir,
        )

    def shutdown(self) -> None:
        """Clean up session resources at server shutdown.

        Clears file caches and prepares for graceful shutdown.
        Future phases will add job cancellation logic here.
        """
        self.editors.clear()
        self.results.clear()
        # TODO Phase 3: Cancel pending jobs and wait for running jobs to complete
