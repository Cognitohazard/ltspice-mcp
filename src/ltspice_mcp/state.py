"""Session state management and simulation job tracking."""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Type

from ltspice_mcp.config import ServerConfig
from ltspice_mcp.lib.cache import FileCache
from ltspice_mcp.lib.library_manager import LibraryManager


@dataclass
class SweepDimension:
    """One axis of a parameter sweep.

    Attributes:
        type: "component" (add_value_sweep) or "parameter" (add_param_sweep)
        name: Component reference (e.g. "R1") or parameter name (e.g. "TEMP")
        start: Start value for sweep range
        stop: Stop value for sweep range
        step: Step size — mutually exclusive with points
        points: Number of points — mutually exclusive with step
        scale: "linear" or "log"
    """

    type: Literal["component", "parameter"]
    name: str
    start: float
    stop: float
    step: float | None = None
    points: int | None = None
    scale: str = "linear"


@dataclass
class SweepConfig:
    """Configuration for a multi-dimensional parameter sweep.

    Attributes:
        netlist: Path to the netlist to sweep (bound at config creation)
        dimensions: List of sweep axes (one per varied parameter)
    """

    netlist: Path
    dimensions: list[SweepDimension] = field(default_factory=list)


@dataclass
class MonteCarloConfig:
    """Configuration for a Monte Carlo analysis run.

    Attributes:
        netlist: Path to the netlist
        type_tolerances: Per-component-type tolerances: prefix -> (tolerance, distribution)
            e.g. {"R": (0.05, "uniform")} means all resistors get 5% uniform tolerance
        component_overrides: Per-component tolerances: ref -> (tolerance, distribution)
            e.g. {"R1": (0.01, "normal")} overrides R1 with 1% normal distribution
        num_runs: Number of Monte Carlo runs (default 100)
        seed: Optional RNG seed for reproducibility
    """

    netlist: Path
    type_tolerances: dict[str, tuple[float, str]] = field(default_factory=dict)
    component_overrides: dict[str, tuple[float, str]] = field(default_factory=dict)
    num_runs: int = 100
    seed: int | None = None


@dataclass
class BatchJob:
    """Track state of a running or completed batch simulation job.

    Attributes:
        job_id: Unique identifier for this batch job
        job_type: "sweep" or "montecarlo"
        netlist: Path to the netlist file being processed
        total_runs: Total number of runs in this batch
        completed_runs: Number of runs completed so far
        failed_runs: Number of runs that failed
        status: Current job status
        started_at: When the batch job started
        completed_at: When the batch job finished (None if still running)
        error: Error message if the whole job failed
        done_event: Event signaled when batch completes or is cancelled
        run_results: Per-run results: run_index -> {raw_file, log_file, params}
        sweep_config: SweepConfig stored for reference during execution
        mc_config: MonteCarloConfig stored for reference during execution
    """

    job_id: str
    job_type: Literal["sweep", "montecarlo"]
    netlist: Path
    total_runs: int
    completed_runs: int = 0
    failed_runs: int = 0
    status: Literal["running", "completed", "failed", "cancelled"] = "running"
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: datetime | None = None
    error: str | None = None
    done_event: asyncio.Event = field(default_factory=asyncio.Event)
    run_results: dict[int, dict] = field(default_factory=dict)
    sweep_config: SweepConfig | None = None
    mc_config: MonteCarloConfig | None = None


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
        libraries: Loaded component libraries
        working_dir: Base directory for relative paths
        sweep_configs: Stored sweep configurations keyed by config_id
        mc_configs: Stored Monte Carlo configurations keyed by config_id
        batch_jobs: Active and completed batch jobs keyed by job_id
    """

    config: ServerConfig
    available_simulators: dict[str, Type]
    default_simulator: Type | None
    editors: FileCache  # FileCache[SpiceEditor] - type parameter for documentation
    results: FileCache  # FileCache[RawRead]
    jobs: dict[str, SimulationJob]
    libraries: LibraryManager
    working_dir: Path
    sweep_configs: dict[str, SweepConfig] = field(default_factory=dict)
    mc_configs: dict[str, MonteCarloConfig] = field(default_factory=dict)
    batch_jobs: dict[str, BatchJob] = field(default_factory=dict)

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
            libraries=LibraryManager(available),
            working_dir=config.working_dir,
        )

    def shutdown(self) -> None:
        """Clean up session resources at server shutdown.

        Clears file caches and cancels running jobs for graceful shutdown.
        """
        self.editors.clear()
        self.results.clear()
        # Cancel any running simulation jobs
        for job in self.jobs.values():
            if job.status in ("running", "queued"):
                job.status = "cancelled"
                job.completed_at = datetime.now()
                job.done_event.set()
        # Cancel any running batch jobs
        for batch_job in self.batch_jobs.values():
            if batch_job.status == "running":
                batch_job.status = "cancelled"
                batch_job.completed_at = datetime.now()
                batch_job.done_event.set()
