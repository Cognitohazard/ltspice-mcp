"""Domain dataclasses for simulation and batch jobs.

This module is the leaf of the job-subsystem dependency graph: it
defines the types and nothing else. Extracting these from ``state.py``
broke a cluster of import cycles where ``state`` imported its
collaborators (``lib.job_registry``, ``lib.job_store``, ``lib.job_lifecycle``,
``lib.observability``) and those collaborators needed the dataclasses
back for their type signatures.

After this split, the graph is strictly layered:

    lib.job_types (this file)
        ↑
        ├── state (SessionState composes JobRegistry)
        ├── lib.job_registry (owns sim_jobs / batch_jobs dicts)
        ├── lib.job_lifecycle (transition chokepoint)
        ├── lib.observability (event emission)
        ├── lib.job_store (disk persistence)
        └── lib.{sim,sweep,montecarlo}_runner

No TYPE_CHECKING guards remain in the collaborators — they import
these classes at runtime cleanly. ``state.py`` re-exports every name
below so downstream code that wrote ``from ltspice_mcp.state import
SimulationJob`` keeps working.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from ltspice_mcp.lib import now

# Terminal statuses — eligible for eviction, represent finished work.
# 'interrupted' is terminal because the owning runner is gone; metadata
# and any partial outputs are preserved but the job cannot resume
# in-process (recovery promotes it via lib.job_lifecycle.recover).
TERMINAL_STATUSES: frozenset[str] = frozenset(
    {"completed", "failed", "timeout", "cancelled", "interrupted"}
)

# Statuses that only make sense while a runner owns the job. Seeing one
# in a persisted record means the prior server died mid-run.
NON_TERMINAL_LIVE_STATUSES: frozenset[str] = frozenset({"queued", "running"})


@dataclass
class SweepDimension:
    """One axis of a parameter sweep.

    Either ``values`` (an explicit discrete list) is set, or the
    ``start``/``stop``/``step``|``points``/``scale`` range fields are. Use
    ``resolved_values()`` to get the concrete sweep values regardless of form.

    Attributes:
        type: "component" (add_value_sweep) or "parameter" (add_param_sweep)
        name: Component reference (e.g. "R1") or parameter name (e.g. "TEMP")
        start: Start value for sweep range
        stop: Stop value for sweep range
        step: Step size — mutually exclusive with points
        points: Number of points — mutually exclusive with step
        scale: "linear" or "log"
        values: Explicit discrete values — mutually exclusive with the range fields
    """

    type: Literal["component", "parameter"]
    name: str
    start: float | None = None
    stop: float | None = None
    step: float | None = None
    points: int | None = None
    scale: str = "linear"
    values: list[float] | None = None

    def resolved_values(self) -> list[float]:
        """Concrete sweep values for this dimension.

        Returns the explicit ``values`` list when set; otherwise generates the
        range from start/stop/step|points/scale.
        """
        from ltspice_mcp.lib.sweep_utils import generate_sweep_range

        if self.values is not None:
            return [float(v) for v in self.values]
        if self.start is None or self.stop is None:
            raise ValueError(
                f"Sweep dimension '{self.name}' has neither explicit values nor a "
                "start/stop range."
            )
        return generate_sweep_range(self.start, self.stop, self.step, self.points, self.scale)


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
        component_overrides: Per-component tolerances: ref -> (tolerance, distribution)
        num_runs: Number of Monte Carlo runs (default 100)
        seed: Optional RNG seed for reproducible runs. None => fresh entropy.
        model_tolerances: Process-variation rules — sampled once per .MODEL
            per run; every transistor instance using the model inherits the
            perturbation. List items are ``ModelTolerance`` from
            ``lib.montecarlo``.
        mismatch_rules: Pelgrom-law mismatch rules per device prefix. List
            items are ``MismatchRule`` from ``lib.montecarlo``. Generates
            per-instance variant ``.MODEL`` cards in the per-run netlist.
        param_tolerances: ``.PARAM`` perturbation rules — sampled once per
            run. List items are ``ParamTolerance`` from ``lib.montecarlo``.
    """

    netlist: Path
    type_tolerances: dict[str, tuple[float, str]] = field(default_factory=dict)
    component_overrides: dict[str, tuple[float, str]] = field(default_factory=dict)
    num_runs: int = 100
    seed: int | None = None
    # Imported lazily-typed (Any) to avoid a circular dep on lib.montecarlo
    # — runtime objects are validated at construction in configure_montecarlo.
    model_tolerances: list = field(default_factory=list)
    mismatch_rules: list = field(default_factory=list)
    param_tolerances: list = field(default_factory=list)


@dataclass(frozen=True)
class RunRef:
    """One result run, the unit of the unified single-run/batch result model.

    A ``SimulationJob`` projects to exactly one ``RunRef`` (index 0 — the
    "batch of one"); a ``BatchJob`` projects to one per ``run_results`` entry.
    Result-extraction routines consume ``RunRef`` so they stay job-agnostic
    instead of branching on the job type. Built by ``services.runs_of``.

    Attributes:
        index: 0-based run index (always 0 for a single run).
        raw_file: Path to this run's .raw, or None if not produced.
        log_file: Path to this run's .log, or None if not produced.
        params: Per-run parameter values (empty for a single run).
    """

    index: int
    raw_file: Path | None
    log_file: Path | None
    params: dict[str, str] = field(default_factory=dict)


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
    status: Literal["running", "completed", "failed", "cancelled", "interrupted"] = "running"
    started_at: datetime = field(default_factory=now)
    completed_at: datetime | None = None
    error: str | None = None
    done_event: asyncio.Event = field(default_factory=asyncio.Event)
    run_results: dict[int, dict] = field(default_factory=dict)
    sweep_config: SweepConfig | None = None
    mc_config: MonteCarloConfig | None = None
    task: asyncio.Task | None = field(default=None, repr=False)
    # Cached convergence-warning scan; populated lazily by
    # ``services.scan_batch_convergence`` once the job is terminal.
    # ``None`` means "not scanned yet"; an empty list means "scanned, no
    # warnings found".
    convergence_warnings: list[dict] | None = field(default=None, repr=False)


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
        task: RunTask from spicelib
        done_event: Event signaled when simulation completes
    """

    job_id: str
    netlist: Path
    simulator: str
    status: Literal[
        "queued", "running", "completed", "failed", "timeout", "cancelled", "interrupted"
    ]
    started_at: datetime
    completed_at: datetime | None = None
    raw_file: Path | None = None
    log_file: Path | None = None
    error: str | None = None
    task: Any | None = None
    done_event: asyncio.Event = field(default_factory=asyncio.Event)
