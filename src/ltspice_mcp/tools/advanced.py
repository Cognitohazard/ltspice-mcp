"""Advanced analysis tools — parametric sweep and Monte Carlo MCP handlers. (Phase 6)"""

import asyncio
import logging
from math import prod
from typing import Literal

from mcp import types
from pydantic import Field

from ltspice_mcp.errors import BatchJobError
from ltspice_mcp.lib import services
from ltspice_mcp.lib.sweep_utils import (
    generate_batch_job_id,
    generate_config_id,
    generate_sweep_range,
)
from ltspice_mcp.state import (
    BatchJob,
    MonteCarloConfig,
    SessionState,
    SweepConfig,
    SweepDimension,
)
from ltspice_mcp.tools._base import (
    StrictModel,
    ToolInput,
    format_response,
    pagination_metadata,
    registry,
    require_simulator,
    resolve_netlist_path,
    resolve_output_folder,
    text_response,
)

logger = logging.getLogger(__name__)


class SweepParameter(StrictModel):
    """Nested sweep parameter definition."""

    name: str
    type: Literal["component", "parameter"]
    start: float
    stop: float
    step: float | None = None
    points: int | None = None
    scale: Literal["linear", "log"] = "linear"


class ConfigureSweepInput(ToolInput):
    netlist: str = Field(description="Path to the netlist file (.cir, .net, .asc)")
    parameters: list[SweepParameter] = Field(description="Sweep dimensions")


class RunBatchInput(ToolInput):
    config_id: str
    max_parallel: int | None = None


class MonteCarloTolerance(StrictModel):
    ref: str
    tolerance: float
    distribution: Literal["uniform", "gaussian", "normal"] = "uniform"


class ConfigureMonteCarloInput(ToolInput):
    netlist: str = Field(description="Path to the netlist file (.cir, .net, .asc)")
    tolerances: list[MonteCarloTolerance]
    num_runs: int = 100


class GetBatchResultsInput(ToolInput):
    job_id: str
    signal: str | None = None
    filters: dict[str, str] | None = None
    offset: int = 0
    limit: int = 50
    raw: bool = False
    format: Literal["json", "text"] | None = None


# ---------------------------------------------------------------------------
# Component type name -> single-char prefix mapping for Monte Carlo tolerances
# ---------------------------------------------------------------------------
_TYPE_NAME_TO_PREFIX: dict[str, str] = {
    # Resistors
    "r": "R",
    "resistor": "R",
    "resistors": "R",
    # Capacitors
    "c": "C",
    "capacitor": "C",
    "capacitors": "C",
    # Inductors
    "l": "L",
    "inductor": "L",
    "inductors": "L",
    # Voltage / Current sources
    "v": "V",
    "voltage": "V",
    "voltages": "V",
    "i": "I",
    "current": "I",
    "currents": "I",
}

# Distribution name normalization: user "gaussian" -> spicelib "normal"
_DISTRIBUTION_MAP: dict[str, str] = {
    "gaussian": "normal",
    "normal": "normal",
    "uniform": "uniform",
}


def _resolve_mc_ref(ref: str) -> tuple[str, bool]:
    """Resolve a Monte Carlo tolerance reference to (prefix, is_type_level).

    Handles:
      - Type names: "resistors", "R", "capacitor", etc. -> ("R", True)
      - Component refs: "R1", "C3", "L2" -> ("R1", False)

    Args:
        ref: Raw ref string from the user

    Returns:
        (resolved_ref, is_type_level) tuple
    """
    lower = ref.lower().strip()

    # Check type name map first
    if lower in _TYPE_NAME_TO_PREFIX:
        return (_TYPE_NAME_TO_PREFIX[lower], True)

    # Single uppercase letter -> treat as type prefix
    if len(ref) == 1 and ref.upper().isalpha():
        return (ref.upper(), True)

    # Otherwise assume component ref (e.g. "R1", "C3", "L2a")
    return (ref, False)


# ---------------------------------------------------------------------------
# Handler 1: configure_sweep
# ---------------------------------------------------------------------------
@registry.tool(
    name="ltspice_configure_sweep",
    description=(
        "Configure a multi-parameter sweep for a netlist and return a config_id "
        "for later execution."
    ),
    input_model=ConfigureSweepInput,
    annotations=types.ToolAnnotations(
        readOnlyHint=False,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    ),
    profiles=("full",),
)
async def handle_configure_sweep(arguments: ConfigureSweepInput, state: SessionState):
    """Configure a multi-parameter sweep and store it for later execution.

    Validates all parameters, creates SweepDimension objects, computes total
    run count, and stores the SweepConfig in session state.

    Args:
        arguments: Tool arguments with netlist and parameters array
        state: Current session state

    Returns:
        TextContent with config ID and summary
    """
    netlist_str = arguments.netlist
    parameters = arguments.parameters

    netlist_path = resolve_netlist_path(netlist_str, state)

    if not parameters:
        raise BatchJobError("At least one parameter dimension is required")

    # Validate and build dimensions
    dimensions: list[SweepDimension] = []
    for i, param in enumerate(parameters):
        name = param.name.strip()
        if not name:
            raise BatchJobError(f"Parameter {i}: name is required and must be non-empty")

        param_type = param.type
        if param_type not in ("component", "parameter"):
            raise BatchJobError(
                f"Parameter '{name}': type must be 'component' or 'parameter', got '{param_type}'"
            )

        start = float(param.start)
        stop = float(param.stop)
        step = param.step
        points = param.points
        scale = param.scale

        # step and points are mutually exclusive
        if step is not None and points is not None:
            raise BatchJobError(
                f"Parameter '{name}': step and points are mutually exclusive — provide one, not both"
            )
        if step is None and points is None:
            raise BatchJobError(f"Parameter '{name}': one of step or points is required")

        if scale not in ("linear", "log"):
            raise BatchJobError(
                f"Parameter '{name}': scale must be 'linear' or 'log', got '{scale}'"
            )

        # Convert types
        if step is not None:
            step = float(step)
        if points is not None:
            points = int(points)

        dimensions.append(
            SweepDimension(
                type=param_type,
                name=name,
                start=start,
                stop=stop,
                step=step,
                points=points,
                scale=scale,
            )
        )

    # Compute total runs: product of each dimension's point count
    dim_sizes: list[int] = []
    for dim in dimensions:
        values = generate_sweep_range(dim.start, dim.stop, dim.step, dim.points, dim.scale)
        dim_sizes.append(len(values))

    total_runs = prod(dim_sizes) if dim_sizes else 0

    # Build and store config
    config = SweepConfig(netlist=netlist_path, dimensions=dimensions)
    config_id = generate_config_id("sweep")
    state.sweep_configs[config_id] = config

    logger.info(
        f"Sweep configured: config_id={config_id}, netlist={netlist_path.name}, "
        f"dimensions={len(dimensions)}, total_runs={total_runs}"
    )

    return text_response(
        f"Sweep configured\n"
        f"Config ID: {config_id}\n"
        f"Netlist: {netlist_path}\n"
        f"Dimensions: {len(dimensions)}\n"
        f"Total simulations: {total_runs}\n\n"
        f"Use ltspice_run_sweep('{config_id}') to execute"
    )


# ---------------------------------------------------------------------------
# Handler 2: run_sweep
# ---------------------------------------------------------------------------
@registry.tool(
    name="ltspice_run_sweep",
    description=(
        "Execute a previously configured parameter sweep asynchronously and "
        "return a job_id immediately."
    ),
    input_model=RunBatchInput,
    annotations=types.ToolAnnotations(
        readOnlyHint=False,
        destructiveHint=False,
        idempotentHint=False,
        openWorldHint=False,
    ),
    profiles=("full", "agentic"),
)
async def handle_run_sweep(arguments: RunBatchInput, state: SessionState):
    """Start a previously configured parameter sweep.

    Looks up the sweep config, creates a BatchJob, and starts execution
    asynchronously. Returns the job ID immediately — never blocks.

    Args:
        arguments: Tool arguments with config_id and optional max_parallel
        state: Current session state

    Returns:
        TextContent with job ID for monitoring
    """
    config_id = arguments.config_id
    max_parallel = arguments.max_parallel

    # Look up config
    config = state.sweep_configs.get(config_id)
    if not config:
        raise BatchJobError(
            f"Sweep config not found: {config_id}\n\n"
            f"Use ltspice_configure_sweep() to create a sweep configuration first"
        )

    require_simulator(state)

    # Compute total runs
    dim_sizes = []
    for dim in config.dimensions:
        values = generate_sweep_range(dim.start, dim.stop, dim.step, dim.points, dim.scale)
        dim_sizes.append(len(values))
    total_runs = prod(dim_sizes) if dim_sizes else 0

    # Create and register batch job
    job_id = generate_batch_job_id("sweep")
    batch_job = BatchJob(
        job_id=job_id,
        job_type="sweep",
        netlist=config.netlist,
        total_runs=total_runs,
        sweep_config=config,
    )
    state.batch_jobs[job_id] = batch_job

    # Get sweep runner and start async task
    default_simulator = state.default_simulator
    if default_simulator is None:
        raise BatchJobError("No simulator available. Check server status.")
    runner = state.runners.get_sweep_runner(
        loop=asyncio.get_running_loop(),
        simulator_class=default_simulator,
        output_folder=resolve_output_folder(state),
        max_parallel=max_parallel or state.config.max_parallel_sims,
    )
    asyncio.create_task(runner.start_sweep(batch_job, state))  # noqa: RUF006

    logger.info(
        f"Sweep job started: job_id={job_id}, config_id={config_id}, total_runs={total_runs}"
    )

    return text_response(
        f"Sweep started\n"
        f"Job ID: {job_id}\n"
        f"Total runs: {total_runs}\n\n"
        f"Use ltspice_get_batch_results('{job_id}') to monitor progress\n"
        f"Use ltspice_get_batch_results('{job_id}', signal='...') to query results"
    )


# ---------------------------------------------------------------------------
# Handler 3: configure_montecarlo
# ---------------------------------------------------------------------------
@registry.tool(
    name="ltspice_configure_montecarlo",
    description=(
        "Configure a Monte Carlo analysis with component tolerances and return "
        "a config_id for later execution."
    ),
    input_model=ConfigureMonteCarloInput,
    annotations=types.ToolAnnotations(
        readOnlyHint=False,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    ),
    profiles=("full",),
)
async def handle_configure_montecarlo(arguments: ConfigureMonteCarloInput, state: SessionState):
    """Configure a Monte Carlo analysis and store it for later execution.

    Parses tolerances with type-name-to-prefix mapping, validates distribution
    names, and stores MonteCarloConfig in session state.

    Args:
        arguments: Tool arguments with netlist, tolerances, num_runs
        state: Current session state

    Returns:
        TextContent with config ID and summary
    """
    netlist_str = arguments.netlist
    tolerances_list = arguments.tolerances
    num_runs = int(arguments.num_runs)

    netlist_path = resolve_netlist_path(netlist_str, state)

    if not tolerances_list:
        raise BatchJobError("At least one tolerance entry is required")

    if num_runs < 1:
        raise BatchJobError(f"num_runs must be >= 1, got {num_runs}")

    # Parse tolerances into type_tolerances and component_overrides
    type_tolerances: dict[str, tuple[float, str]] = {}
    component_overrides: dict[str, tuple[float, str]] = {}

    for entry in tolerances_list:
        ref = entry.ref.strip()
        if not ref:
            raise BatchJobError("Each tolerance entry must have a non-empty 'ref' field")

        tolerance = float(entry.tolerance)

        # Normalize distribution name
        raw_dist = entry.distribution.lower()
        distribution = _DISTRIBUTION_MAP.get(raw_dist)
        if distribution is None:
            raise BatchJobError(
                f"Tolerance entry for '{ref}': distribution must be 'uniform', 'normal', or 'gaussian', "
                f"got '{raw_dist}'"
            )

        resolved_ref, is_type_level = _resolve_mc_ref(ref)

        if is_type_level:
            type_tolerances[resolved_ref] = (tolerance, distribution)
        else:
            component_overrides[resolved_ref] = (tolerance, distribution)

    # Build and store config
    config = MonteCarloConfig(
        netlist=netlist_path,
        type_tolerances=type_tolerances,
        component_overrides=component_overrides,
        num_runs=num_runs,
    )
    config_id = generate_config_id("mc")
    state.mc_configs[config_id] = config

    # Build summary strings
    type_summary = (
        ", ".join(f"{k}: {v[0] * 100:.1f}% {v[1]}" for k, v in type_tolerances.items())
        if type_tolerances
        else "none"
    )
    component_summary = (
        ", ".join(f"{k}: {v[0] * 100:.1f}% {v[1]}" for k, v in component_overrides.items())
        if component_overrides
        else "none"
    )

    logger.info(
        f"Monte Carlo configured: config_id={config_id}, netlist={netlist_path.name}, "
        f"num_runs={num_runs}"
    )

    return text_response(
        f"Monte Carlo configured\n"
        f"Config ID: {config_id}\n"
        f"Netlist: {netlist_path}\n"
        f"Runs: {num_runs}\n"
        f"Type tolerances: {type_summary}\n"
        f"Component overrides: {component_summary}\n"
        f"\n"
        f"Use ltspice_run_montecarlo('{config_id}') to execute"
    )


# ---------------------------------------------------------------------------
# Handler 4: run_montecarlo
# ---------------------------------------------------------------------------
@registry.tool(
    name="ltspice_run_montecarlo",
    description=(
        "Execute a previously configured Monte Carlo analysis asynchronously "
        "and return a job_id immediately."
    ),
    input_model=RunBatchInput,
    annotations=types.ToolAnnotations(
        readOnlyHint=False,
        destructiveHint=False,
        idempotentHint=False,
        openWorldHint=False,
    ),
    profiles=("full", "agentic"),
)
async def handle_run_montecarlo(arguments: RunBatchInput, state: SessionState):
    """Start a previously configured Monte Carlo analysis.

    Looks up the MC config, creates a BatchJob, and starts execution
    asynchronously. Returns the job ID immediately — never blocks.

    Args:
        arguments: Tool arguments with config_id and optional max_parallel
        state: Current session state

    Returns:
        TextContent with job ID for monitoring
    """
    config_id = arguments.config_id
    max_parallel = arguments.max_parallel

    # Look up config
    config = state.mc_configs.get(config_id)
    if not config:
        raise BatchJobError(
            f"Monte Carlo config not found: {config_id}\n\n"
            f"Use ltspice_configure_montecarlo() to create a Monte Carlo configuration first"
        )

    require_simulator(state)

    # Create and register batch job
    job_id = generate_batch_job_id("mc")
    batch_job = BatchJob(
        job_id=job_id,
        job_type="montecarlo",
        netlist=config.netlist,
        total_runs=config.num_runs,
        mc_config=config,
    )
    state.batch_jobs[job_id] = batch_job

    # Get MC runner and start async task
    default_simulator = state.default_simulator
    if default_simulator is None:
        raise BatchJobError("No simulator available. Check server status.")
    runner = state.runners.get_mc_runner(
        loop=asyncio.get_running_loop(),
        simulator_class=default_simulator,
        output_folder=resolve_output_folder(state),
        max_parallel=max_parallel or state.config.max_parallel_sims,
    )
    asyncio.create_task(runner.start_montecarlo(batch_job, state))  # noqa: RUF006

    logger.info(
        f"Monte Carlo job started: job_id={job_id}, config_id={config_id}, "
        f"total_runs={config.num_runs}"
    )

    return text_response(
        f"Monte Carlo started\n"
        f"Job ID: {job_id}\n"
        f"Total runs: {config.num_runs}\n\n"
        f"Use ltspice_get_batch_results('{job_id}') to monitor progress\n"
        f"Use ltspice_get_batch_results('{job_id}', signal='...') to query results"
    )


# ---------------------------------------------------------------------------
# Handler 5: get_batch_results (consolidated: status + results)
# ---------------------------------------------------------------------------
@registry.tool(
    name="ltspice_get_batch_results",
    description=(
        "Query a batch simulation job (sweep or Monte Carlo). Without signal: "
        "returns job status and progress. With signal: returns aggregate statistics "
        "or per-run data for that signal."
    ),
    input_model=GetBatchResultsInput,
    annotations=types.ToolAnnotations(
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    ),
    profiles=("full", "agentic"),
)
async def handle_get_batch_results(arguments: GetBatchResultsInput, state: SessionState):
    """Query a batch simulation job — status/progress or signal results.

    Without signal: returns job status and progress (running/completed/failed).
    With signal: returns aggregate statistics or per-run data for that signal.

    Supports parameter filtering using SPICE notation (exact or range).

    Args:
        arguments: Tool arguments with job_id, optional signal, optional filters/pagination
        state: Current session state

    Returns:
        TextContent with status info or statistics/per-run data
    """
    job_id = arguments.job_id
    signal = arguments.signal
    fmt = arguments.format
    batch_job = services.resolve_batch_job(job_id, state)

    if signal is None:
        data = services.get_batch_status(batch_job)
        return format_response(_format_batch_status_text(data), data, fmt)

    filters = arguments.filters
    offset = arguments.offset
    limit = min(arguments.limit, 50)
    raw_mode = arguments.raw
    data = services.get_batch_signal_data(
        batch_job,
        signal,
        filters=filters,
        raw=raw_mode,
        offset=offset,
        limit=limit,
    )
    if raw_mode:
        data["pagination"] = pagination_metadata(data["total_matching"], offset, limit)
        return format_response(_format_batch_raw_text(data), data, fmt)

    return format_response(_format_batch_aggregate_text(data, batch_job), data, fmt)


def _format_batch_status_text(data: dict) -> str:
    """Format batch status/progress output for humans."""
    status = data["status"]
    if status == "running":
        eta_s = data["eta_s"]
        eta_str = ""
        if eta_s is not None:
            eta_str = (
                f", ~{int(eta_s // 60)}m remaining" if eta_s >= 60 else f", ~{int(eta_s)}s remaining"
            )
        return (
            f"Batch job {data['job_id']} is running\n"
            f"Type: {data['job_type']}\n"
            f"Progress: {data['completed']}/{data['total']} runs complete{eta_str}\n"
            f"Failed: {data['failed']}\n"
            f"Netlist: {data['netlist']}\n\n"
            f"Use ltspice_get_batch_results('{data['job_id']}', signal='...') to query partial results"
        )
    if status == "completed":
        duration = data["duration"] or 0.0
        return (
            f"Batch job {data['job_id']} completed\n"
            f"Type: {data['job_type']}\n"
            f"Total runs: {data['total_runs']}\n"
            f"Successful: {data['successful']}\n"
            f"Failed: {data['failed_runs']}\n"
            f"Duration: {duration:.1f}s\n\n"
            f"Use ltspice_get_batch_results('{data['job_id']}', signal='V(out)') to query results"
        )
    if status == "failed":
        return (
            f"Batch job {data['job_id']} failed\n"
            f"Type: {data['job_type']}\n"
            f"Netlist: {data['netlist']}\n"
            f"Error: {data['error'] or 'Unknown error'}"
        )
    if status == "cancelled":
        return (
            f"Batch job {data['job_id']} was cancelled\n"
            f"Type: {data['job_type']}\n"
            f"Completed {data['completed_runs']} of {data['total_runs']} before cancellation. "
            f"Partial results available via get_batch_results."
        )
    raise BatchJobError(f"Batch job {data['job_id']} has unexpected status: {status}")


def _format_batch_aggregate_text(data: dict, batch_job: BatchJob) -> str:
    """Format aggregate batch signal statistics."""
    stats = data["stats"]

    def _fmt(v: float | None) -> str:
        if v is None:
            return "N/A"
        return f"{v:.6g}"

    lines = [
        f"Batch Results: {data['signal']}",
        f"Job ID: {data['job_id']}",
        f"Type: {data['job_type']}",
        f"Runs analyzed: {data['run_count']}",
    ]
    if data["filtered"]:
        lines.append(f"Filtered to {data['total_matching']} of {data['total_available']} runs")
    lines += [
        "",
        "Aggregate Statistics (peak absolute values across runs):",
        f"  Max:    {_fmt(stats['max_across_runs'])}",
        f"  Min:    {_fmt(stats['min_across_runs'])}",
        f"  Mean:   {_fmt(stats['mean_across_runs'])}",
        f"  Std:    {_fmt(stats['std_across_runs'])}",
        f"  Median: {_fmt(stats['median_across_runs'])}",
    ]

    worst_run = data["worst_case_run"]
    if worst_run is not None:
        worst_params = batch_job.run_results[worst_run].get("params", {})
        params_str = ", ".join(f"{k}={v}" for k, v in worst_params.items()) if worst_params else "no params"
        lines.append(f"\nWorst-case run: #{worst_run} ({params_str})")

    best_run = data["best_case_run"]
    if best_run is not None:
        best_params = batch_job.run_results[best_run].get("params", {})
        params_str = ", ".join(f"{k}={v}" for k, v in best_params.items()) if best_params else "no params"
        lines.append(f"Best-case run:  #{best_run} ({params_str})")

    return "\n".join(lines)


def _format_batch_raw_text(data: dict) -> str:
    """Format raw per-run batch signal data."""
    lines = [
        f"Batch Results (raw): {data['signal']}",
        f"Job ID: {data['job_id']}",
        f"Showing runs {data['offset'] + 1}-{data['offset'] + len(data['runs'])} of {data['total_matching']}",
        "",
        f"{'Run':<6} {'Max':>12} {'Mean':>12} {'Min':>12}  Params",
        "-" * 60,
    ]

    def _fmt_col(v: float | None) -> str:
        return f"{v:>12.6g}" if v is not None else f"{'N/A':>12}"

    for run_summary in data["runs"]:
        run_idx = run_summary["run_index"]
        params = run_summary.get("params", {})
        params_str = " ".join(f"{k}={v}" for k, v in params.items()) if params else "-"
        lines.append(
            f"{run_idx:<6} {_fmt_col(run_summary.get('peak'))} {_fmt_col(run_summary.get('mean'))} "
            f"{_fmt_col(run_summary.get('min'))}  {params_str}"
        )

    pagination = data["pagination"]
    if pagination["has_more"]:
        lines.append(
            f"\nNext page: ltspice_get_batch_results('{data['job_id']}', signal='{data['signal']}', raw=true, offset={pagination['next_offset']})"
        )

    return "\n".join(lines)
