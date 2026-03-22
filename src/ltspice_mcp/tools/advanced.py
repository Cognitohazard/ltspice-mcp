"""Advanced analysis tools — parametric sweep and Monte Carlo MCP handlers. (Phase 6)"""

import asyncio
import logging
from math import prod

from mcp import types

from ltspice_mcp.errors import BatchJobError
from ltspice_mcp.lib.batch_results import (
    compute_batch_stats,
    filter_runs_by_params,
    get_progress_snapshot,
)
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
    FORMAT_PROP, format_response, require_simulator, resolve_netlist_path,
    resolve_output_folder, run_sync, text_response,
)

logger = logging.getLogger(__name__)


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
async def handle_configure_sweep(arguments: dict, state: SessionState) -> list[types.TextContent]:
    """Configure a multi-parameter sweep and store it for later execution.

    Validates all parameters, creates SweepDimension objects, computes total
    run count, and stores the SweepConfig in session state.

    Args:
        arguments: Tool arguments with netlist and parameters array
        state: Current session state

    Returns:
        TextContent with config ID and summary
    """
    netlist_str = arguments["netlist"]
    parameters = arguments["parameters"]

    netlist_path = resolve_netlist_path(netlist_str, state)

    if not parameters:
        raise BatchJobError("At least one parameter dimension is required")

    # Validate and build dimensions
    dimensions: list[SweepDimension] = []
    for i, param in enumerate(parameters):
        name = param.get("name", "").strip()
        if not name:
            raise BatchJobError(f"Parameter {i}: name is required and must be non-empty")

        param_type = param.get("type", "")
        if param_type not in ("component", "parameter"):
            raise BatchJobError(
                f"Parameter '{name}': type must be 'component' or 'parameter', got '{param_type}'"
            )

        try:
            start = float(param["start"])
            stop = float(param["stop"])
        except (KeyError, TypeError, ValueError) as e:
            raise BatchJobError(f"Parameter '{name}': start and stop must be numbers: {e}")

        step = param.get("step")
        points = param.get("points")
        scale = param.get("scale", "linear")

        # step and points are mutually exclusive
        if step is not None and points is not None:
            raise BatchJobError(
                f"Parameter '{name}': step and points are mutually exclusive — provide one, not both"
            )
        if step is None and points is None:
            raise BatchJobError(
                f"Parameter '{name}': one of step or points is required"
            )

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
async def handle_run_sweep(arguments: dict, state: SessionState) -> list[types.TextContent]:
    """Start a previously configured parameter sweep.

    Looks up the sweep config, creates a BatchJob, and starts execution
    asynchronously. Returns the job ID immediately — never blocks.

    Args:
        arguments: Tool arguments with config_id and optional max_parallel
        state: Current session state

    Returns:
        TextContent with job ID for monitoring
    """
    config_id = arguments["config_id"]
    max_parallel = arguments.get("max_parallel")

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
    assert state.default_simulator is not None
    runner = state.runners.get_sweep_runner(
        loop=asyncio.get_running_loop(),
        simulator_class=state.default_simulator,
        output_folder=resolve_output_folder(state),
        max_parallel=max_parallel or state.config.max_parallel_sims,
    )
    asyncio.create_task(runner.start_sweep(batch_job, state))

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
async def handle_configure_montecarlo(
    arguments: dict, state: SessionState
) -> list[types.TextContent]:
    """Configure a Monte Carlo analysis and store it for later execution.

    Parses tolerances with type-name-to-prefix mapping, validates distribution
    names, and stores MonteCarloConfig in session state.

    Args:
        arguments: Tool arguments with netlist, tolerances, num_runs
        state: Current session state

    Returns:
        TextContent with config ID and summary
    """
    netlist_str = arguments["netlist"]
    tolerances_list = arguments["tolerances"]
    num_runs = int(arguments.get("num_runs", 100))

    netlist_path = resolve_netlist_path(netlist_str, state)

    if not tolerances_list:
        raise BatchJobError("At least one tolerance entry is required")

    if num_runs < 1:
        raise BatchJobError(f"num_runs must be >= 1, got {num_runs}")

    # Parse tolerances into type_tolerances and component_overrides
    type_tolerances: dict[str, tuple[float, str]] = {}
    component_overrides: dict[str, tuple[float, str]] = {}

    for entry in tolerances_list:
        ref = entry.get("ref", "").strip()
        if not ref:
            raise BatchJobError("Each tolerance entry must have a non-empty 'ref' field")

        try:
            tolerance = float(entry["tolerance"])
        except (KeyError, TypeError, ValueError) as e:
            raise BatchJobError(f"Tolerance entry for '{ref}': tolerance must be a number: {e}")

        # Normalize distribution name
        raw_dist = entry.get("distribution", "uniform").lower()
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
        ", ".join(f"{k}: {v[0]*100:.1f}% {v[1]}" for k, v in type_tolerances.items())
        if type_tolerances
        else "none"
    )
    component_summary = (
        ", ".join(f"{k}: {v[0]*100:.1f}% {v[1]}" for k, v in component_overrides.items())
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
async def handle_run_montecarlo(arguments: dict, state: SessionState) -> list[types.TextContent]:
    """Start a previously configured Monte Carlo analysis.

    Looks up the MC config, creates a BatchJob, and starts execution
    asynchronously. Returns the job ID immediately — never blocks.

    Args:
        arguments: Tool arguments with config_id and optional max_parallel
        state: Current session state

    Returns:
        TextContent with job ID for monitoring
    """
    config_id = arguments["config_id"]
    max_parallel = arguments.get("max_parallel")

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
    assert state.default_simulator is not None
    runner = state.runners.get_mc_runner(
        loop=asyncio.get_running_loop(),
        simulator_class=state.default_simulator,
        output_folder=resolve_output_folder(state),
        max_parallel=max_parallel or state.config.max_parallel_sims,
    )
    asyncio.create_task(runner.start_montecarlo(batch_job, state))

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
async def handle_get_batch_results(
    arguments: dict, state: SessionState
) -> list[types.TextContent]:
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
    job_id = arguments["job_id"]
    signal = arguments.get("signal")

    # Look up batch job
    batch_job = state.batch_jobs.get(job_id)
    if not batch_job:
        if signal is None:
            # Status mode: softer message
            return text_response(f"Batch job not found: {job_id}\n\nUse ltspice_run_sweep() or ltspice_run_montecarlo() to start a batch job",
            )
        raise BatchJobError(
            f"Batch job not found: {job_id}\n\n"
            f"Use ltspice_run_sweep() or ltspice_run_montecarlo() to start a batch job"
        )

    # --- No signal: return status/progress info ---
    if signal is None:
        netlist_name = batch_job.netlist.name

        if batch_job.status == "running":
            # Compute progress snapshot for ETA
            start_ts = batch_job.started_at.timestamp()
            snap = get_progress_snapshot(batch_job, start_ts)

            completed = snap["completed"]
            total = snap["total"]
            failed = snap["failed"]
            eta_s = snap["eta_s"]

            # Format ETA
            if eta_s is not None:
                if eta_s >= 60:
                    eta_str = f", ~{int(eta_s // 60)}m remaining"
                else:
                    eta_str = f", ~{int(eta_s)}s remaining"
            else:
                eta_str = ""

            progress_str = f"{completed}/{total} runs complete{eta_str}"

            return text_response(
                        f"Batch job {job_id} is running\n"
                        f"Type: {batch_job.job_type}\n"
                        f"Progress: {progress_str}\n"
                        f"Failed: {failed}\n"
                        f"Netlist: {netlist_name}\n\n"
                        f"Use ltspice_get_batch_results('{job_id}', signal='...') to query partial results"
            )

        elif batch_job.status == "completed":
            duration = 0.0
            if batch_job.completed_at and batch_job.started_at:
                duration = (batch_job.completed_at - batch_job.started_at).total_seconds()

            successful = batch_job.completed_runs - batch_job.failed_runs

            return text_response(
                        f"Batch job {job_id} completed\n"
                        f"Type: {batch_job.job_type}\n"
                        f"Total runs: {batch_job.total_runs}\n"
                        f"Successful: {successful}\n"
                        f"Failed: {batch_job.failed_runs}\n"
                        f"Duration: {duration:.1f}s\n\n"
                        f"Use ltspice_get_batch_results('{job_id}', signal='V(out)') to query results"
            )

        elif batch_job.status == "failed":
            error_msg = batch_job.error or "Unknown error"
            return text_response(
                        f"Batch job {job_id} failed\n"
                        f"Type: {batch_job.job_type}\n"
                        f"Netlist: {netlist_name}\n"
                        f"Error: {error_msg}"
            )

        elif batch_job.status == "cancelled":
            return text_response(
                        f"Batch job {job_id} was cancelled\n"
                        f"Type: {batch_job.job_type}\n"
                        f"Completed {batch_job.completed_runs} of {batch_job.total_runs} before cancellation. "
                        f"Partial results available via get_batch_results."
            )

        else:
            return text_response(f"Batch job {job_id} has unexpected status: {batch_job.status}",
            )

    # --- Signal provided: return results ---
    filters = arguments.get("filters")
    offset = int(arguments.get("offset", 0))
    limit = min(int(arguments.get("limit", 50)), 50)  # Capped at 50 per Phase 5 convention
    raw_mode = bool(arguments.get("raw", False))
    fmt = arguments.get("format")

    if batch_job.completed_runs == 0:
        return text_response(
            f"No completed runs yet for job {job_id}. Use ltspice_get_batch_results('{job_id}') to check progress."
        )

    # Apply parameter filters to get matching run indices
    if filters:
        matching_indices = filter_runs_by_params(batch_job.run_results, filters)
        filter_applied = True
    else:
        matching_indices = sorted(batch_job.run_results.keys())
        filter_applied = False

    total_matching = len(matching_indices)

    if total_matching == 0:
        return text_response(
            f"No runs match the specified filters.\n"
            f"Total runs available: {len(batch_job.run_results)}\n"
            f"Filters applied: {filters}"
        )

    # Build the subset of run_results for the matching indices
    matching_run_results = {idx: batch_job.run_results[idx] for idx in matching_indices}

    if not raw_mode:
        # Aggregated statistics mode (default)
        batch_stats = await run_sync(compute_batch_stats, matching_run_results, signal)

        run_count = batch_stats["run_count"]
        stats = batch_stats["stats"]
        worst_run = batch_stats["worst_case_run"]
        best_run = batch_stats["best_case_run"]

        if run_count == 0:
            return text_response(
                        f"Signal '{signal}' not found in any completed run.\n"
                        f"Job ID: {job_id}\n"
                        f"Available runs: {total_matching}\n\n"
                        f"Verify signal name (e.g., 'V(out)', 'I(R1)')"
            )

        # Format stats with sensible precision
        def _fmt(v) -> str:
            if v is None:
                return "N/A"
            return f"{v:.6g}"

        lines = [
            f"Batch Results: {signal}",
            f"Job ID: {job_id}",
            f"Type: {batch_job.job_type}",
            f"Runs analyzed: {run_count}",
        ]

        if filter_applied:
            lines.append(f"Filtered to {total_matching} of {len(batch_job.run_results)} runs")

        lines += [
            "",
            "Aggregate Statistics (peak absolute values across runs):",
            f"  Max:    {_fmt(stats['max_across_runs'])}",
            f"  Min:    {_fmt(stats['min_across_runs'])}",
            f"  Mean:   {_fmt(stats['mean_across_runs'])}",
            f"  Std:    {_fmt(stats['std_across_runs'])}",
            f"  Median: {_fmt(stats['median_across_runs'])}",
        ]

        if worst_run is not None:
            worst_params = batch_job.run_results[worst_run].get("params", {})
            params_str = (
                ", ".join(f"{k}={v}" for k, v in worst_params.items())
                if worst_params
                else "no params"
            )
            lines.append(f"\nWorst-case run: #{worst_run} ({params_str})")

        if best_run is not None:
            best_params = batch_job.run_results[best_run].get("params", {})
            params_str = (
                ", ".join(f"{k}={v}" for k, v in best_params.items())
                if best_params
                else "no params"
            )
            lines.append(f"Best-case run:  #{best_run} ({params_str})")

        json_data = {
            "job_id": job_id,
            "job_type": batch_job.job_type,
            "signal": signal,
            "run_count": run_count,
            "filtered": filter_applied,
            "stats": stats,
            "worst_case_run": worst_run,
            "best_case_run": best_run,
        }
        return format_response("\n".join(lines), json_data, fmt)

    else:
        # Raw per-run mode with pagination
        paginated_indices = matching_indices[offset: offset + limit]
        shown = len(paginated_indices)

        if shown == 0:
            return text_response(
                        f"No runs in requested page range.\n"
                        f"Total matching: {total_matching}, offset: {offset}, limit: {limit}"
            )

        # Compute stats for the paginated subset
        paginated_run_results = {idx: batch_job.run_results[idx] for idx in paginated_indices}
        page_stats = await run_sync(compute_batch_stats, paginated_run_results, signal)

        lines = [
            f"Batch Results (raw): {signal}",
            f"Job ID: {job_id}",
            f"Showing runs {offset + 1}-{offset + shown} of {total_matching}",
            "",
            f"{'Run':<6} {'Max':>12} {'Mean':>12} {'Min':>12}  Params",
            "-" * 60,
        ]

        def _fmt_col(v: float | None) -> str:
            return f"{v:>12.6g}" if v is not None else f"{'N/A':>12}"

        for run_summary in page_stats["runs"]:
            run_idx = run_summary["run_index"]
            params = run_summary.get("params", {})
            params_str = (
                " ".join(f"{k}={v}" for k, v in params.items()) if params else "-"
            )
            lines.append(
                f"{run_idx:<6} {_fmt_col(run_summary.get('peak'))} {_fmt_col(run_summary.get('mean'))} "
                f"{_fmt_col(run_summary.get('min'))}  {params_str}"
            )

        if offset + shown < total_matching:
            next_offset = offset + limit
            lines.append(f"\nNext page: ltspice_get_batch_results('{job_id}', signal='{signal}', raw=true, offset={next_offset})")

        return text_response("\n".join(lines))


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------
TOOL_DEFS: list[types.Tool] = [
    types.Tool(
        name="ltspice_configure_sweep",
        description=(
            "Configure a multi-parameter sweep for a netlist. "
            "Define one or more sweep dimensions (component values or .PARAM parameters), "
            "each with a start/stop range and either a step size or point count. "
            "Returns a config_id to use with ltspice_run_sweep(). "
            "Use this when you want to simulate a circuit across a range of parameter values "
            "to understand how it behaves (e.g., sweep R1 from 1k to 10k in 10 steps)."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "netlist": {
                    "type": "string",
                    "description": "Path to the netlist file (.cir, .net, .asc)",
                },
                "parameters": {
                    "type": "array",
                    "description": "List of sweep dimensions. Each dimension defines one swept parameter.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {
                                "type": "string",
                                "description": "Component reference (e.g. 'R1') or parameter name (e.g. 'TEMP')",
                            },
                            "type": {
                                "type": "string",
                                "description": "'component' for component values (add_value_sweep) or 'parameter' for .PARAM variables (add_param_sweep)",
                                "enum": ["component", "parameter"],
                            },
                            "start": {
                                "type": "number",
                                "description": "Start value of the sweep range",
                            },
                            "stop": {
                                "type": "number",
                                "description": "Stop value of the sweep range",
                            },
                            "step": {
                                "type": "number",
                                "description": "Step size (mutually exclusive with points)",
                            },
                            "points": {
                                "type": "integer",
                                "description": "Number of points in the sweep (mutually exclusive with step)",
                            },
                            "scale": {
                                "type": "string",
                                "description": "Sweep scale: 'linear' (default) or 'log'",
                                "enum": ["linear", "log"],
                            },
                        },
                        "required": ["name", "type", "start", "stop"],
                    },
                },
            },
            "required": ["netlist", "parameters"],
        },
        annotations=types.ToolAnnotations(
            readOnlyHint=False,
            destructiveHint=False,
            idempotentHint=True,
            openWorldHint=False,
        ),
    ),
    types.Tool(
        name="ltspice_run_sweep",
        description=(
            "Execute a previously configured parameter sweep. "
            "Starts the sweep asynchronously and returns a job_id immediately — never blocks. "
            "Use ltspice_get_batch_results(job_id) to monitor progress and "
            "ltspice_get_batch_results(job_id, signal='V(out)') to query results. "
            "Requires ltspice_configure_sweep() to have been called first."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "config_id": {
                    "type": "string",
                    "description": "Sweep config ID returned by ltspice_configure_sweep",
                },
                "max_parallel": {
                    "type": "integer",
                    "description": "Override maximum parallel simulations. Default: server config value.",
                },
            },
            "required": ["config_id"],
        },
        annotations=types.ToolAnnotations(
            readOnlyHint=False,
            destructiveHint=False,
            idempotentHint=False,
            openWorldHint=False,
        ),
    ),
    types.Tool(
        name="ltspice_configure_montecarlo",
        description=(
            "Configure a Monte Carlo analysis with component tolerances. "
            "Supports type-level defaults (e.g., all resistors get 5% tolerance) and "
            "per-component overrides (e.g., R1 gets 1% tolerance). "
            "Returns a config_id to use with ltspice_run_montecarlo(). "
            "Use this to understand circuit behavior under component variation and find "
            "worst-case corners."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "netlist": {
                    "type": "string",
                    "description": "Path to the netlist file (.cir, .net, .asc)",
                },
                "tolerances": {
                    "type": "array",
                    "description": "List of tolerance specifications. Each entry sets a tolerance for a component type or specific component.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "ref": {
                                "type": "string",
                                "description": "Component type ('resistors', 'R', 'capacitors', 'C', 'inductors', 'L') or specific ref ('R1', 'C3')",
                            },
                            "tolerance": {
                                "type": "number",
                                "description": "Tolerance as a fraction (e.g., 0.05 for 5%)",
                            },
                            "distribution": {
                                "type": "string",
                                "description": "Distribution type: 'uniform' (default) or 'gaussian'/'normal'",
                                "enum": ["uniform", "gaussian", "normal"],
                            },
                        },
                        "required": ["ref", "tolerance"],
                    },
                },
                "num_runs": {
                    "type": "integer",
                    "description": "Number of Monte Carlo runs. Default: 100.",
                },
            },
            "required": ["netlist", "tolerances"],
        },
        annotations=types.ToolAnnotations(
            readOnlyHint=False,
            destructiveHint=False,
            idempotentHint=True,
            openWorldHint=False,
        ),
    ),
    types.Tool(
        name="ltspice_run_montecarlo",
        description=(
            "Execute a previously configured Monte Carlo analysis. "
            "Starts the analysis asynchronously and returns a job_id immediately — never blocks. "
            "Use ltspice_get_batch_results(job_id) to monitor progress and "
            "ltspice_get_batch_results(job_id, signal='V(out)') to query statistics. "
            "Requires ltspice_configure_montecarlo() to have been called first."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "config_id": {
                    "type": "string",
                    "description": "Monte Carlo config ID returned by ltspice_configure_montecarlo",
                },
                "max_parallel": {
                    "type": "integer",
                    "description": "Override maximum parallel simulations. Default: server config value.",
                },
            },
            "required": ["config_id"],
        },
        annotations=types.ToolAnnotations(
            readOnlyHint=False,
            destructiveHint=False,
            idempotentHint=False,
            openWorldHint=False,
        ),
    ),
    types.Tool(
        name="ltspice_get_batch_results",
        description=(
            "Query a batch simulation job (sweep or Monte Carlo). "
            "Without signal: returns job status and progress. "
            "With signal: returns aggregate statistics or per-run data for that signal. "
            "Supports parameter filtering using SPICE notation."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "job_id": {
                    "type": "string",
                    "description": "Batch job ID from ltspice_run_sweep or ltspice_run_montecarlo",
                },
                "signal": {
                    "type": "string",
                    "description": "Signal name to analyze (e.g., 'V(out)', 'I(R1)', 'V(n001)'). Omit to get job status/progress instead.",
                },
                "filters": {
                    "type": "object",
                    "description": (
                        "Parameter filters. Keys are parameter names, values are filter expressions. "
                        "Exact: {\"R1\": \"1k\"}, Range: {\"R1\": \"1k..5k\"}"
                    ),
                },
                "offset": {
                    "type": "integer",
                    "description": "Pagination offset for raw mode. Default: 0.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max results per page for raw mode (max 50). Default: 50.",
                },
                "raw": {
                    "type": "boolean",
                    "description": "If true, return per-run data instead of aggregated stats. Default: false.",
                },
                "format": FORMAT_PROP,
            },
            "required": ["job_id"],
        },
        annotations=types.ToolAnnotations(
            readOnlyHint=True,
            destructiveHint=False,
            idempotentHint=True,
            openWorldHint=False,
        ),
    ),
]

TOOL_HANDLERS: dict[str, object] = {
    "ltspice_configure_sweep": handle_configure_sweep,
    "ltspice_run_sweep": handle_run_sweep,
    "ltspice_configure_montecarlo": handle_configure_montecarlo,
    "ltspice_run_montecarlo": handle_run_montecarlo,
    "ltspice_get_batch_results": handle_get_batch_results,
}
