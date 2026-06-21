"""Advanced analysis tools — parametric sweep and Monte Carlo MCP handlers. (Phase 6)"""

import asyncio
import logging
from math import prod
from typing import Literal

from mcp import types
from pydantic import Field

from ltspice_mcp.errors import BatchJobError
from ltspice_mcp.lib import services
from ltspice_mcp.lib.format import parse_spice_value
from ltspice_mcp.lib.montecarlo import (
    MismatchRule,
    ModelTolerance,
    ParamTolerance,
    ToleranceSpec,
)
from ltspice_mcp.lib.spice_lex import lex
from ltspice_mcp.lib.sweep_utils import (
    generate_batch_job_id,
    generate_config_id,
)
from ltspice_mcp.state import (
    BatchJob,
    MonteCarloConfig,
    SessionState,
    SweepConfig,
    SweepDimension,
)
from ltspice_mcp.tools._base import (
    DEFAULT_PAGE_CAP,
    PAGINATION_SCHEMA,
    StrictModel,
    ToolInput,
    format_response,
    pagination_metadata,
    registry,
    require_simulator,
    resolve_output_folder,
    resolve_runnable_netlist,
    text_response,
)

logger = logging.getLogger(__name__)

# Upper bound on a single batch job's run count, shared by sweeps and Monte
# Carlo. A sweep cross-product (e.g. 5x5x16x100) can silently balloon into tens
# of thousands of cold simulator processes; refuse it up front like Monte Carlo
# already does, rather than spawning until the machine falls over.
MAX_BATCH_RUNS = 10_000


class SweepParameter(StrictModel):
    """Nested sweep parameter definition.

    Provide EITHER an explicit ``values`` list (e.g. E-series resistors) OR a
    ``start``/``stop`` range with ``step`` or ``points``. The two forms are
    mutually exclusive.
    """

    name: str = Field(description="Component reference (e.g., 'R1') or parameter name")
    type: Literal["component", "parameter"] = Field(
        description="'component' for ref values, 'parameter' for .PARAM"
    )
    values: list[float | str] | None = Field(
        default=None,
        description=(
            "Explicit discrete sweep values, e.g. [1000, 2200, 4700] for E-series "
            "resistors. Accepts plain numbers or SPICE notation strings, e.g. "
            "['1k', '2.2k', '4.7k']. Mutually exclusive with start/stop/step/points."
        ),
    )
    start: float | str | None = Field(
        default=None,
        description="Start value of sweep range (plain number or SPICE notation, e.g. '1k')",
    )
    stop: float | str | None = Field(
        default=None,
        description="End value of sweep range (plain number or SPICE notation, e.g. '10k')",
    )
    step: float | str | None = Field(
        default=None,
        description=(
            "Step size, plain number or SPICE notation, e.g. '1k' (mutually exclusive with points)"
        ),
    )
    points: int | None = Field(
        default=None, description="Number of points (mutually exclusive with step)"
    )
    scale: Literal["linear", "log"] = Field(default="linear", description="Sweep scale")


class ConfigureSweepInput(ToolInput):
    netlist: str = Field(description="Path to the netlist file (.cir, .net, .asc)")
    parameters: list[SweepParameter] = Field(description="Sweep dimensions")


class RunBatchInput(ToolInput):
    config_id: str = Field(
        description="Configuration ID from configure_sweep or configure_montecarlo"
    )
    max_parallel: int | None = Field(
        default=None, description="Max concurrent simulations (default: server config)"
    )


class MonteCarloTolerance(StrictModel):
    ref: str = Field(
        description="Component ref (e.g., 'R1') or type name (e.g., 'resistors', 'R')"
    )
    tolerance: float = Field(description="Tolerance as fraction (e.g., 0.05 for 5%)")
    distribution: Literal["uniform", "gaussian", "normal"] = Field(
        default="uniform", description="Distribution type"
    )


class MonteCarloParameterTolerance(StrictModel):
    """Tolerance for one parameter — either a model card param or a .PARAM."""

    tolerance: float = Field(
        description=(
            "Tolerance value. Interpreted by 'kind': fraction of nominal "
            "(0.05 = ±5%) for relative; σ/half-range in source units (0.012 = "
            "±12 mV at 3σ for absolute on a voltage parameter). Tolerance "
            "represents ±3σ for normal distributions."
        ),
    )
    distribution: Literal["uniform", "gaussian", "normal"] = Field(
        default="normal",
        description="Distribution type. Default 'normal' matches foundry MC convention.",
    )
    kind: Literal["relative", "absolute"] = Field(
        default="relative",
        description=(
            "'relative' = fraction of nominal (good for KP, RD-like). "
            "'absolute' = σ/half-range in source units (good for VTH where "
            "PDKs spec σ_VTH in volts directly)."
        ),
    )


class MonteCarloModelTolerance(StrictModel):
    """Process-variation rule: sampled once per .MODEL per run.

    Every transistor instance using this model inherits the same perturbed
    parameters in a given run — this matches foundry-correlated process
    variation (wafer-to-wafer / die-to-die). For uncorrelated per-instance
    mismatch, use the 'mismatch' input instead.
    """

    model: str = Field(
        description="Name of the .MODEL card to perturb (case-insensitive, must already exist in the netlist)."
    )
    parameters: dict[str, MonteCarloParameterTolerance] = Field(
        description=(
            "Per-parameter tolerance specs keyed by parameter name (e.g., "
            "{'VTO': {'tolerance': 0.012, 'kind': 'absolute'}, "
            "'KP': {'tolerance': 0.10, 'kind': 'relative'}})."
        ),
    )


class MonteCarloMismatchRule(StrictModel):
    """Pelgrom-law mismatch coefficients applied to one device prefix.

    σ(ΔVTH) = AVT/√(W·L) and σ(ΔK)/K = AK/√(W·L), sampled INDEPENDENTLY per
    instance per run. Generates per-instance variant .MODEL cards inlined
    into the per-run netlist (foundry preprocessor convention).

    Defaults are deliberately not provided — coefficients are technology-
    specific. Typical values: 65nm AVT≈3-5 mV·µm, AK≈1-2 %·µm.
    """

    prefix: str = Field(
        default="M",
        description=(
            "Device prefix to apply mismatch to (case-insensitive). Default 'M' "
            "(MOSFETs). Other letter prefixes (e.g. 'Q' for BJTs, 'J' for JFETs) "
            "work too — the engine matches on the leading character. Mismatch "
            "math (Pelgrom σ ∝ 1/√(W·L)) and the vth_param/k_param defaults are "
            "still MOSFET-shaped, so for non-MOSFETs you'll typically want to "
            "set vth_param/k_param to that device's threshold/gain parameters."
        ),
    )
    AVT: float = Field(
        default=0.0,
        description=(
            "VTH-mismatch coefficient in V·µm (e.g. 3e-3 = 3 mV·µm). 0 disables VTH mismatch."
        ),
    )
    AK: float = Field(
        default=0.0,
        description=(
            "K-mismatch coefficient in fraction·µm (e.g. 0.02 = 2%·µm). 0 disables K mismatch."
        ),
    )
    distribution: Literal["uniform", "gaussian", "normal"] = Field(
        default="normal",
        description="Distribution type for the per-instance offset (default normal).",
    )
    vth_param: str = Field(
        default="VTO",
        description=(
            "Model-card parameter receiving ΔVTH. Defaults to 'VTO' (Level-1 SPICE); "
            "use 'VTH0' for BSIM models."
        ),
    )
    k_param: str = Field(
        default="KP",
        description=(
            "Model-card parameter scaled by (1+ΔK/K). Defaults to 'KP' (Level-1); "
            "use 'U0' for BSIM."
        ),
    )
    min_wl_um2: float = Field(
        default=1e-3,
        description=(
            "Lower bound on W·L (in µm²) used when computing Pelgrom σ — "
            "guards against div-by-zero for behaviorally-described instances."
        ),
    )


class MonteCarloParamRule(StrictModel):
    """Sample-once-per-run perturbation of a .PARAM directive.

    Useful when the user has already wired .PARAM substitution into model
    cards (e.g., '.MODEL NMOS1 NMOS(VTO={vto_n})' with '.PARAM vto_n=0.7').
    """

    name: str = Field(description=".PARAM name to perturb (must already exist).")
    tolerance: float = Field(description="Tolerance value (see kind).")
    distribution: Literal["uniform", "gaussian", "normal"] = Field(default="normal")
    kind: Literal["relative", "absolute"] = Field(default="relative")


class ConfigureMonteCarloInput(ToolInput):
    netlist: str = Field(description="Path to the netlist file (.cir, .net, .asc)")
    tolerances: list[MonteCarloTolerance] = Field(
        default_factory=list,
        description=(
            "R/C/L (and V/I type-level) component tolerance specifications. A "
            "ref-named entry (e.g. 'R1') sets a per-component tolerance; a "
            "type-named entry (e.g. 'R' or 'resistors') sets a type-level tolerance."
        ),
    )
    model_tolerances: list[MonteCarloModelTolerance] = Field(
        default_factory=list,
        description=(
            "Process-variation rules: per-.MODEL parameter perturbations "
            "sampled once per run. All instances of the model see the same "
            "perturbation (correlated)."
        ),
    )
    mismatch: list[MonteCarloMismatchRule] = Field(
        default_factory=list,
        description=(
            "Pelgrom-law mismatch rules per device prefix. Sampled INDEPENDENTLY "
            "per instance per run. Requires explicit AVT/AK — defaults are 0 "
            "(no mismatch) since coefficients are technology-specific."
        ),
    )
    param_tolerances: list[MonteCarloParamRule] = Field(
        default_factory=list,
        description=(
            "Sample-once-per-run perturbation of .PARAM directives. Use this "
            "when the netlist already wires {param} substitutions into model "
            "cards or component values."
        ),
    )
    num_runs: int = Field(default=100, description="Number of Monte Carlo iterations")
    seed: int | None = Field(
        default=None,
        description="Optional RNG seed for reproducible runs. None = fresh entropy each call.",
    )


class GetBatchResultsInput(ToolInput):
    job_id: str = Field(description="Batch job ID from run_sweep or run_montecarlo")
    signal: str | None = Field(
        default=None, description="Signal name for per-signal stats (e.g., 'V(out)')"
    )
    filters: dict[str, str] | None = Field(
        default=None,
        description="Filter runs by parameter values (e.g., {'R1': '10k'}). Applies in both aggregate and raw mode (requires signal).",
    )
    at: str | None = Field(
        default=None,
        description=(
            "Optional time (transient) or frequency (AC) point in SPICE notation "
            "(e.g., '1k', '100u'). When given, each run is sliced to a single sample at that "
            "point before aggregating. Without this, the per-run peak across the full waveform "
            "is used, which conflates startup/roll-off with run-to-run variation on AC sweeps."
        ),
    )
    offset: int = Field(default=0, description="Pagination offset for raw data")
    limit: int = Field(
        default=50, description="Max raw data rows to return (server caps at 50; page with offset)"
    )
    raw: bool = Field(
        default=False, description="Return per-run raw data instead of aggregate stats"
    )
    format: Literal["json", "text"] | None = Field(
        default=None,
        description="Response format: 'json' for structured data, 'text' for human-readable",
    )


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


def _spec_from_input(tol: float, dist: str, kind: str) -> ToleranceSpec:
    """Build a ``ToleranceSpec`` from raw user input, validating distribution."""
    normalized = _DISTRIBUTION_MAP.get(dist.lower())
    if normalized is None:
        raise BatchJobError(
            f"Distribution must be 'uniform', 'normal', or 'gaussian', got {dist!r}"
        )
    return ToleranceSpec(
        tolerance=float(tol),
        distribution=normalized,  # type: ignore[arg-type]
        kind=kind,  # type: ignore[arg-type]
    )


def _normalize_sweep_value(raw: float | str, name: str, field: str) -> float:
    """Coerce a sweep value to float, parsing SPICE notation strings.

    Plain numbers pass through unchanged; strings like '10k' or '4.7k' are
    parsed the same way the rest of the surface (set_component_value, filters)
    interprets component values. Raises BatchJobError on an unparseable token.
    """
    if isinstance(raw, str):
        try:
            return parse_spice_value(raw)
        except (ValueError, TypeError) as e:
            raise BatchJobError(
                f"Parameter '{name}': {field} value {raw!r} is not a number or "
                f"valid SPICE notation (e.g. '10k', '4.7k', '159n'): {e}"
            ) from e
    return float(raw)


def _resolve_mc_ref(ref: str) -> tuple[str, bool]:
    """Resolve a Monte Carlo tolerance reference to (prefix, is_type_level).

    Handles:
      - Type names: "resistors", "R", "capacitor", etc. -> ("R", True)
      - Component refs: "R1", "C3", "L2" -> ("R1", False)

    Surrounding whitespace is stripped before classification so that
    "  R1  " resolves the same as "R1".

    Args:
        ref: Raw ref string from the user

    Returns:
        (resolved_ref, is_type_level) tuple
    """
    ref = ref.strip()
    lower = ref.lower()

    if lower in _TYPE_NAME_TO_PREFIX:
        return (_TYPE_NAME_TO_PREFIX[lower], True)

    # Single letter -> treat as type prefix
    if len(ref) == 1 and ref.isalpha():
        return (ref.upper(), True)

    # Otherwise assume component ref (e.g. "R1", "C3", "L2a")
    return (ref, False)


def _ngspice_preflight_warnings(netlist_path, state: SessionState) -> list[str]:
    """ngspice batch-mode warnings for a base netlist (.meas/.four skipped).

    Reuses the single-run pre-flight so the sweep/MC paths surface the same
    "ngspice cannot evaluate .meas in batch mode" warning instead of silently
    dropping measurements. A ``.step`` blocker is downgraded to a warning here
    (batch substitutes parameters per-run, so it isn't fatal at config time).
    """
    if state.default_simulator is None:
        return []
    from ltspice_mcp.errors import SimulationError

    try:
        return services.ngspice_preflight_warnings(netlist_path, state.default_simulator)
    except SimulationError as e:
        return [str(e)]


def _netlist_component_refs(netlist_path) -> set[str]:
    """Uppercased component reference designators on instance lines of a netlist.

    Enough to validate Monte Carlo component overrides against the netlist so an
    unmatched ref (e.g. ``C99``) is flagged instead of silently perturbing
    nothing. Delegates to the shared ``spice_lex`` tokenizer rather than
    hand-scanning lines, so continuations, CRLF, and malformed input are
    classified consistently with the rest of the codebase.
    """
    try:
        text = netlist_path.read_text(errors="replace")
    except OSError:
        return set()
    return {card.instance_ref.upper() for card in lex(text).cards if card.instance_ref}


# ---------------------------------------------------------------------------
# Handler 1: configure_sweep
# ---------------------------------------------------------------------------
@registry.tool(
    name="configure_sweep",
    description=(
        "Configure a multi-parameter sweep for a netlist and return a config_id "
        "for later execution. Dimensions combine as a full cross-product, so this "
        "also covers deterministic worst-case corner analysis (give each component "
        "a two-value [low, high] set — N parts yields 2^N corners that bound the "
        "true extremes, which random Monte Carlo cannot guarantee) and sensitivity "
        "analysis (sweep one part at a time across its tolerance to rank impact). "
        "Use configure_montecarlo instead for statistical yield/spread."
    ),
    input_model=ConfigureSweepInput,
    annotations=types.ToolAnnotations(
        readOnlyHint=False,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    ),
    profiles=("full", "agentic"),
)
async def handle_configure_sweep(args: ConfigureSweepInput, state: SessionState):
    """Configure a multi-parameter sweep and store it for later execution.

    Validates all parameters, creates SweepDimension objects, computes total
    run count, and stores the SweepConfig in session state.

    Args:
        args: Tool args with netlist and parameters array
        state: Current session state

    Returns:
        TextContent with config ID and summary
    """
    netlist_str = args.netlist
    parameters = args.parameters

    netlist_path = resolve_runnable_netlist(netlist_str, state)

    if not parameters:
        raise BatchJobError("At least one parameter dimension is required")

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

        # Explicit discrete value list (e.g. E-series) — mutually exclusive
        # with the start/stop/step/points range form (F5).
        if param.values is not None:
            if any(v is not None for v in (param.start, param.stop, param.step, param.points)):
                raise BatchJobError(
                    f"Parameter '{name}': 'values' is mutually exclusive with "
                    "start/stop/step/points — provide one form, not both"
                )
            if len(param.values) == 0:
                raise BatchJobError(f"Parameter '{name}': 'values' must be a non-empty list")
            dimensions.append(
                SweepDimension(
                    type=param_type,
                    name=name,
                    values=[_normalize_sweep_value(v, name, "values") for v in param.values],
                )
            )
            continue

        if param.start is None or param.stop is None:
            raise BatchJobError(
                f"Parameter '{name}': start and stop are required (or provide 'values')"
            )
        start = _normalize_sweep_value(param.start, name, "start")
        stop = _normalize_sweep_value(param.stop, name, "stop")
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

        if step is not None:
            step = _normalize_sweep_value(step, name, "step")
            if step <= 0:
                raise BatchJobError(f"Parameter '{name}': step must be > 0, got {step}")
        if points is not None:
            points = int(points)
            if points < 2:
                raise BatchJobError(f"Parameter '{name}': points must be >= 2, got {points}")

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

    # Compute total runs: product of each dimension's point count, and capture
    # the resolved value list per dimension so the response can enumerate them
    # (log vs linear spacing is otherwise unverifiable without running).
    dim_sizes: list[int] = []
    dim_values: list[tuple[str, list[float]]] = []
    for dim in dimensions:
        values = dim.resolved_values()
        dim_sizes.append(len(values))
        dim_values.append((dim.name, values))

    total_runs = prod(dim_sizes) if dim_sizes else 0
    if total_runs > MAX_BATCH_RUNS:
        raise BatchJobError(
            f"Sweep cross-product is {total_runs} runs, over the {MAX_BATCH_RUNS} cap "
            f"({' x '.join(str(s) for s in dim_sizes)}). Narrow a dimension or split "
            "the sweep — each run is a separate simulator process."
        )

    config = SweepConfig(netlist=netlist_path, dimensions=dimensions)
    config_id = generate_config_id("sweep")
    state.sweep_configs[config_id] = config

    logger.info(
        f"Sweep configured: config_id={config_id}, netlist={netlist_path.name}, "
        f"dimensions={len(dimensions)}, total_runs={total_runs}"
    )

    lines = [
        "Sweep configured",
        f"Config ID: {config_id}",
        f"Netlist: {netlist_path}",
        f"Dimensions: {len(dimensions)}",
        f"Total simulations: {total_runs}",
    ]
    for name, values in dim_values:
        if len(values) <= 12:
            preview = ", ".join(f"{v:g}" for v in values)
        else:
            preview = (
                ", ".join(f"{v:g}" for v in values[:6])
                + f", … ({len(values)} points) …, "
                + ", ".join(f"{v:g}" for v in values[-2:])
            )
        lines.append(f"  {name}: [{preview}]")
    for warn in _ngspice_preflight_warnings(netlist_path, state):
        lines.append(f"\n⚠ {warn}")
    lines.append(f"\nUse run_sweep('{config_id}') to execute")
    return text_response("\n".join(lines))


# ---------------------------------------------------------------------------
# Handler 2: run_sweep
# ---------------------------------------------------------------------------
@registry.tool(
    name="run_sweep",
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
async def handle_run_sweep(args: RunBatchInput, state: SessionState):
    """Start a previously configured parameter sweep.

    Looks up the sweep config, creates a BatchJob, and starts execution
    asynchronously. Returns the job ID immediately — never blocks.

    Args:
        args: Tool args with config_id and optional max_parallel
        state: Current session state

    Returns:
        TextContent with job ID for monitoring
    """
    config_id = args.config_id
    max_parallel = args.max_parallel

    # Look up config
    config = state.sweep_configs.get(config_id)
    if not config:
        raise BatchJobError(
            f"Sweep config not found: {config_id}\n\n"
            f"Use configure_sweep() to create a sweep configuration first"
        )

    require_simulator(state)

    # Compute total runs
    dim_sizes = []
    for dim in config.dimensions:
        values = dim.resolved_values()
        dim_sizes.append(len(values))
    total_runs = prod(dim_sizes) if dim_sizes else 0

    job_id = generate_batch_job_id("sweep")
    batch_job = BatchJob(
        job_id=job_id,
        job_type="sweep",
        netlist=config.netlist,
        total_runs=total_runs,
        sweep_config=config,
    )

    default_simulator = state.default_simulator
    assert default_simulator is not None  # guaranteed by require_simulator above
    # Runner first, then register + create_task with no await between —
    # submit-ordering rule, see the concurrency contract in tools/_base.py.
    runner = state.runners.get_sweep_runner(
        loop=asyncio.get_running_loop(),
        simulator_class=default_simulator,
        output_folder=await resolve_output_folder(state, config.netlist),
        max_parallel=max_parallel or state.config.max_parallel_sims,
    )
    state.add_batch_job(batch_job)
    batch_job.task = asyncio.create_task(runner.start_sweep(batch_job, state))

    logger.info(
        f"Sweep job started: job_id={job_id}, config_id={config_id}, total_runs={total_runs}"
    )

    return text_response(
        f"Sweep started\n"
        f"Job ID: {job_id}\n"
        f"Total runs: {total_runs}\n\n"
        f"Use batch_results('{job_id}') to monitor progress\n"
        f"Use batch_results('{job_id}', signal='...') to query results"
    )


# ---------------------------------------------------------------------------
# Handler 3: configure_montecarlo
# ---------------------------------------------------------------------------
@registry.tool(
    name="configure_montecarlo",
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
    profiles=("full", "agentic"),
)
async def handle_configure_montecarlo(args: ConfigureMonteCarloInput, state: SessionState):
    """Configure a Monte Carlo analysis and store it for later execution.

    Parses tolerances with type-name-to-prefix mapping, validates distribution
    names, and stores MonteCarloConfig in session state.

    Args:
        args: Tool args with netlist, tolerances, num_runs
        state: Current session state

    Returns:
        TextContent with config ID and summary
    """
    netlist_str = args.netlist
    tolerances_list = args.tolerances
    model_tolerances_input = args.model_tolerances
    mismatch_input = args.mismatch
    param_tolerances_input = args.param_tolerances
    num_runs = int(args.num_runs)

    netlist_path = resolve_runnable_netlist(netlist_str, state)

    has_any_rule = bool(
        tolerances_list or model_tolerances_input or mismatch_input or param_tolerances_input
    )
    if not has_any_rule:
        raise BatchJobError(
            "At least one tolerance rule is required (tolerances, model_tolerances, "
            "mismatch, or param_tolerances)."
        )

    if num_runs < 1 or num_runs > MAX_BATCH_RUNS:
        raise BatchJobError(f"num_runs must be 1-{MAX_BATCH_RUNS}, got {num_runs}")

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

    model_tolerances: list[ModelTolerance] = []
    for mt in model_tolerances_input:
        if not mt.parameters:
            raise BatchJobError(
                f"model_tolerances entry for '{mt.model}': parameters dict must be non-empty"
            )
        params: dict[str, ToleranceSpec] = {}
        for param_name, spec_input in mt.parameters.items():
            params[param_name.upper()] = _spec_from_input(
                spec_input.tolerance, spec_input.distribution, spec_input.kind
            )
        model_tolerances.append(ModelTolerance(model_name=mt.model, parameters=params))

    mismatch_rules: list[MismatchRule] = []
    for mr in mismatch_input:
        if mr.AVT == 0.0 and mr.AK == 0.0:
            raise BatchJobError(
                f"mismatch entry for prefix '{mr.prefix}': at least one of "
                "AVT or AK must be non-zero (mismatch coefficients are technology-"
                "specific; no defaults are provided)."
            )
        normalized = _DISTRIBUTION_MAP.get(mr.distribution.lower())
        if normalized is None:
            raise BatchJobError(
                f"mismatch entry for prefix '{mr.prefix}': unknown distribution {mr.distribution!r}"
            )
        mismatch_rules.append(
            MismatchRule(
                prefix=mr.prefix,
                avt=float(mr.AVT),
                ak=float(mr.AK),
                distribution=normalized,  # type: ignore[arg-type]
                vth_param=mr.vth_param,
                k_param=mr.k_param,
                min_wl_um2=float(mr.min_wl_um2),
            )
        )

    param_tolerances: list[ParamTolerance] = []
    for pt in param_tolerances_input:
        param_tolerances.append(
            ParamTolerance(
                name=pt.name,
                spec=_spec_from_input(pt.tolerance, pt.distribution, pt.kind),
            )
        )

    config = MonteCarloConfig(
        netlist=netlist_path,
        type_tolerances=type_tolerances,
        component_overrides=component_overrides,
        num_runs=num_runs,
        seed=args.seed,
        model_tolerances=model_tolerances,
        mismatch_rules=mismatch_rules,
        param_tolerances=param_tolerances,
    )
    config_id = generate_config_id("mc")
    state.mc_configs[config_id] = config

    def _summary(items, formatter) -> str:
        return ", ".join(formatter(x) for x in items) if items else "none"

    type_summary = _summary(
        type_tolerances.items(), lambda kv: f"{kv[0]}: {kv[1][0] * 100:.1f}% {kv[1][1]}"
    )
    component_summary = _summary(
        component_overrides.items(),
        lambda kv: f"{kv[0]}: {kv[1][0] * 100:.1f}% {kv[1][1]}",
    )
    model_summary = _summary(
        model_tolerances,
        lambda mt: f"{mt.model_name}({', '.join(mt.parameters.keys())})",
    )
    mismatch_summary = _summary(
        mismatch_rules, lambda r: f"{r.prefix}(AVT={r.avt:.2g}, AK={r.ak:.2g})"
    )
    param_summary = _summary(param_tolerances, lambda p: p.name)

    logger.info(
        f"Monte Carlo configured: config_id={config_id}, netlist={netlist_path.name}, "
        f"num_runs={num_runs}, .MODEL rules={len(model_tolerances)}, "
        f"mismatch rules={len(mismatch_rules)}, .PARAM rules={len(param_tolerances)}"
    )

    # Validate per-component overrides against the netlist so an unmatched ref
    # (e.g. C99) is flagged here rather than silently perturbing nothing and
    # understating the variation with no signal to the user.
    warnings: list[str] = []
    if component_overrides:
        netlist_refs = _netlist_component_refs(netlist_path)
        if netlist_refs:
            unmatched = [r for r in component_overrides if r.upper() not in netlist_refs]
            if unmatched:
                warnings.append(
                    f"Component override(s) {unmatched} match no component in the "
                    f"netlist — they will perturb nothing and the variation will be "
                    f"understated. Check the reference designators."
                )
    warnings.extend(_ngspice_preflight_warnings(netlist_path, state))

    text = (
        f"Monte Carlo configured\n"
        f"Config ID: {config_id}\n"
        f"Netlist: {netlist_path}\n"
        f"Runs: {num_runs}\n"
        f"Type-level tolerances (type names e.g. R/resistors): {type_summary}\n"
        f"Per-component tolerances (refs e.g. R1): {component_summary}\n"
        f".MODEL process variation: {model_summary}\n"
        f"Mismatch (Pelgrom): {mismatch_summary}\n"
        f".PARAM perturbation: {param_summary}\n"
        f"Seed: {args.seed if args.seed is not None else 'fresh entropy'}\n"
    )
    for warn in warnings:
        text += f"\n⚠ {warn}\n"
    text += f"\nUse run_montecarlo('{config_id}') to execute"
    return text_response(text)


# ---------------------------------------------------------------------------
# Handler 4: run_montecarlo
# ---------------------------------------------------------------------------
@registry.tool(
    name="run_montecarlo",
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
async def handle_run_montecarlo(args: RunBatchInput, state: SessionState):
    """Start a previously configured Monte Carlo analysis.

    Looks up the MC config, creates a BatchJob, and starts execution
    asynchronously. Returns the job ID immediately — never blocks.

    Args:
        args: Tool args with config_id and optional max_parallel
        state: Current session state

    Returns:
        TextContent with job ID for monitoring
    """
    config_id = args.config_id
    max_parallel = args.max_parallel

    # Look up config
    config = state.mc_configs.get(config_id)
    if not config:
        raise BatchJobError(
            f"Monte Carlo config not found: {config_id}\n\n"
            f"Use configure_montecarlo() to create a Monte Carlo configuration first"
        )

    require_simulator(state)

    job_id = generate_batch_job_id("mc")
    batch_job = BatchJob(
        job_id=job_id,
        job_type="montecarlo",
        netlist=config.netlist,
        total_runs=config.num_runs,
        mc_config=config,
    )

    default_simulator = state.default_simulator
    assert default_simulator is not None  # guaranteed by require_simulator above
    # Runner first, then register + create_task with no await between —
    # submit-ordering rule, see the concurrency contract in tools/_base.py.
    runner = state.runners.get_mc_runner(
        loop=asyncio.get_running_loop(),
        simulator_class=default_simulator,
        output_folder=await resolve_output_folder(state, config.netlist),
        max_parallel=max_parallel or state.config.max_parallel_sims,
    )
    state.add_batch_job(batch_job)
    batch_job.task = asyncio.create_task(runner.start_montecarlo(batch_job, state))

    logger.info(
        f"Monte Carlo job started: job_id={job_id}, config_id={config_id}, "
        f"total_runs={config.num_runs}"
    )

    return text_response(
        f"Monte Carlo started\n"
        f"Job ID: {job_id}\n"
        f"Total runs: {config.num_runs}\n\n"
        f"Use batch_results('{job_id}') to monitor progress\n"
        f"Use batch_results('{job_id}', signal='...') to query results"
    )


# ---------------------------------------------------------------------------
# Handler 5: get_batch_results (consolidated: status + results)
# ---------------------------------------------------------------------------
@registry.tool(
    name="batch_results",
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
    output_schema={
        "type": "object",
        "properties": {
            "job_id": {"type": "string"},
            "job_type": {"type": "string"},
            "status": {"type": "string"},
            "netlist": {"type": "string"},
            "total_runs": {"type": "integer"},
            "completed_runs": {"type": "integer"},
            "failed_runs": {"type": "integer"},
            "mode": {"type": "string", "enum": ["aggregate", "raw"]},
            "signal": {"type": "string"},
            "run_count": {"type": "integer"},
            "stats": {
                "type": "object",
                "properties": {
                    "max_across_runs": {"type": ["number", "null"]},
                    "min_across_runs": {"type": ["number", "null"]},
                    "mean_across_runs": {"type": ["number", "null"]},
                    "std_across_runs": {"type": ["number", "null"]},
                    "median_across_runs": {"type": ["number", "null"]},
                },
            },
            "max_case_run": {"type": ["integer", "null"]},
            "min_case_run": {"type": ["integer", "null"]},
            "runs": {"type": "array", "items": {"type": "object"}},
            "pagination": PAGINATION_SCHEMA,
            "convergence_warnings": {
                "type": "array",
                "description": (
                    "Per-run convergence-fallback markers (Gmin stepping, "
                    "source stepping, etc.) detected in the per-run logs. "
                    "Present only when at least one run hit a fallback."
                ),
                "items": {
                    "type": "object",
                    "properties": {
                        "run_index": {"type": "integer"},
                        "markers": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    },
                    "required": ["run_index", "markers"],
                },
            },
        },
    },
)
async def handle_batch_results(args: GetBatchResultsInput, state: SessionState):
    """Query a batch simulation job — status/progress or signal results.

    Without signal: returns job status and progress (running/completed/failed).
    With signal: returns aggregate statistics or per-run data for that signal.

    Supports parameter filtering using SPICE notation (exact or range).

    Args:
        args: Tool args with job_id, optional signal, optional filters/pagination
        state: Current session state

    Returns:
        TextContent with status info or statistics/per-run data
    """
    job_id = args.job_id
    signal = args.signal
    fmt = args.format
    batch_job = services.resolve_batch_job(job_id, state)

    if signal is None:
        data = await services.get_batch_status(batch_job)
        return format_response(_format_batch_status_text(data), data, fmt)

    filters = args.filters
    offset = args.offset
    limit = min(args.limit, DEFAULT_PAGE_CAP)
    raw_mode = args.raw
    at_value: float | None = None
    if args.at is not None:
        try:
            at_value = parse_spice_value(args.at)
        except (ValueError, TypeError) as e:
            raise BatchJobError(
                f"Invalid 'at' value {args.at!r} — expected SPICE notation (e.g., '1k', '100u'): {e}"
            ) from e
    data = await services.get_batch_signal_data(
        batch_job,
        signal,
        filters=filters,
        raw=raw_mode,
        offset=offset,
        limit=limit,
        at=at_value,
        dialect=state.raw_dialect,
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
                f", ~{int(eta_s // 60)}m remaining"
                if eta_s >= 60
                else f", ~{int(eta_s)}s remaining"
            )
        return (
            f"Batch job {data['job_id']} is running\n"
            f"Type: {data['job_type']}\n"
            f"Progress: {data['completed']}/{data['total']} runs complete{eta_str}\n"
            f"Failed: {data['failed']}\n"
            f"Netlist: {data['netlist']}\n\n"
            f"Use batch_results('{data['job_id']}', signal='...') to query partial results"
        )
    if status == "completed":
        duration = data["duration"] or 0.0
        text = (
            f"Batch job {data['job_id']} completed\n"
            f"Type: {data['job_type']}\n"
            f"Total runs: {data['total_runs']}\n"
            f"Successful: {data['successful']}\n"
            f"Failed: {data['failed_runs']}\n"
            f"Duration: {duration:.1f}s"
        )
        flagged = data.get("convergence_warnings") or []
        if flagged:
            run_ids = ", ".join(str(f["run_index"]) for f in flagged[:10])
            more = "" if len(flagged) <= 10 else f", … (+{len(flagged) - 10} more)"
            text += (
                f"\n\nWarning: {len(flagged)} of {data['total_runs']} run(s) hit "
                f"convergence fallbacks (Gmin/source stepping or worse) — bias "
                f"point may be degenerate. Run indices: {run_ids}{more}"
            )
        return text + (
            f"\n\nUse batch_results('{data['job_id']}', signal='V(out)') to query results"
        )
    if status == "failed":
        return (
            f"Batch job {data['job_id']} failed\n"
            f"Type: {data['job_type']}\n"
            f"Netlist: {data['netlist']}\n"
            f"Error: {data.get('error') or 'Unknown error'}"
        )
    if status == "cancelled":
        return (
            f"Batch job {data['job_id']} was cancelled\n"
            f"Type: {data['job_type']}\n"
            f"Completed {data['completed_runs']} of {data['total_runs']} before cancellation. "
            f"Partial results available via get_batch_results."
        )
    if status == "interrupted":
        # Terminal status assigned on restart recovery when the owning server
        # stopped mid-batch (job_registry). Treat like cancelled — surface the
        # partial results instead of raising "unexpected status".
        return (
            f"Batch job {data['job_id']} was interrupted\n"
            f"Type: {data['job_type']}\n"
            f"The server stopped while this batch was running; "
            f"{data['completed_runs']} of {data['total_runs']} run(s) completed before the "
            f"interruption. Partial results available via get_batch_results."
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

    max_run = data["max_case_run"]
    if max_run is not None:
        max_params = batch_job.run_results[max_run].get("params", {})
        params_str = (
            ", ".join(f"{k}={v}" for k, v in max_params.items()) if max_params else "no params"
        )
        lines.append(f"\nHighest-peak run: #{max_run} ({params_str})")

    min_run = data["min_case_run"]
    if min_run is not None:
        min_params = batch_job.run_results[min_run].get("params", {})
        params_str = (
            ", ".join(f"{k}={v}" for k, v in min_params.items()) if min_params else "no params"
        )
        lines.append(f"Lowest-peak run:  #{min_run} ({params_str})")

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
        # Runs sliced to a single sample (``at=``/.op-style or an exactly
        # constant waveform) collapse to one ``value`` key; render it in all
        # three columns rather than a row of N/A that hides the data the
        # ``at`` slice just computed.
        value = run_summary.get("value")
        peak = run_summary.get("peak", value)
        mean = run_summary.get("mean", value)
        low = run_summary.get("min", value)
        lines.append(
            f"{run_idx:<6} {_fmt_col(peak)} {_fmt_col(mean)} {_fmt_col(low)}  {params_str}"
        )

    pagination = data["pagination"]
    if pagination["has_more"]:
        lines.append(
            f"\nNext page: batch_results('{data['job_id']}', signal='{data['signal']}', raw=true, offset={pagination['next_offset']})"
        )

    return "\n".join(lines)
