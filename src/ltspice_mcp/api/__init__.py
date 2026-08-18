"""Public in-process Python API and curated analysis primitives.

Every public name resolves lazily (PEP 562): the engine's heavy imports
(scipy behind the signal primitives, the MCP SDK behind the session) cost
over a second, which a catalogue lookup — ``python -m ltspice_mcp.api
reference`` — must not pay. The no-boot property is pinned by a
cold-subprocess test asserting scipy/mcp stay out of ``sys.modules``.
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ltspice_mcp.api._exceptions import (
        ApiCallError,
        ApiClosedError,
        ApiError,
        ApiInternalError,
        ApiInterrupted,
        ApiSessionError,
        ApiValidationError,
    )
    from ltspice_mcp.api._primitives import RawResult
    from ltspice_mcp.api._session import Api
    from ltspice_mcp.lib.ac_analysis import (
        CrossingDirection,
        CrossingWithQuantity,
        Crossover,
        FilterMetricsOutput,
        FilterType,
        GainAtPoint,
        GainMargin,
        NoiseIntegralOutput,
        PhaseMargin,
        Quantity,
        ResonancePeak,
        ResonancesOutput,
        ReturnLossOutput,
        RollOffOutput,
        SearchDirection,
        StabilityLabel,
        StabilityMetricsOutput,
        classify_filter,
        compute_filter_metrics,
        compute_resonances,
        compute_return_loss,
        compute_roll_off,
        compute_stability_metrics,
        detect_crossings,
        find_crossings_any_quantity,
        gain_at_frequencies,
        integrate_noise,
        log_interp,
        log_interp_complex,
        prepare_ac_arrays,
        unwrap_phase_safe,
    )
    from ltspice_mcp.lib.ac_structure import (
        AcStructureResult,
        Corner,
        CornerKind,
        analyze_ac_structure,
    )
    from ltspice_mcp.lib.format import parse_spice_value
    from ltspice_mcp.lib.result_observations import Observation
    from ltspice_mcp.lib.signal_analysis import (
        DisturbanceResponseOutput,
        EdgeMetricsOutput,
        HarmonicEntry,
        HistogramBin,
        MeasurementStatsEntry,
        PeriodicMetricsOutput,
        PulseResponseOutput,
        SignalStatsOutput,
        ThdOutput,
        TimingBetweenOutput,
        analyze_disturbance_response,
        analyze_edge,
        analyze_periodic,
        analyze_pulse_response,
        analyze_thd,
        analyze_timing_between,
        compute_measurement_stats,
        compute_signal_stats,
        window_and_clean,
    )

__all__ = [  # noqa: RUF022 - grouped in the contract's published order
    "Api",
    "RawResult",
    "ApiError",
    "ApiCallError",
    "ApiSessionError",
    "ApiClosedError",
    "ApiInterrupted",
    "ApiInternalError",
    "ApiValidationError",
    "prepare_ac_arrays",
    "unwrap_phase_safe",
    "log_interp",
    "log_interp_complex",
    "detect_crossings",
    "find_crossings_any_quantity",
    "gain_at_frequencies",
    "compute_filter_metrics",
    "compute_stability_metrics",
    "compute_roll_off",
    "compute_resonances",
    "compute_return_loss",
    "integrate_noise",
    "classify_filter",
    "window_and_clean",
    "analyze_edge",
    "analyze_pulse_response",
    "analyze_disturbance_response",
    "analyze_timing_between",
    "analyze_periodic",
    "analyze_thd",
    "compute_signal_stats",
    "compute_measurement_stats",
    "analyze_ac_structure",
    # Variation values round-trip as SPICE literals ('5p'), so the code door
    # needs the same reader the wire door parses them with.
    "parse_spice_value",
    "Quantity",
    "SearchDirection",
    "CrossingDirection",
    "FilterType",
    "StabilityLabel",
    "CornerKind",
    "CrossingWithQuantity",
    "GainAtPoint",
    "ReturnLossOutput",
    "FilterMetricsOutput",
    "StabilityMetricsOutput",
    "Crossover",
    "PhaseMargin",
    "GainMargin",
    "RollOffOutput",
    "ResonancesOutput",
    "ResonancePeak",
    "NoiseIntegralOutput",
    "EdgeMetricsOutput",
    "PulseResponseOutput",
    "DisturbanceResponseOutput",
    "TimingBetweenOutput",
    "PeriodicMetricsOutput",
    "SignalStatsOutput",
    "ThdOutput",
    "HarmonicEntry",
    "MeasurementStatsEntry",
    "HistogramBin",
    "AcStructureResult",
    "Corner",
    "Observation",
]

# name -> defining module, resolved on first attribute access. Grouped by
# module; tests pin that this covers __all__ exactly, so a name added to one
# without the other fails the suite.
_SOURCES: dict[str, str] = {
    name: module
    for module, names in (
        (
            "ltspice_mcp.api._exceptions",
            (
                "ApiCallError",
                "ApiClosedError",
                "ApiError",
                "ApiInternalError",
                "ApiInterrupted",
                "ApiSessionError",
                "ApiValidationError",
            ),
        ),
        ("ltspice_mcp.api._primitives", ("RawResult",)),
        ("ltspice_mcp.api._session", ("Api",)),
        (
            "ltspice_mcp.lib.ac_analysis",
            (
                "CrossingDirection",
                "CrossingWithQuantity",
                "Crossover",
                "FilterMetricsOutput",
                "FilterType",
                "GainAtPoint",
                "GainMargin",
                "NoiseIntegralOutput",
                "PhaseMargin",
                "Quantity",
                "ResonancePeak",
                "ResonancesOutput",
                "ReturnLossOutput",
                "RollOffOutput",
                "SearchDirection",
                "StabilityLabel",
                "StabilityMetricsOutput",
                "classify_filter",
                "compute_filter_metrics",
                "compute_resonances",
                "compute_return_loss",
                "compute_roll_off",
                "compute_stability_metrics",
                "detect_crossings",
                "find_crossings_any_quantity",
                "gain_at_frequencies",
                "integrate_noise",
                "log_interp",
                "log_interp_complex",
                "prepare_ac_arrays",
                "unwrap_phase_safe",
            ),
        ),
        (
            "ltspice_mcp.lib.ac_structure",
            ("AcStructureResult", "Corner", "CornerKind", "analyze_ac_structure"),
        ),
        ("ltspice_mcp.lib.format", ("parse_spice_value",)),
        ("ltspice_mcp.lib.result_observations", ("Observation",)),
        (
            "ltspice_mcp.lib.signal_analysis",
            (
                "DisturbanceResponseOutput",
                "EdgeMetricsOutput",
                "HarmonicEntry",
                "HistogramBin",
                "MeasurementStatsEntry",
                "PeriodicMetricsOutput",
                "PulseResponseOutput",
                "SignalStatsOutput",
                "ThdOutput",
                "TimingBetweenOutput",
                "analyze_disturbance_response",
                "analyze_edge",
                "analyze_periodic",
                "analyze_pulse_response",
                "analyze_thd",
                "analyze_timing_between",
                "compute_measurement_stats",
                "compute_signal_stats",
                "window_and_clean",
            ),
        ),
    )
    for name in names
}


def __getattr__(name: str) -> Any:
    module_name = _SOURCES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    value = getattr(import_module(module_name), name)
    # Cache on the module so later access is a plain namespace hit.
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
