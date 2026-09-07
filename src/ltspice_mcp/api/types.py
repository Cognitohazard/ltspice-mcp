"""Argument models the six operations validate against, importable by name.

The engine validates every call against these pydantic models, so a rejection
names them: *"Input should be a valid dictionary or instance of RenderPolicy"*.
Before this module there was no way to act on that — the type the message named
could not be imported, and reaching it meant guessing at private modules. Here
they are, under the names the errors use, so an error is a pointer rather than a
dead end.

Nothing here is required to call the API: every operation takes plain dicts, and
that stays the ordinary spelling. These are for the two moments dicts are not
enough — following an error back to the shape it wanted, and having an editor or
a type checker validate a request before it is sent.

``Api.reference(op)`` renders the same models as flat, readable argument trees;
this module is the typed counterpart of that text. It is deliberately a separate
import surface from :mod:`ltspice_mcp.api`, whose ``__all__`` is a pinned
stability boundary.
"""

from __future__ import annotations

from ltspice_mcp.lib.recipes import (
    AcStructureRecipe,
    BodeCrossingRecipe,
    BodeFilterRecipe,
    BodePointRecipe,
    BodeSlopeRecipe,
    EdgesRecipe,
    Levels,
    MeasurementsRecipe,
    NoiseIntegralRecipe,
    OperatingPointRecipe,
    PeriodicRecipe,
    PlotRecipe,
    PlotSpan,
    Recipe,
    ResonanceRecipe,
    ReturnLossRecipe,
    SignalStatsRecipe,
    SpecLimits,
    StabilityRecipe,
    StepSelector,
    SummaryRecipe,
    ThdRecipe,
    TimingEndpoint,
    TimingRecipe,
    TransientResponseRecipe,
    ValueRecipe,
    WaveformRecipe,
    Window,
)
from ltspice_mcp.lib.schematic_ops import (
    OpAddComponent,
    OpAddDirective,
    OpAddNetLabel,
    OpMoveComponent,
    OpRemoveComponent,
    OpRemoveDirective,
    OpRemoveNetLabel,
    OpRemoveWire,
    OpSetComponentAttribute,
    OpSetComponentValue,
    OpWirePins,
)
from ltspice_mcp.lib.variations import (
    AssignVariation,
    ComponentRule,
    MismatchRule,
    ModelRule,
    ParamRule,
    RandomRule,
    RandomVariation,
    Variation,
)
from ltspice_mcp.tools.analyze import (
    AnalyzeInclude,
    AnalyzeResultsInput,
    AnalyzeSourceInput,
    CaseSelection,
    ContinueInput,
    PerRunInclude,
)
from ltspice_mcp.tools.experiments import (
    AnalysisInclude,
    AnalysisPerRun,
    AttachedAnalysis,
    ExperimentCircuit,
    ExperimentExecution,
    RunExperimentsInput,
)
from ltspice_mcp.tools.inspect_tools import (
    CapabilitiesQuery,
    ComponentsQuery,
    InspectInput,
    ModelQuery,
    NetQuery,
    Query,
    SymbolQuery,
    SymbolsQuery,
)
from ltspice_mcp.tools.jobs import JobsInput
from ltspice_mcp.tools.schematic_edit import (
    ConsolidatedOp,
    EditSchematicInput,
    EditViewCursors,
)
from ltspice_mcp.tools.verify import (
    CompareSpec,
    RenderPolicy,
    VerifyCircuitInput,
    VerifyCompareSpec,
    VerifyRenderPolicy,
)

# The schematic op models are the ones lib/schematic_ops.py defines and the
# applier dispatches on. Aliased rather than re-declared: one class per op, so
# an isinstance check and a validation error agree about what a caller built.
AddComponentOp = OpAddComponent
AddDirectiveOp = OpAddDirective
AddNetLabelOp = OpAddNetLabel
MoveComponentOp = OpMoveComponent
RemoveComponentOp = OpRemoveComponent
RemoveDirectiveOp = OpRemoveDirective
RemoveNetLabelOp = OpRemoveNetLabel
RemoveWireOp = OpRemoveWire
SetComponentAttributeOp = OpSetComponentAttribute
SetComponentValueOp = OpSetComponentValue
WirePinsOp = OpWirePins
ViewCursors = EditViewCursors

__all__ = [  # noqa: RUF022 - grouped by the operation that takes them
    # The six operations' top-level argument models.
    "AnalyzeResultsInput",
    "EditSchematicInput",
    "InspectInput",
    "JobsInput",
    "RunExperimentsInput",
    "VerifyCircuitInput",
    # The render and compare policies. verify_circuit takes SUBCLASSES of the
    # shared pair (it has checks to skip and an image channel to deliver into,
    # and two comparison modes); edit_schematic takes the shared pair itself. A
    # parent instance is not a child instance, so both spellings are exported —
    # passing the wrong one is a validation error, not a widening.
    "RenderPolicy",
    "CompareSpec",
    "VerifyRenderPolicy",
    "VerifyCompareSpec",
    # analyze_results
    "AnalyzeInclude",
    "AnalyzeSourceInput",
    "CaseSelection",
    "ContinueInput",
    "PerRunInclude",
    # analyze_results recipes (the discriminated union and its members)
    "Recipe",
    "AcStructureRecipe",
    "BodeCrossingRecipe",
    "BodeFilterRecipe",
    "BodePointRecipe",
    "BodeSlopeRecipe",
    "EdgesRecipe",
    "MeasurementsRecipe",
    "NoiseIntegralRecipe",
    "OperatingPointRecipe",
    "PeriodicRecipe",
    "PlotRecipe",
    "ResonanceRecipe",
    "ReturnLossRecipe",
    "SignalStatsRecipe",
    "StabilityRecipe",
    "SummaryRecipe",
    "ThdRecipe",
    "TimingRecipe",
    "TransientResponseRecipe",
    "ValueRecipe",
    "WaveformRecipe",
    # shared recipe fragments
    "Levels",
    "PlotSpan",
    "SpecLimits",
    "StepSelector",
    "TimingEndpoint",
    "Window",
    # run_experiments
    "AnalysisInclude",
    "AnalysisPerRun",
    "AttachedAnalysis",
    "ExperimentCircuit",
    "ExperimentExecution",
    # run_experiments variations
    "Variation",
    "AssignVariation",
    "RandomVariation",
    "RandomRule",
    "ComponentRule",
    "MismatchRule",
    "ModelRule",
    "ParamRule",
    # inspect
    "Query",
    "CapabilitiesQuery",
    "ComponentsQuery",
    "ModelQuery",
    "NetQuery",
    "SymbolQuery",
    "SymbolsQuery",
    # edit_schematic
    "ConsolidatedOp",
    "AddComponentOp",
    "AddDirectiveOp",
    "AddNetLabelOp",
    "MoveComponentOp",
    "RemoveComponentOp",
    "RemoveDirectiveOp",
    "RemoveNetLabelOp",
    "RemoveWireOp",
    "SetComponentAttributeOp",
    "SetComponentValueOp",
    "WirePinsOp",
    "ViewCursors",
]
