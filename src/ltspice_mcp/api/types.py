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

# The schematic op models live module-private in tools/circuit.py and
# tools/schematic_edit.py, which is where the applier dispatches on them. Making
# them public means aliasing those names here, not copying the classes.
# pyright: reportPrivateUsage=false
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
from ltspice_mcp.tools.circuit import (
    _OpAddComponent,
    _OpAddDirective,
    _OpAddNetLabel,
    _OpMoveComponent,
    _OpRemoveComponent,
    _OpRemoveDirective,
    _OpRemoveNetLabel,
    _OpRemoveWire,
    _OpSetComponentAttribute,
    _OpSetComponentValue,
)
from ltspice_mcp.tools.experiments import (
    AnalysisInclude,
    AnalysisPerRun,
    AttachedAnalysis,
    ExperimentCircuit,
    ExperimentExecution,
    JobsInput,
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
from ltspice_mcp.tools.schematic_edit import (
    ConsolidatedOp,
    EditSchematicInput,
    _OpWirePinsStrict,
    _ViewCursors,
)
from ltspice_mcp.tools.verify import RenderPolicy, VerifyCircuitInput

# The schematic op models are shared verbatim with the shipped
# apply_schematic_ops surface, where they are module-private. Aliased rather
# than re-declared: one class per op, so an isinstance check and a validation
# error agree about what a caller constructed.
AddComponentOp = _OpAddComponent
AddDirectiveOp = _OpAddDirective
AddNetLabelOp = _OpAddNetLabel
MoveComponentOp = _OpMoveComponent
RemoveComponentOp = _OpRemoveComponent
RemoveDirectiveOp = _OpRemoveDirective
RemoveNetLabelOp = _OpRemoveNetLabel
RemoveWireOp = _OpRemoveWire
SetComponentAttributeOp = _OpSetComponentAttribute
SetComponentValueOp = _OpSetComponentValue
WirePinsOp = _OpWirePinsStrict
ViewCursors = _ViewCursors

__all__ = [  # noqa: RUF022 - grouped by the operation that takes them
    # The six operations' top-level argument models.
    "AnalyzeResultsInput",
    "EditSchematicInput",
    "InspectInput",
    "JobsInput",
    "RunExperimentsInput",
    "VerifyCircuitInput",
    # verify_circuit
    "RenderPolicy",
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
