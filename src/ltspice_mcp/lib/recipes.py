"""Strict recipe models for the consolidated result-analysis surface.

Each discriminant is a sealed Pydantic model.  Reduction fields live only on
the category that can interpret them, so an unsupported ``reduce``, ``spec``,
or ``field`` is rejected by schema validation instead of being ignored.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, TypeAlias, get_args

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    model_validator,
)

from ltspice_mcp.errors import compact_validation_error

ReduceStat = Literal["min", "max", "mean", "stddev", "p50", "p90", "count"]

REDUCIBLE_FIELDS: dict[str, tuple[str, ...]] = {
    "signal_stats": ("min", "max", "mean", "rms", "peak_to_peak", "stddev"),
    "edges": ("rise_time", "fall_time", "edges_found"),
    "timing": ("delay", "from_time", "to_time"),
    "periodic": ("period", "frequency", "duty_cycle"),
    "transient_response": (
        "overshoot_pct",
        "undershoot_pct",
        "settling_time",
        "final_value",
        "peak_value",
        "peak_time",
        "deviation",
        "recovery_time",
        "undershoot",
        "overshoot",
    ),
    "bode_filter": (
        "passband_gain_db",
        "passband_ripple_db",
        "cutoff_low_hz",
        "cutoff_high_hz",
        "stopband_rejection_db",
        "transition_bandwidth_hz",
        "rolloff_slope_db_per_decade",
        "estimated_order",
    ),
    # unity_gain_hz is the headline leaf analyze promotes from the crossover
    # list at row-build time (tools/analyze.py _HEADLINE_LEAVES), so a reduce or
    # spec reads the same number the row shows.
    "stability": ("phase_margin_deg", "gain_margin_db", "unity_gain_hz", "dc_gain_db"),
    "return_loss": ("return_loss_db", "vswr", "reflection_coefficient"),
}

# Which transient_response fields each mode can reduce. A step response and a
# disturbance response measure different quantities off the same trace, so the
# reducible field must match the mode.
TRANSIENT_FIELDS_BY_MODE: dict[str, frozenset[str]] = {
    "step": frozenset(
        {
            "overshoot_pct",
            "undershoot_pct",
            "settling_time",
            "final_value",
            "peak_value",
            "peak_time",
        }
    ),
    "disturbance": frozenset(
        {
            "deviation",
            "recovery_time",
            "undershoot",
            "overshoot",
        }
    ),
}


class StrictModel(BaseModel):
    """Tool-independent strict model base for recipe schemas."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        validate_assignment=True,
    )


# Appended to every dormant-branch summary so each stub names both discovery
# channels; the helper owning it makes the pointer structural, not a
# per-call-site convention.
_DORMANT_POINTER = " Full arguments: api.reference('analyze_results') or spice://guide."


def _dormant_wire_stub(summary: str) -> ConfigDict:
    """Advertise a recipe branch as its discriminant plus a one-line pointer.

    For branches no recorded workload has ever called: the advertised schema
    shrinks to ``{"metric": <const>}`` and ``summary`` (which must name the
    produced fields; the full-argument pointer is appended here), while the
    model itself — validation, execution, and the ``api.reference`` catalogue,
    which walks model fields rather than this schema — keeps every field. The
    stub deliberately drops ``additionalProperties: false`` so a client
    pre-validating a full call against the wire shape still sends it.
    """
    description = summary + _DORMANT_POINTER

    def _stub(schema: dict[str, Any]) -> None:
        metric = schema["properties"]["metric"]
        schema.clear()
        schema.update(
            {
                "type": "object",
                "description": description,
                "properties": {"metric": metric},
                "required": ["metric"],
            }
        )

    return ConfigDict(json_schema_extra=_stub)


class Window(StrictModel):
    """Optional domain window, in seconds or hertz as appropriate."""

    start: float | str | None = None
    end: float | str | None = None


class StepSelector(StrictModel):
    """Select one inner ``.step`` result by axis value."""

    axis: str
    value: float | str


class SpecLimits(StrictModel):
    """Caller-declared limits used to count a recipe's scalar samples.

    Which number the limits apply to is the recipe's own ``field``, not a
    second copy here: a reduction and a spec on one recipe always read the
    same number, and two spellings of that could only ever agree or be
    refused.
    """

    min: float | None = None
    max: float | None = None
    allow_incomplete: bool = False

    @model_validator(mode="after")
    def _has_limit(self) -> SpecLimits:
        if self.min is None and self.max is None:
            raise ValueError("spec requires at least one of 'min' or 'max'")
        if self.min is not None and self.max is not None and self.min > self.max:
            raise ValueError("spec.min must be less than or equal to spec.max")
        return self


class RecipeBase(StrictModel):
    """Fields every recipe carries.

    ``.step`` selection is deliberately absent: which step iteration a call
    reads is one choice for the whole call, so it lives on
    ``analyze_results``' own ``step``/``all_steps`` arguments (and on the
    attached-analysis block) rather than being restated on each of the
    twenty-one recipes.
    """

    key: str = Field(min_length=1)
    sources: list[str] | None = None

    @model_validator(mode="after")
    def _unique_sources(self) -> RecipeBase:
        if self.sources is not None and (
            not self.sources or len(set(self.sources)) != len(self.sources)
        ):
            raise ValueError("sources must be a non-empty list of unique labels")
        return self


class ScalarRecipe(RecipeBase):
    reduce: list[ReduceStat] = Field(default_factory=list)
    spec: SpecLimits | None = None


class MultiRecipe(RecipeBase):
    reduce: list[ReduceStat] = Field(default_factory=list)
    field: str | None = Field(
        default=None,
        description=(
            "Which of this recipe's numbers a 'reduce' or a 'spec' reads (e.g. "
            "'phase_margin_deg'); required as soon as either is given."
        ),
    )
    spec: SpecLimits | None = None

    @model_validator(mode="after")
    def _field_for_cross_run_work(self) -> MultiRecipe:
        wants_reduction = bool(self.reduce) or self.spec is not None
        if wants_reduction and self.field is None:
            raise ValueError(
                "this recipe returns multiple fields; set 'field' to the one "
                "the reduction or spec should read"
            )
        if self.field is not None:
            # ``metric`` is the Literal discriminant every concrete subclass
            # sets; this abstract base doesn't declare it, so read it dynamically.
            metric: str = getattr(self, "metric")  # noqa: B009
            fields = REDUCIBLE_FIELDS.get(metric, ())
            if self.field not in fields:
                raise ValueError(
                    f"{metric!r} does not produce reducible field {self.field!r}; "
                    f"choose one of: {', '.join(fields)}"
                )
        return self


class KeyedRecipe(RecipeBase):
    reduce: list[ReduceStat] = Field(default_factory=list)
    field: str | None = Field(
        default=None,
        description=(
            "Which key of this recipe's map a 'reduce' or a 'spec' reads — a "
            ".meas name, or a bias-point quantity. Required when 'spec' is "
            "given; without it, 'reduce' covers every key."
        ),
    )
    spec: SpecLimits | None = None

    @model_validator(mode="after")
    def _spec_names_key(self) -> KeyedRecipe:
        if self.spec is not None and not self.field:
            raise ValueError("set 'field' to the key this recipe's spec applies to")
        return self


class VariableRecipe(RecipeBase):
    """Base for recipes whose per-run value has no cross-run reduction.

    These carry no ``reduce``/``spec``/``field`` — an unsupported one is
    rejected as an extra field by ``extra="forbid"``. Marking them with a
    dedicated base makes every recipe inherit exactly one category base, so the
    reducer categorization (isinstance against these four bases) cannot be
    missed.
    """


class SummaryRecipe(VariableRecipe):
    metric: Literal["summary"]


class MeasurementsRecipe(KeyedRecipe):
    metric: Literal["measurements"]
    names: list[str] | None = None
    histogram_bins: int = Field(
        default=0,
        ge=0,
        description=(
            "Bin count for a distribution histogram over each .MEAS name's "
            "per-run values (the shape a Monte Carlo spread is read from); "
            "0, the default, computes none."
        ),
    )


class ValueRecipe(ScalarRecipe):
    metric: Literal["value"]
    expr: str
    at: float | str | None = None


class SignalStatsRecipe(MultiRecipe):
    metric: Literal["signal_stats"]
    signal: str
    window: Window | None = None


class Levels(StrictModel):
    low: float | None = None
    high: float | None = None


class EdgesRecipe(MultiRecipe):
    metric: Literal["edges"]
    signal: str
    levels: Levels | None = None
    edge: Literal["rising", "falling", "auto"] = "auto"
    window: Window | None = None


class TimingEndpoint(StrictModel):
    signal: str
    edge: Literal["rising", "falling"] = "rising"
    level: float | None = None


class TimingRecipe(MultiRecipe):
    metric: Literal["timing"]
    from_: TimingEndpoint = Field(alias="from")
    to: TimingEndpoint
    nth: int = Field(default=1, ge=1)
    window: Window | None = None


class PeriodicRecipe(MultiRecipe):
    metric: Literal["periodic"]
    signal: str
    window: Window | None = None

    model_config = _dormant_wire_stub(
        "Period, frequency, and duty_cycle of a repetitive .tran signal."
    )


class TransientResponseRecipe(MultiRecipe):
    metric: Literal["transient_response"]
    signal: str
    mode: Literal["step", "disturbance"]
    input: str | None = None
    window: Window | None = None

    @model_validator(mode="after")
    def _disturbance_input(self) -> TransientResponseRecipe:
        if self.mode == "disturbance" and not self.input:
            raise ValueError("input is required when mode='disturbance'")
        if self.mode == "step" and self.input is not None:
            raise ValueError("input is accepted only when mode='disturbance'")
        if self.field is not None and self.field not in TRANSIENT_FIELDS_BY_MODE[self.mode]:
            choices = ", ".join(sorted(TRANSIENT_FIELDS_BY_MODE[self.mode]))
            raise ValueError(
                f"transient_response mode={self.mode!r} cannot reduce {self.field!r}; "
                f"choose one of: {choices}"
            )
        return self


class ThdRecipe(ScalarRecipe):
    metric: Literal["thd"]
    signal: str
    fundamental_hz: float | str | None = None
    harmonics: int = Field(default=7, ge=1, le=50)
    window: Window | None = None


class BodeFilterRecipe(MultiRecipe):
    metric: Literal["bode_filter"]
    signal: str


class BodePointRecipe(ScalarRecipe):
    metric: Literal["bode_point"]
    signal: str
    at_hz: float | str


class BodeCrossingRecipe(VariableRecipe):
    metric: Literal["bode_crossing"]
    signal: str
    level_db: float | None = Field(
        default=None,
        description="Magnitude level to cross, in dB; exactly one of level_db/level_deg.",
    )
    # Named for its axis the way level_db is, and reachable under the older
    # 'phase_deg' spelling. The pair reads as one choice of level on one of two
    # axes, which is what it is; 'phase_deg' alone gave no hint that the
    # magnitude axis was the other member of the same either-or.
    level_deg: float | None = Field(
        default=None,
        validation_alias=AliasChoices("level_deg", "phase_deg"),
        description=(
            "Phase level to cross, in degrees, scanned on the unwrapped phase "
            "so a crossing past ±180° is found once; exactly one of "
            "level_db/level_deg."
        ),
    )

    @model_validator(mode="after")
    def _one_crossing_level(self) -> BodeCrossingRecipe:
        if (self.level_db is None) == (self.level_deg is None):
            raise ValueError("provide exactly one of 'level_db' or 'level_deg'")
        return self


class BodeSlopeRecipe(ScalarRecipe):
    metric: Literal["bode_slope"]
    signal: str
    from_hz: float | str
    to_hz: float | str


class StabilityRecipe(MultiRecipe):
    metric: Literal["stability"]
    signal: str


class AcStructureRecipe(VariableRecipe):
    metric: Literal["ac_structure"]
    signal: str


class ResonanceRecipe(VariableRecipe):
    metric: Literal["resonance"]
    signal: str


class ReturnLossRecipe(MultiRecipe):
    metric: Literal["return_loss"]
    signal: str
    z0: float = Field(default=50.0, gt=0)

    model_config = _dormant_wire_stub(
        "return_loss_db, vswr, and reflection_coefficient vs frequency from an "
        ".AC impedance trace (z0 default 50)."
    )


class NoiseIntegralRecipe(ScalarRecipe):
    metric: Literal["noise_integral"]
    signal: str | None = None
    from_hz: float | str | None = None
    to_hz: float | str | None = None

    model_config = _dormant_wire_stub(
        "Integrated RMS noise from a .noise run, optionally windowed to [from_hz, to_hz]."
    )


class OperatingPointRecipe(KeyedRecipe):
    metric: Literal["operating_point"]
    device: str | None = Field(
        default=None,
        description=(
            "Return one device's small-signal params and terminal currents "
            "(e.g. 'M6') instead of the whole bias point — a few hundred bytes "
            "against tens of KB for a real opamp."
        ),
    )


class WaveformRecipe(VariableRecipe):
    metric: Literal["waveform"]
    signals: list[str] = Field(min_length=1, max_length=32)
    max_points: int = Field(default=2000, ge=1, le=2_000_000)
    format: Literal["inline", "csv"] = "inline"
    window: Window | None = None


class PlotSpan(StrictModel):
    start: float | str | None = None
    end: float | str | None = None


class PlotRecipe(VariableRecipe):
    metric: Literal["plot"]
    signals: list[str] = Field(min_length=1, max_length=32)
    title: str | None = None
    log_x: bool | None = None
    span: PlotSpan | None = None


Recipe: TypeAlias = Annotated[
    SummaryRecipe
    | MeasurementsRecipe
    | ValueRecipe
    | SignalStatsRecipe
    | EdgesRecipe
    | TimingRecipe
    | PeriodicRecipe
    | TransientResponseRecipe
    | ThdRecipe
    | BodeFilterRecipe
    | BodePointRecipe
    | BodeCrossingRecipe
    | BodeSlopeRecipe
    | StabilityRecipe
    | AcStructureRecipe
    | ResonanceRecipe
    | ReturnLossRecipe
    | NoiseIntegralRecipe
    | OperatingPointRecipe
    | WaveformRecipe
    | PlotRecipe,
    Field(discriminator="metric"),
]

RECIPE_ADAPTER = TypeAdapter(Recipe)
# The discriminated union is the single source of truth for the recipe set; the
# concrete models and their discriminants are read straight off it so there is
# no parallel list to keep in sync.
RECIPE_MODELS: tuple[type[RecipeBase], ...] = get_args(get_args(Recipe)[0])


def _discriminant_of(model: type[RecipeBase]) -> str:
    return get_args(model.model_fields["metric"].annotation)[0]


DISCRIMINANTS: tuple[str, ...] = tuple(sorted(_discriminant_of(model) for model in RECIPE_MODELS))
_DISCRIMINANT_SET: frozenset[str] = frozenset(DISCRIMINANTS)


def validate_recipe(data: Any) -> Recipe:
    """Validate one recipe and give unknown discriminants an actionable error."""
    if isinstance(data, dict):
        metric = data.get("metric")
        if metric not in _DISCRIMINANT_SET:
            raise ValueError(
                f"unsupported recipe metric {metric!r}; supported values: "
                f"{', '.join(DISCRIMINANTS)}"
            )
    return RECIPE_ADAPTER.validate_python(data)


def recipe_error(exc: ValueError | TypeError) -> str:
    """Compact one-item validation error suitable for the failures channel."""
    return compact_validation_error(exc)
