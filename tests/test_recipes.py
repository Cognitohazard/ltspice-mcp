"""Strict-schema coverage for the 21 analyze_results recipe discriminants."""

from __future__ import annotations

from typing import get_args

import pytest
from pydantic import ValidationError

from ltspice_mcp.lib.recipes import (
    DISCRIMINANTS,
    RECIPE_MODELS,
    BodeCrossingRecipe,
    KeyedRecipe,
    MultiRecipe,
    ScalarRecipe,
    SignalStatsRecipe,
    ValueRecipe,
    VariableRecipe,
    validate_recipe,
)

# A recipe's reducer category is which of the four bases it inherits (exactly
# one). The bases encode the accept/reject behavior the matrix below checks.
_CATEGORY_BASES = {
    "scalar": ScalarRecipe,
    "multi": MultiRecipe,
    "keyed": KeyedRecipe,
    "variable": VariableRecipe,
}


def _category_of(model: type) -> str:
    matches = [name for name, base in _CATEGORY_BASES.items() if issubclass(model, base)]
    assert len(matches) == 1, f"{model.__name__} inherits {matches}, expected exactly one base"
    return matches[0]


_METRIC_CATEGORY = sorted(
    (get_args(model.model_fields["metric"].annotation)[0], _category_of(model))
    for model in RECIPE_MODELS
)

VALID_RECIPES = {
    "summary": {},
    "measurements": {},
    "value": {"expr": "V(out)", "at": "1m"},
    "signal_stats": {"signal": "V(out)"},
    "edges": {"signal": "V(out)"},
    "timing": {
        "from": {"signal": "V(in)"},
        "to": {"signal": "V(out)"},
    },
    "periodic": {"signal": "V(out)"},
    "transient_response": {"signal": "V(out)", "mode": "step"},
    "thd": {"signal": "V(out)"},
    "bode_filter": {"signal": "V(out)"},
    "bode_point": {"signal": "V(out)", "at_hz": "1k"},
    "bode_crossing": {"signal": "V(out)", "level_db": -3.0},
    "bode_slope": {"signal": "V(out)", "from_hz": "1k", "to_hz": "10k"},
    "stability": {"signal": "V(loop)"},
    "ac_structure": {"signal": "V(out)"},
    "resonance": {"signal": "V(out)"},
    "return_loss": {"signal": "V(in)"},
    "noise_integral": {},
    "operating_point": {},
    "waveform": {"signals": ["V(out)"]},
    "plot": {"signals": ["V(out)"]},
}

MULTI_FIELDS = {
    "signal_stats": "mean",
    "edges": "rise_time",
    "timing": "delay",
    "periodic": "period",
    "transient_response": "settling_time",
    "bode_filter": "cutoff_low_hz",
    "stability": "phase_margin_deg",
    "return_loss": "return_loss_db",
}


@pytest.mark.parametrize("metric", DISCRIMINANTS)
def test_every_discriminant_validates(metric: str):
    recipe = validate_recipe({"key": metric, "metric": metric, **VALID_RECIPES[metric]})
    assert recipe.metric == metric
    assert recipe.model_config.get("extra") == "forbid"


def test_every_recipe_inherits_exactly_one_category_base():
    # Categorization is total and non-overlapping: each recipe model inherits
    # exactly one of the four reducer bases, so a missing category is impossible.
    for model in RECIPE_MODELS:
        _category_of(model)


@pytest.mark.parametrize(("metric", "category"), _METRIC_CATEGORY)
def test_category_accept_reject_matrix(metric: str, category: str):
    base = {"key": metric, "metric": metric, **VALID_RECIPES[metric]}
    if category == "variable":
        for unsupported in (
            {"reduce": ["mean"]},
            {"field": "value"},
            {"spec": {"min": 0}},
        ):
            with pytest.raises(ValidationError):
                validate_recipe({**base, **unsupported})
        return

    accepted: dict[str, object] = {**base, "reduce": ["mean"], "spec": {"min": 0}}
    if category == "keyed":
        accepted["field"] = "value"
    if category == "multi":
        accepted["field"] = MULTI_FIELDS[metric]
    validate_recipe(accepted)

    if category == "scalar":
        # One number, so there is nothing for 'field' to choose between.
        with pytest.raises(ValidationError):
            validate_recipe({**base, "field": "value"})


@pytest.mark.parametrize("metric", ["signal_stats", "measurements"])
def test_a_spec_no_longer_carries_its_own_field(metric: str):
    """One spelling: the recipe's 'field' names the number, never spec.field.

    Two spellings could only ever agree — the validator demanded they match —
    so the second one was a way to write the same call wrong."""
    base = {"key": metric, "metric": metric, **VALID_RECIPES[metric]}
    with pytest.raises(ValidationError, match="Extra inputs"):
        validate_recipe({**base, "spec": {"field": "mean", "min": 0}})


def test_a_multi_field_recipe_names_its_field_once_for_reduce_and_spec():
    """'field' drives both: one name, read by the reduction and by the spec."""
    recipe = validate_recipe(
        {
            "key": "stats",
            "metric": "signal_stats",
            "signal": "V(out)",
            "field": "mean",
            "reduce": ["max"],
            "spec": {"min": 0.0},
        }
    )
    assert isinstance(recipe, SignalStatsRecipe)
    assert recipe.field == "mean"
    with pytest.raises(ValidationError, match="set 'field'"):
        validate_recipe(
            {"key": "s", "metric": "signal_stats", "signal": "V(out)", "reduce": ["max"]}
        )


def test_stability_reduces_its_crossover_frequency_and_dc_gain():
    """ "Keep UGBW above 2 MHz" is the most natural stability spec after phase
    margin; the recipe reports unity_gain_hz per case, so refusing a spec or a
    reduce on it made a caller pull the rows and judge by hand."""
    base = {"key": "loop", "metric": "stability", "signal": "V(out)"}
    validate_recipe({**base, "field": "unity_gain_hz", "spec": {"min": 2e6}})
    validate_recipe({**base, "reduce": ["max"], "field": "dc_gain_db"})


@pytest.mark.parametrize("metric", ["summary", "waveform", "plot"])
def test_non_reducible_payload_recipes_reject_reduce_and_spec(metric: str):
    base = {"key": metric, "metric": metric, **VALID_RECIPES[metric]}
    with pytest.raises(ValidationError):
        validate_recipe({**base, "reduce": ["max"]})
    with pytest.raises(ValidationError):
        validate_recipe({**base, "spec": {"max": 1}})


@pytest.mark.parametrize("spelling", ["step", "all_steps"])
def test_step_selection_is_not_a_recipe_field(spelling: str):
    """Which .step iteration to read is one choice for the whole call.

    It lives on analyze_results (and on run_experiments' attached analysis),
    not restated on each of the twenty-one recipes."""
    value = {"axis": "R", "value": "1k"} if spelling == "step" else True
    with pytest.raises(ValidationError, match="Extra inputs"):
        validate_recipe(
            {"key": "stats", "metric": "signal_stats", "signal": "V(out)", spelling: value}
        )


def test_unknown_discriminant_lists_every_supported_value():
    with pytest.raises(ValueError, match="supported values") as caught:
        validate_recipe({"key": "bad", "metric": "not_a_metric"})
    message = str(caught.value)
    for metric in DISCRIMINANTS:
        assert metric in message


def test_disturbance_requires_reference_input():
    with pytest.raises(ValidationError, match="input is required"):
        validate_recipe(
            {
                "key": "load_step",
                "metric": "transient_response",
                "signal": "V(out)",
                "mode": "disturbance",
            }
        )


def test_value_at_is_optional_at_schema_boundary():
    recipe = validate_recipe({"key": "bias", "metric": "value", "expr": "V(out)"})
    assert isinstance(recipe, ValueRecipe)
    assert recipe.at is None


class TestBodeCrossingLevel:
    """One level, on one of two axes, named the same way on both."""

    @staticmethod
    def _crossing(**extra: object) -> BodeCrossingRecipe:
        recipe = validate_recipe(
            {"key": "x", "metric": "bode_crossing", "signal": "V(out)", **extra}
        )
        assert isinstance(recipe, BodeCrossingRecipe)
        return recipe

    def test_level_deg_is_accepted(self):
        assert self._crossing(level_deg=-45.0).level_deg == -45.0

    def test_level_db_is_accepted(self):
        recipe = self._crossing(level_db=-3.0)
        assert recipe.level_db == -3.0
        assert recipe.level_deg is None

    def test_phase_deg_still_reaches_the_same_field(self):
        assert self._crossing(phase_deg=-45.0).level_deg == -45.0

    def test_neither_level_is_refused_naming_both(self):
        with pytest.raises(ValidationError) as excinfo:
            self._crossing()
        message = str(excinfo.value)
        assert "level_db" in message
        assert "level_deg" in message

    def test_both_levels_are_refused_naming_both(self):
        with pytest.raises(ValidationError) as excinfo:
            self._crossing(level_db=-3.0, level_deg=-45.0)
        message = str(excinfo.value)
        assert "level_db" in message
        assert "level_deg" in message


def test_field_accepts_the_result_rows_own_key():
    """A stability row reports ``phase_margin_worst_deg``; passing that key back as
    'field' names the same number as the reducible spelling, so it is accepted
    and canonicalized rather than rejected as a second spelling."""
    base = {"key": "s", "metric": "stability", **VALID_RECIPES["stability"], "reduce": ["min"]}
    recipe = validate_recipe({**base, "field": "phase_margin_worst_deg"})
    assert isinstance(recipe, MultiRecipe) and recipe.field == "phase_margin_deg"
    with pytest.raises(ValidationError, match="does not produce reducible field"):
        validate_recipe({**base, "field": "phase_margin"})
    # Two reducible fields read the same row key: the key alone cannot choose.
    edges = {"key": "e", "metric": "edges", **VALID_RECIPES["edges"], "reduce": ["min"]}
    with pytest.raises(ValidationError, match="does not produce reducible field"):
        validate_recipe({**edges, "field": "transition_time"})


def test_field_dependency_rides_the_advertised_schema():
    """'field' is required once 'reduce' or 'spec' is given. The rule is stated
    in the schema itself (dependentRequired), so a listing that strips every
    description still carries it and a caller need not learn it from an error."""
    import json

    from ltspice_mcp.tools._schema import build_input_schema, strip_argument_descriptions
    from ltspice_mcp.tools.analyze import AnalyzeResultsInput

    schema = build_input_schema(AnalyzeResultsInput)
    multi = '"dependentRequired": {"reduce": ["field"], "spec": ["field"]}'
    keyed = '"dependentRequired": {"spec": ["field"]}'
    for text in (json.dumps(schema), json.dumps(strip_argument_descriptions(schema))):
        assert multi in text
        assert keyed in text
