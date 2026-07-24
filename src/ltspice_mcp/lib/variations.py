"""Strict experiment-variation models and deterministic deck expansion."""

from __future__ import annotations

import fnmatch
import hashlib
import re
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Annotated, Any, Literal, TypeAlias

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictFloat,
    StrictInt,
    field_validator,
    model_validator,
)

from ltspice_mcp.errors import NetlistError
from ltspice_mcp.lib import atomic_write_text, component_value
from ltspice_mcp.lib.format import parse_spice_value
from ltspice_mcp.lib.montecarlo import (
    MCSampler,
    ToleranceSpec,
    extract_model_card,
    extract_mosfet_instances,
    inject_card_before_end,
    parse_model_params,
    parse_value,
    perturb_model_in_text,
    render_variant_model_card,
    rewrite_instance_model,
    sample_instance_mismatch,
    sample_model_perturbation,
    variant_model_name,
)
from ltspice_mcp.lib.montecarlo import MismatchRule as EngineMismatchRule
from ltspice_mcp.lib.spice_lex import SpiceCard, TokenKind, emit, lex, tokenize_body
from ltspice_mcp.lib.spice_lex_views import InstanceLine

ScalarValue: TypeAlias = StrictInt | StrictFloat | str
Distribution: TypeAlias = Literal["normal", "gaussian", "uniform"]
Scale: TypeAlias = Literal["relative", "absolute"]

_CIRCUIT_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")


class VariationError(ValueError):
    """A variation cannot be resolved or expanded without ambiguity."""

    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(message)


class VariationModel(BaseModel):
    """Strict base shared by every public variation branch."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        validate_assignment=True,
    )


def _validate_scalar(value: ScalarValue) -> ScalarValue:
    if isinstance(value, str) and not value:
        raise ValueError("SPICE assignment values cannot be empty")
    return value


class AssignVariation(VariationModel):
    """A deterministic grid or lock-step assignment family."""

    kind: Literal["assign"]
    id: str | None = None
    combine: Literal["grid", "zip"] = "grid"
    applies_to: list[str] | None = None
    assign: dict[str, list[ScalarValue]]

    @field_validator("assign")
    @classmethod
    def _validate_assignments(
        cls, assignments: dict[str, list[ScalarValue]]
    ) -> dict[str, list[ScalarValue]]:
        if not assignments:
            raise ValueError("assign must contain at least one target")
        for target, values in assignments.items():
            if not target.strip():
                raise ValueError("assignment targets cannot be empty")
            if not values:
                raise ValueError(f"assignment target {target!r} has no values")
            assignments[target] = [_validate_scalar(value) for value in values]
        return assignments

    @model_validator(mode="after")
    def _validate_zip_lengths(self) -> AssignVariation:
        if self.combine != "zip":
            return self
        lengths = {len(values) for values in self.assign.values()}
        if len(lengths) > 1:
            detail = ", ".join(f"{target}={len(values)}" for target, values in self.assign.items())
            raise ValueError(f"zip assignment lists must have equal lengths ({detail})")
        return self


class RandomRuleBase(VariationModel):
    target: str
    tolerance: float = Field(gt=0.0)
    scale: Scale = "relative"
    distribution: Distribution = "normal"

    @field_validator("target")
    @classmethod
    def _target_not_empty(cls, value: str) -> str:
        if not value:
            raise ValueError("random-rule target cannot be empty")
        return value


class ComponentRule(RandomRuleBase):
    rule: Literal["component"]


class ParamRule(RandomRuleBase):
    rule: Literal["param"]


class ModelRule(RandomRuleBase):
    rule: Literal["model"]
    param: str


class MismatchRule(VariationModel):
    """Pelgrom mismatch rule kept field-for-field with the shipped tool model."""

    rule: Literal["mismatch"]
    prefix: str = "M"
    AVT: float = 0.0
    AK: float = 0.0
    distribution: Distribution = "normal"
    vth_param: str = "VTO"
    k_param: str = "KP"
    min_wl_um2: float = Field(default=1e-3, gt=0.0)


RandomRule: TypeAlias = Annotated[
    ComponentRule | ParamRule | ModelRule | MismatchRule,
    Field(discriminator="rule"),
]


class RandomVariation(VariationModel):
    """One reproducible Monte Carlo family."""

    kind: Literal["random"]
    id: str | None = None
    runs: int = Field(ge=1)
    seed: int | None = None
    applies_to: list[str] | None = None
    rules: list[RandomRule]

    @field_validator("rules")
    @classmethod
    def _rules_not_empty(cls, value: list[RandomRule]) -> list[RandomRule]:
        if not value:
            raise ValueError("random variation rules cannot be empty")
        return value


Variation: TypeAlias = Annotated[
    AssignVariation | RandomVariation,
    Field(discriminator="kind"),
]


@dataclass(frozen=True)
class CircuitDeck:
    """One uniquely named circuit deck supplied to the expansion engine."""

    circuit_id: str
    path: Path
    text: str


@dataclass(frozen=True)
class ResolvedAssignment:
    """One assignment bound to a concrete deck edit."""

    target: str
    kind: Literal["param", "component", "model"]
    value: ScalarValue
    refs: tuple[str, ...] = ()


@dataclass
class ExpandedCase:
    """A stable pre-materialization case produced by variation expansion."""

    circuit_id: str
    case_index: int
    assignments: dict[str, Any]
    edits: tuple[ResolvedAssignment, ...] = ()
    random: RandomVariation | None = None
    random_index: int | None = None

    @property
    def case_id(self) -> str:
        return format_case_id(self.circuit_id, self.case_index)


@dataclass(frozen=True)
class MaterializedCase:
    """A concrete variant deck ready for an ExperimentCase record."""

    case_id: str
    circuit_id: str
    case_index: int
    path: Path
    text: str
    sha256: str
    assignments: dict[str, Any]


@dataclass(frozen=True)
class _DeckTargets:
    params: dict[str, str]
    components: dict[str, str]
    models_by_ref: dict[str, str]


def normalize_circuit_decks(circuits: list[CircuitDeck]) -> list[CircuitDeck]:
    """Validate circuit ids and preserve caller order."""
    seen: set[str] = set()
    normalized: list[CircuitDeck] = []
    for circuit in circuits:
        circuit_id = circuit.circuit_id
        if _CIRCUIT_ID_RE.fullmatch(circuit_id) is None:
            raise VariationError(
                "invalid_circuit_id",
                f"Circuit id {circuit_id!r} must be 1-64 letters, digits, underscores, "
                "or hyphens, starting with a letter or digit",
            )
        folded = circuit_id.casefold()
        if folded in seen:
            raise VariationError(
                "duplicate_circuit_id",
                f"Circuit id {circuit_id!r} is duplicated (ids are case-insensitive)",
            )
        seen.add(folded)
        normalized.append(circuit)
    if not normalized:
        raise VariationError("circuits_empty", "At least one circuit is required")
    return normalized


def validate_variation_circuit_ids(
    circuits: list[CircuitDeck],
    variations: list[Variation],
) -> None:
    """Reject applies_to ids that name no input circuit."""
    known = {circuit.circuit_id.casefold(): circuit.circuit_id for circuit in circuits}
    for variation in variations:
        for circuit_id in variation.applies_to or []:
            if circuit_id.casefold() not in known:
                raise VariationError(
                    "missing_circuit_id",
                    f"Variation {variation.id or variation.kind!r} applies_to unknown "
                    f"circuit id {circuit_id!r}",
                )


def assignment_family_size(variation: AssignVariation) -> int:
    """Return one assign entry's expansion size without reading a deck."""
    sizes = [len(values) for values in variation.assign.values()]
    if variation.combine == "zip":
        return sizes[0]
    count = 1
    for size in sizes:
        count *= size
    return count


def projected_case_count(circuit_id: str, variations: list[Variation]) -> int:
    """Count cases for one circuit without resolving assignment targets."""
    count = 1
    for variation in variations:
        if not _applies(variation, circuit_id):
            continue
        if isinstance(variation, AssignVariation):
            count *= assignment_family_size(variation)
        else:
            count *= variation.runs
    return count


def check_case_cap(total: int, max_cases: int) -> None:
    """Reject invalid or oversized experiment case counts."""
    if max_cases < 1:
        raise VariationError("case_cap", "max_cases must be at least 1")
    if total > max_cases:
        raise VariationError(
            "case_cap_exceeded",
            f"Variation expansion exceeds the configured maximum of {max_cases} experiment cases",
        )


def format_case_id(circuit_id: str, case_index: int) -> str:
    """Return the stable synthetic id for one expanded circuit case."""
    return f"{circuit_id}-case-{case_index:04d}"


def expand_variations(
    circuits: list[CircuitDeck],
    variations: list[Variation],
    *,
    max_cases: int = 1024,
    validate_applies_to: bool = True,
) -> list[ExpandedCase]:
    """Expand all circuit families in stable circuit/entry/value order."""
    circuits = normalize_circuit_decks(circuits)
    if validate_applies_to:
        validate_variation_circuit_ids(circuits, variations)
    random_entries = [item for item in variations if isinstance(item, RandomVariation)]
    if len(random_entries) > 1:
        raise VariationError(
            "multiple_random_variations",
            "At most one random variation entry is allowed per run_experiments call",
        )
    projected = sum(projected_case_count(circuit.circuit_id, variations) for circuit in circuits)
    check_case_cap(projected, max_cases)

    expanded: list[ExpandedCase] = []
    for circuit in circuits:
        targets, cards = _deck_targets(circuit.text)
        families: list[list[tuple[dict[str, Any], tuple[ResolvedAssignment, ...]]]] = []
        random_variation: RandomVariation | None = None
        for variation in variations:
            if not _applies(variation, circuit.circuit_id):
                continue
            if isinstance(variation, RandomVariation):
                random_variation = variation
                _validate_random_targets(circuit, targets, cards, variation)
                continue
            families.append(_resolve_assign_family(circuit, targets, variation))

        combinations: list[tuple[dict[str, Any], tuple[ResolvedAssignment, ...]]] = [({}, ())]
        for family in families:
            next_combinations = []
            for prior_assignments, prior_edits in combinations:
                for assignments, edits in family:
                    prior_targets = {target.casefold(): target for target in prior_assignments}
                    incoming_targets = {target.casefold(): target for target in assignments}
                    overlap = prior_targets.keys() & incoming_targets.keys()
                    if overlap:
                        targets_text = ", ".join(
                            sorted(prior_targets[target] for target in overlap)
                        )
                        raise VariationError(
                            "duplicate_assignment_target",
                            f"Circuit {circuit.circuit_id!r} assigns target(s) "
                            f"{targets_text} in more than one assign entry",
                        )
                    next_combinations.append(
                        ({**prior_assignments, **assignments}, (*prior_edits, *edits))
                    )
            combinations = next_combinations

        random_indices: tuple[int | None, ...] = (
            tuple(range(random_variation.runs)) if random_variation is not None else (None,)
        )
        circuit_index = 0
        for assignments, edits in combinations:
            for random_index in random_indices:
                expanded.append(
                    ExpandedCase(
                        circuit_id=circuit.circuit_id,
                        case_index=circuit_index,
                        assignments=dict(assignments),
                        edits=edits,
                        random=random_variation,
                        random_index=random_index,
                    )
                )
                circuit_index += 1
                check_case_cap(len(expanded), max_cases)
    return expanded


def materialize_variants(
    circuit: CircuitDeck,
    cases: list[ExpandedCase],
    output_dir: Path,
) -> list[MaterializedCase]:
    """Write stable ``case-NNNN`` deck variants into a staged circuit directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = (
        circuit.path.suffix if circuit.path.suffix.lower() in {".cir", ".net", ".sp"} else ".cir"
    )
    materialized: list[MaterializedCase] = []
    for case in cases:
        if case.circuit_id != circuit.circuit_id:
            continue
        text = _apply_assignments(circuit.text, case.edits)
        assignments = dict(case.assignments)
        if case.random is not None and case.random_index is not None:
            sampler = MCSampler(case.random.seed).derive(
                f"{case.circuit_id}:case{case.case_index}:run{case.random_index + 1}"
            )
            text, draws = _apply_random_rules(text, case.random.rules, sampler)
            assignments.update(draws)
            assignments["_random_run"] = case.random_index
            if case.random.id is not None:
                assignments["_random_id"] = case.random.id
        path = output_dir / f"case-{case.case_index:04d}{suffix}"
        atomic_write_text(path, text, durable=True)
        materialized.append(
            MaterializedCase(
                case_id=case.case_id,
                circuit_id=case.circuit_id,
                case_index=case.case_index,
                path=path,
                text=text,
                sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
                assignments=assignments,
            )
        )
    return materialized


def _applies(variation: AssignVariation | RandomVariation, circuit_id: str) -> bool:
    return variation.applies_to is None or circuit_id.casefold() in {
        item.casefold() for item in variation.applies_to
    }


def _deck_targets(text: str) -> tuple[_DeckTargets, list[SpiceCard]]:
    cards = lex(text).cards
    params: dict[str, str] = {}
    components: dict[str, str] = {}
    models_by_ref: dict[str, str] = {}
    for card in cards:
        if card.kind == "param":
            for token in tokenize_body(card.body)[1:]:
                if token.kind == TokenKind.KEY_VALUE and token.key:
                    params.setdefault(token.key.casefold(), token.key)
        elif card.kind == "instance" and card.name:
            try:
                view = InstanceLine.from_card(card)
            except ValueError:
                continue
            components.setdefault(view.ref.casefold(), view.ref)
            if view.model is not None:
                models_by_ref.setdefault(view.ref.casefold(), view.ref)
    return (
        _DeckTargets(
            params=params,
            components=components,
            models_by_ref=models_by_ref,
        ),
        cards,
    )


def _resolve_assign_family(
    circuit: CircuitDeck,
    targets: _DeckTargets,
    variation: AssignVariation,
) -> list[tuple[dict[str, Any], tuple[ResolvedAssignment, ...]]]:
    target_values = list(variation.assign.items())
    folded_targets = [target.casefold() for target, _ in target_values]
    if len(set(folded_targets)) != len(folded_targets):
        raise VariationError(
            "duplicate_assignment_target",
            f"Circuit {circuit.circuit_id!r} assigns the same target more than "
            "once with different casing",
        )
    if variation.combine == "zip":
        rows = zip(*(values for _, values in target_values), strict=True)
    else:
        rows = product(*(values for _, values in target_values))
    family = []
    for row in rows:
        assignments: dict[str, Any] = {}
        edits: list[ResolvedAssignment] = []
        for (target, _), value in zip(target_values, row, strict=True):
            assignments[target] = value
            edits.append(_resolve_assignment(circuit, targets, target, value))
        family.append((assignments, tuple(edits)))
    return family


def _resolve_assignment(
    circuit: CircuitDeck,
    targets: _DeckTargets,
    target: str,
    value: ScalarValue,
) -> ResolvedAssignment:
    if target.casefold().endswith("@model"):
        pattern = target[:-6]
        refs = tuple(
            ref
            for folded, ref in targets.models_by_ref.items()
            if fnmatch.fnmatchcase(folded, pattern.casefold())
        )
        if not refs:
            raise VariationError(
                "ambiguous_target",
                f"Circuit {circuit.circuit_id!r}: model-swap target {target!r} "
                "matches no model-bearing component",
            )
        return ResolvedAssignment(target=target, kind="model", value=value, refs=refs)

    folded = target.casefold()
    if folded in targets.params:
        _require_numeric_assignment(circuit, target, value)
        return ResolvedAssignment(
            target=target,
            kind="param",
            value=value,
            refs=(targets.params[folded],),
        )
    if folded in targets.components:
        reference = targets.components[folded]
        return ResolvedAssignment(
            target=target,
            kind="component",
            value=value,
            refs=(reference,),
        )
    raise VariationError(
        "ambiguous_target",
        f"Circuit {circuit.circuit_id!r}: target {target!r} is neither a declared "
        ".param, a component reference, nor a REF@model/model-glob target",
    )


def _require_numeric_assignment(
    circuit: CircuitDeck,
    target: str,
    value: ScalarValue,
) -> None:
    if not isinstance(value, str):
        return
    try:
        parse_spice_value(value)
    except ValueError as exc:
        raise VariationError(
            "invalid_assignment_value",
            f"Circuit {circuit.circuit_id!r}: target {target!r} requires a plain "
            f"number or SI-suffix SPICE value, got {value!r}",
        ) from exc


def _validate_random_targets(
    circuit: CircuitDeck,
    targets: _DeckTargets,
    cards: list[SpiceCard],
    variation: RandomVariation,
) -> None:
    model_names = {card.name.casefold() for card in cards if card.kind == "model" and card.name}
    for rule in variation.rules:
        if isinstance(rule, ComponentRule):
            if not any(
                fnmatch.fnmatchcase(folded, rule.target.casefold())
                for folded in targets.components
            ):
                raise VariationError(
                    "ambiguous_target",
                    f"Circuit {circuit.circuit_id!r}: random component target "
                    f"{rule.target!r} matches no component",
                )
        elif isinstance(rule, ParamRule):
            if rule.target.casefold() not in targets.params:
                raise VariationError(
                    "ambiguous_target",
                    f"Circuit {circuit.circuit_id!r}: random param target "
                    f"{rule.target!r} is not a declared .param",
                )
        elif isinstance(rule, ModelRule) and rule.target.casefold() not in model_names:
            raise VariationError(
                "ambiguous_target",
                f"Circuit {circuit.circuit_id!r}: random model target "
                f"{rule.target!r} has no local .model card",
            )


def _apply_assignments(text: str, edits: tuple[ResolvedAssignment, ...]) -> str:
    if not edits:
        return text
    cards = lex(text).cards
    for edit in edits:
        if edit.kind == "param":
            _set_param_value_on_cards(cards, edit.refs[0], edit.value)
        elif edit.kind == "component":
            _set_component_value(cards, edit.refs[0], edit.value)
        else:
            for ref in edit.refs:
                _set_instance_model(cards, ref, str(edit.value))
    return emit(cards)


def _set_param_value(text: str, name: str, value: ScalarValue) -> str:
    cards = lex(text).cards
    _set_param_value_on_cards(cards, name, value)
    return emit(cards)


def _set_param_value_on_cards(
    cards: list[SpiceCard],
    name: str,
    value: ScalarValue,
) -> None:
    target = name.casefold()
    for card in cards:
        if card.kind != "param":
            continue
        for token in tokenize_body(card.body)[1:]:
            if token.kind == TokenKind.KEY_VALUE and token.key and token.key.casefold() == target:
                card.replace_span(
                    token.body_offset,
                    token.body_end,
                    f"{token.key}={_render_scalar(value)}",
                )
                return
    raise VariationError("ambiguous_target", f".param {name!r} disappeared during expansion")


def _set_component_value(
    cards: list[SpiceCard],
    ref: str,
    value: ScalarValue,
) -> None:
    target = ref.casefold()
    for card in cards:
        if card.kind == "instance" and card.name and card.name.casefold() == target:
            try:
                component_value.apply_value_to_instance(card, _render_scalar(value))
            except NetlistError as exc:
                raise VariationError("invalid_assignment_value", str(exc)) from exc
            return
    raise VariationError("ambiguous_target", f"Component {ref!r} disappeared during expansion")


def _set_instance_model(
    cards: list[SpiceCard],
    ref: str,
    value: str,
) -> None:
    target = ref.casefold()
    for card in cards:
        if card.kind != "instance" or not card.name or card.name.casefold() != target:
            continue
        view = InstanceLine.from_card(card)
        if view.model is None:
            raise VariationError(
                "ambiguous_target",
                f"Instance {ref!r} has no model token to rewrite",
            )
        view.set_model(value)
        return
    raise VariationError("ambiguous_target", f"Instance {ref!r} disappeared during expansion")


def _apply_random_rules(
    text: str,
    rules: list[RandomRule],
    sampler: MCSampler,
) -> tuple[str, dict[str, float]]:
    draws: dict[str, float] = {}
    for rule in rules:
        if isinstance(rule, ComponentRule):
            text, sampled = _apply_component_rule(text, rule, sampler)
        elif isinstance(rule, ParamRule):
            text, sampled = _apply_param_rule(text, rule, sampler)
        elif isinstance(rule, ModelRule):
            text, sampled = _apply_model_rule(text, rule, sampler)
        else:
            text, sampled = _apply_mismatch_rule(text, rule, sampler)
        draws.update(sampled)
    return text, draws


def _spec(rule: RandomRuleBase) -> ToleranceSpec:
    return ToleranceSpec(
        tolerance=rule.tolerance,
        distribution=rule.distribution,
        kind=rule.scale,
    )


def _sample_nominal(
    sampler: MCSampler,
    nominal: float,
    spec: ToleranceSpec,
    *,
    stream: str,
) -> float:
    if spec.kind == "relative":
        return sampler.sample(nominal, spec, stream=stream)
    return nominal + sampler.sample_offset(nominal, spec, stream=stream)


def _apply_component_rule(
    text: str,
    rule: ComponentRule,
    sampler: MCSampler,
) -> tuple[str, dict[str, float]]:
    result = lex(text)
    matched: list[tuple[str, float]] = []
    for card in result.cards:
        if card.kind != "instance" or not card.name:
            continue
        view = InstanceLine.from_card(card)
        if not fnmatch.fnmatchcase(view.ref.casefold(), rule.target.casefold()):
            continue
        nominal = parse_value(view.value or "")
        if nominal is None:
            raise VariationError(
                "random_nominal_unavailable",
                f"Random component target {view.ref!r} has no plain numeric nominal value",
            )
        sampled = _sample_nominal(
            sampler,
            nominal,
            _spec(rule),
            stream=f"component:{view.ref}",
        )
        view.set_value(sampled)
        matched.append((view.ref, sampled))
    if not matched:
        raise VariationError(
            "ambiguous_target",
            f"Random component target {rule.target!r} matches no component",
        )
    return emit(result.cards), {f"random:component:{ref}": value for ref, value in matched}


def _apply_param_rule(
    text: str,
    rule: ParamRule,
    sampler: MCSampler,
) -> tuple[str, dict[str, float]]:
    nominal = _param_nominal(text, rule.target)
    if nominal is None:
        raise VariationError(
            "random_nominal_unavailable",
            f"Random .param target {rule.target!r} has no plain numeric nominal value",
        )
    sampled = _sample_nominal(
        sampler,
        nominal,
        _spec(rule),
        stream=f"param:{rule.target}",
    )
    return (
        _set_param_value(text, rule.target, sampled),
        {f"random:param:{rule.target}": sampled},
    )


def _param_nominal(text: str, name: str) -> float | None:
    target = name.casefold()
    for card in lex(text).cards:
        if card.kind != "param":
            continue
        for token in tokenize_body(card.body)[1:]:
            if token.kind == TokenKind.KEY_VALUE and token.key and token.key.casefold() == target:
                return parse_value(token.value or "")
    return None


def _render_scalar(value: ScalarValue) -> str:
    if isinstance(value, float):
        return f"{value:.12g}"
    return str(value)


def _apply_model_rule(
    text: str,
    rule: ModelRule,
    sampler: MCSampler,
) -> tuple[str, dict[str, float]]:
    card = extract_model_card(text, rule.target)
    if card is None:
        raise VariationError(
            "ambiguous_target",
            f"Random model target {rule.target!r} has no local .model card",
        )
    nominals = parse_model_params(card)
    perturbations = sample_model_perturbation(
        sampler,
        rule.target,
        nominals,
        {rule.param: _spec(rule)},
    )
    if rule.param not in perturbations:
        raise VariationError(
            "random_nominal_unavailable",
            f"Model {rule.target!r} parameter {rule.param!r} has no plain numeric nominal",
        )
    sampled = perturbations[rule.param]
    return (
        perturb_model_in_text(text, rule.target, perturbations),
        {f"random:model:{rule.target}.{rule.param}": sampled},
    )


def _apply_mismatch_rule(
    text: str,
    rule: MismatchRule,
    sampler: MCSampler,
) -> tuple[str, dict[str, float]]:
    engine_rule = EngineMismatchRule(
        prefix=rule.prefix,
        avt=rule.AVT,
        ak=rule.AK,
        distribution=rule.distribution,
        vth_param=rule.vth_param,
        k_param=rule.k_param,
        min_wl_um2=rule.min_wl_um2,
    )
    instances = [
        instance
        for instance in extract_mosfet_instances(text)
        if instance.ref.upper().startswith(rule.prefix.upper())
    ]
    if not instances:
        raise VariationError(
            "ambiguous_target",
            f"Mismatch prefix {rule.prefix!r} matches no top-level device with numeric W/L",
        )
    draws: dict[str, float] = {}
    for instance in instances:
        base_card = extract_model_card(text, instance.model_name)
        if base_card is None:
            raise VariationError(
                "model_missing",
                f"Mismatch instance {instance.ref!r} references missing model "
                f"{instance.model_name!r}",
            )
        nominals = parse_model_params(base_card)
        mismatch = sample_instance_mismatch(sampler, instance, engine_rule)
        overrides: dict[str, float] = {}
        vth_nominal = nominals.get(rule.vth_param.upper())
        if vth_nominal is not None:
            overrides[rule.vth_param] = vth_nominal + mismatch["dvth"]
            draws[f"random:mismatch:{instance.ref}.{rule.vth_param}"] = overrides[rule.vth_param]
        k_nominal = nominals.get(rule.k_param.upper())
        if k_nominal is not None:
            overrides[rule.k_param] = k_nominal * (1.0 + mismatch["dk_over_k"])
            draws[f"random:mismatch:{instance.ref}.{rule.k_param}"] = overrides[rule.k_param]
        if not overrides:
            raise VariationError(
                "random_nominal_unavailable",
                f"Mismatch model {instance.model_name!r} declares neither "
                f"{rule.vth_param!r} nor {rule.k_param!r}",
            )
        variant = variant_model_name(instance.model_name, instance.ref)
        text = inject_card_before_end(
            text,
            render_variant_model_card(base_card, variant, overrides),
        )
        text = rewrite_instance_model(text, instance.ref, variant)
    return text, draws
