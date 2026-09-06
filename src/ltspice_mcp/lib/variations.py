"""Strict experiment-variation models and deterministic deck expansion."""

from __future__ import annotations

import fnmatch
import hashlib
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
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
from ltspice_mcp.lib.deck_staging import (
    card_sections,
    closure_depth,
    rewrite_staged_references,
    staged_reference_targets,
)
from ltspice_mcp.lib.format import parse_spice_value
from ltspice_mcp.lib.montecarlo import (
    MCSampler,
    ToleranceSpec,
    extract_mosfet_instances,
    inject_card_before_end,
    matches_prefix,
    parse_model_params,
    parse_value,
    render_variant_model_card,
    rewrite_instance_model,
    sample_instance_mismatch,
    sample_model_perturbation,
    variant_model_name,
)
from ltspice_mcp.lib.montecarlo import MismatchRule as EngineMismatchRule
from ltspice_mcp.lib.spice_lex import SpiceCard, Token, TokenKind, emit, lex, tokenize_body
from ltspice_mcp.lib.spice_lex_views import InstanceLine, ModelCard
from ltspice_mcp.lib.subckt_mismatch import (
    MOBILITY_PARAM,
    VTH_PARAM,
    ClosureFile,
    MismatchPlan,
    MismatchPlanError,
    MismatchValues,
    XFetTarget,
    build_plan,
    draw_mismatch,
    overlapping_claims,
    render,
)

ScalarValue: TypeAlias = StrictInt | StrictFloat | str
Distribution: TypeAlias = Literal["normal", "gaussian", "uniform"]
Scale: TypeAlias = Literal["relative", "absolute"]

_APPLIES_TO_DESCRIPTION = (
    "Circuit ids this variation expands over (default: every circuit). Not a "
    "device filter — devices are selected by each rule's own targeting."
)

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
    id: str | None = Field(
        default=None,
        description=(
            "Label for this entry in error messages only; unlike a random "
            "variation's id it does not reach case assignments. Group cases by "
            "the assigned target names instead."
        ),
    )
    combine: Literal["grid", "zip"] = Field(
        default="grid",
        description=(
            "How the value lists combine: 'grid' runs every combination (the "
            "cartesian product), 'zip' runs the i-th value of every target "
            "together as case i."
        ),
    )
    applies_to: list[str] | None = Field(default=None, description=_APPLIES_TO_DESCRIPTION)
    assign: dict[str, list[ScalarValue]] = Field(
        description=(
            "Target → value list, resolved in order as a 'REF@model' (glob "
            "allowed) model swap, an 'X1:delvto'/'X1:mulu0' per-instance "
            "mismatch delta, a declared .param, then a component reference — "
            "forms in spice://guide."
        )
    )

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
    tolerance: float = Field(
        gt=0.0,
        description=(
            "Spread of the draw: with scale 'relative' a fraction of the nominal "
            "(0.1 = 10%), with 'absolute' in the value's own units. For a normal "
            "draw it is the 3-sigma bound (sigma = tolerance/3, draws clipped at "
            "the bound); for a uniform draw, the half-range."
        ),
    )
    scale: Scale = Field(
        default="relative",
        description="'relative' reads tolerance as a fraction of the nominal, 'absolute' in its units.",
    )
    distribution: Distribution = Field(
        default="normal",
        description="'normal' (or 'gaussian'): truncated at 3 sigma = tolerance; 'uniform': flat over the bound.",
    )

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


# The field descriptions here carry the two facts a caller cannot recover from
# a result: the coefficients' units, and the inversion from a target sigma.
# Prefix conventions and BSIM parameter names are in ``spice://guide``, which a
# caller reads once, rather than on the wire in every session.
class MismatchRule(VariationModel):
    """Pelgrom mismatch rule: σ(ΔVTH) = AVT/√(W·L) and σ(ΔK)/K = AK/√(W·L),
    sampled independently per instance per run. Worked examples, prefix
    conventions and BSIM parameter names: spice://guide."""

    rule: Literal["mismatch"]
    prefix: str = Field(
        default="M",
        description=(
            "Device prefix, case-insensitive, matched on leading characters — "
            "'M1' also claims M10, 'X' reaches subckt-wrapped FETs, and bare "
            "'M' perturbs every MOSFET."
        ),
    )
    AVT: float = Field(
        default=0.0,
        description=(
            "VTH-mismatch coefficient in V·µm (3e-3 = 3 mV·µm); for a target "
            "sigma, invert: AVT = sigma·√(W_µm·L_µm), and 0 disables it."
        ),
    )
    AK: float = Field(
        default=0.0,
        description="K-mismatch coefficient in fraction·µm (0.02 = 2%·µm), 0 disables it.",
    )
    distribution: Distribution = Field(
        default="normal",
        description="Distribution of the per-instance offset.",
    )
    vth_param: str = Field(
        default="VTO",
        description="Model-card parameter receiving ΔVTH ('VTH0' for BSIM).",
    )
    k_param: str = Field(
        default="KP",
        description="Model-card parameter scaled by (1+ΔK/K) ('U0' for BSIM).",
    )
    min_wl_um2: float = Field(
        default=1e-3,
        gt=0.0,
        description="Floor on W·L in µm² for the √(W·L) denominator.",
    )


RandomRule: TypeAlias = Annotated[
    ComponentRule | ParamRule | ModelRule | MismatchRule,
    Field(discriminator="rule"),
]


class RandomVariation(VariationModel):
    """One reproducible Monte Carlo family."""

    kind: Literal["random"]
    id: str | None = Field(
        default=None,
        description=(
            "Label for this entry; also recorded on every case it produces as "
            "the '_random_id' assignment, so it is groupable."
        ),
    )
    runs: int = Field(ge=1)
    seed: int | None = None
    applies_to: list[str] | None = Field(default=None, description=_APPLIES_TO_DESCRIPTION)
    rules: list[RandomRule] = Field(
        description=(
            "What varies — one entry per thing that varies, each naming its "
            "kind in 'rule': 'component' perturbs a part's value (resistor "
            "tolerance), 'param' a declared .param, 'model' one model "
            "parameter (process spread), 'mismatch' the Pelgrom "
            "device-to-device spread of matched transistors."
        )
    )

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
class DeckFile:
    """One staged file a deck pulls in through ``.include``/``.lib``."""

    path: Path
    text: str


@dataclass(frozen=True)
class CircuitDeck:
    """One uniquely named circuit deck supplied to the expansion engine."""

    circuit_id: str
    path: Path
    text: str
    # The staged include closure, root deck excluded. A variation resolves
    # against every file in it, so factoring a circuit into a reusable core
    # does not put that core's components out of a sweep's reach.
    includes: tuple[DeckFile, ...] = ()
    # True when ``circuit_id`` was taken from the file stem because the caller
    # named none. Carried on the deck so the one validator can say where a
    # rejected id came from: the rule is about the id, but the fix is about the
    # argument, and a caller who never wrote an id cannot see the connection.
    id_from_file_stem: bool = False


@dataclass(frozen=True)
class ResolvedAssignment:
    """One assignment bound to a concrete deck edit."""

    target: str
    kind: Literal["param", "component", "model", "instance_param"]
    value: ScalarValue
    refs: tuple[str, ...] = ()
    # Which of the two per-instance mismatch parameters an ``instance_param``
    # edit writes. The instance it writes to is ``refs[0]``, still spelled the
    # way the caller wrote it: which inner device that names is resolved once
    # per case, over every such edit at a time, so the patched device
    # subcircuit is cloned once no matter how many instances use it.
    instance_param: str | None = None
    # Index into the deck closure: 0 is the root deck, 1.. its includes.
    file_index: int = 0
    # Where inside that file resolution chose to edit. Carried so the writer
    # looks the choice up instead of re-deciding it: one precedence rule, in
    # ``_select_site``, and no second copy of it to drift. ``None`` only for a
    # model-swap glob, which is a set query and edits every match it finds.
    site: _Site | None = None


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
class _Site:
    """One declaration of a target name inside a single file.

    A name is unique per file only in the flat, section-less decks; a library
    file routinely declares the same ``.model`` once per corner section, and a
    factored include declares the same ``R1`` inside every ``.subckt`` it
    holds. Recording where each declaration sits is what lets an exact-name
    target refuse to pick one of them at random.
    """

    name: str
    scope: tuple[str, ...]
    section: str | None


@dataclass(frozen=True)
class _DeckTargets:
    params: dict[str, list[_Site]]
    components: dict[str, list[_Site]]
    models_by_ref: dict[str, str]
    model_names: dict[str, list[_Site]]


@dataclass(frozen=True)
class _ClosureFile:
    """One file of a deck's include closure, with its targets resolved."""

    index: int
    path: Path
    text: str
    targets: _DeckTargets

    @property
    def depth(self) -> int:
        return closure_depth(self.index)


@dataclass(frozen=True)
class _RuleTarget:
    """One closure file a random rule perturbs, with what it resolved to there.

    ``site`` is the single declaration an exact ``.param``/``.model`` target
    names. A component glob is a set query with no one site, so it carries the
    folded references it matched instead — the writer perturbs those and does
    not re-evaluate the pattern, which is what keeps what was applied and what
    was reported the same set.
    """

    file: _ClosureFile
    site: _Site | None = None
    refs: frozenset[str] = frozenset()


@dataclass(frozen=True)
class _DeckClosure:
    """A deck and its staged includes, addressed as one target namespace."""

    circuit_id: str
    files: tuple[_ClosureFile, ...]
    # Mismatch plans built over the UNEDITED closure, keyed by what was
    # selected. One circuit's cases select the same devices and differ only in
    # the values written, and a plan costs a lex of every file in the closure —
    # a PDK's worth, per case. Filled and read only by ``_mismatch_plan``.
    plans: dict[tuple[Any, ...], MismatchPlan] = field(default_factory=dict, compare=False)

    def texts(self) -> dict[int, str]:
        return {file.index: file.text for file in self.files}

    def unedited(self, files: Sequence[ClosureFile]) -> bool:
        """Is this the closure's own text, or has a case already edited it?"""
        return all(file.text is self.files[file.index].text for file in files)


def _id_suggestion(circuit_id: str) -> str:
    """A valid id built out of the rejected one, or '' when nothing survives.

    Offered rather than imposed: silently repairing the id would run the file
    under a name the caller never chose and cannot predict.
    """
    cleaned = "".join(
        char for char in circuit_id if char.isascii() and (char.isalnum() or char in "_-")
    )
    return cleaned.lstrip("_-")[:64]


def normalize_circuit_decks(circuits: list[CircuitDeck]) -> list[CircuitDeck]:
    """Validate circuit ids and preserve caller order."""
    seen: set[str] = set()
    normalized: list[CircuitDeck] = []
    for circuit in circuits:
        circuit_id = circuit.circuit_id
        if _CIRCUIT_ID_RE.fullmatch(circuit_id) is None:
            # Spell the positional part of the rule out: "letters, digits,
            # underscores ... starting with a letter or digit" reads as
            # self-contradictory to anyone whose id starts with an underscore.
            raise VariationError(
                "invalid_circuit_id",
                f"Circuit id {circuit_id!r} must be 1-64 characters long, start "
                "with a letter or digit, and use only letters, digits, "
                "underscores and hyphens after that"
                + (
                    f". This id was derived from the file stem of {circuit.path.name!r} "
                    "because the circuit carried no 'id'; pass one explicitly "
                    "(e.g. id='"
                    + (_id_suggestion(circuit_id) or "amp")
                    + "') to run this file under a valid id without renaming it"
                    if circuit.id_from_file_stem
                    else ""
                ),
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
        closure = _build_closure(circuit)
        families: list[list[tuple[dict[str, Any], tuple[ResolvedAssignment, ...]]]] = []
        random_variation: RandomVariation | None = None
        for variation in variations:
            if not _applies(variation, circuit.circuit_id):
                continue
            if isinstance(variation, RandomVariation):
                random_variation = variation
                head = _mismatch_head(variation.rules)
                mismatch_rules = [
                    rule for rule in variation.rules if isinstance(rule, MismatchRule)
                ]
                for rule in variation.rules:
                    if isinstance(rule, MismatchRule):
                        # Validated as a set, at the position of the first of
                        # them, so a rule is reported in declaration order the
                        # way it is applied.
                        if rule is head:
                            _validate_mismatch_rules(closure, mismatch_rules)
                        continue
                    _resolve_random_rule_targets(closure, rule)
                continue
            families.append(_resolve_assign_family(closure, variation))

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
    closure = _build_closure(circuit)
    referrers = _include_referrers(closure)
    materialized: list[MaterializedCase] = []
    for case in cases:
        if case.circuit_id != circuit.circuit_id:
            continue
        texts = closure.texts()
        _apply_assignments(closure, texts, case.edits)
        assignments = dict(case.assignments)
        if case.random is not None and case.random_index is not None:
            sampler = MCSampler(case.random.seed).derive(
                f"{case.circuit_id}:case{case.case_index}:run{case.random_index + 1}"
            )
            assignments.update(_apply_random_rules(closure, texts, case.random.rules, sampler))
            assignments["_random_run"] = case.random_index
            if case.random.id is not None:
                assignments["_random_id"] = case.random.id
        text = _write_case_includes(closure, referrers, texts, case.case_index)
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


def _deck_targets(text: str, source: Path, depth: int) -> _DeckTargets:
    params: dict[str, list[_Site]] = {}
    components: dict[str, list[_Site]] = {}
    models_by_ref: dict[str, str] = {}
    model_names: dict[str, list[_Site]] = {}
    cards = lex(text).cards
    for card, section in zip(cards, card_sections(cards, source, depth), strict=True):
        if card.kind == "param":
            for token in tokenize_body(card.body)[1:]:
                if token.kind == TokenKind.KEY_VALUE and token.key:
                    params.setdefault(token.key.casefold(), []).append(
                        _Site(token.key, card.scope, section)
                    )
        elif card.kind == "model" and card.name:
            model_names.setdefault(card.name.casefold(), []).append(
                _Site(card.name, card.scope, section)
            )
        elif card.kind == "instance" and card.name:
            try:
                view = InstanceLine.from_card(card)
            except ValueError:
                continue
            components.setdefault(view.ref.casefold(), []).append(
                _Site(view.ref, card.scope, section)
            )
            if view.model is not None:
                models_by_ref.setdefault(view.ref.casefold(), view.ref)
    return _DeckTargets(
        params=params,
        components=components,
        models_by_ref=models_by_ref,
        model_names=model_names,
    )


def _build_closure(circuit: CircuitDeck) -> _DeckClosure:
    """Project a deck and its staged includes into one addressable namespace."""
    files = [
        _ClosureFile(
            index=index,
            path=path,
            text=text,
            targets=_deck_targets(text, path, closure_depth(index)),
        )
        for index, (path, text) in enumerate(
            [(circuit.path, circuit.text)]
            + [(include.path, include.text) for include in circuit.includes]
        )
    ]
    return _DeckClosure(circuit_id=circuit.circuit_id, files=tuple(files))


def _select_target_file(
    closure: _DeckClosure,
    matches: list[int],
    *,
    what: str,
    target: str,
) -> int:
    """Resolve an exact-name target that several closure files declare.

    Only exact names come here — a glob or a prefix is a set query and is
    applied to every file it matches, so it has nothing to disambiguate.

    The root deck wins outright: that is where a target resolved before the
    closure was searched at all, so extending the search can only add reach,
    never move an existing target somewhere else. Between two includes there is
    no such tiebreak, so the caller is told which files collided rather than
    handed a silent pick.
    """
    if 0 in matches:
        return 0
    if len(matches) == 1:
        return matches[0]
    names = ", ".join(sorted(closure.files[index].path.name for index in matches))
    raise VariationError(
        "ambiguous_target",
        f"Circuit {closure.circuit_id!r}: {what} {target!r} names one declaration "
        f"but is declared in more than one included file ({names}) and not in the "
        "deck itself; declare it in the deck to say which one the variation means "
        "(a glob or prefix target would instead have been applied to every match)",
    )


def _site_label(site: _Site) -> str:
    where = f".subckt {'.'.join(site.scope)}" if site.scope else "top level"
    if site.section is not None:
        return f"{where} of section {site.section!r}"
    return where


def _select_site(
    closure: _DeckClosure,
    file_index: int,
    sites: list[_Site],
    *,
    what: str,
    target: str,
) -> _Site:
    """Resolve an exact-name target that one file declares more than once.

    Same precedence as the cross-file rule, one level down: the declaration
    that is unconditionally live at the file's top level wins, because that is
    the one a caller naming the deck's own target means. Two declarations that
    are equally buried — one per ``.subckt``, or one per library section — have
    no such tiebreak, and picking either would edit half a circuit and report
    a whole one.
    """
    if len(sites) == 1:
        return sites[0]
    outermost = [site for site in sites if not site.scope and site.section is None]
    if len(outermost) == 1:
        return outermost[0]
    labels = sorted({_site_label(site) for site in sites})
    # Declarations that share a site have no tiebreak to reach for, so the
    # advice cannot be "move one there" — they are already there.
    if len(labels) == 1:
        where, advice = f"all at {labels[0]}", "give them distinct names"
    else:
        where = f"at {', '.join(labels)}"
        advice = "give them distinct names, or move the one you mean to the file's top level"
    raise VariationError(
        "ambiguous_target",
        f"Circuit {closure.circuit_id!r}: {what} {target!r} has {len(sites)} "
        f"declarations in {closure.files[file_index].path.name} ({where}); a "
        f"variation edits one declaration, so {advice}",
    )


def _resolve_assign_family(
    closure: _DeckClosure,
    variation: AssignVariation,
) -> list[tuple[dict[str, Any], tuple[ResolvedAssignment, ...]]]:
    target_values = list(variation.assign.items())
    folded_targets = [target.casefold() for target, _ in target_values]
    if len(set(folded_targets)) != len(folded_targets):
        raise VariationError(
            "duplicate_assignment_target",
            f"Circuit {closure.circuit_id!r} assigns the same target more than "
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
            edits.extend(_resolve_assignment(closure, target, value))
        family.append((assignments, tuple(edits)))
    return family


def _resolve_assignment(
    closure: _DeckClosure,
    target: str,
    value: ScalarValue,
) -> tuple[ResolvedAssignment, ...]:
    """Bind one assignment to every closure file it edits.

    A ``REF@model`` target is a glob, so it fans out: one edit per file that
    holds a match. An exact ``.param``/component name binds to a single file.
    """
    if ":" in target:
        return _resolve_instance_param(closure, target, value)
    if target.casefold().endswith("@model"):
        pattern = target[:-6].casefold()
        edits: list[ResolvedAssignment] = []
        for file in closure.files:
            refs = tuple(
                ref
                for folded, ref in file.targets.models_by_ref.items()
                if fnmatch.fnmatchcase(folded, pattern)
            )
            if refs:
                edits.append(
                    ResolvedAssignment(
                        target=target,
                        kind="model",
                        value=value,
                        refs=refs,
                        file_index=file.index,
                    )
                )
        if not edits:
            raise VariationError(
                "ambiguous_target",
                f"Circuit {closure.circuit_id!r}: model-swap target {target!r} "
                f"matches no model-bearing component{_closure_scope(closure)}",
            )
        return tuple(edits)

    folded = target.casefold()
    # File precedence is the outer test and kind precedence the inner one: a
    # .param still beats a same-named component, but only within the one file
    # that owns the target, so an include can never shadow the root deck.
    matches = [
        file.index
        for file in closure.files
        if folded in file.targets.params or folded in file.targets.components
    ]
    if not matches:
        raise VariationError(
            "ambiguous_target",
            f"Circuit {closure.circuit_id!r}: target {target!r} is neither a declared "
            f".param, a component reference, nor a REF@model/model-glob target"
            f"{_closure_scope(closure)}",
        )
    index = _select_target_file(closure, matches, what="target", target=target)
    targets = closure.files[index].targets
    if folded in targets.params:
        _require_numeric_assignment(closure.circuit_id, target, value)
        site = _select_site(closure, index, targets.params[folded], what="target", target=target)
        return (
            ResolvedAssignment(
                target=target,
                kind="param",
                value=value,
                refs=(site.name,),
                file_index=index,
                site=site,
            ),
        )
    site = _select_site(closure, index, targets.components[folded], what="target", target=target)
    return (
        ResolvedAssignment(
            target=target,
            kind="component",
            value=value,
            refs=(site.name,),
            file_index=index,
            site=site,
        ),
    )


_INSTANCE_PARAM_RE = re.compile(
    r"^(?P<instance>[A-Za-z][^\s:]*):(?P<param>[A-Za-z_][A-Za-z0-9_]*)$"
)


def _resolve_instance_param(
    closure: _DeckClosure,
    target: str,
    value: ScalarValue,
) -> tuple[ResolvedAssignment, ...]:
    """Bind a per-instance mismatch target such as ``X1:delvto``.

    ``X1:delvto`` names the single MOS device inside ``X1``; ``X1.M0:delvto``
    names one device in a body that holds several, and is required there.
    Only the two parameters the mismatch engine owns are addressable this way —
    anything else would be a general hierarchical-parameter feature, which this
    is not.

    Resolution stops at the shape here. Which inner device the instance names
    is settled once per case, over all such targets together, because the
    patched device subcircuit is cloned once and two independently resolved
    targets would clone it twice.
    """
    match = _INSTANCE_PARAM_RE.match(target)
    if match is None:
        raise VariationError(
            "invalid_instance_param_target",
            f"Circuit {closure.circuit_id!r}: target {target!r} is not a per-instance "
            "parameter target; write INSTANCE:PARAM (for example 'X1:delvto', or "
            "'X1.M0:delvto' when the subcircuit holds more than one device)",
        )
    instance = match.group("instance")
    param = match.group("param").casefold()
    if param not in (VTH_PARAM, MOBILITY_PARAM):
        raise VariationError(
            "invalid_instance_param_target",
            f"Circuit {closure.circuit_id!r}: target {target!r} sets {param!r}, but "
            f"only {VTH_PARAM} (a shift of the signed threshold) and {MOBILITY_PARAM} "
            "(a mobility multiplier) can be set per instance",
        )
    folded = target.casefold()
    clashes = sorted(
        {
            file.path.name
            for file in closure.files
            if folded in file.targets.params or folded in file.targets.components
        }
    )
    if clashes:
        raise VariationError(
            "ambiguous_target",
            f"Circuit {closure.circuit_id!r}: target {target!r} reads as a per-instance "
            f"{param} target but is also declared as a .param or component in "
            f"{', '.join(clashes)}; rename one of them so the target names one thing",
        )
    _require_numeric_assignment(closure.circuit_id, target, value)
    return (
        ResolvedAssignment(
            target=target,
            kind="instance_param",
            value=value,
            refs=(instance,),
            instance_param=param,
        ),
    )


def _closure_files(closure: _DeckClosure, texts: dict[int, str]) -> list[ClosureFile]:
    """Project the closure into the mismatch engine's view of it."""
    return [
        ClosureFile(index=file.index, path=file.path, text=texts[file.index])
        for file in closure.files
    ]


def _mismatch_plan(
    closure: _DeckClosure,
    files: list[ClosureFile],
    *,
    prefix: Sequence[str] | None = None,
    selectors: Sequence[str] | None = None,
    exact: bool = False,
) -> MismatchPlan:
    """Build a mismatch plan, relabelling its refusals as variation errors.

    Reused across the cases of one circuit while the closure still reads as it
    did when the plan was built: a plan resolves which devices a selection
    names, and every case of a circuit names the same ones — only the values
    differ. Each build lexes every file in the closure, which on a PDK deck is
    the dominant cost of materializing a case.

    A case that has already edited the closure gets its own plan rather than
    the cached one, because the edit can be the very thing the plan reads.
    """
    prefixes = None if prefix is None else tuple(prefix)
    key = (prefixes, tuple(selectors) if selectors is not None else None, exact)
    unedited = closure.unedited(files)
    if unedited and key in closure.plans:
        return closure.plans[key]
    try:
        plan = build_plan(files, prefix=prefixes, selectors=selectors, exact=exact)
    except MismatchPlanError as exc:
        raise VariationError(exc.code, f"Circuit {closure.circuit_id!r}: {exc}") from exc
    if unedited:
        closure.plans[key] = plan
    return plan


def _apply_instance_params(
    closure: _DeckClosure,
    texts: dict[int, str],
    edits: list[ResolvedAssignment],
) -> None:
    """Write every per-instance mismatch value this case asks for, at once.

    One plan over all of them, so a device subcircuit two instances share is
    cloned once and both instances point at the same patched copy.
    """
    if not edits:
        return
    files = _closure_files(closure, texts)
    plan = _mismatch_plan(
        closure,
        files,
        selectors=[edit.refs[0] for edit in edits],
        exact=True,
    )
    values: dict[str, MismatchValues] = {}
    written: dict[tuple[str, str], str] = {}
    for edit in edits:
        target = _instance_target(plan, edit.refs[0])
        assert edit.instance_param is not None
        key = (target.ref.casefold(), edit.instance_param)
        previous = written.get(key)
        if previous is not None:
            raise VariationError(
                "duplicate_assignment_target",
                f"Circuit {closure.circuit_id!r}: targets {previous!r} and "
                f"{edit.target!r} both set {edit.instance_param} on {target.ref}",
            )
        written[key] = edit.target
        number = float(parse_spice_value(str(edit.value)))
        current = values.get(target.ref, MismatchValues())
        values[target.ref] = (
            MismatchValues(delvto=number, mulu0=current.mulu0)
            if edit.instance_param == VTH_PARAM
            else MismatchValues(delvto=current.delvto, mulu0=number)
        )
    texts.update(render(files, plan, values))


def _instance_target(plan: MismatchPlan, selector: str) -> XFetTarget:
    x_ref, _, inner = selector.partition(".")
    for target in plan.targets:
        if target.x_ref.casefold() != x_ref.casefold():
            continue
        if inner and target.inner_ref.casefold() != inner.casefold():
            continue
        return target
    raise KeyError(selector)


def _closure_scope(closure: _DeckClosure) -> str:
    """Name what was searched, so a miss reads as a miss and not a blind spot."""
    includes = len(closure.files) - 1
    if not includes:
        return " in the deck, which includes no other file"
    return f" in the deck or its {includes} included file(s)"


def _require_numeric_assignment(
    circuit_id: str,
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
            f"Circuit {circuit_id!r}: target {target!r} requires a plain "
            f"number or SI-suffix SPICE value, got {value!r}",
        ) from exc


def _resolve_random_rule_targets(
    closure: _DeckClosure,
    rule: ComponentRule | ParamRule | ModelRule,
) -> tuple[_RuleTarget, ...]:
    """Bind one random rule to every closure file it perturbs.

    Expansion and materialization both route through here, so a rule can never
    validate against one file and then be applied to another. Mismatch rules do
    not: they are resolved as a set rather than one at a time (see
    ``_apply_mismatch_rules``), through ``_descend`` — which expansion calls
    too, so they get the same guarantee.

    A component glob is a set query: it means every match, and a match set that
    stops at the root deck is a Monte Carlo that varies half the devices while
    reporting a full one. An exact ``.param`` or ``.model`` name is a singular
    reference and still resolves to one file and one declaration inside it —
    and that declaration travels with the target, so the edit lands where
    resolution said it would.
    """
    if isinstance(rule, ComponentRule):
        pattern = rule.target.casefold()
        by_ref: dict[str, list[tuple[_ClosureFile, _Site]]] = {}
        for file in closure.files:
            for folded, sites in file.targets.components.items():
                if fnmatch.fnmatchcase(folded, pattern):
                    by_ref.setdefault(folded, []).extend((file, site) for site in sites)
        if not by_ref:
            raise VariationError(
                "ambiguous_target",
                f"Circuit {closure.circuit_id!r}: random component target {rule.target!r} "
                f"matches no component{_closure_scope(closure)}",
            )
        _refuse_repeated_reference(closure, rule.target, by_ref)
        refs_by_file: dict[int, set[str]] = {}
        for folded, hits in by_ref.items():
            for file, _ in hits:
                refs_by_file.setdefault(file.index, set()).add(folded)
        return tuple(
            _RuleTarget(file=closure.files[index], refs=frozenset(refs))
            for index, refs in sorted(refs_by_file.items())
        )

    if isinstance(rule, ParamRule):
        what, miss = "random param target", "is not a declared .param"
        sites_by_file = {file.index: file.targets.params for file in closure.files}
    else:
        what, miss = "random model target", "has no .model card"
        sites_by_file = {file.index: file.targets.model_names for file in closure.files}
    folded = rule.target.casefold()
    matches = [index for index, sites in sites_by_file.items() if folded in sites]
    if not matches:
        raise VariationError(
            "ambiguous_target",
            f"Circuit {closure.circuit_id!r}: {what} {rule.target!r} {miss}"
            f"{_closure_scope(closure)}",
        )
    index = _select_target_file(closure, matches, what=what, target=rule.target)
    site = _select_site(
        closure,
        index,
        sites_by_file[index][folded],
        what=what,
        target=rule.target,
    )
    return (_RuleTarget(file=closure.files[index], site=site),)


def _refuse_repeated_reference(
    closure: _DeckClosure,
    target: str,
    by_ref: dict[str, list[tuple[_ClosureFile, _Site]]],
) -> None:
    """Refuse a glob whose match set holds one reference more than once.

    Every match draws its own value from a stream keyed by the reference, so a
    reference declared at two sites — two ``.subckt`` bodies, or two files — is
    perturbed twice with two different numbers while the receipt records
    whichever was written last. The deck and the receipt then describe
    different circuits. There is no tiebreak to reach for the way an exact
    target has one: the glob asked for every match, and both are matches.
    """
    for hits in by_ref.values():
        if len(hits) < 2:
            continue
        labels = ", ".join(
            sorted({f"{file.path.name} {_site_label(site)}" for file, site in hits})
        )
        raise VariationError(
            "ambiguous_target",
            f"Circuit {closure.circuit_id!r}: random component target {target!r} matches "
            f"{hits[0][1].name!r} at {len(hits)} declarations ({labels}); each match draws "
            "its own value, so one reference declared twice would be perturbed with two "
            "different values and reported with one — give the declarations distinct "
            "references, or narrow the target to the one you mean",
        )


def _apply_assignments(
    closure: _DeckClosure,
    texts: dict[int, str],
    edits: tuple[ResolvedAssignment, ...],
) -> None:
    by_file: dict[int, list[ResolvedAssignment]] = {}
    instance_params: list[ResolvedAssignment] = []
    for edit in edits:
        if edit.kind == "instance_param":
            instance_params.append(edit)
            continue
        by_file.setdefault(edit.file_index, []).append(edit)
    _apply_instance_params(closure, texts, instance_params)
    for index, file_edits in by_file.items():
        file = closure.files[index]
        cards = lex(texts[index]).cards
        for edit in file_edits:
            if edit.kind == "model":
                for ref in edit.refs:
                    _set_instance_model(cards, ref, str(edit.value))
                continue
            assert edit.site is not None
            if edit.kind == "param":
                _set_param_value_on_cards(cards, edit.site, edit.value, file=file)
            else:
                _set_component_value(cards, edit.site, edit.value, file=file)
        texts[index] = emit(cards)


def _card_at_site(
    cards: list[SpiceCard],
    matches: list[int],
    site: _Site,
    *,
    file: _ClosureFile,
    what: str,
) -> int:
    """Find the declaration resolution chose among the cards being rewritten.

    Resolution owns the precedence rule — ``_select_site`` — and records where
    it landed; this only looks that place back up. Writing the rule a second
    time here is what would let the two drift, after which an edit lands
    somewhere the receipt does not describe.
    """
    if len(matches) == 1:
        return matches[0]
    sections = card_sections(cards, file.path, file.depth)
    at_site = [
        index
        for index in matches
        if cards[index].scope == site.scope and sections[index] == site.section
    ]
    if len(at_site) == 1:
        return at_site[0]
    raise VariationError(
        "ambiguous_target",
        f"{what} resolved to {_site_label(site)} in {file.path.name}, which now holds "
        f"{len(at_site)} matching declarations",
    )


def _param_tokens(cards: list[SpiceCard], name: str) -> list[tuple[int, Token]]:
    """Every ``name=value`` token any ``.param`` card declares, with its card."""
    target = name.casefold()
    tokens: list[tuple[int, Token]] = []
    for index, card in enumerate(cards):
        if card.kind != "param":
            continue
        for token in tokenize_body(card.body)[1:]:
            if token.kind == TokenKind.KEY_VALUE and token.key and token.key.casefold() == target:
                tokens.append((index, token))
    return tokens


def _chosen_param_token(
    cards: list[SpiceCard],
    site: _Site,
    *,
    file: _ClosureFile,
) -> tuple[int, Token] | None:
    """The one ``name=value`` token the site names, with its card.

    ``None`` says no card declares that name any more, which is a different
    answer from "several do" — folding the two together reported a naming
    collision as a missing nominal.
    """
    tokens = _param_tokens(cards, site.name)
    if not tokens:
        return None
    chosen = _card_at_site(
        cards,
        [index for index, _ in tokens],
        site,
        file=file,
        what=f".param {site.name!r}",
    )
    return next((index, token) for index, token in tokens if index == chosen)


def _set_param_value_on_cards(
    cards: list[SpiceCard],
    site: _Site,
    value: ScalarValue,
    *,
    file: _ClosureFile,
) -> None:
    chosen = _chosen_param_token(cards, site, file=file)
    if chosen is None:
        raise VariationError(
            "ambiguous_target",
            f".param {site.name!r} disappeared during expansion",
        )
    index, token = chosen
    cards[index].replace_span(
        token.body_offset,
        token.body_end,
        f"{token.key}={_render_scalar(value)}",
    )


def _instance_view(card: SpiceCard, ref: str, what: str) -> InstanceLine:
    """Read an instance card, naming the reference when the card will not parse.

    A declaration the tokenizer refuses is skipped while the target index is
    built, so the ambiguity guards never see it and cannot tell whether it is a
    second declaration of the reference they cleared. The writer meets it
    anyway, and has to say which reference it was rather than surface a raw
    tokenizer fault from underneath a Monte Carlo.
    """
    try:
        return InstanceLine.from_card(card)
    except ValueError as exc:
        raise VariationError(
            "ambiguous_target",
            f"{what} {ref!r} has a declaration on line {card.line_start} that could not "
            f"be read ({exc}), so it cannot be resolved to one declaration",
        ) from exc


def _instance_cards(cards: list[SpiceCard], ref: str) -> list[int]:
    target = ref.casefold()
    return [
        index
        for index, card in enumerate(cards)
        if card.kind == "instance" and card.name and card.name.casefold() == target
    ]


def _set_component_value(
    cards: list[SpiceCard],
    site: _Site,
    value: ScalarValue,
    *,
    file: _ClosureFile,
) -> None:
    matches = _instance_cards(cards, site.name)
    if not matches:
        raise VariationError(
            "ambiguous_target",
            f"Component {site.name!r} disappeared during expansion",
        )
    chosen = _card_at_site(
        cards,
        matches,
        site,
        file=file,
        what=f"component {site.name!r}",
    )
    try:
        component_value.apply_value_to_instance(cards[chosen], _render_scalar(value))
    except NetlistError as exc:
        raise VariationError("invalid_assignment_value", str(exc)) from exc


def _set_instance_model(
    cards: list[SpiceCard],
    ref: str,
    value: str,
) -> None:
    """Rewrite the model token on every instance carrying ``ref``.

    A model swap arrives from a glob, and a glob means every match — including
    the copies of one reference that a file holding several ``.subckt``
    definitions declares.
    """
    matches = _instance_cards(cards, ref)
    if not matches:
        raise VariationError("ambiguous_target", f"Instance {ref!r} disappeared during expansion")
    for index in matches:
        view = _instance_view(cards[index], ref, "Instance")
        if view.model is None:
            raise VariationError(
                "ambiguous_target",
                f"Instance {ref!r} has no model token to rewrite",
            )
        view.set_model(value)


def _apply_random_rules(
    closure: _DeckClosure,
    texts: dict[int, str],
    rules: list[RandomRule],
    sampler: MCSampler,
) -> dict[str, float]:
    draws: dict[str, float] = {}
    head = _mismatch_head(rules)
    for rule in rules:
        if isinstance(rule, MismatchRule):
            # The whole mismatch set is applied where the first of them sits,
            # so the set keeps its place in declaration order. They share one
            # plan: each plan clones the device subcircuit it patches, and two
            # built one after the other would clone one device twice.
            if rule is not head:
                continue
            # One memo for the set, not one per file: each lookup costs a walk
            # of the whole closure, a mismatch rule asks for one per device,
            # and the closure's .model cards do not move while the rules run —
            # a mismatch edit only adds per-instance variants, under names of
            # their own.
            draws.update(
                _apply_mismatch_rules(
                    closure, texts, rules, sampler, _model_card_lookup(closure, texts)
                )
            )
            continue
        for target in _resolve_random_rule_targets(closure, rule):
            index = target.file.index
            text = texts[index]
            site = target.site
            if isinstance(rule, ComponentRule):
                text, sampled = _apply_component_rule(text, rule, sampler, target.refs)
            elif isinstance(rule, ParamRule):
                assert site is not None
                text, sampled = _apply_param_rule(text, rule, sampler, site=site, file=target.file)
            else:
                assert site is not None
                text, sampled = _apply_model_rule(text, rule, sampler, site=site, file=target.file)
            texts[index] = text
            # Draws are keyed by the name of what they perturbed, never by a
            # position in the match list, so merging one file's results into
            # the next one's is order-independent. Two files declaring one
            # reference would collide here, which is why resolution refuses
            # that deck rather than letting the last write win.
            draws.update(sampled)
    return draws


def _model_card_lookup(
    closure: _DeckClosure,
    texts: dict[int, str],
) -> Callable[[str], str | None]:
    """Memoize ``_closure_model_card`` for the span of one mismatch rule.

    The lookup walks every file to see whether a second declaration exists, so
    it costs a full closure parse; a mismatch rule asks it once per device and
    a deck's devices share a handful of models. Caching by name over one rule
    is exact: the rule reads its instances once, before it edits anything, so
    every name it asks for is one that was declared before it started — and the
    variant cards it injects along the way are named for the instance, so they
    can never answer a lookup already in the cache.
    """
    cache: dict[str, str | None] = {}

    def lookup(name: str) -> str | None:
        if name not in cache:
            cache[name] = _closure_model_card(closure, texts, name)
        return cache[name]

    return lookup


def _closure_model_card(closure: _DeckClosure, texts: dict[int, str], name: str) -> str | None:
    """Find the one .model card the closure declares under ``name``.

    A mismatch rule perturbs an instance, and the model that instance names is
    very often the thing the include was factored out for. Two declarations of
    that name — one per corner section of a library, one per file — leave no
    way to tell which set of nominals the instance actually simulates with, so
    the closure says so instead of reading the first one it walks past.

    This reads the live texts rather than the target index staging built. It
    has to: a mismatch rule INJECTS a per-instance variant card and repoints
    its instance at it, so a second rule matching the same device asks for a
    name the index never saw, and answering from the index alone reports a card
    that is right there as missing. Sections are derived only once a second
    declaration turns up, since they feed nothing but that error's label.
    """
    target = name.casefold()
    found: list[tuple[_ClosureFile, list[SpiceCard], int]] = []
    for file in closure.files:
        cards = lex(texts[file.index]).cards
        found.extend(
            (file, cards, index)
            for index, card in enumerate(cards)
            if card.kind == "model" and card.name and card.name.casefold() == target
        )
    if not found:
        return None
    if len(found) > 1:
        labels: set[str] = set()
        for file, cards, index in found:
            card = cards[index]
            sections = card_sections(cards, file.path, file.depth)
            labels.add(
                f"{file.path.name} "
                f"{_site_label(_Site(card.name or name, card.scope, sections[index]))}"
            )
        raise VariationError(
            "ambiguous_target",
            f"Circuit {closure.circuit_id!r}: mismatch model {name!r} has "
            f"{len(found)} declarations ({', '.join(sorted(labels))}); the "
            "nominals a mismatch draw perturbs have to come from one card",
        )
    _, cards, index = found[0]
    return "".join(cards[index].raw_lines)


def _include_referrers(closure: _DeckClosure) -> dict[int, set[int]]:
    """Map each closure file to the files whose references point at it."""
    by_path = {file.path.resolve(): file.index for file in closure.files}
    referrers: dict[int, set[int]] = {file.index: set() for file in closure.files}
    for file in closure.files:
        for target in staged_reference_targets(file.text, file.path, depth=file.depth):
            index = by_path.get(target)
            if index is not None and index != file.index:
                referrers[index].add(file.index)
    return referrers


def _case_copy_name(case_index: int, name: str) -> str:
    return f"case-{case_index:04d}__{name}"


def _write_case_includes(
    closure: _DeckClosure,
    referrers: dict[int, set[int]],
    texts: dict[int, str],
    case_index: int,
) -> str:
    """Write this case's private copies of the includes it edited.

    Isolation is by construction: a case only ever creates new ``case-NNNN__``
    files beside the shared staged originals and never writes to a path any
    other case reads, so the staged closure a sibling case consumes cannot move
    under it. Every file on the include chain above an edited one is copied too
    — otherwise the copy would be written and nothing would point at it.
    """
    edited = {index for index, text in texts.items() if text != closure.files[index].text}
    copies = {0}
    pending = list(edited)
    while pending:
        index = pending.pop()
        if index in copies:
            continue
        copies.add(index)
        pending.extend(referrers.get(index, ()))
    renames = {
        closure.files[index].path.resolve(): _case_copy_name(
            case_index, closure.files[index].path.name
        )
        for index in copies
        if index != 0
    }
    for index in sorted(copies):
        file = closure.files[index]
        texts[index] = rewrite_staged_references(
            texts[index],
            file.path,
            renames,
            depth=file.depth,
        )
        if index == 0:
            continue
        destination = file.path.with_name(renames[file.path.resolve()])
        destination.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(destination, texts[index], durable=True)
    return texts[0]


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
    refs: frozenset[str],
) -> tuple[str, dict[str, float]]:
    """Perturb the references resolution matched in this file, one draw each.

    ``refs`` rather than the pattern: resolution already evaluated the glob,
    and evaluating it a second time here is what would let the set that gets
    perturbed differ from the set that gets reported.
    """
    result = lex(text)
    matched: list[tuple[str, float]] = []
    for card in result.cards:
        if card.kind != "instance" or not card.name or card.name.casefold() not in refs:
            continue
        view = _instance_view(card, card.name, "Random component target")
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
    *,
    site: _Site,
    file: _ClosureFile,
) -> tuple[str, dict[str, float]]:
    cards = lex(text).cards
    chosen = _chosen_param_token(cards, site, file=file)
    nominal = None if chosen is None else parse_value(chosen[1].value or "")
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
    _set_param_value_on_cards(cards, site, sampled, file=file)
    return emit(cards), {f"random:param:{rule.target}": sampled}


def _render_scalar(value: ScalarValue) -> str:
    if isinstance(value, float):
        return f"{value:.12g}"
    return str(value)


def _apply_model_rule(
    text: str,
    rule: ModelRule,
    sampler: MCSampler,
    *,
    site: _Site,
    file: _ClosureFile,
) -> tuple[str, dict[str, float]]:
    cards = lex(text).cards
    matches = [
        index
        for index, card in enumerate(cards)
        if card.kind == "model" and card.name and card.name.casefold() == rule.target.casefold()
    ]
    if not matches:
        raise VariationError(
            "ambiguous_target",
            f"Random model target {rule.target!r} has no local .model card",
        )
    chosen = _card_at_site(
        cards,
        matches,
        site,
        file=file,
        what=f"random model target {rule.target!r}",
    )
    nominals = parse_model_params("".join(cards[chosen].raw_lines))
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
    view = ModelCard.from_card(cards[chosen])
    for param, new_value in perturbations.items():
        view.set_param(param, new_value)
    return emit(cards), {f"random:model:{rule.target}.{rule.param}": sampled}


def _engine_rule(rule: MismatchRule) -> EngineMismatchRule:
    return EngineMismatchRule(
        prefix=rule.prefix,
        avt=rule.AVT,
        ak=rule.AK,
        distribution=rule.distribution,
        vth_param=rule.vth_param,
        k_param=rule.k_param,
        min_wl_um2=rule.min_wl_um2,
    )


def _flat_mismatch_files(
    closure: _DeckClosure,
    texts: dict[int, str],
    rules: list[MismatchRule],
) -> tuple[tuple[int, ...], ...]:
    """Per rule, in the rules' order, the files holding a top-level match.

    One extraction per file for the whole rule set rather than one per rule:
    the parse is what costs, and every rule asks the same question of the same
    text. A file whose reference index holds nothing any prefix could claim is
    not parsed at all — that index covers every scope while the extraction
    reads top level only, so it can over-admit a file but never hide one.
    """
    prefixes = tuple(rule.prefix.casefold() for rule in rules)
    if not prefixes:
        return ()
    hits: list[list[int]] = [[] for _ in rules]
    for file in closure.files:
        if not any(ref.startswith(prefixes) for ref in file.targets.components):
            continue
        refs = [instance.ref for instance in extract_mosfet_instances(texts[file.index])]
        for position, rule in enumerate(rules):
            if any(matches_prefix(ref, rule.prefix) for ref in refs):
                hits[position].append(file.index)
    return tuple(tuple(indexes) for indexes in hits)


def _mismatch_head(rules: list[RandomRule]) -> MismatchRule | None:
    """The mismatch rule that carries the whole set, or None if there is none."""
    return next((rule for rule in rules if isinstance(rule, MismatchRule)), None)


# The answer for a closure no prefix could reach into, so the descent walk is
# skipped rather than run to prove itself empty.
_NO_DESCENT = MismatchPlan(targets=(), skips=(), clones=())


def _descend(
    closure: _DeckClosure,
    texts: dict[int, str],
    rules: list[MismatchRule],
) -> tuple[list[ClosureFile], MismatchPlan]:
    """The plan covering every mismatch rule at once, and the files it read.

    One plan for the whole set, not one per rule: the descent clones the device
    subcircuit it patches, and per-rule plans would clone one device once per
    rule, under a different name each time.
    """
    if not _has_subckt_candidate(closure, rules):
        return [], _NO_DESCENT
    files = _closure_files(closure, texts)
    return files, _mismatch_plan(closure, files, prefix=[rule.prefix for rule in rules])


def _refuse_unreached(
    closure: _DeckClosure,
    rules: list[MismatchRule],
    flat: tuple[tuple[int, ...], ...],
    plan: MismatchPlan,
) -> None:
    """Refuse a mismatch rule that reaches no device at either level.

    A rule that quietly matches nothing leaves every run identical while the
    job still reports success — zero measured spread, which reads as a design
    that is insensitive to mismatch rather than one that was never perturbed.
    """
    unmatched = [
        rule
        for rule, indexes in zip(rules, flat, strict=True)
        if not indexes
        and not any(matches_prefix(target.x_ref, rule.prefix) for target in plan.targets)
    ]
    if not unmatched:
        return
    prefixes = ", ".join(repr(rule.prefix) for rule in unmatched)
    raise VariationError(
        "ambiguous_target",
        f"Circuit {closure.circuit_id!r}: mismatch prefix {prefixes} matches no "
        f"top-level device with numeric W/L, and no X instance reaching a MOS device "
        f"one subcircuit level down{_closure_scope(closure)}",
    )


def _validate_mismatch_rules(closure: _DeckClosure, rules: list[MismatchRule]) -> None:
    """Resolve the mismatch rules against the unedited closure and refuse misses.

    Expansion's half of the resolution materialization runs, so a rule cannot
    validate here and then reach nothing there. The plan it builds is the one
    materialization reuses.
    """
    texts = closure.texts()
    _, plan = _descend(closure, texts, rules)
    _refuse_unreached(closure, rules, _flat_mismatch_files(closure, texts, rules), plan)


def _apply_mismatch_rules(
    closure: _DeckClosure,
    texts: dict[int, str],
    rules: list[RandomRule],
    sampler: MCSampler,
    model_card: Callable[[str], str | None],
) -> dict[str, float]:
    """Perturb every device the mismatch rules reach, at whichever level.

    A rule answered by top-level ``M`` cards is applied on its own, in
    declaration order, exactly as before subcircuit descent existed — each
    injects its own variant model card and the next rule reads what the one
    before it wrote. Descent is then attempted for every rule, not only for the
    ones the top level left empty, because a deck holding plain transistors
    alongside X-wrapped ones is ordinary and each rule should reach the devices
    its prefix names.
    """
    mismatch_rules = [rule for rule in rules if isinstance(rule, MismatchRule)]
    flat = _flat_mismatch_files(closure, texts, mismatch_rules)
    draws: dict[str, float] = {}
    for rule, indexes in zip(mismatch_rules, flat, strict=True):
        for index in indexes:
            text, sampled = _apply_mismatch_rule(
                texts[index], rule, sampler, model_card=model_card
            )
            texts[index] = text
            draws.update(sampled)

    # Descent reads the closure as the flat pass left it: a variant model card
    # injected above is part of the file the clone is copied out of.
    files, plan = _descend(closure, texts, mismatch_rules)
    _refuse_unreached(closure, mismatch_rules, flat, plan)
    if plan.targets:
        _refuse_overlapping_rules(closure, plan, mismatch_rules)
        draws.update(_apply_subckt_mismatch(closure, texts, files, plan, mismatch_rules, sampler))
    return draws


def _has_subckt_candidate(closure: _DeckClosure, rules: list[MismatchRule]) -> bool:
    """Is there an X reference any of these prefixes could reach?

    Read off the target index the closure already built, so a deck of plain
    transistors does not pay for a descent walk that can only come back empty.
    """
    prefixes = tuple(rule.prefix.casefold() for rule in rules)
    if not prefixes:
        return False
    return any(
        ref.startswith("x") and ref.startswith(prefixes)
        for file in closure.files
        for ref in file.targets.components
    )


def _refuse_overlapping_rules(
    closure: _DeckClosure,
    plan: MismatchPlan,
    siblings: list[MismatchRule],
) -> None:
    """Refuse two rules that both claim one X-wrapped device.

    A device reached through a subcircuit takes its values on its own instance
    line, so a second rule would overwrite the first rather than layer on it —
    unlike a top-level device, where each rule injects its own model card and
    the next reads the one before. Rather than pick a winner by declaration
    order, say which prefixes collide.
    """
    collision = overlapping_claims(plan, [rule.prefix for rule in siblings])
    if collision is None:
        return
    x_ref, prefixes = collision
    raise VariationError(
        "overlapping_mismatch_rules",
        f"Circuit {closure.circuit_id!r}: instance {x_ref} is claimed by "
        f"{len(prefixes)} mismatch rules (prefixes "
        f"{', '.join(repr(p) for p in prefixes)}); a device reached "
        "through a subcircuit takes one rule's values, so narrow the prefixes "
        "until each device is claimed once",
    )


def _apply_subckt_mismatch(
    closure: _DeckClosure,
    texts: dict[int, str],
    files: list[ClosureFile],
    plan: MismatchPlan,
    rules: list[MismatchRule],
    sampler: MCSampler,
) -> dict[str, float]:
    """Draw and write per-instance values for devices one subcircuit level down.

    The receipt carries what the rules asked for against what they reached,
    because a device the plan could not descend into is a fact the reader needs
    — an aggregate that reports only the draws it made cannot be told apart
    from one where every device was covered.
    """
    drawn = draw_mismatch(plan, [_engine_rule(rule) for rule in rules], sampler)
    draws: dict[str, float] = {}
    for ref, delvto in drawn.delvto.items():
        draws[f"random:mismatch:{ref}.{VTH_PARAM}"] = delvto
        draws[f"random:mismatch:{ref}.{MOBILITY_PARAM}"] = drawn.mulu0[ref]

    draws["random:mismatch:requested"] = float(len(plan.targets) + len(plan.skips))
    draws["random:mismatch:matched"] = float(len(plan.targets))
    draws["random:mismatch:applied"] = float(len(drawn.values))
    draws["random:mismatch:skipped"] = float(len(plan.skips))
    for skip in plan.skips:
        draws[f"random:mismatch:{skip.x_ref}:{skip.code}"] = 1.0
    if drawn.values:
        try:
            texts.update(render(files, plan, drawn.values))
        except MismatchPlanError as exc:
            raise VariationError(exc.code, f"Circuit {closure.circuit_id!r}: {exc}") from exc
    return draws


def _apply_mismatch_rule(
    text: str,
    rule: MismatchRule,
    sampler: MCSampler,
    *,
    model_card: Callable[[str], str | None],
) -> tuple[str, dict[str, float]]:
    engine_rule = _engine_rule(rule)
    instances = [
        instance
        for instance in extract_mosfet_instances(text)
        if matches_prefix(instance.ref, rule.prefix)
    ]
    if not instances:
        raise VariationError(
            "ambiguous_target",
            f"Mismatch prefix {rule.prefix!r} matches no top-level device with numeric W/L",
        )
    draws: dict[str, float] = {}
    for instance in instances:
        base_card = model_card(instance.model_name)
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
