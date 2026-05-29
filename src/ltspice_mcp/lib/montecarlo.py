"""Monte Carlo perturbation engine.

Owns the math: given component nominal values, tolerances, distributions,
and an optional seed, produce per-run perturbed values. Knows nothing
about runners, asyncio, or filesystem layout — those concerns live in
``montecarlo_runner``.

Replaces spicelib's ``Montecarlo`` because spicelib's ``_get_sim_value``
samples ``random.gauss(value, tolerance/3)`` (absolute σ instead of
multiplicative) and uses a fresh ``random.Random()`` per call (defeats
reproducibility). Owning the math gives us the intended
``value * (1 + N(0, tol/3))`` and a seeded RNG.

Foundry-MC additions
--------------------
Beyond R/C/L value perturbation, the engine supports MOSFET-level MC the
way commercial PDKs do it (Spectre/HSPICE/Eldo conventions):

- **Process variation** — sample once per ``.MODEL`` per run; every
  instance using that model inherits the same perturbed parameters
  (correlated). Implemented by rewriting the ``.MODEL`` card.
- **Mismatch (Pelgrom)** — sample once per instance per run, scaled by
  ``σ ∝ 1/√(W·L)``. Implemented by generating per-instance variant
  ``.MODEL`` cards inlined into the per-run netlist.
- **``.PARAM`` deviation** — sample once per run for ``.PARAM`` directives
  the user already wired into model cards via ``{param}`` substitution.

Both relative (fractional %) and absolute (σ in source units) tolerances
are accepted — VTH variation is naturally absolute (σ=15 mV) while KP is
naturally relative (σ=10%).
"""

from __future__ import annotations

import hashlib
import logging
import math
import random
from dataclasses import dataclass
from typing import Literal

from ltspice_mcp.lib.format import format_spice_value, parse_spice_value
from ltspice_mcp.lib.spice_lex import (
    TokenKind,
    emit,
    lex,
    tokenize_body,
)
from ltspice_mcp.lib.spice_lex_ops import inject_card_before_end as _ops_inject
from ltspice_mcp.lib.spice_lex_views import (
    InstanceLine,
    ModelCard,
    ParamCard,
    find_model,
)

logger = logging.getLogger(__name__)


Distribution = Literal["uniform", "normal", "gaussian"]
ToleranceKind = Literal["relative", "absolute"]


# Component prefixes whose values can be perturbed without breaking
# semantics. Sources, controlled sources, and switches are excluded —
# their values often encode behavior, not magnitudes.
_PERTURBABLE_PREFIXES: frozenset[str] = frozenset({"R", "C", "L"})


@dataclass(frozen=True)
class ToleranceSpec:
    """Tolerance + distribution for one perturbation source.

    ``tolerance`` is interpreted by ``kind``:

    - ``relative`` (default): fraction of the nominal — ``σ = nominal·tol/3``
      for normal, ``±tol·nominal`` half-range for uniform. Matches the
      original R/C/L semantics and LTspice's ``ntol`` macro.
    - ``absolute``: σ (for normal) or half-range (for uniform) **in the
      source units** of whatever's being perturbed. Foundry PDKs spec VTH
      mismatch this way (σ_VTH = 15 mV, not 5%).

    Both forms produce additive offsets internally; the application site
    decides whether to add them to a nominal (e.g. ``VTO + ΔVTH``) or
    multiply (``RD * (1 + ΔR/R)``).
    """

    tolerance: float
    distribution: Distribution = "normal"
    kind: ToleranceKind = "relative"


class MCSampler:
    """Per-run value perturbation with optional seed for reproducibility.

    Streams
    -------
    Each perturbation source (R/C/L value, per-.MODEL params, per-instance
    mismatch, .PARAM) draws from its own seeded sub-stream. Adding or
    removing a source does NOT shift the sample sequence of unrelated
    sources — so adding mismatch on M5 doesn't change R1's perturbation
    history. Sub-streams are derived deterministically from the global
    seed and a string key (sha256-based, stable across Python versions).

    Pass ``stream`` to ``sample`` / ``sample_offset`` to draw from a
    specific stream; omit it to use the default ``"_default"`` stream
    (preserves legacy single-stream behaviour for code that doesn't care).

    Truncation
    ----------
    Normal samples are truncated at ±tolerance (= ±3σ by the convention
    we ship). Foundry MC default is 3σ truncation; without it the rare
    tail samples produce ``Vov < 0`` instances that break otherwise-valid
    designs in ways that don't reflect real silicon. Rejection sampling
    has ~0.27% reject rate at 3σ — negligible.
    """

    def __init__(self, seed: int | None = None):
        self._seed = seed
        # Default stream — used when callers don't pass an explicit key.
        # Backward-compatible with the previous single-RNG behaviour
        # (sample(..., stream="_default") matches sample(...) of the old code
        # for a given seed and call ordering).
        self._streams: dict[str, random.Random] = {}

    def stream(self, key: str = "_default") -> random.Random:
        """Return a per-key sub-stream RNG.

        Sub-streams are seeded from the global seed mixed with a stable
        hash of ``key`` — adding a new stream key cannot perturb existing
        streams. When the global seed is ``None`` (fresh-entropy mode)
        each stream just gets its own ``random.Random()`` instance, still
        independent.
        """
        existing = self._streams.get(key)
        if existing is not None:
            return existing
        if self._seed is None:
            rng = random.Random()
        else:
            # SHA256 of "seed:key" → stable across Python versions and
            # platforms (unlike `hash()`, which is randomised). 64-bit
            # slice is enough entropy for Mersenne Twister seeding.
            digest = hashlib.sha256(f"{self._seed}:{key}".encode()).digest()
            sub_seed = int.from_bytes(digest[:8], "big")
            rng = random.Random(sub_seed)
        self._streams[key] = rng
        return rng

    def derive(self, namespace: str) -> MCSampler:
        """Return a child sampler whose seed is deterministically derived.

        Used by the runner to scope per-run RNG state: each MC iteration
        gets its own ``MCSampler`` derived from ``"run<N>"``. Helpers
        within an iteration then use short stream keys (``"rcl:R1"``,
        ``"model:NMOS1.VTO"``) without worrying about cross-run collision.

        With a ``None`` parent seed (fresh entropy), the child also gets
        fresh entropy — namespacing is meaningless when the parent isn't
        deterministic.
        """
        if self._seed is None:
            return MCSampler(seed=None)
        digest = hashlib.sha256(f"{self._seed}:{namespace}".encode()).digest()
        child_seed = int.from_bytes(digest[:8], "big")
        return MCSampler(seed=child_seed)

    def sample(
        self,
        value: float,
        spec: ToleranceSpec,
        stream: str = "_default",
    ) -> float:
        """Return a perturbed value for an R/C/L-style nominal/tolerance pair.

        - ``normal`` / ``gaussian``: ``value * (1 + N(0, tolerance/3))``,
          truncated to ``±tolerance`` of the multiplier (= ±3σ).
        - ``uniform``: ``U(value*(1-tol), value*(1+tol))``.

        Always interpreted as multiplicative on ``value`` regardless of
        ``spec.kind``. For per-parameter (model card or ``.PARAM``)
        sampling that needs absolute support, use ``sample_offset``.
        """
        offset = _draw_zero_centred(self.stream(stream), spec.tolerance, spec.distribution)
        return value * (1.0 + offset)

    def sample_offset(
        self,
        nominal: float,
        spec: ToleranceSpec,
        stream: str = "_default",
    ) -> float:
        """Sample an additive perturbation for a parameter (model param, .PARAM).

        Returns the *delta*. Caller adds it to the nominal.

        - ``relative``: σ = ``|nominal| * tolerance / 3``; bounded by
          ``±|nominal|*tolerance``.
        - ``absolute``: σ = ``tolerance / 3``; bounded by ``±tolerance``.
        """
        scale = abs(nominal) * spec.tolerance if spec.kind == "relative" else spec.tolerance
        return _draw_zero_centred(self.stream(stream), scale, spec.distribution)


def _draw_zero_centred(
    rng: random.Random,
    bound: float,
    distribution: Distribution,
) -> float:
    """Sample a zero-mean offset of half-range ``bound`` from ``distribution``.

    ``normal`` / ``gaussian``: truncated Gaussian with σ=bound/3 (so
    ±bound = ±3σ, matching the foundry convention).
    ``uniform``: ``U(-bound, +bound)``.

    Shared by ``MCSampler.sample`` (which multiplies by the nominal
    afterwards) and ``MCSampler.sample_offset`` (which returns the raw
    delta) so the dist dispatch lives in one place.
    """
    dist = distribution.lower()
    if dist in ("normal", "gaussian"):
        return _truncated_gauss(rng, bound / 3.0, bound)
    if dist == "uniform":
        return rng.uniform(-bound, +bound)
    raise ValueError(f"Unknown distribution {dist!r}; expected 'uniform' or 'normal'")


# Module-level so it's reachable from tests; not part of the public surface.
_MAX_TRUNCATION_REJECTS = 1000


def _truncated_gauss(rng: random.Random, sigma: float, bound: float) -> float:
    """Sample from N(0, sigma) truncated to [-bound, +bound] via rejection.

    For the typical ±3σ bound (which is what our tolerance convention
    implies) the reject rate is ~0.27% — rejection sampling is fine. The
    1000-iteration cap is a safety net for callers that pass pathological
    bounds (bound much smaller than sigma); in that case we fall through
    to a clamped sample so we never spin.
    """
    if sigma == 0.0:
        return 0.0
    abs_bound = abs(bound)
    for _ in range(_MAX_TRUNCATION_REJECTS):
        x = rng.gauss(0.0, sigma)
        if -abs_bound <= x <= abs_bound:
            return x
    # Pathological case: bound << sigma. Clamp to ±bound so the function
    # is still total. This branch is unreachable for sane inputs.
    return max(-abs_bound, min(abs_bound, rng.gauss(0.0, sigma)))


# ---------------------------------------------------------------------------
# Phase 1: Process variation — per-.MODEL parameter perturbation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelTolerance:
    """Process-variation rule for one ``.MODEL`` card.

    Per Monte Carlo run, each parameter is sampled once and the perturbed
    value is written back to the ``.MODEL`` line; every transistor instance
    using this model inherits the same perturbation (foundry-correlated
    process variation).
    """

    model_name: str
    parameters: dict[str, ToleranceSpec]


def sample_model_perturbation(
    sampler: MCSampler,
    model_name: str,
    nominals: dict[str, float],
    tolerances: dict[str, ToleranceSpec],
) -> dict[str, float]:
    """Sample one run's perturbed parameters for one ``.MODEL`` card.

    For each ``param`` in ``tolerances`` that has a numeric nominal in
    ``nominals``, returns ``perturbed_value = nominal + sample_offset(...)``.
    Params absent from ``nominals`` are silently skipped — the model card
    didn't declare them, so we shouldn't invent them.

    Each ``(model, param)`` pair draws from its own sub-stream
    (``"model:<MODEL>.<PARAM>"``) so adding a new perturbed parameter
    doesn't shift the sample sequence of the existing ones.
    """
    perturbed: dict[str, float] = {}
    for param, spec in tolerances.items():
        nominal = nominals.get(param)
        if nominal is None:
            logger.debug(
                "MC: model %s param %s has no nominal in card; skipping perturbation",
                model_name,
                param,
            )
            continue
        delta = sampler.sample_offset(nominal, spec, stream=f"model:{model_name}.{param}")
        perturbed[param] = nominal + delta
    return perturbed


def perturb_model_in_text(
    netlist_text: str,
    model_name: str,
    perturbations: dict[str, float],
) -> str:
    """Rewrite the ``.MODEL <model_name> ...`` card to apply parameter overrides.

    Existing ``param=value`` tokens are replaced in place; new params are
    appended. Continuation lines and quoted/braced values are handled
    correctly via the shared ``spice_lex`` parser.

    Raises ``ValueError`` if ``model_name`` isn't found.
    """
    if not perturbations:
        return netlist_text
    result = lex(netlist_text)
    view = find_model(result.cards, model_name)
    if view is None:
        raise ValueError(f".MODEL {model_name!r} not found in netlist")
    for param, new_value in perturbations.items():
        view.set_param(param, new_value)
    return emit(result.cards)


# ---------------------------------------------------------------------------
# Phase 2: Mismatch (Pelgrom-scaled per-instance)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MismatchRule:
    """Pelgrom-law mismatch coefficients for one device prefix.

    σ(ΔVTH) = AVT / √(W·L) — typical 65nm: 3-5 mV·µm; 28nm: 1-2 mV·µm.
    σ(ΔK)/K = AK  / √(W·L) — typical 1-2 %·µm.

    Units: ``AVT`` in volts·µm (so ``3e-3`` for 3 mV·µm), ``AK`` as the
    K-mismatch fraction-µm coefficient (so ``0.02`` for 2 %·µm).

    ``vth_param`` / ``k_param`` name which model-card parameters carry the
    threshold and current-factor — defaults match Level-1 SPICE
    (``VTO``/``KP``); BSIM users should pass ``VTH0``/``U0``.

    ``min_wl_um2`` guards against division-by-zero / runaway σ for
    ill-defined instances (W·L < this in µm² uses this floor).
    """

    prefix: str
    avt: float = 0.0
    ak: float = 0.0
    distribution: Distribution = "normal"
    vth_param: str = "VTO"
    k_param: str = "KP"
    min_wl_um2: float = 1e-3


@dataclass(frozen=True)
class InstanceGeometry:
    """Geometric inputs needed to compute Pelgrom σ for one transistor."""

    ref: str
    model_name: str
    width_m: float
    length_m: float


def sample_instance_mismatch(
    sampler: MCSampler,
    instance: InstanceGeometry,
    rule: MismatchRule,
) -> dict[str, float]:
    """Sample ΔVTH and ΔK/K for one transistor instance.

    Returns a dict with keys ``"dvth"`` (volts) and ``"dk_over_k"``
    (dimensionless fraction). Either may be 0.0 if the corresponding
    coefficient was 0. Per-instance independent given a seeded sampler.

    Each ``(instance, channel)`` pair draws from its own sub-stream
    (``"mismatch:<REF>.dvth"`` / ``"mismatch:<REF>.dk_over_k"``) so
    adding a new transistor doesn't perturb existing ones' samples.
    """
    # Convert W·L to µm² for the Pelgrom denominator (coefficients given
    # in mV·µm and %·µm conventions).
    w_um = instance.width_m * 1e6
    l_um = instance.length_m * 1e6
    wl_um2 = max(w_um * l_um, rule.min_wl_um2)
    sqrt_wl = math.sqrt(wl_um2)

    dvth = 0.0
    if rule.avt > 0.0:
        sigma_vth = rule.avt / sqrt_wl
        spec = ToleranceSpec(
            tolerance=3.0 * sigma_vth,  # sample_offset divides by 3
            distribution=rule.distribution,
            kind="absolute",
        )
        dvth = sampler.sample_offset(0.0, spec, stream=f"mismatch:{instance.ref}.dvth")

    dk_over_k = 0.0
    if rule.ak > 0.0:
        sigma_dk_over_k = rule.ak / sqrt_wl
        spec = ToleranceSpec(
            tolerance=3.0 * sigma_dk_over_k,
            distribution=rule.distribution,
            kind="absolute",
        )
        dk_over_k = sampler.sample_offset(0.0, spec, stream=f"mismatch:{instance.ref}.dk_over_k")

    return {"dvth": dvth, "dk_over_k": dk_over_k}


def variant_model_name(base_model: str, instance_ref: str) -> str:
    """Generate a stable variant model name for a per-instance card.

    Foundry preprocessors emit names like ``NMOS_lvt_M1`` — a base-derived,
    instance-derived flat string. Avoid characters SPICE parsers reject
    (e.g. ``.``).
    """
    safe_ref = instance_ref.replace(".", "_")
    return f"{base_model}__{safe_ref}"


def render_variant_model_card(
    base_card: str,
    variant_name: str,
    overrides: dict[str, float],
) -> str:
    """Produce a new ``.MODEL`` card cloning ``base_card`` with overrides.

    ``base_card`` is the merged text of an existing ``.MODEL`` card (after
    process perturbation, if any). Replaces the model name with
    ``variant_name`` and applies the parameter overrides.
    """
    cards = lex(base_card).cards
    if not cards or cards[0].kind != "model":
        return base_card  # malformed — pass through
    view = ModelCard.from_card(cards[0])
    view.set_name(variant_name)
    for param, new_value in overrides.items():
        view.set_param(param, new_value)
    return emit(cards)


def extract_model_card(netlist_text: str, model_name: str) -> str | None:
    """Return the merged text of a ``.MODEL`` card by name, or None.

    The returned text is the original raw lines (continuation lines
    intact, leading/trailing whitespace and the trailing newline
    preserved).
    """
    target = model_name.lower()
    result = lex(netlist_text)
    for c in result.cards:
        if c.kind == "model" and c.name and c.name.lower() == target:
            return "".join(c.raw_lines)
    return None


# ---------------------------------------------------------------------------
# Phase 3: .PARAM perturbation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ParamTolerance:
    """Perturbation rule for one ``.PARAM`` directive.

    Sampled once per Monte Carlo run. The user is expected to wire the
    ``.PARAM`` into model/component cards via ``{param}`` substitution
    (e.g. ``.MODEL NMOS1 NMOS(VTO={vto_n})`` with ``.PARAM vto_n=0.7``);
    we just rewrite the ``.PARAM`` value per run.
    """

    name: str
    spec: ToleranceSpec


def perturb_param_in_text(
    netlist_text: str,
    name: str,
    new_value: float,
) -> str:
    """Rewrite ``.PARAM <name>=<value>`` in place.

    Case-insensitive match on the param name. Raises ``ValueError`` if
    no matching ``.PARAM`` is present.
    """
    target = name.lower()
    result = lex(netlist_text)
    for c in result.cards:
        if c.kind == "param" and c.name and c.name.lower() == target:
            view = ParamCard.from_card(c)
            view.set_value(new_value)
            return emit(result.cards)
    raise ValueError(f".PARAM {name!r} not found in netlist")


# ---------------------------------------------------------------------------
# Netlist parsing helpers — extract nominals for the various MC phases
# ---------------------------------------------------------------------------


def parse_model_params(card_text: str) -> dict[str, float]:
    """Parse all ``KEY=VALUE`` tokens from a ``.MODEL`` card into a float dict.

    Walks tokens via ``tokenize_body`` so quoted values and brace
    expressions (``KP={2*kp_n}``) are handled correctly. Skips tokens
    whose value can't be parsed via ``parse_spice_value`` (parametric
    expressions, behavioural). Keys are upper-cased per SPICE convention.
    """
    out: dict[str, float] = {}

    def _harvest(tokens: list) -> None:
        for tok in tokens:
            if tok.kind == TokenKind.KEY_VALUE:
                parsed = parse_value(tok.value)
                if parsed is not None:
                    out[tok.key.upper()] = parsed
            elif tok.kind == TokenKind.PARENED:
                # Recurse into the .MODEL's parameter group.
                _harvest(tokenize_body(tok.text[1:-1]))
            elif tok.kind == TokenKind.COMMENT_TRAIL:
                break

    _harvest(tokenize_body(card_text))
    return out


def inject_card_before_end(netlist_text: str, card: str) -> str:
    """Insert a ``.MODEL`` card (or any directive block) just before ``.END``.

    SPICE simulators reject definitions after ``.END`` — for variant
    model cards we want them visible to the rest of the deck. If no
    ``.END`` is present, append at the end. Routes through
    ``spice_lex_ops.inject_card_before_end``.
    """
    if not card.endswith("\n"):
        card = card + "\n"
    cards = lex(netlist_text).cards
    _ops_inject(cards, card)
    return emit(cards)


def rewrite_instance_model(
    netlist_text: str,
    instance_ref: str,
    new_model_name: str,
) -> str:
    """Replace the model token on a transistor (M/Q/J) instance line.

    The instance line is matched by reference (case-insensitive).
    Routes through ``InstanceLine.set_model``, which uses the
    classified-token rule to identify the model position correctly
    even with quoted model names and trailing params. Raises
    ``ValueError`` if the instance isn't found.
    """
    target = instance_ref.lower()
    result = lex(netlist_text)
    for c in result.cards:
        if c.kind == "instance" and c.name and c.name.lower() == target:
            view = InstanceLine.from_card(c)
            if view.model is None:
                raise ValueError(
                    f"Instance {instance_ref!r} has no model token to rewrite: {c.body!r}"
                )
            view.set_model(new_model_name)
            return emit(result.cards)
    raise ValueError(f"Instance {instance_ref!r} not found in netlist")


def extract_mosfet_instances(netlist_text: str) -> list[InstanceGeometry]:
    """Find every ``Mxxx`` instance and parse its model name + W/L geometry.

    Top-level only — subcircuit-instantiated transistors (``X1.M1``)
    aren't visible at the top level and so aren't returned. W/L are
    pulled from the ``InstanceLine.params`` dict, parsed via
    ``parse_spice_value`` so engineering suffixes (``10u``, ``180n``)
    work. Instances missing W, L, or a model are skipped (Pelgrom σ
    is undefined without those).
    """
    instances: list[InstanceGeometry] = []
    result = lex(netlist_text)
    for c in result.cards:
        if c.kind != "instance" or not c.name or c.name[:1].upper() != "M":
            continue
        if c.scope != ():  # top-level only
            continue
        view = InstanceLine.from_card(c)
        if view.model is None:
            continue
        w = parse_value(view.params.get("W", ""))
        l_val = parse_value(view.params.get("L", ""))
        if w is None or l_val is None:
            logger.debug(
                "MC: instance %s has no W/L parameter, skipping mismatch (params=%r)",
                view.ref,
                view.params,
            )
            continue
        instances.append(
            InstanceGeometry(
                ref=view.ref,
                model_name=view.model,
                width_m=w,
                length_m=l_val,
            )
        )
    return instances


def find_mismatch_rule(
    instance_ref: str,
    rules: list[MismatchRule],
) -> MismatchRule | None:
    """Return the first rule whose prefix matches the instance ref."""
    for rule in rules:
        if instance_ref.upper().startswith(rule.prefix.upper()):
            return rule
    return None


def parse_param_nominal(netlist_text: str, name: str) -> float | None:
    """Read the nominal value from ``.PARAM <name>=<value>``.

    Returns ``None`` when the param is absent or its value isn't a plain
    number (e.g. expressions like ``{2*Rd}``). Phase 3 silently skips such
    params — they're already user-side expressions.
    """
    target = name.lower()
    result = lex(netlist_text)
    for c in result.cards:
        if c.kind == "param" and c.name and c.name.lower() == target:
            view = ParamCard.from_card(c)
            return parse_value(view.value)
    return None


def expand_tolerances(
    component_refs: list[str],
    type_tolerances: dict[str, tuple[float, str]],
    component_overrides: dict[str, tuple[float, str]],
) -> dict[str, ToleranceSpec]:
    """Resolve type-prefix and per-ref rules to a flat ref -> spec map.

    Resolution order: ``component_overrides`` (specific ref) > matching
    prefix in ``type_tolerances`` > skip. Refs whose prefix isn't in
    ``_PERTURBABLE_PREFIXES`` are dropped (perturbing a voltage source
    would change the stimulus). Drops are logged at debug level so a
    user passing e.g. ``type_tolerances={"V": 0.05}`` can find them.
    """
    resolved: dict[str, ToleranceSpec] = {}
    skipped: list[str] = []
    for ref in component_refs:
        prefix = ref[0].upper() if ref else ""
        if prefix not in _PERTURBABLE_PREFIXES:
            skipped.append(ref)
            continue

        if ref in component_overrides:
            tol, dist = component_overrides[ref]
        elif prefix in type_tolerances:
            tol, dist = type_tolerances[prefix]
        else:
            continue
        resolved[ref] = ToleranceSpec(
            tolerance=tol,
            distribution=dist,  # type: ignore[arg-type]
        )
    if skipped:
        logger.debug(
            "MC: skipped %d non-perturbable refs (prefix not in R/C/L): %s",
            len(skipped),
            ", ".join(skipped[:8]) + ("…" if len(skipped) > 8 else ""),
        )
    return resolved


def parse_value(value: str | float) -> float | None:
    """Parse a SPICE-formatted component value to float, or None.

    Wraps ``lib.format.parse_spice_value`` with two MC-specific behaviors:
    skips parametric expressions (``{Rd}``, ``table(...)``) so they
    don't get corrupted, and accepts numeric passthrough.
    """
    if isinstance(value, int | float):
        return float(value)
    s = str(value).strip()
    if not s or "{" in s or "(" in s:
        return None
    try:
        return parse_spice_value(s)
    except ValueError:
        return None


def format_value(value: float) -> str:
    """Format a perturbed float back to a SPICE-compatible literal.

    Thin wrapper around ``format_spice_value`` kept as a public alias —
    several callers import ``format_value`` from this module by name.
    """
    return format_spice_value(value)
