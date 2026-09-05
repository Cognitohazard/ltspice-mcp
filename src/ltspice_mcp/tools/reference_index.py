"""A searchable index of every branch the seven tools take.

Each tool hides a vocabulary behind a discriminator: ``analyze_results`` has
twenty-one recipe metrics, ``edit_schematic`` eleven ops, ``run_experiments``
its variation kinds and random rules, ``inspect`` its query kinds,
``verify_circuit`` its checks, ``jobs`` its actions. A host that searches tools
sees only names and descriptions, so nothing there can lead it to "phase
margin" or "total harmonic distortion"; and on the ``compact`` tool listing
every per-argument description is stripped, so the fields of a branch are not
on the wire at all. Either way the vocabulary is reachable only by reading the
whole packaged guide.

This module is the index that answers instead: one entry per branch, carrying
the branch name, a one-line summary, how the call is written, and the branch's
fields with their types, defaults, bounds and units. ``inspect(kind="reference")``
searches it; the entries themselves are built by walking the same Pydantic
models the wire validates against, so a recipe or an op added to a union
appears here without anything else being edited — and
``tests/test_reference_index.py`` fails if one arrives without a summary.

Two things are hand-written, and only two: the one-line summary for a branch
whose model carries no docstring, and the plain-English synonyms a person
actually types ("distortion" for ``thd``). Both are keyed by discriminant and
both are checked for completeness, so they cannot silently fall behind the
models.

It lives in ``tools/`` rather than ``lib/`` because what it indexes *is* the
tool layer — the input models of the seven registered tools. A ``lib`` module
may not import ``tools`` (see ``tests/test_dispatch.py::TestLayering``), and an
index of the tool vocabulary that could not name its sources would be an index
of nothing. The field-reading it shares with ``api.reference()`` is the part
that is layer-neutral, and that part is in ``lib/model_fields.py``.
"""

from __future__ import annotations

import functools
import json
import re
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

from ltspice_mcp.lib.model_fields import (
    accepted_annotation,
    constraint_label,
    default_label,
    describe_field,
    field_name,
    first_sentence,
    item_model,
    literal_values,
    model_of,
    non_null,
    strip_annotated,
    type_label,
    union_members,
)

#: How deep a branch's nested argument objects are flattened onto dotted names.
#: Depth 2 reaches ``spec.min`` and ``window.start``, which is every nesting the
#: current surface has; anything deeper is described by its own object line.
_MAX_FIELD_DEPTH = 2

#: Upper bound on the fields one entry lists. No branch is near it; it is what
#: keeps a future recursive model from rendering forever.
_MAX_FIELDS = 60


@dataclass(frozen=True)
class FieldEntry:
    """One argument of one branch, as a caller has to write it."""

    name: str
    type: str
    required: bool
    default: str
    description: str

    def as_dict(self) -> dict[str, Any]:
        entry: dict[str, Any] = {"name": self.name, "type": self.type}
        if self.required:
            entry["required"] = True
        else:
            entry["default"] = self.default
        if self.description:
            entry["description"] = self.description
        return entry


@dataclass(frozen=True)
class BranchEntry:
    """One branch of one tool: what it is called, what it does, what it takes."""

    tool: str
    family: str
    name: str
    summary: str
    call: str
    synonyms: tuple[str, ...]
    fields: tuple[FieldEntry, ...]

    def as_dict(self, *, with_fields: bool = True) -> dict[str, Any]:
        entry: dict[str, Any] = {
            "tool": self.tool,
            "family": self.family,
            "name": self.name,
            "summary": self.summary,
        }
        if with_fields:
            entry["call"] = self.call
            entry["fields"] = [field.as_dict() for field in self.fields]
        return entry


# ---------------------------------------------------------------------------
# The hand-written half: summaries where a model has no docstring, and the
# plain words a person types instead of the discriminant.
# ---------------------------------------------------------------------------

#: ``(tool, branch) -> one-line summary``. Present only where the model carries
#: no docstring of its own, or where the docstring is written for a reader who
#: already knows the surface. A branch with neither is a test failure.
_SUMMARIES: dict[tuple[str, str], str] = {
    # analyze_results recipes
    ("analyze_results", "summary"): (
        "Everything the run says about itself: analysis type, axis ranges, signal "
        "names, .meas table, Fourier and AC bandwidth figures, and log diagnostics."
    ),
    ("analyze_results", "measurements"): (
        "The .meas table the simulator computed, per run, optionally as a histogram "
        "over a Monte Carlo spread."
    ),
    ("analyze_results", "value"): (
        "One SPICE expression evaluated on the result, optionally at one point on the axis."
    ),
    ("analyze_results", "signal_stats"): (
        "Amplitude statistics of a transient signal: min, max, mean, RMS, "
        "peak-to-peak and standard deviation over an optional window."
    ),
    ("analyze_results", "edges"): (
        "Rise and fall times of a transient signal, between level thresholds you "
        "give or ones inferred from the trace."
    ),
    ("analyze_results", "timing"): (
        "Propagation delay between an edge on one signal and an edge on another."
    ),
    ("analyze_results", "periodic"): (
        "Period, frequency and duty cycle of a repetitive transient signal."
    ),
    ("analyze_results", "transient_response"): (
        "Step response (overshoot, settling time, final value) or load-disturbance "
        "response (deviation, recovery time) of a transient signal."
    ),
    ("analyze_results", "thd"): (
        "Total harmonic distortion of a transient signal, as a percentage and in "
        "dB, with the per-harmonic amplitudes."
    ),
    ("analyze_results", "bode_filter"): (
        "Filter characteristics of an .AC sweep: passband gain and ripple, cutoff "
        "frequencies, bandwidth, stopband rejection, roll-off slope and order."
    ),
    ("analyze_results", "bode_point"): "Gain in dB and phase in degrees at one frequency.",
    ("analyze_results", "bode_crossing"): (
        "Every frequency where the gain crosses a dB level, or where the unwrapped "
        "phase crosses a degree level."
    ),
    ("analyze_results", "bode_slope"): (
        "Average slope in dB per decade between two frequencies — the roll-off rate."
    ),
    ("analyze_results", "stability"): (
        "Loop stability from an .AC sweep: phase margin in degrees, gain margin in "
        "dB, the unity-gain crossover frequency, and DC gain."
    ),
    ("analyze_results", "ac_structure"): (
        "Pole and zero corners read off the .AC magnitude and phase shape, with "
        "each corner's frequency and slope change."
    ),
    ("analyze_results", "resonance"): (
        "Resonant peaks in an .AC sweep: peak frequency, peak gain and Q."
    ),
    ("analyze_results", "return_loss"): (
        "Return loss in dB, VSWR and reflection coefficient against a reference "
        "impedance, from an .AC impedance trace."
    ),
    ("analyze_results", "noise_integral"): (
        "Integrated RMS noise over a frequency band from a .noise run."
    ),
    ("analyze_results", "operating_point"): (
        "The DC bias point: node voltages, branch currents, and per-device "
        "small-signal parameters (gm, gds, vth, id) for one device or all of them."
    ),
    ("analyze_results", "waveform"): (
        "Sample data for named signals, either decimated inline or written to a "
        "CSV file whose handle comes back."
    ),
    ("analyze_results", "plot"): (
        "A chart of named signals, written to an image file whose handle comes back."
    ),
    # run_experiments variations
    ("run_experiments", "assign"): (
        "A declared sweep: each target takes a list of values, combined as a "
        "cartesian grid or lock-step. Targets resolve as a model swap, a "
        "per-instance mismatch delta, a .param, or a component value."
    ),
    ("run_experiments", "random"): (
        "One Monte Carlo family: N runs perturbed by the rules below, reproducible "
        "from 'seed'. At most one random entry per call."
    ),
    # run_experiments random rules
    ("run_experiments", "component"): (
        "Perturb a component's value by a tolerance, e.g. 1% resistors. 'target' is "
        "a reference or a glob."
    ),
    ("run_experiments", "param"): "Perturb a declared .param by a tolerance.",
    ("run_experiments", "model"): (
        "Perturb one parameter of a .model card by a tolerance — process spread on a device model."
    ),
    ("run_experiments", "mismatch"): (
        "Per-instance device mismatch in the Pelgrom form: sigma(dVTH) = AVT/sqrt(W*L) "
        "and sigma(dK)/K = AK/sqrt(W*L), drawn independently per device per run."
    ),
    # verify_circuit checks
    ("verify_circuit", "syntax"): (
        "Netlist only: SPICE syntax, directive spelling and element arity, plus "
        "connectivity facts such as a node wired to one terminal."
    ),
    ("verify_circuit", "symbols"): (
        "Schematic only: every symbol and pin the sheet refers to resolves in the "
        "active symbol libraries."
    ),
    ("verify_circuit", "export"): (
        "Schematic only: run the LTspice netlist exporter and report what it "
        "produced, including wires the file appears to contain but the export drops."
    ),
    ("verify_circuit", "layout"): (
        "Schematic only: geometric facts — overlapping bodies, wires crossing a "
        "symbol body, floating pins, dangling wire ends."
    ),
    ("verify_circuit", "quality"): (
        "Hygiene facts: on a sheet, nets joined only by label stubs and text "
        "anchored inside a body; on a netlist, connectivity such as a net with no "
        "DC path to ground."
    ),
    ("verify_circuit", "compare"): (
        "Compare against a reference netlist: 'equivalence' graph-compares "
        "connectivity, 'structural_diff' reports the added, removed and changed "
        "delta. Needs 'compare' (or the retained flat 'reference')."
    ),
}

#: ``(tool, branch) -> the words a person types instead of the discriminant``.
#: Search ranks a synonym hit above a field-name hit, so this is what makes
#: "phase margin", "distortion" and "bias point" reach the right branch. Only
#: where the discriminant is not itself the phrase a caller would search for.
_SYNONYMS: dict[tuple[str, str], tuple[str, ...]] = {
    ("analyze_results", "summary"): ("overview", "what is in this run", "signals available"),
    ("analyze_results", "measurements"): (".meas", "meas", "measure statement"),
    ("analyze_results", "value"): ("expression", "evaluate", "node voltage", "current"),
    ("analyze_results", "signal_stats"): ("rms", "peak to peak", "ripple", "amplitude", "mean"),
    ("analyze_results", "edges"): ("rise time", "fall time", "slew", "transition time"),
    ("analyze_results", "timing"): ("propagation delay", "delay", "skew", "setup"),
    ("analyze_results", "periodic"): ("duty cycle", "oscillation frequency", "period"),
    ("analyze_results", "transient_response"): (
        "overshoot",
        "settling time",
        "step response",
        "load step",
        "load regulation transient",
        "droop",
        "recovery",
    ),
    ("analyze_results", "thd"): (
        "total harmonic distortion",
        "distortion",
        "harmonics",
        "linearity",
    ),
    ("analyze_results", "bode_filter"): (
        "cutoff frequency",
        "corner frequency",
        "bandwidth",
        "passband",
        "stopband",
        "filter order",
        "3 db point",
    ),
    ("analyze_results", "bode_point"): ("gain at a frequency", "phase at a frequency"),
    ("analyze_results", "bode_crossing"): (
        "0 db crossing",
        "unity gain frequency",
        "phase crossing",
        "where the gain crosses",
    ),
    ("analyze_results", "bode_slope"): ("roll off", "db per decade", "slope"),
    ("analyze_results", "stability"): (
        "phase margin",
        "gain margin",
        "unity gain bandwidth",
        "crossover frequency",
        "dc gain",
        "open loop gain",
        "loop stability",
        "is it stable",
    ),
    ("analyze_results", "ac_structure"): (
        "poles",
        "zeros",
        "dominant pole",
        "pole zero",
        "corner frequencies",
    ),
    ("analyze_results", "resonance"): ("q factor", "peaking", "resonant frequency", "tank"),
    ("analyze_results", "return_loss"): ("vswr", "s11", "reflection", "matching", "impedance"),
    ("analyze_results", "noise_integral"): (
        "noise",
        "input referred noise",
        "output noise",
        "integrated noise",
    ),
    ("analyze_results", "operating_point"): (
        "bias point",
        "dc operating point",
        "gm",
        "gds",
        "vth",
        "transconductance",
        "quiescent current",
        "op point",
    ),
    ("analyze_results", "waveform"): ("trace", "samples", "export csv", "raw data", "time series"),
    ("analyze_results", "plot"): ("chart", "graph", "draw the waveform", "png"),
    ("run_experiments", "assign"): (
        "sweep",
        "parameter sweep",
        "corners",
        "grid",
        "step a value",
    ),
    ("run_experiments", "random"): ("monte carlo", "tolerance analysis", "statistical", "yield"),
    ("run_experiments", "component"): ("resistor tolerance", "1 percent", "component spread"),
    ("run_experiments", "param"): ("parameter tolerance", "param spread"),
    ("run_experiments", "model"): ("process variation", "model parameter spread"),
    ("run_experiments", "mismatch"): (
        "pelgrom",
        "device mismatch",
        "offset voltage",
        "avt",
        "threshold mismatch",
    ),
    ("inspect", "capabilities"): (
        "which simulators",
        "server status",
        "what can this server do",
        "limits",
    ),
    ("inspect", "symbols"): ("available parts", "symbol library", "what can i place"),
    ("inspect", "symbol"): ("pin positions", "bounding box", "rotation", "geometry"),
    ("inspect", "net"): ("trace a net", "what is connected", "shorts", "connectivity"),
    ("inspect", "components"): ("list parts", "bill of materials", "what is on the sheet"),
    ("inspect", "model"): ("find a part", "subckt", "transistor model", "library search"),
    ("inspect", "reference"): (
        "what arguments",
        "which recipe",
        "vocabulary",
        "how do i call",
        "options",
    ),
    ("edit_schematic", "add_component"): ("place a part", "new resistor", "insert"),
    ("edit_schematic", "move_component"): ("reposition", "rotate a part"),
    ("edit_schematic", "remove_component"): ("delete a part",),
    ("edit_schematic", "set_component_value"): ("change a resistor", "set capacitance"),
    ("edit_schematic", "set_component_attribute"): ("spiceline", "spicemodel", "value2"),
    ("edit_schematic", "add_net_label"): ("name a net", "label", "flag", "vdd"),
    ("edit_schematic", "remove_net_label"): ("delete a label",),
    ("edit_schematic", "wire_pins"): ("connect", "draw a wire", "route", "join two pins"),
    ("edit_schematic", "remove_wire"): ("delete a wire", "disconnect"),
    ("edit_schematic", "add_directive"): (".tran", ".ac", "spice directive", "simulation command"),
    ("edit_schematic", "remove_directive"): ("delete a directive",),
    ("jobs", "status"): ("is it done", "progress", "check a job"),
    ("jobs", "wait"): ("block until finished", "long poll"),
    ("jobs", "cancel"): ("stop a run", "kill"),
    ("jobs", "list"): ("recent circuits", "what did i run", "find an earlier job"),
    ("jobs", "runs"): ("raw file path", "log path", "per run records"),
    ("verify_circuit", "syntax"): ("lint", "does this deck parse", "arity"),
    ("verify_circuit", "symbols"): ("missing symbol", "unresolved part"),
    ("verify_circuit", "export"): ("netlist a schematic", "asc to net"),
    ("verify_circuit", "layout"): ("overlap", "floating pin", "dangling wire"),
    ("verify_circuit", "quality"): ("label island", "no dc path to ground"),
    ("verify_circuit", "compare"): ("diff", "equivalence", "is it the same circuit"),
}


# ---------------------------------------------------------------------------
# The derived half: branch sets and their fields, walked off the live models.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Source:
    """One family of branches and where its models come from."""

    tool: str
    family: str
    call: str
    #: ``(branch name, model or None, dotted prefix for that model's fields)``.
    branches: tuple[tuple[str, type[BaseModel] | None, str], ...]
    discriminator: str | None


def _members(union: Any) -> tuple[type[BaseModel], ...]:
    """The model branches of a union — annotated, bare, or already a tuple.

    Some unions on the surface already publish their members as a tuple, read
    off the union itself at import (``RECIPE_MODELS``, ``QUERY_MODELS``,
    ``JobsInput.VARIANTS``); others are the union type. Both are the same set,
    so both are accepted rather than made to agree first.
    """
    if isinstance(union, tuple):
        return union
    members = union_members(strip_annotated(union)) or ()
    return tuple(model for model in (model_of(member) for member in members) if model is not None)


def _union_branches(union: Any, discriminator: str) -> tuple[Any, ...]:
    """``(discriminant, model, "")`` for every member of a discriminated union.

    A member declaring several literals (``wire_pins`` also answers to the
    retained ``connect``) is listed under the first, which is the spelling the
    surface advertises.
    """
    branches = []
    for model in _members(union):
        values = literal_values(model, discriminator)
        if not values:  # pragma: no cover - a union member always tags itself
            continue
        branches.append((values[0], model, ""))
    return tuple(branches)


@functools.cache
def _sources() -> tuple[_Source, ...]:
    """Every family, in the order a reader should meet them.

    Imported here rather than at module scope: ``inspect_tools`` imports this
    module, so naming it back at import time would close a cycle, and this way
    ``inspect`` also stays free of ``experiments``' import cost until something
    actually asks for the index.
    """
    from ltspice_mcp.lib.recipes import RECIPE_MODELS
    from ltspice_mcp.lib.schematic_ops import SchematicOp
    from ltspice_mcp.lib.variations import RandomRule, Variation
    from ltspice_mcp.tools import inspect_tools, jobs, verify

    return (
        _Source(
            tool="run_experiments",
            family="variation",
            call='run_experiments(circuits=[...], variations=[{{"kind": "{name}", ...}}])',
            branches=_union_branches(Variation, "kind"),
            discriminator="kind",
        ),
        _Source(
            tool="run_experiments",
            family="random rule",
            call=(
                'run_experiments(variations=[{{"kind": "random", "runs": 100, '
                '"rules": [{{"rule": "{name}", ...}}]}}])'
            ),
            branches=_union_branches(RandomRule, "rule"),
            discriminator="rule",
        ),
        _Source(
            tool="analyze_results",
            family="recipe",
            call=(
                'analyze_results(sources=[{{"job_id": "..."}}], '
                'recipes=[{{"key": "k", "metric": "{name}", ...}}])'
            ),
            branches=_union_branches(RECIPE_MODELS, "metric"),
            discriminator="metric",
        ),
        _Source(
            tool="inspect",
            family="query kind",
            call='inspect(queries=[{{"kind": "{name}", ...}}])',
            branches=_union_branches(inspect_tools.QUERY_MODELS, "kind"),
            discriminator="kind",
        ),
        _Source(
            tool="edit_schematic",
            family="op",
            call='edit_schematic(target="sheet.asc", ops=[{{"op": "{name}", ...}}])',
            branches=_union_branches(SchematicOp, "op"),
            discriminator="op",
        ),
        _Source(
            tool="verify_circuit",
            family="check",
            call='verify_circuit(path="circuit.asc", checks=["{name}"])',
            # A check is a name in a list, not a model, so it has no fields of
            # its own — except `compare`, whose arguments are the compare spec.
            branches=tuple(
                (name, verify.VerifyCompareSpec if name == "compare" else None, "compare.")
                for name in verify.CHECK_ORDER
            ),
            discriminator=None,
        ),
        _Source(
            tool="jobs",
            family="action",
            call='jobs(action="{name}", ...)',
            branches=_union_branches(jobs.JobsInput.VARIANTS, "action"),
            discriminator="action",
        ),
    )


def _default_spelling(field: Any) -> str:
    """The default written the way a caller writes it in a call.

    A tool call is JSON, so ``null``/``true``/``false`` is what a caller types;
    ``repr`` would hand them Python's spelling of the same three values, which
    is the one thing on a reference card that must not be copied verbatim.
    Anything a factory produces keeps its descriptive label ("empty").
    """
    if field.default_factory is not None:
        return default_label(field)
    default = field.default
    if default is None or isinstance(default, (str, int, float, bool)):
        return json.dumps(default)
    return default_label(field)


def _field_entries(
    model: type[BaseModel],
    *,
    discriminator: str | None,
    prefix: str = "",
    depth: int = 0,
    seen: frozenset[type[BaseModel]] = frozenset(),
) -> list[FieldEntry]:
    """One branch's arguments, nested objects flattened onto dotted names."""
    entries: list[FieldEntry] = []
    seen = seen | {model}
    for name, field in model.model_fields.items():
        if depth == 0 and name == discriminator:
            continue
        spelled = f"{prefix}{field_name(model, name, field)}"
        label = type_label(accepted_annotation(field))
        bounds = constraint_label(field)
        entries.append(
            FieldEntry(
                name=spelled,
                type=f"{label} ({bounds})" if bounds else label,
                required=default_label(field) == "REQUIRED",
                default=_default_spelling(field),
                description=first_sentence(describe_field(field), limit=240),
            )
        )
        if depth + 1 >= _MAX_FIELD_DEPTH or len(entries) >= _MAX_FIELDS:
            continue
        nested = model_of(non_null(field.annotation))
        suffix = "."
        if nested is None:
            nested = model_of(item_model(field.annotation))
            suffix = "[]."
        if nested is not None and nested not in seen:
            entries.extend(
                _field_entries(
                    nested,
                    discriminator=None,
                    prefix=f"{spelled}{suffix}",
                    depth=depth + 1,
                    seen=seen,
                )
            )
    return entries[:_MAX_FIELDS]


def _summary_for(tool: str, name: str, model: type[BaseModel] | None) -> str:
    """The branch's one line: the hand-written one, else the model's docstring."""
    written = _SUMMARIES.get((tool, name))
    if written:
        return written
    if model is not None and model.__doc__:
        return first_sentence(" ".join(model.__doc__.split()), limit=240)
    return ""


@functools.cache
def build_index() -> tuple[BranchEntry, ...]:
    """Every branch of every tool, in advertised order."""
    entries: list[BranchEntry] = []
    for source in _sources():
        for name, model, prefix in source.branches:
            fields = (
                _field_entries(model, discriminator=source.discriminator, prefix=prefix)
                if model is not None
                else []
            )
            entries.append(
                BranchEntry(
                    tool=source.tool,
                    family=source.family,
                    name=name,
                    summary=_summary_for(source.tool, name, model),
                    call=source.call.format(name=name),
                    synonyms=_SYNONYMS.get((source.tool, name), ()),
                    fields=tuple(fields),
                )
            )
    return tuple(entries)


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

#: Words that carry no discriminating power in a lookup like "how do I get the
#: phase margin". Dropped before scoring so they cannot pad a weak match.
_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "can",
        "do",
        "does",
        "for",
        "from",
        "get",
        "how",
        "i",
        "in",
        "is",
        "it",
        "its",
        "me",
        "my",
        "of",
        "on",
        "or",
        "the",
        "this",
        "to",
        "want",
        "what",
        "which",
        "with",
    }
)

#: What a token is worth where it lands. The order is the ranking rule: the
#: branch's own name beats the words a person types for it, which beat a field
#: name, which beats prose. One token scores once, at the best place it hits.
_NAME_WEIGHT = 100
_SYNONYM_WEIGHT = 60
_FIELD_WEIGHT = 30
_PROSE_WEIGHT = 10

#: Whole-phrase bonuses, on top of the per-token score. They are what makes an
#: exact name or an exact synonym outrank an entry that merely shares two
#: common words with the query.
_EXACT_NAME_BONUS = 250
_NAME_PHRASE_BONUS = 80
_SYNONYM_PHRASE_BONUS = 120

_WORD = re.compile(r"[a-z0-9]+")


def _normalize(text: str) -> str:
    """Lower-case, with every separator flattened to a single space."""
    return " ".join(_WORD.findall(text.lower()))


def _tokens(text: str) -> list[str]:
    words = [word for word in _WORD.findall(text.lower()) if word not in _STOPWORDS]
    return [word for word in words if len(word) > 1] or words


@dataclass(frozen=True)
class _Haystacks:
    name: str
    synonyms: str
    fields: str
    prose: str


@functools.cache
def _haystacks() -> tuple[_Haystacks, ...]:
    """The searchable text of each entry, in ``build_index()`` order."""
    stacks = []
    for entry in build_index():
        stacks.append(
            _Haystacks(
                name=_normalize(f"{entry.tool} {entry.family} {entry.name}"),
                synonyms=_normalize(" ".join(entry.synonyms)),
                fields=_normalize(" ".join(field.name for field in entry.fields)),
                prose=_normalize(
                    " ".join([entry.summary, *(field.description for field in entry.fields)])
                ),
            )
        )
    return tuple(stacks)


def _score(entry: BranchEntry, stacks: _Haystacks, phrase: str, tokens: list[str]) -> int:
    total = 0
    for token in tokens:
        if token in stacks.name:
            total += _NAME_WEIGHT
        elif token in stacks.synonyms:
            total += _SYNONYM_WEIGHT
        elif token in stacks.fields:
            total += _FIELD_WEIGHT
        elif token in stacks.prose:
            total += _PROSE_WEIGHT
    if not total:
        return 0
    if phrase in (_normalize(entry.name), _normalize(f"{entry.name} {entry.family}")):
        total += _EXACT_NAME_BONUS
    elif phrase and phrase in stacks.name:
        total += _NAME_PHRASE_BONUS
    if phrase and phrase in stacks.synonyms:
        total += _SYNONYM_PHRASE_BONUS
    return total


def search_branches(query: str, *, limit: int) -> tuple[list[BranchEntry], int]:
    """The best ``limit`` branches for ``query``, and how many matched at all.

    Ranking is by where the query's words land — branch name, then the plain
    words a person types for it, then a field name, then prose — and ties break
    on advertised order, so the same query always returns the same list.
    """
    phrase = _normalize(query)
    tokens = _tokens(query)
    scored: list[tuple[int, int, BranchEntry]] = []
    for position, (entry, stacks) in enumerate(zip(build_index(), _haystacks(), strict=True)):
        score = _score(entry, stacks, phrase, tokens)
        if score:
            scored.append((-score, position, entry))
    scored.sort(key=lambda row: (row[0], row[1]))
    return [entry for _, _, entry in scored[:limit]], len(scored)


def table_of_contents() -> list[dict[str, Any]]:
    """Every branch name and its one line, grouped by tool and family."""
    groups: list[dict[str, Any]] = []
    for source in _sources():
        branches = [
            entry.as_dict(with_fields=False)
            for entry in build_index()
            if entry.tool == source.tool and entry.family == source.family
        ]
        groups.append(
            {
                "tool": source.tool,
                "family": source.family,
                "branches": [
                    {"name": branch["name"], "summary": branch["summary"]} for branch in branches
                ],
            }
        )
    return groups
