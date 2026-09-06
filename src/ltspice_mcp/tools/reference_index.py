"""A searchable index of what the seven tools take: their branches and their
own arguments.

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

The tools' own top-level arguments are one more family here, one entry per
tool, named after it. A branch family covers what a caller writes *inside* a
discriminated item, which left arguments like ``all_steps``, ``budget`` and
``expected_sha256`` on neither channel a ``compact`` session has — stripped
from the listing and declared by no branch.

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
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ValidationError

from ltspice_mcp.errors import compact_validation_error
from ltspice_mcp.lib.model_fields import (
    accepted_annotation,
    constraint_label,
    default_label,
    default_spelling,
    describe_field,
    field_name,
    first_sentence,
    item_model,
    literal_values,
    model_of,
    model_union,
    non_null,
    type_label,
)

#: How deep a branch's nested argument objects are flattened onto dotted names.
#: Depth 2 reaches ``spec.min`` and ``window.start``, which is every nesting the
#: current surface has; anything deeper is described by its own object line.
_MAX_FIELD_DEPTH = 2

#: Upper bound on the fields one entry lists. No branch is near it; it is what
#: keeps a future recursive model from rendering forever.
_MAX_FIELDS = 60

#: Where the walk stops descending into nested objects. Separate from the cap
#: on what an entry publishes, because they answer different questions: this
#: one bounds the work, ``_MAX_FIELDS`` bounds the answer.
_MAX_NESTED_FIELDS = _MAX_FIELDS


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

    def as_dict(self) -> dict[str, Any]:
        return {
            "tool": self.tool,
            "family": self.family,
            "name": self.name,
            "summary": self.summary,
            "call": self.call,
            "fields": [field.as_dict() for field in self.fields],
        }


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
        "delta. Needs the 'compare' argument."
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
    #: ``(branch name, one line)`` where the line is derived from something
    #: other than the model — the tool argument tables read the tool's own
    #: registered description. A hand-written summary still wins.
    summaries: tuple[tuple[str, str], ...] = ()


def _union_branches(union: Any, discriminator: str) -> tuple[Any, ...]:
    """``(discriminant, model, "")`` for every member of a discriminated union.

    A member declaring several literals is listed under the first, which is the
    spelling the surface advertises.
    """
    branches = []
    for model in model_union(union):
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
            # its own — except where the tool declares an argument object for
            # one, whose fields are then the check's, under that object's name.
            branches=tuple(
                (name, model, f"{name}." if model is not None else "")
                for name, model in (
                    (name, verify.CHECK_ARGUMENT_MODELS.get(name)) for name in verify.CHECK_ORDER
                )
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
        *_argument_sources(),
    )


def _argument_sources() -> tuple[_Source, ...]:
    """One entry per registered tool: the arguments the tool itself takes.

    The branch families cover what a caller writes *inside* a discriminated
    item, which left a tool's own arguments — ``all_steps``, ``budget``,
    ``expected_sha256`` — described nowhere a ``compact`` session can reach.
    The entry is named after its tool, so asking for the tool by name returns
    its argument table.
    """
    # The registry rather than the package's ``get_tools``: importing a name
    # out of ``ltspice_mcp.tools`` here reads, to the module-closure scan in
    # tests/test_consolidated_contracts.py, as importing a module of that name.
    from ltspice_mcp.tools._base import registry

    _, dispatch = registry.get_tools()
    return tuple(
        _Source(
            tool=name,
            family="argument",
            call="{name}(...)",
            branches=((name, registered.input_model, ""),),
            discriminator=None,
            summaries=(
                (name, first_sentence(registered.definition.description or "", limit=240)),
            ),
        )
        for name, registered in dispatch.items()
        if registered.input_model is not None
    )


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
                default=default_spelling(field),
                # The whole description, not its first sentence: on the compact
                # listing this lookup is the only channel an argument's prose
                # has, so a cut here leaves the rest reaching nobody. The
                # dense per-branch listing the API renders still abbreviates.
                description=" ".join(describe_field(field).split()),
            )
        )
        if depth + 1 >= _MAX_FIELD_DEPTH or len(entries) >= _MAX_NESTED_FIELDS:
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


def _summary_for(tool: str, name: str, model: type[BaseModel] | None, fallback: str = "") -> str:
    """The branch's one line: hand-written, else the model's docstring, else
    whatever its source derived for it."""
    written = _SUMMARIES.get((tool, name))
    if written:
        return written
    if model is not None and model.__doc__:
        return first_sentence(" ".join(model.__doc__.split()), limit=240)
    return fallback


@functools.cache
def build_index() -> tuple[BranchEntry, ...]:
    """Every branch of every tool, plus each tool's own arguments, in
    advertised order."""
    entries: list[BranchEntry] = []
    for source in _sources():
        derived = dict(source.summaries)
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
                    summary=_summary_for(source.tool, name, model, derived.get(name, "")),
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
    #: The normalized spellings an exact-name query has to equal. Held here
    #: with the rest of the searchable text rather than re-normalized on every
    #: entry on every query.
    exact: frozenset[str]


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
                exact=frozenset(
                    {_normalize(entry.name), _normalize(f"{entry.name} {entry.family}")}
                ),
            )
        )
    return tuple(stacks)


def _score(stacks: _Haystacks, phrase: str, tokens: list[str]) -> int:
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
    if phrase in stacks.exact:
        total += _EXACT_NAME_BONUS
    elif phrase and phrase in stacks.name:
        total += _NAME_PHRASE_BONUS
    if phrase and phrase in stacks.synonyms:
        total += _SYNONYM_PHRASE_BONUS
    return total


def search_branches(
    query: str, *, limit: int, tools: frozenset[str] | None = None
) -> tuple[list[BranchEntry], int]:
    """The best ``limit`` branches for ``query``, and how many matched at all.

    Ranking is by where the query's words land — branch name, then the plain
    words a person types for it, then a field name, then prose — and ties break
    on advertised order, so the same query always returns the same list.
    ``tools`` restricts the search to the tools a session serves: the index is
    built from the whole registry, and a tool the operator did not turn on
    must not be findable on a session that would refuse the call.
    """
    phrase = _normalize(query)
    tokens = _tokens(query)
    scored: list[tuple[int, int, BranchEntry]] = []
    for position, (entry, stacks) in enumerate(zip(build_index(), _haystacks(), strict=True)):
        if tools is not None and entry.tool not in tools:
            continue
        score = _score(stacks, phrase, tokens)
        if score:
            scored.append((-score, position, entry))
    scored.sort(key=lambda row: (row[0], row[1]))
    return [entry for _, _, entry in scored[:limit]], len(scored)


def table_of_contents(tools: frozenset[str] | None = None) -> list[dict[str, Any]]:
    """Every entry name and its one line, grouped by tool and family.

    One pass over the index: entries come out in advertised order, so grouping
    preserves it without re-scanning the whole index once per family.
    ``tools`` restricts it to the tools a session serves.
    """
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for entry in build_index():
        if tools is not None and entry.tool not in tools:
            continue
        rows = groups.setdefault((entry.tool, entry.family), [])
        rows.append({"name": entry.name, "summary": entry.summary})
    return [
        {"tool": tool, "family": family, "branches": rows}
        for (tool, family), rows in groups.items()
    ]


_REFERENCE_DESCRIPTION_CHARS = 160


def validation_error_detail(
    tool: str,
    exc: ValidationError | ValueError | TypeError,
    *,
    field_owners: Mapping[str, Sequence[str]] | None = None,
    limit: int = 2,
) -> str:
    """The compact rendering of a validation error, followed by the field table
    of each branch the error names, so a caller corrects the call from the
    error instead of looking the branch up first. A tagged-union error's
    location carries the tag, which is the branch's name in the index."""
    detail = compact_validation_error(exc, field_owners=field_owners)
    if not isinstance(exc, ValidationError):
        return detail
    branches = {
        entry.name: entry
        for entry in build_index()
        if entry.tool == tool and entry.family != "argument"
    }
    named: list[BranchEntry] = []
    for error in exc.errors(include_url=False, include_input=False):
        for part in error["loc"]:
            entry = branches.get(part) if isinstance(part, str) else None
            if entry is not None and entry not in named:
                named.append(entry)
    for entry in named[:limit]:
        detail += (
            f" Reference for {entry.name}: "
            + "; ".join(_field_line(field) for field in entry.fields)
            + "."
        )
    return detail


def _field_line(field: FieldEntry) -> str:
    line = f"{field.name} ({field.type}{', required' if field.required else ''})"
    if field.description:
        line += f": {field.description[:_REFERENCE_DESCRIPTION_CHARS]}"
    return line
