"""Which traces of a result a caller's signal name reads, and why a name reads none.

The one rule for every reader of a signal by name: a result already decoded
(``services.resolve_signal``, which every recipe, plot and the Python API's
``trace`` go through) and the parser worker, which keeps only the traces a
``load_raw(signals=...)`` names and so has only the names a raw's header
declares. Both ask ``read_in_plot`` of a plot's header, and say why a name
reads nothing with ``why_not``.

A name reads the one trace ``find_trace`` finds for it; else a node pair
``V(a,b)`` reads its two nodes' voltages, a ground node (``0``, ``gnd``) the
result holds no voltage for reading as zero. No name reads a variable that
holds each step's stepped value (``signal_names``), and a node pair of a noise
result reads nothing, its traces being densities that do not subtract.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import PureWindowsPath
from typing import Literal

from ltspice_mcp.lib.asc_document import is_asc
from ltspice_mcp.lib.deck_nodes import GROUND_NAMES
from ltspice_mcp.lib.device_names import (
    device_param_name,
    find_trace_by_path,
    parse_device_param,
    parse_shorthand,
    trace_key,
    writes_xvii_command,
)
from ltspice_mcp.lib.raw_header import RawPlotHeader, leading_parameters
from ltspice_mcp.lib.result_kinds import header_analysis, step_variables

#: A node-pair voltage ``V(a,b)``. SPICE syntax names it, but no simulator
#: writes it as a trace, so it is read as ``V(a) - V(b)``. A node name holds
#: no comma, parenthesis or space, which keeps the split unambiguous.
NODE_PAIR_RE = re.compile(r"\s*v\(\s*([^\s(),]+)\s*,\s*([^\s(),]+)\s*\)\s*", re.IGNORECASE)

#: A node voltage trace ``V(<node>)``, the node inside.
V_TRACE_RE = re.compile(r"v\((.+)\)", re.IGNORECASE)

#: The name LTspice's export gives a schematic's net that carries no label:
#: ``N001``, ``N002``, ... (recorded: ``net-naming``).
UNLABELLED_NET_RE = re.compile(r"n\d{3,}", re.IGNORECASE)

#: What a trace of a result holds, as its name and declared type tell.
TraceQuantity = Literal["node_voltage", "current", "device_parameter", "other"]

#: The declared types that say a trace holds a node voltage or a current. A
#: trace declared with any other type holds another quantity, whatever its
#: name: a declared impedance does not become a voltage by its spelling.
_QUANTITY_BY_TYPE: dict[str, TraceQuantity] = {
    "voltage": "node_voltage",
    "current": "current",
    "device_current": "current",
}

#: A current named with no declared type: ``I(R1)``, or a terminal current
#: of a device of more terminals, ``Ic(Q1)`` or ``Id(M1)``.
_CURRENT_NAME = re.compile(r"i[a-z]?\(", re.IGNORECASE)


def trace_quantity(name: str, declared: str | None, *, axis: bool = False) -> TraceQuantity:
    """What the trace ``name``, declared by the simulator as ``declared``
    (None where nothing declares it), holds; ``axis`` when it is the sampled
    axis of its plot.

    A plot's axis is the variable it is swept along, not a quantity of the
    circuit's solution, though a ``.dc V1`` sweep's is typed voltage: LTspice
    names it ``V1`` (recorded: ``raw/dc``) and ngspice ``v(v-sweep)``
    (``dc_current_only`` in ``tests/fixtures/ngspice_runs``). A device's
    parameter (``@m1[gm]``) is one whatever its type and its wrapping: ngspice
    writes a saved ``@m1[vth]`` as ``v(@m1[vth])``, typed voltage, and
    ``@m1[id]`` as ``i(@m1[id])``, typed current (``op_device_params`` there).
    Otherwise the declared type decides, and only a trace with none is read
    by its name: ``V(...)`` a node voltage, ``I(...)`` or ``Ic(...)`` a
    current.

    The one rule for whether a trace holds a node voltage: the operating
    point files its values by it, a recovered bias point is vouched for only
    by the traces it calls node voltages, the value scan compares only those
    with the deck's sources, and an AC summary defaults to the first.

    It reads no plot's analysis, so the readers ask it of a plot's traces
    through ``raw_parser.node_voltages``, which lists none for a noise plot:
    LTspice types its traces as voltages (``v(onoise)``, a source's
    contribution ``v(R1)``; recorded: ``raw/noise``), and they are spectral
    densities.
    """
    if axis:
        return "other"
    if parse_device_param(name) is not None:
        return "device_parameter"
    if declared:
        return _QUANTITY_BY_TYPE.get(declared.strip().lower(), "other")
    if name[:2].lower() == "v(":
        return "node_voltage"
    return "current" if _CURRENT_NAME.match(name) else "other"


def is_node_voltage(name: str, declared: str | None, *, axis: bool = False) -> bool:
    """Whether the trace ``name``, declared as ``declared``, holds a node
    voltage (:func:`trace_quantity`)."""
    return trace_quantity(name, declared, axis=axis) == "node_voltage"


def find_trace(names: Sequence[str], signal: str, *, xvii: bool) -> str | None:
    """The name among ``names`` that ``signal`` addresses, or None.

    Lookup is case-insensitive: SPICE node names are case-insensitive per
    SPICE conventions, but spicelib preserves the case the simulator wrote.
    LTspice writes ``V(out)`` for transient/AC/DC sweep raws but ``v(onoise)``
    for ``.NOISE`` raws — case-sensitive match would reject the user's
    ``V(onoise)`` even though the data exists. ``xvii`` is for names a build
    wrote that names subcircuit members as XVII does. The one rule behind
    every reader that finds a trace or an operating-point value by name.
    """
    if signal in names:
        return signal
    sig_lower = signal.lower()
    by_lower = {name.lower(): name for name in names}
    if sig_lower in by_lower:
        return by_lower[sig_lower]

    # Resolve cross-simulator / shorthand aliases transparently instead of
    # forcing a guaranteed retry on a deterministic rename:
    #   - noise: LTspice V(onoise)/V(inoise) <-> ngspice onoise_spectrum/
    #     inoise_spectrum, plus bare onoise/inoise shorthand
    #   - hierarchical separator: LTspice ':' (V(X1:mid)) <-> ngspice '.'
    #   - a path through subcircuits, which XVII writes without each
    #     instance's X (V(xa.xin.mid) is its V(a:in:mid))
    candidates: list[str] = []
    for kind in ("onoise", "inoise"):
        if sig_lower in (kind, f"v({kind})", f"{kind}_spectrum"):
            candidates += [f"v({kind})", f"{kind}_spectrum", kind]
    if ":" in sig_lower:
        candidates.append(sig_lower.replace(":", "."))
    if "." in sig_lower:
        candidates.append(sig_lower.replace(".", ":"))

    for cand in candidates:
        if cand in by_lower:
            return by_lower[cand]
    return find_trace_by_path(names, signal, xvii=xvii) or device_param_name(names, signal)


Refusal = Literal["missing", "expression", "step_value", "ground_twice", "noise_pair"]


@dataclass(frozen=True)
class NameRead:
    """What a signal name reads of a plot's traces, or why it reads none.

    ``name`` is what a reply reports: the trace as the simulator wrote it, or,
    for a node pair, ``V(a,b)`` with each node spelled as the raw spells it.
    The value is ``plus - minus``; a side is None where it is ground. When
    ``refused`` is set the name reads nothing, and says why: a ``"missing"``
    trace (``absent`` names it: the signal, or a pair's node voltage), an
    ``"expression"``, a ``"step_value"``, a pair naming ground twice
    (``"ground_twice"``), or a ``"noise_pair"``.
    """

    name: str
    plus: str | None = None
    minus: str | None = None
    refused: Refusal | None = None
    absent: str | None = None

    @property
    def traces(self) -> tuple[str, ...]:
        """The traces the name reads, each once."""
        return tuple(dict.fromkeys(t for t in (self.plus, self.minus) if t is not None))


def read_in_plot(plot: RawPlotHeader, signal: str) -> NameRead:
    """What ``signal`` reads of ``plot``'s traces (``read_name``), from what
    the plot's header says: its signals, its step variables, whether it is a
    noise result and whether its writer names subcircuit members as XVII does."""
    return read_name(
        signal_names(plot),
        signal,
        xvii=writes_xvii_command(dict(plot.fields)),
        step_values=step_variables(plot),
        noise=header_analysis(plot) == "noise",
    )


def read_name(
    names: Sequence[str],
    signal: str,
    *,
    xvii: bool,
    step_values: Sequence[str] = (),
    noise: bool = False,
) -> NameRead:
    """What ``signal`` reads among the traces ``names``.

    A trace ``find_trace`` finds wins, one literally named ``V(a,b)``
    included; otherwise a node pair reads as ``V(a) - V(b)``, each side found
    like any single trace. ``step_values`` are the names of the variables that
    hold each step's stepped value, and ``noise`` says the traces are noise
    densities; ``xvii`` is ``find_trace``'s.
    """
    trace = find_trace(names, signal, xvii=xvii)
    if trace is not None:
        return NameRead(trace, trace)
    pair = NODE_PAIR_RE.fullmatch(signal)
    if pair is not None:
        return _node_pair(names, signal, pair.group(1), pair.group(2), xvii=xvii, noise=noise)
    if looks_like_expression(signal):
        return NameRead(signal, refused="expression")
    if signal.lower() in {name.lower() for name in step_values}:
        return NameRead(signal, refused="step_value")
    return NameRead(signal, refused="missing", absent=signal)


def _node_pair(
    names: Sequence[str], signal: str, plus_node: str, minus_node: str, *, xvii: bool, noise: bool
) -> NameRead:
    """``V(plus_node, minus_node)`` read from the two node voltages among
    ``names``. A name a deck reads as ground (``deck_nodes``) that the result
    carries no voltage for contributes nothing, so ``V(a,0)`` is ``V(a)``."""
    traces = {node: find_trace(names, f"V({node})", xvii=xvii) for node in (plus_node, minus_node)}
    grounded = {
        node for node, trace in traces.items() if trace is None and node.casefold() in GROUND_NAMES
    }
    if minus_node not in grounded and noise:
        return NameRead(signal, refused="noise_pair")
    for node, trace in traces.items():
        if trace is None and node not in grounded:
            return NameRead(signal, refused="missing", absent=f"V({node})")
    plus, minus = traces[plus_node], traces[minus_node]
    if minus is None:
        if plus is None:
            return NameRead(signal, refused="ground_twice")
        return NameRead(plus, plus)
    name = f"V({_node_spelling(plus, plus_node)},{_node_spelling(minus, minus_node)})"
    return NameRead(name, plus, minus)


def _node_spelling(trace: str | None, node: str) -> str:
    """``node`` as the raw spells it inside its ``V(...)`` trace, else as asked."""
    match = V_TRACE_RE.fullmatch(trace or "")
    return match.group(1) if match else node


def looks_like_expression(signal: str) -> bool:
    """Whether ``signal`` combines traces rather than naming one.

    True for an operator outside every parenthesis (``V(a)-V(b)``,
    ``2*V(out)``) or a reference nested in another (``abs(V(out))``). A sign
    inside the parentheses is part of a node name — ``V(in-)`` is one trace.
    """
    depth = 0
    for char in signal:
        if char == "(":
            depth += 1
            if depth > 1:
                return True
        elif char == ")":
            depth -= 1
        elif depth == 0 and char in "+-*/^":
            return True
    return False


def why_not(read: NameRead, signal: str, plot: RawPlotHeader, *, listed: str = "") -> str:
    """Why ``signal`` reads no trace of ``plot`` (``read.refused``) and what to
    read instead: the sentences every reader's refusal gives. ``listed`` is
    the reader's own sentence naming the traces there are, placed where it
    helps; a pair of grounds and a noise pair are refused without it.
    """
    if read.refused == "expression":
        return (
            f"Signal '{signal}' not found: it reads as an expression, and a signal names "
            f"one trace or a node-pair voltage V(a,b).{listed} For other trace math, "
            "combine the traces in Python: r = api.load_raw(job_id=..., case_id=...); "
            "t = r.axis(step=0); y = r.trace('V(out)', step=0) * r.trace('I(R1)', step=0); "
            "compute_signal_stats(t, y) gives its time-weighted statistics."
        )
    if read.refused == "step_value":
        return f"Signal '{signal}' not found. {step_value_note(signal)}{listed}"
    if read.refused == "ground_twice":
        return f"Signal '{signal}' names ground twice; it is zero by definition."
    if read.refused == "noise_pair":
        return (
            f"Signal '{signal}' is a node-pair difference, and a .noise result holds "
            "spectral densities: the difference of two densities is not the noise "
            "between the nodes. Put the pair in the .noise directive, "
            "e.g. .noise V(a,b) <source> ..., and read V(onoise)."
        )
    if read.absent is not None and read.absent != signal:
        return (
            f"Signal '{signal}' reads as a node-pair difference, but node voltage "
            f"'{read.absent}' is not in this result.{naming_note(plot, read.absent)}{listed}"
        )
    return f"Signal '{signal}' not found.{_missing_note(plot, signal)}{listed}"


def _missing_note(plot: RawPlotHeader, signal: str) -> str:
    """What a name no trace answers to may mean instead, or nothing: a naming
    rule (``naming_note``), else where a device's parameters are read."""
    from ltspice_mcp.lib.log_parser import device_key  # Its parser is not needed here.

    if note := naming_note(plot, signal):
        return note
    lowered = signal.lower()
    device = parse_shorthand(lowered)
    if device or "@" in lowered:
        save_target = device_key(*device) if device else signal
        return (
            f" Per-device small-signal params (gm/gds/vth/...) come back from "
            f"operating_point(device=...). As a raw trace (for a sweep/plot), ngspice "
            f"needs '.save {save_target}' in the deck; LTspice writes them only to the "
            f".log under '.options logopinfo' (operating_point reads that), not the raw."
        )
    return ""


def naming_note(plot: RawPlotHeader, signal: str) -> str:
    """Why ``signal`` reads no trace of ``plot`` when a simulator names what it
    means otherwise, and the name to read; empty when no naming rule explains it.

    The one answer every reader of a name gives, a value read off an
    operating point's bias point included: the noise traces' names on each
    simulator, a pin of a subcircuit instance (``_pin_note``), an instance's
    X (``_instance_x_note``), and a schematic's unlabelled nets
    (``_unlabelled_net_note``). A device's parameter (``V(@m1[vth])``) names
    no node or instance, so only the noise names can be its answer here.
    """
    lowered = signal.lower()
    traces = {name.lower() for name in signal_names(plot)}
    if lowered in ("v(onoise)", "v(inoise)") and traces & {"onoise_spectrum", "inoise_spectrum"}:
        return " (ngspice names noise signals 'onoise_spectrum'/'inoise_spectrum')"
    if lowered in ("onoise_spectrum", "inoise_spectrum") and traces & {"v(onoise)", "v(inoise)"}:
        return " (LTspice names noise signals 'V(onoise)'/'V(inoise)')"
    if parse_device_param(signal) is not None:
        return ""
    return (
        _pin_note(plot, signal)
        or _instance_x_note(plot, signal)
        or _unlabelled_net_note(plot, signal)
    )


@dataclass(frozen=True)
class _Respelled:
    """The one trace a name reads once an instance on its path gains or loses
    a leading X (``_respelled``)."""

    #: The trace, as the result spells it.
    trace: str
    #: The instances as asked and as the result names them: ``'xx1' as 'x1'``.
    named: str
    #: Why the result names them so, each sentence led by a space; empty where
    #: no recorded rule explains it.
    why: str


def _respelled(plot: RawPlotHeader, signal: str) -> _Respelled | None:
    """The one trace of ``plot`` that ``signal`` names once an instance on its
    path gains or loses a leading X, and why the result spells it so; None
    when no trace does, or more than one.

    A part placed on a schematic is named in a raw by its reference, without
    the X its netlist line puts in front: block X1, exported as ``X§X1`` by
    LTspice 26 and ``XX1`` by XVII, is ``Ix(x1:in)`` (XVII: ``Ix(x1:IN)``),
    and resistor Load, exported as ``R§Load``, is ``I(Load)`` (recorded:
    ``export/hier_plain_top_run``, ``export/load_name_op``). LTspice 26 names
    an instance a deck calls with X by that whole name (``Id(xa:Mdrv)``), and
    XVII every instance without its X (``Id(a:Mdrv)``; recorded:
    ``raw/subckt_names_tran_saved``). ``find_trace`` already reads a deck's
    spelling of an instance on XVII, so a name gets here there only when it
    mixes the two spellings. Only an instance is matched this way: the last
    name on a path is a node or a device, which no X marks.

    The reason given is the rule of the result's own source
    (``from_schematic``): a schematic's rule for a sheet's run on LTspice, a
    deck's for a deck's run. An X the caller added to a deck's instance has no
    rule behind it, and gets none.
    """
    asked_function, asked_path = trace_key(signal, xvii=False)
    if not asked_function or len(asked_path) < 2:
        return None
    found: list[tuple[str, list[tuple[str, str]]]] = []
    for name in signal_names(plot):
        function, path = trace_key(name, xvii=False)
        if function != asked_function or len(path) != len(asked_path):
            continue
        if path[-1] != asked_path[-1]:
            continue
        differ = [(a, b) for a, b in zip(asked_path[:-1], path[:-1], strict=True) if a != b]
        if differ and all(a == f"x{b}" or b == f"x{a}" for a, b in differ):
            found.append((name, differ))
    if len(found) != 1:
        return None
    ((trace, differ),) = found
    named = " and ".join(f"'{asked}' as '{spelled}'" for asked, spelled in differ)
    added = any(asked == f"x{spelled}" for asked, spelled in differ)
    kept = any(spelled == f"x{asked}" for asked, spelled in differ)
    why: list[str] = []
    if writes_xvii_command(dict(plot.fields)):
        why.append(
            "The LTspice build that wrote this result names each subcircuit instance "
            "without the X that starts its name in the netlist."
        )
    elif plot.dialect == "ltspice" and from_schematic(plot):
        if added:
            why.append(
                "LTspice names a part placed on a schematic by its reference, without the X "
                "its netlist line adds in front."
            )
        if kept:
            why.append(
                "LTspice keeps the X of a schematic part whose reference starts with one, "
                "and of an instance a deck line calls with X."
            )
    elif kept:
        why.append("An instance a deck calls with X keeps that X in the names a result gives it.")
    return _Respelled(trace, named, "".join(f" {sentence}" for sentence in why))


def _instance_x_note(plot: RawPlotHeader, signal: str) -> str:
    """Name the one trace ``signal`` reads once an instance on its path gains
    or loses a leading X, and why (``_respelled``); empty when there is none."""
    found = _respelled(plot, signal)
    if found is None:
        return ""
    return f" {found.trace} is in this result, which names {found.named}.{found.why}"


def _unlabelled_net_note(plot: RawPlotHeader, signal: str) -> str:
    """For a node ``V(<node>)`` at the top of a schematic's circuit
    (``from_schematic``), name the nets of ``plot`` LTspice named itself
    because the sheet put no label on them, and say a label names one; empty
    for any other name, or a result that holds no such net.

    LTspice's export names a sheet's net that carries no label ``N001``,
    ``N002``, ... (recorded: ``net-naming``), and the run's raw holds it
    under that name, ``V(n001)``, on both builds (``export/hier_plain_top_run``).
    """
    match = V_TRACE_RE.fullmatch(signal.strip())
    if match is None:
        return ""
    node = match.group(1).strip()
    if re.search(r"[\s:.,]", node) or node.casefold() in GROUND_NAMES:
        return ""
    if not from_schematic(plot):
        return ""
    unlabelled = [
        name
        for name in signal_names(plot)
        if (voltage := V_TRACE_RE.fullmatch(name))
        and UNLABELLED_NET_RE.fullmatch(voltage.group(1))
    ]
    if not unlabelled:
        return ""
    listed = ", ".join(unlabelled[:10])
    if len(unlabelled) > 10:
        listed += f" and {len(unlabelled) - 10} more"
    note = (
        " This result is a schematic's, and LTspice names a net that has no label "
        f"N001, N002, ...: here {listed}."
    )
    if not UNLABELLED_NET_RE.fullmatch(node):
        note += f" Put a net label {node} on the net and run it again to read V({node})."
    return note


def from_schematic(plot: RawPlotHeader) -> bool:
    """Whether ``plot`` is of a run of a schematic, as its title says.

    LTspice XVII titles a raw with its deck's first line, which a sheet's
    export writes as ``* <sheet>.asc``. LTspice 26 titles it with the path of
    the deck it ran, which for a sheet is the ``.net`` it exported; the server
    runs a sheet's export under that suffix too (``deck_prep``). A deck's run
    is titled by the deck's own first line or path (recorded on both builds:
    ``export/hier_plain_top_run`` and ``export/load_name_op`` against every
    deck's run). LTspice 26's title says no more than the suffix, so a deck its
    author named ``.net`` reads as a schematic's. Either title names a Windows
    path, read as one on any platform.
    """
    title = next((value for key, value in plot.fields if key.casefold() == "title"), "")
    path = PureWindowsPath(title.strip())
    return is_asc(path) or path.suffix.casefold() == ".net"


def _pin_note(plot: RawPlotHeader, signal: str) -> str:
    """Say, for ``V(<instance>:<pin>)``, that ``pin`` is a pin of a subcircuit
    instance, which a result holds no voltage for; empty for any other name.

    A raw names a node inside a subcircuit ``V(<instance>:<node>)``, but a pin
    is the net it connects to in the circuit above, under that net's name. The
    pin is known as one by the current into it (``Ix(<instance>:<pin>)``), or
    by the ``Backannotation`` line LTspice writes for the instance, which lists
    its pins in the order the instance line lists their nets (the
    ``export-hierarchical-block`` recordings; XVII writes that line for a
    sheet's block but not for a netlist's instance, ``raw/subckt_names_tran``).
    An instance asked for with an X more or less than the result names it
    (``_respelled``) is known by its current under the result's name, and
    the note says how the result names it.
    """
    match = V_TRACE_RE.fullmatch(signal.strip())
    if match is None or ":" not in match.group(1):
        return ""
    instance, _, pin = match.group(1).strip().rpartition(":")
    pins = _instance_pins(plot, instance)
    # Found as any trace is, so XVII's spelling (Ix(a:TOP) for xa's TOP) counts.
    current = find_trace(
        signal_names(plot),
        f"Ix({instance}:{pin})",
        xvii=writes_xvii_command(dict(plot.fields)),
    )
    respelled = None
    if current is None and not pins:
        # The instance as the result names it once its X is read otherwise
        # (_respelled), known as one by the current into the pin.
        respelled = _respelled(plot, f"Ix({instance}:{pin})")
        if respelled is not None:
            current = respelled.trace
            instance = _current_instance(current)
            pins = _instance_pins(plot, instance)
    elif current is not None and not pins:
        # XVII reads a deck's spelling (xx1 for its x1); its Backannotation
        # line names the instance as the result does.
        pins = _instance_pins(plot, _current_instance(current))
    if current is None and pin.lower() not in (p.lower() for p in pins):
        return ""
    hint = (
        f" '{pin}' is a pin of subcircuit instance {instance}, and a result holds no "
        "voltage under a pin's name: the pin's voltage is the net it connects to in the "
        "circuit above, under that net's name."
    )
    if respelled is not None:
        hint += f" This result names {respelled.named}.{respelled.why}"
    if pins:
        hint += (
            f" {instance}'s pins are {', '.join(pins)}, in the order its instance line "
            "lists their nets (inspect kind='components' with detail='full' shows them)."
        )
    if current is not None:
        hint += f" {current} is the current into the pin."
    return hint


def _current_instance(current: str) -> str:
    """The instance a port current ``Ix(<instance>:<pin>)`` names, as spelled."""
    return current[current.index("(") + 1 : -1].rpartition(":")[0]


def _instance_pins(plot: RawPlotHeader, instance: str) -> list[str]:
    """The pins of ``instance`` as the plot's ``Backannotation`` line for it
    lists them, or none when the raw has no such line."""
    for key, value in plot.fields:
        if key.casefold() != "backannotation":
            continue
        name, *pins = value.split()
        if name.lower() == instance.lower():
            return pins
    return []


def signal_names(plot: RawPlotHeader) -> list[str]:
    """The names of ``plot``'s variables a signal can read: all but those that
    hold each step's stepped value (``result_kinds.step_variables``), which a
    decoded result reads as its steps and leaves out of its traces."""
    stepped = step_variables(plot)
    return [variable.name for variable in plot.variables if variable.name not in stepped]


def step_value_note(signal: str) -> str:
    """Why ``signal``, the name of a variable that holds each step's stepped
    value, reads no trace, and where its values are read instead."""
    return (
        f"'{signal}' holds each step's stepped value, not a signal: read it from "
        "the steps (each row's step_values, or RawResult.steps)."
    )


class SignalsNotFoundError(ValueError):
    """A signal a read was asked to keep reads no trace of the result.

    ``parser_code`` is the code the parser worker reports it under, so the
    caller gets this message rather than a failed parse.
    """

    parser_code = "signals_not_found"


def chosen_traces(
    plots: Sequence[RawPlotHeader], signals: Sequence[str], *, plot_index: int
) -> list[list[int]]:
    """For each plot, the positions of the traces a read of ``signals``
    keeps: its first variable (the axis, where it has one), the step
    parameters it opens with (``leading_parameters``), and every trace a
    signal reads there (``read_in_plot``).

    A signal that reads no trace of plot ``plot_index``, the plot the caller
    reads, is refused, with why (``why_not``) and that plot's traces; a
    ``plot_index`` past the file's plots is left to the reader of a plot to
    refuse.
    """
    kept: list[list[int]] = []
    refused: dict[str, str] = {}
    for plot in plots:
        positions = {variable.name: variable.index for variable in plot.variables}
        leading = leading_parameters(plot)
        keep = {0, *leading}
        for signal in signals:
            read = read_in_plot(plot, signal)
            keep.update(positions[name] for name in read.traces)
            if plot.index == plot_index and read.refused is not None:
                refused.setdefault(signal, why_not(read, signal, plot))
        if keep <= set(leading):
            # A plot holds a value besides its step parameters; one no signal
            # reads keeps its first.
            keep.add(len(leading))
        kept.append(sorted(keep))
    if refused and plot_index < len(plots):
        names = signal_names(plots[plot_index])
        listed = ", ".join(names[:20]) + (
            f", ... ({len(names)} in all)" if len(names) > 20 else ""
        )
        raise SignalsNotFoundError(f"{' '.join(refused.values())} This plot's traces: {listed}.")
    return kept
