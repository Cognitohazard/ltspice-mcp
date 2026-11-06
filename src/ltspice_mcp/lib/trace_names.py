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
from typing import Literal

from ltspice_mcp.lib.device_names import (
    device_param_name,
    find_trace_by_path,
    parse_shorthand,
    writes_xvii_command,
)
from ltspice_mcp.lib.raw_header import RawPlotHeader, leading_parameters
from ltspice_mcp.lib.result_kinds import header_analysis, step_variables

#: The names a netlist gives ground: node 0, and the aliases simulators accept.
GROUND_ALIASES: frozenset[str] = frozenset({"0", "gnd", "gnd!"})

#: A node-pair voltage ``V(a,b)``. SPICE syntax names it, but no simulator
#: writes it as a trace, so it is read as ``V(a) - V(b)``. A node name holds
#: no comma, parenthesis or space, which keeps the split unambiguous.
NODE_PAIR_RE = re.compile(r"\s*v\(\s*([^\s(),]+)\s*,\s*([^\s(),]+)\s*\)\s*", re.IGNORECASE)

#: A node voltage trace ``V(<node>)``, the node inside.
V_TRACE_RE = re.compile(r"v\((.+)\)", re.IGNORECASE)


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
    ``names``. A ground alias the result carries no voltage for is ground and
    contributes nothing, so ``V(a,0)`` is ``V(a)``."""
    traces = {
        node: find_trace(names, f"V({node})", xvii=xvii) for node in (plus_node, minus_node)
    }
    grounded = {
        node for node, trace in traces.items() if trace is None and node.lower() in GROUND_ALIASES
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
            f"'{read.absent}' is not in this result.{pin_note(plot, read.absent)}{listed}"
        )
    return f"Signal '{signal}' not found.{_missing_note(plot, signal)}{listed}"


def _missing_note(plot: RawPlotHeader, signal: str) -> str:
    """What a name no trace answers to may mean instead, or nothing."""
    from ltspice_mcp.lib.log_parser import device_key  # Its parser is not needed here.

    lowered = signal.lower()
    traces = {name.lower() for name in signal_names(plot)}
    if lowered in ("v(onoise)", "v(inoise)") and traces & {"onoise_spectrum", "inoise_spectrum"}:
        return " (ngspice names noise signals 'onoise_spectrum'/'inoise_spectrum')"
    if lowered in ("onoise_spectrum", "inoise_spectrum") and traces & {"v(onoise)", "v(inoise)"}:
        return " (LTspice names noise signals 'V(onoise)'/'V(inoise)')"
    if pin := pin_note(plot, signal):
        return pin
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


def pin_note(plot: RawPlotHeader, signal: str) -> str:
    """Say, for ``V(<instance>:<pin>)``, that ``pin`` is a pin of a subcircuit
    instance, which a result holds no voltage for; empty for any other name.

    A raw names a node inside a subcircuit ``V(<instance>:<node>)``, but a pin
    is the net it connects to in the circuit above, under that net's name. The
    pin is known as one by the current into it (``Ix(<instance>:<pin>)``), or
    by the ``Backannotation`` line LTspice writes for the instance, which lists
    its pins in the order the instance line lists their nets (the
    ``export-hierarchical-block`` recordings; XVII writes that line for a
    sheet's block but not for a netlist's instance, ``raw/subckt_names_tran``).
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
    if current is None and pin.lower() not in (p.lower() for p in pins):
        return ""
    hint = (
        f" '{pin}' is a pin of subcircuit instance {instance}, and a result holds no "
        "voltage under a pin's name: the pin's voltage is the net it connects to in the "
        "circuit above, under that net's name."
    )
    if pins:
        hint += (
            f" {instance}'s pins are {', '.join(pins)}, in the order its instance line "
            "lists their nets (inspect kind='components' with detail='full' shows them)."
        )
    if current is not None:
        hint += f" {current} is the current into the pin."
    return hint


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
