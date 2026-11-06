"""Which traces of a result a caller's signal name reads.

The one rule for every reader of a signal by name: a result already decoded
(``services.resolve_signal``, which every recipe, plot and the Python API's
``trace`` go through) and the parser worker, which decodes only the traces a
``load_raw(signals=...)`` names and so has only the names a raw's header
declares. A name reads the one trace ``find_trace`` finds for it; else a node
pair ``V(a,b)`` reads its two nodes' voltages, a ground node (``0``, ``gnd``)
the result holds no voltage for reading as zero (``traces_read``).
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence

from ltspice_mcp.lib.device_names import device_param_name, find_trace_by_path
from ltspice_mcp.lib.raw_header import RawPlotHeader, leading_parameters

#: The names a netlist gives ground: node 0, and the aliases simulators accept.
GROUND_ALIASES: frozenset[str] = frozenset({"0", "gnd", "gnd!"})

#: A node-pair voltage ``V(a,b)``. SPICE syntax names it, but no simulator
#: writes it as a trace, so it is read as ``V(a) - V(b)``. A node name holds
#: no comma, parenthesis or space, which keeps the split unambiguous.
NODE_PAIR_RE = re.compile(r"\s*v\(\s*([^\s(),]+)\s*,\s*([^\s(),]+)\s*\)\s*", re.IGNORECASE)


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


def traces_read(names: Sequence[str], signal: str, *, xvii: bool) -> tuple[str, ...] | None:
    """The traces among ``names`` that ``signal`` reads, or None when it reads
    none of them: the one ``find_trace`` finds, else a node pair's two node
    voltages, without a ground node the result holds no voltage for.

    ``services.resolve_signal`` reads a decoded result by the same rule and
    says, besides, which side of a pair each trace is and why a name is not
    there; ``tests/test_chosen_signals.py`` holds the two to one answer. They
    differ in one place, on purpose: on a noise result ``resolve_signal``
    refuses a node pair, whose two traces are densities that do not subtract,
    and this still names those two traces, so a read that keeps them refuses
    the pair when it is read, with the same reason.
    """
    trace = find_trace(names, signal, xvii=xvii)
    if trace is not None:
        return (trace,)
    pair = NODE_PAIR_RE.fullmatch(signal)
    if pair is None:
        return None
    read: list[str] = []
    for node in pair.groups():
        voltage = find_trace(names, f"V({node})", xvii=xvii)
        if voltage is None and node.lower() not in GROUND_ALIASES:
            return None
        if voltage is not None and voltage not in read:
            read.append(voltage)
    return tuple(read)


class SignalsNotFoundError(ValueError):
    """A signal a read was asked to keep names no trace of the result.

    ``parser_code`` is the code the parser worker reports it under, so the
    caller gets this message rather than a failed parse.
    """

    parser_code = "signals_not_found"


def chosen_traces(
    plots: Sequence[RawPlotHeader],
    signals: Sequence[str],
    *,
    xvii: Callable[[RawPlotHeader], bool],
) -> list[list[int]]:
    """For each plot, the positions of the traces a read of ``signals``
    keeps: its first variable (the axis, or the first value of a plot with
    none), the step parameters it opens with (``leading_parameters``), and
    every trace a signal reads (``traces_read``). ``xvii`` says whether a plot
    names subcircuit members as XVII does.

    A signal no plot has a trace for is refused, with the traces there are.
    """
    kept: list[list[int]] = []
    found: set[str] = set()
    for plot in plots:
        names = [variable.name for variable in plot.variables]
        positions = {name: index for index, name in enumerate(names)}
        keep = {0, *leading_parameters(plot)}
        for signal in signals:
            read = traces_read(names, signal, xvii=xvii(plot))
            if read is not None:
                found.add(signal)
                keep.update(positions[name] for name in read)
        kept.append(sorted(keep))
    missing = [signal for signal in signals if signal not in found]
    if missing:
        names = [variable.name for variable in plots[0].variables] if plots else []
        listed = ", ".join(names[:20]) + (
            f", ... ({len(names)} in all)" if len(names) > 20 else ""
        )
        raise SignalsNotFoundError(
            f"{', '.join(repr(signal) for signal in missing)} "
            f"{'names' if len(missing) == 1 else 'name'} no trace of this result, so a read of "
            f"the chosen signals cannot keep {'it' if len(missing) == 1 else 'them'}. "
            f"Its traces: {listed}."
        )
    return kept
