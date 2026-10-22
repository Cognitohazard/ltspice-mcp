"""Which device of a deck an operating-point name belongs to.

A device inside a subcircuit has a different name in each place a result
reports it, and the two LTspice builds name it differently again. Recorded on
both builds (behaviour ``subckt-device-names``) for ``Mtail`` inside ``Xin``
inside ``Xa``:

=============================  ======================  ====================
where                          LTspice 26              LTspice XVII
=============================  ======================  ====================
the log's device table         ``xa:xin:Mtail``        ``m:a:in:tail``
a device current in the raw    ``Id(xa:xin:Mtail)``    ``Id(a:in:Mtail)``
a port current in the raw      ``Ix(xa:xin:n)``        ``Ix(a:in:N)``
=============================  ======================  ====================

LTspice 26 writes the path of instance names and the device's own name. XVII
drops each instance's ``X``, and in the table also the device's letter, which
it writes first instead. Every subcircuit instance's name starts with ``X``,
so both read back to one path, here ``("xa", "xin", "mtail")``. ngspice names
the same device ``m.xa.xin.mtail`` (its class letter, then the path).

A raw holds the currents of devices inside subcircuits only when the build's
"save subcircuit device currents" setting is on; it is off by default on both
builds, and the raw then holds only the instances' port currents.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ltspice_mcp.lib.simulator_build import is_cp1252_ltspice_build

if TYPE_CHECKING:
    from ltspice_mcp.lib.decoded_raw import RawData

DevicePath = tuple[str, ...]

#: A ``@device[param]`` name, bare or as ngspice wraps it (``v(@m1[vth])``).
_DEVICE_PARAM = re.compile(r"@([^\[\]()]+)\[([^\]]*)\]")
_CURRENT = re.compile(r"^(i[a-z]?)\(([^)]+)\)$", re.IGNORECASE)
#: A trace of one node or one device: ``V(xa:xin:mid)``, ``Id(a:in:Mtail)``.
_TRACE = re.compile(r"^([a-z]+)\(([^(),]+)\)$", re.IGNORECASE)


def writes_xvii_names(raw: RawData) -> bool:
    """Whether the build that wrote ``raw`` names devices as XVII does:
    LTspice XVII, and LTspice IV before it, by the header's ``Command``."""
    command = raw.raw_params.get("Command")
    return isinstance(command, str) and is_cp1252_ltspice_build(command)


def split_path(text: str) -> DevicePath:
    """A device path as a caller writes it, by dots or colons: ``xa.xin.mtail``."""
    return tuple(segment for segment in re.split(r"[.:]", text.casefold()) if segment)


def _instances(segments: Iterable[str], *, xvii: bool) -> DevicePath:
    """Instance names as the deck writes them: XVII drops each one's ``X``."""
    return tuple(f"x{segment}" for segment in segments) if xvii else tuple(segments)


def table_device_path(name: str) -> DevicePath:
    """The device a name in the log's device table (or an ngspice ``@`` trace)
    belongs to: ``xa:xin:mtail`` and ``m:a:in:tail`` are both
    ``("xa", "xin", "mtail")``, ngspice's ``m.xa.xin.mtail`` too.

    Each name says which build wrote it: LTspice 26 starts with the outermost
    instance, whose name starts with X, and XVII with the device's letter,
    as its table lists no instance."""
    low = name.casefold()
    if ":" not in low:
        segments = [segment for segment in low.split(".") if segment]
        # ngspice writes the device's class letter first: m.x1.mn.
        if len(segments) > 1 and len(segments[0]) == 1:
            segments = segments[1:]
        return tuple(segments)
    first, *rest = low.split(":")
    if first.startswith("x"):
        return (first, *rest)
    # XVII writes the device's letter first and drops it from its name.
    *instances, last = rest
    return (*_instances(instances, xvii=True), first + last)


@dataclass(frozen=True)
class _Owner:
    """What an operating-point value's name belongs to."""

    #: The owner as the name spells it: ``xa:xin:Mtail``, ``m.x1.mn``.
    text: str
    path: DevicePath
    #: A port current (``Ix(xa:out)``), which belongs to the instance.
    port: bool = False
    #: The parameter of a ``@device[param]`` name.
    param: str | None = None


def _param_owner(name: str) -> _Owner | None:
    """The device a ``@device[param]`` name holds a parameter of, as the log's
    table or ngspice spells it; None for any other name."""
    table = _DEVICE_PARAM.search(name)
    if table is None:
        return None
    device = table.group(1)
    return _Owner(device, table_device_path(device), param=table.group(2))


def _owner(name: str, *, xvii: bool) -> _Owner | None:
    if (owner := _param_owner(name)) is not None:
        return owner
    current = _CURRENT.match(name)
    if current is None:
        return None
    text = current.group(2)
    path = _raw_path(text, xvii=xvii)
    if current.group(1).casefold() == "ix" and len(path) > 1:
        # The port's name follows the instance's path.
        return _Owner(text, path[:-1], port=True)
    return _Owner(text, path)


def _raw_path(text: str, *, xvii: bool) -> DevicePath:
    """The path a raw's name for a node or device spells: LTspice's ``:`` or
    ngspice's ``.`` between instances, each instance without its ``X`` on
    XVII."""
    if ":" not in text:
        return split_path(text)
    *instances, last = text.casefold().split(":")
    return (*_instances(instances, xvii=xvii), last)


def raw_name(scope: Iterable[str], name: str, *, xvii: bool) -> str:
    """How an LTspice raw, and so ``.save``, names a node or device inside
    subcircuit instances, the reverse of :func:`trace_key`'s reading: the
    instances as written on LTspice 26 (``xa:xin:mid``), each without its X on
    XVII (``a:in:mid``). Each build's ``.save`` takes only its own spelling
    (recorded: ``raw/save_subckt_names*``)."""
    instances = (segment.lower()[1:] if xvii else segment.lower() for segment in scope)
    return ":".join((*instances, name.lower()))


TraceKey = tuple[str, DevicePath]


def trace_key(name: str, *, xvii: bool) -> TraceKey:
    """One key for a trace however a build or simulator spells it: its
    function and the path to its node or device. LTspice 26's
    ``V(xa:xin:mid)``, XVII's ``V(a:in:mid)`` (``xvii``), ngspice's
    ``v(xa.xin.mid)`` and a caller's ``V(xa.xin.mid)`` are all
    ``("v", ("xa", "xin", "mid"))``. A name of another form is its own key."""
    match = _TRACE.match(name.strip())
    if match is None:
        return "", (name.strip().casefold(),)
    return match.group(1).casefold(), _raw_path(match.group(2), xvii=xvii)


def owner_path(name: str, *, xvii: bool) -> DevicePath | None:
    """The device or instance an operating-point value's name belongs to: a
    device for ``@device[param]`` or ``Id(xa:xin:Mtail)``, the instance for a
    port current ``Ix(xa:out)``. None for a node voltage."""
    owner = _owner(name, xvii=xvii)
    return None if owner is None else owner.path


def current_device(name: str) -> str | None:
    """The device's own name a device current belongs to: ``mtail`` for
    ``Id(xa:xin:Mtail)`` on either build. None for a port current or any other
    name."""
    owner = _owner(name, xvii=False)
    if owner is None or owner.port or owner.param is not None:
        return None
    return owner.path[-1]


def _within(path: DevicePath, want: DevicePath) -> bool:
    return bool(want) and any(
        path[i : i + len(want)] == want for i in range(len(path) - len(want) + 1)
    )


def device_query(wanted: str, *, xvii: bool) -> Callable[[str], bool]:
    """A test of whether an operating-point value's name belongs to what
    ``wanted`` names: the device as the simulator spells it (``m.xa.xleaf.m0``,
    which ``inspect``'s hierarchy addresses use), its path or part of it
    (``mtail``, ``xin.mtail``, ``xa.xin.mtail``; dots or colons), or an
    instance it is inside."""
    literal = wanted.strip().casefold()
    want = split_path(wanted)

    def names(name: str) -> bool:
        owner = _owner(name, xvii=xvii)
        return owner is not None and (
            owner.text.casefold() == literal or _within(owner.path, want)
        )

    return names


def find_device_param(names: Iterable[str], device: str, param: str) -> str | None:
    """The one name among ``names`` that holds ``param`` of ``device``: the
    device as the simulator spells it, or the end of its path (``x1.mn`` for
    ``MN`` inside ``X1``, wherever ``X1`` is), whichever build or simulator
    wrote it. None when there is not exactly one."""
    literal = device.casefold()
    want = split_path(device)
    found = [
        name
        for name in names
        if (owner := _param_owner(name)) is not None
        and owner.param is not None
        and owner.param.casefold() == param.casefold()
        and (owner.text.casefold() == literal or (want and owner.path[-len(want) :] == want))
    ]
    return found[0] if len(found) == 1 else None


def find_trace_by_path(names: Iterable[str], signal: str, *, xvii: bool) -> str | None:
    """The one trace among ``names`` that ``signal`` names by a path through
    subcircuit instances, as either build writes it (``trace_key``):
    ``V(xa.xin.mid)`` is LTspice 26's ``V(xa:xin:mid)`` and XVII's
    ``V(a:in:mid)``, ``Id(xa.xin.mtail)`` is ``Id(a:in:Mtail)`` on XVII, and
    ``Ix(xa.top)`` is ``Ix(a:TOP)``. None when ``signal`` names no such path or
    not exactly one trace matches."""
    if _TRACE.match(signal.strip()) is None:
        return None
    want = trace_key(signal, xvii=False)
    if len(want[1]) < 2:
        return None
    # Every match ends with the path's last segment: a cheap test first.
    tail = f"{want[1][-1]})"
    found = [
        name
        for name in names
        if name.casefold().endswith(tail) and trace_key(name, xvii=xvii) == want
    ]
    return found[0] if len(found) == 1 else None


def path_text(path: DevicePath) -> str:
    """A device path as a caller writes it: ``xa.xin.mtail``."""
    return ".".join(path)


def device_names(
    table_names: Iterable[str], raw_traces: Iterable[str], *, xvii: bool
) -> dict[str, dict[str, str | None]]:
    """Each device's name in the log's device table and in the raw's currents.

    ``table_names`` are the table's device names (``xa:xin:mtail``),
    ``raw_traces`` every trace name of the raw. A device the table lists with
    no current in the raw has ``raw`` None, as a device inside a subcircuit
    with currents and no table entry has ``op_table`` None. Port currents name
    an instance, not a device, and are left out, as is a top-level part with
    no table entry, whose name is the same everywhere.
    """
    names: dict[str, dict[str, str | None]] = {}
    for name in table_names:
        entry = names.setdefault(
            path_text(table_device_path(name)), {"op_table": None, "raw": None}
        )
        entry["op_table"] = name
    for trace in raw_traces:
        owner = _owner(trace, xvii=xvii)
        if owner is None or owner.port or owner.param is not None:
            continue
        key = path_text(owner.path)
        if key in names or len(owner.path) > 1:
            names.setdefault(key, {"op_table": None, "raw": None})["raw"] = owner.text
    return dict(sorted(names.items()))
