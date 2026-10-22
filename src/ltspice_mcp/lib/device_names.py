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
from collections.abc import Iterable, Sequence

from ltspice_mcp.lib.simulator_build import is_cp1252_ltspice_build

DevicePath = tuple[str, ...]

#: A ``@device[param]`` name, bare or as ngspice wraps it (``v(@m1[vth])``).
_DEVICE_PARAM = re.compile(r"@([^\[\]()]+)\[")
_CURRENT = re.compile(r"^(i[a-z]?)\(([^)]+)\)$", re.IGNORECASE)


def writes_xvii_names(raw: object) -> bool:
    """Whether the build that wrote ``raw`` names devices as XVII does:
    LTspice XVII, and LTspice IV before it, by the header's ``Command``."""
    fields = getattr(getattr(raw, "header", None), "fields", ())
    command = next((value for key, value in fields if key.casefold() == "command"), None)
    return command is not None and is_cp1252_ltspice_build(command)


def _path(segments: Sequence[str], *, xvii: bool, letter: str = "") -> DevicePath:
    """The path XVII's segments stand for: each instance's ``X`` put back, and
    the device's ``letter`` when the table dropped it."""
    *instances, name = segments
    if not xvii:
        return (*instances, name)
    return (*(f"x{segment}" for segment in instances), letter + name)


def table_device_path(name: str, *, xvii: bool) -> DevicePath:
    """The device a name in the log's device table (or an ngspice ``@`` trace)
    belongs to: ``xa:xin:mtail`` and ``m:a:in:tail`` are both
    ``("xa", "xin", "mtail")``, ngspice's ``m.xa.xin.mtail`` too."""
    low = name.casefold()
    if ":" not in low:
        segments = [segment for segment in low.split(".") if segment]
        # ngspice writes the device's class letter first: m.x1.mn.
        if len(segments) > 1 and len(segments[0]) == 1:
            segments = segments[1:]
        return tuple(segments)
    segments = low.split(":")
    if xvii and len(segments) > 1:
        return _path(segments[1:], xvii=True, letter=segments[0])
    return tuple(segments)


def current_owner_path(trace: str, *, xvii: bool) -> DevicePath | None:
    """What a raw current belongs to: a device for ``Id(xa:xin:Mtail)``, the
    instance itself for a port current ``Ix(xa:out)``. None for anything else."""
    match = _CURRENT.match(trace)
    if match is None:
        return None
    segments = match.group(2).casefold().split(":")
    if match.group(1).casefold() == "ix" and len(segments) > 1:
        # A port current: the port's name follows the instance's path.
        instances = segments[:-1]
        return tuple(f"x{s}" for s in instances) if xvii else tuple(instances)
    return _path(segments, xvii=xvii)


def owner_path(name: str, *, xvii: bool) -> DevicePath | None:
    """The device or instance an operating-point value's name belongs to: a
    ``@device[param]`` key or a current. None for a node voltage."""
    table = _DEVICE_PARAM.search(name)
    if table is not None:
        return table_device_path(table.group(1), xvii=xvii)
    return current_owner_path(name, xvii=xvii)


def names_device(name: str, wanted: str, *, xvii: bool) -> bool:
    """Whether the operating-point value ``name`` belongs to what ``wanted``
    names: the device as the simulator spells it (``m.xa.xleaf.m0``, which
    ``inspect``'s hierarchy addresses use), its path, part of its path, or an
    instance it is inside (``path_matches``)."""
    table = _DEVICE_PARAM.search(name)
    current = None if table is not None else _CURRENT.match(name)
    owner = table.group(1) if table is not None else current.group(2) if current else None
    if owner is None:
        return False
    if owner.casefold() == wanted.strip().casefold():
        return True
    path = owner_path(name, xvii=xvii)
    return path is not None and path_matches(path, wanted)


def find_device_param(names: Iterable[str], device: str, param: str, *, xvii: bool) -> str | None:
    """The one name among ``names`` that holds ``param`` of the device at the
    path ``device`` (``xa.xin.mtail``), whichever build or simulator spelled
    it; None when there is not exactly one."""
    want = tuple(segment for segment in re.split(r"[.:]", device.casefold()) if segment)
    found = [
        name
        for name in names
        if (match := _DEVICE_PARAM.search(name)) is not None
        and name[match.end() :].split("]", 1)[0].casefold() == param.casefold()
        and table_device_path(match.group(1), xvii=xvii) == want
    ]
    return found[0] if len(found) == 1 else None


def path_text(path: DevicePath) -> str:
    """A device path as a caller writes it: ``xa.xin.mtail``."""
    return ".".join(path)


def path_matches(path: DevicePath, wanted: str) -> bool:
    """Whether ``wanted`` (``mtail``, ``xin.mtail``, ``xa``; dots or colons)
    names ``path``, part of it, or an instance it is inside."""
    want = tuple(segment for segment in re.split(r"[.:]", wanted.casefold()) if segment)
    if not want or len(want) > len(path):
        return False
    return any(path[i : i + len(want)] == want for i in range(len(path) - len(want) + 1))


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
            path_text(table_device_path(name, xvii=xvii)), {"op_table": None, "raw": None}
        )
        entry["op_table"] = name
    for trace in raw_traces:
        match = _CURRENT.match(trace)
        if match is None or match.group(1).casefold() == "ix":
            continue
        key = path_text(_path(match.group(2).casefold().split(":"), xvii=xvii))
        # A top-level part has one name everywhere: listed only beside its table entry.
        if key in names or ":" in match.group(2):
            names.setdefault(key, {"op_table": None, "raw": None})["raw"] = match.group(2)
    return dict(sorted(names.items()))
