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
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ltspice_mcp.lib.spice_lex_ops import SECTION_SIGN, section_sign_names

if TYPE_CHECKING:
    from ltspice_mcp.lib.decoded_raw import RawData
    from ltspice_mcp.lib.simulator_build import Producer

DevicePath = tuple[str, ...]

#: A ``@device[param]`` name, bare or as ngspice wraps it (``v(@m1[vth])``).
_DEVICE_PARAM = re.compile(r"@([^\[\]()]+)\[([^\]]*)\]")
#: A caller's ``dev.param`` shorthand: everything before the last dot is the
#: device, so a flattened subcircuit path (``m.x1.mn.gm``) parses too.
_SHORTHAND = re.compile(r"([a-z][\w.]*)\.([a-z]\w*)")
_CURRENT = re.compile(r"^(i[a-z]?)\(([^)]+)\)$", re.IGNORECASE)
#: A trace of one node or one device: ``V(xa:xin:mid)``, ``Id(a:in:Mtail)``.
_TRACE = re.compile(r"^([a-z]+)\(([^(),]+)\)$", re.IGNORECASE)


#: An LTspice build its own run names, ``LTspice 26.0.2 for Windows``.
_NUMBERED_LTSPICE = re.compile(r"(?i)\bLTspice\s+(\d+)\.")
#: The LTspice builds whose names are recorded (``tests/fixtures/ltspice_recorded``).
_RECORDED_LTSPICE = {26: "LTspice 26"}


@dataclass(frozen=True)
class Naming:
    """How a reader of a deck, or the writer of a result, spells a name
    inside subcircuit instances (:func:`naming_for`).

    ``family`` is the simulator family (``ltspice``, ``ngspice``) or None.
    ``xvii`` says each instance is spelled without its X, as LTspice XVII
    and IV spell it. ``read_as`` is the recorded build whose spelling this
    follows, and ``note`` says why that is not the build's own, when it is
    not: a build with no recording is read as its nearest recorded one.
    """

    family: str | None
    xvii: bool
    read_as: str | None
    note: str | None = None


def naming_for(producer: Producer) -> Naming:
    """The spelling ``producer`` reads a deck's names in and writes a
    result's names in: the one derivation of a build's naming facts, for the
    deck side (``hierarchy.SemanticProfile.of``) and to check a result's
    writer (:func:`naming_disagreement`).

    LTspice XVII and IV write each instance without its X; LTspice 26 and
    every build from 24 on write it with its X (recorded:
    ``raw/save_subckt_names*``, ``raw/subckt_names_tran_saved``). Only 26 and
    XVII are recorded, so IV is read as XVII, and 24, 25, a build after 26 and
    one whose number is not known yet are read as 26, each with a note.
    ngspice spells a path with dots and keeps every X.
    """
    # Here, not at the top: the raw parser reads device names through this
    # module, and simulator_build reads a raw's writer through the raw parser.
    from ltspice_mcp.lib.simulator_build import ltspice_generation_before_24

    if producer.family == "ngspice":
        return Naming("ngspice", xvii=False, read_as="ngspice")
    if producer.family != "ltspice":
        return Naming(producer.family, xvii=False, read_as=None)
    generation = ltspice_generation_before_24(producer.program, producer.build)
    if generation == "xvii":
        return Naming("ltspice", xvii=True, read_as="LTspice XVII")
    if generation == "iv":
        return Naming(
            "ltspice",
            xvii=True,
            read_as="LTspice XVII",
            note=(
                "LTspice IV's names inside subcircuits are not recorded; they are "
                "spelled as LTspice XVII, the nearest recorded build, spells them"
            ),
        )
    numbered = _NUMBERED_LTSPICE.search(producer.build or "")
    if numbered is not None and int(numbered[1]) in _RECORDED_LTSPICE:
        return Naming("ltspice", xvii=False, read_as=_RECORDED_LTSPICE[int(numbered[1])])
    which = (
        f"LTspice {numbered[1]}'s names inside subcircuits are not recorded"
        if numbered is not None
        else "the LTspice build is not known until a run on it names itself"
    )
    return Naming(
        "ltspice",
        xvii=False,
        read_as="LTspice 26",
        note=f"{which}; they are spelled as LTspice 26, the nearest recorded build, spells them",
    )


def naming_disagreement(producer: Producer, command: str | None) -> str | None:
    """Why the raw whose header's ``Command`` is ``command`` names subcircuit
    members otherwise than :func:`naming_for` says ``producer`` does, or None
    when they agree or the raw names no LTspice writer. For a run the server
    made, where the two must agree."""
    from ltspice_mcp.lib.simulator_build import is_cp1252_ltspice_build

    if producer.family != "ltspice" or not command or "ltspice" not in command.casefold():
        return None
    expected = naming_for(producer)
    written = is_cp1252_ltspice_build(command)
    if written == expected.xvii:
        return None
    spelled = "without" if written else "with"
    return (
        f"The raw's writer ({command}) names each subcircuit instance {spelled} its X, "
        f"but the run's executable ({producer.program or 'not recorded'}, build "
        f"{producer.build or 'not reported'}) was read as {expected.read_as}, which "
        f"does not; names inside subcircuits are read as the raw's writer spells them."
    )


def writes_xvii_names(raw: RawData) -> bool:
    """Whether the build that wrote ``raw`` names devices as XVII does:
    LTspice XVII, and LTspice IV before it, by the header's ``Command``."""
    return writes_xvii_command(raw.raw_params)


def writes_xvii_command(fields: Mapping[str, object]) -> bool:
    """The same, from a raw header's fields, whose names are read in any case:
    what the parser worker reads before a raw is decoded."""
    # Here, not at the top: the raw parser reads device names through this
    # module, and simulator_build reads a raw's writer through the raw parser.
    from ltspice_mcp.lib.simulator_build import is_cp1252_ltspice_build

    command = next((value for key, value in fields.items() if key.lower() == "command"), None)
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


def parse_device_param(name: str) -> tuple[str, str] | None:
    """The device and parameter a ``@device[param]`` name holds, bare or as
    ngspice wraps it: ``("m1", "vth")`` for ``v(@m1[vth])``, the log table's
    ``@xa:xin:mtail[gm]`` too. None for any other name. The one reader of the
    form, wherever a result or a caller writes it."""
    match = _DEVICE_PARAM.search(name)
    return None if match is None else (match.group(1), match.group(2))


def parse_shorthand(signal: str) -> tuple[str, str] | None:
    """The device and parameter a caller's ``dev.param`` shorthand names,
    lowercased: ``("x1.mn", "gm")`` for ``X1.MN.gm``. None for another name."""
    match = _SHORTHAND.fullmatch(signal.lower())
    return None if match is None else (match.group(1), match.group(2))


def device_param_name(names: Iterable[str], signal: str) -> str | None:
    """The one name among ``names`` a ``dev.param`` shorthand addresses, or None.

    Its literal forms first: ngspice writes a device parameter bare
    (``@m1[gm]``), v-wrapped (``v(@m1[vth])``) or i-wrapped (``i(@m1[id])``)
    depending on the quantity, and LTspice's ``.log`` block is folded into
    ``device_op_points`` under the bare form. Then the device by the end of
    its path (``x1.mn.gm`` for ngspice's ``@m.x1.mn[gm]``, LTspice 26's
    ``@x1:mn[gm]`` or XVII's ``@m:1:n[gm]``), refused when more than one
    matches: never a guess.
    The one rule behind both readers of the shorthand.
    """
    parsed = parse_shorthand(signal)
    if parsed is None:
        return None
    dev, param = parsed
    by_lower = {name.lower(): name for name in names}
    for form in (f"@{dev}[{param}]", f"v(@{dev}[{param}])", f"i(@{dev}[{param}])"):
        if form in by_lower:
            return by_lower[form]
    return find_device_param(by_lower.values(), *parsed)


def _param_owner(name: str) -> _Owner | None:
    """The device a ``@device[param]`` name holds a parameter of, as the log's
    table or ngspice spells it; None for any other name."""
    parsed = parse_device_param(name)
    if parsed is None:
        return None
    device, param = parsed
    return _Owner(device, table_device_path(device), param=param)


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


def spelled_name(scope: Iterable[str], name: str, naming: Naming) -> str:
    """A node or device inside the instances ``scope``, as ``naming``'s reader
    spells it in a trace and a ``.save``: LTspice's :func:`raw_name`, or
    ngspice's dotted path (``xa.xin.mid``). The one speller of a path."""
    if naming.family == "ngspice":
        return ".".join((*(segment.lower() for segment in scope), name.lower()))
    return raw_name(scope, name, xvii=naming.xvii)


def device_address(path: Sequence[str], naming: Naming) -> str:
    """How a device at ``path`` (its instances, then its own name) is named
    to select it: ngspice's device name, its class letter before a nested
    path (``m.xa.mn``, ``m1`` at the top); for LTspice the path as the
    ``operating_point`` selector reads it on either build (``xa:mn``,
    :func:`device_query`)."""
    lowered = [segment.lower() for segment in path]
    if naming.family == "ngspice":
        device = ".".join(lowered)
        return f"{lowered[-1][0]}.{device}" if len(lowered) > 1 else device
    return ":".join(lowered)


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


def section_sign_spellings(name: str) -> tuple[str, ...]:
    """The other names a result may give what ``name`` names when a segment
    of it carries LTspice's ``§`` (``R§Load``): the part's own name, which
    LTspice's results use (``I(Load)``, recorded: ``export/load_name_op``),
    then the name without the marker, which ngspice's copy of the deck gives
    it (``spice_lex_ops.section_sign_names``). None for a name without one."""
    if SECTION_SIGN not in name:
        return ()
    parts = re.split(r"([(),.:\s]+)", name)
    own = "".join(_own(part, 1) for part in parts)
    bare = "".join(_own(part, 0) for part in parts)
    return tuple(dict.fromkeys(spelled for spelled in (own, bare) if spelled != name))


def _own(segment: str, which: int) -> str:
    """``segment``'s name without the marker (``which`` 0) or its own (1)."""
    names = section_sign_names(segment)
    return names[which] if names else segment


def device_query(wanted: str, *, xvii: bool, among: Iterable[str]) -> Callable[[str], bool]:
    """A test of whether an operating-point value's name belongs to what
    ``wanted`` names: the device as the simulator spells it (``m.xa.xleaf.m0``,
    which ``inspect``'s hierarchy addresses use), its path or part of it
    (``mtail``, ``xin.mtail``, ``xa.xin.mtail``; dots or colons), or an
    instance it is inside.

    A name exported with LTspice's ``§`` (``R§Load``) names the part by the
    one of its names (:func:`section_sign_spellings`) that a name of ``among``,
    every name of the result, belongs to: ``Load`` in LTspice's results,
    ``RLoad`` in ngspice's. Where both are there, they are two parts and the
    name answers to neither, as ``trace_names.find_trace`` refuses it."""
    names = list(among)

    def query(spelled: str) -> Callable[[str], bool]:
        literal = spelled.strip().casefold()
        want = split_path(spelled)

        def belongs(name: str) -> bool:
            owner = _owner(name, xvii=xvii)
            return owner is not None and (
                owner.text.casefold() == literal or _within(owner.path, want)
            )

        return belongs

    spellings = section_sign_spellings(wanted)
    if not spellings:
        return query(wanted)
    found = [q for q in map(query, (wanted, *spellings)) if any(map(q, names))]
    return found[0] if len(found) == 1 else (lambda _name: False)


def find_device_param(names: Iterable[str], device: str, param: str) -> str | None:
    """The one name among ``names`` that holds ``param`` of ``device``: the
    device as the simulator spells it, or the end of its path (``x1.mn`` for
    ``MN`` inside ``X1``, wherever ``X1`` is), whichever build or simulator
    wrote it. None when there is not exactly one."""
    asked = (device, *section_sign_spellings(device))
    literals = {spelled.casefold() for spelled in asked}
    wants = [want for spelled in asked if (want := split_path(spelled))]
    found = [
        name
        for name in names
        if (owner := _param_owner(name)) is not None
        and owner.param is not None
        and owner.param.casefold() == param.casefold()
        and (
            owner.text.casefold() in literals
            or any(owner.path[-len(want) :] == want for want in wants)
        )
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
