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

import itertools
import re
import string
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ltspice_mcp.lib.spice_lex_ops import SECTION_SIGN, section_sign_names

if TYPE_CHECKING:
    from ltspice_mcp.lib.decoded_raw import RawData
    from ltspice_mcp.lib.simulator_build import Producer

DevicePath = tuple[str, ...]

#: ASCII capitals to small letters, and nothing else.
_ASCII_FOLD = str.maketrans(string.ascii_uppercase, string.ascii_lowercase)


def fold_name(name: str) -> str:
    """A node, net, trace or device name in the form two names are compared
    in: the one fold of a name, for the deck, the sheet and the result alike.

    Names that differ only in ASCII case are one name to both simulators
    (recorded: ``node-name-case``, ``deck/case_insensitive``), and that is all
    it folds. What LTspice does with names that differ beyond ASCII case
    (``Maß`` and ``MASS``, ``É`` and ``é``) is not recorded, so they compare
    exactly. ngspice-42 replaces each non-ASCII byte of a node name with
    ``_``, so two such names can be one node; that is reported by the lint
    (``node-non-ascii-ngspice``) and not modelled here.
    """
    return name.translate(_ASCII_FOLD)

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
    from ltspice_mcp.lib.simulator_build import which_ltspice

    if producer.family == "ngspice":
        return Naming("ngspice", xvii=False, read_as="ngspice")
    if producer.family != "ltspice":
        return Naming(producer.family, xvii=False, read_as=None)
    found = which_ltspice(producer.program, producer.build)
    generation = found.generation if found is not None else None
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


def naming_disagreement(expected_xvii: bool, command: str | None) -> str | None:
    """Why the raw whose header's ``Command`` is ``command`` names subcircuit
    members otherwise than the spelling its deck was read with
    (``expected_xvii``, :func:`naming_for`'s ``xvii`` at submission), or None
    when they agree or the raw names no LTspice writer. For a run the server
    made, where the two must agree."""
    from ltspice_mcp.lib.simulator_build import is_cp1252_ltspice_build

    if not command or "ltspice" not in command.casefold():
        return None
    written = is_cp1252_ltspice_build(command)
    if written == expected_xvii:
        return None
    return (
        f"The raw's writer ({command}) names each subcircuit instance "
        f"{'without' if written else 'with'} its X, but the job's decks were read with "
        f"each instance spelled {'without' if expected_xvii else 'with'} it: a name "
        "inside a subcircuit the deck spelled, a .save among them, may have saved "
        "nothing. Names in the results are read as the raw's writer spells them."
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
    return tuple(segment for segment in re.split(r"[.:]", fold_name(text)) if segment)


def _instances(segments: Iterable[str], *, xvii: bool) -> DevicePath:
    """Instance names as the deck writes them: XVII drops each one's ``X``."""
    return tuple(f"x{segment}" for segment in segments) if xvii else tuple(segments)


def table_device_path(name: str, *, xvii: bool) -> DevicePath:
    """The device a name in the log's device table (or an ngspice ``@`` trace)
    belongs to, read as the build that wrote it spells the table (``xvii``,
    :func:`table_naming`): ``xa:xin:mtail`` and ``m:a:in:tail`` are both
    ``("xa", "xin", "mtail")``, ngspice's ``m.xa.xin.mtail`` too.

    LTspice 26 writes the path of instance names, then the device's own name;
    XVII the device's letter, each instance without its X, then the rest of
    the device's name. LTspice 26 writes a block placed as U1, which its
    export names ``X§U1``, as ``u1`` (recorded: ``export/hier_device_logopinfo``,
    ``u1:D1``), where XVII exports it ``XU1`` and writes ``d:u1:1``: both are
    the instance ``xu1``, the export's name without the ``§``, as every
    instance of a deck starts with X."""
    low = fold_name(name)
    if ":" not in low:
        segments = [segment for segment in low.split(".") if segment]
        # ngspice writes the device's class letter first: m.x1.mn.
        if len(segments) > 1 and len(segments[0]) == 1:
            segments = segments[1:]
        return tuple(segments)
    if xvii:
        # XVII writes the device's letter first and drops it from its name.
        first, *instances, last = low.split(":")
        return (*_instances(instances, xvii=True), first + last)
    *instances, last = low.split(":")
    return (*(i if i.startswith("x") else f"x{i}" for i in instances), last)


def table_naming(writer: Producer, printed_by: str | None) -> Naming:
    """The spelling of the names in an LTspice log's device table, written by
    ``writer`` (a job case's producer, or what a bare path names) whose log
    opens with ``printed_by`` (``simulator_build.log_banner_build``), None
    when its first line names no build.

    The build ``writer`` names, else the one the banner names; a log with a
    device table, no banner and nothing else naming its build was written by
    an LTspice before 24, as 24 and later open the log of every run they
    start with the banner (every LTspice 26 recording of a run does; only a
    deck refused before its run lacks it, and prints no table; no XVII
    recording has one), and is read as XVII writes it.
    """
    # Here, not at the top: the raw parser reads device names through this
    # module, and simulator_build reads a raw's writer through the raw parser.
    from ltspice_mcp.lib.simulator_build import Producer, which_ltspice

    if writer.build is None and printed_by is not None:
        writer = Producer(writer.family or "ltspice", printed_by, writer.program)
    if writer.family not in (None, "ltspice") or which_ltspice(writer.program, writer.build):
        return naming_for(writer)
    return Naming(
        "ltspice",
        xvii=True,
        read_as="LTspice XVII",
        note=(
            "the log's first line names no build, as every log of LTspice 24 and later "
            "does, so its device table is read as LTspice XVII writes one"
        ),
    )


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
    #: Every path the name may stand for (:func:`trace_keys`): ``path`` alone
    #: but on an XVII run of a schematic, where ``path`` is the reading of
    #: each instance as its card and the others read some as the sheet's parts.
    readings: tuple[DevicePath, ...] = ()


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
    match = _SHORTHAND.fullmatch(fold_name(signal))
    return None if match is None else (match.group(1), match.group(2))


def device_param_name(names: Iterable[str], signal: str, *, xvii: bool) -> str | None:
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
    by_lower = {fold_name(name): name for name in names}
    for form in (f"@{dev}[{param}]", f"v(@{dev}[{param}])", f"i(@{dev}[{param}])"):
        if form in by_lower:
            return by_lower[form]
    return find_device_param(by_lower.values(), *parsed, xvii=xvii)


def _param_owner(name: str, *, xvii: bool, schematic: bool = False) -> _Owner | None:
    """The device a ``@device[param]`` name holds a parameter of, as the log's
    table, written as ``xvii`` says (:func:`table_device_path`), or ngspice
    spells it; None for any other name.

    ``path`` reads each instance as its card. On a run of a schematic
    (``schematic``) the table's own spelling of an instance is a reading too,
    as :func:`trace_keys` reads a trace: XVII writes part MB1 (card ``XMB1``)
    ``m:mb1:1``, which is ``mb1`` or the card ``xmb1``, and LTspice 26 writes
    it ``mb1:m1`` (``export/sheet_part_names``, ``export/hier_device_logopinfo``)."""
    parsed = parse_device_param(name)
    if parsed is None:
        return None
    device, param = parsed
    path = table_device_path(device, xvii=xvii)
    readings: tuple[DevicePath, ...] = ()
    low = fold_name(device)
    if schematic and ":" in low:
        if xvii:
            first, *instances, last = low.split(":")
            own = first + last
            readings = tuple(
                (*reading, own)
                for reading in itertools.product(*((i, f"x{i}") for i in instances))
            )
        else:
            *instances, last = low.split(":")
            readings = tuple(dict.fromkeys(((*instances, last), path)))
    return _Owner(device, path, param=param, readings=readings)


def _owner(name: str, *, xvii: bool, schematic: bool = False) -> _Owner | None:
    """What the value ``name`` belongs to, its instances read as
    :func:`trace_keys` reads them (``schematic`` for a result of a run of a
    schematic). ``path`` is the reading of each instance as its card, the one
    a deck's run has and the one the log's device table is read in
    (:func:`table_device_path`); ``readings`` holds every reading."""
    if (owner := _param_owner(name, xvii=xvii, schematic=schematic)) is not None:
        return owner
    current = _CURRENT.match(name)
    if current is None:
        return None
    text = current.group(2)
    # The last reading reads every instance as its card (trace_keys).
    readings = tuple(path for _, path in trace_keys(name, xvii=xvii, schematic=schematic))
    if fold_name(current.group(1)) == "ix" and len(readings[-1]) > 1:
        # The port's name follows the instance's path.
        instances = tuple(path[:-1] for path in readings)
        return _Owner(text, instances[-1], port=True, readings=instances)
    return _Owner(text, readings[-1], readings=readings)


def _raw_path(text: str, *, xvii: bool) -> DevicePath:
    """The path a raw's name for a node or device spells: LTspice's ``:`` or
    ngspice's ``.`` between instances, each instance without its ``X`` on
    XVII."""
    if ":" not in text:
        return split_path(text)
    *instances, last = fold_name(text).split(":")
    return (*_instances(instances, xvii=xvii), last)


def raw_name(scope: Iterable[str], name: str, *, xvii: bool) -> str:
    """How an LTspice raw, and so ``.save``, names a node or device inside
    subcircuit instances, the reverse of :func:`trace_keys`'s reading: the
    instances as written on LTspice 26 (``xa:xin:mid``), each without its X on
    XVII (``a:in:mid``). Each build's ``.save`` takes only its own spelling
    (recorded: ``raw/save_subckt_names*``)."""
    instances = (fold_name(segment)[1:] if xvii else fold_name(segment) for segment in scope)
    return ":".join((*instances, fold_name(name)))


def spelled_name(scope: Iterable[str], name: str, naming: Naming) -> str:
    """A node or device inside the instances ``scope``, as ``naming``'s reader
    spells it in a trace and a ``.save``: LTspice's :func:`raw_name`, or
    ngspice's dotted path (``xa.xin.mid``). The one speller of a path."""
    if naming.family == "ngspice":
        return ".".join((*(fold_name(segment) for segment in scope), fold_name(name)))
    return raw_name(scope, name, xvii=naming.xvii)


def device_address(path: Sequence[str], naming: Naming) -> str:
    """How a device at ``path`` (its instances, then its own name) is named
    to select it: ngspice's device name, its class letter before a nested
    path (``m.xa.mn``, ``m1`` at the top); for LTspice the path as the
    ``operating_point`` selector reads it on either build (``xa:mn``,
    :func:`device_query`)."""
    lowered = [fold_name(segment) for segment in path]
    if naming.family == "ngspice":
        device = ".".join(lowered)
        return f"{lowered[-1][0]}.{device}" if len(lowered) > 1 else device
    return ":".join(lowered)


TraceKey = tuple[str, DevicePath]


def trace_keys(name: str, *, xvii: bool, schematic: bool) -> tuple[TraceKey, ...]:
    """Every key the trace ``name`` may stand for, however a build or
    simulator spells it: its function and the path to its node or device, an
    instance by the name it has where it was placed (a deck's card name, a
    sheet's reference). The one answer to which instance path a trace's name
    names; ``xvii`` for a result an XVII-spelling build wrote, ``schematic``
    for one of a run of a schematic (``trace_names.from_schematic``).

    LTspice 26's ``V(xa:xin:mid)``, XVII's ``V(a:in:mid)``, ngspice's
    ``v(xa.xin.mid)`` and a caller's ``V(xa.xin.mid)`` are all
    ``("v", ("xa", "xin", "mid"))``. A name of another form is its own key.

    XVII writes every instance without the X of its netlist card. A deck's
    card is the instance's own name (``Xa`` is ``a``: ``raw/subckt_names_tran_saved``),
    so a deck's trace has one key. A sheet's export writes a part given a
    Prefix of X as ``X`` and its reference (block ``X1`` as ``XX1``, part
    ``MB1`` as ``XMB1``), which XVII writes as the reference (``Ix(x1:IN)``,
    ``Id(mb1:M1)``; LTspice 26 writes ``x1`` and ``mb1`` too), and a card a
    directive on the sheet writes (``XMB2``) without its X (``mb2``, which
    LTspice 26 writes ``xmb2``; recorded: ``export/hier_plain_top_run``,
    ``export/sheet_part_names``). The name alone does not say which an
    instance of a schematic's XVII run is, and XVII's own netlist neither
    (``XMB1`` and ``XMB2`` alike), so each instance is read both ways and the
    name has a key for each reading, the sheet's first. A reader that needs
    one key pairs them against another result's keys (:func:`pair_traces`).
    """
    match = _TRACE.match(name.strip())
    if match is None:
        return (("", (fold_name(name.strip()),)),)
    function, text = fold_name(match.group(1)), match.group(2)
    if not (xvii and schematic and ":" in text):
        return ((function, _raw_path(text, xvii=xvii)),)
    *instances, last = fold_name(text).split(":")
    readings = itertools.product(*((segment, f"x{segment}") for segment in instances))
    return tuple((function, (*reading, last)) for reading in readings)


def _deck_key(name: str, *, xvii: bool) -> TraceKey:
    """The one key of ``name`` read as a deck's run writes it, each instance
    its card (:func:`trace_keys` with no schematic): what a deck's own
    ``.save`` and ``.meas`` name, and what its cards are."""
    (key,) = trace_keys(name, xvii=xvii, schematic=False)
    return key


@dataclass(frozen=True)
class TracePairing:
    """Which trace of one result is which of another (:func:`pair_traces`), by
    index into each list of keys."""

    pairs: tuple[tuple[int, int], ...]
    #: A trace of ``here`` whose keys meet more than one trace of ``there``,
    #: or one trace there that another trace here meets too: it is not paired,
    #: never guessed between. Each to the traces there it could be.
    ambiguous: Mapping[int, tuple[int, ...]]
    #: The same, from ``there``'s side.
    ambiguous_there: Mapping[int, tuple[int, ...]]


def pair_traces(
    here: Sequence[tuple[TraceKey, ...]], there: Sequence[tuple[TraceKey, ...]]
) -> TracePairing:
    """Pair the traces of two results by their keys (:func:`trace_keys`): a
    trace here and a trace there are one when a key of each is the same and
    neither meets another trace so. The one rule for pairing two results'
    traces."""
    index: dict[TraceKey, list[int]] = {}
    for position, keys in enumerate(there):
        for key in dict.fromkeys(keys):
            index.setdefault(key, []).append(position)
    meets = [
        tuple(dict.fromkeys(j for key in keys for j in index.get(key, ()))) for keys in here
    ]
    met_by: dict[int, list[int]] = {}
    for i, found in enumerate(meets):
        for j in found:
            met_by.setdefault(j, []).append(i)
    pairs = tuple(
        (i, found[0])
        for i, found in enumerate(meets)
        if len(found) == 1 and len(met_by[found[0]]) == 1
    )
    paired_here = {i for i, _ in pairs}
    paired_there = {j for _, j in pairs}
    return TracePairing(
        pairs,
        {i: found for i, found in enumerate(meets) if found and i not in paired_here},
        {j: tuple(found) for j, found in met_by.items() if j not in paired_there},
    )


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


def device_query(
    wanted: str, *, xvii: bool, schematic: bool, among: Iterable[str]
) -> Callable[[str], bool]:
    """A test of whether an operating-point value's name belongs to what
    ``wanted`` names: the device as the simulator spells it (``m.xa.xleaf.m0``,
    which ``inspect``'s hierarchy addresses use), its path or part of it
    (``mtail``, ``xin.mtail``, ``xa.xin.mtail``; dots or colons), or an
    instance it is inside.

    A name exported with LTspice's ``§`` (``R§Load``) names the part by the
    one of its names (:func:`section_sign_spellings`) that a name of ``among``,
    every name of the result, belongs to: ``Load`` in LTspice's results,
    ``RLoad`` in ngspice's. Where both are there, they are two parts and the
    name answers to neither, as ``trace_names.find_trace`` refuses it.

    On an XVII run of a schematic (``schematic``) an instance is read both
    as the sheet's part and as a directive's card (:func:`trace_keys`), so
    ``mb1`` and ``xmb1`` both name part MB1 there (``export/sheet_part_names``)."""
    names = list(among)

    def query(spelled: str) -> Callable[[str], bool]:
        literal = fold_name(spelled.strip())
        want = split_path(spelled)

        def belongs(name: str) -> bool:
            owner = _owner(name, xvii=xvii, schematic=schematic)
            return owner is not None and (
                fold_name(owner.text) == literal
                or any(_within(path, want) for path in owner.readings or (owner.path,))
            )

        return belongs

    spellings = section_sign_spellings(wanted)
    if not spellings:
        return query(wanted)
    found = [q for q in map(query, (wanted, *spellings)) if any(map(q, names))]
    return found[0] if len(found) == 1 else (lambda _name: False)


def find_device_param(names: Iterable[str], device: str, param: str, *, xvii: bool) -> str | None:
    """The one name among ``names`` that holds ``param`` of ``device``: the
    device as the simulator spells it, or the end of its path (``x1.mn`` for
    ``MN`` inside ``X1``, wherever ``X1`` is), whichever build or simulator
    wrote it. None when there is not exactly one."""
    asked = (device, *section_sign_spellings(device))
    literals = {fold_name(spelled) for spelled in asked}
    wants = [want for spelled in asked if (want := split_path(spelled))]
    found = [
        name
        for name in names
        if (owner := _param_owner(name, xvii=xvii)) is not None
        and owner.param is not None
        and fold_name(owner.param) == fold_name(param)
        and (
            fold_name(owner.text) in literals
            or any(owner.path[-len(want) :] == want for want in wants)
        )
    ]
    return found[0] if len(found) == 1 else None


@dataclass(frozen=True)
class DeckSignal:
    """What a run's reader makes of a signal a deck names in a ``.save`` or a
    ``.meas`` (:func:`deck_signal`).

    ``key`` is the trace it stores or reads (:func:`trace_keys`, the reading
    of a deck's own names: its instances are its cards), None when
    it stores or reads none; ``why`` says why it is None, or why the reading
    is not established, in which case ``established`` is False and ``key``
    is None too. ``spelled`` is the name as the reader spells it, where the
    deck spelled it otherwise and the reader's spelling is known.
    """

    key: TraceKey | None
    why: str | None = None
    established: bool = True
    spelled: str | None = None
    #: The reader refuses the whole ``.save`` that names it and saves every
    #: signal instead, as XVII does.
    voids_save: bool = False


#: Where each LTspice build's spelling of a name inside subcircuit instances
#: is recorded, for a ``.save`` and a ``.meas``.
_SPELLING_RECORDED = (
    "recorded: raw/save_subckt_names*, deck/meas_subckt_node_*: LTspice 26 "
    "stores and measures a node inside instances only spelled with each "
    "instance's X (xa:xin:mid), XVII only without it (a:in:mid)"
)


def deck_signal(
    text: str,
    naming: Naming,
    instances: Collection[str],
    *,
    saved: bool,
    marked: Collection[str] = (),
) -> DeckSignal:
    """What ``naming``'s reader stores for ``text`` named in a ``.save``
    (``saved``), or reads for it named in a ``.meas``, in a deck whose
    top-level subcircuit instances are ``instances``, folded
    (:func:`fold_name`), with the own names of those LTspice exported with a
    ``§`` in ``marked`` (``U1`` for ``X§U1``, whose name in ``instances`` is
    ``xu1``).

    A name outside every subcircuit is stored and read as written, in any
    ASCII case. A path through instances:

    - ngspice stores a path spelled with dots; a ``.save`` of one spelled
      with colons stores nothing (ngspice-42, measured). A ``.meas`` of a
      colon path is read as the dotted path the run stores, as every reader
      of the result reads it; ngspice's own ``.meas`` does not parse it
      ("can't parse 'x1:mid': ignored", measured), but a run in batch mode
      with a raw evaluates no top-level ``.meas`` (lint
      ``meas-ngspice-batch``), so the server's reading is the one that counts.
    - Each LTspice build stores and measures a path only in its own spelling
      (:data:`_SPELLING_RECORDED`), whose first segment must name an instance
      of the deck. Otherwise LTspice 26 stores and reads nothing for it, and
      XVII reads nothing for it in a ``.meas`` and refuses the whole ``.save``
      that names it, saving every signal instead ("Invalid .save command.
      Just saving everything...", ``voids_save``). Which spelling a build of
      no known spelling takes, a dotted path in an LTspice deck, and a
      segment that names an instance in both spellings or one exported with
      a ``§`` are not recorded: the reading is not established.
    """
    # A probe's spaces are no part of its name: ``v( a )`` is ``v(a)``.
    text = re.sub(r"\s+", "", text)
    match = _TRACE.match(text)
    if match is None:
        return DeckSignal(_deck_key(text, xvii=naming.xvii))
    function, node = match.group(1), match.group(2).strip()
    if ":" not in node and "." not in node:
        return DeckSignal(_deck_key(text, xvii=naming.xvii))
    if naming.family == "ngspice":
        if ":" in node and saved:
            dotted = f"{function}({node.replace(':', '.')})"
            return DeckSignal(
                None,
                "ngspice stores a name inside subcircuit instances only spelled with dots: "
                f"a .save of {text.strip()} stores nothing (ngspice-42, measured)",
                spelled=dotted,
            )
        return DeckSignal(_deck_key(text, xvii=False))
    if naming.family != "ltspice":
        return DeckSignal(
            None,
            f"how {naming.family or 'this simulator'} spells a name inside subcircuit "
            "instances is not modelled",
            established=False,
        )
    if ":" not in node:
        return DeckSignal(
            None,
            "whether LTspice reads a name inside subcircuit instances spelled with dots "
            "as one spelled with colons is not recorded",
            established=False,
        )
    first = fold_name(node.split(":", 1)[0])
    with_x = first in instances
    without_x = f"x{first}" in instances
    if first in {fold_name(name) for name in marked} or (with_x and without_x):
        return DeckSignal(
            None,
            f"{first!r} names more than one instance of the deck as LTspice spells it, "
            "or one exported with a §, and which one it reads is not recorded",
            established=False,
        )
    if not with_x and not without_x and not saved:
        # The same on both builds: neither measures a path through no instance.
        return DeckSignal(
            None,
            f"{text} names no instance of the deck ({first!r} or {('x' + first)!r}), "
            f"so neither LTspice build measures it ({_SPELLING_RECORDED})",
        )
    if naming.note is not None:
        return DeckSignal(
            None,
            f"which spelling the build that runs the deck takes is not established "
            f"({naming.note}; {_SPELLING_RECORDED})",
            established=False,
        )
    if (with_x if not naming.xvii else without_x):
        return DeckSignal(_deck_key(text, xvii=naming.xvii))
    spelled = None
    if with_x or without_x:
        key = _deck_key(text, xvii=not naming.xvii)
        *scope, name = key[1]
        spelled = f"{function}({spelled_name(scope, name, naming)})"
    names = (
        f"names nothing; it is spelled {spelled}"
        if spelled is not None
        else f"names no instance of the deck ({first!r} as {naming.read_as} spells it)"
    )
    if naming.xvii and saved:
        return DeckSignal(
            None,
            f"to {naming.read_as} {text} {names}, so it refuses the whole .save "
            "(\"Invalid .save command. Just saving everything...\") and saves every "
            f"signal outside subcircuits instead ({_SPELLING_RECORDED})",
            spelled=spelled,
            voids_save=True,
        )
    return DeckSignal(
        None,
        f"to {naming.read_as} {text} {names}, and it stores and measures nothing for it "
        f"({_SPELLING_RECORDED})",
        spelled=spelled,
    )


def find_trace_by_path(names: Iterable[str], signal: str, *, xvii: bool) -> str | None:
    """The one trace among ``names`` that ``signal`` names by a path through
    subcircuit instances, as either build writes it (``trace_keys``, as a
    deck's run names them; a schematic's run on XVII is found by its literal
    name, ``trace_names.find_trace``):
    ``V(xa.xin.mid)`` is LTspice 26's ``V(xa:xin:mid)`` and XVII's
    ``V(a:in:mid)``, ``Id(xa.xin.mtail)`` is ``Id(a:in:Mtail)`` on XVII, and
    ``Ix(xa.top)`` is ``Ix(a:TOP)``. None when ``signal`` names no such path or
    not exactly one trace matches."""
    if _TRACE.match(signal.strip()) is None:
        return None
    want = _deck_key(signal, xvii=False)
    if len(want[1]) < 2:
        return None
    # Every match ends with the path's last segment: a cheap test first.
    tail = f"{want[1][-1]})"
    found = [
        name
        for name in names
        if fold_name(name).endswith(tail) and _deck_key(name, xvii=xvii) == want
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
            path_text(table_device_path(name, xvii=xvii)), {"op_table": None, "raw": None}
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
