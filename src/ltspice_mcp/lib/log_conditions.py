"""What a simulator said about a run: its log and console lines, classified.

One answer to "what did the simulator say about this run", for every reader:
the run rows of ``run_experiments`` and ``jobs``, the ``summary`` recipe and
the Python API. A diagnostic is data, ``{code, count, first_text, channel}``,
plus ``severity`` only where the simulator wrote one: ``code`` is the server's
name for the condition, ``severity`` the simulator's own word, kept apart
because LTspice XVII calls an ignored model parameter an error on a run that
finished.

The wording differs by simulator and by build, so the rules are chosen by the
run's producer (:class:`Producer`). Each set of rules is built from logs the
simulator wrote (``tests/fixtures/ltspice_recorded`` for LTspice 26 and XVII,
the ngspice fixtures and a live ngspice run), and every line outside a known
condition and outside the producer's routine output (banners, settings,
timing, step lines, solver statistics, and the result blocks other readers
decode) is counted under ``unclassified`` with the first one as written, so a
line the server does not recognise is never dropped.

The parser worker runs every set over a captured log (:func:`classify_channels`)
without knowing who wrote it, so the decoded facts stay a function of the
captured bytes. A reader picks the set for its producer
(:func:`run_diagnostics`), which takes the producer as a required argument.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from ltspice_mcp.lib.log_types import LogCondition

Channel = Literal["log", "console"]

#: The code of a line no rule of the producer's set recognises.
UNCLASSIFIED = "unclassified"


#: Why a run's diagnostics are not its producer's own recorded reading, by the
#: value a reply carries; a reply without one was read with that reading.
COVERAGE: dict[str, str] = {
    "producer_unknown": (
        "the simulator that wrote the log is not known, so it was read with the "
        "wording of every simulator the server has recordings of"
    ),
    "build_unrecorded": (
        "the run named no build of its simulator that the server has recordings of, "
        "so its log was read with the wording of every recorded build"
    ),
    "unverified": (
        "the server has no recordings of this simulator's log wording, so every "
        "line is counted under 'unclassified'"
    ),
    "log_unread": "the run's log could not be read, so nothing it said is reported",
    "not_classified": (
        "the run was recorded before the server classified simulator logs, or its "
        "log was not read when it finished"
    ),
}


#: What the simulator said about one run, as :func:`run_diagnostics` answers:
#: ``{"diagnostics": [LogCondition, ...], "coverage": <a COVERAGE key>}``, each
#: key only when it says something.
RunDiagnostics = dict[str, Any]


@dataclass(frozen=True)
class Producer:
    """Who wrote a log: the simulator family, the build the run named in its own
    output (``simulator_build.reported_build``), and the program that ran.

    ``family`` is a dialect name (``ltspice``, ``ngspice``, ``qspice``,
    ``xyce``) or None when it is not known; ``build`` and ``program`` are None
    when not known.
    """

    family: str | None
    build: str | None
    program: str | None


# --- Rules -----------------------------------------------------------------

_LT26 = "ltspice26"
_LT17 = "ltspice17"
_NG = "ngspice"

_Follow = Callable[[Sequence[str], int], int]
_Head = re.Pattern[str] | Callable[[Sequence[str], int], bool]


def _no_follow(_lines: Sequence[str], _index: int) -> int:
    return 0


@dataclass(frozen=True)
class _Rule:
    """A line (``head``) and the lines after it that belong to it (``follow``,
    which returns how many). ``code`` None is routine output. A rule with
    ``needs_follow`` applies only when ``follow`` took a line."""

    code: str | None
    head: _Head
    builds: frozenset[str]
    follow: _Follow = _no_follow
    needs_follow: bool = False

    def starts(self, lines: Sequence[str], index: int) -> bool:
        if isinstance(self.head, re.Pattern):
            return self.head.search(lines[index].strip()) is not None
        return self.head(lines, index)


def _stripped(lines: Sequence[str], index: int) -> str:
    return lines[index].strip() if 0 <= index < len(lines) else ""


_CARET = re.compile(r"^\s*\^+\s*$")


def _source_and_caret(source: re.Pattern[str] | None = None) -> _Follow:
    """LTspice 26's ``file(N): message`` block: the card it names and a caret
    line under the word, after at most one blank line, or a caret alone. With
    ``source``, the block belongs only when the card matches it."""

    def follow(lines: Sequence[str], index: int) -> int:
        start = index + 1 if not _stripped(lines, index) else index
        if start < len(lines) - 1 and _CARET.match(lines[start + 1]) and _stripped(lines, start):
            if source is not None and not source.search(_stripped(lines, start)):
                return 0
            return start + 2 - index
        if source is None and index < len(lines) and _CARET.match(lines[index]):
            return 1
        return 0

    return follow


def _next(*patterns: re.Pattern[str]) -> _Follow:
    """The following lines, each matching the next of ``patterns``; stops at the
    first that does not."""

    def follow(lines: Sequence[str], index: int) -> int:
        taken = 0
        for pattern in patterns:
            if not pattern.search(_stripped(lines, index + taken)):
                break
            taken += 1
        return taken

    return follow


def _while(pattern: re.Pattern[str]) -> _Follow:
    """Every following line that matches ``pattern``."""

    def follow(lines: Sequence[str], index: int) -> int:
        taken = 0
        while index + taken < len(lines) and pattern.search(lines[index + taken]):
            taken += 1
        return taken

    return follow


def _until_blank(lines: Sequence[str], index: int) -> int:
    taken = 0
    while _stripped(lines, index + taken):
        taken += 1
    return taken


def _re(pattern: str) -> re.Pattern[str]:
    return re.compile(pattern)


_BOTH_LT = frozenset({_LT26, _LT17})
_ONLY_26 = frozenset({_LT26})
_ONLY_17 = frozenset({_LT17})
_ONLY_NG = frozenset({_NG})

#: LTspice 26's ``<deck>(<line>): <message>`` diagnostic.
_FILE_LINE = r"^.+\(\d+\):\s+"
#: LTspice XVII's ``Error on line <n> : <card>``, which the next line explains.
_LINE_ERROR = _re(r"^Error on line \d+ : ")

# A number as LTspice and ngspice print it in a table row: with a unit tail
# (dB, a degree sign, a percent sign), or the word a failed or infinite one is.
_NUMBER = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:e[-+]?\d+)?(?:dB|°|%)?|[-+]?inf(?:dB)?|failed"
_ROW = _re(rf"^\(?(?:{_NUMBER})\)?(?:[\s,]+\(?(?:{_NUMBER})\)?)*,?$")
_KEYED_NUMBERS = _re(rf"^[A-Za-z]\w*:?(?:\s+(?:{_NUMBER}))+$")


def _step_line(lines: Sequence[str], index: int) -> bool:
    """A ``.step`` line, as the log parser reads one (``read_step_line``)."""
    # Imported here: the log parser imports this module.
    from ltspice_mcp.lib.log_parser import read_step_line

    return read_step_line(lines[index]) is not None


def _native_print_header(lines: Sequence[str], index: int) -> bool:
    """The head of an ngspice ``print`` table: two title lines over a ruled
    ``Index`` header."""
    return (
        bool(_stripped(lines, index))
        and re.fullmatch(r"-{20,}", _stripped(lines, index + 2)) is not None
        and _stripped(lines, index + 3).startswith("Index")
        and re.fullmatch(r"-{20,}", _stripped(lines, index + 4)) is not None
    )


def _native_print_rest(_lines: Sequence[str], _index: int) -> int:
    return 4


# The order matters: the first rule whose head matches a line takes it, so a
# condition comes before the generic rule its wording would also match, and
# every condition comes before routine output.
_RULES: tuple[_Rule, ...] = (
    # An unknown model parameter, which every simulator ignores and runs on.
    _Rule(
        "unknown_model_parameter",
        _re(_FILE_LINE + r"Ignoring unknown model parameter\."),
        _ONLY_26,
        _source_and_caret(),
    ),
    _Rule(
        "unknown_model_parameter",
        _LINE_ERROR,
        _ONLY_17,
        _next(_re(r'^\* Unrecognized parameter ".*" -- ignored$')),
        needs_follow=True,
    ),
    _Rule(
        "unknown_model_parameter",
        _re(r"^Warning: Model issue on line \d+ :"),
        _ONLY_NG,
        _next(_re(r"^\.model\s"), _re(r"^unrecognized parameter \(.*\) - ignored$")),
        needs_follow=True,
    ),
    # A parameter an instance card does not take.
    _Rule(
        "unknown_instance_parameter",
        _re(_FILE_LINE + r"Unknown parameter$"),
        _ONLY_26,
        _source_and_caret(),
    ),
    _Rule(
        "unknown_instance_parameter",
        _LINE_ERROR,
        _ONLY_17,
        _next(_re(r'^Unknown parameter ".*"$')),
        needs_follow=True,
    ),
    _Rule(
        "unknown_directive",
        _LINE_ERROR,
        _ONLY_17,
        _next(_re(r"^Unknown control card$")),
        needs_follow=True,
    ),
    # A file an .include or .lib names that the simulator could not open.
    _Rule(
        "missing_include", _re(_FILE_LINE + r"File not found\.$"), _ONLY_26, _source_and_caret()
    ),
    _Rule(
        "missing_include", _re(r"^Fatal Error: Could not open (?:include|library) file "), _ONLY_17
    ),
    # A model or subcircuit the deck uses and does not define.
    _Rule(
        "missing_model",
        _re(_FILE_LINE + r'(?:Undefined model ".*"\.|This sub-circuit name is not defined\.)$'),
        _ONLY_26,
        _source_and_caret(),
    ),
    _Rule("missing_model", _re(r'^Fatal Error:\s+Unable to find definition of model "'), _ONLY_17),
    _Rule(
        "missing_model",
        _re(r"^Fatal Error: Unknown subcircuit called in:$"),
        _ONLY_17,
        _next(_re(r"\S")),
    ),
    # A parameter an expression uses and nothing defines.
    _Rule(
        "undefined_parameter",
        _re(_FILE_LINE + r"No such parameter defined\.$"),
        _ONLY_26,
        _source_and_caret(),
    ),
    _Rule(
        "undefined_parameter",
        _re(r'^Questionable use of curly braces in ".*"$'),
        _ONLY_17,
        _next(_re(r'^Error: undefined symbol in: ".*"$')),
        needs_follow=True,
    ),
    # A .meas the simulator could not read, and one it could not take.
    _Rule(
        "measurement_error",
        _re(_FILE_LINE),
        _ONLY_26,
        _source_and_caret(_re(r"(?i)^\.meas")),
        needs_follow=True,
    ),
    _Rule(
        "measurement_error",
        _re(r"(?i)^\.meas"),
        _ONLY_17,
        _next(_re(r"^Error: ")),
        needs_follow=True,
    ),
    _Rule("measurement_failed", _re(r"^Measurement \".*\" FAIL'ed$"), _BOTH_LT),
    _Rule("floating_node", _re(r"^ERROR: Node \S+ is floating and connected to "), _BOTH_LT),
    _Rule(
        "singular_matrix",
        _re(r"^(?:Fatal Error: )?.+ are paralleled making an over-defined circuit matrix\.$"),
        _BOTH_LT,
        _next(_re(r"^You will need to correct the circuit or add some series resistance\.$")),
    ),
    _Rule("singular_matrix", _re(r"^Warning: singular matrix:\s+check node \S+$"), _ONLY_NG),
    # The rungs of the bias-point search past its first try; a run that needed
    # them may still have found one.
    _Rule(
        "operating_point_fallback",
        _re(
            r"^(?:Direct Newton iteration failed to find operating point\."
            r"|Starting Gmin stepping"
            r"|Gmin stepping (?:level \d+ )?failed to find operating point\."
            r"|Gmin stepping succeeded in finding the operating point\.$"
            r"|Starting source stepping with srcstepmethod="
            r"|Source stepping (?:level \d+ )?failed to find operating point\."
            r"|Pseudo Transient (?:level \d+ )?failed to find the operating point at )"
        ),
        _ONLY_26,
    ),
    _Rule(
        "operating_point_fallback",
        _re(
            r"^(?:Direct Newton iteration failed to find \.op point\."
            r"|Starting Gmin stepping$"
            r"|Gmin stepping (?:failed|succeeded in finding the operating point\.)$"
            r"|Starting source stepping with srcstepmethod=\d+$"
            r"|Source stepping failed$"
            r"|Pseudo Transient failed in finding the operating point at )"
        ),
        _ONLY_17,
    ),
    _Rule(
        "operating_point_fallback",
        _re(
            r"^(?:Note: Starting (?:dynamic gmin|true gmin|source) stepping$"
            r"|Warning: (?:Dynamic gmin|True gmin|source) stepping failed$"
            r"|Note: Source stepping completed$"
            r"|Note: Transient op (?:started|finished successfully)$)"
        ),
        _ONLY_NG,
    ),
    _Rule(
        "convergence_failed",
        _re(
            r"^(?:Convergence Failure:\s+Time step too small"
            r"|Fatal Error: Analysis:\s+Time step too small"
            r"|Simulation Failed: Iteration limit reached$"
            r"|Trouble finding operating point\.\.\.\.skipping operating point )"
        ),
        _BOTH_LT,
    ),
    _Rule(
        "device_size_warning",
        _re(
            r'^(?:Instance ".*": (?:Length shorter|Width narrower) than recommended for a '
            r"level \d+ MOSFET\.|Warning: P[ds] = \S+ is less than W\.)$"
        ),
        _BOTH_LT,
    ),
    _Rule("shorted_element", _re(r"^\S+: both pins shorted together -- ignoring\.$"), _ONLY_17),
    _Rule(
        "dangling_node",
        _re(r"^WARNING: Less than two connections to node \S+\s+This node is used by \S+$"),
        _ONLY_17,
    ),
    _Rule("source_value_assumed", _re(r"^Note: \S+: has no value, DC 0 assumed$"), _ONLY_NG),
    _Rule(
        "simulator_note",
        _re(r"^lossy: using linear interpolation because trytocompact option specified$"),
        _BOTH_LT,
    ),
    # Anything else a deck had that the simulator refused or ignored.
    _Rule("deck_error", _re(_FILE_LINE + r"\S"), _ONLY_26, _source_and_caret()),
    _Rule(
        "deck_error",
        _re(
            r"^(?:No analysis specified\.|More than one analysis specified\.)$"
            r"|^[A-Za-z][\w.:§]*: [A-Za-z][\w ]* must not be zero\.$"
            r"|^\.options? \w+ should be "
        ),
        _ONLY_26,
    ),
    _Rule("deck_error", _LINE_ERROR, _ONLY_17),
    _Rule("deck_error", _re(r"^Invalid \.save command\."), _ONLY_17),
    # ngspice's preamble names the compatibility mode in the form of a note.
    _Rule(
        None,
        _re(r"^Note: (?:Compatibility modes selected: .+|No compatibility mode selected!)$"),
        _ONLY_NG,
    ),
    # Lines in the simulator's own diagnostic form whose condition has no name here.
    _Rule("simulator_error", _re(r"^(?:Fatal Error|Error|ERROR)\s*:"), _BOTH_LT | _ONLY_NG),
    _Rule("simulator_warning", _re(r"^(?:Warning|WARNING)\s*:"), _BOTH_LT | _ONLY_NG),
    _Rule("simulator_note", _re(r"^Note: "), _ONLY_NG),
    # --- Routine output -----------------------------------------------------
    _Rule(None, _re(r"^LTspice [\d.]+ for Windows$"), _ONLY_26),
    _Rule(None, _re(r"^Start Time: "), _ONLY_26),
    _Rule(None, _re(r"^Circuit: "), _BOTH_LT | _ONLY_NG),
    _Rule(None, _re(r"^Files loaded:$"), _ONLY_26, _until_blank),
    _Rule(None, _re(r"^Date: "), _ONLY_17),
    _Rule(
        None,
        _re(
            r"^(?:\.OP point found by inspection\."
            r"|Direct Newton iteration succeeded in finding operating point\."
            r"|Per \.tran options, skipping operating point for transient analysis\.)$"
        ),
        _ONLY_26,
    ),
    _Rule(
        None,
        _re(
            r"^(?:\.OP point found by inspection\."
            r"|Direct Newton iteration for \.op point succeeded\."
            r"|Per \.tran options, skipping operating point for transient analysis\.)$"
        ),
        _ONLY_17,
    ),
    # Settings and solver statistics.
    _Rule(None, _re(r"^Options:\S"), _ONLY_26),
    _Rule(None, _re(r"^Maximum thread count: \d+$"), _ONLY_26),
    _Rule(None, _re(r"^\w+ = (?:true|false)$"), _ONLY_26),
    _Rule(None, _re(r"^(?:tnom|temp) = \S+$"), _BOTH_LT),
    _Rule(None, _re(r"^method = (?:trap|modified trap|Gear)$"), _BOTH_LT),
    _Rule(None, _re(r"^solver = (?:Normal|Alternate)$"), _BOTH_LT),
    _Rule(None, _re(r"^Total elapsed time: [\d.]+ seconds\.$"), _BOTH_LT),
    _Rule(None, _re(r"^Total iterations: \d+$"), _ONLY_26),
    _Rule(
        None,
        _re(r"^(?:totiter|traniter|tranpoints|accept|rejected|matrix size|fillins) = \d+$"),
        _ONLY_17,
    ),
    _Rule(None, _re(r"^Matrix Compiler\d: "), _ONLY_17),
    # Progress lines of a time step or bias-point search.
    _Rule(None, _re(r"^Changing Tseed to \S+$"), _BOTH_LT),
    _Rule(
        None,
        _re(
            r"^(?:Gmin = \S+|vernier = \S+|Source Step = \S+%"
            r"|Increasing initial diagonal Gmin to \S+)$"
        ),
        _ONLY_17,
    ),
    # Step lines and the result blocks the measurement, Fourier, device
    # operating-point and frequency-response readers decode.
    _Rule(None, _step_line, _BOTH_LT),
    _Rule(None, _re(r"^Measurement: \S+$"), _BOTH_LT),
    _Rule(None, _re(r"^step\t"), _BOTH_LT),
    _Rule(None, _re(r"^[\w.]+: .*="), _BOTH_LT),
    _Rule(None, _re(r"^\w+=\S+ FROM \S+ TO \S+$"), _BOTH_LT),
    _Rule(None, _re(r"^Semiconductor Device Operating Points:$"), _BOTH_LT),
    _Rule(None, _re(r"^--- .+ ---$"), _BOTH_LT),
    _Rule(None, _re(r"^(?:Name|Model):(?:\s+\S+)+$"), _BOTH_LT),
    _Rule(None, _KEYED_NUMBERS, _BOTH_LT),
    _Rule(None, _re(r"^Fourier components of \S"), _BOTH_LT),
    _Rule(None, _re(r"^(?:N-Period=\S+|DC component:\S+)$"), _BOTH_LT),
    _Rule(None, _re(r"^Harmonic\tFrequency\t"), _BOTH_LT),
    _Rule(None, _re(r"^Number \t"), _BOTH_LT),
    _Rule(None, _re(r"^(?:Total|Partial) Harmonic Distortion:\s*\S+$"), _BOTH_LT),
    _Rule(None, _re(r"^FRA @\S+ - Simultaneous frequencies "), _ONLY_26),
    _Rule(None, _re(r"^FRA Instance @\S+: "), _ONLY_26),
    _Rule(None, _ROW, _BOTH_LT | _ONLY_NG),
    # ngspice's preamble, run statistics and printed tables.
    _Rule(
        None,
        _re(
            r'^(?:binary raw file ".*"'
            r"|Doing analysis at TEMP = \S+ and TNOM = \S+"
            r"|Using SPARSE 1\.3 as Direct Linear Solver"
            r"|No\. of Data (?:Columns|Rows) : \d+"
            r"|Total (?:analysis|elapsed) time \(seconds\) = \S+"
            r"|(?:Total DRAM|DRAM currently) available = \S+ MB\."
            r"|(?:Maximum|Current) ngspice program size = \s*\S+ MB\."
            r"|(?:Shared ngspice|Text \(code\)|Library) pages = \s*\S+ MB\."
            r"|Stack = \d+ bytes\."
            r"|ngspice-\S+ done)$"
        ),
        _ONLY_NG,
    ),
    _Rule(
        None,
        _re(r"^List of plots available:$"),
        _ONLY_NG,
        _while(_re(r"^(?:Current \S|\t\S|\s*$)")),
    ),
    _Rule(None, _native_print_header, _ONLY_NG, _native_print_rest),
    # The banner and file notes ngspice writes to its console in batch mode.
    _Rule(None, _re(r"^\*{6}$|^\*\* "), _ONLY_NG),
    _Rule(
        None,
        _re(
            r"^(?:Batch mode|Simulation output goes to rawfile: .+"
            r"|Comments and warnings go to log-file: .+)$"
        ),
        _ONLY_NG,
    ),
    _Rule(None, _re(r"^\S+ = \S+$"), _ONLY_NG),
)

_SEVERITY = re.compile(r"^(Fatal Error|Error|ERROR|Warning|WARNING|Note)\b")


def _rules_of(*builds: str) -> tuple[_Rule, ...]:
    return tuple(rule for rule in _RULES if rule.builds & set(builds))


#: Each set of rules a log can be read with, by name. ``ltspice`` is every
#: recorded LTspice build's, ``wording`` every recorded simulator's, and
#: ``unverified`` none.
_CLASSIFIERS: dict[str, tuple[_Rule, ...]] = {
    _LT26: _rules_of(_LT26),
    _LT17: _rules_of(_LT17),
    "ltspice": _rules_of(_LT26, _LT17),
    _NG: _rules_of(_NG),
    "wording": _RULES,
    "unverified": (),
}
CLASSIFIERS: tuple[str, ...] = tuple(_CLASSIFIERS)


def _occurrences(text: str, rules: Sequence[_Rule]) -> list[tuple[str, list[str]]]:
    """Each condition ``text`` names, with its lines as written, in order."""
    lines = text.splitlines()
    found: list[tuple[str, list[str]]] = []
    index = 0
    while index < len(lines):
        if not lines[index].strip():
            index += 1
            continue
        for rule in rules:
            if not rule.starts(lines, index):
                continue
            taken = rule.follow(lines, index + 1)
            if rule.needs_follow and not taken:
                continue
            if rule.code is not None:
                found.append((rule.code, lines[index : index + 1 + taken]))
            index += 1 + taken
            break
        else:
            found.append((UNCLASSIFIED, [lines[index]]))
            index += 1
    return found


#: How much of a first occurrence an entry holds; the run's log holds all of it.
FIRST_TEXT_CHARS = 600


def _held(text: str) -> str:
    """``text`` as written, cut at :data:`FIRST_TEXT_CHARS` with the length left out."""
    if len(text) <= FIRST_TEXT_CHARS:
        return text
    return f"{text[:FIRST_TEXT_CHARS]}… (+{len(text) - FIRST_TEXT_CHARS} characters)"


def classify(channels: Mapping[Channel, str], rules_name: str) -> list[LogCondition]:
    """The conditions the ``channels``' text names under one set of rules: one
    entry per code, in the order each first appears, log before console."""
    rules = _CLASSIFIERS[rules_name]
    entries: dict[str, LogCondition] = {}
    for channel in ("log", "console"):
        for code, block in _occurrences(channels.get(channel) or "", rules):
            seen = entries.get(code)
            if seen is not None:
                seen["count"] += 1
                continue
            entry: LogCondition = {
                "code": code,
                "count": 1,
                "first_text": _held("\n".join(block)),
                "channel": channel,
            }
            severity = next(
                (m[1] for line in block if (m := _SEVERITY.match(line.strip())) is not None),
                None,
            )
            if severity is not None:
                entry["severity"] = severity
            entries[code] = entry
    return list(entries.values())


def classify_channels(log_text: str, console_text: str) -> dict[str, list[LogCondition]]:
    """What a captured log and console capture say, under every set of rules,
    leaving out a set that found nothing. The parser worker's half: it runs
    without knowing who wrote the log, and :func:`run_diagnostics` picks the
    producer's set."""
    channels: dict[Channel, str] = {"log": log_text, "console": console_text}
    result: dict[str, list[LogCondition]] = {}
    for name in _CLASSIFIERS:
        found = classify(channels, name)
        if found:
            result[name] = found
    return result


_LTSPICE_26 = re.compile(r"^LTspice 26\.")


def rules_for(producer: Producer) -> tuple[str, str | None]:
    """The set of rules ``producer``'s log is read with, and the
    :data:`COVERAGE` key that says why it is not the producer's own, if so."""
    # Here rather than at the top: the parser worker imports this module, and
    # the build check would bring the raw reader into it.
    from ltspice_mcp.lib.simulator_build import ltspice_before_24

    if producer.family is None:
        return "wording", "producer_unknown"
    if producer.family == "ltspice":
        if ltspice_before_24(producer.program, producer.build) is not None:
            return _LT17, None
        if producer.build is not None and _LTSPICE_26.match(producer.build):
            return _LT26, None
        return "ltspice", "build_unrecorded"
    if producer.family == "ngspice":
        return _NG, None
    return "unverified", "unverified"


def run_diagnostics(
    conditions: Mapping[str, Sequence[LogCondition]] | None,
    producer: Producer,
    *,
    unread: bool = False,
) -> RunDiagnostics:
    """What ``producer``'s run said, from the ``conditions`` a log decode
    found (the ``conditions`` of its ``diagnostics`` section). ``unread`` (or
    no ``conditions``) says the log could not be read."""
    if unread or conditions is None:
        return {"coverage": "log_unread"}
    name, coverage = rules_for(producer)
    result: RunDiagnostics = {}
    found = [dict(entry) for entry in conditions.get(name, ())]
    if found:
        result["diagnostics"] = found
    if coverage is not None:
        result["coverage"] = coverage
    return result


def decoded_run_diagnostics(
    logs: Any, producer: Producer, *, unread: bool = False
) -> RunDiagnostics:
    """:func:`run_diagnostics` over a ``DecodedLog``: the conditions of its
    ``diagnostics`` section, or ``log_unread`` when that section was not read."""
    if logs is None or unread:
        return {"coverage": "log_unread"}
    section = logs.section("diagnostics")
    if section["status"] != "parsed":
        return {"coverage": "log_unread"}
    return run_diagnostics(section["value"]["conditions"], producer)


def reply_fields(found: Mapping[str, Any]) -> dict[str, Any]:
    """A :class:`RunDiagnostics` as the fields a run row or a summary carries:
    ``diagnostics`` and ``diagnostics_coverage``, each only when it says
    something."""
    fields: dict[str, Any] = {}
    if found.get("diagnostics"):
        fields["diagnostics"] = [dict(entry) for entry in found["diagnostics"]]
    if found.get("coverage"):
        fields["diagnostics_coverage"] = found["coverage"]
    return fields
