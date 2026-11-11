"""Capture and verify the exact owned bytes consumed by recoverable runs.

These helpers do blocking filesystem work; event-loop callers offload them.
Closure checks use staging's lexer/resolver over the captured file inventory,
so no second recursive include or hierarchy walker is needed.
"""

from __future__ import annotations

import hashlib
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from ltspice_mcp.lib import atomic_write_bytes, capabilities, now
from ltspice_mcp.lib.asc_document import is_asc
from ltspice_mcp.lib.deck_staging import (
    card_sections,
    sha256_file,
    staged_card_reference_targets,
)
from ltspice_mcp.lib.encoding import decode_spice_bytes
from ltspice_mcp.lib.experiment_types import ExperimentCase, SourceRecord
from ltspice_mcp.lib.ngspice_driver import deck_seeds, fixed_seed, seed_option
from ltspice_mcp.lib.pdk_native import ArtifactDigest, NativePaths, driver_bytes
from ltspice_mcp.lib.recovery_records import (
    ExecutionRecord,
    FrozenInputs,
    ProducedArtifacts,
    RecoveryError,
    StartupPolicy,
)
from ltspice_mcp.lib.spice_lex import (
    INCLUDE_HEADS,
    EditCards,
    SpiceCard,
    SpiceLexError,
    Token,
    TokenKind,
    lex_for_edit,
    tokenize_body,
)
from ltspice_mcp.lib.spice_lex_views import read_instance
from ltspice_mcp.lib.store import Store

if TYPE_CHECKING:
    from ltspice_mcp.lib.variations import MaterializedCase

STARTUP_VERSION = "ngspice-inert-spinit-v1"
_SPINIT = b"* Controlled inert system startup for recoverable simulation.\n"
_FUNCTION = re.compile(r"\b([A-Za-z_][A-Za-z_0-9]*)\s*\(")
_RANDOM = frozenset(
    {
        "rand",
        "random",
        "gauss",
        "agauss",
        "unif",
        "aunif",
        "flat",
        "sgauss",
        "limit",
        "mc",
        "poisson",
        "white",
        "flicker",
        "trnoise",
        "trrandom",
    }
)
# ngspice's electrical parser evaluates these through gauss1/drand, both
# backed by the generator setseed resets, as a seed option does. Transient
# noise uses other state.
_SEEDED_RANDOM = frozenset({"agauss", "gauss", "aunif", "unif", "limit"})
_PURE_FUNCTIONS = frozenset(
    {
        "abs",
        "acos",
        "acosh",
        "asin",
        "asinh",
        "atan",
        "atan2",
        "atanh",
        "cos",
        "cosh",
        "sin",
        "sinh",
        "tan",
        "tanh",
        "exp",
        "ln",
        "log",
        "log10",
        "sqrt",
        "pow",
        "pwr",
        "pwrs",
        "min",
        "max",
        "int",
        "nint",
        "floor",
        "ceil",
        "sgn",
        "sign",
        "if",
        "iif",
        "ternary_fcn",
        "u",
        "uramp",
        "ustep",
        "table",
        "buf",
        "inv",
        "d",
        "ddt",
        "idt",
        "idtmod",
        "delay",
        "laplace",
        "v",
        "i",
        "real",
        "imag",
        "mag",
        "phase",
        "db",
        "hypot",
        "cbrt",
        "round",
        "mod",
        "fmod",
        "poly",
        "pulse",
        "pwl",
        "sffm",
        "am",
        "pwlrepeated",
        "pwlrepeatforever",
        "x",
        "par",
        "vntol",
        "j",
        "asym",
    }
)
_DIRECTIVES = frozenset(
    {
        ".model",
        ".param",
        ".parameters",
        ".func",
        ".subckt",
        ".ends",
        ".tran",
        ".ac",
        ".dc",
        ".op",
        ".noise",
        ".meas",
        ".measure",
        ".save",
        ".probe",
        ".option",
        ".options",
        ".temp",
        ".step",
        ".ic",
        ".nodeset",
        ".global",
        ".end",
        ".endl",
        ".title",
        ".backanno",
        *INCLUDE_HEADS,
    }
)
_MODULE_DIRECTIVES = frozenset({".load", ".hdl", ".verilog", ".osdi", ".pre_osdi"})
# A refusal lists this many constructs, each with this many more places, and
# quotes each card to this length, so a long vendor library stays readable.
_GROUPS_SHOWN = 12
_PLACES_SHOWN = 5
_CARD_TEXT_CHARS = 100
_WARNING_LINE = re.compile(r"line (\d+):")


def _contained(path: Path, root: Path) -> Path:
    try:
        resolved_root = root.resolve(strict=True)
        resolved = path.resolve(strict=True)
        if (
            not root.is_absolute()
            or not path.is_absolute()
            or not resolved.is_relative_to(resolved_root)
            or not resolved.is_file()
        ):
            raise RecoveryError(
                "recovery_path_escape", "Artifact is outside its lineage directory"
            )
        return resolved
    except (OSError, RuntimeError) as exc:
        raise RecoveryError(
            "recovery_artifact_missing", "Required recovery artifact is unavailable"
        ) from exc


def _verify(artifact: ArtifactDigest, root: Path, *, code: str) -> None:
    resolved = _contained(artifact.path, root)
    try:
        actual = sha256_file(resolved)
    except OSError as exc:
        raise RecoveryError(
            "recovery_artifact_missing", "Required recovery artifact is unreadable"
        ) from exc
    if actual != artifact.sha256:
        raise RecoveryError(code, "Exact recovery artifact bytes have changed")


def capture_case_inputs(
    case: ExperimentCase,
    source: SourceRecord,
    *,
    lineage_root: Path,
    materialized: MaterializedCase | None = None,
    seeded: bool = False,
) -> FrozenInputs:
    """Freeze staged bytes, using materialization hashes rather than source hashes."""
    expected: dict[Path, str] = {}

    def bind_digest(path: Path, digest: str) -> None:
        if path in expected and expected[path] != digest:
            raise RecoveryError("recovery_input_drift", "Captured input identities disagree")
        expected[path] = digest

    if case.recovery is not None:
        expected.update((item.path, item.sha256) for item in case.recovery.inputs.files)
    if materialized is not None:
        if (
            materialized.case_id != case.case_id
            or materialized.path != case.staged_deck
            or materialized.sha256 != case.deck_sha256
        ):
            raise RecoveryError("recovery_input_drift", "Materialized case identity disagrees")
        for path, digest in materialized.file_digests:
            bind_digest(path, digest)
    bind_digest(case.staged_deck, case.deck_sha256)
    paths = dict.fromkeys(expected)
    live = [
        entry
        for entry in source.manifest
        if entry.live or not entry.staged or entry.staged_path is None
    ]
    if live:
        raise RecoveryError(
            "recovery_live_dependency",
            "Recoverable execution saves a copy of every file the deck reads, and "
            + ("this one is" if len(live) == 1 else "these are")
            + " read in place: "
            + "; ".join(
                f"{entry.path}" + (f" ({entry.reason})" if entry.reason else "") for entry in live
            )
            + ". Make each a file the server can copy, or submit without "
            "execution.recoverable.",
        )
    # Each frozen file as the caller knows it, for a refusal to name: a staged
    # file by the file staging copied, and a case's own copy by its staged file.
    deck = f"the netlist exported from {source.path}" if is_asc(source.path) else str(source.path)
    origins: dict[Path, str] = {}

    def name(path: Path, origin: str) -> None:
        origins.setdefault(path, origin)
        origins.setdefault(path.resolve(), origin)

    name(case.staged_deck, deck)
    name(source.staged_deck, deck)
    for entry in source.manifest:
        if entry.staged_path is not None:  # else refused above as live
            name(entry.staged_path, str(entry.path))
    if materialized is not None:
        for copy, staged in materialized.copied_from:
            origin = origins.get(staged) or origins.get(staged.resolve())
            if origin is not None:
                name(copy, origin)
    for entry in source.manifest:
        if entry.staged_path is None:
            continue
        # An exported schematic's authoring snapshot is provenance, not an
        # electrical dependency. Any include edge to it still fails closure
        # admission unless it was explicitly materialized as an input file.
        if entry.path == source.path and is_asc(source.path):
            continue
        # A variant replaces its standalone source deck. Keep that source only
        # when materialization explicitly includes it as an electrical input.
        if entry.staged_path == source.staged_deck and entry.staged_path not in expected:
            continue
        paths[entry.staged_path] = None
    native = case.native_statistics
    if native is not None:
        try:
            native.validate()
        except ValueError as exc:
            raise RecoveryError(
                "recovery_native_unprepared", "Native sample identity is invalid"
            ) from exc
        if native.sample is None:
            raise RecoveryError(
                "recovery_native_unprepared", "Native recovery requires a validated sample"
            )
        dependencies = (
            native.prepared.dependencies if native.prepared else native.pending_dependencies
        )
        for dependency in dependencies:
            bind_digest(dependency.path, dependency.sha256)
            paths[dependency.path] = None
        if native.prepared is not None:
            for path, digest in (
                (Path(native.prepared.paths.electrical_input), native.prepared.input_sha256),
                (Path(native.prepared.paths.prepared_driver), native.prepared.driver_sha256),
            ):
                bind_digest(path, digest)
                paths[path] = None
    artifacts: list[ArtifactDigest] = []
    for path in paths:
        _contained(path, lineage_root)
        try:
            digest = sha256_file(path)
        except OSError as exc:
            raise RecoveryError("recovery_artifact_missing", "Frozen input is unreadable") from exc
        if path in expected and digest != expected[path]:
            raise RecoveryError("recovery_input_drift", "Materialized input bytes have changed")
        artifacts.append(ArtifactDigest(path, digest))
    result = FrozenInputs(
        lineage_root,
        ArtifactDigest(case.staged_deck, case.deck_sha256),
        tuple(artifacts),
        materialized.source_lineage
        if materialized is not None
        else (case.recovery.inputs.source_lineage if case.recovery is not None else ()),
    )
    # Typed native preparation has its own control/seed contract. The electrical
    # closure is still checked below; only its generated setup is excluded.
    verify_case_inputs(
        result,
        native=native is not None,
        seeded=seeded,
        dialect=source.dialect,
        origins=origins,
    )
    return result


def _quoted_passive_value(card: SpiceCard, tokens: list[Token]) -> bool:
    """Recognize a single-quoted R/C primary value after two bare nodes."""
    if len(tokens) < 4 or tokens[0].text[:1].casefold() not in {"r", "c"}:
        return False
    if any(token.kind != TokenKind.BARE for token in tokens[:3]):
        return False
    value = tokens[3]
    if value.kind != TokenKind.QUOTED or len(value.text) < 3 or value.text[0] != "'":
        return False
    if (
        value.text[-1] != "'"
        or not value.text[1:-1].strip()
        or any(token.kind != TokenKind.KEY_VALUE for token in tokens[4:])
    ):
        return False
    instance = read_instance(card)
    return (
        instance is not None
        and len(instance.nodes) == 2
        and instance.model is None
        and instance.value == value.text
    )


@dataclass(frozen=True)
class _Construct:
    """A check a card fails: its refusal code and the construct it names.

    ``seedable`` marks random functions an ngspice seed admits (``simulator_seed``
    or the deck's own), on a deck that has none, so the refusal offers a seed
    only where it helps.
    """

    code: str
    what: str
    seedable: bool = False


@dataclass(frozen=True)
class _Refused:
    """One construct the frozen inputs cannot be recovered with, where it is."""

    code: str
    what: str
    path: Path
    line: int | None
    text: str
    seedable: bool = False


def _card_text(card: SpiceCard) -> str:
    text = " ".join(card.body.split()) or card.raw_lines[0].strip()
    return text if len(text) <= _CARD_TEXT_CHARS else text[: _CARD_TEXT_CHARS - 1] + "…"


def _refused_constructs(
    card: SpiceCard, functions: set[str], *, native: bool, seeded: bool
) -> _Construct | None:
    """The first check ``card`` fails, or None."""
    if card.kind == "control":
        return _Construct("recovery_control_unsupported", "a `.control` block")
    try:
        tokens = [
            token for token in tokenize_body(card.body) if token.kind != TokenKind.COMMENT_TRAIL
        ]
    except SpiceLexError as exc:
        # Listed in its place among the others, as the first card that failed
        # was refused before: an unreadable card hides no other construct.
        return _Construct(
            "recovery_syntax_unsupported", f"a card the server cannot read ({exc.message})"
        )
    if not tokens:
        return None
    first = tokens[0].text
    head = first.casefold()
    if head in _MODULE_DIRECTIVES:
        return _Construct("recovery_external_module", f"`{first}`, which loads an external module")
    if card.kind == "instance" and head.startswith("a"):
        return _Construct("recovery_external_module", f"`{first}`, an XSPICE device")
    if head.startswith(".") and head not in _DIRECTIVES:
        return _Construct(
            "recovery_directive_unsupported",
            f"`{first}`, a directive outside the supported set",
        )
    if head in INCLUDE_HEADS:
        return None
    text = " ".join(token.text for token in tokens)
    reader = re.search(r"(?i)\b(file|sfile|pwlfile)\s*=|\bpwl\s*\([^)]*[\"']", text)
    if reader:
        return _Construct("recovery_external_reader", f"`{reader.group(0)}`, which reads a file")
    if (
        card.kind == "instance"
        and any(token.kind == TokenKind.QUOTED for token in tokens)
        and not _quoted_passive_value(card, tokens)
    ):
        return _Construct(
            "recovery_external_reader",
            "a quoted value on an instance, which may name a file the server does not save",
        )
    calls = {match.casefold() for match in _FUNCTION.findall(text)}
    stochastic = calls.intersection(_RANDOM) or set(
        name.casefold() for name in re.findall(r"(?i)\b(trnoise|trrandom)\b", text)
    )
    admitted = _SEEDED_RANDOM if seeded else set()
    if stochastic - admitted and not native:
        return _Construct(
            "recovery_random_unsupported",
            f"{_names(stochastic - admitted)}, random"
            + (
                " in a way the seed does not set" if seeded else " with no seed the server records"
            ),
            seedable=not seeded and bool(stochastic & _SEEDED_RANDOM),
        )
    unknown = calls - _PURE_FUNCTIONS - functions - (_RANDOM if native else admitted)
    # A .model name precedes its parenthesized parameter block.
    if card.kind == "model" and len(tokens) >= 3:
        unknown.discard(tokens[2].text.casefold().split("(")[0])
    if unknown and not native:
        return _Construct(
            "recovery_expression_unsupported", f"{_names(unknown)}, not known to the server"
        )
    return None


def _names(names: set[str]) -> str:
    listed = ", ".join(f"`{name}`" for name in sorted(names))
    return f"function {listed}" if len(names) == 1 else f"functions {listed}"


def _validate_cards(
    cards: list[SpiceCard], functions: set[str], *, native: bool, seeded: bool
) -> list[tuple[SpiceCard, _Construct]]:
    """Every card that fails a check, with the first check it fails."""
    refused: list[tuple[SpiceCard, _Construct]] = []
    previous: SpiceCard | None = None
    for card in cards:
        if card.kind in {"comment", "blank"}:
            continue
        # Each line of a .control block is its own card: the block is named
        # once, at the line that opens it.
        opens = card.kind != "control" or previous is None or previous.kind != "control"
        previous = card
        found = _refused_constructs(card, functions, native=native, seeded=seeded)
        if found is not None and opens:
            refused.append((card, found))
    return refused


def _refusal(refused: list[_Refused], origins: Mapping[Path, str]) -> RecoveryError:
    """One refusal naming every construct refused, grouped by construct.

    The code is the first construct's: a card the lexer cannot classify, then
    a seeded deck's analyses, then a seeded deck's own seed options, then each
    file's cards in turn, the deck's first. The message lists each construct with its code, so a deck is fixed
    in one pass rather than one submission per construct.
    """
    groups: dict[tuple[str, str], dict[str, _Refused]] = {}
    for item in refused:
        where = origins.get(item.path, str(item.path))
        at = f"{where} line {item.line}" if item.line is not None else where
        # One place once: a library frozen both as shared and as a case's
        # own copy names the same line of the same file twice.
        groups.setdefault((item.code, item.what), {}).setdefault(at, item)
    lines: list[str] = []
    for (code, what), places in list(groups.items())[:_GROUPS_SHOWN]:
        at, first = next(iter(places.items()))
        line = f"- {what} ({code}): {at}: {first.text}"
        others = list(places)[1:]
        if others:
            shown = others[:_PLACES_SHOWN]
            more = len(others) - len(shown)
            line += "; also " + ", ".join(shown) + (f" and {more} more" if more else "")
        lines.append(line)
    hidden = len(groups) - _GROUPS_SHOWN
    if hidden > 0:
        lines.append(
            f"- and {hidden} more construct{'s' if hidden != 1 else ''}, listed when "
            "these are fixed and the request is sent again"
        )
    count = len(groups)
    seedable = any(item.seedable for item in refused)
    message = (
        f"Recoverable execution refuses {count} construct{'s' if count != 1 else ''} "
        "in this deck and the files it includes, because a resumed case could not "
        "be relied on to run them the same way:\n"
        + "\n".join(lines)
        + "\nChange or remove each, or submit without execution.recoverable"
        + (
            "; on ngspice, execution.simulator_seed, or a seed the deck fixes itself with "
            "`.options seed=<n>` at its top level, admits agauss, gauss, aunif, unif and limit"
            if seedable
            else ""
        )
        + "."
    )
    return RecoveryError(refused[0].code, message)


def _native_setup(inputs: FrozenInputs, artifact: ArtifactDigest, content: bytes) -> bool:
    """Recognize the exact existing seeded adapter, never arbitrary control text."""
    suffix = ".setup.cir"
    if not artifact.path.name.endswith(suffix) or artifact.path.parent != inputs.lineage_root:
        return False
    token = artifact.path.name.removesuffix(suffix)
    seed = re.search(rb"\nsetseed ([0-9]+)\n", content)
    if seed is None:
        return False
    root = inputs.lineage_root
    paths = NativePaths(
        root,
        root / f"{token}.input.cir",
        artifact.path,
        root / f"{token}.cir",
        root / f"{token}.raw",
        root / f"{token}.log",
    )
    if not any(item.path == paths.electrical_input for item in inputs.files):
        return False
    return content == driver_bytes(int(seed.group(1)), paths, token)


def _verify_inputs(
    inputs: FrozenInputs,
    *,
    native: bool,
    seeded: bool,
    dialect: str | None,
    origins: Mapping[Path, str],
) -> None:
    inventory = {item.path.resolve(): item for item in inputs.files}
    contents: dict[str, bytes] = {}
    native_setups: set[Path] = set()
    for artifact in inputs.files:
        _contained(artifact.path, inputs.lineage_root)
        content = artifact.path.read_bytes()
        if hashlib.sha256(content).hexdigest() != artifact.sha256:
            raise RecoveryError("recovery_input_drift", "Exact frozen input bytes have changed")
        contents.setdefault(artifact.sha256, content)
        if native and _native_setup(inputs, artifact, content):
            native_setups.add(artifact.path)
    roots = {inputs.electrical.path}
    roots.update(
        path.with_name(path.name.removesuffix(".setup.cir") + ".input.cir")
        for path in native_setups
    )
    # Share syntax only within this verification. Every physical file above
    # still proves its bytes, and every include below resolves from its own path.
    file_keys: dict[Path, tuple[str, bool]] = {}
    cards_by_content: dict[tuple[str, bool], EditCards] = {}
    # Every construct the inputs cannot be recovered with, so one refusal names
    # them all (``_refusal`` says in which order).
    refused: list[_Refused] = []
    for artifact in inputs.files:
        path = artifact.path
        if path in native_setups:
            continue
        key = (artifact.sha256, path in roots)
        file_keys[path] = key
        if key in cards_by_content:
            continue
        content = contents[artifact.sha256]
        # A card after a file's end marker is not read, nor a root deck's
        # line 1, its title (lex_for_edit). Every card is still classified, so
        # the warnings and the constructs of an unread card refuse as well.
        parsed = lex_for_edit(decode_spice_bytes(content), deck=path in roots)
        for warning in parsed.warnings:
            found = _WARNING_LINE.match(warning)
            refused.append(
                _Refused(
                    "recovery_syntax_unsupported",
                    "a card the server cannot classify",
                    path,
                    int(found.group(1)) if found else None,
                    warning[found.end() :].strip() if found else warning,
                )
            )
        cards_by_content[key] = parsed
    functions = {
        match.group(1).casefold()
        for parsed in cards_by_content.values()
        for card in parsed.cards
        if card.body.lstrip().casefold().startswith(".func ")
        if (match := _FUNCTION.search(card.body)) is not None
    }
    if seeded:
        # The audited driver writes the current plot. Multiple analyses and
        # noise can create additional plots that an ordinary batch run retains.
        # An analysis is one the simulator reads, as the run's requirements
        # count them (runner_base.deck_requirements).
        analyses = [
            head
            for key in file_keys.values()
            for card in cards_by_content[key].read
            if card.kind == "directive"
            if (head := card.body.split(maxsplit=1)[0].casefold())
            in {".op", ".ac", ".dc", ".tran", ".noise", ".step"}
        ]
        if len(analyses) != 1 or analyses[0] not in {".op", ".ac", ".dc", ".tran"}:
            refused.append(
                _Refused(
                    "recovery_seed_analysis_unsupported",
                    "a seeded ngspice deck needs exactly one .op, .ac, .dc or .tran analysis",
                    inputs.electrical.path,
                    None,
                    f"it has {', '.join(analyses) or 'none'}",
                )
            )
    # Which seed ngspice draws the deck's static random functions from
    # (ngspice_driver.seed_option). ngspice reads a seed option when it loads
    # the deck, after the driver's setseed, so the deck's own seed wins and a
    # seeded run of it would record a seed ngspice never drew from (measured
    # on ngspice-42).
    first_path: dict[tuple[str, bool], Path] = {}
    for path, key in file_keys.items():
        first_path.setdefault(key, path)
    keys = list(first_path)
    seed_cards = deck_seeds([cards_by_content[key] for key in keys])
    seeds = [value for _, _, value in seed_cards]
    if seeded:
        refused.extend(
            _Refused(
                "recovery_seed_unsupported",
                "a seed option, which ngspice reads after execution.simulator_seed "
                "and so overrides it",
                first_path[keys[index]],
                card.line_start,
                _card_text(card),
            )
            for index, card, _ in seed_cards
        )
    # A deck that fixes ngspice's seed itself draws alike on every run, as the
    # driver's setseed does. What another simulator makes of the card is not
    # recorded, so only an ngspice deck counts.
    seeded_itself = dialect == "ngspice" and _deck_seeds_itself(
        inputs.electrical.path, seeds, file_keys, cards_by_content
    )
    validated: set[tuple[str, bool]] = set()
    for path, key in file_keys.items():
        if key not in validated:
            refused.extend(
                _Refused(
                    found.code, found.what, path, card.line_start, _card_text(card), found.seedable
                )
                for card, found in _validate_cards(
                    cards_by_content[key].cards,
                    functions,
                    native=native,
                    seeded=seeded or seeded_itself,
                )
            )
            validated.add(key)
    if refused:
        raise _refusal(refused, origins)
    for path, key in file_keys.items():
        parsed = cards_by_content[key]
        try:
            # The includes staging followed (deck_staging.stage_deck): none in
            # a root deck's title or after a file's end marker.
            targets = staged_card_reference_targets(
                parsed.read, path, depth=0 if path == inputs.electrical.path else 1
            )
        except (ValueError, OSError) as exc:
            raise RecoveryError(
                "recovery_closure_incomplete", "Cannot resolve frozen input closure"
            ) from exc
        if any(target not in inventory for target in targets):
            raise RecoveryError(
                "recovery_closure_incomplete",
                "Referenced input is absent from the frozen manifest",
            )


def _deck_seeds_itself(
    root: Path,
    seeds: list[str],
    file_keys: dict[Path, tuple[str, bool]],
    cards_by_content: dict[tuple[str, bool], EditCards],
) -> bool:
    """Whether the frozen deck fixes ngspice's seed with an option card of its
    own, given every seed value (``seeds``) its closure reads.

    ngspice reads the first seed option in the deck, so every one the closure
    reads must be fixed, and one at the top level of the root deck, outside a
    subcircuit and a ``.lib`` section, makes sure ngspice reads one at all: it
    drops a subcircuit no instance calls, seed and all (measured on
    ngspice-42), and skips a ``.lib`` section the deck does not select.
    """
    if not seeds or not all(fixed_seed(value) for value in seeds):
        return False
    key = file_keys.get(root)
    if key is None:
        return False
    parsed = cards_by_content[key]
    return any(
        parsed.reads(card) and card.scope == () and section is None and seed_option(card)
        for card, section in zip(parsed.cards, card_sections(parsed.cards, root), strict=True)
    )


def verify_case_inputs(
    inputs: FrozenInputs,
    *,
    native: bool = False,
    seeded: bool = False,
    dialect: str | None = None,
    origins: Mapping[Path, str] | None = None,
) -> None:
    """Refuse byte drift, escapes and any uncaptured staged include edge.

    ``dialect`` is the simulator family that runs the deck. Only an ngspice
    deck that fixes its own seed may use static random functions without the
    seeded driver, so a caller that leaves it out gets the stricter answer.
    ``origins`` names a frozen file as the caller knows it (the file it was
    copied from) in a refusal; a file it leaves out is named by its frozen path.
    """
    if type(seeded) is not bool or type(native) is not bool or (seeded and native):
        raise RecoveryError(
            "recovery_seed_unsupported", "Seeded and native contracts are distinct"
        )
    try:
        _verify_inputs(
            inputs, native=native, seeded=seeded, dialect=dialect, origins=origins or {}
        )
    except OSError as exc:
        raise RecoveryError("recovery_artifact_missing", "Frozen input is unreadable") from exc
    except ValueError as exc:
        raise RecoveryError(
            "recovery_syntax_unsupported", "Frozen input cannot be classified"
        ) from exc


#: The hosts each family's controlled startup is verified on: LTspice's
#: captured ``LTspice.ini`` needs native Windows, ngspice's immutable spinit is
#: verified on Linux (WSL included) and Windows. A family absent here has no
#: policy. The capability table (``capabilities``) declares the same, and a
#: test holds the two to each other.
def _startup_unavailable(family: str, host: str) -> bool:
    """Whether the capability table declares no controlled startup for
    ``family`` on ``host``."""
    found = capabilities.entry(family, "controlled_startup")
    return found is None or found.status == "gap" or found.unavailable_on(host) is not None


def prepare_startup(
    store: Store,
    root_job_id: str,
    simulator: type | None = None,
    *,
    ini_source: Path | None = None,
) -> StartupPolicy:
    """Capture simulator startup without changing the user's settings or environment."""
    from spicelib.simulators.ltspice_simulator import LTspice
    from spicelib.simulators.ngspice_simulator import NGspiceSimulator

    from ltspice_mcp.lib.simulator import platform_key

    host = platform_key()
    if simulator is not None and issubclass(simulator, LTspice):
        from ltspice_mcp.lib.controlled_ltspice import STARTUP_VERSION as LTSPICE_STARTUP_VERSION
        from ltspice_mcp.lib.controlled_ltspice import capture_ini_template

        if _startup_unavailable("ltspice", host):
            raise RecoveryError(
                "recovery_startup_unsupported",
                "Controlled LTspice startup requires native Windows",
            )
        if ini_source is None:
            appdata = os.environ.get("APPDATA")
            if not appdata:
                raise RecoveryError(
                    "recovery_startup_unsupported",
                    "Set simulator.ltspice_ini to established settings",
                )
            ini_source = Path(appdata) / "LTspice.ini"
        elif not ini_source.is_absolute():
            ini_source = store.working_dir / ini_source
        root = store.lineage_run_dir(root_job_id, store.run_dir(root_job_id, simulator), simulator)
        template = capture_ini_template(
            ini_source, store.recovery_ini_template(root_job_id, simulator), root
        )
        return StartupPolicy(LTSPICE_STARTUP_VERSION, False, ini_template=template)
    if simulator is not None and not issubclass(simulator, NGspiceSimulator):
        raise RecoveryError(
            "recovery_startup_unsupported", "Controlled spinit policy requires ngspice"
        )
    if _startup_unavailable("ngspice", host):
        raise RecoveryError(
            "recovery_startup_unsupported",
            "Controlled system startup is not verified on this platform",
        )
    path = store.recovery_spinit(root_job_id, simulator)
    try:
        root = store.lineage_run_dir(root_job_id, store.run_dir(root_job_id, simulator), simulator)
        if not path.parent.resolve().is_relative_to(root):
            raise RecoveryError("recovery_path_escape", "Startup directory is outside its lineage")
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_bytes(path, _SPINIT, overwrite=False, durable=True)
    except OSError as exc:
        raise RecoveryError(
            "recovery_startup_io", "Cannot prepare immutable controlled startup"
        ) from exc
    return StartupPolicy(
        STARTUP_VERSION,
        True,
        ArtifactDigest(path, hashlib.sha256(_SPINIT).hexdigest()),
        (("SPICE_SCRIPTS", str(path.parent)),),
    )


def verify_startup(policy: StartupPolicy, lineage_root: Path) -> None:
    """Bind system startup bytes, user-init suppression and the local env override."""
    if policy.ini_template is not None:
        from ltspice_mcp.lib.controlled_ltspice import STARTUP_VERSION as LTSPICE_STARTUP_VERSION
        from ltspice_mcp.lib.controlled_ltspice import verify_ini_template

        if (
            policy.version != LTSPICE_STARTUP_VERSION
            or policy.user_init_disabled
            or policy.spinit is not None
            or policy.environment
        ):
            raise RecoveryError(
                "recovery_startup_drift", "Controlled LTspice policy is incompatible"
            )
        verify_ini_template(policy.ini_template, lineage_root)
        return
    if (
        policy.version != STARTUP_VERSION
        or not policy.user_init_disabled
        or policy.spinit is None
        or policy.environment != (("SPICE_SCRIPTS", str(policy.spinit.path.parent)),)
        or policy.spinit.sha256 != hashlib.sha256(_SPINIT).hexdigest()
    ):
        raise RecoveryError("recovery_startup_drift", "Controlled startup policy is incompatible")
    _verify(policy.spinit, lineage_root, code="recovery_startup_drift")


def prepare_case_startup(
    case: ExperimentCase,
    execution: ExecutionRecord,
    store: Store,
    root_job_id: str,
    simulator: type,
) -> None:
    """Give each LTspice attempt its own settings copy before the job is claimed."""
    template = execution.startup.ini_template
    if template is None:
        return
    from ltspice_mcp.lib.controlled_ltspice import prepare_attempt_ini

    if case.recovery is None:
        raise RecoveryError("recovery_record_invalid", "Startup copy requires frozen inputs")
    prepare_attempt_ini(
        template,
        store.recovery_ini(root_job_id, case.run_token, simulator),
        case.recovery.inputs.lineage_root,
    )


def capture_produced_artifacts(raw: Path, log: Path, *, lineage_root: Path) -> ProducedArtifacts:
    """Snapshot bytes after the coordinator established completed solve evidence."""
    artifacts: list[ArtifactDigest] = []
    for path in (raw, log):
        _contained(path, lineage_root)
        try:
            artifacts.append(ArtifactDigest(path, sha256_file(path)))
        except OSError as exc:
            raise RecoveryError(
                "recovery_artifact_missing", "Produced artifact is unreadable"
            ) from exc
    return ProducedArtifacts(artifacts[0], artifacts[1], now())


def verify_produced_artifacts(outputs: ProducedArtifacts, lineage_root: Path) -> None:
    """Missing/corrupt success data is unavailable, never an implicit retry."""
    for artifact in (outputs.raw, outputs.log):
        _verify(artifact, lineage_root, code="recovery_output_drift")
