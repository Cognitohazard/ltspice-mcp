"""Snapshot SPICE decks and their local dependency graph for experiments."""

from __future__ import annotations

import hashlib
import os
import re
from collections.abc import Callable, Collection, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PureWindowsPath
from typing import Any

from ltspice_mcp.errors import PathSecurityError
from ltspice_mcp.lib import atomic_write_bytes, wsl
from ltspice_mcp.lib.encoding import (
    decode_spice_bytes_with_encoding,
    encode_spice_text,
    rewrite_codec,
)
from ltspice_mcp.lib.experiment_types import ManifestEntry
from ltspice_mcp.lib.ngspice_compat import reads_lib_as_includes, rejects_lib
from ltspice_mcp.lib.pathutil import resolve_safe_path
from ltspice_mcp.lib.spice_lex import (
    INCLUDE_HEADS,
    SpiceCard,
    SpiceLexError,
    Token,
    TokenKind,
    directive_head,
    emit,
    lex_for_edit,
    tokenize_body,
    unreadable_cards,
)
from ltspice_mcp.lib.spice_lex_ops import (
    MICRO_SIGN_READERS,
    ValueSuffixSite,
    fold_micro_suffix_cards,
)
from ltspice_mcp.lib.spice_lex_views import WAVEFILE
from ltspice_mcp.lib.stored_values import (
    CP1252_BUILD,
    SHEET_UNREAD,
    OtherReading,
    SessionBuilds,
    readings_of_cards,
    readings_of_sheet,
    summaries,
)

# Sized for real foundry PDKs, which fan out further than a hand-written deck:
# sky130 reaches a device model five levels down (deck -> sky130.lib.spice ->
# corners/<corner>.spice -> <device>__<corner>.corner.spice -> <device>.pm3.spice),
# and gf180 nests comparably. Cycles are caught separately by the in-progress
# set, so this bound is a resource guard, not the loop guard.
DEFAULT_INCLUDE_DEPTH = 8

_WINDOWS_DRIVE_RE = re.compile(r"^[A-Za-z]:[\\/]")

# The two ways out of a root-escape refusal, named in the refusal itself: a
# caller who cannot see the sandbox boundary cannot guess either one, and the
# option is not called what the message used to imply.
_ROOT_ESCAPE_REMEDY = (
    "Either add its directory to [security] allowed_paths "
    "(LTSPICE_MCP_ALLOWED_PATHS) so it is snapshotted with the deck, or pass "
    "allow_live_includes=true to read it in place — a live include is read at "
    "run time and its content is not covered by the deck's provenance hash."
)


class DeckStagingError(ValueError):
    """A deck snapshot could not satisfy staged-only provenance."""

    def __init__(self, code: str, message: str, *, reference: str | None = None):
        self.code = code
        self.reference = reference
        super().__init__(message)


@dataclass(frozen=True)
class ExperimentPaths:
    """Platform-routed staging and output locations."""

    staging_root: Path
    output_folder: Path
    # True when these paths live on the Windows filesystem for a Windows
    # simulator driven across the WSL boundary — that simulator needs any
    # absolute path written into a deck spelled the Windows way.
    windows_native: bool = False


@dataclass(frozen=True)
class StagedFile:
    """One staged file in a deck's include closure."""

    source: Path
    staged_path: Path
    text: str
    sha256: str = ""
    # The codec a copy written from ``text`` is encoded in
    # (``encoding.rewrite_codec``), so it keeps every byte it did not change.
    codec: str = "utf-8"


@dataclass
class StagedDeck:
    """One staged primary deck plus its dependency manifest."""

    source_path: Path
    staged_deck: Path
    text: str
    sha256: str
    manifest: list[ManifestEntry]
    # Digest of the file the author edits: the ``origin`` when the primary deck
    # was generated from one (a schematic exported to a netlist), else the
    # primary deck itself. This is the digest that describes the path a caller
    # named, which is a different file whenever an export sits in between.
    origin_sha256: str = ""
    observations: list[dict[str, Any]] = field(default_factory=list)
    # Every staged file EXCEPT the primary deck, in discovery order. The
    # manifest records where each dependency came from; this records what the
    # staged copy says, which is what a variation has to read to find a
    # component that the root deck only reaches through an include.
    includes: list[StagedFile] = field(default_factory=list)
    # Original bytes from the same reads that established the manifest hashes.
    # Profile validation must not re-read possibly edited authoring files or
    # mistake rewritten include paths for changes to the original model.
    source_contents: dict[Path, bytes] = field(default_factory=dict)
    # The primary deck's codec, as ``StagedFile.codec``.
    codec: str = "utf-8"


@dataclass(frozen=True)
class IncludeReference:
    """One quote-aware include or library reference in a parsed deck.

    ``token`` spans the path as written: for LTspice, which reads the rest of
    a ``.lib`` line as one path, every word after ``.lib``. ``head`` is the
    card's head, folded (``.lib``, ``.include``). ``first_word`` is, for such
    an LTspice path of more than one word, its first word when that alone
    names a file: what a reader that splits off a section would have read.
    """

    card: SpiceCard
    token: Token
    raw_path: str
    section: str | None
    head: str = ".include"
    first_word: str | None = None


def _micro_fold_observation(
    path: Path, folded: Sequence[ValueSuffixSite], reader: str
) -> dict[str, Any]:
    """The fact that a staged copy spells a source's micro-sign suffixes ``u``.

    Reported only for a simulator that decodes decks as cp1252 (``reader``),
    the one reader for which the staged copy and its source mean different
    things.
    """
    return {
        "code": "micro_sign_folded",
        "kind": "provenance",
        "detail": (
            f"The staged copy of {path.name} spells {len(folded)} micro-sign "
            f"suffix(es) as 'u'. {MICRO_SIGN_READERS} This job's simulator decodes "
            "decks as cp1252, so there the source and its staged copy can differ."
        ),
        "evidence": {
            "file": str(path),
            "tokens": [site.token for site in folded],
            "lines": [site.line for site in folded],
            "reader": reader,
        },
    }


def _read_otherwise_observations(
    readings: Sequence[OtherReading], builds: SessionBuilds | None
) -> list[dict[str, Any]]:
    """The values of the run's source files that an LTspice build the session
    runs reads as another number than the run uses (``stored_values``): one
    observation per build and kind of reading, over every staged file. The run reads each value
    as the server does, whatever the simulator, but the file opened in that
    build is another circuit, and a number drawn or checked from it here (a
    Monte Carlo nominal, a device's size) is not the one that build reads."""
    return [
        {
            "code": summary["code"],
            "kind": "provenance",
            "detail": summary["detail"],
            "evidence": {
                key: value for key, value in summary.items() if key not in ("code", "detail")
            },
        }
        for summary in summaries(readings, builds or {})
    ]


def _sheet_unread_observation(sheet: Path) -> dict[str, Any]:
    """That a sheet run as the server's export could not be read, so how a
    build reads its own values was not checked."""
    return {
        "code": "value_read_otherwise",  # stored_values.CODE
        "kind": "provenance",
        "detail": f"{sheet.name}: {SHEET_UNREAD}.",
        "evidence": {"file": str(sheet), "not_checked": SHEET_UNREAD},
    }


def sha256_file(path: Path) -> str:
    """Return a file's SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_experiment_paths(
    working_dir: Path,
    job_id: str,
    circuit_id: str,
    simulator: type,
) -> ExperimentPaths:
    """Ask the store where this simulator's staged decks and output may go.

    The routing rule itself (WSL + LTspice needs a Windows-native tree, because
    the SQLite ``.db`` behind ``.MEAS`` cannot be written over the UNC share)
    lives in ``Store.artifact_base`` so every writer applies one decision. This
    wrapper exists to translate its refusal into the staging error code callers
    already handle.
    """
    from ltspice_mcp.lib.store import Store, WindowsNativeStorageUnavailable

    store = Store(working_dir)
    try:
        base = store.artifact_base(simulator)
    except WindowsNativeStorageUnavailable as exc:
        raise DeckStagingError("windows_native_storage_unavailable", str(exc)) from exc
    return ExperimentPaths(
        # Staged decks live inside the job's own run directory: they are part of
        # what that run produced, they must sit on the same filesystem as the
        # output for the WSL case, and grouping them there is what makes a job's
        # files enumerable as a set. The Store names it, so this is not a second
        # copy of the layout that can drift from the one that reads it back.
        staging_root=store.staged_deck_root(job_id, circuit_id, simulator),
        output_folder=base.runs,
        windows_native=base.windows_native,
    )


def stage_deck(
    source_path: Path,
    staging_root: Path,
    allowed_roots: list[Path] | None,
    *,
    origin: Path,
    dialect: str | None,
    allow_live_includes: bool = False,
    max_depth: int = DEFAULT_INCLUDE_DEPTH,
    windows_paths: bool = False,
    simulator_roots: Sequence[Path] = (),
    compact_digests: Collection[str] = (),
    cp1252_reader: str | None = None,
    stated_builds: SessionBuilds | None = None,
) -> StagedDeck:
    """Copy a primary deck, its include/lib closure and the data files it
    reads into ``staging_root``.

    ``compact_digests`` identifies audited model contents whose only external
    references use the include/lib grammar below. Those dependencies receive
    short filenames; all other sources retain their original relative layout.

    ``allowed_roots`` is the sandbox, or None when there is none. With a
    sandbox, the deck and every file it reaches must lie in one of its roots
    (or, for a reached file, in ``simulator_roots``). With none, the deck's own
    folder is the first root, and a reached file outside every root so far
    adds its own folder as another, so every file that exists is staged.

    Relative dependencies retain their topology within their root. An
    absolute or cross-root reference is rewritten only in the staged copy so
    the simulator still consumes the snapshot. The authoring files are never
    modified.

    ``origin`` is the file the primary deck was generated FROM — an ``.asc``
    schematic exported to the ``.net`` handed here, or the deck itself when it
    was hand-written. It is snapshotted and digested like any other source even
    though no simulator reads it, because it is the file the author edits:
    without it the manifest records only the export, and a caller who changes
    the schematic changes nothing this deck's provenance can see. Required
    rather than defaulted, so a new caller has to answer the question instead
    of inheriting an answer that silently records the wrong file.

    A primary deck generated from ``origin`` is staged as if it stood where the
    netlister wrote it, beside ``origin``, wherever it is kept (the server keeps
    exports in its store, ``Store.exports_dir``): it is admitted because
    ``origin`` is, so the store need not sit inside the allowed roots, and its
    relative references resolve against ``origin``'s directory, which is the
    directory the schematic's own ``.include`` lines name files in.

    ``dialect`` is the simulator that reads the deck (``simulator_dialect``),
    which decides the data files it names (``scan_data_references``). Each is
    resolved against the deck's folder, where both simulators look for one
    (``relative_data_path_established``), and refused when it does not exist
    or lies outside ``allowed_roots``. A data file named by a relative path is
    copied, as bytes, and the card is rewritten to name the copy by its
    absolute path, wherever the card is: the run starts from a copy of the
    deck somewhere else. One named by an absolute path is read where it is, a
    live manifest entry, and spelled the Windows way when ``windows_paths``
    is set and it was not.

    ``windows_paths`` renders the root deck's rewritten references in Windows
    form, for a Windows simulator reached across the WSL boundary: it cannot
    open the ``/mnt/c/...`` spelling of the very file it is being handed.

    ``simulator_roots`` are the detected simulator's own library directories
    (``simulator.simulator_library_roots``). They are appended AFTER the
    caller's roots, so a reference inside one is staged and hashed like any
    other dependency while every existing root keeps its index — but the deck
    itself must still live in an allowed root, so this cannot be used to reach
    a deck the sandbox denies.

    Every staged copy spells a micro-sign scale suffix ``u``, and a copy that
    had to be rewritten is written in its source's encoding, so the characters
    it did not rewrite keep their bytes. ``cp1252_reader`` names the target
    simulator when it decodes decks as cp1252 (``services.cp1252_ltspice``):
    only then, and only for a file whose signs that build reads otherwise
    (``stored_values``: UTF-8, not cp1252 or UTF-16), does the fold change what
    a value means to the simulator, and only then is it reported, as a
    ``micro_sign_folded`` observation. ``stated_builds`` are the LTspice
    builds the session runs (``services.session_ltspice_builds``): a value of
    a source that one of them reads as another number is reported as a
    ``value_read_otherwise`` observation.
    """
    if max_depth < 0:
        raise ValueError("max_depth must be non-negative")
    source = source_path.resolve(strict=True)
    authored = origin.resolve(strict=True)
    exported = source != authored
    # Where the primary deck stands for the sandbox, for its place in the
    # staging tree, and for its relative references.
    stands_at = authored.parent / source.name if exported else source
    confined = allowed_roots is not None
    allowed = _resolved_roots(allowed_roots) if confined else [stands_at.parent]
    roots = allowed + _resolved_roots(simulator_roots)
    # Authored files are checked against ``allowed`` alone — a prefix of
    # ``roots``, so the index means the same thing in both — which is what
    # keeps the simulator's library a place references may POINT, never a
    # place a deck may be RUN FROM.
    root_index = _containing_root(stands_at, allowed)
    if root_index is None:
        outside = authored if exported else source
        raise DeckStagingError(
            "include_unstaged",
            f"Primary deck {outside} is outside the configured allowed roots",
            reference=str(outside),
        )

    def _render_absolute(path: Path) -> str:
        if windows_paths:
            return wsl.to_windows_path(path)
        # The platform's own spelling, as the netlister writes it: POSIX off
        # Windows, backslashes on it.
        return str(path)

    staging_root.mkdir(parents=True, exist_ok=True)
    manifest: list[ManifestEntry] = []
    manifest_keys: set[tuple[Path, str | None]] = set()
    source_destinations: dict[Path, Path] = {}
    destination_owners: dict[str, Path] = {}
    compact_digests = frozenset(compact_digests)
    source_bytes: dict[Path, bytes] = {}
    source_digests: dict[Path, str] = {}
    staged_texts: dict[Path, str] = {}
    staged_digests: dict[Path, str] = {}
    staged_codecs: dict[Path, str] = {}
    processed_depths: dict[Path, int] = {}
    processing: set[Path] = set()
    observations: list[dict[str, Any]] = []
    # The values of the staged files another build reads otherwise.
    stated: list[OtherReading] = []
    # The data files read in place, each reported once.
    live_data: set[Path] = set()
    # The data references rewritten to an absolute path, for the LTspice note.
    absolute_data: list[str] = []

    primary_destination = _destination_for(stands_at, staging_root, roots, root_index)

    def add_manifest(entry: ManifestEntry) -> None:
        key = (entry.path, entry.section)
        if key in manifest_keys:
            return
        manifest_keys.add(key)
        manifest.append(entry)

    def snapshot_file(path: Path, destination: Path, depth: int, *, walk: bool = True) -> Path:
        """Copy one source into the staging tree and record it in the manifest.

        ``walk=False`` stages a file whose references are not ours to follow —
        the authoring origin, which is not SPICE. It still goes through here so
        it lands in the same bookkeeping: a path staged twice is copied once,
        and the copy the walk wrote is never overwritten by a plain one.
        """
        resolved = path.resolve(strict=True)
        prior = source_destinations.get(resolved)
        if prior is None:
            data = resolved.read_bytes()
            source_bytes[resolved] = data
            source_digest = hashlib.sha256(data).hexdigest()
            source_digests[resolved] = source_digest
            if walk and depth > 0 and source_digest in compact_digests:
                destination = _compact_destination(resolved, staging_root)
            owner = destination_owners.setdefault(str(destination).casefold(), resolved)
            if owner != resolved:
                raise DeckStagingError(
                    "staging_collision", "Two captured sources resolve to the same staged filename"
                )
            source_destinations[resolved] = destination
            add_manifest(
                ManifestEntry(
                    path=resolved,
                    sha256=source_digest,
                    staged=True,
                    live=False,
                    staged_path=destination,
                )
            )
        else:
            destination = prior
            data = source_bytes[resolved]
        if not walk:
            if prior is None:
                destination.parent.mkdir(parents=True, exist_ok=True)
                atomic_write_bytes(destination, data, durable=True)
            return destination
        if resolved in processing:
            return destination
        prior_depth = processed_depths.get(resolved)
        if prior_depth is not None and prior_depth <= depth:
            return destination
        processing.add(resolved)
        try:
            text, encoding = decode_spice_bytes_with_encoding(data)
            # The deck's references and values are the cards LTspice reads: an
            # include named in its title or after its .END is not staged, and
            # those cards are copied as they are.
            parsed = lex_for_edit(text, deck=depth == 0)
            changed = False
            anchor = stands_at if resolved == source else resolved
            for card, error in unreadable_references(parsed.read):
                raise DeckStagingError(
                    "include_unstaged",
                    f"Cannot stage the reference on line {card.line_start} of "
                    f"{resolved}: {error.message}, so which file it names is not "
                    f"known. {error.suggestion[:1].upper()}{error.suggestion[1:]}.",
                    reference=" ".join(card.body.split()),
                )
            for reference in scan_include_references(
                parsed.read, anchor, family=dialect, depth=depth
            ):
                target = resolve_reference(anchor.parent, reference.raw_path)
                target_resolved = resolve_existing(target)
                target_root = None
                if target_resolved is not None:
                    target_root = _containing_root(target_resolved, roots)
                    if target_root is None and not confined:
                        roots.append(target_resolved.parent)
                        target_root = len(roots) - 1
                reason = _unstaged_reason(
                    target,
                    target_resolved,
                    target_root,
                    depth=depth,
                    max_depth=max_depth,
                )
                if target_resolved is None and dialect == "ltspice":
                    # What LTspice reads instead, and why it is nothing.
                    reason = (
                        library_read(reference, None, LibraryReader("ltspice")).unread or reason
                    )
                if reason is not None:
                    if not allow_live_includes:
                        # Only the root escape has these two remedies; a
                        # missing file or an over-deep chain is not fixed by
                        # either, and naming them there would misdirect.
                        escaped = target_resolved is not None and target_root is None
                        raise DeckStagingError(
                            "include_unstaged",
                            f"Cannot stage reference {reference.raw_path!r} from "
                            f"{resolved}: {reason}"
                            + (f". {_ROOT_ESCAPE_REMEDY}" if escaped else ""),
                            reference=reference.raw_path,
                        )
                    live_path = target_resolved or target
                    add_manifest(
                        ManifestEntry(
                            path=live_path,
                            sha256="",
                            staged=False,
                            live=True,
                            reason=reason,
                            section=reference.section,
                        )
                    )
                    _replace_reference(reference, str(live_path))
                    changed = True
                    observations.append(
                        {
                            "code": "live_include",
                            "kind": "provenance",
                            "detail": (
                                f"Reference {reference.raw_path!r} will be read live; "
                                "its content is not covered by the staged snapshot hash."
                            ),
                            "evidence": {
                                "file": str(resolved),
                                "reference": reference.raw_path,
                                "reason": reason,
                            },
                        }
                    )
                    continue

                assert target_resolved is not None
                assert target_root is not None
                target_destination = _destination_for(
                    target_resolved,
                    staging_root,
                    roots,
                    target_root,
                )
                actual_destination = snapshot_file(
                    target_resolved,
                    target_destination,
                    depth + 1,
                )
                add_manifest(
                    ManifestEntry(
                        path=target_resolved,
                        sha256=source_digests[target_resolved],
                        staged=True,
                        live=False,
                        staged_path=actual_destination,
                        section=reference.section,
                    )
                )
                if depth == 0:
                    # The root deck is what gets handed to the simulator, which
                    # runs it from its own output folder — the deck moves and
                    # its siblings do not. EVERY reference it carries must
                    # therefore be absolute, including a plain same-directory
                    # ".include core.inc": that one needs no rewrite to be
                    # correct in the staging tree, which is exactly why it used
                    # to survive staging and then die at run time.
                    _replace_reference(reference, _render_absolute(actual_destination))
                    changed = True
                else:
                    # Deeper files never move relative to each other, so they
                    # stay relative and the bundle stays relocatable.
                    expected = (
                        destination.parent / _portable_relative(reference.raw_path)
                    ).resolve()
                    if (
                        is_absolute_reference(reference.raw_path)
                        or expected != actual_destination.resolve()
                        # An LTspice path of more than one word is written in
                        # quotes, so every reader of the staged copy reads
                        # one path (``scan_include_references``).
                        or _unquoted_words(reference)
                    ):
                        _replace_reference(
                            reference,
                            Path(
                                os.path.relpath(actual_destination, destination.parent)
                            ).as_posix(),
                        )
                        changed = True

            rewrites = [
                (named, rewritten)
                for named in scan_data_references(parsed.read, dialect)
                if (rewritten := stage_data_file(named, resolved, depth)) is not None
            ]
            # Last first, so a rewrite never moves a span still to be rewritten
            # on the same card.
            for named, rewritten in sorted(rewrites, key=lambda item: -item[0].start):
                try:
                    named.card.replace_span(named.start, named.end, rewritten)
                except SpiceLexError as exc:
                    raise DeckStagingError(
                        "data_file_unstaged",
                        f"Cannot name the staged copy of {named.keyword}{named.raw_path} on "
                        f"line {named.card.line_start} of {resolved}: {exc.message}. Write "
                        "the path on one line of the card.",
                        reference=named.raw_path,
                    ) from exc
                changed = True

            destination.parent.mkdir(parents=True, exist_ok=True)
            # Every staged SPICE file is a simulator input. A micro-sign suffix
            # is spelled 'u', which every simulator in every encoding reads as
            # micro: a UTF-8 µ reaches LTspice XVII as the two cp1252
            # characters 'Âµ' and silently loses its scale.
            # Read before the fold, which spells the signs the readings name.
            # A sheet's export is the server's file, not one the caller stored:
            # how a build reads its bytes says nothing of how it reads the sheet.
            readings = readings_of_cards(parsed.read, encoding, file=resolved)
            if exported and resolved == source:
                # A sheet's export is the server's file, not one the caller
                # stored: the sheet's own bytes are what a build reads.
                of_sheet = readings_of_sheet(parsed.read, authored)
                if of_sheet is None:
                    observations.append(_sheet_unread_observation(authored))
                else:
                    stated.extend(of_sheet)
            else:
                stated.extend(readings)
            folded = fold_micro_suffix_cards(parsed.read)
            if folded:
                changed = True
                # The fold changes what a value means only where the cp1252
                # reader reads the file's sign otherwise (stored_values).
                if cp1252_reader is not None and any(
                    reading.build == CP1252_BUILD for reading in readings
                ):
                    observations.append(_micro_fold_observation(resolved, folded, cp1252_reader))
            codec = rewrite_codec(encoding)
            staged_text = emit(parsed.cards) if changed else text
            staged_bytes = encode_spice_text(staged_text, codec) if changed else data
            atomic_write_bytes(destination, staged_bytes, durable=True)
            staged_texts[resolved] = staged_text
            staged_digests[resolved] = hashlib.sha256(staged_bytes).hexdigest()
            staged_codecs[resolved] = codec
        finally:
            processing.discard(resolved)
        processed_depths[resolved] = depth
        return destination

    def stage_data_file(data: DataReference, reader: Path, depth: int) -> str | None:
        """Stage the data file ``data`` names on a card of ``reader``; return
        the text that replaces its path on the card, or None to keep it."""
        where = f"{data.keyword}{data.raw_path} on line {data.card.line_start} of {reader}"
        absolute = is_absolute_reference(data.raw_path)
        if not absolute and not relative_data_path_established(dialect, in_deck=depth == 0):
            raise DeckStagingError(
                "data_file_unestablished",
                f"Cannot stage {where}: LTspice builds read a relative data path on "
                "a card in an included file from different folders (LTspice 26 from "
                "the included file's folder, LTspice XVII from the deck's folder; on "
                "a card in the deck both read the deck's folder). Name the file by an "
                "absolute path, or move the card into the deck.",
                reference=data.raw_path,
            )
        target = resolve_reference(stands_at.parent, data.raw_path)
        try:
            located = resolve_safe_path(str(target), allowed_roots)
        except PathSecurityError as exc:
            raise DeckStagingError(
                "data_file_outside_allowed_paths",
                f"Cannot stage {where}: {exc}. Add its folder to [security] "
                "allowed_paths (LTSPICE_MCP_ALLOWED_PATHS), or move the file into "
                "an allowed folder.",
                reference=data.raw_path,
            ) from exc
        if not located.is_file():
            raise DeckStagingError(
                "data_file_missing",
                f"Cannot stage {where}: {target} "
                + ("is not a file" if located.exists() else "does not exist")
                + (
                    ""
                    if absolute
                    else f". A relative path is read from the deck's folder ({stands_at.parent})"
                )
                + ". Put the file there, or correct the path on the card.",
                reference=data.raw_path,
            )
        if absolute:
            if located not in live_data:
                live_data.add(located)
                add_manifest(
                    ManifestEntry(
                        path=located,
                        sha256="",
                        staged=False,
                        live=True,
                        reason="data file named by an absolute path, read where it is",
                    )
                )
                observations.append(
                    {
                        "code": "live_data_file",
                        "kind": "provenance",
                        "detail": (
                            f"{data.keyword}{data.raw_path} is named by an absolute path, "
                            "so the run reads it where it is; its content is not covered "
                            "by the staged snapshot hash. Name it relative to the deck to "
                            "have it staged."
                        ),
                        "evidence": {"file": str(reader), "reference": data.raw_path},
                    }
                )
            if windows_paths and not (
                _WINDOWS_DRIVE_RE.match(data.raw_path) or data.raw_path.startswith("\\\\")
            ):
                return _render_data_path(wsl.to_windows_path(located), data.quoted)
            return None
        root = _containing_root(located, roots)
        if root is None:
            roots.append(located.parent)
            root = len(roots) - 1
        copied = snapshot_file(
            located, _destination_for(located, staging_root, roots, root), depth + 1, walk=False
        )
        rendered = _render_absolute(copied)
        if rendered not in absolute_data:
            absolute_data.append(rendered)
        return _render_data_path(rendered, data.quoted)

    def _snapshot_origin(resolved: Path) -> str:
        """Snapshot the file the primary deck was generated from; return its digest.

        It carries no include references — it is not SPICE — so it is staged
        without walking. It joins the manifest as a staged entry so every later
        check over that manifest sees it: drift verification is the one that
        matters, since an edit to the schematic leaves the previously exported
        netlist on disk byte-identical.
        """
        if not exported:
            return primary_sha
        origin_root = _containing_root(resolved, allowed)
        if origin_root is None:
            raise DeckStagingError(
                "include_unstaged",
                f"Source {resolved} is outside the configured allowed roots",
                reference=str(resolved),
            )
        snapshot_file(
            resolved,
            _destination_for(resolved, staging_root, roots, origin_root),
            0,
            walk=False,
        )
        return source_digests[resolved]

    staged_primary = snapshot_file(source, primary_destination, 0)
    # Every staged file's readings, stated together: one observation per build
    # and kind of reading, whatever the number of files and values.
    observations.extend(_read_otherwise_observations(stated, stated_builds))
    primary_sha = next(
        entry.sha256 for entry in manifest if entry.path == source and entry.section is None
    )
    origin_sha = _snapshot_origin(authored)
    if absolute_data and dialect == "ltspice":
        observations.append(
            {
                "code": "data_path_unrecorded",
                "kind": "provenance",
                "detail": (
                    "The staged deck names the staged copy of each data file it reads "
                    "by an absolute path, in quotes when it holds a space. That LTspice "
                    "reads an absolute wavefile= or PWL file= path, and a quoted one, "
                    "is not recorded (data-file-absolute-path in the LTspice "
                    "recordings, pending): if a case reports the file missing, this is "
                    "why."
                ),
                "evidence": {"paths": absolute_data},
            }
        )
    return StagedDeck(
        source_path=source,
        staged_deck=staged_primary,
        text=staged_texts[source],
        sha256=primary_sha,
        origin_sha256=origin_sha,
        manifest=manifest,
        observations=observations,
        source_contents=source_bytes,
        includes=[
            StagedFile(
                source=path,
                staged_path=source_destinations[path],
                text=text,
                sha256=staged_digests[path],
                codec=staged_codecs[path],
            )
            for path, text in staged_texts.items()
            if path != source
        ],
        codec=staged_codecs[source],
    )


#: The family a staged copy's references are split as, by the readers of
#: staged copies below: every family reads a staged copy alike, since
#: staging writes an LTspice path of more than one word in quotes
#: (``stage_deck``) and splits every other family's as ngspice does.
STAGED_COPY_FAMILY: str | None = None


def staged_reference_targets(text: str, source: Path, *, depth: int) -> list[Path]:
    """Return the resolved paths one staged file's include references name.

    At ``depth`` 0 the file is the deck, and its references are the ones
    staging followed: none in its title or after its ``.END``.
    """
    return staged_card_reference_targets(
        lex_for_edit(text, deck=depth == 0).read, source, depth=depth
    )


def staged_card_reference_targets(
    cards: list[SpiceCard], source: Path, *, depth: int
) -> list[Path]:
    """Resolve staged references from cards already parsed for this file."""
    return [
        resolve_reference(source.parent, reference.raw_path).resolve()
        for reference in scan_include_references(
            cards, source, family=STAGED_COPY_FAMILY, depth=depth
        )
    ]


def rewrite_staged_references(
    text: str,
    source: Path,
    renames: dict[Path, str],
    *,
    depth: int,
) -> str:
    """Repoint include references at renamed copies sitting beside the originals.

    Only the reference's final path segment is rewritten, so whatever spelling
    staging chose for the rest of it — a POSIX relative hop between two deep
    files, a Windows absolute path handed to LTspice across the WSL boundary —
    survives untouched. That is why ``renames`` is keyed by resolved path and
    valued by bare filename: a copy that is not a sibling of its original
    cannot be addressed this way. At ``depth`` 0 the file is the deck, and a
    reference in its title or after its ``.END`` is left as it is.
    """
    lexed = lex_for_edit(text, deck=depth == 0)
    return (
        emit(lexed.cards)
        if rewrite_staged_reference_cards(lexed.read, source, renames, depth=depth)
        else text
    )


def rewrite_staged_reference_cards(
    cards: list[SpiceCard],
    source: Path,
    renames: dict[Path, str],
    *,
    depth: int,
) -> bool:
    """``rewrite_staged_references`` over cards already lexed, in place.

    Returns whether any reference was rewritten.
    """
    changed = False
    for reference in scan_include_references(
        cards, source, family=STAGED_COPY_FAMILY, depth=depth
    ):
        target = resolve_reference(source.parent, reference.raw_path).resolve()
        name = renames.get(target)
        if name is None:
            continue
        _replace_reference(reference, _replace_last_segment(reference.raw_path, name))
        changed = True
    return changed


def _replace_last_segment(raw_path: str, name: str) -> str:
    cut = max(raw_path.rfind("/"), raw_path.rfind("\\"))
    return name if cut < 0 else raw_path[: cut + 1] + name


def verify_staged_manifest(manifest: list[ManifestEntry]) -> list[dict[str, Any]]:
    """Surface source drift without changing which staged bytes are consumed."""
    observations: list[dict[str, Any]] = []
    checked: set[Path] = set()
    for entry in manifest:
        if not entry.staged or entry.live or entry.path in checked:
            continue
        checked.add(entry.path)
        try:
            current = sha256_file(entry.path)
        except OSError as exc:
            observations.append(
                {
                    "code": "source_unavailable_after_staging",
                    "kind": "provenance",
                    "detail": (
                        f"Source {entry.path} is no longer readable; the staged copy "
                        "remains the experiment input."
                    ),
                    "evidence": {"path": str(entry.path), "error": str(exc)},
                }
            )
            continue
        if current != entry.sha256:
            observations.append(
                {
                    "code": "source_modified_after_staging",
                    "kind": "provenance",
                    "detail": (
                        f"Source {entry.path} changed after the manifest was created; "
                        "the experiment continues to use the staged copy."
                    ),
                    "evidence": {
                        "path": str(entry.path),
                        "manifest_sha256": entry.sha256,
                        "current_sha256": current,
                    },
                }
            )
    return observations


def _resolved_roots(roots: Sequence[Path]) -> list[Path]:
    resolved = []
    for root in roots:
        try:
            resolved.append(root.resolve(strict=True))
        except OSError:
            resolved.append(root.resolve(strict=False))
    return resolved


def _containing_root(path: Path, roots: list[Path]) -> int | None:
    for index, root in enumerate(roots):
        if path == root or path.is_relative_to(root):
            return index
    return None


def _compact_destination(source: Path, staging_root: Path) -> Path:
    """Short, source-specific names in a namespace disjoint from case clones."""
    suffixes = [suffix.casefold() for suffix in source.suffixes]
    suffix = source.suffix.casefold()
    for marker in (".lib", ".sub"):
        if marker in suffixes[:-1]:
            suffix = marker + suffix
            break
    identity = hashlib.sha256(str(source).encode("utf-8")).hexdigest()[:32]
    return staging_root / f"model-{identity}{suffix}"


def _destination_for(
    source: Path,
    staging_root: Path,
    roots: list[Path],
    root_index: int,
) -> Path:
    relative = source.relative_to(roots[root_index])
    return staging_root / f"root-{root_index}" / relative


def scan_include_references(
    cards: list[SpiceCard],
    source: Path,
    *,
    family: str | None,
    depth: int = 0,
    exists: Callable[[Path], bool] | None = None,
) -> list[IncludeReference]:
    """Return quote-aware include/library references from parsed cards.

    ``family`` is the simulator family that reads the cards. The one place a
    ``.lib`` card is split into a path and a section: LTspice reads the whole
    rest of the line as one path, quoted or not (recorded on both builds:
    ``deck/lib_path_space_unquoted``, ``deck/lib_path_space_quoted``,
    ``deck/lib_section``), so its references name no section. Every other
    family reads the second word as the path and a third as the section, as
    ngspice does. A staged copy writes an LTspice path of more than one word
    in quotes (``stage_deck``), so the two readings of a staged copy agree.

    ``depth`` is 0 for the deck itself and >0 for a file reached by following
    a reference out of it — inside such a file a bare ``.lib <name>`` is a
    section declaration, not an include.
    """
    references: list[IncludeReference] = []
    for card in cards:
        if card.kind != "directive":
            continue
        try:
            tokens = tokenize_body(card.body)
        except SpiceLexError:
            if directive_head(card) in INCLUDE_HEADS:
                # Which file it names is not known, so the references read
                # are not all of them. Staging refuses the card by its line
                # (``unreadable_references``) before reading any.
                raise
            continue  # a directive that names no file
        tokens = [token for token in tokens if token.kind != TokenKind.COMMENT_TRAIL]
        if len(tokens) < 2 or tokens[0].text.casefold() not in INCLUDE_HEADS:
            continue
        head = tokens[0].text.casefold()
        is_lib = head == ".lib"
        path_token = tokens[1]
        raw_path = unquote(path_token.text)
        first_word: str | None = None
        if is_lib and family == "ltspice" and len(tokens) > 2:
            # The rest of the line, as written, is one path.
            start, end = tokens[1].body_offset, tokens[-1].body_end
            path_token = Token(
                TokenKind.BARE, card.body[start:end], None, None, start, end - start
            )
            raw_path = path_token.text
            first_word = unquote(tokens[1].text)
            if not (exists or Path.exists)(resolve_reference(source.parent, first_word)):
                first_word = None
            section = None
        else:
            section = unquote(tokens[2].text) if is_lib and len(tokens) > 2 else None
        if not raw_path:
            continue
        if (
            is_lib
            and section is None
            and len(tokens) == 2
            and looks_like_section_declaration(raw_path, source, depth, exists=exists)
        ):
            continue
        references.append(
            IncludeReference(
                card=card,
                token=path_token,
                raw_path=raw_path,
                section=section,
                head=head,
                first_word=first_word,
            )
        )
    return references


@dataclass(frozen=True)
class DataReference:
    """One data file a card reads when it runs, named in the card's text.

    ``start`` and ``end`` are the span of the path in the card's body as
    written, quotes included, so a rewrite replaces the path and nothing else.
    """

    card: SpiceCard
    start: int
    end: int
    raw_path: str
    quoted: bool
    #: How the card names the file, as a refusal quotes it (``wavefile=``).
    keyword: str


#: The data files each simulator reads, by the words that name them on a card
#: it reads. LTspice: a V or I source's ``wavefile=`` (the
#: ``wavefile-source-nodes`` recordings), the same written ``wavefile level.wav``
#: with no ``=`` (``wavefile-without-equals``), and a PWL source's ``file=``
#: (``data-file-relative-path``). ngspice: a ``filesource`` model's ``file=``
#: (measured on ngspice-42; ``tests/test_staged_data_files.py``). Other
#: simulators' data cards are not read here.
DATA_READING_DIALECTS = frozenset({"ltspice", "ngspice"})


def scan_data_references(cards: list[SpiceCard], dialect: str | None) -> list[DataReference]:
    """The data files ``cards`` name, as the simulator of ``dialect`` reads them.

    The one reader of these cards: staging copies what it returns, and a
    recoverable run checks that each file it returns was frozen with the deck
    (``experiment_inputs._verify_inputs``). A data file is read when the case
    runs, so the staged copy of the deck must name a file the run can open.
    Whether it is known where a simulator looks for a relative path is
    ``relative_data_path_established``.
    """
    if dialect not in DATA_READING_DIALECTS:
        return []
    found: list[DataReference] = []
    for card in cards:
        try:
            tokens = [
                token
                for token in tokenize_body(card.body)
                if token.kind != TokenKind.COMMENT_TRAIL
            ]
        except SpiceLexError:
            continue  # reported by the lint's unreadable-card rule
        if dialect == "ltspice":
            found.extend(_ltspice_data_references(card, tokens))
        else:
            found.extend(_ngspice_data_references(card, tokens))
    return found


def _ltspice_data_references(card: SpiceCard, tokens: list[Token]) -> list[DataReference]:
    if card.kind != "instance" or not tokens or tokens[0].text[:1].upper() not in "VI":
        return []
    found = []
    for index, token in enumerate(tokens):
        if token.kind == TokenKind.KEY_VALUE and (token.key or "").casefold() in {
            WAVEFILE,
            "file",
        }:
            reference = _value_reference(card, card.body, token, 0, f"{token.key}=")
            if reference is not None:
                found.append(reference)
        elif (
            token.kind == TokenKind.BARE
            and token.text.casefold() == WAVEFILE
            and index + 1 < len(tokens)
            and tokens[index + 1].kind in (TokenKind.BARE, TokenKind.QUOTED)
        ):
            path = tokens[index + 1]
            raw = unquote(path.text)
            if raw:
                found.append(
                    DataReference(
                        card,
                        path.body_offset,
                        path.body_end,
                        raw,
                        path.kind == TokenKind.QUOTED,
                        f"{token.text} ",
                    )
                )
    return found


def _ngspice_data_references(card: SpiceCard, tokens: list[Token]) -> list[DataReference]:
    if card.kind != "model" or len(tokens) < 3:
        return []
    if tokens[2].text.split("(", 1)[0].casefold() != "filesource":
        return []
    found = []
    for token in tokens[2:]:
        if token.kind == TokenKind.KEY_VALUE:
            candidates, text, offset = [token], card.body, 0
        elif token.kind == TokenKind.PARENED:
            text = token.text[1:-1]
            offset = token.body_offset + 1
            try:
                candidates = tokenize_body(text)
            except SpiceLexError:
                continue
        else:
            continue
        for candidate in candidates:
            if (
                candidate.kind == TokenKind.KEY_VALUE
                and (candidate.key or "").casefold() == "file"
            ):
                reference = _value_reference(card, text, candidate, offset, f"{candidate.key}=")
                if reference is not None:
                    found.append(reference)
    return found


def _value_reference(
    card: SpiceCard, text: str, token: Token, offset: int, keyword: str
) -> DataReference | None:
    """The path a ``key=path`` token names, with its span in ``card.body``.

    ``text`` is the text the token was read from, which starts at ``offset``
    in the card's body (the inside of a model's parentheses).
    """
    written = text[token.body_offset : token.body_end]
    value = written.split("=", 1)[1]
    start = token.body_offset + len(written) - len(value) + (len(value) - len(value.lstrip()))
    value = value.strip()
    raw = unquote(value)
    if not raw:
        return None
    quoted = raw != value
    return DataReference(card, offset + start, offset + start + len(value), raw, quoted, keyword)


def relative_data_path_established(dialect: str | None, *, in_deck: bool) -> bool:
    """Whether every build of the simulator of ``dialect`` reads a relative
    data path from one folder, the deck's, so staging can copy the file from
    there.

    LTspice reads a relative path on a card in the deck from the deck's
    folder, never from the folder it was started in (the
    ``data-file-relative-path`` recordings, both builds). For a card in an
    included file the builds differ (``data-file-relative-path-in-include``):
    LTspice 26 reads it from the included file's folder, LTspice XVII from
    the deck's folder, and staging does not know which build runs the deck,
    so no folder is established. ngspice-42 reads it
    from the deck's folder for a card in the deck and in an included file
    alike, and only then from the folder it was started in, which the server
    does not choose for a run (``tests/test_staged_data_files.py``).
    """
    return in_deck or dialect != "ltspice"


def unreadable_references(cards: list[SpiceCard]) -> list[tuple[SpiceCard, SpiceLexError]]:
    """The include and library cards whose path cannot be read, because the
    card cannot be split into parts (``spice_lex.unreadable_cards``): an
    unclosed quote, brace or parenthesis, or a closer with nothing open.
    ``scan_include_references`` raises the lexer's error on any of them and
    ``card_sections`` on a ``.lib``, so staging refuses them by line first,
    instead of running a deck whose files it could not check."""
    return [
        (card, error)
        for card, error in unreadable_cards(cards)
        if card.kind == "directive" and directive_head(card) in INCLUDE_HEADS
    ]


def closure_depth(index: int) -> int:
    """0 for the deck itself, 1 for anything it pulls in.

    Only that distinction reaches the section predicate — a bare ``.lib X`` is
    a section declaration in any file reached by following a reference — so a
    file three includes deep still reads as depth 1. Written once here, beside
    the predicate that consumes it, so a closure cannot number its files one
    way and have them read another.
    """
    return 0 if index == 0 else 1


def card_sections(
    cards: list[SpiceCard],
    source: Path,
    depth: int = 0,
    *,
    exists: Callable[[Path], bool] | None = None,
) -> list[str | None]:
    """Name the ``.lib``/``.endl`` section each card sits in, or ``None``.

    The lexer tracks ``.SUBCKT`` nesting but not library sections, so this walks
    the card list once and pairs each card with its enclosing section. With a
    second argument a ``.lib`` is a *select* (``.lib mos.lib ff``) and never
    opens anything; whether a single-argument one names a file or opens a
    section is answered by ``looks_like_section_declaration`` right here, so a
    caller cannot pair up an answer that disagrees with staging's.

    Lives beside that predicate because a second copy that reads a plain include
    as a section opens a section nothing closes: every later card is stamped
    with it, no declaration reads as top level any more, and an exact-name
    target that resolves today is refused as ambiguous.
    """
    sections: list[str | None] = []
    stack: list[str] = []
    for card in cards:
        sections.append(stack[-1] if stack else None)
        if card.kind != "directive":
            continue
        try:
            body_tokens = tokenize_body(card.body)
        except SpiceLexError:
            head = directive_head(card)
            if head == ".lib":
                # Whether it opens a section is not known; staging refuses it
                # by its line (``unreadable_references``).
                raise
            if head == ".endl" and stack:
                stack.pop()
            continue
        tokens = [token for token in body_tokens if token.kind != TokenKind.COMMENT_TRAIL]
        if not tokens:
            continue
        head = tokens[0].text.casefold()
        if head == ".endl" and stack:
            stack.pop()
        elif head == ".lib" and len(tokens) == 2:
            name = unquote(tokens[1].text)
            if name and looks_like_section_declaration(name, source, depth, exists=exists):
                stack.append(name)
    return sections


@dataclass(frozen=True)
class LibraryReader:
    """Who reads a deck's ``.lib`` cards: the simulator family and, for
    ngspice, the compatibility mode it reads the deck in (``""`` for native),
    which decides what a ``.lib`` card brings in (``library_read``)."""

    family: str | None
    mode: str | None = None

    def __post_init__(self) -> None:
        if self.family == "ngspice" and self.mode is None:
            raise ValueError("ngspice reads a .lib card by its mode; give the run's mode")


#: The reader for a deck whose simulator's reading of a library is not
#: decided (QSPICE, Xyce, or none named): the reading the server had before
#: it knew ngspice's, kept for them (``library_read``).
UNDECIDED_READER = LibraryReader(None)


@dataclass(frozen=True)
class LibraryRead:
    """What one include or library card brings in (``library_read``).

    ``unread`` says why nothing is read, and ``rule_id`` names the refusal the
    lint gives it; with neither, ``reads`` says which cards of the file are
    read: ``selected`` names the one section read (None reads every
    section), and ``outside`` whether the cards outside every section are.
    """

    selected: str | None = None
    outside: bool = True
    unread: str | None = None
    rule_id: str | None = None

    def reads(self, owner: str | None) -> bool:
        """Whether a card of the file in section ``owner`` (``card_sections``;
        None outside every section) is read. The one comparison of a card's
        section with the section a card selects."""
        if self.unread is not None:
            return False
        if owner is None:
            return self.outside
        return self.selected is None or owner.casefold() == self.selected.casefold()


#: What ngspice does after it fails to find a file a ``.lib`` line names as an
#: include, which differs by build: none of the file's cards are read in
#: either, and only whether the run stops differs (ngspice-42 measured here,
#: ngspice-45 in CI, ``tests/test_library_card_read.py``).
AFTER_A_MISSING_INCLUDE = (
    "whether the run then stops on the missing include or goes on without them, "
    "a wrong number, depends on the ngspice build (ngspice-42 goes on, ngspice-45 stops)"
)

#: The file itself, or a file pulled in whole: every card is read.
WHOLE_FILE = LibraryRead()


def library_read(
    reference: IncludeReference | None,
    sections: Sequence[str | None] | None,
    reader: LibraryReader,
) -> LibraryRead:
    """What ``reference`` brings in for ``reader``: one section's cards, the
    whole file, or nothing, with the reason.

    ``reference`` is None for the deck itself, which is read whole.
    ``sections`` are the ``card_sections`` of the file it names, or None when
    no such file exists. The one answer every reader of a deck's library
    cards asks: hierarchy discovery, the linter, the analyses a run expects,
    the comparison of ``verify_circuit`` and the mismatch plan.

    - LTspice reads the rest of the line as one path
      (``scan_include_references``) and a file by that name whole. No such
      file brings in nothing. What LTspice makes of a file that declares
      sections, named by a ``.lib`` with no section, is not recorded: it is
      read whole, as before (to record: ``lib-whole-sectioned`` in ``inputs/cases.toml``).
    - ngspice, measured on ngspice-42 and, where it differs, ngspice-45
      (``tests/test_library_card_read.py``):
      with a section named, only that section's cards, not the cards outside
      every section; in a mode that reads ``.lib`` as an include (``lt`` or
      ``ps`` set) or not at all (``s3``), nothing. With no section, in a mode
      that reads ``.lib`` as an include, the whole file when it declares no
      section, and nothing when it does: the file's own ``.lib tt`` is read as
      an include of a file ``tt`` that does not exist, and no card of the
      file is read. ngspice-42 then runs on without them, a wrong number;
      ngspice-45 stops on the missing include (``AFTER_A_MISSING_INCLUDE``).
      In any other mode ngspice stops on the card
      ("unimplemented dot command '.lib'"). An include of a file that
      declares sections reads its section lines as the deck's own: a mode
      that reads ``.lib`` as an include drops every card of the file, and
      any other stops on the first one.
    - Any other simulator's reading is not decided: a section named reads
      that section and the cards outside every section, and none reads the
      whole file, as before.
    """
    if reference is None:
        return WHOLE_FILE
    missing = LibraryRead(
        unread=_no_file(reference, reader), rule_id=_missing_rule(reference, reader)
    )
    if reader.family != "ngspice":
        if sections is None:
            return missing
        if reference.head != ".lib" or reader.family == "ltspice":
            return WHOLE_FILE
        return LibraryRead(selected=reference.section) if reference.section else WHOLE_FILE
    if reference.head != ".lib":
        # ngspice reads an include's own section lines as it reads them in a
        # deck: a mode that reads .lib as an include drops every card of the
        # file, and any other stops on the line (measured on ngspice-42).
        if sections is None:
            return missing
        named = sorted({owner for owner in sections if owner is not None}, key=str.casefold)
        if not named:
            return WHOLE_FILE
        mode = reader.mode or ""
        fix = f"Read one section with .lib {reference.raw_path} {named[0]} (sections: " + (
            f"{', '.join(named)})"
        )
        if reads_lib_as_includes(mode):
            return LibraryRead(
                unread=(
                    f"ngspice in mode {mode!r} reads the section line of {reference.raw_path} "
                    f"(.lib {named[0]}) as an include of a file that does not exist, so it "
                    f"drops every card of the file: {AFTER_A_MISSING_INCLUDE}. "
                    f'{fix}, in a mode that reads sections ([simulator] ngbehavior = "hsa")'
                ),
                rule_id="lib-sections-dropped-ngspice",
            )
        return LibraryRead(
            unread=(
                f"ngspice in mode {mode!r} stops on the section line of {reference.raw_path} "
                f"(.lib {named[0]}: \"unimplemented dot command '.lib'\"). {fix}"
            ),
            rule_id="lib-without-section-ngspice",
        )
    # ngspice. What a named section, or a .lib in a mode that does not read
    # .lib as an include, brings in does not depend on what the file holds.
    mode = reader.mode or ""
    if reference.section is not None:
        if reads_lib_as_includes(mode):
            return LibraryRead(
                unread=(
                    f"ngspice in mode {mode!r} reads .lib {reference.raw_path} "
                    f"{reference.section} as two plain includes: it finds no file "
                    f"{reference.section!r} and reads none of the library's cards: "
                    f"{AFTER_A_MISSING_INCLUDE}"
                ),
                rule_id="lib-section-ngspice",
            )
        if rejects_lib(mode):
            return LibraryRead(
                unread=f"ngspice in mode {mode!r} (s3) does not read .lib at all",
                rule_id="lib-section-ngspice",
            )
        if sections is None:
            return missing
        return LibraryRead(selected=reference.section, outside=False)
    if not reads_lib_as_includes(mode):
        declares = sections is not None and any(owner is not None for owner in sections)
        fix = (
            f"Name the section the file declares: .lib {reference.raw_path} <section>"
            if declares
            else f"Use .include {reference.raw_path} for a file with no sections"
            if sections is not None
            else f"Name the section to read (.lib {reference.raw_path} <section>), or use "
            ".include for a file with no sections"
        )
        return LibraryRead(
            unread=(
                f"ngspice in mode {mode!r} stops on a .lib card that names no section "
                f"(\"unimplemented dot command '.lib'\"). {fix}"
            ),
            rule_id="lib-without-section-ngspice",
        )
    if sections is None:
        return missing
    named = sorted({owner for owner in sections if owner is not None}, key=str.casefold)
    if not named:
        return WHOLE_FILE
    return LibraryRead(
        unread=(
            f"ngspice in mode {mode!r} reads .lib {reference.raw_path} as an include, "
            f"and the file's own section line (.lib {named[0]}) as an include of a "
            f"file that does not exist, so it drops every card of the file: "
            f"{AFTER_A_MISSING_INCLUDE}. Name the section: .lib {reference.raw_path} {named[0]} "
            f"(sections: {', '.join(named)}), in a mode that reads sections "
            '([simulator] ngbehavior = "hsa")'
        ),
        rule_id="lib-sections-dropped-ngspice",
    )


def _no_file(reference: IncludeReference, reader: LibraryReader) -> str:
    """Why a reference to a file that does not exist reads nothing."""
    if reader.family == "ltspice" and reference.first_word is not None:
        return (
            "LTspice has no library sections: it reads the rest of the line, "
            f"{reference.raw_path!r}, as one file name, and no such file exists; "
            f"{reference.first_word!r} alone names a file. Include the file the "
            "section's models are in, or a copy holding only that section"
        )
    return f"no file {reference.raw_path!r} exists"


def _missing_rule(reference: IncludeReference, reader: LibraryReader) -> str | None:
    if (
        reader.family == "ltspice"
        and reference.head == ".lib"
        and any(char.isspace() for char in reference.raw_path)
    ):
        return "lib-section-ltspice"
    return None


def looks_like_section_declaration(
    raw_path: str,
    source: Path,
    depth: int = 0,
    *,
    exists: Callable[[Path], bool] | None = None,
) -> bool:
    """True when a single-token ``.lib X`` declares a section rather than
    naming a file to include.

    Library-context test, in order of authority: any file reached by following
    a reference is one (that is how a sectioned library is entered), and at the
    top level the name has to look like a library. Matching ``.lib`` anywhere
    in the suffixes — not just the last one — is what admits the near-universal
    PDK naming ``<pdk>.lib.spice`` (sky130, gf180); keying on the final suffix
    alone read every corner declaration inside them as a missing file.

    Public because it is the *only* answer to this question: a second one
    written elsewhere can disagree, and then one caller reads a file the other
    reads as a section.

    ``exists`` may supply captured existence facts for immutable readers.
    Omitting it retains the live filesystem classification used by staging.
    """
    if depth == 0 and not any(suffix.casefold() in {".lib", ".sub"} for suffix in source.suffixes):
        return False
    if any(char in raw_path for char in ("/", "\\", ".")):
        return False
    return not (exists or Path.exists)(source.parent / raw_path)


def unquote(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        return value[1:-1]
    return value


def _render_data_path(path: str, quoted: bool) -> str:
    """A data path as a card names it: in quotes when it was, or when it holds a space."""
    return f'"{path}"' if quoted or any(char.isspace() for char in path) else path


def _unquoted_words(reference: IncludeReference) -> bool:
    """Whether the path of ``reference`` is more than one word, unquoted."""
    return reference.token.kind != TokenKind.QUOTED and any(
        char.isspace() for char in reference.raw_path
    )


def _replace_reference(reference: IncludeReference, path: str) -> None:
    rendered = f'"{path}"' if any(char.isspace() for char in path) else path
    reference.card.replace_span(
        reference.token.body_offset,
        reference.token.body_end,
        rendered,
    )


def is_absolute_reference(raw_path: str) -> bool:
    """Return whether a SPICE file reference is POSIX, drive, or UNC absolute."""
    return (
        Path(raw_path).is_absolute()
        or _WINDOWS_DRIVE_RE.match(raw_path) is not None
        or raw_path.startswith("\\\\")
    )


def _portable_relative(raw_path: str) -> Path:
    return Path(raw_path.replace("\\", "/"))


def resolve_reference(parent: Path, raw_path: str) -> Path:
    """Resolve a SPICE file reference — POSIX, Windows drive (via ``/mnt``
    on WSL), or UNC — against ``parent``.

    Public because it is the *only* answer to this question: a second resolver
    written elsewhere can disagree, and then another layer (the linter) reads
    a different file than the one staging staged.
    """
    if wsl.is_wsl():  # the mount spellings are WSL interop, never the general path
        if _WINDOWS_DRIVE_RE.match(raw_path):
            windows = PureWindowsPath(raw_path)
            drive = windows.drive.rstrip(":").lower()
            return Path("/mnt") / drive / Path(*windows.parts[1:])
        if raw_path.startswith("\\\\"):
            windows = PureWindowsPath(raw_path)
            return Path("/") / Path(*windows.parts)
    normalized = _portable_relative(raw_path)
    return normalized if normalized.is_absolute() else parent / normalized


def resolve_existing(path: Path) -> Path | None:
    try:
        return path.resolve(strict=True)
    except OSError:
        return None


def _unstaged_reason(
    target: Path,
    resolved: Path | None,
    root_index: int | None,
    *,
    depth: int,
    max_depth: int,
) -> str | None:
    if depth >= max_depth:
        return f"include recursion exceeds depth {max_depth}"
    if resolved is None:
        return f"referenced file does not exist: {target}"
    if root_index is None:
        return (
            "referenced file resolves outside allowed roots and outside the "
            f"simulator's own library: {resolved}"
        )
    if not resolved.is_file():
        return f"referenced path is not a file: {resolved}"
    return None
