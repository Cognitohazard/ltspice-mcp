"""Snapshot SPICE decks and their local dependency graph for experiments."""

from __future__ import annotations

import hashlib
import os
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path, PureWindowsPath
from typing import Any

from ltspice_mcp.lib import atomic_write_bytes, atomic_write_text
from ltspice_mcp.lib.encoding import decode_spice_bytes
from ltspice_mcp.lib.experiment_types import ManifestEntry
from ltspice_mcp.lib.spice_lex import SpiceCard, Token, TokenKind, emit, lex, tokenize_body

# Sized for real foundry PDKs, which fan out further than a hand-written deck:
# sky130 reaches a device model five levels down (deck -> sky130.lib.spice ->
# corners/<corner>.spice -> <device>__<corner>.corner.spice -> <device>.pm3.spice),
# and gf180 nests comparably. Cycles are caught separately by the in-progress
# set, so this bound is a resource guard, not the loop guard.
DEFAULT_INCLUDE_DEPTH = 8

INCLUDE_HEADS = frozenset({".include", ".inc", ".lib", ".libfile"})
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


@dataclass(frozen=True)
class IncludeReference:
    """One quote-aware include or library reference in a parsed deck."""

    card: SpiceCard
    token: Token
    raw_path: str
    section: str | None


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
    """Route experiment staging and output to simulator-safe filesystems."""
    from spicelib.simulators.ltspice_simulator import LTspice

    from ltspice_mcp.lib import wsl

    is_ltspice = isinstance(simulator, type) and issubclass(simulator, LTspice)
    if wsl.is_wsl() and is_ltspice:
        windows_root = wsl.get_windows_output_dir()
        if windows_root is None:
            raise DeckStagingError(
                "windows_native_storage_unavailable",
                "WSL LTspice experiments require a Windows-native directory for both "
                "staged decks and simulator output, but no Windows temp directory "
                "is available",
            )
        base = windows_root / "experiments"
        return ExperimentPaths(
            staging_root=base / "jobs" / job_id / "staged" / circuit_id,
            output_folder=base / "runs",
            windows_native=True,
        )

    return ExperimentPaths(
        staging_root=(working_dir / ".ltspice-mcp" / "jobs" / job_id / "staged" / circuit_id),
        output_folder=working_dir / ".ltspice-mcp" / "runs",
    )


def stage_deck(
    source_path: Path,
    staging_root: Path,
    allowed_roots: list[Path],
    *,
    origin: Path,
    allow_live_includes: bool = False,
    max_depth: int = DEFAULT_INCLUDE_DEPTH,
    windows_paths: bool = False,
    simulator_roots: Sequence[Path] = (),
) -> StagedDeck:
    """Copy a primary deck and its include/lib closure into ``staging_root``.

    Relative dependencies retain their topology within the allowed root. An
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

    ``windows_paths`` renders the root deck's rewritten references in Windows
    form, for a Windows simulator reached across the WSL boundary: it cannot
    open the ``/mnt/c/...`` spelling of the very file it is being handed.

    ``simulator_roots`` are the detected simulator's own library directories
    (``simulator.simulator_library_roots``). They are appended AFTER the
    caller's roots, so a reference inside one is staged and hashed like any
    other dependency while every existing root keeps its index — but the deck
    itself must still live in an allowed root, so this cannot be used to reach
    a deck the sandbox denies.
    """
    if max_depth < 0:
        raise ValueError("max_depth must be non-negative")
    allowed = _resolved_roots(allowed_roots)
    roots = allowed + _resolved_roots(list(simulator_roots), required=False)
    source = source_path.resolve(strict=True)
    # Authored files are checked against ``allowed`` alone — a prefix of
    # ``roots``, so the index means the same thing in both — which is what
    # keeps the simulator's library a place references may POINT, never a
    # place a deck may be RUN FROM.
    root_index = _containing_root(source, allowed)
    if root_index is None:
        raise DeckStagingError(
            "include_unstaged",
            f"Primary deck {source} is outside the configured allowed roots",
            reference=str(source),
        )

    def _render_absolute(path: Path) -> str:
        if windows_paths:
            from ltspice_mcp.lib import wsl

            return wsl.to_windows_path(path)
        return path.as_posix()

    staging_root.mkdir(parents=True, exist_ok=True)
    manifest: list[ManifestEntry] = []
    manifest_keys: set[tuple[Path, str | None]] = set()
    source_destinations: dict[Path, Path] = {}
    source_bytes: dict[Path, bytes] = {}
    source_digests: dict[Path, str] = {}
    staged_texts: dict[Path, str] = {}
    processed_depths: dict[Path, int] = {}
    processing: set[Path] = set()
    observations: list[dict[str, Any]] = []

    primary_destination = _destination_for(source, staging_root, roots, root_index)

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
            source_destinations[resolved] = destination
            data = resolved.read_bytes()
            source_bytes[resolved] = data
            source_digest = hashlib.sha256(data).hexdigest()
            source_digests[resolved] = source_digest
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
            text = decode_spice_bytes(data)
            parsed = lex(text)
            changed = False
            for reference in scan_include_references(parsed.cards, resolved, depth=depth):
                target = resolve_reference(resolved.parent, reference.raw_path)
                target_resolved = resolve_existing(target)
                target_root = (
                    _containing_root(target_resolved, roots)
                    if target_resolved is not None
                    else None
                )
                reason = _unstaged_reason(
                    target,
                    target_resolved,
                    target_root,
                    depth=depth,
                    max_depth=max_depth,
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
                    ):
                        _replace_reference(
                            reference,
                            Path(
                                os.path.relpath(actual_destination, destination.parent)
                            ).as_posix(),
                        )
                        changed = True

            destination.parent.mkdir(parents=True, exist_ok=True)
            if changed:
                staged_text = emit(parsed.cards)
                atomic_write_text(destination, staged_text, durable=True)
            else:
                staged_text = text
                atomic_write_bytes(destination, data, durable=True)
            staged_texts[resolved] = staged_text
        finally:
            processing.discard(resolved)
        processed_depths[resolved] = depth
        return destination

    def _snapshot_origin(authoring_source: Path) -> str:
        """Snapshot the file the primary deck was generated from; return its digest.

        It carries no include references — it is not SPICE — so it is staged
        without walking. It joins the manifest as a staged entry so every later
        check over that manifest sees it: drift verification is the one that
        matters, since an edit to the schematic leaves the previously exported
        netlist on disk byte-identical.
        """
        resolved = authoring_source.resolve(strict=True)
        if resolved == source:
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
    primary_sha = next(
        entry.sha256 for entry in manifest if entry.path == source and entry.section is None
    )
    origin_sha = _snapshot_origin(origin)
    return StagedDeck(
        source_path=source,
        staged_deck=staged_primary,
        text=staged_texts[source],
        sha256=primary_sha,
        origin_sha256=origin_sha,
        manifest=manifest,
        observations=observations,
        includes=[
            StagedFile(source=path, staged_path=source_destinations[path], text=text)
            for path, text in staged_texts.items()
            if path != source
        ],
    )


def staged_reference_targets(text: str, source: Path, *, depth: int) -> list[Path]:
    """Return the resolved paths one staged file's include references name."""
    return [
        resolve_reference(source.parent, reference.raw_path).resolve()
        for reference in scan_include_references(lex(text).cards, source, depth=depth)
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
    cannot be addressed this way.
    """
    cards = lex(text).cards
    changed = False
    for reference in scan_include_references(cards, source, depth=depth):
        target = resolve_reference(source.parent, reference.raw_path).resolve()
        name = renames.get(target)
        if name is None:
            continue
        _replace_reference(reference, _replace_last_segment(reference.raw_path, name))
        changed = True
    return emit(cards) if changed else text


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


def _resolved_roots(roots: list[Path], *, required: bool = True) -> list[Path]:
    if not roots:
        if required:
            raise DeckStagingError("include_unstaged", "No allowed roots are configured")
        return []
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
    depth: int = 0,
) -> list[IncludeReference]:
    """Return quote-aware include/library references from parsed cards.

    ``depth`` is 0 for the deck itself and >0 for a file reached by following
    a reference out of it — inside such a file a bare ``.lib <name>`` is a
    section declaration, not an include.
    """
    references: list[IncludeReference] = []
    for card in cards:
        if card.kind != "directive":
            continue
        tokens = tokenize_body(card.body)
        if len(tokens) < 2 or tokens[0].text.casefold() not in INCLUDE_HEADS:
            continue
        path_token = tokens[1]
        raw_path = unquote(path_token.text)
        if not raw_path:
            continue
        is_lib = tokens[0].text.casefold() == ".lib"
        section = unquote(tokens[2].text) if is_lib and len(tokens) > 2 else None
        if is_lib and section is None and looks_like_section_declaration(raw_path, source, depth):
            continue
        references.append(
            IncludeReference(
                card=card,
                token=path_token,
                raw_path=raw_path,
                section=section,
            )
        )
    return references


def closure_depth(index: int) -> int:
    """0 for the deck itself, 1 for anything it pulls in.

    Only that distinction reaches the section predicate — a bare ``.lib X`` is
    a section declaration in any file reached by following a reference — so a
    file three includes deep still reads as depth 1. Written once here, beside
    the predicate that consumes it, so a closure cannot number its files one
    way and have them read another.
    """
    return 0 if index == 0 else 1


def card_sections(cards: list[SpiceCard], source: Path, depth: int = 0) -> list[str | None]:
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
        tokens = [
            token for token in tokenize_body(card.body) if token.kind != TokenKind.COMMENT_TRAIL
        ]
        if not tokens:
            continue
        head = tokens[0].text.casefold()
        if head == ".endl" and stack:
            stack.pop()
        elif head == ".lib" and len(tokens) == 2:
            name = unquote(tokens[1].text)
            if name and looks_like_section_declaration(name, source, depth):
                stack.append(name)
    return sections


def looks_like_section_declaration(raw_path: str, source: Path, depth: int = 0) -> bool:
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
    """
    if depth == 0 and not any(suffix.casefold() in {".lib", ".sub"} for suffix in source.suffixes):
        return False
    if any(char in raw_path for char in ("/", "\\", ".")):
        return False
    return not (source.parent / raw_path).exists()


def unquote(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        return value[1:-1]
    return value


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
    """Resolve a SPICE file reference — POSIX, Windows drive (via ``/mnt``),
    or UNC — against ``parent``.

    Public because it is the *only* answer to this question: a second resolver
    written elsewhere can disagree, and then another layer (the linter) reads
    a different file than the one staging staged.
    """
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
