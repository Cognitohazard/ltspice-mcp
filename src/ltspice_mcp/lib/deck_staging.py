"""Snapshot SPICE decks and their local dependency graph for experiments."""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass, field
from pathlib import Path, PureWindowsPath
from typing import Any

from ltspice_mcp.lib import atomic_write_bytes, atomic_write_text
from ltspice_mcp.lib.encoding import decode_spice_bytes
from ltspice_mcp.lib.experiment_types import ManifestEntry
from ltspice_mcp.lib.spice_lex import SpiceCard, Token, emit, lex, tokenize_body

DEFAULT_INCLUDE_DEPTH = 3

INCLUDE_HEADS = frozenset({".include", ".inc", ".lib", ".libfile"})
_WINDOWS_DRIVE_RE = re.compile(r"^[A-Za-z]:[\\/]")


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


@dataclass
class StagedDeck:
    """One staged primary deck plus its dependency manifest."""

    source_path: Path
    staged_deck: Path
    text: str
    sha256: str
    manifest: list[ManifestEntry]
    observations: list[dict[str, Any]] = field(default_factory=list)


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
    allow_live_includes: bool = False,
    max_depth: int = DEFAULT_INCLUDE_DEPTH,
) -> StagedDeck:
    """Copy a primary deck and its include/lib closure into ``staging_root``.

    Relative dependencies retain their topology within the allowed root. An
    absolute or cross-root reference is rewritten only in the staged copy so
    the simulator still consumes the snapshot. The authoring files are never
    modified.
    """
    if max_depth < 0:
        raise ValueError("max_depth must be non-negative")
    roots = _resolved_roots(allowed_roots)
    source = source_path.resolve(strict=True)
    root_index = _containing_root(source, roots)
    if root_index is None:
        raise DeckStagingError(
            "include_unstaged",
            f"Primary deck {source} is outside the configured allowed roots",
            reference=str(source),
        )

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

    def snapshot_file(path: Path, destination: Path, depth: int) -> Path:
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
            for reference in scan_include_references(parsed.cards, resolved):
                target = _resolve_reference(resolved.parent, reference.raw_path)
                target_resolved = _resolve_existing(target)
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
                        raise DeckStagingError(
                            "include_unstaged",
                            f"Cannot stage reference {reference.raw_path!r} from "
                            f"{resolved}: {reason}",
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
                expected = (destination.parent / _portable_relative(reference.raw_path)).resolve()
                if (
                    is_absolute_reference(reference.raw_path)
                    or expected != actual_destination.resolve()
                ):
                    staged_relative = Path(
                        os.path.relpath(actual_destination, destination.parent)
                    ).as_posix()
                    _replace_reference(reference, staged_relative)
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

    staged_primary = snapshot_file(source, primary_destination, 0)
    primary_sha = next(
        entry.sha256 for entry in manifest if entry.path == source and entry.section is None
    )
    return StagedDeck(
        source_path=source,
        staged_deck=staged_primary,
        text=staged_texts[source],
        sha256=primary_sha,
        manifest=manifest,
        observations=observations,
    )


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


def _resolved_roots(roots: list[Path]) -> list[Path]:
    if not roots:
        raise DeckStagingError("include_unstaged", "No allowed roots are configured")
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
) -> list[IncludeReference]:
    """Return quote-aware include/library references from parsed cards."""
    references: list[IncludeReference] = []
    for card in cards:
        if card.kind != "directive":
            continue
        tokens = tokenize_body(card.body)
        if len(tokens) < 2 or tokens[0].text.casefold() not in INCLUDE_HEADS:
            continue
        path_token = tokens[1]
        raw_path = _unquote(path_token.text)
        if not raw_path:
            continue
        is_lib = tokens[0].text.casefold() == ".lib"
        section = _unquote(tokens[2].text) if is_lib and len(tokens) > 2 else None
        if is_lib and section is None and _looks_like_section_declaration(raw_path, source):
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


def _looks_like_section_declaration(raw_path: str, source: Path) -> bool:
    if source.suffix.casefold() not in {".lib", ".sub"}:
        return False
    if any(char in raw_path for char in ("/", "\\", ".")):
        return False
    return not (source.parent / raw_path).exists()


def _unquote(value: str) -> str:
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


def _resolve_reference(parent: Path, raw_path: str) -> Path:
    if _WINDOWS_DRIVE_RE.match(raw_path):
        windows = PureWindowsPath(raw_path)
        drive = windows.drive.rstrip(":").lower()
        return Path("/mnt") / drive / Path(*windows.parts[1:])
    if raw_path.startswith("\\\\"):
        windows = PureWindowsPath(raw_path)
        return Path("/") / Path(*windows.parts)
    normalized = _portable_relative(raw_path)
    return normalized if normalized.is_absolute() else parent / normalized


def _resolve_existing(path: Path) -> Path | None:
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
        return f"referenced file resolves outside allowed roots: {resolved}"
    if not resolved.is_file():
        return f"referenced path is not a file: {resolved}"
    return None
