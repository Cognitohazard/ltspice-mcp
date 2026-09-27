"""Bind final experiment cases to the captured inputs of a native PDK profile."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path, PurePath

from ltspice_mcp.errors import NetlistError
from ltspice_mcp.lib.deck_staging import StagedDeck, sha256_file
from ltspice_mcp.lib.encoding import decode_spice_bytes
from ltspice_mcp.lib.hierarchy import Hierarchy, SemanticProfile, Source, load_hierarchy
from ltspice_mcp.lib.native_records import NativeCaseRecord
from ltspice_mcp.lib.pdk_native import (
    ENTRYPOINT,
    NGBEHAVIOR,
    ActiveCard,
    ArtifactDigest,
    CaseInputs,
    FinalAssignment,
    LibraryBinding,
    MosCoverage,
    NativeCaseError,
    NativeRequest,
    NativeRequestError,
    Occurrence,
    OriginalCapture,
    VerifiedOriginals,
    validate_sample,
    verify_originals,
)
from ltspice_mcp.lib.spice_lex import SpiceCard, TokenKind, lex, tokenize_body
from ltspice_mcp.lib.variations import ExpandedCase, MaterializedCase


class _Sources:
    """One mapping from materialized occurrences back to same-read original bytes."""

    def __init__(
        self,
        staged: StagedDeck,
        variant: MaterializedCase,
        cards: dict[Path, dict[int, SpiceCard]],
    ):
        self.staged = staged
        self.originals = {
            entry.staged_path.resolve(): entry.path.resolve()
            for entry in staged.manifest
            if entry.staged and entry.staged_path is not None
        }
        self.texts = {
            path.resolve(): text
            for path, text in [
                (staged.staged_deck, staged.text),
                *((item.staged_path, item.text) for item in staged.includes),
            ]
        }
        self.cards = cards
        self.lineage: dict[tuple[str, int, str | None], Source] = {}
        for item in variant.source_lineage:
            key = (item.case_source.path, item.case_source.line, item.case_source.section)
            previous = self.lineage.setdefault(key, item.staged_source)
            if previous != item.staged_source:
                raise NativeCaseError("lineage", "final source occurrence has ambiguous origins")
        self.files: dict[Path, Path] = {}
        for item in variant.source_lineage:
            final = Path(item.case_source.path).resolve()
            original = self.originals[Path(item.staged_source.path).resolve()]
            previous = self.files.setdefault(final, original)
            if previous != original:
                raise NativeCaseError("lineage", "one final file mixes original captures")
        self.files[variant.path.resolve()] = staged.source_path.resolve()
        self.identities: dict[Path, str] = {}

    def original_path(self, final: Path) -> Path:
        original = self.files.get(final.resolve(), self.originals.get(final.resolve()))
        if original is None:
            raise NativeCaseError("lineage", "final dependency lacks an original capture")
        return original

    def origin(self, source: Source) -> Source | None:
        return self.lineage.get((source.path, source.line, source.section))

    def occurrence(self, source: Source) -> Occurrence:
        staged = self.origin(source)
        if staged is None:
            raise NativeCaseError("lineage", "final active occurrence lacks source lineage")
        path = Path(staged.path).resolve()
        original = self.originals[path]
        if path not in self.cards:
            before = lex(decode_spice_bytes(self.staged.source_contents[original])).cards
            after = lex(self.texts[path]).cards
            if len(before) != len(after) or any(
                old.kind != new.kind for old, new in zip(before, after, strict=True)
            ):
                raise NativeCaseError("lineage", "staging changed the source card population")
            self.cards[path] = {
                new.line_start: old for old, new in zip(before, after, strict=True)
            }
        card = self.cards[path][staged.line]
        return Occurrence(self.identities[original], card.line_start, staged.section, card.scope)


def _coverage(hierarchy: Hierarchy, sources: _Sources) -> tuple[MosCoverage, ...]:
    rows = {tuple(p.casefold() for p in row.instance): row for row in hierarchy.instances}
    coverage = []
    for row in hierarchy.instances:
        if row.element != "M":
            continue
        parent = rows.get(tuple(p.casefold() for p in row.instance[:-1]))
        if parent is None or parent.model_source is None:
            raise NativeCaseError("coverage", "physical MOS has no resolved PDK wrapper")
        geometry = dict(row.geometry)
        environment = dict(row.environment)
        parameters = dict(row.parameters)

        mult = environment.get("mult")
        nf = parameters.get("nf", environment.get("nf"))

        # SPICE defaults an omitted device/call multiplier to one. An unrelated
        # parameter named m is not a multiplier unless the instance uses it.
        multipliers = [
            dict(rows[tuple(p.casefold() for p in row.instance[:depth])].parameters).get("m")
            for depth in range(1, len(row.instance) + 1)
        ]
        multiplicity = (
            1.0 if all(fact is None or fact.value == 1 for fact in multipliers) else None
        )
        coverage.append(
            MosCoverage(
                row.instance,
                sources.occurrence(row.source),
                sources.occurrence(parent.model_source),
                tuple(sources.occurrence(model.source) for model in row.model_family),
                geometry["w"].value if "w" in geometry else None,
                geometry["l"].value if "l" in geometry else None,
                hierarchy.scale.value,
                multiplicity,
                mult.value if mult is not None else None,
                nf.value if nf is not None else None,
            )
        )
    return tuple(coverage)


def _assignments(
    staged: StagedDeck, expanded: ExpandedCase, hierarchy: Hierarchy, sources: _Sources
) -> tuple[FinalAssignment, ...]:
    files = [staged.staged_deck, *(item.staged_path for item in staged.includes)]
    rows = {tuple(p.casefold() for p in row.instance): row for row in hierarchy.instances}
    result = []
    for edit in expanded.edits:
        if edit.kind in {"model", "instance_param"}:
            raise NativeRequestError("native statistics do not permit model or mismatch injection")
        if edit.instance_edit is not None:
            instance_edit = edit.instance_edit
            row = rows.get(tuple(p.casefold() for p in instance_edit.instance))
            if row is None:
                raise NativeCaseError(
                    "assignment", "assigned instance is absent from final hierarchy"
                )
            result.append(
                FinalAssignment(
                    "model" if instance_edit.attribute == "model" else "instance",
                    sources.occurrence(row.source),
                    instance_edit.parameter or instance_edit.attribute,
                    edit.value,
                    row.instance,
                )
            )
            continue
        site = edit.site
        if site is None:
            raise NativeCaseError("assignment", "assignment lacks one resolved declaration")
        candidates = []
        for active in hierarchy.occurrences:
            origin = sources.origin(active.source)
            if origin is None or Path(origin.path).resolve() != files[edit.file_index].resolve():
                continue
            occurrence = sources.occurrence(active.source)
            if tuple(p.casefold() for p in occurrence.scope) != tuple(
                p.casefold() for p in site.scope
            ):
                continue
            if (occurrence.section or "").casefold() != (site.section or "").casefold():
                continue
            card = active.card
            matches = (
                card.kind == "param"
                and any(
                    t.kind == TokenKind.KEY_VALUE
                    and (t.key or "").casefold() == site.name.casefold()
                    for t in tokenize_body(card.body)
                )
                if edit.kind == "param"
                else (card.name or "").casefold() == site.name.casefold()
            )
            if matches:
                candidates.append(occurrence)
        unique = set(candidates)
        if len(unique) != 1:
            raise NativeCaseError(
                "assignment", "final assignment has ambiguous original occurrence"
            )
        result.append(
            FinalAssignment(
                "parameter" if edit.kind == "param" else "instance",
                unique.pop(),
                site.name if edit.kind == "param" else "value",
                edit.value,
            )
        )
    return tuple(result)


def _bench_identities(paths: Sequence[PurePath]) -> dict[PurePath, str]:
    """Name captured bench files without putting host drives in sample identity."""
    volumes: dict[PurePath, list[PurePath]] = {}
    for path in paths:
        if not path.is_absolute():
            raise NativeCaseError("source_identity", "bench capture path is not absolute")
        volumes.setdefault(path.parents[-1], []).append(path)
    result = {}
    # Capture traversal order is stable; drive letters and configured root
    # ordering must not choose the namespace when the same bench is relocated.
    for index, group in enumerate(volumes.values()):
        root = group[0].parent
        for path in group[1:]:
            while not path.is_relative_to(root):
                root = root.parent
        prefix = f"bench/root-{index}/" if len(volumes) > 1 else "bench/"
        for path in group:
            result[path] = prefix + path.relative_to(root).as_posix()
    return result


class NativeCaseValidator:
    """Share immutable captured-card parsing within one circuit's preparation."""

    def __init__(self, staged: StagedDeck, staging_root: Path):
        self.staged = staged
        self.staging_root = staging_root
        self.cards: dict[Path, dict[int, SpiceCard]] = {}
        self._originals: dict[tuple[OriginalCapture, ...], VerifiedOriginals] = {}

    def _verified_originals(self, captures: tuple[OriginalCapture, ...]) -> VerifiedOriginals:
        originals = self._originals.get(captures)
        if originals is None:
            originals = verify_originals(captures)
            self._originals[captures] = originals
        return originals

    def validate(
        self, request: NativeRequest, variant: MaterializedCase, expanded: ExpandedCase
    ) -> NativeCaseRecord:
        """Validate each final case against its captured inputs and written bytes."""
        staged = self.staged
        if any(entry.live or not entry.staged for entry in staged.manifest):
            raise NativeCaseError("closure", "native statistics require completely staged inputs")
        expected = dict(variant.file_digests)
        if expected.get(variant.path.resolve()) != variant.sha256:
            raise NativeCaseError("artifact_drift", "electrical case lacks its written-byte hash")
        try:
            for path, digest in expected.items():
                if sha256_file(path) != digest:
                    raise NativeCaseError("artifact_drift", "materialized case bytes changed")
        except OSError as exc:
            raise NativeCaseError(
                "artifact_drift", "materialized case file is unavailable"
            ) from exc
        try:
            hierarchy = load_hierarchy(
                str(variant.path), [self.staging_root], SemanticProfile("ngspice", NGBEHAVIOR)
            )
        except (NetlistError, OSError) as exc:
            raise NativeCaseError("hierarchy", str(exc)) from exc
        if any(expected.get(item.path) != item.digest for item in hierarchy.inputs):
            raise NativeCaseError(
                "artifact_drift", "captured hierarchy differs from written bytes"
            )
        sources = _Sources(staged, variant, self.cards)
        bindings = [
            binding
            for binding in hierarchy.library_bindings
            if sources.original_path(binding.target).as_posix().endswith("/" + ENTRYPOINT)
        ]
        if len(bindings) != 1:
            raise NativeCaseError("binding", "one active pinned PDK entrypoint is required")
        binding = bindings[0]
        pdk_root = sources.original_path(binding.target).parents[len(Path(ENTRYPOINT).parts) - 1]
        originals = {sources.original_path(item.path) for item in hierarchy.inputs}
        bench = _bench_identities(
            [
                sources.original_path(item.path)
                for item in hierarchy.inputs
                if not sources.original_path(item.path).is_relative_to(pdk_root)
            ]
        )
        captures = []
        for original in sorted(originals):
            relative = (
                original.relative_to(pdk_root).as_posix()
                if original.is_relative_to(pdk_root)
                else None
            )
            identity = "pdk/" + relative if relative else bench[original]
            sources.identities[original] = identity
            captures.append(OriginalCapture(identity, staged.source_contents[original], relative))
        root_identity = sources.identities[staged.source_path.resolve()]
        inputs = CaseInputs(
            root_identity,
            tuple(captures),
            tuple(
                ActiveCard(sources.occurrence(item.source), item.card)
                for item in hierarchy.occurrences
            ),
            (
                LibraryBinding(
                    sources.occurrence(binding.source),
                    sources.identities[sources.original_path(binding.target)],
                    binding.section,
                ),
            ),
            _coverage(hierarchy, sources),
            _assignments(staged, expanded, hierarchy, sources),
            hashlib.sha256(repr(hierarchy.binding()).encode()).hexdigest(),
            True,
            True,
        )
        sample = validate_sample(
            request, inputs, originals=self._verified_originals(inputs.captures)
        )
        dependencies = []
        for item in hierarchy.inputs:
            if item.path == variant.path.resolve():
                continue
            identity = sources.identities[sources.original_path(item.path)]
            if identity == root_identity:
                raise NativeCaseError(
                    "lineage", "an included clone of the electrical root is unsupported"
                )
            dependencies.append(ArtifactDigest(item.path, item.digest, identity))
        return NativeCaseRecord(request, sample=sample, pending_dependencies=tuple(dependencies))
