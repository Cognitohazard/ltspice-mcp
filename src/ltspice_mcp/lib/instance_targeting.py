"""Validated, occurrence-exact edits with private runtime ancestry clones."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

from ltspice_mcp.errors import NetlistError
from ltspice_mcp.lib.component_value import apply_value_to_instance
from ltspice_mcp.lib.deck_staging import (
    resolve_reference,
    rewrite_staged_references,
    scan_include_references,
)
from ltspice_mcp.lib.hierarchy import (
    CapturedFile,
    Hierarchy,
    SemanticProfile,
    Source,
    parse_hierarchy_cards,
    resolve_hierarchy,
    walk_active,
)
from ltspice_mcp.lib.montecarlo import render_variant_model_card
from ltspice_mcp.lib.spice_lex import SpiceCard, TokenKind, emit, lex, tokenize_body
from ltspice_mcp.lib.spice_lex_ops import rename_subckt
from ltspice_mcp.lib.spice_lex_views import InstanceLine
from ltspice_mcp.lib.subckt_mismatch import ClosureFile

Attribute = Literal["value", "model", "parameter"]
_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
_MODEL = re.compile(r"[A-Za-z_][A-Za-z0-9_.$:-]*\Z")


def canonical_target(instance: Sequence[str], attribute: str, parameter: str | None = None) -> str:
    return "instance:" + json.dumps(
        [
            *[p.casefold() for p in instance],
            attribute,
            parameter.casefold() if parameter else None,
        ],
        separators=(",", ":"),
    )


def validate_segments(instance: Sequence[str]) -> None:
    if not instance or any(
        not p or p != p.strip() or any(c.isspace() for c in p) for p in instance
    ):
        raise ValueError("instance must be a nonempty list of nonempty reference segments")


def validate_payload(attribute: Attribute, parameter: str | None, value: str | float | int) -> str:
    if (parameter is not None) != (attribute == "parameter"):
        raise ValueError("parameter is required iff attribute is parameter")
    if parameter is not None and not _IDENTIFIER.fullmatch(parameter):
        raise ValueError("parameter must be an identifier")
    if isinstance(value, (int, float)) and not math.isfinite(value):
        raise ValueError("assignment must be finite")
    text = str(value)
    if not text.strip() or any(c in text for c in "\r\n;$"):
        raise ValueError("payload must be nonempty, single-line and comment-free")
    tokens = tokenize_body(text)
    if attribute == "model":
        if len(tokens) != 1 or tokens[0].kind != TokenKind.BARE or not _MODEL.fullmatch(text):
            raise ValueError("model must be one legal model reference token")
    elif attribute == "parameter":
        # Lex in the destination slot; exactly one complete KEY_VALUE must survive.
        slot = tokenize_body("p=" + text)
        if (
            len(tokens) != 1
            or len(slot) != 1
            or slot[0].kind != TokenKind.KEY_VALUE
            or slot[0].value != text
        ):
            raise ValueError("parameter payload must be one expression")
        if any(
            t.kind in {TokenKind.KEY_VALUE, TokenKind.EQUALS, TokenKind.COMMENT_TRAIL}
            for t in tokens
        ):
            raise ValueError("parameter payload contains another assignment")
        if "=" in text:
            raise ValueError(
                "assignment/comparison delimiters are unsupported in parameter payloads"
            )
    return text


@dataclass(frozen=True)
class InstanceEdit:
    instance: tuple[str, ...]
    attribute: Attribute
    value: str | float | int
    parameter: str | None = None

    @property
    def target(self) -> str:
        return canonical_target(self.instance, self.attribute, self.parameter)


def apply_edit(card: SpiceCard, edit: InstanceEdit) -> set[str]:
    """Validate and write through the existing typed dispatcher; return actual fields."""
    text = validate_payload(edit.attribute, edit.parameter, edit.value)
    view = InstanceLine.from_card(card)
    if edit.attribute == "model":
        if view.model is None:
            raise NetlistError(f"{view.ref} has no model slot")
        view.set_model(text)
        return {"model"}
    if edit.attribute == "parameter":
        assert edit.parameter is not None
        view.set_param(edit.parameter, text)
        return {"parameter:" + edit.parameter.casefold()}
    tokens = tokenize_body(text)
    fields = {
        "parameter:" + t.key.casefold() for t in tokens if t.kind == TokenKind.KEY_VALUE and t.key
    }
    if any(t.kind != TokenKind.KEY_VALUE for t in tokens):
        fields.add(
            "model" if view.model is not None and view.ref[0].upper() in "MQJXD" else "value"
        )
    apply_value_to_instance(card, text)
    return fields


def _captured_inputs(files: Sequence[ClosureFile]) -> dict[Path, CapturedFile]:
    """Use the shared include scanner with an immutable in-memory existence snapshot."""
    by_path = {file.path.resolve(): file for file in files}
    captured = {}
    for file in files:
        path = file.path.resolve()
        existence: dict[Path, bool] = {}

        def exists(candidate: Path, captured_existence: dict[Path, bool] = existence) -> bool:
            captured_existence[candidate] = candidate.resolve() in by_path
            return captured_existence[candidate]

        refs = scan_include_references(
            parse_hierarchy_cards(file.text.encode(), root=file.index == 0),
            path,
            depth=file.depth,
            exists=exists,
        )
        targets = []
        for ref in refs:
            candidate = resolve_reference(path.parent, ref.raw_path).resolve()
            targets.append((ref.raw_path, candidate if candidate in by_path else None))
        captured[path] = CapturedFile(
            path, file.text.encode(), tuple(existence.items()), tuple(targets)
        )
    return captured


def captured_hierarchy(files: Sequence[ClosureFile], profile: SemanticProfile) -> Hierarchy:
    return resolve_hierarchy(files[0].path.resolve(), _captured_inputs(files), profile)


@dataclass(frozen=True)
class SourceLineage:
    """A final context and its origin, whose definition is None when shared."""

    case_source: Source
    staged_source: Source


def source_lineage(
    before: Sequence[ClosureFile], after: Sequence[ClosureFile], profile: SemanticProfile
) -> tuple[SourceLineage, ...]:
    """Map active contexts through ordinary edits that preserve card population."""
    captured = _captured_inputs(before)
    occurrences = walk_active(
        before[0].path.resolve(), lambda path, depth: captured[path], profile
    )
    origins: dict[tuple[str, int], Source] = {}
    for occurrence in occurrences:
        source = occurrence.source
        key = (source.path, source.line)
        previous = origins.get(key)
        if previous is not None:
            if previous.section != source.section:
                raise NetlistError(
                    f"source lineage spans distinct sections at {source.path}:{source.line}"
                )
            if previous.definition != source.definition:
                source = replace(source, definition=None)
        origins[key] = source
    positions: dict[tuple[str, int], tuple[str, int]] = {}
    for old, new in zip(before, after, strict=True):
        old_path, new_path = str(old.path.resolve()), str(new.path.resolve())
        old_cards, new_cards = lex(old.text).cards, lex(new.text).cards
        if len(old_cards) != len(new_cards):
            raise NetlistError(
                "ordinary edits changed card population; source lineage is unavailable"
            )
        for old_card, new_card in zip(old_cards, new_cards, strict=True):
            key = (old_path, old_card.line_start)
            if key in origins and old_card.kind == new_card.kind:
                positions[key] = (new_path, new_card.line_start)
    result = []
    for occurrence in occurrences:
        source = occurrence.source
        key = (source.path, source.line)
        if key in positions:
            path, line = positions[key]
            result.append(SourceLineage(replace(source, path=path, line=line), origins[key]))
    return tuple(result)


def source_origins(
    before: Sequence[ClosureFile], after: Sequence[ClosureFile], profile: SemanticProfile
) -> dict[tuple[str, int], Source]:
    """Return physical origins, retaining a definition only when unambiguous."""
    return {
        (item.case_source.path, item.case_source.line): item.staged_source
        for item in source_lineage(before, after, profile)
    }


def select(hierarchy: Hierarchy, instance: Sequence[str]):
    folded = tuple(p.casefold() for p in instance)
    matches = [
        row for row in hierarchy.instances if tuple(p.casefold() for p in row.instance) == folded
    ]
    if len(matches) != 1:
        raise NetlistError(f"instance {list(instance)!r} does not resolve exactly once")
    return matches[0]


class TargetEditor:
    """One in-memory case edit transaction; all output uses the existing case writer."""

    def __init__(
        self,
        files: Sequence[ClosureFile],
        profile: SemanticProfile,
        cloned: set[tuple[str, ...]] | None = None,
        origins: dict[tuple[str, int], Source] | None = None,
    ):
        self.files = list(files)
        self.profile = profile
        self.hierarchy = captured_hierarchy(files, profile)
        self.cards = {file.path.resolve(): lex(file.text).cards for file in files}
        self.original = {path: list(cards) for path, cards in self.cards.items()}
        self.cloned = set(cloned or ())
        self.maps: dict[tuple[str, ...], dict[tuple[str, int], SpiceCard]] = {}
        self.names = {d.view.name.casefold() for d in self.hierarchy.definitions}
        self.root_map = {
            (str(path), card.line_start): card
            for path, cards in self.cards.items()
            for card in cards
        }
        input_origins = origins if origins is not None else source_origins(files, files, profile)
        self.origins = {
            id(card): input_origins[key]
            for key, card in self.root_map.items()
            if key in input_origins
        }

    def _name(self, path: tuple[str, ...]) -> str:
        stem = "target_" + hashlib.sha256(json.dumps(path).encode()).hexdigest()[:20]
        name = stem
        serial = 0
        while name.casefold() in self.names:
            serial += 1
            name = f"{stem}_{serial}"
        self.names.add(name.casefold())
        return name

    def _clone_fragment(
        self,
        source: Path,
        owner: tuple[str, ...],
        mapping: dict[tuple[str, int], SpiceCard],
        copies: dict[Path, Path],
    ) -> Path:
        if source in copies:
            return copies[source]
        digest = hashlib.sha256(json.dumps([owner, str(source)]).encode()).hexdigest()[:16]
        destination = source.with_name(f"target_{digest}__{source.name}")
        if destination in self.cards:
            raise NetlistError(f"generated include name collision: {destination.name}")
        copies[source] = destination
        block = copy.deepcopy(self.original[source])
        self.cards[destination] = block
        self.files.append(ClosureFile(len(self.files), destination, ""))
        self._map_block(source, block, mapping)
        self._copy_includes(source, block, owner, mapping, copies)
        return destination

    def _map_block(
        self, source: Path, block: list[SpiceCard], mapping: dict[tuple[str, int], SpiceCard]
    ) -> None:
        for card in block:
            key = (str(source), card.line_start)
            if key in mapping:
                raise NetlistError(
                    "repeated source occurrence in cloned definition is unsupported"
                )
            mapping[key] = card
            original = self.root_map.get(key)
            if original is not None and id(original) in self.origins:
                self.origins[id(card)] = self.origins[id(original)]

    def _copy_includes(
        self,
        source: Path,
        block: list[SpiceCard],
        owner: tuple[str, ...],
        mapping: dict[tuple[str, int], SpiceCard],
        copies: dict[Path, Path],
    ) -> None:
        refs = scan_include_references(
            block, source, depth=1, exists=lambda p: p.resolve() in self.original
        )
        for ref in refs:
            target = resolve_reference(source.parent, ref.raw_path).resolve()
            if target not in self.original:
                raise NetlistError(f"missing cloned include {ref.raw_path}")
            destination = self._clone_fragment(target, owner, mapping, copies)
            rewritten = rewrite_staged_references(
                emit([ref.card]), source, {target: destination.name}, depth=1
            )
            replacement = lex(rewritten).cards[0]
            ref.card.raw_lines = replacement.raw_lines
            ref.card.body = replacement.body
            ref.card.body_layout = replacement.body_layout

    def _ancestry(self, path: tuple[str, ...]) -> dict[tuple[str, int], SpiceCard]:
        if not path:
            return self.root_map
        folded = tuple(p.casefold() for p in path)
        if folded in self.maps:
            return self.maps[folded]
        row = select(self.hierarchy, path)
        parent_map = self._ancestry(path[:-1])
        if folded in self.cloned:
            self.maps[folded] = self.root_map
            return self.root_map
        definition = next(
            (d for d in self.hierarchy.definitions if d.source == row.model_source), None
        )
        if definition is None:
            raise NetlistError(f"{path} has no exact subcircuit definition")
        if definition.source.path != definition.closer.path:
            raise NetlistError("subcircuit boundaries across includes cannot be cloned")
        source = Path(definition.source.path)
        original = self.original[source]
        block = copy.deepcopy(
            [
                c
                for c in original
                if definition.source.line <= c.line_start <= definition.closer.line
            ]
        )
        mapping: dict[tuple[str, int], SpiceCard] = {}
        self._map_block(source, block, mapping)
        self._copy_includes(source, block, folded, mapping, {})
        name = self._name(folded)
        rename_subckt(block, definition.view.name, name)
        caller = parent_map[(row.source.path, row.source.line)]
        InstanceLine.from_card(caller).set_model(name)
        closer = self.root_map[(definition.closer.path, definition.closer.line)]
        at = self.cards[source].index(closer) + 1
        if not closer.raw_lines[-1].endswith("\n"):
            closer.raw_lines[-1] += "\n"
        if not block[-1].raw_lines[-1].endswith("\n"):
            block[-1].raw_lines[-1] += "\n"
        self.cards[source][at:at] = block
        self.maps[folded] = mapping
        self.cloned.add(folded)
        return mapping

    def apply(self, edits: Sequence[InstanceEdit]) -> None:
        for edit in edits:
            row = select(self.hierarchy, edit.instance)
            mapping = self._ancestry(row.instance[:-1])
            card = mapping[(row.source.path, row.source.line)]
            apply_edit(card, edit)

    def result(self) -> list[ClosureFile]:
        return [
            ClosureFile(file.index, file.path, emit(self.cards[file.path.resolve()]))
            for file in self.files
        ]

    def clone_flat_model(
        self, source: Source, instance: tuple[str, ...], overrides: dict[str, float]
    ) -> str:
        """Retain the flat model-variant engine using the resolved model occurrence."""
        card = self.root_map[(source.path, source.line)]
        names = {
            c.name.casefold()
            for cards in self.cards.values()
            for c in cards
            if c.kind == "model" and c.name
        }
        self.names.update(names)
        name = self._name(("model", *[part.casefold() for part in instance]))
        clone = lex(render_variant_model_card(emit([card]), name, overrides)).cards[0]
        if id(card) in self.origins:
            self.origins[id(clone)] = self.origins[id(card)]
        cards = self.cards[Path(source.path)]
        if not card.raw_lines[-1].endswith("\n"):
            card.raw_lines[-1] += "\n"
        if not clone.raw_lines[-1].endswith("\n"):
            clone.raw_lines[-1] += "\n"
        cards.insert(cards.index(card) + 1, clone)
        return name

    def output_origins(self) -> dict[tuple[str, int], Source]:
        result = {}
        for path, cards in self.cards.items():
            line = 1
            for card in cards:
                if id(card) in self.origins:
                    result[(str(path), line)] = self.origins[id(card)]
                line += len(emit([card]).splitlines())
        return result

    def lineage(self) -> tuple[SourceLineage, ...]:
        origins = self.output_origins()
        hierarchy = captured_hierarchy(self.result(), self.profile)
        return tuple(
            SourceLineage(
                occurrence.source, origins[(occurrence.source.path, occurrence.source.line)]
            )
            for occurrence in hierarchy.occurrences
            if (occurrence.source.path, occurrence.source.line) in origins
        )
