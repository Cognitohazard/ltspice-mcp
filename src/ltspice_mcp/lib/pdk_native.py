"""Bounded Sky130 native statistics, independent of experiment orchestration.

The parent supplies the complete active closure, original captures, final
hierarchy and assignments. This module never discovers includes, edits PDK
controls, chooses Store locations or launches a simulator. Capture identities
are stable logical relative names; execution paths never enter sample keys.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from importlib.resources import files
from pathlib import Path, PurePath, PurePosixPath, PureWindowsPath
from typing import Any, Literal

from ltspice_mcp.lib import atomic_write_bytes
from ltspice_mcp.lib.cursor_codec import canonical_hash, canonical_json
from ltspice_mcp.lib.deck_staging import INCLUDE_HEADS, sha256_file
from ltspice_mcp.lib.encoding import decode_spice_bytes
from ltspice_mcp.lib.spice_lex import SpiceCard, TokenKind, lex, tokenize_body

PROFILE = "sky130-e6f9c887-ngspice-v1"
PIN_MANIFEST_SHA256 = "48f8c0953abca720520bfcc3729579d29b7f501c72dffac6ca564f0b21c2c589"
ENTRYPOINT = "libs.tech/ngspice/sky130.lib.spice"
WRAPPER = "sky130_fd_pr__nfet_01v8"
MODEL = WRAPPER + "__model"
DERIVATION_VERSION = "pdk-native-sha256-v1"
ADAPTER_VERSION = "ngspice-preload-v1"
NGBEHAVIOR = "hsa"
SEED_MAX = 2147483646
Mode = Literal["nominal", "mismatch", "process", "combined"]
_MODES: dict[str, tuple[str, int, int]] = {
    "nominal": ("tt", 0, 0),
    "mismatch": ("tt_mm", 1, 0),
    "process": ("mc", 0, 1),
    "combined": ("mc", 1, 1),
}
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}\Z")
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_OWNED = frozenset(
    {"seed", "randomseed", "mc_mm_switch", "mc_pr_switch", "ngbehavior", "ng_nomodcheck", "scale"}
)
_ANALYSES = frozenset({".op", ".tran", ".ac"})
_FORBIDDEN = frozenset(
    {
        ".step",
        ".alter",
        ".dc",
        ".noise",
        ".pz",
        ".sens",
        ".tf",
        ".disto",
        ".four",
        ".reset",
        ".setseed",
    }
)


class NativeRequestError(ValueError):
    """Reject the request: unsupported profile, conflicting ownership or collision."""


class NativeCaseError(ValueError):
    """Expected isolated validation/preparation failure; retain the case descriptor."""

    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(message)


@dataclass(frozen=True)
class NativeRequest:
    circuit_id: str
    family_id: str
    profile: str
    mode: str
    root_seed: int
    sample_index: int | None


@dataclass(frozen=True)
class OriginalCapture:
    """Captured bytes, not a path to reread. Parent classifies protected sources.

    ``identity`` is a stable project-relative logical name. ``pdk_relative``
    is the acquisition-relative path for EVERY active source within the PDK
    installation, including unused definitions in an active file. Bench files
    outside that namespace have None. Never substitute staged paths here.
    """

    identity: str
    content: bytes
    pdk_relative: str | None = None

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.content).hexdigest()


@dataclass(frozen=True)
class Occurrence:
    """Original physical card identity, including selected section and scope."""

    capture: str
    line: int
    section: str | None = None
    scope: tuple[str, ...] = ()


@dataclass(frozen=True)
class ActiveCard:
    """Final card from the parent's active flattened stream, in evaluation order."""

    source: Occurrence
    card: SpiceCard


@dataclass(frozen=True)
class LibraryBinding:
    source: Occurrence
    target_capture: str
    section: str


@dataclass(frozen=True)
class MosCoverage:
    """One physical MOS leaf and its actual resolved wrapper/model occurrences.

    Geometry is the FINAL resolved SI geometry and scale, including inherited
    multiplicity. Unknown values must be None, never supplied as defaults.
    ``models`` retains every candidate bin instead of guessing a selected bin.
    """

    instance: tuple[str, ...]
    source: Occurrence
    wrapper: Occurrence
    models: tuple[Occurrence, ...]
    width_m: float | None
    length_m: float | None
    scale: float | None
    m: float | None
    mult: float | None
    nf: float | None


@dataclass(frozen=True)
class FinalAssignment:
    """Resolved target, not selector spelling or generated clone identity.

    Ordinary root parameters also carry their original source occurrence.
    Model/library/control targets are profile-owned and cannot be assigned.
    """

    kind: Literal["parameter", "instance", "model", "library", "control"]
    source: Occurrence
    field: str
    value: str | int | float
    instance: tuple[str, ...] = ()


@dataclass(frozen=True)
class CaseInputs:
    root_capture: str
    captures: tuple[OriginalCapture, ...]
    active_cards: tuple[ActiveCard, ...]
    library_bindings: tuple[LibraryBinding, ...]
    coverage: tuple[MosCoverage, ...]
    assignments: tuple[FinalAssignment, ...]
    hierarchy_revision: str
    closure_complete: bool
    coverage_complete: bool


@dataclass(frozen=True)
class ValidatedSample:
    request: NativeRequest
    sample_key: str
    effective_seed: int
    input_digest: str
    model_digest: str
    population_digest: str
    coverage: tuple[MosCoverage, ...]
    analysis: str
    hierarchy_revision: str
    dependency_captures: tuple[str, ...]


@dataclass(frozen=True)
class ArtifactDigest:
    path: Path
    sha256: str
    original_capture: str = ""


@dataclass(frozen=True)
class NativePaths:
    """Explicit Store-owned paths, all beside each other in the existing run dir."""

    cwd: PurePath
    electrical_input: PurePath
    prepared_driver: PurePath
    executed_driver: PurePath
    raw: PurePath
    log: PurePath

    @property
    def artifacts(self) -> tuple[PurePath, ...]:
        return (
            self.electrical_input,
            self.prepared_driver,
            self.executed_driver,
            self.raw,
            self.log,
        )


@dataclass(frozen=True)
class PreparedLaunch:
    paths: NativePaths
    input_sha256: str
    driver_sha256: str
    dependencies: tuple[ArtifactDigest, ...]
    sample_key: str
    effective_seed: int
    analysis: str

    @property
    def switches(self) -> tuple[str, ...]:
        return ("-n",)

    @property
    def ngbehavior(self) -> str:
        return NGBEHAVIOR


@dataclass(frozen=True)
class SimulatorFacts:
    """Observed environment, never inferred from the native profile name."""

    version: str | None = None
    build: str | None = None
    executable_sha256: str | None = None
    platform: str | None = None


def validate_request(request: NativeRequest, *, backend: str = "ngspice") -> None:
    if request.profile != PROFILE or backend != "ngspice":
        raise NativeRequestError("unsupported native profile or backend")
    if request.mode not in _MODES:
        raise NativeRequestError("unsupported native mode")
    for name in (request.circuit_id, request.family_id):
        if not _IDENTIFIER.fullmatch(name):
            raise NativeRequestError("native circuit and family ids must be stable identifiers")
    if type(request.root_seed) is not int or not 0 <= request.root_seed <= 2**63 - 1:
        raise NativeRequestError("native root seed must be an integer in 0..2**63-1")
    if request.sample_index is not None and (
        type(request.sample_index) is not int or request.sample_index < 0
    ):
        raise NativeRequestError("logical sample index must be a nonnegative integer")


def validate_family_ownership(native_family_ids: Sequence[str], *, caller_random: bool) -> None:
    if len(native_family_ids) > 1 or (native_family_ids and caller_random):
        raise NativeRequestError(
            "one stochastic family per circuit; native and caller-random conflict"
        )


def profile_pins() -> dict[str, str]:
    content = (
        files("ltspice_mcp").joinpath("assets/pdk_profiles/sky130-e6f9c887.json").read_bytes()
    )
    if hashlib.sha256(content).hexdigest() != PIN_MANIFEST_SHA256:
        raise NativeCaseError("profile_manifest", "packaged acquisition pins have changed")
    return {item["path"]: item["sha256"] for item in json.loads(content)}


def _relative_identity(value: str) -> None:
    posix, windows = PurePosixPath(value), PureWindowsPath(value)
    if (
        not value
        or posix.is_absolute()
        or windows.drive
        or windows.root
        or "\\" in value
        or any(p in {".", ".."} for p in value.split("/"))
        or any(ord(c) < 32 for c in value)
        or ":" in value
        or str(posix) != value
    ):
        raise NativeCaseError(
            "source_identity", "source identity must be a canonical logical relative path"
        )


def validate_captures(captures: Sequence[OriginalCapture]) -> dict[str, OriginalCapture]:
    pins = profile_pins()
    result: dict[str, OriginalCapture] = {}
    protected: set[str] = set()
    for capture in captures:
        _relative_identity(capture.identity)
        if capture.identity in result:
            raise NativeCaseError("source_identity", "duplicate original capture identity")
        result[capture.identity] = capture
        if capture.pdk_relative is not None:
            _relative_identity(capture.pdk_relative)
            if capture.pdk_relative in protected:
                raise NativeCaseError(
                    "source_identity", "duplicate protected acquisition identity"
                )
            protected.add(capture.pdk_relative)
            if pins.get(capture.pdk_relative) != capture.sha256:
                raise NativeCaseError(
                    "pdk_pin",
                    f"protected capture differs from acquisition pin: {capture.pdk_relative}",
                )
    return result


def _source_key(source: Occurrence) -> dict[str, Any]:
    _relative_identity(source.capture)
    if type(source.line) is not int or source.line < 1:
        raise NativeCaseError("source_identity", "source line must be a positive integer")
    return {
        "capture": source.capture,
        "line": source.line,
        "section": source.section.casefold() if source.section else None,
        "scope": [p.casefold() for p in source.scope],
    }


def _original_cards(captures: dict[str, OriginalCapture]) -> dict[tuple[str, int], SpiceCard]:
    return {
        (identity, card.line_start): card
        for identity, capture in captures.items()
        for card in lex(decode_spice_bytes(capture.content)).cards
    }


def _original(source: Occurrence, cards: dict[tuple[str, int], SpiceCard]) -> SpiceCard:
    _source_key(source)
    card = cards.get((source.capture, source.line))
    if card is None or tuple(s.casefold() for s in card.scope) != tuple(
        s.casefold() for s in source.scope
    ):
        raise NativeCaseError("binding", "source occurrence does not identify a captured card")
    return card


def _validate_coverage(
    inputs: CaseInputs,
    captures: dict[str, OriginalCapture],
    cards: dict[tuple[str, int], SpiceCard],
) -> None:
    if not inputs.coverage_complete or not inputs.coverage:
        raise NativeCaseError("coverage", "complete nonempty physical MOS coverage is required")
    seen: set[tuple[str, ...]] = set()
    active_sources = {a.source for a in inputs.active_cards}
    section = inputs.library_bindings[0].section.casefold()
    expected_model_path = (
        f"libs.ref/sky130_fd_pr/spice/{WRAPPER}{'__tt' if section != 'mc' else ''}.pm3.spice"
    )
    for leaf in inputs.coverage:
        if any(
            source not in active_sources for source in (leaf.source, leaf.wrapper, *leaf.models)
        ):
            raise NativeCaseError(
                "coverage", "covered wrapper, leaf and models must be active occurrences"
            )
        instance = tuple(p.casefold() for p in leaf.instance)
        if not instance or instance in seen:
            raise NativeCaseError(
                "coverage", "physical MOS identities must be nonempty and unique"
            )
        seen.add(instance)
        wrapper = _original(leaf.wrapper, cards)
        device = _original(leaf.source, cards)
        capture = captures[leaf.wrapper.capture]
        if (
            capture.pdk_relative != expected_model_path
            or wrapper.kind != "subckt"
            or (wrapper.name or "").casefold() != WRAPPER
            or leaf.source.capture != leaf.wrapper.capture
            or device.kind != "instance"
            or (device.name or "").casefold() != "m" + WRAPPER
            or tuple(p.casefold() for p in device.scope) != (WRAPPER,)
            or not leaf.models
        ):
            raise NativeCaseError(
                "coverage", "MOS must bind to the protected wrapper and physical leaf occurrences"
            )
        for model in leaf.models:
            card = _original(model, cards)
            name = (card.name or "").casefold()
            if (
                model.capture != leaf.wrapper.capture
                or card.kind != "model"
                or tuple(p.casefold() for p in card.scope) != (WRAPPER,)
                or not re.fullmatch(re.escape(MODEL) + r"\.\d+", name)
            ):
                raise NativeCaseError(
                    "coverage", "MOS model must bind to protected local bin occurrences"
                )
        for value in (leaf.width_m, leaf.length_m):
            if value is None or isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise NativeCaseError("geometry", "final W/L must be finite positive SI geometry")
        if leaf.scale != 1e-6 or any(
            v is None or isinstance(v, bool) or v != 1 for v in (leaf.m, leaf.mult, leaf.nf)
        ):
            raise NativeCaseError(
                "geometry", "native profile requires scale=1e-6 and unit m, mult and nf"
            )


def _owned_parameters(inputs: CaseInputs, captures: dict[str, OriginalCapture]) -> frozenset[str]:
    names = set(_OWNED)
    for active in inputs.active_cards:
        if (
            captures[active.source.capture].pdk_relative is not None
            and active.card.kind == "param"
            and not active.card.scope
        ):
            names.update(
                (token.key or "").casefold()
                for token in tokenize_body(active.card.body)
                if token.kind == TokenKind.KEY_VALUE
            )
    return frozenset(names)


def _validate_controls(
    request: NativeRequest,
    inputs: CaseInputs,
    captures: dict[str, OriginalCapture],
    owned_parameters: frozenset[str],
) -> str:
    analyses: list[str] = []
    overrides: list[tuple[int, ActiveCard, str, str]] = []
    binding = inputs.library_bindings[0]
    binding_order: int | None = None
    for index, active in enumerate(inputs.active_cards):
        card = active.card
        capture = captures.get(active.source.capture)
        if capture is None:
            raise NativeCaseError("binding", "active card lacks original capture")
        if active.source == binding.source:
            binding_tokens = tokenize_body(card.body)
            if (
                len(binding_tokens) != 3
                or binding_tokens[0].text.casefold() != ".lib"
                or binding_tokens[2].text.strip("\"'").casefold() != binding.section.casefold()
            ):
                raise NativeCaseError(
                    "binding", "final library card disagrees with active binding section"
                )
            binding_order = index
        if card.trailing:
            raise NativeCaseError("binding", "active stream contains a trailing card")
        if card.kind == "control":
            raise NativeRequestError("native adapter owns control and seed initialization")
        # Model bodies can be hundreds of kilobytes. Their protected bytes are
        # checked separately; only the directive name matters in this pass.
        words = card.body.split(maxsplit=1)
        head = words[0].casefold() if words else ""
        if head in _FORBIDDEN:
            raise NativeRequestError(
                "native adapter requires one .op/.tran/.ac and owns analysis sequencing"
            )
        if head in _ANALYSES:
            if card.scope:
                raise NativeRequestError("native analysis must be at root scope")
            analyses.append(head)
        if capture.pdk_relative is not None:
            continue
        for token in tokenize_body(card.body):
            if (
                token.kind == TokenKind.KEY_VALUE
                and (token.key or "").casefold() in owned_parameters
            ):
                overrides.append((index, active, (token.key or "").casefold(), token.value or ""))
            elif head in {".option", ".options"} and token.text.casefold() in _OWNED:
                overrides.append((index, active, token.text.casefold(), ""))
    if len(analyses) != 1:
        raise NativeRequestError("native adapter requires exactly one .op/.tran/.ac analysis")
    if binding_order is None:
        raise NativeCaseError("binding", "library binding is absent from active stream")
    expected = 1 if request.mode == "combined" else 0
    if len(overrides) != expected:
        raise NativeRequestError("unexpected or missing profile-owned control override")
    if overrides:
        order, active, key, value = overrides[0]
        if (
            key != "mc_mm_switch"
            or value != "1"
            or active.card.kind != "param"
            or active.card.scope
            or active.source.capture != inputs.root_capture
            or order <= binding_order
        ):
            raise NativeRequestError(
                "combined mode requires one root mc_mm_switch=1 after the library binding"
            )
    return analyses[0]


def _assignment_keys(
    assignments: Sequence[FinalAssignment],
    captures: dict[str, OriginalCapture],
    owned_parameters: frozenset[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[bytes] = set()
    for assignment in assignments:
        capture = captures.get(assignment.source.capture)
        if capture is None:
            raise NativeCaseError("assignment", "final assignment lacks an original source")
        if (
            capture.pdk_relative is not None
            or assignment.kind in {"model", "library", "control"}
            or assignment.field.casefold() in owned_parameters
        ):
            raise NativeRequestError(
                "assignment conflicts with profile-owned model, control or library binding"
            )
        target = {
            "kind": assignment.kind,
            "source": _source_key(assignment.source),
            "field": assignment.field.casefold(),
            "instance": [p.casefold() for p in assignment.instance],
        }
        key = canonical_json(target)
        if key in seen:
            raise NativeRequestError("duplicate final assignment target")
        seen.add(key)
        try:
            canonical_json(assignment.value)
        except ValueError as exc:
            raise NativeCaseError("assignment", "final assignment is not finite") from exc
        rows.append({**target, "value": assignment.value})
    return sorted(rows, key=canonical_json)


def validate_sample(request: NativeRequest, inputs: CaseInputs) -> ValidatedSample:
    """Validate final parent facts and derive a seed only from complete inputs."""
    validate_request(request)
    if request.sample_index is None:
        raise NativeCaseError("sample_index", "logical sample index is unavailable")
    if not inputs.closure_complete:
        raise NativeCaseError(
            "closure", "native closure must be complete, captured and staged; no live includes"
        )
    captures = validate_captures(inputs.captures)
    if inputs.root_capture not in captures or not inputs.hierarchy_revision:
        raise NativeCaseError(
            "input", "original electrical input or hierarchy revision is unavailable"
        )
    cards = _original_cards(captures)
    bindings = inputs.library_bindings
    if len(bindings) != 1:
        raise NativeCaseError(
            "binding", "one unambiguous active profile library binding is required"
        )
    binding = bindings[0]
    target = captures.get(binding.target_capture)
    if target is None or target.pdk_relative != ENTRYPOINT:
        raise NativeCaseError("binding", "entrypoint must identify a pinned original capture")
    if binding.section.casefold() != _MODES[request.mode][0]:
        raise NativeRequestError("library section conflicts with requested native mode")
    if _original(binding.source, cards).body.split(maxsplit=1)[0].casefold() != ".lib":
        raise NativeCaseError("binding", "library binding must identify an actual .lib occurrence")
    for active in inputs.active_cards:
        original = _original(active.source, cards)
        head = original.body.split(maxsplit=1)[0].casefold() if original.body else ""
        if (
            captures[active.source.capture].pdk_relative is not None
            and head not in INCLUDE_HEADS
            and active.card.body != original.body
        ):
            raise NativeRequestError("final active card modifies protected PDK content")
    for assignment in inputs.assignments:
        _original(assignment.source, cards)
    owned_parameters = _owned_parameters(inputs, captures)
    analysis = _validate_controls(request, inputs, captures, owned_parameters)
    _validate_coverage(inputs, captures, cards)
    originals = sorted(
        [
            {"identity": c.identity, "sha256": c.sha256, "pdk_relative": c.pdk_relative}
            for c in captures.values()
        ],
        key=lambda row: row["identity"] or "",
    )
    input_digest = canonical_hash({"root": inputs.root_capture, "captures": originals})
    model_digest = canonical_hash([row for row in originals if row["pdk_relative"] is not None])
    key = canonical_json(
        {
            "version": DERIVATION_VERSION,
            "circuit": request.circuit_id.casefold(),
            "family": request.family_id.casefold(),
            "inputs": input_digest,
            "assignments": _assignment_keys(inputs.assignments, captures, owned_parameters),
            "profile": request.profile,
            "mode": request.mode,
            "section": binding.section.casefold(),
            "sample_index": request.sample_index,
        }
    ).decode("ascii")
    digest = canonical_hash(
        {"domain": DERIVATION_VERSION, "root_seed": request.root_seed, "key": key}
    )
    seed = 1 + int(digest, 16) % SEED_MAX
    population = sorted([asdict(leaf) for leaf in inputs.coverage], key=canonical_json)
    dependencies = tuple(
        sorted(identity for identity in captures if identity != inputs.root_capture)
    )
    return ValidatedSample(
        request,
        key,
        seed,
        input_digest,
        model_digest,
        canonical_hash(population),
        inputs.coverage,
        analysis,
        inputs.hierarchy_revision,
        dependencies,
    )


def validate_seed_collisions(samples: Sequence[ValidatedSample]) -> None:
    seen: dict[int, tuple[int, str]] = {}
    for sample in samples:
        identity = (sample.request.root_seed, sample.sample_key)
        previous = seen.setdefault(sample.effective_seed, identity)
        if previous != identity:
            raise NativeRequestError(
                f"native effective-seed collision at {sample.effective_seed}: {previous!r} and {identity!r}"
            )


def validate_paths(paths: NativePaths, token: str) -> None:
    """Pure native Windows/POSIX spelling check; no host path conversion."""
    if not _IDENTIFIER.fullmatch(token) or token.casefold() in {
        "con",
        "prn",
        "aux",
        "nul",
        *(f"com{i}" for i in range(1, 10)),
        *(f"lpt{i}" for i in range(1, 10)),
    }:
        raise NativeCaseError("paths", "run token must be a generated portable basename")
    windows = isinstance(paths.cwd, PureWindowsPath)
    if not paths.cwd.is_absolute() or ".." in paths.cwd.parts:
        raise NativeCaseError("paths", "native cwd must be an absolute run directory")
    for path, suffix in (
        (paths.electrical_input, ".input.cir"),
        (paths.prepared_driver, ".setup.cir"),
        (paths.executed_driver, ".cir"),
        (paths.raw, ".raw"),
        (paths.log, ".log"),
    ):
        if (
            isinstance(path, PureWindowsPath) != windows
            or path.parent != paths.cwd
            or path.name != token + suffix
        ):
            raise NativeCaseError(
                "paths", "native artifacts must have distinct token basenames in the supplied cwd"
            )


def driver_bytes(seed: int, paths: NativePaths, token: str) -> bytes:
    validate_paths(paths, token)
    if type(seed) is not int or not 1 <= seed <= SEED_MAX:
        raise NativeCaseError("seed", "effective ngspice seed is outside the verified domain")
    return (
        "Native PDK initialization\n.control\nset ngbehavior=hsa\nset ng_nomodcheck\n"
        f"setseed {seed}\nsource {paths.electrical_input.name}\nrun\nwrite {paths.raw.name}\n"
        "quit\n.endc\n.end\n"
    ).encode("ascii")


def _verify_artifact(artifact: ArtifactDigest) -> None:
    if not _DIGEST.fullmatch(artifact.sha256):
        raise NativeCaseError("artifact_digest", "expected artifact digest is missing or invalid")
    try:
        actual = sha256_file(artifact.path)
    except OSError as exc:
        raise NativeCaseError(
            "artifact_missing", "required launch artifact is unavailable"
        ) from exc
    if actual != artifact.sha256:
        raise NativeCaseError("artifact_drift", "exact launch artifact bytes have changed")


def _unused(paths: Sequence[Path]) -> None:
    if any(p.exists() or p.is_symlink() for p in paths):
        raise NativeCaseError("artifact_exists", "refusing to overwrite a previous native attempt")


def _validate_run_boundary(paths: NativePaths, dependencies: Sequence[ArtifactDigest]) -> None:
    try:
        cwd = Path(paths.cwd).resolve(strict=True)
        if not cwd.is_dir():
            raise NativeCaseError("paths", "native cwd must be an existing run directory")
        dependency_paths = [d.path.resolve() for d in dependencies]
        artifact_paths = {Path(p).resolve() for p in paths.artifacts}
    except (OSError, RuntimeError) as exc:
        raise NativeCaseError(
            "paths", "could not resolve the existing run directory or its paths"
        ) from exc
    if any(not path.is_relative_to(cwd) for path in dependency_paths):
        raise NativeCaseError(
            "dependencies", "staged dependencies must resolve within the run directory"
        )
    if (
        any(not d.path.is_absolute() for d in dependencies)
        or len(set(dependency_paths)) != len(dependency_paths)
        or artifact_paths.intersection(dependency_paths)
    ):
        raise NativeCaseError(
            "dependencies",
            "dependency paths must be absolute, distinct and separate from run artifacts",
        )


def prepare_launch(
    sample: ValidatedSample,
    *,
    paths: NativePaths,
    token: str,
    electrical_bytes: bytes,
    dependencies: tuple[ArtifactDigest, ...],
    skipped: bool = False,
) -> PreparedLaunch | None:
    """Write exact bytes only at supplied paths; expected failures stay case-local.

    Parent must supply bytes with include references already valid from cwd and
    every rewritten dependency digest within an existing Store-owned run
    directory. Parent creates that directory and calls verify_launch before
    SimRunner.run. The runner copies the submitted setup to executed_driver;
    checking an already-existing copy does not guarantee prelaunch ordering.
    Partial preparation is preserved on failure and never silently reused.
    """
    if skipped:
        return None
    driver = driver_bytes(sample.effective_seed, paths, token)
    if not all(isinstance(p, Path) for p in (paths.cwd, *paths.artifacts)):
        raise NativeCaseError("paths", "preparation requires host-native concrete Paths")
    input_path, setup_path = Path(paths.electrical_input), Path(paths.prepared_driver)
    if not electrical_bytes or not dependencies:
        raise NativeCaseError(
            "input", "electrical bytes and complete staged dependencies are required"
        )
    if {d.original_capture for d in dependencies} != set(sample.dependency_captures):
        raise NativeCaseError(
            "dependencies",
            "rewritten dependency manifest must cover every original dependency and no others",
        )
    _validate_run_boundary(paths, dependencies)
    _unused([Path(p) for p in paths.artifacts])
    for dependency in dependencies:
        _verify_artifact(dependency)
    try:
        atomic_write_bytes(input_path, electrical_bytes, overwrite=False)
        atomic_write_bytes(setup_path, driver, overwrite=False)
    except OSError as exc:
        raise NativeCaseError(
            "preparation_io", "could not prepare immutable native artifacts"
        ) from exc
    return PreparedLaunch(
        paths,
        hashlib.sha256(electrical_bytes).hexdigest(),
        hashlib.sha256(driver).hexdigest(),
        dependencies,
        sample.sample_key,
        sample.effective_seed,
        sample.analysis,
    )


def verify_launch(prepared: PreparedLaunch, *, executed_copy: bool = False) -> None:
    """Verify the run boundary and exact rewritten bytes before submission.

    Parent calls this before SimRunner.run and serializes submission by token;
    this check is not a filesystem lock. Dependency paths are resolved again
    so a symlink redirected outside the run directory is refused.
    ``executed_copy=False`` refuses an existing executed destination; True
    checks an already-existing copy against the prepared driver. It provides
    no postcopy-prelaunch scheduling guarantee: the worker may already run.
    """
    paths = prepared.paths
    _validate_run_boundary(paths, prepared.dependencies)
    _unused([Path(paths.raw), Path(paths.log)])
    artifacts = (
        ArtifactDigest(Path(paths.electrical_input), prepared.input_sha256),
        ArtifactDigest(Path(paths.prepared_driver), prepared.driver_sha256),
        *prepared.dependencies,
    )
    for artifact in artifacts:
        _verify_artifact(artifact)
    if executed_copy:
        _verify_artifact(ArtifactDigest(Path(paths.executed_driver), prepared.driver_sha256))
    else:
        _unused([Path(paths.executed_driver)])


def _dependency_summary(dependencies: Sequence[ArtifactDigest]) -> dict[str, str | int]:
    """Bind the multiset of captured-source identities and rewritten hashes.

    This compact digest preserves multiplicity but excludes local path spelling.
    The full persisted dependencies retain the exact final-file membership.

    Hash the version tag and newline, then each canonical JSON pair followed
    by a newline, sorted by (original_capture, sha256). JSON escapes embedded
    newlines, so records remain unambiguous without serializing a full list.
    """
    version = "pdk-native-staged-dependencies-sha256-v1"
    digest = hashlib.sha256((version + "\n").encode("ascii"))
    for dependency in sorted(dependencies, key=lambda d: (d.original_capture, d.sha256)):
        digest.update(canonical_json((dependency.original_capture, dependency.sha256)))
        digest.update(b"\n")
    return {
        "dependency_count": len(dependencies),
        "dependency_digest": digest.hexdigest(),
        "dependency_digest_version": version,
    }


def provenance(
    request: NativeRequest,
    *,
    sample: ValidatedSample | None = None,
    prepared: PreparedLaunch | None = None,
    simulator: SimulatorFacts | None = None,
    unavailable_reason: str = "not validated",
    error: NativeCaseError | None = None,
    detailed: bool = True,
) -> dict[str, Any]:
    """JSON-ready facts; detailed=False replaces large lists with counts/digests.

    Compact coverage uses the existing population digest without traversing
    the population. Dependencies use stable capture identities and rewritten
    hashes, excluding paths; artifact paths are also omitted. Full detail is
    available through the existing jobs run-field projection. Submission and
    the persisted NativeCaseRecord remain parent-owned.
    """
    validate_request(request)
    requested = asdict(request)
    if request.sample_index is None:
        requested.pop("sample_index")
    result: dict[str, Any] = {
        "origin": "pdk_native",
        "requested": requested,
        "unavailable": {"internal_draws": "simulator-internal random draws are not observed"},
    }
    missing = result["unavailable"]
    if request.sample_index is None:
        missing["sample_index"] = unavailable_reason
    if sample is not None:
        if sample.request != request:
            raise NativeRequestError("provenance request and validated sample disagree")
        section, mm, pr = _MODES[request.mode]
        result["validated"] = {
            "sample_key": sample.sample_key,
            "effective_seed": sample.effective_seed,
            "derivation_version": DERIVATION_VERSION,
            "section": section,
            "controls": {"mc_mm_switch": mm, "mc_pr_switch": pr},
            "input_digest": sample.input_digest,
            "model_digest": sample.model_digest,
            "population_digest": sample.population_digest,
            **(
                {"coverage": [asdict(leaf) for leaf in sample.coverage]}
                if detailed
                else {"coverage_count": len(sample.coverage)}
            ),
            "hierarchy_revision": sample.hierarchy_revision,
            "analysis": sample.analysis,
            "pin_manifest_sha256": PIN_MANIFEST_SHA256,
            "profile_identity": {
                "open_pdks_revision": "e6f9c8876da77220403014b116761b0b2d79aab4",
                "open_pdks_version": "1.0.393",
                "skywater_revision": "f70d8ca46961ff92719d8870a18a076370b85f6c",
                "primitive_revision": "f62031a1be9aefe902d6d54cddd6f59b57627436",
                "supported_wrapper": WRAPPER,
                "compatibility": NGBEHAVIOR,
            },
        }
    else:
        for fact in ("sample_key", "effective_seed", "input_digest", "model_digest", "coverage"):
            missing[fact] = unavailable_reason
    if prepared is not None:
        if (
            sample is None
            or prepared.sample_key != sample.sample_key
            or prepared.effective_seed != sample.effective_seed
        ):
            raise NativeRequestError("prepared artifacts and validated sample disagree")
        result["prepared"] = {
            "adapter_version": ADAPTER_VERSION,
            **(
                {"paths": {key: str(value) for key, value in asdict(prepared.paths).items()}}
                if detailed
                else {}
            ),
            "input_sha256": prepared.input_sha256,
            "prepared_driver_sha256": prepared.driver_sha256,
            "expected_executed_driver_sha256": prepared.driver_sha256,
            **(
                {
                    "dependencies": [
                        {
                            "path": str(d.path),
                            "sha256": d.sha256,
                            "original_capture": d.original_capture,
                        }
                        for d in prepared.dependencies
                    ]
                }
                if detailed
                else _dependency_summary(prepared.dependencies)
            ),
            "switches": list(prepared.switches),
            "ngbehavior": prepared.ngbehavior,
            "ng_nomodcheck": True,
        }
    else:
        missing["artifacts"] = unavailable_reason
    facts = asdict(simulator or SimulatorFacts())
    result["simulator"] = {key: value for key, value in facts.items() if value is not None}
    for key, value in facts.items():
        if value is None:
            missing["simulator." + key] = "not observed"
    if error is not None:
        result["error"] = {"code": error.code, "message": str(error)}
    if not detailed and sample is not None:
        result["detail_hint"] = (
            "Full coverage, dependency manifest and artifact paths are available using "
            "jobs(runs, run_fields=['native_statistics']) for this case."
        )
    return result
