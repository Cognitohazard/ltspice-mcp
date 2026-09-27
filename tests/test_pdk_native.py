"""Pure checks for the bounded native adapter; no simulator process is started."""

import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path, PureWindowsPath

import pytest

from ltspice_mcp.lib import pdk_native as native
from ltspice_mcp.lib.spice_lex import lex


def test_unknown_profile_is_request_rejection():
    with pytest.raises(native.NativeRequestError, match="profile"):
        native.validate_request(native.NativeRequest("amp", "native", "unknown", "nominal", 1, 0))


def request(mode="nominal", index: int | None = 0):
    return native.NativeRequest("amp", "native", native.PROFILE, mode, 17, index)


@pytest.fixture
def inputs(monkeypatch):
    """Synthetic acquisition bytes exercise validation without a PDK install.

    Only acquisition data is replaced; hashing, occurrence resolution, controls,
    geometry, seed derivation and artifact verification all run normally.
    """
    root = native.OriginalCapture(
        "bench/main.cir", b'* bench\n.lib "models.lib" tt\nR1 d 0 1k\n.op\n.end\n'
    )
    library = native.OriginalCapture(
        "pdk/library",
        b".lib tt\n.param mc_mm_switch=0 mc_pr_switch=0\n.endl tt\n",
        native.ENTRYPOINT,
    )
    wrapper = native.OriginalCapture(
        "pdk/nmos",
        (
            f".subckt {native.WRAPPER} d g s b\n"
            f"m{native.WRAPPER} d g s b {native.MODEL} l={{l}} w={{w}}\n"
            f".model {native.MODEL}.0 nmos level=54\n"
            ".ends\n"
        ).encode(),
        f"libs.ref/sky130_fd_pr/spice/{native.WRAPPER}__tt.pm3.spice",
    )
    critical = native.OriginalCapture(
        "pdk/critical", b".param variation=1\n", "libs.tech/ngspice/parameters/critical.spice"
    )
    captures = (root, library, wrapper, critical)
    pins = {c.pdk_relative: c.sha256 for c in captures if c.pdk_relative}
    monkeypatch.setattr(native, "profile_pins", lambda: pins)
    active = tuple(
        native.ActiveCard(native.Occurrence(c.identity, card.line_start, scope=card.scope), card)
        for c in captures
        for card in lex(c.content.decode()).cards
        if not card.trailing
    )
    leaf = native.MosCoverage(
        ("Xtop", "Xn", "m" + native.WRAPPER),
        native.Occurrence(wrapper.identity, 2, scope=(native.WRAPPER,)),
        native.Occurrence(wrapper.identity, 1),
        (native.Occurrence(wrapper.identity, 3, scope=(native.WRAPPER,)),),
        1e-6,
        0.15e-6,
        1e-6,
        1,
        1,
        1,
    )
    return native.CaseInputs(
        root.identity,
        captures,
        active,
        (native.LibraryBinding(native.Occurrence(root.identity, 2), library.identity, "tt"),),
        (leaf,),
        (),
        "hierarchy-v1",
        True,
        True,
    )


def test_packaged_manifest_is_exact_acquired_map():
    pins = native.profile_pins()
    assert len(pins) == 821
    assert native.ENTRYPOINT in pins
    assert all(not Path(path).is_absolute() for path in pins)


def test_validated_nominal_and_passive_bench(inputs):
    sample = native.validate_sample(request(), inputs)
    assert sample.analysis == ".op"
    assert 1 <= sample.effective_seed <= native.SEED_MAX
    assert sample.coverage == inputs.coverage


@pytest.mark.parametrize("protected_index", [1, 2, 3])
def test_every_active_protected_capture_is_pinned(inputs, protected_index):
    captures = list(inputs.captures)
    captures[protected_index] = replace(
        captures[protected_index], content=captures[protected_index].content + b"* drift\n"
    )
    with pytest.raises(native.NativeCaseError, match="pin"):
        native.validate_sample(request(), replace(inputs, captures=tuple(captures)))


def test_unknown_protected_source_rejected(inputs):
    extra = native.OriginalCapture("pdk/unpinned", b".param seed=1", "libs.tech/ngspice/new.spice")
    with pytest.raises(native.NativeCaseError, match="pin"):
        native.validate_sample(request(), replace(inputs, captures=(*inputs.captures, extra)))


def test_spoofed_wrapper_name_is_not_coverage(inputs):
    clone = replace(inputs.captures[2], identity="bench/spoof", pdk_relative=None)
    leaf = inputs.coverage[0]
    spoof = replace(
        leaf,
        wrapper=replace(leaf.wrapper, capture=clone.identity),
        source=replace(leaf.source, capture=clone.identity),
        models=tuple(replace(m, capture=clone.identity) for m in leaf.models),
    )
    active_spoof = tuple(
        native.ActiveCard(native.Occurrence(clone.identity, c.line_start, scope=c.scope), c)
        for c in lex(clone.content.decode()).cards
    )
    with pytest.raises(native.NativeCaseError, match="protected"):
        native.validate_sample(
            request(),
            replace(
                inputs,
                captures=(*inputs.captures, clone),
                coverage=(spoof,),
                active_cards=(*inputs.active_cards, *active_spoof),
            ),
        )


@pytest.mark.parametrize("change", ["line", "model", "scope", "duplicate", "empty", "incomplete"])
def test_coverage_requires_actual_occurrences(inputs, change):
    leaf = inputs.coverage[0]
    coverage = (leaf,)
    if change == "line":
        coverage = (replace(leaf, wrapper=replace(leaf.wrapper, line=2)),)
    elif change == "model":
        coverage = (replace(leaf, models=(leaf.source,)),)
    elif change == "scope":
        coverage = (replace(leaf, source=replace(leaf.source, scope=())),)
    elif change == "duplicate":
        coverage = (leaf, leaf)
    elif change == "empty":
        coverage = ()
    with pytest.raises(native.NativeCaseError):
        native.validate_sample(
            request(), replace(inputs, coverage=coverage, coverage_complete=change != "incomplete")
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("width_m", None),
        ("length_m", 0),
        ("width_m", float("nan")),
        ("length_m", float("inf")),
        ("m", 2),
        ("mult", None),
        ("nf", 2),
        ("scale", 1),
        ("m", True),
    ],
)
def test_final_geometry_is_bounded(inputs, field, value):
    leaf = replace(inputs.coverage[0], **{field: value})
    with pytest.raises(native.NativeCaseError, match=r"geometry|scale"):
        native.validate_sample(request(), replace(inputs, coverage=(leaf,)))


def with_root(inputs, text):
    root = replace(inputs.captures[0], content=text.encode())
    cards = tuple(
        native.ActiveCard(native.Occurrence(root.identity, c.line_start, scope=c.scope), c)
        for c in lex(text).cards
        if not c.trailing
    )
    return replace(
        inputs,
        captures=(root, *inputs.captures[1:]),
        active_cards=(
            *cards,
            *(a for a in inputs.active_cards if a.source.capture != root.identity),
        ),
    )


@pytest.mark.parametrize(
    "directive",
    [
        ".control\nsetseed 1\n.endc",
        ".step param x 1 2 1",
        ".alter",
        ".dc V1 0 1 .1",
        ".param seed=2",
        ".option seed = 2",
        ".param mc_mm_switch=1",
        ".option scale=1",
        ".ac dec 1 1 10",
    ],
)
def test_control_and_analysis_ownership(inputs, directive):
    changed = with_root(inputs, '* bench\n.lib "models.lib" tt\n' + directive + "\n.op\n.end\n")
    with pytest.raises(native.NativeRequestError):
        native.validate_sample(request(), changed)


def test_combined_override_must_be_root_after_binding(inputs):
    # Process models differ by acquisition identity, while this synthetic fixture
    # uses the same small content to test the occurrence and control mechanism.
    model = replace(
        inputs.captures[2], pdk_relative=f"libs.ref/sky130_fd_pr/spice/{native.WRAPPER}.pm3.spice"
    )
    # The fixture's acquisition table is a mutable test data map.
    pins = native.profile_pins()
    pins[model.pdk_relative] = model.sha256
    changed = replace(
        inputs,
        captures=(*inputs.captures[:2], model, inputs.captures[3]),
        library_bindings=(replace(inputs.library_bindings[0], section="mc"),),
    )
    changed = with_root(
        changed, '* bench\n.lib "models.lib" mc\n.param mc_mm_switch=1\n.op\n.end\n'
    )
    sample = native.validate_sample(request("combined"), changed)
    assert native.provenance(request("combined"), sample=sample)["validated"]["controls"] == {
        "mc_mm_switch": 1,
        "mc_pr_switch": 1,
    }
    changed = with_root(
        changed, '* bench\n.param mc_mm_switch=1\n.lib "models.lib" mc\n.op\n.end\n'
    )
    changed = replace(
        changed,
        library_bindings=(
            replace(changed.library_bindings[0], source=native.Occurrence("bench/main.cir", 3)),
        ),
    )
    with pytest.raises(native.NativeRequestError, match="after"):
        native.validate_sample(request("combined"), changed)


@pytest.mark.parametrize("change", ["missing", "live", "root", "binding", "index"])
def test_incomplete_inputs_cannot_produce_seed(inputs, change):
    if change == "missing":
        inputs = replace(inputs, captures=inputs.captures[:1])
    elif change == "live":
        inputs = replace(inputs, closure_complete=False)
    elif change == "root":
        inputs = replace(inputs, root_capture="missing")
    elif change == "binding":
        inputs = replace(inputs, library_bindings=())
    with pytest.raises(native.NativeCaseError):
        native.validate_sample(request(index=None if change == "index" else 0), inputs)


def test_assignments_use_canonical_original_occurrences(inputs):
    first = native.FinalAssignment(
        "instance", native.Occurrence("bench/main.cir", 3), "value", 2000, ("R1",)
    )
    second = native.FinalAssignment(
        "parameter", native.Occurrence("bench/main.cir", 3), "temp", 27
    )
    one = native.validate_sample(request(), replace(inputs, assignments=(first, second)))
    two = native.validate_sample(
        replace(request(), circuit_id="AMP", family_id="NATIVE"),
        replace(
            inputs,
            captures=tuple(reversed(inputs.captures)),
            assignments=(second, replace(first, instance=("r1",), field="VALUE")),
        ),
    )
    assert one.sample_key == two.sample_key
    assert one.effective_seed == two.effective_seed
    different = native.validate_sample(
        request(), replace(inputs, assignments=(replace(first, value=3000), second))
    )
    assert one.sample_key != different.sample_key
    assert "staged" not in one.sample_key


def test_selected_sample_and_order_independence(inputs):
    samples = [native.validate_sample(request(index=i), inputs) for i in range(10)]
    assert native.validate_sample(request(index=7), inputs) == samples[7]
    native.validate_seed_collisions(list(reversed(samples)))
    native.validate_seed_collisions([samples[0], samples[0]])
    with pytest.raises(native.NativeRequestError, match="collision"):
        native.validate_seed_collisions(
            [samples[0], replace(samples[1], effective_seed=samples[0].effective_seed)]
        )


@pytest.mark.parametrize("field", ["mc_mm_switch", "mc_pr_switch", "seed", "scale"])
def test_assignments_cannot_own_profile_controls(inputs, field):
    assignment = native.FinalAssignment(
        "parameter", native.Occurrence("bench/main.cir", 3), field, 1
    )
    with pytest.raises(native.NativeRequestError, match="assignment"):
        native.validate_sample(request(), replace(inputs, assignments=(assignment,)))


def test_family_ownership():
    native.validate_family_ownership(["native"], caller_random=False)
    for families, random in [(["native"], True), (["one", "two"], False)]:
        with pytest.raises(native.NativeRequestError):
            native.validate_family_ownership(families, caller_random=random)


@pytest.mark.parametrize("bad", [True, -1, 2**63, 1.5])
def test_seed_request_is_strict(bad):
    with pytest.raises(native.NativeRequestError):
        native.validate_request(replace(request(), root_seed=bad))


def paths_at(cwd):
    return native.NativePaths(
        cwd,
        cwd / "token.input.cir",
        cwd / "token.setup.cir",
        cwd / "token.cir",
        cwd / "token.raw",
        cwd / "token.log",
    )


def test_windows_paths_with_spaces_use_only_generated_relative_names():
    paths = paths_at(PureWindowsPath("C:/simulation workspace/run directory"))
    driver = native.driver_bytes(1, paths, "token")
    assert b"source token.input.cir\nrun\nwrite token.raw" in driver
    assert b"C:" not in driver and b"workspace" not in driver
    assert driver.index(b"ngbehavior=hsa") < driver.index(b"setseed") < driver.index(b"source")
    native.driver_bytes(native.SEED_MAX, paths, "token")
    with pytest.raises(native.NativeCaseError):
        native.driver_bytes(1, replace(paths, raw=paths.cwd / "other.raw"), "token")
    with pytest.raises(native.NativeCaseError):
        native.driver_bytes(1, paths, "token\nquit")


@pytest.fixture
def preparation(inputs, tmp_path):
    sample = native.validate_sample(request(), inputs)
    run_dir = tmp_path / "run directory"
    run_dir.mkdir()
    paths = paths_at(run_dir)
    dependencies = []
    for index, identity in enumerate(sample.dependency_captures):
        dep = run_dir / f"rewritten dependency {index}.spice"
        dep.write_bytes(b"* rewritten\r\n.param x=1\r\n")
        dependencies.append(
            native.ArtifactDigest(dep, hashlib.sha256(dep.read_bytes()).hexdigest(), identity)
        )
    electrical = b"* binary cp1252 \x96\r\n.op\r\n.end\r\n"
    return sample, paths, tuple(dependencies), electrical


@pytest.fixture
def prepared(preparation):
    sample, paths, dependencies, electrical = preparation
    launch = native.prepare_launch(
        sample,
        paths=paths,
        token="token",
        electrical_bytes=electrical,
        dependencies=dependencies,
    )
    assert launch is not None
    return sample, launch, electrical


def test_exact_binary_bytes_and_distinct_driver_identities(prepared):
    sample, launch, electrical = prepared
    assert Path(launch.paths.electrical_input).read_bytes() == electrical
    assert launch.input_sha256 == hashlib.sha256(electrical).hexdigest()
    assert launch.switches == ("-n",)
    native.verify_launch(launch)
    driver = Path(launch.paths.prepared_driver).read_bytes()
    Path(launch.paths.executed_driver).write_bytes(driver)
    native.verify_launch(launch, executed_copy=True)
    descriptor = native.provenance(request(), sample=sample, prepared=launch)
    assert "prepared" in descriptor and "executed" not in descriptor
    assert "submitted_at" not in descriptor
    assert json.loads(json.dumps(descriptor))["prepared"]["paths"]["executed_driver"].endswith(
        "token.cir"
    )


@pytest.mark.parametrize("artifact", ["input", "setup", "dependency", "executed"])
def test_launch_refuses_every_artifact_drift(prepared, artifact):
    _, launch, _ = prepared
    if artifact == "input":
        path = Path(launch.paths.electrical_input)
    elif artifact == "setup":
        path = Path(launch.paths.prepared_driver)
    elif artifact == "dependency":
        path = launch.dependencies[0].path
    else:
        path = Path(launch.paths.executed_driver)
    path.write_bytes(b"drift")
    with pytest.raises(native.NativeCaseError, match="bytes"):
        native.verify_launch(launch, executed_copy=artifact == "executed")


def test_previous_attempt_is_never_overwritten(prepared):
    sample, launch, electrical = prepared
    before = Path(launch.paths.prepared_driver).read_bytes()
    with pytest.raises(native.NativeCaseError, match="overwrite"):
        native.prepare_launch(
            sample,
            paths=launch.paths,
            token="token",
            electrical_bytes=electrical,
            dependencies=launch.dependencies,
        )
    assert Path(launch.paths.prepared_driver).read_bytes() == before
    Path(launch.paths.raw).write_bytes(b"old output")
    with pytest.raises(native.NativeCaseError, match="overwrite"):
        native.verify_launch(launch)


def test_skipped_cases_never_prepare(prepared):
    sample, launch, _ = prepared
    assert (
        native.prepare_launch(
            sample,
            paths=launch.paths,
            token="unsafe\n",
            electrical_bytes=b"",
            dependencies=(),
            skipped=True,
        )
        is None
    )


def test_missing_facts_are_absent_with_reasons():
    descriptor = native.provenance(
        request(index=None),
        unavailable_reason="input resolution failed",
        error=native.NativeCaseError("input", "unavailable"),
    )
    assert "validated" not in descriptor and "prepared" not in descriptor
    assert "sample_index" not in descriptor["requested"]
    assert "effective_seed" not in descriptor
    assert descriptor["unavailable"]["effective_seed"] == "input resolution failed"
    assert descriptor["simulator"] == {}
    assert descriptor["error"]["code"] == "input"


def test_coverage_occurrences_must_be_active(inputs):
    leaf = inputs.coverage[0]
    inactive = replace(
        inputs, active_cards=tuple(a for a in inputs.active_cards if a.source != leaf.models[0])
    )
    with pytest.raises(native.NativeCaseError, match="active"):
        native.validate_sample(request(), inactive)


def test_declared_binding_section_must_match_final_library_card(inputs):
    changed = with_root(inputs, '* bench\n.lib "models.lib" ss\n.op\n.end\n')
    with pytest.raises(native.NativeCaseError, match="binding"):
        native.validate_sample(request(), changed)


def test_dependency_manifest_must_cover_all_original_dependencies(prepared):
    sample, launch, electrical = prepared
    other = paths_at(Path(launch.paths.cwd) / "second")
    with pytest.raises(native.NativeCaseError, match="dependenc"):
        native.prepare_launch(
            sample,
            paths=other,
            token="token",
            electrical_bytes=electrical,
            dependencies=launch.dependencies[:1],
        )


def test_final_protected_model_cannot_differ_from_captured_original(inputs):
    model_source = inputs.coverage[0].models[0]
    active = tuple(
        replace(a, card=replace(a.card, body=a.card.body.replace("level=54", "level=1")))
        if a.source == model_source
        else a
        for a in inputs.active_cards
    )
    with pytest.raises(native.NativeRequestError, match="protected"):
        native.validate_sample(request(), replace(inputs, active_cards=active))


@pytest.mark.parametrize("directive", [".param variation=2", ".option ng_nomodcheck"])
def test_bench_cannot_override_profile_owned_parameters_or_startup_flags(inputs, directive):
    changed = with_root(inputs, '* bench\n.lib "models.lib" tt\n' + directive + "\n.op\n.end\n")
    with pytest.raises(native.NativeRequestError):
        native.validate_sample(request(), changed)


@pytest.mark.parametrize(
    ("mode", "section", "controls"),
    [
        ("mismatch", "tt_mm", {"mc_mm_switch": 1, "mc_pr_switch": 0}),
        ("process", "mc", {"mc_mm_switch": 0, "mc_pr_switch": 1}),
    ],
)
def test_native_modes_preserve_authored_sections(inputs, mode, section, controls):
    if mode == "process":
        model = replace(
            inputs.captures[2],
            pdk_relative=f"libs.ref/sky130_fd_pr/spice/{native.WRAPPER}.pm3.spice",
        )
        native.profile_pins()[model.pdk_relative] = model.sha256
        inputs = replace(inputs, captures=(*inputs.captures[:2], model, inputs.captures[3]))
    inputs = with_root(inputs, f'* bench\n.lib "models.lib" {section}\n.op\n.end\n')
    inputs = replace(
        inputs, library_bindings=(replace(inputs.library_bindings[0], section=section),)
    )
    sample = native.validate_sample(request(mode), inputs)
    descriptor = native.provenance(request(mode), sample=sample)
    assert descriptor["validated"]["controls"] == controls
    assert descriptor["validated"]["section"] == section


def test_profile_rejects_ltspice():
    with pytest.raises(native.NativeRequestError, match="backend"):
        native.validate_request(request(), backend="ltspice")


def test_missing_dependency_is_case_failure(prepared):
    _, launch, _ = prepared
    launch.dependencies[0].path.unlink()
    with pytest.raises(native.NativeCaseError) as caught:
        native.verify_launch(launch)
    assert caught.value.code == "artifact_missing"


def symlink_or_skip(link: Path, target: Path):
    try:
        link.symlink_to(target)
    except OSError:
        if os.name == "nt":
            pytest.skip("Creating symlinks requires Windows symlink privileges")
        raise


@pytest.mark.parametrize("cwd_kind", ["missing", "file"])
def test_preparation_requires_existing_run_directory(preparation, tmp_path, cwd_kind):
    sample, _, dependencies, electrical = preparation
    cwd = tmp_path / "unallocated run"
    if cwd_kind == "file":
        cwd.write_bytes(b"existing file")
    paths = paths_at(cwd)
    with pytest.raises(native.NativeCaseError, match="directory") as caught:
        native.prepare_launch(
            sample,
            paths=paths,
            token="token",
            electrical_bytes=electrical,
            dependencies=dependencies,
        )
    assert caught.value.code == "paths"
    if cwd_kind == "missing":
        assert not cwd.exists()
    else:
        assert cwd.read_bytes() == b"existing file"


@pytest.mark.parametrize("via_symlink", [False, True])
def test_preparation_refuses_dependencies_outside_run_directory(
    preparation, tmp_path, via_symlink
):
    sample, paths, dependencies, electrical = preparation
    original = dependencies[0]
    # A sibling with the same name prefix is not a descendant of cwd.
    outside_dir = tmp_path / "run directory sibling"
    outside_dir.mkdir()
    outside = outside_dir / "external.spice"
    outside.write_bytes(original.path.read_bytes())
    dependency_path = outside
    if via_symlink:
        dependency_path = Path(paths.cwd) / "linked.spice"
        symlink_or_skip(dependency_path, outside)
    dependencies = (replace(original, path=dependency_path), *dependencies[1:])
    with pytest.raises(native.NativeCaseError, match="within"):
        native.prepare_launch(
            sample,
            paths=paths,
            token="token",
            electrical_bytes=electrical,
            dependencies=dependencies,
        )
    assert not any(Path(p).exists() for p in paths.artifacts)


@pytest.mark.parametrize("executed_copy", [False, True])
def test_verification_refuses_redirected_dependency_symlink(preparation, tmp_path, executed_copy):
    sample, paths, dependencies, electrical = preparation
    original = dependencies[0]
    link = Path(paths.cwd) / "linked.spice"
    symlink_or_skip(link, original.path)
    dependencies = (replace(original, path=link), *dependencies[1:])
    launch = native.prepare_launch(
        sample, paths=paths, token="token", electrical_bytes=electrical, dependencies=dependencies
    )
    assert launch is not None
    if executed_copy:
        Path(paths.executed_driver).write_bytes(Path(paths.prepared_driver).read_bytes())
    native.verify_launch(launch, executed_copy=executed_copy)
    outside = tmp_path / "external.spice"
    outside.write_bytes(original.path.read_bytes())
    link.unlink()
    symlink_or_skip(link, outside)
    with pytest.raises(native.NativeCaseError, match="within"):
        native.verify_launch(launch, executed_copy=executed_copy)


def test_nested_dependencies_remain_within_run_directory(preparation):
    sample, paths, dependencies, electrical = preparation
    original = dependencies[0]
    nested = Path(paths.cwd) / "models" / "nested"
    nested.mkdir(parents=True)
    destination = nested / original.path.name
    original.path.rename(destination)
    dependencies = (replace(original, path=destination), *dependencies[1:])
    launch = native.prepare_launch(
        sample, paths=paths, token="token", electrical_bytes=electrical, dependencies=dependencies
    )
    assert launch is not None
    native.verify_launch(launch)


def test_provenance_default_retains_full_shape(prepared):
    sample, launch, _ = prepared
    full = native.provenance(request(), sample=sample, prepared=launch)
    assert set(full) == {
        "origin",
        "requested",
        "unavailable",
        "validated",
        "prepared",
        "simulator",
    }
    assert set(full["validated"]) == {
        "sample_key",
        "effective_seed",
        "derivation_version",
        "section",
        "controls",
        "input_digest",
        "model_digest",
        "population_digest",
        "coverage",
        "hierarchy_revision",
        "analysis",
        "pin_manifest_sha256",
        "profile_identity",
    }
    assert set(full["prepared"]) == {
        "adapter_version",
        "paths",
        "input_sha256",
        "prepared_driver_sha256",
        "expected_executed_driver_sha256",
        "dependencies",
        "switches",
        "ngbehavior",
        "ng_nomodcheck",
    }
    assert full["validated"]["coverage"] == [native.asdict(leaf) for leaf in sample.coverage]
    assert full["prepared"]["dependencies"] == [
        {"path": str(d.path), "sha256": d.sha256, "original_capture": d.original_capture}
        for d in launch.dependencies
    ]
    assert full == native.provenance(request(), sample=sample, prepared=launch, detailed=True)


def test_compact_provenance_preserves_scientific_facts(prepared):
    sample, launch, _ = prepared
    simulator = native.SimulatorFacts("42", "observed build", "a" * 64, "linux")
    full = native.provenance(request(), sample=sample, prepared=launch, simulator=simulator)
    compact = native.provenance(
        request(), sample=sample, prepared=launch, simulator=simulator, detailed=False
    )
    for field in ("origin", "requested", "unavailable", "simulator"):
        assert compact[field] == full[field]
    assert {k: v for k, v in compact["validated"].items() if k != "coverage_count"} == {
        k: v for k, v in full["validated"].items() if k != "coverage"
    }
    summary_fields = {"dependency_count", "dependency_digest", "dependency_digest_version"}
    assert {k: v for k, v in compact["prepared"].items() if k not in summary_fields} == {
        k: v for k, v in full["prepared"].items() if k not in {"dependencies", "paths"}
    }
    assert compact["validated"]["coverage_count"] == len(sample.coverage)
    assert compact["prepared"]["dependency_count"] == len(launch.dependencies)
    assert len(compact["prepared"]["dependency_digest"]) == 64
    assert "jobs(runs, run_fields=['native_statistics'])" in compact["detail_hint"]
    assert len(json.dumps(compact)) < len(json.dumps(full))


def test_compact_provenance_does_not_materialize_omitted_detail(prepared, monkeypatch):
    from typing import cast

    class UniterableCoverage(tuple):
        def __iter__(self):
            raise AssertionError("compact provenance traversed coverage")

    class UnstringifiablePath:
        def __str__(self):
            raise AssertionError("compact provenance stringified a dependency path")

    sample, launch, _ = prepared
    sample = replace(sample, coverage=UniterableCoverage(sample.coverage))
    launch = replace(
        launch,
        dependencies=tuple(
            replace(d, path=cast(Path, UnstringifiablePath())) for d in launch.dependencies
        ),
    )
    original_asdict = native.asdict

    def scalar_asdict(value):
        assert isinstance(value, (native.NativeRequest, native.SimulatorFacts))
        return original_asdict(value)

    monkeypatch.setattr(native, "asdict", scalar_asdict)
    compact = native.provenance(request(), sample=sample, prepared=launch, detailed=False)
    assert compact["validated"]["coverage_count"] == len(sample.coverage)
    assert compact["prepared"]["dependency_count"] == len(launch.dependencies)


def test_compact_dependency_digest_binds_stable_identities_and_exact_hashes(prepared, tmp_path):
    sample, launch, _ = prepared

    def digest(prepared_launch):
        return native.provenance(
            request(), sample=sample, prepared=prepared_launch, detailed=False
        )["prepared"]["dependency_digest"]

    original = digest(launch)
    relocated = tuple(
        replace(d, path=tmp_path / f"relocated-{i}.spice")
        for i, d in enumerate(reversed(launch.dependencies))
    )
    assert digest(replace(launch, dependencies=relocated)) == original
    first, *remaining = launch.dependencies
    assert (
        digest(replace(launch, dependencies=(replace(first, sha256="b" * 64), *remaining)))
        != original
    )
    assert (
        digest(
            replace(
                launch, dependencies=(replace(first, original_capture="pdk/other"), *remaining)
            )
        )
        != original
    )


def test_compact_provenance_preserves_missing_facts():
    error = native.NativeCaseError("input", "missing input")
    full = native.provenance(
        request(index=None), unavailable_reason="input resolution failed", error=error
    )
    compact = native.provenance(
        request(index=None),
        unavailable_reason="input resolution failed",
        error=error,
        detailed=False,
    )
    assert compact == full
    assert "validated" not in compact and "prepared" not in compact


def test_compact_validated_provenance_does_not_invent_prepared_facts(inputs):
    sample = native.validate_sample(request(), inputs)
    compact = native.provenance(
        request(), sample=sample, detailed=False, unavailable_reason="preparation failed"
    )
    assert "prepared" not in compact
    assert compact["unavailable"]["artifacts"] == "preparation failed"
    assert compact["validated"]["population_digest"] == sample.population_digest
    assert compact["validated"]["coverage_count"] == len(sample.coverage)


def test_preparation_tracks_distinct_clones_of_one_original(preparation):
    sample, paths, dependencies, electrical = preparation
    first = dependencies[0]
    clone = Path(paths.cwd) / "selected-clone.spice"
    clone.write_bytes(first.path.read_bytes())
    artifacts = (*dependencies, replace(first, path=clone))
    launch = native.prepare_launch(
        sample,
        paths=paths,
        token="token",
        electrical_bytes=electrical,
        dependencies=artifacts,
    )
    assert launch is not None
    assert launch.dependencies == artifacts
    native.verify_launch(launch)
    compact = native.provenance(request(), sample=sample, prepared=launch, detailed=False)
    assert compact["prepared"]["dependency_count"] == len(artifacts)
    clone.write_bytes(b"changed clone")
    with pytest.raises(native.NativeCaseError, match="bytes"):
        native.verify_launch(launch)


def test_repeated_dependency_path_is_refused_even_with_complete_capture_set(preparation):
    sample, paths, dependencies, electrical = preparation
    with pytest.raises(native.NativeCaseError, match="distinct"):
        native.prepare_launch(
            sample,
            paths=paths,
            token="token",
            electrical_bytes=electrical,
            dependencies=(*dependencies, dependencies[0]),
        )
    assert not any(Path(p).exists() for p in paths.artifacts)
