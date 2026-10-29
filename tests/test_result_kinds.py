"""The stepped LTspice kinds against the recordings that establish them.

Each kind names its recordings; each of those decodes as that kind on every
build that recorded it, and every stepped LTspice raw in the recordings is of
some kind, so a stepped result recorded later needs a kind before it passes.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ltspice_mcp.errors import ResultError
from ltspice_mcp.lib.result_kinds import STEPPED_KINDS, SteppedKind
from ltspice_mcp.state import SessionState
from tests import _ltspice_recorded as rec
from tests._asc_ops import inspect_one
from tests.conftest import SyncApi


def _cases(kind: SteppedKind) -> list[tuple[str, str]]:
    """The (build, raw) pairs a kind names: a case id on every build that has
    its raw, or ``<build>:<case id>`` on that build alone."""
    pairs = []
    for name in kind.recorded:
        build, _, case_id = name.rpartition(":")
        builds = [build] if build else rec.BUILDS
        pairs += [(b, case_id) for b in builds if rec.has(b, f"{case_id}.raw")]
    return pairs


def _stepped_raws() -> list[tuple[str, str]]:
    """Every recorded LTspice raw whose header marks it stepped."""
    found = []
    for build in rec.BUILDS:
        for entry in rec.manifest(build)["cases"].values():
            for output in entry["outputs"]:
                if not output.endswith(".raw"):
                    continue
                stem = output.removesuffix(".raw")
                flags = rec.header_fields(build, stem).get("Flags", [""])[0].split()
                if "stepped" in flags:
                    found.append((build, stem))
    return sorted(found)


@pytest.mark.parametrize("kind", STEPPED_KINDS, ids=lambda kind: kind.name)
def test_each_kind_names_recordings_that_exist(kind: SteppedKind):
    named = {name.rpartition(":")[2] for name in kind.recorded}
    assert {case_id for _, case_id in _cases(kind)} == named


@pytest.mark.parametrize(
    ("kind", "build", "case_id"),
    [(kind, build, case_id) for kind in STEPPED_KINDS for build, case_id in _cases(kind)],
    ids=lambda value: value.name if isinstance(value, SteppedKind) else str(value),
)
def test_a_recording_decodes_as_the_kind_that_names_it(
    kind: SteppedKind, build: str, case_id: str, tmp_path: Path
):
    descriptor = rec.decode(build, case_id, tmp_path).raw.descriptor
    assert descriptor.stepped_kind == kind.name
    assert descriptor.analysis in kind.analyses
    if kind.step_values == "raw":
        assert descriptor.step_variables
        assert {name for step in descriptor.steps for name, _ in step.parameters} == set(
            descriptor.step_variables
        )
    else:
        assert descriptor.step_variables == ()
    assert (descriptor.axis is not None) == (kind.axis == "raw")


@pytest.mark.parametrize(("build", "stem"), _stepped_raws())
def test_every_stepped_recording_is_of_a_kind(build: str, stem: str, tmp_path: Path):
    assert rec.decode(build, stem, tmp_path).raw.descriptor.stepped_kind is not None


#: The recordings of the kinds that keep each step's values in the raw.
POINT_A_STEP = [
    pair for kind in STEPPED_KINDS if kind.step_values == "raw" for pair in _cases(kind)
]


def _step_variable(build: str, case_id: str, scratch: Path) -> str:
    (variable,) = rec.decode(build, case_id, scratch).raw.descriptor.step_variables
    return variable


@pytest.mark.parametrize(("build", "case_id"), POINT_A_STEP)
def test_the_api_reads_a_step_variable_as_a_step_value(
    build: str, case_id: str, state_no_sim: SessionState, work_dir: Path
):
    """Its values are in the steps and it is no signal; asked for as one, it
    says where its values are."""
    variable = _step_variable(build, case_id, work_dir / "decoded")
    loaded = SyncApi(state_no_sim).load_raw(rec.stage_run(build, case_id, work_dir / "run"))
    assert variable not in loaded.signals
    assert len(loaded.steps) == rec.declared_points(build, case_id)
    assert all(variable in step for step in loaded.steps)
    with pytest.raises(ResultError, match="stepped value, not a signal"):
        loaded.trace(variable)


@pytest.mark.parametrize(("build", "case_id"), POINT_A_STEP)
async def test_inspect_lists_no_step_variable_among_the_signals(
    build: str, case_id: str, state_no_sim: SessionState, work_dir: Path
):
    variable = _step_variable(build, case_id, work_dir / "decoded")
    raw = rec.stage_run(build, case_id, work_dir / "run")
    listed = await inspect_one(
        state_no_sim, {"kind": "results", "view": "signals", "path": str(raw)}
    )
    names = [row["name"] for row in listed["signals"]]
    assert names and variable not in names
