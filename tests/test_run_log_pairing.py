"""Which log belongs to a results file read by its path.

The one answer is ``raw_parser.run_log_path``, which ``services.source_for_raw_path``
asks for every reader given a path (``analyze_results``, ``plot_waveform``,
``inspect``, the Python API's ``load_raw``). A run of ``<case>`` writes
``<case>.log`` beside its raw and beside each other raw it writes there
(``<case>.op.raw``, ``<case>.fra_1.raw``), and no log of that raw's own name:
each recorded run's ``written`` list says which suffixes it wrote.
"""

from __future__ import annotations

import ast
import asyncio
import shutil
from pathlib import Path

import pytest

from ltspice_mcp.lib import services
from ltspice_mcp.lib.decoded_raw import unresolved_steps_reason
from ltspice_mcp.lib.raw_parser import OTHER_RAW_KINDS, other_raw_path, run_log_path
from ltspice_mcp.state import SessionState
from tests import _ltspice_recorded as rec
from tests._asc_ops import inspect_one
from tests.conftest import SyncApi, dispatched

#: Suffixes of a results file a recorded run writes whose log is not
#: established, each with why. LTspice writes ``<case>.log.raw`` for some
#: runs, but no recording keeps one, so what it holds is not shown.
NOT_ESTABLISHED = {
    ".log.raw": "no recording keeps one: what it holds, and so its log, is not shown"
}


def _results_suffixes(written: list[str]) -> list[str]:
    return [suffix for suffix in written if suffix.endswith(".raw")]


@pytest.mark.parametrize("build", rec.BUILDS)
def test_every_results_file_a_recorded_run_writes_is_paired_with_the_log_it_writes(build: str):
    """For each results file a recorded run wrote, by its suffix, the log it
    is paired with is one the same run wrote, or its pairing is said not to
    be established."""
    wrong: list[str] = []
    seen: set[str] = set()
    for case_id, entry in rec.manifest(build)["cases"].items():
        written = entry.get("written", [])
        for suffix in _results_suffixes(written):
            seen.add(suffix)
            if suffix in NOT_ESTABLISHED:
                continue
            log = run_log_path(Path(f"case{suffix}"))
            if log.name.removeprefix("case") not in written:
                wrong.append(f"{case_id}: {suffix} -> {log.name}, written {written}")
    assert not wrong, wrong
    # Every kind of other raw the code models is among what the runs wrote,
    # but the frequency response XVII does not run.
    kinds = {f".{kind}.raw" for kind in OTHER_RAW_KINDS}
    assert kinds - seen <= ({".fra_1.raw"} if build == "ltspice17" else set()), seen


@pytest.mark.parametrize("build", rec.BUILDS)
def test_every_recorded_results_file_is_paired_with_the_recorded_log_of_its_run(build: str):
    """A recorded results file whose run's log is kept is paired with that
    log; one whose case kept no log is paired with the name the run wrote it
    under, which its ``written`` list holds."""
    for case_id, entry in rec.manifest(build)["cases"].items():
        for output in entry["outputs"]:
            if not output.endswith(".raw"):
                continue
            log = run_log_path(Path(output))
            assert log.as_posix() == f"{case_id}.log", (output, log)
            assert ".log" in entry.get("written", [".log"]) or rec.has(build, log.as_posix())


def test_the_log_of_each_results_file_is_the_runs_by_name():
    assert run_log_path(Path("a/b.raw")) == Path("a/b.log")
    for kind in OTHER_RAW_KINDS:
        assert run_log_path(other_raw_path(Path("a/b.raw"), kind)) == Path("a/b.log")
        assert run_log_path(Path(f"a/B.{kind.upper()}.RAW")) == Path("a/B.log")
    # A name that is only the kind's suffix keeps its own stem.
    assert run_log_path(Path("a/.op.raw")) == Path("a/.op.log")


@pytest.mark.asyncio
async def test_a_stepped_frequency_response_read_by_its_path_reads_its_steps_on_every_route(
    state_no_sim: SessionState, work_dir: Path
):
    """``raw/fra_step`` on LTspice 26: its ``.fra_1.raw`` read by its path is
    paired with the run's log, and every reader of a path reads its two
    steps."""
    build, case_id = "ltspice26", "raw/fra_step"
    if not rec.has(build, f"{case_id}.fra_1.raw"):
        pytest.skip("raw/fra_step is not recorded on LTspice 26")
    raw = rec.stage_run(build, case_id, work_dir / "run")
    fra = other_raw_path(raw, "fra_1")
    shutil.copyfile(rec.recorded(build, f"{case_id}.fra_1.raw"), fra)
    source = services.source_for_raw_path(fra, state_no_sim)
    assert source.log == raw.with_suffix(".log")
    assert source.log is not None and source.log.is_file()
    assert source.console == raw.with_suffix(".exe.log")

    loaded = await asyncio.to_thread(SyncApi(state_no_sim).load_raw, fra)
    assert loaded.step_count == 2
    plots = await inspect_one(state_no_sim, {"kind": "results", "view": "plots", "path": str(fra)})
    assert all(len(plot["steps"]) == 2 for plot in plots["plots"]), plots["plots"]
    signal = next(name for name in loaded.signals if name.lower().startswith("gain"))
    reply = await dispatched(
        state_no_sim,
        "analyze_results",
        sources=[{"path": str(fra), "label": "dut"}],
        recipes=[{"key": "k", "metric": "signal_stats", "signal": signal}],
        all_steps=True,
        include={"per_run": True},
    )
    assert not reply.get("failures"), reply
    assert len(reply["results"]["k"]["per_run"]["items"]) == 2
    plotted = await dispatched(
        state_no_sim, "plot_waveform", path=str(fra), signals=[signal], open=False
    )
    assert "error" not in plotted, plotted
    assert plotted["traces"], plotted


@pytest.mark.asyncio
async def test_without_the_runs_log_the_reply_says_the_steps_cannot_be_told_apart(
    state_no_sim: SessionState, work_dir: Path
):
    """With no log beside it, the ``.fra_1.raw`` is refused as before: its
    steps cannot be told apart, and the source names the log it looked for."""
    build, case_id = "ltspice26", "raw/fra_step"
    if not rec.has(build, f"{case_id}.fra_1.raw"):
        pytest.skip("raw/fra_step is not recorded on LTspice 26")
    fra = work_dir / "alone" / "fra_step.fra_1.raw"
    fra.parent.mkdir(parents=True)
    shutil.copyfile(rec.recorded(build, f"{case_id}.fra_1.raw"), fra)
    assert services.source_for_raw_path(fra, state_no_sim).log == fra.parent / "fra_step.log"
    reply = await dispatched(
        state_no_sim,
        "analyze_results",
        sources=[{"path": str(fra), "label": "dut"}],
        recipes=[{"key": "k", "metric": "signal_stats", "signal": "gain_1"}],
        all_steps=True,
    )
    assert "Step boundaries are unresolved" in str(reply), reply


@pytest.mark.parametrize("build", rec.BUILDS)
def test_an_operating_point_raw_beside_a_stepped_run_reads_as_before_with_the_runs_log(
    build: str, work_dir: Path
):
    """The ``.op.raw`` beside a stepped transient, paired with the run's log
    as a job's case pairs it, is the same kind and refused for the same
    reason as without it (``raw/step_tran_op_raw``, whose run is
    ``raw/step_tran``'s deck)."""
    folder = work_dir / build
    folder.mkdir()
    op = folder / "x.op.raw"
    shutil.copyfile(rec.recorded(build, "raw/step_tran_op_raw.op.raw"), op)
    alone = rec.decode_file(op, folder / "alone", log=None).raw.descriptor
    shutil.copyfile(rec.recorded(build, "raw/step_tran.log"), run_log_path(op))
    paired = rec.decode_file(op, folder / "paired", log=run_log_path(op)).raw.descriptor
    assert paired.stepped_kind == alone.stepped_kind
    assert len(paired.steps) == len(alone.steps)
    assert unresolved_steps_reason(paired.stepped_kind) == unresolved_steps_reason(
        alone.stepped_kind
    )


#: The scopes that name a log by another file's name, each with what that
#: file is: none of them a results file.
_LOG_NAMERS = {
    ("lib/raw_parser.py", "run_log_path"): "the one answer for a results file",
    ("lib/ngspice_mode.py", "run_batch"): "a probe deck's own log",
    ("lib/controlled_ngspice.py", "run"): "a deck's own log",
    ("lib/sheet_results.py", "files_beside"): "a circuit's own log",
}


def test_only_run_log_path_names_the_log_of_a_results_file():
    """Every place that names a log after another file is the one answer for
    a results file or names a deck's or a circuit's log: a second pairing of
    a results file with a log is how ``.fra_1.raw`` came to be read with a
    log no build writes."""
    from tests._source_scan import modules, scopes

    found: set[tuple[str, str]] = set()
    for path, tree in modules().items():
        for scope, nodes in scopes(tree):
            for node in nodes:
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr in ("with_suffix", "with_name")
                    and any(
                        isinstance(part, ast.Constant)
                        and isinstance(part.value, str)
                        and part.value.endswith(".log")
                        and not part.value.endswith(".exe.log")
                        for arg in node.args
                        for part in ast.walk(arg)
                    )
                ):
                    found.add((path, scope.rsplit(".", 1)[-1]))
    assert found <= set(_LOG_NAMERS), sorted(found - set(_LOG_NAMERS))


def test_a_log_of_the_files_own_name_is_its_runs(tmp_path: Path):
    """A circuit named ``amp.op`` writes ``amp.op.raw`` and ``amp.op.log``: its
    own-named log is its run's, not ``amp.log`` of another circuit beside it.
    No run writes a log of an other raw's own name, so one that is there
    belongs to that file's run."""
    for name in ("amp.op.raw", "amp.op.log", "amp.raw", "amp.log"):
        (tmp_path / name).write_bytes(b"")
    assert run_log_path(tmp_path / "amp.op.raw") == tmp_path / "amp.op.log"
    (tmp_path / "amp.op.log").unlink()
    assert run_log_path(tmp_path / "amp.op.raw") == tmp_path / "amp.log"
