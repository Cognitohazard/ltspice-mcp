"""What a session puts on disk, and where — plus who the store says owns it.

The store's whole point is that one object decides the layout, so the layout is
checkable. Two tests are that check from both ends: what a real session
actually creates, and what the ``Store`` API is capable of creating. Adding a
root is then a deliberate edit here rather than a directory that quietly
appears in someone's project. The owner-liveness probe lives here too: it is
the store's answer to "is the process that wrote this record still running?".
"""

from __future__ import annotations

import inspect
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import psutil
import pytest

from ltspice_mcp.lib import store as store_module
from ltspice_mcp.lib.deck_staging import resolve_experiment_paths
from ltspice_mcp.lib.store import Store, StoreError
from ltspice_mcp.state import SessionState
from ltspice_mcp.tools.analyze import AnalyzeResultsInput, handle_analyze_results
from ltspice_mcp.tools.experiments import RunExperimentsInput, handle_run_experiments
from tests.conftest import fake_simulator

# Every directory the working-directory store may hold, and what it is for.
# A new entry here is a new place the server writes; a missing one means
# something created a root nobody declared.
DECLARED_ROOTS: dict[str, str] = {
    "experiments": "job records plus the request and per-circuit indexes",
    "runs": "one directory per job: its staged decks and its raw/log artifacts",
    "results": "immutable analyze_results sets and the files they point at",
    "detached": "hand-off files and console logs for per-job detached owners",
    "renders": "schematic images",
    "verify": "verify_circuit exports and scratch",
    "edit-exports": "edit_schematic exports",
    "locks": "cross-process store locks",
}

# The store's own version stamp, which is a file rather than a directory.
DECLARED_FILES: frozenset[str] = frozenset({"store.json"})


class FakeNonLTspice:
    """Any simulator that is not LTspice, so artifact routing stays in-store."""


def _normalize(store: Store, path: Path, job_id: str) -> str:
    """One on-disk path as a shape, with the ids that vary spelled generically."""
    parts = [part.replace(job_id, "{job_id}") for part in path.relative_to(store.root).parts]
    return "/".join(parts)


def _tree(store: Store, job_id: str) -> set[str]:
    return {
        _normalize(store, path, job_id)
        for path in store.root.rglob("*")
        if path.is_dir() or path.suffix == ".json"
    }


@pytest.mark.asyncio
async def test_a_finished_run_creates_only_declared_roots(
    state_with_sim: SessionState,
    work_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Run one experiment and read one result set, then inventory the store.

    This is the end that catches a writer reaching past the Store: it walks
    what is really there rather than what the API says it would create.
    """
    # Keep the user-global recent index inside the test's own directory: it is
    # the one store record that lives outside the working directory.
    monkeypatch.setenv("LTSPICE_MCP_HOME", str(work_dir / "recent-state"))
    fake_simulator(monkeypatch)
    deck = work_dir / "dut.cir"
    deck.write_text("V1 in 0 1\nR1 in 0 1k\n.op\n.end\n")

    submitted = await handle_run_experiments(
        RunExperimentsInput.model_validate(
            {
                "request_id": "store-layout",
                "circuits": [{"path": str(deck), "id": "dut"}],
                "execution": {"wait_s": 5.0},
            }
        ),
        state_with_sim,
    )
    receipt = submitted.structured_content
    assert receipt is not None, submitted.content[0].text
    job_id = receipt["job_id"]

    analyzed = await handle_analyze_results(
        AnalyzeResultsInput.model_validate(
            {
                "sources": [{"job_id": job_id, "label": "dut"}],
                "recipes": [{"key": "vin", "metric": "value", "expr": "V(in)"}],
            }
        ),
        state_with_sim,
    )
    assert analyzed.structured_content is not None

    tree = _tree(Store(work_dir), job_id)
    top_level = {entry.split("/")[0] for entry in tree}

    undeclared = top_level - set(DECLARED_ROOTS) - DECLARED_FILES
    assert not undeclared, (
        f"the session created roots nothing declares: {sorted(undeclared)}. "
        "Add them to DECLARED_ROOTS (and to the layout in lib/store.py) if they "
        "are meant to exist."
    )
    # The version stamp is written by the first durable write, not at startup.
    assert "store.json" in tree

    # The parts a run must have produced, spelled as shapes so the assertion
    # says what the layout IS rather than merely that nothing new appeared.
    assert {
        "experiments",
        "experiments/{job_id}.json",
        "experiments/by-request",
        "experiments/by-circuit",
        "runs",
        "runs/{job_id}",
        "runs/{job_id}/staged",
        "runs/{job_id}/staged/dut",
        "results",
        "locks",
    } <= tree

    # The job's own artifacts are inside its own directory, not loose in the
    # shared runs folder under a job-id-prefixed name. That folder is one per
    # box, so every session's results used to sit in it together.
    run_dir = Store(work_dir).run_dir(job_id)
    assert sorted(path.suffix for path in run_dir.glob("*.*")) == [".log", ".raw"]
    assert not list(Store(work_dir).runs_root().glob("*.raw"))


# Placeholder arguments for every path-returning member of Store, so each can
# be called and asked which root it lands in. A member missing from this table
# fails the test below by name: that is the prompt to decide, deliberately,
# which root a new path belongs to.
_PATH_MEMBERS: dict[str, Any] = {
    "root": (),
    "manifest": (),
    "experiments_dir": (),
    "job_record": ("exp_1",),
    "request_index": ("some-request-id",),
    "circuit_index_dir": (Path("/tmp/deck.cir"),),
    "circuit_index": (Path("/tmp/deck.cir"), "exp_1"),
    "cancellation": ("exp_1",),
    "lock": ("a-lock",),
    "request_lock": ("some-request-id",),
    "cancellation_lock": ("exp_1",),
    "runs_root": (),
    "run_dir": ("exp_1",),
    "staged_deck_root": ("exp_1", "dut"),
    "detached_dir": (),
    "detached_request": ("some-request-id", "0123abcd"),
    "detached_receipt": ("some-request-id", "0123abcd"),
    "detached_log": ("some-request-id", "0123abcd"),
    "results_dir": (),
    "result_set": ("rs_" + "0" * 32,),
    "result_artifacts": ("rs_" + "0" * 32,),
    "renders_dir": (),
    "verify_artifact": ("export",),
    "edit_export": ("build_1",),
}

# Paths that deliberately live outside the working-directory store, and why.
_OUTSIDE_THE_STORE: dict[str, str] = {
    "circuit_sidecar": "belongs to the user's circuit, not to a session",
    "circuit_exports": "a receipt's provenance names it; it outlives the session",
    "circuit_plots": "a plot belongs beside the circuit it was made from",
    "artifact_base": "returns the routing decision, not a path",
}


def test_every_store_path_lands_in_a_declared_root(tmp_path: Path) -> None:
    """The other end: what the API can create, whether or not a test drove it.

    A root that no test happens to exercise (a render, an edit export) still
    has to be declared, and a new path method has to say where it belongs.
    """
    public = {
        name
        for name, member in inspect.getmembers(Store)
        if not name.startswith("_")
        and (isinstance(member, property) or inspect.isfunction(member))
        and name not in {"ensure_root", "store_version_on_disk"}
    }
    untabled = public - set(_PATH_MEMBERS) - set(_OUTSIDE_THE_STORE)
    assert not untabled, (
        f"new Store members with no declared root: {sorted(untabled)}. "
        "Add each to _PATH_MEMBERS with sample arguments, or to "
        "_OUTSIDE_THE_STORE with the reason it lives elsewhere."
    )

    store = Store(tmp_path)
    roots: set[str] = set()
    for name, args in _PATH_MEMBERS.items():
        member = getattr(Store, name)
        path = (
            getattr(store, name) if isinstance(member, property) else getattr(store, name)(*args)
        )
        assert isinstance(path, Path)
        relative = path.relative_to(store.root)
        if relative.parts:
            roots.add(relative.parts[0])

    assert roots == set(DECLARED_ROOTS) | DECLARED_FILES


@pytest.mark.parametrize("circuit_id", ["../escape", "a/b", "", "."])
def test_staged_deck_root_refuses_a_circuit_id_that_is_not_one_segment(
    tmp_path: Path, circuit_id: str
) -> None:
    """Every other caller-supplied segment is validated; this one was not.

    ``circuit_id`` reaches the store from the experiment request, so a value
    that walks out of the job's directory has to be refused where the path is
    built, not wherever someone remembers to check.
    """
    with pytest.raises(StoreError):
        Store(tmp_path).staged_deck_root("exp_1", circuit_id)


def test_deck_staging_reads_its_staging_root_from_the_store(tmp_path: Path) -> None:
    """One formula for where a staged deck lands, not two that agree today."""
    paths = resolve_experiment_paths(tmp_path, "exp_staging", "dut", FakeNonLTspice)
    assert paths.staging_root == Store(tmp_path).staged_deck_root(
        "exp_staging", "dut", FakeNonLTspice
    )


_FOREIGN_PID = 999_999_999


class TestOwnerLivenessUnknown:
    """A probe that could not reach an answer must not read as "owner dead".

    Sessions share a working directory, and "the owner is gone" is exactly the
    reading that licenses one session to rewrite another's running job as
    interrupted. A psutil call that raises is not evidence of anything, so the
    record stands as the owning server wrote it. The job records this probe
    guards are exercised in tests/test_experiment_job.py; what is pinned here
    is the probe's own three answers.
    """

    @staticmethod
    def _break_the_probe(monkeypatch: Any) -> None:
        def boom(pid: int) -> bool:
            raise OSError("process table unavailable")

        monkeypatch.setattr(store_module.psutil, "pid_exists", boom)

    def test_probe_reports_unknown_rather_than_dead(self, monkeypatch: Any) -> None:
        self._break_the_probe(monkeypatch)
        liveness = store_module.owner_liveness(_FOREIGN_PID)
        assert liveness is store_module.OwnerLiveness.UNKNOWN
        assert liveness.is_dead is False

    def test_probe_still_answers_dead_and_alive(self, monkeypatch: Any) -> None:
        """The two real answers must survive the third one being added."""
        monkeypatch.setattr(store_module.psutil, "pid_exists", lambda pid: False)
        assert store_module.owner_liveness(_FOREIGN_PID) is store_module.OwnerLiveness.DEAD
        monkeypatch.undo()
        # A real live process for the ALIVE answer: the probe now also asks
        # what the process is doing, and a pid that exists only in a stub has
        # nothing to answer with.
        assert store_module.owner_liveness(os.getppid()) is store_module.OwnerLiveness.ALIVE
        # A record with no pid predates pid tracking; recovering those jobs is
        # the behaviour this probe was added to, not something it takes away.
        assert store_module.owner_liveness(None) is store_module.OwnerLiveness.DEAD


class TestOwnerLivenessExitedProcess:
    """A process that has exited is not an owner, collected or not.

    An exited child keeps its pid in the process table until the process that
    started it collects it, so the pid alone still reads as "there". A job
    whose owner stopped there has nobody supervising it, and reading that as
    running is what keeps the restart reconciliation from ever running.
    """

    def test_probe_reports_an_exited_uncollected_process_as_dead(self) -> None:
        child = subprocess.Popen([sys.executable, "-c", ""])
        try:
            deadline = time.monotonic() + 30
            # Deliberately never poll() or wait() here: either would collect
            # the child and remove the state under test.
            while psutil.Process(child.pid).status() != psutil.STATUS_ZOMBIE:
                if time.monotonic() >= deadline:
                    pytest.fail("the child process never exited")
                time.sleep(0.02)
            assert psutil.pid_exists(child.pid)
            assert store_module.owner_liveness(child.pid) is store_module.OwnerLiveness.DEAD
        finally:
            child.wait(timeout=30)
