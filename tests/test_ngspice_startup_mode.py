"""Which compatibility mode ngspice reads a deck in.

The server reads a deck in the mode a run passes (``[simulator] ngbehavior``,
read back by ``simulator.run_ngbehavior``). Hierarchy discovery, the
``lib-section-ngspice`` lint rule and the failure classifier all answer from
it, and the server hands it to ngspice as ``-D ngbehavior=<mode>``. ngspice
applies that define before it runs its startup files, so a
``set ngbehavior`` in a user's ``.spiceinit`` replaces it. ngspice looks for
that file in the folder ``SPICE_USERINIT_DIR`` names, then in its working
directory, then in ``HOME``, then in ``USERPROFILE`` (measured with
ngspice-42).

The first tests put a ``.spiceinit`` that sets ``hsa`` in one of those
places, with the server configured for ``kiltpsa``, and check what ngspice
did against what the server says it does. The ordinary launch and the
controlled launch of a recoverable job must agree.

The installation's ``spinit`` can set ``ngbehavior`` too, and an ordinary run
still reads it for ngspice's code models. The last tests point
``SPICE_SCRIPTS`` at one that does, and check that the server compares the
mode a run's log names with the mode it passed, and says when it cannot.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from spicelib.simulators.ngspice_simulator import NGspiceSimulator

from ltspice_mcp.config import ServerConfig
from ltspice_mcp.lib.ngspice_mode import mode_notes
from ltspice_mcp.lib.services import resolve_job
from ltspice_mcp.lib.simulator import detect_simulators
from ltspice_mcp.state import SessionState
from tests.conftest import needs_controlled_startup, terminal_experiment
from tests.test_inspect_tools import _run
from tests.test_ngspice_e2e import _analyze

pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.skipif(shutil.which("ngspice") is None, reason="ngspice not on PATH"),
]

LOCATIONS = ("SPICE_USERINIT_DIR", "working directory", "home directory")
"""Where ngspice looks for a user's ``.spiceinit``, in the order it looks."""

MULTIPLIER = (
    "* m on an instance and on the card it calls\n"
    "V1 a 0 1\n"
    "X1 a r1 m=4\n"
    ".subckt r1 p\nR1 p 0 1k m=3\n.ends\n"
    ".op\n.end\n"
)
"""Outside the HSPICE modes ngspice gives R1 the instance's m, so 4 copies
of 1k draw 4 mA from 1 V. In ``hsa`` it multiplies the two, 12 copies and
12 mA."""

KILTPSA_FLAGS = "ps lt ki a"
"""How ngspice-42 names the flags of ``kiltpsa`` in its log."""


def _kiltpsa_session(work_dir: Path, monkeypatch: pytest.MonkeyPatch) -> SessionState:
    """A session that runs real ngspice and models decks in ``kiltpsa``."""
    # detect_simulators sets the mode on spicelib's class for the whole
    # process; recording the current value first restores it afterwards.
    monkeypatch.setattr(
        NGspiceSimulator, "_compatibility_mode", NGspiceSimulator._compatibility_mode
    )
    config = ServerConfig(
        simulator="ngspice",
        working_dir=work_dir,
        allowed_paths=[work_dir],
        ngbehavior="kiltpsa",
    )
    available = detect_simulators(config)
    assert "ngspice" in available, available
    return SessionState.create(config, available)


def _spiceinit_sets_hsa(location: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Put a ``.spiceinit`` that sets ``hsa`` at ``location``, and leave the
    places ngspice looks in before it empty, so that file is the one read."""
    folder = tmp_path / "user-startup"
    folder.mkdir()
    (folder / ".spiceinit").write_text("set ngbehavior=hsa\n", encoding="utf-8")
    empty = tmp_path / "no-startup"
    empty.mkdir()
    monkeypatch.delenv("SPICE_USERINIT_DIR", raising=False)
    # A run's ngspice inherits the server's working directory.
    monkeypatch.chdir(folder if location == "working directory" else empty)
    if location == "SPICE_USERINIT_DIR":
        monkeypatch.setenv("SPICE_USERINIT_DIR", str(folder))
    elif location == "home directory":
        # ngspice reads HOME, then USERPROFILE; a Windows session may set
        # only the second.
        monkeypatch.setenv("HOME", str(folder))
        monkeypatch.setenv("USERPROFILE", str(folder))


def _log(state: SessionState, job_id: str) -> str:
    job = resolve_job(job_id, state)
    log_file = job.cases[0].log_file
    assert log_file is not None
    return log_file.read_text(encoding="utf-8")


@pytest.mark.parametrize("location", LOCATIONS)
@pytest.mark.parametrize(
    "recoverable", [False, pytest.param(True, marks=needs_controlled_startup)]
)
async def test_ngspice_reads_the_deck_in_the_mode_the_server_models(
    location: str,
    recoverable: bool,
    work_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    state = _kiltpsa_session(work_dir, monkeypatch)
    _spiceinit_sets_hsa(location, tmp_path, monkeypatch)
    deck = work_dir / "multiplier.cir"
    deck.write_text(MULTIPLIER, encoding="utf-8")

    (discovered,) = await _run(
        state, [{"kind": "hierarchy", "path": str(deck), "simulator": "ngspice"}]
    )
    assert discovered["ok"], discovered
    assert discovered["data"]["profile"]["ngbehavior"] == "kiltpsa"

    receipt = await terminal_experiment(
        state,
        {
            "request_id": f"startup-mode-{recoverable}",
            "circuits": [{"path": str(deck)}],
            "execution": {"simulator": "ngspice", "recoverable": recoverable, "wait_s": 90},
        },
    )
    assert receipt["status"] == "completed", receipt
    log = _log(state, receipt["job_id"])
    assert f"Compatibility modes selected: {KILTPSA_FLAGS}" in log, log
    result = await _analyze(
        state, receipt["job_id"], [{"key": "op", "metric": "operating_point", "device": "V1"}]
    )
    op = result["results"]["op"]["values"][0]["value"]
    (current,) = op["currents"].values()
    assert abs(current) == pytest.approx(4e-3)


@pytest.mark.parametrize("location", LOCATIONS)
async def test_a_sectioned_lib_fails_where_the_server_says_it_will(
    location: str,
    work_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """``kiltpsa`` reads ``.lib <file> <section>`` as two plain includes, so
    the lint refuses the deck and, with the lint off, the classifier names
    the mode as the cause of the failed run. Both read the configured mode;
    ngspice must read the same one."""
    state = _kiltpsa_session(work_dir, monkeypatch)
    _spiceinit_sets_hsa(location, tmp_path, monkeypatch)
    (work_dir / "models.lib").write_text(
        ".lib tt\n.model dmod d is=1e-14\n.endl tt\n", encoding="utf-8"
    )
    deck = work_dir / "corner.cir"
    deck.write_text(
        "* corner select\n.lib models.lib tt\nV1 a 0 1\nR1 a b 1k\nD1 b 0 dmod\n.op\n.end\n",
        encoding="utf-8",
    )

    def request(lint: str) -> dict:
        return {
            "request_id": f"corner-{lint}",
            "circuits": [{"path": str(deck)}],
            "execution": {"simulator": "ngspice", "wait_s": 90},
            "lint": lint,
        }

    refused = await terminal_experiment(state, request("block"))
    assert refused["failures"][0]["code"] == "lint_blocked", refused

    receipt = await terminal_experiment(state, request("off"))
    (failure,) = receipt["failures"]
    assert failure["code"] == "ngspice_lib_section", receipt
    assert failure["evidence"]["ngbehavior"] == "kiltpsa"
    # The failure itself shows ngspice did not read hsa, which loads the
    # section. Whether the log also names the mode depends on the build:
    # ngspice-42 does so before the missing include, later builds only once
    # the deck is read, which this one never is. Either way no line in it
    # names another mode.
    log = _log(state, receipt["job_id"])
    assert all(flags == tuple(KILTPSA_FLAGS.split()) for flags in mode_notes(log)), log
    # The .spiceinit this test wrote did not fix the run, so the hint names
    # the setting that does and does not offer a .spiceinit as another way.
    assert '[simulator] ngbehavior = "hsa"' in failure["hint"]
    assert "to a .spiceinit" not in failure["hint"]


async def test_admission_and_launch_read_the_mode_the_runs_own_ngspice_class_passes(
    work_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    """The lint, the hierarchy profile and the launch take a run's mode from
    one place (``simulator.run_ngbehavior``): the mode the run's own ngspice
    class passes. Here that class passes ``hsa``, which reads a ``.lib``
    section, while spicelib's base class still holds ``kiltpsa``, which would
    drop it."""
    state = _kiltpsa_session(work_dir, monkeypatch)
    base = state.available_simulators["ngspice"]
    own = type(base.__name__, (base,), {"_compatibility_mode": "hsa"})
    monkeypatch.setitem(state.available_simulators, "ngspice", own)
    (work_dir / "models.lib").write_text(
        ".lib tt\n.model dmod d is=1e-14\n.endl tt\n", encoding="utf-8"
    )
    deck = work_dir / "corner.cir"
    deck.write_text(
        "* corner select\n.lib models.lib tt\nV1 a 0 1\nR1 a b 1k\nD1 b 0 dmod\n.op\n.end\n",
        encoding="utf-8",
    )

    (discovered,) = await _run(
        state, [{"kind": "hierarchy", "path": str(deck), "simulator": "ngspice"}]
    )
    assert discovered["ok"], discovered
    assert discovered["data"]["profile"]["ngbehavior"] == "hsa"

    receipt = await terminal_experiment(
        state,
        {
            "request_id": "own-class-mode",
            "circuits": [{"path": str(deck)}],
            "execution": {"simulator": "ngspice", "wait_s": 90},
        },
    )
    assert receipt["status"] == "completed", receipt
    assert mode_notes(_log(state, receipt["job_id"]))[0] == ("hs", "a")


def _spinit_sets(mode: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Point ``SPICE_SCRIPTS`` at a ``spinit`` that sets ``mode``, as an
    installation's own could."""
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "spinit").write_text(f"set ngbehavior={mode}\n", encoding="utf-8")
    monkeypatch.setenv("SPICE_SCRIPTS", str(scripts))


async def _run_deck(state: SessionState, deck: Path, *, recoverable: bool = False, **request):
    return await terminal_experiment(
        state,
        {
            "request_id": f"{deck.stem}-{recoverable}-{request.get('lint', 'block')}",
            "circuits": [{"path": str(deck)}],
            "execution": {"simulator": "ngspice", "recoverable": recoverable, "wait_s": 90},
            **request,
        },
    )


MISMATCH_EVIDENCE = {
    "ngbehavior": "kiltpsa",
    "expected_modes": KILTPSA_FLAGS.split(),
    "logged_modes": ["hs", "a"],
}
"""What a run read in ``hsa`` while the server passed ``kiltpsa`` reports."""


async def test_a_run_ngspice_read_in_another_mode_fails(
    work_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    state = _kiltpsa_session(work_dir, monkeypatch)
    _spinit_sets("hsa", tmp_path, monkeypatch)
    deck = work_dir / "multiplier.cir"
    deck.write_text(MULTIPLIER, encoding="utf-8")

    (discovered,) = await _run(
        state, [{"kind": "hierarchy", "path": str(deck), "simulator": "ngspice"}]
    )
    assert discovered["data"]["profile"]["ngbehavior"] == "kiltpsa"

    receipt = await _run_deck(state, deck)
    (failure,) = receipt["failures"]
    assert failure["code"] == "ngspice_mode_mismatch", receipt
    assert failure["evidence"] == MISMATCH_EVIDENCE
    assert "spinit" in failure["hint"]


@needs_controlled_startup
async def test_a_recoverable_run_does_not_read_the_spinit_that_sets_another_mode(
    work_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    state = _kiltpsa_session(work_dir, monkeypatch)
    _spinit_sets("hsa", tmp_path, monkeypatch)
    deck = work_dir / "multiplier.cir"
    deck.write_text(MULTIPLIER, encoding="utf-8")

    receipt = await _run_deck(state, deck, recoverable=True)
    assert receipt["status"] == "completed", receipt
    assert f"Compatibility modes selected: {KILTPSA_FLAGS}" in _log(state, receipt["job_id"])


async def test_a_sectioned_lib_read_in_another_mode_fails_on_the_mode(
    work_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """The lint refuses the deck in the mode the server passes. Run anyway,
    ngspice reads it in the ``spinit``'s ``hsa`` and loads the section, and
    the run fails on the mode instead of returning results the lint said
    could not exist."""
    state = _kiltpsa_session(work_dir, monkeypatch)
    _spinit_sets("hsa", tmp_path, monkeypatch)
    (work_dir / "models.lib").write_text(
        ".lib tt\n.model dmod d is=1e-14\n.endl tt\n", encoding="utf-8"
    )
    deck = work_dir / "corner.cir"
    deck.write_text(
        "* corner select\n.lib models.lib tt\nV1 a 0 1\nR1 a b 1k\nD1 b 0 dmod\n.op\n.end\n",
        encoding="utf-8",
    )

    refused = await _run_deck(state, deck, lint="block")
    assert refused["failures"][0]["code"] == "lint_blocked", refused
    receipt = await _run_deck(state, deck, lint="off")
    (failure,) = receipt["failures"]
    assert failure["code"] == "ngspice_mode_mismatch", receipt
    assert failure["evidence"] == MISMATCH_EVIDENCE


async def test_a_run_whose_log_names_no_mode_says_the_mode_is_unchecked(
    work_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """ngspice's ``mc`` mode prints no compatibility note, so the log cannot
    show which mode ngspice read the deck in. The run keeps its results and
    says the check did not run."""
    state = _kiltpsa_session(work_dir, monkeypatch)
    _spinit_sets("mc", tmp_path, monkeypatch)
    deck = work_dir / "divider.cir"
    deck.write_text("* divider\nV1 a 0 1\nR1 a b 1k\nR2 b 0 1k\n.op\n.end\n", encoding="utf-8")

    receipt = await _run_deck(state, deck)
    assert receipt["status"] == "completed", receipt
    assert "Compatibility mode" not in _log(state, receipt["job_id"])
    (unchecked,) = [
        item
        for item in receipt.get("observations", [])
        if item["code"] == "ngspice_mode_unverified"
    ]
    assert unchecked["evidence"] == {"ngbehavior": "kiltpsa", "reason": "no_mode_in_log"}
