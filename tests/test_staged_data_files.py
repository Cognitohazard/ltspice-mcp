"""The data files a deck reads, copied with it when the server stages it.

A run starts the simulator on a staged copy of the deck in a folder of its
own, so a data file the deck names by a relative path (LTspice's
``wavefile=`` and PWL ``file=``, ngspice's ``filesource`` ``file=``) is
missing beside that copy, or read from somewhere else, unless staging copies
it and the copy names it. Each path that stages a deck is run here: a plain
run, assign and random variations (several staged copies), a detached job,
and the Windows-native route LTspice takes across the WSL boundary.

Where each simulator looks for a relative data path: LTspice, the deck's
folder (the ``data-file-relative-path`` recordings, both builds); ngspice-42,
the deck's folder, for a card in an included file too, and then the folder it
was started in (measured below).
"""

from __future__ import annotations

import dataclasses
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from ltspice_mcp.api import Api
from ltspice_mcp.config import ServerConfig
from ltspice_mcp.lib import wsl
from ltspice_mcp.lib.deck_staging import (
    DeckStagingError,
    resolve_experiment_paths,
    scan_data_references,
    stage_deck,
)
from ltspice_mcp.lib.experiment_runner import ExperimentRunner
from ltspice_mcp.lib.simulator import NGSPICE_NO_USER_INIT, detect_simulators
from ltspice_mcp.lib.spice_lex import lex_deck, lex_for_edit
from ltspice_mcp.state import SessionState
from ltspice_mcp.tools import experiments as experiments_mod
from ltspice_mcp.tools.analyze import AnalyzeResultsInput, handle_analyze_results
from ltspice_mcp.tools.experiments import RunExperimentsInput, handle_run_experiments
from tests.conftest import LIVENESS_S, FakeSim, fake_artifact_paths, terminal_experiment
from tests.ltspice_recorder import INPUTS

needs_ngspice = pytest.mark.skipif(shutil.which("ngspice") is None, reason="ngspice not on PATH")

#: The level the copy of the data file beside the deck holds, and the level of
#: a decoy copy elsewhere: a run that reads the wrong file reads the decoy.
BESIDE_THE_DECK = 0.5
DECOY = 0.75

FILESOURCE_MODEL = (
    '.model src filesource (file="data/level.txt" amploffset=[0] amplscale=[1] '
    "timeoffset=0 timescale=1 timerelative=false amplstep=false)\n"
)
FILESOURCE_DECK = (
    "* a filesource reading a file named by a relative path\n"
    "a1 %vd([a 0]) src\n" + FILESOURCE_MODEL + "R1 a 0 1k\n.tran 0.1 1\n.end\n"
)
FILESOURCE_INCLUDED = (
    "* a filesource model in an included file in another folder\n"
    ".include models/src.inc\n"
    "a1 %vd([a 0]) src\nR1 a 0 1k\n.tran 0.1 1\n.end\n"
)


def _level_file(level: float) -> str:
    return f"0 {level}\n1 {level}\n"


def _bench(root: Path, deck: str = FILESOURCE_DECK) -> Path:
    """A deck in ``root/bench`` beside ``data/level.txt``, and a decoy copy of
    the data file in ``root`` and in the include's folder."""
    bench = root / "bench"
    (bench / "data").mkdir(parents=True)
    (bench / "data" / "level.txt").write_text(_level_file(BESIDE_THE_DECK))
    (bench / "models" / "data").mkdir(parents=True)
    (bench / "models" / "src.inc").write_text(FILESOURCE_MODEL)
    (bench / "models" / "data" / "level.txt").write_text(_level_file(DECOY))
    (root / "data").mkdir(exist_ok=True)
    (root / "data" / "level.txt").write_text(_level_file(DECOY))
    deck_path = bench / "fs.cir"
    deck_path.write_text(deck)
    return deck_path


# --------------------------------------------------------------------------
# Where ngspice looks for a relative data path
# --------------------------------------------------------------------------


def _ngspice_level(deck: Path, started_in: Path) -> float | None:
    """v(a) half way through the run of ``deck`` by ngspice started in
    ``started_in``, in the mode the server runs it in; None when it fails."""
    control = deck.with_name(deck.stem + "_probe.cir")
    control.write_text(
        deck.read_text().replace(
            ".end\n", ".control\nrun\nmeas tran va find v(a) at=0.5\n.endc\n.end\n"
        )
    )
    completed = subprocess.run(
        ["ngspice", NGSPICE_NO_USER_INIT, "-D", "ngbehavior=kiltpsa", "-b", str(control)],
        cwd=started_in,
        capture_output=True,
        text=True,
        timeout=LIVENESS_S,
        check=False,
    )
    found = re.search(r"^va\s*=\s*(\S+)", completed.stdout, re.MULTILINE)
    return float(found.group(1)) if found else None


@needs_ngspice
@pytest.mark.parametrize("deck", [FILESOURCE_DECK, FILESOURCE_INCLUDED], ids=["deck", "include"])
def test_ngspice_reads_a_relative_data_path_from_the_decks_folder(tmp_path: Path, deck: str):
    """Started in a folder that holds a decoy, ngspice reads the copy beside
    the deck, for a card in an included file too (not the copy beside the
    include); with no copy beside the deck it falls back to the folder it was
    started in, which the server does not choose for a run."""
    deck_path = _bench(tmp_path, deck)
    assert _ngspice_level(deck_path, tmp_path) == BESIDE_THE_DECK
    (deck_path.parent / "data" / "level.txt").unlink()
    assert _ngspice_level(deck_path, tmp_path) == DECOY


# --------------------------------------------------------------------------
# The one reader of data cards
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("card", "dialect", "path"),
    [
        ("V1 a 0 wavefile=data/level.wav", "ltspice", "data/level.wav"),
        ('V1 a b wavefile="level.wav"', "ltspice", "level.wav"),
        ("V1 a 0 wavefile level.wav", "ltspice", "level.wav"),
        ("I1 d 0 wavefile=level.wav", "ltspice", "level.wav"),
        ("V1 a 0 PWL file=data/level.txt", "ltspice", "data/level.txt"),
        ('V1 a 0 PWL file = "my data.txt"', "ltspice", "my data.txt"),
        (FILESOURCE_MODEL.strip(), "ngspice", "data/level.txt"),
        ('.model src filesource file="x.txt"', "ngspice", "x.txt"),
    ],
)
def test_the_reader_names_each_data_file(card: str, dialect: str, path: str):
    cards = lex_deck(f"* t\n{card}\n.end\n").cards
    (reference,) = scan_data_references(cards, dialect)
    assert reference.raw_path == path
    assert reference.card.body[reference.start : reference.end].strip("\"'") == path


@pytest.mark.parametrize(
    ("card", "dialect"),
    [
        # Each simulator's own cards only: ngspice runs no wavefile, and
        # LTspice has no filesource.
        ("V1 a 0 wavefile=level.wav", "ngspice"),
        (FILESOURCE_MODEL.strip(), "ltspice"),
        ("V1 a 0 wavefile=level.wav", None),
        ("R1 a 0 file=1k", "ltspice"),
        (".model d1 d (is=1e-14)", "ngspice"),
    ],
)
def test_the_reader_names_no_file_another_simulator_reads(card: str, dialect: str | None):
    assert scan_data_references(lex_deck(f"* t\n{card}\n.end\n").cards, dialect) == []


# --------------------------------------------------------------------------
# Staging
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("stem", "data"),
    [
        ("relative_wavefile", "data/level.wav"),
        ("relative_pwl_file", "data/level.txt"),
        ("wavefile_no_equals", "level.wav"),
    ],
)
def test_staging_copies_a_relative_data_file_and_names_the_copy(
    tmp_path: Path, stem: str, data: str
):
    """The recorded decks, staged for LTspice: the data file is copied byte
    for byte and the card names the copy by its whole path, since the run
    starts from a copy of the deck in another folder. That LTspice reads an
    absolute data path is not recorded, and the staging says so."""
    root = tmp_path / stem
    deck = root / f"{stem}.cir"
    (root / data).parent.mkdir(parents=True, exist_ok=True)
    deck.write_bytes((INPUTS / "deck" / f"{stem}.cir").read_bytes())
    source = INPUTS / "deck" / data
    (root / data).write_bytes(source.read_bytes())

    staged = stage_deck(deck, tmp_path / "stage", [root], origin=deck, dialect="ltspice")

    (reference,) = scan_data_references(lex_deck(staged.text).cards, "ltspice")
    copy = Path(reference.raw_path)
    assert copy.is_absolute()
    assert copy.is_relative_to(tmp_path / "stage")
    assert copy.read_bytes() == source.read_bytes()
    entry = next(item for item in staged.manifest if item.path == (root / data).resolve())
    assert (entry.staged, entry.live, entry.staged_path) == (True, False, copy)
    (note,) = [item for item in staged.observations if item["code"] == "data_path_unrecorded"]
    assert note["evidence"]["paths"] == [str(copy)]


def test_two_data_paths_on_one_card_are_each_rewritten(tmp_path: Path):
    """Each rewrite lengthens the card, so they are made last first: one made
    first would move the span of the next."""
    for name in ("a.wav", "bb.wav"):
        (tmp_path / name).write_bytes(name.encode())
    deck = tmp_path / "tb.cir"
    deck.write_text("* t\nV1 a 0 wavefile=a.wav wavefile=bb.wav\nR1 a 0 1k\n.tran 1\n.end\n")

    staged = stage_deck(deck, tmp_path / "stage", [tmp_path], origin=deck, dialect="ltspice")

    named = [ref.raw_path for ref in scan_data_references(lex_deck(staged.text).cards, "ltspice")]
    assert [Path(path).read_bytes() for path in named] == [b"a.wav", b"bb.wav"]


def test_a_data_path_on_a_continuation_line_is_rewritten_there(tmp_path: Path):
    (tmp_path / "level.wav").write_bytes(b"RIFF")
    deck = tmp_path / "tb.cir"
    deck.write_text("* t\nV1 a 0 wavefile=\n+ level.wav\nR1 a 0 1k\n.tran 1\n.end\n")

    staged = stage_deck(deck, tmp_path / "stage", [tmp_path], origin=deck, dialect="ltspice")

    (reference,) = scan_data_references(lex_deck(staged.text).cards, "ltspice")
    assert Path(reference.raw_path).read_bytes() == b"RIFF"
    assert staged.text.splitlines()[1:3] == ["V1 a 0 wavefile=", f"+ {reference.raw_path}"]


def test_ngspice_staging_reads_an_included_files_data_path_from_the_decks_folder(
    tmp_path: Path,
):
    deck = _bench(tmp_path, FILESOURCE_INCLUDED)

    staged = stage_deck(deck, tmp_path / "stage", [tmp_path], origin=deck, dialect="ngspice")

    (included,) = staged.includes
    (reference,) = scan_data_references(lex_for_edit(included.text, deck=False).read, "ngspice")
    assert Path(reference.raw_path).read_text() == _level_file(BESIDE_THE_DECK)
    assert not [item for item in staged.observations if item["code"] == "data_path_unrecorded"]


def test_ltspice_staging_refuses_a_relative_data_path_in_an_included_file(tmp_path: Path):
    """Which folder LTspice reads it from is not recorded, so staging does
    not guess."""
    root = tmp_path / "root"
    (root / "sub" / "data").mkdir(parents=True)
    (root / "sub" / "data" / "level.txt").write_text(_level_file(BESIDE_THE_DECK))
    (root / "sub" / "src.inc").write_text("V1 a 0 PWL file=data/level.txt\n")
    deck = root / "tb.cir"
    deck.write_text("* t\n.include sub/src.inc\nR1 a 0 1k\n.tran 1\n.end\n")

    with pytest.raises(DeckStagingError) as caught:
        stage_deck(deck, tmp_path / "stage", [root], origin=deck, dialect="ltspice")

    assert caught.value.code == "data_file_unestablished"
    assert "not recorded" in str(caught.value)
    assert "absolute path" in str(caught.value)


def test_a_missing_data_file_is_refused_with_where_it_was_looked_for(tmp_path: Path):
    deck = tmp_path / "tb.cir"
    deck.write_text("* t\nV1 a 0 wavefile=data/level.wav\nR1 a 0 1k\n.tran 1\n.end\n")

    with pytest.raises(DeckStagingError) as caught:
        stage_deck(deck, tmp_path / "stage", [tmp_path], origin=deck, dialect="ltspice")

    assert caught.value.code == "data_file_missing"
    message = str(caught.value)
    assert str(tmp_path / "data" / "level.wav") in message
    assert "line 2" in message
    assert "deck's folder" in message


def test_a_data_file_outside_the_sandbox_is_refused(tmp_path: Path):
    inside, outside = tmp_path / "inside", tmp_path / "outside"
    inside.mkdir()
    outside.mkdir()
    (outside / "level.txt").write_text(_level_file(BESIDE_THE_DECK))
    deck = inside / "tb.cir"
    deck.write_text("* t\nV1 a 0 PWL file=../outside/level.txt\nR1 a 0 1k\n.tran 1\n.end\n")

    with pytest.raises(DeckStagingError) as caught:
        stage_deck(deck, tmp_path / "stage", [inside], origin=deck, dialect="ltspice")

    assert caught.value.code == "data_file_outside_allowed_paths"
    assert "allowed_paths" in str(caught.value)
    # With no sandbox it is staged like any other.
    staged = stage_deck(deck, tmp_path / "open", None, origin=deck, dialect="ltspice")
    (reference,) = scan_data_references(lex_deck(staged.text).cards, "ltspice")
    assert Path(reference.raw_path).read_text() == _level_file(BESIDE_THE_DECK)


def test_an_absolute_data_path_is_read_where_it_is_and_reported(tmp_path: Path):
    data = tmp_path / "level.txt"
    data.write_text(_level_file(BESIDE_THE_DECK))
    deck = tmp_path / "tb.cir"
    card = f"V1 a 0 PWL file={data}"
    deck.write_text(f"* t\n{card}\nR1 a 0 1k\n.tran 1\n.end\n")

    staged = stage_deck(deck, tmp_path / "stage", [tmp_path], origin=deck, dialect="ltspice")

    assert card in staged.text
    entry = next(item for item in staged.manifest if item.path == data.resolve())
    assert (entry.staged, entry.live) == (False, True)
    assert [item["code"] for item in staged.observations] == ["live_data_file"]


def test_windows_paths_name_the_data_copy_through_the_conversion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """LTspice across the WSL boundary opens neither the Linux spelling of the
    copy nor of a file named by an absolute path."""
    monkeypatch.setattr(wsl, "to_windows_path", lambda path: f"Z:{path}".replace("/", "\\"))
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "level.wav").write_bytes(
        (INPUTS / "deck" / "data" / "level.wav").read_bytes()
    )
    absolute = tmp_path / "other.txt"
    absolute.write_text(_level_file(DECOY))
    deck = tmp_path / "tb.cir"
    deck.write_text(
        f"* t\nV1 a 0 wavefile=data/level.wav\nV2 b 0 PWL file={absolute}\n"
        "R1 a 0 1k\nR2 b 0 1k\n.tran 1\n.end\n"
    )

    staged = stage_deck(
        deck, tmp_path / "stage", [tmp_path], origin=deck, dialect="ltspice", windows_paths=True
    )

    copy = next(item.staged_path for item in staged.manifest if item.path.name == "level.wav")
    assert copy is not None
    named = [ref.raw_path for ref in scan_data_references(lex_deck(staged.text).cards, "ltspice")]
    assert named == [wsl.to_windows_path(copy), wsl.to_windows_path(absolute.resolve())]


# --------------------------------------------------------------------------
# Every path that stages a deck, LTspice (no simulator: the deck it is handed)
# --------------------------------------------------------------------------

LTspice = type("LTspice", (FakeSim,), {})


def _capture_submitted(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Stand in for the simulator, keeping the text of each deck it is handed."""
    from ltspice_mcp.lib.runner_base import _raw_completion_outcome

    texts: list[str] = []

    def submit(self, netlist: Path, run_filename: str, callback, **_kwargs):
        texts.append(Path(netlist).read_text())
        raw, log = fake_artifact_paths(self.output_folder, run_filename)
        raw.write_bytes(b"Title: mock")
        log.write_text("ok")
        outcome = _raw_completion_outcome(str(raw), str(log))
        self.loop.call_soon_threadsafe(callback, outcome)
        return object()

    monkeypatch.setattr(ExperimentRunner, "submit_netlist", submit)
    return texts


def _assert_names_the_copy(text: str, source: Path, work_dir: Path, *, windows_native: bool):
    """The one data card of a submitted deck names a staged copy of ``source``."""
    (reference,) = scan_data_references(lex_deck(text).cards, "ltspice")
    named = reference.raw_path
    if windows_native:
        assert named.startswith("Z:")
        named = named.removeprefix("Z:").replace("\\", "/")
    assert Path(named).read_bytes() == source.read_bytes()
    assert Path(named).is_relative_to(work_dir / ".ltspice-mcp")


_VARIATIONS = {
    "plain": None,
    "assign": [{"kind": "assign", "assign": {"R1": ["1k", "2k"]}}],
    "random": [
        {
            "kind": "random",
            "runs": 2,
            "seed": 7,
            "rules": [{"rule": "component", "target": "R1", "tolerance": 0.05}],
        }
    ],
}


@pytest.mark.asyncio
@pytest.mark.parametrize("windows_native", [False, True], ids=["local", "windows_native"])
@pytest.mark.parametrize("variations", list(_VARIATIONS))
async def test_every_ltspice_case_names_the_staged_copy(
    config: ServerConfig,
    work_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    variations: str,
    windows_native: bool,
):
    texts = _capture_submitted(monkeypatch)
    if windows_native:
        route = resolve_experiment_paths

        def windows(working_dir, job_id, circuit_id, simulator):
            return dataclasses.replace(
                route(working_dir, job_id, circuit_id, simulator), windows_native=True
            )

        monkeypatch.setattr(experiments_mod, "resolve_experiment_paths", windows)
        monkeypatch.setattr(wsl, "to_windows_path", lambda path: f"Z:{path}".replace("/", "\\"))
    bench = work_dir / "bench"
    (bench / "data").mkdir(parents=True)
    source = INPUTS / "deck" / "data" / "level.wav"
    (bench / "data" / "level.wav").write_bytes(source.read_bytes())
    deck = bench / "tb.cir"
    deck.write_text("* t\nV1 a 0 wavefile=data/level.wav\nR1 a 0 1k\n.tran 10m\n.end\n")
    state = SessionState.create(config, available={"sim": LTspice})
    payload: dict = {
        "request_id": f"data-{variations}-{windows_native}",
        "circuits": [{"path": str(deck), "id": "dut"}],
        "execution": {"wait_s": LIVENESS_S},
        "lint": "off",
    }
    if _VARIATIONS[variations] is not None:
        payload["variations"] = _VARIATIONS[variations]

    result = await handle_run_experiments(RunExperimentsInput.model_validate(payload), state)

    data = result.structured_content
    assert data is not None and not result.is_error, data
    assert texts and len(texts) == data["completeness"]["expanded"]
    for text in texts:
        _assert_names_the_copy(text, source, work_dir, windows_native=windows_native)


@pytest.mark.asyncio
async def test_a_case_whose_data_file_is_missing_fails_before_it_runs(
    config: ServerConfig, work_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    texts = _capture_submitted(monkeypatch)
    deck = work_dir / "tb.cir"
    deck.write_text("* t\nV1 a 0 PWL file=data/level.txt\nR1 a 0 1k\n.tran 1\n.end\n")
    state = SessionState.create(config, available={"sim": LTspice})

    result = await handle_run_experiments(
        RunExperimentsInput.model_validate(
            {
                "request_id": "data-missing",
                "circuits": [{"path": str(deck), "id": "dut"}],
                "execution": {"wait_s": LIVENESS_S},
                "lint": "off",
            }
        ),
        state,
    )

    data = result.structured_content
    assert data is not None
    assert texts == []
    (failure,) = data["failures"]
    assert failure["code"] == "data_file_missing", failure


# --------------------------------------------------------------------------
# Every path that stages a deck, ngspice (live)
# --------------------------------------------------------------------------


@pytest.fixture
def ngspice_state(work_dir: Path) -> SessionState:
    config = ServerConfig(
        simulator="ngspice", working_dir=work_dir, allowed_paths=[work_dir], log_level="DEBUG"
    )
    return SessionState.create(config, detect_simulators(config))


async def _levels(state: SessionState, job_id: str) -> list[float]:
    result = await handle_analyze_results(
        AnalyzeResultsInput.model_validate(
            {
                "sources": [{"job_id": job_id, "runs": "all", "label": "dut"}],
                "recipes": [{"key": "va", "metric": "value", "expr": "v(a)", "at": "0.5"}],
            }
        ),
        state,
    )
    data = result.structured_content
    assert data is not None, result.content
    return [row["value"]["value"] for row in data["results"]["va"]["values"]]


@needs_ngspice
@pytest.mark.asyncio
@pytest.mark.parametrize("deck", [FILESOURCE_DECK, FILESOURCE_INCLUDED], ids=["deck", "include"])
@pytest.mark.parametrize("variations", list(_VARIATIONS))
async def test_every_ngspice_case_reads_the_data_file_beside_the_deck(
    ngspice_state: SessionState, work_dir: Path, variations: str, deck: str
):
    deck_path = _bench(work_dir, deck)
    payload: dict = {
        "request_id": f"ng-data-{variations}-{len(deck)}",
        "circuits": [{"path": str(deck_path), "id": "dut"}],
        "execution": {"wait_s": 90, "simulator": "ngspice"},
    }
    if _VARIATIONS[variations] is not None:
        payload["variations"] = _VARIATIONS[variations]

    receipt = await terminal_experiment(ngspice_state, payload)

    assert receipt["status"] == "completed", receipt
    levels = await _levels(ngspice_state, receipt["job_id"])
    assert len(levels) == receipt["completeness"]["expanded"]
    assert levels == pytest.approx([BESIDE_THE_DECK] * len(levels))


@needs_ngspice
def test_a_detached_ngspice_job_reads_the_data_file_beside_the_deck(work_dir: Path):
    deck_path = _bench(work_dir)
    with Api(
        working_dir=work_dir, simulator="ngspice", allowed_paths=[work_dir], max_parallel_sims=1
    ) as api:
        receipt = api.run_experiments(
            wait=False,
            detach=True,
            request_id="ng-data-detached",
            circuits=[{"path": str(deck_path), "id": "dut"}],
        )
        final = api.wait(receipt["job_id"], timeout=120)
        assert final["status"] == "completed", final
        analysis = api.analyze_results(
            sources=[{"job_id": receipt["job_id"], "label": "dut"}],
            recipes=[{"key": "va", "metric": "value", "expr": "v(a)", "at": "0.5"}],
        )
    measured = analysis["results"]["va"]["values"][0]["value"]
    assert measured["value"] == pytest.approx(BESIDE_THE_DECK)
