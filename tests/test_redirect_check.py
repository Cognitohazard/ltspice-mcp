"""Whether a path the server is about to use redirects to another location.

On Windows, ``Path.resolve`` of a path that does not exist asks the system for
the final path twice: once at the start, and once more to decide whether to
strip the ``\\\\?\\`` prefix from its answer (``ntpath.realpath``). It strips
it only when both lookups fail the same way, so a folder another thread
creates or removes between the two leaves the prefix on, and a check that
compares the resolved text with the path it built refuses a path that
redirects nowhere. The tests below make that window on purpose: another thread
changes the folder between the two lookups, through a handshake, and the check
must still answer from what is on disk.
"""

from __future__ import annotations

import sys
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest

from ltspice_mcp.lib.store import Store, parser_file_in
from tests.conftest import LIVENESS_S

windows_only = pytest.mark.skipif(
    sys.platform != "win32",
    reason="the resolve race is in ntpath.realpath, which only Windows runs",
)


@contextmanager
def _between_lookups(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    *,
    found: bool,
    change: Callable[[], None],
) -> Iterator[None]:
    """Run ``change`` on another thread right after the system's first final-path
    lookup of a path named ``name`` that fails (``found=False``) or succeeds
    (``found=True``), and hold that lookup until ``change`` is done.

    Every lookup still goes to the system; only the moment another thread acts
    is chosen. When the code under test never makes such a lookup, ``change``
    runs once it returns, so the folder ends in the same state either way.
    """
    import ntpath

    real = ntpath._getfinalpathname  # pyright: ignore[reportAttributeAccessIssue]
    asked = threading.Event()
    done = threading.Event()
    failures: list[BaseException] = []

    def hold(path: str) -> None:
        if asked.is_set() or ntpath.basename(path) != name:
            return
        asked.set()
        assert done.wait(LIVENESS_S), "the other thread never changed the folder"

    def lookup(path: str) -> str:
        try:
            result = real(path)
        except OSError:
            if not found:
                hold(path)
            raise
        if found:
            hold(path)
        return result

    def other() -> None:
        try:
            if asked.wait(LIVENESS_S):
                change()
        except BaseException as exc:  # reported on the asking thread
            failures.append(exc)
        finally:
            done.set()

    monkeypatch.setattr(ntpath, "_getfinalpathname", lookup)
    thread = threading.Thread(target=other, name="folder-changer")
    thread.start()
    try:
        yield
    finally:
        asked.set()
        thread.join(LIVENESS_S)
    assert not thread.is_alive()
    if failures:
        raise failures[0]


@windows_only
def test_parser_dir_whose_folder_appears_between_the_two_lookups(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """The sighting: a fresh store, and a peer parse creates ``parsing`` while
    this one asks for its own folder."""
    store = Store(tmp_path)
    store.root.mkdir(parents=True)
    parsing = store.root / "parsing"

    with _between_lookups(monkeypatch, "probe", found=False, change=parsing.mkdir):
        directory = store.parser_dir("probe")

    assert directory.name == "probe"
    assert directory.parent.samefile(parsing)


def _parse_at_once(store: Store, threads: int) -> list[BaseException]:
    """``threads`` parses of ``store`` started behind one barrier, each making
    its parser folder the way ``parser_service`` does; what they raised."""
    barrier = threading.Barrier(threads)
    failures: list[BaseException] = []

    def parse(index: int) -> None:
        try:
            barrier.wait(LIVENESS_S)
            directory = store.parser_dir(f"parse-{index}")
            directory.parent.mkdir(parents=True, exist_ok=True)
            directory.mkdir()
        except BaseException as exc:
            failures.append(exc)

    workers = [threading.Thread(target=parse, args=(i,)) for i in range(threads)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(LIVENESS_S)
    return failures


@windows_only
def test_parser_dirs_asked_for_at_once_in_a_fresh_store(tmp_path: Path):
    """Many cases start to read their results at the same moment."""
    for round_ in range(20):
        store = Store(tmp_path / f"round-{round_}")
        store.root.mkdir(parents=True)
        failures = _parse_at_once(store, 16)
        assert not failures, f"round {round_}: {failures[0]!r}"


@windows_only
def test_parser_record_removed_between_the_two_lookups(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    store = Store(tmp_path)
    directory = store.parser_dir("records")
    directory.mkdir(parents=True)
    record = directory / "result.json"
    record.write_bytes(b"{}")

    with _between_lookups(monkeypatch, "result.json", found=True, change=record.unlink):
        named = parser_file_in(directory, "result.json")

    assert named == directory / "result.json"


@windows_only
def test_job_record_removed_between_the_two_lookups(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from ltspice_mcp.lib import experiment_store
    from tests.conftest import persist_experiment_record

    circuit = tmp_path / "deck.cir"
    circuit.write_text("* deck\n.end\n", encoding="utf-8")
    path = persist_experiment_record(tmp_path, circuit, "exp_race")
    job = experiment_store.load_job_from_path(path, tmp_path)
    assert job is not None

    with _between_lookups(monkeypatch, path.name, found=True, change=path.unlink):
        saved = experiment_store.save_job(job)

    assert saved.samefile(path)


@windows_only
def test_lineage_dir_removed_between_the_two_lookups(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    store = Store(tmp_path)
    recorded = store.run_dir("exp_race")
    recorded.mkdir(parents=True)

    with _between_lookups(monkeypatch, "exp_race", found=True, change=recorded.rmdir):
        root = store.lineage_run_dir("exp_race", recorded)

    assert root.name == "exp_race"
    assert root.parent.samefile(store.runs_root())


@windows_only
def test_ltspice_attempt_whose_folder_appears_between_the_two_lookups(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from ltspice_mcp.lib import controlled_ltspice
    from tests.test_controlled_ltspice import _QUERY, _template

    monkeypatch.setattr("ltspice_mcp.lib.controlled_ltspice.time.time", lambda: _QUERY)
    template = _template(tmp_path)
    path = tmp_path / "startup" / "attempt-1" / "LTspice.ini"

    with _between_lookups(monkeypatch, "LTspice.ini", found=False, change=path.parent.mkdir):
        controlled_ltspice.prepare_attempt_ini(template, path, tmp_path)

    assert path.is_file()
