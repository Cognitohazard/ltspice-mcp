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

import ast
import os
import sys
import threading
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ltspice_mcp.lib.experiment_types import ExperimentJob
    from ltspice_mcp.lib.pdk_native import ArtifactDigest

import pytest

from ltspice_mcp.lib.redirect import canonical, redirect_below, same_location, without_added_prefix
from ltspice_mcp.lib.store import Store, parser_file_in
from tests._source_scan import modules, scopes
from tests.conftest import LIVENESS_S, needs_controlled_startup, symlink_or_skip

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
    on: Sequence[tuple[object, str]] = (),
    must_make: bool = True,
) -> Iterator[None]:
    """Run ``change`` on another thread right after the first lookup of a path
    named ``name`` that fails (``found=False``) or succeeds (``found=True``),
    and hold that lookup until ``change`` is done.

    The lookups watched are the functions ``on`` names: by default the
    system's final-path lookup that ``Path.resolve`` makes on Windows (the old
    window), and ``os.stat`` and ``os.lstat`` for the lookups the redirect
    checks make now. Every lookup still goes to the system; only the moment
    another thread acts is chosen. When the code under test never makes such
    a lookup, ``change`` does not run: the code answered without asking the
    system the question whose answer the window changes. A window watched on
    ``os.stat`` and ``os.lstat`` must be made, unless ``must_make`` is false:
    the test fails otherwise.
    """
    import ntpath

    watched = list(on) or [(ntpath, "_getfinalpathname")]
    asked = threading.Event()
    done = threading.Event()
    closing = threading.Event()
    failures: list[BaseException] = []
    made: list[str] = []

    def hold(path: object) -> None:
        if asked.is_set() or not isinstance(path, (str, os.PathLike)):
            return
        if os.path.basename(os.fspath(path)) != name:
            return
        asked.set()
        made.append(name)
        assert done.wait(LIVENESS_S), "the other thread never changed the folder"

    def watching(real: Callable[..., object]) -> Callable[..., object]:
        def lookup(path: object, *args: object, **kwargs: object) -> object:
            try:
                result = real(path, *args, **kwargs)
            except OSError:
                if not found:
                    hold(path)
                raise
            if found:
                hold(path)
            return result

        return lookup

    def other() -> None:
        try:
            if asked.wait(LIVENESS_S) and not closing.is_set():
                change()
        except BaseException as exc:  # reported on the asking thread
            failures.append(exc)
        finally:
            done.set()

    for owner, attribute in watched:
        monkeypatch.setattr(owner, attribute, watching(getattr(owner, attribute)))
    thread = threading.Thread(target=other, name="folder-changer")
    thread.start()
    try:
        yield
    finally:
        closing.set()
        asked.set()
        thread.join(LIVENESS_S)
    assert not thread.is_alive()
    if failures:
        raise failures[0]
    # The lookups a check makes now are watched to make the window on them, so
    # a test that never made it would prove nothing about the race.
    assert made or not on or not must_make, (
        f"no lookup of {name!r} happened, so no window was made"
    )


#: The lookups the redirect checks make now: ``same_location`` follows links
#: (``os.stat``), ``redirect_below`` looks at each part as it is (``os.lstat``).
STAT = ((os, "stat"), (os, "lstat"))
FOLLOWED = ((os, "stat"),)


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

    assert same_location(directory, parsing / "probe")


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
        assert not worker.is_alive(), "a parse never finished"
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


# -- the checks as they are now, with the folder changed while they look ----
#
# The same windows, made on the lookups the checks make now (``os.stat`` and
# ``os.lstat``), on every platform.


def test_same_location_while_the_folder_appears(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "link"
    symlink_or_skip(link, target, target_is_directory=True)
    folder = target / "exp_a"

    with _between_lookups(monkeypatch, "exp_a", found=False, change=folder.mkdir, on=STAT):
        assert same_location(link / "exp_a", target / "exp_a")
    assert same_location(folder, folder)


def test_same_location_while_the_folder_vanishes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    target = tmp_path / "target"
    folder = target / "exp_a"
    folder.mkdir(parents=True)
    link = tmp_path / "link"
    symlink_or_skip(link, target, target_is_directory=True)

    with _between_lookups(monkeypatch, "exp_a", found=True, change=folder.rmdir, on=STAT):
        assert same_location(link / "exp_a", target / "exp_a")


def test_redirect_below_while_a_folder_appears_or_vanishes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    root = tmp_path / "root"
    root.mkdir()
    sub = root / "sub"

    with _between_lookups(monkeypatch, "sub", found=False, change=sub.mkdir, on=STAT):
        assert redirect_below(sub / "file", root) is None
    assert sub.is_dir()
    with _between_lookups(monkeypatch, "sub", found=True, change=sub.rmdir, on=STAT):
        assert redirect_below(sub / "file", root) is None


def test_parser_dir_while_its_folder_appears(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    store = Store(tmp_path)
    store.root.mkdir(parents=True)
    parsing = store.root / "parsing"

    with _between_lookups(monkeypatch, "parsing", found=False, change=parsing.mkdir, on=STAT):
        directory = store.parser_dir("probe")

    assert directory == canonical(store.root) / "parsing" / "probe"


def test_parser_record_while_it_is_removed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    store = Store(tmp_path)
    directory = store.parser_dir("records")
    directory.mkdir(parents=True)
    record = directory / "result.json"
    record.write_bytes(b"{}")

    with _between_lookups(monkeypatch, "result.json", found=True, change=record.unlink, on=STAT):
        assert parser_file_in(directory, "result.json") == record


def _job(tmp_path: Path, job_id: str) -> tuple[Path, ExperimentJob]:
    from ltspice_mcp.lib import experiment_store
    from tests.conftest import persist_experiment_record

    circuit = tmp_path / "deck.cir"
    circuit.write_text("* deck\n.end\n", encoding="utf-8")
    path = persist_experiment_record(tmp_path, circuit, job_id)
    job = experiment_store.load_job_from_path(path, tmp_path)
    assert job is not None
    return path, job


def test_job_record_saved_while_it_is_removed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from ltspice_mcp.lib import experiment_store

    path, job = _job(tmp_path, "exp_race")

    with _between_lookups(monkeypatch, path.name, found=True, change=path.unlink, on=STAT):
        saved = experiment_store.save_job(job)

    assert saved.samefile(path)


@pytest.mark.parametrize(
    ("on", "must_make"),
    [(STAT, True), (FOLLOWED, False)],
    ids=["any-lookup", "followed-lookup"],
)
def test_job_record_read_while_its_folder_appears(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    on: Sequence[tuple[object, str]],
    must_make: bool,
):
    """A fresh store: a peer creates ``experiments`` while a job is looked up.
    The window is made on the first lookup of the folder, and then on the
    first lookup that follows links, which is where an answer that compared
    which folders existed went wrong; one that compares the two spellings
    first makes none."""
    from ltspice_mcp.lib import experiment_store

    store = Store(tmp_path)
    store.root.mkdir(parents=True)
    record = store.job_record("exp_zz")
    folder = store.root / "experiments"

    with _between_lookups(
        monkeypatch, "experiments", found=False, change=folder.mkdir, on=on, must_make=must_make
    ):
        assert experiment_store.load_job_from_path(record, tmp_path) is None


@pytest.mark.parametrize("found", [False, True], ids=["appears", "vanishes"])
def test_lineage_dir_while_it_appears_or_vanishes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, found: bool
):
    """The runs folder kept behind a link, so the recorded lineage folder is
    spelled through the link and the expected one through its target."""
    store = Store(tmp_path)
    target = tmp_path / "canonical-runs"
    target.mkdir()
    runs = store.runs_root()
    runs.parent.mkdir(parents=True)
    symlink_or_skip(runs, target, target_is_directory=True)
    recorded = store.run_dir("exp_race")
    lineage = target / "exp_race"
    if found:
        lineage.mkdir()

    change = lineage.rmdir if found else lineage.mkdir
    with _between_lookups(monkeypatch, "exp_race", found=found, change=change, on=FOLLOWED):
        assert store.lineage_run_dir("exp_race", recorded) == canonical(target) / "exp_race"


def _attempt_template(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ArtifactDigest:
    """A frozen LTspice settings template under ``tmp_path``, read as
    ``test_controlled_ltspice`` reads it off Windows: the profile reader is
    the Windows API, so the template's values are given there."""
    from ltspice_mcp.lib import controlled_ltspice
    from tests.test_controlled_ltspice import _QUERY, _template

    template = _template(tmp_path)
    monkeypatch.setattr("ltspice_mcp.lib.controlled_ltspice.time.time", lambda: _QUERY)
    if sys.platform != "win32":
        monkeypatch.setattr(sys, "platform", "win32")
        monkeypatch.setattr(
            controlled_ltspice,
            "_read_profile",
            lambda path: {
                ("options", "uuid"): "12345678901234567890",
                ("options", "captureanalytics"): "false",
                ("options", "lastwebupdatequery"): str(_QUERY),
            },
        )
    return template


def test_ltspice_attempt_while_its_folder_appears(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from ltspice_mcp.lib import controlled_ltspice

    template = _attempt_template(tmp_path, monkeypatch)
    path = tmp_path / "startup" / "attempt-1" / "LTspice.ini"

    with _between_lookups(
        monkeypatch, "attempt-1", found=False, change=path.parent.mkdir, on=STAT
    ):
        controlled_ltspice.prepare_attempt_ini(template, path, tmp_path)

    assert path.is_file()


@needs_controlled_startup
def test_ngspice_startup_while_its_folder_appears(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from ltspice_mcp.lib.experiment_inputs import prepare_startup

    store = Store(tmp_path)
    store.run_dir("exp_root").mkdir(parents=True)
    startup = store.run_dir("exp_root") / "startup"

    with _between_lookups(monkeypatch, "startup", found=False, change=startup.mkdir, on=STAT):
        prepare_startup(store, "exp_root")

    assert store.recovery_spinit("exp_root").is_file()


def test_a_store_spelled_with_parent_parts_after_a_link(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """``..`` after a link is folded as the system folds it, for the parser
    folder as for every other store path, so the parser folder lies in the
    same store as every other store path. Where that store is depends on the
    platform's own rule: Windows folds ``..`` by spelling before it follows a
    link (``ntpath.realpath`` normalizes the path first), so the store lies
    beside the link; POSIX follows the link first, so it lies beside the
    link's target."""
    deep = tmp_path / "real" / "deep" / "w"
    deep.mkdir(parents=True)
    working = tmp_path / "w"
    symlink_or_skip(working, deep, target_is_directory=True)
    monkeypatch.setenv("LTSPICE_MCP_STORE_DIR", "../stores")
    store = Store(working)

    directory = store.parser_dir("p")

    assert directory.parent == store.experiments_dir.parent / "parsing"
    if sys.platform == "win32":
        expected, other = tmp_path / "stores", tmp_path / "real" / "deep" / "stores"
    else:
        expected, other = tmp_path / "real" / "deep" / "stores", tmp_path / "stores"
    assert canonical(expected) in directory.parents
    assert canonical(other) not in directory.parents


def test_a_link_between_a_path_and_its_root_is_seen(tmp_path: Path):
    """A link below the root that leads back above it: the root is met again
    below the link, and the link is still what redirects."""
    root = tmp_path / "root"
    root.mkdir()
    symlink_or_skip(root / "a", tmp_path, target_is_directory=True)

    assert redirect_below(root / "a" / "root" / "x", root) == root / "a"


@pytest.mark.parametrize(
    ("spelled", "resolved", "expected"),
    [
        ("C:\\w\\x", "\\\\?\\C:\\w\\x", "C:\\w\\x"),
        ("\\\\server\\share\\x", "\\\\?\\UNC\\server\\share\\x", "\\\\server\\share\\x"),
        ("\\\\?\\C:\\w", "\\\\?\\C:\\w", "\\\\?\\C:\\w"),
        ("C:\\w", "\\\\?\\Volume{1}\\w", "\\\\?\\Volume{1}\\w"),
        ("/w/x", "/w/x", "/w/x"),
    ],
)
def test_a_resolved_spelling_carries_no_prefix_the_input_did_not(
    spelled: str, resolved: str, expected: str
):
    assert without_added_prefix(spelled, resolved) == expected


@windows_only
def test_canonical_while_a_folder_appears_between_the_two_lookups(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    parent = tmp_path / "parsing"

    with _between_lookups(monkeypatch, "probe", found=False, change=parent.mkdir):
        spelled = canonical(parent / "probe")

    assert spelled == canonical(tmp_path) / "parsing" / "probe"


# -- a path that does redirect is still refused ------------------------------


def _junction_or_skip(link: Path, target: Path) -> None:
    """Create a junction ``link`` -> ``target``; a junction needs no privilege,
    so only another platform skips."""
    if sys.platform != "win32":
        pytest.skip("junctions are a native Windows facility")
    if TYPE_CHECKING:

        def create_junction(target: str, link: str) -> None: ...

    else:
        from _winapi import CreateJunction as create_junction

    create_junction(str(target), str(link))


_LINKS = {
    "symlink": lambda link, target: symlink_or_skip(link, target, target_is_directory=True),
    "junction": _junction_or_skip,
}


@pytest.mark.parametrize("kind", sorted(_LINKS))
def test_a_link_below_the_root_redirects(tmp_path: Path, kind: str):
    root = tmp_path / "root"
    elsewhere = tmp_path / "elsewhere"
    (root / "sub").mkdir(parents=True)
    elsewhere.mkdir()
    _LINKS[kind](root / "sub" / "link", elsewhere)

    assert redirect_below(root / "sub" / "link" / "file", root) == root / "sub" / "link"
    assert redirect_below(root / "sub" / "link", root) == root / "sub" / "link"
    assert redirect_below(root / "sub" / "missing" / "file", root) is None


@pytest.mark.parametrize("kind", sorted(_LINKS))
def test_the_root_itself_may_redirect(tmp_path: Path, kind: str):
    """What lies at and above the root is the caller's choice (a store kept
    behind a link is still the store); only what lies below it is asked."""
    target = tmp_path / "target"
    (target / "sub").mkdir(parents=True)
    root = tmp_path / "root"
    _LINKS[kind](root, target)

    assert redirect_below(root / "sub" / "file", root) is None


def test_a_store_spelled_with_parent_parts_still_parses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """A relative ``LTSPICE_MCP_STORE_DIR`` such as ``../stores`` spells the
    store root with ``..``; its parser folders are still its own."""
    working = tmp_path / "work"
    working.mkdir()
    monkeypatch.chdir(working)
    monkeypatch.setenv("LTSPICE_MCP_STORE_DIR", "../stores")
    store = Store(working)

    directory = store.parser_dir("parse_1")

    assert ".." not in directory.parts
    assert tmp_path / "stores" in directory.parents


def test_a_path_outside_the_root_or_with_parent_parts_is_refused(tmp_path: Path):
    root = tmp_path / "root"
    root.mkdir()
    (tmp_path / "other").mkdir()

    assert redirect_below(tmp_path / "other" / "file", root) == tmp_path / "other" / "file"
    escape = root / ".." / "other" / "file"
    assert redirect_below(escape, root) == escape
    assert redirect_below(Path("relative") / "file", root) == Path("relative") / "file"


def test_nothing_below_a_root_that_does_not_exist_redirects(tmp_path: Path):
    root = tmp_path / "not-yet" / "root"

    assert redirect_below(root / "a" / "b", root) is None
    assert redirect_below(tmp_path / "elsewhere", root) == tmp_path / "elsewhere"


def test_same_location_follows_a_link_to_the_folder_that_exists(tmp_path: Path):
    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "link"
    symlink_or_skip(link, target, target_is_directory=True)

    assert same_location(link / "missing" / "dir", target / "missing" / "dir")
    assert not same_location(link / "missing" / "dir", target / "missing" / "other")
    assert not same_location(tmp_path / "a", tmp_path / "b")


@pytest.mark.skipif(sys.platform != "win32", reason="short names are a Windows facility")
def test_same_location_and_containment_hold_over_a_short_name(tmp_path: Path):
    import ctypes

    long_dir = tmp_path / "a folder with a long name"
    (long_dir / "sub").mkdir(parents=True)
    buffer = ctypes.create_unicode_buffer(1024)
    if not ctypes.windll.kernel32.GetShortPathNameW(str(long_dir), buffer, 1024):  # type: ignore[attr-defined]
        pytest.skip("this volume keeps no short names")
    short_dir = Path(buffer.value)
    if short_dir == long_dir:
        pytest.skip("this volume keeps no short names")

    assert same_location(short_dir / "sub" / "missing", long_dir / "sub" / "missing")
    assert redirect_below(short_dir / "sub" / "missing", long_dir) is None


@pytest.mark.parametrize("kind", sorted(_LINKS))
def test_every_check_refuses_a_link_in_its_path(tmp_path: Path, kind: str):
    """The same input through every check that asks whether a path redirects."""
    from ltspice_mcp.lib import experiment_store
    from ltspice_mcp.lib.store import StoreError
    from tests.conftest import persist_experiment_record

    store = Store(tmp_path)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()

    # The parser folder of one parse.
    parsing = store.root / "parsing"
    parsing.mkdir(parents=True)
    _LINKS[kind](parsing / "linked", elsewhere)
    with pytest.raises(StoreError, match="must not redirect"):
        store.parser_dir("linked")

    # A record in a parser folder.
    directory = store.parser_dir("records")
    directory.mkdir()
    _LINKS[kind](directory / "result.json", elsewhere)
    with pytest.raises(StoreError):
        parser_file_in(directory, "result.json")

    # A job's lineage folder.
    store.runs_root().mkdir(parents=True)
    _LINKS[kind](store.run_dir("exp_linked"), elsewhere)
    with pytest.raises(StoreError):
        store.lineage_run_dir("exp_linked", store.run_dir("exp_linked"))

    # A job's record.
    circuit = tmp_path / "deck.cir"
    circuit.write_text("* deck\n.end\n", encoding="utf-8")
    path = persist_experiment_record(tmp_path, circuit, "exp_record")
    job = experiment_store.load_job_from_path(path, tmp_path)
    assert job is not None
    path.unlink()
    _LINKS[kind](path, elsewhere)
    with pytest.raises(ValueError, match="not contained"):
        experiment_store.save_job(job)
    with pytest.raises(ValueError, match="outside the working store"):
        experiment_store.load_job_from_path(path, tmp_path)


@pytest.mark.parametrize("kind", sorted(_LINKS))
def test_an_ltspice_attempt_through_a_link_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str
):
    from ltspice_mcp.lib import controlled_ltspice
    from ltspice_mcp.lib.recovery_records import RecoveryError
    from tests.test_controlled_ltspice import _QUERY, _template

    template = _template(tmp_path)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    _LINKS[kind](tmp_path / "startup" / "attempt-1", elsewhere)
    monkeypatch.setattr("ltspice_mcp.lib.controlled_ltspice.time.time", lambda: _QUERY)
    if sys.platform != "win32":
        # As ``test_controlled_ltspice`` runs these checks off Windows: the
        # profile reader is the Windows API, so the template's values are given.
        monkeypatch.setattr(sys, "platform", "win32")
        monkeypatch.setattr(
            controlled_ltspice,
            "_read_profile",
            lambda path: {
                ("options", "uuid"): "12345678901234567890",
                ("options", "captureanalytics"): "false",
                ("options", "lastwebupdatequery"): str(_QUERY),
            },
        )
    path = tmp_path / "startup" / "attempt-1" / "LTspice.ini"

    with pytest.raises(RecoveryError, match="escaped"):
        controlled_ltspice.prepare_attempt_ini(template, path, tmp_path)
    assert list(elsewhere.iterdir()) == []


# -- the guard ---------------------------------------------------------------

# Why each comparison of a resolved path below is not a redirect check.
_SOURCE_KEYS = (
    "keys files that exist and that this call reads, by their resolved name, to "
    "tell one from another (a visited set, a digest by file, the staged copy of "
    "a source); a missed match costs a duplicate entry or a rewrite of a path "
    "that reaches the same file, never a redirect let through"
)
_EXISTING_CONTAINMENT = (
    "follows links on purpose to check that a file that must exist lands in its "
    "folder; the read that follows fails anyway if the file is removed at that "
    "moment"
)
_NOT_A_PATH = "a name bound to a resolved path in the same function, compared as something else"
_SANDBOX = "the sandbox: where a caller's path lands, links followed on purpose"

#: Every comparison in the package with a resolved path on one side, with why
#: it is not a redirect check. A new one fails the test below: ask
#: ``redirect.redirect_below`` whether a path redirects,
#: ``redirect.same_location`` whether two paths are one place, and spell a
#: store folder with ``redirect.canonical``.
_RESOLVED_COMPARISONS_ALLOWED: dict[tuple[str, str], str] = {
    (
        "lib/controlled_ngspice.py",
        "driver.path.resolve(strict=True).is_relative_to(root.resolve(strict=True))",
    ): _EXISTING_CONTAINMENT,
    ("lib/experiment_inputs.py", "resolved.is_relative_to(resolved_root)"): _EXISTING_CONTAINMENT,
    ("lib/experiment_runner.py", "copied.is_relative_to(root)"): _EXISTING_CONTAINMENT,
    ("lib/pdk_native.py", "path.is_relative_to(cwd)"): _EXISTING_CONTAINMENT,
    ("lib/pathutil.py", "resolved.is_relative_to(allowed_dir.resolve())"): _SANDBOX,
    ("lib/deck_staging.py", "entry.path == source"): _SOURCE_KEYS,
    ("lib/deck_staging.py", "expected != actual_destination.resolve()"): _SOURCE_KEYS,
    ("lib/deck_staging.py", "owner != resolved"): _SOURCE_KEYS,
    ("lib/deck_staging.py", "path != source"): _SOURCE_KEYS,
    ("lib/deck_staging.py", "resolved == source"): _SOURCE_KEYS,
    ("lib/deck_staging.py", "resolved in processing"): _SOURCE_KEYS,
    ("lib/deck_staging.py", "source != authored"): _SOURCE_KEYS,
    ("lib/experiment_inputs.py", "path.resolve() in data"): _SOURCE_KEYS,
    ("lib/experiment_inputs.py", "resolved in spice"): _SOURCE_KEYS,
    ("lib/experiment_inputs.py", "resolved not in by_digest"): _SOURCE_KEYS,
    ("lib/experiment_inputs.py", "target not in inventory"): _SOURCE_KEYS,
    ("lib/experiment_store.py", "circuit in seen"): _SOURCE_KEYS,
    ("lib/instance_targeting.py", "candidate in by_path"): _SOURCE_KEYS,
    ("lib/instance_targeting.py", "candidate.resolve() in by_path"): _SOURCE_KEYS,
    ("lib/instance_targeting.py", "p.resolve() in self.original"): _SOURCE_KEYS,
    ("lib/instance_targeting.py", "target not in self.original"): _SOURCE_KEYS,
    ("lib/job_registry.py", "resolved in self._loaded_circuits"): _SOURCE_KEYS,
    ("lib/lint_rules.py", "(resolved, reference.section) in visited"): _SOURCE_KEYS,
    ("lib/lint_rules.py", "resolved in snapshot"): _SOURCE_KEYS,
    (
        "lib/native_inputs.py",
        "Path(origin.path).resolve() != files[edit.file_index].resolve()",
    ): _SOURCE_KEYS,
    (
        "lib/native_inputs.py",
        "expected.get(variant.path.resolve()) != variant.sha256",
    ): _SOURCE_KEYS,
    ("lib/native_inputs.py", "identity == root_identity"): _SOURCE_KEYS,
    ("lib/native_inputs.py", "item.path == variant.path.resolve()"): _SOURCE_KEYS,
    ("lib/native_inputs.py", "path not in self.cards"): _SOURCE_KEYS,
    ("lib/native_inputs.py", "previous != original"): _SOURCE_KEYS,
    ("lib/pdk_native.py", "len(set(dependency_paths)) != len(dependency_paths)"): _SOURCE_KEYS,
    ("lib/recent.py", "e.get('path') != resolved"): _SOURCE_KEYS,
    ("lib/runner_base.py", "(key, selected) in seen"): _SOURCE_KEYS,
    ("lib/simulator.py", "resolved not in found"): _SOURCE_KEYS,
    ("lib/variations.py", "source in written"): _SOURCE_KEYS,
    ("lib/variations.py", "case.circuit_id != circuit.circuit_id"): _NOT_A_PATH,
    ("lib/variations.py", "circuit.path.suffix.lower() in RUN_DECK_SUFFIXES"): _NOT_A_PATH,
    ("lib/wsl.py", "len(parts[2]) == 1"): _NOT_A_PATH,
    ("lib/wsl.py", "parts[0] == '/'"): _NOT_A_PATH,
    ("lib/wsl.py", "parts[1] == 'mnt'"): _NOT_A_PATH,
    (
        "tools/jobs.py",
        "(source.path.resolve() if source.path.exists() else source.path) == resolved_circuit",
    ): _SOURCE_KEYS,
    ("tools/jobs.py", "key in seen"): _SOURCE_KEYS,
}

#: Calls whose result is a resolved path.
_RESOLVING_METHODS = {"resolve", "realpath"}
_RESOLVING_FUNCTIONS = {"realpath", "canonical", "_resolved"}


def _resolves(node: ast.AST, names: set[str]) -> bool:
    """Whether ``node`` holds a call that resolves a path, or a name bound to one."""
    for inner in ast.walk(node):
        if isinstance(inner, ast.Call):
            func = inner.func
            if isinstance(func, ast.Attribute) and func.attr in _RESOLVING_METHODS:
                return True
            if isinstance(func, ast.Name) and func.id in _RESOLVING_FUNCTIONS:
                return True
        if isinstance(inner, ast.Name) and inner.id in names:
            return True
    return False


def _bound(target: ast.AST) -> list[str]:
    return [node.id for node in ast.walk(target) if isinstance(node, ast.Name)]


def _resolved_comparisons() -> set[tuple[str, str]]:
    found: set[tuple[str, str]] = set()
    for module, tree in modules().items():
        if module == "lib/redirect.py":
            continue
        for _scope, nodes in scopes(tree):
            names: set[str] = set()
            for node in nodes:
                if isinstance(node, ast.Assign) and _resolves(node.value, set()):
                    for target in node.targets:
                        names.update(_bound(target))
                elif (
                    isinstance(node, (ast.AnnAssign, ast.NamedExpr))
                    and node.value is not None
                    and _resolves(node.value, set())
                ):
                    names.update(_bound(node.target))
            for node in nodes:
                if isinstance(node, ast.Compare) and all(
                    isinstance(op, (ast.Eq, ast.NotEq, ast.In, ast.NotIn)) for op in node.ops
                ):
                    hit = any(_resolves(side, names) for side in [node.left, *node.comparators])
                elif (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "is_relative_to"
                ):
                    hit = _resolves(node.func.value, names) or any(
                        _resolves(arg, names) for arg in node.args
                    )
                else:
                    hit = False
                if hit:
                    found.add((module, ast.unparse(node)))
    return found


def test_no_check_compares_the_text_of_a_resolved_path():
    """Every ``==``, ``!=``, ``in``, ``not in`` and ``is_relative_to`` with a
    resolved path on one side: a call to ``resolve``, ``realpath``,
    ``canonical`` or ``_resolved`` there, or a name bound to one earlier in
    the same function (by an assignment, a tuple unpacking or ``:=``).

    A scan cannot see a resolved value that reaches the comparison through a
    function's result, an attribute or a property. Those are kept safe by
    taking every store folder's spelling from ``redirect.canonical``, which
    gives one spelling whatever a peer creates or removes meanwhile, and by
    asking ``redirect_below`` and ``same_location`` instead of comparing.
    ``redirect.py`` itself is not scanned: it is where these answers live."""
    assert _resolved_comparisons() == set(_RESOLVED_COMPARISONS_ALLOWED)
