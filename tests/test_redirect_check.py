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
import sys
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from ltspice_mcp.lib.redirect import redirect_below, same_location
from ltspice_mcp.lib.store import Store, parser_file_in
from tests._source_scan import modules, scopes
from tests.conftest import LIVENESS_S, symlink_or_skip

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
    does not run: the code answered without asking the system the question
    whose answer the window changes.
    """
    import ntpath

    real = ntpath._getfinalpathname  # pyright: ignore[reportAttributeAccessIssue]
    asked = threading.Event()
    done = threading.Event()
    closing = threading.Event()
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
            if asked.wait(LIVENESS_S) and not closing.is_set():
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
        closing.set()
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

#: Every comparison in the package with a ``resolve()`` result on one side,
#: with why it is not a redirect check. A new one fails the test below: ask
#: ``redirect.redirect_below`` whether a path redirects, and
#: ``redirect.same_location`` whether two paths are one place.
_RESOLVED_COMPARISONS_ALLOWED = {
    # Whether a relative include already reaches its staged file, so its text
    # can stay as written. A false "no" rewrites it to a path that reaches the
    # same file, so a wrong answer costs a rewrite, never a wrong file.
    ("lib/deck_staging.py", "expected != actual_destination.resolve()"),
    # Two spellings of a staged source and its recorded edit, both files that
    # exist and that no other thread changes: which file an edit belongs to.
    (
        "lib/native_inputs.py",
        "Path(origin.path).resolve() != files[edit.file_index].resolve()",
    ),
    ("lib/native_inputs.py", "item.path == variant.path.resolve()"),
    # A lookup by a resolved key, compared with a digest, not with a path.
    ("lib/native_inputs.py", "expected.get(variant.path.resolve()) != variant.sha256"),
    # Deck staging names each source by its strict resolve (a file that
    # exists and that staging reads, never one being created or removed), and
    # these ask whether two such names are one source: whether the deck was
    # exported from another, which source a staged copy came from, and
    # whether two sources would land on one staged name.
    ("lib/deck_staging.py", "source != authored"),
    ("lib/deck_staging.py", "owner != resolved"),
    ("lib/deck_staging.py", "resolved == source"),
    ("lib/deck_staging.py", "entry.path == source"),
    ("lib/deck_staging.py", "path != source"),
    # Native inputs: which original capture a staged file came from, and
    # whether a dependency is the root deck: keys of resolved names and
    # identity digests, of files that exist, not a redirect check.
    ("lib/native_inputs.py", "previous != original"),
    ("lib/native_inputs.py", "identity == root_identity"),
    # Whether two staged dependencies are one file: a count, not a location.
    ("lib/pdk_native.py", "len(set(dependency_paths)) != len(dependency_paths)"),
    # The recent-circuits list, keyed by a circuit's resolved name; a
    # mismatch keeps a duplicate entry, never a wrong file.
    ("lib/recent.py", "e.get('path') != resolved"),
    # Whether a path lies on a WSL drive mount: the form of a path.
    ("lib/wsl.py", "parts[0] == '/'"),
    ("lib/wsl.py", "parts[1] == 'mnt'"),
    ("lib/wsl.py", "len(parts[2]) == 1"),
    # Which of a circuit's jobs ran it: both sides name a circuit file that
    # exists, so neither is being created or removed by the server meanwhile.
    (
        "tools/jobs.py",
        "(source.path.resolve() if source.path.exists() else source.path) == resolved_circuit",
    ),
}


def _resolves(node: ast.AST, names: set[str]) -> bool:
    """Whether ``node`` holds a ``resolve()`` call, or a name bound to one."""
    return any(
        (
            isinstance(inner, ast.Call)
            and isinstance(inner.func, ast.Attribute)
            and inner.func.attr == "resolve"
        )
        or (isinstance(inner, ast.Name) and inner.id in names)
        for inner in ast.walk(node)
    )


def test_no_check_compares_the_text_of_a_resolved_path():
    """Every ``==`` or ``!=`` with a ``resolve()`` result on one side, called
    there or bound to a name earlier in the same function."""
    found: set[tuple[str, str]] = set()
    for module, tree in modules().items():
        for _scope, nodes in scopes(tree):
            names: set[str] = set()
            for node in nodes:
                if isinstance(node, (ast.Assign, ast.AnnAssign)) and node.value is not None:
                    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                    if _resolves(node.value, set()):
                        names.update(t.id for t in targets if isinstance(t, ast.Name))
            for node in nodes:
                if (
                    isinstance(node, ast.Compare)
                    and all(isinstance(op, (ast.Eq, ast.NotEq)) for op in node.ops)
                    and any(_resolves(side, names) for side in [node.left, *node.comparators])
                ):
                    found.add((module, ast.unparse(node)))
    assert found == _RESOLVED_COMPARISONS_ALLOWED
