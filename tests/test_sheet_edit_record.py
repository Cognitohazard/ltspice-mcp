"""The record of what every ``edit_schematic`` call answers (``tests/sheet_edit_record.py``).

Its comparison of sheets must not tell apart the ways two editors write one
sheet (record order across kinds, line endings, an attribute line's place, a
directive's trailing spaces), and must tell apart any change to a record. A
call on two spellings of one sheet must have one key, and a call answering
differently from the record must be found.

A call's key must hold what the call asked and the files it can read, and
nothing else, so an input field added later or a sheet left beside it changes
no key. A record whose entries no call matches any more must fail a run of the
whole suite, and must not fail a run of part of it.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path, PurePath, PurePosixPath, PureWindowsPath
from typing import Any

import pytest
from pydantic import Field, create_model

from ltspice_mcp.lib.schematic_ops import OpAddDirective
from ltspice_mcp.state import SessionState
from ltspice_mcp.tools.schematic_edit import EditSchematicInput
from tests import sheet_edit_record
from tests._asc_ops import build_sheet
from tests._schematic_fixtures import TESTS, suite_resolver
from tests.conftest import LIVENESS_S, symlink_or_skip
from tests.sheet_edit_record import key_of, scrubbed, sheet_form

_SHEET = [
    "Version 4",
    "SHEET 1 880 680",
    "WIRE 16 96 112 96",
    "FLAG 16 96 in",
    "SYMBOL res 112 48 R0",
    "SYMATTR InstName R1",
    "SYMATTR Value 1k",
    "TEXT 16 200 Left 2 !.op",
]


def _bytes(lines: list[str], newline: str = "\n", encoding: str = "ascii") -> bytes:
    return "".join(line + newline for line in lines).encode(encoding)


class TestTheSheetComparison:
    def test_reads_two_writings_of_one_sheet_as_one(self):
        regrouped = [_SHEET[0], _SHEET[1], *_SHEET[4:7], _SHEET[2], _SHEET[3], _SHEET[7]]
        attributes_swapped = [*_SHEET[:5], _SHEET[6], _SHEET[5], _SHEET[7]]
        trailing_space = [*_SHEET[:7], _SHEET[7] + "  "]
        first = sheet_form(_bytes(_SHEET))
        assert sheet_form(_bytes(_SHEET, "\r\n")) == first
        assert sheet_form(_bytes(regrouped)) == first
        assert sheet_form(_bytes(attributes_swapped)) == first
        assert sheet_form(_bytes(trailing_space)) == first

    @pytest.mark.parametrize(
        ("index", "line"),
        [
            (2, "WIRE 16 96 128 96"),
            (3, "FLAG 16 96 out"),
            (4, "SYMBOL res 112 48 R90"),
            (6, "SYMATTR Value 2k"),
            (7, "TEXT 16 200 Left 2 !.tran 1m"),
        ],
    )
    def test_tells_any_changed_record_apart(self, index: int, line: str):
        changed = list(_SHEET)
        changed[index] = line
        assert sheet_form(_bytes(changed)) != sheet_form(_bytes(_SHEET))

    def test_tells_records_in_another_order_within_a_kind_apart(self):
        two_wires = [*_SHEET[:3], "WIRE 0 0 16 0", *_SHEET[3:]]
        swapped = [*_SHEET[:2], "WIRE 0 0 16 0", _SHEET[2], *_SHEET[3:]]
        assert sheet_form(_bytes(two_wires)) != sheet_form(_bytes(swapped))

    def test_tells_the_codec_apart(self):
        micro = [*_SHEET[:6], "SYMATTR Value 1µ"]
        assert sheet_form(_bytes(micro, encoding="cp1252")) != sheet_form(
            _bytes(micro, encoding="utf-8")
        )


class TestTextReadsTheSameOnEveryPlatform:
    """The record is made on one platform and compared on both."""

    @pytest.mark.parametrize(
        ("folder", "text"),
        [
            (PurePosixPath("/tmp/run/work"), "Cannot open /tmp/run/work/sub/bus.asc"),
            (PureWindowsPath(r"C:\Temp\run\work"), r"Cannot open C:\Temp\run\work\sub\bus.asc"),
        ],
    )
    def test_a_folder_reads_as_work(self, folder: PurePath, text: str):
        assert scrubbed(text, (folder,), {}) == "Cannot open WORK/sub/bus.asc"

    @pytest.mark.parametrize("drive", ["C", "D"])
    def test_an_absolute_posix_argument_reads_as_given_on_any_drive(self, drive: str):
        text = rf"Path {drive}:\outside\ref.cir is outside allowed directories"
        assert (
            scrubbed(text, (), {}, ("/outside/ref.cir",))
            == "Path /outside/ref.cir is outside allowed directories"
        )

    @pytest.mark.skipif(
        sys.platform == "win32", reason="the arguments the record maps are POSIX paths"
    )
    def test_an_absolute_posix_argument_reads_as_given_through_a_symlink(self, tmp_path: Path):
        """macOS resolves /etc to /private/etc; any host resolves a linked folder."""
        (tmp_path / "real").mkdir()
        symlink_or_skip(tmp_path / "link", tmp_path / "real", target_is_directory=True)
        given = str(tmp_path / "link" / "ref.cir")
        text = f"Path {os.path.realpath(given)} is outside allowed directories"
        assert scrubbed(text, (), {}, (given,)) == f"Path {given} is outside allowed directories"

    def test_the_posix_spelling_is_left_alone(self):
        text = "Path /outside/ref.cir is outside allowed directories"
        assert scrubbed(text, (), {}, ("/outside/ref.cir",)) == text

    def test_a_digest_of_bytes_a_test_wrote_reads_as_sha256(self):
        """Such bytes have the platform's line breaks; a call's own sheets keep their tokens."""
        before, written = "a" * 64, "b" * 64
        text = f"expected_sha256 {before} does not match the current file ({written})"
        assert (
            scrubbed(text, (), {before: "BEFORE"})
            == "expected_sha256 BEFORE does not match the current file (SHA256)"
        )


def test_a_call_on_two_writings_of_one_sheet_has_one_key(tmp_path: Path):
    """Each handed the digest of its own sheet, as a caller does."""
    keys = []
    for name, newline in (("lf", "\n"), ("crlf", "\r\n")):
        folder = tmp_path / name
        folder.mkdir()
        target = folder / "sheet.asc"
        target.write_bytes(_bytes(_SHEET, newline))
        args = EditSchematicInput.model_validate(
            {
                "path": str(target),
                "expected_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                "ops": [{"op": "add_directive", "instruction": ".tran 1m"}],
            }
        )
        keys.append(
            key_of(target, target.read_bytes(), args, (folder,), resolver=suite_resolver(target))
        )
    assert keys[0] == keys[1]


@pytest.mark.parametrize("encoding", ["ascii", "utf-16-le"])
def test_a_file_beside_the_sheet_makes_one_key_in_either_line_ending(
    tmp_path: Path, encoding: str
):
    """A test writes the files beside its sheet with the platform's line breaks."""
    keys = []
    for name, newline in (("lf", "\n"), ("crlf", "\r\n")):
        folder = tmp_path / name
        folder.mkdir()
        target = folder / "sheet.asc"
        target.write_bytes(_bytes(_SHEET))
        reference = _bytes(["* reference", "R1 in 0 1k", ".end"], newline, encoding)
        (folder / "ref.cir").write_bytes(reference)
        args = EditSchematicInput.model_validate(
            {
                "path": str(target),
                "ops": [{"op": "add_directive", "instruction": ".tran 1m"}],
            }
        )
        keys.append(
            key_of(target, target.read_bytes(), args, (folder,), resolver=suite_resolver(target))
        )
    assert keys[0] == keys[1]


def test_a_call_whose_reply_differs_by_platform_has_a_key_of_its_own(tmp_path: Path):
    target = tmp_path / "sheet.asc"
    target.write_bytes(_bytes(_SHEET))
    args = EditSchematicInput.model_validate(
        {"path": str(target), "ops": [{"op": "add_directive", "instruction": ".op"}]}
    )
    resolver = suite_resolver(target)
    plain = key_of(target, target.read_bytes(), args, (tmp_path,), resolver=resolver)
    assert (
        key_of(
            target, target.read_bytes(), args, (tmp_path,), resolver=resolver, per_platform=True
        )
        != plain
    )


class _AnswersOtherwise(dict[str, Any]):
    """A record that holds every key, each answered otherwise."""

    def get(self, key: str, default: Any = None) -> Any:
        return {"sheet": "recorded", "reply": "recorded"}


async def test_a_call_answering_differently_from_the_record_is_found(
    asc_state: SessionState, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(sheet_edit_record, "_recording", False)
    monkeypatch.setattr(sheet_edit_record, "_record", _AnswersOtherwise())
    await build_sheet(asc_state, "found", [{"op": "add_directive", "instruction": ".op"}])
    found = sheet_edit_record.take_mismatches()
    assert len(found) == 1
    assert found[0].startswith("call ")


def _key(
    target: Path, payload: dict[str, Any], model: type[EditSchematicInput] = EditSchematicInput
):
    args = model.model_validate({"path": str(target), **payload})
    return key_of(
        target, target.read_bytes(), args, (target.parent,), resolver=suite_resolver(target)
    )


_DIRECTIVE = {"ops": [{"op": "add_directive", "instruction": ".tran 1m"}]}


class _DirectiveWithMore(OpAddDirective):
    """``add_directive`` with an optional field it does not have yet."""

    spacing: int = 16


class TestTheKeyHoldsWhatTheCallAsked:
    """A call is keyed on the arguments it gives, not on every field the input has."""

    @pytest.fixture
    def target(self, tmp_path: Path) -> Path:
        target = tmp_path / "sheet.asc"
        target.write_bytes(_bytes(_SHEET))
        return target

    def test_an_optional_field_added_to_the_input_changes_no_key(self, target: Path):
        """As ``window_pid`` did when it was added: no call gave it, and every
        key changed, so the record matched no call."""
        widened = create_model(
            "_InputWithMore",
            __base__=EditSchematicInput,
            window_title=(str | None, None),
            windows=(list[int], Field(default_factory=list)),
            ops=(list[_DirectiveWithMore], ...),
        )
        assert _key(target, _DIRECTIVE, widened) == _key(target, _DIRECTIVE)

    def test_an_argument_given_at_its_default_is_the_argument_left_out(self, target: Path):
        defaults = EditSchematicInput.model_fields
        given = {
            "base": defaults["base"].default,
            "dry_run": False,
            "expected_sha256": None,
            "return_views": ["touched"],
            "view_limit": defaults["view_limit"].default,
            "ops": [
                {"op": "add_directive", "instruction": ".tran 1m", "kind": "directive", "size": 2}
            ],
        }
        assert _key(target, given) == _key(target, _DIRECTIVE)

    @pytest.mark.parametrize(
        "changed",
        [
            {"dry_run": True},
            # Unlike a reply's field, an argument left out reads as its default,
            # so an empty one asks for something else.
            {"return_views": []},
            {"ops": [{"op": "add_directive", "instruction": ".tran 1m", "kind": "comment"}]},
            {"ops": [{"op": "remove_directive", "instruction": ".tran 1m"}]},
        ],
        ids=["flag", "empty list", "nested field", "op"],
    )
    def test_an_argument_away_from_its_default_changes_the_key(
        self, target: Path, changed: dict[str, Any]
    ):
        assert _key(target, {**_DIRECTIVE, **changed}) != _key(target, _DIRECTIVE)


class TestTheKeyHoldsTheFilesBesideTheSheetAnEditReads:
    """A sheet or plot settings file left beside a sheet is never read by an
    edit of it, so it changes no key; one the edit reads does."""

    @pytest.fixture
    def target(self, tmp_path: Path) -> Path:
        target = tmp_path / "sheet.asc"
        target.write_bytes(_bytes(_SHEET))
        return target

    @pytest.mark.parametrize("name", ["Draft1.asc", "other.plt"])
    def test_a_file_the_edit_does_not_read_changes_no_key(self, target: Path, name: str):
        plain = _key(target, _DIRECTIVE)
        (target.parent / name).write_bytes(_bytes(_SHEET))
        assert _key(target, _DIRECTIVE) == plain

    @pytest.mark.parametrize(
        ("names", "payload"),
        [
            (["sheet.plt"], _DIRECTIVE),
            (["models.lib"], _DIRECTIVE),
            (["block.asy", "block.asc"], _DIRECTIVE),
            (["res.asc"], _DIRECTIVE),
            (["ref.asc"], {**_DIRECTIVE, "compare": {"reference": "ref.asc"}}),
        ],
        ids=["own plot settings", "library", "block sheet", "sheet of a part", "reference"],
    )
    def test_a_file_the_edit_reads_is_in_the_key(
        self, target: Path, names: list[str], payload: dict[str, Any]
    ):
        before = _key(target, payload)
        for name in names:
            (target.parent / name).write_bytes(_bytes(_SHEET))
        written = _key(target, payload)
        assert written != before
        (target.parent / names[-1]).write_bytes(_bytes([*_SHEET, "TEXT 0 0 Left 2 ;more"]))
        assert _key(target, payload) != written


# -- A record that has gone stale ---------------------------------------------
#
# These run a small suite of their own in a separate pytest process, with this
# plugin loaded as the suite's conftest loads it and the record in the inner
# suite's folder: how a run tells the whole suite from part of it is decided
# by pytest and xdist, so only a real run shows it.

_INNER_RUN_TIMEOUT_S = 10 * LIVENESS_S
"""A cap on a hang of the inner pytest process, which imports the server."""

_INNER_INI = "[pytest]\nasyncio_mode = auto\nasyncio_default_fixture_loop_scope = function\n"

_INNER_CONFTEST = """\
from pathlib import Path

from tests import sheet_edit_record

sheet_edit_record.RECORD = Path(__file__).with_name("record.json")
pytest_plugins = ["tests.conftest"]
"""

_INNER_TESTS = """\
import os

import pytest

from tests._asc_ops import build_sheet


def test_fails_where_asked():
    assert not os.environ.get("INNER_FAIL")


async def test_one(asc_state):
    await build_sheet(asc_state, "one", [{"op": "add_directive", "instruction": ".op"}])


async def test_two(asc_state):
    await build_sheet(asc_state, "two", [{"op": "add_directive", "instruction": ".tran 1m"}])


async def test_three(asc_state):
    # As a test needing a simulator the machine lacks skips before its call.
    if not os.environ.get("INNER_ALL"):
        pytest.skip("INNER_ALL is not set")
    await build_sheet(asc_state, "three", [{"op": "add_directive", "instruction": ".dc V1 0 1 1"}])


@pytest.mark.edit_reply_per_platform("stands in for a reply that differs by platform")
async def test_four(asc_state):
    await build_sheet(asc_state, "four", [{"op": "add_directive", "instruction": ".op"}])
"""

_OTHER_PLATFORM = "posix" if sys.platform == "win32" else "windows"


def _inner_suite(folder: Path, record: dict[str, Any] | None = None) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "pytest.ini").write_text(_INNER_INI, encoding="utf-8")
    (folder / "conftest.py").write_text(_INNER_CONFTEST, encoding="utf-8")
    (folder / "test_calls.py").write_text(_INNER_TESTS, encoding="utf-8")
    if record is not None:
        (folder / "record.json").write_text(json.dumps(record), encoding="utf-8")
    return folder


def _run_inner(folder: Path, *args: str, **env: str) -> subprocess.CompletedProcess[str]:
    """Run the inner suite in ``folder`` with ``args``, in an environment that
    has none of this run's pytest, xdist or recording settings."""
    environment = {
        name: value
        for name, value in os.environ.items()
        if not name.startswith(("PYTEST_", "COV_CORE_"))
        and name not in (sheet_edit_record.RECORD_ENV, "INNER_ALL", "INNER_FAIL")
    }
    environment["PYTHONPATH"] = os.pathsep.join(
        filter(None, [str(TESTS.parent), os.environ.get("PYTHONPATH")])
    )
    environment["PYTHONIOENCODING"] = "utf-8"
    environment.update(env)
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", "-p", "no:randomly", *args],
        cwd=folder,
        env=environment,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=_INNER_RUN_TIMEOUT_S,
        check=False,
    )


@pytest.fixture(scope="module")
def inner_record(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """The record a recording of the whole inner suite writes."""
    folder = _inner_suite(tmp_path_factory.mktemp("inner_recording"))
    ran = _run_inner(folder, INNER_ALL="1", **{sheet_edit_record.RECORD_ENV: "1"})
    assert ran.returncode == 0, ran.stdout + ran.stderr
    return json.loads((folder / "record.json").read_text(encoding="utf-8"))


def _rotated(record: dict[str, Any]) -> dict[str, Any]:
    """``record`` with every key changed, as an input field added to every call
    once changed them."""
    return {hashlib.sha256(key.encode()).hexdigest()[:16]: entry for key, entry in record.items()}


def _made_by(record: dict[str, Any], test: str) -> str:
    return next(key for key, entry in record.items() if f"test_calls.py::{test}" in entry["tests"])


def _said(ran: subprocess.CompletedProcess[str]) -> str:
    return ran.stdout + ran.stderr


def test_the_record_names_the_tests_that_made_each_call(inner_record: dict[str, Any]):
    tests = sorted(test for entry in inner_record.values() for test in entry["tests"])
    assert tests == [f"test_calls.py::test_{n}" for n in ("four", "one", "three", "two")]
    platforms = {entry.get("platform") for entry in inner_record.values()}
    assert platforms == {None, sheet_edit_record._PLATFORM}
    assert inner_record[_made_by(inner_record, "test_four")]["platform"] == (
        sheet_edit_record._PLATFORM
    )


def test_a_record_its_calls_still_match_passes_a_run_of_the_whole_suite(
    tmp_path: Path, inner_record: dict[str, Any]
):
    """The call of a test skipped here is not made, and its entry is not
    stale; a varying entry is matched by its key."""
    record = dict(inner_record)
    varying = _made_by(record, "test_two")
    record[varying] = {"varies": True, "tests": record[varying]["tests"]}
    ran = _run_inner(_inner_suite(tmp_path, record))
    assert ran.returncode == 0, _said(ran)
    assert "2 compared, 1 varying" in ran.stdout
    assert "stale" not in ran.stdout


@pytest.mark.parametrize("parallel", [[], ["-n", "2"]], ids=["serial", "xdist"])
def test_a_record_no_call_matches_fails_a_run_of_the_whole_suite(
    tmp_path: Path, inner_record: dict[str, Any], parallel: list[str]
):
    """Every key rotated, and one entry whose test is gone: each is stale."""
    record = _rotated(inner_record)
    gone = next(iter(inner_record.values()))
    record["0" * 16] = {**gone, "tests": ["test_calls.py::test_gone"]}
    ran = _run_inner(_inner_suite(tmp_path, record), *parallel, INNER_ALL="1")
    assert ran.returncode == 1, _said(ran)
    assert "3 not recorded, 1 not recorded on this platform" in ran.stdout
    assert f"record.json is stale: {len(record)} of its entries matched no call" in ran.stdout
    assert "test_calls.py::test_gone" in ran.stdout
    assert "LTSPICE_MCP_RECORD_SHEET_EDITS=1" in ran.stdout


@pytest.mark.parametrize(
    "partial",
    [
        ["-k", "one or two"],
        ["test_calls.py"],
        ["test_calls.py::test_one"],
        ["-n", "2", "-k", "one"],
    ],
    ids=["deselected", "a file", "a test", "xdist deselected"],
)
def test_a_run_of_part_of_the_suite_does_not_fail_on_a_stale_record(
    tmp_path: Path, inner_record: dict[str, Any], partial: list[str]
):
    """It cannot tell an entry no test makes from one whose test did not run,
    so it reports only the entries of tests that passed without making them."""
    ran = _run_inner(_inner_suite(tmp_path, _rotated(inner_record)), *partial, INNER_ALL="1")
    assert ran.returncode == 0, _said(ran)
    assert "is stale" not in ran.stdout
    assert "matched no call of a test that passed" in ran.stdout
    assert "not a run of the whole suite" in ran.stdout


def test_a_run_stopped_early_does_not_take_the_tests_it_did_not_run_for_gone(
    tmp_path: Path, inner_record: dict[str, Any]
):
    ran = _run_inner(
        _inner_suite(tmp_path, _rotated(inner_record)), "-x", INNER_ALL="1", INNER_FAIL="1"
    )
    assert ran.returncode == 1, _said(ran)
    assert "1 failed" in ran.stdout
    assert "is stale" not in ran.stdout


def test_an_entry_kept_for_the_other_platform_is_not_stale_here(
    tmp_path: Path, inner_record: dict[str, Any]
):
    """A recording on the other platform keeps its own keys for the calls of a
    test marked ``edit_reply_per_platform``; this platform's call matches none."""
    record = dict(inner_record)
    kept = record.pop(_made_by(record, "test_four"))
    record["f" * 16] = {**kept, "platform": _OTHER_PLATFORM}
    ran = _run_inner(_inner_suite(tmp_path, record), INNER_ALL="1")
    assert ran.returncode == 0, _said(ran)
    assert "3 compared, 1 not recorded on this platform" in ran.stdout
    assert "stale" not in ran.stdout


def test_a_recording_of_part_of_the_suite_writes_nothing(
    tmp_path: Path, inner_record: dict[str, Any]
):
    """Its record would hold only the calls of the tests that ran."""
    folder = _inner_suite(tmp_path, inner_record)
    ran = _run_inner(folder, "-k", "one", **{sheet_edit_record.RECORD_ENV: "1"})
    assert ran.returncode == 0, _said(ran)
    assert "record.json not written: not a run of the whole suite" in ran.stdout
    assert json.loads((folder / "record.json").read_text(encoding="utf-8")) == inner_record
