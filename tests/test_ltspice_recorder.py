"""The LTspice recorder, the recordings it committed, and whether a build still matches them.

Three layers, all in this module because they share one object: the tree under
``tests/fixtures/ltspice_recorded``.

* The recorder's own mechanics (scrubbing, the privacy guard, the settings
  copy) run everywhere, on bytes built here.
* The committed tree is checked for integrity everywhere: every file a manifest
  lists is present with the digest it was recorded with, every input is the
  one that was run, and every behaviour in the case list is recorded on every
  build or says why it is not.
* With ``LTSPICE_MCP_RUN_LTSPICE_INTEGRATION=1`` each installed build is
  recorded again into a temporary directory and compared with what is
  committed, so a release that changes behaviour fails here by name. A build
  that is not installed skips with that reason.
"""

from __future__ import annotations

import codecs
import json
import re
import struct
from pathlib import Path

import pytest

from ltspice_mcp.lib import pe_menu
from tests import ltspice_recorder as recorder
from tests._ltspice_recorded import installed_counterpart
from tests.ltspice_recorder import (
    BEHAVIOUR_KEYS,
    FIXTURES,
    INPUTS,
    MANIFEST,
    NEUTRAL_DATE,
    NEUTRAL_DIR,
    NEUTRAL_HOME,
    RecorderError,
    Scrubber,
    assert_private,
    load_cases,
    load_manifest,
    neutral_settings,
    read_settings,
    recorded_builds,
    sha256_bytes,
    split_raw,
)
from tests.privacy_scan import scan_bytes

REPO = Path(__file__).resolve().parents[1]
CASES = load_cases()
BUILDS = recorded_builds()

# A made-up home directory, under a name the repository's privacy scan takes
# for a placeholder; the scrubber's own stand-in is a different one.
WORK = "C:\\Users\\dev\\AppData\\Local\\Temp\\rec\\ltspice26\\raw__tran"
HOME = "C:\\Users\\dev"

# A user name and the ANSI code page of a machine it would be found on. The
# katakana one is two bytes a character with a capital letter for the second,
# and the first character of the last has a backslash for its second byte.
LOCAL_NAMES = [
    ("cp936", "\u7528\u6237\u540d"),
    ("cp932", "\u30a2\u30a4\u30a6"),
    ("cp932", "\u30bd\u30d5\u30c8"),
    ("cp1251", "\u0418\u0432\u0430\u043d"),
]


#: XVII names itself only in a raw's Command field, and on the Alternate solver
#: that field ends with the solver. The build is the same one, but
#: ``reported_build`` reads the whole field as its name.
SOLVER_IN_THE_BUILD_NAME_ON_THE_SERVER_ONLY = {("ltspice17", "deck/solver_setting_1")}


def scrubber() -> Scrubber:
    return Scrubber.for_run(Path(WORK), home=Path(HOME))


def utf16(text: str) -> bytes:
    return text.encode("utf-16-le")


class TestScrubbing:
    """What a recording carries from the machine it ran on is rewritten, in
    the file's own encoding, and nothing else is."""

    def test_the_run_directory_becomes_the_neutral_one(self):
        log = f"LTspice 26.1.1 for Windows\nCircuit: {WORK}\\tran.cir\n".encode()
        assert scrubber().bytes(log, "tran.log") == (
            b"LTspice 26.1.1 for Windows\nCircuit: " + NEUTRAL_DIR.encode() + b"\\tran.cir\n"
        )

    def test_a_path_is_found_in_either_separator_and_any_case(self):
        forward = WORK.replace("\\", "/").upper()
        scrubbed = scrubber().bytes(f"* {forward}/sheet.asc\n".encode(), "sheet.net")
        # The neutral directory is spelled with the separator the file used.
        assert scrubbed == b"* " + NEUTRAL_DIR.replace("\\", "/").encode() + b"/sheet.asc\n"

    def test_the_home_directory_becomes_the_placeholder_home(self):
        net = f".lib {HOME}\\Documents\\LTspiceXVII\\lib\\cmp\\standard.mos\r\n".encode()
        assert scrubber().bytes(net, "sheet.net") == (
            b".lib "
            + NEUTRAL_HOME.encode()
            + b"\\Documents\\LTspiceXVII\\lib\\cmp\\standard.mos\r\n"
        )

    def test_a_run_directory_inside_the_home_directory_is_replaced_whole(self):
        # Replacing the home prefix first would leave the rest of the run
        # directory behind it.
        scrubbed = scrubber().bytes(f"Circuit: {WORK}\\a.cir\n".encode(), "a.log")
        assert b"AppData" not in scrubbed
        assert NEUTRAL_HOME.encode() not in scrubbed

    @pytest.mark.parametrize(
        "line",
        [
            "Start Time: Tue Oct  6 18:08:06 2026",  # LTspice 26 log: space-padded day
            "Date: Tue Oct 06 18:08:07 2026",  # LTspice XVII log and raw: zero-padded day
            "Date: Fri Dec 25 23:59:59 2026",
        ],
    )
    def test_a_date_is_replaced_by_the_fixed_one(self, line: str):
        key = line.split(":", 1)[0]
        assert scrubber().text(f"x\n{line}\ny\n") == f"x\n{key}: {NEUTRAL_DATE}\ny\n"

    def test_duration_and_thread_count_are_fixed(self):
        text = "Maximum thread count: 32\nTotal elapsed time: 12.345 seconds.\n"
        assert scrubber().text(text) == (
            "Maximum thread count: 1\nTotal elapsed time: 0.000 seconds.\n"
        )

    def test_the_matrix_compiler_report_is_replaced(self):
        # LTspice XVII keeps whichever of two compilers was faster on this run.
        for report in ("off  [0.0]/0.0/0.0", "175 bytes object code size  0.0/0.0/[0.0]"):
            assert scrubber().text(f"Matrix Compiler2: {report}\n") == (
                "Matrix Compiler2: (timing-dependent)\n"
            )

    def test_a_byte_that_is_not_utf8_survives(self):
        # LTspice XVII writes a degree sign as the single cp1252 byte B0.
        log = b"Circuit: * t\n.step temp=-40\xb0C\n"
        assert scrubber().bytes(log, "t.log") == log

    def test_a_utf16_log_stays_utf16(self):
        # The form LTspice XVII leaves when a run fails.
        log = utf16(f"Circuit: * t\n\nFatal Error: cannot open {WORK}\\x.lib\n")
        scrubbed = scrubber().bytes(log, "t.log")
        assert scrubbed == utf16(
            f"Circuit: * t\n\nFatal Error: cannot open {NEUTRAL_DIR}\\x.lib\n"
        )

    def test_a_raw_header_is_scrubbed_and_its_samples_are_not(self):
        header = f"Title: {WORK}\\tran.cir\nDate: Tue Oct  6 18:08:06 2026\nBinary:\n"
        # Samples that happen to spell the run directory must come through as written.
        samples = utf16(WORK) + bytes(range(256))
        scrubbed = scrubber().bytes(utf16(header) + samples, "tran.raw")
        assert scrubbed == (
            utf16(f"Title: {NEUTRAL_DIR}\\tran.cir\nDate: {NEUTRAL_DATE}\nBinary:\n") + samples
        )

    def test_an_eight_bit_raw_header_is_scrubbed(self):
        # LTspice 26 writes its ASCII raw as 8-bit text, header included.
        raw = f"Title: {WORK}\\tran.cir\nValues:\n0\t\t0.0e+00\n".encode()
        assert scrubber().bytes(raw, "tran.raw") == (
            f"Title: {NEUTRAL_DIR}\\tran.cir\nValues:\n0\t\t0.0e+00\n".encode()
        )

    def test_a_raw_cut_off_inside_its_header_is_all_header(self):
        cut = utf16(f"Title: {WORK}\\tran.cir\nPlotname: Trans")
        assert split_raw(cut) == (cut, b"")
        assert utf16(NEUTRAL_DIR) in scrubber().bytes(cut, "tran.raw")

    @pytest.mark.parametrize(
        "name", ["\u7528\u6237\u540d", "Zo\u00eb", "\u05e9\u05dc\u05d5\u05dd"]
    )
    def test_a_home_directory_as_xvii_spells_it_is_replaced(self, name):
        # LTspice XVII writes a path in cp1252 on a machine of any code page,
        # with a question mark for each character cp1252 does not have.
        home = f"C:\\Users\\{name}"
        local = Scrubber.for_run(Path(f"{home}\\rec"), home=Path(home))
        written = f".lib {home}\\Documents\\LTspiceXVII\\lib\\cmp\\standard.mos\r\n"
        assert local.bytes(written.encode("cp1252", "replace"), "sheet.net") == (
            b".lib "
            + NEUTRAL_HOME.encode()
            + b"\\Documents\\LTspiceXVII\\lib\\cmp\\standard.mos\r\n"
        )

    @pytest.mark.parametrize(("ansi", "name"), LOCAL_NAMES)
    def test_a_home_directory_in_the_machines_own_code_page_is_replaced(self, ansi, name):
        # Not a spelling either recorded build writes; covered so the scrub
        # does not depend on that.
        home = f"C:\\Users\\{name}"
        local = Scrubber.for_run(Path(f"{home}\\rec"), home=Path(home), ansi=ansi)
        net = f".lib {home}\\Documents\\LTspiceXVII\\lib\\cmp\\standard.mos\r\n".encode(ansi)
        assert local.bytes(net, "sheet.net") == (
            b".lib "
            + NEUTRAL_HOME.encode()
            + b"\\Documents\\LTspiceXVII\\lib\\cmp\\standard.mos\r\n"
        )

    def test_this_machines_code_page_is_one_python_can_encode_in(self):
        codecs.lookup(recorder.ansi_codec())


class TestPrivacyGuard:
    """A recording that still names the recording machine is refused."""

    @pytest.mark.parametrize(("ansi", "name"), LOCAL_NAMES)
    def test_a_name_in_the_machines_own_code_page_is_refused(self, ansi, name):
        net = f".lib C:\\Users\\{name}\\models.lib\r\n".encode(ansi)
        with pytest.raises(RecorderError):
            assert_private({"sheet.net": net}, [name], ansi=ansi)

    def test_a_clean_recording_passes(self):
        assert_private({"a.log": b"Circuit: C:\\recording\\a.cir\n"}, ["Someone", HOME])

    @pytest.mark.parametrize("encode", [str.encode, utf16])
    def test_a_surviving_name_is_refused_in_either_encoding(self, encode):
        with pytest.raises(RecorderError) as refused:
            assert_private({"a.log": encode("Circuit: D:\\work\\SOMEONE\\a.cir\n")}, ["Someone"])
        # The message names the file, never the private string.
        assert "a.log" in str(refused.value)
        assert "someone" not in str(refused.value).lower()

    def test_a_raw_header_is_checked_and_its_samples_are_not(self):
        header = utf16("Title: * t\nBinary:\n")
        assert_private({"t.raw": header + b"someone"}, ["Someone"])
        with pytest.raises(RecorderError):
            assert_private({"t.raw": utf16("Title: Someone\nBinary:\n")}, ["Someone"])


class TestSettingsCopy:
    """The copy of a build's settings file a case runs against."""

    # grid in [Options] is the waveform window's grid, which changes what a
    # build saves; Grid in [Colors] is a colour.
    ANSI = (
        b"[Options]\r\nLastRunVersion=26.1.1\r\nDefaultTrtol=2\r\nNoGreekMus=true\r\n"
        b"grid=on\r\nSchFontSize=28\r\nBoldSchFont=true\r\nSchematicFont=3\r\n"
        b"ShowCheckSum=false\r\n[Colors]\r\nGrid=1\r\n"
        b"[Recent File List]\r\nFile1=C:\\x\\y.asc\r\n"
    )

    def test_keys_that_change_results_are_removed_and_the_rest_kept(self):
        copy = neutral_settings(self.ANSI, {})
        assert copy == (
            b"[Options]\r\nLastRunVersion=26.1.1\r\nShowCheckSum=false\r\n[Colors]\r\nGrid=1\r\n"
        )

    def test_a_default_on_record_stands_and_one_it_lacks_is_added(self):
        on_record = {"Solver": "0", "RadianMeasure": "false"}
        written_back = {"Solver": "1", "grid": "off", "RawTempDir": "C:\\Users\\dev\\raw"}
        assert recorder.merged_defaults(on_record, written_back) == {
            "Solver": "0",
            "RadianMeasure": "false",
            "grid": "off",
        }
        assert recorder.merged_defaults({}, written_back) == {"Solver": "1", "grid": "off"}

    def test_the_copy_is_never_empty_of_what_marks_a_used_install(self):
        # A build that starts on an empty settings file runs its first-launch
        # steps; what says the build has run before must survive the copy.
        assert b"LastRunVersion=26.1.1" in neutral_settings(self.ANSI, {})

    def test_a_case_can_set_a_key_back(self):
        copy = neutral_settings(self.ANSI, {"NoGreekMus": "true"})
        assert read_settings(copy, BEHAVIOUR_KEYS) == {"NoGreekMus": "true"}
        # Written into [Options], before the next section.
        assert copy.index(b"NoGreekMus=true") < copy.index(b"[Colors]")

    def test_a_utf16_file_stays_utf16(self):
        # LTspice 26 writes its settings as UTF-16 with a byte order mark.
        source = b"\xff\xfe" + self.ANSI.decode("ascii").encode("utf-16-le")
        copy = neutral_settings(source, {})
        assert copy.startswith(b"\xff\xfe")
        assert copy[2:].decode("utf-16-le") == neutral_settings(self.ANSI, {}).decode("ascii")

    def test_a_byte_outside_cp1252_round_trips(self):
        # The file is in the recording machine's code page, which need not be
        # cp1252: 0x81 has no cp1252 character, and 0x85 read as Latin-1 is a
        # character Python treats as a line break. Both must come back as read.
        source = b"[Options]\r\nSchematicFontName=\x81\x85\x40\r\n"
        assert neutral_settings(source, {}) == source


class TestCaseList:
    """``inputs/cases.toml`` is the inventory of modelled LTspice behaviour."""

    def test_every_behaviour_is_recorded_or_says_why_not(self):
        unexplained = [
            key
            for key, behaviour in CASES.behaviours.items()
            if not CASES.of(key) and not behaviour.evidence and not behaviour.unrecordable
        ]
        assert not unexplained, (
            "every LTspice behaviour the server models needs a recording, or a written "
            f"reason there is none: {unexplained}"
        )

    def test_every_behaviour_names_the_code_that_models_it(self):
        for key, behaviour in CASES.behaviours.items():
            assert behaviour.model, f"{key} names no code"
            for name in behaviour.model:
                path = (
                    REPO / name if name.startswith("tests/") else REPO / "src/ltspice_mcp" / name
                )
                assert path.exists(), f"{key}: {name} is not in the repository"

    def test_every_input_file_belongs_to_a_case(self):
        used = {name for case in CASES.cases for name in case.copies}
        present = {
            path.relative_to(INPUTS).as_posix()
            for path in INPUTS.rglob("*")
            if path.is_file() and path.name != recorder.CASES_FILE
        }
        assert present == used

    def test_a_case_stopped_part_way_is_marked_as_varying(self):
        # How far a stopped run got differs every time, so comparing its
        # samples would fail on a build that had not changed.
        for case in CASES.cases:
            if case.kind == "kill":
                assert case.volatile, case.case_id


@pytest.mark.parametrize("build", BUILDS)
class TestCommittedRecordings:
    """Each build's directory is exactly what its manifest says was recorded."""

    def test_the_manifest_identifies_the_build(self, build: str):
        manifest = load_manifest(FIXTURES / build)
        assert manifest["schema"] == recorder.MANIFEST_SCHEMA
        assert manifest["build"] == build
        executable = manifest["executable"]
        assert re.fullmatch(r"[0-9a-f]{64}", executable["sha256"])
        # The digest identifies the build; where it was installed is not kept.
        assert executable["name"] == Path(executable["name"]).name
        assert executable["name"].lower().endswith(".exe")
        assert build == f"ltspice{executable['file_version'].split('.')[0]}"
        assert manifest["reported_build"]

    def test_every_default_on_record_is_for_a_key_the_recorder_removes(self, build: str):
        """The defaults are what a build wrote back for the keys removed from
        its settings copy. LTspice XVII writes back the ones it knows; LTspice
        26 writes none, so what it does on a default is in its recorded files."""
        manifest = load_manifest(FIXTURES / build)
        defaults = manifest["settings"]["defaults"]
        assert {key.casefold() for key in defaults} <= BEHAVIOUR_KEYS
        assert bool(defaults) == (manifest["generation"] == "xvii")

    def test_every_applicable_case_is_recorded_and_no_other(self, build: str):
        manifest = load_manifest(FIXTURES / build)
        generation = manifest["generation"]
        expected = {
            case.case_id
            for case in CASES.cases
            if not case.builds or generation in case.builds or build in case.builds
        }
        assert set(manifest["cases"]) == expected

    def test_every_behaviour_with_a_case_is_recorded_on_this_build(self, build: str):
        recorded = {
            entry["behaviour"] for entry in load_manifest(FIXTURES / build)["cases"].values()
        }
        expected = {case.behaviour for case in CASES.cases}
        assert recorded == expected

    def test_every_recorded_file_is_present_and_unchanged(self, build: str):
        directory = FIXTURES / build
        listed: set[str] = set()
        for case_id, entry in load_manifest(directory)["cases"].items():
            for name, facts in entry["outputs"].items():
                data = (directory / name).read_bytes()
                assert len(data) == facts["bytes"], f"{case_id}: {name}"
                assert sha256_bytes(data) == facts["sha256"], f"{case_id}: {name}"
                listed.add(name)
        present = {
            path.relative_to(directory).as_posix()
            for path in directory.rglob("*")
            if path.is_file() and path.name != MANIFEST
        }
        assert present == listed

    def test_every_case_was_run_on_the_input_that_is_committed(self, build: str):
        stale = [
            f"{case_id}: {name}"
            for case_id, entry in load_manifest(FIXTURES / build)["cases"].items()
            for name, digest in entry["inputs"].items()
            if sha256_bytes((INPUTS / name).read_bytes()) != digest
        ]
        assert not stale, f"inputs changed since they were recorded; record again: {stale}"

    def test_every_command_is_the_one_the_server_launches(self, build: str):
        """A plot case, a save case and a draw case are the exceptions: the
        first runs the sheet in the window, as the person the sheet is handed
        to does, and the others only open it there."""
        manifest = load_manifest(FIXTURES / build)
        for case_id, entry in manifest["cases"].items():
            command = entry["command"]
            assert command[0] == manifest["executable"]["name"], case_id
            modes = {"netlist": ["-netlist"], "plot": ["-Run"], "save": [], "draw": []}
            mode = modes.get(entry["kind"], ["-Run", "-b"])
            assert command[1 : 1 + len(mode)] == mode, case_id
            assert command[1 + len(mode)].startswith("<dir>/"), case_id

    def test_nothing_recorded_names_a_person_or_a_machine(self, build: str):
        directory = FIXTURES / build
        found = sorted(
            f"{path.relative_to(directory).as_posix()}: {finding.category}"
            for path in directory.rglob("*")
            if path.is_file()
            for finding in scan_bytes(path.read_bytes())
        )
        assert found == []

    def test_every_run_names_the_build_the_way_the_manifest_does(self, build: str):
        """Build detection reads each recorded run's own output to the same answer.

        A run that failed before it wrote a banner, and left no raw, names no
        build; every other run names the one the manifest records.
        """
        from ltspice_mcp.lib.simulator_build import reported_build

        directory = FIXTURES / build
        manifest = load_manifest(directory)
        named = 0
        for case_id in manifest["cases"]:
            log = directory / f"{case_id}.log"
            raw = directory / f"{case_id}.raw"
            if not log.is_file():
                continue  # an export, or a case that kept only its raw
            detected = reported_build(log, raw if raw.is_file() and raw.stat().st_size else None)
            if detected is not None:
                expected = manifest["reported_build"]
                if (build, case_id) in SOLVER_IN_THE_BUILD_NAME_ON_THE_SERVER_ONLY:
                    expected += " -- Alternate Solver"
                assert detected == expected, case_id
                named += 1
        assert named > 50


def test_both_generations_are_recorded():
    """The suite's model of LTspice is checked against the current build and
    against XVII, the two generations the server tells apart."""
    generations = {load_manifest(FIXTURES / build)["generation"] for build in BUILDS}
    assert generations == {"current", "xvii"}


# --------------------------------------------------------------------------
# The opt-in tier: is each installed build still the one that was recorded?
# --------------------------------------------------------------------------

GROUPS = sorted({case.case_id.split("/", 1)[0] for case in CASES.cases})


@pytest.mark.parametrize("group", GROUPS)
@pytest.mark.parametrize("label", BUILDS)
def test_an_installed_build_still_behaves_as_recorded(label: str, group: str, tmp_path: Path):
    """Record the group again on the installed build and compare.

    A difference is a change in what LTspice does, or something of the
    machine or person recording that the recorder let through. Look at it
    first. For the one, fix the model if the server depended on the old
    behaviour, then record again with ``scripts/record_ltspice_fixtures.py``.
    For the other, teach the recorder: the scrubber for a path, date or
    duration, ``BEHAVIOUR_KEYS`` for a setting.
    """
    build = installed_counterpart(label)
    if isinstance(build, str):
        pytest.skip(build)
    only = [f"{group}/*"]
    recorder.record_build(build, tmp_path / "fresh", only=only, work_root=tmp_path / "work")
    differences = recorder.compare(FIXTURES / label, tmp_path / "fresh" / build.label, only=only)
    version = build.file_version or build.exe.name
    assert not differences, (
        f"LTspice {version} no longer matches the {label} recording:\n  "
        + "\n  ".join(differences[:40])
    )


class TestSettingsSwitch:
    """A build ignores ``-ini`` when the deck's path holds the letters ``ini``."""

    @pytest.mark.parametrize(
        "path",
        ["C:/rec/x_ini0.cir", "C:/rec/mini.cir", "C:/rec/MINI.cir", "D:/Gemini/rec/x.cir"],
    )
    def test_a_path_that_holds_them_is_refused(self, path: str):
        with pytest.raises(RecorderError, match="ignores -ini"):
            recorder.refuse_unread_settings(Path(path), "x")

    def test_the_letters_apart_are_not(self):
        recorder.refuse_unread_settings(Path("C:/rec/in_i.cir"), "x")

    def _cases(self, tmp_path: Path, extra: str = "") -> recorder.CaseFile:
        return load_cases(
            _inputs(
                tmp_path,
                "raw/x.cir",
                '[[case]]\nid = "raw/x_ini0"\nbehaviour = "b"\nsource = "raw/x.cir"\n' + extra,
            )
        )

    def test_a_case_named_with_them_is_refused_before_anything_runs(self, tmp_path: Path):
        with pytest.raises(RecorderError, match="ignores -ini"):
            self._cases(tmp_path)

    def test_a_case_that_names_no_settings_file_has_no_switch_to_lose(self, tmp_path: Path):
        assert self._cases(tmp_path, "settings = false\n").case("raw/x_ini0")


def _menu_item(flags: int, text: str, command: int | None = None) -> bytes:
    """One item of a classic menu template, the form both builds' menus are in."""
    head = flags.to_bytes(2, "little")
    if command is not None:
        head += command.to_bytes(2, "little")
    return head + text.encode("utf-16-le") + b"\0\0"


def _inputs(tmp_path: Path, source: str, cases: str) -> Path:
    """An inputs folder: the bare sheet ``source`` and a cases file of one
    behaviour, ``b``, and ``cases``."""
    (tmp_path / source).parent.mkdir()
    (tmp_path / source).write_text("Version 4\n", encoding="utf-8")
    (tmp_path / "cases.toml").write_text(
        '[behaviour.b]\nsummary = "s"\nmodel = ["m"]\n\n' + cases, encoding="utf-8"
    )
    return tmp_path


def _emf_record(kind: int, *fields: bytes) -> bytes:
    """One record of an enhanced metafile: its type, its size, what it holds."""
    body = b"".join(fields)
    return struct.pack("<II", kind, 8 + len(body)) + body


def _emf_point(x: int, y: int) -> bytes:
    return struct.pack("<2i", x, y)


def _emf_font(handle: int, height: int, escapement: int, weight: int, face: str) -> bytes:
    """An EMR_EXTCREATEFONTINDIRECTW record, the fields a drawing keeps filled in."""
    return _emf_record(
        82,
        struct.pack("<I5i", handle, height, 0, escapement, escapement, weight),
        bytes(8),
        face.encode("utf-16-le").ljust(64, b"\0"),
    )


def _emf_text(
    text: str, at: tuple[int, int], bounds: tuple[int, int, int, int], advances: list[int]
) -> bytes:
    """An EMR_EXTTEXTOUTW record, laid out as both builds write one: the string
    after the fixed fields, and the advances after the string."""
    chars = text.encode("utf-16-le")
    string_at = 76
    return _emf_record(
        84,
        struct.pack("<4i", *bounds),
        struct.pack("<Iff", 1, 1.0, 1.0),
        _emf_point(*at),
        struct.pack("<III", len(text), string_at, 0),
        struct.pack("<4i", 0, 0, -1, -1),
        struct.pack("<I", string_at + len(chars)),
        chars,
        struct.pack(f"<{len(text)}i", *advances),
    )


class TestDrawingOfASheet:
    """What is read from the image a build writes of a sheet."""

    def test_a_stroke_runs_from_where_the_pen_was(self):
        drawing = recorder.read_emf(
            _emf_record(27, _emf_point(1, 2))
            + _emf_record(54, _emf_point(3, 4))
            + _emf_record(54, _emf_point(5, 6))
            + _emf_record(27, _emf_point(7, 8))
            + _emf_record(54, _emf_point(9, 10))
        )
        assert drawing["strokes"] == [[1, 2, 3, 4], [3, 4, 5, 6], [7, 8, 9, 10]]

    def test_a_text_keeps_the_font_and_alignment_it_was_drawn_with(self):
        drawing = recorder.read_emf(
            _emf_font(1, 159, 0, 700, "Tahoma")
            + _emf_font(2, 159, 900, 700, "Tahoma")
            + _emf_record(37, struct.pack("<I", 2))
            + _emf_record(22, struct.pack("<I", 10))
            + _emf_text("66\u00b5", (5, 6), (1, 2, 3, 4), [90, 84, 80])
        )
        assert drawing["texts"] == [
            {
                "text": "66\u00b5",
                "at": [5, 6],
                "bounds": [1, 2, 3, 4],
                "advances": [90, 84, 80],
                "align": 10,
                "font": "Tahoma",
                "height": 159,
                "escapement": 900,
                "weight": 700,
            }
        ]

    def test_an_ellipse_is_kept_by_its_box(self):
        drawing = recorder.read_emf(_emf_record(42, struct.pack("<4i", 1, 2, 3, 4)))
        assert drawing["ellipses"] == [[1, 2, 3, 4]]

    def test_a_record_that_draws_something_unread_is_counted(self):
        # An arc or a polygon would otherwise leave no trace of being missed.
        # The pixel a build sets at the end of a stroke adds nothing to it.
        pixel = _emf_record(15, _emf_point(1, 2), struct.pack("<I", 0))
        arc = _emf_record(45, bytes(32))
        polygon = _emf_record(86, bytes(24))
        assert recorder.read_emf(pixel + polygon + arc + polygon)["other"] == [[45, 1], [86, 2]]

    def test_nothing_of_the_machine_is_kept(self):
        """A file's header gives the display it was written on, and LTspice
        26's description runs on into strings of the program's own."""
        header = _emf_record(1, b"\x00\x0a\x00\x00" * 20, "LTspice image".encode("utf-16-le"))
        drawing = recorder.read_emf(header + _emf_record(27, _emf_point(1, 2)))
        rendered = recorder.render_drawing(drawing)
        assert b"LTspice image" not in rendered
        assert b"2560" not in rendered
        assert json.loads(rendered) == drawing

    def test_a_record_cut_off_ends_the_reading(self):
        whole = _emf_record(27, _emf_point(1, 2)) + _emf_record(54, _emf_point(3, 4))
        assert recorder.read_emf(whole + b"\x36\x00")["strokes"] == [[1, 2, 3, 4]]

    def test_a_drawing_is_ascii_with_one_item_a_line(self):
        drawing = recorder.read_emf(
            _emf_record(27, _emf_point(1, 2))
            + _emf_record(54, _emf_point(3, 4))
            + _emf_record(54, _emf_point(5, 6))
            + _emf_font(1, 159, 0, 700, "Tahoma")
            + _emf_record(37, struct.pack("<I", 1))
            + _emf_text("66\u00b5", (5, 6), (1, 2, 3, 4), [90, 84, 80])
        )
        rendered = recorder.render_drawing(drawing)
        assert rendered.isascii()
        assert rendered.endswith(b"\n")
        lines = rendered.decode("ascii").splitlines()
        assert "[1, 2, 3, 4]," in lines
        assert "[3, 4, 5, 6]" in lines
        assert sum("66" in line for line in lines) == 1

    def test_the_same_drawing_gives_the_same_bytes(self):
        one = recorder.read_emf(_emf_record(27, _emf_point(1, 2)))
        assert recorder.render_drawing(one) == recorder.render_drawing(dict(reversed(one.items())))


@pytest.mark.parametrize(("kind", "kept"), [("save", "asc"), ("draw", "drawn")])
class TestSheetWindowCases:
    """The parts of a case that opens a sheet in the window that run anywhere."""

    def _case(self, tmp_path: Path, kind: str, extra: str = "") -> recorder.Case:
        inputs = _inputs(
            tmp_path,
            "sheets/sheet.asc",
            f'[[case]]\nid = "sheets/x"\nbehaviour = "b"\nkind = "{kind}"\n'
            'source = "sheets/sheet.asc"\n' + extra,
        )
        return load_cases(inputs).case("sheets/x")

    def test_it_opens_the_sheet_and_nothing_else(self, tmp_path: Path, kind: str, kept: str):
        case = self._case(tmp_path, kind)
        build = recorder.Build(Path("C:/apps/LTspice.exe"), "ltspice26", "current", None)
        deck, ini = tmp_path / "work" / case.work_name, tmp_path / "x.ini"
        assert recorder._command(build, case, deck, ini) == [
            str(build.exe),
            deck.as_posix(),
            "-ini",
            str(ini),
        ]

    def test_what_it_keeps_is_what_the_window_wrote(self, tmp_path: Path, kind: str, kept: str):
        case = self._case(tmp_path, kind)
        assert case.kept() == (kept,)
        assert case.work_name == "x.asc"

    def test_it_takes_no_steps(self, tmp_path: Path, kind: str, kept: str):
        with pytest.raises(RecorderError, match="only a plot case"):
            self._case(tmp_path, kind, 'steps = [{ command = "Save" }]\n')


class TestSaveCases:
    """The menu a save case's Save is read from, which runs anywhere."""

    def test_its_save_is_the_schematic_menus_own(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """Several of a build's menus have a Save; the one sent is in the menu
        that also has Label Net, which only the schematic window's does."""
        waveform = (
            b"\0\0\0\0"
            + _menu_item(0x10, "&File")
            + _menu_item(0x80, "&Save Plot Settings\tCtrl+S", 111)
            + _menu_item(0x10 | 0x80, "&Plot Settings")
            + _menu_item(0x80, "Add trace\tCtrl+A", 112)
        )
        schematic = (
            b"\0\0\0\0"
            + _menu_item(0x10, "&File")
            + _menu_item(0x80, "&Save\tCtrl+S", 221)
            + _menu_item(0x10 | 0x80, "&Edit")
            + _menu_item(0x80, "Label &Net\tF4", 222)
        )
        exe = tmp_path / "LTspice.exe"
        exe.write_bytes(b"")
        menus = [pe_menu.menu_items(waveform), pe_menu.menu_items(schematic)]
        monkeypatch.setattr(pe_menu, "menus", lambda _executable: menus)
        recorder.schematic_commands.cache_clear()
        recorder.waveform_commands.cache_clear()
        try:
            assert recorder.schematic_commands(exe) == {"Save": 221, "Label Net": 222}
            assert recorder.waveform_commands(exe) == {
                "Save Plot Settings": 111,
                "Add trace": 112,
            }
            monkeypatch.setattr(pe_menu, "menus", lambda _executable: menus[:1])
            recorder.schematic_commands.cache_clear()
            with pytest.raises(RecorderError, match="no schematic-window menu"):
                recorder.schematic_commands(exe)
        finally:
            recorder.schematic_commands.cache_clear()
            recorder.waveform_commands.cache_clear()


class TestPlotCases:
    """The parts of a plot case that run anywhere: its steps and the menu it reads."""

    def _case_file(self, tmp_path: Path, case: str) -> Path:
        return _inputs(tmp_path, "plot/rc.asc", case)

    def test_a_step_is_a_trace_or_a_command(self, tmp_path: Path):
        inputs = self._case_file(
            tmp_path,
            '[[case]]\nid = "plot/x"\nbehaviour = "b"\nkind = "plot"\nsource = "plot/rc.asc"\n'
            'steps = [{ trace = "V(out)" }, { command = "Add Plot Pane" }]\n',
        )
        (case,) = load_cases(inputs).cases
        assert case.steps == (("trace", "V(out)"), ("command", "Add Plot Pane"))
        assert recorder.DEFAULT_KEEP[case.kind] == ("plt",)

    @pytest.mark.parametrize(
        "steps", ['[{ trace = "V(out)", command = "Add Plot Pane" }]', '[{ pane = "x" }]']
    )
    def test_a_step_that_is_neither_is_refused(self, tmp_path: Path, steps: str):
        inputs = self._case_file(
            tmp_path,
            '[[case]]\nid = "plot/x"\nbehaviour = "b"\nkind = "plot"\nsource = "plot/rc.asc"\n'
            f"steps = {steps}\n",
        )
        with pytest.raises(RecorderError, match="a step is one of"):
            load_cases(inputs)

    def test_a_case_of_another_kind_with_steps_is_refused(self, tmp_path: Path):
        inputs = self._case_file(
            tmp_path,
            '[[case]]\nid = "plot/x"\nbehaviour = "b"\nsource = "plot/rc.asc"\n'
            'steps = [{ trace = "V(out)" }]\n',
        )
        with pytest.raises(RecorderError, match="only a plot case"):
            load_cases(inputs)

    def test_the_plot_settings_go_beside_the_sheet_under_its_name(self, tmp_path: Path):
        inputs = self._case_file(
            tmp_path,
            '[[case]]\nid = "plot/read_x"\nbehaviour = "b"\nkind = "plot"\n'
            'source = "plot/rc.asc"\nplot = "plot/x.plt"\n',
        )
        (inputs / "plot" / "x.plt").write_bytes(b"")
        (case,) = load_cases(inputs).cases
        assert case.copies == {"plot/rc.asc": "read_x.asc", "plot/x.plt": "read_x.plt"}

    def test_a_plot_case_with_nothing_to_read_or_make_is_refused(self, tmp_path: Path):
        inputs = self._case_file(
            tmp_path,
            '[[case]]\nid = "plot/x"\nbehaviour = "b"\nkind = "plot"\nsource = "plot/rc.asc"\n',
        )
        with pytest.raises(RecorderError, match="reads plot settings or makes them"):
            load_cases(inputs)

    def test_a_menu_label_is_the_text_a_case_names(self):
        assert pe_menu.menu_label("&Save Plot Settings\tCtrl+S") == "Save Plot Settings"
        assert pe_menu.menu_label("Save Plot Settings &As...") == "Save Plot Settings As"
        assert pe_menu.menu_label("Add &Plot Pane Below Active Pane") == (
            "Add Plot Pane Below Active Pane"
        )

    def test_a_menu_template_gives_each_item_its_command(self):
        """The classic template form both builds' waveform menus are in."""
        template = (
            b"\0\0\0\0"
            + _menu_item(0x10, "&File")
            + _menu_item(0x80, "&Save Plot Settings\tCtrl+S", 57603)
            + _menu_item(0x10 | 0x80, "&Plot Settings")
            + _menu_item(0, "Add trace\tCtrl+A", 32855)
            + _menu_item(0x80, "Save Plot Settings As...", 32914)
        )
        assert pe_menu.menu_items(template) == [
            (None, "&File"),
            (57603, "&Save Plot Settings\tCtrl+S"),
            (None, "&Plot Settings"),
            (32855, "Add trace\tCtrl+A"),
            (32914, "Save Plot Settings As..."),
        ]
