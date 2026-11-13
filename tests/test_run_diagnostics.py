"""What the simulator said about a run, on every path that answers it.

Both LTspice builds ignore a model parameter they do not know and finish the
run (``log/unknown_model_param`` in the recordings): LTspice 26 writes
``<deck>(5): Ignoring unknown model parameter.`` once per step, LTspice XVII
writes ``Error on line 5 : .model ...`` and ``* Unrecognized parameter
"bogus" -- ignored``. Both are one condition, ``unknown_model_parameter``, and
the run still counts as produced. The answer is the same from the reader, the
decoded log, the ``summary`` recipe, a run row of ``run_experiments`` and of
``jobs``, and the Python API.

Every recorded log line is a known condition or routine output, so a run's
``unclassified`` count is zero on every recording; a line no rule knows is
counted, never dropped.
"""

from __future__ import annotations

import asyncio
import inspect
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, ClassVar

import pytest
from spicelib.simulators.ltspice_simulator import LTspice

from ltspice_mcp.config import ServerConfig
from ltspice_mcp.lib import log_conditions, wsl
from ltspice_mcp.lib.encoding import read_spice_text
from ltspice_mcp.lib.log_parser import extract_log_diagnostics
from ltspice_mcp.lib.services import AnalysisSource, load_logs_sync
from ltspice_mcp.lib.simulator import detect_simulators
from ltspice_mcp.lib.simulator_build import Producer
from ltspice_mcp.state import SessionState
from ltspice_mcp.tools.analyze import AnalyzeResultsInput, handle_analyze_results
from ltspice_mcp.tools.jobs import JobsInput, handle_jobs
from tests import _ltspice_recorded as rec
from tests.conftest import FIXTURES_DIR, SyncApi, stand_in_program, terminal_experiment

_SRC = Path(__file__).resolve().parents[1] / "src" / "ltspice_mcp"
_NGSPICE_FIXTURES = (
    FIXTURES_DIR / "ngspice_ac_no_meas.log",
    *sorted((FIXTURES_DIR / "native_log_tables").glob("*.log")),
    *sorted((FIXTURES_DIR / "ngspice_runs").glob("*.log")),
)

# What each build said, as each path reports it: the first occurrence with its
# lines as written, how often, from the log, and the simulator's own word only
# where it wrote one (LTspice 26 writes none).
EXPECTED = {
    "ltspice26": {
        "diagnostics": [
            {
                "code": "unknown_model_parameter",
                "count": 3,
                "first_text": (
                    "C:\\recording\\unknown_model_param.cir(5): Ignoring unknown model "
                    "parameter. This may or may not be a problem.\n"
                    ".model DX D(Is=1e-14 Bogus=3)\n"
                    "                     ^^^^^^^"
                ),
                "channel": "log",
            }
        ]
    },
    "ltspice17": {
        "diagnostics": [
            {
                "code": "unknown_model_parameter",
                "count": 1,
                "first_text": (
                    "Error on line 5 : .model dx d(is=1e-14 bogus=3)\n"
                    '\t* Unrecognized parameter "bogus" -- ignored'
                ),
                "channel": "log",
                "severity": "Error",
            }
        ]
    },
}

# What the summary recipe's ``errors`` held for the same logs before runs
# carried diagnostics; it holds the same.
SUMMARY_ERRORS = {
    "ltspice26": [
        "C:\\recording\\unknown_model_param.cir(5): Ignoring unknown model parameter. "
        "This may or may not be a problem.\n.model DX D(Is=1e-14 Bogus=3)\n"
        "^^^^^^^ (repeated 3 times)"
    ],
    "ltspice17": ["Error on line 5 : .model dx d(is=1e-14 bogus=3)"],
}

# The program each build runs as: XVII is known by its stock name, 26 by the
# banner its log opens with.
PROGRAM = {"ltspice26": "LTspice.exe", "ltspice17": "XVIIx64.exe"}
REPORTED = {
    "ltspice26": "LTspice 26.1.1 for Windows",
    "ltspice17": "Linear Technology Corporation LTspice XVII",
}


def _recorded_simulator(work_dir: Path, build: str) -> type:
    recorded = rec.recorded(build, "log")

    class _Recorded(LTspice):
        spice_exe: ClassVar[list[str]] = []

        @classmethod
        def run(
            cls,
            netlist_file,
            cmd_line_switches=None,
            timeout=None,
            stdout=None,
            stderr=None,
            cwd=None,
            exe_log=False,
        ) -> int:
            netlist = Path(netlist_file)
            shutil.copy(recorded / "unknown_model_param.log", netlist.with_suffix(".log"))
            shutil.copy(recorded / "unknown_model_param.raw", netlist.with_suffix(".raw"))
            return 0

        @classmethod
        def valid_switch(cls, switch: str, path: str = "") -> list:
            return []

    program = stand_in_program(work_dir / "sim" / PROGRAM[build], build.encode())
    return type("LTspice", (_Recorded,), {"spice_exe": [str(program)]})


def _session(work_dir: Path, simulator: type) -> SessionState:
    config = ServerConfig(working_dir=work_dir, allowed_paths=[work_dir], simulator="ltspice")
    return SessionState.create(config, available={"ltspice": simulator})


def _row_diagnostics(row: dict[str, Any]) -> dict[str, Any]:
    """The diagnostics fields of a run row or a summary, as one value."""
    found: dict[str, Any] = {}
    if "diagnostics" in row:
        found["diagnostics"] = row["diagnostics"]
    if "diagnostics_coverage" in row:
        found["coverage"] = row["diagnostics_coverage"]
    return found


async def _summary(state: SessionState, job_id: str) -> dict[str, Any]:
    result = await handle_analyze_results(
        AnalyzeResultsInput.model_validate(
            {
                "sources": [{"job_id": job_id, "label": "run"}],
                "recipes": [{"key": "s", "metric": "summary"}],
                "include": {"per_run": {"limit": 10}, "fields": ["value"]},
            }
        ),
        state,
    )
    data = result.structured_content
    assert data is not None, result.content
    (row,) = data["results"]["s"]["per_run"]["items"]
    return row["value"]


@pytest.mark.asyncio
@pytest.mark.parametrize("build", ["ltspice26", "ltspice17"])
async def test_the_recorded_unknown_model_parameter_on_every_path(
    build: str, work_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    # WSL interop is off whatever the host, as for every replayed LTspice.
    monkeypatch.setattr(wsl, "is_wsl", lambda: False)
    expected = EXPECTED[build]
    recorded_log = rec.recorded(build, "log") / "unknown_model_param.log"
    producer = Producer("ltspice", REPORTED[build], PROGRAM[build])

    # The reader, over the decode the parser worker returns.
    conditions = extract_log_diagnostics(recorded_log)["conditions"]
    assert log_conditions.run_diagnostics(conditions, producer) == expected

    simulator = _recorded_simulator(work_dir, build)
    state = _session(work_dir, simulator)

    # The decoded log, read through the parser process.
    logs = await asyncio.to_thread(
        load_logs_sync,
        AnalysisSource(
            raw=None,
            log=recorded_log,
            netlist=None,
            dialect="ltspice",
            identity=None,
            trusted_job_artifact=True,
        ),
        state,
    )
    assert log_conditions.decoded_run_diagnostics(logs, producer) == expected

    deck = work_dir / "unknown_model_param.cir"
    shutil.copy(rec.INPUTS / "log" / "unknown_model_param.cir", deck)
    reply = await terminal_experiment(
        state,
        {
            "request_id": f"unknown-param-{build}",
            "circuits": [{"path": str(deck), "id": "dut"}],
            "execution": {"wait_s": 30},
            "lint": "off",
        },
    )
    # Both builds finish this run: the parameter is ignored, not refused.
    assert reply["status"] == "completed", reply
    (row,) = reply["runs"]["items"]
    assert row["status"] == "produced"
    assert row["simulator_version"] == REPORTED[build]
    assert _row_diagnostics(row) == expected

    runs = await handle_jobs(
        JobsInput.model_validate({"action": "runs", "job_id": reply["job_id"]}), state
    )
    assert runs.structured_content is not None
    (jobs_row,) = runs.structured_content["items"]
    assert _row_diagnostics(jobs_row) == expected

    summary = await _summary(state, reply["job_id"])
    assert _row_diagnostics(summary) == expected
    assert summary["errors"] == SUMMARY_ERRORS[build]
    assert "warnings" not in summary or not any(
        "unknown model parameter" in warning.lower() for warning in summary["warnings"]
    )

    api = SyncApi(state)
    api_runs = await asyncio.to_thread(api.jobs, action="runs", job_id=reply["job_id"])
    (api_row,) = api_runs["items"]
    assert _row_diagnostics(api_row) == expected
    api_summary = await asyncio.to_thread(
        api.analyze_results,
        sources=[{"job_id": reply["job_id"], "label": "run"}],
        recipes=[{"key": "s", "metric": "summary"}],
        include={"per_run": {"limit": 10}, "fields": ["value"]},
    )
    (api_summary_row,) = api_summary["results"]["s"]["per_run"]["items"]
    assert _row_diagnostics(api_summary_row["value"]) == expected


def _recorded_logs(build: str) -> list[Path]:
    return sorted((rec.FIXTURES / build).rglob("*.log"))


def _unclassified(path: Path, rules: str, console: Path | None = None) -> list[dict]:
    channels: dict[str, str] = {"log": read_spice_text(path)}
    if console is not None and console.exists():
        channels["console"] = read_spice_text(console)
    return [
        entry
        for entry in log_conditions.classify(channels, rules)  # type: ignore[arg-type]
        if entry["code"] == log_conditions.UNCLASSIFIED
    ]


@pytest.mark.parametrize("build", ["ltspice26", "ltspice17"])
def test_every_recorded_ltspice_line_is_a_known_condition_or_routine(build: str):
    logs = _recorded_logs(build)
    assert len(logs) > 150, "the recordings were not found"
    leftover = {
        str(path.relative_to(rec.FIXTURES)): found
        for path in logs
        if (found := _unclassified(path, build))
    }
    assert leftover == {}


def test_every_ngspice_fixture_line_is_a_known_condition_or_routine():
    leftover = {
        path.name: found
        for path in _NGSPICE_FIXTURES
        if (found := _unclassified(path, "ngspice", path.with_suffix(".exe.log")))
    }
    assert leftover == {}


@pytest.mark.parametrize("build", ["ltspice26", "ltspice17"])
def test_a_line_no_rule_knows_is_counted_once_as_written(build: str):
    text = read_spice_text(rec.recorded(build, "log") / "unknown_model_param.log")
    added = "Something the simulator never said before: 42"
    lines = text.splitlines()
    lines.insert(2, added)
    found = log_conditions.classify({"log": "\n".join(lines)}, build)
    unknown = [entry for entry in found if entry["code"] == log_conditions.UNCLASSIFIED]
    assert unknown == [{"code": "unclassified", "count": 1, "first_text": added, "channel": "log"}]
    # The known condition is still read beside it.
    assert [entry["code"] for entry in found if entry["code"] != "unclassified"] == [
        "unknown_model_parameter"
    ]


def test_log_and_console_lines_keep_their_channel():
    directory = FIXTURES_DIR / "ngspice_runs"
    found = log_conditions.classify(
        {
            "log": read_spice_text(directory / "unknown_model_param.log"),
            "console": "Error: something only the console said\n",
        },
        "ngspice",
    )
    by_code = {entry["code"]: entry for entry in found}
    assert by_code["unknown_model_parameter"]["channel"] == "log"
    assert by_code["simulator_error"] == {
        "code": "simulator_error",
        "count": 1,
        "first_text": "Error: something only the console said",
        "channel": "console",
        "severity": "Error",
    }


def test_the_producer_chooses_the_rules_and_says_when_they_are_not_its_own():
    log = rec.recorded("ltspice26", "log") / "unknown_model_param.log"
    conditions = extract_log_diagnostics(log)["conditions"]
    # A family with no build the server has recordings of is read with every
    # recorded build's wording, and says so.
    unrecorded = log_conditions.run_diagnostics(
        conditions, Producer("ltspice", "LTspice 24.0.12 for Windows", "LTspice.exe")
    )
    assert unrecorded["coverage"] == "build_unrecorded"
    assert [entry["code"] for entry in unrecorded["diagnostics"]] == ["unknown_model_parameter"]
    # No known producer: classified by wording alone, and said.
    unknown = log_conditions.run_diagnostics(conditions, Producer(None, None, None))
    assert unknown["coverage"] == "producer_unknown"
    assert [entry["code"] for entry in unknown["diagnostics"]] == ["unknown_model_parameter"]
    # A simulator with no recordings has every line counted as unclassified.
    xyce = log_conditions.run_diagnostics(conditions, Producer("xyce", None, None))
    assert xyce["coverage"] == "unverified"
    ((entry),) = xyce["diagnostics"]
    assert entry["code"] == "unclassified"
    assert entry["count"] == sum(1 for line in read_spice_text(log).splitlines() if line.strip())
    # A log that was not read says so.
    assert log_conditions.run_diagnostics(None, Producer("ltspice", None, None)) == {
        "coverage": "log_unread"
    }


def test_a_run_with_nothing_to_report_carries_no_field():
    clean = rec.recorded("ltspice26", "raw") / "op.log"
    conditions = extract_log_diagnostics(clean)["conditions"]
    producer = Producer("ltspice", "LTspice 26.1.1 for Windows", "LTspice.exe")
    assert log_conditions.run_diagnostics(conditions, producer) == {}


def test_a_record_written_before_classification_says_its_log_was_not_classified(
    work_dir: Path,
):
    from ltspice_mcp.lib.experiment_types import ExperimentCase
    from ltspice_mcp.tools.receipts import runs_page

    def case(**fields: Any) -> ExperimentCase:
        return ExperimentCase(
            case_id="c",
            run_index=0,
            circuit="dut",
            circuit_path=work_dir / "dut.cir",
            staged_deck=work_dir / "dut.cir",
            deck_sha256="0" * 64,
            **fields,
        )

    old = case(status="produced", submitted_at=_now())
    unread = case(status="failed", submitted_at=_now(), log_diagnostics={"coverage": "log_unread"})
    clean = case(status="produced", submitted_at=_now(), log_diagnostics={})
    queued = case()
    rows = runs_page([old, unread, clean, queued])["items"]
    assert [_row_diagnostics(row) for row in rows] == [
        {"coverage": "not_classified"},
        {"coverage": "log_unread"},
        {},
        {},
    ]


def _now():
    from ltspice_mcp.lib import now

    return now()


def test_the_design_sentence_holds_for_every_reply_it_names():
    """``docs/DESIGN.md`` says the simulator's diagnostics are on the replies
    of ``run_experiments`` and ``jobs`` and on the ``summary`` recipe; each of
    those declares them."""
    from ltspice_mcp.tools.analyze import _ATTRIBUTED_VALUE_SCHEMA
    from ltspice_mcp.tools.receipts import RUN_RECORD_SCHEMA

    summary_schema = _ATTRIBUTED_VALUE_SCHEMA["properties"]["value"]

    design = (_SRC.parents[1] / "docs" / "DESIGN.md").read_text(encoding="utf-8")
    section = design.split("## Diagnostics", 1)[1].split("\n## ", 1)[0]
    for name in ("run_experiments", "jobs", "summary", "diagnostics", "unclassified"):
        assert name in section, name
    for schema in (RUN_RECORD_SCHEMA, summary_schema):
        assert {"diagnostics", "diagnostics_coverage"} <= set(schema["properties"])
        item = schema["properties"]["diagnostics"]["items"]
        assert set(item["required"]) == {"code", "count", "first_text", "channel"}
        assert set(schema["properties"]["diagnostics_coverage"]["enum"]) == set(
            log_conditions.COVERAGE
        )


def test_the_producer_is_required_and_never_assumed():
    """The readers take the producer with no default, and no caller writes one
    that stands for LTspice."""
    for reader in (log_conditions.run_diagnostics, log_conditions.decoded_run_diagnostics):
        parameter = inspect.signature(reader).parameters["producer"]
        assert parameter.default is inspect.Parameter.empty, reader.__name__
    assumed = re.compile(r"Producer\(\s*(?:family\s*=\s*)?[\"']ltspice")
    offenders = [
        str(path.relative_to(_SRC))
        for path in _SRC.rglob("*.py")
        if assumed.search(path.read_text(encoding="utf-8"))
    ]
    assert offenders == []


def _ngspice_version() -> str | None:
    program = shutil.which("ngspice")
    if program is None:
        return None
    done = subprocess.run([program, "-v"], capture_output=True, text=True, timeout=60, check=False)
    return next((line.strip("* ") for line in done.stdout.splitlines() if "ngspice-" in line), "")


@pytest.mark.asyncio
async def test_ngspice_names_the_same_condition_live(work_dir: Path):
    """ngspice ignores the parameter too and finishes. The recording's deck
    has a ``.step``, which ngspice does not run, so the live deck leaves it out."""
    version = _ngspice_version()
    if version is None:
        pytest.skip("ngspice is not on PATH")
    print(f"ngspice that ran: {version}")
    deck = work_dir / "unknown_model_param.cir"
    shutil.copy(FIXTURES_DIR / "ngspice_runs" / "unknown_model_param.cir", deck)
    config = ServerConfig(working_dir=work_dir, allowed_paths=[work_dir], simulator="ngspice")
    state = SessionState.create(config, available=detect_simulators(config))
    reply = await terminal_experiment(
        state,
        {
            "request_id": "unknown-param-ngspice",
            "circuits": [{"path": str(deck), "id": "dut"}],
            "execution": {"wait_s": 60},
            "lint": "off",
        },
    )
    assert reply["status"] == "completed", reply
    (row,) = reply["runs"]["items"]
    assert row["status"] == "produced"
    assert "diagnostics_coverage" not in row, row
    by_code = {entry["code"]: entry for entry in row["diagnostics"]}
    entry = by_code.pop("unknown_model_parameter")
    assert entry["severity"] == "Warning"
    assert entry["channel"] == "log"
    assert entry["first_text"].startswith("Warning: Model issue on line 5 :")
    assert "unrecognized parameter (bogus) - ignored" in entry["first_text"]
    # A build the fixtures do not record may say more than they show; its
    # lines are counted, not dropped, and printed here so a new wording is
    # seen and recorded (the fixture tests above hold every recorded line).
    if "unclassified" in by_code:
        print(f"unclassified in {version}: {by_code['unclassified']}")


def test_one_code_on_both_channels_is_one_entry_per_channel():
    found = log_conditions.classify(
        {"log": "Error: said in the log\n", "console": "Error: said on the console\n"},
        "ngspice",
    )
    assert [(entry["code"], entry["channel"], entry["count"]) for entry in found] == [
        ("simulator_error", "log", 1),
        ("simulator_error", "console", 1),
    ]


def test_a_case_that_has_not_ended_or_never_ran_says_nothing(work_dir: Path):
    """A retried case keeps no earlier attempt's answer on its row while it runs,
    and a case whose submission failed ran no simulator to say anything."""
    from ltspice_mcp.lib.experiment_types import ExperimentCase
    from ltspice_mcp.tools.receipts import runs_page

    stale = {"diagnostics": EXPECTED["ltspice26"]["diagnostics"]}

    def case(**fields: Any) -> ExperimentCase:
        return ExperimentCase(
            case_id="c",
            run_index=0,
            circuit="dut",
            circuit_path=work_dir / "dut.cir",
            staged_deck=work_dir / "dut.cir",
            deck_sha256="0" * 64,
            **fields,
        )

    rows = runs_page(
        [
            case(status="running", submitted_at=_now(), log_diagnostics=stale),
            case(status="queued", log_diagnostics=stale),
            case(status="failed", submitted_at=_now(), failure_code="submission_failed"),
        ]
    )["items"]
    assert [_row_diagnostics(row) for row in rows] == [{}, {}, {}]


@pytest.mark.asyncio
@pytest.mark.parametrize("dialect", [None, "ltspice"])
async def test_a_bare_paths_summary_reads_the_producer_its_files_name(
    dialect: str | None, work_dir: Path, state_no_sim: SessionState
):
    """A raw read by path has no case: the summary takes the simulator from the
    caller or the raw's header, and the build from the log's banner, so a
    recorded LTspice 26 run reads as its run row would."""
    for suffix in (".raw", ".log"):
        shutil.copy(
            rec.recorded("ltspice26", "log") / f"unknown_model_param{suffix}",
            work_dir / f"unknown_model_param{suffix}",
        )
    source: dict[str, Any] = {"path": str(work_dir / "unknown_model_param.raw"), "label": "p"}
    if dialect is not None:
        source["dialect"] = dialect
    result = await handle_analyze_results(
        AnalyzeResultsInput.model_validate(
            {
                "sources": [source],
                "recipes": [{"key": "s", "metric": "summary"}],
                "include": {"per_run": {"limit": 10}, "fields": ["value"]},
            }
        ),
        state_no_sim,
    )
    data = result.structured_content
    assert data is not None, result.content
    (row,) = data["results"]["s"]["per_run"]["items"]
    assert _row_diagnostics(row["value"]) == EXPECTED["ltspice26"]


def test_a_result_set_stored_before_runs_named_their_producer_reads_as_a_bare_path():
    from ltspice_mcp.tools.analyze import _stored_producer

    assert _stored_producer(None) is None
    assert _stored_producer({"family": "ltspice"}) is None
    assert _stored_producer({"family": 3, "build": None, "program": None}) is None
    assert _stored_producer({"family": "ngspice", "build": None, "program": None}) == Producer(
        "ngspice", None, None
    )


def test_only_xvii_of_the_builds_before_24_reads_with_xviis_rules():
    assert log_conditions.rules_for(Producer("ltspice", None, "C:/LTC/XVIIx64.exe")) == (
        "ltspice17",
        None,
    )
    assert log_conditions.rules_for(Producer("ltspice", None, "C:/LTC/scad3.exe")) == (
        "ltspice",
        "build_unrecorded",
    )
    # A folder named for XVII does not make 17.1 the build the rules were recorded from.
    assert log_conditions.rules_for(
        Producer("ltspice", "LTspice 17.1.8 for Windows", "C:/LTspiceXVII/LTspice.exe")
    ) == ("ltspice", "build_unrecorded")
    assert log_conditions.rules_for(
        Producer("ltspice", "Linear Technology Corporation LTspice XVII", "LTspice.exe")
    ) == ("ltspice17", None)
    # The same reading of a build's name as every other answer about XVII.
    assert log_conditions.rules_for(Producer("ltspice", "LTSPICE  XVII", None)) == (
        "ltspice17",
        None,
    )


@pytest.mark.asyncio
async def test_a_produced_runs_log_is_read_once_it_lands_and_without_its_raw(
    work_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    """The read happens after the run's time limit stopped counting, not inside
    the completion the limit waits on, and copies only the log and console: a
    raw past the parser's input limit would refuse a read that captured it."""
    from ltspice_mcp.lib import services

    monkeypatch.setattr(wsl, "is_wsl", lambda: False)
    reads: list[AnalysisSource] = []
    real = services.load_logs_sync

    def recording(source: AnalysisSource, state: SessionState, **kwargs: Any):
        reads.append(source)
        return real(source, state, **kwargs)

    monkeypatch.setattr(services, "load_logs_sync", recording)
    state = _session(work_dir, _recorded_simulator(work_dir, "ltspice26"))
    deck = work_dir / "unknown_model_param.cir"
    shutil.copy(rec.INPUTS / "log" / "unknown_model_param.cir", deck)
    reply = await terminal_experiment(
        state,
        {
            "request_id": "read-once",
            "circuits": [{"path": str(deck), "id": "dut"}],
            "execution": {"wait_s": 30},
            "lint": "off",
        },
    )
    (row,) = reply["runs"]["items"]
    assert row["status"] == "produced"
    assert _row_diagnostics(row) == EXPECTED["ltspice26"]
    (read,) = [source for source in reads if source.log == Path(row["log"])]
    assert read.raw is None


@pytest.mark.asyncio
async def test_the_replaying_stand_in_leaves_the_log_read_to_the_runner(
    config: ServerConfig, work_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    """The stand-in most run tests replay recordings through returns what the
    real completion returns for a run that produced, no log facts, so the read
    those tests cover is the runner's own: once, without the raw."""
    from ltspice_mcp.lib import services
    from tests.conftest import FakeSim, replaying_simulator

    reads: list[AnalysisSource] = []
    real = services.load_logs_sync

    def recording(source: AnalysisSource, state: SessionState, **kwargs: Any):
        reads.append(source)
        return real(source, state, **kwargs)

    monkeypatch.setattr(services, "load_logs_sync", recording)
    recorded = rec.recorded("ltspice26", "log")
    replaying_simulator(
        monkeypatch, recorded / "unknown_model_param.raw", recorded / "unknown_model_param.log"
    )
    state = SessionState.create(config, available={"ltspice": type("LTspice", (FakeSim,), {})})
    deck = work_dir / "unknown_model_param.cir"
    shutil.copy(rec.INPUTS / "log" / "unknown_model_param.cir", deck)
    reply = await terminal_experiment(
        state,
        {
            "request_id": "replayed",
            "circuits": [{"path": str(deck), "id": "dut"}],
            "execution": {"wait_s": 30},
            "lint": "off",
        },
    )
    (row,) = reply["runs"]["items"]
    assert _row_diagnostics(row) == EXPECTED["ltspice26"]
    (read,) = [source for source in reads if source.log == Path(row["log"])]
    assert read.raw is None
