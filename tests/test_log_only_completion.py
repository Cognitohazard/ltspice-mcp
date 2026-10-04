"""Only recognized, captured output can establish a log-only produced case."""

import pytest
from spicelib.simulators.ngspice_simulator import NGspiceSimulator

from ltspice_mcp.lib.runner_base import DeckRequirements, collect_run_outcome
from tests.conftest import FIXTURES_DIR
from tests.test_summary_log_facts import captured_log_facts


@pytest.mark.parametrize(
    ("name", "analysis"),
    [
        ("tf_table", ".tf"),
        ("pz_table", ".pz"),
        ("sens_dc_table", ".sens"),
        ("sens_ac_table", ".sens"),
        ("disto_harm_table", ".disto"),
        ("disto_two_table", ".disto"),
    ],
)
def test_recorded_native_results_can_produce_without_raw(tmp_path, name, analysis):
    source = FIXTURES_DIR / "native_log_tables" / f"{name}.log"
    facts = captured_log_facts(tmp_path, source)
    outcome = collect_run_outcome(
        "absent.raw",
        str(source),
        DeckRequirements((analysis,), has_control=True),
        exit_code=0,
        logs=facts,
        simulator=NGspiceSimulator,
    )

    assert outcome.error is None
    assert outcome.raw_file == ""
    assert outcome.log_file == str(source)
    assert outcome.observations[0]["code"] == "native_log_output"
    assert outcome.observations[0]["evidence"]["analysis_extent"] == "unknown"


@pytest.mark.parametrize("text", ["", "Circuit: quiet\n", "ngspice-42 done\n"])
@pytest.mark.parametrize("analyses", [(), (".tf",)])
def test_quiet_log_or_footer_is_not_a_result(tmp_path, text, analyses):
    outcome = collect_run_outcome(
        "absent.raw",
        "run.log",
        DeckRequirements(analyses, has_control=True),
        exit_code=0,
        logs=captured_log_facts(tmp_path, text=text),
        simulator=NGspiceSimulator,
    )
    assert outcome.error is not None


@pytest.mark.parametrize("analyses", [(), (".pz",), (".tf", ".tran")])
def test_printed_family_must_match_every_requested_analysis(tmp_path, analyses):
    facts = captured_log_facts(tmp_path, FIXTURES_DIR / "native_log_tables/tf_table.log")
    outcome = collect_run_outcome(
        "absent.raw",
        "run.log",
        DeckRequirements(analyses, has_control=True),
        exit_code=0,
        logs=facts,
        simulator=NGspiceSimulator,
    )
    assert outcome.error is not None


@pytest.mark.parametrize("exit_code", [None, -9, 1])
def test_printed_output_requires_confirmed_successful_exit(tmp_path, exit_code):
    facts = captured_log_facts(tmp_path, FIXTURES_DIR / "native_log_tables/tf_table.log")
    outcome = collect_run_outcome(
        "absent.raw",
        "run.log",
        DeckRequirements((".tf",), has_control=True),
        exit_code=exit_code,
        logs=facts,
        simulator=NGspiceSimulator,
    )
    assert outcome.error is not None


@pytest.mark.parametrize("corruption", ["open", "malformed", "diagnostic"])
def test_native_parse_or_solve_failure_cannot_produce(tmp_path, corruption):
    text = (FIXTURES_DIR / "native_log_tables/tf_table.log").read_text(encoding="utf-8")
    if corruption == "open":
        text = text.replace("ngspice-42 done", "")
    elif corruption == "malformed":
        text = text.replace("6.666666666666666e-01", "not-a-number")
    else:
        text = "Fatal Error: convergence failed\n" + text
    outcome = collect_run_outcome(
        "absent.raw",
        "run.log",
        DeckRequirements((".tf",), has_control=True),
        exit_code=0,
        logs=captured_log_facts(tmp_path, text=text),
        simulator=NGspiceSimulator,
    )
    assert outcome.error is not None


def test_native_prints_do_not_hide_an_unrecovered_stepping_failure(tmp_path):
    text = (FIXTURES_DIR / "native_log_tables/tf_table.log").read_text(encoding="utf-8")
    facts = captured_log_facts(
        tmp_path, text=text + "Gmin stepping failed to find operating point.\n"
    )
    assert facts.value("diagnostics")["errors"]
    outcome = collect_run_outcome(
        "absent.raw",
        "run.log",
        DeckRequirements((".tf",), has_control=True),
        exit_code=0,
        logs=facts,
        simulator=NGspiceSimulator,
    )
    assert outcome.error is not None
