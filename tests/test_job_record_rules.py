"""The rules a job's records and run files are read by, each kept in one place.

A run token, a digest, a run deck's suffix, a raw's extension and whether a
recorded job is an LTspice run: each is one rule its readers share, so a change
to it reaches all of them.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest
from spicelib.simulators.ltspice_simulator import LTspice
from spicelib.simulators.ngspice_simulator import NGspiceSimulator
from spicelib.simulators.qspice_simulator import Qspice
from spicelib.simulators.xyce_simulator import XyceSimulator

from ltspice_mcp.lib import RUN_DECK_SUFFIXES, experiment_store
from ltspice_mcp.lib.proc_kill import _names_run_deck
from ltspice_mcp.lib.simulator import raw_extension_for
from ltspice_mcp.lib.store import is_sha256, is_store_name
from tests.test_jobs_tool import _circuit, _experiment

_SRC = Path(__file__).resolve().parents[1] / "src" / "ltspice_mcp"


@pytest.mark.parametrize("suffix", sorted(RUN_DECK_SUFFIXES))
def test_every_run_deck_suffix_names_a_case_s_run_deck(suffix: str):
    token = "exp_job_case_3"
    assert _names_run_deck(token, f"/runs/{token}{suffix}")
    assert _names_run_deck(token, f"C:\\runs\\{token}{suffix.upper()}")
    # A longer token it prefixes, and a file that only mentions it, are not.
    assert not _names_run_deck(token, f"/runs/{token}0{suffix}")
    assert not _names_run_deck(token, f"/runs/{token}.py")


@pytest.mark.parametrize("simulator", [LTspice, NGspiceSimulator, Qspice, XyceSimulator])
def test_a_recorded_job_s_raw_has_its_simulator_s_extension(simulator: type, work_dir: Path):
    """Where a case's raw is found from its run token, and where the runner
    looks to remove one, both read the simulator class's ``raw_extension``."""
    job = _experiment(work_dir, _circuit(work_dir), status="running")
    job.simulator = simulator.__name__
    job.output_folder = work_dir / "runs"
    case = job.cases[0]
    case.run_token = "exp_job_case_0"

    raw = experiment_store.case_raw_path(job, case)

    assert raw == job.output_folder / f"exp_job_case_0{simulator.raw_extension}"
    assert raw_extension_for(simulator) == raw_extension_for(simulator.__name__)


def test_identifiers_and_digests_are_checked_by_the_store_s_rules() -> None:
    assert is_store_name("exp_a1b2_case_0") and is_store_name("A.b-c_1")
    assert (
        not is_store_name("") and not is_store_name("../escape") and not is_store_name(".hidden")
    )
    assert not is_store_name("a" * 129) and not is_store_name(7)
    assert is_sha256("0" * 64) and not is_sha256("0" * 63) and not is_sha256("A" * 64)
    assert not is_sha256(("0" * 64) + "\n") and not is_sha256(None)


#: The identifier and digest patterns, as a regex literal spells them.
_RULE_LITERALS = {
    "store name": re.compile(r"\[A-Za-z0-9\]\[A-Za-z0-9\._-\]\{0,127\}"),
    "sha256 digest": re.compile(r"\[0-9a-f\]\{64\}"),
}
#: Parser-side copies of the digest pattern, outside the job store; each checks
#: a parser cache key or a capture digest, and none reads a job record.
_PARSER_DIGEST_COPIES = frozenset(
    {
        "lib/decoded_log.py",
        "lib/log_decode.py",
        "lib/parser_protocol.py",
        "lib/parser_service.py",
        "lib/raw_decode.py",
        "lib/result_cache.py",
    }
)


def test_the_store_alone_spells_the_identifier_and_digest_patterns() -> None:
    found = sorted(
        f"{path.relative_to(_SRC).as_posix()}:{node.lineno} ({rule})"
        for path in _SRC.rglob("*.py")
        if path.relative_to(_SRC).as_posix() != "lib/store.py"
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
        for rule, pattern in _RULE_LITERALS.items()
        if pattern.search(node.value)
        and not (
            rule == "sha256 digest" and path.relative_to(_SRC).as_posix() in _PARSER_DIGEST_COPIES
        )
    )
    assert not found, f"patterns spelled outside lib/store.py: {found}"


def test_no_module_reads_a_recorded_simulator_name_by_substring() -> None:
    """``'ltspice' in job.simulator.lower()`` and its kind: a recorded class
    name is read through ``lib/simulator.py`` (``dialect_for_simulator_name``,
    ``raw_extension_for``)."""
    found = sorted(
        f"{path.relative_to(_SRC).as_posix()}:{node.lineno}"
        for path in _SRC.rglob("*.py")
        if path.relative_to(_SRC).as_posix() != "lib/simulator.py"
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "lower"
        and isinstance(node.func.value, ast.Attribute)
        and node.func.value.attr == "simulator"
    )
    assert not found, f"recorded simulator names read by substring: {found}"
