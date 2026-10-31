"""The capability table is held to the code it describes and to its evidence.

``ltspice_mcp.lib.capabilities`` declares, per simulator family, what the
server supports with evidence, what is a declared gap, and what is
implemented but unverified. The tests here keep that table honest:

- every baseline capability has an entry for every family, so a family that
  lacks one carries a reason instead of a silence;
- every piece of evidence resolves: a recorded LTspice behaviour is a key of
  the inventory, a fixture is a file under ``tests/fixtures``, a test is a
  module in ``tests/``;
- the table agrees with the module that implements each capability, through
  the declaration that module carries (``lint_rules.LintRule.family``,
  ``spice_validator.VALIDATED_SIMULATORS``, ``hierarchy.SUPPORTED_FAMILIES``,
  ``deck_prep.EXPORT_READERS``, ``experiment_inputs.CONTROLLED_STARTUP_HOSTS``,
  ``raw_header``'s dialects): a family a module declares is not a gap here,
  and one it does not declare is;
- the refusals that used to compare a family's name read the table.
"""

from __future__ import annotations

import json
import tomllib
from pathlib import Path

import pytest

from ltspice_mcp.config import ServerConfig
from ltspice_mcp.lib import (
    capabilities as cap,
)
from ltspice_mcp.lib import (
    deck_prep,
    experiment_inputs,
    hierarchy,
    lint_rules,
    raw_header,
    spice_validator,
)
from ltspice_mcp.lib import simulator as simulator_mod
from ltspice_mcp.lib.families import DISPLAY, FAMILIES, HOSTS
from ltspice_mcp.lib.simulator import SIMULATORS, asc_export_refusal, family_refusal
from ltspice_mcp.resources import handle_read_resource
from ltspice_mcp.state import SessionState

ROOT = Path(__file__).resolve().parent.parent
INVENTORY = ROOT / "tests" / "fixtures" / "ltspice_recorded" / "inputs" / "cases.toml"
FIXTURES = ROOT / "tests" / "fixtures"

BASELINE = [c.name for c in cap.CAPABILITIES if c.kind == "baseline"]
REFINEMENTS = [c.name for c in cap.CAPABILITIES if c.kind == "refinement"]
EVIDENCE = sorted({ref for e in cap.ENTRIES for ref in e.evidence})


def _host(monkeypatch: pytest.MonkeyPatch, platform_key: str) -> None:
    monkeypatch.setattr(simulator_mod, "_platform_key", lambda: platform_key)


# ---------------------------------------------------------------------------
# Shape
# ---------------------------------------------------------------------------


class TestShape:
    @pytest.mark.parametrize("name", BASELINE)
    def test_every_baseline_capability_has_an_entry_for_every_family(self, name: str):
        missing = [family for family in FAMILIES if cap.entry(family, name) is None]
        assert missing == [], f"{name} is baseline, so each of these needs an entry or a gap"

    def test_every_capability_has_at_least_one_entry(self):
        unused = [
            c.name
            for c in cap.CAPABILITIES
            if not any(e.capability == c.name for e in cap.ENTRIES)
        ]
        assert unused == []

    def test_a_supported_entry_needs_evidence(self):
        with pytest.raises(ValueError, match="needs evidence"):
            cap.Entry("ltspice", "run", "supported")

    @pytest.mark.parametrize("status", ["gap", "unverified"])
    def test_a_gap_or_unverified_entry_needs_a_reason(self, status: str):
        with pytest.raises(ValueError, match="needs a reason"):
            cap.Entry("xyce", "run", status)  # type: ignore[arg-type]

    def test_an_evidence_reference_names_its_kind(self):
        with pytest.raises(ValueError, match="bad evidence reference"):
            cap.Entry("ltspice", "run", "supported", ("raw-transient",))

    def test_hosts_are_the_known_ones(self):
        with pytest.raises(ValueError, match="unknown host"):
            cap.Entry(
                "ltspice",
                "run",
                "supported",
                ("recorded:raw-transient",),
                hosts=frozenset({"beos"}),
            )
        for found in cap.ENTRIES:
            assert found.hosts is None or found.hosts <= set(HOSTS)

    def test_entries_of_a_family_follow_the_capabilities_order(self):
        order = [c.name for c in cap.CAPABILITIES]
        for family in FAMILIES:
            names = [e.capability for e in cap.entries_of(family)]
            assert names == sorted(names, key=order.index)


# ---------------------------------------------------------------------------
# Evidence resolves
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def behaviour_keys() -> set[str]:
    data = tomllib.loads(INVENTORY.read_text(encoding="utf-8"))
    return set(data["behaviour"])


@pytest.mark.parametrize("ref", EVIDENCE)
def test_every_evidence_reference_resolves(ref: str, behaviour_keys: set[str]):
    kind, _, target = ref.partition(":")
    if kind == "recorded":
        assert target in behaviour_keys, f"{ref}: not a behaviour of {INVENTORY.name}"
    elif kind == "fixture":
        assert (FIXTURES / target).is_file(), f"{ref}: no such fixture"
    elif kind == "test":
        assert (ROOT / target).is_file(), f"{ref}: no such test module"
    else:  # pragma: no cover - the Entry constructor refuses other kinds
        pytest.fail(f"{ref}: unknown evidence kind")


# ---------------------------------------------------------------------------
# Bound to the code
# ---------------------------------------------------------------------------


def _status(family: str, name: str) -> str:
    found = cap.entry(family, name)
    assert found is not None, f"{family}/{name} has no entry"
    return found.status


class TestBoundToTheCode:
    @pytest.mark.parametrize("family", FAMILIES)
    def test_every_family_the_server_can_launch_is_not_a_run_gap(self, family: str):
        assert family in SIMULATORS
        assert _status(family, "run") != "gap"

    @pytest.mark.parametrize("family", FAMILIES)
    def test_reads_asc_export_agrees_with_deck_prep(self, family: str):
        implemented = family in deck_prep.EXPORT_READERS
        assert (_status(family, "reads_asc_export") != "gap") == implemented

    @pytest.mark.parametrize("family", FAMILIES)
    def test_lint_rules_agrees_with_the_rules_declared_for_the_family(self, family: str):
        implemented = any(rule.family == family for rule in lint_rules.RULES)
        assert (_status(family, "lint_rules") != "gap") == implemented

    @pytest.mark.parametrize("family", FAMILIES)
    def test_validator_rules_agrees_with_the_validator(self, family: str):
        implemented = DISPLAY[family] in spice_validator.VALIDATED_SIMULATORS
        assert (_status(family, "validator_rules") != "gap") == implemented

    @pytest.mark.parametrize("family", FAMILIES)
    def test_hierarchy_agrees_with_the_hierarchy_reader(self, family: str):
        implemented = family in hierarchy.SUPPORTED_FAMILIES
        assert (_status(family, "hierarchy") != "gap") == implemented

    @pytest.mark.parametrize("family", FAMILIES)
    def test_results_agrees_with_the_raw_readers_dialects(self, family: str):
        assert family in raw_header._DIALECTS  # pyright: ignore[reportPrivateUsage]
        assert _status(family, "results") != "gap"

    @pytest.mark.parametrize("family", FAMILIES)
    def test_controlled_startup_agrees_with_the_startup_policies(self, family: str):
        found = cap.entry(family, "controlled_startup")
        assert found is not None
        hosts = experiment_inputs.CONTROLLED_STARTUP_HOSTS.get(family)
        if hosts is None:
            assert found.status == "gap"
        else:
            assert found.status != "gap"
            assert found.hosts == hosts

    def test_a_family_specific_lint_rule_runs_for_its_family_only(self, tmp_path: Path):
        deck = (
            "* step\nV1 in 0 1\nR1 in 0 {r}\n.param r=1k\n.step param r 1k 2k 1k\n.tran 1m\n.end\n"
        )
        path = tmp_path / "step.cir"
        for_ngspice = lint_deck_ids(deck, path, "ngspice")
        for_xyce = lint_deck_ids(deck, path, "xyce")
        assert "step-ngspice" in for_ngspice
        assert "step-ngspice" not in for_xyce
        assert all(lint_rules.RULES_BY_ID[rule_id].family is None for rule_id in for_xyce)


def lint_deck_ids(deck: str, path: Path, dialect: str) -> set[str]:
    return {f["rule_id"] for f in lint_rules.lint_deck(deck, path, dialect, None)}


# ---------------------------------------------------------------------------
# The refusals read the table
# ---------------------------------------------------------------------------


class TestRefusals:
    @pytest.mark.parametrize(
        ("host", "where"), [("wsl", "WSL"), ("linux", "Wine"), ("darwin", "Wine")]
    )
    def test_qspice_is_refused_off_windows_naming_the_host(
        self, monkeypatch: pytest.MonkeyPatch, host: str, where: str
    ):
        _host(monkeypatch, host)
        refusal = family_refusal("qspice")
        assert refusal is not None
        assert "only when this server itself runs on Windows" in refusal
        assert f"under {where}" in refusal

    def test_qspice_runs_on_windows(self, monkeypatch: pytest.MonkeyPatch):
        _host(monkeypatch, "windows")
        assert family_refusal("qspice") is None

    @pytest.mark.parametrize("host", HOSTS)
    @pytest.mark.parametrize("family", ["ltspice", "ngspice", "xyce"])
    def test_the_other_families_are_not_refused_by_host(
        self, monkeypatch: pytest.MonkeyPatch, host: str, family: str
    ):
        _host(monkeypatch, host)
        assert family_refusal(family) is None

    def test_an_unknown_family_is_not_refused(self):
        assert family_refusal(None) is None
        assert cap.host_refusal("spectre", "linux") is None

    @pytest.mark.parametrize("family", ["qspice", "xyce"])
    def test_a_schematic_run_is_refused_with_the_declared_gap(self, family: str):
        found = cap.entry(family, "reads_asc_export")
        assert found is not None and found.status == "gap"
        assert asc_export_refusal(SIMULATORS[family]) == found.reason
        assert f"prepared for {DISPLAY[family]}" in (found.reason or "")

    @pytest.mark.parametrize("family", ["ltspice", "ngspice"])
    def test_a_schematic_runs_on_the_families_that_read_the_export(self, family: str):
        assert asc_export_refusal(SIMULATORS[family]) is None

    def test_gap_reason_is_none_unless_the_entry_is_a_gap(self):
        assert cap.gap_reason("ltspice", "lint_rules") is None
        assert cap.gap_reason("qspice", "results") is None  # unverified, not a gap
        assert cap.gap_reason("ltspice", "no_such_capability") is None
        assert cap.gap_reason(None, "lint_rules") is None
        assert cap.gap_reason("xyce", "lint_rules") is not None


# ---------------------------------------------------------------------------
# The session's view
# ---------------------------------------------------------------------------


class TestSessionView:
    def test_lists_the_given_families_in_the_tables_order(self):
        view = cap.session_view("linux", ["qspice", "ngspice"])
        assert list(view["families"]) == ["ngspice", "qspice"]
        assert view["evidence"] == cap.RESOURCE_URI

    def test_each_row_carries_kind_and_status_and_no_evidence(self):
        view = cap.session_view("windows", FAMILIES)
        for family, rows in view["families"].items():
            assert set(rows) == {e.capability for e in cap.entries_of(family)}
            for row in rows.values():
                assert row["kind"] in ("baseline", "refinement")
                assert row["status"] in ("supported", "gap", "unverified", "unavailable_here")
                assert "evidence" not in row
                if row["status"] != "supported":
                    assert row["reason"]

    def test_a_host_restriction_reads_as_unavailable_here(self):
        off_windows = cap.session_view("linux", ["qspice", "ltspice"])["families"]
        assert off_windows["qspice"]["run"]["status"] == "unavailable_here"
        assert "under Wine" in off_windows["qspice"]["run"]["reason"]
        assert off_windows["ltspice"]["open_window"]["status"] == "unavailable_here"
        on_windows = cap.session_view("windows", ["qspice", "ltspice"])["families"]
        assert on_windows["qspice"]["run"]["status"] == "unverified"
        assert on_windows["ltspice"]["open_window"]["status"] == "supported"

    def test_a_declared_gap_carries_its_reason(self):
        rows = cap.session_view("linux", ["ngspice"])["families"]["ngspice"]
        assert rows["native_sweeps"]["status"] == "gap"
        assert "step-ngspice" in rows["native_sweeps"]["reason"]


# ---------------------------------------------------------------------------
# The resource
# ---------------------------------------------------------------------------


@pytest.fixture
def bare_state(config: ServerConfig) -> SessionState:
    return SessionState.create(config, available={})


def _read_json(uri: str, state: SessionState) -> dict:
    result = handle_read_resource(uri, state)
    (content,) = result.contents
    assert getattr(content, "mime_type", None) == "application/json"
    return json.loads(content.text)  # type: ignore[union-attr]


class TestResource:
    def test_the_whole_table_with_its_evidence(self, bare_state: SessionState):
        data = _read_json(cap.RESOURCE_URI, bare_state)
        assert [c["name"] for c in data["capabilities"]] == [c.name for c in cap.CAPABILITIES]
        assert set(data["families"]) == set(FAMILIES)
        lint = data["families"]["ltspice"]["lint_rules"]
        assert lint["status"] == "supported"
        assert "recorded:log-meas-angle-unit" in lint["evidence"]
        assert set(data["statuses"]) == {"supported", "gap", "unverified"}
        assert set(data["evidence_kinds"]) == set(cap.EVIDENCE_KINDS)

    def test_one_family(self, bare_state: SessionState):
        data = _read_json(f"{cap.RESOURCE_URI}/xyce", bare_state)
        assert data["family"] == "xyce"
        assert data["display"] == "Xyce"
        assert data["capabilities"]["hierarchy"]["status"] == "gap"
        assert "nested instance targeting" in data["capabilities"]["hierarchy"]["reason"]
        assert data["capabilities"]["controlled_startup"]["status"] == "gap"

    def test_an_unknown_family_is_refused(self, bare_state: SessionState):
        with pytest.raises(ValueError, match="unknown simulator family"):
            handle_read_resource(f"{cap.RESOURCE_URI}/spectre", bare_state)
