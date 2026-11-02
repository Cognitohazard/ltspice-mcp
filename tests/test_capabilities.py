"""The capability table is held to the code it describes and to its evidence.

``ltspice_mcp.lib.capabilities`` declares, per simulator family, what the
server supports with evidence, what is a declared gap, and what is
implemented but unverified. The tests here keep that table honest:

- every baseline capability has an entry for every family, so a family that
  lacks one carries a reason instead of a silence;
- every piece of evidence resolves: a recorded LTspice behaviour is a key of
  the inventory, a fixture is a file under ``tests/fixtures``, a test is a
  module in ``tests/``; every implementing module imports;
- the table agrees with the code's behaviour for each capability: the lint
  rules declared for a family, the validator's directive rules, the hierarchy
  reader's families, the raw readers' dialects, the controlled-startup
  policies;
- the refusals that used to compare a family's name read the table.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest

from ltspice_mcp.errors import NetlistError
from ltspice_mcp.lib import capabilities as cap
from ltspice_mcp.lib import experiment_inputs, hierarchy, lint_rules, raw_header, spice_validator
from ltspice_mcp.lib.families import DISPLAY, FAMILIES, HOSTS
from ltspice_mcp.lib.simulator import SIMULATORS, asc_export_refusal, family_refusal
from ltspice_mcp.resources import handle_read_resource
from ltspice_mcp.state import SessionState
from tests import _ltspice_recorded as rec
from tests.conftest import FIXTURES_DIR, fake_host

ROOT = FIXTURES_DIR.parents[1]

BASELINE = [c.name for c in cap.CAPABILITIES if c.kind == "baseline"]
EVIDENCE = sorted({ref for e in cap.ENTRIES for ref in e.evidence})


# ---------------------------------------------------------------------------
# Shape
# ---------------------------------------------------------------------------


class TestShape:
    @pytest.mark.parametrize("name", BASELINE)
    def test_every_baseline_capability_has_an_entry_for_every_family(self, name: str):
        missing = [family for family in FAMILIES if cap.entry(family, name) is None]
        assert missing == [], f"{name} is baseline, so each of these needs an entry or a gap"

    def test_every_capability_has_at_least_one_entry(self):
        declared = {e.capability for e in cap.ENTRIES}
        assert [c.name for c in cap.CAPABILITIES if c.name not in declared] == []

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

    def test_entries_of_a_family_follow_the_capabilities_order(self):
        order = [c.name for c in cap.CAPABILITIES]
        for family in FAMILIES:
            names = [e.capability for e in cap.entries_of(family)]
            assert names == sorted(names, key=order.index)

    def test_the_families_are_the_simulators_a_run_selects(self):
        assert tuple(SIMULATORS) == FAMILIES


# ---------------------------------------------------------------------------
# Evidence resolves
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ref", EVIDENCE)
def test_every_evidence_reference_resolves(ref: str):
    kind, _, target = ref.partition(":")
    if kind == "recorded":
        assert target in rec.CASES.behaviours, f"{ref}: not a behaviour of the inventory"
    elif kind == "fixture":
        assert (FIXTURES_DIR / target).is_file(), f"{ref}: no such fixture"
    else:
        assert kind == "test"
        assert (ROOT / target).is_file(), f"{ref}: no such test module"


@pytest.mark.parametrize("name", [c.name for c in cap.CAPABILITIES])
def test_every_implementing_module_imports(name: str):
    importlib.import_module(cap.capability(name).implemented_by)


# ---------------------------------------------------------------------------
# Bound to the code's behaviour
# ---------------------------------------------------------------------------


def _is_gap(family: str, name: str) -> bool:
    found = cap.entry(family, name)
    assert found is not None, f"{family}/{name} has no entry"
    return found.status == "gap"


#: A directive each validator dialect refuses and the others accept: a
#: ``.meas`` named after a constant for LTspice, a zero-step ``.tran`` for
#: ngspice. A family with no rules of its own refuses neither.
_VALIDATOR_PROBES = (".meas tran q FIND V(out) AT 1m", ".tran 0 1m")


def _hierarchy_models(family: str) -> bool:
    try:
        hierarchy.SemanticProfile(family, "kiltpsa" if family == "ngspice" else None)  # type: ignore[arg-type]
    except NetlistError:
        return False
    return True


_PROBES = {
    "lint_rules": lambda family: any(rule.family == family for rule in lint_rules.RULES),
    "validator_rules": lambda family: any(
        spice_validator.validate_directive(probe, DISPLAY[family]) is not None
        for probe in _VALIDATOR_PROBES
    ),
    "hierarchy": _hierarchy_models,
    "reads_asc_export": lambda family: asc_export_refusal(SIMULATORS[family]) is None,
}


class TestBoundToTheCode:
    @pytest.mark.parametrize("family", FAMILIES)
    @pytest.mark.parametrize("name", sorted(_PROBES))
    def test_a_gap_is_declared_exactly_where_the_code_has_nothing(self, name: str, family: str):
        assert _is_gap(family, name) == (not _PROBES[name](family))

    @pytest.mark.parametrize("family", FAMILIES)
    def test_results_are_read_for_every_family_the_raw_readers_know(self, family: str):
        assert family in raw_header._DIALECTS  # pyright: ignore[reportPrivateUsage]
        assert not _is_gap(family, "results")

    @pytest.mark.parametrize(
        ("family", "host", "unavailable"),
        [
            ("ltspice", "windows", False),
            ("ltspice", "linux", True),
            ("ngspice", "linux", False),
            ("ngspice", "wsl", False),
            ("ngspice", "darwin", True),
            ("xyce", "windows", True),
            ("qspice", "windows", True),
        ],
    )
    def test_controlled_startup_reads_the_table(self, family: str, host: str, unavailable: bool):
        assert experiment_inputs._startup_unavailable(family, host) is unavailable  # pyright: ignore[reportPrivateUsage]

    def test_a_family_specific_lint_rule_runs_for_its_family_only(self, tmp_path: Path):
        deck = (
            "* step\nV1 in 0 1\nR1 in 0 {r}\n.param r=1k\n.step param r 1k 2k 1k\n.tran 1m\n.end\n"
        )
        path = tmp_path / "step.cir"
        for_ngspice = _lint_ids(deck, path, "ngspice")
        for_xyce = _lint_ids(deck, path, "xyce")
        assert "step-ngspice" in for_ngspice
        assert "step-ngspice" not in for_xyce
        assert all(lint_rules.RULES_BY_ID[rule_id].family is None for rule_id in for_xyce)


def _lint_ids(deck: str, path: Path, dialect: str) -> set[str]:
    return {f["rule_id"] for f in lint_rules.lint_deck(deck, path, dialect, None)}


# ---------------------------------------------------------------------------
# The refusals read the table
# ---------------------------------------------------------------------------


class TestRefusals:
    @pytest.mark.parametrize("host", [h for h in HOSTS if h != "windows"])
    def test_qspice_is_refused_off_windows_by_its_run_entry(
        self, monkeypatch: pytest.MonkeyPatch, host: str
    ):
        fake_host(monkeypatch, host)
        run = cap.entry("qspice", "run")
        assert run is not None
        assert family_refusal("qspice") == run.unavailable_on(host) is not None

    @pytest.mark.parametrize("host", HOSTS)
    @pytest.mark.parametrize("family", ["ltspice", "ngspice", "xyce"])
    def test_the_other_families_are_not_refused_by_host(
        self, monkeypatch: pytest.MonkeyPatch, host: str, family: str
    ):
        fake_host(monkeypatch, host)
        assert family_refusal(family) is None

    def test_qspice_runs_on_windows(self, monkeypatch: pytest.MonkeyPatch):
        fake_host(monkeypatch, "windows")
        assert family_refusal("qspice") is None

    def test_an_unknown_family_is_not_refused(self):
        assert family_refusal(None) is None
        assert cap.host_refusal("spectre", "linux") is None

    @pytest.mark.parametrize("family", ["qspice", "xyce"])
    def test_a_schematic_run_is_refused_with_the_declared_gap(self, family: str):
        found = cap.entry(family, "reads_asc_export")
        assert found is not None and found.status == "gap"
        assert asc_export_refusal(SIMULATORS[family]) == found.reason

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
        assert off_windows["ltspice"]["open_window"] == {
            "kind": "refinement",
            "status": "unavailable_here",
            "reason": "LTspice open window is available on windows only, not on linux.",
        }
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


def _read_json(uri: str, state: SessionState) -> dict:
    result = handle_read_resource(uri, state)
    (content,) = result.contents
    assert getattr(content, "mime_type", None) == "application/json"
    return json.loads(content.text)  # type: ignore[union-attr]


class TestResource:
    def test_the_whole_table_with_its_evidence(self, state_no_sim: SessionState):
        data = _read_json(cap.RESOURCE_URI, state_no_sim)
        assert [c["name"] for c in data["capabilities"]] == [c.name for c in cap.CAPABILITIES]
        assert set(data["families"]) == set(FAMILIES)
        lint = data["families"]["ltspice"]["lint_rules"]
        assert lint["status"] == "supported"
        assert "recorded:log-meas-angle-unit" in lint["evidence"]
        assert set(data["evidence_kinds"]) == set(cap.EVIDENCE_KINDS)

    def test_one_family(self, state_no_sim: SessionState):
        data = _read_json(f"{cap.RESOURCE_URI}/xyce", state_no_sim)
        assert data["family"] == "xyce"
        assert data["display"] == "Xyce"
        assert data["capabilities"]["hierarchy"]["status"] == "gap"
        assert "nested instance targeting" in data["capabilities"]["hierarchy"]["reason"]
        assert data["capabilities"]["controlled_startup"]["status"] == "gap"

    def test_an_unknown_family_is_refused(self, state_no_sim: SessionState):
        with pytest.raises(ValueError, match="unknown simulator family"):
            handle_read_resource(f"{cap.RESOURCE_URI}/spectre", state_no_sim)
