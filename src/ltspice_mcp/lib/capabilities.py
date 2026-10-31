"""What each simulator family can do here, declared with its evidence.

One table says, for every family and every capability the server offers,
whether the capability is **supported**, with the evidence that establishes
it; a **gap**, with the reason; or **unverified**, implemented but with no
recording or live test behind it. A *baseline* capability has an entry for
every family, so a family that lacks it carries a declared gap instead of a
silence. A *refinement* is declared only by the families that have it, and
reaches the tool surface through an argument or action gated on it.

Features ask this table rather than compare a family's name:
``simulator.family_refusal`` and ``simulator.asc_export_refusal`` read it,
``experiment_inputs.prepare_startup`` reads where a controlled startup holds,
``verify_circuit`` reads it to say when a deck was checked with the
dialect-neutral rules only, ``inspect(kind="capabilities")`` shows the view
for this session (``session_view``), and the ``spice://capabilities``
resource serves the whole table with its evidence (``table_report``).

Each entry names the module that implements its capability, and each piece
of evidence is a typed reference (``EVIDENCE_KINDS``) that
``tests/test_capabilities.py`` resolves. The same test holds the table to the
code: a baseline capability the code has for a family must not be a gap
here, one it lacks must be, and a supported entry carries evidence.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Literal

from ltspice_mcp.lib.families import DISPLAY, FAMILIES, HOSTS

Kind = Literal["baseline", "refinement"]
Status = Literal["supported", "gap", "unverified"]

#: The prefixes an evidence reference may carry, and what each names.
EVIDENCE_KINDS: dict[str, str] = {
    "recorded": (
        "a behaviour key of the LTspice inventory, "
        "tests/fixtures/ltspice_recorded/inputs/cases.toml"
    ),
    "fixture": "a file under tests/fixtures that a simulator wrote",
    "test": "a test module that runs the simulator live or reads its files",
}

#: The URI of the resource that serves this table with its evidence.
RESOURCE_URI = "spice://capabilities"

_HOSTS = frozenset(HOSTS)


@dataclass(frozen=True)
class Capability:
    """One thing the server offers per simulator family."""

    name: str
    kind: Kind
    #: The need, stated without any simulator's mechanism.
    need: str
    #: The module that implements it; the binding test imports it.
    implemented_by: str


@dataclass(frozen=True)
class Entry:
    """One family's standing on one capability."""

    family: str
    capability: str
    status: Status
    #: Typed references (``EVIDENCE_KINDS``) that establish a supported entry.
    evidence: tuple[str, ...] = ()
    #: Why the entry is a gap or unverified.
    reason: str | None = None
    #: The hosts the capability holds on; None for every host.
    hosts: frozenset[str] | None = None
    #: How this family meets the need, where the mechanism is its own.
    mechanism: str | None = None
    #: Why the capability is not available on a host outside ``hosts``, with
    #: ``{where}`` standing for the layer that host runs Windows programs
    #: through (WSL, or Wine). Without it a plain sentence names the host.
    host_reason: str | None = None

    def __post_init__(self) -> None:
        if self.family not in FAMILIES:
            raise ValueError(f"unknown family {self.family!r}")
        if self.status == "supported" and not self.evidence:
            raise ValueError(f"{self.family}/{self.capability}: a supported entry needs evidence")
        if self.status != "supported" and not self.reason:
            raise ValueError(
                f"{self.family}/{self.capability}: a {self.status} entry needs a reason"
            )
        for ref in self.evidence:
            kind, sep, _ = ref.partition(":")
            if not sep or kind not in EVIDENCE_KINDS:
                raise ValueError(
                    f"{self.family}/{self.capability}: bad evidence reference {ref!r}"
                )
        if self.hosts is not None and not self.hosts <= _HOSTS:
            raise ValueError(
                f"{self.family}/{self.capability}: unknown host in {sorted(self.hosts)}"
            )
        if self.host_reason is not None and self.hosts is None:
            raise ValueError(f"{self.family}/{self.capability}: host_reason without hosts")

    def unavailable_on(self, host: str) -> str | None:
        """Why the capability is not available on ``host``, or None when it is."""
        if self.hosts is None or host in self.hosts:
            return None
        if self.host_reason is not None:
            return self.host_reason.format(where="WSL" if host == "wsl" else "Wine")
        what = self.capability.replace("_", " ")
        return (
            f"{DISPLAY[self.family]} {what} is available on {' or '.join(sorted(self.hosts))} "
            f"only, not on {host}."
        )


CAPABILITIES: tuple[Capability, ...] = (
    Capability(
        "run",
        "baseline",
        "Run a deck and its variations as points on this host.",
        "ltspice_mcp.lib.runner_base",
    ),
    Capability(
        "reads_asc_export",
        "baseline",
        "Run a schematic through the netlist LTspice exports from it.",
        "ltspice_mcp.lib.deck_prep",
    ),
    Capability(
        "lint_rules",
        "baseline",
        "Lint a deck for the constructs this simulator silently misreads.",
        "ltspice_mcp.lib.lint_rules",
    ),
    Capability(
        "validator_rules",
        "baseline",
        "Check a deck's directives and elements against this simulator's dialect before a run.",
        "ltspice_mcp.lib.spice_validator",
    ),
    Capability(
        "hierarchy",
        "baseline",
        "Resolve nested instances, models and parameters as this simulator does, for "
        "instance targeting and inspect(kind='hierarchy').",
        "ltspice_mcp.lib.hierarchy",
    ),
    Capability(
        "diagnostics",
        "baseline",
        "Read this simulator's log and console into warnings and errors.",
        "ltspice_mcp.lib.log_parser",
    ),
    Capability(
        "results",
        "baseline",
        "Read the results this simulator writes as traces against their axis.",
        "ltspice_mcp.lib.raw_header",
    ),
    Capability(
        "measurements",
        "baseline",
        "Measurements per point: the simulator's own where it computes them, the "
        "server's recipes otherwise.",
        "ltspice_mcp.lib.metrics",
    ),
    Capability(
        "progress_in_point",
        "refinement",
        "Report how far a running point has got, from its partial results.",
        "ltspice_mcp.lib.raw_parser",
    ),
    Capability(
        "controlled_startup",
        "refinement",
        "Start the simulator with captured settings, so a recoverable run repeats exactly.",
        "ltspice_mcp.lib.experiment_inputs",
    ),
    Capability(
        "native_measurements",
        "refinement",
        "Measurements the simulator computes in a batch run, read per step from its output.",
        "ltspice_mcp.lib.log_parser",
    ),
    Capability(
        "native_sweeps",
        "refinement",
        "A parameter sweep inside one run, every step in one results file.",
        "ltspice_mcp.lib.result_kinds",
    ),
    Capability(
        "compatibility_modes",
        "refinement",
        "Read a deck under another simulator's rules, selected per session.",
        "ltspice_mcp.lib.simulator",
    ),
    Capability(
        "pdk_statistics",
        "refinement",
        "Statistical variation computed by the simulator from a PDK's own model cards.",
        "ltspice_mcp.lib.pdk_native",
    ),
    Capability(
        "open_window",
        "refinement",
        "Show a sheet or a run in the simulator's own window, and keep an open sheet in "
        "step with the file.",
        "ltspice_mcp.lib.ltspice_bridge",
    ),
    Capability(
        "results_beside_sheet",
        "refinement",
        "Put a run's results where the simulator's window finds them for the sheet.",
        "ltspice_mcp.lib.sheet_results",
    ),
)

_CAPABILITY_BY_NAME: dict[str, Capability] = {c.name: c for c in CAPABILITIES}

#: The families the server runs through spicelib's common interface only:
#: nothing of theirs is recorded, and the deck-language code models none.
_UNMODELLED: tuple[str, ...] = ("qspice", "xyce")
_WINDOWS = frozenset({"windows"})


def _each(families: Iterable[str], capability: str, status: Status, reason: str) -> list[Entry]:
    """One entry per family, with ``{display}`` in ``reason`` spelled as a person
    names the family."""
    return [
        Entry(family, capability, status, reason=reason.format(display=DISPLAY[family]))
        for family in families
    ]


ENTRIES: tuple[Entry, ...] = (
    # --- run ---------------------------------------------------------------
    Entry("ltspice", "run", "supported", ("recorded:raw-transient", "recorded:log-banner")),
    Entry("ngspice", "run", "supported", ("test:tests/test_ngspice_e2e.py",)),
    Entry(
        "qspice",
        "run",
        "unverified",
        reason="No recording or live test of a QSPICE run exists.",
        hosts=_WINDOWS,
        host_reason=(
            "QSPICE runs only when this server itself runs on Windows: it is a Windows "
            "program, spicelib starts it with this host's own file paths, and under "
            "{where} it cannot open them (only LTspice has a path adapter here)."
        ),
    ),
    Entry("xyce", "run", "unverified", reason="No recording or live test of a Xyce run exists."),
    # --- reads_asc_export --------------------------------------------------
    Entry(
        "ltspice",
        "reads_asc_export",
        "supported",
        ("recorded:export-boilerplate", "recorded:export-instance-names"),
    ),
    Entry(
        "ngspice",
        "reads_asc_export",
        "supported",
        ("recorded:export-boilerplate", "test:tests/test_deck_prep.py"),
        mechanism="deck_prep drops the exporter's .backanno and the § in instance names.",
    ),
    *_each(
        _UNMODELLED,
        "reads_asc_export",
        "gap",
        "LTspice exports a schematic in its own netlist dialect, which is not prepared for "
        "{display}, so a run on {display} takes a hand-written .cir/.net/.sp netlist.",
    ),
    # --- lint_rules --------------------------------------------------------
    Entry(
        "ltspice",
        "lint_rules",
        "supported",
        (
            "recorded:log-meas-angle-unit",
            "recorded:log-meas-reserved-name",
            "recorded:deck-forms",
            "recorded:value-suffixes",
        ),
    ),
    Entry(
        "ngspice",
        "lint_rules",
        "supported",
        ("test:tests/test_lint_rules.py", "fixture:ngspice_ac_no_meas.log"),
    ),
    *_each(
        _UNMODELLED,
        "lint_rules",
        "gap",
        "No lint rules for {display} are declared; the dialect-neutral rules run.",
    ),
    # --- validator_rules ---------------------------------------------------
    Entry(
        "ltspice",
        "validator_rules",
        "supported",
        ("recorded:log-meas-db", "recorded:deck-forms"),
    ),
    Entry("ngspice", "validator_rules", "supported", ("test:tests/test_spice_validator.py",)),
    *_each(
        _UNMODELLED,
        "validator_rules",
        "gap",
        "No validator rules for {display} are declared; the dialect-neutral checks run.",
    ),
    # --- hierarchy ---------------------------------------------------------
    Entry(
        "ltspice",
        "hierarchy",
        "supported",
        ("recorded:subckt-device-names", "recorded:save-subckt-names", "recorded:deck-semantics"),
    ),
    Entry("ngspice", "hierarchy", "supported", ("test:tests/test_hierarchy.py",)),
    *_each(
        _UNMODELLED,
        "hierarchy",
        "gap",
        "The hierarchy reader models LTspice and ngspice only; a run on {display} gets no "
        "semantic profile, so nested instance targeting is unavailable.",
    ),
    # --- diagnostics -------------------------------------------------------
    Entry(
        "ltspice",
        "diagnostics",
        "supported",
        ("recorded:log-diagnostics", "recorded:log-convergence"),
    ),
    Entry(
        "ngspice",
        "diagnostics",
        "supported",
        (
            "fixture:ngspice_ac_no_meas.log",
            "fixture:native_log_tables/pz_table.log",
            "test:tests/test_ngspice_e2e.py",
        ),
    ),
    *_each(
        _UNMODELLED,
        "diagnostics",
        "unverified",
        "Convergence and fatal errors are read with the generic patterns; no recording of "
        "{display}'s wording exists.",
    ),
    # --- results -----------------------------------------------------------
    Entry(
        "ltspice",
        "results",
        "supported",
        ("recorded:raw-transient", "recorded:raw-ac", "recorded:raw-stepped"),
    ),
    Entry(
        "ngspice",
        "results",
        "supported",
        ("fixture:ngspice_noise_2plot.raw", "test:tests/test_ngspice_e2e.py"),
    ),
    Entry(
        "qspice",
        "results",
        "unverified",
        reason="A .qraw is read by its header's layout; no recording of one exists.",
    ),
    Entry("xyce", "results", "unverified", reason="No recording of a Xyce raw exists."),
    # --- measurements ------------------------------------------------------
    Entry("ltspice", "measurements", "supported", ("recorded:log-meas",)),
    Entry(
        "ngspice",
        "measurements",
        "supported",
        ("test:tests/test_ngspice_e2e.py",),
        mechanism="The server's recipes; see native_measurements for the simulator's own.",
    ),
    *_each(
        _UNMODELLED,
        "measurements",
        "unverified",
        "The server's recipes read the results, which are unverified for {display}.",
    ),
    # --- refinements -------------------------------------------------------
    Entry("ltspice", "progress_in_point", "supported", ("recorded:raw-partial",)),
    Entry("ngspice", "progress_in_point", "supported", ("test:tests/test_raw_parser.py",)),
    Entry(
        "qspice",
        "progress_in_point",
        "unverified",
        reason=(
            "The partial-results reader assumes every value is a double; no QSPICE raw is "
            "recorded."
        ),
    ),
    Entry(
        "xyce",
        "progress_in_point",
        "unverified",
        reason="No Xyce raw is recorded to establish the partial-results layout.",
    ),
    Entry(
        "ltspice",
        "controlled_startup",
        "supported",
        ("test:tests/test_recovery_launch.py",),
        hosts=_WINDOWS,
        mechanism="A captured LTspice.ini template.",
    ),
    Entry(
        "ngspice",
        "controlled_startup",
        "supported",
        ("test:tests/test_controlled_ngspice.py",),
        hosts=frozenset({"linux", "wsl", "windows"}),
        mechanism="An immutable spinit, selected through SPICE_SCRIPTS.",
    ),
    *_each(
        _UNMODELLED,
        "controlled_startup",
        "gap",
        "No controlled startup policy exists for {display}; a recoverable run is refused.",
    ),
    Entry(
        "ltspice",
        "native_measurements",
        "supported",
        ("recorded:log-meas", "recorded:log-step", "recorded:log-meas-failed"),
        mechanism="The .meas table in the log, one row per step.",
    ),
    Entry(
        "ngspice",
        "native_measurements",
        "gap",
        reason=(
            "ngspice run in batch mode with a raw output does not evaluate a top-level .meas "
            "(lint meas-ngspice-batch); measure with a recipe, or with a dot-less meas inside "
            ".control."
        ),
    ),
    Entry(
        "ltspice",
        "native_sweeps",
        "supported",
        ("recorded:raw-stepped", "recorded:log-step"),
        mechanism=".step, every step in one raw.",
    ),
    Entry(
        "ngspice",
        "native_sweeps",
        "gap",
        reason=(
            "ngspice has no .step and ignores the line in batch mode (lint step-ngspice); "
            "sweep with run_experiments variations, one case per point."
        ),
    ),
    Entry(
        "ngspice",
        "compatibility_modes",
        "supported",
        ("test:tests/test_config.py", "test:tests/test_hierarchy.py"),
        mechanism="[simulator] ngbehavior, passed as -D ngbehavior= on every run.",
    ),
    Entry(
        "ngspice",
        "pdk_statistics",
        "supported",
        ("test:tests/test_native_experiments.py",),
        mechanism="The pdk_native variation, run by ngspice's own statistical blocks.",
    ),
    Entry(
        "ltspice",
        "open_window",
        "supported",
        (
            "recorded:bridge-replies",
            "recorded:open-window-copy",
            "test:tests/test_ltspice_bridge.py",
        ),
        hosts=_WINDOWS,
        mechanism="The bridge LTspice 26.1 and later install.",
    ),
    Entry(
        "ltspice",
        "results_beside_sheet",
        "supported",
        ("recorded:sheet-results-command",),
        mechanism="A copy of the run's files beside the sheet, under the sheet's name.",
    ),
)


def _index(entries: Iterable[Entry]) -> dict[tuple[str, str], Entry]:
    """The entries by (family, capability); a repeat or an unknown capability is a bug."""
    by_key: dict[tuple[str, str], Entry] = {}
    for found in entries:
        if found.capability not in _CAPABILITY_BY_NAME:
            raise ValueError(f"entry for unknown capability {found.capability!r}")
        if (found.family, found.capability) in by_key:
            raise ValueError(f"{found.family}/{found.capability} is declared twice")
        by_key[found.family, found.capability] = found
    return by_key


_BY_KEY: dict[tuple[str, str], Entry] = _index(ENTRIES)


def capability(name: str) -> Capability:
    """The capability named ``name``."""
    return _CAPABILITY_BY_NAME[name]


def entry(family: str, name: str) -> Entry | None:
    """``family``'s entry for capability ``name``, or None where none is declared."""
    return _BY_KEY.get((family, name))


def entries_of(family: str) -> tuple[Entry, ...]:
    """Every entry declared for ``family``, in the capabilities' order."""
    return tuple(e for c in CAPABILITIES if (e := _BY_KEY.get((family, c.name))) is not None)


def host_refusal(family: str | None, host: str) -> str | None:
    """Why ``family`` cannot run on ``host``, or None when it can (or is unknown)."""
    if family is None:
        return None
    run = entry(family, "run")
    return None if run is None else run.unavailable_on(host)


def gap_reason(family: str | None, name: str) -> str | None:
    """The reason ``family``'s capability ``name`` is a declared gap, or None when
    it is not one (supported, unverified, undeclared or an unknown family)."""
    if family is None:
        return None
    found = entry(family, name)
    return found.reason if found is not None and found.status == "gap" else None


def session_view(host: str, families: Iterable[str]) -> dict[str, Any]:
    """The table as this session sees it: each of ``families`` with every
    declared capability's kind and status, where a capability a host
    restriction excludes here reads ``unavailable_here``. The evidence is left
    to the resource the view points at."""
    wanted = set(families)
    report: dict[str, Any] = {}
    for family in (f for f in FAMILIES if f in wanted):
        rows: dict[str, Any] = {}
        for found in entries_of(family):
            row: dict[str, Any] = {"kind": capability(found.capability).kind}
            unavailable = found.unavailable_on(host)
            if unavailable is not None:
                row["status"] = "unavailable_here"
                row["reason"] = unavailable
            else:
                row["status"] = found.status
                if found.reason is not None:
                    row["reason"] = found.reason
            rows[found.capability] = row
        report[family] = rows
    return {"families": report, "evidence": RESOURCE_URI}


def _entry_report(found: Entry) -> dict[str, Any]:
    report: dict[str, Any] = {"status": found.status}
    if found.evidence:
        report["evidence"] = list(found.evidence)
    if found.reason is not None:
        report["reason"] = found.reason
    if found.hosts is not None:
        report["hosts"] = sorted(found.hosts)
    if found.mechanism is not None:
        report["mechanism"] = found.mechanism
    return report


def _capabilities_of(family: str) -> dict[str, Any]:
    return {found.capability: _entry_report(found) for found in entries_of(family)}


def family_report(family: str) -> dict[str, Any]:
    """One family's entries with their evidence, for the resource."""
    if family not in FAMILIES:
        raise ValueError(f"unknown simulator family {family!r}; one of {list(FAMILIES)}")
    return {"family": family, "display": DISPLAY[family], "capabilities": _capabilities_of(family)}


def table_report() -> dict[str, Any]:
    """The whole table with its evidence, for the resource."""
    return {
        "capabilities": [
            {"name": c.name, "kind": c.kind, "need": c.need, "implemented_by": c.implemented_by}
            for c in CAPABILITIES
        ],
        "families": {family: _capabilities_of(family) for family in FAMILIES},
        "evidence_kinds": dict(EVIDENCE_KINDS),
    }
