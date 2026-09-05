"""Shared fixtures and helpers for ltspice-mcp tests."""

import asyncio
import shutil
import typing
from collections.abc import Coroutine, Iterator
from datetime import timedelta
from pathlib import Path

import pytest
from spicelib import AscEditor

from ltspice_mcp.api import _session as _api_session
from ltspice_mcp.api._methods import ApiMethodsMixin
from ltspice_mcp.config import ServerConfig
from ltspice_mcp.engine import BootstrapResult
from ltspice_mcp.lib import now
from ltspice_mcp.lib.experiment_runner import ExperimentRunner
from ltspice_mcp.lib.experiment_types import (
    Completeness,
    ExperimentCase,
    ExperimentJob,
    SourceRecord,
)
from ltspice_mcp.lib.runner_base import RunOutcome
from ltspice_mcp.state import BatchJob, SessionState, SimulationJob

_T = typing.TypeVar("_T")

FIXTURES_DIR = Path(__file__).parent / "fixtures"
_FIXTURE_SYMBOLS = FIXTURES_DIR / "symbols"
_FIXTURE_DRAFT = FIXTURES_DIR / "Draft1.asc"

# Recorded real-LTspice fixture values shared across test modules.
# Single transient run of an RC low-pass (R=1k, C=100n, 1 V step input); its
# log holds the one .MEAS line ``vfinal: V(out)=0.999876166042 at 0.0009``.
LTSPICE_TRAN_RC_LOG = FIXTURES_DIR / "ltspice_tran_rc.log"
LTSPICE_TRAN_RC_VFINAL = 0.999876166042
# 3-run LTspice parameter sweep of the same RC low-pass (R1 = 1k / 2.2k /
# 4.7k), one .MEAS log per run as the sweep/MC runners record them.
LTSPICE_SWEEP_RUN_LOGS = [FIXTURES_DIR / f"ltspice_sweep_meas_run{i}.log" for i in range(3)]


# ---------------------------------------------------------------------------
# The registered tool surface, shared by every test that names it
# ---------------------------------------------------------------------------

# The six ops that share the ratified response envelope.
CONSOLIDATED_TOOLS = (
    "run_experiments",
    "jobs",
    "analyze_results",
    "edit_schematic",
    "verify_circuit",
    "inspect",
)

# The advertised surface: the six plus the plot widget, which is registered by
# ruling but predates the envelope — it joins surface-wide checks (size pins,
# completeness) and stays out of the envelope contract matrix. Membership is
# pinned BY NAME, never derived from schema shape: a tool that lost its
# envelope marker must fail a contract test, not silently reclassify.
REGISTERED_TOOLS = (*CONSOLIDATED_TOOLS, "plot_waveform")

# Every tool name removed in 0.6.0 when the consolidated profile became the
# product (frozen history; test_doc_drift composes its dead-name gate from it).
TOOLS_REMOVED_IN_0_6: tuple[str, ...] = (
    "create_netlist",
    "read_circuit",
    "list_components",
    "set_component_value",
    "parameter",
    "edit_directive",
    "export_netlist",
    "reset_schematic",
    "symbol_info",
    "component_info",
    "wire_pins",
    "create_schematic",
    "trace_net",
    "validate_netlist",
    "diff_circuit",
    "apply_schematic_ops",
    "signal_stats",
    "get_waveform",
    "export_waveform",
    "query_value",
    "operating_point",
    "simulation_summary",
    "edge_metrics",
    "transient_response",
    "timing_between",
    "periodic_metrics",
    "thd",
    "noise_integral",
    "measurement_stats",
    "stability_metrics",
    "bode_metrics",
    "resonance",
    "return_loss",
    "ac_structure",
    "run_simulation",
    "check_job",
    "cancel_job",
    "configure_sweep",
    "run_sweep",
    "configure_montecarlo",
    "run_montecarlo",
    "batch_results",
    "find_model",
    "load_library",
    "unload_library",
    "list_libraries",
    "server_status",
    "recent",
)

# Delegated handlers that legitimately declare NO structuredContent contract:
# name -> (reason, emits_intermediate_structured_content). The second field
# derives the conformance hook's walk-stop set — an adapter that emits
# intermediate structuredContent needs the frame walk stopped at it, while a
# text-only one must NOT stop the walk (that would subtract coverage for
# anything emitting beneath its frame). test_conformance_hook_armed.py's
# closure test pins the exemptions fail-closed (a name that gains a contract,
# or stops being delegated to, fails the suite).
NO_CONTRACT_DELEGATES: dict[str, tuple[str, bool]] = {
    "handle_pulse_response": ("transient adapter; its former dispatcher had no schema", True),
    "handle_disturbance_response": (
        "transient adapter; its former dispatcher had no schema",
        True,
    ),
    "handle_cancel_job": ("text-only confirmation; emits no structuredContent of its own", False),
}

INTERMEDIATE_NO_SCHEMA_ADAPTERS = tuple(
    name for name, (_, emits_intermediate) in NO_CONTRACT_DELEGATES.items() if emits_intermediate
)


class FakeSim:
    """Stub simulator class for tests that need a default simulator."""

    spice_exe: typing.ClassVar[list[str]] = ["/fake/path/sim.exe"]


async def terminal_experiment(state, payload: dict, *, wait_timeout_s: int = 120) -> dict:
    """Submit an experiment and return its TERMINAL receipt.

    A receipt whose dwell expired is followed through the real ``jobs(wait)``
    control plane — the route a client has — rather than by polling the store.
    Shared by the live ngspice/LTspice end-to-end files.
    """
    from ltspice_mcp.tools.experiments import (
        JobsInput,
        RunExperimentsInput,
        handle_jobs,
        handle_run_experiments,
    )

    result = await handle_run_experiments(RunExperimentsInput.model_validate(payload), state)
    data = result.structuredContent
    assert data is not None, result.content[0].text
    if data["outcome"] == "in_progress":
        waited = await handle_jobs(
            JobsInput.model_validate(
                {"action": "wait", "job_id": data["job_id"], "timeout_s": wait_timeout_s}
            ),
            state,
        )
        data = waited.structuredContent
        assert data is not None
    assert not data.get("timed_out"), f"job {data.get('job_id')} never went terminal: {data}"
    return data


def stage_recorded_fixture(work_dir: Path, name: str) -> Path:
    """Copy a recorded fixture's .raw (and .log when recorded) into work_dir.

    Returns the staged .raw path. The .log lands next to it so a handler's
    automatic ``raw_file`` -> ``.log`` derivation is exercised for real.
    """
    raw = work_dir / f"{name}.raw"
    shutil.copy(FIXTURES_DIR / f"{name}.raw", raw)
    log = FIXTURES_DIR / f"{name}.log"
    if log.exists():
        shutil.copy(log, work_dir / f"{name}.log")
    return raw


def fake_simulator(
    monkeypatch: pytest.MonkeyPatch,
    submissions: list[str] | None = None,
    *,
    delay_s: float | None = 0.0,
) -> list[str]:
    """Stand in for the simulator behind ``ExperimentRunner.submit_netlist``.

    A case is always accepted and recorded; ``delay_s`` decides when — and
    whether — it comes back:

    * ``0`` finishes it with a readable artifact pair before submit returns;
    * a positive delay finishes it that many seconds later on the loop, which
      is what makes a caller that failed to block print a receipt for a job
      still in flight;
    * ``None`` never calls back at all.

    Returns the list run filenames are appended to, so a caller that passed
    none can still read what was submitted.
    """
    recorded = [] if submissions is None else submissions

    def submit(self, _netlist: Path, run_filename: str, callback):
        recorded.append(run_filename)
        if delay_s is None:
            return object()
        raw = self.output_folder / f"{Path(run_filename).stem}.raw"
        log = self.output_folder / f"{Path(run_filename).stem}.log"

        def finish() -> None:
            raw.write_bytes(b"Title: mock")
            log.write_text("ok")
            callback(RunOutcome(str(raw), str(log), raw.stat().st_size, None))

        if delay_s > 0:
            self.loop.call_later(delay_s, finish)
        else:
            raw.write_bytes(b"Title: mock")
            log.write_text("ok")
            outcome = RunOutcome(str(raw), str(log), raw.stat().st_size, None)
            self.loop.call_soon_threadsafe(callback, outcome)
        return object()

    monkeypatch.setattr(ExperimentRunner, "submit_netlist", submit)
    return recorded


def recorded_fixture_simulator(monkeypatch: pytest.MonkeyPatch) -> None:
    """Instant engine behind ``ExperimentRunner.submit_netlist`` that hands back
    the recorded real-LTspice ``ltspice_tran_rc`` raw+log pair, so analysis
    stages parse genuine simulator artifacts rather than a mock byte string."""

    def submit(self, _netlist: Path, run_filename: str, callback):
        stem = Path(run_filename).stem
        raw = self.output_folder / f"{stem}.raw"
        log = self.output_folder / f"{stem}.log"
        shutil.copy(FIXTURES_DIR / "ltspice_tran_rc.raw", raw)
        shutil.copy(FIXTURES_DIR / "ltspice_tran_rc.log", log)
        outcome = RunOutcome(str(raw), str(log), raw.stat().st_size, None)
        self.loop.call_soon_threadsafe(callback, outcome)
        return object()

    monkeypatch.setattr(ExperimentRunner, "submit_netlist", submit)


def resolve_local_ref(schema: dict, node: dict) -> dict:
    """Follow a local ``$ref`` (possibly allOf-wrapped) into ``schema['$defs']``.

    Input schemas keep ``$defs`` instead of inlining (followups item 30);
    contract tests that assert on a nested submodel's shape resolve it the
    way a conformant client would.
    """
    while True:
        if "$ref" in node:
            node = schema["$defs"][node["$ref"].split("/")[-1]]
        elif "allOf" in node and len(node["allOf"]) == 1 and "$ref" in node["allOf"][0]:
            node = schema["$defs"][node["allOf"][0]["$ref"].split("/")[-1]]
        else:
            return node


def make_sim_job(job_id: str = "j1", *, status: str = "completed", **overrides) -> SimulationJob:
    """SimulationJob with test defaults; any dataclass field is overridable.

    A ``completed`` job gets a ``completed_at`` one second after
    ``started_at`` unless the caller overrides it.
    """
    started_at = overrides.pop("started_at", None) or now()
    fields: dict = {
        "netlist": Path("/tmp/test.cir"),
        "simulator": "ltspice",
        "completed_at": started_at + timedelta(seconds=1) if status == "completed" else None,
    }
    fields.update(overrides)
    return SimulationJob(
        job_id=job_id,
        status=status,  # type: ignore[arg-type]
        started_at=started_at,
        **fields,
    )


class SyncApi(ApiMethodsMixin):
    """Synchronous host that exercises the public mixin over a real state.

    Stands in for :class:`ltspice_mcp.api.Api` where the private loop thread
    and the process lease are not what a test is about: every call runs to
    completion on a throwaway loop, and the marshalling flags are discarded.
    """

    def __init__(self, state: SessionState) -> None:
        self._state = state

    def _check_process_and_thread(self) -> None:
        return None

    def _call(
        self,
        coroutine: Coroutine[typing.Any, typing.Any, _T],
        *,
        cancelable: bool = False,
        cancel_on_interrupt: bool = False,
        preserve_interrupt: bool = False,
    ) -> _T:
        del cancelable, cancel_on_interrupt, preserve_interrupt
        return asyncio.run(coroutine)


def patch_stub_bootstrap(monkeypatch: pytest.MonkeyPatch, state: object) -> None:
    """Make a real ``Api()`` adopt ``state`` instead of bootstrapping its own."""

    async def bootstrap(**kwargs: object) -> BootstrapResult:
        del kwargs
        return BootstrapResult(
            state=typing.cast(SessionState, state),
            preloaded_circuits=0,
        )

    monkeypatch.setattr(_api_session, "bootstrap_library_engine", bootstrap)


def make_experiment_job(
    state: SessionState,
    *,
    job_id: str,
    count: int = 1,
    status: str = "completed",
    case_id: str | None = None,
    run_index: int | None = None,
    raw: Path | None = None,
) -> ExperimentJob:
    """ExperimentJob registered on ``state``, for the Python API door tests.

    ``count`` expands that many synthetic cases (``case-0000``…), each with its
    own unwritten artifact pair. ``case_id`` / ``run_index`` name a single case
    instead, and ``raw`` points every case at one real recorded artifact pair —
    which is what the run-addressing tests resolve through.
    """
    deck = state.working_dir / f"{job_id}.cir"
    deck.write_text(".tran 1m\n.end\n", encoding="utf-8")
    complete = status == "completed"
    cases = [
        ExperimentCase(
            case_id=case_id or f"case-{index:04d}",
            run_index=index if run_index is None else run_index,
            circuit="dut",
            circuit_path=deck,
            staged_deck=deck,
            deck_sha256="a" * 64,
            assignments={"R": index},
            status="produced" if complete else "queued",
            raw_file=raw or state.working_dir / f"case-{index}.raw",
            log_file=(raw or state.working_dir / f"case-{index}.raw").with_suffix(".log"),
        )
        for index in range(count)
    ]
    job = ExperimentJob(
        job_id=job_id,
        request_id=f"request-{job_id}",
        fingerprint="fingerprint",
        canonicalizer_version=1,
        control_token="control-token",
        store_path=state.working_dir / f"{job_id}.json",
        cases=cases,
        sources=[
            SourceRecord(
                circuit="dut",
                path=deck,
                sha256="a" * 64,
                staged_deck=deck,
                simulator="ltspice",
            )
        ],
        simulator="ltspice",
        completeness=Completeness(
            declared=count,
            expanded=count,
            submitted=count if complete else 0,
            produced=count if complete else 0,
        ),
        status=typing.cast(typing.Any, status),
        completed_at=now() if complete else None,
    )
    state.add_experiment_job(job, already_persisted=True)
    return job


def make_batch_job(job_id: str = "b1", *, status: str = "completed", **overrides) -> BatchJob:
    """BatchJob with test defaults; any dataclass field is overridable.

    A ``completed`` job gets a ``completed_at`` one second after
    ``started_at`` unless the caller overrides it.
    """
    fields: dict = {
        "job_type": "sweep",
        "netlist": Path("/tmp/test.cir"),
        "total_runs": 2,
    }
    fields.update(overrides)
    bj = BatchJob(
        job_id=job_id,
        status=status,  # type: ignore[arg-type]
        **fields,
    )
    if status == "completed" and bj.completed_at is None:
        bj.completed_at = bj.started_at + timedelta(seconds=1)
    return bj


class _FakeSession:
    """Stub MCP session — log/progress calls are no-ops."""

    async def send_log_message(self, **kwargs):
        pass

    async def send_progress_notification(self, **kwargs):
        pass


class _FakeRequestContext:
    def __init__(self, state: SessionState):
        self.lifespan_context = {"state": state}
        self.session = _FakeSession()
        self.meta = None


class _FakeServer:
    """Stands in for the module-level MCP server so dispatch-level tests can
    drive call_tool/read_resource against a plain SessionState."""

    def __init__(self, state: SessionState):
        self.request_context = _FakeRequestContext(state)


@pytest.fixture
def work_dir(tmp_path: Path) -> Path:
    """Temporary working directory for tests."""
    return tmp_path


@pytest.fixture
def config(work_dir: Path) -> ServerConfig:
    """ServerConfig pointing at tmp working directory."""
    return ServerConfig(
        working_dir=work_dir,
        allowed_paths=[work_dir],
        log_level="DEBUG",
    )


@pytest.fixture
def state_no_sim(config: ServerConfig) -> SessionState:
    """SessionState with no simulators available (degraded mode)."""
    return SessionState.create(config, available={})


@pytest.fixture
def state_with_sim(config: ServerConfig) -> SessionState:
    """SessionState with a (fake) default simulator, so handlers that guard on
    simulator availability (e.g. cancel_job's require_simulator) get past it."""
    return SessionState.create(config, available={"fake": FakeSim})


@pytest.fixture(scope="session")
def _asc_symbol_cache() -> Iterator[Path]:
    """Warm AscEditor's class-level symbol cache with the .asy fixtures once.

    ``AscEditor._asy_file_find`` otherwise walks ``os.path.curdir`` (the project
    root, with ``.venv`` and ``.git``) on every cold load — ~1s per symbol
    lookup. Keeping the cache warm across the session eliminates that walk for
    every test after the first.
    """
    for asy in _FIXTURE_SYMBOLS.glob("*.asy"):
        AscEditor.symbol_cache[asy.name] = str(asy)
    yield _FIXTURE_SYMBOLS
    AscEditor.symbol_cache = {}


@pytest.fixture
def asc_symbols(_asc_symbol_cache: Path) -> Iterator[Path]:
    """Point symbol resolution at the .asy fixture library for this test.

    Re-asserted per test rather than once per session: booting the engine (any
    Api or server test) sets ``AscEditor.custom_lib_paths`` process-wide to the
    host's real LTspice symbol library and never puts it back, so a
    session-scoped assignment silently loses to whichever test ran first in the
    worker — a real symbol then resolves in place of the fixture one. The
    geometry cache is keyed by symbol name, so it is dropped alongside the paths
    or a name parsed from the real library would survive the switch.
    """
    from ltspice_mcp.lib import symbol_geometry

    previous_paths = AscEditor.custom_lib_paths
    previous_geometry = dict(symbol_geometry._symbol_cache)
    AscEditor.set_custom_library_paths(str(_FIXTURE_SYMBOLS))
    symbol_geometry._symbol_cache.clear()
    try:
        yield _FIXTURE_SYMBOLS
    finally:
        AscEditor.custom_lib_paths = previous_paths
        symbol_geometry._symbol_cache.clear()
        symbol_geometry._symbol_cache.update(previous_geometry)


@pytest.fixture
def asc_state(state_no_sim: SessionState, work_dir: Path, asc_symbols: Path) -> SessionState:
    """SessionState with .asc editor available and a Draft1.asc copied into work_dir."""
    dest = work_dir / "Draft1.asc"
    shutil.copy(_FIXTURE_DRAFT, dest)
    return state_no_sim


@pytest.fixture
def asc_file(asc_state: SessionState, work_dir: Path) -> Path:
    """Path to Draft1.asc within the test work_dir."""
    return work_dir / "Draft1.asc"


@pytest.fixture
def sample_netlist(work_dir: Path) -> Path:
    """Create a simple RC filter netlist and return its path."""
    p = work_dir / "rc_filter.cir"
    p.write_text(
        "* RC Low-Pass Filter\n"
        "R1 in out 1k\n"
        "C1 out 0 100n\n"
        "V1 in 0 AC 1\n"
        ".ac dec 100 1 1Meg\n"
        ".meas AC fc WHEN mag(V(out))=0.707\n"
        ".param Rval=1k\n"
        ".END\n"
    )
    return p


# ---------------------------------------------------------------------------
# Output-schema conformance hook
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session", autouse=True)
def _enforce_output_schema_conformance():
    """Validate every structuredContent emitted during the suite against the
    emitting tool's declared output_schema.

    Motivating bug: check_job's batch branch emitted ``"error": null``
    against a schema typing error as a non-nullable string — every
    schema-validating MCP client (including the official python SDK) raised
    on every batch-job poll, and no test caught it because handler tests
    read structuredContent as a plain dict. This hook turns every existing
    handler test into a conformance test: the emitting tool is identified
    by walking the call stack for a registered handler's frame, and its
    declared schema is enforced at the moment of emission.
    """
    import sys

    # The contract belongs to the handler, not to its registration: both
    # @registry.tool and @declare_output_schema stamp __output_schema__ on the
    # module-visible handler, so ONE scan over the tool modules finds every
    # contract — registered tool or internal adapter alike — and a delegated
    # emission validates against the ADAPTER's contract instead of falling
    # through to the delegating tool's outer envelope. Keyed by code object
    # (the frame that emits; __wrapped__ unwraps the registry's validation
    # wrapper), so two same-named handlers in different modules cannot share
    # a validator.
    from types import ModuleType

    import jsonschema

    import ltspice_mcp.tools as tools_pkg
    from ltspice_mcp.tools import _base as base_mod

    tool_modules = {
        mod
        for mod in vars(tools_pkg).values()
        if isinstance(mod, ModuleType) and mod.__name__.startswith("ltspice_mcp.tools.")
    }
    contracts: dict = {}  # code object -> (handler name, compiled validator)
    for mod in tool_modules:
        for obj in vars(mod).values():
            schema = getattr(obj, "__output_schema__", None)
            if schema is None or not callable(obj):
                continue
            code = getattr(obj, "__wrapped__", obj).__code__
            if code not in contracts:
                contracts[code] = (obj.__name__, jsonschema.Draft202012Validator(schema))

    # Intermediate-emitter frames whose structuredContent must NOT be validated
    # against the calling tool's schema. analyze_results delegates to compute
    # handlers to read a value out of each one's CallToolResult; that emission
    # is not analyze_results' returned envelope. A delegate that declares a
    # contract (via @registry.tool or @declare_output_schema) is caught by its
    # own frame above. The two transient adapters declare NONE — their former
    # dispatcher (transient_response) had no schema, so nothing ever validated
    # them — and the walk would otherwise fall through to analyze_results'
    # schema. Stop the walk at exactly those; the closure test in
    # test_conformance_hook_armed.py pins the full schema-less delegate set
    # fail-closed.
    from ltspice_mcp.tools import analysis as _analysis_mod

    _skip_codes: set = {
        getattr(fn, "__wrapped__", fn).__code__
        for fn in (getattr(_analysis_mod, name) for name in INTERMEDIATE_NO_SCHEMA_ADAPTERS)
    }

    def _validate(result) -> None:
        sc = result.structuredContent
        if sc is None:
            return
        # 0=_validate, 1=checked_* wrapper, 2=the wrapper's caller.
        frame = sys._getframe(2)
        for _ in range(25):
            if frame is None:
                return
            if frame.f_code in _skip_codes:
                # An unregistered compute adapter's intermediate emission — the
                # delegating tool reads a value from it but does not return it.
                return
            contract = contracts.get(frame.f_code)
            if contract is not None:
                name, validator = contract
                errors = list(validator.iter_errors(sc))
                if errors:
                    raise AssertionError(
                        f"{name}: structuredContent violates its declared output_schema: "
                        + "; ".join(e.message for e in errors[:3])
                    )
                return
            frame = frame.f_back

    original_format = base_mod.format_response
    original_json = base_mod.json_response

    def checked_format_response(text, data, fmt=None):
        result = original_format(text, data, fmt)
        # fmt="json" delegates to the (also patched) json_response, which
        # already validated this result — don't pay the stack walk and
        # schema validation twice per emission.
        if fmt != "json":
            _validate(result)
        return result

    def checked_json_response(data):
        result = original_json(data)
        _validate(result)
        return result

    # Handlers bind these helpers at import time (``from _base import
    # format_response``), so patch the binding in every tool module, not just
    # the defining module. The module set is the same package-derived set the
    # contract scan used — deriving it from REGISTERED handlers would silently
    # drop modules whose tools became unregistered adapters (circuit.py and
    # simulation.py carry contracts but no registrations), leaving their
    # emissions unvalidated.
    #
    # Known limitation: only bindings literally named format_response /
    # json_response are patched — an aliased import (``from _base import
    # format_response as fr``) or a module-local wrapper would escape the
    # check. tests/test_conformance_hook_armed.py proves the patch chain is
    # live for the canonical binding; it cannot prove no alias exists. Keep
    # the canonical names when adding tool modules.
    saved = []
    for mod in tool_modules | {base_mod, tools_pkg}:
        for attr, checked, orig in (
            ("format_response", checked_format_response, original_format),
            ("json_response", checked_json_response, original_json),
        ):
            if getattr(mod, attr, None) is orig:
                saved.append((mod, attr, orig))
                setattr(mod, attr, checked)
    yield
    for mod, attr, orig in saved:
        setattr(mod, attr, orig)
