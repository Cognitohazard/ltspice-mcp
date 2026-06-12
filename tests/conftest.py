"""Shared fixtures and helpers for ltspice-mcp tests."""

import shutil
import typing
from collections.abc import Iterator
from datetime import timedelta
from pathlib import Path

import pytest
from spicelib import AscEditor

from ltspice_mcp.config import ServerConfig
from ltspice_mcp.lib import now
from ltspice_mcp.state import BatchJob, SessionState, SimulationJob

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


class FakeSim:
    """Stub simulator class for tests that need a default simulator."""

    spice_exe: typing.ClassVar[list[str]] = ["/fake/path/sim.exe"]


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


@pytest.fixture(autouse=True)
def _reset_runners():
    """No-op: runners now live on SessionState instances (RunnerManager).

    Each test creates a fresh SessionState with a fresh RunnerManager,
    so there are no module-level globals to reset. This fixture is kept
    as a guard — if runner management ever regresses to globals, tests
    will still pass.
    """
    return


@pytest.fixture(scope="session")
def asc_symbols() -> Iterator[Path]:
    """Register tiny .asy fixture symbols with AscEditor (class-level).

    Session-scoped so the class-level ``symbol_cache`` is populated once and
    reused across all tests. ``AscEditor._asy_file_find`` otherwise walks
    ``os.path.curdir`` (the project root, with ``.venv`` and ``.git``) on every
    cold load — ~1s per symbol lookup. Keeping the cache warm across the
    session eliminates that walk for every test after the first.
    """
    AscEditor.set_custom_library_paths(str(_FIXTURE_SYMBOLS))
    for asy in _FIXTURE_SYMBOLS.glob("*.asy"):
        AscEditor.symbol_cache[asy.name] = str(asy)
    yield _FIXTURE_SYMBOLS
    AscEditor.custom_lib_paths = []
    AscEditor.symbol_cache = {}


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

    import jsonschema

    import ltspice_mcp.tools as tools_pkg
    from ltspice_mcp.tools import _base as base_mod
    from ltspice_mcp.tools import get_tools_for_profile

    _, dispatch = get_tools_for_profile("full")
    code_to_tool: dict = {}
    validators: dict = {}
    for name, reg in dispatch.items():
        if reg.definition.outputSchema is None:
            continue
        # reg.handler is the registry's validation wrapper — a closure whose
        # code object is SHARED by every tool, so it can't identify the
        # emitter. @wraps preserves the original under __wrapped__; its code
        # object is unique per handler and is the frame that actually calls
        # format_response.
        target = getattr(reg.handler, "__wrapped__", reg.handler)
        code_to_tool[target.__code__] = name
        validators[name] = jsonschema.Draft202012Validator(reg.definition.outputSchema)

    def _validate(result) -> None:
        sc = result.structuredContent
        if sc is None:
            return
        # 0=_validate, 1=checked_* wrapper, 2=the wrapper's caller.
        frame = sys._getframe(2)
        for _ in range(25):
            if frame is None:
                return
            tool = code_to_tool.get(frame.f_code)
            if tool is not None:
                errors = list(validators[tool].iter_errors(sc))
                if errors:
                    raise AssertionError(
                        f"{tool}: structuredContent violates its declared output_schema: "
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
    # format_response``), so patch the binding in every tool module, not
    # just the defining module. The module set is derived from the
    # registered handlers themselves so a new tool module can't silently
    # escape conformance checking.
    import sys as _sys

    saved = []
    handler_modules = {
        _sys.modules[getattr(reg.handler, "__wrapped__", reg.handler).__module__]
        for reg in dispatch.values()
    }
    for mod in handler_modules | {base_mod, tools_pkg}:
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
