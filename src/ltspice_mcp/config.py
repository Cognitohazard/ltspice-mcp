"""Server configuration with TOML and environment variable support."""

import logging
import os
import tempfile
import tomllib
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import tomlkit
from tomlkit import comment, document, nl, table

from ltspice_mcp.lib import atomic_write_text

logger = logging.getLogger(__name__)

ToolListing = Literal["full", "compact"]
VALID_TOOL_LISTINGS: frozenset[str] = frozenset({"full", "compact"})

VALID_LOG_LEVELS: frozenset[str] = frozenset({"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"})


def _validate_numeric(
    config_dict: dict,
    key: str,
    type_fn: type,
    min_val: float,
    max_val: float,
    *,
    exclusive_min: bool = False,
    source: str = "config",
) -> None:
    """Validate a numeric config value in-place, dropping it on failure.

    Unlike ``_load_bounded_env`` this operates on an already-loaded dict
    so it can validate TOML values using the same bounds as env overrides.
    """
    if key not in config_dict:
        return
    raw = config_dict[key]
    try:
        val = type_fn(raw)
    except (ValueError, TypeError):
        logger.warning("%s: invalid value %r for %s; ignoring", source, raw, key)
        del config_dict[key]
        return
    too_low = val <= min_val if exclusive_min else val < min_val
    if too_low or val > max_val:
        low = f">{min_val}" if exclusive_min else str(min_val)
        logger.warning("%s: %s must be %s-%s, got %s; ignoring", source, key, low, max_val, val)
        del config_dict[key]
    else:
        config_dict[key] = val


def _load_bounded_env(
    env_var: str,
    config_dict: dict,
    key: str,
    type_fn: type,
    min_val: float,
    max_val: float,
    *,
    exclusive_min: bool = False,
) -> None:
    """Load a numeric env var into *config_dict* if it passes bounds validation."""
    raw = os.getenv(env_var)
    if raw is None:
        return
    try:
        val = type_fn(raw)
    except (ValueError, TypeError):
        logger.warning("%s: invalid value %r; ignoring", env_var, raw)
        return
    too_low = val <= min_val if exclusive_min else val < min_val
    if too_low or val > max_val:
        low = f">{min_val}" if exclusive_min else str(min_val)
        logger.warning("%s must be %s-%s, got %s; ignoring", env_var, low, max_val, val)
    else:
        config_dict[key] = val


def _validated_listing(value: object, source: str) -> str | None:
    """Return value if it names a tool-listing mode, else warn and return None."""
    if isinstance(value, str) and value in VALID_TOOL_LISTINGS:
        return value
    logger.warning(
        "Unknown tool listing %r in %s, using 'full'; valid values are %s",
        value,
        source,
        ", ".join(sorted(VALID_TOOL_LISTINGS)),
    )
    return None


def _validated_string_list(
    value: object, field_name: str, source: str = "config"
) -> list[str] | None:
    """Return *value* if it is a genuine list of strings, else warn and return None.

    Guards the list-of-strings TOML fields against the common mistake of writing
    a bare scalar (``allowed_paths = "/home/me"`` instead of ``["/home/me"]``).
    A scalar string is iterable, so ``[Path(p) for p in value]`` would silently
    expand it character-wise into one-character Paths (including ``Path("/")``
    for every slash) rather than failing — which, for a sandbox path list, would
    widen the sandbox to the whole filesystem. Callers keep their default when
    this returns None.
    """
    if isinstance(value, list) and all(isinstance(x, str) for x in value):
        return value
    logger.warning("%s: %s must be a list of strings; ignoring %r", source, field_name, value)
    return None


# The exact simulator-resolution keys the loader reads, exported as constants
# and used AT THE READ SITES below: the capabilities remediation and the
# no-simulator error compose their guidance from these same names, so the key
# a message tells the user to set is the key the loader honors — a
# hand-written key name in guidance text drifts; a shared constant cannot.
SIM_SECTION = "simulator"
SIM_PATH_KEY = "path"
SIM_ENABLED_KEY = "enabled"
SIM_PATH_ENV = "LTSPICE_MCP_SIMULATOR_EXE"
SIM_ENABLED_ENV = "LTSPICE_MCP_ENABLED_SIMULATORS"


#: Returned by a coercer that rejected its input: the field keeps whatever the
#: lower-precedence source (or the dataclass default) already put there. It is
#: not the same as ``None``, which several fields use as a real value.
_SKIP: Any = object()


def _toml_simulator(value: Any) -> Any:
    """An empty ``default`` means auto-select, not the empty string."""
    return value or None


def _toml_simulator_exe(value: Any) -> Any:
    return Path(value) if value else _SKIP


def _toml_enabled_simulators(value: Any) -> Any:
    names = _validated_string_list(value, f"{SIM_SECTION}.{SIM_ENABLED_KEY}")
    return _SKIP if names is None else [x.strip().lower() for x in names]


def _toml_ngbehavior(value: Any) -> Any:
    if not isinstance(value, str):
        logger.warning("config: simulator.ngbehavior must be a string; ignoring %r", value)
        return _SKIP
    return value.strip() or _SKIP


def _toml_path_list(field_name: str) -> Callable[[Any], Any]:
    """Coercer for a TOML list-of-strings read as filesystem paths."""

    def coerce(value: Any) -> Any:
        paths = _validated_string_list(value, field_name)
        return _SKIP if paths is None else [Path(p) for p in paths]

    return coerce


def _toml_log_level(value: Any) -> Any:
    level = str(value).upper()
    if level in VALID_LOG_LEVELS:
        return level
    logger.warning(
        "config: invalid log level %r; must be one of %s", value, sorted(VALID_LOG_LEVELS)
    )
    return _SKIP


def _toml_tool_listing(value: Any) -> Any:
    return _validated_listing(value, "config") or _SKIP


def _toml_bool(name: str) -> Callable[[Any], Any]:
    """A TOML reader for one boolean key, refusing anything but a real boolean."""

    def read(value: Any) -> Any:
        if isinstance(value, bool):
            return value
        logger.warning("config: %s must be boolean; ignoring %r", name, value)
        return _SKIP

    return read


def _toml_preload_recent_count(value: Any) -> Any:
    if isinstance(value, int) and value >= 0:
        return value
    logger.warning(
        "config: state.preload_recent_count must be a non-negative integer; ignoring %r", value
    )
    return _SKIP


def _env_path(value: str) -> Any:
    return Path(value)


def _env_path_list(value: str) -> Any:
    return [Path(p) for p in value.split(os.pathsep)]


def _env_enabled_simulators(value: str) -> Any:
    # Comma- or os.pathsep-separated list of simulator names.
    sep = "," if "," in value else os.pathsep
    return [x.strip().lower() for x in value.split(sep) if x.strip()]


def _env_ngbehavior(value: str) -> Any:
    return value.strip() or _SKIP


def _env_log_level(value: str) -> Any:
    level = value.upper()
    if level in VALID_LOG_LEVELS:
        return level
    logger.warning(
        "LTSPICE_MCP_LOG_LEVEL: invalid value %r; must be one of %s",
        value,
        sorted(VALID_LOG_LEVELS),
    )
    return _SKIP


def _env_tool_listing(value: str) -> Any:
    return _validated_listing(value.strip(), "LTSPICE_MCP_TOOL_LISTING") or _SKIP


def _env_bool(name: str) -> Callable[[str], Any]:
    """An environment reader for one boolean variable (1/true/yes/on, 0/false/no/off)."""

    def read(value: str) -> Any:
        normalized = value.strip().lower()
        if normalized in ("1", "true", "yes", "on"):
            return True
        if normalized in ("0", "false", "no", "off"):
            return False
        logger.warning("%s: invalid boolean %r; ignoring", name, value)
        return _SKIP

    return read


def _env_preload_recent_count(value: str) -> Any:
    try:
        parsed = int(value)
        if parsed < 0:
            raise ValueError("must be >= 0")
    except ValueError as e:
        logger.warning(
            "LTSPICE_MCP_PRELOAD_RECENT_COUNT: invalid integer %r (%s); ignoring", value, e
        )
        return _SKIP
    return parsed


@dataclass(frozen=True)
class _Bounds:
    """Range a numeric setting is validated against, from either source."""

    type_fn: type
    min_val: float
    max_val: float
    exclusive_min: bool = False


@dataclass(frozen=True)
class _Setting:
    """One config field and the two places a value for it can come from.

    ``bounds`` replaces both coercers for a numeric field: the TOML value is
    taken as written and range-checked after the whole file is read (so the
    file's own values are validated with the same bounds as the environment's),
    and the environment value goes through the same check as it is read.
    """

    field: str
    section: str | None = None
    key: str | None = None
    from_toml: Callable[[Any], Any] | None = None
    env: str | None = None
    from_env: Callable[[str], Any] | None = None
    bounds: _Bounds | None = None
    env_accepts_empty: bool = False
    """Whether an empty environment value is a value to validate (and warn
    about) rather than an absent override. True only for the two settings
    whose readers distinguish 'set to something invalid' from 'unset'."""


#: Every setting the loader reads, in TOML-section order. Driven once for the
#: TOML file and once for the environment, so a new setting is one row rather
#: than one branch in each pass — the shape that let a key be added to one pass
#: and forgotten in the other.
_SETTINGS: tuple[_Setting, ...] = (
    _Setting(
        field="simulator",
        section=SIM_SECTION,
        key="default",
        from_toml=_toml_simulator,
        env="LTSPICE_MCP_SIMULATOR",
    ),
    _Setting(
        field="simulator_exe",
        section=SIM_SECTION,
        key=SIM_PATH_KEY,
        from_toml=_toml_simulator_exe,
        env=SIM_PATH_ENV,
        from_env=_env_path,
    ),
    _Setting(
        field="enabled_simulators",
        section=SIM_SECTION,
        key=SIM_ENABLED_KEY,
        from_toml=_toml_enabled_simulators,
        env=SIM_ENABLED_ENV,
        from_env=_env_enabled_simulators,
    ),
    _Setting(
        field="ngbehavior",
        section=SIM_SECTION,
        key="ngbehavior",
        from_toml=_toml_ngbehavior,
        env="LTSPICE_MCP_NGBEHAVIOR",
        from_env=_env_ngbehavior,
    ),
    _Setting(
        field="allowed_paths",
        section="security",
        key="allowed_paths",
        from_toml=_toml_path_list("security.allowed_paths"),
        env="LTSPICE_MCP_ALLOWED_PATHS",
        from_env=_env_path_list,
    ),
    _Setting(
        field="max_parallel_sims",
        section="simulation",
        key="max_parallel",
        env="LTSPICE_MCP_MAX_PARALLEL",
        bounds=_Bounds(int, 1, 128),
    ),
    _Setting(
        field="max_experiment_cases",
        section="simulation",
        key="max_experiment_cases",
        env="LTSPICE_MCP_MAX_EXPERIMENT_CASES",
        bounds=_Bounds(int, 1, 1_000_000),
    ),
    _Setting(
        field="default_timeout",
        section="simulation",
        key="timeout",
        env="LTSPICE_MCP_TIMEOUT",
        bounds=_Bounds(float, 0, 86400, exclusive_min=True),
    ),
    _Setting(
        field="max_estimated_points",
        section="simulation",
        key="max_estimated_points",
        env="LTSPICE_MCP_MAX_ESTIMATED_POINTS",
        bounds=_Bounds(int, 1, 100_000_000_000),
    ),
    _Setting(
        field="max_raw_mb",
        section="simulation",
        key="max_raw_mb",
        env="LTSPICE_MCP_MAX_RAW_MB",
        bounds=_Bounds(int, 1, 10_000_000),
    ),
    _Setting(
        field="max_points_returned",
        section="analysis",
        key="max_points",
        env="LTSPICE_MCP_MAX_POINTS",
        bounds=_Bounds(int, 1, 10_000_000),
    ),
    _Setting(
        field="analysis_budget_s",
        section="analysis",
        key="analysis_budget_s",
        env="LTSPICE_MCP_ANALYSIS_BUDGET_S",
        bounds=_Bounds(float, 0, 3600, exclusive_min=True),
    ),
    _Setting(
        field="default_budget",
        section="analysis",
        key="default_budget",
        env="LTSPICE_MCP_DEFAULT_BUDGET",
        bounds=_Bounds(int, 0, 10_000_000),
    ),
    _Setting(
        field="result_set_ttl_hours",
        section="analysis",
        key="result_set_ttl_hours",
        env="LTSPICE_MCP_RESULT_SET_TTL_HOURS",
        bounds=_Bounds(float, 0, 87600, exclusive_min=True),
    ),
    _Setting(
        field="log_level",
        section="logging",
        key="level",
        from_toml=_toml_log_level,
        env="LTSPICE_MCP_LOG_LEVEL",
        from_env=_env_log_level,
    ),
    _Setting(
        field="symbol_paths",
        section="schematic",
        key="symbol_paths",
        from_toml=_toml_path_list("schematic.symbol_paths"),
        env="LTSPICE_MCP_SYMBOL_PATHS",
        from_env=_env_path_list,
    ),
    _Setting(
        field="tool_listing",
        section="tools",
        key="listing",
        from_toml=_toml_tool_listing,
        env="LTSPICE_MCP_TOOL_LISTING",
        from_env=_env_tool_listing,
    ),
    _Setting(
        field="run_code",
        section="tools",
        key="run_code",
        from_toml=_toml_bool("tools.run_code"),
        env="LTSPICE_MCP_RUN_CODE",
        from_env=_env_bool("LTSPICE_MCP_RUN_CODE"),
    ),
    _Setting(
        field="persist_jobs",
        section="state",
        key="persist_jobs",
        from_toml=_toml_bool("state.persist_jobs"),
        env="LTSPICE_MCP_PERSIST_JOBS",
        from_env=_env_bool("LTSPICE_MCP_PERSIST_JOBS"),
        env_accepts_empty=True,
    ),
    _Setting(
        field="preload_recent_count",
        section="state",
        key="preload_recent_count",
        from_toml=_toml_preload_recent_count,
        env="LTSPICE_MCP_PRELOAD_RECENT_COUNT",
        from_env=_env_preload_recent_count,
        env_accepts_empty=True,
    ),
    # Environment-only: the working directory is where the config file itself
    # is looked up, so it cannot be configured from inside that file.
    _Setting(
        field="working_dir",
        env="LTSPICE_MCP_WORKING_DIR",
        from_env=_env_path,
    ),
)


def config_key(field: str) -> str:
    """The ``section.key`` an operator writes for one ``ServerConfig`` field."""
    for setting in _SETTINGS:
        if setting.field == field and setting.section is not None:
            return f"{setting.section}.{setting.key}"
    raise KeyError(field)


@dataclass
class ServerConfig:
    """Configuration for the LTSpice MCP server.

    Configuration sources (in order of precedence, highest to lowest):
    1. Environment variables (LTSPICE_MCP_*)
    2. TOML configuration file
    3. Hardcoded defaults
    """

    simulator: str | None = None
    """Preferred (default) simulator name (ltspice, ngspice, qspice, xyce).
    None = auto-select. Must be one of ``enabled_simulators`` when that list
    is non-empty."""

    enabled_simulators: list[str] = field(default_factory=list)
    """Allowlist of simulators to make available (ltspice, ngspice, qspice,
    xyce). Empty (default) = auto-detect every supported simulator. When
    non-empty, only the listed simulators are probed/exposed."""

    simulator_exe: Path | None = None
    """Explicit path to simulator executable. Overrides auto-detection."""

    ngbehavior: str | None = None
    """ngspice compatibility mode (``ngbehavior``). ``None`` leaves spicelib's
    default (``kiltpsa``), whose LTspice (``lt``) and PSPICE (``ps``) tokens each
    make ngspice read a sectioned ``.lib <file> <section>`` (the PDK corner idiom)
    as two plain includes, dropping the section. Set a mode with neither token
    (e.g. ``"hsa"`` or ``"kia"``) for standard-SPICE / PDK decks. Applied at
    startup to ``NGspiceSimulator._compatibility_mode``; ngspice-only, ignored by
    other simulators."""

    working_dir: Path = field(default_factory=Path.cwd)
    """Working directory for circuit files."""

    allowed_paths: list[Path] = field(default_factory=list)
    """Sandbox paths. Defaults to [working_dir] if empty."""

    max_parallel_sims: int = field(default_factory=lambda: min(os.cpu_count() or 4, 8))
    """Maximum concurrent simulations.

    Defaults to the host core count capped at 8 (a 64-core box otherwise sat at
    4). The cap keeps parallel cold simulator processes from thrashing memory/IO;
    raise it via ``[simulation] max_parallel`` or ``LTSPICE_MCP_MAX_PARALLEL`` (to
    128) when the box can take it.
    """

    max_experiment_cases: int = 1024
    """Maximum expanded cases accepted by one ``run_experiments`` call."""

    default_timeout: float = 300.0
    """Simulation timeout in seconds."""

    max_estimated_points: int = 20_000_000
    """Preflight WARN threshold: a .tran/.ac/.dc whose estimated point count
    exceeds this gets a warning (large raws are slow to produce and parse).
    Estimate only — .tran uses adaptive stepping. ``[simulation] max_estimated_points``."""

    max_raw_mb: int = 4096
    """Preflight REFUSE threshold: reject a run whose estimated raw size exceeds
    this. Estimated as a single-trace lower bound (8 bytes/point), so it never
    false-refuses (a real multi-trace raw is only larger) — it catches a runaway
    directive (e.g. a fs step for a ns run) before it fills the disk.
    ``[simulation] max_raw_mb``."""

    max_points_returned: int = 10000
    """Maximum waveform data points to return."""

    analysis_budget_s: float = 60.0
    """Whole-call work budget for ``analyze_results``."""

    result_set_ttl_hours: float = 24.0
    """Retention for raw-path-only immutable analysis result sets."""

    default_budget: int = 4000
    """Server-side response budget, in estimated tokens, for a consolidated-profile
    call that sets no ``budget`` of its own. It engages only the ladder's trim rung
    — empty presentation blocks and the identity echo — so it can never cut a fact
    or revoke a detail the caller explicitly asked for. Set 0 to leave every default
    response undegraded. ``[analysis] default_budget``."""

    log_level: str = "WARNING"
    """Stderr logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).

    WARNING by default because the server's stderr is not always a log file. A
    caller driving the engine in-process, or spawning a server per script, gets
    it interleaved with their own output — and a ~34-line INFO startup banner
    there is answered with a blanket ``2>/dev/null``, which then hides the
    tracebacks that mattered. Startup detail is still one setting away
    (``[logging] level`` or ``LTSPICE_MCP_LOG_LEVEL``), and an ``inspect``
    capabilities query reports the same facts on demand. Stderr is the server's
    only log channel: the MCP logging capability that once carried these
    messages to the client is deprecated as of the 2026-07-28 revision and is
    no longer served."""

    symbol_paths: list[Path] = field(default_factory=list)
    """Custom paths to LTspice symbol (.asy) files for .asc schematic support.
    On Windows and WSL these are auto-detected; set this to override."""

    tool_listing: ToolListing = "full"
    """How much of each tool definition the tool list carries.

    ``"full"`` (the default) advertises the seven tools exactly as they are
    registered. ``"compact"`` advertises the same seven with every per-argument
    description removed from the published schema; structure, enums, defaults
    and ``$defs`` are untouched, and the tools accept exactly what they did.
    Both listings are static — the same for every connection, and unchanged by
    anything called on it — so a client may cache either one."""

    run_code: bool = False
    """Advertise the ``run_code`` tool: a Python snippet run in a warm worker
    process that holds this server's engine as ``api``. Off by default because
    the snippet runs with the server process's own file and process authority,
    not inside ``allowed_paths``; turning it on is the operator's decision, and
    it takes effect at the next start."""

    persist_jobs: bool = True
    """Persist experiment job records to the working directory's
    ``.ltspice-mcp/`` store so a restarted server can surface prior runs."""

    preload_recent_count: int = 10
    """At startup, eagerly load persisted jobs for this many recently-touched
    circuits (capped by ``recent.json``). Set to 0 to disable preload and
    fall back to lazy loading on first tool call. Bounded-IO; typical
    cost is a handful of millisecond-scale JSON reads."""

    config_path: Path = field(default_factory=lambda: Path.cwd() / "ltspice-mcp.toml")
    """Path that was resolved for the config file (set by load())."""

    def __post_init__(self) -> None:
        """A config without allowed_paths gets the default sandbox."""
        if not self.allowed_paths:
            self.allowed_paths = default_allowed_paths(self.working_dir)

    @classmethod
    def load(
        cls,
        config_path: Path | None = None,
        *,
        overrides: Mapping[str, object] | None = None,
    ) -> "ServerConfig":
        """Load configuration from defaults, TOML, environment, and overrides.

        Args:
            config_path: Path to TOML config file. If None, looks for ltspice-mcp.toml
                        in the current working directory.
            overrides: Explicit values applied after environment variables. Callers
                       are responsible for validating names and value types.

        Returns:
            Populated ServerConfig instance.
        """
        config_dict: dict = {}

        if config_path is None:
            env_config = os.getenv("LTSPICE_MCP_CONFIG")
            config_path = Path(env_config) if env_config else Path.cwd() / "ltspice-mcp.toml"

        if config_path.exists():
            with open(config_path, "rb") as f:
                toml_data = tomllib.load(f)

            for setting in _SETTINGS:
                if setting.section is None or setting.section not in toml_data:
                    continue
                section = toml_data[setting.section]
                if setting.key not in section:
                    continue
                raw = section[setting.key]
                value = setting.from_toml(raw) if setting.from_toml else raw
                if value is not _SKIP:
                    config_dict[setting.field] = value

            # Range-checked after the whole file is read, so a numeric key gets
            # the same bounds (and the same warning) from the file as from the
            # environment; a value outside them is dropped, not clamped.
            for setting in _SETTINGS:
                if setting.bounds is not None:
                    _validate_numeric(
                        config_dict,
                        setting.field,
                        setting.bounds.type_fn,
                        setting.bounds.min_val,
                        setting.bounds.max_val,
                        exclusive_min=setting.bounds.exclusive_min,
                        source="config",
                    )

        for setting in _SETTINGS:
            if setting.env is None:
                continue
            if setting.bounds is not None:
                _load_bounded_env(
                    setting.env,
                    config_dict,
                    setting.field,
                    setting.bounds.type_fn,
                    setting.bounds.min_val,
                    setting.bounds.max_val,
                    exclusive_min=setting.bounds.exclusive_min,
                )
                continue
            raw_env = os.getenv(setting.env)
            if raw_env is None or (not raw_env and not setting.env_accepts_empty):
                continue
            value = setting.from_env(raw_env) if setting.from_env else raw_env
            if value is not _SKIP:
                config_dict[setting.field] = value

        if overrides:
            config_dict.update(overrides)

        config_dict["config_path"] = config_path
        return cls(**config_dict)


def claude_scratch_root() -> Path | None:
    """Where Claude Code keeps a session's scratch files. Its system prompt tells
    an agent to write throwaway files there rather than in the working
    directory, so a deck an agent authors lands outside a sandbox of ["."] and
    every run of it costs a copy first. POSIX layout only; on Windows the
    location is not known, so nothing is added."""
    if os.name != "posix":
        return None
    return Path(tempfile.gettempdir()) / f"claude-{os.getuid()}"


def default_allowed_paths(working_dir: Path) -> list[Path]:
    """The sandbox a config without ``allowed_paths`` gets."""
    scratch = claude_scratch_root()
    return [working_dir] + ([scratch] if scratch else [])


def generate_default_config(path: Path) -> None:
    """Generate a self-documenting default configuration file.

    Args:
        path: Path where the TOML config file should be written.
    """
    doc = document()

    # Simulator section
    doc.add(comment("LTSpice MCP Server Configuration"))
    doc.add(
        comment(
            "All settings have sensible defaults and can be overridden with environment variables"
        )
    )
    doc.add(nl())

    sim = table()
    sim.add(comment("Preferred simulator: ltspice, ngspice, qspice, xyce"))
    sim.add(
        comment("Leave empty or set to null for auto-detection (prefers LTSpice if available)")
    )
    sim.add("default", "ltspice")
    sim.add(nl())
    sim.add(comment('Allowlist of simulators to expose, e.g. ["ltspice", "ngspice"].'))
    sim.add(comment("Empty = auto-detect every supported simulator."))
    sim.add(SIM_ENABLED_KEY, [])
    sim.add(nl())
    sim.add(comment("Explicit path to simulator executable (overrides auto-detection)"))
    sim.add(comment("Leave empty for auto-detection"))
    sim.add(SIM_PATH_KEY, "")
    sim.add(nl())
    sim.add(comment("ngspice compatibility mode (ngbehavior). Unset = spicelib's default"))
    sim.add(comment("'kiltpsa'; its lt (LTspice) and ps (PSPICE) tokens both break sectioned"))
    sim.add(comment("'.lib <file> <section>' PDK corner selection. Set a mode with neither,"))
    sim.add(comment('"hsa" or "kia", for standard-SPICE / PDK decks.'))
    sim.add(comment('ngbehavior = "hsa"'))
    doc.add(SIM_SECTION, sim)
    doc.add(nl())

    # Security section
    sec = table()
    sec.add(comment("Paths accessible to the server (sandbox). Left unset, the default is the"))
    sec.add(comment("working directory plus the Claude Code scratch directory"))
    sec.add(comment("(<tempdir>/claude-<uid>), where an agent writes its throwaway decks."))
    sec.add(comment("Set your own list to replace that default:"))
    sec.add(comment('allowed_paths = ["."]'))
    doc.add("security", sec)
    doc.add(nl())

    # Simulation section
    sim_conf = table()
    sim_conf.add(comment("Maximum number of concurrent simulations."))
    sim_conf.add(comment("Default: number of CPU cores, capped at 8. Uncomment to override."))
    sim_conf.add(comment("max_parallel = 4"))
    sim_conf.add(nl())
    sim_conf.add(comment("Maximum cases after run_experiments variation expansion."))
    sim_conf.add("max_experiment_cases", 1024)
    sim_conf.add(nl())
    sim_conf.add(comment("Default simulation timeout in seconds"))
    sim_conf.add("timeout", 300.0)
    sim_conf.add(nl())
    sim_conf.add(comment("Preflight size guard, estimated from .tran/.ac/.dc directives."))
    sim_conf.add(comment("Warn when the estimated point count exceeds this:"))
    sim_conf.add("max_estimated_points", 20_000_000)
    sim_conf.add(comment("Refuse a run whose estimated raw (MB, single-trace) exceeds this:"))
    sim_conf.add("max_raw_mb", 4096)
    doc.add("simulation", sim_conf)
    doc.add(nl())

    # Analysis section
    analysis = table()
    analysis.add(comment("Maximum waveform data points to return per trace"))
    analysis.add("max_points", 10000)
    analysis.add(comment("Whole-call work budget for analyze_results, in seconds"))
    analysis.add("analysis_budget_s", 60.0)
    analysis.add(comment("Default response budget in tokens for consolidated calls (0 disables)"))
    analysis.add("default_budget", 4000)
    analysis.add(comment("Retention for raw-path-only analysis result sets, in hours"))
    analysis.add("result_set_ttl_hours", 24.0)
    doc.add("analysis", analysis)
    doc.add(nl())

    # Tools section
    tools_tbl = table()
    tools_tbl.add(comment('How much of each tool definition the tool list carries. "full"'))
    tools_tbl.add(comment('(the default) advertises the seven tools as registered; "compact"'))
    tools_tbl.add(comment("advertises the same seven with the per-argument descriptions"))
    tools_tbl.add(comment("removed. No tool gains or loses a capability either way."))
    tools_tbl.add("listing", "full")
    tools_tbl.add(
        comment("run_code = true adds a tool that runs a Python snippet with the engine in")
    )
    tools_tbl.add(comment("scope (for loops over runs and numpy on samples). The snippet has the"))
    tools_tbl.add(comment("server's own file and process authority, not the sandbox above, so"))
    tools_tbl.add(comment("permission it in your client the way you would a shell."))
    tools_tbl.add("run_code", False)
    doc.add("tools", tools_tbl)
    doc.add(nl())

    # Schematic section
    schem = table()
    schem.add(comment("Custom paths to LTspice symbol (.asy) files for .asc schematic support"))
    schem.add(comment("On Windows and WSL these are auto-detected from the LTspice installation"))
    schem.add(comment("Set this to override auto-detection or for non-standard installs"))
    schem.add(comment('Example: symbol_paths = ["/path/to/LTspice/lib/sym"]'))
    schem.add("symbol_paths", [])
    doc.add("schematic", schem)
    doc.add(nl())

    # Logging section
    logging_tbl = table()
    logging_tbl.add(comment("Stderr logging level: DEBUG, INFO, WARNING, ERROR, CRITICAL."))
    logging_tbl.add(comment('Set "INFO" for the startup banner and per-run detail.'))
    logging_tbl.add("level", "WARNING")
    doc.add("logging", logging_tbl)
    doc.add(nl())

    # State section
    state_tbl = table()
    state_tbl.add(
        comment(
            "Persist simulation/batch job metadata to .ltspice-mcp/jobs/ next to each circuit."
        )
    )
    state_tbl.add(
        comment(
            "Lets a restarted server surface prior runs and recent circuits; set to false to disable."
        )
    )
    state_tbl.add("persist_jobs", True)
    state_tbl.add(
        comment(
            "preload_recent_count: at startup, eagerly load persisted jobs for this many "
            "recently-touched circuits. 0 disables preload (lazy-only)."
        )
    )
    state_tbl.add("preload_recent_count", 10)
    doc.add("state", state_tbl)

    atomic_write_text(path, tomlkit.dumps(doc), durable=False)
