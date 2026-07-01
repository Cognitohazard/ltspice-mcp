"""Server configuration with TOML and environment variable support."""

import logging
import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import tomlkit
from tomlkit import comment, document, nl, table

from ltspice_mcp.lib import atomic_write_text

logger = logging.getLogger(__name__)

ToolProfile = Literal["full", "agentic"]
VALID_PROFILES: frozenset[str] = frozenset({"full", "agentic"})
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


def _validated_profile(value: str, source: str) -> str | None:
    """Return value if it's a valid profile, else warn and return None."""
    if value in VALID_PROFILES:
        return value
    logger.warning("Unknown tool profile %r in %s, using 'full'", value, source)
    return None


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

    default_timeout: float = 300.0
    """Simulation timeout in seconds."""

    max_points_returned: int = 10000
    """Maximum waveform data points to return."""

    log_level: str = "INFO"
    """Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)."""

    symbol_paths: list[Path] = field(default_factory=list)
    """Custom paths to LTspice symbol (.asy) files for .asc schematic support.
    On Windows and WSL these are auto-detected; set this to override."""

    tool_profile: ToolProfile = "full"
    """Tool profile: "full" exposes all tools, "agentic" exposes a subset
    for LLM agents with native file access (Read/Edit/Write)."""

    persist_jobs: bool = True
    """Persist simulation/batch job metadata to ``.ltspice-mcp/jobs/`` next
    to each circuit file so a restarted server can surface prior runs."""

    preload_recent_count: int = 10
    """At startup, eagerly load persisted jobs for this many recently-touched
    circuits (capped by ``recent.json``). Set to 0 to disable preload and
    fall back to lazy loading on first tool call. Bounded-IO; typical
    cost is a handful of millisecond-scale JSON reads."""

    config_path: Path = field(default_factory=lambda: Path.cwd() / "ltspice-mcp.toml")
    """Path that was resolved for the config file (set by load())."""

    def __post_init__(self) -> None:
        """Ensure allowed_paths defaults to [working_dir] if not set."""
        if not self.allowed_paths:
            self.allowed_paths = [self.working_dir]

    @classmethod
    def load(cls, config_path: Path | None = None) -> "ServerConfig":
        """Load configuration from defaults, TOML file, and environment variables.

        Args:
            config_path: Path to TOML config file. If None, looks for ltspice-mcp.toml
                        in the current working directory.

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

            # Map TOML structure to config fields
            if "simulator" in toml_data:
                if "default" in toml_data["simulator"]:
                    config_dict["simulator"] = toml_data["simulator"]["default"] or None
                if "path" in toml_data["simulator"] and toml_data["simulator"]["path"]:
                    config_dict["simulator_exe"] = Path(toml_data["simulator"]["path"])
                if "enabled" in toml_data["simulator"]:
                    raw = toml_data["simulator"]["enabled"]
                    if isinstance(raw, list) and all(isinstance(x, str) for x in raw):
                        config_dict["enabled_simulators"] = [x.strip().lower() for x in raw]
                    else:
                        logger.warning(
                            "config: simulator.enabled must be a list of strings; ignoring %r",
                            raw,
                        )
                if "ngbehavior" in toml_data["simulator"]:
                    raw = toml_data["simulator"]["ngbehavior"]
                    if isinstance(raw, str):
                        if raw.strip():
                            config_dict["ngbehavior"] = raw.strip()
                    else:
                        logger.warning(
                            "config: simulator.ngbehavior must be a string; ignoring %r", raw
                        )

            if "security" in toml_data and "allowed_paths" in toml_data["security"]:
                config_dict["allowed_paths"] = [
                    Path(p) for p in toml_data["security"]["allowed_paths"]
                ]

            if "simulation" in toml_data:
                if "max_parallel" in toml_data["simulation"]:
                    config_dict["max_parallel_sims"] = toml_data["simulation"]["max_parallel"]
                if "timeout" in toml_data["simulation"]:
                    config_dict["default_timeout"] = toml_data["simulation"]["timeout"]

            if "analysis" in toml_data and "max_points" in toml_data["analysis"]:
                config_dict["max_points_returned"] = toml_data["analysis"]["max_points"]

            if "logging" in toml_data and "level" in toml_data["logging"]:
                level = str(toml_data["logging"]["level"]).upper()
                if level in VALID_LOG_LEVELS:
                    config_dict["log_level"] = level
                else:
                    logger.warning(
                        "config: invalid log level %r; must be one of %s",
                        toml_data["logging"]["level"],
                        sorted(VALID_LOG_LEVELS),
                    )

            if "schematic" in toml_data and "symbol_paths" in toml_data["schematic"]:
                config_dict["symbol_paths"] = [
                    Path(p) for p in toml_data["schematic"]["symbol_paths"]
                ]

            if (
                "tools" in toml_data
                and "profile" in toml_data["tools"]
                and (p := _validated_profile(toml_data["tools"]["profile"], "config"))
            ):
                config_dict["tool_profile"] = p

            if "state" in toml_data and "persist_jobs" in toml_data["state"]:
                raw = toml_data["state"]["persist_jobs"]
                if isinstance(raw, bool):
                    config_dict["persist_jobs"] = raw
                else:
                    logger.warning("config: state.persist_jobs must be boolean; ignoring %r", raw)

            if "state" in toml_data and "preload_recent_count" in toml_data["state"]:
                raw = toml_data["state"]["preload_recent_count"]
                if isinstance(raw, int) and raw >= 0:
                    config_dict["preload_recent_count"] = raw
                else:
                    logger.warning(
                        "config: state.preload_recent_count must be a non-negative "
                        "integer; ignoring %r",
                        raw,
                    )

            _validate_numeric(config_dict, "max_parallel_sims", int, 1, 128, source="config")
            _validate_numeric(
                config_dict,
                "default_timeout",
                float,
                0,
                86400,
                exclusive_min=True,
                source="config",
            )
            _validate_numeric(
                config_dict,
                "max_points_returned",
                int,
                1,
                10_000_000,
                source="config",
            )

        if env_sim := os.getenv("LTSPICE_MCP_SIMULATOR"):
            config_dict["simulator"] = env_sim

        if env_enabled := os.getenv("LTSPICE_MCP_ENABLED_SIMULATORS"):
            # Comma- or os.pathsep-separated list of simulator names.
            sep = "," if "," in env_enabled else os.pathsep
            config_dict["enabled_simulators"] = [
                x.strip().lower() for x in env_enabled.split(sep) if x.strip()
            ]

        if env_exe := os.getenv("LTSPICE_MCP_SIMULATOR_EXE"):
            config_dict["simulator_exe"] = Path(env_exe)

        if (env_ngb := os.getenv("LTSPICE_MCP_NGBEHAVIOR")) and env_ngb.strip():
            config_dict["ngbehavior"] = env_ngb.strip()

        if env_wd := os.getenv("LTSPICE_MCP_WORKING_DIR"):
            config_dict["working_dir"] = Path(env_wd)

        if env_paths := os.getenv("LTSPICE_MCP_ALLOWED_PATHS"):
            config_dict["allowed_paths"] = [Path(p) for p in env_paths.split(os.pathsep)]

        _load_bounded_env(
            "LTSPICE_MCP_MAX_PARALLEL", config_dict, "max_parallel_sims", int, 1, 128
        )
        _load_bounded_env(
            "LTSPICE_MCP_TIMEOUT",
            config_dict,
            "default_timeout",
            float,
            0,
            86400,
            exclusive_min=True,
        )
        _load_bounded_env(
            "LTSPICE_MCP_MAX_POINTS", config_dict, "max_points_returned", int, 1, 10_000_000
        )
        if env_log := os.getenv("LTSPICE_MCP_LOG_LEVEL"):
            env_log_upper = env_log.upper()
            if env_log_upper in VALID_LOG_LEVELS:
                config_dict["log_level"] = env_log_upper
            else:
                logger.warning(
                    "LTSPICE_MCP_LOG_LEVEL: invalid value %r; must be one of %s",
                    env_log,
                    sorted(VALID_LOG_LEVELS),
                )

        if env_sym := os.getenv("LTSPICE_MCP_SYMBOL_PATHS"):
            config_dict["symbol_paths"] = [Path(p) for p in env_sym.split(os.pathsep)]

        if (env_profile := os.getenv("LTSPICE_MCP_TOOL_PROFILE")) and (
            p := _validated_profile(env_profile, "LTSPICE_MCP_TOOL_PROFILE")
        ):
            config_dict["tool_profile"] = p

        if (env_persist := os.getenv("LTSPICE_MCP_PERSIST_JOBS")) is not None:
            normalized = env_persist.strip().lower()
            if normalized in ("1", "true", "yes", "on"):
                config_dict["persist_jobs"] = True
            elif normalized in ("0", "false", "no", "off"):
                config_dict["persist_jobs"] = False
            else:
                logger.warning(
                    "LTSPICE_MCP_PERSIST_JOBS: invalid boolean %r; ignoring", env_persist
                )

        if (env_preload := os.getenv("LTSPICE_MCP_PRELOAD_RECENT_COUNT")) is not None:
            try:
                parsed = int(env_preload)
                if parsed < 0:
                    raise ValueError("must be >= 0")
                config_dict["preload_recent_count"] = parsed
            except ValueError as e:
                logger.warning(
                    "LTSPICE_MCP_PRELOAD_RECENT_COUNT: invalid integer %r (%s); ignoring",
                    env_preload,
                    e,
                )

        config_dict["config_path"] = config_path
        return cls(**config_dict)


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
    sim.add("enabled", [])
    sim.add(nl())
    sim.add(comment("Explicit path to simulator executable (overrides auto-detection)"))
    sim.add(comment("Leave empty for auto-detection"))
    sim.add("path", "")
    sim.add(nl())
    sim.add(comment("ngspice compatibility mode (ngbehavior). Unset = spicelib's default"))
    sim.add(comment("'kiltpsa'; its lt (LTspice) and ps (PSPICE) tokens both break sectioned"))
    sim.add(comment("'.lib <file> <section>' PDK corner selection. Set a mode with neither,"))
    sim.add(comment('"hsa" or "kia", for standard-SPICE / PDK decks.'))
    sim.add(comment('ngbehavior = "hsa"'))
    doc.add("simulator", sim)
    doc.add(nl())

    # Security section
    sec = table()
    sec.add(comment("Paths accessible to the server (sandbox)"))
    sec.add(comment('Default: ["."] (current working directory)'))
    sec.add("allowed_paths", ["."])
    doc.add("security", sec)
    doc.add(nl())

    # Simulation section
    sim_conf = table()
    sim_conf.add(comment("Maximum number of concurrent simulations"))
    sim_conf.add("max_parallel", 4)
    sim_conf.add(nl())
    sim_conf.add(comment("Default simulation timeout in seconds"))
    sim_conf.add("timeout", 300.0)
    doc.add("simulation", sim_conf)
    doc.add(nl())

    # Analysis section
    analysis = table()
    analysis.add(comment("Maximum waveform data points to return per trace"))
    analysis.add("max_points", 10000)
    doc.add("analysis", analysis)
    doc.add(nl())

    # Tools section
    tools_tbl = table()
    tools_tbl.add(comment('Tool profile: "full" (all tools) or "agentic" (subset for LLM agents)'))
    tools_tbl.add(
        comment('"agentic" removes netlist-editing tools that capable agents handle natively')
    )
    tools_tbl.add("profile", "full")
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
    logging_tbl.add(comment("Logging level: DEBUG, INFO, WARNING, ERROR, CRITICAL"))
    logging_tbl.add("level", "INFO")
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
