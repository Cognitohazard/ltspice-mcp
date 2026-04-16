"""Server configuration with TOML and environment variable support."""

import logging
import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import tomlkit
from tomlkit import comment, document, nl, table

logger = logging.getLogger(__name__)

ToolProfile = Literal["full", "agentic"]
VALID_PROFILES: frozenset[str] = frozenset({"full", "agentic"})
VALID_LOG_LEVELS: frozenset[str] = frozenset(
    {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
)


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
        logger.warning(
            "%s: %s must be %s-%s, got %s; ignoring", source, key, low, max_val, val
        )
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
    """Preferred simulator name (ltspice, ngspice, qspice, xyce). None = auto-select."""

    simulator_exe: Path | None = None
    """Explicit path to simulator executable. Overrides auto-detection."""

    working_dir: Path = field(default_factory=Path.cwd)
    """Working directory for circuit files."""

    allowed_paths: list[Path] = field(default_factory=list)
    """Sandbox paths. Defaults to [working_dir] if empty."""

    max_parallel_sims: int = 4
    """Maximum concurrent simulations."""

    default_timeout: float = 300.0
    """Simulation timeout in seconds."""

    max_points_returned: int = 10000
    """Maximum waveform data points to return."""

    plot_dpi: int = 150
    """Plot resolution in DPI."""

    plot_style: str = "seaborn-v0_8-darkgrid"
    """Matplotlib style."""

    log_level: str = "INFO"
    """Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)."""

    symbol_paths: list[Path] = field(default_factory=list)
    """Custom paths to LTspice symbol (.asy) files for .asc schematic support.
    On Windows and WSL these are auto-detected; set this to override."""

    tool_profile: ToolProfile = "full"
    """Tool profile: "full" exposes all tools, "agentic" exposes a subset
    for LLM agents with native file access (Read/Edit/Write)."""

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

            if "plotting" in toml_data:
                if "dpi" in toml_data["plotting"]:
                    config_dict["plot_dpi"] = toml_data["plotting"]["dpi"]
                if "style" in toml_data["plotting"]:
                    config_dict["plot_style"] = toml_data["plotting"]["style"]

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

            _validate_numeric(
                config_dict, "max_parallel_sims", int, 1, 128, source="config"
            )
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
            _validate_numeric(config_dict, "plot_dpi", int, 50, 600, source="config")

        if env_sim := os.getenv("LTSPICE_MCP_SIMULATOR"):
            config_dict["simulator"] = env_sim

        if env_exe := os.getenv("LTSPICE_MCP_SIMULATOR_EXE"):
            config_dict["simulator_exe"] = Path(env_exe)

        if env_wd := os.getenv("LTSPICE_MCP_WORKING_DIR"):
            config_dict["working_dir"] = Path(env_wd)

        if env_paths := os.getenv("LTSPICE_MCP_ALLOWED_PATHS"):
            config_dict["allowed_paths"] = [Path(p) for p in env_paths.split(os.pathsep)]

        _load_bounded_env(
            "LTSPICE_MCP_MAX_PARALLEL", config_dict, "max_parallel_sims", int, 1, 128
        )
        _load_bounded_env(
            "LTSPICE_MCP_TIMEOUT", config_dict, "default_timeout", float, 0, 86400, exclusive_min=True
        )
        _load_bounded_env(
            "LTSPICE_MCP_MAX_POINTS", config_dict, "max_points_returned", int, 1, 10_000_000
        )
        _load_bounded_env("LTSPICE_MCP_PLOT_DPI", config_dict, "plot_dpi", int, 50, 600)

        if env_style := os.getenv("LTSPICE_MCP_PLOT_STYLE"):
            config_dict["plot_style"] = env_style

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
    sim.add(comment("Explicit path to simulator executable (overrides auto-detection)"))
    sim.add(comment("Leave empty for auto-detection"))
    sim.add("path", "")
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

    # Plotting section
    plotting = table()
    plotting.add(comment("Plot resolution in DPI"))
    plotting.add("dpi", 150)
    plotting.add(nl())
    plotting.add(comment("Matplotlib style (e.g., seaborn-v0_8-darkgrid, ggplot, bmh)"))
    plotting.add("style", "seaborn-v0_8-darkgrid")
    doc.add("plotting", plotting)
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

    path.write_text(tomlkit.dumps(doc))
