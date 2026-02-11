"""Server configuration with TOML and environment variable support."""

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

import tomlkit
from tomlkit import comment, document, nl, table


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
        # Start with defaults
        config_dict: dict = {}

        # Load from TOML if it exists
        if config_path is None:
            config_path = Path.cwd() / "ltspice-mcp.toml"

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
                config_dict["log_level"] = toml_data["logging"]["level"]

        # Override with environment variables (highest precedence)
        if env_sim := os.getenv("LTSPICE_MCP_SIMULATOR"):
            config_dict["simulator"] = env_sim

        if env_exe := os.getenv("LTSPICE_MCP_SIMULATOR_EXE"):
            config_dict["simulator_exe"] = Path(env_exe)

        if env_wd := os.getenv("LTSPICE_MCP_WORKING_DIR"):
            config_dict["working_dir"] = Path(env_wd)

        if env_paths := os.getenv("LTSPICE_MCP_ALLOWED_PATHS"):
            config_dict["allowed_paths"] = [Path(p) for p in env_paths.split(":")]

        if env_parallel := os.getenv("LTSPICE_MCP_MAX_PARALLEL"):
            config_dict["max_parallel_sims"] = int(env_parallel)

        if env_timeout := os.getenv("LTSPICE_MCP_TIMEOUT"):
            config_dict["default_timeout"] = float(env_timeout)

        if env_points := os.getenv("LTSPICE_MCP_MAX_POINTS"):
            config_dict["max_points_returned"] = int(env_points)

        if env_dpi := os.getenv("LTSPICE_MCP_PLOT_DPI"):
            config_dict["plot_dpi"] = int(env_dpi)

        if env_style := os.getenv("LTSPICE_MCP_PLOT_STYLE"):
            config_dict["plot_style"] = env_style

        if env_log := os.getenv("LTSPICE_MCP_LOG_LEVEL"):
            config_dict["log_level"] = env_log

        return cls(**config_dict)


def generate_default_config(path: Path) -> None:
    """Generate a self-documenting default configuration file.

    Args:
        path: Path where the TOML config file should be written.
    """
    doc = document()

    # Simulator section
    doc.add(comment("LTSpice MCP Server Configuration"))
    doc.add(comment("All settings have sensible defaults and can be overridden with environment variables"))
    doc.add(nl())

    sim = table()
    sim.add(comment("Preferred simulator: ltspice, ngspice, qspice, xyce"))
    sim.add(comment("Leave empty or set to null for auto-detection (prefers LTSpice if available)"))
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
    sec.add(comment("Default: [\".\"] (current working directory)"))
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

    # Logging section
    logging = table()
    logging.add(comment("Logging level: DEBUG, INFO, WARNING, ERROR, CRITICAL"))
    logging.add("level", "INFO")
    doc.add("logging", logging)

    # Write to file
    path.write_text(tomlkit.dumps(doc))
