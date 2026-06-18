"""Invariants for the Claude Desktop extension manifest (packaging/mcpb).

Guards the MCPB ``user_config`` -> ``env`` substitution. The MCPB resolver
only replaces a ``${user_config.KEY}`` placeholder when KEY is in the merged
config, which holds only if the entry is ``required`` (Claude Desktop blocks
install when it is missing) or carries a ``default``. An optional key with no
default leaks the literal ``${user_config.KEY}`` into the environment; the
server then treats a non-empty ``LTSPICE_MCP_SIMULATOR_EXE`` as a real
executable path and silently skips simulator auto-detection.
"""

import json
import re
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MCPB_DIR = REPO_ROOT / "packaging" / "mcpb"
MANIFEST = MCPB_DIR / "manifest.json"

_PLACEHOLDER = re.compile(r"\$\{user_config\.([^}]+)\}")


def _manifest() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def test_manifest_parses_and_names_the_server() -> None:
    assert _manifest()["name"] == "ltspice-mcp"


def test_env_user_config_placeholders_always_resolve() -> None:
    manifest = _manifest()
    user_config = manifest.get("user_config", {})
    env = manifest["server"]["mcp_config"].get("env", {})

    referenced = {key for value in env.values() for key in _PLACEHOLDER.findall(value)}
    assert referenced, "expected the bundle env to reference user_config keys"

    for key in referenced:
        assert key in user_config, f"env references undefined user_config key {key!r}"
        option = user_config[key]
        assert option.get("required") or "default" in option, (
            f"user_config {key!r} is optional with no default; the MCPB resolver "
            f"would leak the literal ${{user_config.{key}}} into the environment"
        )


def test_uv_bundle_declares_its_dependency() -> None:
    """A ``type: "uv"`` bundle resolves deps from a root pyproject.toml. Type
    ``"python"`` would instead require vendored deps the bundle does not ship
    (numpy/scipy are per-platform wheels), so the type and the dep manifest
    must stay in lockstep — otherwise a conformant host fails before handshake.
    """
    server = _manifest()["server"]
    assert server["type"] == "uv"
    # Host launches via `uv run --directory <bundle> <entry_point>`.
    assert server["mcp_config"]["command"] == "uv"
    assert "run" in server["mcp_config"]["args"]

    pyproject = MCPB_DIR / "pyproject.toml"
    assert pyproject.is_file(), "type='uv' bundle must ship a pyproject.toml"
    deps = tomllib.loads(pyproject.read_text("utf-8"))["project"]["dependencies"]
    assert any(d == "ltspice-mcp" or d.startswith("ltspice-mcp") for d in deps)


def test_bundle_versions_agree() -> None:
    """The plugin and Desktop-extension version literals must move together.

    None of these derive from the VCS-tagged package version (``uvx`` always
    pulls the latest published release), so they are hand-maintained across
    four spots. Pin them to one value so a release bump cannot silently leave
    one behind.
    """
    plugin = json.loads((REPO_ROOT / ".claude-plugin" / "plugin.json").read_text("utf-8"))
    market = json.loads((REPO_ROOT / ".claude-plugin" / "marketplace.json").read_text("utf-8"))
    versions = {
        "plugin.json": plugin["version"],
        "marketplace metadata": market["metadata"]["version"],
        "marketplace plugin entry": market["plugins"][0]["version"],
        "mcpb manifest": _manifest()["version"],
    }
    assert len(set(versions.values())) == 1, f"bundle versions disagree: {versions}"
