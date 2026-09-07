# LTspice-MCP Desktop Extension (.mcpb)

This extension installs the server in **Claude Desktop** (macOS/Windows). On
install, Claude Desktop shows a native folder picker for your circuits
directory; there is no JSON config to edit.

## What it does and does not bundle

This is a **`type: "uv"` bundle**: `manifest.json` plus a `pyproject.toml` that
declares `ltspice-mcp` as its only dependency. The host runs
`uv run --directory <bundle> server/run.py`, and `uv` installs the published
package and its native dependencies (numpy/scipy) into an on-demand environment
on first use. It does **not** vendor those. They ship as per-platform binary wheels, so a
vendored bundle would be locked to one OS and Python ABI; `uv` pulls the
correct wheels for the host instead.

**Requirements on the host machine:**

- **`uv`** on `PATH`. See <https://docs.astral.sh/uv/> for installation.
- **A simulator** — **LTspice** (Windows/macOS) or **ngspice**. The bundle
  cannot ship either (LTspice is a licensed app; ngspice is a native binary).
- **Network access on first run**, so `uv` can download the package.

Netlist editing (`.cir`/`.net`) works with no simulator at all. Editing an
`.asc` schematic needs LTspice's `.asy` symbol libraries on disk — the
geometry comes from those symbol files, so an install of LTspice is what
supplies them (auto-detected on Windows and macOS). Running simulations
needs a simulator.

## Build

Requires Node (for the `mcpb` CLI). From this directory:

```bash
npx @anthropic-ai/mcpb validate manifest.json       # check it against the schema
npx @anthropic-ai/mcpb pack . ltspice-mcp.mcpb      # zip this directory into the bundle
```

`pack` zips this directory (`manifest.json` + `pyproject.toml` + `server/`).

## Install and test

1. Drag `ltspice-mcp.mcpb` onto Claude Desktop (Settings → Extensions).
2. When prompted, pick your circuits directory; leave the simulator path blank
   to auto-detect.
3. In a chat, confirm the `ltspice` tools appear and run a trivial request
   (e.g. "validate this netlist": a two-resistor divider).

If the server fails to start, the usual cause is `uv` not being on the
GUI app's `PATH` (notably on macOS, where launched apps don't inherit your
shell `PATH`). Fixes: install `uv` system-wide, or set the simulator/uv paths
explicitly. Report what you see and we'll adjust the manifest.
