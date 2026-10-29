# LTspice-MCP Desktop Extension (.mcpb)

This extension installs the server in **Claude Desktop** (macOS/Windows). On
install, Claude Desktop asks for two optional settings, a sandbox folder and
the simulator path; there is no JSON config to edit.

## Contents and host requirements

This is a **`type: "uv"` bundle**, with a `manifest.json` and a
`pyproject.toml` that declares `ltspice-mcp` as its only dependency. The host
runs `uv run --directory <bundle> server/run.py`, and `uv` installs the
published package and its native dependencies (numpy/scipy) into an on-demand
environment on first use. The bundle does **not** vendor those, because they
ship as per-platform binary wheels and a vendored bundle would be locked to
one OS and Python ABI. `uv` downloads the correct wheels for the host instead.

**Requirements on the host machine:**

- **`uv`** on `PATH`. See <https://docs.astral.sh/uv/> for installation.
- **A simulator** — **LTspice** (Windows/macOS) or **ngspice**. The bundle
  cannot ship either (LTspice is a licensed app; ngspice is a native binary).
- **Network access on first run**, so `uv` can download the package.

Reading and checking netlists (`.cir`/`.net`) works without a simulator.
Editing an `.asc` schematic needs LTspice's `.asy` symbol libraries on disk,
because the geometry comes from those symbol files. An LTspice install
supplies them, and the server finds them automatically on Windows. Running
simulations needs a simulator.

**PNG schematic renders.** The bundle installs the `raster` extra. Its
rasterizer, `resvg-py`, ships as a prebuilt wheel and needs no native
library, so PNG renders work without further setup.
`inspect(kind="capabilities")` reports `render.png`, and when it is false it
names what is missing and how to install it. See
[PNG rendering](../../docs/USAGE.md#png-rendering).

## Build

Requires Node (for the `mcpb` CLI). From this directory:

```bash
npx @anthropic-ai/mcpb validate manifest.json       # check it against the schema
npx @anthropic-ai/mcpb pack . ltspice-mcp.mcpb      # zip this directory into the bundle
```

`pack` zips this directory (`manifest.json` + `pyproject.toml` + `server/`).

## Install and test

1. Drag `ltspice-mcp.mcpb` onto Claude Desktop (Settings → Extensions).
2. When prompted, leave the sandbox folder blank for no sandbox, or pick the
   folder the tools should be kept inside; leave the simulator path blank to
   auto-detect.
3. In a chat, confirm the `ltspice-mcp` tools appear and run a trivial
   request (e.g. "validate this netlist": a two-resistor divider).

If the server fails to start, the usual cause is `uv` not being on the
GUI app's `PATH` (notably on macOS, where launched apps don't inherit your
shell `PATH`). To fix it, install `uv` system-wide; the extension runs the
command `uv` and has no setting for its path. If the server starts but finds
no simulator, set the simulator path in the extension's settings. Report what
you see and we'll adjust the manifest.
