# LTspice-MCP Desktop Extension (.mcpb)

A one-click install of the server for **Claude Desktop** (macOS/Windows). On
install, Claude Desktop shows a native folder picker for your circuits
directory — no JSON config to edit.

## What it does and does not bundle

This is a **`type: "uv"` bundle**: `manifest.json` plus a `pyproject.toml` that
declares `ltspice-mcp` as its only dependency. The host runs
`uv run --directory <bundle> server/run.py`, and `uv` installs the published
package and its native dependencies (numpy/scipy) into an on-demand environment
on first use. It does **not** vendor those — they ship as per-platform binary
wheels, so a vendored bundle would be locked to one OS and Python ABI; `uv`
pulls the correct wheels for the host instead.

**It therefore requires, already on the machine:**

- **`uv`** on `PATH` — <https://docs.astral.sh/uv/>. One-line install; tiny.
- **A simulator** — **LTspice** (Windows/macOS) or **ngspice**. The bundle
  cannot ship either (LTspice is a licensed app; ngspice is a native binary).
- **Network access on first run**, so `uvx` can download the package.

Circuit editing works with no simulator; running simulations needs one.

## Build

Requires Node (for the `mcpb` CLI). From this directory:

```bash
npx @anthropic-ai/mcpb validate    # check manifest.json against the schema
npx @anthropic-ai/mcpb pack        # produces ltspice-mcp.mcpb
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
