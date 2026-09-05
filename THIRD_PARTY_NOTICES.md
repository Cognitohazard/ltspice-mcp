# Third-party notices

`ltspice-mcp` itself is licensed GPL-3.0-or-later (see `LICENSE`). It ships
some third-party code inside the package, and depends on third-party packages
installed alongside it. Those are listed here.

Each vendored component keeps its own full licence text next to its files; the
paths below point at it.

---

## Vendored in the distributed package

### uPlot 1.6.32

- **Files:** `src/ltspice_mcp/assets/uplot/uPlot.iife.min.js`,
  `src/ltspice_mcp/assets/uplot/uPlot.min.css`
- **Upstream:** https://github.com/leeoniya/uPlot
- **Licence:** MIT — full text at `src/ltspice_mcp/assets/uplot/LICENSE`
- **Copyright:** Copyright (c) 2022 Leon Sorokin
- **Used for:** the interactive waveform chart. It is inlined into the
  generated plot HTML and into the MCP Apps widget resource, so a plot renders
  with no network access and no CDN.

The version is the one stamped in the vendored file's own header banner
(`/*! https://github.com/leeoniya/uPlot (v1.6.32) */`).

### Model Context Protocol `ext-apps` browser runtime

- **File:** `src/ltspice_mcp/assets/ext-apps/app-with-deps.js`
- **Upstream:** the Model Context Protocol project
  (https://github.com/modelcontextprotocol)
- **Licence file:** `src/ltspice_mcp/assets/ext-apps/LICENSE`
- **Copyright:** Copyright (c) 2024-2025 Model Context Protocol a Series of
  LF Projects, LLC.
- **Used for:** the host-side bridge in the MCP Apps waveform widget. The
  widget template receives its per-call chart spec through this runtime's
  `app.ontoolresult` hook.

That LICENSE file states the project's licensing position in full; quoting its
opening verbatim:

> The MCP project is undergoing a licensing transition from the MIT License to
> the Apache License, Version 2.0 ("Apache-2.0"). All new code and
> specification contributions to the project are licensed under Apache-2.0.
> Documentation contributions (excluding specifications) are licensed under
> CC-BY-4.0.
>
> Contributions for which relicensing consent has been obtained are licensed
> under Apache-2.0. Contributions made by authors who originally licensed their
> work under the MIT License and who have not yet granted explicit permission
> to relicense remain licensed under the MIT License.
>
> No rights beyond those granted by the applicable original license are
> conveyed for such contributions.

The file then carries the full Apache-2.0 text, the MIT text with the copyright
line above, and a pointer to CC-BY-4.0 for documentation.

The bundle is a pre-built minified artifact and carries no version string of
its own, so none is recorded here. It was vendored on 2026-06-14; the MCP
protocol revision it negotiates is `2025-11-21`.

---

## Runtime dependencies

Installed by the package manager rather than vendored, and each subject to its
own licence. Version ranges are in `pyproject.toml`; exact resolved versions
are in `uv.lock`.

**spicelib** (`>=1.4.9,<1.6`) — GPL-3.0, per its distribution metadata. The
simulation and netlist-editing backend: it drives the simulator, parses `.raw`
and `.log` output, and provides the `.asc` / `.cir` editors this project builds
on. Known defects in the pinned range, with reproductions and the workarounds
we carry for them, are catalogued in `docs/spicelib_bugs.md`.

The other runtime dependencies are not redistributed inside this package.
Their licences, as declared in the installed distribution metadata:

| package | declared licence |
|-|-|
| `anyio` | MIT |
| `mcp` | MIT |
| `numpy` | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 |
| `psutil` | BSD-3-Clause |
| `pydantic` | MIT |
| `rapidfuzz` | MIT |
| `scipy` | BSD (per its trove classifier) |
| `tomlkit` | MIT |
| `cairosvg` (optional `raster` extra) | LGPL-3.0-or-later |

Each package's own metadata is the authoritative text; the table above is a
convenience and can go stale as versions move.

---

## Not bundled

LTspice, ngspice, QSPICE and Xyce are external programs this project invokes.
They are not distributed with it and are governed entirely by their own
licences.
