# Security Policy

## Supported versions

Only the latest release line on PyPI receives security fixes. This project is
pre-1.0 and ships from `master`; patches are released as new minor or patch
versions tagged `vX.Y.Z`.

## Reporting a vulnerability

**Please do not open a public issue for security-sensitive reports.**

Use GitHub's private [Security Advisory](https://github.com/cognitohazard/ltspice-mcp/security/advisories/new)
flow to report a vulnerability. If you cannot use GitHub advisories, open an
empty public issue titled `Security contact request` and a maintainer will
reach out off-channel.

When reporting, please include:

- A description of the vulnerability and its impact
- The affected version or commit SHA
- A minimal reproduction (netlist, tool arguments, or config excerpt)
- Any suggested mitigation, if known

You should expect an acknowledgement within **7 days**. Once a fix is
available, we will coordinate a disclosure window with the reporter before
publishing a release and advisory.

## Threat model

`ltspice-mcp` is designed to run as a **local MCP server** used by a trusted
MCP client (Claude Desktop, Claude Code, etc.) **and as an importable Python
library** (`ltspice_mcp.api`) inside a trusted local process. It is not
hardened as a public, multi-tenant network service. The library runs with the
importing process's own privileges, and the same `allowed_paths` sandbox
applies to both the server and the library. Relevant considerations:

- **Filesystem access** — tool calls can read and write files under the
  configured `[security] allowed_paths`. Paths outside that sandbox are
  rejected via `PathSecurityError`. The server uses stdio only; exposing it
  over a network (for example, by wrapping that transport in a network proxy)
  without restricting `allowed_paths` is out of scope for the default threat
  model.
- **Simulator subprocesses** — tools spawn LTspice, ngspice, qspice, or xyce
  as child processes and read their output. A maliciously crafted netlist
  can therefore do anything the simulator binary can do on your machine.
  Treat third-party `.asc`, `.cir`, and `.lib` files the same way you would
  treat any untrusted executable input.
- **Job-addressed result reads bypass the sandbox by design** — when
  `analyze_results` or a job tool reads a `.raw` or `.log` for a `job_id`, the
  path comes from the server's own job record rather than from the caller, so
  it is not re-checked against `allowed_paths`. Those files are artifacts this
  server produced. A result addressed by path instead (`raw_path`, `log_file`)
  goes through the normal `allowed_paths` check.
- **Simulation artifacts can sit outside `allowed_paths`** — normally staged
  decks, `.raw` and `.log` files live under the working directory's
  `.ltspice-mcp/runs/`; on WSL with LTspice they are written to a
  Windows-native temp directory instead, because LTspice cannot write the
  SQLite file behind `.MEAS` over a `wsl.localhost` share. That directory is
  one per machine, not one per server process. Each job gets its own
  `runs/{job_id}/` subdirectory under it, so a job's artifacts are enumerable
  as a set rather than by guessing at a filename prefix — but the root is
  shared, and everything in it is readable by anything running as that user.
  An artifact's provenance comes from the job record that names it, never
  from its presence in that folder.
- **Deck staging trust roots** — before a run, the deck and every file its
  `.include`/`.lib` directives reach are copied into a staging directory and
  hashed. A reference is staged only if it resolves inside
  `[security] allowed_paths` or inside the detected simulator's own shipped
  library directory (LTspice's netlister appends a `.lib` into its install for
  any schematic with a MOSFET on it). Anything else is refused and the run does
  not start. `allow_live_includes=true` overrides that refusal: the file is
  then read in place at run time, and the response records that its content is
  not covered by the deck's snapshot hash. Include recursion is bounded at 8
  levels by default.
- **`plot_waveform` opens a local window by default** — where the client
  cannot render the chart in-chat, the tool writes an HTML file and, with
  `open` (default `true`), spawns a detached local process to display it in a
  browser or app window. Pass `open=false` to write the file only.
- **The Claude Desktop extension defaults to your Documents folder** — the
  `.mcpb` folder picker sets `allowed_paths`, and its default is
  `${HOME}/Documents`. Point it at your circuits directory for a narrower
  sandbox.
- **Dependency supply chain** — runtime dependencies are pinned in
  `uv.lock`. The `publish.yml` workflow builds and publishes to PyPI from
  version tags only, via PyPI Trusted Publishing (OIDC).

## Known vulnerabilities

Run a local audit against the pinned lockfile at any time:

```bash
uv export --format requirements-txt --no-hashes --no-dev --no-emit-project \
    --output-file /tmp/reqs.txt
uv run --with pip-audit python -m pip_audit -r /tmp/reqs.txt --no-deps --strict
```
