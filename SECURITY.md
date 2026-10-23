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
contact you privately.

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
importing process's own privileges, and the `allowed_paths` sandbox, when one
is set, applies to both the server and the library. Relevant considerations:

- **Filesystem access** — unless `[security] allowed_paths` (or
  `LTSPICE_MCP_ALLOWED_PATHS`) is set, there is no sandbox, and tool calls can
  read and write any file the server process can. With it set, tool calls can
  read and write files only under those directories, and a path outside them
  is rejected with `PathSecurityError`. The trusted client's own permission
  controls decide which tool calls run. The server uses stdio only;
  exposing it over a network (for example, by wrapping that transport in a
  network proxy) without setting `allowed_paths` and `[tools] run_code =
  false` is out of scope for the default threat model.
- **Simulator subprocesses** — tools spawn LTspice, ngspice, qspice, or xyce
  as child processes and read their output. A maliciously crafted netlist
  can therefore do anything the simulator binary can do on your machine.
  Treat third-party `.asc`, `.cir`, and `.lib` files as untrusted executable
  input.
- **`run_code` runs Python with the server's own authority** — the tool is on
  by default and runs a snippet in a worker process that holds the engine.
  The snippet has the server process's file and process rights, which the
  `allowed_paths` sandbox does not confine. The tool is meant for a local
  server with a single client. Grant it permission in the client as you would
  a shell, and set `[tools] run_code = false` when the server is reachable by
  more than one trusted client, for example through a proxy in front of it.
- **Job-addressed result reads are not checked against a sandbox** — when
  `analyze_results` or a job tool reads a `.raw` or `.log` for a `job_id`, the
  path comes from the server's own job record rather than from the caller, so
  it is not re-checked against `allowed_paths`. Those files are artifacts this
  server produced. A result addressed by path instead (`raw_path`, `log_file`)
  goes through the `allowed_paths` check when a sandbox is set.
- **Simulation artifacts can be stored outside `allowed_paths`** — staged
  decks, `.raw` and `.log` files are normally stored under the working
  directory's `.ltspice-mcp/runs/`; on WSL with LTspice they are written to a
  Windows-native temp directory instead, because LTspice cannot write the
  SQLite file behind `.MEAS` over a `wsl.localhost` share. The machine has one
  such directory, shared by every server process. Each job gets its own
  `runs/{job_id}/` subdirectory under it, so a job's artifacts can be listed
  as a set without guessing at a filename prefix. The root is still shared,
  and any process running as that user can read everything in it. The server
  takes an artifact's provenance from the job record that names it; a file's
  presence in that folder does not establish provenance.
- **Deck staging** — before a run, the deck and every file its
  `.include`/`.lib` directives reach are copied into a staging directory and
  hashed. With no sandbox, every file that exists is staged. With a sandbox, a
  reference is staged only if it resolves inside `[security] allowed_paths` or
  inside the detected simulator's own shipped library directory (LTspice's
  netlister appends a `.lib` into its install for any schematic with a MOSFET
  on it); anything else is refused and the run does not start.
  `allow_live_includes=true` overrides a refusal. The file is then read in
  place at run time, and the response records that the deck's snapshot hash
  does not cover its content. Include recursion is bounded at 8 levels by
  default. A schematic runs as the netlist LTspice exports from it,
  which the server keeps in its store; that export is admitted because the
  schematic is, and its relative references resolve from the schematic's own
  folder, so the store itself need not be inside `allowed_paths`.
- **Where the server writes** — the working directory's `.ltspice-mcp/` store
  (or, with `LTSPICE_MCP_STORE_DIR`, a per-working-directory store under that
  directory, which may be outside `allowed_paths`), the run directory above,
  and a per-user home (`%LOCALAPPDATA%\ltspice-mcp` on Windows, the XDG state
  directory elsewhere, `LTSPICE_MCP_HOME` to override) holding the
  recent-circuits index and one lock file per circuit ever edited, named by a
  digest of its path. Beside a circuit, the server writes only the files a
  caller asks for (an edited sheet, a `verify_circuit` `sidecar` export, a
  `plot_waveform` `out_dir`) and the `<name>.net` that LTspice's own exporter
  writes beside a schematic it is asked to run.
- **`plot_waveform` opens a local window by default** — where the client
  cannot render the chart in-chat, the tool writes an HTML file and, with
  `open` (default `true`), spawns a detached local process to display it in a
  browser or app window. Pass `open=false` to write the file only.
- **The Claude Desktop extension sets a sandbox only when asked** — its
  optional folder picker sets `allowed_paths`, and if the picker is left
  empty, the extension runs with no sandbox.
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
