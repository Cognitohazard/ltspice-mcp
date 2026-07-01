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

`ltspice-mcp` is designed to run as a **local MCP server** spoken to by a
trusted MCP client (Claude Desktop, Claude Code, etc.). It is not hardened as
a public, multi-tenant network service. Relevant considerations:

- **Filesystem access** — tool calls can read and write files under the
  configured `[security] allowed_paths`. Paths outside that sandbox are
  rejected via `PathSecurityError`. The server only speaks stdio; exposing it
  over a network (for example, by wrapping that transport in a network proxy)
  without restricting `allowed_paths` is out of scope for the default threat
  model.
- **Simulator subprocesses** — tools spawn LTspice, ngspice, qspice, or xyce
  as child processes and read their output. A maliciously crafted netlist
  can therefore do anything the simulator binary can do on your machine.
  Treat third-party `.asc`, `.cir`, and `.lib` files the same way you would
  treat any untrusted executable input.
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
