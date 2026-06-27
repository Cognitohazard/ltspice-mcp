# Alias packages

Thin redirect packages so the server is discoverable and installable under more
than one name. Each depends on the canonical `ltspice-mcp` and exposes a console
script that runs the same server, so `uvx circuit-mcp` / `uvx ngspice-mcp` work
as drop-in equivalents of `uvx ltspice-mcp`.

Their version tracks the canonical release: each derives its version from the
same git tag via `hatch-vcs`, so a `v0.3.0` tag ships `ltspice-mcp`,
`circuit-mcp`, and `ngspice-mcp` all at `0.3.0`.

## Publishing

Published by the **Publish aliases to PyPI** workflow
(`.github/workflows/publish-aliases.yml`), which runs automatically on every
`v*` release tag (alongside `publish.yml`) and publishes all aliases. It uses
PyPI Trusted Publishing — the same OIDC mechanism and `pypi` environment as the
canonical release, no tokens — and `skip-existing`, so re-runs (and an alias
whose version is already on PyPI) are no-ops rather than errors.

Each alias is a separate PyPI project, so each needs its own trusted publisher.
Before the first run, add a **pending publisher** on PyPI for each project name
(PyPI account → Publishing → add a pending publisher) with these fields:

| Field | Value |
|-|-|
| PyPI Project Name | `circuit-mcp` / `ngspice-mcp` |
| Owner | `cognitohazard` |
| Repository name | `ltspice-mcp` |
| Workflow name | `publish-aliases.yml` |
| Environment name | `pypi` |

"Workflow name" is the file name (`publish-aliases.yml`), not the YAML `name:`.
The "Environment name" must match the workflow's `environment:` (`pypi`); leaving
it blank on PyPI (shown as *Any*) also works, since blank imposes no constraint.

Manual dispatch (Actions tab → Run workflow) is for **catch-up**: if an alias's
trusted publisher is registered only after a release, dispatch the workflow
against that release's tag and set the `alias` input to the single package
(default `all`). Run it against a tag ref, not a branch — `hatch-vcs` only
produces a clean, PyPI-acceptable version on an exact tag (off-tag builds carry
a `.devN+g<sha>` local segment that PyPI rejects).

The `ltspice-mcp` dependency is intentionally unpinned, so a fresh install of an
alias always pulls the latest canonical release at runtime regardless of the
alias's own version.
