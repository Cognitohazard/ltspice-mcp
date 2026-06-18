# Alias packages

Thin redirect packages so the server is discoverable and installable under more
than one name. Each depends on the canonical `ltspice-mcp` and exposes a console
script that runs the same server, so `uvx circuit-mcp` / `uvx ngspice-mcp` work
as drop-in equivalents of `uvx ltspice-mcp`.

These are intentionally **not** part of the tag-triggered canonical release
(`publish.yml`, which publishes `ltspice-mcp`). They are publish-once and change
rarely.

## Publishing

Published via the manually-triggered **Publish aliases to PyPI** workflow
(`.github/workflows/publish-aliases.yml`) from the Actions tab. It uses PyPI
Trusted Publishing — the same OIDC mechanism and `pypi` environment as the
canonical release, no tokens — and `skip-existing`, so re-runs are idempotent.

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

To ship a new version of an alias later, bump its `version` in its
`pyproject.toml` and re-run the workflow (`skip-existing` skips the unchanged
one). The `ltspice-mcp` dependency is intentionally unpinned, so a fresh install
of an alias always pulls the latest canonical release without re-publishing it.
