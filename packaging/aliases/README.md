# Alias packages

`circuit-mcp`, `ngspice-mcp` and `osic-mcp` are aliases for the canonical
`ltspice-mcp` package, so users can find and install the server under more
than one name. Each depends on `ltspice-mcp` and provides a console script, so
`uvx circuit-mcp`, `uvx ngspice-mcp` and `uvx osic-mcp` run the same server as
`uvx ltspice-mcp`. `osic-mcp` (open-source IC) is the name the project may
move to as it extends beyond LTspice.

Each alias derives its version from the same git tag as the canonical release,
through `hatch-vcs`, so a `v0.3.0` tag ships `ltspice-mcp`, `circuit-mcp`,
`ngspice-mcp` and `osic-mcp` all at `0.3.0`.

## Publishing

The **Publish aliases to PyPI** workflow
(`.github/workflows/publish-aliases.yml`) publishes all aliases. It runs
automatically on every `v*` release tag (alongside `publish.yml`). It uses
PyPI Trusted Publishing (the same OIDC mechanism and `pypi` environment as the
canonical release; no tokens) and `skip-existing`, so a re-run, or an alias
whose version is already on PyPI, is skipped without an error.

Each alias is a separate PyPI project, so each needs its own trusted publisher.
Before the first run, add a **pending publisher** on PyPI for each project name
(PyPI account → Publishing → add a pending publisher) with these fields:

| Field | Value |
|-|-|
| PyPI Project Name | `circuit-mcp` / `ngspice-mcp` / `osic-mcp` |
| Owner | `cognitohazard` |
| Repository name | `ltspice-mcp` |
| Workflow name | `publish-aliases.yml` |
| Environment name | `pypi` |

"Workflow name" is the file name (`publish-aliases.yml`), not the YAML `name:`.
The "Environment name" must match the workflow's `environment:` (`pypi`); leaving
it blank on PyPI (shown as *Any*) also works, since blank imposes no constraint.

A pending publisher does not reserve the name, because the first upload
claims a name on PyPI. A new alias is reserved only once a release tag that
contains its folder has been published.

If an alias's trusted publisher is registered only after a release, use
manual dispatch (Actions tab → Run workflow). Run the workflow against that
release's tag and set the `alias` input to the single package (default
`all`). Run it against a tag ref, because the workflow takes the version from
the ref name and stops with an error when the ref is not a release tag
(`v` followed by a version such as `0.3.0`).

In the repository, the `ltspice-mcp` dependency is a `>=` lower bound used
only for local development. At release, `publish-aliases.yml` rewrites it to an
exact `ltspice-mcp==<tag>` pin and waits for that version to be installable
from PyPI, so a published alias always installs the co-released canonical
version and cannot resolve an older one.
