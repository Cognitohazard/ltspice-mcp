# Contributing

Thanks for looking. This is a small project; bug reports with a reproduction
are as welcome as patches.

By participating you agree to the [Code of Conduct](CODE_OF_CONDUCT.md).
Contributions are licensed under GPL-3.0-or-later, matching the project.

## Setting up

```bash
uv sync              # dev tools and dependencies come from pyproject.toml
uv run ltspice-mcp   # start the server on stdio
```

Python 3.11 or newer. A simulator is optional for most of the suite: recorded
`.raw` and `.log` fixtures cover the parse paths offline. `ngspice` on PATH
enables the live end-to-end tests; the LTspice integration tests are opt-in
behind an environment flag.

## The gate

Run all three before opening a pull request. CI runs the same commands.

```bash
uv run pyright
uv run ruff check --fix src/ tests/
uv run pytest tests/ -v
```

The suite runs serially by default. `-n auto` parallelises it locally; `-n0`
turns parallelism off, which is what you want when you are reading one
failure:

```bash
uv run pytest -n0 tests/test_pathutil.py::TestName::test_case -v
```

## Where things live

`docs/DESIGN.md` is the architecture and the reasoning behind it.
`CLAUDE.md` is the working map of the source tree — layers, module
responsibilities, and the patterns that are load-bearing.

Two documents are contracts rather than descriptions, and changing behaviour
they cover means changing them too:

- `docs/design/mcp_surface.md` — the MCP tools: the shared response envelope,
  each tool's argument shape, and why each rule exists.
- `docs/design/python_api.md` — the in-process Python API (`ltspice_mcp.api`),
  including `__all__` as its stability boundary.

Third-party code is listed in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Three rules that will come up in review

- **A regression test must fail before the fix and pass after.** If it never
  failed, it does not pin the behaviour. Run it against the unfixed code to
  prove it bites.
- **Test through real code paths and assert on real values.** A test that
  stubs the mechanism it claims to cover passes for the wrong reason. See
  `docs/TESTING.md` for what the suite does and does not mock.
- **Plain technical language in code, docstrings and commit messages.**
  Describe the actual behaviour, condition or bug. No severity codes,
  internal codenames, or references to a numbered backlog — those are not
  readable outside the project. Name a branch for what it changes
  (`fix/asc-export-lock`).

## Reporting a spicelib bug

spicelib is a pinned dependency we cannot patch in place, so we work around
its bugs and delete the workaround when upstream fixes them. Record every one
you hit in `docs/spicelib_bugs.md`, even if you also ship a workaround: that
file is the record of what to delete later, and its reproductions are what let
someone confirm a fix.

Follow the existing sections: summary, affected code and version,
reproduction, impact, proposed fix, a suggested upstream test, and a
cross-reference to our workaround and the test that pins it.

## Adding a tool or an op

Read `docs/TESTING.md` first. It describes the class of bug that shipped past
a long run of adversarial testing — a capability that is *missing*, or unusable
for an input class nobody fed it — and the mechanisms that now catch it:
inverse-operation closure over the schematic op surface, the archetype build
battery across device classes, task-down coverage, and judging the artifact
rather than the call sequence. A new op with no inverse, or a workflow that
was only ever exercised on passives, fails those checks by design.

The registration convention itself (`@registry.tool`, input models, output
schemas, the shared response helpers) is in `CLAUDE.md` under *Tool Module
Convention*.
