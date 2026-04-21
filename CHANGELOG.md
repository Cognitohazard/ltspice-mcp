# Changelog

All notable changes to this project are documented here. The format loosely
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the
project will adopt [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
once it reaches `1.0.0`. Until then, minor versions may contain breaking
tool-surface changes.

## [Unreleased]

### Security

- Upgraded transitive dependencies `cryptography` (46.0.6 → 46.0.7,
  GHSA-p423-j2cm-9vmq / CVE-2026-39892) and `python-multipart`
  (0.0.24 → 0.0.26, GHSA-mj87-hwqh-73pj / CVE-2026-40347) via
  `uv.lock` refresh.

### Added

- `SECURITY.md` describing the vulnerability reporting process and the
  local threat model.
- `CHANGELOG.md` (this file).
- `.github/dependabot.yml` for weekly GitHub Actions and Python
  dependency updates.
- `pip-audit` step in CI to fail the build on newly published advisories
  against pinned dependencies.
- `twine check --strict` on built artefacts before PyPI publish so
  broken metadata fails the release instead of landing on PyPI.

### Changed

- `pyproject.toml` now declares Trove `classifiers` so PyPI renders
  category, license, and Python-version metadata correctly.
- CI and publish workflows hardened: default-read `GITHUB_TOKEN`
  (top-level `permissions: contents: read`), `persist-credentials:
  false` on every checkout, `timeout-minutes` on every job,
  `concurrency` groups that cancel redundant PR runs but serialise
  release builds, and `if-no-files-found: error` on the dist
  artefact upload.
- `pypa/gh-action-pypi-publish` pinned to the `v1.14.0` commit SHA
  (`cef2210…`) instead of the mutable `release/v1` branch — the
  publish step holds the OIDC token used for PyPI Trusted
  Publishing, so a floating ref there is the weakest link in the
  supply chain.

## Earlier history

Release history before the first tagged version lives in `git log`; the
`feat:` / `fix:` / `refactor:` prefixes and PR descriptions describe
each change.
