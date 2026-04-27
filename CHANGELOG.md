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
- New `lib/montecarlo.py` engine replacing `spicelib.Montecarlo`. Adds
  three foundry-grade MOSFET perturbation classes — per-`.MODEL`
  process variation, Pelgrom-scaled per-instance mismatch via inline
  variant cards, and `.PARAM` rewriting — alongside the existing
  R/C/L engine. Uses keyed sub-streams and ±tolerance-truncated
  Gaussians.
- `MonteCarloConfig.seed` field for reproducible MC runs; per-run
  realised perturbed values are now stored in
  `run_results[i]["params"]` so `batch_results` and
  `measurement_stats` can correlate measurements with realised
  values.
- `ToleranceSpec.kind` (`relative` | `absolute`) so component
  tolerances can be expressed as either a fraction of nominal or an
  absolute σ.
- `lib/spice_validator.py` — Layer A directive validator that blocks
  unsupported expressions (`vdb()`, `phase()`, `group_delay()`) in
  `.MEAS` at directive-write time with concrete suggestions. Fast
  path skips non-`.MEAS` directives.
- Layer B `.MEAS` error surfacing: `extract_log_diagnostics` returns
  a structured `meas_errors` list with the offending directive and an
  optional suggestion from the validator blocklist, propagated
  through `run_simulation` and `simulation_summary`.

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
- Removed the misleading legacy `phase_margin` / `gain_margin` fields
  from `simulation_summary`; loop-gain analysis now goes through
  `ltspice_stability_metrics` exclusively.
- `SimulationRunner._kill` renamed to `kill` — it is a documented
  cross-class API used by both the runner's `cancel()` and the
  tool-layer timeout path, so the leading underscore was misleading.
- Typed tightening across runners and parsers: `Literal` for
  distribution names, `TypedDict` for log diagnostics, and
  `parse_value` reuses `lib.format.parse_spice_value`.
- Ruff `allowed-confusables` list expanded with `σ × · − ∝ µ √ °`
  so RUF001/002/003 stop flagging deliberate scientific / EE
  notation in docstrings (Pelgrom's σ ∝ 1/√(W·L), ±3σ truncation,
  V·µm units, etc.).
- `pyrightconfig.json` adds an `executionEnvironments` override that
  silences `reportPrivateUsage` for the `tests/` tree only — `src/`
  still reports it.

### Fixed

- Parallel sweep parameter mislabelling when `max_parallel > 1`:
  `run_results` was keyed by completion order while `sim_info` was
  zipped by `runno`, so completed-out-of-order runs got the wrong
  parameter labels. `wrap_runner_for_runno_callbacks` now captures
  `task.runno` at submission and the per-run callback receives it
  explicitly, so `run_results` pairs with `sim_info[runno]`
  regardless of completion order.
- Replaced `spicelib.Montecarlo`'s broken Gaussian sampler, which
  used `random.gauss(value, tol/3)` (absolute σ instead of
  multiplicative) and an unseeded `random.Random()` per call.
- AC `operating_point` no longer silently returns complex magnitudes
  as voltages.
- MOSFET `set_component_value` no longer duplicates `W` / `L` tokens.
- `find_model` now sees LTspice's UTF-16 libraries.
- `.NOISE` signal lookup is now case-insensitive.
- `batch_results` AC aggregation no longer collapses to peak-only.
- `active_jobs` no longer counts completed records.
- `pulse_response` refuses the auto `initial_value` / `final_value`
  when the leading/trailing 10 % has stddev > 10 % of
  `|final - initial|` — the window straddles the edge or hasn't
  settled, which was silently producing wrong overshoot numbers.

## Earlier history

Release history before the first tagged version lives in `git log`; the
`feat:` / `fix:` / `refactor:` prefixes and PR descriptions describe
each change.
