# CLAUDE.md

Guidance for contributors and coding agents working in this repository. `docs/DESIGN.md` describes the architecture; `docs/TESTING.md` describes testing practice.

## Project overview

An MCP server that exposes SPICE circuit simulation to LLMs. LTspice is the primary simulator; ngspice, QSPICE and Xyce are supported but secondary. The server is built on the low-level `mcp.server.lowlevel.Server` API (not FastMCP), with spicelib as the simulation backend. The same engine is importable in-process as `ltspice_mcp.api`.

The server has 8 registered tools: `run_experiments` and `jobs` (run and track simulations), `analyze_results` and `inspect` (read results and circuits), `edit_schematic` and `verify_circuit` (author and check circuits), `plot_waveform` (the interactive waveform widget), and `run_code` (runs a Python snippet with the engine in scope as `api`; served unless `[tools] run_code = false`). The first six share one response envelope. `docs/design/mcp_surface.md` defines the envelope and each tool's arguments; read it before changing any of the six. `docs/design/python_api.md` specifies the Python API.

## Setup and commands

Run `uv sync` on Python 3.11 or newer. Most of the suite needs no simulator: recorded `.raw`/`.log` fixtures cover the parse paths offline. `ngspice` on PATH enables the live end-to-end tests; the LTspice integration tests are opt-in (`LTSPICE_MCP_RUN_LTSPICE_INTEGRATION=1`).

```bash
uv run ruff check --fix src/ tests/
uv run pyright
uv run pytest tests/ -v          # serial by default; add -n auto to run in parallel
uv run pytest -n0 tests/test_pathutil.py::TestName::test_case -v   # one test, readable output
python -m ltspice_mcp            # run the server without the console script
```

There is no Makefile. CI: `.github/workflows/ci.yml` runs `checks.yml` (ruff, pyright, the suite with coverage, and the suite under two jitter seeds) on pushes and pull requests to `master`; `publish.yml` runs the same checks and publishes to PyPI on version tags.

### Working practice

- Don't use `uv run python -c` to explore an API; read the source.
- A regression test must fail before the fix and pass after. Run it against the unfixed code to confirm it fails.
- Test through real code paths, never a bypass, and assert on real values. Never stub the mechanism a test claims to cover.
- Re-run a passing suite only after the tree changes.
- In tests, wait for the value the assertion reads, never for a side effect or a fixed time. Await work the server started through its owner (`await runner.settled(job)`, `await state.settled()`), and anything else (a thread, a fake simulator's callback, another process) through an event or a handshake. A timeout on a wait only caps a hang (`LIVENESS_S` in `tests/conftest.py`); it is not a speed requirement. `tests/test_test_hygiene.py` checks part of this rule mechanically. `--jitter-seed=N` reruns the suite with thread hand-offs, timers and process starts perturbed to imitate Windows scheduling (`tests/schedule_jitter.py`). A test that fails only under a seed has a race.

Read `docs/TESTING.md` before adding a tool or an op. Some tests check the code against a committed snapshot. After a deliberate change, regenerate the snapshot by running the test with its variable set to `1`: `LTSPICE_MCP_RECORD_SHEET_FINDINGS`, `LTSPICE_MCP_RECORD_ROUTE_PLANNER`, `LTSPICE_MCP_RECORD_LABELS_AND_MOVES`, `LTSPICE_MCP_RECORD_BUILD_BENCHMARK`, `LTSPICE_MCP_RECORD_INSPECT_SHEETS`.

### Recorded LTspice behaviour

`tests/fixtures/ltspice_recorded/` contains the files LTspice 26 and LTspice XVII each wrote for the inputs in `inputs/cases.toml`, which lists every LTspice behaviour the code models. `tests/test_recorded_ltspice_*.py` check the server against those files on any machine, and `scripts/record_ltspice_fixtures.py` regenerates them on Windows. Code that models a new LTspice behaviour needs an input in `inputs/cases.toml` and a test against its recording, not an expected value derived from the assumption. The two builds differ in log and export encodings, error wording and `.op` output, so behaviour checked on one build is not established for the other. See `docs/TESTING.md`, "Recorded LTspice behaviour".

## Windows is the primary platform

Most users run the server natively on Windows. Development happens on WSL2 and CI runs on Linux, so a change that passes there can still break on Windows. Every path, file write, signal, process, newline and encoding decision must work on Windows:

- `os.open` without `O_BINARY` opens in text mode.
- `signal.SIGKILL` and `os.killpg` do not exist.
- Symlinks need a privilege.
- `\r\n` is the platform newline.
- The platform codec is the machine's ANSI code page (cp1252, cp936, cp932 and so on), not UTF-8 unless Python is configured for it.
- A drive letter does not map to `/mnt`. The `/mnt` mapping, `wslpath` and the Windows-side temp directory are WSL interop, used only when `is_wsl()` is true.
- LTspice opens a window and takes the keyboard focus for every batch run and export, so code that launches LTspice on Windows must call `ltspice_windows.run_on_desktop`, which starts it on a hidden desktop.

The release gate includes a native Windows run of the suite (`docs/TESTING.md`, "Platforms and the release gate"). A test that cannot pass on a platform skips there with a reason instead of failing.

## Comments, docstrings, commit messages, and branch names

People outside the project read these, so keep internal shorthand out of them: no severity codes, internal codenames for findings, backlog item numbers, or internal batch or version labels. Describe the behaviour, condition or bug in plain technical terms. Name a branch after the change it makes (`fix/asc-export-lock`). Internal planning notes may use shorthand; shipped code, git history and branch names may not.

## spicelib bugs

spicelib is a pinned dependency we cannot fix in place, so we work around its bugs and remove each workaround once upstream is fixed. When you hit a spicelib bug or limitation (a wrong parse, a hang, a silently dropped field, an unapplied header value), record it in `docs/spicelib_bugs.md` as a self-contained section ready to send upstream: summary, affected code and version, reproduction, impact, proposed fix, a suggested upstream test, and a cross-reference to our workaround and the test that pins it. Do this even when you also ship a workaround. Follow the format of the existing entries.

## Where things live

All source is under `src/ltspice_mcp/`.

- **MCP protocol**: `server.py` (lifespan, dispatch, request routing, error-to-result conversion), `resources.py` (resources and URI templates), `prompts.py`, `main.py` (stdio entry point).
- **Tools**: `tools/`, one module per tool: `experiments.py`, `jobs.py`, `analyze.py`, `inspect_tools.py`, `schematic_edit.py`, `verify.py`, `analysis.py` (`plot_waveform`), `run_code.py`. Shared code: `_base.py` (response helpers, `ToolInput`, the response envelope), `_schema.py` (builds the published JSON Schema), `receipts.py` (the job receipt), `reference_index.py` (the table behind `inspect(kind="reference")`).
- **Python API**: `api/` (`Api`, booted by `engine.py`; `__all__` is the pinned stability boundary), `code_worker.py` (the worker process behind `run_code`), `detached_owner.py` (supervises a detached job).
- **Config and state**: `config.py` (`ServerConfig`), `state.py` (`SessionState`), `errors.py` (the error hierarchy).
- **Core logic**: `lib/`, by area:
  - Services and storage: `services.py` (job resolution, cached result loading), `store.py` (every path the server writes is a method on `Store`), `recent.py`, `cache.py`.
  - Jobs and experiments: `job_types.py`, `job_registry.py`, `job_lifecycle.py` (status state machine), `experiment_types.py`, `experiment_store.py`, `experiment_runner.py`, `variations.py`, `montecarlo.py`, `deck_prep.py`, `deck_staging.py`, `lint_rules.py`, `background.py`.
  - Running simulators: `runner_base.py` (`submit_netlist`), `runner_manager.py`, `simulator.py` (detection, WSL/Wine selection), `simulator_build.py` (identifies the build that ran), `proc_kill.py`, `ltspice_wsl.py`, `wsl.py`, `ltspice_windows.py`, `hidden_desktop.py`.
  - Nested targeting and PDK statistics: `hierarchy.py`, `hierarchy_expr.py`, `instance_targeting.py`, `native_inputs.py`, `pdk_native.py`, `native_execution.py`, `native_records.py`.
  - Netlists and libraries: `spice_lex.py`, `spice_lex_ops.py`, `spice_lex_views.py`, `spice_validator.py`, `library_manager.py`, `library_parser.py`, `encoding.py`.
  - Results: `raw_parser.py`, `log_parser.py`, `parser_service.py` (every `.raw`/`.log` read calls `load_artifacts_sync`, which parses in a subprocess under a deadline and memory limit), `parser_process.py`, `parser_worker.py`, `result_cache.py`, `result_store.py`, `recipes.py`, `metrics.py` (one function per `analyze_results` recipe), `ac_analysis.py`, `signal_analysis.py`, `ac_structure.py`, `result_observations.py`, `pagination.py`, `cursor_codec.py`, `projection.py`.
  - Schematics: `schematic_ops.py` (the `.asc` edit engine behind `edit_schematic`), `asc_document.py`, `sheet.py`, `symbol_file.py`, `symbol_library.py`, `symbol_geometry.py`, `geometry.py`, `connectivity.py` (groups a sheet's points into nets), `routing.py` (route rules), `sheet_findings.py` (the sheet rules that both `edit_schematic` and `verify_circuit` report), `component_value.py`, `plot_settings.py` (`.plt` files).
  - Rendering and comparison: `schematic_scene.py`, `schematic_renderer.py` (SVG), `raster.py` (PNG, optional `raster` extra), `netlist_graph.py`, `netlist_diff.py`, `pin_legend.py`, `plot_html.py`, `plot_svg.py`.
  - Open LTspice windows (native Windows): `ltspice_bridge.py`, `ltspice_window.py`, `ltspice_frame.py`, `pe_menu.py`.
  - Other: `guide.py` (the agent-facing guide), `simulator_docs.py`, `observability.py` (structured events on stderr), `pathutil.py` (`safe_path`), `filelock.py` (`circuit_file_lock`), `model_fields.py`.

The agent-facing guide has one copy: a core (`assets/guide/core.md`) and one file per section, listed in `guide.py:SECTION_ORDER`. Add guidance there, never under `skills/`.

## Tool conventions

A registered tool uses the `@registry.tool()` decorator:

```python
@registry.tool(
    name="foo",
    title="Do The Thing",           # required: the label a client shows a person
    description="...",
    input_model=FooInput,           # subclass of ToolInput (Pydantic)
    annotations=RO_ANNOTATIONS,     # or custom ToolAnnotations
    output_schema={...},            # JSON Schema for structuredContent
)
async def handle_foo(args: FooInput, state: SessionState) -> types.CallToolResult:
    ...
```

- `tools/__init__.py` imports every tool module to register it. Its import order is the advertised tool order; do not alphabetise it.
- Every tool declares a `title`.
- Every registered tool's `output_schema` must be an object schema at the top level. A bare `oneOf`/`anyOf` makes strict clients, Claude Code included, reject the whole `tools/list` response.
- A tool with an `output_schema` must return `structuredContent` on every code path, text-only replies included.
- `structuredContent` must carry everything the caller needs, because structured-aware clients drop the text channel when it is present. Copy any hint, referral, recovery step or caveat from the text into the data, conventionally as an optional `hint` key declared in the schema.
- A handler that a registered tool delegates to declares its own output schema with `@declare_output_schema(...)`; `tests/test_conformance_hook_armed.py` fails a delegation without one.
- Keep argument descriptions to one or two plain sentences (unit, sign convention, default, interaction with a sibling field). Put longer explanation in a guide section and point to it (`tests/test_guide_delivery.py` checks that pointers resolve).
- A file argument is named `path`.
- Raise errors from the hierarchy in `errors.py`. `server.py:call_tool` converts every failure into an `is_error` result, with one recovery hint per error type in `_ERROR_HINTS`.

## Rules the code depends on

- **Blocking work**: every request runs on one shared event loop, so offload slow filesystem and parse work with `asyncio.to_thread`. Build responses (`format_response`, `json_response`) in the handler coroutine, and touch cached editors only on the loop.
- **Background work**: start an unawaited task with its owner's `BackgroundTasks.spawn` (`lib/background.py`), never a bare `create_task`. A direct spawn elsewhere must be listed in `tests/test_test_hygiene.py`.
- **Paths**: every user-provided path is resolved through `state.resolve_path()` (`safe_path()` on the tool side).
- **Runners**: get a runner from `state.runners` (`RunnerManager`); never create one directly.
- **Circuit files**: every circuit-file mutation or export runs inside `circuit_file_lock`, taken before the cached editor is fetched. Resolve any geometry the mutation depends on inside the lock.

## Configuration

The server reads `ltspice-mcp.toml` from the working directory and, if it is missing, writes one with defaults on the first tool call (unless `LTSPICE_MCP_WRITE_CONFIG=false`). Environment variables with the `LTSPICE_MCP_` prefix override TOML values. `config.py:ServerConfig` lists every option. Sections: `[simulator]`, `[security]`, `[simulation]`, `[analysis]`, `[logging]`, `[schematic]`, `[tools]`, `[state]`.

- **Sandbox**: paths are unrestricted unless `[security] allowed_paths` (or `LTSPICE_MCP_ALLOWED_PATHS`) is set; then a path outside the list raises `PathSecurityError`.
- **Named executables**: `[simulator.executables]` (or `LTSPICE_MCP_SIMULATOR_EXECUTABLES`) adds more builds of a simulator, selected per run as `execution.simulator = "ltspice:xvii"`. Each keeps its family's class name, so code that identifies a build reads the executable (`simulator_build.executable_identity`), not the class name.
- **Symbol libraries**: `.asc` editing needs LTspice's `.asy` symbol files. spicelib locates them on Windows and under Wine; `wsl.get_ltspice_lib_paths()` locates them on WSL. `[schematic] symbol_paths` (or `LTSPICE_MCP_SYMBOL_PATHS`) overrides both. Without LTspice there is no `.asc` support.
- **ngspice mode**: spicelib runs ngspice in `kiltpsa` mode by default, which reads a sectioned `.lib <file> <section>` as two plain includes. `[simulator] ngbehavior` (or `LTSPICE_MCP_NGBEHAVIOR`) sets another mode, such as `"hsa"`.

## WSL

On WSL, `LTspice.exe` runs through Windows interop, not Wine. `lib/ltspice_wsl.py` (`LTspiceWSL`) converts paths with `wslpath` and is selected when `is_wsl()` is true. spicelib cannot detect LTspice across the WSL boundary, so set `simulator.path` in `ltspice-mcp.toml` to the Windows-side executable, for example `/mnt/c/Program Files/ADI/LTspice/LTspice.exe`. LTspice cannot write the `.MEAS` database over the `wsl.localhost` share, so on WSL its run artifacts are written to a Windows-native temp directory (`Store.artifact_base`). LTspice requires a netlist file extension (`.cir`, `.net`, `.sp`); `runner_base.submit_netlist` preserves it.
