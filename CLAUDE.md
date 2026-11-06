# CLAUDE.md

Guidance for contributors and coding agents working in this repository. `docs/DESIGN.md` describes the architecture; `docs/TESTING.md` describes testing practice.

## Project overview

An MCP server that exposes SPICE circuit simulation to LLMs. LTspice is the primary simulator; ngspice, QSPICE and Xyce are supported but secondary. The server is built on the low-level `mcp.server.lowlevel.Server` API (not FastMCP), with spicelib as the simulation backend. The same engine is importable in-process as `ltspice_mcp.api`.

The server has 8 registered tools: `run_experiments` and `jobs` (run and track simulations), `analyze_results` and `inspect` (read results and circuits), `edit_schematic` and `verify_circuit` (author and check circuits), `plot_waveform` (the interactive waveform widget), and `run_code` (runs a Python snippet with the engine in scope as `api`; served unless `[tools] run_code = false`). The first six share one response envelope. `docs/design/mcp_surface.md` defines the envelope and each tool's arguments; read it before changing any of the six. `docs/design/python_api.md` specifies the Python API.

## Setup and commands

Run `uv sync` on Python 3.11 or newer. Most of the suite needs no simulator, because recorded `.raw`/`.log` fixtures cover the parse paths offline. `ngspice` on PATH enables the live end-to-end tests; the LTspice integration tests are opt-in (`LTSPICE_MCP_RUN_LTSPICE_INTEGRATION=1`).

```bash
uv run ruff check --fix src/ tests/
uv run pyright
uv run pytest tests/ -v          # serial by default; add -n auto to run in parallel
uv run pytest -n0 tests/test_pathutil.py::TestName::test_case -v   # one test, readable output
python -m ltspice_mcp            # run the server without the console script
```

There is no Makefile. CI: `.github/workflows/ci.yml` runs `checks.yml` (ruff, pyright, the suite with coverage on Linux, the suite on Windows, the suite on macOS with the tests that read a result skipped, and a dependency audit) on pushes and pull requests to `master`; `publish.yml` runs the same checks and publishes to PyPI on version tags.

### Working practice

- Don't use `uv run python -c` to explore an API; read the source.
- A regression test must fail before the fix and pass after. Run it against the unfixed code to confirm it fails. When the bug is a wrong answer, the test runs the failing input through every code path that answers the same question, not only the one it was reported through.
- Before fixing a wrong answer, name the question it answers ("which cards of a deck does LTspice read", "which points of a sheet are one net"), not the tool or op that showed it. Then find every other place that answers it: the callers of the function that decides it, other code reading the same text or record, and the modules listed for the behaviour in `inputs/cases.toml` (add any it misses). Fix it where the question is decided and move the other answers there in the same change. If no single place decides it, make one that answers the whole question, so no caller has a second step to remember. If two paths must differ, say why beside both and pin the difference in a test whose name states it.
- Changing what a shared function returns or accepts changes every caller's answer. Read each caller and run the changed input through it, because a caller can rely on the old result in a way no test of the function shows. Before a pull request with such a change is merged, have an agent that did not write it review the diff against these rules, without the author's conclusions, and fix or answer what it finds; say in the pull request which shared functions changed.
- A docstring saying a function is the one answer stops nobody from writing a second one. When only its owner may call a function, or a replaced function keeps callers for a while, ban its name in `[tool.ruff.lint.flake8-tidy-imports.banned-api]` in `pyproject.toml`, also under any module that imports it, and mark each allowed use with `# noqa: TID251 - <reason>`. Ruff then refuses any other use and flags a marker its line no longer needs (RUF100), so the allowed uses only shrink. A rule Ruff cannot express (a literal, a pattern, a method) gets a test that scans the source (`tests/_source_scan.py`).
- Test through real code paths, never a bypass, and assert on real values. Never stub the mechanism a test claims to cover.
- Re-run a passing suite only after the tree changes.
- In tests, wait for the value the assertion reads, never for a side effect or a fixed time. Await work the server started through the object that started it (`await runner.settled(job)`, `await state.settled()`), and anything else (a thread, a fake simulator's callback, another process) through an event or a handshake. A timeout on a wait only caps a hang (`LIVENESS_S` in `tests/conftest.py`), so it is not a test of speed. `tests/test_test_hygiene.py` checks part of this rule mechanically. To reproduce on Linux a test that fails only on Windows, rerun it with `--jitter-seed=random`, which perturbs thread hand-offs, timers and process starts as Windows scheduling does (`tests/schedule_jitter.py`) and prints the seed that reproduces a failure. A test that fails only under a seed has a race. No gate runs it; use it to investigate a flaky test.

Read `docs/TESTING.md` before adding a tool or an op. Some tests check the code against a committed snapshot. After a deliberate change, regenerate the snapshot by running the test with its variable (`LTSPICE_MCP_RECORD_SHEET_FINDINGS`, `LTSPICE_MCP_RECORD_ROUTE_PLANNER`, `LTSPICE_MCP_RECORD_LABELS_AND_MOVES`, `LTSPICE_MCP_RECORD_BUILD_BENCHMARK`, `LTSPICE_MCP_RECORD_INSPECT_SHEETS` or `LTSPICE_MCP_RECORD_TOOL_ARGUMENTS`) set to `1`. `LTSPICE_MCP_RECORD_SHEET_EDITS` re-records the result of every `edit_schematic` call in the suite (`tests/sheet_edit_record.py`), so it needs a run of the whole suite with no LTspice library installed or visible.

### Recorded LTspice behaviour

`tests/fixtures/ltspice_recorded/` contains the files LTspice 26 and LTspice XVII each wrote for the inputs in `inputs/cases.toml`, which lists every LTspice behaviour the code models. `tests/test_recorded_ltspice_*.py` check the server against those files on any machine, and `scripts/record_ltspice_fixtures.py` regenerates them on Windows. Code that models a new LTspice behaviour needs an input in `inputs/cases.toml` and a test against its recording instead of an expected value derived from the assumption. The two builds differ in log and export encodings, error wording and `.op` output, so behaviour checked on one build is not established for the other. See `docs/TESTING.md`, "Recorded LTspice behaviour".

## Windows is the primary platform

Most users run the server natively on Windows. Development happens on WSL2 and CI runs on Linux, so a change that passes there can still break on Windows. Every path, file write, signal, process, newline and encoding decision must work on Windows:

- `os.open` without `O_BINARY` opens in text mode.
- `signal.SIGKILL` and `os.killpg` do not exist.
- Symlinks need a privilege.
- `\r\n` is the platform newline.
- The platform codec is the machine's ANSI code page (cp1252, cp936, cp932 and so on); it is UTF-8 only when Python is configured for it.
- A drive letter does not map to `/mnt`. The `/mnt` mapping, `wslpath` and the Windows-side temp directory are WSL interop, used only when `is_wsl()` is true.
- LTspice opens a window and takes the keyboard focus for every batch run and export, so code that launches LTspice on Windows must call `ltspice_windows.run_on_desktop`, which starts it on a hidden desktop.

The release gate includes a native Windows run of the suite (`docs/TESTING.md`, "Platforms and the release gate"). A test that cannot pass on a platform skips there with a reason instead of failing.

## Comments, docstrings, commit messages, and branch names

People outside the project read these, so keep severity codes, internal codenames for findings, backlog item numbers, and internal batch or version labels out of them. Describe the behaviour, condition or bug in plain technical terms. Name a branch after the change it makes (`fix/asc-export-lock`). Shorthand is allowed only in internal planning notes, never in shipped code, git history or branch names.

## spicelib bugs

spicelib is a pinned dependency we cannot fix in place, so we work around its bugs and remove each workaround once upstream is fixed. When you find a spicelib bug or limitation (a wrong parse, a hang, a silently dropped field, an unapplied header value), record it in `docs/spicelib_bugs.md` as a self-contained section ready to send upstream, with a summary, the affected code and version, a reproduction, the impact, a proposed fix, a suggested upstream test, and a cross-reference to our workaround and the test that pins it. Do this even when you also ship a workaround. Follow the format of the existing entries.

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
  - Replies: `reply_fields.py` (the rule that leaves the fields that say nothing out of every reply).
  - Results: `result_kinds.py` (what a plot is: its analysis, its axis and, for a stepped LTspice plot, its kind, with where each fact is kept and the recordings that establish it), `raw_parser.py`, `log_parser.py`, `parser_service.py` (every `.raw`/`.log` read calls `load_artifacts_sync`, which parses in a subprocess under a deadline and memory limit), `parser_process.py`, `parser_worker.py`, `result_cache.py`, `result_store.py`, `recipes.py`, `metrics.py` (one function per `analyze_results` recipe), `ac_analysis.py`, `signal_analysis.py`, `ac_structure.py`, `result_observations.py`, `pagination.py`, `cursor_codec.py`, `projection.py`.
  - Schematics: `schematic_ops.py` (the `.asc` edit engine behind `edit_schematic`), `asc_document.py`, `sheet.py`, `symbol_file.py`, `symbol_library.py`, `symbol_geometry.py`, `geometry.py`, `connectivity.py` (LTspice's rules for which points of a sheet are one net), `nets.py` (a sheet's nets in a form that holds no tool's rules, which every reader of nets uses), `routing.py` (route rules), `sheet_findings.py` (the sheet rules that both `edit_schematic` and `verify_circuit` report), `component_value.py`, `plot_settings.py` (`.plt` files).
  - Rendering and comparison: `schematic_scene.py`, `schematic_renderer.py` (SVG), `raster.py` (PNG, optional `raster` extra), `netlist_graph.py`, `netlist_diff.py`, `pin_legend.py`, `plot_html.py`, `plot_svg.py`.
  - Open LTspice windows (native Windows): `ltspice_bridge.py`, `ltspice_window.py`, `ltspice_frame.py`, `pe_menu.py`.
  - Other: `guide.py` (the agent-facing guide), `simulator_docs.py`, `observability.py` (structured events on stderr), `pathutil.py` (`safe_path`), `filelock.py` (`circuit_file_lock`), `model_fields.py`.

The agent-facing guide exists only once, as a core (`assets/guide/core.md`) and one file per section, listed in `guide.py:SECTION_ORDER`. Add guidance there, never under `skills/`.

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

- Hold a new tool, op or reply to the design principles in `docs/DESIGN.md`: validate before write and name the fix in every refusal; a check says what it did not look at; one call per intent; a short reply with the data behind an id; repeatable requests; LTspice's own tools as peers; evidence before assumption; fast enough to try. Say in the change which principle it keeps or restores.
- `tools/__init__.py` imports every tool module to register it. Do not alphabetise the imports, because each module's dependencies load ahead of it. `ADVERTISED_ORDER` in the same file sets the order `tools/list` advertises, because clients cache the list and prompt caching depends on its exact bytes; a registered tool missing from it stops the server at startup.
- Every tool declares a `title`.
- Every input schema is an object with no `oneOf`, `allOf` or `anyOf` at its top level: Claude's API refuses such a tool, and a client then drops it from the session. A tool whose arguments are a union of actions publishes one flat object, lists each action's fields in its tool description, which the compact listing keeps (`jobs_input_schema`, `jobs_fields_by_action`), and validates each action in its input model. `tests/test_dispatch.py` checks every tool.
- Every registered tool's `output_schema` must be an object schema at the top level, as MCP requires. `tools/list` leaves output schemas out (`ToolRegistry.get_tools`), but advertising a bare `oneOf`/`anyOf` would make strict clients, Claude Code included, reject the whole response.
- A tool with an `output_schema` must return `structuredContent` on every code path, text-only replies included.
- `structuredContent` must contain everything the caller needs: structured-aware clients read only it, and the server makes the text block that data serialized (`server._with_serialized_text`), as the MCP specification recommends. Put any hint, referral, recovery step or caveat in the data, conventionally as an optional `hint` key declared in the schema.
- Everything the server sends a caller carries information, and nothing else. A reply leaves out an empty list or object, a null that means "nothing here", a flag at its default value, and a partial answer the overall answer already implies (when a comparison is `equivalent`, its parts match too). Declare such a field optional in the output schema. A missing field means "checked, and nothing to report", so a check that did not run, a step that was skipped or a value that could not be read is stated in the reply, never left out. The same holds for text: a description, hint or guide line tells the reader something they could not infer.
- The reply rule above is applied in one place: `reply_payload` in `_base.py` runs `lib/reply_fields.py` over every reply, and the Python API runs it over its results (`as_sent`, in `_CataloguedMethod`), so a handler builds its payload in full and need not drop an empty field itself; output schemas are relaxed to match when a tool registers. A field whose empty, null or false value is the answer goes on the tool's `ALWAYS_PRESENT` list in `_base.py`, with the reason; a registered tool needs an entry. The conformance hook fails a reply that carries any other such field (`docs/design/mcp_surface.md`, "Fields left out").
- A handler that a registered tool delegates to declares its own output schema with `@declare_output_schema(...)`, so the suite checks its `structuredContent` against that schema instead of the delegating tool's; `tests/test_conformance_hook_armed.py` fails a delegation without one.
- Keep argument descriptions to one or two plain sentences (unit, sign convention, default, interaction with a sibling field). Put longer explanation in a guide section and point to it (`tests/test_guide_delivery.py` checks that pointers resolve).
- A file argument is named `path`.
- Raise errors from the hierarchy in `errors.py`. `server.py:call_tool` converts every failure into an `is_error` result, with one recovery hint per error family in `errors.ERROR_HINTS` (`errors.error_hint`).

## Rules the code depends on

- **Blocking work**: every request runs on one shared event loop, so offload slow filesystem and parse work with `asyncio.to_thread`. Build responses (`format_response`) in the handler coroutine, and use the cached editors (`state.editors`, the parsed circuit files the session keeps between calls) only on the loop.
- **Background work**: start a task you do not await with `spawn` on the `BackgroundTasks` (`lib/background.py`) of the object that starts it, never with a bare `create_task`, so that object's `settled()` can wait for the work. A direct spawn elsewhere must be listed in `tests/test_test_hygiene.py`.
- **Paths**: every user-provided path is resolved through `state.resolve_path()` (`safe_path()` on the tool side), which resolves a relative path and refuses a path outside `[security] allowed_paths` when that is set.
- **Runners**: get a runner from `state.runners` (`RunnerManager`); never create one directly. The manager keeps one runner per simulator and output folder, so runs share its limit on parallel simulations and its per-job cancel events.
- **Circuit files**: every circuit-file mutation or export runs inside `circuit_file_lock`, taken before the cached editor is fetched. The lock serializes edits from parallel server processes, whose saves would otherwise overwrite each other, and taking it first means the editor cache sees another writer's finished save. Resolve any geometry the mutation depends on inside the lock.

## Configuration

The server reads `ltspice-mcp.toml` from the working directory and, if it is missing, writes one with defaults on the first tool call (unless `LTSPICE_MCP_WRITE_CONFIG=false`). Environment variables with the `LTSPICE_MCP_` prefix override TOML values. `config.py:ServerConfig` lists every option. Sections: `[simulator]`, `[security]`, `[simulation]`, `[analysis]`, `[logging]`, `[schematic]`, `[tools]`, `[state]`.

- **Sandbox**: paths are unrestricted unless `[security] allowed_paths` (or `LTSPICE_MCP_ALLOWED_PATHS`) is set; then a path outside the list raises `PathSecurityError`.
- **Named executables**: `[simulator.executables]` (or `LTSPICE_MCP_SIMULATOR_EXECUTABLES`) adds more builds of a simulator, selected per run as `execution.simulator = "ltspice:xvii"`. Each keeps its family's class name, so code that identifies a build reads the executable (`simulator_build.executable_identity`) instead of the class name.
- **Symbol libraries**: `.asc` editing needs LTspice's `.asy` symbol files. A session reads them through one `SymbolLibrary` (`state.symbols`, built by `simulator.symbol_library_for`), which searches the sheet's own folder, then the folders in `[schematic] symbol_paths` (or `LTSPICE_MCP_SYMBOL_PATHS`), then the `sym` folder of the LTspice build the session runs. `simulator_library_roots` finds that build's library on Windows, under Wine and on WSL (`wsl.get_ltspice_lib_paths()`). Without LTspice there is no `.asc` support.
- **ngspice mode**: spicelib runs ngspice in `kiltpsa` mode by default, which reads a sectioned `.lib <file> <section>` as plain includes and drops the section, so `run_experiments` refuses such a deck (lint rule `lib-section-ngspice`). `[simulator] ngbehavior` (or `LTSPICE_MCP_NGBEHAVIOR`) sets another mode, such as `"hsa"`. ngspice applies `-D ngbehavior=` before its startup files, so every ngspice launch carries `simulator.NGSPICE_NO_USER_INIT` (`-n`); without it a user's `.spiceinit` would replace the mode the server models the deck in.

## WSL

On WSL, `LTspice.exe` runs through Windows interop instead of Wine. `lib/ltspice_wsl.py` (`LTspiceWSL`) converts paths with `wslpath` and is selected when `is_wsl()` is true. spicelib cannot detect LTspice across the WSL boundary, so set `simulator.path` in `ltspice-mcp.toml` to the Windows-side executable, for example `/mnt/c/Program Files/ADI/LTspice/LTspice.exe`. LTspice cannot write the `.MEAS` database over the `wsl.localhost` share, so on WSL its run artifacts are written to a Windows-native temp directory (`Store.artifact_base`). LTspice requires a netlist file extension (`.cir`, `.net`, `.sp`); `runner_base.submit_netlist` preserves it.
