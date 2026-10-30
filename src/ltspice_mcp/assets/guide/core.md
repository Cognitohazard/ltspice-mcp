# SPICE simulation guide

This is the guide's core. The index at its end lists the topic sections and
task playbooks. Before you start a task, read the section whose line in the
index matches it.

Read one with `inspect(queries=[{"kind": "guide", "section": "<name>"}])`, or
`api.guide("<name>")` in Python. A pointer elsewhere such as
"guide section 'signals'" names one of them.

## How to work

- Write `.cir`, `.net` and `.sp` decks with your own file tools. Change an
  `.asc` schematic only through `edit_schematic`, which checks geometry and
  writes a whole batch of ops at once, or nothing if one fails. Iterate on a
  netlist; build the `.asc` last.
- Simulate instead of reasoning it out, because runs are cheap and a sweep's
  cases run in parallel.
- Use `run_experiments` for LTspice, for sweeps, corners and Monte Carlo, and
  for jobs that outlive a call. You may also make a quick one-off ngspice run
  yourself and read its raw with `analyze_results` (a `path` source) or
  `api.load_raw(path)`.
- Check a new deck with `verify_circuit` and `checks: ["syntax"]` before a
  sweep.
- On LTspice, put each scalar you need in the deck as a `.meas` and read it
  back with the `measurements` recipe. On ngspice, read the trace with a
  recipe instead (see the rules below). Use the other recipes for what `.meas`
  cannot express.
- Pass a `request_id` to `run_experiments`. A repeat call with the same id and
  arguments returns the original receipt instead of running again, and `jobs`
  finds the job by that id.
- A completed run can still hold a degenerate result, such as a coerced value
  or a skipped `.meas`. Read `observations`, `warnings`, `failures` and
  `completeness` before trusting a number.

## Python or tools

Both run the same engine on the same job records, so either one can read a
job the other started, by its `job_id`. Among the tools, `run_experiments`
runs decks, sweeps and corners; `jobs` follows, waits on and cancels them;
`analyze_results` measures finished runs; `inspect` reads circuits, symbols,
libraries and this guide; `edit_schematic` edits `.asc` sheets;
`verify_circuit` checks and compares circuits; `plot_waveform` draws a chart;
`run_code` runs Python.

| | Python (`run_code`, or `ltspice_mcp.api`) | Tools |
|-|-|-|
| Cost | loops, decisions and trace math in one call; far fewer tokens over many steps | one step per call |
| Results | complete (every row, numpy arrays, no cursors) | structured replies, paged and budgeted |
| Authority | `run_code` has the server's own file and process authority | a path outside `[security] allowed_paths` is refused when that is set |
| Long jobs | owned by the worker, and stopped by a worker restart unless submitted with `detach=True` | owned by the server |
| Limits | 60 s per snippet by default (600 at most), one snippet at a time | per-tool caps |
| Charts | HTML chart files from the `plot` recipe | `plot_waveform`: interactive, or an inline PNG |
| Failures | exceptions with tracebacks | error replies with a recovery hint |

Use Python for anything past a single call. Use a tool for one call whose
reply you will read as it is, for a chart you want to see, and for a job that
must outlive the `run_code` worker.

## The API on one page

`run_code` opens `api` for you. In your own Python, run
`from ltspice_mcp.api import Api, compute_signal_stats, window_and_clean`
and `api = Api(working_dir=".")`.

```python
print(api.reference("run_experiments"))  # every argument, with an example
receipt = api.run_experiments(  # waits until the job is terminal
    request_id="ldo-load-1",
    circuits=[{"path": "ldo.cir"}],
    variations=[{"kind": "assign", "assign": {"ILOAD": ["1m", "10m", "100m"]}}],
)
for row in receipt["runs"]["items"]:  # one row per case
    r = api.load_raw(job_id=receipt["job_id"], case_id=row["case_id"])
    t, v = r.axis(step=0), r.trace("V(out)", step=0)
    t_w, v_w, _ = window_and_clean(t, v, 4e-3, None)  # from 4 ms to the end
    print(row["assignments"], compute_signal_stats(t_w, v_w)["mean"])
```

- The methods take the tools' arguments as keywords and return complete
  results.
- `wait=False` returns the receipt at once and `api.wait(job_id)` blocks;
  adding `detach=True` lets the job outlive this process.
- Take statistics of a transient trace with `compute_signal_stats` and
  `time_weighted_quantiles`, which weight by time; `np.mean` does not (guide
  section 'signals').
- `ApiValidationError` means bad arguments and `ApiCallError` a failed call.
  A per-item failure does not raise; it is returned in the data.

Guide section 'python' covers the rest of the API.

## Rules that cause silent errors

- `M` is milli, not mega: `1M` is 0.001. Write `MEG` for 1e6. A suffix letter
  SPICE does not know is ignored without an error.
- Write micro as `u`. LTspice 24 writes µ in UTF-8 and reads it back; LTspice
  XVII misreads that µ and drops the scale, so `23µ` runs as 23.
- A deck with `.step` writes all its steps to one raw; `analyze_results`
  reads the first step unless you name `step` or `all_steps`. The
  `measurements` recipe gives statistics over every step unless you name them.
- ngspice has no `.step`, so sweep with `run_experiments` variations. Once a
  deck has a `.save` line, ngspice keeps only what `.save` names. Under this
  server, ngspice skips a top-level `.meas`, so the deck runs but the value is
  missing from the result. Read the trace with a recipe instead.
- Inline comments are `;` (or a `$` standing alone) in LTspice and `$` in
  ngspice. Each ends at the end of its line.
- Ground is `0`; `00` is a different node.
