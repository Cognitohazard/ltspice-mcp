# SPICE simulation guide

This is the core: how to work with this server, when to use Python or the
tools, the rules that cause silent errors, and an index of the topic sections
and task playbooks. Before you start a task, read the section whose line in the
index matches it.

Read one with `inspect(queries=[{"kind": "guide", "section": "<name>"}])`, or
`api.guide("<name>")` in Python. A pointer elsewhere such as
"guide section 'signals'" names one of them.

## How to work

- Write `.cir`, `.net` and `.sp` decks with your own file tools. Change an
  `.asc` schematic only through `edit_schematic`, which checks geometry and
  writes in one guarded transaction. Iterate on a netlist; build the `.asc`
  last.
- Simulate instead of reasoning it out: runs are cheap, and a sweep's cases
  run in parallel.
- Use `run_experiments` for LTspice, for sweeps, corners and Monte Carlo, and
  for jobs that outlive a call. A quick one-off ngspice run you make yourself
  is fine too: `analyze_results` with a `raw_path` source, or
  `api.load_raw(path)`, reads its raw.
- Check a new deck before a sweep: `verify_circuit` with `checks: ["syntax"]`.
- On LTspice, put each scalar you need in the deck as a `.meas` and read it
  back with the `measurements` recipe. On ngspice, read the trace with a
  recipe instead (see the rules below). Use the other recipes for what `.meas`
  cannot express.
- Pass a `request_id` to `run_experiments`: the same id and arguments return
  the original receipt instead of running again, and `jobs` finds the job by
  it.
- A completed run can still hold a degenerate result, such as a coerced value
  or a skipped `.meas`. Read `observations`, `warnings`, `failures` and
  `completeness` before trusting a number.
- Paths must lie in the sandbox; a refused path names the config line that
  widens it.

## Python or tools

Both run the same engine on the same job records: a job either one starts,
the other reads by `job_id`. The tools: `run_experiments` runs decks, sweeps
and corners; `jobs` follows, waits on and cancels them; `analyze_results`
measures finished runs; `inspect` reads circuits, symbols, libraries and this
guide; `edit_schematic` edits `.asc` sheets; `verify_circuit` checks and
compares circuits; `plot_waveform` draws a chart; `run_code` runs Python.

| | Python (`run_code`, or `ltspice_mcp.api`) | Tools |
|-|-|-|
| Cost | loops, decisions and trace math in one call; far fewer tokens over many steps | one step per call |
| Results | complete: every row, numpy arrays, no cursors | structured replies, paged and budgeted |
| Authority | `run_code` has the server's own file and process authority, outside the sandbox | paths confined to the sandbox |
| Long jobs | owned by the worker, and stopped by a worker restart unless submitted with `detach=True` | owned by the server |
| Limits | 60 s per snippet by default (600 at most), one snippet at a time | per-tool caps |
| Charts | HTML chart files from the `plot` recipe | `plot_waveform`: interactive, or an inline PNG |
| Failures | exceptions with tracebacks | error replies with a recovery hint |

Use Python for anything past a single call. Use a tool for one call whose
reply you will read as it is, for a chart you want to see, and for a job that
must outlive the `run_code` worker.

## The API on one page

In `run_code`, `api` is already open. In your own Python,
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

- The methods take the tools' arguments as keywords, and their results come
  back whole.
- `wait=False` returns the receipt at once and `api.wait(job_id)` blocks;
  adding `detach=True` lets the job outlive this process.
- Take statistics of a transient trace with `compute_signal_stats` and
  `time_weighted_quantiles`, which weight by time; `np.mean` does not (guide
  section 'signals').
- `ApiValidationError` means bad arguments and `ApiCallError` a failed call.
  A per-item failure is returned data, not an exception.

The rest is guide section 'python'.

## Rules that cause silent errors

- `M` is milli, not mega: `1M` is 0.001. Write `MEG` for 1e6. A suffix letter
  SPICE does not know is ignored without an error.
- Write micro as `u`. LTspice 24 writes µ in UTF-8 and reads it back; LTspice
  XVII misreads that µ and drops the scale, so `23µ` runs as 23.
- A deck that carries `.step` holds several sweeps in one raw;
  `analyze_results` reads the first unless you name `step` or `all_steps`.
- ngspice has no `.step`: sweep with `run_experiments` variations. Once a deck
  has a `.save` line, ngspice keeps only what `.save` names. As this server
  runs it, ngspice skips a top-level `.meas`: the deck runs and the value
  comes back absent. Read the trace with a recipe instead.
- Inline comments are `;` in LTspice and `$` in ngspice.
- Ground is `0`; `00` is a different node.
