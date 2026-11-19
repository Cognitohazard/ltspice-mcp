`run_experiments` and `jobs` replies now go through the same presenter as
`verify_circuit`, and two things a caller may read change:

- Run rows are cut by size. `budget` (or `[analysis] default_budget`) is the
  reply's target. A receipt over it keeps at least one run row, and its new
  `cut` field names what was left out, with a cursor that reads the rest from
  the job's record through `jobs(action="runs")`. The new `runs_not_shown`
  names, per kind, the runs left out that carry a diagnostic, a measurement
  that gave no value or was not read, or a status other than produced, each
  with a cursor to its row. With no target, an MCP reply holds 50 run rows.
  A reply that cannot meet its target says so once (`budget_not_met`).
  Offset cursors from earlier releases (`o:<n>`) are refused.
- A note that describes the job or the request (`micro_sign_folded`,
  `ngspice_mode_unverified`, `results_not_beside_sheet`, the request's
  `lint` and `source`, and the route to the `measurements` recipe) is sent in
  full once per session. After that, a `wait`, a run receipt or a `resume`
  sends it as its code with `said_before` and no `detail`.
  `jobs(action="status")` and a replay by `request_id` send every note in
  full, and the Python API returns everything in full.

Also: a `jobs(action="runs")` page shows the job's `status` and `progress` as
it stands when read; `jobs(action="list")` is stored and cut by circuit
groups; a receipt carries `job_type`, `analysis_status` and `dialect`; a
receipt for a deck with `.meas` cards names the `measurements` recipe; and the
route for a thinned attached waveform names `max_points` and the CSV format.
