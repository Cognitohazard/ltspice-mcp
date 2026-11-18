`run_experiments` and `jobs` replies now go through the same presenter as
`verify_circuit`. `budget` (or `[analysis] default_budget`) is the reply's
target: a receipt over it is cut by run rows, always keeping at least one, and
its new `cut` field names what was left out with the cursor that reads the
rest from the job's record through `jobs(action="runs")`. A reply that cannot
meet its target says so once (`budget_not_met`), and no note claims that
nothing was dropped; before, a small budget could leave a receipt with no run
row while saying no fact was dropped. A `jobs(action="runs")` page shows the
job's `status` and `progress` as it stands when read. `jobs(action="list")` is
stored and cut by circuit groups. Cursors from earlier releases are refused.

A note that describes the job or the request (`micro_sign_folded`,
`ngspice_mode_unverified`, `results_not_beside_sheet`, the request's `lint`
and `source`, and the route to the `measurements` recipe) is sent in full once
per session and as its code with `said_before` after that, on a `wait` and a
run receipt; `jobs(action="status")` always sends every note in full, and the
Python API returns everything in full. A receipt now also carries `job_type`,
`analysis_status` and `dialect`, as `jobs` replies do. The route sentence for a
thinned attached waveform now names `max_points` and the CSV format.
