The `measurements` recipe reads a stepped run's `.meas` results a step at a
time, where it refused `step` and `all_steps` and gave only statistics over
every step. With `all_steps` it returns one row per step, and with `step`
the row of the step the selector picks, each with that step's
`step_values`, so an `analyze` block's `group_by` can name a stepped
parameter. Row n of a stepped `.meas` table is the step of the log's n-th
`.step` line on both LTspice builds (recorded: `log/meas_step`,
`log/step_two_params`, `log/step_temp`); a log whose `.step` lines do not
number its rows is refused. Without either, the recipe still gives the
statistics.
