The Python API's `measurements` and `device_op_points` return a `LogResult`,
a `dict` with `observations`, so a stopped case's log says, as `load_raw`
does, that its results end where the run was stopped (`partial_raw`).
