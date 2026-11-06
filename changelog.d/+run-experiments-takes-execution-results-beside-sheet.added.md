`run_experiments` takes `execution.results_beside_sheet`. With it set, each
sheet's results are copied beside it, under its name, once the runs
finish, so LTspice opens them with the sheet, without a separate
`jobs(action="place_beside_sheet")` call. The receipt says, per sheet, what
was copied (`results_placed_beside_sheet`) or why not
(`results_not_placed`, for example a sheet edited during the run). A request
is refused before it runs when a circuit is a netlist, the job does not run
on LTspice, or a sheet would run more than one case; it cannot be combined
with `recoverable`. The jobs action and the run option decide whether a case
can be copied by the same rules.
