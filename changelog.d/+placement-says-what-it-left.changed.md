A placement of results beside a circuit (`jobs(action="place_beside_sheet")`,
`execution.results_beside_sheet`, `plot_waveform` with `in_ltspice`) now
names in `left_in_place` each of the circuit's result files from another run
that it left beside it, such as an earlier `.op.raw`, and `not_checked` says
that what LTspice's window does with it is not recorded. `not_checked` also
says that the files the circuit includes or uses were not compared with the
ones the run read.
