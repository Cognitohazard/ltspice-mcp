A run's reply now says what the run did not give. Each run row of
`run_experiments` and `jobs` names the `.meas` LTspice printed as failed
(`failed_measurements`, with `failed_measurement_steps` for a stepped run that
failed at some steps only) and the steps whose printed value is not
established (`measurements_not_established`, as LTspice XVII prints 0 for a
failed step), and each requested `.meas` the log printed nothing for
(`measurements_not_printed`). A row of a run recorded before this says
`measurements_coverage: "not_read"`, and one of a QSPICE or Xyce run, whose
measure output the server does not read, `"simulator_unsupported"`. The
receipt's hint names the
`measurements` recipe that reads the values. The receipt and `analyze_results`
add a `meas_failed_in_every_run` observation naming each `.meas` that failed in
every run read, and saying so when that is every `.meas` the deck requests. A
run whose raw holds no node voltage inside the deck's subcircuits, where a
subcircuit has a net of its own, says so (`subcircuit_nodes_not_saved`) on its
receipt and on every read by job, naming the deck's `.save` cards and LTspice's
"Save subcircuit node voltages" setting as the causes. A read or a recipe that
takes samples from a transient LTspice XVII saved with waveform compression on
says so (`compressed_transient`). The guide's core lists the recipe that
answers each common question, and its example calls one.
