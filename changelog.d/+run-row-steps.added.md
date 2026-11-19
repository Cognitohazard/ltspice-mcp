A run row says how a stepped case's results are stepped: `steps` gives the
number of steps and the names of the stepped parameters, on the receipt of
`run_experiments`, `jobs` status, wait and runs, and the Python API. It is
decided once when the run ends, from the raw's header and the log's `.step`
lines, and kept on the case's record; where the two disagree, or a step's
start cannot be read, `steps` says why instead of giving a number. A case
whose results are not stepped carries none. A chart of a job's case, of one
case or of several, names in a `steps_not_as_recorded` observation a case
whose results file holds another number of steps than its record says.
