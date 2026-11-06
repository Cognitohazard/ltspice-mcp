`jobs(action="place_beside_sheet", job_id | request_id, case_id?)` copies a
finished case's results beside the `.asc` it ran, under the sheet's name,
where LTspice keeps a sheet's own results and opens them from: the raw, the
log and each other raw the run wrote. The reply lists each file copied and
whether it replaced one, and the case's `assignments`, which the sheet does
not show. A netlist's run, another simulator's run, a case with no results,
a sheet changed since the run and a job whose cases are still running are
refused. The job keeps its own copy, and the sheet's `.plt` is left as it is.
