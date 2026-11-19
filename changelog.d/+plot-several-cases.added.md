`plot_waveform` draws several cases of one job in one chart: `runs`, beside
`job_id`, takes `"all"`, a list of run indices or `{case_ids: [...]}`, as an
`analyze_results` source does, and the two tools read the same cases for one
selection. Each trace's legend label names the assignments that differ among
the cases drawn and the step's values (`V(out) [R1=1000;r=22]`), each trace
summary names its `case_id` and `run_index`, and the reply lists the cases
drawn with their assignments and steps. A case with no results, without the
`step` asked for, or missing from the job is named in `cases_not_drawn`, and
a signal some cases lack is drawn for the others and named in
`signals_not_in_cases`. Cases of different analyses, and `in_ltspice` with
several cases, are refused with the reason.
