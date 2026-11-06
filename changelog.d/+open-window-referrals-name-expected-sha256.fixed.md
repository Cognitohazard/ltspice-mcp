Replies that send the caller to `edit_schematic`'s `open_window` named a
call that was refused: `plot_waveform`'s note on a plot not tied to its
sheet, `verify_circuit`'s hint on a window showing another sheet than the
file, and `edit_schematic`'s note on a window left showing the sheet as it
was all said `ops=[]` and `open_window` without `expected_sha256`, and the
call was refused as `expected_sha256_required`. Every such reply now writes
the call out with each argument it needs, and says where to find the sha256:
the one the refused call sent, the reply's own after a commit, or the one
`inspect(kind="open_in_ltspice")` gives for the sheet. A refusal under two
windows that hold different copies names the save with `window_pid` and the
discard without it as two calls, where one sentence named both.
