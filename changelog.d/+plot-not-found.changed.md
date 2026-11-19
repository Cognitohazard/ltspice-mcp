A `plot_index` a results file does not hold is refused by `inspect` and
`plot_waveform` with the code `plot_not_found`, naming how many plots the file
holds, in place of `result_unreadable`: the file can be read, and the request
does not fit it. A chart of several cases names each case that lacks the plot
the same way. `analyze_results` gives such a recipe's failure the code
`recipe_failed`, as before, with the new message.
