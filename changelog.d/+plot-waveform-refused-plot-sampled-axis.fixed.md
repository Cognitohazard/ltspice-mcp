`plot_waveform(in_ltspice=true)` refused a plot with no sampled axis (a
stepped `.op` or `.tf`, or a stepped `.ac` or `.noise` at one frequency,
which hold one point a step), so no call could put such a run of a sheet
beside the sheet for LTspice to open. It now opens one in LTspice without
a chart, from its sheet when it ran one, names no traces for it and says
so in a `no_chart` observation. Without `in_ltspice` the refusal is now
`no_axis`, says what the plot holds (one point a step, and the stepped
parameter it starts with) and where its values are read, and no longer
appends the generic hint to check that the run finished.
