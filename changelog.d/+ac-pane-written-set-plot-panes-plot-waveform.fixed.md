An AC pane written by `set_plot_panes`, or by `plot_waveform` with
`in_ltspice`, lacked the two lines that switch its magnitude and phase
curves on (`PltMag: 1`, `PltPhi: 1 0`), which both builds write for an AC
pane they make. LTspice XVII opened such a pane with both curves off, so it
drew nothing, a trace added by hand included. Every AC pane written now has
them. A plot settings file written by an earlier version keeps its blank
pane until `set_plot_panes` writes it again.
