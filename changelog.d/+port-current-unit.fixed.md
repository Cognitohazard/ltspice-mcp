A block's port current, such as `Ix(u1:out)`, which LTspice declares with the
type `subckt_current`, is now read as a current in amperes: `plot_waveform`
charts it with its unit, `inspect` lists it in amperes, and an operating point
files it among the currents instead of "other". Every type a recorded LTspice
raw declares is now known, and a trace of a type the server does not know is
described with `unit_evidence: "declared_type_unknown"` instead of passing as
another quantity without a word.
