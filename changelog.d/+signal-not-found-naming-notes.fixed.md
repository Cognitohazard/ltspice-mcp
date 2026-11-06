A signal that read no trace because LTspice names it otherwise got no
explanation. Every reader of a signal by name (`analyze_results`,
`plot_waveform`, `RawResult.trace`, `load_raw(signals=...)` and the `value`
recipe on an operating point) now says so in two more cases. When a name
misses only by the X of an instance on its path, the refusal names the one
trace it matches and why: LTspice names a part placed on a schematic by its
reference (block X1 is `Ix(x1:in)`, not the netlist's `Ix(xx1:in)`),
LTspice 26 keeps the X of an instance a deck calls with X (`Id(xa:Mdrv)`),
and LTspice XVII drops it from every instance. A pin of an instance asked
for under its other spelling (`V(xx1:in)`) is told it is a pin, under the
instance's name in the result. When a result from a schematic holds no
`V(<name>)` but holds nets LTspice named itself, the refusal lists them
(`V(n001)`, ...) and says that a net label on the net gives it the name. A
result counts as a schematic's when its title names a `.asc` (LTspice XVII)
or a `.net` (LTspice 26).
