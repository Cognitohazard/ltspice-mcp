A capacitor or inductor defined by its charge or flux (`C1 b 0 Q=100n*x`,
`L1 c 0 Flux=1m*x`) was read as one node and a value of 0, so
`inspect(kind="net")` left it off ground and the equivalence compare called
a rewiring of it a value change. Both LTspice builds run it between its two
nodes (recorded: `deck/keyword_values`), and both are now read.
