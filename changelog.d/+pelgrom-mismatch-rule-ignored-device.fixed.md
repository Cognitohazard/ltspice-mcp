A Pelgrom `mismatch` rule ignored a device's multiplier `m`. The simulator
builds m parallel copies from one card and they share its one draw, so
their area is `m·W·L`, but the draw took the area of a single copy: a device
written as `W=1u m=12` got a threshold spread √12 (about 3.5×) too large,
with no warning. The area now counts the device's own `m`, including an
inner `m={mult}` set from the X line, and under ngspice the `m` of every
subcircuit instance above it, on a top-level `M` card, a device one
subcircuit level down and a nested `instance` target alike. An `m` that does
not resolve to a positive number refuses the request with
`mismatch_geometry_unresolved` instead of being drawn as one copy. A refusal
while drawing for a device one subcircuit level down now reaches the caller
as a coded error rather than an uncoded one.
