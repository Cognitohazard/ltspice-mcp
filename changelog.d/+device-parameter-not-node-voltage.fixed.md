ngspice writes a saved device parameter that it types as a voltage, such as
`.save @m1[vth]`, as the trace `v(@m1[vth])`, and the variable of a `.dc V1`
sweep as `v(v-sweep)`. The operating point already filed `v(@m1[vth])` with
the device's parameters, but three other readers took a `v(` name for a node
voltage. When an ngspice log reported `gmin stepping failed` or `source
stepping failed` and the raw held no node voltage, only such traces, the
summary turned those errors into warnings that said the run "produced finite
node data". The summary's value scan compared a device parameter with the
deck's voltage sources, so a 1.8 V threshold on a deck with 20 mV supplies
was reported as about 90 times the largest source. An AC summary with no
`signal` named could compute its bandwidth on a device parameter. Every
reader now uses one rule: a plot's sweep variable and a device parameter are
never node voltages, nor is any trace of a noise plot (LTspice types its
noise densities as voltages), a trace's declared type decides otherwise, and
a name is read only when the trace has no type. A stepping failure in a run
that saved no node voltage stays an error, and an AC summary with no node
voltage computes no default bandwidth and says why. The operating point read
from a `.dc` sweep now lists the sweep variable (`V1` from LTspice,
`v(v-sweep)` from ngspice) under `other` rather than `voltages`;
`sweep_value` still gives it when `at` is set.

Separately, ngspice's own line that a stepping method found the bias point
(`Source stepping completed`) is now read as the recovery it is, as LTspice's
are, so the failed rungs before it are no longer reported as the run's
errors. ngspice's `Transient op finished successfully` is not read so: ngspice
prints it for a circuit with no DC solution too.
