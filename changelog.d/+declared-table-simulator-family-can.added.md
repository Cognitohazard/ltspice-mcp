A declared table of what each simulator family can do: every capability
the server offers, marked baseline or refinement, is `supported` with its
evidence (a recorded LTspice behaviour, a file another simulator wrote, or
a test that runs it), a `gap` with the reason, or `unverified`. The first
declared gaps: QSPICE and Xyce have no lint or validator rules, no
hierarchy reader (so no nested instance targeting) and no controlled
startup; ngspice has no `.step` and evaluates no top-level `.meas` in a
batch run. `inspect(kind="capabilities")` reports the detected families'
statuses under `simulator_capabilities`, and the `spice://capabilities`
resource (and `spice://capabilities/{family}`) serves the whole table with
its evidence. The refusals of a QSPICE run off Windows and of a schematic
run on QSPICE or Xyce now read the table.
