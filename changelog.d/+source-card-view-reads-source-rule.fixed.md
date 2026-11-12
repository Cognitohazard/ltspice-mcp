A V or I source card is now read by one rule wherever the server reads it:
the words after its two nodes are its value, except its own parameters
(`Rser=`, `Cpar=`, an `AC` magnitude written `AC=2` or `AC 2`, and a current
source's `load`), which both LTspice builds read after any form of value.

- A source whose PWL has a keyword before its points, such as
  `V4 d 0 PWL TIME_SCALE_FACTOR=2 (0 4 1 4) Rser=1k`, is now listed by
  `inspect(kind="components")` and the netlist info. `verify_circuit` and the
  pre-run lint no longer warn `value-expression-remnant` on it.
  `inspect(kind="hierarchy")` no longer refuses a deck that holds one.
- The comparison behind `verify_circuit` now reports a change to such a
  keyword, as in `TIME_SCALE_FACTOR=2` to `TIME_SCALE_FACTOR=3`.
- A random component rule on a source whose value is followed by its own
  parameters, as in `V1 a 0 5V AC 1`, now draws the value and keeps `AC 1`.
  Before, the rule was refused.
- Setting a parameter of a source that is a keyword of its value, such as
  `TIME_SCALE_FACTOR`, is refused with a message to edit the value instead.
- The source-relative `extreme_value` observation is no longer turned off by
  a source with an `AC` magnitude after its value.
