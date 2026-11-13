A name means the same thing to every tool, told who reads the deck or wrote
the result:

- A part LTspice exported as `R§Load` is read by its own trace: `I(R§Load)`
  reads `I(Load)` in `analyze_results`, `plot_waveform` and the Python API's
  `trace` and `load_raw(signals=...)`, and the `device=` filter of the
  operating-point recipe finds it. A name that misses a trace only by its
  element letter (`I(RLoad)` against LTspice's `I(Load)`, or the reverse) is
  refused naming the trace it means.
- For a session that runs ngspice in its native or hsa mode,
  `verify_circuit`'s connectivity comparison and its floating-node,
  dangling-node and undefined-reference checks, and `inspect`'s net query,
  read a subcircuit port named `gnd` as ground and a port named like a
  `.global` net as that net, as ngspice runs them; they read both as the port
  before. In kiltpsa mode, where ngspice's reading of `gnd` differs between
  versions, they keep LTspice's reading.
- `run_experiments` reads a deck's subcircuits with the spelling of the
  LTspice build that runs it, XVII's included, as `inspect(kind="hierarchy")`
  already did.
