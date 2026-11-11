The server runs ngspice with `-n`, as recoverable and `pdk_native` runs
already did, so it no longer reads a user's `.spiceinit` or `spice.rc`
from `SPICE_USERINIT_DIR`, the working directory, `HOME` or `USERPROFILE`.
Such a file could replace the compatibility mode the server reads the deck
in (see Fixed). Everything else it did is gone too, so move it: the mode to
`[simulator] ngbehavior`, options to an `.options` line in the deck, and
commands such as `osdi` or `codemodel` to the deck's `.control` block as
`pre_osdi` or `pre_codemodel`, which run before ngspice reads the circuit.
Drop a `seed` option rather than move it: a seed in the deck makes a
`random` variation with `simulator_draws` or a run with
`execution.simulator_seed` refuse the deck, and those set the seed.
The installation's `spinit`, which loads ngspice's own code models, is
still read.
