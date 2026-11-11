The server runs ngspice with `-n`, as recoverable and `pdk_native` runs
already did, so it no longer reads a user's `.spiceinit` or `spice.rc`
from `SPICE_USERINIT_DIR`, the working directory, `HOME` or `USERPROFILE`.
Such a file could replace the compatibility mode the server reads the deck
in (see Fixed). Everything else it did is gone too, so move it: the mode to
`[simulator] ngbehavior`, options to an `.options` line in the deck, and
commands such as `osdi` or `codemodel` to the deck's `.control` block as
`pre_osdi` or `pre_codemodel`, which run before ngspice reads the circuit.
The installation's `spinit`, which loads ngspice's own code models, is
still read.
