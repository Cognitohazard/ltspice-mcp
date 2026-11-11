Each ngspice run is checked against the compatibility mode the server
passed it and read the deck in, so a run that used to complete can now
fail. ngspice names the mode it read a circuit in at the top of its log
(`Note: Compatibility modes selected: ps lt ki a`). The server compares the
deck's note (the first; the second when a native setup's or a seeded run's
driver sources the deck) with the flags the same ngspice gives the passed
mode, which it learns by running ngspice once per mode with no startup
file. A run read in another mode, as when the installation's `spinit` sets
`ngbehavior`, fails with `ngspice_mode_mismatch`, whose evidence names both
and whose hint says where to look; its results used to come back as if
read in the configured mode. A run whose mode could not be checked keeps
its results and carries an `ngspice_mode_unverified` observation whose
evidence gives the reason: `no_mode_in_log` (ngspice's `mc` mode, an older
build), `deck_note_missing` (the driver's deck was not read),
`log_unreadable`, `ngspice_not_asked`, or `no_log` on a run that produced
results without one.
