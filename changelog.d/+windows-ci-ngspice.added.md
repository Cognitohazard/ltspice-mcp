CI: the `checks / check-windows` job installs the official Windows build
of ngspice (47, pinned by SHA-256) at `C:\Spice64` and puts its console
program on PATH, so the real-ngspice tests run on Windows too instead of
skipping there. That build writes ASCII raw files, as its `spinit` sets
`filetype=ascii`. Three are expected to fail there until they are fixed:
two that read ASCII raw files, which also fail in their ASCII variant
everywhere (an AC run that saves a device parameter, `docs/spicelib_bugs.md`
Bug 42, and a stopped run's partial raw), and the multiplier test's deck with
a code model, for which the ngspice that test starts reports no `spinit`.
