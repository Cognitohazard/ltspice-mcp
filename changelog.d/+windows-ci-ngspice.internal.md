CI: the `checks / check-windows` job installs the official Windows build
of ngspice (47, pinned by SHA-256) at `C:\Spice64` and puts its console
program on PATH, so the real-ngspice tests run on Windows too instead of
skipping there. That build writes ASCII raw files, as its `spinit` sets
`filetype=ascii`. Two are expected to fail there: a stopped run's partial
raw, which the server does not keep when the raw file is ASCII (its ASCII
variant fails everywhere), and the multiplier test's deck with a code model,
for which the ngspice that test starts reports no `spinit`.
