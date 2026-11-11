CI: the `checks / check-windows` job installs the official Windows build
of ngspice (47, pinned by SHA-256) and puts its console program on PATH, so
the real-ngspice tests run on Windows too instead of skipping there. That
build writes ASCII raw files, as its `spinit` sets `filetype=ascii`.
