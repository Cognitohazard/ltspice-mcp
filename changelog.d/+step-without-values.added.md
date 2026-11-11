The lint rule `step-without-values-ltspice` refuses an LTspice deck with a
`.step` that has nothing to step (`.step` alone, or `.step param r` with no
values), which both LTspice builds refuse before the run. `edit_schematic`
refuses to write such a directive, and `verify_circuit`'s `syntax` check
reports it.
