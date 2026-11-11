The server now reads an ngspice compatibility mode as ngspice does: `spe`
clears `ps`, `lt`, `ki` and `eg`, `ps` then clears `hs`, `mc` turns off
every flag but `s3`, and `s3` skips a `.lib <file> <section>` line. Before,
the `lib-section-ngspice` lint, the `ngspice_lib_section` failure and
hierarchy discovery read any `lt` or `ps` in a mode as dropping a `.lib`
section and nothing else as dropping one. So `run_experiments` refused a
sectioned deck in `spelt` or `ltmc`, where ngspice reads the section, and
let one through in `s3`, where it does not. Discovery also multiplied a
card's `m` by its call's in `hsps` and `hsmc`, where ngspice replaces it.
