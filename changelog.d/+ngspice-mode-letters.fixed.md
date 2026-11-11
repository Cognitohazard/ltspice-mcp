The server now reads an ngspice compatibility mode as ngspice does. ngspice
matches a mode's letters case-sensitively, `spe` clears `ps`, `lt`, `ki` and
`eg`, `ps` then clears `hs`, and `mc` turns the others off. Before, the
`lib-section-ngspice` lint, the `ngspice_lib_section` failure, and hierarchy
discovery read any `lt` or `ps` in any case as dropping a `.lib` section, so
`run_experiments` refused a sectioned deck in `spelt` or `LTPS`, where ngspice
reads the section. Discovery also multiplied a card's `m` by its call's in
`hsps`, `HSA` and `hsmc`, where ngspice replaces it.
