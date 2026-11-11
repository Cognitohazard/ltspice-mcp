ngspice reads a deck in the compatibility mode the server models,
`[simulator] ngbehavior`. ngspice applies the `-D ngbehavior=` the server
passes before its startup files, so a `set ngbehavior` in a user's
`.spiceinit` replaced it. The run then used another mode than the one that
hierarchy discovery, the `lib-section-ngspice` lint rule and the failure
classifier read the deck in. For example, with a `.spiceinit` that set
`hsa`, ngspice built 12 copies of `R1 p 0 1k m=3` called by `X1 a r1 m=4`,
where `kiltpsa`, the configured mode, builds 4; and a sectioned `.lib`
that the lint refused ran without error. Every ngspice run now starts with
`-n` (see Breaking). The `ngspice_lib_section` hint no longer suggests a
`.spiceinit`, and names only `[simulator] ngbehavior`.
