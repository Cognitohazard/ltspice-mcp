The `save-meas-coverage` lint refused a `.meas` on `V(a,b)` in a deck whose
`.save` lists `V(a)` and `V(b)`, though both LTspice builds measure it, and
`.save V(a,b)` saves `V(a)` and `V(b)` (recorded:
`deck/save_pair_both_nodes`, `deck/save_pair_as_pair`). A voltage between
two nodes, neither of them ground, now stands for its two nodes in the
`.meas` and in the `.save`. A voltage against ground is compared as
written, since neither build measures `V(a,0)` with only `V(a)` saved
(recorded: `deck/save_pair_to_ground`). The linter version is 11.
