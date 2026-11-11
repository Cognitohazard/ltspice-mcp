A run on ngspice configured with a mode other than `hsa`, `kiltpsa` or
native (`""`), such as the `kia` the `ngbehavior` setting's own
documentation suggests, no longer fails every case with `submission_failed`.
Building the hierarchy profile for the deck refused the mode, though a run
without instance edits never reads the deck through it.
