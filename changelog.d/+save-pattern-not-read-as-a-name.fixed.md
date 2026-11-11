`run_experiments` refused a deck whose `.save` named signals by a pattern, such
as `.save V(*)`: the `save-meas-coverage` lint read `V(*)` as a node named `*`
and reported every node a `.meas` reads as unsaved, at error severity. Which
signals a pattern saves is not recorded for LTspice, so beside a pattern the
lint no longer refuses: a new warning, `save-pattern-unrecorded`, lists the
signals it did not check and says that naming them in the `.save` has them
checked. A `.save` that names its signals one by one is held as before.
