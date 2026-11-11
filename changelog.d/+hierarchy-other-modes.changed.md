An ngspice mode other than `hsa`, `kiltpsa` and native (`""`) is no
longer refused by `inspect(kind="hierarchy")` outright. The query answers
when ngspice is installed and confirms every value discovery resolved, and
is otherwise refused with what the check found and the modes to use. In
such a mode discovery refuses `gnd`, whose reading there is not recorded.
Editing a deck by instance is still refused in such a mode, and a
recoverable run is refused before the job is recorded, with
`recovery_ngbehavior_unsupported`; both messages name the supported modes.
A query's `ngbehavior` is lower-cased as `[simulator] ngbehavior` is, so
`HSA` names `hsa` in both.
