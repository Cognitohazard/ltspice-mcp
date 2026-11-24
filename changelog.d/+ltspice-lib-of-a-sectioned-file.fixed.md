A `.lib` card with no section that names a file declaring sections
(`.lib tt` ... `.endl tt`) was read as the whole file for LTspice. LTspice 26
and XVII read the file's own `.lib tt` as a library file named `tt`, find
none and run nothing. `run_experiments` now refuses such a deck (lint
`lib-section-ltspice`), and `inspect` and `verify_circuit` read none of the
file's cards.
