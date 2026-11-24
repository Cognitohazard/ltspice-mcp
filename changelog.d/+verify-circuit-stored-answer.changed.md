`verify_circuit` sends every finding when its reply fits `[analysis]
default_budget` (4000 estimated tokens), instead of at most 25 a rule. A reply
over it stores the complete answer in the store's results folder, cuts the
findings from the end, and adds `cut` and `stored_answer`; its hint names the
stored file, the call with `findings_cursor`, and the Python call. Reading on
with `findings_cursor` slices the stored answer and runs no check, so it no
longer fails as stale after the file changed: the page carries the snapshot
the answer was computed from. `findings_omitted` keeps its shape, filled from
the cut, until a later breaking release.
