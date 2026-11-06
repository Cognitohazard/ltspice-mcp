A stepped LTspice result that keeps each step's values in variables of
its own, a stepped `.op`, `.tf`, or `.ac` or `.noise` at one frequency,
lists those variables among its steps' values and no longer among its
signals: the Python API's `RawResult.signals`, `table()` and `compare()`,
and `inspect`'s signals and table views. Asked for as a signal, such a
variable is refused with where its values are.
