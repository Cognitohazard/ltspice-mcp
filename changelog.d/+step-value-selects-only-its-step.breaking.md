A `.step` value given as a number selects only a step with that value, to the
six significant digits LTspice's log prints a step value with, however it is
written (`"1k"`, `1e3` and `1000` select a step of 1000, and the deck's
2154.4346900318838 the step the log prints as 2154.43), in `analyze_results`'
and `run_experiments.analyze`'s `step`, `plot_waveform`'s `step` and the Python
API's `analyze_results`. A step that holds the value itself is chosen over a
neighbour within those digits, and a value the digits would join to two steps
the run holds apart is refused. A value no step has is refused with the values
the run has, as a text value already was. Before, a number selected the nearest step
without saying so, so `"6.8k"` on a run stepping 1k, 2k and 4k read the run at
4k. `RawResult.compare` pairs a step with the other run's by the same rule,
finding a step parameter's name in any case, and its refusal now lists the
other run's steps.
