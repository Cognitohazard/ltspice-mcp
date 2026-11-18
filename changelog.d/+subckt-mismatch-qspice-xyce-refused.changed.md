`run_experiments` now refuses a mismatch draw for a transistor inside a
subcircuit on QSPICE and Xyce, with the code
`mismatch_inside_subcircuit_unmodelled`. This covers a `mismatch` rule whose
prefix names an `X` instance, a rule with `instance`, and `X1:delvto` /
`X1:mulu0`. The server does not model how those simulators read a subcircuit
body, and the draw it made there read the device's size off the `X` line and
evaluated the body as ngspice would, so the reported spread could be several
times too large. Run such a deck on LTspice or ngspice; a mismatch rule on
top-level `M` devices still runs on QSPICE and Xyce. The codes that path alone
returned are gone: `ambiguous_inner_device`, `ambiguous_instance_ref`,
`ambiguous_scale`, `ambiguous_subckt`, `clone_include_unsupported`,
`geometry_not_literal`, `inner_device_not_found`, `inner_model_unresolved`,
`instance_not_found`, `invalid_instance_target`, `nested_fet_unsupported`,
`non_bsim_inner_device`, `param_namespace_collision`,
`preexisting_mismatch_param`, `subckt_unresolved`, `unencodable_device_ref` and
`unplanned_instance`.
