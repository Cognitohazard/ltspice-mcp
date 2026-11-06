A `mismatch` rule reaches a MOSFET inside a subcircuit on LTspice. LTspice
takes neither `mulu0` nor `delvto` on a BSIM3 device, so the device gets its
own copy of its `.model` card inside a private copy of its subcircuit, with
the drawn shift and scale applied to the rule's `vth_param` and `k_param`,
as a top-level device does. Any model level works if the model has one
definition; a binned model is refused with `model_missing`. LTspice ignores
`m` on an X line, so the draw's area counts only the device's own `m`.
Such a rule used to be refused with `unsupported_backend`, a code the server
no longer returns.
