`verify_circuit` with `checks=["compare"]` on an `.asc` compared nothing,
even with LTspice: it skipped the export as not requested, then skipped the
comparison for want of the exported netlist it reads. Asking to compare a
sheet now asks for its export too, which runs and reports as a check that
ran, with its `export` block and findings. Without LTspice the `hint` names
the export as the check that could not run, with the comparison it stopped.
