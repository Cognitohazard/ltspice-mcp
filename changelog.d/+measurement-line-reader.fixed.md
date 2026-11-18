A measurement whose name holds a dash, which LTspice XVII prints as
`t-fall: v(out)=0.5 AT ...`, is no longer counted under `unclassified` in a
run's `diagnostics`. The log classifier now asks the measurement reader which
lines print a measurement, so the two read the same lines.
