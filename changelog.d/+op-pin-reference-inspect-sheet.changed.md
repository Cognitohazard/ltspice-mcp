**Breaking:** every op, every pin reference and `inspect(kind="net")` on a
sheet name a part by its reference in any case, as LTspice compares
references: the reference as spelled, else the one that differs from it
only in case. A reference that names no part is refused with one message
that lists the parts there are.
