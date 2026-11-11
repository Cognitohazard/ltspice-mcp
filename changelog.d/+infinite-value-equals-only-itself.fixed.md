`verify_circuit`'s equivalence comparison counted a value written `inf` as
equal to every finite value, because its tolerance is a share of the larger
magnitude. An infinite value now equals only the same infinity.
