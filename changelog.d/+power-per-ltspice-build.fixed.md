A power in a parameter expression read for LTspice follows the build that runs
the deck. `0**0` is 1, and `-2**2` is 4, as LTspice binds a sign to the number
after it. A negative base with a fractional exponent is 0 on LTspice 26 and
refused on LTspice XVII, so it is read as 0 only where the build is known to
be LTspice 26, and declined otherwise; a zero base with a negative exponent is
declined. An LTspice element value with no braces that is neither a number nor
one name is reported as the refusal LTspice gives it, with the braced
spelling to use.
