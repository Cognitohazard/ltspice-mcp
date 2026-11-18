The package now requires NumPy 2.0 or later. It already called
`np.trapezoid`, which NumPy 1.26 does not have, so an install that resolved
NumPy 1.x failed in the signal, AC and loop-structure readers. The Python
guide says which NumPy 1 names fail in a `run_code` snippet and what to write
instead.
