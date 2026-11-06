In the Python API, `r.steps[0]["Rv"]` raised `KeyError` for a parameter
the deck steps as `Rv`, since LTspice logs it as `rv`. Each entry of
`RawResult.steps` now finds a name as a step selector does, whatever its
case, and keeps the keys as the run wrote them.
