The Python API's `compare` now pairs the traces of an LTspice XVII run of a
schematic with another run's. XVII writes a part the sheet names `X1` or
`MB1` under that name (`Ix(x1:IN)`, `Id(mb1:M1)`), and the server read every
XVII instance as one whose X was dropped, so such a trace did not pair. The
name does not say whether such an instance is a part of the sheet or a card a
directive on it wrote, so the trace now pairs with the one trace either
reading names; a trace that could be more than one is not compared, and is
listed under `ambiguous` with the traces it could be.
