Names inside a block placed on a schematic are now read alike on LTspice 26
and XVII. Such a run writes a part the sheet names `X1` or `MB1` under that
name (`Ix(x1:IN)`, `Id(mb1:M1)`) and a card a directive on the sheet writes
under its own, and the name does not say which an instance is.

- A name the result writes stands for that device, then that instance,
  first. One that matches nothing as written is read with an instance's X
  dropped, so the netlist's spelling (`Ix(xx1:in)`, `xmb1.m1.gm`,
  `device="xmb1"`, `device="xin"` on XVII) reads the part on either build,
  under every parent that has it. Where two such readings select different
  things, the read is refused and names each one instead of merging them.
- A device's table values and its currents are selected by the same names,
  so `device="mb1"` on an XVII sheet run keeps both, and `mb1.m1.gm` reads
  the value.
- The Python API's `compare` pairs a trace with the trace of its own name
  when both results are of one build and one kind of run, and otherwise by
  the instances each name may stand for, so two runs of one sheet on
  LTspice 26 and XVII now pair. A trace that could be more than one is not
  compared and is listed under `ambiguous` with the traces it could be,
  whether or not `signals` was given.
