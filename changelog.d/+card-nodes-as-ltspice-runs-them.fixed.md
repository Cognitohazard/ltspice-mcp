The server read some cards' nodes differently from LTspice, so `inspect`,
`verify_circuit`'s comparison and node checks, and the lint before a run
could miss a node or count a word that is not one. They now read each card
as LTspice 26 and XVII were recorded running it:

- a source that plays a .wav file (`V1 a b wavefile="in.wav"`) has both
  words as its nodes, where the second was read as its value;
- a lossless line (`T1 a 0 b 0 Td=1n Z0=50`) has four nodes and no model,
  where the fourth node was read as a model;
- a current-controlled switch (`W1 a 0 Vsense sw`) has two nodes, where the
  source that sets it was read as a third, and a change of that source now
  shows in a comparison;
- a MESFET's word after its model (`Z2 d g s nmf 2`) is its area, where it
  was read as the model and the model as a fourth node;
- LTspice 26's frequency response analyzer (`@1 a b ...`) has two nodes,
  where the second was read as a model.

Such cards with too few nodes are now refused by the `element-arity` check.
