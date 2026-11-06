A signal named `V(<instance>:<pin>)` for a subcircuit's pin is refused with
the reason: a result holds no voltage under a pin's name, because the pin's
voltage is the net it connects to in the circuit above. The refusal lists
the instance's pins in the order its instance line lists their nets, where
LTspice wrote them, and names the current into the pin. Every reader of a
signal says so, `analyze_results` on an operating point included.
