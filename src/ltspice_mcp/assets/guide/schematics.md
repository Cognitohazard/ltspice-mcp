---
name: schematics
description: >
  Building or editing an LTspice `.asc` schematic: the workflow, symbol pin
  offsets, MOSFET orientation, and layout practice.
---

# LTspice `.asc` Schematics

## Design workflow: .cir first, .asc last

**Design and iterate over `.cir` netlists** — plain text, no placement overhead, fast to edit and simulate. Only build `.asc` schematics after the circuit design is finalized or when the user needs a visual schematic for review. The `.asc` tools are for presentation, not design iteration.

## Building and editing a sheet

`.asc` files are structured text representing the schematic graphically. While technically readable, hand-editing is error-prone — use `edit_schematic` or LTspice's GUI. It gives geometry-aware editing (orthogonal routing, pin-collision and junction checks) that hand-writing the file can't match. `edit_schematic(target=..., base="blank")` starts a new sheet; every mutation below is an entry in its `ops` list, applied as one guarded transaction, so batch a whole build into one call. Place components with the `add_component` op; the response's `touched` view gives their placed pins and nets, `warnings` any overlap, and `inspect(kind="symbol")` previews the same geometry before you place anything.

- Component attributes: Value, Value2, SpiceLine, SpiceLine2.
- Export to netlist for direct text editing when needed.
- Bus notation: `Data[0:7]` creates 8 nets (cosmetic — netlister flattens to individual nets).

### Common symbol pin offsets (at R0)

| Symbol | Pins (name: x,y) | Size (WxH) |
|-|-|-|
| nmos | D:(48,0) G:(0,80) S:(48,96) | 48x96 |
| pmos | D:(48,0) G:(0,80) S:(48,96) | 48x96 |
| voltage | +:(0,16) -:(0,96) | 64x80 |
| current | +:(0,0) -:(0,80) | 64x80 |
| res | A:(16,16) B:(16,96) | 32x80 |
| cap | A:(16,0) B:(16,64) | 32x64 |

Rotations transform pin (x,y) as: R90→(-y,x), R180→(-x,-y), R270→(y,-x), M0→(-x,y), M90→(y,x), M180→(x,-y), M270→(-y,-x). Use `inspect(kind="symbol")` for exact positions.

A pin is addressed as `REF.PIN` (`M1.D`) by its name. When no pin has that name and it is all digits, it is the pin's 1-based SpiceOrder, the terminal number a netlist uses, so `X1.2` reaches the second pin of a block whose pins are lettered. Names are matched first because some symbols name their pins `1`/`2` in an order that need not be their SpiceOrder. `inspect(kind="symbol")` lists each pin's `name` and `order`.

**3- vs 4-terminal devices**: The basic `nmos`/`pmos` and `npn`/`pnp` symbols are 3-terminal — a MOSFET's bulk ties internally to its source, and a BJT has no separate substrate pin. When you need the body/substrate on its own net (e.g. a non-source bulk bias), use the 4-terminal variants (`nmos4`/`pmos4`, `npn4`/`pnp4`), which expose bulk/substrate as a 4th pin.

### MOSFET orientation conventions

| Rotation | Gate side | D/S vertical | Typical use |
|-|-|-|-|
| R0 | Left | D top, S bottom | NMOS (drain up) |
| M0 | Right | D top, S bottom | NMOS mirrored (symmetric diff pair) |
| M180 | Left | D bottom, S top | PMOS (source to VDD at top) |
| R180 | Right | D bottom, S top | PMOS mirrored (gate faces right) |

**Choose orientation based on where the gate connects:**
- Gate wire must not cross through the component's own body. Pick the rotation that puts the gate on the side facing the signal source.
- Example: if M3's gate connects to M5 on the right → use M0 (gate right), not R0 (gate left).
- For diff pairs: M1 at R0 (gate left, toward Vinp), M2 at M0 (gate right, toward Vinn).
- For PMOS current mirrors: M4a at R180 (gate right, toward center), M4b at M180 (gate left, toward center) — gates face each other.
- Use `inspect(kind="symbol")` and read the pins for the intended rotation to verify pin directions before placing.

### Schematic layout best practices

**Delegate the build when you can.** Placement and wiring is detailed,
mechanical work. An agent doing design and layout in one pass tends to tag
pins with net labels instead of routing wires. If your environment supports
subagents, hand the schematic build to one whose only brief is this section: give it the final netlist, have it
read guide section 'schematics', require it to build with `edit_schematic` (never by hand-writing the
`.asc`), and have it verify before returning — `verify_circuit` with the
`export` and `compare` checks must match the source netlist, and
`inspect(kind="net")` must show no multi-label shorts. Review the result with
`inspect(kind="components")`.

**Component placement:**
- **Tier alignment**: Matched/mirrored transistors (diff pairs, current mirrors, bias mirrors) must share the same y-coordinate. Plan horizontal tiers: VDD rail → PMOS loads → diff pair → tail/bias → VSS.
- **Drain/source alignment on each branch**: Within a vertical branch (e.g., PMOS load stacked above NMOS input), position components so the drain pin of the upper device is on the same x-column as the drain pin of the lower device. This eliminates horizontal jogs between stacked transistors.
- **Pin-to-rail alignment**: Place voltage/current sources so their pins land directly on the rail they connect to — no wire through the source body. For a VDD source, position it so the `+` pin y-coordinate equals the VDD rail y-coordinate. Use `inspect(kind="symbol")` to compute the exact placement origin from the desired pin position (e.g., for voltage `+` at y=128, place origin at y=128-16=112).
- **Minimum 128 units vertical spacing between pin levels** of adjacent tiers (e.g., between PMOS drain y and NMOS drain y). This leaves room for horizontal buses and net labels between tiers. With MOSFET bbox height of 96, plan tier origins ~192 units apart.
- **Bias circuit alignment**: Bias devices (e.g., M5/Ibias) should share the y-level of their functional counterpart (e.g., M3 tail current source).
- **Plan the full layout before placing**: Decide VDD rail y, tier y-coordinates, and bus y-coordinates first. Verify that buses fit between bounding boxes of adjacent tiers. Use `inspect(kind="symbol")` to check bbox extents at the intended rotation.

**Wiring:**
- **All wires must be orthogonal** — strictly horizontal or vertical. Never route diagonal wires. Use waypoints in the `wire_pins` op for L-shaped or multi-segment routes.
- **Horizontal buses must route outside all component bounding boxes.** Use `inspect(kind="symbol")` to check bbox extents. For PMOS M180 with bbox top at y=160, a gate bus at y=176 is inside the bbox — route at y=144 (between VDD rail and bbox top) instead. Plan bus y-coordinates before placing components.
- **Vertical wires must not pass through component bodies to reach a bus.** When connecting a drain to a horizontal bus, jog the wire horizontally outside the bbox first, then route vertically to the bus. Example for PMOS M180 diode connection: route drain (400,256) → right to (448,256) → up to (448,144) → along bus to label, not straight up through the body at x=400.
- **Leave room for buses between tiers.** The minimum 128-unit tier spacing must account for bounding box height plus bus clearance. For PMOS M180 (bbox height 96), if VDD rail is at y=128 and PMOS origins at y=288: bbox occupies y=192–288, bus fits at y=144–160 (between rail and bbox top).
- **Tap an existing wire with a T-junction**: give the `wire_pins` op a coordinate endpoint on the wire, `to_pin: {"x": 240, "y": 196}`. LTspice joins a wire end, pin or label anywhere along a wire, so the wire stays whole and a `remove_wire` op on the new segment undoes it. The op's entry in the response's `results` names the wire it joined under `junctions`. Two wires that only cross are not joined. A waypoint that touches another net's wiring is refused rather than silently merging it; `inspect(kind="net")` at a point on a wire traces that wire (`snapped_to_wire`).
- **Heed the `wire_pins` op's warnings and errors**: it refuses diagonal wires, pin collisions, and wire junction overlaps. Non-blocking warnings (long runs, bbox crossings) should still be addressed.
- **Read the `wiring` profile `edit_schematic` returns.** It reports `pins_wired` and `pins_label_only` out of `pins_total`. `pins_label_only` high with `wire_segments` near zero means you tagged pins with net-labels instead of drawing wires. That is a wiring list, not a routed schematic, and whether it connects as intended depends only on the label names, which the profile does not check. Draw wires with the `wire_pins` op for local nets; reserve net-labels for ground, power rails, and distant nets. Also heed the `label_over_component` validation warning (a net-label whose anchor fell inside a symbol's bounding box).
- **On an existing sheet, the reported findings are the edit's.** `warnings` and `wiring.label_only_pins` list only what your ops introduced or named; older ones are counted in `preexisting`, not listed. Before calling a sheet done, list them with `return_views: ["preexisting"]` (on an op-less read, `ops: []`, that is the whole sheet) or run `verify_circuit`.

**Ground and net labels:**
- **Local ground flags**: Place a ground (`0`) label directly at each grounded pin via an `edit_schematic` `add_net_label` op. Never route wires to a distant ground flag.
- **One ground per pin**: Each component's ground connection gets its own `add_net_label` op at the pin's coordinates — do not share ground flags between components.
- **Do not use the `wire_pins` op with `net:0`** when multiple ground labels exist — it errors on ambiguous net references. Place ground flags directly at pin coordinates with an `add_net_label` op (`net="0", pin="M3.S"`) — no wire needed when the flag is on the pin.
- **Named nets (VDD, outp, etc.)**: Repeating the same label at distant pins ties them together — the netlist merges same-name labels into one net (correct, not a short), and no routing is needed. Wire nearby pins with the `wire_pins` op. Caveat: once a name carries duplicate labels, `wire_pins` with `net:NAME` is ambiguous — target a component pin (`Ref.Pin`) instead.
- **Label any net you reference by name in a directive.** The `wire_pins` op wires pins but assigns no name — at export an unlabeled net becomes `N001`, `N002`, …. So a `.meas V(vref)`, a `.param` expression using `V(x)`, or a behavioral `B`-source referencing `V(name)` silently breaks unless that exact net carries an `add_net_label`. A wire with no label is enough for a net you never name; **label any net a directive mentions by name.**

**Sources:**
- **Voltage source polarity**: `+` pin is at the top (smaller y), `-` at bottom. For VDD sources, `+` connects to the supply rail, `-` to ground.
- **Current source direction**: Current flows from `+` (top) to `-` (bottom) externally. Place with `+` on the higher-voltage rail.

**Models:**
- **Model names must not collide with type keywords**: Use `NMOS_3V3` not `NMOS` for `.model` names when the symbol Value is also a MOSFET type.
- **Diode default-model collision**: The `diode` symbol defaults its Value to `D`, which LTspice resolves to a built-in ideal diode. Adding your own `.model D D(...)` collides with that built-in — give the model a unique name (`.model MYDIODE D(...)`) and set the symbol's Value to `MYDIODE`, rather than reusing `D`.
