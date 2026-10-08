---
name: schematics
description: >
  Building or editing an LTspice `.asc` schematic: the workflow, symbol pin
  offsets, MOSFET orientation, and layout practice.
---

# LTspice `.asc` Schematics

## Building and editing a sheet

Settle the design as a netlist first, then build the sheet, which is for
presentation and review. Edit `.asc` files only through `edit_schematic` (or
LTspice's GUI): it routes wires orthogonally and checks pin collisions and
junctions. `edit_schematic(target=..., base="blank")` starts a new sheet; every
mutation below is an entry in its `ops` list, applied as one guarded
transaction, so batch a whole build into one call. Place components with the
`add_component` op; the response's `touched` view gives their placed pins and
nets, `warnings` any overlap, and `inspect(kind="symbol")` previews the same
geometry before you place anything. It also lists every attribute the
symbol carries (`Prefix`, `SpiceModel`, `Value`, `SpiceLine`, `ModelFile`):
the model name and parameters an instance is netlisted with.

When the user says "this circuit" and names no file, ask what they have open:
`inspect(kind="open_in_ltspice")` lists the sheets and netlists open in LTspice
and marks the one in front `active`. Work on that path as on any other. A
sheet with `differs_from_file: true` has changes in the window that are not in
the file, so a run would simulate something else than they see: say so, and
ask them to save.

When they ask to see a sheet you built or changed, `verify_circuit(path=...,
in_ltspice=true)` opens it in their LTspice window, in front. If the reply's
`ltspice` block says `differs_from_file: true`, the window already had the
sheet open and is showing an older copy, not the one you checked: tell them.
With no LTspice window open, `in_ltspice` starts LTspice (`started: true` in
the block), which comes to the front of their screen: pass it when they asked
to see something there, never to check your own work. Where the block says
none was started, starting is turned off (`[schematic] start_ltspice`): ask
them to open LTspice.

When they want to plot nets by clicking the sheet in LTspice, show them a run
of the sheet: `plot_waveform(job_id=..., signals=[...], in_ltspice=true)` for a
job that ran an `.asc`. The run's results are put beside the sheet under its
name, replacing the ones there, and opened from the sheet, which is what makes
LTspice tie the plot to it: the reply's `ltspice` block names the `sheet`, and
a click on a net there plots it. Results beside a sheet of the same name,
which is what a run in LTspice leaves, open the same way by `raw_file`. Two
replies ask something of the user. LTspice already has that plot open, and
goes on showing what it read: they close it there, and you ask again. Or
LTspice had the sheet open before it had any results, and looks for them only
as it opens a sheet: they close the sheet there, and you ask again, since the
results are beside it now. A run of a netlist has no sheet to tie to and opens
on its own, as before.

LTspice ships an MCP server of its own, which a session may have beside this
one (its tools include `set_design_content` and `start_simulation`). It works
on the window, not the file. Make edits with `edit_schematic`: a sheet changed
through LTspice's server is changed in the window only, and `edit_schematic`
then refuses that sheet until the user saves it. Run with `run_experiments`:
a run started through LTspice's server is of the window's copy and leaves no
job. When one of its tools answers that it could not start or attach an
LTspice instance, no LTspice window is open: do not try its `start_headless`,
show the user what they asked for with `in_ltspice`, which opens one.

A sheet the user has open in LTspice (Windows, LTspice 26.1 or later) is kept
in step: a commit's reply lists the window under `open_in_ltspice`, and with
`shown: true` the user is already looking at the edit. `shown: false` means
the window still shows the old sheet and a save from it would overwrite yours,
so tell the user to close it without saving and open it again. An edit that
fails as `open_window_differs` wrote nothing: the window's copy is not the
file's. Only the user can settle that, so ask them to save the sheet (Ctrl+S)
or close it, then read its `sha256` again and resubmit.

- Component attributes: Value, Value2, SpiceLine, SpiceLine2.
- Bus notation: `Data[0:7]` creates 8 nets (cosmetic — netlister flattens to
  individual nets).

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

A pin is addressed as `REF.PIN` (`M1.D`) by its name. When no pin has that name
and it is all digits, it is the pin's 1-based SpiceOrder, the terminal number a
netlist uses, so `X1.2` reaches the second pin of a block whose pins are
lettered. Names are matched first because some symbols name their pins `1`/`2`
in an order that need not be their SpiceOrder. `inspect(kind="symbol")` lists
each pin's `name` and `order`.

**3- vs 4-terminal devices**: The basic `nmos`/`pmos` and `npn`/`pnp` symbols
are 3-terminal — a MOSFET's bulk is tied to its source in the exported netlist,
and a BJT has no separate substrate pin. When you need the body/substrate on
its own net (e.g. a non-source bulk bias), use the 4-terminal variants
(`nmos4`/`pmos4`, `npn4`/`pnp4`), which expose bulk/substrate as a 4th pin.

### MOSFET orientation conventions

| Rotation | Gate side | D/S vertical | Typical use |
|-|-|-|-|
| R0 | Left | D top, S bottom | NMOS (drain up) |
| M0 | Right | D top, S bottom | NMOS mirrored (symmetric diff pair) |
| M180 | Left | D bottom, S top | PMOS (source to VDD at top) |
| R180 | Right | D bottom, S top | PMOS mirrored (gate faces right) |

**Choose orientation based on where the gate connects:**
- Gate wire must not cross through the component's own body. Pick the rotation
  that puts the gate on the side facing the signal source.
- Example: if M3's gate connects to M5 on the right → use M0 (gate right), not
  R0 (gate left).
- For diff pairs: M1 at R0 (gate left, toward Vinp), M2 at M0 (gate right,
  toward Vinn).
- For PMOS current mirrors: M4a at R180 (gate right, toward center), M4b at
  M180 (gate left, toward center) — gates face each other.
- Use `inspect(kind="symbol")` and read the pins for the intended rotation to
  verify pin directions before placing.

### Schematic layout best practices

**Delegate the build when you can.** Placement and wiring is detailed,
mechanical work, and an agent doing design and layout in one pass tends to tag
pins with net labels instead of routing wires. If your environment supports
subagents, give one the final netlist and guide section 'schematics' as its
whole brief. It builds with `edit_schematic`, never by writing the `.asc`
itself, and verifies before returning: `verify_circuit` with the `export` and
`compare` checks must match the source netlist, and `inspect(kind="net")` must
show no multi-label shorts. Review the result with `inspect(kind="components")`.

**Component placement:**
- **Tier alignment**: Matched/mirrored transistors (diff pairs, current
  mirrors, bias mirrors) must share the same y-coordinate. Plan horizontal
  tiers: VDD rail → PMOS loads → diff pair → tail/bias → VSS.
- **Drain/source alignment on each branch**: Within a vertical branch (e.g.,
  PMOS load stacked above NMOS input), position components so the drain pin of
  the upper device is on the same x-column as the drain pin of the lower
  device. This eliminates horizontal jogs between stacked transistors.
- **Pin-to-rail alignment**: Place voltage/current sources so their pins land
  directly on the rail they connect to — no wire through the source body. For a
  VDD source, position it so the `+` pin y-coordinate equals the VDD rail
  y-coordinate. Use `inspect(kind="symbol")` to compute the exact placement
  origin from the desired pin position (e.g., for voltage `+` at y=128, place
  origin at y=128-16=112).
- **Minimum 128 units vertical spacing between pin levels** of adjacent tiers
  (e.g., between PMOS drain y and NMOS drain y), so a horizontal bus and its
  labels fit between one tier's bounding boxes and the next. With a MOSFET bbox
  height of 96, plan tier origins ~192 units apart.
- **Bias circuit alignment**: Bias devices (e.g., M5/Ibias) should share the
  y-level of their functional counterpart (e.g., M3 tail current source).
- **Plan the full layout before placing**: Decide VDD rail y, tier
  y-coordinates, and bus y-coordinates first. Verify that buses fit between
  bounding boxes of adjacent tiers. Use `inspect(kind="symbol")` to check bbox
  extents at the intended rotation.

**Wiring:**
- **All wires must be orthogonal** — strictly horizontal or vertical. Never
  route diagonal wires. Use waypoints in the `wire_pins` op for L-shaped or
  multi-segment routes.
- **Horizontal buses must route outside all component bounding boxes.** Use
  `inspect(kind="symbol")` to check bbox extents. For PMOS M180 with bbox top
  at y=160, a gate bus at y=176 is inside the bbox — route at y=144 (between
  VDD rail and bbox top) instead. Plan bus y-coordinates before placing
  components.
- **Vertical wires must not pass through component bodies to reach a bus.**
  When connecting a drain to a horizontal bus, jog the wire sideways outside
  the bbox first, then run it vertically to the bus.
- **Tap an existing wire with a T-junction**: give the `wire_pins` op a
  coordinate endpoint on the wire, `to_pin: {"x": 240, "y": 196}`. LTspice
  joins a wire end, pin or label anywhere along a wire, so the wire stays whole
  and a `remove_wire` op on the new segment undoes it. The op's entry in the
  response's `results` names the wire it joined under `junctions`. Two wires
  that only cross are not joined. A waypoint that touches another net's wiring
  is refused rather than silently merging it; `inspect(kind="net")` at a point
  on a wire traces that wire (`snapped_to_wire`).
- **Heed the `wire_pins` op's warnings and errors**: it refuses diagonal wires,
  pin collisions, and wire junction overlaps. Non-blocking warnings (long runs,
  bbox crossings) should still be addressed.
- **Read the `wiring` profile `edit_schematic` returns.** It reports
  `pins_wired` and `pins_label_only` out of `pins_total`. `pins_label_only`
  high with `wire_segments` near zero means you tagged pins with net-labels
  instead of drawing wires. Such a sheet connects only through its label names,
  which the profile does not check. Draw wires with the `wire_pins` op for
  local nets; reserve net-labels for ground, power rails, and distant nets.
  Also heed the sheet findings in `warnings`: they are what `verify_circuit`
  reports of the same sheet (a floating pin, a loose wire end, parts that
  overlap, a wire through a part, a label or text inside one, a wire LTspice
  leaves out of the netlist), said as soon as the edit that caused them.
- **On an existing sheet, the reported findings are the edit's.** `warnings`
  and `wiring.label_only_pins` list only what your ops introduced or named;
  older ones are counted in `preexisting`, not listed. Before calling a sheet
  done, list them with `return_views: ["preexisting"]` (on an op-less read,
  `ops: []`, that is the whole sheet) or run `verify_circuit`.

**Ground and net labels:**
- **Ground**: give each grounded pin its own ground (`0`) label with an
  `add_net_label` op on that pin (`net="0", pin="M3.S"`); no wire is needed,
  and no ground flag is shared or reached by a long wire. Once several ground
  labels exist, the `wire_pins` op with `net:0` is ambiguous and errors.
- **Named nets (VDD, outp, etc.)**: Repeating the same label at distant pins
  ties them together — the netlist merges same-name labels into one net
  (correct, not a short), and no routing is needed. Wire nearby pins with the
  `wire_pins` op. Caveat: once a name carries duplicate labels, `wire_pins`
  with `net:NAME` is ambiguous — target a component pin (`Ref.Pin`) instead.
- **Label any net you reference by name in a directive.** The `wire_pins` op
  wires pins but assigns no name — at export an unlabeled net becomes `N001`,
  `N002`, …. So a `.meas V(vref)`, a `.param` expression using `V(x)`, or a
  behavioral `B`-source referencing `V(name)` silently breaks unless that exact
  net carries an `add_net_label`. A net you never name needs no label.

**Sources:**
- **Voltage source polarity**: `+` pin is at the top (smaller y), `-` at
  bottom. For VDD sources, `+` connects to the supply rail, `-` to ground.
- **Current source direction**: Current flows from `+` (top) to `-` (bottom)
  externally. Place with `+` on the higher-voltage rail.

**Models:**
- **Model names must not collide with type keywords**: Use `NMOS_3V3` not
  `NMOS` for `.model` names when the symbol Value is also a MOSFET type.
- **Diode default-model collision**: The `diode` symbol defaults its Value to
  `D`, which LTspice resolves to a built-in ideal diode. Adding your own
  `.model D D(...)` collides with that built-in — give the model a unique name
  (`.model MYDIODE D(...)`) and set the symbol's Value to `MYDIODE`, rather
  than reusing `D`.

### Waveform panes for the person the sheet is for

The `set_plot_panes` op writes the `.plt` beside the sheet, which LTspice's
waveform window reads when the sheet is run there:
`{"op": "set_plot_panes", "analysis": "tran", "panes": [{"traces": ["V(out)"]},
{"traces": ["V(in)", "I(R1)"]}]}` opens `V(out)` in a pane above `V(in)` and
`I(R1)`. Panes are listed top to bottom. A trace is an expression as typed into
LTspice's Add Traces dialog, written without spaces (`V(in)-V(out)`): LTspice
reads a trace in the file only up to its first space, so the op refuses one.
The run ranges every axis to its data; `x_scale` and `y_scale` set a pane's log
or dB scales and default to LTspice's own (linear for `tran`, log frequency and
dB magnitude for `ac`). The op replaces that analysis's panes, leaves the
file's other analyses alone and does not change the sheet. New panes keep the
waveform grid the replaced ones all had; a pane in a new file has none, even
for a person whose LTspice draws one on panes it makes. Its `results` entry
names the file and the `replaced_panes`, which passed back as `panes` restore
them; `panes: []` removes them.
