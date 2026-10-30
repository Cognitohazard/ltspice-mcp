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
LTspice's GUI), because it routes wires orthogonally and checks pin collisions
and junctions. `edit_schematic(path=..., base="blank")` starts a new sheet;
every mutation below is an entry in its `ops` list. The ops are applied
together, and the first one that fails aborts the batch with nothing written,
so batch a whole build into one call. Place components with the
`add_component` op. The response's `touched` view lists their placed pins and
nets, and `warnings` reports any overlap. A pin on a net that no label names
has `net: null`; with a `compare`, it also has `net_export`, the name LTspice's
export gives it (`N001`). `inspect(kind="symbol")` previews the same geometry
before you place anything, and lists every attribute the symbol has
(`Prefix`, `SpiceModel`, `Value`, `SpiceLine`, `ModelFile`). These attributes
hold the model name and parameters that an instance is netlisted with.

When the user says "this circuit" and names no file, find out what they have
open with `inspect(kind="open_in_ltspice")`. It lists the sheets and netlists
open in LTspice and marks the one in front `active`. Work on that path as on
any other. A sheet with `differs_from_file: true` has changes in the window
that are not in the file, so a run would simulate something other than what
they see. Tell them, and ask them to save. `only_in_window` and `only_in_file`
list the `.asc` lines where the two differ; a part whose value changed appears
in both, once with each value. Use them to tell the user what is unsaved, and
do not treat them as the circuit to work on.

When they ask to see a sheet you built or changed, `verify_circuit(path=...,
in_ltspice=true)` opens it in their LTspice window, in front. If the reply's
`ltspice` block has `differs_from_file: true`, the window already had the
sheet open and shows an older copy than the one you checked, so tell them.
With no LTspice window open, `in_ltspice` starts LTspice (`started: true` in
the block), and LTspice comes to the front of their screen. Pass it only when
they asked to see something there, never to check your own work. If the block
reports that none was started, starting is turned off
(`[schematic] start_ltspice`), so ask them to open LTspice.

When they want to plot nets by clicking the sheet in LTspice, show them a run
of the sheet with `plot_waveform(job_id=..., signals=[...], in_ltspice=true)`
on a job that ran an `.asc`. The run's results are written beside the sheet
under its name, replacing any results there, and opened from the sheet, which
makes LTspice tie the plot to the sheet. The reply's `ltspice` block then
names the `sheet` and has `tied: true`, and a click on a net in the sheet
plots it. Results already beside a sheet under its name, as a run in LTspice
leaves them, open the same way when you pass them as `path`. LTspice reads
a plot only when it opens it, so a plot of these results that is already open
is closed and reopened, and the block reports `reopened: true`. Traces the
user added to that plot by hand are lost, so tell them if they had been
working in it. LTspice also looks for a sheet's results
only when it opens the sheet, so a sheet it opened before it had results is
closed and reopened too (`sheet_reopened: true`), which clears its undo
history in LTspice. This is done only when the sheet in LTspice matches the
file. If it has changes that are not in the file, the sheet stays open and the
plot opens separately, with `tied: false` and `differs_from_file: true` in the
block. Tell the user that clicking nets will work after they save or close the
sheet in LTspice and you show the run again. A netlist's run has no sheet to
tie to and opens as a separate plot; it is also closed and reopened if it was
already open.

LTspice ships its own MCP server, which a session may have beside this one
(its tools include `set_design_content` and `start_simulation`). It acts on
the copy in the LTspice window instead of the file. Make edits with
`edit_schematic`. A sheet changed through LTspice's server changes only in the
window, and `edit_schematic` then refuses that sheet until the user saves it.
A run started through LTspice's server simulates the window's copy and leaves
no job, so run with `run_experiments`. When one of its tools reports that it
could not start or attach an LTspice instance, no LTspice window is open. Do
not try its `start_headless`; show the user what they asked for with
`in_ltspice`, which opens one.

LTspice does not read an open file again, so when the user has a sheet open in
LTspice (Windows, LTspice 26.1 or later), the server shows each commit to it
in that window as one step of its undo history. A commit's reply lists the
window under `open_in_ltspice`, and `shown: true` means the user already sees
the edit.
`shown: false` means the window still shows the old sheet and a save from it
would overwrite yours, so tell the user to close it without saving and open it
again. An edit that fails with `open_window_differs` wrote nothing, because the
window's copy differs from the file. Only the user can resolve that, so ask
them to save the sheet (Ctrl+S) or close it, then read its `sha256` again and
resubmit.

- Component attributes: Value, Value2, SpiceLine, SpiceLine2. A part's own
  `Prefix` replaces its symbol's element letter on export, on both builds, so
  `Prefix X` with the subcircuit's name in `Value` makes a resistor or MOSFET
  symbol a subcircuit call (`X§R2 ... mysub`) without a new symbol file.
- Bus notation: `Data[0:7]` creates 8 nets. The bus is cosmetic, because the
  netlister flattens it to individual nets.

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
are 3-terminal, and the export writes four nodes for them, with a MOSFET's
bulk at its source and a BJT's substrate at `0`. When you need the
body/substrate on its own net (e.g. a non-source bulk bias), use the
4-terminal variants (`nmos4`/`pmos4`, `npn4`/`pnp4`), which expose
bulk/substrate as a 4th pin.
In a reference netlist for the `compare` check, write such a MOSFET with four
nodes, `M1 d g s s NMOS_MOD`, because LTspice does not read a three-node card
with a level-1 model as having its bulk at the source. A three-node
`Q1 c b e QN` card has its substrate at `0` and compares equal.

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
itself, and verifies before it returns. `verify_circuit` with the `export` and
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
- **Pin-to-rail alignment**: Place voltage/current sources so their pins are
  directly on the rail they connect to, with no wire through the source body.
  For a VDD source, position it so the `+` pin y-coordinate equals the VDD rail
  y-coordinate. Use `inspect(kind="symbol")` to compute the exact placement
  origin from the desired pin position (e.g., for voltage `+` at y=128, place
  origin at y=128-16=112).
- **Keep at least 128 units of vertical space between the pin levels** of
  adjacent tiers (e.g., between PMOS drain y and NMOS drain y), so a
  horizontal bus and its labels fit between one tier's bounding boxes and the
  next. With a MOSFET bbox height of 96, plan tier origins ~192 units apart.
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
- **Heed the `wire_pins` op's errors and junctions**: it refuses diagonal
  wires, pin collisions, and wire junction overlaps. Where the route joins its
  own net on the way, its `results` entry lists the point under `junctions`.
  The sheet findings below report a crossed wire or a route through a part.
- **Keep wires short; nothing reports wire length.** In LTspice's own example
  schematics 50% of wire segments are 32 units or shorter, 90% are 192 or
  shorter, and about 3% exceed 400. A long wire is not an error, but if it
  only runs a signal across the sheet, a net label at each end is easier
  to read.
- **Read the `wiring` profile `edit_schematic` returns.** It reports
  `pins_wired` and `pins_label_only` out of `pins_total`. A high
  `pins_label_only` with `wire_segments` near zero means you tagged pins with
  net-labels instead of drawing wires. Such a sheet connects only through its
  label names, which the profile does not check. Draw wires with the
  `wire_pins` op for local nets; reserve net-labels for ground, power rails,
  and distant nets. Also heed the sheet findings in `warnings`. They are the
  findings `verify_circuit` reports for the same sheet (a floating pin, a
  loose wire end, parts that overlap, a wire through a part, a label or text
  inside one, a part's name or value drawn across a wire, another part or
  other text, a wire LTspice leaves out of the netlist), returned by the edit
  that caused them. A part's text moves with the part and no op moves the text
  alone, so move the part (`move_component`, or `move` for a region) or the
  item its text crosses.
- **On an existing sheet, the reply lists only your edit's findings.**
  `warnings` lists only the findings your ops made or changed, and
  `wiring.label_only_pins` only the pins of parts they rewired or named. A
  part that is moved or given a new value keeps the findings it had. Older
  findings are only counted, in `preexisting`. A batch that only sets values
  or attributes lists no pins at all, since it leaves every pin joined as it
  was. Before calling a sheet done, list the older findings with
  `return_views: ["preexisting"]` (on an op-less read,
  `ops: []`, that is the whole sheet) or run `verify_circuit`.

**Moving what is drawn:** the `move` op shifts parts, wires, labels and
texts by one offset (`dx`, `dy`), chosen by name or by a `region`. With
`drag: true`, other wires' ends and the labels and grounds on a moved pin move
with it, so they stay attached. A wire dragged off its axis turns diagonal,
and the sheet findings report it (`diagonal_wire`). To move one part to a
position, or to turn it, use `move_component`.

**Ground and net labels:**
- **Ground**: give each grounded pin its own ground (`0`) label with an
  `add_net_label` op on that pin (`net="0", pin="M3.S"`); no wire is needed,
  and no ground flag is shared or reached by a long wire. A `GND` label in any
  letter case is ground too. Once several ground labels exist, `net:0` is
  ambiguous; address one by its point, `to_pin: {"x": 400, "y": 340}`.
- **A label or ground flag at a crossing joins the two wires.** Place it on
  the one wire you mean. `add_net_label` refuses it when both wires are on
  named nets; otherwise it is placed and the sheet reports the join
  (`joined_at_crossing`).
- **Named nets (VDD, outp, etc.)**: Repeating the same label at distant pins
  ties them together, because the netlist merges same-name labels into one
  net. That merge is intended and is not a short, and no routing is needed.
  Names that differ only in case (`OUT`, `out`) are one name. Wire nearby pins
  with the `wire_pins` op. Once several labels have the same name, `net:NAME`
  is ambiguous; end the route on the one you mean by its point instead.
- **A wire ending in a label**: put the label on the free point first, then
  route to that point with the `wire_pins` op. A coordinate endpoint must be
  on a pin, a wire or a label; one on nothing is refused.
- **Label any net you reference by name in a directive.** The `wire_pins` op
  wires pins but assigns no name — at export an unlabeled net becomes `N001`,
  `N002`, …. So a `.meas V(vref)`, a `.param` expression using `V(x)`, or a
  behavioral `B`-source referencing `V(name)` silently breaks unless that exact
  net has an `add_net_label`. A net you never name needs no label.

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
waveform window reads when the sheet is run there. For example,
`{"op": "set_plot_panes", "analysis": "tran", "panes": [{"traces": ["V(out)"]},
{"traces": ["V(in)", "I(R1)"]}]}` opens `V(out)` in a pane above `V(in)` and
`I(R1)`. Panes are listed top to bottom. A trace is an expression as typed into
LTspice's Add Traces dialog, written without spaces (`V(in)-V(out)`). LTspice
reads a trace in the file only up to its first space, so the op refuses a
trace that contains one.
`analysis` is `tran`, `ac`, `dc` (a DC sweep) or `noise`. The run sets each
axis range to fit its data; `x_scale` and `y_scale` set a pane's log or dB
scales and default to LTspice's own (linear for `tran` and `dc`, log frequency
and dB magnitude for `ac`, log frequency for `noise`). The op replaces that
analysis's panes, leaves the file's other analyses alone and does not change
the sheet. New panes keep the waveform grid the replaced ones all had; a pane
in a new file has none, even for a person whose LTspice draws one on panes it
makes. Its `results` entry names the file and lists the `replaced_panes`; pass
them back as `panes` to restore them. `panes: []` removes that analysis's
panes.

`plot_waveform(in_ltspice=true)` writes these panes itself for the traces it
plots, unless the `.plt` there was saved from LTspice. That file belongs to
the user and is left unchanged. The reply's `ltspice` block then lists the
panes the window draws instead under `saved_panes`, in the form this op takes.
When the user wants different or more panes, set them with this op; to keep
the user's panes as well, pass `saved_panes` with your own panes added. The
axis ranges they saved with them are not kept.
