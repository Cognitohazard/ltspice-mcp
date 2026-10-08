# The schematic engine — capability target and architecture

What the `.asc` authoring side of the server should be able to do, what that
requires of its internals, and the order the internals change in. The tool
contracts stay in `docs/design/mcp_surface.md`; this document is about the
engine behind `edit_schematic`, `inspect` and `verify_circuit`.

The code is the authority. Where this document and the source disagree, the
source wins and this document is the thing to fix.

---

## 1. The stance

The agent decides what the drawing looks like. It knows what a readable
schematic is: signal flow left to right, supplies above and ground below,
matched devices mirrored about an axis, a mirror's gates joined by one
straight bus. The server does not decide any of that, and it does not lay out
or route a circuit on its own.

What the server owns is everything the agent cannot cheaply get right from
text:

- **Arithmetic.** Where a pin is after a rotation, where the origin must go
  for a pin to land on a rail, where the corners of an L-shaped wire are.
- **Legality.** Whether a proposed drawing connects what LTspice will say it
  connects, by rules recorded from LTspice.
- **Facts.** What the sheet contains, what an edit changed, and the measurable
  properties of the drawing (crossings, bends, overlaps) that the agent's own
  judgement can be checked against.

So the unit of authoring is an *intent the agent states and the server
resolves*: "put this pin there", "join these two with an L that leaves
horizontally", "run this net's trunk along y = 176". The agent chooses the
shape; the server computes the coordinates and refuses what is not legal,
saying what blocked it and what nearby choice would not be blocked.

This replaces the earlier reason given for having no routing (that routing is
NP-hard). Joining two pins on a grid is not hard to compute. The reason the
server does not do it unasked is that the choice of path is the drawing, and
the drawing is the agent's.

One published comparison bears on this directly. SchGen (arXiv 2605.30345,
Table 2) had a language model write the same schematics in four
representations and scored functional correctness: relative placement with
connection by pin name 60.5%, absolute coordinates with connection by pin name
33.0%, absolute coordinates with coordinate wires 6.0%, the raw file 3.0%. The
model was a fine-tuned 20B one and the circuits were KiCad boards, so the
numbers do not transfer; the ordering is the point. `edit_schematic` today is
the second and third rows: placement is an absolute origin, and a wire's ends
are pins by name but every corner is a coordinate.

## 2. Capability inventory

"Seen in" names where a capability exists elsewhere, as that project's own
documentation or paper describes it. None of those tools was run here, so a
row is a statement about what exists, not about how well it works. "Order" is
when it is built: 0 is the internal rework in §5, which comes first; 1 is what
closes the gap to the field; 2 and 3 go past it; "shown first" marks a row
that is not built until the rule below has been met for it.

### Placement

| Capability | Today | Seen in | Order |
|-|-|-|-|
| Place by pin: a named pin lands on a point or on another pin, with an offset | no, origin only | SchGen; schemdraw anchors | 1 |
| Place beside a part: a side, a gap, a pin to align on | no | Lcapy direction hints | 1 |
| Report a coordinate off the 16-unit grid | no, any integer is written silently | KiCAD-MCP-Server lint; kicad-sch-api | 1 |
| Orientation by intent ("gate left, drain up") | no, a rotation code read off a table in the guide | SKiDL orientation hints | 2 |
| Align or distribute a set of parts | no | any GUI | shown first |
| Move, rotate or mirror a group with its wiring | no | any GUI | 2 |
| Mirror-copy or repeat a group, renumbering references | no | any GUI | shown first |
| Save a group and place it again | no | KiCad design blocks | 3 |

### Wiring

| Capability | Today | Seen in | Order |
|-|-|-|-|
| Connect two ends, LTspice's joining rules enforced | yes, recorded | not seen elsewhere | — |
| T-junction onto a wire | yes | mcp-server-kicad | — |
| Route by shape (L or Z, optional track); the server computes corners | no, the caller gives each corner | SchGen; kicad-sch-api | 1 |
| A refusal as structured conflicts, with the nearest clear tracks | prose in an error | not seen elsewhere | 1 |
| A multi-pin net on a named trunk (rail, bus) | no | not verified elsewhere | 1 |
| A junction where two wires cross | no | any GUI | 1 |
| Drag: a move keeps wires attached | no, wires are left behind | LTspice, xschem, KiCad | 1 |
| The facts of each route shape the caller names, on a dry run | no | not seen elsewhere | 2 |
| Insert a two-terminal part into a wire | no | LTspice | 2 |
| Tidy: merge collinear runs, drop duplicates and stubs | duplicates are not drawn | xschem | 2 |
| Rename a net; turn a label link into a wire or back | no | not seen elsewhere | 2 |
| Replace a symbol and keep its connections | no | any GUI | 2 |

### Text, annotation, hierarchy

| Capability | Today | Seen in | Order |
|-|-|-|-|
| Waveform panes for the sheet | yes | not seen elsewhere | — |
| Move or hide a part's name and value text | no | any GUI | 1 |
| Attribute text counted in overlap facts | no, a stated blind spot | — | 1 |
| Annotation shapes; aligned and turned comments | comments only, left-aligned | any GUI | 2 |
| A port direction on a label | kept on rewrite, cannot be set | any GUI | 2 |
| Define a block: child sheet, symbol and ports | designed in `mcp_surface.md`, not built | any GUI | 2 |
| Make a symbol for a `.subckt` from its ports | no | LTspice | 2 |
| Edit a symbol file | no | any GUI | 3 |

### Reading, checking, committing

| Capability | Today | Seen in | Order |
|-|-|-|-|
| Equivalence with a reference netlist, through LTspice's own export | yes | Weave certifies the same way | — |
| Layout facts: overlap, wire through a body, floating pin, loose end, label-only nets | yes | eda-agent | — |
| All-or-nothing commit guarded by the file's hash | yes | — | — |
| Open any sheet and keep every record of it | no: an unmodelled record is fatal, `DATAFLAG` is dropped | kicad-sch-api | 0 |
| An edit leaves untouched records' bytes and order alone | no, the file is regrouped | kicad-sch-api | 0 |
| An edit that only changes the drawing is shown to leave the circuit unchanged | no | not seen elsewhere | 0 |
| A measured build benchmark | builds at scale are tested for completeness, not scored | none published | 0 |
| Readability facts: crossings, bends, lengths, off-grid, alignment of named groups | no | eda-agent (crossings) | 1 |
| Render: junction dots, text alignment, every visible attribute | partial | xuio (native capture) | 1 |
| What changed between two revisions, as geometry | no | not seen elsewhere | shown first |
| Render overlays: ruler, pin names, markers on findings, a region | no | not seen elsewhere | 2 |

**Each row is held to the project's rule for new functionality.** Before a
row is built, the task is done through what exists (the ops, `inspect`, and
a caller's own code in `run_code`), and what is written down is the
primitive that was missing and the correctness burden the server would own.
The rows of order 1 have that burden on their face: placing by pin and
routing by shape are the stated-intent form of ops whose legality is already
the server's, a trunk and a drag must keep LTspice's recorded joining rules,
and a readability fact needs one definition of a crossing, which is a
connectivity rule. The rows marked "shown first" are arithmetic over records
a caller can already read, so each waits for a `run_code` demonstration that
says what, if anything, the caller could not do reliably. SchGen measured a
model writing coordinates in its reply; it says nothing about a model
writing code that computes them.

The dry-run row is worded on purpose: the server reports on the shapes the
caller names and does not search for one.

Two things the field does that this server will not, by the stance in §1:
generate a schematic from a netlist with no layout input (Weave,
`daviditkin/ltspice-mcp`, `xuio/ltspice-mcp`), and recognise sub-circuits and
stamp a canned layout onto them (eda-agent). Both put the layout decision in
the tool. Weave's own result marks the limit of that approach: it certifies
connectivity on 88.4% of 3,460 Analog Devices demo circuits and reports no
measure of how the drawings read.

## 3. What those capabilities require

Each requirement names the capabilities that force it.

1. **A lossless document.** Every record of a sheet is kept, in order, with
   its encoding and line endings; a record the server has no type for is kept
   as text. A kept line whose kind is not known to be cosmetic is reported
   by everything that reads connectivity, since it may be a connection the
   server cannot see. *Forced by:* opening any sheet, untouched bytes, port
   directions, attribute text, annotation shapes.
2. **A sheet is a value.** An edit takes a sheet and returns a new one; the
   old one is still there. *Forced by:* dry runs and candidate routes (try,
   measure, discard), group operations (build the result, check it, commit or
   not), the before-and-after reads `edit_schematic` already does, and the
   geometric diff.
3. **One symbol library.** A symbol is parsed once into everything any caller
   needs: pins with names and order, the body, the attribute windows, the
   attributes. One resolver decides which file a name means. The library can
   also write a symbol. *Forced by:* place by pin, orientation by intent,
   attribute text, symbol generation, a render that agrees with the editor.
4. **Orientation arithmetic in one place.** The eight orientations compose;
   a point, a box, a part or a group transforms the same way; "which origin
   puts this pin there" and "which orientation faces this pin that way" are
   solved, not looked up. *Forced by:* place by pin, place beside, orientation
   by intent, group mirror and rotate.
5. **One connectivity index.** Nets from wires, pins and labels by LTspice's
   recorded rules, kept up to date as segments are added and removed, able to
   answer "what would this join" for segments not yet drawn, and reducible to
   a signature two sheets can be compared by. *Forced by:* every wiring
   capability, and sheets of several hundred parts.
6. **Legality apart from construction.** Whatever proposes segments — the
   caller's waypoints, a shape solver, a drag, a mirror — the same predicate
   judges them and returns conflicts as data. *Forced by:* route by shape,
   trunks, drag, group operations, structured refusals.
7. **Findings are data.** One type for a fact about a sheet or a conflict
   with a proposal: its kind, the parts, wires and labels involved, where, and
   for a conflict what would not conflict. Prose is produced at the tool
   boundary. *Forced by:* structured refusals, candidate routes, readability
   facts, and any loop a caller writes in `run_code`.
8. **An op states what it may change.** An op that only changes the drawing
   (drag, tidy, mirror, moving attribute text, turning a label link into a
   wire) must leave the connectivity signature equal, and the engine checks
   that before the commit. The signature is the circuit LTspice would
   netlist, so it applies every recorded rule, including the wire LTspice
   drops when it runs straight between two pins of one part. *Forced by:*
   every capability that rearranges an existing, working circuit.
9. **A scene derived from the sheet.** The render and the layout facts read
   the same sheet and the same library the editor does, with text extents.
   *Forced by:* render fidelity, overlays, text in overlap facts.
10. **A commit of several files.** A block is a child sheet and a symbol; a
    sheet already commits with its `.plt`. *Forced by:* blocks, symbol
    generation.
11. **A benchmark.** Fixed netlists built through the tools, scored on
    equivalence, calls, refusals and the readability facts. *Forced by:* the
    need to tell whether a new op helps. The removed `occupancy` view was
    decided by a measurement; this makes that the routine. It grows out of
    `tests/test_ops_at_scale.py`, which already builds every archetype and
    a 51-part ladder in one call and checks that each is complete.
12. **A sheet names its children.** A block symbol resolves to its sheet
    through the library, and nothing reads a child unless asked to. *Forced
    by:* blocks, and by what exists today: spicelib loads a block's sheet
    when it opens the parent, and the commit refuses a parent whose loaded
    child was changed.

## 4. Where the present structure cannot carry them

- **Two stacks.** The editor reads a sheet through spicelib's `AscEditor` and
  `lib/symbol_geometry.py`; the checker and renderer read it through
  `lib/schematic_scene.py`. Each has its own sheet parser, symbol parser,
  symbol resolver, point-on-segment test and floating-pin rule. Only the
  rotation table and the dropped-wire rule are shared. The two already
  disagree in small ways: a symbol in a subfolder beside the sheet resolves
  for the renderer, which searches the folder, and for spicelib, which walks
  it ignoring case, and not for the editor's geometry, which tries the one
  exact path; two pins of one part on one point
  are connected to the editor and floating to the checker; and a sheet with a
  record spicelib has no case for opens for the renderer and fails for the
  editor. Requirements 1, 3, 5 and 9 are each "one of these, not two".
- **The sheet is another library's mutable object.** A batch mutates a cached
  `AscEditor` and undoes a failure by dropping it from the cache, at five
  places in `tools/schematic_edit.py`. Port records survive a save only
  because the saved text is patched afterwards. Symbol search paths are class
  attributes on `AscEditor`. Eight of the twenty-four entries in
  `docs/spicelib_bugs.md` are in this small part of spicelib (the ones on
  directive placement, an empty attribute value, directive removal, ports, a
  save that writes child sheets, reference prefixes, the symbol cache, and a
  block with no sheet), and the dependency is capped below 1.6 because 1.6
  changed the editor's model. Requirements 1 and 2 cannot be met through it.
- **Results are sentences.** A refused route raises an error whose message
  lists what blocked it; op advisories are strings and whole-sheet findings
  are dicts; a repeated advisory is folded by comparing strings. Requirement 7.
- **Legality is one function with construction inside it.**
  `_plan_connect_route` resolves the ends, builds the segments, judges them
  and writes the advisories, and rebuilds every part's geometry and the whole
  net partition on each call. Requirements 5 and 6.

What is sound and stays: the recorded connectivity rules, the commit protocol
and revision guard, edit-scoped findings, the netlist comparison, and the op
contract callers already use.

## 5. Target structure

Flat modules under `lib/`, as the rest of the engine is. Each layer imports
only the ones above it in this list.

```
format   asc_document.py      .asc bytes <-> records; lossless
         symbol_file.py       .asy bytes <-> one symbol definition
model    symbol_library.py    name -> definition; the one resolver; session-owned
         sheet.py             a sheet as a value: its document and its library;
                              addressing (reference, net, region); edit primitives;
                              the child sheets its blocks name, read on request
kernel   geometry.py          boxes, orthogonal segments, text extents
         placement.py         orientations, placed pins and boxes, the two solvers
         connectivity.py      the net index, overlays, the signature
policy   sheet_findings.py    finding and conflict types; layout and readability facts
         routing.py           shape solvers; the legality predicate
         schematic_ops.py     op models; each op is sheet -> (sheet, facts)
view     schematic_scene.py   scene from a sheet; overlays
         schematic_renderer.py
tools    schematic_edit.py    lock, revision guard, commit of one or more files
         verify.py, inspect_tools.py
```

Rules the structure holds to:

- **One parse.** Nothing reads `.asc` or `.asy` text except the two format
  modules.
- **Unmodified records are written as they were read.** A record carries the
  text it was read from, and that text is written back as long as it still
  describes the record. Only a changed or new record is formatted. A build
  from a blank sheet writes the bytes the present engine writes, which is the
  form the recordings show both LTspice builds read.
- **Records keep their order.** A new record goes after the last of its kind.
- **The library belongs to the session.** Its search roots are set once at
  startup from the configuration and the detected simulator; nothing reads
  them from a class attribute.
- **Every connectivity rule cites its recording.** `connectivity.py` is the
  only module that encodes how LTspice joins a sheet, and each rule names the
  case in `tests/fixtures/ltspice_recorded/` that shows it.
- **Nothing here needs the event loop.** Sheets, indexes and scenes are
  immutable, so a batch can be evaluated on a worker thread; the rule that a
  cached editor is touched only on the loop goes away with the cached editor.
- **spicelib launches simulators and exports netlists.** It no longer reads or
  writes a sheet.

## 6. The rework, in order

Each step leaves the suite passing. The tool contracts change only by
addition: no field is removed and no severity changed; a list of kinds may
grow, and a count may differ where two passes become one. A step's gate is
what must hold before the next one starts.

1. **The lossless document** (`asc_document.py`). Additive. *Done.*
   *Gate:* every `.asc` under `tests/` reads and writes back to its own bytes;
   records with no type, mixed line endings and each encoding in the fixtures
   are kept; each record's formatted form is the line spicelib writes for it,
   and a sheet built through `edit_schematic` is rebuilt from records to the
   same bytes (`tests/test_asc_document.py`). The sheets each build writes
   when it saves one are recorded (the `save` cases) and read, written back
   and formatted the same way (`tests/test_recorded_ltspice_sheet_save.py`).
   A symbol with a line under it that does not read stays a symbol and
   carries that line.
2. **One symbol library** (`symbol_file.py`, `symbol_library.py`).
   `symbol_geometry` and `schematic_scene` take their symbols from it; the
   session owns it. *The reader is done:* both former parsers are
   `symbol_file.read_symbol`, and agree on every symbol in the suite
   (`tests/test_symbol_file.py`). Two things changed for a hand-broken file,
   each towards one answer: a bare `PIN` line now makes the symbol unusable to
   the editor, as any other unreadable pin line did, and an `ARC` with fewer
   than eight numbers is left out of the editor's box as it was out of the
   drawing. *The search is done too:* where each build finds a symbol is
   recorded, and the editor's pin geometry and the renderer both ask
   `symbol_library.find_symbol` (§7), held to every recorded case through
   each of them (`tests/test_recorded_ltspice_schematics.py`). *Still to
   do:* the session owning the search roots. They stay class attributes of
   spicelib's editor for as long as it opens sheets, so this moves with
   steps 5 and 6, when a sheet carries its library. *Gate for that:* the
   existing
   symbol, scene and render tests pass unchanged, and one resolution test
   runs against both former entry points.
3. **The connectivity index** (`connectivity.py`), taken out of
   `net_partition` with row and column buckets, reading plain segments, pins
   and labels. *Done:* the editor, the checker and the trace all read it. It
   is held to a plain reference partition on every sheet in the suite and on
   generated ones (`tests/test_connectivity.py`), and its signature to what
   both builds netlisted for every recorded connectivity sheet.
4. **One findings pass**, on the rule model of §8, in five parts. It reads a
   plain index of a sheet (pins, labels, segments, boxes, text anchors), as
   `connectivity.partition` already does, so that the editor can build it from
   the editor it holds and the checker from the file, until step 5 gives both
   the sheet.
   - *The type and the registry.* *Done* (`lib/sheet_findings.py`). One
     finding type; each rule's id, family, scope, provenance and what a
     finding of it is called on a whole sheet in a registry; the whole-sheet
     checks of both tools moved under it as they are, each tool still shown
     the rules it showed. A rule's disposition on a proposal joins the
     registry with the refusals, two parts on. *Gate, met:* what both tools
     say of every `.asc` in the suite was recorded beforehand
     (`tests/fixtures/sheet_findings.json`) and is the same after. That
     record is the gate of the three parts that follow: each changes it on
     purpose, in the commit that changes the rule.
   - *The disagreements settled* (§7). *Done.* A floating pin is one rule,
     the recorded one, for both tools; two parts overlap by their boxes with
     pins; the editor reports a part whose symbol is not found. *Gate, met:*
     the record changed for three sheets and no others: a pin on a diagonal
     wire's interior, the sheet of parts with two pins on one point, where
     both tools now call floating exactly the pins LTspice left on a node of
     their own, and the sheet whose symbol is not found.
   - *The two memberships merged.* The editor reports what the checker did and
     the reverse; an edit's findings are scoped as §8 says. *Gate:* the
     response contracts, and the `preexisting` counts reconciled. *Done*
     (`sheet_findings.findings`), as its review left the plan, which follows.
     *Gate, met:* for every sheet in the suite that the editor opens, an edit
     and a check find the same things, rule by rule and place by place
     (`tests/test_sheet_findings_snapshot.py`), and the same through both
     handlers, sentence for sentence (`tests/test_edit_schematic.py`); the
     record changed for 31 sheets, each tool gaining there what the other
     already said. `verify_circuit`'s published description names the four
     facts it gained, which raised its size bound by 60 characters.
     - The view the editor builds of a sheet cannot carry the checker's
       rules. It has no box of what a part draws without its pins and no
       anchor of a part's attribute text, which a wire through a part and a
       text inside one are judged by, and working either out from spicelib's
       editor would be a second copy of the scene's placement. So
       `edit_schematic` builds the scene of the text it is about to write and
       reads the view `verify_circuit` reads; the view built from the editor
       goes. What an edit reports is then what the checker would say of the
       file once written. Measured on LTspice's own example sheets: a scene
       takes 30 to 110 ms on the largest and every rule together under 20 ms,
       where spicelib takes 0.5 to 3.4 s to open the same sheet; and with an
       arc bounded by what is drawn of it (§7) the two views already agree on
       every part's box and on every finding of the rules both hold, on all
       146 of the 150 largest that spicelib opens.
     - The sheet before the batch is read from the editor's rendering of it
       too, taken before the ops run, and not from the file. spicelib does
       not write a sheet back as it read it: two parts of one reference
       become one, a text line it cannot read is dropped, and the records
       are regrouped. Read from the file, every such difference on a sheet
       the batch did not touch would be reported as the batch's.
     - Both scenes are built off the event loop, in one call: they are pure
       functions of text and symbol files. The editor is rendered and
       changed on the loop, as it must be.
     - The scene an edit reads is resolved from the roots the editor's pin
       geometry searches, and no others, with a test that the two are the
       same list. `verify_circuit`'s resolver also searches the stock
       library in one configuration where the editor's does not (WSL with
       `[schematic] symbol_paths` set, where the configured paths replace the
       stock ones for the editor), and an edit whose findings saw a part's
       pins while its pin counts did not would contradict itself in one
       reply. That the two tools search different roots there is the
       leftover of step 2 and goes when the session owns the library.
     - One list of findings in the registry's order, and one sentence for
       each that names its parts and its place, since an edit's reply shows
       the sentence alone. Everything a sentence says is also in the
       finding's parts, points and facts (the text of a text inside a part
       is a fact, not only words), so two findings are the same finding when
       those are the same. A part whose symbol is not found is one finding a
       part; `verify_circuit` groups them by symbol where it words them.
     - `edit_schematic` gains overlapping parts, a wire through a part, a
       loose wire end, text inside a part, a net joined only by labels, and a
       wire LTspice leaves out. `verify_circuit` gains a wire drawn twice and
       a label on nothing under `layout`, and a label inside a part and
       stacked directives under `quality`, all as observations. A label is
       inside a part by the part's box with its pins and a text by what the
       part draws, and the description of the `quality` check says so.
     - An edit's findings are scoped as findings, by every part and every
       point one names, where today a row's first part and first point are
       looked at; and the row an edit's `preexisting` view lists carries all
       of them, where today it carries the first.
     - The criterion §8 adds, a net whose membership the batch changed, is
       not built. Worked through, every case it was meant for is already
       caught: a pin left floating at the far end of removed wires is a new
       finding, and so is what is said of the other labels of a net when one
       is removed. What it adds is only what was there before and is
       unchanged, on any net the batch touched, which on a rail is every
       loose end and repeated wire of the rail, listed again on every batch
       that connects to it.
     - An op's own advisory and a finding of the sheet already say one thing
       twice today, for a label placed on nothing. Overlap and a wire through
       a part will be said twice the same way until the next part turns the
       ops' checks into rules.
     - Nothing is said of the extent of a part whose symbol is not found.
       The checker drew it as a placeholder and judged overlaps and crossings
       by the placeholder's box, which is not the part's; with one list that
       would have reached an edit's reply too.
   - *Refusals as rules*, the part most likely to go wrong. The route planner's
     and the label op's refusals become rules over a transition, in the order
     they are raised today, wrong intent first. *Gate:* every test that pins a
     refusal's wording, the archetype and at-scale builds with no rejection,
     the recorded connectivity sheets through the planner, and the planner's
     answers captured beforehand and compared after. *The capture is made:*
     `tests/fixtures/route_planner_record.json` holds what the planner answers
     to some 8,500 proposals on the 79 sheets of the suite the editor opens
     (pin to pin straight and by each corner, pin to the middle of each wire,
     a detour level with each part), every refusal and advisory it has among
     them, and `tests/test_route_planner_record.py` holds it to that. The
     plan's review found two routes the planner drew that join named nets, an
     end given by net name and a crossing at a labelled point; both are
     refused now and the record asks for routes by net name too, 11,000
     proposals in all. How the rest is to be done, as the review left it:
     - The checks become functions of plain data in `routing.py`, reading
       the view and the partition the sheet rules read, filled from the
       editor by an adapter that step 6 discards. No answer changes: the
       record is the gate. *Done* (`routing.judge`), with the record
       unchanged and each rule held to plain sheets in `tests/test_routing.py`.
     - They are not all separate. The overlap check, the check of a leg
       along the wire it ends on and the contact check hand a set of wires
       already refused from one to the next, and stay one stage. The pin
       check's notion of a pin on the route's net (a wire runs straight from
       it to an end) is narrower than the net lookup the contact check uses,
       and both are kept as they are.
     - The checks raised alone before any other (both ends on one point, no
       length, the two named-net checks, a wire LTspice drops) and the
       resolving of an end are checks of what was asked, and stay in the
       planner with the wording of a refusal.
     - Route rules have a registry of their own. In the sheet rules' registry
       they would be listed as run by every edit, published among the kinds
       of a sheet finding, and expected to have a whole-sheet finder.
     - Then, each on purpose and with the record changing: a part's box
       crossed by a route is judged as `wire_through_symbol` is, by what the
       part draws and for the route's own end parts too; the advisory for a
       run over 400 units goes, a length the server picked being no ground
       for advice, and the guide names what is usual instead; and placing or
       moving a part is checked for the one way it can join two named nets,
       a pin landing where two wires cross. *The last is done*
       (`schematic_ops._pins_on_crossings`); the other two are not.
   - *What ran.* The list of rules that ran, added to both replies; and, if
     wanted, a caller's limits and waivers, which are new arguments and need
     the published size of both tools raised on purpose. *The list is done:*
     `rules_run` in both replies names each sheet rule that ran with its
     count, zero included. Limits and waivers are not built.
5. **Readers on the sheet.** `inspect`'s net and component queries and the
   schematic resource stop using the cached editor; a cache of sheets keyed
   by the file's stamp takes its place.
6. **Ops on the sheet**, in two parts, and the step most likely to go wrong.
   First the ops that move records (parts, wires, labels): each becomes
   sheet -> (sheet, facts), the batch is a fold, and the transaction keeps
   the before and after values. Then the ops that edit text, where
   spicelib's behaviour has to be written again on the project's own lexer:
   a value split into its model and parameters, a parameter set inside an
   attribute line, where a directive with no position is put, and a
   directive that replaces the one of its kind. *Gate for each part:* the
   whole schematic suite passes; a blank build of every archetype is
   byte-identical to the build the present engine makes; every sheet the
   present suite commits is captured beforehand and compared after; and on
   every fixture, an op followed by its inverse gives back the sheet's own
   bytes.
7. **Markers at the tool boundary.** With step 4 done a conflict is already
   a finding; this step adds the structured form to the replies beside the
   prose.
8. **Remove the old path.** `AscEditor` leaves the schematic path, with the
   workarounds for the eight spicelib entries it needed; the test fixtures
   that set its symbol paths and cache move to the library; `docs/DESIGN.md`,
   `CLAUDE.md`, `mcp_surface.md` and the guide are brought in line.

The benchmark (requirement 11) is built alongside steps 1 to 3, against the
present engine, so that it has a baseline before any op changes.

## 7. Decisions

Taken:

- Existing ops keep their names and shapes. Everything in §2 is additive: a
  new field on an existing op where the intent is the same act (`add_component`
  placed by pin, `wire_pins` given a shape), a new op where it is not.
- An existing sheet is rewritten in its own record order, not regrouped.
- There is still no undo and no server-side layout or routing without a
  stated intent.
- A coordinate off the 16-unit grid is written and reported, not refused.
  A sheet that already holds one must stay editable.
- The editor reports a part whose symbol is not found, as the checker does.
  Today it leaves the part out of every geometry pass without a word, so
  its pins are missing from the floating-pin check and from the pin counts.
- A move is checked for joining two named nets, as drawing a wire and
  placing a label are, and so is a placement. Worked out from the joining
  rules, there is one way either can: a pin landing on a point where two
  wires cross, which joins them. On a wire's interior or end, a label or
  another part's pin, a pin lands on one net. Between two named nets that
  is refused; with an unnamed side the part is placed and the join said,
  a pin put on a crossing being taken as meant where a route's waypoint is
  not.
- A wire is through a part, and a text inside one, by what the part draws
  without its pins. The box with pins is for two parts overlapping. With a
  pin drawn apart from the body, the box reaches out to it over the whole
  side, and an ordinary wire to a nearer pin on that side would be through
  it.
- A part's box, for overlap, includes its pins. It is the box `inspect` and
  `add_component` already give a caller to plan with, so a finding is judged
  against the box the caller was shown. The choice is a small one: in the
  libraries of both builds every pin lies within what the symbol draws for
  about 98 symbols in 100, the common parts all among them, so the box with
  the pins and the box without are the same box, and parts that meet pin to
  pin share an edge, which is not an overlap. They differ only for the few
  symbols with a pin drawn apart from the body.
- An arc counts, in a part's box, as what is drawn of it. The editor took
  the whole ellipse an `ARC` line is cut from and the checker the points of
  the arc it drew, and on LTspice's own example sheets the two boxes differed
  for some part on 43 of the 60 largest: every inductor, whose coil stops
  short of its ellipses, and every polarized capacitor, whose curved plate is
  a sliver of a circle 64 units across. `SymbolArc.extent` is the one
  answer, `SymbolFile.body` and `SymbolFile.bbox` are built on it, and both
  tools' views take a part's two boxes from there. Which way an arc turns is
  read off the stock inductor, whose arcs make a coil one way round and three
  scraps the other; it is the renderer's rule and is not from a recording.
- A second schematic dialect is not built now. The format layer is where one
  would enter.
- A formatted sheet is in LTspice's own form. Both builds were recorded
  saving a sheet: LF line endings, 8-bit text, and the kinds in the order
  the document puts a new record in. Two things a build does on a save are
  not copied, because each would rewrite records an edit did not touch: it
  puts the wires in an order of its own, and it writes the attribute a
  window line names ahead of the instance name.

Settled by a recording of both builds:

- **A part's pins that share a point.** With no wire and no label on the
  point they are not joined to each other. Only the pin highest in
  SpiceOrder touches another part's pin there, and the part's other pins on
  the point are each connected to nothing; a wire or a label on the point
  joins them all. `connectivity.signature` applies this, and so does
  `sheet_findings.floating_pins`, the one reading of a floating pin both
  tools now report. The partition is by coordinate and still groups them.
- **Where a symbol is found.** Beside the sheet, neither build looks into a
  folder for a bare name. For a name that says a folder they differ:
  LTspice 26 looks in that folder, XVII drops the folder and looks right
  beside the sheet. In its own library a build finds a bare name in any
  folder, and a name that says a folder the library does not have.
  `symbol_library.find_symbol` finds a symbol wherever either build would,
  because a part whose symbol is not found has no pins and a connection to
  it goes unseen. spicelib still walks the sheet's folder when it opens a
  sheet, until step 8 removes it.

Open:

- What reading of a block's child sheet is kept once spicelib no longer
  loads it.
- What an edit may add to a sheet saved in a double-byte code page. Such a
  sheet is read as 8-bit text and keeps its bytes, but a character an edit
  adds is written as one byte, which a reader in that code page takes for
  half of a two-byte character. This is so today and is not made worse.

## 8. The rules, after layout rule checking

A layout is checked against its design rules by one small machine: a deck of
rules, results that are markers on the layout carrying what was measured, and
the same deck run while routing, on save and at sign-off. Step 4's one pass
takes its shape from that. The direction is agreed and it was reviewed against
the code; this section is as the review left it. It is not yet built.

**What there is to hold.** The code makes about forty checks today, not the
fifteen first counted: twelve refusals and five advisories in the route
planner, four checks on a label, three on placing a part, three on moving or
removing one, the guard on removing a wire, two on values and directives, five
in the editor's whole-sheet pass, five in the checker's, three more in
`verify_circuit`, and two in the net trace. They live in four modules and
`inspect`. About a third are one kind of check over one set of geometry
(touches, lies in a box, passes through a box, how many at a point, how long,
off the grid). The electrical ones are questions about the net partition and
how an edit changes it. The rest are checks of an op's arguments.

So a rule is not a line of declaration. What is declared is what a caller and
a maintainer need to know about it; what it does stays a function.

- **One registry.** Each rule has an id, a family, a scope (a point, a part,
  a net, the sheet), a disposition at each moment below, a provenance, and a
  limit if it has one. It is the shape a deck rule already has
  (`mcp_surface.md` §6), with the disposition per moment.
- **One index.** Every rule reads the same plain view of a sheet and the same
  net partition, so two rules cannot disagree about where a pin is or what is
  joined.
- **A rule reads a transition.** Its inputs are the sheet before, the sheet
  after, and for a proposal what the op names: its endpoints and the nets
  they were on. Many refusals cannot be told from the sheet as it would be
  alone. A pin under a new wire is a short or the route's own end. A net with
  two names is two named nets joined, or one net given a second name. Only
  the sheet before, or the op, says which.
- **Three moments, one rule.** A proposal is the transition from the sheet to
  the sheet as it would be, with the op; a committed edit is the transition
  from before to after; a whole sheet is the transition from nothing. A
  refusal is then a finding of a rule that blocks at the first moment, and
  what it found is the conflict set. A rule that does not read the op is the
  same check at all three.
- **A finding is a marker.** It carries the rule, the parts, wires and labels
  involved, the points, segments or boxes to look at, and what was measured.
- **Three families, kept apart.** An *electrical* rule says the sheet will
  netlist differently from how it is drawn; it has no limit, and its
  provenance is the recording that shows the behaviour. A *structural* rule
  says something is left undone: a pin on nothing, a label on nothing, a wire
  drawn twice. It has no limit either. A *drawing* rule measures how the sheet
  reads. It reports what it measured, always. A limit is the caller's: given
  one, the rule lists what passes it. The server holds no default, because a
  number of the server's own is a bare magnitude, the weakest ground the
  project's rule on result trust allows; the guide says what limits are
  usual, and each layout practice there names the rule that measures it.
- **What an edit is told.** As now: a finding that is new, any change to it
  making it new, or one that names a part or a point the batch named, by
  any of the parts and points the finding names. An edit can change a
  finding far from where it touched, as when removing a part with its wires
  leaves a pin floating at the far end, or removing one label changes what
  is said of every other label on its net, so the scope is not a region of
  the sheet; each of those is a new finding and is told for that. A further
  criterion was considered, a finding on a net whose membership the batch
  changed, and is not taken: it adds only findings that are unchanged, and
  on a rail it lists every one of them again on every batch (§6, step 4).
- **Every reply says what ran.** The rules that ran are listed with their
  counts, and a rule the caller waived is listed as waived with its count, so
  that no findings is never read as no rules.
- **Text.** A rule about where text is anchored is exact. A rule about how
  far text reaches rests on an extent the server estimates from a font it
  does not have, so its id says it is an estimate and its evidence carries
  the basis.

Severity on a whole sheet stays what it is today, rule by rule. A rule that
blocks a proposal is not thereby an error on a sheet: `verify_circuit` calls a
result partial when a finding is a warning or an error, and a sheet under
construction would be partial throughout.

What is not taken from layout checking: a verdict that a sheet passes, a
general polygon engine (the geometry here is boxes, orthogonal segments and
points), a stored database of results or waivers, and the window, which is
how an incremental check bounds its work and is no way to say what an edit
caused.

Two things the one pass will newly make uniform. A move has no check that it
joins two named nets, where drawing a wire and placing a label both have. And
the tool contract says a wire over a symbol's body is an error, where the code
warns.

The three checks a layout is signed off with are already the three things
checked here, which is a way to read the tools and not a change to them:
rule checking is the `layout` and `quality` checks, electrical rule checking
is the connectivity rules run on a netlist, and layout against schematic is
`compare`, with the signature of §3 as the same check made on every edit that
only changes the drawing.
