# Declared native PDK profiles

**Status: proposal for review. Nothing here is implemented.** The contract
in force is "Native PDK experiment contract" in
[mcp_surface.md](mcp_surface.md); it stays as written until this design is
accepted and built.

This document proposes how a user declares a native PDK statistical profile
of their own, so that `pdk_native` variations can sample PDKs, devices and
device forms that the one packaged profile does not cover. It covers the
profile file and its configuration, the trust label on results, the checks
the server can make without knowing the PDK, how a device built as `m`
parallel copies is sampled, how one die can be held across temperature
corners, an `inspect` query that helps an agent draft a profile, a test plan,
and the questions left for the maintainer.

---

## 1. Summary

A `pdk_native` variation (`run_experiments`, `kind: "pdk_native"`) runs each
sample in its own ngspice process, seeded before the PDK's models load, so the
PDK's own statistical expressions (`agauss` and the like) draw the process
and mismatch values. Today one profile exists, `sky130-e6f9c887-ngspice-v1`,
and it accepts one device, `sky130_fd_pr__nfet_01v8`, with `m`, `mult` and
`nf` all 1. A PMOS load, another PDK, or a device written as `m` fingers
cannot get native statistics.

The proposal:

- **A profile file** (TOML) declares what is PDK-specific: the library entry
  file, the `.lib` section and deck-written switch parameters for each mode,
  the devices that carry statistics, how `m` and `nf` are treated, and
  optionally a hash manifest. `[simulation] pdk_profiles` in
  `ltspice-mcp.toml` (or `LTSPICE_MCP_PDK_PROFILES`) lists the files (§3).
- **Trust labels.** Every `native_statistics` record says `trust:
  "recorded"` for the packaged profile and `trust: "declared"` for a profile
  from the configuration. `inspect(kind="capabilities")` lists every profile
  with its label (§4).
- **Three checks that need no knowledge of the PDK** run for every profile:
  the hashes of the PDK files the deck actually read are recorded; a section
  whose random functions cannot reach the devices is refused, so a misnamed
  switch cannot produce N identical runs labelled as Monte Carlo; and every
  MOSFET must resolve to a declared device (§5).
- **Multiplicity.** Sky130's mismatch expressions divide by
  `sqrt(l*w*mult)`, and ngspice builds `m` copies that share one draw. So a
  device is sampled correctly exactly when `mult` equals its total `m`. The
  proposal passes `m` through and checks that equality for profiles that name
  such a parameter (Option B), and keeps expansion into separate instances
  (Option A) for PDKs whose expressions have no such parameter. Measurements
  are in §6.
- **Temperature.** One seed draws the same die at every temperature on
  ngspice (measured: offset correlation 0.9998 between 27 °C and 85 °C). A
  `hold_die_across` list on the variation drops the named condition
  parameters from the seed derivation, after checking that they cannot change
  the sequence of draws (§7).
- **`inspect(kind="pdk_library")`** lists a library's sections, its switch
  parameters and the models whose expressions draw random values, and can
  check a draft profile against the library before the user declares it (§8).

PDKs with only Spectre or HSPICE views stay out of scope (§10).

---

## 2. What is PDK-specific today

Most of the native machinery knows nothing about Sky130: the per-sample seed
derivation, one ngspice process per sample (`-n`, `ngbehavior=hsa`), the
staging of every input, the capture and hashing of original bytes, the
preparation and verification of launch artifacts, retry from frozen inputs,
and the `native_statistics` record. The driver each sample runs (`set
ngbehavior=hsa`, `set ng_nomodcheck`, `setseed`, `source`, `run`, `write`) has
no PDK in it.

What is Sky130-specific is a handful of constants and two checks in
`src/ltspice_mcp/lib/pdk_native.py`:

| Today | What it encodes | Becomes |
|---|---|---|
| `PROFILE` | the profile name | `name` |
| `ENTRYPOINT = "libs.tech/ngspice/sky130.lib.spice"` | the library file the deck's one `.lib` line names; the PDK root is inferred from it (`native_inputs.py`) | `entry` |
| `_MODES` | `nominal` → `tt`, `mismatch` → `tt_mm`, `process` → `mc`, `combined` → `mc` | `modes.<mode>.section` |
| `_validate_controls`, combined branch | `combined` needs one root `.param mc_mm_switch=1` after the `.lib` line | `modes.combined.deck_params` |
| `_OWNED` (`mc_mm_switch`, `mc_pr_switch`) | switches a deck may not set | derived (§3.4) |
| `WRAPPER`, `MODEL`, the expected model file in `_validate_coverage` | the one accepted device, its file per section, and its bin names | `devices` (§5.3) |
| `leaf.scale != 1e-6` | the scale Sky130's `all.spice` sets | `scale` |
| `m`, `mult`, `nf` must be 1 | no multiplicity | `multiplicity` (§6) |
| `profile_pins()`, `PIN_MANIFEST_SHA256` | SHA-256 of every PDK file at one open_pdks revision | `manifest` |
| `profile_identity` in `provenance()` | the open_pdks and skywater revisions | `identity` |

The hash pin is a trust choice rather than a technical need: it makes
"supported" mean "recorded on these exact bytes". A declared profile keeps
the option (`manifest`) but does not require it, and its results say so.

---

## 3. The profile file

### 3.1 Configuration

The profiles a session may use are listed in `[simulation]`:

```toml
[simulation]
# Native PDK statistical profiles you declare, one file each. A pdk_native
# variation names a profile by the name written inside its file.
pdk_profiles = ["C:/pdks/profiles/sky130-local.toml"]
```

- `ServerConfig.pdk_profiles: list[Path]`, default empty. One `_Setting` row:
  section `simulation`, key `pdk_profiles`, read with
  `_toml_path_list("simulation.pdk_profiles")` (a bare string is refused, as
  for `symbol_paths`), environment variable `LTSPICE_MCP_PDK_PROFILES` read
  with `_env_path_list` (entries separated by `os.pathsep`, as
  `LTSPICE_MCP_SYMBOL_PATHS` is). `Api(pdk_profiles=[...])` overrides both,
  following `python_api.md` §3.
- A relative entry resolves as `symbol_paths` entries do.
- The generated default config shows the key commented out with an example,
  through `_shown_default`.
- The list is re-read when the config file changes, as `[security]
  allowed_paths` is, so a profile can be added without a restart.
- The profile file is configuration, like `ltspice_ini`, so `[security]
  allowed_paths` does not restrict where it lives. The PDK it describes is
  read through the deck, as today, and must lie inside the sandbox when one
  is set.

A list of files is used rather than a `name = path` table (the form of
`[simulator.executables]`) so that a profile's name is written once, in the
file that defines it, and the name the record carries cannot disagree with
the file.

The profile is read from configuration rather than from the request, so the
declaration lives in one place the user controls and that
`inspect(kind="capabilities")` can show. The label states what the profile
is, not who wrote it: with `run_code` served, a snippet can construct an
`Api` with any profile, and its results are still labelled `declared`. The
drafting loop an agent needs is served by `inspect` (§8.4) without letting a
request declare trust.

### 3.2 Example

A declared profile for the Sky130A ngspice library, covering the 1.8 V NMOS
and PMOS devices and devices written with `m` fingers:

```toml
schema = 1
name = "sky130-local"
simulator = "ngspice"
description = "Sky130A from a local open_pdks build: 1.8 V nfet and pfet"
entry = "libs.tech/ngspice/sky130.lib.spice"
scale = 1e-6
manifest = "sky130-local.manifest.json"   # optional

[modes.nominal]
section = "tt"

[modes.mismatch]
section = "tt_mm"

[modes.process]
section = "mc"

[modes.combined]
section = "mc"
deck_params = { mc_mm_switch = 1 }

[[devices]]
subckt = "sky130_fd_pr__nfet_01v8"

[[devices]]
subckt = "sky130_fd_pr__pfet_01v8"

[multiplicity]
m = "formula"
parameter = "mult"
nf = "pass"

[identity]   # copied into the record under the declared label; never checked
open_pdks_revision = "e6f9c8876da77220403014b116761b0b2d79aab4"
```

A profile per corner is also possible: one whose `mismatch` mode selects
`ss_mm` gives mismatch samples at the slow corner.

### 3.3 Schema (version 1)

| Field | Type | Required | Meaning |
|---|---|---|---|
| `schema` | integer | yes | `1`. Any other number is refused, so a later format cannot be misread. |
| `name` | string | yes | What `pdk_native.profile` names. `[A-Za-z0-9][A-Za-z0-9_.-]{0,63}`. It may not equal a packaged profile's name or another declared profile's name. |
| `simulator` | string | yes | `"ngspice"`, the only family with the per-sample launch. |
| `description` | string | no | Shown by `inspect(kind="capabilities")`. |
| `entry` | string | yes | The library file, relative to the PDK root, `/`-separated, with no `..`, drive or leading `/` (the rule `_relative_identity` applies today). The deck's one `.lib` line must name a file whose path ends with it, and the PDK root is inferred from that path, as today. |
| `scale` | number | no | The scale the library sets (`.option scale`). When given, every covered device's resolved scale must equal it. |
| `manifest` | string | no | A JSON file, relative to the profile file, in the packaged manifest's format: a list of `{path, sha256, size}`. When given, every PDK file the deck reads must be listed with that hash. |
| `modes.<mode>.section` | string | per mode | The `.lib` section the deck's line must select for that mode. `<mode>` is one of `nominal`, `mismatch`, `process`, `combined`. A mode left out is refused for this profile. |
| `modes.<mode>.deck_params` | table of name → number | no | Root `.param` assignments the deck must write after its `.lib` line for that mode, exactly these, with these values. Each name must be a parameter the PDK defines in that section (§5.4). |
| `devices[].subckt` | string | one of the two | A subcircuit the deck calls with an X line, as Sky130 wraps each transistor. |
| `devices[].model` | string | one of the two | A model, or a family of bins `name.1`, `name.2`…, that an M line names directly. |
| `multiplicity.m` | `"refuse"`, `"formula"` or `"expand"` | no; default `"refuse"` | How a device built as `m` parallel copies is sampled (§6). Under `"refuse"` every `m` on a covered device's path, and the named `parameter` if any, must be 1. `"expand"` is reserved and refused until it is built. |
| `multiplicity.parameter` | string | with `"formula"`; optional otherwise | The device parameter by which the PDK's statistical expressions scale the area (`mult` in Sky130). |
| `multiplicity.nf` | `"refuse"` or `"pass"` | no; default `"refuse"` | Whether a covered device may have `nf` other than 1. `"pass"` is right when the PDK's statistical area uses the total width (§6.1). |
| `identity` | table of strings | no | Free-form facts, such as a revision, copied into the record under the declared label. Never checked. |

The four mode names stay fixed, so the `mode` field of the published input
schema does not change.

### 3.4 Loading and validation

- The file is read as bytes and parsed with `tomllib`. TOML is UTF-8 by
  definition, so the platform code page never applies. A size cap (64 KiB)
  bounds the read, and the manifest has its own cap (16 MiB).
- Unknown keys are refused, so a misspelt `deck_param` cannot silently drop a
  requirement.
- At startup (and on a config change) each file is parsed and checked
  against the schema. A file that fails becomes a startup diagnostic, which
  `inspect(kind="capabilities")` already reports, and its name, when it can be
  read, refuses requests with that diagnostic as `pdk_native_request`. The PDK
  is not read at load time; it is not known until a deck names it.
- When a request names the profile, its file is read again and hashed, and
  the bytes read then are the ones the request uses and records, as deck
  inputs are captured once per request.
- The owned parameters, which the deck may not set, are the adapter's
  (`seed`, `randomseed`, `ngbehavior`, `ng_nomodcheck`, `scale`), every
  root-level `.param` that the active PDK files define (computed per case,
  as `_owned_parameters` does today), and every name in any mode's
  `deck_params`. The deck may set only the current mode's `deck_params`, with
  exactly those values, at root scope, after the `.lib` line. This is
  today's rule, with the names taken from the profile. A profile names only
  the switches the deck writes; the switches a section sets itself are the
  PDK's business, and the reach check (§5.2) verifies their effect.

### 3.5 The packaged profile in the same form

The packaged profile becomes data in this schema
(`assets/pdk_profiles/sky130-e6f9c887-ngspice-v1.toml` beside its existing
manifest), loaded by the same code, so there is one code path. Its values:
the entry and sections above, `deck_params = { mc_mm_switch = 1 }` for
`combined`, one device `sky130_fd_pr__nfet_01v8`, `scale = 1e-6`,
`multiplicity.m = "refuse"`, `multiplicity.parameter = "mult"` (so `mult`
must be 1, as today), `nf = "refuse"`, and the manifest. The manifest's own
digest stays pinned in code (`PIN_MANIFEST_SHA256`), so a packaged profile
cannot be edited in place.

Two packaged checks are not fields: the expected model file for each section
(`sky130_fd_pr__nfet_01v8__tt.pm3.spice` for `tt` and `tt_mm`,
`sky130_fd_pr__nfet_01v8.pm3.spice` for `mc`) and the bin names
(`sky130_fd_pr__nfet_01v8__model.N`). The generic device check (§5.3)
requires the wrapper's active definition and the leaf's candidate models to
come from PDK files. Under the packaged manifest the only active definition
of the wrapper in each section is that file, so the generic check accepts and
refuses the same decks. The existing tests in `tests/test_pdk_native.py` and
`tests/test_native_inputs.py` are kept unchanged to show it.

### 3.6 Identity and the seed

The seed key today is the canonical JSON of the derivation version, circuit
id, family id, input digest, assignments, profile name, mode, section and
sample index. It changes as follows:

- **Packaged profile:** unchanged, byte for byte, so every existing replay
  still reproduces. A test pins keys and seeds derived by the current code as
  golden values before the refactor.
- **Declared profile:** the key gains `profile_digest`, the SHA-256 of the
  canonical JSON of the fields that decide how a deck is checked and run:
  `schema`, `simulator`, `entry`, `scale`, `modes`, `devices`,
  `multiplicity`, and the manifest's SHA-256. The name is already in the key.
  `description` and `identity` are left out, so editing a description does
  not change the dice. Any other edit draws new dice, which is right: a
  different declaration is a different experiment.
- `DERIVATION_VERSION` stays `pdk-native-sha256-v1`: packaged keys do not
  change, and declared keys carry a profile name no packaged key has.

---

## 4. Trust labels

### 4.1 Meaning

- **`recorded`**: the project established the profile. Its four modes have
  been run live against the pinned PDK and compared with ngspice run directly
  (`tests/test_native_experiments.py`), and every PDK file hash is packaged.
- **`declared`**: the server's configuration declares the profile. The
  server applies every check in §5 and the profile's own fields, but nobody
  has recorded that its section, switch and device table is right for that
  PDK.

A manifest in a declared profile does not make it recorded. It pins the
bytes, which is a separate fact (`file_pins` below), and says nothing about
whether the mode table fits them.

The capability table already uses `recorded:` as an evidence kind, meaning a
behaviour key of the LTspice inventory. The profile label is a different
thing, and a packaged profile's evidence uses the existing `test:` kind. §12
asks whether to rename one of the two.

### 4.2 In `native_statistics`

The record gains one top-level field and a few under `validated`:

```text
native_statistics:
  origin: "pdk_native"
  trust: "recorded" | "declared"              # new, on every case, failed ones included
  requested: {circuit_id, family_id, profile, mode, root_seed, sample_index?}
  validated:
    ...existing fields...
    file_pins: "packaged" | "declared_manifest" | "none"      # new
    pdk_files: [{path, sha256, size}]       # new, full form; compact form: pdk_file_count
    random_reach: {...}                     # new, §5.2
    multiplicity: {m, parameter?, nf}       # new, the policy applied
    held_assignments: [name, ...]           # new, §7, when used
    controls: {name: value}                 # packaged: as today; declared: §5.2
    profile_identity:
      packaged: unchanged
      declared: {profile_sha256, entry, identity?}
```

`trust` sits at the top because the profile is resolved at request time, so
it is known for a case whose validation failed. It is in the compact form, so
it reaches receipts and the source identity of every result read from the
case (`services.py` puts the compact record in that identity). Both labels
are non-empty strings, so the reply rule in `lib/reply_fields.py` never drops
them; `file_pins: "none"` is a string for the same reason.

`NativeCaseRecord.from_record` must read job records written before these
fields existed: `trust` defaults to `recorded`, since only the packaged
profile existed then, and the missing facts go under `unavailable` with the
reason that the record predates them.

### 4.3 In the capability table

`capabilities.py` describes simulator families, and a profile is not a
family. The `ngspice` / `pdk_statistics` entry stays `supported` with its
evidence, because the machinery it names is the same for every profile. Its
`mechanism` text gains a clause: "packaged profiles are recorded; profiles
from `[simulation] pdk_profiles` are declared".

Profiles get a section of their own:

- `capabilities.PROFILES` lists the packaged profiles, each with `trust:
  "recorded"`, its evidence (`test:tests/test_native_experiments.py`) and its
  modes and devices. `tests/test_capabilities.py` resolves the evidence as it
  does for entries. `table_report()`, behind `spice://capabilities`, includes
  them.
- `inspect(kind="capabilities")` gains a `pdk_profiles` field (a new
  `CapabilityField`; `tests/test_inspect_tools.py` holds the list equal to
  the report). It lists every profile this session can use, keyed by name:

```json
{"sky130-e6f9c887-ngspice-v1": {"trust": "recorded", "simulator": "ngspice",
   "entry": "libs.tech/ngspice/sky130.lib.spice",
   "modes": {"nominal": "tt", "mismatch": "tt_mm", "process": "mc", "combined": "mc"},
   "devices": ["sky130_fd_pr__nfet_01v8"],
   "multiplicity": {"m": "refuse", "parameter": "mult", "nf": "refuse"},
   "file_pins": "packaged"},
 "sky130-local": {"trust": "declared", "path": "C:/pdks/profiles/sky130-local.toml",
   "profile_sha256": "<64 hex digits>", "simulator": "ngspice",
   "entry": "libs.tech/ngspice/sky130.lib.spice",
   "modes": {"nominal": "tt", "mismatch": "tt_mm", "process": "mc", "combined": "mc"},
   "devices": ["sky130_fd_pr__nfet_01v8", "sky130_fd_pr__pfet_01v8"],
   "multiplicity": {"m": "formula", "parameter": "mult", "nf": "pass"},
   "file_pins": "declared_manifest"},
 "broken": {"trust": "declared", "path": "C:/pdks/profiles/broken.toml",
   "problem": "modes.combined: 'section' is required"}}
```

Declared profiles are session configuration, so they appear in the session
view only, never in the static resource.

The `profile` field's description in the `pdk_native` input schema becomes a
fixed sentence ("the packaged sky130-e6f9c887-ngspice-v1, or a profile
declared in [simulation] pdk_profiles; inspect(kind='capabilities') lists
them"). It must not depend on the configuration, because `tools/list` bytes
are cached by clients and by prompt caching.

---

## 5. Checks that need no knowledge of the PDK

### 5.1 Hashes of the PDK files read

The validator already captures the original bytes of every file the final
hierarchy reads (`hierarchy.inputs`), staged copies included, and classifies
each as a PDK file (`pdk_relative`) or a bench file. The record exposes a
digest over the PDK files (`model_digest`) and the staged copies'
hashes (`prepared.dependencies`), but not each original PDK file's hash,
because staging can rewrite include paths.

Proposed:

- `ValidatedSample` gains `pdk_files`: each PDK file read, as its
  root-relative path, the SHA-256 of its original bytes, and its size. These
  are the files ngspice reads for the selected section, not the whole
  installation: for Sky130 a walk of the include lines finds 169 files
  (6.9 MB) for `tt` and `tt_mm` and 124 files (5.3 MB) for `mc`. The full
  record lists them; the compact form gives `pdk_file_count` beside the
  existing `model_digest`.
- With a manifest, every PDK file must be listed with the same hash
  (`validate_captures`, as today), and `file_pins` is `packaged` or
  `declared_manifest`. Without one, nothing is compared, `file_pins` is
  `none`, and the list is the evidence of what ran. It is also a manifest the
  user can save to pin the profile later (§8.2, the `files` view).
- For a declared profile, a file that a PDK file includes from outside the
  inferred root is refused (`closure`), so that `pdk_files` is the whole of
  what the PDK loaded. Under the packaged manifest such a file cannot occur.

Hashes are of the bytes as installed. A PDK checked out on Windows with
line-ending conversion has different hashes from the same revision on Linux,
so a manifest made on one does not match the other. The archives that the
`volare` and `ciel` installers publish are not converted. The guide should
say so.

Placement: the list is built in `pdk_native.validate_sample`, beside
`model_digest`, from the captures `native_inputs.NativeCaseValidator.validate`
already supplies. The closure check sits in `NativeCaseValidator.validate`,
where the PDK root is inferred.

### 5.2 Random reach

**The failure it prevents.** In Sky130, `tt` and `tt_mm` include the same
files (`corners/tt.spice`), and both contain the same 540
`sky130_fd_pr__nfet_01v8` mismatch terms of the form
`MC_MM_SWITCH*AGAUSS(...)`. Only the section's `.param
mc_mm_switch=1` makes them draw. A profile that maps `mismatch` to `tt` and
asks the deck for `.param mm_switch=1` (a misspelt name) runs, and every
sample is identical. Measured with ngspice 42 on the pinned files: seeds 1
and 2 both give 0.2956537 µA for the same device. A purely textual check
("the selected closure contains a random function") passes this profile,
because the terms are there. The check has to evaluate whether a draw
changes anything.

**Random functions.** The ngspice functions evaluated when the deck is read:
`gauss`, `agauss`, `unif`, `aunif` and `limit`. Functions evaluated during an
analysis, such as transient noise sources, are not process statistics and are
not counted.

**Classes.** A random call in a root-level `.param` or root-level model card
is *global*: ngspice evaluates it once per run, so every device that reads it
shares the draw, which is process variation. A random call inside a `.subckt`
body (its `.param` lines or its model cards) is *per-instance*: ngspice
evaluates it once for each instance, which is mismatch. Sky130 has both:
`parameters/critical.spice` holds 67 global terms such as

```spice
.param lvn_threshold = 0.0 + MC_PR_SWITCH*AGAUSS(0,0.6,1)
```

which `parameters/montecarlo.spice` feeds into each device's model through
parameters such as

```spice
+ sky130_fd_pr__nfet_01v8__vth0_diff_0 = '-3.20594e-04*lvn_threshold*lvn_threshold+1.97749e-02*lvn_threshold+-2.44410e-02'
```

and each device wrapper's model cards hold the per-instance terms quoted in
§6.1.

**Liveness.** Each expression that contains a random call of a class, or
reads (through parameters) a root parameter that does, is evaluated twice:
with each random call of that class at its nominal value, and at one
standard deviation above it (`agauss(n,a,s)` → `n+a/s`, `gauss(n,r,s)` →
`n·(1+r/s)`, `aunif(n,a)` → `n+a`, `unif(n,r)` → `n·(1+r)`, `limit(n,a)` →
`n+a`). The call is *live* if the two values differ, *dead* if they are
equal, and *undetermined* if the expression cannot be evaluated (an
unresolved name, or a function the evaluator does not model). A zero
variation (`agauss(0,0,1)`) is dead. Values come from the case's own
closure: the section's parameters, the deck's `deck_params` (later root
definitions win, as the existing `combined` rule relies on), and, for a
per-instance term, the instance's resolved parameters (`l`, `w`, `mult` and
so on, which the hierarchy already resolves per instance).

**Reaching a device.** A covered device (§5.3) is reached by a class when at
least one parameter of one of its candidate model bins, or of its wrapper,
changes under that class's perturbation. Only the candidate bins the
hierarchy already keeps per device (`MosCoverage.models`) are evaluated, and
within them only the expressions that contain or read a random term: three
or four per bin in Sky130 (§6.1). The dependency pass over root parameters
is done once per capture set and cached, as `NativeCaseValidator` already
caches verified originals.

**What each mode requires**, of every covered device:

| Mode | Global class | Per-instance class |
|---|---|---|
| `nominal` | reaches none | reaches none |
| `mismatch` | reaches none | reaches every device |
| `process` | reaches every device | reaches none |
| `combined` | reaches every device | reaches every device |

"Every device" is deliberate: a statistical run in which half the circuit
stays nominal is the same silent hole as a run in which all of it does. The
refusal names the instances that are not reached. "None" catches a section
that would put process variation into a mismatch study, or vary a nominal
run.

**Outcome.** A determinate violation fails the case with
`pdk_native_validation`, reason `random_reach`, naming the class, the
instances and the gating parameters with their values. If the outcome is
undetermined for some device, the case runs, and the record says so:
`validated.random_reach = {status: "undetermined", unresolved: [{file, line,
reason}]}` in full, with counts in the compact form, plus an observation on
the job. Otherwise `random_reach.status` is `"live"` or, for `nominal`,
`"none"`, with per-class counts. `validated.controls` for a declared profile
lists the root parameters set to 0 or 1 that the random terms reaching the
covered devices read, with their effective values: for Sky130 `combined`,
`{"mc_mm_switch": 1, "mc_pr_switch": 1}`, the same block the packaged profile
reports today.

**Evaluator.** `hierarchy_expr.evaluate` resolves static arithmetic only (no
calls) and caps an expression at 128 syntax nodes. The reach check needs a
call node for the five random functions (stubbed as above) and for the
common math functions (`sqrt`, `abs`, `exp`, `log`, `min`, `max`, `pow`). It
should be an explicit table, with anything else undetermined. Sky130's
statistical expressions use only `sqrt` and arithmetic and fit the node
cap. The extension lives in a new module, `lib/random_reach.py`, which takes
the active cards and per-instance environments and returns a `RandomReach`
value, so it can be tested without a simulator.

Placement: computed in `NativeCaseValidator.validate` after `load_hierarchy`,
carried in `CaseInputs` (a new `reach` field), and judged against the mode in
`pdk_native.validate_sample`, where today's mode checks are. Keeping the
verdict in `validate_sample` keeps it testable with synthetic `CaseInputs`,
as `tests/test_pdk_native.py` does now.

The packaged profile runs the same check. On the pinned files its four
sections give exactly the table above (§11.3 tests that), so no case that
passes today is refused.

### 5.3 Every MOSFET resolves to a declared device

`native_inputs._coverage` already lists every physical MOS leaf (element
`M`) in the final hierarchy, with its wrapper, its candidate model bins, its
resolved geometry and its multiplicity facts. `pdk_native._validate_coverage`
then requires the Sky130 specifics. The generic rule:

1. A leaf is attributed to a `devices[].subckt` when its nearest X ancestor
   instantiates that subcircuit, and that subcircuit's active definition is
   in a PDK file.
2. Otherwise a leaf is attributed to a `devices[].model` when its candidate
   models (`model_family`) are that model or its bins, defined in a PDK file.
3. Any other MOS leaf fails the case with reason `coverage`, naming the
   instance and what it resolved to: an undeclared PDK device, a model
   defined in the bench (which carries no PDK statistics; a `random`
   variation with a `mismatch` rule covers that case), or no model at all.
4. A covered leaf needs W and L that resolve to finite positive SI values,
   the profile's `scale` when it gives one, and multiplicity facts that meet
   the profile's policy (§6). These checks exist today; only the constants
   come from the profile.

A wrapper may hold more than one MOS leaf, for example a device with a gate
resistor or extra parasitic transistors. Each leaf is listed and attributed
to the same device.

Non-MOS PDK devices (resistors, capacitors, bipolars) are not checked today
and are not checked here. They take whatever statistics the selected section
gives them, and the record does not list them. §12 asks whether a profile
should be able to declare them.

Placement: attribution in `native_inputs._coverage`, which gains a `device`
field on `MosCoverage`; the policy in `pdk_native._validate_coverage`, which
takes the profile.

### 5.4 Smaller checks

- Each `deck_params` name must be a root parameter that an active PDK file
  defines in the selected section; otherwise the case fails with reason
  `deck_params`. This catches a misspelt switch before the reach check, with
  a plainer message.
- A request for a mode the profile does not declare is refused at request
  time (`pdk_native_request`).
- A request naming a profile whose file failed to load is refused at request
  time, with the load problem.

### 5.5 Where everything sits

```text
run_experiments
  validate_request (pdk_native)           profile known and loaded, mode declared,
                                          simulator matches           -> pdk_native_request
  staging, variation expansion,
  materialization                         unchanged
  NativeCaseValidator.validate (native_inputs), per case
    load_hierarchy (ngspice, hsa)         unchanged
    binding: one .lib naming entry        exists; entry from the profile
    captures of every file read           exists
      closure under the PDK root          new, declared profiles      (5.1)
      manifest comparison                 exists; manifest from the profile
    coverage rows with device attribution generalized                 (5.3, 6)
    random reach                          new                         (5.2)
  validate_sample (pdk_native)
    controls and deck_params              generalized                 (3.4, 5.4)
    coverage policy, multiplicity         generalized                 (5.3, 6)
    reach verdict                         new                         (5.2)
    pdk_files, seed key                   new field; key per 3.6 and 7
  prepare_native_cases (native_execution) unchanged: the driver does not depend on the profile
  NativeCaseRecord, provenance            trust and the new fields    (4.2)
```

`prepare_native_retry` reuses the frozen preparation and reads neither the
PDK nor the profile, as today; the record's `profile_sha256` says which
declaration the case was prepared under.

---

## 6. Multiplicity

### 6.1 What Sky130's expressions do with `mult` and `nf`

Read from the open_pdks build of revision
`e6f9c8876da77220403014b116761b0b2d79aab4`, taken from the archives the
`volare`/`ciel` installers publish for that revision. All 821 files match the
packaged manifest byte for byte.

The NMOS wrapper, identical in the `tt` file
(`libs.ref/sky130_fd_pr/spice/sky130_fd_pr__nfet_01v8__tt.pm3.spice`) and
the `mc` file (`sky130_fd_pr__nfet_01v8.pm3.spice`):

```spice
.subckt  sky130_fd_pr__nfet_01v8 d g s b
+ 
.param  l = 1 w = 1 nf = 1.0 ad = 0 as = 0 pd = 0 ps = 0 nrd = 0 nrs = 0 sa = 0 sb = 0 sd = 0 mult = 1
msky130_fd_pr__nfet_01v8 d g s b sky130_fd_pr__nfet_01v8__model l = {l} w = {w} nf = {nf} ad = {ad} as = {as} pd = {pd} ps = {ps} nrd = {nrd} nrs = {nrs} sa = {sa} sb = {sb} sd = {sd}
```

The mismatch terms of bin 0 of the `tt` file:

```spice
+ toxe = {4.148e-09+MC_MM_SWITCH*AGAUSS(0,1.0,1)*(4.148e-09*1.0*(sky130_fd_pr__nfet_01v8__toxe_slope/sqrt(l*w*mult)))}
+ vth0 = {0.5190093+MC_MM_SWITCH*AGAUSS(0,1.0,1)*(sky130_fd_pr__nfet_01v8__vth0_slope/sqrt(l*w*mult))}
+ voff = {-0.1052686+MC_MM_SWITCH*AGAUSS(0,1.0,1)*(sky130_fd_pr__nfet_01v8__voff_slope/sqrt(l*w*mult))}
```

with the coefficients in `sky130_fd_pr__nfet_01v8__mismatch.corner.spice`:

```spice
.param sky130_fd_pr__nfet_01v8__toxe_slope=3.443e-03
.param sky130_fd_pr__nfet_01v8__voff_slope=0.007
.param sky130_fd_pr__nfet_01v8__vth0_slope=3.356e-03
```

In the `mc` file one expression carries both process and mismatch:

```spice
+ vth0 = {0.49439+sky130_fd_pr__nfet_01v8__vth0_diff_0+MC_MM_SWITCH*AGAUSS(0,1.0,1)*(sky130_fd_pr__nfet_01v8__vth0_slope/sqrt(l*w*mult))}
```

Every random term in the four files that matter here divides by
`sqrt(l*w*mult)`:

| File | Bins | Random terms | With `sqrt(l*w*mult)` | Parameters drawn |
|---|---|---|---|---|
| `sky130_fd_pr__nfet_01v8__tt.pm3.spice` (`tt`, `tt_mm`) | 180 | 540 | 540 | `toxe`, `vth0`, `voff` |
| `sky130_fd_pr__nfet_01v8.pm3.spice` (`mc`) | 63 | 241 | 241 | `toxe`, `vth0`, `voff`, `nfactor` (52 bins) |
| `sky130_fd_pr__pfet_01v8__tt.pm3.spice` (`tt`, `tt_mm`) | 108 | 432 | 432 | `toxe`, `vth0`, `voff`, `nfactor` |
| `sky130_fd_pr__pfet_01v8.pm3.spice` (`mc`) | 66 | 264 | 264 | `toxe`, `vth0`, `voff`, `nfactor` |

What follows from these lines:

- **`nf` never appears in a statistical term.** `w` is the device's total
  width (BSIM4 divides it among `nf` fingers), so `l*w` is the whole gate area
  and the Pelgrom scaling is right for any `nf`. `nf` changes only the
  per-finger width that selects the bin, and the diffusion geometry.
- **`mult` is a parameter of the wrapper alone.** It is not passed to the
  inner device, whose card has no `m=`. So it changes the spread and not the
  current.
- **`m` on the X line** makes ngspice build the inner device as `m`
  parallel copies of one card. They share that card's single evaluation, and
  so its single draw, and the expressions do not see `m`.
- So the PDK samples a device correctly exactly when `mult` equals the
  device's total `m`: the product of every `m` from the root down to the
  wrapper, as `hierarchy_expr.multiplier_product` computes it for the
  `random` mismatch rule (guide section 'variations', "Pelgrom mismatch").

gmid-workbench's deck generator (`tools/verify/gen_decks.py`) writes each
device as the characterization device times `m`:

```text
L=<L in µm> W=1 nf=1 m=<W / 1 µm>
```

with no `mult`. Passed through unchanged, such a device draws √m times its
correct spread: ten times at `m=100`.

### 6.2 Measurements

ngspice 42 on a four-core Linux machine, using the pinned files, section
`tt_mm` (mismatch only), L = 0.5 µm, Vgs = 0.6 V, Vds = 0.9 V, 100 seeds,
every variant in one deck so each seed is one library load. "sd/mean" is
the drain current's relative spread. Its standard error is about 7 % of
itself (±1.4 points at 19 %, ±0.15 at 2 %).

| Deck form | Mean current | sd/mean |
|---|---|---|
| `W=1` (one finger) | 0.307 µA | 19.4 % |
| `W=1 m=100` | 29.3 µA | **18.9 %** |
| `W=1 m=100 mult=100` | 29.6 µA | 1.91 % |
| 100 separate `W=1` instances | 29.8 µA | 2.13 % |
| `W=99` | 55.7 µA | 2.08 % |
| `W=99 nf=99` | 29.3 µA | 1.88 % |
| `W=1 mult=100`, no `m` | 0.296 µA | 2.07 % |

The one-finger spread divided by √100 is 1.94 %. (`W=100` is outside the
bins, so the wide devices use 99 µm.) Reading the table:

- `m=100` alone keeps the one-finger spread: the copies share one draw.
- `m=100 mult=100` matches 100 independent instances and one wide device.
- `nf` passes: `W=99 nf=99` and `W=99 nf=1` both scale by area.
- `mult` without `m` is a one-finger device with the spread of a 100-finger
  one, so it is wrong in the other direction.
- The mean of `m=100 mult=100` is 0.9 % below that of 100 instances, about
  three standard errors. Summing independent copies of a current that is
  convex in the threshold voltage predicts a difference of this sign; a
  single scaled draw does not reproduce it. It is a second-order effect near
  threshold.

The cost of expanding into instances, one seeded load and `.op`:

| Deck | Load and `.op` | Peak resident memory |
|---|---|---|
| 1 instance | 6.9 s | 158 MiB |
| 1 instance, `m=100` | 6.8 s | 157 MiB |
| 100 instances | 7.1 s | 158 MiB |
| 1000 instances | 8.5 s | 158 MiB |
| 3000 instances | 12.4 s | 177 MiB |

and with `.tran 1n 10u` (10,008 points) into a resistor load:

| Deck | Load and `.tran` |
|---|---|
| 1 instance, `m=100 mult=100` | 6.4 s |
| 100 instances | 9.0 s |
| 1000 instances | 32.0 s |

Loading the library dominates: expansion adds about 1.8 ms per instance at
load, and about 2.5 µs per instance per time point in a transient. An OTA
written with 300 µm of total width in 1 µm fingers becomes 300 instances:
about half a second more per sample at load, and about 7.5 s more per sample
in a 10,000-point transient.

### 6.3 Option A: expand `m` into instances

Each X or M line with `m = k > 1` on the path to a covered device is written
as `k` lines without `m` in the staged copy, with generated names, so each
copy is evaluated, and drawn, separately.

- **Correct without reading the PDK's formulas**, provided its mismatch is
  per instance (which the reach check establishes), and it reproduces the
  second-order effect above.
- **It does need the multiplicity parameter anyway** when the PDK has one.
  A deck written `m=100 mult=100` and expanded gives each copy the spread of
  a 100-finger device, so the sum of the 100 copies is ten times too tight.
  Under expansion the parameter must be 1 on every copy, so the server must
  know its name.
- **It renames devices.** A probe of the authored device
  (`@m.xm1.msky130_fd_pr__nfet_01v8[gm]`, a `.save` of its current, an
  `analyze_results` recipe reading it) no longer exists under that name.
  Keeping the first copy's name would read a hundredth of the current.
- **Some decks cannot be expanded textually.** An `m` inside a subcircuit
  body that differs between instances of that subcircuit (`m={nfing}`)
  needs a clone of the subcircuit per value. Those would be refused.
- **Lineage and records grow.** Every copy must map back to its authored
  card for coverage, assignments and the record. Listing copies as a count on
  the authored row (`copies: 100`) keeps the record small.
- **Cost** as measured: small at load, proportional to instances in a
  transient.

### 6.4 Option B: pass `m` through when the PDK's formulas handle it

The deck runs as written. A profile with `multiplicity.m = "formula"` names
the parameter its expressions scale by, and the server checks, for every
covered device, that the parameter resolves to a positive number equal to the
device's total `m`, within a relative 1e-9. Both facts are already resolved by
the hierarchy (`environment["mult"]` and the `m` of every level on the
path). A device that fails is refused with reason `multiplicity` and the
exact value to write ("write `mult=100` on `x1`; its total m is 100").

- **Exact to first order, at no cost**, with device names and probes intact.
- **Relies on the PDK's formulas**, which is a declaration. For Sky130 the
  evidence above supports it. For any other PDK it is what the profile author
  asserts, and the result is labelled `declared`.
- **Verifiable**: the one condition it relies on, equality of the parameter
  and the total `m`, is checked per device, so a deck that omits `mult` is
  refused, not sampled ten times too wide.
- **A deck change for gmid-workbench**: its generator writes `mult=` with
  the same value as `m`. Its decks already need a native variant, because
  they `.include corners/tt.spice` and set `mc_mm_switch` themselves, which
  the native contract refuses in favour of one `.lib` line and owned
  switches.

### 6.5 Decision

- `multiplicity.m = "formula"` (Option B) is built first, for profiles that
  name a multiplicity parameter. Sky130 is one, and B is what gmid-workbench
  needs. Under it, `nf = "pass"` is safe for Sky130.
- `multiplicity.m = "expand"` (Option A) is reserved in the schema and built
  when a PDK whose expressions have no multiplicity parameter needs it. For
  such a PDK it is the only correct option. When the profile also names a
  parameter, the parameter must then be 1 on each covered device.
- `multiplicity.m = "refuse"` stays the default and is the packaged profile's
  setting, so nothing changes for existing users.

A packaged, recorded Sky130 profile with formula multiplicity and the PMOS
device would give gmid-workbench sign-off a `recorded` label instead of
`declared`. §12 asks whether to add one.

---

## 7. Temperature corners on ngspice

### 7.1 The problem

ngspice ignores `.step`, so a temperature corner is a separate case: either
a separate circuit file or an `assign` variation over a parameter that the
deck's `.temp` reads. The seed key includes the input digest and the
assignments, so case 85 °C of sample 7 derives a different seed from case
27 °C of sample 7. One sample index therefore draws a different die at each
temperature.

That is the wrong population for sign-off. A die is one chip, and the yield
across temperature is the probability that one die passes at every
temperature. Independent dice per temperature estimate each temperature's
marginal distribution correctly, but not the joint one, and per-die
quantities such as offset drift between −40 °C and 125 °C have no meaning.

### 7.2 Measurements

The same bias and geometry as §6.2, with section `mc` and `.param
mc_mm_switch=1` (`combined`), two matched devices, and `vp p 0
{lvn_threshold}` to expose one global process draw:

- With seed 11, `lvn_threshold` is 0.1619226 at 27 °C and at 85 °C. The 85 °C
  currents are identical whether the temperature is written `.temp 85`,
  `.param tc=85` with `.temp {tc}`, or `.param tc=85` with `.option
  temp={tc}`.
- A run at 27 °C followed in the same process by `option temp=85` and `run`
  gives exactly the currents of a fresh load at 85 °C with the same seed.
- Over 24 seeds, the correlation of the pair's offset, ln(I1/I2), at 27 °C
  and at 85 °C is 0.9998 with the same seed, and 0.29 with different seeds
  (consistent with zero for 24 pairs).

On ngspice, then, the die is a function of the seed and the sequence of
random calls. Temperature does not enter that sequence.

### 7.3 Proposal: hold the die across declared condition parameters

The `pdk_native` variation gains `hold_die_across`, a list of parameter
names:

```json
"variations": [
  {"kind": "pdk_native", "id": "dice", "profile": "sky130-local", "mode": "combined",
   "runs": 200, "seed": 1947, "hold_die_across": ["tc"]},
  {"kind": "assign", "assign": {"tc": [-40, 27, 125]}}
]
```

with `.param tc=27` and `.temp {tc}` in the deck. (`tc` rather than `temp`:
the linter's `temp-as-param` rule blocks a parameter named `temp`, because
such a parameter is not the simulation temperature.)

- Each name must be an `assign` target of the same request that resolves to
  a root `.param` in a bench file. A component, an instance parameter or a
  PDK parameter is refused at request time (`pdk_native_request`).
- In each case's final hierarchy, the held parameter, directly or through
  other parameters, may not be read by any card in a PDK file, by any
  expression that contains a random call, by any instance multiplier `m`, or
  by the `.lib` line. Those are the ways a value could change which random
  calls are evaluated, or in what order. A violation fails the case with
  reason `hold_die`. Element values, sources and `.temp` / `.option temp=`
  may read it freely.
- The seed key leaves out the held assignments. Original captures are
  pre-assignment, so the input digest is already the same for every
  temperature, and cases that differ only in held assignments derive the same
  seed for the same sample index. `validate_seed_collisions` treats them as
  one identity (same root seed and key), so they are not collisions.
- The record lists the names under `validated.held_assignments`. The case's
  own assignments still carry the values.
- A temperature corner written as separate circuit files cannot be held,
  because the circuit id and input digest differ. The guide should recommend
  the parameter form.

The rule is not specific to temperature: a supply voltage or a load can be
held the same way, under the same conditions.

### 7.4 Alternative considered: several temperatures in one process

The measurement in §7.2 shows that one ngspice process can draw once and run
each temperature with `option temp=` and `run`, writing one raw file per
temperature. That holds the die by construction and pays the library load
(about 7 s here) once instead of once per temperature. It breaks the rule of
one case per process, on which the runner, cancellation, timeouts, recovery
and receipts all rely, and a failure at one temperature would take the
others with it. It is worth revisiting if load time dominates corner runs;
`hold_die_across` gets the right population without it.

### 7.5 Why not leave it as it is

Independent dice per temperature are not wrong for per-temperature
statistics. The current behaviour can stay the default, since without
`hold_die_across` nothing changes. What it cannot answer is the question
sign-off asks, so the option should exist.

---

## 8. `inspect(kind="pdk_library")`

### 8.1 Purpose and place

Drafting a profile needs three things from a PDK library: its sections, the
switch parameters that turn statistics on, and which models draw random
values and from what. Today an agent can get `.lib` section names only by
reading the file. `model` searches definitions by name, and `hierarchy`
expands a deck, not a library. The new kind reads a library and one of its
sections the way the hierarchy reader does (ngspice `hsa`, no simulator
installed, bounded by the same limits), and reports the facts a profile
needs.

### 8.2 Query

```text
{kind: "pdk_library", path, section?, entry?, view?: "summary"|"random"|"files"|"check",
 draft?, prefix?, cursor?}
```

- `path` (required): the library file, such as
  `…/sky130A/libs.tech/ngspice/sky130.lib.spice`. The sandbox applies.
- `entry`: the library's path relative to the PDK root, as a profile's
  `entry`. It sets the root that file paths are relative to. Without it, the
  root is the deepest directory that holds every file read, and the reply
  says which.
- Without `section`: `sections`, one row per `.lib` … `.endl` block, as
  `{name, line_start, line_end, params, includes}`, where `params` holds the
  `.param` assignments written in the section itself as text
  (`{"mc_mm_switch": "1", "mc_pr_switch": "0"}`) and `includes` counts its
  include and `.lib` lines. Paged.
- With `section`, `view: "summary"` (the default):
  `{section, file_count, model_digest, switches, random}`.
  - `switches`: root parameters set to 0 or 1 in the section's closure and
    read by at least one expression that contains a random call. They are
    found by what they do, not by their name, so no `_switch` suffix is
    assumed. Each row is
    `{name, value, set_at: {file, line}, gates: {global, per_instance}}`, the
    last two counting the random terms that read it. For Sky130 `tt_mm`:
    `mc_mm_switch` 1, `mc_pr_switch` 0.
  - `random`: counts per class (§5.2) of terms, models and subcircuits, each
    split into live, dead and undetermined.
- `view: "random"`: the random terms, paged, with `prefix` filtering by
  subcircuit or parameter name. Global rows are `{class: "global", param,
  file, line, functions, reads, reach}`. Per-instance rows are grouped by
  subcircuit: `{class: "per_instance", subckt, file, line, bins,
  parameters, functions, reads, instance_parameters, reach}`, where
  `parameters` are the model parameters with random terms (`toxe`, `vth0`,
  `voff`), `reads` the root parameters the terms read (slopes, switches), and
  `instance_parameters` the subcircuit's own parameters the terms read
  (`l`, `w`, `mult`). `instance_parameters` is what tells a profile author
  whether the PDK's formula has a multiplicity parameter. `reach` is
  `live`, `dead` or `undetermined`, evaluated with the section's values and
  the subcircuit's default parameters.
- `view: "files"`: the closure's files as `{path, sha256, size}`, paged. That
  is the manifest format, so saving the pages gives a manifest of the local
  installation.
- `view: "check"` with `draft` (an object in the profile schema, the same
  fields as the TOML file) and no `section`, since the draft names the
  sections: for each mode the draft declares,
  `{section_found, deck_params, reach: {global, per_instance}, devices,
  verdict, problems}`. `deck_params` says whether each name is defined in the
  section, `devices` whether each declared device is defined in the closure,
  is reached as the mode requires, and has terms that read the declared
  multiplicity parameter, and `verdict` is `consistent`, `inconsistent` or
  `undetermined`. Devices are evaluated at their default parameters. Nothing
  is declared by this view; the user puts the file in the configuration.

### 8.3 Reply conventions

- `reach` and `verdict` are strings, so a negative answer is never a `false`
  that the reply rule would leave out. The shared rule (`reply_payload`)
  leaves out empty lists and objects; counts are numbers and stay, zero
  included. The kind needs no `ALWAYS_PRESENT` entry.
- Cursors bind the query and the captured content of every file read, as the
  `hierarchy` kind's do, so an edit invalidates them (`invalid_cursor`).
- The walk is offloaded with `asyncio.to_thread`, as the other file-backed
  kinds are. Sky130's `tt` closure is 169 files, 6.9 MB.
- The kind joins `QUERY_MODELS` in `inspect_tools.py`, the reference index
  (`inspect(kind="reference")`), the `inspect` contract in
  `mcp_surface.md` §3.6 and Appendix A.4, and is reachable through
  `Api.inspect` like every kind.

### 8.4 How an agent uses it

1. `pdk_library` without `section` lists the sections.
2. `summary` on the candidate sections shows which switches each sets.
3. `random` with a device `prefix` shows which subcircuits draw, from what,
   and whether `mult` is among the instance parameters.
4. `check` with a draft confirms the mode table and devices.
5. The agent gives the user the TOML file and the config line to add. The
   user's configuration is the declaration.

---

## 9. Contract, guide and compatibility

- **Contract.** "Native PDK experiment contract" in `mcp_surface.md` is
  rewritten around profiles: the packaged profile's table stays as its
  example, and the new failure reasons (`closure`, `deck_params`,
  `random_reach`, `multiplicity`, `hold_die`) and record fields are added.
  Its §3.1 and Appendix A.1 document `hold_die_across`, its §3.6 and
  Appendix A.4 the new `inspect` kind, and its `capabilities` field list
  gains `pdk_profiles`.
- **Guide.** The 'sky130' section becomes a section on native PDK profiles,
  with Sky130 as its worked example, the declared-profile file, the agent
  workflow of §8.4, the line-ending note of §5.1 and the temperature form of
  §7.3. Pointers are checked by `tests/test_guide_delivery.py`.
- **Python API.** `Api(pdk_profiles=...)` joins the constructor overrides in
  `python_api.md` §3.
- **Compatibility.** Existing requests, seeds and records are unchanged for
  the packaged profile. Replies gain fields: `trust`, the new `validated`
  facts and the capabilities field. `tools/list` changes once, through the
  `profile` description, the `hold_die_across` field and the new `inspect`
  kind, and never with configuration. Job records written before the change
  still load (§4.2).

---

## 10. Out of scope

- **PDKs with only Spectre or HSPICE views.** ngspice cannot evaluate their
  statistical blocks (Spectre `statistics { process { … } mismatch { … } }`,
  HSPICE `.variation`), and the native launch is ngspice's. They stay on the
  generic `random` variation with a `mismatch` rule, which applies the
  Pelgrom law and counts `m` (guide section 'variations', "Pelgrom
  mismatch"). A vendor-converted ngspice view is in scope as a declared
  profile.
- **LTspice.** Its statistical functions and `.step` semantics differ, and
  the per-sample launch policy is ngspice's. `simulator = "ltspice"` is
  refused.
- **Corner sweeps that are not statistical** (`ss`, `ff` as plain sections)
  remain deck settings or model-swap variations. A profile can still map
  `nominal` to any section.
- **Non-MOS coverage**: see §5.3 and §12.

---

## 11. Test plan

Following `docs/TESTING.md`: tests go through the real code paths and assert
on real values. A regression test fails before its fix. Waits read the value
they assert on.

### 11.1 Offline, no simulator

A small fixture PDK written for these tests (not copied from a real PDK):
`tests/fixtures/pdk_fixture/`, a few kilobytes, with a library
`libs.tech/ngspice/fixture.lib.spice` and sections `tt`, `tt_mm`, `mc` and
deliberately broken variants. It has two wrapper subcircuits (`fx_nfet`,
`fx_pfet`), each with two BSIM4 (`level=54`) bins whose cards carry
`MC_MM_SWITCH*AGAUSS(0,1,1)*(slope/sqrt(l*w*mult))` terms, and a
`critical`-style global parameter gated by `MC_PR_SWITCH`. The variants: a
section whose switch is misspelt, a device whose random terms omit `mult`, a
term with `agauss(0,0,1)`, terms using `unif` and `limit`, and a term with a
function the evaluator does not model.

- **Profile files**: every schema field valid and invalid, with the field
  path in the diagnostic; unknown keys; a name equal to the packaged
  profile's or duplicated; `entry` with `..`, a drive letter, a backslash or
  a leading `/`; a reserved `expand`; size caps; a manifest with bad paths or
  digests.
- **Configuration**: the TOML list, a bare string refused as for
  `symbol_paths`, `LTSPICE_MCP_PDK_PROFILES` split on `os.pathsep`,
  `Api(pdk_profiles=...)`, the default config's commented example, a re-read
  after a config change, and a startup diagnostic for a broken file.
- **Packaged profile as data**: every existing test in
  `tests/test_pdk_native.py` and `tests/test_native_inputs.py` passes
  unchanged, and golden keys and seeds recorded from today's code before the
  refactor stay byte-identical.
- **Seed key**: a declared profile's key includes `profile_digest`; the
  digest ignores `description` and `identity` and changes with `modes`,
  `devices`, `multiplicity`, `scale` and the manifest.
- **Random reach** on the fixture: the mode × section matrix gives the §5.2
  table; the misspelt switch is refused with reason `random_reach`; the
  zero-variance term is dead; the unmodelled function is undetermined,
  recorded and not refused; a device with no random terms is refused in
  `mismatch` mode and named.
- **Coverage**: an undeclared PDK subcircuit, a bench-defined MOS model and
  an M with no model are each refused and named; a declared device inside a
  user block is covered; a wrapper with two leaves lists both.
- **Multiplicity, formula**: `m=100 mult=100` accepted; `m=100` alone
  refused with the value to write; nested `m` (a block with `m=2` holding a
  device with `m=50 mult=100`) accepted; `mult` not equal to the total `m`
  refused; `nf` under `pass` and `refuse`.
- **Hashes**: `pdk_files` equals the files under the root that the hierarchy
  read, with original-byte hashes; a manifest mismatch is refused
  (`pdk_pin`); `file_pins` takes each of its three values; a CRLF copy of a
  fixture file has a different hash; a file included from outside the root is
  refused (`closure`).
- **Records**: the new fields survive `to_record` and `from_record`; a job
  record written before them, kept as a fixture, still loads with `trust:
  "recorded"` and the missing facts listed under `unavailable`.
- **Labels and capabilities**: `trust` in the compact and full forms; the
  `pdk_profiles` capability field; `CapabilityField` equal to the report;
  `table_report()` holds packaged profiles with evidence that
  `tests/test_capabilities.py` resolves.
- **`inspect(kind="pdk_library")`**: sections, switches, random rows grouped
  by subcircuit with `instance_parameters`, the files view, paging and cursor
  binding, a sandbox refusal, and `check` on good and bad drafts.
- **`hold_die_across`**: held assignments are left out of the key; a held
  parameter read by a random expression, an `m` or a PDK file is refused; a
  non-`.param` target is refused at request time; the collision check
  accepts same-key cases.
- **Fixed tool listing**: `tools/list` bytes do not depend on configured
  profiles.

### 11.2 Live, ngspice and the fixture PDK

ngspice is installed on the Linux CI runner, so these run in the gate there,
and in the native Windows release run where ngspice is on PATH.

- The four modes through `run_experiments` with a declared fixture profile:
  `nominal` samples identical; `mismatch`, `process` and `combined` samples
  different; sample *k* alone equal to sample *k* in the batch.
- Multiplicity as values, through `run_experiments` with a formula profile:
  60 samples each of one finger, `m=100 mult=100`, and 100 separate
  instances. The spread of `m=100 mult=100` lies between 0.7 and 1.4 times
  that of the 100 instances, and the one finger's spread is between 6 and 16
  times it; `m=100` alone is refused before launch. Exact values hold only
  for one ngspice build, so the assertions use bands, not equality.
- `hold_die_across` with `tc` in {27, 85}: a probe of a global draw
  (`vp p 0 {param}`) is equal across temperatures for each sample, and
  differs without the option.
- A deck using the misspelt-switch section fails with reason `random_reach`,
  and the job's `submitted` count is 0.

### 11.3 Real PDK, opt-in

These need `LTSPICE_MCP_TEST_PDK_ROOT` pointing at the pinned Sky130A
installation and ngspice, as the existing integration tests do.

- The existing packaged-profile tests, unchanged.
- Random reach on Sky130's four sections gives the §5.2 table exactly
  (`tt` none, `tt_mm` per-instance only, `mc` global only, `mc` with
  `mc_mm_switch=1` both), for `nfet_01v8` and `pfet_01v8`.
- A declared Sky130 profile (nfet and pfet, formula, `nf` pass), kept as a
  test fixture TOML: a PMOS-loaded five-transistor OTA in gmid-workbench's
  device form with `mult` added, through all four modes; coverage lists both
  device types; the record carries `trust: "declared"`; the spread of a
  `m=100` device matches the §6.2 table within the band.
- `inspect(kind="pdk_library")` on `sky130.lib.spice`: the sections include
  `tt`, `tt_mm` and `mc`; `mc_mm_switch` and `mc_pr_switch` are switches;
  `sky130_fd_pr__nfet_01v8` in `tt` shows 180 bins, `toxe`, `vth0`, `voff`,
  instance parameters `l`, `w`, `mult`, dead in `tt` and live in `tt_mm`; and
  `check` on the packaged profile's data is consistent.

What needs a real PDK and is not planned for the gate: GF180MCU and IHP
SG13G2. Neither has been examined, so no profile for either is proposed
here.

### 11.4 Windows

- Profile and manifest paths with drive letters and spaces, and an `entry`
  compared with the inferred root as the filesystem compares names (without
  case on Windows).
- The TOML and JSON files are read as bytes. A PDK installed with CRLF line
  endings fails a manifest made from LF files; the guide says so.

---

## 12. Open questions for the maintainer

1. **A recorded Sky130 profile with PMOS and multiplicity.** Should the
   project ship `sky130-e6f9c887-ngspice-v2` (`nfet_01v8` and `pfet_01v8`,
   formula multiplicity, `nf` pass), recorded by extending the opt-in live
   test? gmid-workbench sign-off would then be `recorded` rather than
   `declared`.
2. **Filling `mult`.** Under `formula`, should the server write `mult=<total
   m>` into the staged copy when a device has no `mult`, instead of refusing?
   It would accept gmid-workbench decks unchanged, but it edits the deck, and
   a block instantiated with different `m` needs a clone.
3. **Undetermined reach.** Run with a warning (proposed), or refuse for
   declared profiles?
4. **A backstop after the run.** Add a job observation when two samples of a
   statistical mode produce identical data? It catches whatever the static
   check calls undetermined.
5. **Option A.** Build `expand` now, or wait for a PDK without a
   multiplicity parameter?
6. **`ngbehavior` per profile.** Keep `hsa` for every profile (proposed), or
   let a profile choose another mode the hierarchy reader models?
7. **Non-MOS devices.** Should a profile be able to declare resistors,
   capacitors and bipolars, so that coverage lists them and reach checks
   them?
8. **The label "recorded".** It overlaps the `recorded:` evidence kind of
   the capability table. Keep both, or rename one?
9. **Include-form decks.** gmid-workbench includes `corners/tt.spice`
   directly and sets the switches itself. Should a profile allow that form,
   or stay with one `.lib` line (proposed), whose section is the unit the
   reach check and the record name?
10. **Manifests.** Is the `files` view enough to make a manifest, or should
    a script under `scripts/` write one in the exact packaged format?
11. **Holding other conditions.** Is `hold_die_across` for any qualifying
    bench parameter (proposed) acceptable, or should it be limited to
    temperature?
12. **Stages.** Proposed order: (1) the packaged profile as data, no
    behaviour change; (2) configuration, loading, trust labels and records;
    (3) the three checks; (4) formula multiplicity; (5) `pdk_library`;
    (6) `hold_die_across`; later, `expand` and a recorded v2 profile.

---

## Appendix: measurement decks

The measurements in §6.2 and §7.2 came from decks of this form, each run by a
driver that does what the native driver does (`set ngbehavior=hsa`, `set
ng_nomodcheck`, `setseed <n>`, `source`, `run`), repeated per seed:

```spice
* multiplicity, section tt_mm
.lib "<sky130A>/libs.tech/ngspice/sky130.lib.spice" tt_mm
vg g 0 0.6
vdE dE 0 0.9
xe dE g 0 0 sky130_fd_pr__nfet_01v8 L=0.5 W=1 nf=1
vdA dA 0 0.9
xa dA g 0 0 sky130_fd_pr__nfet_01v8 L=0.5 W=1 nf=1 m=100
vdB dB 0 0.9
xb dB g 0 0 sky130_fd_pr__nfet_01v8 L=0.5 W=1 nf=1 m=100 mult=100
* ... 100 lines xc0 .. xc99, each "dC g 0 0 sky130_fd_pr__nfet_01v8 L=0.5 W=1 nf=1"
* ... xd W=99, xf W=99 nf=99, xg W=1 mult=100
.op
.end
```

```spice
* one die at two temperatures, combined mode
.lib "<sky130A>/libs.tech/ngspice/sky130.lib.spice" mc
.param mc_mm_switch=1
.param tc=85
.temp {tc}
vg g 0 0.6
vd1 d1 0 0.9
vd2 d2 0 0.9
x1 d1 g 0 0 sky130_fd_pr__nfet_01v8 L=0.5 W=1 nf=1
x2 d2 g 0 0 sky130_fd_pr__nfet_01v8 L=0.5 W=1 nf=1
vp p 0 {lvn_threshold}
rp p 0 1k
.op
.end
```
