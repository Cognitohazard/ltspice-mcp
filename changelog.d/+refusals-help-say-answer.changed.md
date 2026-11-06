Refusals and help say where the answer is:
- a `run_code` NameError hint says each call starts from a fresh
  namespace, so a name an earlier call set is gone;
- an unknown top-level field this tool takes further down is named
  there first (`run_experiments` takes `path` as `circuits[].path`),
  before the tools that take it at their top level;
- a sandbox refusal names the directory a relative path is taken from;
- a `jobs` resume refused for a reused `resume_request_id`, a stale
  parent, a running parent or a case selection names the jobs and cases
  involved and what to do instead;
- `RawResult`'s docstring, `load_raw`'s and `api.reference()` name its
  members (`signals`, `steps`, `trace(name, step=k)` and the rest);
- the `variations` description and guide section 'variations' give the
  one-value `assign` list that holds a target fixed in every case;
- guide section 'fundamentals' says LTspice prints every AC `.meas`
  result in its log as dB and angle, a real one included, or under
  `.options meascplxfmt` as magnitude and angle or real and imaginary
  parts, and that `measurements` decodes all three;
- a read of one step of an LTspice 26 `.op.raw` from a stepped `.tran`
  or `.ac`, which holds one point and does not say whose step it is, is
  refused with that cause and the remedy (run `.op` under the same
  `.step`), where it said only that step boundaries were unresolved.
