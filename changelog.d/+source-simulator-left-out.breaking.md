A `source[]` entry of a `run_experiments` receipt and of the `jobs` replies no
longer carries `simulator`, which held the simulator's class name (`LTspice`,
`LTspiceWSL`, `NGspiceSimulator`), different per platform. Read the job-level
`simulator` instead, which names the simulator by the selector a run passes
back (`ltspice`, `ltspice:xvii`).
