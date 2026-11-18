`run_experiments` and `jobs` replies now go through the same presenter as the
other tools: a small `budget` cuts run rows but always keeps at least one, the
reply's `cut` field names the rest, and a reply that cannot meet its budget
says so. A standing note about a job is sent in full once per session and by
its code after that; `jobs(action="status")` always sends every note in full.
