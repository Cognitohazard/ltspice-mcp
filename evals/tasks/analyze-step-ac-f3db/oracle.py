"""One analyze_results call reads every step's cutoff."""

import json

from ltspice_eval.scripted import Call


def script(ctx):
    [reply] = yield Call(
        "mcp__spice__analyze_results",
        {
            "sources": [{"path": "step_ac.raw"}],
            "recipes": [{"key": "f", "metric": "bode_filter", "signal": "V(out)"}],
            "all_steps": True,
        },
    )
    rows = sorted(reply.json["results"]["f"]["values"], key=lambda row: row["step_index"])
    answer = {"f3db_hz": [row["value"]["cutoff_high_hz"] for row in rows]}
    yield Call("write_file", {"path": "answer.json", "content": json.dumps(answer)})
