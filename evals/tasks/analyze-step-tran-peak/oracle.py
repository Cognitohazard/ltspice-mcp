"""signal_stats over every step; the server pairs each with its step values."""

import json

from ltspice_eval.scripted import Call


def script(ctx):
    [reply] = yield Call(
        "mcp__spice__analyze_results",
        {
            "sources": [{"path": "step_tran.raw"}],
            "recipes": [{"key": "s", "metric": "signal_stats", "signal": "V(out)"}],
            "all_steps": True,
        },
    )
    best = max(reply.json["results"]["s"]["values"], key=lambda row: row["value"]["max"])
    answer = {"r": best["step_values"]["r"], "peak_v": best["value"]["max"]}
    yield Call("write_file", {"path": "answer.json", "content": json.dumps(answer)})
