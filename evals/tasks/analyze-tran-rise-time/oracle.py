"""The edges recipe measures a 10%-90% transition."""

import json

from ltspice_eval.scripted import Call


def script(ctx):
    [reply] = yield Call(
        "mcp__spice__analyze_results",
        {
            "sources": [{"path": "tran_rc.raw"}],
            "recipes": [{"key": "e", "metric": "edges", "signal": "V(out)"}],
        },
    )
    [row] = reply.json["results"]["e"]["values"]
    answer = {"rise_time_s": row["value"]["transition_time"]}
    yield Call("write_file", {"path": "answer.json", "content": json.dumps(answer)})
