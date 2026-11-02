"""Ask for the trace; a failure naming it as missing means the answer is null."""

import json

from ltspice_eval.scripted import Call


def script(ctx):
    [reply] = yield Call(
        "mcp__spice__analyze_results",
        {
            "sources": [{"path": "ac_rc.raw"}],
            "recipes": [{"key": "g", "metric": "bode_point", "signal": "V(fb)", "at_hz": 1000}],
        },
    )
    data = reply.json
    rows = data.get("results", {}).get("g", {}).get("values", [])
    gain = rows[0]["value"].get("gain_db") if rows else None
    if gain is None and not any(
        "not found" in failure.get("message", "") for failure in data["failures"]
    ):
        raise AssertionError(f"unexpected reply: {data}")
    yield Call("write_file", {"path": "answer.json", "content": json.dumps({"gain_db": gain})})
