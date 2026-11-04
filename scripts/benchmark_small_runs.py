"""Time a few small runs and compare them with a stored baseline.

Design principle "Fast enough to try" (docs/DESIGN.md): small runs stay fast
enough that an agent runs an experiment instead of reasoning one out, and work
added to the run path is weighed against that. The test suite times nothing
(CLAUDE.md, "Working practice"), so this script is where a slowdown shows. It
reports one and never fails: it exits 0 whatever it measures.

It times, each as the best of --repeat tries, through the Python API:

- booting the engine;
- a 5-case .op sweep on ngspice, when ngspice is installed;
- reading ten of the recorded LTspice raws and one trace of each;
- one edit_schematic batch that draws a divider, compared with a reference
  netlist when LTspice is installed to export the sheet.

A timing that cannot be taken here is reported as skipped, with the reason.

    uv run python scripts/benchmark_small_runs.py --record   # store the baseline
    uv run python scripts/benchmark_small_runs.py            # compare with it

The baseline belongs to the machine it was taken on, so it is kept out of the
repository (.benchmark_small_runs.json, ignored by git).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from tests._asc_ops import DIVIDER_OPS  # noqa: E402
from tests._schematic_fixtures import SUITE_LIBRARY_DIR  # noqa: E402

DEFAULT_BASELINE = REPO / ".benchmark_small_runs.json"
RECORDED_RAWS = REPO / "tests" / "fixtures" / "ltspice_recorded" / "ltspice26" / "raw"

_SWEEP_DECK = "* divider\nV1 in 0 5\nR1 in out 1k\nR2 out 0 2k\n.op\n.end\n"
#: LTspice's export of the sheet DIVIDER_OPS draws: R1 and R2 joined at N001.
_DIVIDER_REFERENCE = "R1 N001 vout R\nR2 N001 N002 R\n.end\n"


def _best_of(repeat: int, run: Callable[[int], None]) -> float:
    times = []
    for attempt in range(repeat):
        started = time.perf_counter()
        run(attempt)
        times.append(time.perf_counter() - started)
    return min(times)


def _available(api: Any) -> set[str]:
    report = api.inspect(queries=[{"kind": "capabilities"}])
    simulators = report["results"][0]["data"].get("simulators", {})
    return {name for name, row in simulators.items() if row.get("available")}


def measure(repeat: int) -> dict[str, float | str]:
    from ltspice_mcp.api import Api

    timings: dict[str, float | str] = {}
    with tempfile.TemporaryDirectory(prefix="benchmark-small-runs-") as scratch:
        work = Path(scratch)
        symbols = [str(SUITE_LIBRARY_DIR)]

        def boot(attempt: int) -> None:
            Api(work / f"boot{attempt}", symbol_paths=symbols).close()

        for attempt in range(repeat):
            (work / f"boot{attempt}").mkdir()
        timings["engine boot"] = _best_of(repeat, boot)

        with Api(work, symbol_paths=symbols) as api:
            available = _available(api)
            deck = work / "divider.cir"
            deck.write_text(_SWEEP_DECK, encoding="utf-8", newline="\n")

            def sweep(attempt: int) -> None:
                receipt = api.run_experiments(
                    request_id=f"sweep-{attempt}",
                    circuits=[{"path": str(deck)}],
                    variations=[
                        {"kind": "assign", "assign": {"R1": ["1k", "2k", "3k", "4k", "5k"]}}
                    ],
                    execution={"simulator": "ngspice", "wait_s": 120},
                )
                if receipt["status"] != "completed":
                    raise RuntimeError(f"the sweep ended {receipt['status']}: {receipt}")

            timings["ngspice 5-case .op sweep"] = (
                _best_of(repeat, sweep)
                if "ngspice" in available
                else "skipped: ngspice is not installed"
            )

            # Fresh copies for each try, made before timing, so the result cache
            # does not answer and only the reads are timed.
            raws = sorted(RECORDED_RAWS.glob("*.raw"))[:10]
            for attempt in range(repeat):
                folder = work / f"raws{attempt}"
                folder.mkdir()
                for path in raws:
                    for source in (path, path.with_suffix(".log")):
                        if source.exists():
                            shutil.copyfile(source, folder / source.name)

            def read(attempt: int) -> None:
                for path in raws:
                    raw = api.load_raw(work / f"raws{attempt}" / path.name)
                    raw.trace(raw.signals[-1])

            timings[f"read {len(raws)} recorded raws"] = _best_of(repeat, read)

            def edit(attempt: int, compare: bool) -> None:
                arguments: dict[str, Any] = {
                    "path": str(work / f"divider{attempt}{'-compared' if compare else ''}.asc"),
                    "base": "blank",
                    "ops": DIVIDER_OPS,
                }
                if compare:
                    arguments["compare"] = {"reference": _DIVIDER_REFERENCE}
                reply = api.edit_schematic(**arguments)
                if reply.get("commit_state") != "committed":
                    raise RuntimeError(f"the batch did not commit: {reply}")

            timings["edit_schematic batch"] = _best_of(repeat, lambda a: edit(a, False))
            timings["edit_schematic batch with compare"] = (
                _best_of(repeat, lambda a: edit(a, True))
                if "ltspice" in available
                else "skipped: LTspice is not installed"
            )
    return timings


def report(
    timings: dict[str, float | str], baseline: dict[str, float] | None, tolerance: float
) -> list[str]:
    """The table to print, and the slowdowns in it."""
    slower: list[str] = []
    print(f"{'run':40} {'now':>9} {'baseline':>9}")
    for name, now in timings.items():
        if isinstance(now, str):
            print(f"{name:40} {now}")
            continue
        before = (baseline or {}).get(name)
        if before is None:
            print(f"{name:40} {now:8.3f}s {'-':>9}")
            continue
        flag = ""
        # A tenth of a second is below what a run's noise allows saying.
        if now > before * (1 + tolerance) and now - before > 0.1:
            flag = f"  slower by {now / before - 1:.0%}"
            slower.append(name)
        print(f"{name:40} {now:8.3f}s {before:8.3f}s{flag}")
    return slower


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Time a few small runs and compare them with a stored baseline."
    )
    parser.add_argument(
        "--record", action="store_true", help="store these timings as the baseline"
    )
    parser.add_argument("--repeat", type=int, default=3, help="tries per run; the best counts")
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument(
        "--tolerance", type=float, default=0.25, help="slowdown reported above this fraction"
    )
    args = parser.parse_args(argv)

    timings = measure(max(1, args.repeat))
    if args.record:
        taken = {name: now for name, now in timings.items() if isinstance(now, float)}
        args.baseline.write_text(json.dumps(taken, indent=2) + "\n", encoding="utf-8")
        report(timings, None, args.tolerance)
        print(f"baseline written to {args.baseline}")
        return 0
    baseline = (
        json.loads(args.baseline.read_text(encoding="utf-8")) if args.baseline.exists() else None
    )
    slower = report(timings, baseline, args.tolerance)
    if baseline is None:
        print(f"no baseline at {args.baseline}: run with --record to store one")
    elif slower:
        print(f"slower than the baseline: {', '.join(slower)}")
    else:
        print("no run is slower than the baseline")
    return 0


if __name__ == "__main__":
    sys.exit(main())
