"""Records what LTspice's own MCP bridge does, for the tests that model it.

``lib/ltspice_bridge.py`` and ``lib/ltspice_window.py`` encode how a running
LTspice answers through ``ltspice-mcp-bridge.exe``: what a window hands back
for a sheet it has open, that replacing it leaves the file alone, that it
never reads the file again, that a run in the window is of the window's copy,
and that a bridge told where LTspice is not cannot start one. Each of those is recorded here from an installed build, under
``tests/fixtures/ltspice_bridge_recorded/<build>/``:

- ``sheets/<name>.asc``: the window's copy of ``inputs/<name>.asc``, as UTF-8;
- ``conversation.json``: every call made and what came back, in order;
- ``manifest.json``: the build, and the digest of each input and recording.

It is a recorder of its own because it needs a window: the main recorder
(``tests/ltspice_recorder.py``) runs a build once per input and reads the
files it leaves. This one starts LTspice with its window on a desktop of its
own, where it cannot take the keyboard focus, and talks to it with the
server's own client, whose bridge runs on the server's desktop: another one,
so every recording is also of a bridge reaching a window on a desktop that is
not its own, as it does for a person's.

LTspice XVII has no bridge, so only builds from 26.1 on are recorded.

    uv run python scripts/record_ltspice_bridge.py
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import shutil
import tempfile
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ltspice_mcp.lib.encoding import decode_spice_bytes
from ltspice_mcp.lib.guide import split_front_matter
from ltspice_mcp.lib.hidden_desktop import HiddenDesktop
from ltspice_mcp.lib.ltspice_bridge import BridgeError, BridgeSession, bridge_command
from tests.ltspice_recorder import (
    Build,
    RecorderError,
    assert_private,
    discover_builds,
    neutral_settings,
    private_strings,
    unavailable_reason,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "ltspice_bridge_recorded"
INPUTS = FIXTURES / "inputs"
MANIFEST = "manifest.json"
CONVERSATION = "conversation.json"
MANIFEST_SCHEMA = 1

NEUTRAL_DIR = "C:\\recording"
NEUTRAL_PID = 1000
NEUTRAL_PORT = 50000
NOT_A_PROCESS = 999999
#: The sheet the write and read-back steps are made on.
EDITED = "older_version"
_STARTED_S = 60.0


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def input_names() -> list[str]:
    return sorted(path.stem for path in INPUTS.glob("*.asc"))


def recorded_builds(root: Path = FIXTURES) -> list[str]:
    return sorted(path.parent.name for path in root.glob(f"*/{MANIFEST}"))


def load_manifest(directory: Path) -> dict[str, Any]:
    return json.loads((directory / MANIFEST).read_text(encoding="utf-8"))


def load_conversation(directory: Path) -> list[dict[str, Any]]:
    return json.loads((directory / CONVERSATION).read_text(encoding="utf-8"))


class _Scrub:
    """Replaces what differs between two recordings of one build: the run
    directory, the process id and the port."""

    def __init__(self, work: Path) -> None:
        spelled = re.escape(str(work)).replace(r"\\", r"[\\/]")
        self._work = re.compile(spelled, re.IGNORECASE)
        self.pid: int | None = None
        self.port: int | None = None

    def text(self, value: str) -> str:
        return self._work.sub(lambda _match: NEUTRAL_DIR, value)

    def __call__(self, value: Any, key: str = "") -> Any:
        if isinstance(value, dict):
            return {name: self(item, name) for name, item in value.items()}
        if isinstance(value, list):
            return [self(item, key) for item in value]
        if isinstance(value, str):
            return self.text(value)
        if isinstance(value, int) and not isinstance(value, bool):
            if key.lower().endswith("pid") and value in (self.pid,):
                return NEUTRAL_PID
            if key == "port" and value == self.port:
                return NEUTRAL_PORT
        return value


class _Recording:
    """The conversation so far: each call as it was made and what it returned."""

    def __init__(self, scrub: _Scrub) -> None:
        self.steps: list[dict[str, Any]] = []
        self._scrub = scrub

    def call(
        self, session: BridgeSession, note: str, name: str, **arguments: Any
    ) -> dict[str, Any]:
        """Make one call and record it. A refusal is recorded and returned as ``{}``."""
        step: dict[str, Any] = {"note": note, "call": name, "arguments": self._scrub(arguments)}
        try:
            reply = session.call(name, **arguments)
        except BridgeError as error:
            step["error"] = self._scrub.text(str(error))
            self.steps.append(step)
            return {}
        kept = dict(reply)
        if isinstance(kept.get("text"), str):
            # The sheet itself is recorded beside this file, once.
            kept["text"] = f"<{len(kept['text'].splitlines())} lines>"
        if name == "get_raw_info":
            # The server opens a results file with this call and reads nothing
            # of what it describes, which is the main recorder's subject.
            kept = {"path": kept.get("path")}
        step["reply"] = self._scrub(kept)
        self.steps.append(step)
        return reply

    def fact(self, note: str, value: Any) -> None:
        self.steps.append({"note": note, "observed": value})


def wait_for_window(command: Sequence[str], pid: int, sheet: Path) -> tuple[int, int]:
    """The port of the window ``pid`` once it has ``sheet`` open.

    LTspice offers a window to the bridge before it has finished opening the
    sheet it was started with, and says so nowhere but to the bridge, so this
    asks until the sheet is listed.
    """
    deadline = time.monotonic() + _STARTED_S
    while time.monotonic() < deadline:
        with contextlib.suppress(BridgeError), BridgeSession(command) as session:
            for row in session.call("status").get("instances", []):
                if row.get("pid") != pid or row.get("mode") != "gui":
                    continue
                session.attach(pid)
                if any(Path(spelled) == sheet for spelled in session.open_designs()):
                    return pid, int(row["port"])
        time.sleep(0.25)  # timing: between two looks; what is waited for is the listing
    raise RecorderError(f"LTspice process {pid} never offered its window to the bridge")


def _run_in_window(session: BridgeSession, recording: _Recording, sheet: Path) -> Path:
    """Run ``sheet`` in the window, which is how a results file comes to be."""
    recording.call(session, "run the sheet in the window", "start_simulation", path=str(sheet))
    deadline = time.monotonic() + _STARTED_S
    while time.monotonic() < deadline:
        running = session.call("is_simulation_running", path=str(sheet))
        if str(running.get("result")).lower() == "false":
            break
        time.sleep(0.25)  # timing: between two looks; what is waited for is the run's end
    results = sheet.with_suffix(".raw")
    deadline = time.monotonic() + _STARTED_S
    while not results.is_file() and time.monotonic() < deadline:
        time.sleep(0.25)  # timing: between two looks; what is waited for is the file
    if not results.is_file():
        raise RecorderError(f"running {sheet.name} in the window left no results file")
    return results


def _resistor_value(netlist: Path) -> str | None:
    """The value of R1 in the netlist a run in the window wrote beside its sheet.

    When the run is made the window holds ``2k`` and the file ``3k``, so this
    says which of the two LTspice simulated.
    """
    if not netlist.is_file():
        return None
    for line in decode_spice_bytes(netlist.read_bytes()).splitlines():
        words = line.split()
        if words and words[0] == "R1":
            return words[-1]
    return None


def _record(build: Build, out: Path, desktop: HiddenDesktop) -> None:
    """The recording itself, with LTspice's window on ``desktop``."""
    command = bridge_command(build.exe)
    if command is None:
        raise RecorderError(f"{build.exe} has no bridge beside it")
    settings = build.settings_file
    assert settings is not None
    with tempfile.TemporaryDirectory(prefix="ltspice-bridge-rec-") as scratch:
        work = Path(scratch).resolve() / "sheets"
        work.mkdir()
        for name in input_names():
            shutil.copyfile(INPUTS / f"{name}.asc", work / f"{name}.asc")
        ini = work.parent / settings.name
        ini.write_bytes(neutral_settings(settings.read_bytes(), {}))
        scrub = _Scrub(work)
        recording = _Recording(scrub)
        sheets: dict[str, bytes] = {}
        names = input_names()
        first = work / f"{names[0]}.asc"

        with BridgeSession(command) as session:
            recording.call(session, "no LTspice is running", "status")
            recording.call(
                session, "attach to a process that is not LTspice", "attach", pid=NOT_A_PROCESS
            )
            recording.call(
                session, "a request that needs LTspice, with none running", "list_open_designs"
            )

        window = desktop.start([str(build.exe), str(first), "-ini", str(ini)])
        try:
            scrub.pid, scrub.port = wait_for_window(command, window.pid, first)
            with BridgeSession(command) as session:
                recording.call(session, "one window is open", "status")
                recording.call(session, "attach to the window", "attach", pid=window.pid)
                recording.call(session, "attached", "status")
                for name in names[1:]:
                    recording.call(
                        session,
                        "open another sheet",
                        "open_design",
                        path=str(work / f"{name}.asc"),
                    )
                recording.call(session, "every sheet is open", "list_open_designs")
                recording.call(session, "the document in front", "get_active_design_path")
                recording.call(
                    session, "open a sheet that is already open", "open_design", path=str(first)
                )
                recording.call(session, "put it in front", "bring_to_front", path=str(first))
                recording.call(session, "the document in front now", "get_active_design_path")
                recording.call(
                    session,
                    "open a sheet that is not there",
                    "open_design",
                    path=str(work / "absent.asc"),
                )
                for name in names:
                    reply = recording.call(
                        session,
                        "the window's copy",
                        "get_design_content",
                        path=str(work / f"{name}.asc"),
                    )
                    sheets[f"sheets/{name}.asc"] = str(reply.get("text", "")).encode("utf-8")

                edited = work / f"{EDITED}.asc"
                on_disk = edited.read_bytes()
                held = sheets[f"sheets/{EDITED}.asc"].decode("utf-8")
                recording.call(
                    session,
                    "replace a sheet with the text it already holds",
                    "set_design_content",
                    path=str(edited),
                    text=held,
                )
                changed = held.replace("SYMATTR Value 1k", "SYMATTR Value 2k")
                recording.call(
                    session,
                    "replace a sheet with a changed one",
                    "set_design_content",
                    path=str(edited),
                    text=changed,
                )
                recording.fact(
                    "the window then reads back exactly what it was given",
                    session.design_text(str(edited)) == changed,
                )
                recording.fact("the file is as it was", edited.read_bytes() == on_disk)
                rewritten = on_disk.replace(b"SYMATTR Value 1k", b"SYMATTR Value 3k")
                edited.write_bytes(rewritten)
                recording.fact(
                    "after the file is rewritten the window still holds its own copy",
                    session.design_text(str(edited)) == changed,
                )
                recording.call(
                    session,
                    "a file no window has open",
                    "get_design_content",
                    path=str(work / "not_open.asc"),
                )
                recording.call(
                    session,
                    "replace a file no window has open",
                    "set_design_content",
                    path=str(work / "not_open.asc"),
                    text=held,
                )

                results = _run_in_window(session, recording, edited)
                recording.fact(
                    "a run in the window is of the window's copy and not of the file",
                    _resistor_value(edited.with_suffix(".net")),
                )
                recording.fact("the run did not write the sheet", edited.read_bytes() == rewritten)
                recording.fact(
                    "what the run left beside the sheet",
                    sorted(path.name[len(EDITED) :] for path in work.glob(f"{EDITED}.*")),
                )
                recording.call(session, "open a results file", "get_raw_info", path=str(results))
                recording.call(session, "put it in front", "bring_to_front", path=str(results))
                recording.fact(
                    "the results file is then the one LTspice has in front",
                    session.call("get_raw_info").get("path") == str(results),
                )
                recording.call(
                    session,
                    "a results file that is not there",
                    "get_raw_info",
                    path=str(work / "absent.raw"),
                )

                window.kill()
                window.wait(timeout=30)
                recording.call(session, "the window has closed", "list_open_designs")
                recording.call(session, "the window has closed", "status")
                recording.fact(
                    "no LTspice was started in its place",
                    not any(
                        row.get("mode") for row in session.call("status").get("instances", [])
                    ),
                )
        finally:
            window.close()

    conversation = (
        json.dumps(recording.steps, indent=1, ensure_ascii=False).encode("utf-8") + b"\n"
    )
    files = {CONVERSATION: conversation, **sheets}
    assert_private(files, private_strings())
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "build": {
            "label": build.label,
            "file_version": build.file_version,
            "generation": build.generation,
        },
        "bridge": {
            "size": Path(command[0]).stat().st_size,
            "sha256": sha256_bytes(Path(command[0]).read_bytes()),
        },
        "inputs": {name: sha256_bytes((INPUTS / f"{name}.asc").read_bytes()) for name in names},
        "reference": reference_record(build),
        "files": {name: sha256_bytes(data) for name, data in sorted(files.items())},
    }
    if out.exists():
        shutil.rmtree(out)
    for name, data in files.items():
        target = out / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    (out / MANIFEST).write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")


def reference_record(build: Build) -> dict[str, Any] | None:
    """The reference documents ``build`` installs: their names and the keys of
    their front matter. The documents are the vendor's and are not recorded."""
    library = build.library_root
    directory = library.parent / "reference" if library is not None else None
    if directory is None or not directory.is_dir():
        return None
    keys: set[str] = set()
    names: list[str] = []
    for path in sorted(directory.iterdir(), key=lambda entry: entry.name.casefold()):
        if path.is_file():
            names.append(path.name)
            head = path.read_bytes().decode("utf-8-sig", errors="replace").replace("\r\n", "\n")
            keys |= set(split_front_matter(head)[0])
    return {"directory": directory.name, "files": names, "front_matter": sorted(keys)}


def bridge_builds() -> list[Build]:
    """The installed builds that ship a bridge."""
    return [build for build in discover_builds() if bridge_command(build.exe) is not None]


def record(build: Build, out: Path) -> None:
    """Record ``build`` into ``out``, its window on a desktop of its own."""
    reason = unavailable_reason(build)
    if reason:
        raise RecorderError(reason)
    with HiddenDesktop(f"ltspice-bridge-recorder-{os.getpid()}") as desktop:
        if not desktop.available:
            raise RecorderError(
                "Windows gave no desktop to record on; not starting LTspice on this one"
            )
        _record(build, out, desktop)


def main() -> int:
    builds = bridge_builds()
    if not builds:
        print("No installed LTspice has ltspice-mcp-bridge.exe beside it (26.1 or later).")
        return 1
    for build in builds:
        record(build, FIXTURES / build.label)
        print(f"recorded {build.label} ({build.file_version}) into {FIXTURES / build.label}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
