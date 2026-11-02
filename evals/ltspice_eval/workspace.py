"""One trial's directories, the file tools confined to its workspace, and the
sweep that stops whatever the trial's server left running."""

from __future__ import annotations

import hashlib
import shutil
import tempfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import psutil

from ltspice_eval.tasks import Task

#: What the server keeps in its working directory; never graded.
SERVER_FILES = frozenset({".ltspice-mcp", "ltspice-mcp.toml"})

#: How long the sweep waits for a killed process to go before reporting it.
SWEEP_WAIT_S = 10.0


@dataclass(frozen=True)
class TrialDirs:
    root: Path
    #: The agent's workspace: the server's working directory and sandbox.
    workspace: Path
    #: The server's private ``LTSPICE_MCP_HOME``, outside the workspace.
    home: Path


def make_trial_dirs(task: Task) -> TrialDirs:
    root = Path(tempfile.mkdtemp(prefix="ltspice-eval-")).resolve()
    workspace = root / "workspace"
    home = root / "home"
    if task.workspace.is_dir():
        shutil.copytree(task.workspace, workspace)
    else:
        workspace.mkdir()
    home.mkdir()
    return TrialDirs(root=root, workspace=workspace, home=home)


def remove_trial_dirs(dirs: TrialDirs) -> str | None:
    """Remove the trial's directories; the reason it could not, if it could not."""
    try:
        shutil.rmtree(dirs.root)
    except OSError as exc:
        return f"{type(exc).__name__}: {exc}"
    return None


def agent_files(workspace: Path) -> Iterator[Path]:
    """The workspace's files, less the server's own."""
    for path in sorted(workspace.rglob("*")):
        relative = path.relative_to(workspace)
        if relative.parts[0] in SERVER_FILES or not path.is_file():
            continue
        yield path


def snapshot(workspace: Path) -> dict[str, str]:
    """Each file's SHA-256, by its workspace-relative POSIX path."""
    return {
        path.relative_to(workspace).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in agent_files(workspace)
    }


def changed_inputs(
    before: dict[str, str], after: dict[str, str], editable: tuple[str, ...]
) -> list[str]:
    """Inputs the task marks read-only that the agent changed or removed."""
    return sorted(
        name
        for name, digest in before.items()
        if name not in editable and after.get(name) != digest
    )


def sweep(home: Path) -> int:
    """Kill every process that carries this trial's ``LTSPICE_MCP_HOME``.

    The MCP SDK stops the server's process group (a Job object on Windows),
    but the server starts some processes in a session of their own, such as
    the parser worker and a detached job's owner, and those outlive it. Each
    inherits the server's environment, so the trial's private home names
    them. Returns how many were still running.
    """
    marker = str(home)
    found: list[psutil.Process] = []
    for process in psutil.process_iter(["environ"], ad_value=None):
        environ = process.info.get("environ")
        if environ and environ.get("LTSPICE_MCP_HOME") == marker:
            found.append(process)
    for process in found:
        try:
            process.kill()
        except psutil.NoSuchProcess:
            continue
        except psutil.AccessDenied:
            pass
    psutil.wait_procs(found, timeout=SWEEP_WAIT_S)
    return len(found)


# ---------------------------------------------------------------------------
# File tools. The agent reads and writes its workspace through these; there is
# no shell, so the simulator is reached only through the server.

READ_LIMIT_LINES = 2000

FILE_TOOLS: list[dict[str, Any]] = [
    {
        "name": "read_file",
        "description": (
            "Read a text file in the workspace. Returns its lines numbered from 1. "
            "Reads up to 2000 lines from `offset`."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "A path relative to the workspace."},
                "offset": {"type": "integer", "minimum": 1, "description": "First line, from 1."},
                "limit": {"type": "integer", "minimum": 1, "description": "Lines to read."},
            },
            "required": ["path"],
            "additionalProperties": False,
        },
    },
    {
        "name": "write_file",
        "description": "Write a file in the workspace, replacing it if it exists.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "A path relative to the workspace."},
                "content": {"type": "string"},
            },
            "required": ["path", "content"],
            "additionalProperties": False,
        },
    },
    {
        "name": "edit_file",
        "description": (
            "Replace text in a workspace file. `old_string` must occur exactly once "
            "unless `replace_all` is true."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "A path relative to the workspace."},
                "old_string": {"type": "string"},
                "new_string": {"type": "string"},
                "replace_all": {"type": "boolean"},
            },
            "required": ["path", "old_string", "new_string"],
            "additionalProperties": False,
        },
    },
    {
        "name": "list_dir",
        "description": "List a workspace directory: names, with a trailing / on directories, and sizes.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Relative to the workspace; default '.'.",
                }
            },
            "additionalProperties": False,
        },
    },
]
FILE_TOOL_NAMES = frozenset(tool["name"] for tool in FILE_TOOLS)


class FileToolError(Exception):
    """A file tool call the tool refuses; its message is what the model reads."""


class FileTools:
    def __init__(self, workspace: Path) -> None:
        self.workspace = workspace.resolve()

    def call(self, name: str, arguments: dict[str, Any]) -> str:
        handler = {
            "read_file": self._read,
            "write_file": self._write,
            "edit_file": self._edit,
            "list_dir": self._list,
        }[name]
        try:
            return handler(**arguments)
        except TypeError as exc:
            raise FileToolError(f"invalid arguments: {exc}") from exc
        except OSError as exc:
            raise FileToolError(f"{type(exc).__name__}: {exc.strerror or exc}") from exc

    def _path(self, path: str) -> Path:
        candidate = (self.workspace / path).resolve()
        if candidate != self.workspace and self.workspace not in candidate.parents:
            raise FileToolError(f"{path}: outside the workspace")
        return candidate

    def _read(self, path: str, offset: int = 1, limit: int = READ_LIMIT_LINES) -> str:
        target = self._path(path)
        data = target.read_bytes()
        if b"\x00" in data[:8192]:
            return f"{path} is a binary file of {len(data)} bytes."
        lines = data.decode("utf-8", errors="replace").splitlines()
        start = max(offset, 1) - 1
        chosen = lines[start : start + max(limit, 1)]
        numbered = "\n".join(f"{start + i + 1:6d}\t{line}" for i, line in enumerate(chosen))
        if start + len(chosen) < len(lines):
            numbered += f"\n(lines {start + 1}-{start + len(chosen)} of {len(lines)})"
        return numbered or f"{path} is empty."

    def _write(self, path: str, content: str) -> str:
        target = self._path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8", newline="") as handle:
            handle.write(content)
        return f"Wrote {path}."

    def _edit(self, path: str, old_string: str, new_string: str, replace_all: bool = False) -> str:
        target = self._path(path)
        with target.open(encoding="utf-8", newline="") as handle:
            text = handle.read()
        count = text.count(old_string)
        if count == 0:
            raise FileToolError(f"{path}: old_string not found")
        if count > 1 and not replace_all:
            raise FileToolError(f"{path}: old_string occurs {count} times; set replace_all")
        text = (
            text.replace(old_string, new_string)
            if replace_all
            else text.replace(old_string, new_string, 1)
        )
        with target.open("w", encoding="utf-8", newline="") as handle:
            handle.write(text)
        return f"Edited {path}."

    def _list(self, path: str = ".") -> str:
        target = self._path(path)
        entries = []
        for child in sorted(target.iterdir()):
            if child.is_dir():
                entries.append(f"{child.name}/")
            else:
                entries.append(f"{child.name}  {child.stat().st_size} bytes")
        return "\n".join(entries) or "(empty)"
