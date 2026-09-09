"""No tracked file carries a real person's path, address, or session link.

A recorded simulator artifact keeps whatever path it was produced from, and
the sdist ships ``tests/``, so a fixture's bytes travel with every release.
Neither is a reason to stop recording real artifacts — a hand-written one
tests our idea of a format rather than the format — but it does decide how
they have to be checked.

LTspice writes its raw header in UTF-16LE, which a plain text search reads
as interleaved NUL bytes and never matches. So the question worth asking is
not "is the pattern absent from this file's text" but "is it absent from
every encoding this file could be in". These tests decode each tracked file
both ways before looking.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

#: Stand-in user names a fixture or doc is allowed to contain. A real name
#: reaching this list is the bug the module exists to prevent, so add to it
#: only for something that is plainly not a person.
PLACEHOLDER_NAMES = frozenset({"user", "me", "dev", "test", "u", "...", "youruser"})

_HOME_PATTERNS = (
    ("Windows home", re.compile(r"C:\\Users\\([A-Za-z0-9_.-]+)", re.IGNORECASE)),
    ("POSIX home", re.compile(r"/home/([A-Za-z0-9_.-]+)")),
    ("root home", re.compile(r"/root/([A-Za-z0-9_.-]+)")),
)
_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_SESSION_LINK = re.compile(r"claude\.ai/code/session|Claude-Session:", re.IGNORECASE)

#: The encodings a tracked file can plausibly be in. UTF-16LE is the one that
#: matters: it is what LTspice writes, and it is invisible to a plain search.
_ENCODINGS = ("utf-8", "utf-16-le")


#: This module is the one tracked file that must contain the strings it hunts
#: for, because it defines them. Excluding it is derived, never a list anyone
#: can add to: the test below asserts the scan skipped exactly this file.
_PATTERN_SOURCE = Path(__file__).resolve()


def _tracked_files() -> list[Path]:
    try:
        listed = subprocess.run(
            ["git", "ls-files", "-z"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:  # pragma: no cover
        pytest.skip(f"git is not usable here, so the tracked set is unknown: {exc}")
    return [ROOT / name for name in listed.split("\0") if name]


def _readings(path: Path) -> list[str]:
    """The file's bytes decoded every way it could have been written."""
    try:
        data = path.read_bytes()
    except OSError:  # pragma: no cover - a tracked path that cannot be read
        return []
    return [data.decode(encoding, errors="ignore") for encoding in _ENCODINGS]


def _findings(text: str) -> list[str]:
    found = []
    for label, pattern in _HOME_PATTERNS:
        for name in pattern.findall(text):
            if name.lower() not in PLACEHOLDER_NAMES:
                found.append(f"{label} of {name!r}")
    found.extend(f"e-mail address {address!r}" for address in set(_EMAIL.findall(text)))
    if _SESSION_LINK.search(text):
        found.append("session link")
    return found


class TestTrackedFilesCarryNoPrivateInformation:
    def test_no_tracked_file_names_a_real_person(self) -> None:
        offenders: dict[str, list[str]] = {}
        skipped: list[Path] = []
        for path in _tracked_files():
            if path.resolve() == _PATTERN_SOURCE:
                skipped.append(path)
                continue
            found = sorted({item for text in _readings(path) for item in _findings(text)})
            if found:
                offenders[str(path.relative_to(ROOT))] = found

        assert skipped == [_PATTERN_SOURCE], (
            "the scan must skip this module and nothing else; "
            f"it skipped {[str(p) for p in skipped]}"
        )

        assert not offenders, "private information in tracked files: " + "; ".join(
            f"{name}: {', '.join(items)}" for name, items in sorted(offenders.items())
        )

    def test_the_scan_reads_utf16_and_not_only_text(self, tmp_path: Path) -> None:
        """The instrument's own blind spot, pinned.

        A text-only version of this scan passes on the file that started all
        this, so the guard is worth no more than its ability to decode.
        """
        planted = tmp_path / "ltspice_shaped.raw"
        planted.write_bytes("Title: C:\\Users\\realname\\sim.cir\n".encode("utf-16-le"))

        assert "Windows home of 'realname'" in _findings(planted.read_bytes().decode("utf-16-le"))
        assert not _findings(planted.read_bytes().decode("utf-8", errors="ignore"))
        assert _findings(_readings(planted)[1])
