"""Check tracked content and commits about to leave this repository.

The same byte scanner serves the checkout test and the local push gates.
Diagnostics deliberately omit matched text and filenames.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PLACEHOLDERS = frozenset({"user", "me", "dev", "test", "u", "...", "youruser", "foo"})
ENCODINGS = ("utf-8", "utf-16-le", "utf-16-be")
ZERO_OIDS = {"0" * 40, "0" * 64}
OID = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})\Z", re.IGNORECASE)
REMOTE_NAME = re.compile(r"[A-Za-z0-9._/-]+\Z")

HOME_PATHS = (
    (
        "Windows home",
        re.compile(
            r"(?:[A-Z]:[\\/]+|/mnt/[a-z]/)Users[\\/]+(?P<name>[A-Za-z0-9_.-]+)", re.IGNORECASE
        ),
    ),
    (
        "POSIX home",
        re.compile(
            r"(?<![A-Za-z]:)(?<!/mnt/[A-Za-z])/(?:home|root|Users)/(?P<name>[A-Za-z0-9_.-]+)"
        ),
    ),
)
PATTERNS = (
    ("email address", re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")),
    (
        "session link",
        re.compile(r"claude\.ai/code/" r"session|Claude-" r"Session:", re.IGNORECASE),
    ),
    (
        "GitHub token",
        re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{36}|github_pat_[A-Za-z0-9_]{22,})\b"),
    ),
    ("OpenAI token", re.compile(r"\bsk-(?!ant-)[A-Za-z0-9_-]{20,}\b")),
    ("Anthropic token", re.compile(r"\bsk-ant-[A-Za-z0-9_-]{20,}\b")),
    ("AWS access key", re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b")),
    ("Slack token", re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{20,}\b")),
    ("Google API key", re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b")),
    ("private key", re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----")),
)
TRAILER = re.compile(r"([A-Za-z][A-Za-z-]*):[ \t]+(.+)\Z")
AI_ATTRIBUTION = re.compile(
    r"(?:Claude(?: (?:Opus|Sonnet|Haiku) [0-9]+(?:\.[0-9]+)*"
    r"(?: \([0-9]+[KM] context\))?)?|Codex|ChatGPT|OpenAI) "
    r"<(?P<address>noreply@(?:anthropic|openai)\.com)>\Z"
)


class ScanError(Exception):
    """The expected Git range or content could not be inspected."""


@dataclass(frozen=True)
class Finding:
    category: str
    line: int


def scan_text(text: str) -> set[Finding]:
    findings = set()
    for category, pattern in HOME_PATHS:
        for match in pattern.finditer(text):
            if match.group("name").lower() not in PLACEHOLDERS:
                findings.add(Finding(category, text.count("\n", 0, match.start()) + 1))
    for category, pattern in PATTERNS:
        for match in pattern.finditer(text):
            findings.add(Finding(category, text.count("\n", 0, match.start()) + 1))
    return findings


def scan_bytes(data: bytes) -> set[Finding]:
    return {
        finding
        for encoding in ENCODINGS
        for finding in scan_text(data.decode(encoding, errors="ignore"))
    }


def scan_message(data: bytes) -> set[Finding]:
    findings = set()
    for encoding in ENCODINGS:
        text = data.decode(encoding, errors="ignore")
        lines = text.rstrip("\r\n").splitlines()
        footer_start = len(lines)
        while footer_start and TRAILER.fullmatch(lines[footer_start - 1]):
            footer_start -= 1
        if footer_start < len(lines):
            separated = footer_start > 0 and lines[footer_start - 1] == ""
            for index in range(footer_start, len(lines)):
                match = TRAILER.fullmatch(lines[index])
                assert match is not None
                attribution = AI_ATTRIBUTION.fullmatch(match.group(2))
                if separated and match.group(1) == "Co-Authored-By" and attribution:
                    lines[index] = lines[index].replace(attribution.group("address"), "", 1)
                else:
                    findings.add(Finding("disallowed trailer", index + 1))
        findings.update(scan_text("\n".join(lines)))
    return findings


def git(*args: str) -> bytes:
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ScanError("Git could not provide the expected range or content") from exc


def oid(value: str) -> str:
    if not OID.fullmatch(value):
        raise ScanError("invalid object ID in push update")
    return value


def commit_exists(value: str) -> None:
    git("cat-file", "-e", f"{oid(value)}^{{commit}}")


def tracking_oids(remote: str) -> list[str]:
    """Cached tips are suitable only for the explicitly offline local preview."""
    if not REMOTE_NAME.fullmatch(remote):
        return []
    data = git("for-each-ref", "--format=%(objectname)", f"refs/remotes/{remote}/")
    return [oid(item) for item in data.decode("ascii").splitlines()]


def advertised_refs(remote_url: str) -> dict[str, str]:
    if not remote_url:
        raise ScanError("receiving remote URL is required")
    refs = {}
    for line in git("ls-remote", "--refs", "--", remote_url).splitlines():
        value, separator, name = line.partition(b"\t")
        if not separator:
            raise ScanError("invalid remote advertisement")
        refs[name.decode("utf-8", errors="surrogateescape")] = oid(value.decode("ascii"))
    return refs


def peel_tags(value: str) -> tuple[str, dict[str, bytes]]:
    """Read every annotation in a tag chain and return its target commit."""
    tags = {}
    while git("cat-file", "-t", oid(value)).strip() == b"tag":
        header, separator, message = git("cat-file", "tag", value).partition(b"\n\n")
        first_line = header.partition(b"\n")[0]
        if not separator or not first_line.startswith(b"object "):
            raise ScanError("invalid annotated tag")
        tags[value] = message
        value = oid(first_line.removeprefix(b"object ").decode("ascii"))
    commit_exists(value)
    return value, tags


def new_commits(local: str, excluded: list[str]) -> list[str]:
    commit_exists(local)
    for base in excluded:
        commit_exists(base)
    data = git("rev-list", "--reverse", local, *[f"^{base}" for base in excluded])
    return [oid(item) for item in data.decode("ascii").splitlines()]


def tracked_findings() -> list[str]:
    results = []
    paths = [name for name in git("ls-files", "-z").split(b"\0") if name]
    for index, name in enumerate(paths, 1):
        path = ROOT / name.decode("utf-8", errors="surrogateescape")
        try:
            data = path.read_bytes()
        except OSError as exc:
            raise ScanError("tracked file could not be read") from exc
        filename_findings = scan_bytes(name)
        for finding in sorted(filename_findings, key=lambda f: (f.line, f.category)):
            results.append(f"{finding.category}: tracked file #{index}, filename")
        for finding in sorted(scan_bytes(data), key=lambda f: (f.line, f.category)):
            results.append(f"{finding.category}: tracked file #{index}, line {finding.line}")
    return results


def range_findings(commits: list[str]) -> list[str]:
    results = []
    for commit in commits:
        for finding in sorted(
            scan_message(git("log", "-1", "--format=%B", commit)),
            key=lambda f: (f.line, f.category),
        ):
            results.append(f"{finding.category}: commit {commit}, message line {finding.line}")
        revisions = [
            name
            for name in git(
                "diff-tree",
                "--root",
                "-m",
                "-r",
                "--no-commit-id",
                "--name-only",
                "-z",
                "--diff-filter=ACMRT",
                commit,
            ).split(b"\0")
            if name
        ]
        for index, name in enumerate(dict.fromkeys(revisions), 1):
            filename_findings = scan_bytes(name)
            path = name.decode("utf-8", errors="surrogateescape")
            data = git("cat-file", "blob", f"{commit}:{path}")
            for finding in sorted(filename_findings, key=lambda f: (f.line, f.category)):
                results.append(f"{finding.category}: commit {commit}, file #{index}, filename")
            for finding in sorted(scan_bytes(data), key=lambda f: (f.line, f.category)):
                results.append(
                    f"{finding.category}: commit {commit}, file #{index}, line {finding.line}"
                )
    return results


def push_findings(remote_url: str, updates: bytes) -> list[str]:
    pending = []
    if not updates.strip():
        raise ScanError("no push updates supplied")
    for update in updates.splitlines():
        parts = update.decode("utf-8", errors="surrogateescape").split()
        if len(parts) != 4:
            raise ScanError("invalid push update")
        _, local, remote_ref, previous = parts
        oid(local)
        oid(previous)
        if local not in ZERO_OIDS:
            pending.append((local, remote_ref, previous))
    if not pending:
        return []  # Deleted refs introduce no content and need no range lookup.

    advertised = advertised_refs(remote_url)
    excluded = []
    known_tags = set()
    for value in dict.fromkeys(advertised.values()):
        try:
            commit, tags = peel_tags(value)
        except ScanError:
            # An optional exclusion we cannot inspect excludes nothing. The
            # actual target and introduced history are still required below.
            continue
        excluded.append(commit)
        known_tags.update(tags)

    results = []
    for local, remote_ref, previous in pending:
        if previous not in ZERO_OIDS:
            peel_tags(previous)  # An expected object must be available locally.
        if advertised.get(remote_ref, "0" * len(previous)) != previous:
            raise ScanError("remote ref changed since the push update")
        commit, tags = peel_tags(local)
        for tag, message in tags.items():
            if tag not in known_tags:
                for finding in sorted(scan_message(message), key=lambda f: (f.line, f.category)):
                    results.append(f"{finding.category}: tag {tag}, message line {finding.line}")
        results.extend(range_findings(new_commits(commit, excluded)))
    return list(dict.fromkeys(results))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("tracked", "local", "pre-push"))
    parser.add_argument("remote", nargs="?", default="origin")
    parser.add_argument("remote_url", nargs="?")
    args = parser.parse_args(argv)
    try:
        if args.mode == "tracked":
            findings = tracked_findings()
        elif args.mode == "local":
            excluded = tracking_oids(args.remote)
            if not excluded:
                raise ScanError("no remote tracking range is available")
            head = git("rev-parse", "--verify", "HEAD^{commit}").decode("ascii").strip()
            findings = range_findings(new_commits(head, excluded))
        else:
            findings = push_findings(args.remote_url or "", sys.stdin.buffer.read())
    except ScanError as exc:
        print(f"privacy range error: {exc}", file=sys.stderr)
        return 2
    for finding in findings:
        print(finding, file=sys.stderr)
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
