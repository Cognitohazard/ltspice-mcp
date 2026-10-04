"""The checkout and push-range guards share one privacy scanner."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from tests import privacy_scan

ROOT = Path(__file__).resolve().parent.parent


def _run(repo: Path, *args: str) -> bytes:
    return subprocess.run(list(args), cwd=repo, capture_output=True, check=True).stdout


def _commit(repo: Path, message: str) -> str:
    _run(repo, "git", "add", "-A")
    _run(
        repo,
        "git",
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test" + "@" + "example.invalid",
        "commit",
        "-qm",
        message,
    )
    return _run(repo, "git", "rev-parse", "HEAD").decode().strip()


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _run(repo, "git", "init", "-q")
    (repo / ".gitignore").write_text("__pycache__/\n")
    (repo / "scripts").mkdir()
    (repo / ".githooks").mkdir()
    if not (ROOT / "scripts/privacy_scan.py").exists():
        pytest.skip("maintainer scripts are absent from the sdist")
    (repo / "tests").mkdir()
    shutil.copy2(ROOT / "tests/__init__.py", repo / "tests/__init__.py")
    shutil.copy2(ROOT / "tests/privacy_scan.py", repo / "tests/privacy_scan.py")
    shutil.copy2(ROOT / "scripts/privacy_scan.py", repo / "scripts/privacy_scan.py")
    shutil.copy2(ROOT / ".githooks/pre-push", repo / ".githooks/pre-push")
    _run(repo, "git", "config", "core.hooksPath", ".githooks")
    monkeypatch.setattr(privacy_scan, "ROOT", repo)
    return repo


def _bare_remote(repo: Path, base: str) -> Path:
    remote = repo.parent / "remote.git"
    _run(repo, "git", "init", "--bare", "-q", str(remote))
    _run(repo, "git", "remote", "add", "origin", str(remote))
    _run(repo, "git", "push", "-q", "origin", f"{base}:refs/heads/main")
    return remote


def _push(repo: Path, refspec: str) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(["git", "push", "-q", "origin", refspec], cwd=repo, capture_output=True)


def _cli(repo: Path, remote: Path, update: bytes) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        [sys.executable, "scripts/privacy_scan.py", "pre-push", "origin", str(remote)],
        cwd=repo,
        input=update,
        capture_output=True,
    )


def _categories(data: bytes) -> set[str]:
    return {finding.category for finding in privacy_scan.scan_bytes(data)}


def test_current_tracked_tree_is_private() -> None:
    try:
        top = _run(ROOT, "git", "rev-parse", "--show-toplevel").decode().strip()
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("Git tracked files are unavailable in this source tree")
    if Path(top).resolve() != ROOT.resolve():
        pytest.skip("source tree is outside its Git checkout")
    assert privacy_scan.tracked_findings() == []


@pytest.mark.parametrize(
    "path",
    [
        lambda n: "D:" + "\\" + "Users" + "\\" + n + "\\file.cir",
        lambda n: "E:" + "//Users/" + n + "/file.cir",
        lambda n: "F:" + "\\\\Users\\\\" + n + "\\\\file.cir",
        lambda n: "/mnt/c/Users/" + n + "/file.cir",
        lambda n: "/Users/" + n + "/file.cir",
        lambda n: "/home/" + n + "/file.cir",
        lambda n: "/root/" + n + "/file.cir",
    ],
)
def test_home_paths_and_both_utf16_orders(path) -> None:
    value = path("syntheticperson")
    for encoding in privacy_scan.ENCODINGS:
        assert any("home" in category for category in _categories(value.encode(encoding)))


@pytest.mark.parametrize("name", ["user", "me", "dev", "test", "u", "youruser"])
def test_neutral_home_placeholders_are_allowed(name: str) -> None:
    value = "/home/" + name + "/fixture.cir"
    assert not _categories(value.encode())


def test_common_token_shapes_and_safe_findings() -> None:
    values = (
        "ghp_" + "A" * 36,
        "github_pat_" + "B" * 24,
        "sk-" + "C" * 32,
        "sk-ant-" + "D" * 32,
        "AKIA" + "E" * 16,
        "xoxb-" + "F" * 24,
        "AIza" + "G" * 35,
    )
    for value in values:
        findings = privacy_scan.scan_bytes(value.encode("utf-16-be"))
        assert findings
        assert value not in repr(findings)


def test_ai_attribution_requires_proper_trailer() -> None:
    address = "noreply" + "@" + "anthropic.com"
    valid = "Change behavior\n\nCo-Authored-By: Claude <" + address + ">\n"
    assert not privacy_scan.scan_message(valid.encode())
    assert privacy_scan.scan_message(("Change behavior " + address).encode())
    for encoding in ("utf-16-le", "utf-16-be"):
        assert privacy_scan.scan_message(("Change behavior " + address).encode(encoding))
    assert privacy_scan.scan_message(
        ("Change behavior\n\nSigned-off-by: Claude <" + address + ">\n").encode()
    )
    assert privacy_scan.scan_message(
        ("Change behavior\nCo-Authored-By: Claude <" + address + ">\n").encode()
    )


def test_intermediate_revision_and_filename_are_checked(repo: Path) -> None:
    safe = repo / "safe.txt"
    safe.write_text("ordinary text")
    base = _commit(repo, "Base")
    _bare_remote(repo, base)

    secret = "sk-" + "Z" * 32
    unsafe_name = "syntheticperson" + "@" + "example.invalid.raw"
    unsafe = repo / unsafe_name
    unsafe.write_bytes(secret.encode("utf-16-be"))
    introduced = _commit(repo, "Introduce artifact")
    unsafe.unlink()
    safe.write_text("ordinary revision")
    final = _commit(repo, "Remove artifact")

    findings = privacy_scan.range_findings(privacy_scan.new_commits(final, [base]))
    assert any(
        "email address" in finding and introduced in finding and "filename" in finding
        for finding in findings
    )
    assert any("OpenAI token" in finding and introduced in finding for finding in findings)
    assert secret not in repr(findings)
    assert unsafe_name not in repr(findings)
    assert privacy_scan.new_commits(final, [base]) == [introduced, final]
    result = _push(repo, "HEAD:refs/heads/main")
    assert result.returncode == 1
    assert introduced.encode() in result.stderr
    assert unsafe_name.encode() not in result.stderr
    assert secret.encode() not in result.stderr


def test_push_cli_checks_range_and_fails_closed(repo: Path) -> None:
    (repo / "safe.txt").write_text("ordinary text")
    base = _commit(repo, "Base")
    remote = _bare_remote(repo, base)
    (repo / "safe.txt").write_text("updated")
    new = _commit(repo, "Change\n\nReviewed-by: Someone")
    update = f"refs/heads/main {new} refs/heads/main {base}\n".encode()
    result = _cli(repo, remote, update)
    assert result.returncode == 1
    assert b"disallowed trailer" in result.stderr
    assert b"Someone" not in result.stderr

    missing = f"refs/heads/main {new} refs/heads/main {'a' * 40}\n".encode()
    result = _cli(repo, remote, missing)
    assert result.returncode == 2
    assert b"range error" in result.stderr

    deleted = f"refs/heads/main {'0' * 40} refs/heads/main {base}\n".encode()
    unavailable = repo / "missing-remote"
    result = _cli(repo, unavailable, update)
    assert result.returncode == 2
    assert str(unavailable).encode() not in result.stderr

    result = _cli(repo, unavailable, deleted)
    assert result.returncode == 0


def test_pre_push_new_branch_checks_introduced_revision(repo: Path) -> None:
    (repo / "safe.txt").write_text("ordinary text")
    base = _commit(repo, "Base")
    _bare_remote(repo, base)
    (repo / "safe.txt").write_text("sk-" + "Q" * 32)
    _commit(repo, "Add data")
    result = _push(repo, "HEAD:refs/heads/topic")
    assert result.returncode == 1
    assert b"OpenAI token" in result.stderr
    assert b"sk-" not in result.stderr


def test_push_ignores_stale_tracking_ref(repo: Path) -> None:
    (repo / "safe.txt").write_text("ordinary text")
    base = _commit(repo, "Base")
    _bare_remote(repo, base)
    (repo / "safe.txt").write_text("sk-" + "S" * 32)
    new = _commit(repo, "Add data")
    _run(repo, "git", "update-ref", "refs/remotes/origin/stale", new)

    result = _push(repo, "HEAD:refs/heads/new-branch")
    assert result.returncode != 0
    assert b"OpenAI token" in result.stderr


def test_ai_trailer_does_not_hide_secret() -> None:
    token = "ghp_" + "X" * 36
    address = "noreply" + "@" + "anthropic.com"
    message = "Change\n\nCo-Authored-By: Claude " + token + " <" + address + ">\n"
    findings = privacy_scan.scan_message(message.encode())
    assert "GitHub token" in {finding.category for finding in findings}
    assert token not in repr(findings)


@pytest.mark.parametrize("nested", [False, True])
def test_push_scans_annotated_tag_messages(repo: Path, nested: bool) -> None:
    (repo / "safe.txt").write_text("ordinary text")
    base = _commit(repo, "Base")
    _bare_remote(repo, base)
    token = "ghp_" + "Y" * 36
    identity = ["-c", "user.name=Test", "-c", "user.email=test" + "@" + "example.invalid"]
    _run(repo, "git", *identity, "tag", "-am", "Release " + token, "inner")
    tag = "inner"
    if nested:
        _run(repo, "git", *identity, "tag", "-am", "Release", "outer", "inner")
        tag = "outer"

    result = _push(repo, f"refs/tags/{tag}:refs/tags/{tag}")
    assert result.returncode != 0
    assert b"GitHub token" in result.stderr
    assert token.encode() not in result.stderr


def _remote_only_commit(remote: Path, base: str, ref: str = "refs/heads/other") -> str:
    remote_only = (
        _run(
            remote,
            "git",
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test" + "@" + "example.invalid",
            "commit-tree",
            f"{base}^{{tree}}",
            "-p",
            base,
            "-m",
            "Remote update",
        )
        .decode()
        .strip()
    )
    _run(remote, "git", "update-ref", ref, remote_only)
    return remote_only


@pytest.mark.parametrize("target", ["main", "topic"])
def test_push_accepts_clean_range_with_unavailable_unrelated_ref(repo: Path, target: str) -> None:
    (repo / "safe.txt").write_text("ordinary text")
    base = _commit(repo, "Base")
    remote = _bare_remote(repo, base)
    remote_only = _remote_only_commit(remote, base, "refs/pull/1/head")
    (repo / "safe.txt").write_text("ordinary update")
    new = _commit(repo, "Update")

    result = _push(repo, f"HEAD:refs/heads/{target}")
    assert result.returncode == 0, result.stderr.decode(errors="replace")
    assert _run(remote, "git", "rev-parse", f"refs/heads/{target}").decode().strip() == new
    with pytest.raises(privacy_scan.ScanError):
        privacy_scan.commit_exists(remote_only)


def test_push_checks_deleted_private_revision_with_unknown_unrelated_ref(repo: Path) -> None:
    safe = repo / "safe.txt"
    safe.write_text("ordinary text")
    base = _commit(repo, "Base")
    remote = _bare_remote(repo, base)
    _remote_only_commit(remote, base)
    unsafe = repo / "artifact.txt"
    token = "sk-" + "W" * 32
    unsafe.write_bytes(token.encode("utf-16-be"))
    introduced = _commit(repo, "Add artifact")
    unsafe.unlink()
    _commit(repo, "Remove artifact")

    result = _push(repo, "HEAD:refs/heads/topic")
    assert result.returncode != 0
    assert b"OpenAI token" in result.stderr
    assert introduced.encode() in result.stderr
    assert token.encode() not in result.stderr
    assert _run(remote, "git", "for-each-ref", "refs/heads/topic") == b""
    assert _run(remote, "git", "rev-parse", "refs/heads/main").decode().strip() == base


def test_push_requires_old_target_even_when_other_refs_are_unknown(repo: Path) -> None:
    (repo / "safe.txt").write_text("ordinary text")
    base = _commit(repo, "Base")
    remote = _bare_remote(repo, base)
    _remote_only_commit(remote, base)
    previous = _remote_only_commit(remote, base, "refs/heads/main")
    update = f"refs/heads/main {base} refs/heads/main {previous}\n".encode()
    result = _cli(repo, remote, update)
    assert result.returncode == 2
    assert b"range error" in result.stderr


def test_push_refuses_a_changed_target_with_an_unknown_unrelated_ref(repo: Path) -> None:
    (repo / "safe.txt").write_text("ordinary text")
    base = _commit(repo, "Base")
    remote = _bare_remote(repo, base)
    (repo / "safe.txt").write_text("ordinary update")
    new = _commit(repo, "Update")
    _remote_only_commit(remote, base)
    # Both expected objects exist, but the receiver still advertises base.
    update = f"refs/heads/main {new} refs/heads/main {new}\n".encode()
    result = _cli(repo, remote, update)
    assert result.returncode == 2
    assert b"remote ref changed" in result.stderr


def test_push_refuses_missing_introduced_content_with_unknown_unrelated_ref(repo: Path) -> None:
    (repo / "safe.txt").write_text("ordinary text")
    base = _commit(repo, "Base")
    remote = _bare_remote(repo, base)
    _remote_only_commit(remote, base)
    (repo / "safe.txt").write_text("ordinary update")
    new = _commit(repo, "Update")
    blob = _run(repo, "git", "rev-parse", f"{new}:safe.txt").decode().strip()
    blob_path = repo / ".git" / "objects" / blob[:2] / blob[2:]
    blob_path.chmod(0o600)
    blob_path.unlink()
    update = f"refs/heads/main {new} refs/heads/main {base}\n".encode()
    result = _cli(repo, remote, update)
    assert result.returncode == 2
    assert b"range error" in result.stderr


def test_push_accepts_clean_annotated_tag(repo: Path) -> None:
    (repo / "safe.txt").write_text("ordinary text")
    base = _commit(repo, "Base")
    _bare_remote(repo, base)
    address = "noreply" + "@" + "openai.com"
    message = "Release\n\nCo-Authored-By: Codex <" + address + ">"
    _run(
        repo,
        "git",
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test" + "@" + "example.invalid",
        "tag",
        "-am",
        message,
        "clean",
    )
    result = _push(repo, "refs/tags/clean:refs/tags/clean")
    assert result.returncode == 0
