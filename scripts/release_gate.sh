#!/usr/bin/env bash
# The release gate: every shape the runners and the users have, run before a
# push, so the push confirms a result instead of testing a guess. Each shape
# prints PASS, FAIL or SKIP (with why); the script exits non-zero on any FAIL.
#
# Shapes:
#   1. Linux, the suite serially (what CI runs)
#   2. Linux with WSL detection forced off (Linux CI is not WSL; this box is)
#   3. Ubuntu container, non-root, with an init process, ngspice + libcairo2
#   4. Windows native, Python 3.12, checkout with line-ending conversion ON
#   5. Windows native, Python 3.13, same checkout
#   6. The publisher's own metadata check on a fresh build (the PyPI action's
#      bundled checker, which lags the build backend's default metadata version)
#
# Shapes 4 and 5 need a clone of this repository on a Windows disk, reachable
# from WSL, with the Windows-side uv on PATH: set LTSPICE_MCP_WINDOWS_CLONE to
# its WSL path (for example /mnt/c/Users/<you>/ltspice-mcp-ci). The Windows
# path is derived with wslpath. Without it those shapes SKIP loudly.
#
# Shape 6 needs docker; the image tag must match the SHA pinned on the publish
# step in .github/workflows/publish.yml.
set -u
cd "$(dirname "$0")/.."
PUBLISH_IMAGE=ghcr.io/pypa/gh-action-pypi-publish:dc37677b2e1c63e2034f94d8a5b11f265b73ba33
scratch=$(mktemp -d)
status=0
report() { printf '%-4s %s\n' "$1" "$2"; [ "$1" = FAIL ] && status=1; return 0; }
summary() { grep -E "^[0-9]+ (passed|failed)|[0-9]+ passed" "$1" | tail -1; }

# 1. Linux serial
if uv run pytest tests/ -q -p no:cacheprovider --no-header > "$scratch/linux.txt" 2>&1; then
  report PASS "linux serial: $(summary "$scratch/linux.txt")"
else
  report FAIL "linux serial: $(summary "$scratch/linux.txt") (see $scratch/linux.txt)"
fi

# 2. Linux, WSL detection off
if PYTHONPATH=scripts uv run pytest -p nonwsl_plugin tests/ -q -p no:cacheprovider --no-header > "$scratch/nonwsl.txt" 2>&1; then
  report PASS "linux non-WSL: $(summary "$scratch/nonwsl.txt")"
else
  report FAIL "linux non-WSL: $(summary "$scratch/nonwsl.txt") (see $scratch/nonwsl.txt)"
fi

# 3. Container, non-root, --init (clones the LOCAL master, so commit first or
#    accept that the container tests the last commit, not the working tree)
if command -v docker >/dev/null; then
  docker run --rm --init -v "$PWD":/src:ro ubuntu:24.04 bash -c '
set -o pipefail; export DEBIAN_FRONTEND=noninteractive CI=true GITHUB_ACTIONS=true
apt-get update -qq >/dev/null && apt-get install -y -qq git ngspice libcairo2 curl ca-certificates >/dev/null 2>&1
useradd -m runner; git config --system --add safe.directory "*"
git clone -q --depth 1 --branch master file:///src /work && chown -R runner /work
su runner -s /bin/bash -c "cd /work; curl -LsSf https://astral.sh/uv/install.sh 2>/dev/null | sh >/dev/null 2>&1; export PATH=\$HOME/.local/bin:\$PATH
uv sync -q; uv run pytest tests/ -q -p no:cacheprovider --no-header --cov=ltspice_mcp --cov-fail-under=80 2>&1 | tail -2"' > "$scratch/container.txt" 2>&1
  if grep -qE "^[0-9]+ passed" "$scratch/container.txt" && ! grep -qE "[0-9]+ failed|error" "$scratch/container.txt"; then
    report PASS "container (last commit on master): $(summary "$scratch/container.txt")"
  else
    report FAIL "container: $(tail -3 "$scratch/container.txt" | tr '\n' ' ')"
  fi
else
  report SKIP "container: docker not on PATH"
fi

# 4/5. Windows native, both interpreters, checkout with conversion on
if [ -n "${LTSPICE_MCP_WINDOWS_CLONE:-}" ] && [ -d "$LTSPICE_MCP_WINDOWS_CLONE/.git" ]; then
  clone=$LTSPICE_MCP_WINDOWS_CLONE
  winclone=$(wslpath -w "$clone")
  # The clone's own commit, re-checked-out with conversion on (the bytes a
  # runner sees), then this working tree's files on top, new ones included.
  git -C "$clone" config core.autocrlf true
  git -C "$clone" reset -q --hard HEAD
  git ls-files -z --cached --others --exclude-standard | while IFS= read -r -d '' f; do
    mkdir -p "$clone/$(dirname "$f")" && cp "$f" "$clone/$f"
  done
  for py in 3.12 3.13; do
    cmd.exe /c "cd /d $winclone && uv python pin $py >nul && uv run pytest tests -q -p no:cacheprovider --no-header > pytest_gate_$py.txt 2>&1" >/dev/null 2>&1
    tr -d '\r' < "$clone/pytest_gate_$py.txt" > "$scratch/windows_$py.txt"
    if grep -qE "^[0-9]+ passed" "$scratch/windows_$py.txt" && ! grep -qE "[0-9]+ failed|[0-9]+ error" "$scratch/windows_$py.txt"; then
      report PASS "windows $py: $(summary "$scratch/windows_$py.txt")"
    else
      report FAIL "windows $py: $(summary "$scratch/windows_$py.txt") (see $scratch/windows_$py.txt)"
    fi
  done
  git -C "$clone" checkout -q -- .python-version 2>/dev/null
else
  report SKIP "windows 3.12 / 3.13: set LTSPICE_MCP_WINDOWS_CLONE to a Windows-disk clone"
fi

# 6. The publisher's own metadata check
if command -v docker >/dev/null; then
  rm -rf "$scratch/dist" && uv build -q -o "$scratch/dist" > "$scratch/build.txt" 2>&1 \
    && docker run --rm --entrypoint python -v "$scratch/dist":/dist:ro "$PUBLISH_IMAGE" -m twine check --strict /dist/* > "$scratch/twine.txt" 2>&1
  if [ $? -eq 0 ]; then
    report PASS "publisher metadata check: $(grep -c PASSED "$scratch/twine.txt") files"
  else
    report FAIL "publisher metadata check: $(grep -E 'ERROR|error' "$scratch/twine.txt" | head -1)"
  fi
else
  report SKIP "publisher metadata check: docker not on PATH"
fi

echo "logs: $scratch"
exit $status
