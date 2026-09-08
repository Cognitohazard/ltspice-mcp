#!/usr/bin/env bash
# Cut a release: stamp the plugin manifests, commit, tag.
#
# The package version comes from the git tag via hatch-vcs. The plugin
# manifests are static JSON that Claude Code reads from the repo, so they
# can't derive the version dynamically — this script is the single input
# that keeps every manifest "version" slot in lockstep with the tag.
#
# Usage: scripts/release.sh 0.5.1
set -euo pipefail

ver="${1:?usage: release.sh <version>  (e.g. 0.5.1)}"
ver="${ver#v}"  # tolerate a leading v
root="$(cd "$(dirname "$0")/.." && pwd)"

# Refuse to tag a dirty tree or an unwritten release note: the tag is what
# PyPI builds from, and the changelog section is what users read for it.
if [ -n "$(git -C "$root" status --porcelain)" ]; then
  echo "release.sh: working tree is not clean; commit or stash first" >&2
  exit 1
fi
if ! grep -qE "^## \[$ver\] - [0-9]{4}-[0-9]{2}-[0-9]{2}" "$root/CHANGELOG.md"; then
  echo "release.sh: CHANGELOG.md has no dated '## [$ver] - YYYY-MM-DD' section" >&2
  exit 1
fi

# Rewrite every `"version": "..."` slot across the plugin + Desktop-extension
# manifests (the four spots test_bundle_versions_agree pins together). sed, not
# a JSON round-trip, so the diff is one line per slot and em dashes survive.
manifests=(
  .claude-plugin/plugin.json
  .claude-plugin/marketplace.json
  packaging/mcpb/manifest.json
)
sed -i -E 's/("version": *)"[^"]*"/\1"'"$ver"'"/' "${manifests[@]/#/$root/}"

git -C "$root" add "${manifests[@]}"
git -C "$root" commit -m "release: v$ver"
# Annotated, so the tag carries its own object and message rather than
# borrowing the release commit's subject when something reads it back.
git -C "$root" tag -a "v$ver" -m "Release $ver"

echo "Tagged v$ver. Publish (triggers PyPI) with:  git push && git push origin v$ver"
