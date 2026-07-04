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
git -C "$root" tag "v$ver"

echo "Tagged v$ver. Publish (triggers PyPI) with:  git push && git push origin v$ver"
