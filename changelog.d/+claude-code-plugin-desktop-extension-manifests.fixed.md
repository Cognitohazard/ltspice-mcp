The Claude Code plugin and Desktop-extension manifests still said 0.6.1
after the 0.7.0 release, because the 0.7.0 tag was cut without
`scripts/release.sh`, which stamps them. They now say 0.7.0, and the
publish workflow refuses a tag whose manifests name another version.
