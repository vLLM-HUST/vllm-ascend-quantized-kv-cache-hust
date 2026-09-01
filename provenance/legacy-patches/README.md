# Legacy patch archive

This directory preserves original contribution commits as unmodified
`git format-patch` exports, retaining authors, dates, messages, trailers, and
exact diffs. It contains core PRs #118 and #181 and Ascend PRs #116, #160, and
#207.

Core PR #161 is deliberately excluded from blind archival because its branch
contains a large mixed integration history unrelated to one extension. It must
be mined file-by-file. These files are evidence, not patches to apply blindly.
