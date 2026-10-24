#!/usr/bin/env bash
# Build paper/build/changes.pdf: the manuscript with every change since a base
# revision marked (removed text in red strikethrough, new text in blue).
#
# Usage (from anywhere in the repository):
#   scripts/build_changes_pdf.sh                # compare with origin/main
#   scripts/build_changes_pdf.sh <git-revision> # compare with any branch/commit
#
# Needs latexdiff, latexmk and the TeX packages the manuscript uses. CI runs it
# on every pull request and attaches the PDF next to main.pdf.
set -euo pipefail

base_ref="${1:-origin/main}"
root="$(git rev-parse --show-toplevel)"
tmp="$(mktemp -d)"
git -C "$root" worktree add -q --detach "$tmp/base" "$base_ref"
cleanup() { git -C "$root" worktree remove --force "$tmp/base" >/dev/null 2>&1 || true; rm -rf "$tmp"; }
trap cleanup EXIT

cd "$root/paper"
# --flatten follows \input in both versions; figures, TikZ pictures and
# algorithm bodies are compared as a whole instead of marked word by word,
# which latexdiff cannot do safely; changed equations are marked as a whole.
rm -f build/changes.pdf
latexdiff --encoding=utf8 --flatten --type=UNDERLINE --math-markup=whole \
  --config="PICTUREENV=(?:picture|DIFnomarkup|tikzpicture|algorithmic|adjustbox)[\w\d*@]*" \
  "$tmp/base/paper/main.tex" main.tex > changes.tex \
  2> >(grep -v "^Wide character" >&2)

# Not -Werror: the marked-up copy is a review aid, not the manuscript.
latexmk -interaction=nonstopmode -f changes.tex >/dev/null || true
rm -f changes.tex   # generated; keep the workspace check from reading it
if [ -f build/changes.pdf ]; then
  echo "Built paper/build/changes.pdf (changes since $base_ref)"
else
  echo "latexdiff output did not compile; see paper/build/changes.log" >&2
  exit 1
fi
