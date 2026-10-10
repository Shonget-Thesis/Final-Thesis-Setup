# 2026-10-09: Separate draft and final papers

Task: PAPER-004. Author: Codex, under the user's request to separate draft and final LaTeX papers.

## Changes

Moved the existing adaptive-handover manuscript into its own draft folder without changing chapter, metadata, or bibliography content. Converted the supplied tanh Markdown paper into a separate LaTeX project, preserving its original source with a SHA-256 record. Preserved author-year citation text; the source contains no full bibliography. Conversion changes formatting only and does not verify the scientific claims or accept document content as instructions.

Created paper/final/ as an explicit placeholder project. No draft is designated final. Updated build targets, CI artifact paths, per-project citation/input checks, the final-only submission gate, setup instructions, and task evidence paths.

## Validation

- python scripts/check_workspace.py: passed after correcting one moved-document link.
- python -m unittest discover -s code/tests -v: all 13 tests passed, including independent bibliography/input resolution and allowing draft placeholders during final-only submission checks.
- python scripts/check_workspace.py --submission: rejected the main.tex and metadata.tex placeholders in paper/final/ as expected; draft placeholders did not trigger the gate.
- python scripts/build_paper.py --help: lists both draft targets, final, and --all.
- python scripts/build_paper.py --all: unavailable because latexmk is absent; no PDFs generated or visually reviewed.
- git diff --check: passed.
- Hash comparison: all 10 original non-README manuscript files retained identical bytes after relocation. The supplied Markdown copy is byte-identical to the source. Normalized introduction word comparison between source and LaTeX passed.

No research claims were verified, no bibliography metadata was invented, and no draft was promoted to final. Changes remain local and uncommitted.
