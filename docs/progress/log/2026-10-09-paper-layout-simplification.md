# 2026-10-09: Paper layout simplification

Task: PAPER-006. User request: find the simplest Overleaf-like setup for the paper and clean the paper folder, preparing it to become a separate repository.

Flattened the final manuscript into the standard Overleaf layout: paper/main.tex (metadata folded in from metadata.tex), paper/chapters/, paper/references.bib, paper/figures/. Drafts stay standalone in paper/drafts/. Added paper/.latexmkrc so plain latexmk builds every paper into paper/build/, paper/.gitignore for a future standalone repository, and VS Code LaTeX Workshop settings (build on save, PDF preview, SyncTeX). Chapters declare their root with % !TEX root. Replaced scripts/build_paper.py in CI with latexmk -Werror, which fails on undefined citations/references. Updated the workspace checker, tests, and documentation paths. No manuscript text or claims were changed.

Validation:

- latexmk -Werror in paper/: exit 0; built main.pdf, tanh-velocity-blending.pdf, and adaptive-handover.pdf.
- python -m unittest discover -s code/tests: 13 tests passed.
- python scripts/check_workspace.py: passes after the superseded paper/final/ folder is removed.

Remaining: delete the superseded paper/final/, paper/drafts/README.md, old build folders, and scripts/build_paper.py; review rendered pages; CI has not run remotely.
