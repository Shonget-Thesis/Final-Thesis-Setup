# 2026-10-25: Marked-changes PDF on every pull request

Task: INFRA-003. User request (Mark Vincent D. Limpahan): let reviewers preview a pull request before merging, with a PDF that shows exactly what a member changed.

## Changes

- `scripts/build_changes_pdf.sh <base>`: checks out the base revision in a temporary git worktree, runs latexdiff (`--flatten` to follow `\input`, `--math-markup=whole`, TikZ pictures, algorithm bodies and resized boxes compared as a whole) on paper/main.tex, and compiles `paper/build/changes.pdf` without `-Werror`. Removed text prints red and struck through; new text prints blue and underlined. The generated changes.tex is deleted after the build (and ignored by git) so the workspace check never reads it.
- `.github/workflows/thesis-checks.yml` (paper job): full-history checkout; installs latexdiff and texlive-plain-generic (ulem, for the strikethrough); main.pdf is uploaded as **thesis-papers**; on pull requests the changes PDF is built against the base commit and uploaded as **changes-vs-<base branch>**. That step is `continue-on-error`, so a latexdiff failure never blocks a merge. Every run writes a short summary telling reviewers which artifact is which.
- `paper/README.md`: new "Reviewing a pull request" section with the download steps, the local command and the suggested routine (one branch and draft PR per change; merge after green checks and one approval).

## Checks run

- `scripts/build_changes_pdf.sh origin/main` on this branch: 69-page changes.pdf, no LaTeX errors, outline fonts only; rendered the Objectives page and the Decision Signals page and inspected the markup; curly quotes and dashes survive (`--encoding=utf8`).
- Workflow YAML parses; `python scripts/check_workspace.py --base origin/main` passes.

## Open

- First remote run is the CI on pull request #4.
- latexdiff cannot mark changes inside figures, TikZ code or algorithm bodies; reviewers should check those in main.pdf and the file diff.
- Requiring green checks and an approval before merging is a branch-protection setting only a repository admin can enable.

## Follow-up: build on every branch

The first remote runs showed the PDF only on the `pull_request` run; each push to a PR branch also starts a `push` run, which is what the Checks tab may open first. The changes PDF is now built on every run except pushes to main, against the pull request's base commit or, for push runs, `origin/main`, and the artifact is always named **changes-vs-main**.
