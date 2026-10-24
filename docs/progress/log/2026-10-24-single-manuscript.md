# 2026-10-24: Single manuscript with fill-in front matter

Tasks: PAPER-009, PAPER-002. User: Mark Vincent D. Limpahan.

## Changes

- Front matter: every preliminary-page value is now in one section at the top of paper/frontmatter.tex (between the `=== FILL IN ===` lines): title, department, degree, submission date, authors, adviser, panel, dean, acknowledgement, dedication (with an on/off switch), acronyms, abstract and keywords. main.tex no longer defines these. An empty second panel member prints one centered member on the approval sheet.
- Values entered as supplied by the user: submission date OCTOBER 2026; authors Mark Vincent Daculos Limpahan, Roxanne Manticajon Locsin, Heart Galarido Chiong, Rhenel Jhon Rodriguez Sajol (in that order); adviser Marylene S. Eder, MIT; chair Maricel A. Esclamado, PhD; member Cheryll S. Pagal, MSAMS; dean Junar A. Landicho, PhD. Keywords chosen by Claude at the user's request from the proposal's content: text-to-speech, flow matching, knowledge distillation, adaptive inference, risk control.
- Drafts: the user deleted paper/drafts/; it is now removed with git rm, and latexmk (.latexmkrc), the CI step name, scripts/manuscripts.py and scripts/check_workspace.py build and check only paper/main.tex. Two draft-specific tests were replaced by one test for missing TeX inputs. AGENTS.md, WORKFLOW.md, SETUP.md, README.md and paper/README.md were updated, and ADR-004 records the decision. The drafts remain in git history at commit 2e51c49.
- Task records: PAPER-001 and PAPER-004 evidence no longer points at the deleted drafts; PAPER-002 moved to in_progress with the supplied metadata.
- Cleanup earlier the same day: the untracked superseded paper/final/, paper/drafts/README.md, stale draft PDFs and stale build intermediates in paper/ were deleted (paper/final/main.tex was making VS Code build the wrong root file).

## Checks run

- `python -m unittest discover -s code/tests`: 14 tests passed.
- `python scripts/check_workspace.py`: passed.
- `python scripts/check_workspace.py --submission`: fails only on frontmatter.tex (acknowledgement, dedication and abstract placeholders) and the placeholder Chapters 4-5, as expected.
- `latexmk -Werror` in paper/: exit 0, build/main.pdf only.
- A temporary copy with `\citcproposalfalse` built without errors; its title page, approval sheet and abstract page were rendered and inspected.

## Open

- Acknowledgement, dedication and abstract text.
- Whether the CITC format is the required institutional template; the signed approval form replaces the printed sheet before submission.
- Signatory names were supplied by a team member and are not panel-confirmed.
