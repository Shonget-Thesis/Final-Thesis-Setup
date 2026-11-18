# 2026-11-18: Build fixes, and committing the working-tree Chapter 3 rewrite

## Build fixes (Claude Code, at Mark Limpahan's request)

- `paper/citc.sty`: `\ttdefault` = `lmtt`, `\sfdefault` = `lmss` after `mathpazo`. Before this, `\texttt`/`\textsf` fell back to T1 Computer Modern, which MiKTeX embedded as bitmap Type 3 fonts (`ectt*`, `ecss*`, `ecit*` `.pk` files) because cm-super is not installed. Body text stays Palatino (CITC 1.4 unchanged).
- `paper/chapters/02-related-work.tex`, Table `tab:synthesis`: Domain column 1.7 cm -> 1.8 cm, Objective column 2.8 cm -> 2.7 cm (total width unchanged). This removed the only overfull box (2.4 pt, "Language; image" cell).
- Not changed: the three `ignored error: Infinite glue shrinkage found in box being split` lines in the log. They come from `longtable` page breaks, which the LaTeX kernel deliberately ignores. They are harmless.
- Not changed: LaTeX Workshop's `autoBuild.run: onSave`. Running it at the same time as a terminal `latexmk` into the same `build/` caused the earlier stale `main.bcf` problem.

## Chapter 3 rewrite (`PAPER-020`, from another session)

The working tree also contained a large rewrite of `paper/chapters/03-methodology.tex` (about 960 changed lines), plus changes to `paper/figures/fig_method.tex` and `paper/figures/fig_workflow.tex`. The files were saved about 19:51–19:56 local time.

The rewrite moves the method toward the SNR plus invariant-residual handover of Appendix E (`EXP-034`). It adds:
- the condition model and invariance ladder;
- the quantile payload (`def:quantile`, `prop:invariance`);
- the invariant residual (`def:residual`, `prop:null`, `prop:w2`);
- the excess-rate formulation (`prop:rate`);
- new policy properties (`prop:zero`, `prop:mono`).

Mark Limpahan asked for it to be committed and merged together with the fixes above ("Everything").

- **Approval:** this is not adviser approval. EXP-034 remains development evidence on development speakers only, with no held-out validation and no listening tests.
- **Task record:** the rewrite's own logs (`2026-11-18-chapter3-v2-router-draft.md`, `2026-11-18-chapter3-router-replaces-scheduler.md`) cite `PAPER-020`, but it was missing from `tasks.json`; it is now registered there as `review`. The draft log's links to the removed `-v2` files were turned into plain file names so the workspace check passes.
- **Consistency:** Appendix E text and earlier logs still say "Chapter 3 remains unchanged". They need to be reconciled with the rewrite.
- **Possibly stale text:** the Appendix C prototype paragraph and Table C.1 describe the earlier drift rule ($\lambda=0.35$) and may need updating.

## Checks run

- `latexmk -g -Werror -interaction=nonstopmode main.tex` in `paper/`: exit 0, 106 pages, 0 overfull boxes, 0 undefined references. `pdffonts build/main.pdf` shows 0 Type 3 fonts.
- `latexmk -g -Werror revision-matrix.tex`: exit 0.
- `python scripts/check_workspace.py`: PASSED, after `PAPER-020` was registered and the stale `-v2` links were fixed.
- `python -m unittest discover -s code/tests`: 434 tests OK, run earlier today on the same `code/` tree. This commit does not change `code/`.
