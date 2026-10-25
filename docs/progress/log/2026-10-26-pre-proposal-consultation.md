# 2026-10-26: Manuscript ready for the pre-proposal consultation

The team reports the proposal manuscript is in its final form for the pre-proposal consultation. This is the team's statement, not adviser approval, and no approval is recorded.

## Changed

- Chapter 1 (paper/chapters/01-introduction.tex): added a theoretical framework (`fig:theoretical`, four bodies of theory, each supplying one part of the method) and a conceptual framework (`fig:conceptual`, input-process-output, with the process stage organized by the four specific objectives). New figure files: paper/figures/fig_theoretical.tex and paper/figures/fig_conceptual.tex. Other Chapter 1 text was also revised.
- Chapter 2: added the T-Stitch, Adaptive Guidance and adaptive ASR correction comparisons to the related-work table and synthesis, and revised surrounding text. paper/references.bib gained the matching entries (41 lines). The citations are the team's; they have not been re-verified against the primary sources by this log.
- Front matter: the acronyms page now prints in the proposal (it previously printed only in the final thesis), with dB, GB and ms added; the abstract stays out of the proposal.
- presentation/: index.html and README.md updated, and presentation/proposal-defense-slides.pdf added (19 image-only pages captured from index.html in headless Edge). The slides still follow the proposal text from before the October 25 audit and pilot, so scenes 13-18 do not show those findings (PRES-001).
- scripts/build_paper.py added and fixed: it imported `DEFAULT_TARGET` from scripts/manuscripts.py, which did not define it, so it crashed on import. manuscripts.py now defines `DEFAULT_TARGET = "final"`. The script writes to paper/build/final/; plain `latexmk` in paper/ still writes to paper/build/ and is what CI uses.

## Checks run

- python scripts/check_workspace.py: passed.
- python -m unittest discover -s code/tests: 32 tests, OK.
- python scripts/build_paper.py: exit 0, no unresolved citations or references, PDF produced at paper/build/final/main.pdf.
- latexmk -g -Werror -interaction=nonstopmode in paper/: exit 0, no undefined citations or `<empty citation>` in the log. Run before the last edits to the files above were committed, not repeated after the final commit.

## Remaining

- Rendered pages have not been reviewed (the log shows two "Infinite glue shrinkage" errors that pdfTeX skipped, on the pages around Figure `fig_workflow`; check them visually).
- Adviser approval, primary-literature verification of the new citations, and confirmation of the CITC format are still not recorded.
- The presentation does not reflect the October 25 audit or the EXP-001 pilot.
- No task in tasks.json covers this edit; owner and acceptance criterion are for the team to assign.
- A stuck terminal latexmk can lock paper/build/main.bcf and give an empty bibliography (`<empty citation>`); build with `-interaction=nonstopmode` and not alongside LaTeX Workshop's build on save.
