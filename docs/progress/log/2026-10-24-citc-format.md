# 2026-10-24: CITC manuscript format

Task: PAPER-007. User request: pull, then format the paper with https://github.com/hdmGOAT/citc_latex_style.

`git pull` on setup/workspace fetched origin/main (ee182ba, the merge of this branch); the working tree was already up to date.

Copied citc.sty, frontmatter.tex, and LICENSE (as paper/CITC-LICENSE) from hdmGOAT/citc_latex_style at b1b5bd3082d19145d1a4021072075efe3a244f3d. Upstream SHA-256: citc.sty fa84e847892f8bbb9b20c4c6d4b78c9b45b1aac562ecd585f66d0ec48d99918d (unchanged here), frontmatter.tex 39de80e35b7dacf8feec5ff017601aca8237f1a5b212acdbaa88ab6283f9e3af (its fill-in fields were moved into main.tex so the submission check sees them), LICENSE 7474ab356d3916f06cb41626d2b56e2c45178b998a9e83a3ab2d9dad071e50f4.

paper/main.tex now uses the article class with citc.sty, cleveref, csquotes, biblatex-apa/biber, and the CITC preliminary pages. Every title-page and approval field is TBD; no names, title, department, or degree were filled in. Chapter headings were renamed to the CITC preferred titles (INTRODUCTION, REVIEW OF LITERATURE, METHODOLOGY, RESULTS AND DISCUSSION, SUMMARY, CONCLUSION, AND RECOMMENDATIONS); chapter content is still placeholder text. Drafts are unchanged.

Two fixes live in main.tex rather than the vendored style: headheight 14.5pt with the top margin raised 0.5pt (fancyhdr overflow), and `\tocloftpagestyle{prelim}`, because tocloft stamps the contents and lists with the plain style, which citc.sty redefines as the main-body style, putting their roman numerals in the upper right.

The workspace check now requires citc.sty and frontmatter.tex, scans every top-level paper .tex file for the final target, recognises biblatex cite commands (including `\workcite{Name}{key}`), and treats `\placeholder{` as unfinished in submission mode. CI installs texlive-bibtex-extra and biber.

Validation:

- latexmk -Werror in paper/: exit 0; built main.pdf (15 pages, zero log warnings), tanh-velocity-blending.pdf, and adaptive-handover.pdf.
- Rendered main.pdf pages at low resolution and inspected them: title, approval, acknowledgement, dedication, contents, lists, acronyms, and abstract pages with roman numerals bottom center (title and approval unnumbered); Chapter I-V headings centered with the chapter-opening page number suppressed.
- python -m unittest discover -s code/tests: 15 tests passed (2 new).
- python scripts/check_workspace.py: fails only on the untracked, superseded paper/final/README.md link already noted in the 2026-10-09 layout log.
- python scripts/check_workspace.py --submission: rejects main.tex, frontmatter.tex, and all chapters as unfinished, as expected.

Remaining: confirm metadata (PAPER-002) and whether USTP CITC formatting is officially required; replace the approval page with the signed ISO form; full-resolution page review against the guidelines; CI has not run remotely.
