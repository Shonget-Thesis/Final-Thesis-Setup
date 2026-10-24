# Paper

A self-contained LaTeX project in the standard Overleaf layout. It can be uploaded to Overleaf as-is or split into its own repository.

| Path | Contents |
| --- | --- |
| [main.tex](main.tex) | Manuscript in the USTP CITC format: preamble, proposal/final switch, chapter list |
| [citc.sty](citc.sty), [frontmatter.tex](frontmatter.tex) | CITC style and preliminary pages, from [hdmGOAT/citc_latex_style](https://github.com/hdmGOAT/citc_latex_style) at b1b5bd3 (MIT, [CITC-LICENSE](CITC-LICENSE)) |
| [chapters/](chapters/) | One file per chapter: the proposal's Chapters 1-3 and appendices; Chapters 4-5 are placeholders for the final thesis |
| [references.bib](references.bib) | Verified BibTeX entries for the final manuscript |
| figures/ | Images for the final manuscript (`\includegraphics{name}`) |
| [sources/](sources/tanh-velocity-blending/original.md) | Supplied original Markdown and its hash record (do not edit) |

## CITC format

main.tex follows the CITC manuscript guidelines through citc.sty: a `\section` is a chapter (`Chapter I` over the uppercase title, new page), `\subsection` is 1.1., `\subsubsection` is 1.1.1., and `\phead{...}` is a run-in paragraph heading. Tables, figures, and equations are numbered per chapter; tag an equation with `\eqentry{name}` to list it. Cite with `\parencite`/`\textcite` (APA 7 via biblatex and biber). See the upstream README for every construct.

Fill in the section between the `=== FILL IN ===` lines at the top of frontmatter.tex: title, authors, date, adviser, panel, dean, acknowledgement, dedication, acronyms, abstract and keywords. Anything still `TBD` or `\placeholder{...}` is rejected by the submission check. `\citcproposaltrue` in main.tex (the default) builds the proposal: title page, contents and lists, Chapters 1-3, references and appendices. Set it to false for the final thesis. The approval page must be replaced with the signed, scanned ISO form FM-USTP-ACAD-11 before submission. Draft notes (`\todo`, `\citationneeded`, `\unverified`) are hidden by `\draftnotesfalse`.

## Editing (Overleaf-style, in VS Code)

1. Install the recommended LaTeX Workshop extension (VS Code prompts for it).
2. Open any .tex file and save: it rebuilds and the PDF preview refreshes in a side tab.
3. Ctrl+click in the PDF jumps to the source line; Ctrl+Alt+J jumps from source to PDF.
4. Real-time co-editing: Live Share. Async collaboration: Git branches and pull requests.

Chapter files carry `% !TEX root = ../main.tex`, so saving a chapter builds the whole manuscript.

## Command line and CI

~~~powershell
cd paper
latexmk          # builds main.tex into paper/build/
latexmk -c       # remove intermediates
~~~

CI runs `latexmk -Werror`, which fails on undefined citations or references, and uploads the PDF.

## Reviewing a pull request

Every pull request builds two PDFs. Open the pull request, go to **Checks → Thesis checks**, and download them from **Artifacts** at the bottom of the run's summary page:

- **thesis-papers** (`main.pdf`): the manuscript as it would look after merging.
- **changes-vs-main** (`changes.pdf`): the same manuscript with the pull request's changes marked: removed text red and struck through, new text blue and underlined. Changed equations are marked as a whole, and figures, TikZ pictures and algorithm bodies are not marked inside.

To build the changes PDF on your own computer (needs latexdiff, and `ulem` from texlive-plain-generic on Linux):

~~~powershell
scripts/build_changes_pdf.sh             # changes since origin/main
scripts/build_changes_pdf.sh my-branch   # changes since any branch or commit
~~~

Suggested routine: one branch and pull request per change, opened as a draft while writing; merge when the checks are green and one groupmate has read both PDFs and approved.

## Overleaf

Upload this folder as a zip (New Project → Upload Project). main.tex is detected as the main document. Overleaf's GitHub sync requires a paid or institutional plan; without it, download the source zip and commit it here.

## Rules

- Earlier standalone drafts were removed on 2026-10-24 and live in git history (commit 2e51c49 and earlier; see ADR-004). When reusing their text, record the source commit in the progress log.
- Run python scripts/check_workspace.py --submission before submission; current placeholders intentionally fail it.
