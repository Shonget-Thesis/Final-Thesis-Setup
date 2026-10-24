# Paper

A self-contained LaTeX project in the standard Overleaf layout. It can be uploaded to Overleaf as-is or split into its own repository.

| Path | Contents |
| --- | --- |
| [main.tex](main.tex) | Final manuscript in the USTP CITC format: preamble, title-page/approval metadata, chapter list |
| [citc.sty](citc.sty), [frontmatter.tex](frontmatter.tex) | CITC style and preliminary pages, from [hdmGOAT/citc_latex_style](https://github.com/hdmGOAT/citc_latex_style) at b1b5bd3 (MIT, [CITC-LICENSE](CITC-LICENSE)) |
| [chapters/](chapters/) | One file per chapter: the proposal's Chapters 1-3 and appendices; Chapters 4-5 are placeholders for the final thesis |
| [references.bib](references.bib) | Verified BibTeX entries for the final manuscript |
| figures/ | Images for the final manuscript (`\includegraphics{name}`) |
| [drafts/](drafts/) | Standalone single-file drafts, each with its own inline metadata and bibliography |
| [sources/](sources/tanh-velocity-blending/original.md) | Supplied original Markdown and its hash record (do not edit) |

## CITC format

main.tex follows the CITC manuscript guidelines through citc.sty: a `\section` is a chapter (`Chapter I` over the uppercase title, new page), `\subsection` is 1.1., `\subsubsection` is 1.1.1., and `\phead{...}` is a run-in paragraph heading. Tables, figures, and equations are numbered per chapter; tag an equation with `\eqentry{name}` to list it. Cite with `\parencite`/`\textcite` (APA 7 via biblatex and biber). See the upstream README for every construct.

Fill the `TBD` fields in main.tex and the `\placeholder{...}` pages (acknowledgement, dedication, abstract) in frontmatter.tex; the submission check rejects both. `\citcproposaltrue` in main.tex (the default) builds the proposal: title page, contents and lists, Chapters 1-3, references and appendices. Set it to false for the final thesis. The approval page must be replaced with the signed, scanned ISO form FM-USTP-ACAD-11 before submission. Draft notes (`\todo`, `\citationneeded`, `\unverified`) are hidden by `\draftnotesfalse`. Drafts keep their own formatting.

## Editing (Overleaf-style, in VS Code)

1. Install the recommended LaTeX Workshop extension (VS Code prompts for it).
2. Open any .tex file and save: it rebuilds and the PDF preview refreshes in a side tab.
3. Ctrl+click in the PDF jumps to the source line; Ctrl+Alt+J jumps from source to PDF.
4. Real-time co-editing: Live Share. Async collaboration: Git branches and pull requests.

Chapter files carry `% !TEX root = ../main.tex`, so saving a chapter builds the whole manuscript.

## Command line and CI

~~~powershell
cd paper
latexmk          # builds main.tex and every draft into paper/build/
latexmk main.tex # one target only
latexmk -c       # remove intermediates
~~~

CI runs `latexmk -Werror`, which fails on undefined citations or references, and uploads the PDFs.

## Overleaf

Upload this folder as a zip (New Project → Upload Project). main.tex is detected as the main document; to compile a draft, change Menu → Main document. Overleaf's GitHub sync requires a paid or institutional plan; without it, download the source zip and commit it here.

## Rules

- Drafts stay single-file: no `\input`, `\include`, or external bibliography.
- When promoting a draft into the final chapters, record its source draft and revision in the progress log.
- Run python scripts/check_workspace.py --submission before submission; current placeholders intentionally fail it.
