# Paper

A self-contained LaTeX project in the standard Overleaf layout. It can be uploaded to Overleaf as-is or split into its own repository.

| Path | Contents |
| --- | --- |
| [main.tex](main.tex) | Final manuscript: preamble, title/author metadata, chapter list |
| [chapters/](chapters/) | One file per final chapter (placeholders until reviewed content is promoted) |
| [references.bib](references.bib) | Verified BibTeX entries for the final manuscript |
| figures/ | Images for the final manuscript (`\includegraphics{name}`) |
| [drafts/](drafts/) | Standalone single-file drafts, each with its own inline metadata and bibliography |
| [sources/](sources/tanh-velocity-blending/original.md) | Supplied original Markdown and its hash record (do not edit) |

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
