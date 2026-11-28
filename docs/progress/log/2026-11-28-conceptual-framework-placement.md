# Conceptual framework moved after the background; submission date November 2026 (2026-11-28)

Requested by Mark Limpahan.

- Placement: the CITC "Formatting and Submission Guidelines" (2nd ed., as encoded in `paper/citc.sty`) cover layout and numbering only and set no order for Chapter 1 sections; the guideline document itself is not in the repository or the team Drive search. Following the traditional thesis order, the Conceptual Framework (Figure 1.1) is now Section 1.2, between the Background and the Problem Statement. Later Chapter 1 sections renumber (Problem Statement 1.3 ... Definition of Terms 1.7).
- `paper/frontmatter.tex`: `\citcSubmissionDate` set to NOVEMBER 2026 (was OCTOBER 2026).
- Checks: `latexmk -pdf -Werror` clean; Figure 1.1 placement checked visually; revision matrix rebuilt (page numbers updated from `build/main.aux`).

- Same day: the unidentified "L. Zhang et al. (2025)" citation in the Background was dropped at the team's request; the sentence remains supported by Lu et al. (2026) and Wang et al. (2024).
- Same day: Chapters 4 and 5 (placeholders) left out of `main.tex` for the proposal manuscript; the chapter files are kept for later.
