# Current status

Updated: 2026-10-24 (Asia/Manila). Phase: workspace setup and reproducibility reconciliation.

The repository now separates implementation, the LaTeX manuscript, research evidence, and project documentation. The portable timing replay and automated bookkeeping checks pass, with all 14 current tests passing. Historical evidence is development-only; no new synthesis or held-out evaluation has been performed.

paper/ holds one LaTeX project: paper/main.tex in the USTP CITC style (citc.sty, APA 7 via biblatex/biber), with the team's thesis proposal (Chapters 1-3 and Appendices A-D, "A Risk-Calibrated Adaptive Handover Method for Teacher-Guided Flow-Matching Text-to-Speech Inference") in paper/chapters/. A proposal switch hides the final-thesis pages and the placeholder Chapters 4-5. It is a working draft, not adviser-approved. See the [CITC format log](log/2026-10-24-citc-format.md) and the [proposal draft log](log/2026-10-24-proposal-draft.md).

All title-page and approval fields (authors, date, adviser, panel, dean) and the keywords are filled in at the top of paper/frontmatter.tex; the acknowledgement, dedication and abstract are still placeholders. The earlier standalone drafts were removed at the team's request and remain in git history (ADR-004). See the [single-manuscript log](log/2026-10-24-single-manuscript.md).

## Current priorities

1. Review the completed workspace setup and its recorded checks.
2. Visually review the compiled papers and address remaining layout warnings.
3. Assign task owners and confirm author/adviser/institution details.
4. Recover generation provenance and reconcile schedule/time/cost accounting.
5. Freeze the evaluation protocol before collecting held-out results.

## Blockers and limits

- MiKTeX and Strawberry Perl are installed; the manuscript compiles. Rendered-page review is pending. See the [installation record](log/2026-10-09-latex-installation.md).
- The source snapshot excludes audio, checkpoints, and datasets; inference has not been smoke-tested.
- Existing hybrid telemetry is not independent scheduler-runtime evidence.
- Adviser approval and primary-literature verification are not recorded. Author and signatory names were supplied by a team member, not confirmed by the panel. The CITC template was adopted at a team member's request; confirmation that it is the required institutional format is not recorded.
- CI has not run remotely yet.

See the [task board](TASKS.md) and [setup work log](log/2026-10-09-workspace-setup.md). Dates for team reviews and experiment delivery remain unassigned.
