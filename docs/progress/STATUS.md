# Current status

Updated: 2026-10-09 (Asia/Manila). Phase: workspace setup and reproducibility reconciliation.

The repository now separates implementation, the LaTeX manuscript, research evidence, and project documentation. The portable timing replay and automated bookkeeping checks pass, with all 13 current tests passing. Historical evidence is development-only; no new synthesis or held-out evaluation has been performed.

The supplied tanh paper is now a standalone LaTeX draft in paper/drafts/tanh-velocity-blending.tex. The earlier adaptive-handover scaffold is preserved in paper/drafts/adaptive-handover.tex. paper/main.tex is reserved for the selected submission manuscript and loads separate placeholder chapter files from paper/chapters/. paper/ is a self-contained, Overleaf-compatible project built with latexmk or VS Code LaTeX Workshop. See the [paper organization log](log/2026-10-09-paper-organization.md) and the [paper layout simplification log](log/2026-10-09-paper-layout-simplification.md).

## Current priorities

1. Review the completed workspace setup and its recorded checks.
2. Visually review the compiled papers and address remaining layout warnings.
3. Assign task owners and confirm author/adviser/institution details.
4. Recover generation provenance and reconcile schedule/time/cost accounting.
5. Freeze the evaluation protocol before collecting held-out results.

## Blockers and limits

- MiKTeX and Strawberry Perl are installed; all three papers compile. Rendered-page review is pending. See the [installation record](log/2026-10-09-latex-installation.md).
- The source snapshot excludes audio, checkpoints, and datasets; inference has not been smoke-tested.
- Existing hybrid telemetry is not independent scheduler-runtime evidence.
- Adviser approval, institutional template, final title, and primary-literature verification are not recorded.
- CI is configured locally but has not run remotely; changes have not been committed or pushed.

See the [task board](TASKS.md) and [setup work log](log/2026-10-09-workspace-setup.md). Dates for team reviews and experiment delivery remain unassigned.
