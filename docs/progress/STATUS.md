# Current status

Updated: 2026-10-25 (Asia/Manila). Phase: workspace setup and reproducibility reconciliation.

The repository now separates implementation, the LaTeX manuscript, research evidence, and project documentation. The portable timing replay and automated bookkeeping checks pass, with all 14 current tests passing. Synthesis has been run on Colab (Tesla T4) for all 500 split speakers (research/evidence/benchmarks/), and the reports in research/reports/ include calibration and test-split results. The test split was inspected before the protocol was fixed, so all existing results are development evidence; no independent held-out evaluation exists. See the [October 25 audit](../../research/audits/2026-10-25/FINDINGS.md).

The thesis now uses Colab T4 results only. The verified T4 runs (grid, timing, signal selection, novel algorithms, drift features) are in research/evidence/benchmarks/ with source commits and SHA-256 hashes ([manifest](../../research/evidence/benchmarks/MANIFEST.json)); CPU runs and the comprehensive evaluation suite were not imported. The follow-up tests (seed-noise floor, the 166 unused speakers, degraded prompts) are designed in [EXP-001](../../research/experiments/EXP-001-followup-t4/README.md) with a Colab notebook and runner, not yet run (EVAL-003). See the [import and follow-up log](log/2026-10-25-t4-import-and-followup.md).

An edit basing Appendices A–C on the T4 development data was made and then reverted at the team's request; the manuscript's chapters are identical to main. See the [appendices log](log/2026-10-25-appendices-real-data.md).

The EXP-001 development pilot ran on the T4 (2026-10-25): the pre-experiment checks pass and the T4 grid reproduces exactly; the 20 dB SNR gate misses 18–22 dB white noise; drift helps as a low-drift noise safeguard, not as a stopping signal. Results are in research/experiments/EXP-001-followup-t4/results/ and Part 6 of the compiled reports; see the [pilot log](log/2026-10-25-pilot-results.md). An early freeze in that session is void.

The October 25 audit recomputed the T4 grid: with the manuscript's λ grid (0.20–1.20), Learn-Then-Test on the calibration split certifies every threshold (λ̂ = 1.20, not the reports' CPU-label 0.50), and the policy then equals SNR-only on the test split (4/200 speaker failures, 17.04 NFE). The stabilization score shows no discrimination on clean development prompts (AUC 0.474, three failures). Each report now carries a dated audit note; its original text is unchanged.

paper/ holds one LaTeX project: paper/main.tex in the USTP CITC style (citc.sty, APA 7 via biblatex/biber), with the team's thesis proposal (Chapters 1-3 and Appendices A-D, "A Risk-Calibrated Adaptive Handover Method for Teacher-Guided Flow-Matching Text-to-Speech Inference") in paper/chapters/. A proposal switch hides the final-thesis pages and the placeholder Chapters 4-5. It is a working draft, not adviser-approved. See the [CITC format log](log/2026-10-24-citc-format.md) and the [proposal draft log](log/2026-10-24-proposal-draft.md).

All title-page and approval fields (authors, date, adviser, panel, dean) and the keywords are filled in at the top of paper/frontmatter.tex; the acknowledgement, dedication and abstract are still placeholders. The earlier standalone drafts were removed at the team's request and remain in git history (ADR-004). See the [single-manuscript log](log/2026-10-24-single-manuscript.md).

The endpoint-stabilization score is framed as a candidate signal still to be tested against the prompt SNR, with a post-proposal investigation plan (summarized in Objective 4, detailed in Appendix D). All synthesis, scoring and timing run on an NVIDIA Tesla T4 GPU through Google Colab (the Ryzen laptop is no longer used). The restart level stays fixed. See the [drift framing log](log/2026-10-25-drift-framing.md).

Every pull request now gets two PDF artifacts: main.pdf and a changes PDF with the pull request's edits marked (latexdiff against the base branch). See the [changes PDF log](log/2026-10-25-changes-pdf.md) and the paper guide's review section.

The adviser's comments are applied on branch paper/adviser-revisions: Chapter 2 follows the adviser's outline with only Chapter 3-related formulas, Chapter 3 has a contribution figure and a step-by-step methodology flowchart, defines every formula it cites, and abbreviations are written out at first use. See the [adviser revisions log](log/2026-10-25-adviser-revisions.md).

Follow-up on the adviser revisions: Chapter 2 formulas and figures now cite their sources and point to their Chapter 3 definitions, Figure 3.3 is a compact flowchart with a step table (Table 3.2), and Chapters 2 and 3 follow the proposal tense rules. See the follow-up section of the [adviser revisions log](log/2026-10-25-adviser-revisions.md).

The active research implementation from the sibling working tree has been migrated into the workspace: `code/scheduling/` holds the deterministic adaptive handover sampler and metrics, `code/colab/` holds the turnkey Google Colab GPU-accelerated pipelines and notebooks (focusing entirely on Colab GPU inference and excluding local CPU-only sweeps), `research/protocols/splits/` holds the 500-speaker cohort manifests, and `research/reports/` holds the comprehensive research reports. Checkpoint downloads with verified SHA-256 hashes are automated via `code/scheduling/download_ckpts.py`. See the [implementation migration log](log/2026-10-25-implementation-migration.md).

A proposal-defense walkthrough is in presentation/: an animated 19-scene page covering Chapters 1–3 with a slide mode, citation footers and a presenter script (PRES-001, awaiting team review). It follows the proposal PDF from before the October 25 audit, so its Chapter III scenes do not yet show the audit and pilot findings above. See the [walkthrough log](log/2026-10-25-defense-walkthrough.md).

## Current priorities

1. Review the completed workspace setup and its recorded checks.
2. Visually review the compiled papers and address remaining layout warnings.
3. Assign task owners and confirm author/adviser/institution details.
4. Recover generation provenance and reconcile schedule/time/cost accounting.
5. Freeze the evaluation protocol before collecting held-out results.

## Blockers and limits

- MiKTeX and Strawberry Perl are installed; the manuscript compiles. Rendered-page review is pending. See the [installation record](log/2026-10-09-latex-installation.md).
- The source snapshot excludes audio, checkpoints, and datasets. Inference runs on Colab from a clone of Shonget-DMOSpeech2, not from this workspace's code/ copy; the clone commit is not recorded for the T4 grid and timing runs.
- The CPU grids behind the reports' λ̂ = 0.50 and their scripts were not migrated.
- A fresh held-out test needs the 166 prompt speakers in no split, after the protocol is frozen (EVAL-001).
- Existing hybrid telemetry is not independent scheduler-runtime evidence.
- Adviser approval and primary-literature verification are not recorded. Author and signatory names were supplied by a team member, not confirmed by the panel. The CITC template was adopted at a team member's request; confirmation that it is the required institutional format is not recorded.
- CI has not run remotely yet.

See the [task board](TASKS.md) and [setup work log](log/2026-10-09-workspace-setup.md). Dates for team reviews and experiment delivery remain unassigned.
