# 2026-10-25: Adviser revisions to Chapters 2 and 3

Task: PAPER-011. Request (Mark Vincent D. Limpahan): apply the adviser's comments (simplify the writing; a figure that highlights the contribution; a figure of the whole methodology; define abbreviations before use; define every cited formula in Chapter 3) and the team's change list.

## Changes

- Chapter 2 rewritten to the adviser's outline: flow-matching TTS and its cost; methods for reducing sampling cost; teacher-guided sampling and re-noising; input-adaptive computation; statistical risk control; evaluation of zero-shot TTS; assistive applications; synthesis and research gap. Only formulas used in or closely related to Chapter 3 remain (Euler step, endpoint estimate, sway map, teacher-guided cost, re-noising, zero-failure bound, LTT test, WER, SIM, RTF). The synthesis table is kept; draft notes and figure placeholders are removed.
- Chapter 3: new Figure 3.1 (contribution: released DMOSpeech 2 versus the added per-utterance decision, signals and calibrated threshold). The old objective-aligned workflow figure and Table 3.2 are replaced by one 15-step methodology flowchart with bands for the common setup and Objectives 1-4, plus a short list that walks through it.
- Chapter 3 now defines the Euler step, the endpoint estimate, WER, SIM and RTF itself, each with a symbol table and a List of Equations entry; it no longer points to Chapter 2 equations.
- Objective 4's stabilization-signal subsection is cut to one paragraph; the four-step investigation plan is now Appendix D, "Plan for Further Investigation of the Stabilization Signal".
- Abbreviations: ASD, SNR, GPU (Chapter 1), ROC and CPU (Appendix D) are now written out at first use; the List of Abbreviations gains ACT, ASR, CALM, CPU, DMD, GPU, MOS, ODE and ROC.

## Checks run

- `latexmk -Werror` in paper/: 64 pages, no undefined references, no overfull boxes.
- `scripts/build_changes_pdf.sh origin/main`: 72-page changes.pdf.
- Rendered and inspected the pages with Figures 3.1 and 3.3.
- `python scripts/check_workspace.py --base origin/main` and `python -m unittest discover -s code/tests` (14 tests) pass.

## Open

- A full plain-language pass over the rest of Chapter 3 (Objectives 2 and 3) is still to be done; only the touched paragraphs were simplified.
- latexdiff shows figures as whole blocks; check the two new figures in main.pdf.

## Follow-up (same day)

Request: no symbol definitions in Chapter 2, but every Chapter 2 formula defined or cited in Chapter 3; sources for Chapter 2 figures and equations; a shorter Figure 3.3, as a longtable if possible; Chapter 3 aligned with Figure 3.3; tense rules.

- Chapter 2: the ten where-tables are removed. Each equation's introducing sentence now cites its source and names the Chapter 3 equation that defines it (Euler step, endpoint estimate, sway grid, cost identity, re-noising, zero-failure bound, LTT guarantee, WER, SIM, RTF). Both figure captions now say they were made by the researchers and from which sources.
- Chapter 3: the zero-failure sample-size bound is now defined in Section 3.6.3 (new equation). Figure 3.3 is a compact flowchart, one shaded row per stage, about a third of a page. A new longtable (Table 3.2) gives each step's details, data and Chapter 3 section, replacing the bullet list. Each objective section now names its steps in Figure 3.3. The cost-matched random rule mentioned in the stabilization-signal test is added to the compared-methods table.
- Tense: the Chapter 2 statements about this study now use the future tense; past tense for prior studies. In Chapter 3 the architecture description and several remaining present-tense actions are now future; facts about the released code and mathematical statements stay present. Chapter 1 had no future tense and was left as is.
- Checks: latexmk -Werror (62 pages, clean), changes.pdf against main (73 pages), workspace check and unit tests pass.
