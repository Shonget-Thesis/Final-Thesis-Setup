# 2026-10-25: Animated proposal-defense walkthrough

Task: PRES-001. User request (Mark Vincent D. Limpahan): an animated walkthrough of Chapters 1–3 for the proposal defense, with a slide version, a presenter script and citation footers, styled after endurosat.com and animated in the manner of LEMMiNO's videos; then add the HTML to the repository as a presentation folder.

## Changes

- `presentation/index.html`: one self-contained page with 19 scenes (title, outline, six Chapter I scenes, five Chapter II scenes, six Chapter III scenes, summary). Play mode auto-advances on timed scenes (about 11 minutes); arrow keys or clicks step through it as slides; a script panel and a full-script view carry the narration. Every content scene has a source footer citing the works its claims rest on, using the proposal's reference list.
- `presentation/SCRIPT.md`: the same narration as a readable document, one section per scene with its auto-play duration.
- `presentation/README.md`: how to open and present it, the Figma copy, and the provenance and limits below.

## Checks run

- Opened `presentation/index.html` from `file://` in headless Chromium: standards mode, 19 scenes, no script errors; arrow-key stepping, the hidden script panel and the full-script overlay behave as intended.
- During authoring, scenes were rendered and inspected at 1600×900 for overlap and clipping; the Figma copy was built from the same rendered layout.
- `python scripts/check_workspace.py` and `python -m unittest discover -s code/tests` (results in the pull request description).

## Open

- The slides follow the proposal PDF from before the October 25 audit and the EXP-001 pilot. They do not show the audit's λ̂ = 1.20 on the T4 grid, the stabilization score's lack of discrimination on clean development prompts, or the SNR gate missing 18–22 dB noise; the calibration example (λ̂ = 0.50) is Appendix A's illustration and is labeled as such. Scenes 13–18 need an update if the manuscript changes.
- Changes on the unmerged `paper/adviser-revisions` branch are not reflected.
- The team has not reviewed the content.
