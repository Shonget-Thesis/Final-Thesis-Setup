# Proposal defense walkthrough

An animated, full-screen presentation of the thesis proposal (Chapters 1–3) for the proposal defense, styled with a starry cosmic field, monochrome gradients, blue shade gradients (`#38BDF8` to `#0070E0`) for the teacher, grey for the student, and warm orange shade gradients (`#FFA04D` to `#D84300`) for what this study adds, with LEMMiNO-style citation footers on every content scene.

| File | What it is |
| --- | --- |
| [index.html](index.html) | The walkthrough: 19 scenes, one HTML file, full-screen layout |
| [SCRIPT.md](SCRIPT.md) | The presenter script, one section per scene, about 11 minutes at the auto-play pace |

## Opening it

Open `index.html` in a current Chrome, Edge or Firefox (double-click works; no server needed). It loads Google Sans from Google Fonts (with variable weights for typography hierarchy and light weight for `--mono` labels/metadata) and Computer Modern for mathematical formulas, so present with an internet connection or fonts fall back to system fonts.

- **← / →** (or **Space** / clicking the stage) steps through each beat and scene. If a scene is still building, the first press finishes it and the next moves on.
- **F** toggles fullscreen to fill the entire display.
- **1, 2, 3, 4** jumps directly to Outline, Chapter I, Chapter II, or Chapter III.
- **Home / End** jumps to Title or Summary.
- **S** opens full presenter script notes with a copy button; **Esc** closes notes.

## Other copies

- Editable Figma version with the same 19 scenes, motion timelines, script blocks and a click-through prototype: [Adaptive Handover — Proposal Deck](https://www.figma.com/design/MsZHwBo3nKbgjZMXXbhtGE). Edits made there are not synced to this file.
- A published copy of the page exists as a private claude.ai artifact on Mark Vincent D. Limpahan's account.

## Provenance and limits

The content follows the proposal PDF of October 2026 (`paper/main.tex` as compiled then): Chapters 1–3, Appendix A's sample calculations, Appendix B's simulation and Appendix C's prototype figure. It adds no results of its own.

- Numbers marked illustrative on the slides come from the appendices, not from experiments: the stabilization-score curves and SNR values on scene 14 (Appendix A.2–A.4), the λ 0.40–0.70 calibration walk-through on scene 16 (Appendix A.6), the per-input savings on scene 18 (Figure C.1) and the simulation bars on scene 18 (Table B.1).
- The October 25 audit and the EXP-001 pilot postdate that PDF and are not reflected in the slides: on the T4 grid, Learn-Then-Test certifies λ̂ = 1.20 rather than the illustrative 0.50, the stabilization score shows no discrimination on clean development prompts, and the 20 dB SNR gate misses 18–22 dB white noise. See [STATUS.md](../docs/progress/STATUS.md). Update scenes 13–18 before presenting if the manuscript changes accordingly.
- Changes on the unmerged `paper/adviser-revisions` branch are not reflected either.
