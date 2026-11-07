# Proposal defense walkthrough

An animated, full-screen presentation of the thesis proposal (Chapters 1–3) for the proposal defense, styled with a starry cosmic field, monochrome gradients, blue shade gradients (`#38BDF8` to `#0070E0`) for the teacher, grey for the student, and warm orange shade gradients (`#FFA04D` to `#D84300`) for what this study adds, with LEMMiNO-style citation footers on every content scene.

| File / Directory | What it is |
| --- | --- |
| [index.html](index.html) | The compiled standalone walkthrough: 21 scenes, single HTML file, full-screen layout (double-click ready) |
| [slides/](slides/) | 21 modular slide files containing slide markup, presenter script notes, and slide-specific styles |
| [template.html](template.html) | Base HTML shell, layout, cosmic field background, typography, and presentation controller |
| [build.py](build.py) | Fast builder (<30ms) to compile modular slides into `index.html` (supports `--watch`, `--serve`, `--sync-md`, `--pdf`) |
| [SCRIPT.md](SCRIPT.md) | The presenter script, one section per scene, about 24 minutes at the auto-play pace (about 17 without the appendix) |

## Editing slides

Each slide is modular and located in `slides/` (e.g. `slides/01-title.html`, `slides/05-one-handover-for-every-input.html`).

Inside each slide file, you'll find:
1. `<section class="scene ...">` — The slide HTML markup.
2. `<script type="text/slide-script">` — The spoken presenter script for that slide.
3. `<style>` — CSS styling specific to that slide.
4. `<script>` — Dynamic SVG or canvas animations (if applicable to that slide).

### Compiling changes
- **Build once**:
  ```bash
  python presentation/build.py
  ```
- **Auto-rebuild while editing (watch mode)**:
  ```bash
  python presentation/build.py --watch
  ```
- **Local dev server + watch mode**:
  ```bash
  python presentation/build.py --serve
  ```
- **Keep `SCRIPT.md` in sync**:
  ```bash
  python presentation/build.py --sync-md
  ```
- **Export to vector PDF with selectable text**:
  ```bash
  python presentation/build.py --pdf
  ```

## Deck structure (compact version, 2026-10-27)

- **Chapter I (slides 3–7):** background (flow matching, cost, distillation), the model at a glance (zooms into inference and the handover), the problem with the T4 evidence of Appendix D.1, objectives, scope.
- **Chapter II (slides 8–9):** the comparative table of adaptive methods (all seven rows of Table 2.1) and the gap map.
- **Chapter III (slides 10–16):** architecture, candidate switching times, the two signals, the stopping rule, calibration with the guarantee, research design and workflow, and contribution and summary.
- **Appendix (slides 17–21):** preliminary and two-grid rehearsal experiments (D.1–D.3 & EXP-017), signal diagnostic & premature stabilization (D.4–D.5 & EXP-011–015), live T4 online early-stopping & risk calibration (B–C & EXP-018), proposed STAC / 1D-CRC-STAC modular architecture (App. E.1), and multi-grid / cohort / OOD benchmark comparison (App. E.2).
- There is no separate thank-you slide: stepping past the last appendix slide returns to the title.

## Opening it

Open `index.html` in a current Chrome, Edge or Firefox (double-click works; no server needed). It loads Google Sans from Google Fonts (with variable weights for typography hierarchy and light weight for `--mono` labels/metadata) and Computer Modern for mathematical formulas, so present with an internet connection or fonts fall back to system fonts.

- **← / →** (or **Space** / clicking the stage) steps through each beat and scene. If a scene is still building, the first press finishes it and the next moves on. The ‹ › buttons at the bottom right do the same; ‹ and the ← key go back one slide.
- **F** toggles fullscreen to fill the entire display.
- **1, 2, 3, 4** jumps directly to Outline, Chapter I, Chapter II, or Chapter III.
- **Home / End** jumps to Title or Summary.
- **S** opens full presenter script notes with a copy button; **Esc** closes notes.

## Other copies

- Editable Figma version with the original 19-scene version (it does not reflect the compact restructure), motion timelines, script blocks and a click-through prototype: [Adaptive Handover — Proposal Deck](https://www.figma.com/design/MsZHwBo3nKbgjZMXXbhtGE). Edits made there are not synced to this file.
- A published copy of the page exists as a private claude.ai artifact on Mark Vincent D. Limpahan's account.

## Provenance and limits

The main proposal slides (`01`–`16`) follow the proposal PDF of October 2026 (`paper/main.tex` as compiled then): Chapters 1–3 and Appendix A's sample calculations. The Appendix slides (`17`–`21`) present both the proposal appendices and the post-proposal Tesla T4 development findings (`AUDIT-001`, `EXP-011` through `EXP-018`, including `STAC` and `1D-CRC-STAC`), explicitly labeled as development evidence rather than unseen held-out validation.

- Numbers marked illustrative on main slides `12` and `14` come from Appendix A, and the right-hand stress-test bars on slide `19` come from Appendix B (Table B.1).
- Update main Chapter III slides `10`–`16` before presenting if the team and adviser adopt `STAC` / `1D-CRC-STAC` into the main body of `paper/chapters/03-methodology.tex`.
- Changes on the unmerged `paper/adviser-revisions` branch are not reflected either.
