# Proposal defense walkthrough

An animated, full-screen presentation of the thesis proposal (Chapters 1–3) for the proposal defense, styled with a starry cosmic field, monochrome gradients, blue shade gradients (`#38BDF8` to `#0070E0`) for the teacher, grey for the student, and warm orange shade gradients (`#FFA04D` to `#D84300`) for what this study adds, with LEMMiNO-style citation footers on every content scene.

| File / Directory | What it is |
| --- | --- |
| [index.html](index.html) | The compiled standalone walkthrough: 19 scenes, single HTML file, full-screen layout (double-click ready) |
| [slides/](slides/) | 19 modular slide files containing slide markup, presenter script notes, and slide-specific styles |
| [template.html](template.html) | Base HTML shell, layout, cosmic field background, typography, and presentation controller |
| [build.py](build.py) | Fast builder (<30ms) to compile modular slides into `index.html` (supports `--watch`, `--serve`, `--sync-md`) |
| [SCRIPT.md](SCRIPT.md) | The presenter script, one section per scene, about 13 minutes at the auto-play pace |

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
- **Appendix (slides 17–19):** preliminary experiments (D.1–D.3), the signal diagnostic (D.4–D.5), simulation and prototype (B–C).
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

The content follows the proposal PDF of October 2026 (`paper/main.tex` as compiled then): Chapters 1–3, Appendix A's sample calculations, Appendix B's simulation and Appendix C's prototype figure. It adds no results of its own.

- Numbers marked illustrative on the slides come from the appendices, not from experiments: the stabilization-score curves and SNR values on slide 12 (Appendix A.2–A.4), the λ 0.40–0.70 calibration walk-through on slide 14 (Appendix A.6), the per-input savings on slide 19 (Figure C.1) and the simulation bars on slide 19 (Table B.1).
- The October 25 audit and the EXP-001 pilot postdate that PDF and are not reflected in the slides: on the T4 grid, Learn-Then-Test certifies λ̂ = 1.20 rather than the illustrative 0.50, the stabilization score shows no discrimination on clean development prompts, and the 20 dB SNR gate misses 18–22 dB white noise. See [STATUS.md](../docs/progress/STATUS.md). Update slides 10–15 before presenting if the manuscript changes accordingly.
- Changes on the unmerged `paper/adviser-revisions` branch are not reflected either.
