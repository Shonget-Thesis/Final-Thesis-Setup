# 2026-11-14: Appendix slide 22, Two-Axis Handover architecture

The requester asked for the Two-Axis Handover's architecture as a new appendix slide in `presentation/index.html`, with every component present, a clear statement of what each does, and a legend for each piece.

## What changed

- **`presentation/slides/22-appendix-two-axis-architecture.html` (new).** The diagram follows slide 10's component styles: blue for the teacher, grey for the student, white for released parts, and orange dashed for what the design adds.

  | Component | What the slide says it does |
  |---|---|
  | Input | the prompt audio, and the prompt and target text |
  | SNR router | axis 1, what crosses; it measures how noisy the voice prompt is, at 0 passes |
  | Three lanes, each with its SNR range | clean → the teacher's whole estimate; borderline → Voice-Swap; noisy → nothing crosses |
  | Teacher (clean lane) | guided Steps 0–3, 2 passes per step |
  | Movement monitor | axis 2, when; computes A₃ = d₁ + d₂ + d₃ at 0 extra passes |
  | Decision diamond | A₃ ≥ λ₃? |
  | Teacher Step 4 | one more step when the teacher has not moved enough |
  | Teacher and student draft (borderline lane) | the teacher's Steps 0–4, and one student pass from noise |
  | Voice-Swap | voice(S) + slow(T₄) + fine(T₄) |
  | Student alone | the noisy lane: no teacher at all |
  | Student restart | finishes from the handed estimate, at τ = 0.0761 |
  | Vocos | turns the mel into a waveform |
  | Offline fit | θ = 26.55 dB, a 2 dB band and λ₃ = 1.2154, fitted once on stored data |

  - **Pass counts:** every route is labeled with its cost: 11, 13, 14 or 4.
  - **Three cards** sit under the diagram:
    - **Legend:** colours, shapes and arrow types.
    - **Symbols:** Ŝ, θ, d_k, A₃, λ₃, T₃ and T₄, S, τ, and voice, slow and fine.
    - **Cost and evidence:** 6.73 mean passes out of fold (−68%), the EXP-023 G3 pass, and the status: development evidence, a combination of known parts, frozen test pending.
  - **The rest of the slide:** a citation footer and a presenter script.
- **`presentation/index.html`:** rebuilt with `python presentation/build.py --sync-md`, now 22 slides. `presentation/SCRIPT.md` was synced.
- **`presentation/README.md`:** the scene count and appendix list now include slide 22. It also notes that `proposal-defense-slides.pdf` still has 21 slides, and how to clear the leftover headless Edge processes that can block a re-export.
- **`docs/progress/STATUS.md`:** the presentation line now describes a 22-slide deck.

## Checks run

- **Deck build:** `build.py` built 22 slides.
- **Layout check:** four PDF exports into the session scratchpad. The rendered page 22 was checked visually for overlaps, clipped labels and the animated pulses. The pulses now run along the wires only and start at their routes rather than at the SVG origin. The committed PDF was not regenerated.
- **Numbers on the slide, checked against the sources:**
  - **The 6.73 mean passes and the 60 development speakers:** they match `2026-11-13-offline-solution-screen.md`.
  - **The pitch-variety figure:** checked against `results/pilot_results.json`.
    - **The value:** `two_axis.pitch.TA` has mean −0.0013, interval −0.0042 to +0.0010, n = 10.
    - **What it measures:** the routed Two-Axis track on 10 clean-prompt speakers, all on the whole-estimate branch. For 95% of seeds that branch handed over at Step 3; Step 3 alone gives −0.0009.
    - **The correction:** the first version labelled it "at Step 3". The card and the presenter script now say what it measures.
- **Leftover Edge processes:** the third export hung, because earlier exports had left headless Edge processes on port 9229. Only processes whose command lines contain `slide_pdf_` were stopped, and the export then completed. After the fourth export, 8 more such processes were stopped.
- **`python scripts/check_workspace.py`:** PASSED.
- **`python -m unittest discover -s code/tests`:** 188 tests OK.
- **`latexmk -Werror` in `paper/`:** exit 0, nothing to rebuild. The manuscript sources did not change.

## Remaining blockers

- **The committed PDF:** `proposal-defense-slides.pdf` was not regenerated.
- **Not in the manuscript:** the slide presents a development candidate. Chapter 3 and its figures are unchanged, and adopting the method would need a dated decision record and adviser review.
