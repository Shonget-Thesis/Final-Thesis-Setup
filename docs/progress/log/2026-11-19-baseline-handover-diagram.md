# 2026-11-19: Create existing DMOSpeech 2 baseline handover architecture diagram and embed in Appendix E

Task: `PRES-003` (`review`). Requested by user (`"so our teahcer wants to see baselindes handover achtecture can you create it as iamge?"`, `"EXISITNG HANDOVER ONLY NOT THE ONE WE MADE"`, `"remove thise jsut architecure is snough ahahaha"`, `"is the exising dmospeech 2 handover in archtecutre in appendix ??"` -> `"add it"`).

## Changes

- Created byte-identical [`presentation/baseline_handover_architectures.png`](../../../presentation/baseline_handover_architectures.png) and [`paper/figures/baseline_handover_architectures.png`](../../../paper/figures/baseline_handover_architectures.png) (`3200 x 1800` RGB PNG, `906,686` bytes, SHA-256 `ebb57733896dc1324ad41bde24ac09826032254c9f838363887cec166e4b27bf`), rendered with headless Chrome in the same visual palette as [`presentation/exp034_architecture.png`](../../../presentation/exp034_architecture.png).
- Depicts **strictly the existing DMOSpeech 2 baseline handover architecture** (Li et al., 2025/2026; [`code/vendor/dmospeech2/src/infer.py`](../../../code/vendor/dmospeech2/src/infer.py) `_teacher_sampling` / `_student_sampling`) with zero thesis-proposed mechanisms and no separate variant-card strip:
  1. **Inputs & Setup**: Voice prompt audio, prompt/target text, `Duration Pred. (SLP)`, `step_cond + cond_mask`, and `Init y_0 ~ N(0, I)` with no input screening.
  2. **Guided Teacher ODE (F5-TTS)**: `16`-interval sway grid ($t_i = 1 - \cos((\pi/2)\cdot i/16)$), explicit Euler steps along guided velocity $v = v_c + \gamma(v_c - v_u)$ ($\gamma = 2.0$, `2` passes/eval, optional $\eta$).
  3. **Static Handover Switch & Endpoint Conversion**: Loop terminates at the first grid point $t_k > t_{\text{switch}}$ (no adaptive signal or gate), runs one final CFG evaluation at $t_k$, and hands over raw endpoint mel prediction $\hat{x}_1 = x + (1 - t_k)\,v(x, t_k)$ (`18` teacher passes at $s = 0.25$, $k = 8$, $t_8 = 0.2929$; signature default $s = 0.07$: `10`).
  4. **Student (DMOSpeech, `j = 1`)**: Starts at `student_start_step = 1` across $\tau \in \{0.0761, 0.2929, 0.6173\}$, applies SDEdit-style re-noising $\phi = (1 - \tau)\varepsilon + \tau x_1$, and updates $x_1 \leftarrow \phi + (1 - \tau)\,g(\phi, \tau)$ (`3` unguided student passes; `4` passes from $\tau=0$ if `teacher_steps = 0`).
  5. **Vocos Vocoder**: Strips prompt frames and decodes mel to `24 kHz` audio (`C = 2(k+1) + 3`, `21` passes at $s = 0.25$).
- Added [`paper/figures/fig_baseline_handover.tex`](../../../paper/figures/fig_baseline_handover.tex) and embedded it as **`Figure E.1`** inside [`paper/chapters/appendices.tex`](../../../paper/chapters/appendices.tex) (`app:snr-r-arch`, right before `Figure E.2` `\ref{fig:snr-r}`), plus a cross-reference from Appendix D (`Student Restart Level and Paper Baselines`). Chapters 1–3 remain untouched.

## Checks Run

- Rendered `3200 x 1800` PNG via headless Chrome (`--force-device-scale-factor=2`) and visually verified zero clipping or overlap across both `presentation/` and `paper/figures/` copies.
- Built `paper/build/main.pdf` via `latexmk -interaction=nonstopmode -halt-on-error -Werror main.tex` (`109` pages, `3,124,040` bytes, exit `0`) and visually inspected rendered `pdftoppm` Appendix E pages (`94`–`96`) confirming clean `Figure E.1` and `Figure E.2` placement.
- Ran `python scripts/render_progress.py`, `python scripts/check_workspace.py`, `python scripts/check_workspace.py --base origin/main`, and `python -m unittest discover -s code/tests`.
