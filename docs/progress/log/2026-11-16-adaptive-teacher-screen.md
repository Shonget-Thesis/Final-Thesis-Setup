# 2026-11-16: A restart-masked adaptive teacher: derivation, literature audit and local screen (negative)

Follow-up to EVAL-026 (EXP-033); no new task ID.
- **The request:** Rhenel Jhon R. Sajol asked for a brand-new handover algorithm "like how other algorithms become novel". The example was Huffman coding: an existing procedure (Shannon–Fano coding), one changed rule, and a proof.
- **Status:** a design, an audit and a local CPU screen (screening only, not evidence). The candidate failed its pre-set screen. Nothing is adopted.

## Motivation from EXP-033

With three teacher steps to the commit point (`CPH6`, 10 passes), the outcome depends on the sentence (clean prompts, T4, against `W8R2`):
- **18 of 30 sentences** were fine or better;
- **5 broke badly** (−1.0 to −2.5 UTMOS);
- **what drives it:** the damage tracked how far the coarse path's estimate ended from the 16-interval Step 8 estimate (Spearman −0.39). The stored 16-interval drifts correlated with that distance (+0.44 to +0.56).

So a fixed grid is wrong both ways. That motivated an adaptive teacher.

## The candidate

**Existing algorithm:** adaptive step-size control with an embedded error estimate (as in Runge–Kutta–Fehlberg and Dormand–Prince), applied to DMOSpeech 2's teacher before its handover.

**Derived parts:**
- **Estimate-change identity (exact algebra).** For one Euler step from t_a to t_b (step h), x̂_b − x̂_a = (1 − t_b)(v_b − v_a). Euler's local error is therefore e ≈ (h/2)(v_b − v_a) = h / (2(1 − t_b)) · (x̂_b − x̂_a), and it needs no extra pass.
  - This is Chapter 3's drift d_k times ‖x̂_b‖ · h / (2(1 − t_b)). The drift is, up to that factor, the teacher's local truncation error.
- **Restart masking.** The student sees the handed-over estimate through φ = τ·x̂ + (1 − τ)·ε, so a teacher error counts in units of the restart noise: r = τ / (1 − τ) · rms(e), with τ = t_c = 0.2929 (factor 0.414).
- **Rule:** walk three coarse steps to t_c. Give a flagged step one Heun correction (one extra guided evaluation, 2 passes) when r > θ. Hand over at t_c with the 0.2929 restart.

## Literature audit (2026-11-16, targeted searches)

| Work | What it already does | Relation |
|---|---|---|
| [Gotta Go Fast](https://arxiv.org/abs/2105.14080) (Jolicoeur-Martineau et al., 2021) | Adaptive step size with an error tolerance for score-based SDE samplers | The adaptive-step part is known |
| ["Error as Signal"](https://en.papernotes.org/ICLR2026/image_generation/error_as_signal_stiffness-aware_diffusion_sampling_via_embedded_runge-kutta_guid/) (ICLR 2026) | Uses the free Euler/Heun difference of each step to gate a correction behind a threshold (images) | The gated-correction part is known |
| [Adaptive Second-Order Solvers](https://www.opentrain.ai/papers/adaptive-second-order-solvers-for-fast-stochastic-diffusion-sampling--arxiv-2610.03034/) (2026; listing only) | PI step-size control with a noise-normalised error estimator; per-sample adaptivity helps little on images and language | Close; its normalisation is not by a downstream restart |
| [Restart Sampling](https://arxiv.org/abs/2306.14878) (Xu et al., 2023) | Injected noise contracts accumulated error | The masking principle is known |
| [Fast F5-TTS / EPSS](https://arxiv.org/abs/2505.19931) (Interspeech 2025) | Prunes F5-TTS's sway schedule to 7 evaluations, training-free | Fewer teacher steps (our `CPH` arms) is known for F5-TTS |
| [Optimal Stepsize for Diffusion Sampling](https://arxiv.org/html/2503.21774v1) (2025) | Dynamic programming over step sizes to match a many-step teacher | Fixed schedules, known |
| [T-Stitch](https://arxiv.org/abs/2402.14167) (ICLR 2025) | A small model first, a large one later (fixed schedule) | `SST`'s order is known |
| [DMOSpeech 2](https://arxiv.org/abs/2507.14988) demo ([app.py](https://huggingface.co/spaces/mrfakename/DMOSpeech2/blob/main/app.py)) | A "High Diversity" preset uses `student_start_step = 2` | The late restart for prosody is the authors' own setting |

**Not found:** an error budget set by a distilled student's restart, or a teacher integrated adaptively only up to a handover point. That would have been the new part, but the screen below rejected it.

## Local screen

**Script:** `code/analysis/adaptive_teacher_screen.py`. Output: `research/experiments/EXP-033-cheaper-teacher-path/offline/adaptive_teacher_screen.json`. The pass rules are fixed in the script's docstring before the run.

**Setup:**
- **Sentences:** the five clean EXP-033 sentences where `CPH6` broke worst on the T4, and the five where it did best (input seeds, development speakers).
- **Every correction pattern** on the three coarse steps (eight leaves, prefixes shared).
- **The known competitor:** AB2, a second-order multistep solver at no extra cost.
- **References:** `W8R2` and `CPH8`.
- **Scores:** UTMOS and the estimate's distance from the local Step 8 estimate.
- **Time:** about 24 minutes on this computer's CPU.

| Pre-set rule | Result |
|---|---|
| 1. The local `CPH6` distances reproduce EXP-033's T4 values (Spearman ≥ 0.8) | **Pass** (1.0; equal to 3 decimals) |
| 2. The indicator separates the broken from the good sentences (AUC ≥ 0.8) | **Fail** (0.16: broken sentences give *smaller* indicators) |
| 3. Some threshold repairs every broken sentence at ≤ 11 mean passes, and AB2 does not already repair them | **Fail** (no threshold does; AB2 repairs 1 of 5) |

**What the eight leaves show** (UTMOS; correcting step 1, 2 or 3 is marked H):

| Leaf | Passes | Mean UTMOS | Mean distance from Step 8 | Broken sentences repaired |
|---|---:|---:|---:|---:|
| No correction (`CPH6`) | 10 | 3.03 | 0.694 | 0 of 5 |
| Step 1 corrected (`HEE`) | 12 | 3.53 | 0.435 | 3 of 5 |
| All three corrected (`HHH`) | 16 | 3.62 | 0.412 | 3 of 5 |
| `CPH8` (four Euler steps) | 12 | 3.66 | 0.255 | 3 of 5 |
| `W8R2` | 20 | 3.77 | 0 | (reference) |
| AB2 | 10 | 2.00 | 1.828 | 1 of 5 |

**Notes on the table:**
- **The fifth broken sentence** (`common_voice_en_15734837`) is not repairable by any arm. `W8R2` itself scores 2.30 there.
- **AB2 is unstable** on this grid: it wrecks the good sentences too (UTMOS 1.3–1.9).

## What this means

1. **The damage of a coarse teacher is made in its first step,** from pure noise to t = 0.034.
   - **What repairs it:** correcting that step alone repairs most broken sentences; correcting the later steps adds little.
   - **Why a local error check cannot see it:** the coarse path is sent to a different branch and then stays smooth on it. The free error check measures smooth curvature, so it reads *lower* on the broken sentences.
   - **The same pattern as before:** "wrong but stable", as in EXP-012 (noise lowers drift) and EXP-013 (the endpoint changes direction).
2. **The first-step correction costs 12 passes and is no better than `CPH8` (12)**, so it adds nothing.
3. **The adaptive teacher is set aside.** It is the third candidate with a new part (after `CWH4` and `SST` in EXP-033) that the data rejected.
4. **The cheapest handover shown to keep the paper's variety and quality is still `CPH8`:** the released sampler with fewer teacher steps and the paper's restart, with the SNR gate for noisy prompts. It is a configuration, not a new mechanism.

## Checks run

- **The screen** as above (exit 0). The local teacher reproduces EXP-033's T4 coarse-path distances exactly.
- **Tests:** `code/tests/test_adaptive_teacher_screen.py` (the threshold rule, the AUC, the grid and the case selection).
- **Workspace check:** `python scripts/check_workspace.py` PASSED.
- **Full suite:** `python -m unittest discover -s code/tests` ran 382 tests, OK (1 skipped).
