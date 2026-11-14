# 2026-11-14: EXP-024 Analytic Unconditional Guidance pilot prepared

Task: EVAL-017 (todo). Requested by Rhenel Jhon R. Sajol, who asked for one simple mechanism derived from theory, new as a procedure, and screened before any GPU run.

## Why

- **How the candidate was chosen:**
  - **Earlier rounds:** the team's routers and payloads did not meet the requester's novelty bar, and on these white-noise data prompt SNR beats every model-internal routing signal tried (EXP-008, EXP-012, EXP-023). So the remaining room was in what the teacher computes.
  - **Dropped the same day after checks:**
    - A time-compressed teacher: this is the Half-Rate Teacher set aside on 2026-11-12.
    - Cropping the unconditional pass to the target frames: Lite-CFG (DC-DiT) already gives the unconditional pass fewer tokens.
    - Prompt-prefix caching: diffusion language models already do it.
  - **Kept:** Analytic Unconditional Guidance (AUG) was the one candidate whose procedure survived a family-level novelty search.
- **The theory:**
  - **At Step 0:** for an ideal network, the unconditional velocity is exactly μ − x at t = 0, because the state is pure noise.
  - **At later steps:** it is μ plus a term of order t.
  - **Why it applies here:** the teacher in teacher-guided sampling hands over by t = 0.0761 (Step 4) or t = 0.2929 (Step 8).
- **What the stored data showed beforehand (exploratory, development data):**
  - **What the teacher adds on clean prompts:** the student alone matches the reference's mean quality there (WER −0.006, similarity +0.001, UTMOS +0.008 against Step 8). The teacher's contribution is across-seed pitch variety.
  - **Only the slow part reaches the student:** swapping the teacher's fine detail for the student's draft changes nothing measurable (EXP-020 STT_4 against EXP-022 STS_4, clean and degraded; every 95% interval includes 0).
  - **Steps matter in the window:** halving the teacher's steps inside the handover window lowers clean UTMOS by 0.04–0.05 at Steps 1–3. So AUG keeps every step and only removes the unconditional pass.

## What changed

- **`code/analysis/exp024_power_check.py` and `power_check.json`:** a pre-run power check on stored outputs. It sized the pilot before the freeze, using the proxy of a 16- against a 32-interval grid at the same sway time:
  - 40 speakers rather than 20, because clean UTMOS passes the proxy only 69% of the time with 20.
  - A clean Step-8 WER bound of +0.015 rather than +0.01: the proxy passes +0.01 only 33% of the time at Step 7.
  - On noisy prompts, Step-4 handovers are not judged, and the Step-8 criterion is secondary, because the grid change itself moves similarity there.
- **`code/colab/run_analytic_guidance_pilot.py`:**
  - **calibration:** μ from 20 cases at Quick-set positions 41–60;
  - **diagnostic and quality on 80 cases:**
    - the closed-form gap at Steps 0–8;
    - the Step-0 split into seed-specific and position parts;
    - AUG's change of the Step-4 and Step-8 estimates next to the grid change;
    - W4_K and W8_K for K = 2 and 4, scored against EXP-018's stored outputs of the same cases and seeds;
  - **pitch variety:** 10 speakers × 20 seeds, against EXP-020's stored CVf0;
  - **timing:** the teacher on 10 cases on the T4;
  - **analysis:** reads only the CSV files.
- **`research/experiments/EXP-024-analytic-unconditional-guidance-pilot/`:**
  - the README;
  - `preregistration.json`: sha256 `993c306d13cfb32d...`, created 2026-11-14T00:03 UTC at thesis commit `7d35e81`, before any synthesis;
  - the power check.
- **`code/tests/test_analytic_guidance.py`:** 29 tests, covering:
  - pass accounting;
  - disjoint pilot and calibration speakers;
  - the seed streams;
  - the closed-form gap, the Step-0 split and the slow-part change;
  - the released loop with AUG switched off;
  - the skipped unconditional calls;
  - **AUG reproducing the released teacher exactly when the unconditional branch equals the closed form;**
  - every run loop on a fake model;
  - the decision rule on synthetic outputs;
  - preregistration tampering and changed code.
- **`code/colab/DMOSpeech2_EXP024_Analytic_Guidance_Pilot.ipynb`:** 17 cells, built from the EXP-023 layout. It keeps that notebook's RAM safeguards and resume-after-disconnect steps.
- **Other files:**
  - the EVAL-017 task, with TASKS.md re-rendered;
  - a row in `research/experiments/README.md`;
  - STATUS.md;
  - `.gitattributes`, which keeps future EXP-024 result bytes unchanged.

## Novelty check (family-level searches, 2026-11-14)

No method was found that keeps the conditional pass and computes the unconditional term of classifier-free guidance in closed form. The closest methods, and how they differ:
- **Lite-CFG (DC-DiT):** fewer tokens in the unconditional pass; needs a trained model.
- **Path-Drop Guidance (SPRINT):** skips middle blocks; needs a trained architecture.
- **The guidance interval:** switches guidance off.
- **CFG-Zero⋆:** zeroes the first steps.
- **Independent-condition and time-step guidance:** still run a network pass.
- **Wang and Vastola's teleportation:** replaces the whole model and skips steps.
- **Analyses of CFG in Gaussian or linear settings:** no sampling shortcut; the 2026 one's sampler still runs the unconditional pass.

The theory is known. Links are in the EXP-024 README. This is not a full literature review.

**Correction the same day, after the freeze and before any synthesis.** A search of practitioner tools found ComfyUI "Uncond Zero" nodes. They skip the unconditional pass by treating its prediction as zero. Replacing that pass with a cheap fixed substitute is therefore already practised, though not published. AUG is a principled variant of it: the substitute is the derived closed form, used only where that form is accurate. By the requester's standard it is partly novel. The EXP-024 README records this; the preregistration is unchanged.

## Checks run

- `python code/analysis/exp024_power_check.py`: wrote `power_check.json`.
- `python -m unittest test_analytic_guidance` (from `code/tests`): 29 tests, OK, against the frozen preregistration.
- Notebook builder: 17 cells, and all 8 code cells compile.
- `python -m unittest discover -s code/tests`: 217 tests, OK.
- `python scripts/check_workspace.py`: PASSED.
- The manuscript was not changed, so it was not rebuilt.

## Remaining blockers

- **GPU run:** it needs a Colab Tesla T4. Nothing has been synthesized.
- **Push first:** the runner and preregistration must be pushed to branch `rhnl-3` before the notebook can clone them.
- **Evidence status:** a pass would earn a frozen test on new speakers, not a claim. A change to the thesis method needs a dated decision record and adviser review.
