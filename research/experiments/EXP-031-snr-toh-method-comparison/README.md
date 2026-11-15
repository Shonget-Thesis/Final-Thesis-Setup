# EXP-031: three single-formula handovers against the previous handover methods

**Status:** ran on 2026-11-15 (Colab Tesla T4) under the frozen preregistration `a28c78a6d87a...`; results imported on 2026-11-16 (see [Results](#results-imported-2026-11-16)). Prepared and preregistered on 2026-11-15, before any synthesis.
- **First freeze:** `ec06ed0568b4...`.
- **Amendment 1 (same day, still before any synthesis):** QH3 and BTH3 were added, and the primary methods became three single-formula handovers with no routing. The preregistration was re-frozen as `df26da1412bf...`.
- **Amendment 2 (same day, after the first Colab smoke test stopped and before any main-run synthesis):** the blind teacher's placeholder condition is now a one-frame zero mel instead of a 512-sample silent waveform, which was too short for the input routine's spectrogram padding. The placeholder is dropped in both guidance branches, so the method is unchanged. Re-frozen as `a28c78a6d87a...`.

Task EVAL-024 (review). The run took a Colab Tesla T4, about 3 hours. Files: [notebook](../../../code/colab/DMOSpeech2_EXP031_SNR_TOH_Method_Comparison.ipynb), [runner](../../../code/colab/run_method_comparison.py), [tests](../../../code/tests/test_method_comparison.py).

- **A development experiment:** the speakers are development speakers. It is not held-out evidence.
- **Not adopted:** nothing here changes the thesis method, and no adviser has reviewed it.

## Why this experiment

- **EXP-030** found TOH3 `supported` under its frozen rule. A mirrored map did about as well, so whether the teacher's own timing is needed is open.
- **The requester asked for three things on 2026-11-15:**
  - **Every earlier method compared** on the same cases.
  - **A handover that is a formula, not an if-else policy:** the first design's SNR + TOH routing was judged "just a policy".
  - **Two more candidates:** QH, from the [theory note](../../../docs/progress/log/2026-11-15-handover-invariance-theory.md), and BTH, from the [Blind-Teacher design](../../../docs/progress/log/2026-11-15-blind-teacher-handover-design.md).

## The methods under test (applied to every input, 12 passes each)

DMOSpeech 2's handover is kept: the guided teacher runs 3 sway steps, then the student finishes from its first restart level. The three candidates change only what crosses.

| Candidate | What crosses | Formula |
|---|---|---|
| `TOH3` (EXP-030) | The teacher's timing | S₀ re-timed along a slope-limited DTW map from the teacher's estimate T₃ |
| `QH3` (Quantile Handover) | Each band's time order | P_b(t) = sort(S₀,b)[rank of T₃,b(t)]. Every value is the student's own value in that band. Exactly unchanged by any stationary channel or noise floor |
| `BTH3` (Blind-Teacher Handover) | The timing of a teacher that never receives the prompt | The teacher is given the target text only, over the target's frames, with the audio condition dropped in both guidance branches. Then TOH's map onto S₀. The prompt's audio never reaches the teacher, and its passes run on the target's frames only |

**Correction (2026-11-16):** an earlier version of this row said that no property of the prompt can reach the blind teacher. It can, in two ways:
- **The target length:** the pipeline's duration estimate sets it from the prompt, and in this run it changed with the condition for most speakers.
- **The student's draft:** the map is aligned to S₀, which hears the prompt.

Only the teacher's estimate at a given length is free of the prompt. See the [results log](../../../docs/progress/log/2026-11-16-exp031-results.md).

S₀ is the student's one-step draft. The student's remaining three levels run from the payload with the same seeds as the student alone.

## What they are compared with

All thresholds were fixed by earlier work. The comparison is paired: the same teacher run, student draft and restart noises.

| Method | Source | Rule | Passes |
|---|---|---|---|
| `W4` | DMOSpeech 2's code default | Teacher 4 steps, then the student | 13 |
| `W8` | The study's reference | Teacher 8 steps, then the student | 21 |
| `STAC` | EXP-018 | First step k in {4, 5, 6} whose cumulative drift A_k reaches λ_k = 1.513, 1.647, 1.716; otherwise 8 | 13–21 |
| `CH3` | Chapter 3's stopping policy | SNR gate (20 dB); then the first step k in {4, 5, 6} with d_k ≤ 0.35 (four-fold CV, Appendix D); otherwise 8 | 13–21 |
| `CH3_LTT` | Chapter 3, LTT-certified | Same, with λ = 1.20, the value fixed-sequence LTT certified on the 200 calibration speakers (Appendix D) | 13–21 |
| `RND` | Control | S₀ re-timed by a smooth random map, with no teacher | 4 |
| `SNR_TOH`, `SNR_W4`, `SNR_RND` | First design, now exploratory | The SNR gate, with the student alone below it | 4–13 |

**The random map (`RND`):**
- **Shape:** smoothed white noise, with both ends tied to zero.
- **Size:** fixed from EXP-030's TOH3 maps (RMS displacement 2.7% of the length; largest displacement about 2 × RMS). The settings are RMS target 2.9% and smoothing 10% of the length, which give 2.6% and 1.9 in simulation.
- **Limits:** local slopes within 1 ± 0.45 and inside TOH's band.
- **Seed:** each case's seed.

**Previous lines that are not re-run:**
- **Endpoint extrapolation and other handover operators (EXP-013, EXP-014):** saved no compute.
- **SNR plus other signals (EXP-012, EXP-015):** these are represented by `CH3`.
- **Voice-part and slow-part swaps (EXP-020, EXP-022) and the Decision Handover (EXP-023):** did not advance.
- **The transmissibility line (EXP-026 to EXP-029):** closed by EXP-029.
- **EXP-030's mirrored map:** replaced by `RND`.
- **EXP-021, EXP-024 and EXP-025:** never run.

## What runs

| Stage | Cases | Arms synthesized and scored |
|---|---|---|
| Quality | Development speakers 1–30 × 6 conditions = 180 | `STU`, `W4`, `W5`, `W6`, `W8`, `TOH3`, `QH3`, `BTH3`, `RND` |
| Pitch variety | Speakers 1–20 × 12 seeds, clean prompts | `STU`, `W4`, `W8`, `TOH3`, `QH3`, `BTH3`, `RND` |

**Recorded at no extra cost:**
- the prompt SNR and the teacher's drifts (STAC and CH3 are read from these);
- the teacher's and the blind teacher's sequence lengths, for BTH's compute share;
- the map sizes, and how far QH moved the teacher's estimate.

**Conditions** (EXP-011's definitions):

| Condition | What it is | What EXP-011 found on speakers 1–20 |
|---|---|---|
| `clean` | The prompt as recorded | SNR median 38.7 dB |
| `gauss25`, `noisy15` | White noise at 25 and 15 dB | `gauss25` passes the gate and the teacher copies it |
| `reverb03` | Synthetic room, RT60 0.3 s | Read as clean (median 28.5 dB), yet the teacher copies the room: W4 UTMOS 2.71 against 3.86 on clean |
| `reverb09` | RT60 0.9 s | Around the gate (median 19.0 dB); W4 UTMOS 1.47 |
| `babble15` | DEMAND cafeteria babble at 15 dB | Around the gate (median 19.4 dB); W4 UTMOS 3.34 |

- **Telephone prompts:** dropped. In EXP-030 no arm differed on them.
- **Speakers:** development speakers 1–30 have never been through TOH. The first 20 are EXP-011's pilot speakers with the same seeds, which allows a reproduction check. The calibration and test splits are untouched.

**Measures:**
- **Quality:**
  - UTMOS;
  - WER against the text;
  - speaker similarity against the clean ground-truth recording.
  - Quality is judged on these directly, not by agreement with W8, which copies the prompt's condition.
- **The thesis's failure count:** Definition 3.1's failures against W8 are reported for continuity.
- **Pitch variety:**
  - CVf0;
  - **aligned CVf0:** each seed is aligned to the first in time before the comparison, so timing differences no longer count;
  - **timing spread:** how far the seeds' timings differ, in frames.

**Smoke test:** a run stops before synthesis if any of these fails:
- **Exact identities (≤ 1e-5):**
  - the split student loop matches the full one;
  - the identity map leaves the draft unchanged;
  - TOH with the identity map equals the student alone;
  - QH of the draft onto itself equals the draft;
  - QH's values equal the draft's values in every band;
  - the blind teacher gives the same payload twice.
- **The maps** of TOH3, BTH3 and the random map on real teacher runs: monotone, within their slope limits and inside the band.
- **The policies:** STAC's and CH3's step choices are valid.
- **EXP-027's W4:** reproduced (≤ 0.01).
- **Micro runs:** both stages, including one babble prompt.

## Hypotheses and verdict rule (frozen, Amendments 1 and 2)

These apply to each candidate X in TOH3, QH3 and BTH3.

| | Question | Test |
|---|---|---|
| T1 | Does the teacher add variety beyond random re-timing? | CVf0(X) − CVf0(RND): lower bound > 0 |
| T2 | Does the teacher's contribution sound more natural than random re-timing? | Clean UTMOS(X) − UTMOS(RND): lower bound > 0 |
| C | Does X keep out conditions other than white noise better than DMOSpeech 2's handover? | Per-speaker mean over reverb03, reverb09 and babble15 of UTMOS(X) − UTMOS(W4): lower bound > 0 |
| M | Does X beat a previous method C? | **(a) Clean:** mean ΔUTMOS ≥ −0.08, ΔWER upper bound ≤ +0.01, ΔSIM lower bound ≥ −0.01. **(b) Degraded** (per-speaker mean over the five degraded conditions): ΔUTMOS lower bound > 0, ΔWER upper bound ≤ +0.01, ΔSIM lower bound ≥ −0.01 |

Bounds are 95% speaker-level bootstrap intervals (B = 2000, seed 2026).

**Verdicts (three labels per candidate):**

| Label | Values |
|---|---|
| Teacher | `teacher_needed` if T1 or T2 holds; otherwise `teacher_not_shown_needed` |
| Method | `beats_all_previous` if M holds against W4, W8, STAC, CH3 and CH3_LTT; `beats_some_previous` if against at least one; otherwise `beats_no_previous` |
| Condition | `condition_safety_shown` if C holds; otherwise `condition_safety_not_shown` |

- **No ranking:** the frozen rule does not rank the three candidates.
- **No correction for three candidates:** this is a development pilot.
- **Exploratory:**
  - contrasts between the candidates: aligned CVf0 for QH3 against TOH3, which the theory predicts is positive, and reverb and babble UTMOS for BTH3 against TOH3;
  - the SNR-routed methods, with the first design's S1;
  - the student alone against the previous methods;
  - per-condition tables;
  - BTH's teacher sequence share;
  - the EXP-030 replication;
  - Definition 3.1 failures;
  - the policies' step choices;
  - runaway outputs (WER > 1);
  - the EXP-011 reproduction.
- **No retuning on these data.**

## What a result would mean

| Result | Meaning |
|---|---|
| A candidate with `teacher_needed`, `beats_all_previous` and `condition_safety_shown` | A single-formula handover that needs no routing, beats every earlier method on these data, and keeps conditions out. It is the candidate to bring to the team and adviser, after a full novelty audit |
| All three `teacher_not_shown_needed` | Random re-timing does as well, so the teacher's passes are not justified by these measures. That is a team decision about a teacher-free route |
| QH3 above TOH3 on aligned CVf0 | The teacher's variety includes pitch movement that timing alone misses, and QH carries it |
| BTH3 equal to TOH3 on quality and variety | The teacher does not need the prompt. Each BTH teacher pass runs on the target only: its share of the full sequence is reported |

**Known biases and limits:**
- **In favour of the previous methods:** STAC's and CH3's thresholds were set on development data that include these speakers.
- **White noise is easy to win on:** two of the five degraded conditions are white noise, which is why C and the per-condition tables are reported separately.
- **No listening test, and no held-out speakers.**

## Results (imported 2026-11-16)

Both stages completed: 180/180 quality cases and 20/20 × 12 diversity.
- **Checks:** the smoke identities were exactly 0. The manifest and preregistration hashes match. `analyze()` reruns locally to the same results. The EXP-011 reproduction is within 1.1e-3 UTMOS.
- **Full account:** the [results log](../../../docs/progress/log/2026-11-16-exp031-results.md).
- **Descriptive diagnostics:** `results/diagnostics.json`, from `code/analysis/exp031_diagnostics.py`.

**Frozen verdicts:**

| Candidate | Teacher (T1, T2) | Method (M) | Condition safety (C) |
|---|---|---|---|
| `TOH3` | `teacher_not_shown_needed` | `beats_all_previous` | `condition_safety_shown` (lower bound +0.001) |
| `QH3` | `teacher_not_shown_needed` | `beats_no_previous` (one utterance's inserted word puts the clean WER bound at +0.0125) | `condition_safety_not_shown` |
| `BTH3` | `teacher_not_shown_needed` | `beats_all_previous` | `condition_safety_not_shown` |

**What it means:**
- **The teacher is not shown to be needed.** The teacher-free random re-timing (`RND`, 4 passes) matches every candidate on raw pitch variety and clean UTMOS. This is the outcome listed above as "random re-timing does as well".
- **"Beats all previous" does not credit the teacher.** Under the same frozen M function (checked post hoc), the student alone (4 passes) also beats all five previous methods. The margins come almost entirely from white noise, which the teacher copies and the student does not.
- **The teacher's measurable contribution is pitch variety beyond timing.** On aligned CVf0, W4 adds +0.0175 over the student. Some of it is carried by two candidates (exploratory):
  - QH3: 64% of W4's gain, at W4's clean UTMOS;
  - BTH3: 46%;
  - TOH3 and RND: about 15%.
- **Variety trades against UTMOS.** Arms with more of this variety have lower clean UTMOS, as in EXP-029.
- **For the team:** whether the variety is wanted needs a listening test, and it is a team decision.
- **BTH3's teacher** ran on about half the sequence (share 0.504), but no time was measured.

## Data and evidence status

- **Development evidence only.**
- **Audio** is not committed.
- **Results** are stored under `results/`:
  - the nine files from the results zip, byte-identical, with a SHA-256 manifest and the DEMAND file's hash (`demand_sha256.txt`, `854dd0e5ef33...`);
  - `diagnostics.json`.
