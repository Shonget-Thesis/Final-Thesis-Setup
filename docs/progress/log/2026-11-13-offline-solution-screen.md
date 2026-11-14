# 2026-11-13: Offline screen of handover solutions

Requested by Rhenel Jhon R. Sajol: test different solutions and cases locally and fast, and test on Colab whatever passes well. No task ID; this screen informs EVAL-014 (parked) and EVAL-016.

## What was run

`code/analysis/solution_screen.py` routes among outputs that were already synthesized and scored, on the 360 EXP-012 Quick-set cases (60 development speakers × clean, 25, 22, 20 and 18 dB white noise, and noisy15). The sources are:
- **EXP-018:** whole estimate at Steps 1–8, and drift.
- **EXP-017:** student alone.
- **EXP-020:** Voice-Swap, prompt SNR and pitch variety.
- **EXP-022:** slow part and pitch variety.

It compares 16 solutions:
- fixed handovers;
- the student alone;
- SNR only;
- SNR + drift (Definition 3.4);
- STAC;
- SNR routing that sends noisy prompts to the student and clean-like prompts to one of several teacher handovers.

The script fixes the pass rules in its docstring, and they were set before it was run. The reference is the whole estimate at Step 8:
- **Clean:** the capped-WER difference's upper 95% bound is at most +0.01, and the similarity and UTMOS lower bounds are at least −0.01 and −0.10. There are at most 3 of 60 Definition 3.1 failures.
- **Degraded:** the capped-WER upper bound is at most +0.025, with the same similarity and UTMOS bounds.
- **Cost:** at most 14.7 mean passes.
- **Pitch variety:** clean CVf0 minus the reference is at least −0.01, wherever every routed output has stored CVf0.

Thresholds are fitted out of fold over 5 speaker folds. The bootstrap is over speakers (B = 2000, seed 2026), with the same resamples for every solution.

## Results (exploratory, development data, white noise only)

- **Only Voice-Swap at Step 4 passes every rule:**
  - 14.00 mean passes;
  - 2 of 60 clean failures;
  - degraded WER +0.000 [−0.019, +0.016], similarity +0.062 and UTMOS +1.08 against the reference;
  - clean CVf0 −0.0050.
- **Three cheaper solutions fail narrowly.** Each is SNR routing that sends noisy prompts to the student alone:

  | Clean-like prompts go to | Mean passes | Rule missed |
  |---|---|---|
  | whole estimate at Step 4 | 5.78 | clean CVf0 −0.0107 |
  | Voice-Swap at Step 4 | 5.97 | clean CVf0 −0.0105 |
  | slow part at Step 3 | 5.58 | clean WER upper bound +0.011 |

  Their degraded-prompt quality beats the reference. The pitch-variety misses come from the 3 clean prompts that the SNR rule sends to the student.
- **SNR + drift (Definition 3.4) costs the same as SNR only:** 19.43 against 19.42 mean passes.
- **STAC fails under the same calibration as the other solutions.** It costs 15.06 mean passes and fails:
  - degraded quality (similarity lower bound −0.014);
  - clean quality;
  - the cost rule.
- **The student alone fails** clean failures (8 of 60) and pitch variety (−0.043).
- **The Decision Handover candidates cannot be screened** because no teacher estimates were stored.

## Round 2 (same day, after round 1 was seen)

The requester asked for more solutions, so 20 were added to the same script under the same pass rules. They were designed after round 1 on the same data, so a round-2 pass is weaker evidence than a round-1 pass. The new ideas were:
- three-way SNR routing, where prompts just below the threshold go to Voice-Swap instead of the student;
- strict SNR, where only prompts below the lowest clean SNR of the training folds go to the student;
- drift-protected routing;
- payload routing at Step 4;
- STAC over Voice-Swap and over the slow part;
- a learned SNR table;
- the other Step-4 part swaps;
- a hindsight row.

**Passed:**

| Solution | Mean passes (clean / degraded prompts) | Clean CVf0 |
|---|---|---|
| Three-way SNR, 2 dB band: whole estimate Step 4 / Voice-Swap Step 4 / student | **7.05** (12.7 / 5.9) | −0.0052 |
| Strict SNR → student, else whole estimate Step 4 | 8.20 (12.8 / 7.3) | −0.0007 |
| Three-way SNR, 4 dB band: whole / Voice-Swap / student | 8.66 | −0.0052 |
| Strict SNR → student, else Voice-Swap Step 4 | 8.67 (13.8 / 7.6) | −0.0050 |
| Three-way SNR, 4 dB band: Voice-Swap / Voice-Swap / student | 8.86 | −0.0050 |
| Drift-protected SNR → student, else Voice-Swap Step 4 | 12.73 | −0.0050 |
| STAC over the slow part, Steps 2–7 | 13.74 (10.5 / 14.4) | −0.0060 |
| Payload routing at Step 4: whole / Voice-Swap | 13.80 | −0.0052 |

All of these keep 2 or fewer of 60 clean failures, and their degraded-prompt quality beats the reference.

**Failed:**
- the learned SNR table: 5.33, clean WER upper bound +0.011;
- STAC over Voice-Swap: 16.05, which misses the cost rule;
- drift-protected routing with the whole estimate: 11.94, degraded WER upper bound +0.043;
- every other Step-4 part swap, fixed or routed: clean failures, pitch variety or degraded quality.

**Hindsight:** the best stored output per case gives 5.64 mean passes. This is not a solution.

None of these is novel by the requester's standard:
- **The passing ones** route with SNR or STAC among outputs that already exist.
- **The three-way and strict rules** fix round 1's near misses, where clean prompts were sent to the student and lost pitch variety.

## Round 3 (same day, after round 2)

The requester asked for a novel method tested locally. This PC has no PyTorch, TTS packages or checkpoints, and the stored results hold only scores, so no new handover can be synthesized here. Round 3 therefore replays only decision logic, with stored outputs standing in for the handover:
- **Candidate 4's stop rule (Decision Commitment Handover):**
  - **The Step-1 "empty decision" test replays exactly.** It is the stored d16_1 = ‖T₁ − T₀‖ / ‖T₁‖.
  - **The Step-2 "settled" test does not.** It needs the decision's direction, so d₂ / d₁ stands in for it.
- **A new stop formula, the geometric tail:** the predicted remaining change after Step k, d_k·r/(1 − r) with r = d_k / d_{k−1}.

**First run:** no rule found any feasible threshold, because the fit kept EXP-018's single-case guard (no case more than 0.25 capped WER above the reference). The pass rules have no such guard. The round-3 fits were then rerun without it. That change was made after the first run.

**Results:**
- **The geometric tail fails.** The teacher's drift grows over the early sway steps (median d₂/d₁ 3.2–4.6 in every condition), so the formula predicts no early stop. Its best version, over Voice-Swap Steps 4–7, cost 14.89 mean passes and missed the cost rule.
- **Candidate 4's logic is inconclusive offline.** The fit has no pitch-variety constraint, so it sent almost every input to the student. That cost about 8 mean passes and failed clean failures (8 of 60) and pitch variety (−0.043). The drift ratio is no substitute for the decision's direction, which the EXP-023 pilot records (`dc`), together with the decision's size relative to the draft (`dr`).
- **One supportive observation:** the Step-1 decision is larger on clean prompts (median d16_1 0.274) than on degraded ones (0.163–0.178). The ranges overlap: the clean minimum is 0.059, and the degraded maxima are 0.32–0.35.

## Round 4 (same day): Two-Axis Handover, after a literature check

The requester asked for proposed algorithms, a literature check of each, and a local test of any that were not found. Targeted web searches on 2026-11-13, not a full review:

| Algorithm | Closest work found | Verdict |
|---|---|---|
| A policy choosing both when and what crosses | [Starodubcev et al.](https://arxiv.org/pdf/2312.10835) decide per sample whether the teacher refines a student sample, with no component choice. [OPTD](https://arxiv.org/pdf/2608.02942) chooses how many teacher transitions to compress. | not found |
| Routing noisy-prompt inputs to the distilled student by prompt SNR | noise-robustness studies of flow-matching zero-shot TTS ([Interspeech 2024](https://www.isca-archive.org/interspeech_2024/wang24v_interspeech.pdf)); distilled zero-shot TTS ([StyleTTS-ZS](https://arxiv.org/pdf/2409.10058), [ZipVoice](https://arxiv.org/pdf/2506.13053)) | not found |
| Per-sample switching once the trajectory has committed | [Gandikota and Bau](https://arxiv.org/abs/2503.10637) use a fixed first-step switch; a third-party summary lists adaptive switching as future work | not found |
| A three-tier router with a partial-transfer middle tier | building blocks only, such as a query-level teacher router for LLM distillation ([PerSyn](https://arxiv.org/html/2510.10925v1)) | not found |

Each component of these policies is known on its own.

**Two-Axis Handover (tested here):**
- **Axis 1, what crosses (0 passes):** the prompt SNR decides.
  - clean-like prompts → the whole estimate;
  - prompts within 2 dB below the threshold → Voice-Swap at Step 4;
  - the rest → the student alone.
- **Axis 2, when (0 passes):** on the whole-estimate branch, STAC's action decides. Hand over at an early step if its stage threshold is met, otherwise at Step 4.
- **Calibration:** the rule was written into the script before the first run. It mirrors the pass rules on means, per branch, out of fold.

**Results (exploratory, development data):**
- **Early exit at Step 3 only: passes every measured rule.**
  - **Cost:** 6.73 mean passes, against 7.05 for three-way SNR. That is 11.1 on clean prompts and 5.9 on degraded ones.
  - **Clean quality:** WER +0.001 [−0.004, +0.006], similarity +0.001, UTMOS −0.032 [−0.087, +0.020], 1 of 60 clean failures.
  - **Degraded quality:** WER −0.026, similarity +0.075, UTMOS +1.10.
  - **Routing:** 48 of 60 clean prompts exit at Step 3 and 9 stay at Step 4. Fixed Step 3 failed clean quality in round 1, so the action-based axis adds value.
  - **Pitch variety: unmeasured.** No CVf0 was stored for the whole estimate at Step 3.
- **Early exit at Steps 2 or 3:** 6.53 mean passes, but fails clean quality. Its UTMOS lower bound is −0.110, against −0.10.
- **4 dB band:** 8.14 mean passes, fails clean quality.
- **With EXP-018's single-case guard:** the same as without it.

**What testing it on new speakers needs** (not in EXP-021): the whole estimate at Step 3, the teacher's per-step drift d₁–d₃, and pitch variety for the Step-3 and Step-4 handovers.

## Round 5 (same day): Agreement-Routed Handover, after a literature check

The requester asked for one last candidate through the same process.

**Proposed: the Agreement-Routed Handover (ARH).** What crosses is decided by whether the teacher and the student agree on the voice part, each band's average over time, rather than by the prompt SNR. Any recording condition the teacher copies shows up in that part, so the rule covers noise, reverberation and telephone band-limiting, while SNR covers noise only. The gap is measured at Step 0, between T₀ and the student's draft S. T₀ is in theory the same for every seed, so the gap reflects the input, not the seed. When is decided as in the Two-Axis Handover.

**Literature check (targeted, 2026-11-13).** Two searches found no method that routes inputs by teacher-student disagreement on the voice, or compares a teacher's first-step estimate with a student's output to decide what to transfer.

The closest work:
- **[StyleTTS-ZS](https://arxiv.org/pdf/2409.10058):** compares teacher and student speaker similarity, but only to evaluate distillation.
- **[Dysarthric-speaker TTS](https://arxiv.org/pdf/2508.10412):** uses a teacher speaker encoder as a training-time filter.
- **[Self-distilled disentanglement](https://arxiv.org/pdf/2501.08566):** uses a fixed real/synthetic mixing ratio.

**Local test.** Only EXP-020's Step-4 gap (`voice_rms`) is stored, so it stands in for the Step-0 gap.
- **Passes charged (the Step-0 method's):** whole estimate +1 for the draft; Voice-Swap 14; student alone 4 + 2 for T₀.
- **Cuts (out of fold, set before the run):**
  - whole estimate up to the gap that keeps 95% of training clean prompts;
  - Voice-Swap up to the largest training clean gap;
  - the student alone beyond that.

**Results:**
- **The main version passed every rule, pitch variety included:**
  - **Cost:** 7.60 mean passes, 13.9 on clean prompts and 6.3 on degraded ones.
  - **Clean quality:** WER −0.004 [−0.010, +0.000], UTMOS +0.002 [−0.045, +0.050], 3 of 60 clean failures.
  - **Degraded quality:** WER −0.026, similarity +0.077, UTMOS +1.20.
  - **Pitch variety:** clean CVf0 −0.0052.
  - **Routing:**
    - clean: W4 56, Voice-Swap 3, student 1;
    - 25 dB: student 49, Voice-Swap 9, W4 2;
    - 22 and 20 dB: student 59 each, Voice-Swap 1 each;
    - 18 dB and noisy15: all to the student.
  - **No Step-3 exits:** the Step-3 fit found no safe exit. Its whole-estimate branch holds almost no degraded prompts, and the fit stops at the first failing threshold.
- **The variant without the Voice-Swap band** cost 7.29 mean passes and failed pitch variety (−0.0107).
- **The variant allowing Step-2 exits** was the same as the main version.

**Compared with the Two-Axis Handover (6.73),** ARH costs 0.9 more mean passes on white noise. It needs no SNR estimator. Its claimed advantage, other recording conditions, cannot be tested on these white-noise data.

## Round 6 (same day): Variety-Value Routing, derived from a theory, negative

The requester asked for one more candidate derived from a theory, like ARH.

**The theory.** DMOSpeech 2's student is distilled by distribution matching, which is mode-seeking. Such a student should lose pitch variety only on inputs where the teacher's output varies. The teacher's decision after Step 1, d16_1 = ‖T₁ − T₀‖ / ‖T₁‖, estimates that variation for one seed.

**Check of the prediction, before the rule was built** (20 clean diversity prompts; variety gap = stored reference CVf0 minus student CVf0):
- **Supportive:** Spearman(gap, d16_1) = +0.42.
- **Not supportive:** Spearman with d16_2 −0.35, with A₃ −0.10, and with the prompt SNR +0.05.
- **Every prompt loses something:** the smallest gap is +0.0067, so the student loses some variety on every clean prompt.

**Literature check (targeted, 2026-11-13).** Two searches found no per-input teacher call triggered by the teacher's early deviation. The closest work:
- **[Diversity-Preserved DMD](https://arxiv.org/html/2602.03139):** trains the first step with a teacher-derived objective. It names choosing per prompt "by teacher uncertainty" as future work.
- **[ReSeed](https://arxiv.org/abs/2610.37110):** a trained first-step adapter.
- **[Gandikota and Bau](https://arxiv.org/html/2503.10637):** a fixed first-step switch.

**The rule (out of fold), on top of ARH.**
- **Large Step-0 voice gap:** the student alone (6 passes).
- **Otherwise:** the teacher runs Step 1.
  - **d16_1 < λ_v:** the student finishes from its draft (8 passes).
  - **Otherwise:** ARH's handover at Step 4 (14).
- **λ_v:** the largest threshold whose training cases keep the axis rule on means, and whose training clean diversity prompts keep stored CVf0 within −0.01.

**Result: fails.**
- **Cost:** 7.28 mean passes, 12.8 on clean prompts, against ARH's 7.60 and 13.9.
- **Clean failures:** 5 of 60, over the limit of 3.
- **Pitch variety:** −0.0123.
- **Why:** 11 clean prompts were stopped after Step 1. On held-out folds that cost more pitch variety than the training folds allowed, and added the student's own failures.

**What remains.** The pilot records the theory's sharper quantity: the teacher's decision relative to the student's own distance from T₀ (`dr_g_1`). It can be re-examined there, descriptively. The rule is not added to EXP-023.

## Round 7 (same day): theory-derived ideas after Variety-Value Routing, negative

The requester asked for another theory-derived candidate. Three theories were examined.

- **Speciation (a peak in the rate of change marks commitment): no adaptivity on these data.** Every stored drift profile peaks at Step 2 in every condition. In clean prompts the peak is at Step 2 in 42 of 60 and Step 3 in 17; in degraded prompts, at Step 2 in nearly all. "Hand over after the peak" therefore reduces to a fixed Step 3. What differs is the collapse after the peak: median d₃/d₂ is about 0.81 on clean prompts and 0.36 on noisy15. That is another cleanliness signal, but it needs three teacher steps where SNR or ARH need none.
- **Restart noise: no new candidate.** The released student loop draws fresh noise at every level (seed + τ·100), so the teacher's own noise is discarded at the handover. Reusing it would amount to handing over the teacher's state, which T-Stitch and Gandikota and Bau already do. It would also need synthesis.
- **Commitment-Timed ARH (tested).** On ARH's agreement branch, where prompt noise has already been routed away, the rule was: hand over at Step 3 if d₃/d₂ ≤ λ_r, otherwise the whole estimate at Step 4. The Step-3 payload was either the slow part (pitch variety stored) or the whole estimate.
  - **The pre-specified fit made no Step-3 exits.** The few degraded prompts on the branch collapse fastest, as the theory predicts, so they are scanned first, and each fails the degraded mean. The result was identical to ARH: 7.60 mean passes.
  - **A fit on the branch's clean cases, added after that run, made exits for nearly everyone:** 55 of 60 clean prompts with the slow part, 54 with the whole estimate.
    - **Slow part:** 7.28 mean passes, but fails the clean WER bound (+0.011).
    - **Whole estimate:** 7.29 mean passes, passes the measured rules; pitch variety at Step 3 unmeasured.
  - **The commitment signal does not discriminate in either case,** so the saving comes from handing over at Step 3, not from the rule. It is not added as a candidate, and no literature check was run.

The pitch variety of the whole estimate at Step 3 will be measured in EXP-023 (`cvf0_W3`, under G3). That also tells whether ARH can hand over at Step 3.

## Round 8 (same day): theory-derived candidates that need the pilot's signals

The requester asked for more. Every signal stored here had now been used, so the next theory-derived candidates need measurements only EXP-023 records.
- **GAR (Guidance-Amplification Routing).** Classifier-free guidance amplifies what the prompt imposes, so the guided and conditional-branch Step-0 voice gaps should separate copied recording conditions from model mismatch. A literature check found no such router.
- **Collapse-Gated Handover.** This is round 6's theory with its exact quantity.

Both, and the candidate-4 replay, were pre-specified in `code/analysis/exp023_followups.py` before the run. See the EXP-023 log.

**Two more theories, the same day, also pre-specified there:**
- **PAS (alignment theory):** the student's Step-0 prosody gap to T₀ flags misaligned drafts on ARH's student branch.
- **GCH (early-guidance theory):** the Step-0 ratio `vg_g / vg_u` chooses the guided or unguided teacher's decision.

A restart-visibility stop was dropped, because its supposedly parameter-free threshold depends on an unknown pooling constant. Neither PAS nor GCH can be tested on the stored data, which hold neither Step-0 gap.

## Consequence

- **Round 1's passing solution already has a Colab test:** EXP-021, which is parked.
- **EXP-021 also covers the round-2 winners that use only Voice-Swap and the student.** It records `prompt_snr`, `plain4` (the whole estimate at Step 4), `vs` and `student` on every case: 40 new speakers × clean, noisy15, gauss20, babble15, reverb09 and telephone. Its pitch variety covers `ref`, `vs` and `student`, but not `plain4`.
- **Those checks are exploratory unless frozen first.** That would need EXP-021's preregistration amended before synthesis, with the SNR thresholds fixed from the 60 Quick-set speakers.
- **The novel candidates' Colab test is EXP-023.**
- **EXP-023 results (same day).** The Two-Axis Handover's Step-3 pitch variety held (G3 passes, −0.0013), so it advances to a frozen test on new speakers. The Agreement-Routed Handover failed with the real Step-0 gap. Its AUC was 0.951, against 0.999 for the screen's Step-4 stand-in, and its largest-clean-gap cut was fragile, giving 23 misroutes against 0 for SNR. See the [EXP-023 results log](2026-11-13-exp023-decision-handover-results.md).
- **Rounds 4 and 5 are now part of EXP-023.** The Two-Axis Handover (G3, pitch variety at Step 3) and the Agreement-Routed Handover (G4, routing by the real Step-0 voice gap against the SNR rule) were added by amendment before any synthesis (preregistration `469c90ed...`). Both are on development speakers with white noise only, so a positive result leads to a frozen test on new speakers. For the Agreement-Routed Handover, that test should include reverberant and telephone prompts.

## Checks run

- **`python code/analysis/solution_screen.py`:** completed in 11 s.
- **`python scripts/check_workspace.py`:** PASSED.
- **`python -m unittest discover -s code/tests`:** OK, 169 tests.

## Remaining blockers

- **The data designed these methods.** The 60 speakers were used to design them, so a pass here only earns a test on new speakers.
- **The rules are not a frozen protocol.** They are this screen's own.
- **Pitch variety is thin.** It rests on 20 speakers, and only for outputs whose CVf0 was stored.
