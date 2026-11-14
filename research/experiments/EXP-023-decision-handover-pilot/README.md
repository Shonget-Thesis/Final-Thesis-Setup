# EXP-023: Decision Handover (development pilot)

**Status:** run on 2026-11-13 on a Colab Tesla T4, with 60 of 60 cases, 20 of 20 seed pairs and 10 of 10 diversity prompts. The results are in [results/](results/MANIFEST.json) with SHA-256 hashes; see the [results log](../../../docs/progress/log/2026-11-13-exp023-decision-handover-results.md). Task EVAL-016 (review). Preregistered 2026-11-13 and amended three times the same day, all before any synthesis (`469c90edf7c6094d...`; chain `79752afb...` → `03006f5c...` → `59040438...` → `469c90ed...`).

**Results (development data, white noise):**
- **Decision Handover does not advance.**
  - **G1 fails:** the teacher's change after Step 0 is mostly shared across seeds. The shared share was 0.71–0.83 on clean prompts with the guided teacher, against 0.50 for independent decisions. So it is not this seed's decision.
  - **G2 fails for all six outputs:** the guided versions raise WER by up to +0.116, and the unguided ones lose pitch variety (−0.027 to −0.038).
- **The Two-Axis Handover advances (G3 passes).** Its clean pitch variety is −0.0013 [−0.0042, +0.0010] against the reference. The Step-3 handover alone is −0.0009.
- **The Agreement-Routed Handover does not advance (G4 fails).** The real Step-0 voice gap misroutes 23 of 60 cases, against 0 for SNR.
- **None of the five pre-specified follow-ups is promising.**
- **A pilot:** it decides whether a frozen test on new speakers is worth running, and cannot support a claim.
- **Not adopted:** nothing here is adopted by the thesis, and no adviser has reviewed it.
- **Three candidates:**
  - Decision Handover;
  - the [Two-Axis Handover](#second-candidate-two-axis-handover);
  - the [Agreement-Routed Handover](#third-candidate-agreement-routed-handover).

  Nothing about Decision Handover changed in the amendments.

- Runner: [run_decision_handover_pilot.py](../../../code/colab/run_decision_handover_pilot.py)
- Notebook: [DMOSpeech2_EXP023_Decision_Handover_Pilot.ipynb](../../../code/colab/DMOSpeech2_EXP023_Decision_Handover_Pilot.ipynb)
- Tests: [test_decision_handover.py](../../../code/tests/test_decision_handover.py)
- Preregistration: [preregistration.json](preregistration.json)

## Idea

In flow matching, the guided teacher's first endpoint estimate T₀ (Step 0) is in theory the same for every random seed. It is the average output for this text and speaker, including whatever the prompt's recording conditions add. Everything the teacher adds after that, T_k − T₀, is specific to this seed: its **decision**.

That decision is what the student lacks:
- **Prosody is what the student loses.** The DMOSpeech 2 paper reports that the student's loss is mainly prosody.
- **Everything else, the student matches.** On this project's development data (EXP-017, EXP-018 and EXP-020), the student alone matches or beats the 0.25 reference on WER, speaker similarity and UTMOS in every prompt condition, and the reference's only measurable extra is pitch variety on clean prompts.

So the handover passes only the decision, on top of the student's own one-pass draft S:

**x_k = S + slow(T_k − T₀)**

slow() is the prosody band of EXP-020's split. Three properties follow, with no detector and no SNR estimate:
- **Shared contamination cancels.** Anything the teacher adds identically for every seed cancels in the difference, including the prompt's recording conditions.
- **Voice-Swap is a special case of the principle.** The voice part is shared across seeds, so its difference is zero and the student's voice is kept.
- **A stop signal can be measured on the decision itself.** The earlier signals (drift, slow-part change) divided by the whole estimate, which includes T₀ and the prompt's noise. That may be why noise lowered drift in EXP-012; this is untested.

## Outputs

| Output | Handover input on generated frames | Teacher | Passes |
|---|---|---|---:|
| DH_2, DH_3 | S + slow(T_k − T₀) | guided, as released | 10, 12 |
| DHc_2, DHc_3 | voice(S) + fine(S) + slow(T_k) + g_k (slow(S) − slow(T₀)) | guided | 10, 12 |
| DHu_2, DHu_3 | S + slow(T_k − T₀) | conditional branch only, 1 pass per step | 7, 8 |

For comparison: the 0.25 reference costs 21 passes and Voice-Swap 14.
- **The family:** with g = 0, the formula is EXP-022's slow-part handover (STS); with g = 1, it is DH.
- **DHc's weight:** g_k = √(1 − ρ_k), the variance-preserving completion. ρ_k is the share of the reference's extra pitch variety, over the student alone, that STS reached at Step k in EXP-022: ρ₂ = 0.639 and ρ₃ = 0.941, so g₂ = 0.601 and g₃ = 0.242. It protects against counting the student's prosody twice.
- **DHu:** the teacher only has to decide, and guidance reduces diversity, so it decides without guidance at half the teacher cost.

## What is measured

- **Assumptions (teacher only):** 10 speakers × clean and noisy15 × 4 seeds, guided and unguided, Steps 0–4.
  - **Spread:** of each part of T_k across seeds, at each step.
  - **Shared share:** how much of the decision every seed shares.
- **Quality:** 20 speakers × clean, Gaussian 20 dB and 15 dB (60 cases). The six outputs are scored against the stored 0.25 reference (EXP-018), STS at the same step (EXP-022), Voice-Swap (EXP-020) and the student alone (EXP-017), all on the same cases and seeds.
- **Pitch variety:** CVf0 of the six outputs on 10 speakers × clean × 20 seeds. The seeds are EXP-020/EXP-022's, so the stored CVf0 of the reference, STS and the student are comparable. Since the amendment, the same block also measures the Two-Axis Handover's Step-3 and Step-4 handovers and its per-seed choice between them.
- **Decision signals per case (descriptive):** the decision's size, its turning between steps, its size relative to the draft's distance from T₀, and the Step-0 voice gap between T₀ and S.

## Go / no-go (frozen)

| Rule | Requirement |
|---|---|
| G1, first estimate shared | guided teacher, slow part: median over pairs of (spread across seeds at Step 0) / (spread at Step 3) at most 0.25, on clean and on noisy15 |
| G2, per output | passes at most 12; pooled capped-WER minus the reference, upper bound at most +0.025; similarity minus the reference, lower bound at least −0.01; mean WER minus STS at the same step on degraded prompts at most 0; clean CVf0 minus the stored reference, mean at least −0.01 |

A frozen test on new speakers follows only if G1 holds and at least one output meets G2. The output carried forward is the cheapest one that meets G2 (ties: lower pooled WER). Otherwise this is a negative pilot, with no retuning on these data.

| Rule (added by the amendment) | Requirement |
|---|---|
| G3, the Two-Axis Handover keeps pitch variety | on the 10 clean diversity prompts, its CVf0 minus the stored reference, mean at least −0.01. Its CVf0 is its per-seed choice between the Step-3 and Step-4 handovers on the whole-estimate branch, or the stored Voice-Swap or student-alone CVf0 on the other branches. |

| G4, the Agreement-Routed Handover routes as well as SNR | out of fold over 5 speaker folds, on the 60 quality cases. A misroute is a clean case sent to the student alone, or a degraded case sent to the whole estimate. Requirements: misroutes by the Step-0 voice gap at most 1 more than by the prompt-SNR rule on the same cases and folds; at most 14.7 mean passes; clean CVf0 minus the stored reference, mean at least −0.01. Absolute quality is reported, not judged. |

The Two-Axis Handover advances to a frozen test on new speakers only if G3 holds, and the Agreement-Routed Handover only if G4 holds. Both decisions are independent of G1 and G2.

## Second candidate: Two-Axis Handover

Added on 2026-11-13 by amendment, before any synthesis, at the requester's request. It came from the offline screen ([solution_screen.py](../../../code/analysis/solution_screen.py), round 4; [log](../../../docs/progress/log/2026-11-13-offline-solution-screen.md)).

| Axis | Signal (0 passes) | Decision |
|---|---|---|
| What crosses | prompt SNR | at or above 26.55 dB: the whole estimate; within 2 dB below: Voice-Swap at Step 4 (14 passes); below that: nothing, the student alone (4 passes) |
| When (whole-estimate branch) | the teacher's movement by Step 3, A₃ = d₁ + d₂ + d₃ (the stored drift of EXP-018) | A₃ ≥ 1.2154: hand over at Step 3 (11 passes); otherwise Step 4 (13 passes) |

- **Frozen parameters.** They are fitted on the stored data of the 60 Quick-set speakers with the screen's rules, applied to all speakers at once:
  - **The SNR cut** keeps 95% of the clean prompts on the whole-estimate branch.
  - **The action threshold** is the most permissive one that keeps the mean quality rule.
- **On stored data:**
  - **Out of fold** (the screen): 6.73 mean passes, and every measured rule passed.
  - **In sample:** 7.16 mean passes, 10.8 on clean prompts, with 0 clean failures. The clean UTMOS lower bound is −0.098, just inside the screen's −0.10.
- **What this pilot adds.** The Step-3 handover's pitch variety was never stored, so it is the one rule not yet measured. The diversity block now also runs:
  - the whole-estimate handovers at Steps 3 and 4 (the teacher runs to Step 4);
  - each seed's A₃, computed with the sampler's own drift function;
  - the routed choice between those two handovers for each seed.

  Step 4 is also checked against EXP-020's stored value on the same seeds.
- **Novelty.**
  - **Not found:** targeted searches found no published policy that chooses both when and what crosses.
  - **Known parts:** SNR routing, STAC's action and Voice-Swap are each known.
- **Weak evidence.** It was designed after three earlier screen rounds on the same speakers, so it needs a frozen test on new speakers.

## Third candidate: Agreement-Routed Handover

Added on 2026-11-13 by amendment, before any synthesis, at the requester's request as the last candidate. It came from the offline screen's round 5.

| Step | What happens | Passes |
|---|---|---:|
| 1 | The teacher's first estimate T₀ and the student's one-pass draft S | 3 |
| 2 | Voice gap: RMS difference of their voice parts (each band's average over time) on generated frames | 0 |
| 3 | Agreement (gap up to the cut that keeps 95% of clean prompts): the whole estimate at Step 4 | 14 in all |
| | A small disagreement (up to the largest clean gap): Voice-Swap at Step 4 | 14 in all |
| | A large disagreement: the student finishes alone from S | 6 in all |

- **Why it differs from the Two-Axis Handover.**
  - **Agreement, not SNR:** it asks whether the two models agree on the voice, instead of estimating the prompt's SNR. Any recording condition the teacher copies, whether noise, reverberation or a telephone band, shows up in the voice part, while SNR sees noise only.
  - **Measured on T₀:** it compares the teacher's first estimate, which is in theory the same for every seed, so the gap reflects the input, not the seed.
- **On stored data** (screen round 5, with EXP-020's Step-4 gap standing in for the Step-0 gap):
  - **Cost:** 7.60 mean passes out of fold.
  - **Result:** it passed every rule, with clean CVf0 −0.0052.
  - **No Step-3 exits:** the screen found no safe Step-3 exit for its whole-estimate branch, so it hands over at Step 4.
- **What this pilot adds: the real Step-0 gap.** The quality block already records it for every case (`vg_g`), so G4 needs no extra synthesis. G4 compares its routing with the prompt-SNR rule on the same cases and folds.
- **Why G4 judges routing, not quality.** An earlier G4 (`59040438...`) used absolute quality bounds. A check before any synthesis showed it could not pass on these 20 speakers even with perfect routing: the stored whole-estimate outputs at Step 4 give a clean UTMOS lower bound of −0.132, against −0.10. So it was replaced before the run.
- **What this pilot cannot show.** Its claimed advantage over SNR (reverberation, telephone band) needs prompt conditions this white-noise pilot does not have.
- **Novelty.**
  - **Not found:** targeted searches found no method that routes inputs by teacher-student disagreement on the voice, or compares a teacher's first-step estimate with a student's output to decide what crosses.
  - **Known parts:** Voice-Swap and the student alone are known.

## Pre-specified follow-ups (exploratory, not part of the go/no-go)

[exp023_followups.py](../../../code/analysis/exp023_followups.py) was written on 2026-11-13, before the pilot ran, at the requester's request for more theory-derived candidates. It uses only signals the pilot records and the stored outputs of the same cases, so it needs no synthesis and leaves the frozen preregistration unchanged. Its rules are fixed in the file; run it on the results folder after the pilot. A "promising" result leads only to a frozen test on new speakers.

| Follow-up | Theory | Rule (out of fold, the pilot's 5 speaker folds) | Passes |
|---|---|---|---|
| **GAR (Guidance-Amplification Routing)** | classifier-free guidance amplifies whatever the prompt imposes (w = 2), but not plain teacher-student mismatch | ARH's cut rule on the conditional-branch Step-0 voice gap `vg_u` (GAR-u), or on the ratio `vg_g / vg_u` (GAR-ratio) | GAR-u 14, 14, 5 |
| **CGH (Collapse-Gated Handover)** | the mode-seeking student collapses where the teacher varies | on ARH's teacher branches, the student finishes alone if `dr_g_1` (the teacher's Step-1 decision relative to the student's distance from T₀) is small | 8 when gated |
| **Decision Commitment Handover replay** | candidate 4 | `dr_g_1` ≤ l0: student alone; else `dc_g_2` ≥ lc: DH_2; else DH_3 | 8, 10, 12 |
| **PAS (Prosody-Agreement Safeguard)** | alignment theory: word timing is set by the text and prompt, so it is already in the teacher's average T₀ | on ARH's student branch, a Step-0 prosody gap ‖slow(S) − slow(T₀)‖ (RMS, from `dn_g_1 / dr_g_1`) above the cut that keeps 95% of clean prompts sends the case to Voice-Swap | +8 for each case it moves |
| **GCH (Guidance-Choice Handover)** | early-guidance theory: guidance at the first steps pushes past the conditional mean and amplifies what the prompt imposes | at Step 0, `vg_g / vg_u` ≥ λ: the unguided teacher decides (DHu_3); otherwise the guided one (DH_3) | 9, 12 |

**When a follow-up counts as promising:**
- **Routing rules:** at most 1 misroute more than the SNR rule and no more than ARH, at most 14.7 mean passes, and pitch variety kept.
- **The replay:** the screen's quality rule on means, at most 12 mean passes, and pitch variety kept.
- **PAS:**
  - fewer Definition 3.1 failures on degraded prompts than ARH;
  - degraded WER no higher than ARH's;
  - at most 1 mean pass more than ARH;
  - pitch variety kept.
- **GCH:**
  - the quality and pitch rules;
  - under 12 mean passes;
  - DHu_3 alone must fail one of those rules, otherwise the cheaper fixed choice suffices.

**Literature checks for PAS and GCH** (targeted, 2026-11-13): neither was found.
- **Closest to PAS:** mean-velocity distillation in [dots.tts](https://arxiv.org/pdf/2606.07080), and teacher alignment in CTC distillation ([Oracle Teacher](https://arxiv.org/pdf/2111.03664)). No TTS method compares the student's draft with the teacher's average to catch timing failures.
- **Closest to GCH:** [TamedFlow](https://arxiv.org/html/2610.26025)'s teacher rescue, and [Rethinking CFG in on-policy distillation](https://arxiv.org/pdf/2607.24731). No per-input choice between guided and unguided teacher decisions was found.

**Literature check for GAR** (targeted searches, 2026-11-13): no method found compares guided and unguided first estimates to separate prompt-induced from model-induced disagreement, or routes on it. Closely related work:
- **[First-step unguided sampling](https://arxiv.org/html/2611.11179):** in few-step flow-matching TTS, leaving the first step unguided restores speaker similarity, because a guided first step pushes trajectories beyond the conditional mean.
- **[CFG-Zero⋆](https://arxiv.org/html/2503.18886v1):** zeroes the first solver step.

Both also bear on DHu, whose teacher decides without guidance.

## Candidate methods built on this mechanism

Recorded on 2026-11-13 at the requester's request; none is tested. Every candidate has the same novel part, what crosses (the decision).
- **Candidates 1–3:** the rule that decides when to hand over is known, so combining the decision with SNR or STAC adds no novelty.
- **Candidate 4:** the rule measures the decision itself. That is a new signal for a known kind of decision.

| Candidate | When the teacher hands over | What crosses | What testing it needs |
|---|---|---|---|
| 1. Decision Handover | a fixed step, 2 or 3 | the decision | this pilot |
| 2. SNR + Decision Handover | the prompt SNR gate (Chapter 3) | the decision | this pilot's outputs, then an offline replay |
| 3. STAC + Decision Handover | STAC's summed drift with stage thresholds (Appendix E) | the decision | Decision Handover at Steps 1–7, one more run |
| 4. Decision Commitment Handover | when the decision has stopped turning: the student finishes alone if the decision is empty after Step 1; otherwise the teacher hands over at Step 2 if the decision has settled, or at Step 3 | the decision | this pilot's outputs, then an offline replay |

- **Candidate 4, designed from the stored results** (2026-11-13, at the requester's request):
  - **Design, one result per choice:**
    - **The student makes the speech,** because it matches or beats the reference on WER, similarity and UTMOS.
    - **Only the prosody decision crosses,** because that is the teacher's one measurable extra.
    - **The teacher stops by Step 3,** where its slow part already reaches 94% of the reference's extra pitch variety (EXP-022).
    - **An empty decision ends the teacher's run,** because noisy prompts suppress the teacher's guidance updates (EXP-018).
    - **The stop waits for the decision's direction to settle, not for small steps,** because the teacher's step changes turn (EXP-013) and settledness drift failed (EXP-012, EXP-018).
  - **Rule:**
    - **Empty decision:** ‖D₁‖ ≤ λ₀ ‖slow(S) − slow(T₀)‖ after Step 1, and the student finishes alone.
    - **Committed decision:** cos(D₂, D₁) ≥ λ_c at Step 2, and the student gets x = S + D₂.
    - **Otherwise:** x = S + D₃ at Step 3.
    - **Calibration:** λ₀ and λ_c are calibrated with LTT on quality and pitch variety.
  - **Passes:** empty at Step 1, committed at Step 2, or Step 3.
    - **Guided teacher:** 8, 10 or 12.
    - **Unguided teacher:** 6, 7 or 8.
    - **An optional SNR filter** in front sends noisy prompts to the student at 4 passes.
  - **Testing:** this pilot already records `dr` (the decision's size relative to the draft) and `dc` (its turning) per case for both teachers, plus Decision Handover outputs at Steps 2 and 3. The method can therefore be replayed offline when the results arrive. In that replay, the student-alone branch uses the stored student-alone output, whose seed differs from the pilot's draft.

- **Drift (Definition 3.4) is left out:**
  - **Stopping when the estimate has settled fails:** it certified nothing in EXP-012 and saved 0% in EXP-018.
  - **The same number used the other way round works:** stage-wise drift, d_k ≥ λ_k, saved 19.5% in EXP-018, and STAC builds on that.
- **Candidate 2, noisy prompts:**
  - **Student alone saves the most:** sending noisy prompts to the student alone would cost about 5.2–5.6 mean passes on the development mix, from an offline SNR routing replay on stored data ([payload_routing_screen.py](../../../code/analysis/payload_routing_screen.py)).
  - **But it fails the risk rule:** Definition 3.1 counts 51 of 300 student failures there, because it measures agreement with a reference that copies the noise (EXP-017).
  - **Chapter 3's fallback to the reference** keeps the guarantee but saves only on clean prompts.
- **Candidate 3, cost:**
  - **The draft adds a pass:** with STAC's current exits, the student draft adds one pass to every early handover (about 17.3 instead of 16.82 passes). It saves only if Decision Handover lets the thresholds drop.
  - **STAC's numbers have open problems:** its 19.9% point has 34 Definition 3.1 failures (9.4%, above the 5% limit), and Appendix E's 1D-CRC-STAC guarantee does not match `fit_crc_stac`.
  - **No extra teacher pass:** the decision is the running sum of the slow part of the teacher's step-to-step changes, whose relative sizes STAC already adds up, so only the student draft adds a pass.
- **Routing what crosses, screened offline** (2026-11-13, exploratory, development data, [payload_routing_screen.py](../../../code/analysis/payload_routing_screen.py)):
  - **The idea:** with the payloads already stored, SNR chooses what crosses at a fixed step, the whole estimate on clean prompts and the slow part or Voice-Swap on noisy ones. No handover router found chooses what crosses.
  - **The result:** it loses to sending noisy prompts to the student alone. At Step 3 it costs 11.80 mean passes against 5.38, and on degraded prompts its WER is +0.013 against the reference, where the student's is −0.026.
  - **When it would pay:** only if some payload beats the student on noisy prompts. The decision is the only one left to try, and this pilot measures it.
  - **The baseline to beat:** SNR sending clean prompts to the whole estimate at Step 3 and noisy prompts to the student costs 5.38 mean passes with 0 of 60 clean failures. This routing is known, but any candidate should be compared with it.

## How to run

Push the branch, open the notebook on a Colab T4 and run all cells. It takes about 1.5 to 2 hours by estimate, not measured; the amendment added two handovers per seed to the pitch-variety block.
- **Smoke test:** it must reproduce the stored STS_3 scores of EXP-022, and show that the pilot's own teacher loop matches the released guided teacher (max difference at most 1e-4), before the full run can start.
- **The runner refuses to start if** any of its seven code files, the preregistration or the stored input files have changed, if the GPU is not a T4, or if Shonget-DMOSpeech2 is not at `8a7868f`.
- **On the branch:** the EXP-022 results must be there, because the runner reuses them.
- **When it finishes:** send `DMOSpeech2_EXP023_Decision_Handover_Pilot_Results.zip`.

## Prior work and novelty

To our knowledge, no published handover passes only the teacher's seed-specific change onto the student's own sample. This rests on targeted searches on 2026-11-13, not a full literature review.

The closest work found:
- **[DiFA](https://www.alphaxiv.org/abs/2607.17972) (2026)** adds a deviation term, but inside one model's own trajectory, against its own recent predictions. It has no teacher, no student and no handover.
- **[Gandikota and Bau](https://arxiv.org/html/2503.10637v4) (WACV 2026)** restore a distilled model's diversity by running the base model for the first step only and passing the whole state. They report that zero guidance works best for that step.
- **[T-Stitch](https://arxiv.org/abs/2402.14167) (2024)** switches from a small to a large model along the trajectory and passes the whole state.
- **[Distillation++](https://arxiv.org/pdf/2412.08871) (ICCV 2025)** has the teacher refine the student's estimates during sampling.
- **[STRIDE](https://arxiv.org/html/2605.11494) (2026)** injects structured noise into a distilled model's features.
- **[DMOSpeech 2](https://arxiv.org/pdf/2507.14988)** hands over the teacher's whole estimate.
- **[First-step unguided sampling](https://arxiv.org/html/2611.11179) (2026)** leaves the first sampling step unguided in few-step flow-matching TTS to restore speaker similarity. It is close to DHu's choice of a teacher without guidance, though it does not hand over a decision. Found on 2026-11-13, after the preregistration.

The building blocks are known:
- **Differences that cancel what two predictions share:** classifier-free guidance is one. [Delta Denoising Score](https://arxiv.org/abs/2304.07090) (ICCV 2023) subtracts a reference branch's prediction so that the error both branches share cancels. That is the principle behind "shared contamination cancels" here; DDS applies it within one model, for image editing.
- **Differences across noise levels within one model:** [Self-Guidance](https://arxiv.org/abs/2412.05827) (TPAMI 2025) and DiFA.
- **Adding up differences from a shared baseline:** [Composable Diffusion](https://arxiv.org/abs/2206.01714) (ECCV 2022) combines conditions this way.
- **The first estimate is the conditional average:** a standard property of flow matching at t = 0, on which the Idea above rests.
- **Commitment at particular times:** the theory that diffusion trajectories commit at particular times ([speciation](https://arxiv.org/abs/2602.04404v1)).
- **Skipping guidance where diversity is decided.**

Further searches on 2026-11-13 found no method that transfers a base model's trajectory difference onto a distilled model's sample, and no inference-time teacher-to-student prosody transfer in flow-matching TTS.

In this project's handover taxonomy, Decision Handover is a new answer to "what crosses": a decision, not a component of the estimate. The claim is therefore narrow. The thing handed over is new, but the operations that build it are not, and a reviewer could call it a new combination of known differences. An adaptive stop on the decision would be a new signal for a known kind of decision (when to stop); it is measured here, not tested.

## Caveats

- **Development data:** the speakers designed EXP-020 and EXP-022, and the noise is white Gaussian only.
- **Shared first estimate:** T₀ may not be the same across seeds for the real teacher. EXP-022's voice part changes a lot over Steps 1–4 on clean prompts, which is a warning; G1 tests it.
- **Double counting:** the student's draft is already crisp, and T_k − T₀ also contains the teacher's sharpening away from its blurry average, so DH may exaggerate prosody. DHc is the corrected variant.
- **Timing:** adding the teacher's timing change to a draft with different timing may smear words.
- **Turning decisions:** EXP-013 found that the teacher's step-to-step changes turn direction, so the decision signals may be noisy.
- **Weights from the same data:** DHc's weights come from EXP-022's pitch data on development speakers.
- **Wide pitch intervals:** pitch variety on 10 speakers has wide intervals, so G2 uses its mean.
