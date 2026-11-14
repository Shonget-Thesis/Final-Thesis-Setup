# Presenter script

Read-aloud script for the animated walkthrough in [index.html](index.html). Timings are the auto-play durations of each scene (about 25 minutes in total).

## Scene 01 · Title

*Proposal · about 45 s*

Good day. Our proposal is titled "A Risk-Calibrated Adaptive Handover Method for Teacher-Guided Flow-Matching Text-to-Speech Inference." The line under the title is the whole idea in one picture. Speech is generated from noise at time zero to speech at time one. A large teacher model, in blue, does the first part of that journey, and a small, fast student, in grey, finishes it. The orange mark is the handover point. Today that point is fixed. Our study asks where it should be for each sentence, and how we can promise that moving it does not hurt quality.

## Scene 02 · Proposal Overview

*Proposal · about 45 s*

This is the path of the talk. In Chapter One we set up the background, show the problem with evidence from our own preliminary runs, and state our objectives and scope. Chapter Two is kept short: one comparison table of related methods and one diagram that shows the gap we fill. Chapter Three is most of the talk: the architecture, the decision signals and stopping rule, how we calibrate it with a statistical guarantee, the experimental plan, and finally our contribution. The appendix at the end holds our preliminary test results and supplementary material, which we can return to during questions.

## Scene 03 · Background: Flow Matching, Cost & Distillation

*Chapter I · Introduction · about 60 s*

Here is the background in four panels, read left to right. First panel: flow matching. The model learns a velocity field, dx over dt, that moves random noise toward a mel-spectrogram step by step. Every step is one full network evaluation, and F5-TTS, our teacher, works this way. Second panel: classifier-free guidance. Each guided step runs the network twice, once with the text condition and once without, so the cost doubles. Third panel: distillation. DMOSpeech trained a small student that needs four steps instead of 128. Fourth panel: the catch. The student's speech is less varied, which is called mode shrinkage, so DMOSpeech 2 lets the teacher run the first steps and the student finish. The bar at the bottom shows what that costs in the released code: 18 teacher passes and 3 student passes, 21 forward passes for every utterance, easy or hard.

## Scene 04 · Teacher-Guided TTS: Where This Study Works

*Chapter I · Introduction · about 65 s*

This diagram shows the whole system and where we work. On the left is training, done once and by others: the F5-TTS teacher, and a student distilled from it. We do not retrain or change either model. On the right, in the orange box, is inference, which runs for every utterance. Text and a voice prompt go in, the teacher takes its steps, then hands over at the orange diamond, the student finishes, and speech comes out. Notice the counts above the boxes: 18 teacher passes and 3 student passes. Now the view zooms into inference, and then into the handover itself. The top card is how it works today: one fixed switch, at time 0.25, for every utterance. The bottom card is our study: we pick the switch per utterance, and we bound the risk with a calibrated guarantee. That single decision is our entire focus; everything else stays exactly as released.

## Scene 05 · Problem Statement

*Chapter I · Introduction · about 80 s*

This slide states the problem. On the left, three kinds of input: a clean and easy one, a moderate one, and one with a noisy voice prompt. With a fixed handover, the dashed orange line, all three pay the same 21 passes. Switching too late wastes teacher passes; switching too early may degrade the speech. So we ask: does every utterance really need 18 teacher passes? The orange box shows that this is not a hypothetical. In our preliminary runs on the Tesla T4, with 500 speakers and a clean and noisy version of each prompt, every fixed early switch failed on a quarter to a third of inputs, almost entirely the noisy ones, while clean prompts failed only one to two percent of the time. Choosing the switch per input in hindsight would need under 15 of the 21 passes. No single early switch is safe for everyone. On the right are the four specific problems this leaves: the effects are not characterized, we do not know which signals can pick the switch, adaptive selection alone gives no reliability guarantee, and speed by itself is not evidence of benefit to end users.

## Scene 06 · Research Objectives

*Chapter I · Introduction · about 60 s*

Our general objective is to develop and evaluate a training-free, risk-calibrated method that chooses the teacher's switching time per utterance. Each card is one specific objective, with its output written underneath. One, characterize: measure how word error rate, speaker similarity, predicted naturalness, forward passes and runtime change across switching times and inputs, which gives us a quality-cost map. Two, formulate: state the choice as a cost minimization with a limit on risk, design a policy that needs no extra network evaluations, and prove its properties, four propositions in all. Three, calibrate: use Learn-Then-Test so that the rate of added failures stays within a tolerance alpha with confidence one minus delta, and check this on held-out speakers. Four, compare: benchmark the SNR-only, stabilization-only and combined variants against fixed switching times and an oracle, on clean and noisy prompts, to see whether each signal adds value.

## Scene 07 · Scope & Delimitations

*Chapter I · Introduction · about 65 s*

On the left is what is in and out of scope. In scope: only the switching time varies, and the student's restart level stays at the released default as a controlled setting. All four networks, the teacher, student, duration predictor and vocoder, stay frozen. We use English zero-shot prompts from the Seed-TTS test set, each in a clean and a 15-decibel noisy version, and everything runs on an NVIDIA Tesla T4 through Google Colab. Out of scope, in grey: the guarantee holds on average over inputs, not for every single utterance, and we do not build or test an assistive application or measure clinical outcomes. On the right are the beneficiaries: speech researchers get a formulation and proofs that apply to teacher-guided samplers in general; TTS developers get lower cost with a stated bound and no retraining; assistive-technology developers get cheaper, predictable synthesis; and future researchers get a reusable protocol.

## Scene 08 · Input-Adaptive Computation

*Chapter II · Review of related literature · about 55 s*

Chapter Two places our work among input-adaptive methods, ones that spend more or less computation depending on the input. Read the table by its last column, the guarantee. ACT, BranchyNet, adaptive teacher-student collaboration, T-Stitch and Adaptive Guidance all adapt something, from the number of layers to which model runs each step, but none of them states a bound on how much quality is lost. Two rows do carry risk control: CALM, which calibrates early exit in language models, and adaptive ASR correction, which uses Learn-Then-Test, but for speech recognition, not synthesis. The highlighted bottom row is our study: we adapt the teacher-to-student switching time, decide from the prompt's SNR and the endpoint-stabilization score, use a calibrated threshold, and attach Learn-Then-Test risk control.

## Scene 09 · Gap Map of the Related Literature

*Chapter II · Review of related literature · about 60 s*

This diagram turns the table into a picture of the gap. The three circles are three properties: in blue, deciding per input; in dark grey, a calibrated risk bound; and in light grey, at the bottom, a teacher-to-student handover. Each reviewed method sits where its properties overlap. ACT, BranchyNet and Adaptive Guidance are adaptive only. CALM and adaptive ASR correction are adaptive and bounded, but do not hand over between models. DMOSpeech 2 and T-Stitch hand over, but at a fixed or preset point. Adaptive teacher-student collaboration is adaptive with a handover, but has no bound. The cards on the right summarize these three gaps. The centre, where all three properties meet, is empty in the sources we reviewed. That is where this study sits. To be precise, this describes the sources we examined, not every adaptive method that exists.

## Scene 10 · System Architecture

*Chapter III · Methodology · about 65 s*

Chapter Three starts with the architecture. Blue and grey boxes are the released, frozen parts; orange boxes are what this study adds. Follow the loop from the top. The input is the prompt audio and the prompt and target text. The teacher takes one guided step, costing two passes. The endpoint monitor then reads the teacher's current estimate of the final output and computes the stabilization score, at no extra cost because it reuses values already computed. In parallel, the prompt SNR is estimated once from frame energies. The decision box asks: stop here? If not, the teacher takes another Euler step and we loop. If yes, the student restarts from the teacher's estimate at its fixed level and finishes in three passes, and the Vocos vocoder turns the mel-spectrogram into a waveform. The threshold used in the decision comes from the offline calibrator, which runs Learn-Then-Test once, before deployment.

## Scene 11 · Sampling Discretization & Handover

*Chapter III · Methodology · about 75 s*

This slide shows exactly where the switch can happen and what each choice costs. The curve on the left is the teacher's time grid under sway sampling: the steps are packed near the start, so half of the sixteen steps fall below time 0.29. The released code stops the teacher at the first grid point above the switching argument s, the dashed line at 0.25, which is step 8. Our candidates are the orange points, steps 4, 5 and 6. On the right, each bar is one candidate, with its cost from the formula two times k plus one, plus three. Stopping at step 4 costs 13 passes and saves 38 percent; step 5 costs 15 and saves 29; step 6 costs 17 and saves 19; and the reference costs 21. The box at the bottom explains why earlier handovers are possible: the student always re-noises the teacher's estimate to the same level, tau equals 0.0761, which matches the teacher's fourth grid point, so the student only needs a teacher state good enough to re-enter there.

## Scene 12 · Inference Decision Signals

*Chapter III · Methodology · about 70 s*

The stopping decision uses two signals, both free to compute. The top formula is the endpoint-stabilization score. At each teacher step, we take the teacher's current guess of the final output and measure how much it changed since the last step, relative to its size. A small score means the guess has settled. The chart shows two illustrative inputs: the blue easy input drops below the threshold lambda at step 5 and hands over there, at 15 passes; the grey hard input never settles and runs to the reference. The right side is the second signal, the prompt SNR, estimated from the energies of the loudest and quietest frames. With a gate at 20 decibels, a clean prompt at 30.5 decibels may stop early, while a noisy one at 17.8 decibels goes straight to the reference. The note at the bottom is important: our development diagnostics suggest the SNR may carry most of the benefit, so we treat the stabilization score as a hypothesis to test, not an assumption.

## Scene 13 · Stopping Policy Formulation

*Chapter III · Methodology · about 75 s*

This slide is the formal version. The top line is the problem: find the policy with the lowest expected cost whose risk of added failures stays below alpha. The policy we propose walks through the teacher's steps in order and stops at the first candidate step where the stabilization score is below lambda and the prompt SNR is above the gate theta; otherwise it uses the reference. It can never stop later than step 8. The middle box defines an added failure, fixed before calibration: the output fails if its word error rate exceeds the larger of 12.5 percent and the reference's rate plus one point, or if its speaker similarity drops more than 0.02 below the reference, on the same seeds. Setting lambda to infinity gives the SNR-only rule, and removing the gate gives stabilization only, so one family covers all variants. The four cards are the propositions we prove: zero overhead, cost that rises monotonically as lambda shrinks, a safe fallback that never exceeds 21 passes, and an oracle lower bound that tells us the best possible saving.

## Scene 14 · Risk-Calibrated Threshold Selection

*Chapter III · Methodology · about 80 s*

Now, how do we choose lambda with a guarantee? The bar at the top shows our speaker-disjoint splits from Seed-TTS: 100 speakers for development, 200 for calibration in orange, and 200 for testing in blue, one utterance per speaker. The four cards walk through an illustrative calibration from Appendix A, with 100 inputs. Thresholds are tested from conservative to aggressive. Each gets a binomial p-value from its number of failures. Lambda 0.40, with zero failures, gives p equal to 0.0059, below delta, so it is certified. Lambda 0.50, with one failure, gives 0.0371, also certified. Lambda 0.60, with three failures, gives 0.2578, above delta, so testing stops and 0.70 is never tested. We lock lambda-hat at 0.50. The line at the bottom states the guarantee: with probability at least one minus delta, the added-failure rate on new speakers stays at or below alpha. It also shows the cost of certainty: with zero failures, certifying five percent at ninety percent confidence needs at least 45 inputs. We fix the grid in advance and verify on the held-out test split and on 100 random re-splits.

## Scene 15 · Experimental Workflow & Benchmarks

*Chapter III · Methodology · about 70 s*

This slide puts the whole study in order: fifteen steps across five rows. Setup prepares the speaker splits, creates clean and noisy prompts, and checks that the sampler is deterministic, that the fallback reproduces the reference, and that pass counts match the formula. Objective One, on the development split, is the exploratory stage: one teacher run per input, every candidate scored, and the quality-cost map used to fix the SNR gate. Objective Two needs no data; it is the formulation and proofs. Objectives Three and Four, in the confirmatory stage, use speakers never seen before: we fix alpha, delta and the grid, run the tests in order, lock lambda-hat or fall back, and then benchmark the locked variants with paired comparisons, a noise shift and different calibration sizes, reporting every result, including negative ones. The three panels list what we compare against, what we measure, and the statistical tests we use. The unit of analysis throughout is the speaker.

## Scene 16 · Contribution and Summary

*Chapter III · Methodology · about 80 s*

To close, here is what this study adds, row by row. When to hand over: prior work uses one switching time for every utterance, fixed at 0.25 in DMOSpeech 2 and preset in T-Stitch; we choose it per utterance, from the prompt's SNR and the stabilization score, signals the sampler already computes. How sure we can be: prior adaptive rules have no bound on quality loss, and the calibrated bounds that exist are for exit layers and recognition; we use a Learn-Then-Test threshold, so that added failures stay at or below five percent with at least ninety percent confidence, checked on held-out speakers. What it costs: some prior methods need training or an extra estimator; ours needs no training and no extra forward pass, and if no threshold is certified it falls back to the released sampler, so it never costs more. The bottom line states what we do not claim: a proven advantage of the hybrid, any change to duration prediction, or any model training. In summary: decide per utterance, bound the risk, never cost more. The following slides are the appendix with our preliminary results.

## Scene 17 · Appendix: Preliminary & Rehearsal Experiments

*Appendix · about 75 s*

This appendix slide summarizes our foundational and rehearsal experiments on the Tesla T4; all are development evidence. On the left, the 500-speaker fixed-switching study shows that switching early at Step 4 fails on just 2.4 percent of clean prompts, compared with 64.4 percent of 15-decibel noisy prompts. Hindsight per-input selection saves 29.3 percent of passes, and EXP-017 confirmed the exact same pattern across both 16-interval and 32-interval grids, with 29.7 to 33.7 percent oracle headroom. On the right, the top card updates student restart and baseline checks with EXP-017 on the T4: the paper's 14-plus-2 j-equals-2 setting added 140 failures out of 360 with 0.342 noisy WER, Algorithm 2 without re-noising added 172 failures, and the 32-grid reference matched 16-grid Step 8 at 57 percent more compute—justifying our controlled j-equals-1 setting on the 16-grid. The bottom card reports two critical audit discoveries: Definition 3.4's original low-drift inequality certified zero savings on the 16-grid under Learn-Then-Test, and the binary per-utterance failure rule flagged 22.5 percent false failures even between two equal-quality references due to stochastic student noise, motivating continuous paired TOST equivalence in EXP-018.

## Scene 18 · Appendix: Signal Diagnostic & Premature Stabilization

*Appendix · about 75 s*

This slide explains why both parts of the initial heuristic policy broke when tested beyond clean versus 15-decibel white noise, and how that discovery led directly to STAC. On the left, the 20-decibel SNR gate cleanly separated 15-decibel white noise, but in EXP-011 and EXP-012 moderate 18 to 22 decibel noise crossed the 20-decibel line, causing 86 failures out of 360—nearly five times the five percent tolerance. Even worse, out-of-distribution telephone bandpass filtering silenced pause frames so the SNR estimator reported 44.2 decibels—ultra-clean—and exited early on degraded speech. On the right, EXP-012 uncovered why Definition 3.4's stabilization rule failed: prompt noise causes premature stabilization. Because noisy prompts weaken teacher guidance contrast, median Step-4 drift drops from 0.61 on clean speech down to 0.17 at 15 decibels, so stopping when drift is small stops noisy prompts first. After eight state-modifying operators in EXP-013 through EXP-015 failed our routing gate and single-threshold SNR rules collapsed to 1.5 percent savings across Steps 1 to 8, we inverted the rule and accumulated trajectory path length A-sub-i greater than or equal to lambda-hat-sub-i—eliminating external SNR completely.

## Scene 19 · Appendix: Live T4 Speedup & Risk Calibration

*Appendix · about 70 s*

This slide pairs our live Tesla T4 early-stopping execution from EXP-018 with our risk-control calibration analysis. On the left, instead of an offline prototype diagram, these bars show 30 live GPU runs where the teacher loop terminated online at STAC's chosen step, producing bitwise-identical endpoints with zero point zero difference from full replay and 0.12 milliseconds of gate overhead. Clean utterances averaged 13.2 passes, saving 37.1 percent of compute for a 1.49-times end-to-end wall-clock speedup; moderate 20-decibel noisy utterances averaged 17.8 passes for a 1.21-times speedup; and 15-decibel noisy utterances fell back 100 percent of the time to Step 8 with zero added error. On the right is the Appendix B simulation showing calibrated Learn-Then-Test holding risk below alpha 92 to 99 percent of the time versus 40 to 47 percent for naive selection, alongside our T4 verification where 1D Conformal Risk Control STAC held continuous risk within alpha equals 0.05 while slightly improving WER over the Step-8 reference.

## Scene 20 · Appendix: Proposed STAC Handover Architecture

*Appendix · about 75 s*

This slide introduces our updated architecture: the Stage-Wise Trajectory Action Cascade, or STAC, together with its conformal risk-controlled variant, 1D-CRC-STAC. Compare this directly with Slide 10. First, the external Prompt SNR branch at the top is completely eliminated—saving 1.85 milliseconds of waveform overhead and removing vulnerability to bandpass or babble noise. Second, the endpoint monitor becomes the STAC Action Monitor: at each guided teacher step, reusing tensors already in GPU memory at 0.12 milliseconds overhead and zero extra passes, it accumulates the running trajectory path length A-sub-i equals the sum of step drifts d-sub-1 through d-sub-i. Third, the decision diamond flips the inequality and checks a stage-specific threshold vector calibrated offline: has cumulative action reached lambda-hat-sub-i? The three cards at the bottom trace Utterance 1 from our Tesla T4 run. Clean speech builds strong guidance immediately—A-4 hits 3.65 against threshold 1.59 and exits at Step 4 in 13 passes. Moderate 20-decibel noise reaches only 1.52 at Step 4, stays in the teacher loop until A-7 reaches 2.06 at Step 7, and exits safely in 19 passes. Heavy 15-decibel noise stays prematurely flat below every threshold and automatically completes all 8 teacher steps.

## Scene 21 · Appendix: STAC & Cross-Experiment Benchmark

*Appendix · about 80 s*

Finally, this slide compares STAC and 1D-CRC-STAC head-to-head against every handover baseline on the Tesla T4. Across all eight steps of the 16-interval grid on 360 utterances, STAC ranks number one: without any Prompt SNR estimator and with 0.12 milliseconds of overhead, it reduces forward passes by 19.9 percent to 16.82 NFE, cutting synthesis time from 3.23 seconds to 2.60 seconds—a 1.25-times wall-clock speedup and 1.79-times real-time—while passing TOST equivalence on both WER and speaker similarity. Compared with full 32-step teacher synthesis at 9.65 seconds, STAC is 3.7 times faster with far better noisy intelligibility. The second row, 1D-CRC-STAC, provides formal alpha-equals-0.05 risk control at 15.8 percent savings while achieving an even lower WER than the full Step-8 reference. Notice how matched-degree-of-freedom SNR Cascade lags behind at 13.7 percent, single-threshold Calibrated SNR and TAS collapse to 1.5 percent when Step 1 is included, and original Definition 3.4 saves zero percent. On the 32-interval grid, STAC leads again at 20.7 percent savings, beating memoryless single-step drift by 6.6 percentage points. The three bottom cards confirm generalization and scope limits: 20.8 percent savings on the 500-speaker cohort, 32.7 percent versus 17.1 percent on unseen out-of-distribution corruptions, a minimum time floor of 0.019 to prevent ultra-early Whisper repetition outliers on 14-stage 32-grids, and our remaining step—running the frozen unseen-speaker test. Thank you, and we welcome your questions.

## Scene 22 · Appendix: Two-Axis Handover Architecture

*Appendix · about 85 s*

This appendix slide shows the Two-Axis Handover, the one candidate that passed every rule on our development data. It makes two separate decisions for every input. Axis one, what crosses, is decided first and for free: the SNR router measures how noisy the voice prompt is. A clean prompt, at or above 26.55 decibels, gets the teacher's whole estimate. A borderline prompt, within two decibels below that cut, gets Voice-Swap: the teacher runs to Step 4, but the student's clean voice replaces the teacher's voice part, so the teacher contributes only its prosody. A noisy prompt gets nothing from the teacher: the student makes the speech alone in four passes, because on our data the student beats the reference there. Axis two, when, applies to clean prompts only. The movement monitor adds up how far the teacher's estimate has moved by Step 3, at no extra cost. If it has moved enough, the teacher hands over at Step 3, for 11 passes in total; if not, it takes one more step and hands over at Step 4, for 13. The thresholds come from an offline fit on the stored development data, done once before deployment. On sixty development speakers this averaged 6.73 forward passes instead of 21, kept every quality rule, and EXP-023 confirmed on ten clean-prompt speakers that the routed handover, which stopped at Step 3 for 95 percent of seeds, keeps the teacher's pitch variety. It is development evidence on white noise only, and a combination of known parts, so a frozen test on new speakers comes next.
