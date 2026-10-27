# Presenter script

Read-aloud script for the animated walkthrough in [index.html](index.html). Timings are the auto-play durations of each scene (about 21 minutes in total).

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

## Scene 17 · Appendix: Preliminary Experiments

*Appendix · about 75 s*

This appendix slide holds our preliminary results; all of them are development evidence. The table on the left is the fixed-switching study on the Tesla T4, 500 speakers each with a clean and a noisy prompt. Read across each row: switching at step 4 fails on only 2.4 percent of clean prompts but 64 percent of noisy ones, and the pattern holds for steps 5 and 6; the reference never fails by definition. Choosing per input in hindsight would need 14.85 passes instead of 21, and every input that needed the reference was noisy. The top card on the right is the student restart level: restarting one level later departs from the j-equals-1 reference more often, but this small CPU run measures difference from the reference, not absolute quality, so we keep the default as a controlled setting. The bottom card is a cross-validated stabilization policy: it saved about ten percent with one added failure in 200, but its threshold was chosen with zero training failures, which is not a certificate. That is why Objective Three uses Learn-Then-Test.

## Scene 18 · Appendix: Signal Diagnostic

*Appendix · about 70 s*

This slide shows how the two signals behaved on the 100 development speakers. On the left, the SNR bars: clean prompts measured 23 to 50 decibels and noisy prompts 17 to 18, with the black line marking the 20-decibel gate. The two groups do not overlap, so the gate sent every noisy prompt to the reference. On the right, the stabilization score on clean prompts did not separate failures from passes, with an area under the curve of about 0.47, although that rests on only two or three failures. At lambda 0.50 the policy also waited longer than needed on most clean prompts. The bottom card is our plan to test the score properly, fixed before we look at the data: measure discrimination, compare against controls, test prompts that the SNR cannot detect, such as reverberation, and try richer zero-cost signals. If the score adds nothing, we will recommend the SNR-gated policy and report the score as a negative result.

## Scene 19 · Appendix: Simulation and Prototype

*Appendix · about 65 s*

The last appendix slide shows what we expect, from the simulation and the prototype in Appendices B and C. On the left, the prototype view: a clean, easy input hands over early and saves up to 38 percent, shown by the hatched orange cells; a moderate input saves less; a noisy, hard input keeps the full teacher run and costs exactly the baseline, never more. On the right is a simulation, with made-up data, not experimental results. Each pair of bars is a calibration size. The orange bars show how often Learn-Then-Test kept the true risk within the tolerance: 92 to 99 percent of the time, above the required 90. The grey bars are a naive rule that just picks the best-looking threshold: it met the tolerance only 40 to 47 percent of the time. The guarantee costs some computation, between 15.9 and 18.2 passes on average instead of 21. Thank you; we welcome your questions.
