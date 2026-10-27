# Presenter script

Read-aloud script for the animated walkthrough in [index.html](index.html). Timings are the auto-play durations of each scene (about 12 minutes in total).

## Scene 01 · Title

*Proposal · about 20 s*

Good day. Our study is titled "A Risk-Calibrated Adaptive Handover Method for Teacher-Guided Flow-Matching Text-to-Speech Inference." In one line: today's fast speech synthesizers let a large teacher model start every utterance and a small student model finish it. We propose deciding, for each utterance, when that handover happens, and attaching a statistical guarantee to that decision.

## Scene 02 · Proposal Overview

*Proposal · about 25 s*

This presentation follows the three chapters of the proposal. Chapter One sets up the background, the problem and our objectives. Chapter Two reviews the work we build on, from flow matching to statistical risk control, and identifies the gap. Chapter Three describes the proposed method, how we will calibrate it, and how we will evaluate it.

## Scene 03 · Background: Flow Matching, Cost & Distillation

*Chapter I · Introduction · about 45 s*

Text-to-speech converts written text into speech. Many recent systems use flow matching: the model learns a velocity field that moves random noise toward speech step by step, and every step is a full evaluation of a large network. F5-TTS, our teacher, works this way. That cost doubles under classifier-free guidance, because each guided step runs the network twice: once with the condition and once without. Distillation trains a small student that needs only a few steps; DMOSpeech cut sampling from 128 steps to four. But the student loses diversity. DMOSpeech 2 restores it by letting the teacher run the first steps and the student finish. In the released code that costs 21 forward passes per utterance: 18 for the teacher and 3 for the student.

## Scene 04 · Teacher-Guided TTS: Where This Study Works

*Chapter I · Introduction · about 50 s*

Here is the whole model in one outline. On the left, done once and by others: a large flow-matching teacher, F5-TTS, and a small student distilled from it. We do not train or change either one. On the right is inference, which happens for every utterance. The teacher runs the first steps, then hands over to the student to finish. In the released sampler that handover is one fixed point, eighteen teacher passes and three student passes, for everyone. Now we zoom in. Inside inference, the single open choice is where the handover falls. Today it is fixed at a switching time of 0.25 for every utterance. Our study picks it per utterance, and bounds the risk with a calibrated guarantee. That one decision is our focus, and the rest of the model stays exactly as released.

## Scene 05 · Problem Statement

*Chapter I · Introduction · about 35 s*

The problem is that the handover point is fixed. Every input, easy or hard, clean or noisy, pays the same 18 teacher passes. A switch later than an utterance needs wastes computation; an earlier one may degrade speech. This raises four problems: the effects of the switching time are not characterized; the handover boundary is not chosen per input; an adaptive choice comes with no reliability guarantee; and the assistive motivation does not by itself show any benefit to users. Our preliminary T4 data already show this: on a thousand development instances, fixed early switches fail on a quarter to a third of inputs, almost all with noisy prompts, while choosing per input in hindsight would need under 15 of the 21 passes.

## Scene 06 · Research Objectives

*Chapter I · Introduction · about 35 s*

Our general objective is to develop and evaluate a training-free, risk-calibrated method that selects the teacher's switching time per utterance. First, we will characterize how quality and cost change across switching times. Second, we will formulate selection as a constrained cost-minimization problem and design a policy with provable properties. Third, we will calibrate it with Learn-Then-Test. Fourth, we will compare its signals and benchmark it against fixed switching times and an oracle.

## Scene 07 · Scope & Delimitations

*Chapter I · Introduction · about 30 s*

The study varies only the switching time. Every network stays frozen, and the student's restart level stays at the released default. We evaluate on English Seed-TTS prompts under clean and noisy conditions, on an NVIDIA Tesla T4 through Google Colab. The guarantee is statistical, not per utterance. Assistive communication for autistic users motivates efficient synthesis, but we will not build or evaluate any assistive application or clinical outcome. The results serve speech researchers, TTS developers, assistive-technology developers and future researchers.

## Scene 08 · Input-Adaptive Computation

*Chapter II · Review of related literature · about 35 s*

Choosing the switch per utterance is input-adaptive computation. Adaptive Computation Time and BranchyNet saved computation, but gave no statement of how much quality they lose. CALM, for language models, is our closest precedent: it calibrated its early-exit threshold with risk control. Adaptive teacher-student collaboration exists in image diffusion, but not for the handover time in speech. T-Stitch switches from a small to a large model at a preset point, Adaptive Guidance skips the unguided pass when updates agree, and adaptive ASR correction calibrates with Learn-Then-Test, but for recognition, not synthesis. Our study adapts the switching time and attaches a calibrated guarantee.

## Scene 09 · Gap Map of the Related Literature

*Chapter II · Review of related literature · about 45 s*

Let us place the related work on three properties. The first is deciding per input. Adaptive Computation Time, BranchyNet, Adaptive Guidance and adaptive teacher-student collaboration all adapt to the input, but none states how much quality the saving may cost. The second is a calibrated risk bound. CALM and adaptive ASR correction have one, but for exit layers and recognition hypotheses, not for handing over from a teacher to a student in speech synthesis. The third is the handover itself. DMOSpeech 2 and T-Stitch switch between a large and a small model, but at one fixed or preset point for every input. Where all three meet, in the centre, we found no method in the sources we reviewed. That is where this study sits. This is a statement about the sources we examined, not a claim that no other adaptive method exists.

## Scene 10 · System Architecture

*Chapter III · Methodology · about 40 s*

Chapter Three. The method adds a decision layer to the released sampler without touching any network. At each teacher step, a controller reads the endpoint estimate and the prompt's signal-to-noise ratio. If both indicate the utterance has settled, the teacher stops and hands over to the student; otherwise it takes the next step. If no earlier candidate qualifies, it stops at the reference point. The threshold is set once, offline, by a calibrator.

## Scene 11 · Sampling Discretization & Handover

*Chapter III · Methodology · about 40 s*

Here is where the switch lives. Sway sampling packs the teacher's steps near the start of the flow. The released code stops the teacher at the first grid point above the switching argument; for 0.25, that is step eight. Our candidates are steps four, five, six and eight, costing 13, 15, 17 and 21 forward passes. At the handover, the student re-noises the teacher's estimate to a fixed level, tau equals 0.0761. This helps explain why an earlier, imperfect handover can still work.

## Scene 12 · Inference Decision Signals

*Chapter III · Methodology · about 40 s*

Two signals drive the decision. The endpoint-stabilization score measures how much the teacher's prediction of the final output changed since the last step; a small score means it has settled. In this illustration, the easy input settles by step five and stops there, at 15 passes; the hard input never settles and runs to the reference. The prompt SNR estimates how noisy the recording is. Both reuse values already computed, so the decision costs no extra forward pass. Our development diagnostic suggests the SNR may carry most of the benefit, so we treat the stabilization score as a hypothesis to test.

## Scene 13 · Stopping Policy Formulation

*Chapter III · Methodology · about 40 s*

Formally, we seek the cheapest policy whose risk of added failures stays below alpha. An added failure means the output is clearly worse than the reference on the same input: its word error rate passes the ceiling and margin, or its speaker similarity drops by more than 0.02. The policy stops at the first candidate that passes both signals. We prove four properties: zero overhead, monotone cost in the threshold, a safe fallback to the reference, and an oracle lower bound.

## Scene 14 · Risk-Calibrated Threshold Selection

*Chapter III · Methodology · about 40 s*

A rule that stops early will sometimes fail, so we promise instead that failures are rare: the added-failure risk stays below alpha, and the promise is wrong with probability at most delta. With zero failures, certifying five percent at ninety percent confidence needs 45 inputs, against about 461 for Hoeffding's bound. We calibrate the threshold with Learn-Then-Test on speaker-disjoint data: 100 speakers for development, 200 for calibration and 200 for testing. Thresholds are tested from conservative to aggressive. Each gets a binomial p-value; we keep going while it is at most delta and stop at the first that is not. In this illustrative example, 0.40 and 0.50 pass and 0.60 fails, so we lock 0.50. We will verify the guarantee on held-out speakers and over 100 random re-splits.

## Scene 15 · Experimental Workflow & Benchmarks

*Chapter III · Methodology · about 40 s*

The workflow has fifteen steps across a common setup and four objectives, each on its own data. We will benchmark the calibrated policy against fixed switching times, an SNR-only rule, controls that separate the stabilization signal from simply waiting longer, and an offline oracle. We measure word error rate, speaker similarity, UTMOS, exact forward passes and runtime, with paired statistical tests, and repeat under noise shift and smaller calibration sets. The design has two stages on different speakers: an exploratory stage on the development split fixes the candidates and the SNR gate, and a confirmatory stage calibrates on one split and tests once on another, with the speaker as the unit of analysis.

## Scene 16 · Contribution and Summary

*Chapter III · Methodology · about 45 s*

Here is what is new, stated plainly. Within the sources we reviewed, no method combines three things. First, when to hand over. Prior work uses one switching time for every utterance. We choose it per utterance, from signals the sampler already computes: the prompt's signal-to-noise ratio and the endpoint-stabilization score. Second, how sure we can be. Adaptive rules usually give no bound on quality loss. We calibrate the threshold with Learn-Then-Test, so that added failures stay below alpha, five percent, with at least ninety percent confidence, and we check that on held-out speakers. Third, what it costs. No new training and no extra forward pass, and if no threshold can be certified, we fall back to the released sampler, so we never cost more. What we do not claim: a proven advantage of the hybrid over its parts, any change to duration prediction, or any new model training. To summarize: we decide the handover per utterance from signals the sampler already has, we bound the rate of added failures with a calibrated guarantee, and we never cost more than the released sampler.

## Scene 17 · Appendix: Preliminary Experiments

*Appendix · about 40 s*

These are our preliminary results, all on development data. On the Tesla T4, with all 500 speakers under a clean and a noisy prompt, every fixed early switch fails on a quarter to a third of inputs, and almost all of those failures come from noisy prompts. Choosing per input in hindsight would need about 15 of the 21 passes. Restarting the student one level later departs from the reference more often, but that small run measures difference from the reference, not absolute quality, so we keep the default as a controlled setting. A cross-validated stabilization policy saved about ten percent with one added failure in 200, but chose its threshold with zero training failures, which is not a risk certificate. That is why Objective 3 uses Learn-Then-Test.

## Scene 18 · Appendix: Signal Diagnostic

*Appendix · about 35 s*

On the development speakers, the prompt SNR separated clean from noisy prompts completely, so a 20 decibel gate sent every noisy prompt to the reference. The stabilization score did not separate failures from passes on clean prompts, though that rests on only two or three failures. These figures come from the split that serves design decisions, so they motivate our tests rather than settle them. The plan is fixed in advance: measure discrimination, compare against controls, test prompts the SNR cannot see, and try richer zero-cost signals. If the score adds nothing, we recommend the SNR-gated policy and report the score as a negative result.

## Scene 19 · Appendix: Simulation and Prototype

*Appendix · about 35 s*

What do we expect? Settled, easy inputs hand over early and save up to 38 percent of the forward passes; hard inputs keep the full teacher run and cost exactly the baseline. In simulation, Learn-Then-Test kept its promise at every calibration size, while a naive threshold choice exceeded the tolerance in more than half of the runs. These are simulations; real speech results are what this study will measure.
