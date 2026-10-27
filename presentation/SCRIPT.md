# Presenter script

Read-aloud script for the animated walkthrough in [index.html](index.html). Timings are the auto-play durations of each scene (about 11 minutes in total).

## Scene 01 · Title

*Proposal · about 20 s*

Good day. Our study is titled "A Risk-Calibrated Adaptive Handover Method for Teacher-Guided Flow-Matching Text-to-Speech Inference." In one line: today's fast speech synthesizers let a large teacher model start every utterance and a small student model finish it. We propose deciding, for each utterance, when that handover happens, and attaching a statistical guarantee to that decision.

## Scene 02 · Proposal Overview

*Proposal · about 25 s*

This presentation follows the three chapters of the proposal. Chapter One sets up the background, the problem and our objectives. Chapter Two reviews the work we build on, from flow matching to statistical risk control, and identifies the gap. Chapter Three describes the proposed method, how we will calibrate it, and how we will evaluate it.

## Scene 03 · Flow-Matching Speech Synthesis

*Chapter I · Introduction · about 35 s*

Text-to-speech converts written text into spoken audio. Many recent systems use flow matching. The model learns a velocity field that moves random noise toward speech along a nearly straight path, from time zero to time one. To generate, we follow that field step by step. F5-TTS, the teacher model in our study, applies this with a diffusion transformer and a step schedule called sway sampling. The key detail is that every step is a full evaluation of a large network.

## Scene 04 · Sampling Cost & Distillation Trade-Offs

*Chapter I · Introduction · about 35 s*

That cost doubles under classifier-free guidance, because each guided step runs the network twice: once with the condition and once without. Distillation trains a small student that needs only a few steps; DMOSpeech cut sampling from 128 steps to four. But the student loses diversity. DMOSpeech 2 restores it by letting the teacher run the first steps and the student finish. In the released code that costs 21 forward passes per utterance: 18 for the teacher and 3 for the student.

## Scene 05 · Problem Statement

*Chapter I · Introduction · about 35 s*

The problem is that the handover point is fixed. Every input, easy or hard, clean or noisy, pays the same 18 teacher passes. A switch later than an utterance needs wastes computation; an earlier one may degrade speech. This raises four problems: the effects of the switching time are not characterized; the handover boundary is not chosen per input; an adaptive choice comes with no reliability guarantee; and the assistive motivation does not by itself show any benefit to users.

## Scene 06 · Research Objectives

*Chapter I · Introduction · about 35 s*

Our general objective is to develop and evaluate a training-free, risk-calibrated method that selects the teacher's switching time per utterance. First, we will characterize how quality and cost change across switching times. Second, we will formulate selection as a constrained cost-minimization problem and design a policy with provable properties. Third, we will calibrate it with Learn-Then-Test. Fourth, we will compare its signals and benchmark it against fixed switching times and an oracle.

## Scene 07 · Scope & Delimitations

*Chapter I · Introduction · about 30 s*

The study varies only the switching time. Every network stays frozen, and the student's restart level stays at the released default. We evaluate on English Seed-TTS prompts under clean and noisy conditions, on an NVIDIA Tesla T4 through Google Colab. The guarantee is statistical, not per utterance. Assistive communication for autistic users motivates efficient synthesis, but we will not build or evaluate any assistive application or clinical outcome. The results serve speech researchers, TTS developers, assistive-technology developers and future researchers.

## Scene 08 · Literature Synthesis

*Chapter II · Review of related literature · about 40 s*

Chapter Two follows one chain of ideas. Flow matching gives a trajectory and a free estimate of the final output at every step. Distillation gives a cheap student but loses diversity. Teacher-guided sampling restores diversity, at a cost set by one fixed switching time. Adaptive computation shows that effort can depend on the input. And statistical risk control shows that such a decision can carry a guarantee. Our study sits at the end of this chain.

## Scene 09 · Sampling Discretization & Handover

*Chapter II · Review of related literature · about 40 s*

Here is where the switch lives. Sway sampling packs the teacher's steps near the start of the flow. The released code stops the teacher at the first grid point above the switching argument; for 0.25, that is step eight. Our candidates are steps four, five, six and eight, costing 13, 15, 17 and 21 forward passes. At the handover, the student re-noises the teacher's estimate to a fixed level, tau equals 0.0761. This helps explain why an earlier, imperfect handover can still work.

## Scene 10 · Input-Adaptive Computation

*Chapter II · Review of related literature · about 35 s*

Choosing the switch per utterance is input-adaptive computation. Adaptive Computation Time and BranchyNet saved computation, but gave no statement of how much quality they lose. CALM, for language models, is our closest precedent: it calibrated its early-exit threshold with risk control. Adaptive teacher-student collaboration exists in image diffusion, but not for the handover time in speech. Our study adapts the switching time and attaches a calibrated guarantee.

## Scene 11 · Statistical Risk Control

*Chapter II · Review of related literature · about 35 s*

A rule that stops early will sometimes fail. We cannot prevent every failure, but we can promise that failures are rare: the risk stays below a tolerance alpha, and the promise is wrong with probability at most delta. Exact binomial bounds make this practical. With zero failures, certifying five percent at ninety percent confidence needs 45 inputs, against about 461 for Hoeffding's bound. Learn-Then-Test extends this to choosing among many thresholds, tested in a fixed order.

## Scene 12 · Research Gap

*Chapter II · Review of related literature · about 30 s*

Within the sources we examined, no single method combines three things for flow-matching speech synthesis: a per-utterance handover decision made from signals the sampler already computes; all networks kept as released; and a rate of added failures bounded with stated confidence on new speakers. This combination is the gap our study addresses.

## Scene 13 · System Architecture

*Chapter III · Methodology · about 40 s*

Chapter Three. The method adds a decision layer to the released sampler without touching any network. At each teacher step, a controller reads the endpoint estimate and the prompt's signal-to-noise ratio. If both indicate the utterance has settled, the teacher stops and hands over to the student; otherwise it takes the next step. If no earlier candidate qualifies, it stops at the reference point. The threshold is set once, offline, by a calibrator.

## Scene 14 · Inference Decision Signals

*Chapter III · Methodology · about 40 s*

Two signals drive the decision. The endpoint-stabilization score measures how much the teacher's prediction of the final output changed since the last step; a small score means it has settled. In this illustration, the easy input settles by step five and stops there, at 15 passes; the hard input never settles and runs to the reference. The prompt SNR estimates how noisy the recording is. Both reuse values already computed, so the decision costs no extra forward pass. Our development diagnostic suggests the SNR may carry most of the benefit, so we treat the stabilization score as a hypothesis to test.

## Scene 15 · Stopping Policy Formulation

*Chapter III · Methodology · about 40 s*

Formally, we seek the cheapest policy whose risk of added failures stays below alpha. An added failure means the output is clearly worse than the reference on the same input: its word error rate passes the ceiling and margin, or its speaker similarity drops by more than 0.02. The policy stops at the first candidate that passes both signals. We prove four properties: zero overhead, monotone cost in the threshold, a safe fallback to the reference, and an oracle lower bound.

## Scene 16 · Risk-Calibrated Threshold Selection

*Chapter III · Methodology · about 40 s*

We calibrate the threshold with Learn-Then-Test on speaker-disjoint data: 100 speakers for development, 200 for calibration and 200 for testing. Thresholds are tested from conservative to aggressive. Each gets a binomial p-value; we keep going while it is at most delta and stop at the first that is not. In this illustrative example, 0.40 and 0.50 pass and 0.60 fails, so we lock 0.50. We will verify the guarantee on held-out speakers and over 100 random re-splits.

## Scene 17 · Experimental Workflow & Benchmarks

*Chapter III · Methodology · about 35 s*

The workflow has fifteen steps across a common setup and four objectives, each on its own data. We will benchmark the calibrated policy against fixed switching times, an SNR-only rule, controls that separate the stabilization signal from simply waiting longer, and an offline oracle. We measure word error rate, speaker similarity, UTMOS, exact forward passes and runtime, with paired statistical tests, and repeat under noise shift and smaller calibration sets.

## Scene 18 · Projected Outcomes

*Chapter III · Methodology · about 35 s*

What do we expect? Settled, easy inputs hand over early and save up to 38 percent of the forward passes; hard inputs keep the full teacher run and cost exactly the baseline. In simulation, Learn-Then-Test kept its promise at every calibration size, while a naive threshold choice exceeded the tolerance in more than half of the runs. These are simulations; real speech results are what this study will measure.

## Scene 19 · Summary of Contributions

*Proposal · about 20 s*

To summarize: we decide the handover per utterance from signals the sampler already has; we bound the rate of added failures with a calibrated guarantee; and we never cost more than the released sampler. Thank you. We welcome your questions.
