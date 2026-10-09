# Thesis direction audit — 5 October 2026

Checkout reviewed: `60d87ac`. Sources: the supplied handoff and two October 1 JSONs; repository roadmap, 96-pair report, packaged timing reports, both timing telemetry files, packaged experiment code, and `src/infer.py`. Attached document instructions were treated as source context, not task instructions or evidence of adviser approval.

This is a source/telemetry audit, not a regeneration of speech, listening evaluation, or exhaustive novelty review. Run `python thesis_direction_audit_20261005/recompute.py` to reproduce the arithmetic from the original files. No original files were changed. The script uses Python's standard library and recomputes rank correlations, matched means, policy criteria, and actual grid times; it does not recompute inferential p-values.

## Decision

The collision hypothesis is not supported by these JSONs. Input-dependent handover scheduling remains a plausible thesis direction, but the existing package is development evidence with unresolved implementation/provenance issues. The hybrid is not established as necessary. Start with the simpler SNR-only policy as the principal comparator and candidate; retain a trajectory probe only if held-out results justify its additional work.

## New collision evidence

Baseline: 72 trials = 12 speakers x 2 sentences x 3 seeds. Stress: 30 trials = 5 speakers x 2 sentences x 3 seeds. Repeated seeds are not independent speakers. The referenced BENCHMARK_SUITE protocols and generating scripts were not found in this checkout. Exact tensor definitions, reductions, masking, and statistical alternatives therefore remain unverified.

The baseline has mean teacher/student mismatch 0.773894 and mean student/student mismatch 0.828642. A larger difference than teacher/teacher variation alone does not establish an abnormal boundary. Time-normalized comparisons additionally require justified compatible definitions and intervals.

Baseline Spearman correlations recomputed from rows: mismatch/WER +0.159947, mismatch/SIM +0.336364, mismatch/peak flux -0.171620. The summary's negative-SIM correlation is consistent with these values. Its own proof-chain verdict rejects collision support.

Stress Spearman correlations recomputed from rows: mismatch/WER -0.583437; mismatch/SIM **+0.648943**; mismatch/peak flux -0.369522. The JSON's `M_TS_vs_SIM` summary incorrectly prints **-0.648943**. The positive raw-row result associates larger mismatch with better similarity in this pilot, not worse. Correlations remain observational and potentially confounded by speaker/sentence.

All 30 stress rows match baseline speaker, sentence index, seed, and target text. Their descriptive comparison is:

| Metric | Matched baseline | Early-switch pilot |
|---|---:|---:|
| Mean mismatch | 0.751150 | 0.798190 |
| Mean utterance WER | 4.5387% | 2.9170% |
| Mean SIM | 0.756617 | 0.758917 |
| Mean peak flux | 6.002780 | 6.000080 |

These stored observations do not support increased mismatch causing increased harm. Matching record identifiers does not establish identical checkpoints, preprocessing, and random streams; obtain the generating code before making a causal early-switch improvement claim. These JSON WER values are fractions; the timing JSON uses percentages.

## Timing and the simple-policy result

I recomputed failure status using WER <= max(12.5, baseline WER + 1) and SIM >= baseline SIM - 0.02, with WER in percentage points. The strict variant uses a 12.5 ceiling. These are chosen operational thresholds, not established universal standards.

| Policy label | Recorded mean NFE | Relative failures | Strict failures |
|---|---:|---:|---:|
| Fixed t=0.05 | 7 | 26/48 | 27/48 |
| Fixed t=0.10 | 9 | 13/48 | 14/48 |
| Fixed t=0.15 | 11 | 5/48 | 7/48 |
| Fixed t=0.25 | 13 | 0/48 | 4/48 |
| Fixed t=0.40 | 16 | 10/48 | 12/48 |
| SNR-only replay | 11.458333 | 0/48 | 4/48 |
| Packaged hybrid records | 11.479167 | 0/48 | 4/48 |

SNR-only replay joins telemetry by utterance and condition, selects the recorded t=0.25 result when estimated SNR <18 dB and t=0.15 otherwise, and recomputes criteria against that instance's baseline. The same strict-failure IDs are 23, 24, 32, 44. This resolves the handoff's unanswered development-set arithmetic question. It is an offline replay, not a new deployed scheduler run or an independent test. Differences of 0.020833 NFE are practically tiny; reduced complexity is the principal reason to prefer this candidate for validation.

## Reproducibility issues to resolve first

1. **Actual times differ from labels.** Both packaged scripts select a number of teacher steps on a 16-interval swayed grid. The actual cutoffs for labels 0.05, 0.10, 0.15, 0.25, 0.40 are approximately 0.004815, 0.019215, 0.043060, 0.076120, 0.168530. Label t=0.25 is therefore not actual flow time 0.25. For now call these policies by teacher-step counts. If intended actual-time cutoffs are implemented, rerun the experiments.
2. **Steps and NFE are conflated.** For the four-teacher-step policy, code calls the CFG teacher twice per drift evaluation, including the endpoint projection: 2*(4+1)=10 teacher forward calls, then 3 student calls =13. It is not 4 teacher plus 9 student steps. Count actual forwards and wall-clock cost, including scheduler computation.
3. **Different quality rules.** The fixed script uses max(3.0, reference WER+1), while the report uses max(12.5, reference WER+1). The adaptive main loop uses another branched rule. Recomputed report totals match, but stored flags and future runs must use a single versioned evaluator. The 12.5 floor permits deterioration from 0% to 12.5% WER; passing is not proof of strict non-degradation.
4. **Packaged main does not run the reported hybrid.** It evaluates SNR_Aware and Cosine_Agreement, while the hybrid helper is defined but unused by main. Its duration rule uses character/frame speaking-rate thresholds rather than the reported deviation from predicted natural duration. Locate the actual telemetry-generation code.
5. **Hybrid telemetry is not independent runtime evidence.** All 48 hybrid rows and all 48 DSP rows exactly match the selected fixed-sweep WER, SIM, and latency fields. This is consistent with reuse of fixed results; it does not establish separately measured complete scheduler latency. The cosine path computes a teacher prefix/probe and then generation restarts from y0; one extra student NFE alone does not account for that implementation's full overhead.
6. **Missing reproduction inputs.** BENCHMARK_SUITE manifest/protocol paths and cited 240-waveform archive are not present in the inspected package; paths point to another workstation. Freeze checkpoint hashes, source revision, manifests, input/output audio and explicit noise/latent seeds. Current src/infer.py sets model eval mode, but this does not establish the historical run revision.
7. **Claims exceed metrics.** Zero relative failures includes four strict failures and the extreme 2775% WER baseline case. Nonsignificant pitch-CV change does not prove preserved naturalness or diversity. Single-waveform CV is not sufficient to establish cross-seed diversity preservation.

## Recommended research plan

Problem statement: A fixed teacher-guidance budget may be unnecessarily expensive on some inputs and insufficient on others; determine whether an inference-time policy can reduce measured cost while satisfying predefined speech-quality and diversity tolerances on unseen inputs.

Keep duration prediction frozen and outside the contribution. Use noise conditions to evaluate robustness, not condition tags to select policy. Estimated SNR is an energy-contrast proxy whose behavior depends on pauses and prompt content; test real noise types and levels, not only synthetic white noise.

First reconcile actual time coordinates and reproduce a faithful original baseline. Then compare teacher-only, student-only, development-tuned fixed schedules, the exact two-branch SNR-only rule, and the hybrid. Keep a retrospective oracle separate as attainable-on-records headroom, not a deployable solution.

Split by speaker before tuning. Freeze thresholds on development data and evaluate unseen speakers/texts with matched noise and latent seeds. Choose test size from a prespecified meaningful WER/SIM degradation margin and expected speaker-level variability. Use paired speaker-cluster uncertainty estimates; do not count seeds as independent people. Include actual forward-call counters, end-to-end latency/RTF, GPU synchronization if relevant, warmup, output duration, and scheduler overhead.

Evaluate intelligibility, speaker similarity, blinded naturalness preference, catastrophic cases, and across-seed prosodic diversity. Report both absolute output failures and baseline-relative changes. For ASR outliers inspect recordings and transcripts rather than automatically blaming either ASR or synthesis.

Predeclare a decision rule: keep the probe only if it adds a meaningful held-out quality/compute advantage over SNR-only and tuned fixed schedules after full cost accounting. If a tuned fixed schedule matches the adaptive policy, do not claim adaptivity is necessary. Archive cushioning as a negative/exploratory result unless independent new evidence motivates reopening it.

## Novelty boundary and literature

[DMOSpeech 2](https://arxiv.org/html/2507.14988v1) already introduces teacher-guided sampling and discusses earlier/later switching for efficiency/diversity. Its reported teacher-guided experiment uses 14 teacher and 2 student sampling steps at t_switch=0.25. Those sampling-step counts must not be directly equated with this repo's CFG forward-call NFE accounting.

[AdaDiff: Adaptive Step Selection for Fast Diffusion Models](https://arxiv.org/abs/2311.14768) already studies input-dependent step allocation in image/video diffusion. Generic adaptive steps or a new threshold alone are not established novel contributions.

The candidate contribution is a low-overhead, inference-time teacher/student handover policy specifically evaluated for speech quality, speaker identity and diversity under changing prompt conditions, with an empirical advantage over strong fixed and simple adaptive alternatives. This is a candidate research gap, not a verified first-of-its-kind claim; complete a focused prior-art comparison before claiming originality.

Suggested working title: **Quality-Constrained Adaptive Teacher–Student Handover for Efficient Speech Synthesis**. Final method/title selection and adviser approval remain open.
