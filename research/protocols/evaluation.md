# Evaluation protocol draft

Version: draft-1. Status: not frozen; no held-out evaluation is authorized as a final result by this document.

1. Recover actual generation scripts, checkpoint hashes, source revision, and input/output manifests. Reconcile schedule labels to teacher steps and actual grid times.
2. Reproduce teacher-only, student-only, and the faithful original guided baseline with duration prediction frozen. Add instrumented actual teacher/student forward calls and complete scheduler cost.
3. Assign speakers to development/test before tuning. Record unseen texts, matched latent/noise seeds, corruption type/level, and exclusion rules. Choose sample size from a prespecified meaningful effect/margin and speaker-level variability.
4. Freeze one evaluator and explicit WER units, ASR model/text normalization, speaker-similarity model, quality tolerances, and absolute failure criteria. The legacy 12.5% floor is an audited historical rule, not an endorsed universal threshold.
5. Compare development-tuned fixed schedules, the exact SNR-only rule, and hybrid variants with matched inputs. Keep a retrospective oracle labeled as offline headroom.
6. Measure synchronized end-to-end latency, RTF, output duration, warmup, teacher/student calls, and all probe/restart work on identified hardware.
7. Report intelligibility, speaker similarity, blinded naturalness preference, catastrophic cases, and cross-seed prosodic diversity. Use paired speaker-cluster uncertainty; do not treat seeds as independent participants.
8. Predeclare when added probe complexity is justified. Record adverse and null results as well as improvements.

Before freezing: assign owner, review margins and sample size, record dataset permissions/participant procedures where applicable, and obtain the team's/adviser's protocol decision. Link that decision here.
