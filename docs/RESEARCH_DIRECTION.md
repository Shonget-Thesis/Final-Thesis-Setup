# Research direction

Status: working research plan, recorded 2026-10-09; adviser approval is not recorded.

Question: can an inference-time teacher-guidance policy reduce measured synthesis cost while meeting predefined intelligibility, speaker-similarity, naturalness, and diversity tolerances on unseen inputs?

The [October 5 audit](../research/audits/2026-10-05/FINDINGS.md) is the current interpretive reference. Historical reports remain preserved as evidence of earlier work, with their original wording.

- Treat existing telemetry as development evidence. The replay is not a fresh scheduler execution.
- Compare teacher-only, student-only, tuned fixed schedules, SNR-only, and a hybrid only if its overhead and benefit can be measured.
- Reconcile teacher-step counts, actual swayed-grid time, and real forward-call counts before new experiments.
- Freeze duration prediction. Use noise conditions for evaluation rather than as policy labels.
- Split by speaker before tuning. Freeze the evaluator, thresholds, noise seeds, latent seeds, and quality margins before held-out testing.
- Report absolute failures alongside baseline-relative changes. Repeated seeds do not create independent speakers.
- Keep the probe only if it adds meaningful held-out benefit over SNR-only and tuned fixed schedules after complete cost accounting.
- Do not present collision, naturalness preservation, hybrid necessity, or novelty as established by the current records.

The [protocol draft](../research/protocols/evaluation.md) and task board turn these requirements into pending work. Changes to this scope require a dated decision record with supporting evidence.
