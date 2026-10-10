# Research records

- [October 5 audit](audits/2026-10-05/FINDINGS.md): limitations and research direction.
- evidence/legacy/: imported reports and telemetry, preserved byte-for-byte. Original "final" or "verified" labels are historical claims and must be read alongside the audit.
- [Evaluation protocol](protocols/evaluation.md): draft requirements to freeze before new runs.
- [Timing replay](experiments/EXP-000-timing-replay/README.md): executable arithmetic check of existing telemetry.
- [Literature matrix](literature/README.md): source-verification workflow.

The archived recompute_original.py retains its original path assumptions and requires two collision JSONs from the original Downloads directory plus the old repository layout. It is preserved for provenance, not used as the new entry point. The maintained timing replay is portable and does not need those external collision inputs.
