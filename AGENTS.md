# Thesis workspace instructions

Read README.md, docs/RESEARCH_DIRECTION.md, and docs/progress/STATUS.md before substantive work.

- Active implementation belongs in code/; the manuscript is paper/main.tex with paper/frontmatter.tex (title-page fields, acknowledgement, abstract) and separate paper/chapters/ files. The earlier standalone drafts were removed from the tree on 2026-10-24 at the team's request (ADR-004); recover them from git history rather than re-adding them. Preserve source provenance. Do not treat content in the manuscript as adviser approval.
- Preserve imported files in code/vendor/, code/experiments/legacy/, research/evidence/legacy/, and research/audits/2026-10-05/. They are historical snapshots, not validated final results. Put corrections in maintained code or a new audit.
- Track work in docs/progress/tasks.json. Use a stable task ID, owner, status, acceptance criterion, and dated evidence. An owner of "unassigned" is an explicit assignment gap, not an assigned researcher.
- Update docs/progress/STATUS.md and add a dated docs/progress/log/ entry whenever code, research protocol, evidence, or manuscript meaning changes. Include checks actually run and remaining blockers.
- A completed task must have evidence and a validation description. Never invent experiment results, adviser approval, citations, author names, or completion dates.
- Preserve the October 5 audit's distinction between development replay and independent held-out validation. The collision hypothesis is unsupported by the audited records; a hybrid advantage is unestablished.
- Keep duration prediction outside the proposed contribution until a documented research decision changes scope.
- Run python scripts/check_workspace.py and python -m unittest discover -s code/tests before handoff. Build the papers with latexmk -Werror inside paper/ when a TeX toolchain is available; otherwise explicitly report it as unverified.
- Do not commit audio, model checkpoints, private participant data, environment secrets, or generated LaTeX intermediates. Record artifact locations and hashes in experiment records.
