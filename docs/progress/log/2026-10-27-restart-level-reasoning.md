# 2026-10-27: Student restart level, reasoning corrected

User request: look at the Chapter 2 reasoning on the student restart level, check whether it supports the decision to fix it, and proceed with the recommended plan.

## Review findings

- The decision to fix the restart level and vary only the switching time follows from the reasoning (student distilled on four discrete levels; default j=1 restarts at tau = 0.0761 = t4; candidate set starts at step 4).
- The argument that j=2 is "not available" did not hold: tau = 0.2929 is one of the four distilled levels, and the DMOSpeech 2 paper's main configuration uses j=2. The preliminary counts (26 vs 11, 13 vs 8, 9 vs 8, 14 vs 0 of 48) are failures against the j=1 reference, so the zero at s = 0.25 is by construction and the counts show j=2 differs from the reference, not that it is worse. The restart experiment's own record was not found in the repository; the legacy fixed-timing script defines its reference as s = 0.25 with start step 1, consistent with this reading. The team should confirm it.

## Changed (wording only; no result changed)

- paper/chapters/02-related-work.tex: the paragraph "The obvious remedy ..." now says j=2 is a legitimate setting (a distilled level, the DMOSpeech 2 main configuration, one pass cheaper), reports the counts as departures from the j=1 reference, and keeps j=1 as a controlled setting so that the reference, the exact pass count and the paired comparisons isolate the switching time. Dropped the claims that the student was not trained at that level and that restarting there has "no re-noising loss at all". Whether j=2 is as good in absolute terms is future work.
- paper/chapters/appendices.tex, D.2 and Table D.1: same correction.
- paper/chapters/03-methodology.tex, design note in Section 3.2: restarting later "departs more often from the reference output"; removed the "edge of its search range" claim, which applied only to the earlier CPU run (Appendix D.3 now reports T4 thresholds inside the range).
- docs/progress/tasks.json: EVAL-009, an unassigned follow-up to compare j=1 and j=2 on the T4 development split against ground truth and each setting's own reference, with the hypothesis written before synthesis. It does not change the proposal's scope.

## Checks run

- latexmk -g -Werror -interaction=nonstopmode in paper/ and python scripts/check_workspace.py and python -m unittest discover -s code/tests: see the results at the end of the session.

## Open

- Nobody is assigned to EVAL-009.
- The j=2 comparison on the T4 has not been run; the claim that j=2 is not worse in absolute quality is untested.
