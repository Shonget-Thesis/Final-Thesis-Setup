# 2026-11-18: Fixed 13-pass cost made explicit; commit-point ablation and teacher-variety test

**Task:** `PAPER-020` (`review`)

Following the discussion of whether the router's fixed 13-pass cost is a concern (Vincent, 18 Nov):

- **Chapter 1, Scope and Limitations:** the scope paragraphs now describe the payload router. "Adaptive" is defined as adaptive in the content handed over, not in the amount of computation; the teacher stops at one fixed point (13 vs 21 passes); per-input computation is out of scope; the student restarts where the teacher stops; prosodic variety is now measured. The rest of Chapter 1 (problem, research questions, objectives, conceptual framework, terms) still describes the old scheduler.
- **Chapter 2, Section 2.5 opening:** states that most reviewed methods adapt the amount of computation, while this study keeps it fixed and adapts the content.
- **Chapter 3:**
  - design paragraph says the method adapts content, not cost;
  - the cost proposition notes the draft pass is spent even when the teacher payload is chosen;
  - two commit-point variants added as ablations ($t_2$, $j=1$, 10 passes; $t_6$, $j=3$, 16 passes), the only other 8-grid points that coincide with the student schedule, each calibrated with the same procedure (Tables 3.3 and 3.10, Section 3.8.1, Step 12);
  - the main test of the teacher's value is fixed in advance: router vs STUR2 on CVf0 (raw and aligned), one-sided Wilcoxon, plus a team listening check reported as supporting evidence only;
  - the negative-results paragraph states that if the router's variety does not exceed STUR2's, the three-pass student will be reported as the cost-effective choice.

## Checks Run

- `latexmk -Werror main.tex`: exit 0, 111 pages, zero undefined references or citations, zero Overfull boxes.
