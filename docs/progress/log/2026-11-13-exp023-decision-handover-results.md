# 2026-11-13: EXP-023 Decision Handover pilot, results

Task: EVAL-016, now in review. The run was on a Colab Tesla T4 and the results zip was created at 15:52 UTC. The requester sent `DMOSpeech2_EXP023_Decision_Handover_Pilot_Results.zip`.

## Outcome

These are development data and white noise only. A pilot cannot support a claim.

| Candidate | Frozen rule | Result |
|---|---|---|
| Decision Handover (DH, DHc, DHu at Steps 2 and 3) | G1 and G2 | **Does not advance.** G1 fails, and G2 fails for all six outputs. |
| Two-Axis Handover | G3 | **Advances** to a frozen test on new speakers. |
| Agreement-Routed Handover | G4 | **Does not advance.** |

The five pre-specified follow-ups (`code/analysis/exp023_followups.py`) are all **not promising**:
- GAR-u;
- GAR-ratio;
- Collapse-Gated Handover;
- the Decision Commitment replay;
- the Prosody-Agreement Safeguard;
- the Guidance-Choice Handover.

## Provenance (checked before any analysis)

- **Untrusted handling:** the zip was extracted into its own empty folder and read only with `python -I`. No member path was unsafe. Its SHA-256 is `a75b30448c16ebae32874056e7ba833227b53f29dca584c3062cb50e0bc84f91`.
- **Run info:** `RUN_INFO.txt` records the thesis at `7d35e81` (the pushed notebook fix), Shonget-DMOSpeech2 at `8a7868f` and a Tesla T4.
- **Code and preregistration:** the zipped preregistration is identical to the committed one (`469c90ed...`). All seven zipped runner modules are identical to `code/colab/`, with line endings normalised.
- **Completeness:** 60 of 60 quality cases, 20 of 20 seed pairs and 10 of 10 diversity prompts, all on a Tesla T4.
- **Smoke test:** it reproduced the stored STS_3 scores exactly (max |dWER| and |dSIM| 0.0000). The pilot's teacher loop matched the released guided teacher exactly (max difference 0).
- **Re-analysis:** running `run_decision_handover_pilot.py analyze` on copies of the CSV files gave a `pilot_results.json` identical to the zip's.
- **Import:** `research/experiments/EXP-023-decision-handover-pilot/results/` holds byte-identical copies, with a `MANIFEST.json` of file hashes. The follow-up outputs are generated in place by the committed script and listed separately as derived files.

## Decision Handover: the premise does not hold

**G1, whether the first estimate is shared across seeds.** It requires the median ratio of the slow-part spread at Step 0 to that at Step 3 to be at most 0.25, on both prompt types.

| Condition | Ratio, guided teacher | Within 0.25? |
|---|---|---|
| clean | 0.146 | yes |
| noisy15 | 0.296 | no |

So G1 fails. The deeper finding is that the teacher's change after Step 0 is mostly **shared** across seeds, not seed-specific. The shared share of that change at Steps 1–4 is:

| Teacher | Clean | Noisy15 |
|---|---|---|
| Guided | 0.71–0.83 | 0.89–0.94 |
| Unguided | 0.56–0.61 | 0.60–0.78 |

Independent decisions would give 0.50. Most of what the teacher adds after Step 0 is input-driven sharpening away from its blurry average, so T_k − T₀ is not "this seed's decision". This is the double-counting risk the README listed.

**G2, whether any output keeps quality and pitch variety at 12 passes or fewer.** It fails for all six outputs. Pooled capped WER and clean CVf0 are differences from the stored reference.

| Output | Passes | Pooled WER (95% CI) | Clean CVf0 | What fails |
|---|---|---|---|---|
| DH_2 | 10 | +0.116 [+0.064, +0.174] | −0.029 | WER, pitch variety |
| DH_3 | 12 | +0.115 [+0.057, +0.182] | −0.011 | WER, pitch variety |
| DHc_2 | 10 | +0.062 | −0.023 | WER, pitch variety |
| DHc_3 | 12 | +0.009 [−0.044, +0.056] | −0.012 | WER upper bound; WER above STS on degraded prompts (+0.021); pitch variety |
| DHu_2 | 7 | −0.025 [−0.061, +0.006] | −0.038 | pitch variety only |
| DHu_3 | 8 | −0.013 [−0.056, +0.024] | −0.027 | pitch variety only |

- **Guided decisions:** adding the guided teacher's mostly shared sharpening to the student's already sharp draft damages intelligibility.
- **Unguided decisions:** they keep quality but lose most of the teacher's pitch variety.
- **The student alone, on the same cases:** WER −0.032, similarity +0.048 and UTMOS +0.745 against the reference, again better than the reference.

## Two-Axis Handover: G3 passes

| Clean pitch variety (CVf0) | Against the stored reference |
|---|---|
| The Two-Axis Handover (G3) | **−0.0013 [−0.0042, +0.0010]** |
| Whole estimate at Step 3 alone | −0.0009 |
| Whole estimate at Step 4 alone | −0.0010 |

- **Routing:** all 10 diversity prompts were on the whole-estimate branch, and 95% of their seeds were handed over at Step 3.
- **Reproduction:** the pilot's Step-4 handover reproduced EXP-020's stored CVf0 exactly (max |difference| 0.0000).
- **Together with the offline screen:** the screen found 6.73 mean passes out of fold with every measured quality rule passing. So the Two-Axis Handover now meets every rule on development data. It advances to a frozen test on new speakers, which has not been designed yet.

## Agreement-Routed Handover: G4 fails

Out of fold, routing by the real Step-0 voice gap misrouted 23 of 60 cases, against 0 for the prompt-SNR rule on the same folds.
- **Routes:** 15 of 20 gauss20 cases and 7 of 20 noisy15 cases were sent to the whole estimate.
- **Cost:** 11.47 mean passes.
- **Pitch variety:** clean CVf0 −0.021.

An exploratory check after the results shows why. The AUC for telling degraded from clean prompts on the pilot's cases:

| Signal | AUC |
|---|---|
| Prompt SNR | 1.000 |
| Stored Step-4 voice gap (the screen's stand-in) | 0.999 |
| Step-0 guided gap (ARH) | 0.951 |
| Step-0 unguided gap | 0.781 |
| Guided/unguided ratio | 0.725 |

- **The gap is weaker at Step 0.** There the teacher's estimate is still its blurry average, so clean and degraded prompts overlap more than at Step 4, where the screen measured the gap.
- **The frozen cut was fragile.** The Voice-Swap cut, the largest clean gap of the training folds, sat above most degraded prompts because one clean prompt had a gap of 1.85, against a degraded median of 1.76.
- **No retuning.** ARH's claimed advantage over SNR, on reverberant or telephone prompts, remains untested.

## Pre-specified follow-ups

All ran out of fold with the rules fixed before the run. None is promising:

| Follow-up | Result |
|---|---|
| GAR-u | 24 misroutes |
| GAR-ratio | 27 misroutes. Guidance amplified the voice gap 3.6–13.0 times on clean prompts (median 8.8) and 4.3–21.2 times on degraded ones (median 11.1). The ranges overlap heavily, so the ratio does not separate them. |
| Collapse-Gated Handover | identical to ARH: its gate never fired within the quality and pitch rules |
| Prosody-Agreement Safeguard | identical to ARH: no student-branch case exceeded the cut |
| Decision Commitment replay | fell back to DH_3 for every case, because no thresholds kept the quality and pitch rules |
| Guidance-Choice Handover | fell back to DH_3 for every case, for the same reason |

## Consequences

- **The Two-Axis Handover is the only candidate that advances.** It is a combination of known parts, as noted when it was proposed.
  - **What its frozen test needs:** new speakers (for example development speakers 61–100, also named by EXP-019 and EXP-021). It needs the Step-3 and Step-4 handovers, Voice-Swap, the student alone and the reference, plus the prompt SNR, the teacher's drift d₁–d₃ and pitch variety.
  - **Its parameters:** frozen from the stored data (SNR cut 26.55 dB, 2 dB band, λ₃ = 1.2154).
- **The theory-derived candidates did not survive the pilot.** That covers Decision Handover, ARH, GAR, Collapse-Gated, Decision Commitment, PAS and GCH.
- **The thesis method is unchanged.**

## Checks run

- **Provenance and re-analysis:** as above.
- **`python code/analysis/exp023_followups.py --out <results>`:** wrote `FOLLOWUPS_REPORT.md` and `followups_results.json`.
- **`python scripts/check_workspace.py`, `python -m unittest discover -s code/tests`, `latexmk -Werror` in `paper/`:** see the final message of this change.

## Remaining blockers

- **Evidence and conditions:** development speakers and white noise only.
- **No design yet:** the Two-Axis Handover's frozen test on new speakers has not been designed.
- **Not adopted:** adoption of any method would need a dated decision record and adviser review.
