# 2026-11-13: EXP-023 Decision Handover pilot prepared

Task: EVAL-016 (todo). Requested by Rhenel Jhon R. Sajol, who asked for a method whose mechanism is fully new, and parked EXP-021 ("I don't like EXP-21").

## Why

- **What the stored data show:** on the development data of EXP-017, EXP-018 and EXP-020, the student alone (4 passes) matches or beats the 0.25 reference (21 passes) on WER, similarity and UTMOS in every prompt condition. The reference's only measurable extra is pitch variety on clean prompts. These checks are exploratory, run after the fact on inspected speakers.
- **Earlier proposals did not meet the requester's standard.** A teacher gate on the cross-model voice disagreement was screened offline on the same data and looked strong (7.29 mean passes). It was rejected as not fully novel: deciding per input whether to call the teacher is a known kind of decision, and on white noise a plain SNR estimate gated almost as well (7.58).
- **Decision Handover has a new mechanism.** It hands the student only the teacher's seed-specific change since its first estimate, x_k = S + slow(T_k − T₀), and anything the teacher shares across seeds cancels.
- **A pilot comes first, not a frozen test.** Offline screening is impossible here (no stored T₀ or T_k vectors), and EXP-020 and EXP-022 failed only after full runs.

## What changed

- **`code/colab/run_decision_handover_pilot.py`:**
  - **Algorithm:** the handover family x_k(g) = voice(S) + fine(S) + slow(T_k) + g (slow(S) − slow(T₀)): DH (g = 1), DHc (g = √(1 − ρ_k)) and DHu (DH with the teacher's conditional branch only).
  - **Teacher loop:** for the unguided teacher, written to repeat the released single-pass teacher.
  - **Measurements:** decision signals, the seed check and pitch variety.
  - **Analysis:** against the stored reference, STS, Voice-Swap and the student, with the frozen go/no-go rules G1 and G2.
  - **Commands:** `freeze`, `smoke`, `run` and `analyze`, with code and input hashes, as in EXP-022.
- **`research/experiments/EXP-023-decision-handover-pilot/`:** README and `preregistration.json` (sha256 `79752afbf4d52688...`, created 2026-11-13 UTC), frozen before any synthesis. ρ₂ = 0.639 and ρ₃ = 0.941 are computed from the committed EXP-022 and EXP-020 diversity files and recorded in it.
- **`code/colab/DMOSpeech2_EXP023_Decision_Handover_Pilot.ipynb`:** 17 cells, built from the EXP-022 layout, including the Drive cache cell.
- **`code/tests/test_decision_handover.py`:** 24 tests:
  - the pass accounting;
  - stored-label and pitch-data coverage of every pilot case;
  - the formula: g = 0 is STS, g = 1 is S + slow(T_k − T₀), and a component shared by T₀ and T_k cancels;
  - the teacher loop against a reference copy of the released single pass, with the unguided branch calling only the conditional model;
  - the decision signals and seed statistics;
  - the run loops on a fake model;
  - the go/no-go analysis on synthetic data;
  - preregistration tampering.
- **Tracking and status:** EVAL-016 in `tasks.json`, the rendered `TASKS.md`, `STATUS.md` and `research/experiments/README.md`. `.gitattributes` keeps future EXP-023 result bytes unchanged.
- **EXP-021:** EVAL-014 is set to blocked, with the blocker "parked at the requester's request". Its amended preregistration stays as it is.

## Decisions in this change

- **Development speakers only:** the first 20 Quick-set speakers, the 10 seed-check speakers and the 10 pitch-variety speakers were all used to design earlier methods. The pilot cannot support a claim; a positive result would lead to a frozen test on new speakers.
- **Variants:** DHc guards against counting the student's prosody twice, and DHu tests deciding without guidance (half the teacher cost). The cheapest variant that meets G2 is carried forward, as fixed in the preregistration.
- **The shared first estimate is a precondition.** G1 must hold for the pilot to advance, because the method's justification rests on it.

## Novelty check (targeted searches, 2026-11-13)

No published method found passes only the teacher's seed-specific change onto a student's sample. The closest work:
- **DiFA (2026):** a deviation term within one model's trajectory.
- **Gandikota and Bau (WACV 2026):** the base model runs only the first step, then the whole state is passed to the distilled model.
- **Distillation++ (ICCV 2025):** the teacher refines the student's estimates.
- **STRIDE (2026):** noise is injected into a distilled model's features.
- **T-Stitch (2024):** switches from a small to a large model and passes the whole state.

The building blocks are known. Follow-up searches the same day, after the requester asked whether the method is novel, added three precedents to the README:
- **Delta Denoising Score (ICCV 2023):** subtracting a reference branch cancels the error two predictions share. This is the principle behind "shared contamination cancels".
- **Self-Guidance (TPAMI 2025):** differences across noise levels within one model.
- **Composable Diffusion (ECCV 2022):** adding up differences from a shared baseline.

Those searches found no transfer of a base model's trajectory difference onto a distilled model's sample, and no inference-time teacher-to-student prosody transfer in flow-matching TTS. Commitment theory and skipping guidance are known as well. The preregistration's novelty statement is unchanged, because it already names only the handed-over change as new. This is not a full literature review.

## Candidates recorded (same day)

The requester asked for SNR-based and STAC-based candidates with a novel part. The README now lists three:
- Decision Handover at a fixed step;
- SNR + Decision Handover;
- STAC + Decision Handover.

The decision is the only novel part of each, because SNR, drift and STAC decide when to hand over, which is a known kind of decision. Drift (Definition 3.4) is left out: it certified nothing in EXP-012 and saved 0% in EXP-018. The pass counts in the README are computed from stored data:
- **SNR routing replay:** clean prompts kept on the teacher in 57 of 60 cases, degraded prompts sent to the student in 286 of 300.
- **STAC's EXP-018 exits:** 178 of 360 hand over before Step 8.

Neither candidate is tested; both need Decision Handover outputs first.

The requester then asked whether the routing itself, in SNR or STAC, can be made novel. The only new thing a handover router can choose is what crosses, so `code/analysis/payload_routing_screen.py` replays that choice offline on the stored 360 development cases. This is exploratory and uses the same out-of-fold SNR routing as above:
- **Routing what crosses loses:** at a fixed step, the whole estimate on clean prompts and the slow part or Voice-Swap on noisy ones cost 9.80, 11.80 and 13.80 mean passes at Steps 2, 3 and 4. On degraded prompts, WER was +0.020, +0.013 and +0.004 against the reference, with 96, 87 and 72 of 300 Definition 3.1 failures.
- **Sending noisy prompts to the student alone wins:** it cost 4.99, 5.38 and 5.78 mean passes. On degraded prompts, WER was −0.026, similarity +0.075 and UTMOS +1.15 against the reference, but Definition 3.1 counted 48–53 of 300 failures, because the rule measures agreement with a reference that copies the noise.
- **SNR choosing the step, as in Chapter 3,** cost 18.63–19.42 mean passes.

So routing what crosses pays only if a payload beats the student on noisy prompts. The decision is the only payload left, and the pilot measures it. For STAC, the best new idea found, stopping when the decision has committed (its displacement over STAC's distance travelled), is a new signal for a known decision, which is partly novel by the requester's standard.

The requester then asked for a method built from these results, with no further searching. The README now records candidate 4, **Decision Commitment Handover**:
- **Empty decision:** after Step 1, the student finishes alone.
- **Settled at Step 2:** otherwise, if the decision has stopped turning, the student gets x = S + D₂.
- **Step 3:** otherwise x = S + D₃.

Each choice is tied to a stored result in the README. The frozen pilot already records the signals the method needs (`dr`, `dc`) and its Step-2 and Step-3 outputs, so it will be replayed offline after the run. The runner was not changed.

## Amendment (same day, before any synthesis): the Two-Axis Handover as a second candidate

The requester asked for the Two-Axis Handover, from the offline screen's round 4, to be added to this notebook as another candidate.
- **Why only pitch variety.** EXP-023's speakers and seeds are the stored ones, so re-synthesizing its quality outputs would only reproduce stored scores. The pilot therefore measures the one rule the screen could not: pitch variety when clean prompts are handed over at Step 3.
- **Runner changes (`run_decision_handover_pilot.py`).** The diversity block now runs the guided teacher to Step 4 and also synthesizes:
  - the whole-estimate handovers at Steps 3 and 4 (W3, W4);
  - each seed's A₃, from the sampler's `compute_stabilization_drift`, the function that produced the stored d16;
  - the Two-Axis Handover's per-seed choice between W3 and W4 (TA).
- **Analysis changes.**
  - **New rule:** G3 (TA's clean CVf0 minus the stored reference, mean ≥ −0.01), with its own advance decision.
  - **Checks:** W4 against EXP-020's stored plain_4.
  - **Descriptive:** an in-sample replay on the 360 stored cases.
- **Frozen parameters.** They were fitted on all 60 Quick-set speakers with the screen's rules:
  - SNR cut 26.546 dB;
  - Voice-Swap band 2 dB;
  - Step-3 action threshold λ₃ = 1.21543.

  In sample, this gives 7.16 mean passes, with routes W3 89, W4 7, Voice-Swap 45 and student alone 219, and 0 clean failures. The clean UTMOS lower bound is −0.098.
- **Unchanged.** Decision Handover's outputs, data, G1 and G2 did not change.
- **Preregistration.** It was re-frozen as `03006f5c53a8ecf16b7cafc932e72bc88ef0cca02fd3edcccd40bfaf4ea2fbf0` (2026-11-13 11:59 UTC). It records the amendment and the hash it replaces, `79752afbf4d5268847ec113ae4366d42c329711f2a4d01fa2dd7410d7309c688`. The earlier file was never committed, and no synthesis had used it.
- **Notebook.** The overview table, the "What runs" list and the time estimate (now 1.5 to 2 hours, estimated) were updated. The code cells are unchanged.
- **Tests (`test_decision_handover.py`, now 32).**
  - the costs of W3 and W4;
  - the diversity loop on and off the whole-estimate branch;
  - G3 deciding independently of G1 and G2;
  - stored CVf0 used off the whole-estimate branch;
  - the parameter fit and the routing;
  - the in-sample replay keeping its rule;
  - the amendment record.

## Second and third amendments (same day, before any synthesis): the Agreement-Routed Handover

The requester asked for one last candidate through the same process: proposal, literature check, local test, and addition to the notebook. The process and results are in the offline screen log, round 5.
- **Second amendment.** It added the Agreement-Routed Handover as a third candidate, in analysis only. The quality block already records the Step-0 voice gap between T₀ and the draft S (`vg_g`), so G4 replays the routing out of fold over 5 speaker folds with the stored outputs of the same cases. It was frozen as `590404389927c4888c8059acf8a0548fb84d5cb562442eafb39d64ac1eb39b7c` (12:16 UTC).
- **A check of that G4 before any synthesis** used synthetic gaps that route perfectly: every clean case to the whole estimate at Step 4, and every degraded case to the student. G4 still failed. The stored Step-4 outputs on these 20 speakers give a clean UTMOS of −0.048 [−0.132, +0.026] against the reference, below the −0.10 bound. That G4 would have judged the stored outputs on 20 speakers, not the new signal.
- **Third amendment.** G4 now judges what this pilot can test:
  - **Routing:** the Step-0 voice gap misroutes at most 1 more case than the prompt-SNR rule, on the same cases and folds. A misroute is a clean case sent to the student, or a degraded case sent to the whole estimate.
  - **Cost:** at most 14.7 mean passes.
  - **Pitch variety:** kept.
  - **Absolute quality:** reported, not judged.

  With perfect routing, the new G4 passes: 0 misroutes, against 0 for SNR, at 8.67 mean passes. The SNR rule costs 7.02 there. It was re-frozen as `469c90edf7c6094d0cc2eb31ed0d6fbf8c847847af42f06b717ba9d6f9182600` (2026-11-13 12:18 UTC). The `amendments` list records all three amendments and the hashes they replace.
- **Unchanged.** The run loops, outputs, data, G1, G2 and G3 did not change. The notebook's overview now lists the third candidate; its code cells are unchanged.
- **Tests (`test_decision_handover.py`, now 37).**
  - the cuts, branches and folds;
  - the G4 rule, including the one-misroute allowance;
  - routing that follows separable gaps;
  - an inverted gap, which hands noisy prompts to the teacher and fails G4;
  - the amendment chain.

## Pre-specified follow-ups (same day, before the run)

The requester asked for more theory-derived candidates after rounds 6 and 7 of the offline screen came out negative. The stored data hold no further signal to test them on. They were therefore written into `code/analysis/exp023_followups.py` before any pilot data existed, using only signals the pilot records and the stored outputs of the same cases:
- **GAR (Guidance-Amplification Routing).** Guidance amplifies what the prompt imposes, so the conditional-branch Step-0 voice gap (`vg_u`), or the ratio `vg_g / vg_u`, should separate copied recording conditions from plain teacher-student mismatch. Its literature check found no such router. It did find first-step unguided sampling in few-step TTS ([2611.11179](https://arxiv.org/html/2611.11179)) and CFG-Zero⋆, which are now cited as prior work for DHu.
- **CGH (Collapse-Gated Handover).** This is the theory-exact version of round 6, gated on `dr_g_1`.
- **Candidate 4 (Decision Commitment Handover) replay.**

All are out of fold over the pilot's speaker folds, with fixed criteria for "promising". They are not part of the frozen go/no-go, and the runner and preregistration are unchanged. `code/tests/test_exp023_followups.py` (4 tests) runs them on synthetic pilot files.

The requester then asked for two more theories, added the same day and also before the run:
- **PAS (Prosody-Agreement Safeguard), from alignment theory.** Word timing is set by the text and prompt, so it is already in T₀. On ARH's student branch, a draft whose Step-0 prosody gap to T₀ is beyond 95% of clean drafts goes to Voice-Swap.
- **GCH (Guidance-Choice Handover), from early-guidance theory.** The Step-0 ratio `vg_g / vg_u` chooses between the unguided teacher's decision (DHu_3, 9 passes) and the guided one (DH_3, 12).

A third idea, a "restart-visibility" stop with a supposedly parameter-free threshold, was dropped. Its constant depends on how much the student pools across the utterance, which is unknown.

Literature checks found neither PAS nor GCH. The tests now cover both (6 tests).

## First smoke attempt on Colab: killed for lack of RAM

The first smoke run (branch `rhnl-3` at `6528959`) was killed with exit code −9 while the transformers library loaded Whisper large-v3 (1,259 weight tensors) for the first score. The first case's synthesis had already run, so the GPU was fine. No row was written, and nothing was compared.
- **Cause:** −9 is the operating system's kill when the machine runs out of system RAM. A standard Colab VM has about 12.7 GB. The runner loads the teacher, the student, Whisper large-v3, WavLM and UTMOS, and the notebook kernel also held a PyTorch and CUDA context from the Step-1 GPU check. EXP-022 ran with the same layout earlier the same day, so the margin is small.
- **Fix, in the notebook only:**
  - the GPU check uses `nvidia-smi` instead of importing PyTorch into the kernel;
  - the runner starts with `MALLOC_ARENA_MAX=2`, which limits allocator arenas while the large models load;
  - the available RAM is printed before the smoke test and the full run;
  - exit code −9 now prints what to do: Runtime > Disconnect and delete runtime, then Run all.
- **Preregistration still valid.** The notebook is not a hashed dependency, and none of these changes can alter a result. `check_prereg()` still accepts `469c90ed...`.
- **If it recurs:** use a High-RAM runtime. The alternative is a memory change inside a frozen file, which would need a re-freeze before the run.

## Checks run

- **`python -m unittest discover -s code/tests -p test_decision_handover.py`:** OK, 24 tests.
- **`analyze --allow_partial` on an empty folder:** writes a report without errors.
- **The notebook:** builds from the EXP-022 notebook, and all 8 code cells compile with their notebook magics removed.

- **`python -m unittest discover -s code/tests`:** OK, 169 tests.
- **`python scripts/check_workspace.py`:** PASSED.
- **`latexmk -Werror` in `paper/`:** exit 0. The manuscript was not changed.
- **`python code/analysis/payload_routing_screen.py`:** runs on the committed result files and gives the numbers above; the scratchpad version it was copied from gave the same output.
- **After the third amendment:**
  - `check_prereg()` accepts `469c90ed...`;
  - `python -m unittest discover -s code/tests`: OK, 182 tests;
  - `analyze --allow_partial` on an empty folder writes the report with the Agreement-Routed section;
  - the notebook's 8 code cells compile and are unchanged;
  - `check_workspace.py` PASSED;
  - `latexmk -Werror` in `paper/` exits 0.
- **After the Two-Axis amendment:**
  - `check_prereg()` accepts `03006f5c...`;
  - `python -m unittest discover -s code/tests`: OK, 177 tests;
  - `analyze --allow_partial` on an empty folder writes the report with G3 as n/a and the Two-Axis section;
  - the notebook's 8 code cells compile and are unchanged;
  - `check_workspace.py` PASSED;
  - `latexmk -Werror` in `paper/` exits 0.
- **After the novelty update (documentation only):** `check_workspace.py` PASSED again, the 169 tests are OK, and `check_prereg()` accepts the frozen preregistration (`79752afbf4d52688...`).

## Remaining blockers

- **Not run yet, so no results.** The torch path through the new teacher loop runs first in the notebook's smoke test, which compares it with the released teacher.
- **Push needed:** the notebook clones `rhnl-3`, which must include the runner, the preregistration and the EXP-022 results.
- **Not adopted:** novelty is from targeted searches only, and adoption would need a dated decision record and adviser review.
