# 2026-11-13: EXP-021 Voice-Swap Handover confirmation prepared

Task: EVAL-014 (todo). Requested by Rhenel Jhon R. Sajol: build a frozen test of fixed Voice-Swap Handover against the 0.25 reference, and report whether its novelty holds.

## Why

- **The claim needs a frozen test.** EXP-020's diagnostic that fixed Voice-Swap at Step 4 (14 passes) matches the reference (21 passes) was chosen after its results were seen, so it cannot be claimed from those data.
- **Why Voice-Swap rather than Half-Rate.** Of the candidates discussed on 2026-11-12 and 2026-11-13, the requester wanted a method whose procedure is new rather than an adaptation. The Half-Rate Teacher idea was set aside because its procedure (draft at low resolution, upsample, re-noise, refine) is published for images: MrFlow, Bottleneck Sampling, DemoFusion.

## What changed

- **`code/colab/run_voice_swap_confirm.py`:**
  - **Algorithm:** Voice-Swap is imported unchanged from the EXP-020 runner.
  - **Outputs:** two noise draws for the reference and for Voice-Swap, plus Step 4 and student only.
  - **Timing:** both paths are timed separately on the T4, with CUDA synchronized, alternating order and a warm-up case.
  - **Diversity:** pitch variety (CVf0) on 30 speakers.
  - **Smoke test:** development speakers only, and it must reproduce the stored EXP-017, EXP-018 and EXP-020 scores.
  - **Freeze and safety checks:** `freeze` pins seven code files and five input files by SHA-256, and the runner refuses to start on any change.
- **`research/experiments/EXP-021-voice-swap-confirmation/`:** README and `preregistration.json` (sha256 `1082c66869fd2f33...`, created 2026-11-13T00:17 UTC), frozen before any synthesis on speakers 61-100.
- **`code/colab/DMOSpeech2_EXP021_Voice_Swap_Confirmation.ipynb`:** 19 cells, built from the EXP-020 layout, with Step 2c added to download the DEMAND cafeteria recording once and cache it in Google Drive.
- **`code/tests/test_voice_swap_confirm.py`:** 22 tests, covering:
  - the pass accounting;
  - the fresh speaker list, which is disjoint from speakers 1-60 and identical to EXP-019's fresh list;
  - seed separation;
  - the run loops on a fake numpy model (warm-up, alternating order, resume, the two draws, Step-4 identity of the timed paths);
  - the Voice-Swap input equalling EXP-020's STT;
  - the analysis on synthetic data (a pass, and a failure for each of P1 to P4, plus partial data);
  - preregistration tampering and code changes.
- Pointer from the EXP-020 README; rows in `research/experiments/README.md`, `tasks.json`, the rendered `TASKS.md` and `STATUS.md`; `.gitattributes` keeps future EXP-021 result bytes unchanged.

## Decisions in this change

- **Re-frozen before any synthesis (same day).** The first freeze hashed text inputs byte for byte. `code/scheduling/failure_config.json` and `research/protocols/splits/development_manifest.json` have CRLF line endings in the Windows checkout but LF on Colab, so the runner would have refused to start there. Both runners now hash text inputs with line endings normalized (`input_hash`, with a regression test). Because that changed the runner code, both preregistrations were deleted, unused, and frozen again.
- **Speakers:** development speakers 61-100, the only Seed-TTS speakers that no method in this project was designed on. EXP-019's preregistration names the same 40 speakers, so the team should decide whether running EXP-021 first is acceptable for EXP-019.
- **Margins sized from EXP-020** by resampling 40-speaker studies. Degraded-prompt WER uses +0.025 because +0.01 or +0.015 would fail a method as good as EXP-020's estimate 60 to 80% of the time. Clean WER uses +0.01, similarity -0.01, CVf0 -0.02, and time at most 0.80 of the reference.
- **Variance reduction:** two noise draws per method and 30 pitch-variety speakers (EXP-020 used 20) were added to reduce sampling noise.
- **CVf0 conditions:** CVf0 uses clean and white-noise prompts, because babble prompts can carry other voices into the pitch tracks.

## Novelty check (targeted searches, 2026-11-08 to 2026-11-13)

No published method was found that chooses which parts of the teacher's estimate cross a teacher-to-student handover. The closest work found:
- **Whole-state handovers:** DMOSpeech 2, SRDiffusion, and Gandikota and Bau.
- **Teacher-student collaboration:** Starodubcev et al. and Distillation++.
- **Frequency mixing:** within one model (frequency-guided image translation, ICCV 2025) and whole-trajectory coupling of two models (coupled diffusion sampling, CVPR 2026).
- **Noisy prompts in TTS:** trained approaches only.
- **The classic building block:** cepstral mean normalization uses the same kind of quantity, a band's log-spectrum average over time. This is not a full literature review.

## Checks run

- `python -m unittest discover -s code/tests`: OK, 120 tests at the first freeze; after the re-freeze below, OK, 140 tests (EXP-021 and EXP-022 together), including the 22 tests in `test_voice_swap_confirm.py`.
- `python scripts/check_workspace.py`: PASSED.
- `latexmk -Werror` in `paper/`: exit 0. The manuscript was not changed.
- Every code cell of the notebook compiles with its notebook magics removed. The DEMAND cell was run locally against a simulated Colab and Drive: a first download that is cached, reuse on a fresh machine, a file already present, and a full Drive all behaved as intended.
- `analyze --allow_partial` on an empty folder writes a report without errors.

## Remaining blockers

- **Not run yet, so no results.** The torch path through this runner runs first in the notebook's smoke test.
- **Push needed:** the branch must include the EXP-020 results, which the smoke test reads. They are recorded locally but not yet committed.
- **Shared speakers:** EXP-019 is also planned for speakers 61-100.
- **Novelty not established:** a full literature review is outstanding, and adoption would need a dated decision record and adviser review.
- **Appendix E** still reports STAC's 19.9% figure, which has 34 Definition 3.1 failures (9.4%), above the 5% limit. Its 1D-CRC-STAC guarantee also does not match `fit_crc_stac`. Neither is fixed here.
