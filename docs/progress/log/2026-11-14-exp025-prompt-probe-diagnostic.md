# 2026-11-14: EXP-025 Prompt-Probe Handover diagnostic prepared

Task: EVAL-018 (todo). Requested by Rhenel Jhon R. Sajol, who asked for a fully novel mechanism: not a variant of a known method, not a known method moved to TTS, and not a combination. The requester also asked that ideas not be limited to the project's stored results.

## Why

- **Earlier rounds the same day found nothing that qualified:**
  - Analytic Unconditional Guidance (EXP-024) turned out to be a principled variant of a practised idea: ComfyUI "Uncond Zero" nodes skip the unconditional pass too.
  - A further round of eight candidates all mapped onto known techniques: FreeInit-style noise reshaping, a TTS plus voice-conversion cascade, RePaint, ILVR, autoguidance, student-then-teacher refinement, NIRVANA caching, and shared-prefix sampling.
- **The Prompt-Probe Handover survived** searches of papers and of practitioner tools (F5-TTS code and tools).
- **How it works:** it uses a property of prompt-infilling TTS. Part of the input's answer, the voice prompt, is known. Hiding one second of it from the teacher creates a ground-truth probe of the teacher's current error, at no extra forward pass. A correction fitted on that probe is applied to the target before the handover.
- **The theory:** for an ideal teacher the correction is the identity, because a posterior mean is calibrated. Anything the correction fixes is therefore the network's own bias at that step.
- **What is claimed:** the correction, not the use of the probe error as a when-signal, because that use alone would be a new signal for a known kind of decision.

**Correction the same day, after the freeze and before any synthesis.** Neighbouring fields already have the core move. For Deep Image Prior, held-out measured pixels serve as a reference for early stopping ([arXiv 2605.25299](https://arxiv.org/html/2605.25299v1); see also [Self-Validation](https://arxiv.org/abs/2110.12271)). The Prompt-Probe Handover therefore moves a known idea to TTS and extends it with correction transfer. By the requester's standard that is partly novel, not fully novel. The preregistration is unchanged.

## What changed

- **`code/colab/run_prompt_probe_diagnostic.py`:**
  - **The probe:** 94 frames ending at the prompt's last frame louder than its median, masked in the teacher's conditioning only. F5-TTS advises about 1 s of trailing silence, so the literal last second could be silence.
  - **Corrections:** A, a per-band affine correction with the slope shrunk halfway to 1; B, a per-band voice offset and one slow-part gain.
  - **Measurements:** the probe error at Steps 0–8; each correction's transfer from one half of the probe to the other at Steps 2–4; and the corrected estimate's distance to the teacher's own Step-8 estimate.
  - **Outputs:** P2–P4 (hidden, uncorrected), A2, A3, B2 and B3, scored on 40 development speakers with clean prompts against EXP-018's stored outputs of the released teacher for the same cases and seeds.
  - **Analysis:** reads only probe.csv.
- **`research/experiments/EXP-025-prompt-probe-handover-diagnostic/`:** the README, and `preregistration.json` (sha256 `50d4e86399612b85...`), frozen before any synthesis.
- **`code/tests/test_prompt_probe.py`:** 20 tests, covering:
  - pass accounting and the case lists;
  - the shrunk slope;
  - the identity for a calibrated estimate;
  - removal of a band bias seen on the probe, and the transfer ratio with and without bias;
  - probe placement, which skips trailing silence;
  - a mask that changes only the probe;
  - the teacher loop against the released teacher;
  - the case and smoke loops on a fake model;
  - the decision rule on synthetic outputs;
  - preregistration tampering.
- **`code/colab/DMOSpeech2_EXP025_Prompt_Probe_Diagnostic.ipynb`:** 17 cells from the EXP-023 layout. All 8 code cells compile.
- **Other files:**
  - the EVAL-018 task, with TASKS.md re-rendered;
  - a row in `research/experiments/README.md`;
  - STATUS.md;
  - `.gitattributes`.

## Checks run

- `python -m unittest test_prompt_probe` (from `code/tests`): 20 tests, OK, against the frozen preregistration.
- `python -m unittest discover -s code/tests`: 237 tests, OK.
- `python scripts/check_workspace.py`: PASSED.
- The manuscript was not changed, so it was not rebuilt.

## Remaining blockers

- **GPU run:** it needs a Colab Tesla T4. Nothing has been synthesized.
- **Push first:** the runner and preregistration must be pushed to branch `rhnl-3` before the notebook can clone them.
- **Scope of a pass:** a pass earns a pilot with pitch variety, noisy prompts and an adaptive handover from the probe error, not a claim.
- **Not exhaustive:** the novelty searches cover papers and practitioner tools, but they are not a full review.
