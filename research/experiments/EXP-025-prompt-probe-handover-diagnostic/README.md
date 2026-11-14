# EXP-025: Prompt-Probe Handover (development diagnostic)

**Status:** prepared, not yet run. Preregistration frozen 2026-11-14 before any synthesis (`50d4e86399612b85...`). Task EVAL-018 (todo). The run needs a Colab Tesla T4: [notebook](../../../code/colab/DMOSpeech2_EXP025_Prompt_Probe_Diagnostic.ipynb), [runner](../../../code/colab/run_prompt_probe_diagnostic.py), [tests](../../../code/tests/test_prompt_probe.py).

- **A diagnostic:** it decides whether a pilot is worth building, and cannot support a claim.
- **Not adopted:** nothing here changes the thesis method, and no adviser has reviewed it.

## Idea

In prompt-infilling TTS, part of the input's answer is known: the voice prompt.
1. **Hide one second.** The teacher does not see the prompt's last second of speech. Its words stay in the text and the canvas length is unchanged, so that second simply joins the part the teacher must generate.
2. **Measure the teacher.** At every step the teacher's estimate of the hidden second can be compared with the real audio. That is a ground-truth measurement of the teacher's current error, for this speaker, this sentence and this step, at no extra forward pass.
3. **Correct the handover.** A small correction is fitted on the hidden second (teacher estimate → real audio) and applied to the target before the student finishes. The student keeps the full prompt.

**Theory:** for an ideal teacher, the endpoint estimate is the posterior mean, which is calibrated, so the fitted correction is the identity. A correction that is not the identity therefore measures the network's own bias at that step, where the answer is known. Earlier data hint that such a bias exists: 70–94% of the teacher's early change is the same for every seed (EXP-023).

**Probe placement:** the hidden second is the 94 frames ending at the prompt's last frame louder than its median frame. F5-TTS advises about 1 s of trailing silence in references, so the literal last second could be silence.

| Output | What it is | Passes |
|---|---|---:|
| P2, P3, P4 | second hidden, no correction | 9, 11, 13 |
| A2, A3 | correction A, per mel band y = a x + c, slope shrunk halfway to 1 | 9, 11 |
| B2, B3 | correction B, per-band voice offset and one gain on the 9-frame slow part | 9, 11 |
| W4, W8 | released Step-4 handover and the 0.25 reference (stored, EXP-018) | 13, 21 |

## Novelty check (papers and practitioner tools, 2026-11-14)

No method was found that hides part of the conditioning to create a known-answer probe of the generator at inference, and uses it to correct what is handed over. The closest work:
- [CoPaint](https://proceedings.mlr.press/v202/zhang23q/zhang23q.pdf) uses the known region as a **constraint** during sampling.
- [TD-Paint](https://arxiv.org/pdf/2410.09306) measures known-region error as a **diagnostic**.
- [Vox-Evaluator](https://arxiv.org/pdf/2510.20210) masks and regenerates **generated** speech.
- [VoiceStar](https://arxiv.org/pdf/2505.19462) masks prompts **during training**.
- [NoiseShift](https://arxiv.org/html/2510.02307v1) and [cache calibration](https://arxiv.org/pdf/2605.24870) fit corrections **offline**.
- A search of F5-TTS code and tools found nothing similar.

Using the probe's error only to decide *when* to hand over would be a new signal for a known kind of decision, so that use alone is not claimed as a new mechanism. Not a full literature review.

**Correction after the freeze (2026-11-14, before any synthesis).** A search of neighbouring fields found the core move already published for Deep Image Prior:
- [arXiv 2605.25299](https://arxiv.org/html/2605.25299v1) holds out a small set of measured pixels as a reference that a generative reconstruction never trains on, and uses it to decide when to stop.
- [Self-Validation](https://arxiv.org/abs/2110.12271) is a related early-stopping method.

So "hold out part of the known input to grade a generative process" is a known kind of idea. What this diagnostic adds is applying it to a teacher-student handover in TTS, and transferring a correction fitted on the held-out part to the unknown part. By the requester's standard that is partly novel (a known idea moved to a new domain, plus an extension), not fully novel. The preregistration is unchanged.

## What runs

40 development speakers (Quick-set positions 1–40), clean prompts, synthesis seed = input seed. Per case:
- the teacher runs to Step 8 with the second hidden;
- the probe error is recorded at Steps 0–8;
- at Steps 2–4, each correction is fitted on the first half of the hidden second and measured on the second half (transfer);
- the distance of the corrected estimate to the teacher's own Step-8 estimate is recorded;
- the seven outputs are scored.

The smoke test (Quick-set positions 59–60) must pass three checks:
1. this runner's teacher loop matches the released teacher (≤ 1e-4);
2. hiding the probe changes the conditioning only inside it;
3. the stored W3 and W4 scores are reproduced.

## Criteria (frozen)

| Criterion | Rule |
|---|---|
| G1 transfer, per variant | median over cases of corrected/plain error on the unseen half of the probe, at Step 3, ≤ 0.9 |
| G2 quality, per variant and Step 2 or 3 | the corrected handover against the released Step-4 handover of the same case and seed (speaker bootstrap, 95%): capped-WER upper bound ≤ +0.01, similarity lower bound ≥ −0.01, UTMOS lower bound ≥ −0.10; and its UTMOS minus the plain hidden-probe handover at the same step, mean > 0 |

**Decision rule:**
- **Build a pilot** only if some variant meets G1 and G2 at some step. The pilot would add pitch variety, noisy prompts and an adaptive handover from the probe error.
- **Carry forward** the earliest step; ties go to the larger UTMOS gain over plain.
- **Otherwise** report a negative diagnostic, with no retuning on these data.

**Power:** with 40 speakers, the clean Step-4 bounds pass a change as small as refining the teacher's grid in 100% of 300 subsets. This comes from the EXP-024 power check, [`code/analysis/exp024_power_check.py`](../../../code/analysis/exp024_power_check.py).

## Risks known before the run

- If the network is not biased, the correction stays near the identity and nothing is gained.
- Hiding one second of the prompt reduces the teacher's voice context; P4 against W4 measures that cost.
- On noisy prompts the hidden second's "truth" contains noise, so the correction would pull toward it. Noisy prompts are therefore not in this diagnostic.
- Pitch variety is not measured here.

## Data and evidence status

Development evidence only, on speakers that designed earlier experiments, with clean prompts. Results will be stored under `results/` with a SHA-256 manifest. Audio is not committed.
