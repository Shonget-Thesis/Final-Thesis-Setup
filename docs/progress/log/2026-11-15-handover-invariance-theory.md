# 2026-11-15: Condition-safe handover theory (TOH proposition, invariance ladder, Quantile Handover): a theory note, not tested on the model

No task ID yet. Requested by Rhenel Jhon R. Sajol:
1. a proposition and proof for TOH, as a plain note, not in the manuscript;
2. a "deep think" for a rarer, theory-derived contribution of the team's own.

Nothing here has been run on the model, nothing is adopted, and no adviser has reviewed it. The manuscript is unchanged.

## Setting

- **Arrays:** log-mel arrays over the generated frames. T ∈ ℝ^{n×B} is the teacher's endpoint estimate and S ∈ ℝ^{n×B} is the student's one-step draft, with B = 100 bands.
- **Payload:** a handover payload H(T, S) ∈ ℝ^{n×B} is what the student is re-noised from at its first restart level. The released handover W_k has H = T.
- **The condition model:** a stationary recording condition acts on each band in the same way at every frame: T_b(t) ↦ g_b(T_b(t)).
  - **Channel and noise:** for a linear channel gain h_b > 0 and a stationary additive noise power ν_b ≥ 0 (power domain), g_b(x) = log(h_b·eˣ + ν_b), which is strictly increasing.
  - **Classes of condition:**
    - **G1:** per-band shifts, x ↦ x + c_b (pure gain);
    - **G2:** per-band positive affine maps, x ↦ a_b·x + c_b;
    - **G3:** per-band, time-invariant, strictly increasing maps.
  - **Nesting:** G1 ⊂ G2 ⊂ G3, and G3 contains every stationary channel combined with a stationary noise floor.
  - **Idealization:** the noise power is taken as deterministic. Real noise adds a random texture per frame, which is not a group action. Reverberation, babble and band-limiting (telephone) are outside G3.

**Two safety notions:**
- **Value-safe:** every payload value in band b is a convex combination of the student's values in band b. No teacher value crosses.
- **G-safe:** H(g·T, S) = H(T, S) for every condition g in G. The condition cannot change the payload.

## Proposition 1 (Timing-Only Handover, EXP-030)

Let w be TOH's map, computed by `run_timing_only_handover.timing_map`:
- **Features:** 10 band-group means, a 3-frame moving average with edge padding, and a z-score per group over time.
- **Alignment:** constrained DTW.
- **Map:** the mean student frame per teacher frame.

The payload is S re-timed along w by linear interpolation; the prompt frames are unchanged. Then:

- **(a) Invariance of the map:** w is unchanged by every per-band map g_b(x) = a_γ·x + c_b, where a_γ > 0 is the same for all bands in a group γ.
  *Proof.* A group mean becomes a_γ·f_γ(t) + const. The moving average is linear, and with edge padding it keeps constants. So the smoothed feature is a_γ·f̃_γ(t) + const, and the per-group z-score removes both. The cost matrix, the DTW path (ties are broken deterministically) and w are therefore unchanged. ∎
- **(b) Value safety:** each payload value is x[lo]·(1 − frac) + x[hi]·frac, a convex combination of two of the student's own values in the same band. ∎
- **(c) The student alone is a special case:** with w = identity, frac = 0 everywhere, so the payload is S and TOH equals the student alone exactly. EXP-030's smoke test measured this difference as 0.0. ∎
- **(d) Bounded influence:** the teacher can only move the student's frames in time, with steps between 0 and 2 and |w(i) − i| ≤ band + 1 (Amendment 1).

**What it does not say:**
- **Additive noise can still affect the map.** The map is not G3-invariant, because noise is not affine in log-mel. A noisy teacher can change *where* the map says events happen, but not *which values* reach the student.
- **Nothing is claimed about pitch.** TOH carries only timing.

## Theorem 2 (the invariance ladder)

For each class G, a payload is G-safe if and only if it depends on T only through a maximal invariant m_G(T). The maximal invariants are:

| Class | Maximal invariant | Most informative safe payload |
|---|---|---|
| G1 | Per-band centred estimate, T_b − mean_t T_b | Add back the student's per-band mean. This is Voice-Swap (EXP-020) |
| G2 | Per-band z-scores (bands with non-zero variance) | z_T,b·σ_S,b + μ_S,b. Call it the **Affine Handover** |
| G3 | Per-band rank vectors r_b(T), the time order of each band's values | Q(T, S)_b(t) = sort(S_b)[r_b(T)(t)]: the student's values in each band, arranged in the teacher's time order. Call it the **Quantile Handover (QH)** |

*Proof sketch.* A safe payload is constant on each orbit of G, so it is a function of any maximal invariant.
- **G3:** for vectors with distinct entries, x and y lie in the same orbit exactly when they have the same rank vector. Strictly increasing maps keep ranks, and when ranks agree, linear interpolation through the sorted pairs (x₍ᵢ₎, y₍ᵢ₎) is a strictly increasing map with φ(x) = y. Per band, this gives the per-band rank vectors (with ties, the tie pattern counts too).
- **G1 and G2:** the same argument applies, with centring and z-scoring. ∎

The rank result is textbook invariance theory, used in rank tests. What is new here is its use to characterize which teacher-to-student handovers can carry a recording condition.

**Properties of QH that follow:**
- **Exactly safe against stationary conditions:** it is G3-safe, so it is unchanged by any stationary channel together with any stationary noise floor, in the idealized model.
- **Stronger than value-safe:** each band of the payload holds exactly the student's values, so the payload keeps the student's noise floor, levels and dynamic range, only reordered.
- **Maximally informative among G3-safe payloads:** with the student's values distinct, different teacher rank patterns give different payloads.
- **Reduces to the student alone:** if T differs from S only by a stationary condition (T_b = φ_b(S_b) with φ_b strictly increasing), then QH(T, S) = S.
- **Same cost as TOH:** 2k + 6 passes. QH needs the student's draft for its values but no DTW: in each band it is one sort and one argsort. It does not change the utterance's length.

**The ladder:** W (no invariance) → Voice-Swap (G1) → Affine Handover (G2) → QH (G3). Going up the ladder, the payload is safe against more conditions but carries less of the teacher's information. TOH is not on this chain: it is value-safe, its map is invariant to G1 plus a uniform scaling per band group, and it carries only timing.

## Predictions the theory makes (testable on the model)

| | Prediction |
|---|---|
| P1 | Noise-level leakage on stationary noise: W > Voice-Swap > Affine ≥ QH, and QH is exactly zero. The Affine Handover's residual grows at lower SNR |
| P2 | Teacher information kept (pitch pattern, alignment-aware pitch variety): W ≥ Voice-Swap ≥ Affine ≥ QH ≥ TOH ≈ student alone |
| P3 | On white-noise prompts, QH is as clean as TOH and keeps more of the teacher's pitch variety. On reverberation or babble, which are outside G3, QH may let the teacher's temporal smearing through, while TOH passes no patterns |

**First test on the model (EXP-031, 2026-11-16; exploratory, development speakers):**
- **P2, consistent:** aligned CVf0 ran W4 0.0584 ≥ QH3 0.0521 ≥ TOH3 0.0435 ≈ student alone 0.0410. QH3 − TOH3 was +0.0086 [+0.0022, +0.0160], and TOH3 − STU +0.0025 [−0.0021, +0.0074]. Voice-Swap and the Affine Handover were not run.
- **P3, white noise:** QH3 was as clean as TOH3 (gauss25 3.479 against 3.470; noisy15 2.009 against 1.883).
- **P3, reverberation and babble:** no difference was shown (QH3 − TOH3: −0.007 [−0.033, +0.019]).
- **P1:** not measured.
- **The frozen verdicts:** under these, QH3 did not show that its teacher is needed, because a random re-timing matched it on raw CVf0. See the [EXP-031 results log](2026-11-16-exp031-results.md).

**It explains an earlier result.** White noise acts on log-mel as g(x) = log(eˣ + ν), which a per-band shift cannot undo. That is why Voice-Swap only reduced noisy-prompt failures from 126 to 74 of 300 in EXP-020, while the student alone did better.

## Toy check (synthetic spectrograms, not model outputs)

- **Script:** `code/analysis/handover_ladder_toy.py`.
- **Setup:**
  - harmonic speech-like log-mel, with teacher and student renderings that differ in timing and pitch movement;
  - the teacher copies white noise, additive in power, with a gamma-distributed texture;
  - 12 trials per SNR.
- **Measures:**
  - **Floor:** the per-band 10th-percentile level minus the clean teacher's (0 dB means no level leakage).
  - **Pattern:** the correlation of the low-band harmonic pattern with the clean teacher over speech frames (how much of the teacher's pitch structure is kept).

| Payload | Floor at 25 dB | Floor at 15 dB | Teacher pattern (25 / 15 dB) |
|---|---:|---:|---|
| W (released) | +4.3 | +12.2 | 0.978 / 0.972 |
| Voice-Swap (G1) | +1.9 | +3.4 | 0.944 / 0.938 |
| Affine (G2) | −0.0 | −1.4 | 0.821 / 0.830 |
| QH (G3) | +0.0 | +0.0 | 0.659 / 0.659 |
| TOH | +0.0 | +0.0 | 0.466 / 0.471 |
| Student alone | +0.0 | +0.0 | 0.465 / 0.465 |

- **Ordering:** it is the one the ladder predicts.
- **Voice-Swap:** it leaves part of the floor, as on the real model.
- **QH:** it is the most informative payload with exactly zero level leakage.
- **TOH:** it carries no more of the teacher's pitch pattern than the student alone.
- **The Affine Handover:** it is close behind QH on level in this toy, with a small over-correction at 15 dB.
- **A dropped measure:** "silence texture" was tried and dropped, because timing differences between the two renderings confound it.

**Added the same day: clean prompts in the toy.**
- **QH is not a pass-through on clean prompts.** It does not return the teacher's estimate unchanged.
- **Its output is the same with and without noise:**

  | Payload | RMS distance from the clean teacher | Teacher pattern | Same at 25 and 15 dB? |
  |---|---|---|---|
  | QH | 12.3 dB | 0.69 | Yes |
  | TOH | 15.3 dB | 0.47 | Yes |
  | Student alone | 20.7 dB | 0.47 | Yes |

So in this toy, QH and TOH are condition-invariant: they need no switch between clean and noisy prompts.

This is a sanity check of the logic only. It says nothing about how the student re-renders a payload or how it sounds.

## Novelty audit (targeted, 2026-11-15; six searches)

**Known: the QH operator.**
- **Quantile-based histogram equalization** maps Mel filterbank outputs onto reference distributions for noise-robust ASR ([Hilger and Ney, Eurospeech 2001](https://www.isca-archive.org/eurospeech_2001/hilger01_eurospeech.html); [ICSLP 2002](https://www.isca-archive.org/icslp_2002/hilger02_icslp.html); [IEEE TSAP 2006](https://bvbvf1.bib-bvb.de/vufind/PrimoRecord/cdi_proquest_miscellaneous_889383655)). Its motivation is the same: noise distorts each band's distribution, so match the distributions.
- **Image histogram matching:** ComfyUI nodes match a diffusion output's histogram or colours to a reference image (for example [ImageHistogramMatch+](https://www.runcomfy.com/comfyui-nodes/ComfyUI_essentials/ImageHistogramMatch-), [Color Transfer](https://docs.comfy.org/built-in-nodes/ColorTransfer.md), [KJNodes ColorMatch](https://www.runcomfy.com/comfyui-nodes/ComfyUI-KJNodes/ColorMatch)).
- **The rank result:** a standard invariance fact.

**Not found:**
- **At a handover:** histogram or quantile matching at a teacher-to-student handover, or using the student's own draft as the reference distribution. The closest results were distillation papers in general ([DMD2](https://proceedings.neurips.cc/paper_files/paper/2024/file/54dcf25318f9de5a7a01f0a4125c541e-Paper-Conference.pdf); [progressive distillation](https://arxiv.org/pdf/2202.00512)).
- **A characterization of condition-safe handovers:** which handover payloads can carry a recording condition. Invariance transfer appears only for classifiers or for training-time adaptation ([TransInv](https://link.springer.com/chapter/10.1007/978-981-96-6594-5_28); [adversarial teacher-student](https://arxiv.org/pdf/1804.00644)).
- **Training-free fixes for zero-shot TTS:** an inference-time fix for prompt-condition copying. Published fixes retrain: noise-augmented fine-tuning ([Interspeech 2024](https://arxiv.org/pdf/2406.05699)), controllable background removal ([arXiv 2502.07345](https://arxiv.org/pdf/2502.07345)), and [OZSpeech](https://arxiv.org/pdf/2505.12800).

**Status:**
- **QH is partly novel:** a known operator, used in a new place and with a new reference.
- **The framework is the candidate for the team's own contribution:** condition-safe handovers, the ladder of maximal invariants, and its predictions, including the explanation of EXP-020. It still needs a fuller audit: invariance and safety of transfer between generative models, robust zero-shot TTS, and neighbouring fields such as climate bias correction by quantile mapping.

**Prompt enhancement and noise-floor removal were already tested (corrected the same day):**
- **EXP-015 (development, A100) and EXP-016 (T4 replication):**
  - **Prompt enhancement:** a pretrained DNS64 Demucs enhancer on the prompt, routed by SNR.
  - **SWC:** a Wiener gain applied to the prompt, from its quietest frames.
  - **PIEC:** cancels the prompt's noise floor from the output.
- **How QH differs:** all three estimate the noise from the prompt. QH needs no noise estimate, because it takes its reference levels from the student's own draft and is invariant to any stationary monotone condition.
- **The gap:** neither EXP-030 nor EXP-031 compares TOH or QH with the enhancer in the same run.

## Checks run

- `python code/analysis/handover_ladder_toy.py`: the output is the table above.
- The six web searches listed above.
- No model run. LaTeX build: not run, because the manuscript did not change.

## Open decisions for the requester

1. **Testing QH:**
   - **Option A:** add QH3 (and possibly the Affine Handover) to EXP-031 by a dated amendment and re-freeze, before any synthesis. That adds about two scored arms, roughly 20 more minutes.
   - **Option B:** run EXP-031 as frozen, and test QH in EXP-032 only if TOH3's aligned pitch variety falls short of W4's, which is what P2 predicts.
2. **A same-run prompt-enhancement baseline:** EXP-015's Demucs front-end, in the same run as TOH and QH.
3. **Whether to pursue the framework as the thesis's own theoretical contribution.** This is a team and adviser decision.
