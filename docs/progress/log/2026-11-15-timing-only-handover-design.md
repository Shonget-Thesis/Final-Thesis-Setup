# 2026-11-15: Timing-Only Handover proposed (design and offline checks; not tested)

No task ID yet. Requested by Rhenel Jhon R. Sajol after EXP-029. The requester's bar is now "an existing algorithm with something new inside it", not a fully new mechanism. This entry records the proposal, the offline checks run first, and the literature check. Nothing has been synthesized, and nothing here is adopted.

## Starting point (from EXP-017, EXP-018, EXP-020 and EXP-029)

- **The bar is 13 passes, not 21.** W4 matches the 21-pass reference on clean quality and pitch variety (EXP-026 and EXP-029). W4 is also DMOSpeech 2's own code default.
- **The teacher's estimate brings the prompt's noise back at every handover step.**
  - Mean UTMOS at 25 dB white noise, on the 60 Quick-set speakers (`grid32.csv` and `steps1to8.csv`): student alone 3.55; W1 1.27; W2 1.85; W3 2.10; W4 2.16; W8 2.08.
  - The student alone also has the best similarity and WER in every degraded condition.
  - Earlier handovers are worse, not better: at 20 dB, W1's mean WER is 4.35.
- **The teacher's pitch variety lives in its slow part** (EXP-020 part swaps, 20 clean speakers):
  - Every output that takes the teacher's slow part has CVf0 0.090–0.094.
  - Every output that takes the student's slow part has 0.050–0.054 (student alone 0.0515).
  - The slow part holds both timing and slow pitch movement, and the stored data cannot separate them.

## Offline check of a coherence-length explanation (negative)

- **The idea tested:** re-noising passes a component more strongly the longer it stays constant, so a constant noise floor should pass more for longer utterances. Under mild noise, the handover's penalty against the student alone should then grow with utterance length.
- **Data:** stored data only, run before anything was built (`grid32.csv`, `audio_seconds`).

| Condition | UTMOS student − W4 | Spearman with length (permutation p) |
|---|---|---|
| clean | +0.011 | +0.00 (0.99) |
| gauss25 | +1.389 | −0.05 (0.73) |
| gauss22 | +1.661 | +0.10 (0.44) |
| gauss20 | +1.426 | +0.11 (0.38) |
| gauss18 | +1.040 | +0.18 (0.16) |

- **Result:** copying is already saturated at 25 dB for 3–7 s utterances. Length gives no lever, so no method was built on this.

## The proposal: Timing-Only Handover (TOH)

**The existing algorithm is kept:** DMOSpeech 2's teacher-guided sampling (guided teacher for k sway steps, then the distilled student from its first restart level with fresh noise).

**The new part is what crosses the handover.** Only the teacher's timing crosses, as a time map applied to the student's own draft:

1. **Student draft:** S₀ = the student's one-step output from pure noise (1 pass). This is the student-alone sampler's first step.
2. **Teacher:** the guided teacher runs k sway steps and gives its endpoint estimate T_k (2k + 2 passes), with k = 2 or 3.
3. **Timing map:** a monotone, slope-limited dynamic-time-warping path w between S₀ and T_k.
   - **Features:** per-frame band-group log energies, z-scored over time within the utterance, which removes per-band offsets.
   - **Constraints:** slopes between 1/2 and 2, and a band around the diagonal.
4. **Payload:** S̃ = S₀ resampled along w.
5. **Student completion:** y = τ₁ S̃ + (1 − τ₁) ε′ with fresh noise, then the student's three remaining levels (3 passes).

**Cost:** 2k + 6 passes, so 10 at k = 2 and 12 at k = 3. For comparison: 13 for W4 (DMOSpeech 2's default) and 21 for the reference.

**Why it could work:**
- **The noise can't cross.** w only moves the student's own frames in time. No spectral value from the teacher reaches the student, so the recording condition the teacher copies is excluded by the payload's type, not by a detector. Voice-Swap removes only the per-band average.
- **A bad teacher is bounded.** The slope limits mean a bad teacher can only re-time the student's draft within bounds.
- **The restart cleans up.** The restart at τ₁ keeps broadband structure such as timing and regenerates fine detail, which should erase interpolation artifacts.
- **A clean control exists.** With w = identity, TOH equals the student alone exactly. The warp is the only difference.

**What is known and what is new.**
- **Known:** DTW alignment and rhythm transfer (speech processing and voice conversion), SDEdit-style re-noising of an edited input, DMOSpeech 2's handover, and warping in video diffusion (noise warped along motion for temporal consistency).
- **Not found:** a teacher-to-student handover whose only payload is a time map of the student's draft.
- **Status:** partly novel under the strict rubric; it meets the requester's current bar.

## Literature check (targeted, 2026-11-15)

Seven searches looked for:
- DTW or time-warp handovers between teacher and student diffusion TTS models;
- noise and latent warping in diffusion;
- rhythm transfer with DTW in zero-shot TTS;
- large-model-layout, small-model-detail cascades;
- noise copying from prompts;
- time warping with SDEdit for speech.

The closest work:
- **[How I Warped Your Noise](https://iclr.cc/virtual/2024/oral/19723) (ICLR 2024):** warps noise along optical flow for temporal correlation in video. It warps noise, not a payload between models.
- **[Clockwork Diffusion](https://openaccess.thecvf.com/content/CVPR2024/papers/Habibian_Clockwork_Diffusion_Efficient_Generation_With_Model-Step_Distillation_CVPR_2024_paper.pdf) (CVPR 2024):** a layout/detail split inside one network, with no warp.
- **[LeapTalk](https://arxiv.org/pdf/2608.00079):** warps the noise schedule (the diffusion time variable) between a teacher and a student, not the signal's time axis.
- **[Noise robustness of flow-matching zero-shot TTS](https://arxiv.org/abs/2406.05699) (Interspeech 2024):** zero-shot TTS copies prompt noise. Its fixes are trained.
- **[arXiv 2502.07345](https://web3.arxiv.org/pdf/2502.07345):** removing the prompt's background can conflict with applications that want it kept.
- **No source found** says a distilled TTS student is cleaner than its teacher on noisy prompts. That is this project's own development finding (EXP-017 and EXP-018).

This is not a full review.

## Predictions a pilot would test

| | Prediction | Distinguishes |
|---|---|---|
| P1 | On clean prompts, TOH_k keeps W_k's pitch variety | Timing carries the variety. A failure would point to slow pitch movement instead. |
| P2 | On degraded prompts, TOH_k matches the student alone and beats W4 on UTMOS | The condition cannot cross. |
| P3 | Clean quality within the −0.08 UTMOS bar against W8 | The restart cleans up warp artifacts. |
| P4 | No more runaway outputs (WER > 1) than the student alone | The slope limit bounds a bad teacher. |

**Controls:**
- the identity warp, which equals the student alone;
- a warp from another seed's teacher, which tests seed-specific timing.

## Risks

- **Variety may be pitch, not timing.** EXP-020 cannot separate them. P1 is the deciding test.
- **The student may re-impose its own timing** at τ₁.
- **Compute:** 10–12 passes is only 8–23% below 13. On white noise, routing noisy prompts to the student alone (4 passes) is cheaper. TOH's case rests on one operator for all prompt conditions, including ones SNR cannot detect, such as telephone or reverberation.

## Checks run

- The two stored-data checks above (`grid32.csv`, `steps1to8.csv`, EXP-020 `diversity.csv`), run as inline scripts. They are exploratory and post hoc on development speakers.
- No code, preregistration or task was added.

## Remaining blockers

- **Requester decision:** whether to build a pilot (EXP-030).
  - **Proposed arms:** student alone, W2, W3, W4, TOH₂, TOH₃, and the two controls.
  - **Pitch variety:** 20 clean development speakers × 12 seeds.
  - **Quality:** 30 development speakers × clean, gauss25, gauss20 and noisy15.
  - **Speakers:** development speakers 91–120, so the calibration and test speakers stay untouched.
- **No offline screen is possible.** No stored data hold a teacher or student draft from which a warp could be built.

**Update (same day):** the requester approved the pilot. It is EXP-030 (EVAL-023); see the [setup log](2026-11-15-exp030-timing-only-handover-setup.md). The other-seed control was replaced by a mirrored-map control (MIR3), and TOH4 was dropped; the setup log gives the reasons.
