# 2026-11-15: What all experiments so far say about the handover algorithm (synthesis, no new data)

No task ID. Requested by Rhenel Jhon R. Sajol: "analyze all our experiment results, maybe we could get an idea on making the inference handover algorithm".

**Sources read:**
- the experiments index (`research/experiments/README.md`, EXP-000 to EXP-031);
- the READMEs of EXP-015, EXP-016, EXP-018, EXP-019 and EXP-020;
- the offline solution screen (`2026-11-13-offline-solution-screen.md`, rounds 1–8);
- the EXP-023, EXP-029 and EXP-030 result logs.

Every result cited is development evidence. Most failure counts use Definition 3.1, so they are judged against W8.

## What is established

| | Finding | Experiments |
|---|---|---|
| E1 | One fixed switching time does not suit every input. Noisy prompts fail at early steps (322 of 500 at Step 4), clean prompts rarely (12 of 500). A per-input oracle would need 14.85 instead of 21 passes | EXP-001, EXP-002 |
| E2 | **SNR is the only reliable router.** Drift adds nothing beyond SNR, LTT reduces the Chapter 3 policy to SNR-only (λ̂ = 1.20), and noise lowers drift, so noisy prompts look "settled" early. No teacher feature or agreement signal beats SNR | EXP-002, 004, 006, 008, 009, 012, 017, 023 (ARH: 23 misroutes against 0), 029 |
| E3 | **Teacher steps after 3 or 4 add nothing measurable on clean prompts.** Step 3 is as safe as Step 4 on clean prompts, W4 matches W8 on quality and variety, and later restarts trade quality for variety | EXP-003, 026, 027, 029 |
| E4 | **Every handover copies the prompt's recording condition, at every step.** The student alone is cleaner on noisy prompts. Fixes that clean the payload or the prompt help in proportion to how much of the condition they block: Voice-Swap 126 → 74 failures; prompt enhancement and SWC about 126 → 21–31; TOH as good as the student alone | EXP-013, 014, 015, 017, 020, 029, 030 |
| E5 | **The teacher's pitch variety lives in its slow part.** The teacher's change after Step 0 is mostly shared across seeds. A random re-timing of the same size (the mirrored map) restores the teacher's CVf0 | EXP-020, 022, 023, 030 |
| E6 | **The 20 dB gate misses noise at 18–22 dB, and SNR cannot see reverberation or telephone band-limiting** (telephone reads about 44 dB) | EXP-011, 018 (OOD), 029 |
| E7 | **Routing by SNR, with noisy prompts going to the student alone, is the cheapest structure that passes on stored data:** about 6–8 mean passes on the white-noise Quick set. Its only miss was pitch variety, when clean prompts were sent to the student. A 2026-11-13 targeted search found no published method that routes noisy prompts to the distilled student | Offline screen, rounds 1, 2 and 4 |

## What this implies for the algorithm

1. **The "when" axis is used up.** Every timing signal tried reduces to SNR (E2), and the step choice on clean prompts is only Step 3 against Step 4, two passes (E3). A new stopping rule is unlikely to be a contribution.
2. **The "who" decision is simple and known.** SNR sends noisy prompts to the student alone (E4, E7). The direction is new compared with Chapter 3's policy, which sends noisy prompts to Step 8, the step that copies the noise.
3. **What remains open is "what crosses":**
   - **The trade-off:** passing the whole estimate copies the condition (E4), while passing nothing loses variety (E5).
   - **The requirement:** a payload that keeps the teacher's useful structure and cannot carry the condition.
   - **What it buys:** the router's mistakes (E6) become harmless in the dangerous direction, because a noisy prompt the gate misses still cannot pass its noise.
4. **The resulting algorithm:**

   | Step | What happens | Cost |
   |---|---|---|
   | 1 | Compute the prompt SNR | Free |
   | 2 | Below the gate: the student alone | 4 passes |
   | 3 | Otherwise: the teacher to Step 3, then a condition-safe payload (TOH, or the Quantile Handover from the [theory note](2026-11-15-handover-invariance-theory.md)), then the student finishes | 12 passes |

   This is EXP-031's `SNR_TOH`.

   **Known parts:** the SNR estimator, the 20 dB gate and DMOSpeech 2's handover.

   **New parts:** the condition-safe payload, and the routing direction justified by E4.

## Open questions the evidence leaves (and where they are tested)

| Question | Where it is tested |
|---|---|
| Is the teacher needed at all for variety, given E5? | EXP-031's random-map control (`RND`). If it holds, a teacher-free 4-pass student with re-timing would be the cheaper route, but it would change the thesis's "teacher-guided" framing (team and adviser decision) and touch the scope rule on duration (internal timing only) |
| Is TOH's variety real pitch variety, or only timing? | EXP-031's aligned CVf0. If TOH falls short, the Quantile Handover is the payload the theory predicts would keep pitch movement |
| Does the safety net help where SNR is blind? | EXP-031's S1, on reverberation and babble that pass the gate |
| How does it compare with cleaning the prompt first? | EXP-015 has the Demucs enhancer, but not in the same run as TOH or QH |
| Can the gate move once the payload is condition-safe? | With a condition-safe payload, the gate trades only compute (4 against 12 passes) and variety, not quality. It could then be set from compute rather than from failure risk. Untested |

## Checks run

None. This is a reading of committed results, and no numbers were recomputed.
