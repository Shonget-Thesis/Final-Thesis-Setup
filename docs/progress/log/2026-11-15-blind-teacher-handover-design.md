# 2026-11-15: Blind-Teacher Handover proposed (design and literature check; not tested)

No task ID yet. Requested by Rhenel Jhon R. Sajol: set the title and paper aside, focus on the compute-cost problem, and find another novel teacher-to-student handover after SNR, TOH and QH. Built on the [evidence synthesis](2026-11-15-evidence-synthesis-handover-design.md). Nothing has been synthesized, and nothing is adopted.

## Reasoning from the evidence

- **Where the cost goes:** the student costs 4 fixed passes. Each teacher step costs 2 passes (classifier-free guidance), and every teacher pass processes the prompt frames as well as the target frames.
- **What the teacher contributes:**
  - **useful:** per-seed structure, mainly timing (EXP-020, 023, 030), and a little naturalness;
  - **harmful:** the prompt's recording condition, which it copies at every step (EXP-013 to 015, 017, 020, 029, 030).
- **What follows:** once only timing crosses (TOH), the teacher no longer needs to know who is speaking. The prompt is then pure cost for the teacher, and the only route by which the condition reaches it.

## The proposal: Blind-Teacher Handover (BTH)

**Kept from DMOSpeech 2:** the teacher runs k guided steps; the student finishes from its first restart level.

**Changed:** the teacher never receives the prompt.

1. **Student draft:** S₀ = the student's one-step draft, conditioned on the prompt as usual. It carries the voice (1 pass).
2. **Blind teacher:**
   - **Input:** the **target text only**, over the target's frames only, with the audio condition dropped. This is the model's text-only branch, the same one used for audio-dropout CFG in training.
   - **Guidance:** the usual 2 passes per step, text-conditioned against unconditioned.
   - **Output:** T̃_k after k steps.
3. **Payload:** TOH's time map w from T̃_k to S₀, applied to S₀. Only timing crosses.
4. **Finish:** the student's three remaining levels (3 passes).

**What follows from the design:**
- **Exact independence from the prompt through the teacher path.** The map w is a function of the text, the duration and the teacher's seed only. No property of the prompt can affect it: no noise, reverberation, babble, band-limiting, or even the speaker. The prompt reaches the output only through the student, and the student is the model the data show to be robust to white noise (EXP-017, 030).
  - **Compared with TOH:** TOH's map is computed from a teacher that has copied the condition, so a condition can distort *where* the teacher says events happen.
  - **Compared with QH:** QH is invariant only to stationary conditions.
- **Cheaper teacher passes.** The pass count is unchanged (2k + 6, so 12 at k = 3), but each teacher pass runs on the target's frames instead of prompt plus target. Attention cost falls faster than linearly with length, and the saving grows with the prompt's share of the sequence. It still has to be measured in the pilot.
- **No routing needed** for robustness.
- **Latency:** the blind teacher does not depend on the prompt, so it can run in parallel with the student's draft.

## Risks

- **Generic rhythm.** The teacher's local rhythm is speaker-independent. Only the total length, set by the duration predictor from the prompt, follows the speaker. Speaker-specific timing could be lost.
- **Alignment across voices.** TOH's map aligns z-scored band-group energy contours, which follow phonetic structure more than voice. That should hold, but it is untested.
- **The teacher may not be needed at all.** If EXP-031's random-map control (`RND`) matches TOH3, the teacher's timing does not matter, and BTH is moot like every other teacher-timing idea.
- **Implementation:** the teacher needs its own context (target text only, target length, audio condition dropped). It cannot be checked here without PyTorch; a smoke test on the T4 would have to.

## Literature check (targeted, 2026-11-15; three searches)

**Closest:**
- **Staged models:** [DMOSpeech 2](https://arxiv.org/pdf/2507.14988) itself (teacher for early prosodic structure, student for acoustics), Hybrid SD (large model plans early, small model refines), and [T-Stitch](https://mlanthology.org/iclr/2025/pan2025iclr-tstitch) (the reverse order). In all of them the early model sees the full conditioning.
- **Prosody transfer between voices:** [Exact Prosody Cloning](https://ar5iv.labs.arxiv.org/html/2206.12229), R-VC (rhythm transfer for voice conversion; known from a search summary only, source not opened) and [ASR-TTS cascade voice conversion with prosody transfer](https://arxiv.org/pdf/2009.01475). All are trained systems.
- **Distillation with a differently conditioned teacher:** [arXiv 2609.11523](https://arxiv.org/html/2609.11523) (instruction paraphrase, training time) and [arXiv 2501.08566](https://www.arxiv.org/pdf/2501.08566) (teacher-generated parallel data, training time).

**Not found:** an inference-time handover in which the teacher is deliberately blind to the speaker prompt and only its timing crosses to a prompt-conditioned student.

**Status:**
- **Partly novel:** an existing algorithm with a new element. The element is that the teacher runs without the prompt, which follows from the project's own findings: the teacher's useful payload is timing, and its harmful payload comes from the prompt.
- **Before any claim:** a fuller audit is needed.

## Candidates considered and set aside the same day

| Candidate | Why it was set aside |
|---|---|
| A teacher without guidance (1 pass per step) | Halves the teacher's passes, but risks the text alignment that guidance secures. It is a known kind of change |
| A teacher that calibrates a speaker's timing variation once and a student that samples it per utterance | Large saving only when a prompt is reused for many sentences, which the protocol does not measure. It also depends on random re-timing working (EXP-031's `RND`) |
| A single teacher evaluation on the student's re-noised draft (about 6 passes) | Student-then-teacher refinement, a known family. A teacher that starts from the student's structure adds little per-seed variety |
| Earlier handover for TOH (Steps 1–2) | A parameter choice, not a mechanism |

## Testing it

**Option A (in EXP-031):** add a `BTH3` arm by a dated amendment and re-freeze before any synthesis.
- **Cost:** one blind-teacher run per case plus 3 student passes and scoring, roughly 25–30 minutes more on a T4.
- **Comparisons:**
  - against TOH3: the same payload, but a teacher that hears the prompt;
  - against `RND`: no teacher;
  - against the previous methods.
- **Measures:** variety (including aligned CVf0), quality, and the reverberation and babble conditions, where the exact independence should show.

**Option B:** wait for EXP-031's `RND` result. If the teacher's timing is not needed, BTH is not worth building.

## Checks run

The three web searches above. No code was written and no model was run.
