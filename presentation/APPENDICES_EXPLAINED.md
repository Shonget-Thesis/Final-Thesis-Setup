# Appendices A–D: presenter notes

Short version for presenting. Numbers are from the manuscript. Real results (Appendix D) are **development evidence**, not the final held-out test.

**Key terms**
- **Teacher / student:** slow careful model / fast finishing model.
- **Handover (Step 4, 5, 6, or 8):** when the teacher passes the work to the student. **Step 8 = today's default.**
- **Forward pass:** one run of the model. Our cost unit. Step 8 costs **21**.
- **SNR:** how clean the voice sample is. **Gate = 20 dB:** below it, no early handover.
- **Drift (d):** how much the teacher's guess still changes. Small = settled.
- **λ (lambda):** how settled is "settled enough".
- **Added failure:** early speech clearly worse than Step 8 speech.
- **α = 5%, δ = 10%:** at most 5% failures, 90% sure.
- **Learn-Then-Test (LTT):** our way of picking λ *and proving* it safe.
- **WER / SIM / UTMOS:** clarity / voice match / naturalness.

---

## Appendix A: Sample Calculations

**What it is:** every formula done by hand with tiny made-up numbers. Not results.

**Shows:**
- Cost: Step 4 / 5 / 6 / 8 = **13 / 15 / 17 / 21** calls, saving **38% / 29% / 19% / 0**.
- SNR and drift use values already computed: **0 extra calls**.
- Example: drift 0.675 at Step 4 (too high), 0.154 at Step 5 (settled) → hand over at Step 5.
- Failure = WER or voice match drops past a set limit.
- **Need at least 45 calibration sentences** to prove 5% / 90%; with 36, impossible.

> **Say:** "Appendix A shows every formula works on paper, and that we need at least 45 calibration sentences."

---

## Appendix B: Sample Simulation

**What it is:** testing LTT on **fake data where we know the right answer**. Like a driving simulator.

**What we did:** fake sentences (¼ noisy, ¾ clean). Found the true risk of every λ using 400,000 of them. Then let LTT and a **naive rule** (pick whatever *looks* under 5%) choose λ, **1,000 times** per sample size (50–400).

**Found:**
- **Truly safe λ goes up to 0.85** (15.4 calls instead of 21).
- **LTT kept its promise every time:** safe in **92–99%** of runs (needs 90%).
- **Naive broke it:** safe only **40–47%** of the time. It picks lucky-looking values ("winner's curse").
- **More data = more saving:** 18.2 calls with 50 sentences → **15.9** with 400.

> **Say:** "In a simulator where we know the answer, our method kept its 5% promise every time; just picking the best-looking value failed more than half the time."

---

## Appendix C: Prototype

**What it is:** the same DMOSpeech 2 models plus a **stopping controller**. Not an app. No results; it's the design.

**Shows (Figure C.1):**
- Today: every sentence costs **21**.
- Ours: easy → Step 4 (**saves 8**), moderate → Step 6 (**saves 4**), hard/noisy → Step 8 (**same as today**).

**Three promises:** **Decide** (0 extra calls) · **Guarantee** (≤ 5%, 90% sure) · **Fall back** (never later than today).

> **Say:** "Same models; only the handover point moves, per sentence, and never later than today."

---

## Appendix D: Preliminary Experiments (real results)

### D.1 Fixed switching times: **the problem**
**Did:** 1,000 cases (500 speakers, clean + noisy), every sentence at the same step.

| Step | Calls | Failures (clean) | Failures (noisy) |
|---|---|---|---|
| 4 | 13 | 2.4% | **64.4%** |
| 5 | 15 | 1.8% | 53.8% |
| 6 | 17 | 1.0% | 49.4% |
| 8 | 21 | 0 | 0 |

**Found:** earlier = more failures, **mostly noisy voices**. But each sentence has its own best step: picked in hindsight, only **14.85 of 21** calls (−29%).

> **Say:** "One step for everyone doesn't work. Choosing per sentence could save 29%."

### D.2 Student restart level
**Found:** starting the student later (j = 2) moved further from the reference at every step. **Kept the default (j = 1)**, so only the handover changes.

### D.3 Our policy on real speech
**Did:** 200 cases, 4-fold **cross-validation** (choose λ on 75 speakers, test on the other 25).

**Found:**
- **1 failure in 200**, vs **50–66** for fixed early steps (McNemar p < 10⁻¹³).
- Worst-case failure rate **2.8%**, under 5%.
- **10% fewer calls** overall, **20% on clean voices**.
- **27% faster on clean voices** (3.29 s → 2.40 s).
- **No significant change** in WER, SIM, UTMOS.
- SNR gate is key: drift alone saved nothing.

**Trade-off:** SNR-only saves more (19%) with 3 failures; ours is more careful (1 failure). Too few failures to tell them apart yet.

**Limit:** not yet a safety certificate. LTT on new speakers comes next.

> **Say:** "Almost no failures, fewer calls, 27% faster on clean voices, same quality as far as we can tell. The held-out test decides SNR-only vs ours."

### D.4 Signal diagnostic
- **SNR separates clean vs noisy perfectly** (clean 23–50 dB, noisy 17–18 dB).
- **Drift is unproven:** AUC ≈ 0.47 (about chance), but only 2–3 failures to judge.

### D.5 Plan for drift
Four pre-set tests: does drift separate failures? Beat simple controls? Help on echo / phone audio? Better free variants?
**If drift adds nothing → recommend SNR-only.** The contribution holds either way.

---

## One-line summary

> "Fixed early handovers break many sentences (D.1). Our policy picks per sentence: 1 failure in 200, 27% faster on clean voices (D.3). In simulation, our calibration keeps its 5% promise (A, B). Next: the held-out test."
