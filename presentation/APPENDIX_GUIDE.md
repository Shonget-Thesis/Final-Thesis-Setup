# Appendix and results guide: what each experiment is and how to explain it

For the defense. Every experiment uses the same layout so you can answer any of them the same way:

> **What it is → What we did → What we found → What it means → One line to remember**

Numbers are from the manuscript on branch `rhenelAudit` (PDF page numbers in brackets are the printed page numbers). Chapter 4 (Results) is still a placeholder; **all current results live in Appendix D**. Everything there is **development evidence**: the speakers had already been looked at, so it is not yet the held-out test.

---

## 0. One-page cheat sheet

| Appendix | Name | What it shows in one line |
|---|---|---|
| **A** | Sample calculations | Every formula worked by hand with tiny numbers |
| **B** | Sample simulation | Our calibration keeps its 5% promise; the naive way does not |
| **C** | Prototype | What our method changes: only *where* the handover happens |
| **D.1** | Fixed switching times | One early step for everyone saves calls but breaks many sentences (**the problem**) |
| **D.2** | Student restart level | Keep the student's default start; only the handover time varies |
| **D.3** | Stabilization policy (SNR + drift) | Our policy: far fewer quality drops than any fixed early step, still saves calls and time |
| **D.4** | Development diagnostic | SNR cleanly splits clean vs noisy; drift alone is not proven yet |
| **D.5** | Plan for drift | How the full study will test whether drift is worth keeping |
| **D.6** | Summary table | What each experiment found and what it changed in the design |
| *(E, hidden)* | TAS and SWC | Backup ideas for noisy voices; not printed in the paper |

**The whole story in four lines:**
1. **Problem (D.1):** one fixed early handover saves up to 8 of 21 calls but breaks up to a third of sentences.
2. **Opportunity (D.1, Table D.2):** each sentence has its own best step; in hindsight, 14.85 calls instead of 21.
3. **Our answer (D.3):** SNR + drift picks the step per sentence: 1 quality drop in 200, 18.88 calls, 27% faster on clean voices.
4. **Proof it's safe (A, B):** the calibration math works, and in simulation it keeps its promise every time.

---

## 1. Words you will hear (technical → plain)

| Term | Plain meaning | Memory hook |
|---|---|---|
| **Teacher** (F5-TTS) | The slow, careful model that starts the speech | Senior chef |
| **Student** (DMOSpeech 2) | The fast model that finishes it in 3 steps | Junior chef |
| **Handover / switching time** | The step where the teacher passes the work to the student | Passing the dish |
| **Step 4, 5, 6** | The *candidate* handover points we allow | Early exits |
| **Step 8 / reference / s = 0.25** | The default handover of the original DMOSpeech 2 | The safe default |
| **Euler step** | One small cleaning move of the teacher | One brush stroke |
| **Forward pass (NFE)** | One full run of the neural network; our unit of cost | One "call" |
| **Cost 2(k+1)+3** | Calls for handing over at step k: 13, 15, 17, 21 | Teacher ×2, student +3 |
| **Endpoint estimate x̂₁** | The teacher's free guess of the finished speech | GPS arrival estimate |
| **Drift / stabilization score d** | How much that guess changed since the last step | Did the GPS estimate move? |
| **λ (lambda)** | Most drift allowed before we hand over | "Settled enough" line |
| **SNR, Ŝ** | How clean the voice sample is, in dB | Cleanliness score |
| **θ (theta), SNR gate** | Minimum cleanliness to try early: 20 dB | The door |
| **Added failure** | Early-handover output clearly worse than step 8 (more word errors or less voice match) | A quality drop |
| **WER** | Word error rate: % of words a speech recognizer gets wrong | Readability |
| **SIM** | Speaker similarity, 0 to 1: does it sound like the person? | Voice match |
| **UTMOS** | Predicted listener score, 1 to 5 | Naturalness guess |
| **RTF** | Real-time factor: seconds to make 1 second of audio (lower is faster) | Speed |
| **α = 0.05** | Our tolerance: at most 5% quality drops | The 5% limit |
| **δ = 0.10** | Allowed chance the promise is wrong: 10% (so 90% confidence) | 90% sure |
| **Learn-Then-Test (LTT)** | Approve a threshold only if the data prove it safe | Quality control |
| **Oracle** | Picks each sentence's best step *after* seeing results; impossible in practice | Answer key |
| **Out of fold / cross-validation** | Pick the threshold on some speakers, score it on others | Practice test vs real test |
| **Development evidence** | Data we already looked at; not a final test | Rehearsal |

**Statistics, in one line each:**

| Test | What it answers | How to read it |
|---|---|---|
| **Clopper–Pearson (Upper 95%)** | "Worst-case failure rate, given this sample size" | Under 5% = safe even in the worst case |
| **Exact McNemar** | "Do two methods break *different* sentences, more than chance?" | p < 0.05 = real difference |
| **Wilcoxon signed-rank + Holm** | "Did a measure (calls, WER, SIM, UTMOS) really change vs step 8?" | p < 0.05 = real change; Holm corrects for many comparisons |
| **Bootstrap CI** | "Range the true time saving likely lies in" | Narrow range = reliable |
| **TOST** | "Is quality *equal*, not just 'no difference found'?" | Not run yet; needs margins fixed before the held-out test |

---

## 2. Appendix A: Sample calculations

**What it is:** every formula from Chapter 3, worked by hand with small numbers. These are **illustrations, not results**.

**What it walks through (one toy sentence, λ = 0.50, θ = 20 dB):**

| Section | Formula | Toy numbers | Plain meaning |
|---|---|---|---|
| A.1 | Time grid + cost | Step 4 → t = 0.076, 13 calls; step 8 → 21 calls | Where each step sits and what it costs |
| A.2 | SNR | Loud frames 0.0112 vs quiet 0.00001 → **30.5 dB** | Clean, so the gate opens |
| A.3 | Endpoint estimate | 0.8 + 0.8819 × 1.5 = **2.12** | The free guess of the final value |
| A.4 | Drift + decision | Step 4: d = **0.675** > 0.50 → continue; step 5: d = **0.154** → stop | Hand over once the guess settles |
| A.5 | Failure check | WER 23.08% vs limit 16.38% → **failure** | How a quality drop is counted |
| A.6 | Calibration | 0 failures → p = 0.0059 ✅; 1 → 0.0371 ✅; 3 → 0.2578 ❌ stop | Approve thresholds in order, stop at the first unsafe one |

**Key fact from A.6:** with zero failures, certifying 5% at 90% confidence needs **at least 45 independent inputs** (0.95⁴⁵ = 0.099). That's why the earlier 36-input test could not certify anything.

### Table A.1 made easy: "the price list"

**In one sentence:** *where each handover step sits, and how many model calls it costs.*

| Step k | t_k | Role | Teacher passes | Cost C | Saving vs 21 |
|---|---|---|---|---|---|
| 0–3 | 0 to 0.0431 | Euler steps only | – | – | – |
| **4** | 0.0761 | candidate (s = 0.07) | 10 | **13** | 8 (38.1%) |
| **5** | 0.1181 | candidate (s = 0.10) | 12 | **15** | 6 (28.6%) |
| **6** | 0.1685 | candidate (s = 0.15) | 14 | **17** | 4 (19.0%) |
| 7 | 0.2270 | not a candidate | – | – | – |
| **8** | 0.2929 | reference (s = 0.25) | 18 | **21** | 0 |

**How to read each column:**
- **Step k:** which teacher step.
- **t_k:** how far along the cleaning is (0 = noise, 1 = finished). Step 8 is only at 0.29, because the grid packs steps early, where the big picture is decided.
- **Role:** steps 0–3 are too early to stop; 4, 5, 6 are allowed early exits; 7 isn't used; 8 is the default.
- **Teacher passes:** 2 × (k + 1), because guidance runs the teacher twice per step.
- **Cost C:** teacher passes + 3 for the student.
- **Saving vs 21:** calls avoided compared with step 8.

**Read one row out loud:** *"Stop at step 4: the teacher is 7.6% of the way, uses 10 calls, plus 3 for the student, so 13 total. That's 8 fewer than the default 21."*

**Takeaway:** each step earlier saves **2 calls**. The earliest exit saves 8 of 21.

**Say:** "Table A.1 is the price list. Each teacher step costs two calls and the student adds three, so handing over at step 4 costs 13, and the default step 8 costs 21."

**Likely Q: "Why is step 7 not a candidate?"** "Our candidates are the switching values s = 0.07, 0.10 and 0.15 from the team's earlier experiments, plus the reference 0.25. The released code stops at the first grid point above s, which turns them into steps 4, 5, 6 and 8. Step 7 would need a value like s = 0.20, which wasn't in that set. It's a design choice; the paper doesn't test step 7."

### Table A.2 made easy: "the safety approval line"

**In one sentence:** *how the calibration picks λ: test thresholds one by one, strictest first, and stop at the first one that can't prove it's safe.*

| Order | λ | Failures in 100 | p-value | ≤ 0.10? | Action |
|---|---|---|---|---|---|
| 1 | 0.40 | 0 | 0.0059 | yes | certify, continue |
| 2 | 0.50 | 1 | 0.0371 | yes | certify, continue |
| 3 | **0.60** | **3** | **0.2578** | **no** | **stop** |
| 4 | 0.70 | 6 | not tested | – | – |

**How to read each column:**
- **Order:** the testing order, strict to loose (small λ = waits longer = safer).
- **λ:** the "settled enough" threshold being tested.
- **Failures in 100:** quality drops this λ caused on 100 calibration sentences (**made-up numbers**).
- **p-value:** *"If this λ were actually unsafe, how likely is it to see this few failures by luck?"* Small = strong proof it's safe.
- **≤ 0.10?** The pass mark. 0.10 comes from δ = 0.10, which gives 90% confidence.
- **Action:** pass → approve and try the next one; no → **stop**.

**Read the table out loud:** *"0.40 had no failures, so it's proven safe. 0.50 had one, still proven safe. 0.60 had three, and that's not enough proof, so we stop. 0.70 is never tested. The answer is 0.50."*

**Takeaway:** we pick **the loosest λ that's still proven safe**: the most saving without breaking the 5% promise.

**Why stop instead of skipping ahead?** Each extra test is another chance for an unsafe λ to pass by luck. Stopping at the first "no" keeps the total risk of a wrong approval at 10%. *Like climbing a ladder: stop at the first rung that doesn't hold.*

**Say:** "Table A.2 is the safety check. We test thresholds from strictest to loosest and approve each only if the data prove it's under 5%. At 0.60 three failures weren't enough proof, so we stop and use 0.50."

**Likely Q: "Where does 0.2578 come from?"** "It's the chance of seeing 3 or fewer failures in 100 if the true rate were 5%: 0.0059 + 0.0312 + 0.0812 + 0.1396."

**Remember:** *"Appendix A shows every formula works on paper, step by step."*

**If asked "where does 13 and 21 come from?":** "2 × (steps + 1) for the teacher, because guidance runs it twice, plus 3 for the student. Step 4 is 2 × 5 + 3 = 13; step 8 is 2 × 9 + 3 = 21."

---

## 3. Appendix B: Sample simulation

**What it is:** a test of our calibration on **fake data with known answers**, to check it keeps its promise. No speech is generated.

**What we did:**
- Simulated sentences: ¼ noisy (SNR ≈ 16 dB), ¾ clean (≈ 33 dB), with drift scores and failure chances.
- We know the **true** risk of every threshold (400,000 simulated sentences). The largest truly safe threshold is **λ = 0.85**.
- Ran calibration **1,000 times** for each sample size n = 50, 100, 200, 400.
- Compared **Learn-Then-Test** vs a **naive rule** (pick the most aggressive threshold that *looks* under 5%).

### Table B.1 made easy: "the report card"

**In one sentence:** *we ran the calibration 1,000 times on fake data where we know the right answer, and counted how often it chose a truly safe λ.*

| n | Method | P(R ≤ α) | Mean R | Mean passes | Reference only | Median λ̂ |
|---|---|---|---|---|---|---|
| 50 | **Learn-Then-Test** | **0.967** | 0.013 | 18.20 | 10.1% | 0.35 |
| | Naive | 0.427 | 0.052 | 15.54 | 0.0% | 0.95 |
| 100 | **Learn-Then-Test** | **0.988** | 0.015 | 17.70 | 2.2% | 0.40 |
| | Naive | 0.399 | 0.054 | 15.44 | 0.0% | 0.95 |
| 200 | **Learn-Then-Test** | **0.967** | 0.027 | 16.46 | 0.0% | 0.55 |
| | Naive | 0.420 | 0.052 | 15.44 | 0.0% | 0.90 |
| 400 | **Learn-Then-Test** | **0.920** | 0.035 | 15.92 | 0.0% | 0.70 |
| | Naive | 0.469 | 0.051 | 15.43 | 0.0% | 0.90 |

**How to read each column:**
- **n:** how many calibration sentences each run had.
- **Method:** **Learn-Then-Test** (ours, proof-based) vs **Naive** (pick the loosest λ that *looks* under 5%).
- **P(R ≤ α):** **the main column.** Out of 1,000 runs, how often the chosen λ was **truly** safe (real risk ≤ 5%). Must be **at least 0.90**.
- **Mean R:** the average true failure rate of the chosen λ. Under 0.05 = safe on average.
- **Mean passes:** average calls per sentence with the chosen λ (21 = default). Lower = more saving.
- **Reference only:** how often nothing could be proven safe, so it fell back to step 8.
- **Median λ̂:** the typical λ chosen. The truly safe limit is **0.85** (Figure B.1); above that is unsafe.

**Read one pair out loud (n = 100):** *"With 100 sentences, Learn-Then-Test picked a truly safe λ 98.8% of the time. The naive way only 39.9%: it usually picked 0.95, which is past the safe limit of 0.85."*

**Takeaways:**
1. **Learn-Then-Test always passes** (0.92–0.99, all above 0.90). **Naive always fails** (0.40–0.47).
2. **Naive looks fine on average** (Mean R ≈ 0.05) but **goes over the limit more than half the time**. Averages hide the danger.
3. **Price of safety:** with little data, Learn-Then-Test is cautious (18.20 calls at n = 50, falls back 10% of the time).
4. **More data → more saving:** at n = 400 it uses 15.92 calls, close to the best safe value of 15.4.

**Say:** "Table B.1 is the report card for our calibration. On fake data where we know the truth, Learn-Then-Test picked a safe threshold at least 92% of the time, above the 90% required. Just picking the best-looking threshold was safe less than half the time. The more calibration data, the closer our method gets to the best possible saving."

**Likely Q: "Why is the naive one wrong if its average is 5%?"** "It picks whatever looks best on one sample, which is often just lucky. On average it's near 5%, but more than half the time the real risk is above the limit."

**Remember:** *"Picking the best-looking threshold is a coin flip; Learn-Then-Test keeps its promise every time."*

**If asked "it's fake data, so what?":** "It shows our procedure is implemented correctly and behaves as the theory says. Whether real speech meets the theory's assumptions is what the held-out test checks."

---

## 4. Appendix C: Prototype

**What it is:** a picture of what our method changes (Figure C.1). It isn't an app; it's the same DMOSpeech 2 sampler with a **stopping controller** added.

**What it shows:**
- **Baseline:** every input pays **21 calls**, whether it needs them or not.
- **Ours:** easy input → hand over at step 4 (**saves 8 calls**); moderate → step 6 (**saves 4**); noisy/hard → step 8 (**same as baseline**).

**The three promises (bottom of the figure):**
1. **Decide:** stop when the guess stops changing, using values already computed (**0 extra calls**).
2. **Guarantee:** λ is calibrated so quality drops stay ≤ 5% with ≥ 90% confidence.
3. **Fall back:** never later than the baseline; if nothing is certified, it *is* the baseline.

**Inputs → outputs:** voice sample + its transcript + text + λ + θ → audio + chosen step + calls used + SNR + drift trace.

**Remember:** *"Same models, same audio pipeline; only the handover point moves, and never later than today."*

---

## 5. Appendix D: Preliminary experiments (the results)

### D.1 Fixed switching times: **the problem** [p. 61–62]

**What it is:** what happens if **every** sentence hands over at the **same** early step.

**What we did:** 500 speakers × clean and 15 dB noisy = **1,000 cases**, Tesla T4 GPU, step 4, 5, 6 and 8 for every case, scored against step 8.

**What we found, Table D.1 (one step for everyone):**

| Step | Calls | Quality drops: clean | Noisy | All |
|---|---|---|---|---|
| 4 | 13 | 12 (2.4%) | **322 (64.4%)** | **334 (33.4%)** |
| 5 | 15 | 9 (1.8%) | 269 (53.8%) | 278 (27.8%) |
| 6 | 17 | 5 (1.0%) | 247 (49.4%) | 252 (25.2%) |
| 8 | 21 | 0 | 0 | 0 |

**Table D.2 (best step per sentence, in hindsight):**

| Prompts | Step 4 | 5 | 6 | 8 | Avg calls |
|---|---|---|---|---|---|
| Clean | 488 | 9 | 3 | 0 | 13.06 |
| Noisy | 178 | 91 | 52 | 179 | 16.64 |
| All | **666** | 100 | 55 | 179 | **14.85** |

**What it means:**
- **D.1:** earlier = cheaper but more quality drops, almost all on **noisy** voices. No single step works for everyone.
- **D.2:** each sentence has its own best step. Two-thirds were fine at step 4; 179 noisy ones needed step 8. Choosing per sentence could cut **21 → 14.85 calls** with no quality drops, but that needs hindsight.
- **Why step 8 shows 0:** step 8 is the yardstick; compared with itself it can't be worse.

**Remember:** *"D.1 is the problem: one size doesn't fit all. D.2 is the goal: the right step per sentence."*

**If asked "why not just use step 4?":** "It saves 8 of 21 calls but a third of sentences get worse, two-thirds of the noisy ones. That's the problem we solve."

---

### D.2 Student restart level [p. 62]

**What it is:** could the student start one step later (j = 2) to save one more call?

**What we found:** j = 2 departed from the reference more often at every handover: 26 vs 11, 13 vs 8, 9 vs 8 and 14 vs 0 (of 48). Small CPU test, lighter models.

**What it means:** keep the student's default (j = 1). The **handover time is the only thing our method changes**.

**Remember:** *"We tried moving the student; it hurt, so only the teacher's stop moves."*

---

### D.3 Stabilization policy (SNR + drift): **our method** [p. 63–66]

**What it is:** our actual policy, tested fairly against every method Chapter 3 plans to compare.

**What we did:**
- 100 development speakers × clean/noisy = **200 cases** (T4).
- **4-fold cross-validation:** choose λ on 75 speakers, score on the other 25, rotate. Chosen λ: 0.35, 0.35, 0.35, 0.50.
- Ran every compared method on the same cases, plus a real **timing run** on the T4.

**Table D.3: quality drops and cost**

| Method | What it does | Drops clean / noisy | Worst case (Upper 95%) | Calls | McNemar vs ours |
|---|---|---|---|---|---|
| Fixed step 4 / 5 / 6 | Same step for everyone | 3–0 / **63–49** | 31.6–40.0% | 13–17 | p < 10⁻¹³ |
| Step 8 (reference) | Today's default | 0 / 0 | 1.8% | 21.00 | – |
| SNR-only | Clean → step 4, noisy → step 8 | 3 / 0 | 4.3% | 17.00 | 0.50 |
| SNR + fixed step 5 | Clean → step 5 | 2 / 0 | 3.6% | 18.00 | 1.00 |
| SNR + fixed step 6 | Clean → step 6 | 0 / 0 | 1.8% | 19.00 | 1.00 |
| Stabilization only | Drift with no SNR gate | 0 / 0 | 1.8% | 21.00 | 1.00 |
| **Full policy (ours)** | **SNR gate + drift** | **1 / 0** | **2.8%** | **18.88** | – |
| Cost-matched random | Our steps, shuffled at random | 0.75 avg / 0 | – | 18.88 | – |
| Oracle | Hindsight answer key | 0 / 0 | 1.8% | 14.70 | – |

**Table D.4: quality (average).** On clean voices every method is practically the same: WER about 2.0–2.3%, SIM 0.934, UTMOS about 3.86–3.88. Ours: **WER 2.13%, SIM 0.934, UTMOS 3.88**.

**Table D.5: real time on the T4**

| | Clean: calls | Clean: time | Clean: RTF | Noisy: time |
|---|---|---|---|---|
| Step 8 | 21.00 | 3.29 s | 0.73 | 3.37 s |
| SNR-only | 13.08 | 2.06 s | 0.45 | 3.37 s |
| **Full policy (λ = 0.50)** | **15.30** | **2.40 s** | **0.53** | **3.37 s** |

**What it means (positive, all true):**
1. **Far fewer quality drops than any fixed early step:** 1 vs 50–66 of 200 (McNemar p < 10⁻¹³).
2. **Within 5% even at the worst case:** upper bound **2.8%**.
3. **Fewer calls than step 8, significantly** (Wilcoxon-Holm p < 10⁻¹⁵), with **no significant change in WER, SIM or UTMOS** (p = 1.00).
4. **Real speed:** clean voices **3.29 s → 2.40 s (−27%)**; overall **0.44 s saved per sentence** (95% CI 0.42–0.47 s). Noisy voices cost the same as before, so the checks add no measurable time.
5. **Adapts per sentence:** chose steps 4 / 5 / 6 for 6 / 28 / 52 clean sentences.

**The trade-off (say it calmly):**
- **SNR-only is faster** (17.00 calls), **ours is more careful** (1 drop vs 3; drift prevented 2 of SNR-only's 3 and added none). The gap is about **2 calls per sentence**.
- With so few drops, ours and the SNR-gated controls can't yet be told apart (p ≥ 0.50). The held-out test decides.
- **Drift alone saves nothing**, because noise makes the guess look settled too early. That's **why the SNR gate comes first**: the two signals cover each other.
- Noisy voices currently save nothing; the gate sends them to the safe default.

**Remember:** *"Almost no quality drops, fewer calls, 27% faster on clean voices. SNR-only is faster, ours is more careful; the full test decides."*

**If asked "is drift better than SNR alone?":** "It's more careful: it prevented two quality drops SNR allowed, for about two more calls per sentence. Whether that trade is worth it is what our held-out comparison measures."

**If asked "is the quality the same?":** "We found no significant difference in WER, SIM or UTMOS. Claiming they're *equal* needs an equivalence test, which we run on the held-out data with margins fixed in advance."

**Don't say:** "ours is better than SNR-only," "drift is proven," "same quality," "38% faster."

---

### D.4 Development-split signal diagnostic [p. 66]

**What it is:** a first look at how well each signal separates good and bad handovers.

**What we found:**
- **SNR separates clean from noisy completely:** clean 23.3–50.0 dB, noisy 16.8–18.2 dB, so a 20 dB gate sent every noisy voice to step 8.
- Fixed early handovers: **49–63%** drops on noisy voices, only **2–4%** on clean.
- **Drift on clean voices did not separate drops from passes** (AUC ≈ 0.47, ≈ chance), but there were only 2–3 drops to learn from.
- At λ = 0.50 the policy waited past step 4 on 78 of 100 clean voices; 75–77 of those would have been fine at step 4.

**What it means:** SNR is strong. Drift is **a hypothesis still to be tested**, not an assumption; too few clean drops to judge it.

**Remember:** *"SNR is clear-cut; drift is the open question, and we say so."*

---

### D.5 Plan for the drift signal [p. 66–67]

**What it is:** how the full study will decide whether drift is worth keeping. Four steps, each fixed before seeing its data:
1. **Discrimination:** does drift separate drops from passes? (ROC/AUC, correlation, value beyond SNR)
2. **Controls:** beat SNR + fixed later step and cost-matched random? (McNemar)
3. **Harder conditions:** echo, telephone audio and others that SNR may miss
4. **Richer free signals:** per-frame, per-band, guidance gap (development only)

**Decision rule:** if drift adds nothing, recommend **SNR-only** and report drift as a tested negative result. **The contribution doesn't depend on which signal wins.**

**Remember:** *"Every outcome gives a result: drift wins → keep it; drift doesn't → SNR-only, reported honestly."*

---

### D.6 Summary table [p. 67]

| Experiment | Finding | What it changed |
|---|---|---|
| Fixed switching times | Earlier = more drops (25–33%); best step varies; hindsight 14.85 of 21 | Per-sentence switching (Objective 1) |
| Student restart level | j = 2 departs more from the reference | Restart fixed at j = 1 |
| Stabilization policy | 10% fewer calls, 1 drop in 200, 27% less time on clean voices; no certificate yet | Decision signals (Obj. 2) + LTT calibration (Obj. 3) |

---

## 6. Appendix E (hidden: not printed; answer only if asked)

Removed from the paper on 2026-10-27; kept in `paper/chapters/appendix-e-hidden.tex` and the EXP-015 record. **Backups for noisy voices** if SNR + drift doesn't save enough there.

- **TAS (Trajectory Anomaly Score):** flags guesses that settle *suspiciously early* (noise makes the model go quiet). Plain: "a GPS that stopped updating because it lost signal."
- **SWC (SNR-Wiener Conditioning):** cleans the noise out of the voice sample before the models copy it. Plain: "noise-cancelling the voice sample."
- On 360 cases with moderate noise, SNR-only had 86 drops; **TAS 12 at 18.20 calls**, **SWC (heavy noise) 11 at 16.98 calls**. Development only; SWC ran on a different GPU (A100).

**If asked:** "We explored backups for noisy voices, a stricter warning check and prompt noise cleaning. They're promising on development data but not part of this proposal."

---

## 7. Fixed values and why we chose them

The panel may ask "why this number?" for any of these. **Strength** tells you how solid the answer is:
- **Code:** taken from the released DMOSpeech 2 code
- **Data:** our own measurements support it
- **Design:** our choice, with a stated reason but no outside source; say "fixed in advance" and give the reason

### Model and sampler settings (used everywhere)

| Value | Where | Why (as the paper says) | Strength |
|---|---|---|---|
| **16-step sway grid, s = −1** | Ch. 3, A.1 | The released code's default; the F5-TTS sway schedule | Code |
| **γ = 2 (guidance)** | Ch. 3 | Released code default (`cfg_strength`) | Code |
| **Reference: step 8 (s = 0.25)** | everywhere | Matches the DMOSpeech 2 paper's switching time; later steps can't save anything | Code + paper |
| **Candidates: steps 4, 5, 6 (s = 0.07, 0.10, 0.15)** | Ch. 3, A.1 | Step 4 is the code's default switch; every candidate is earlier than step 8, so it can save | Code |
| **Student start j = 1 (3 steps)** | D.2 | Released default; j = 2 departed from the reference more often (D.2) | Code + data |
| **Cost 2(k+1)+3** | A.1 | Counted from the code, and confirmed by a counter in the pilot (3,000 of 3,000 timing runs matched) | Code + data |

### Decision rule

| Value | Where | Why | Strength |
|---|---|---|---|
| **θ = 20 dB (SNR gate)** | D.3, A.2, B | Ch. 3: set on the development split as the lowest SNR above which early handovers rarely fail. D.4: clean voices measured 23.3–50 dB and noisy ones 16.8–18.2 dB, so 20 dB sits in the gap | Data |
| **SNR = loudest 50% vs quietest 10% of frames, 25 ms frames** | Ch. 3 Def. 3.3, A.2 | Loud frames ≈ speech, quiet frames ≈ background; simple, no extra model. The paper calls it a proxy, checked against the known added noise | Design + data |
| **λ grid 0.20–1.20, steps of 0.05 (21 values)** | Ch. 3, B | Ordered strict → loose, same as Appendix B; reaches well past 0.60, where the earlier search ran out, so the data and not the grid limits λ | Design |
| **λ grid 0.20–0.60 in D.3** | D.3 | Same range as the earlier CPU run; widening it to 1.20 gave the same choices | Data |
| **λ = 0.50 in the timing run (Table D.5)** | D.3 | Taken from the earlier CPU labels, not calibrated on the T4 (stated as a caveat) | Weak: say it's a development setting |

### Quality-drop rule (Definition 3.1)

| Value | Why (as the paper says) | Strength |
|---|---|---|
| **WER ceiling c = 12.5%** | Keeps a short sentence from counting as a drop because of one misheard word (1 error in 8 words = 12.5%) | Design |
| **WER margin 1 point** | Allows for small run-to-run differences in the speech recognizer | Design |
| **SIM margin 0.02** | Allows for small run-to-run differences in the speaker model | Design |

**Honest note:** no outside source is cited for these three. The pilot showed that running step 8 twice with different seeds already "fails" for 4 of 10 speakers, so the margins sit near the natural noise level. **Say:** "They were fixed before calibration to absorb measurement noise, and anything above the ceiling is checked by listening."

### Calibration and statistics

| Value | Why | Strength |
|---|---|---|
| **α = 0.05 (5% tolerance)** | Fixed before calibration; matches Appendices A and B | Design: "a common, strict tolerance we fixed in advance" |
| **δ = 0.10 (90% confidence)** | Fixed before calibration; with δ = 0.05, 50 inputs can't certify anything even with zero failures (Ch. 3 sample-size table) | Design + math |
| **p ≤ 0.10 to certify** | Equals δ | Math |
| **At least 45 inputs** | 0.95⁴⁵ = 0.099 ≤ 0.10, but 0.95⁴⁴ = 0.105 is not | Math |
| **Calibration split: 200 speakers** | Well above 45, so a threshold can be certified even with a few failures | Design + math |
| **95% intervals (Clopper–Pearson, bootstrap)** | Standard reporting level | Convention |

### Experiment setup

| Value | Where | Why | Strength |
|---|---|---|---|
| **15 dB white noise** | D.1, D.3 | Same as the earlier experiments; tests whether the signals react to degraded voices; checked within 15.0 ± 0.1 dB | Design |
| **500 speakers, split 100 / 200 / 200** | Ch. 3, D.1 | Speaker-disjoint development / calibration / test; one utterance per speaker so observations are independent | Design |
| **4 folds (75 train / 25 test)** | D.3 | Score λ on speakers it wasn't chosen on | Design |
| **Cost-matched random: 2,000 shuffles** | D.3 | Enough repetitions for a stable average; same cost as our policy, so only the step choice differs | Design |
| **Tesla T4, FP32, fixed seeds** | everywhere | Free-tier-class Colab GPU; same seeds so only the handover changes | Design |

### Toy and simulation numbers (Appendices A and B)

| Value | Why | What to say |
|---|---|---|
| **A.2–A.6 toy numbers** (frame energies, 0.8 and 1.5, d = 0.675 / 0.154, WER 23.08%, failure counts 0/1/3/6) | Made up to show **each branch** of the rule: gate open, keep going, stop, pass, drop, certify, stop | "Illustrations to check the formulas by hand, not results." |
| **Simulation: noisy ~16 dB (¼), clean ~33 dB (¾)** | Chosen to look like the real data (D.4: noisy 16.8–18.2, clean 23.3–50) | "Fake data shaped like our real data." |
| **Simulation failure model, difficulty, 400,000 / 1,000 runs** | Gives a known true risk (safe up to λ = 0.85) to check against; 1,000 repeats per size for stable percentages | "We needed a case where we know the right answer." |

**The one-line answer for any "why this number?":**
"It's either the released code's default, supported by our own data, or fixed in advance with a stated reason, so it couldn't be tuned to make results look better."

## 8. Answer any experiment with this template

1. **"This experiment tests…"** (one line from the cheat sheet)
2. **"We used…"** (how many speakers and cases, which GPU)
3. **"The key number is…"** (one or two numbers, no more)
4. **"That means…"** (plain meaning)
5. **"The limitation is…"** (development data / small sample / the full test decides)

**Example (D.1):** "This tests one fixed handover for everyone. We used 1,000 cases on a T4. At step 4, a third of sentences got worse, mostly noisy ones. That means no single step fits everyone, which is our problem. It's development data; the held-out test uses new speakers."

## 9. Five numbers to memorize

| Number | What it is |
|---|---|
| **13 / 21** | Calls at step 4 vs step 8 |
| **33%** | Sentences that get worse if everyone hands over at step 4 |
| **14.85** | Calls with the perfect per-sentence choice (hindsight) |
| **1 of 200, 2.8%** | Our quality drops, and the worst-case bound (under 5%) |
| **3.29 → 2.40 s** | Our real time on clean voices (−27%) |
