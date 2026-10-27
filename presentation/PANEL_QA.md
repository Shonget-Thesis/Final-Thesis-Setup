# Panel Q&A prep: proposal defense

Likely questions from the panel (the Dean and a PhD in computing), with answers the team can give.

Each question has:

- **Say this:** a plain-language answer you can give in about 30 seconds.
- **If they push:** the technical version, with the paper section to point to.
- **Don't say:** claims our evidence does not support.

Numbers come from the manuscript (`paper/`, branch `rhenelAudit`) unless a line names a report in `research/`. All preliminary results are **development evidence**: the speakers they use were already looked at, so none of it is a held-out test yet. Say so whenever you quote one.

---

## Analogies to reuse

| Concept | Plain-language picture |
|---|---|
| Flow matching | Fog slowly clearing into a picture. The model learns which way each bit of fog should move at every moment. |
| Endpoint estimate x̂₁ | A GPS arrival estimate: from where you are and your current speed, it predicts where you will end up, for free. |
| Teacher and student | A senior chef starts the dish (slow, careful). A junior chef trained by them finishes it (fast). |
| Switching time | The moment the senior chef hands the dish over. |
| Forward pass | One "call" to the neural network. This is our unit of cost, like counting fuel instead of minutes. |
| Drift (stabilization score dᵢ) | How much the GPS arrival estimate moved since the last check. Small means it has settled. |
| Learn-Then-Test | Quality control before shipping: test a batch, and only approve a setting if the evidence shows at most 5 bad in 100, with 90% confidence. |
| Fixed-sequence testing | Climbing a ladder rung by rung and stopping at the first rung that doesn't hold. |
| Premature stabilization | A GPS that loses signal stops updating. It looks "settled", but only because it stopped receiving information. |
| SWC | Noise-cancelling the voice sample before the model copies the voice. |

---

# Part A: Dean-style questions (purpose, value, feasibility, ethics)

### A1. In one or two sentences, what is your study?

**Say this:** Some AI voice systems use a slow, high-quality model to start the speech and a fast model to finish it. Right now the handover happens at the same moment for every sentence. We make that moment adapt to each sentence, so easy sentences finish sooner, and we use a statistical test to promise in advance how often this makes the speech worse.

**If they push:** A training-free, per-utterance teacher-to-student switching rule for DMOSpeech 2. It decides from signals the sampler already computes and is calibrated with Learn-Then-Test so that the added-failure rate is ≤ 5% with ≥ 90% confidence (Ch. I objectives; Ch. III).

### A2. Why does this matter? Who benefits?

**Say this:** Every network call costs time and electricity. If many sentences don't need the full slow model, skipping it saves both. Speech devices for people who communicate through a device are one place where faster speech matters. That is our motivation, but we only measure speech quality and computing cost. We don't test the system with users.

**If they push:** Significance section (Ch. I). We cite AAC research (Therrien 2025; Muharib 2024) for motivation only. Scope explicitly excludes clinical outcomes, usability, mobile devices and listening tests (Ch. I Scope).

**Don't say:** "This helps autistic learners" or "this improves AAC." We do not test that.

### A3. Isn't this just DMOSpeech 2? What is new?

**Say this:** DMOSpeech 2 built the slow-start, fast-finish system, but it hands over at one fixed point for everyone. What's new is choosing the handover per sentence, at no extra cost, *and* attaching a statistical guarantee on how often that hurts quality. Among the papers we reviewed, no one has combined those in speech synthesis.

**If they push:** Gap table, Ch. II (Table 2.1). CALM calibrates early exits for language models, and an adaptive ASR corrector uses LTT for speech *recognition*. None of them selects a teacher-to-student handover in synthesis with a calibrated bound. We say "within the sources examined", not "first ever."

### A4. Why not just use the fast student model alone, or buy a faster GPU?

**Say this:** The fast model alone gives speech that sounds more alike from one output to the next. The slow model's first steps add variety and structure. A faster GPU speeds up everything equally, but it doesn't remove calls that weren't needed. Our method is about not doing unnecessary work, and it would help on any hardware.

**If they push:** DMOSpeech 2 reports "mode shrinkage" in the student. Teacher guidance restores diversity (Ch. II §2.3). Our savings are counted in forward passes, which don't depend on the hardware.

### A5. How much faster does it actually make things?

**Say this:** In network calls, the earliest handover uses 13 calls instead of 21, about 38% fewer for those sentences. On real development data, a simple version of our rule averaged 18.88 calls instead of 21, about 10% fewer. Actual clock time saves less than that, because some parts, like turning the spectrogram into audio, don't shrink.

**If they push:** Cost identity C(s) = 2(k + 1) + 3 (Prop. 3.1): 13, 15, 17 and 21 passes. Appendix D cross-validation gave 18.88 passes. DMOSpeech 2 itself shows the gap between pass savings and runtime savings: a 2.1× pass ratio gave about 1.8× runtime. Wall-clock time and RTF on the T4 are planned outputs (Ch. III, Objective 4). We have no runtime result for the policy yet.

**Don't say:** "38% faster." That is passes, for the inputs that stop at step 4, not runtime.

### A6. How do you know the speech doesn't get worse?

**Say this:** For every sentence we make two versions, one with the early handover and one with the normal handover, using the same random seed. We compare them. If the early one has clearly more word errors or sounds less like the speaker, we count it as a failure. Our goal is to keep those failures at 5% or less.

**If they push:** Definition 3.1: fail if WER > max(12.5%, ref + 1 pt) or SIM < ref − 0.02. WER uses Whisper-large-v3 and SIM uses the WavLM-Base+ speaker verifier. UTMOS is reported but not used in the rule, because predicted naturalness is unreliable for single utterances (Huang et al., 2022).

### A7. Ethics: you're cloning voices.

**Say this:** We only use public research recordings under their licenses. We report numbers, not cloned voices of identifiable people. We don't involve users. Any listening check stays inside the team.

**If they push:** Ch. III Ethical Considerations. There is an open item to check institutional requirements for any listening test.

### A8. Is this feasible within your time and resources?

**Say this:** Yes. Everything runs on a free-tier-class Tesla T4 GPU through Google Colab. The data is public. We have already run the full pipeline on 1,000 test cases, and our pre-experiment checks pass.

**If they push:**
- The EXP-011 pilot passed all checks: deterministic outputs, a forced fallback reproduces the reference exactly, and the pass counter matches the formula for steps 4, 5, 6 and 8.
- Appendix D covers 1,000 T4 instances.
- Appendix B simulates the calibration step.

### A9. What if your method doesn't work?

**Say this:** That is still a result, and we will report it. Our design checks for exactly that: whether any setting can be certified, and whether we beat a simpler rule. If a simple noise check does just as well, we'll recommend the simple check and say so.

**If they push:** Ch. III Research Workflow lists the negative outcomes per objective. Appendix D.6 has a plan to test the stabilization score, with the SNR-only rule as the fallback recommendation. If SNR + drift falls short in the full run, Appendix E has two backup options already tried on development data (TAS and SWC). They are backups, not part of what we propose.

### A10. Why only English? Why Seed-TTS?

**Say this:** It's the same English test set the original DMOSpeech 2 paper used, so our numbers can be compared with theirs. We don't claim the results carry over to other languages.

### A11. Will it run on a phone?

**Say this:** We don't test that. All timing is on one cloud GPU. Phones are future work.

---

# Part B: Technical questions (PhD in computing)

### B1. Explain flow matching and the endpoint estimate without equations.

**Say this:** The model starts from random noise and moves it, step by step, toward a speech spectrogram, like fog clearing. At every step it knows its current direction. If you extend that direction all the way to the end, you get a guess of the final result. Our method reads that guess at no extra cost.

**If they push:** x̂₁(t) = x_t + (1 − t)·v(x_t, t). It needs no extra network evaluation because v is already computed for the Euler step (Prop. 3.2, Zero overhead). It is exact for straight probability paths and approximate otherwise.

### B2. Why does a teacher step cost two passes but a student step one?

**Say this:** The teacher uses a technique where it predicts twice, once *with* the text and voice prompt and once *without*, and then exaggerates the difference to follow the prompt more closely. That's two network calls. The student was trained to imitate the final result directly, so it needs only one.

**If they push:** Classifier-free guidance: v = v_c + γ(v_c − v_u) with γ = 2 (Eq. 3.2). The student is DMD-distilled from the guided teacher.

### B3. "Step 8" but "21 passes": how many steps are there?

**Say this:** The teacher takes 8 steps, then does 1 more evaluation to get the final guess it hands over. That's 9 evaluations at 2 calls each, so 18. The student adds 3 steps at 1 call each. Total: 21.

**If they push:** C(s) = 2(k(s) + 1) + 3. Step 4 → 13, step 5 → 15, step 6 → 17, step 8 → 21 (Table 3.6, Appendix A.1).

### B4. The DMOSpeech 2 paper says 14 teacher steps at t = 0.25. You say 8. Who is right?

**Say this:** Both are, on different time grids. The paper's description fits a 32-step grid. The released code uses a 16-step grid and stops at the first point after 0.25. We use the released code, because that's what actually runs.

**If they push:** Remark 3.1. On the 32-interval grid, t₁₄ ≈ 0.227 is the last point ≤ 0.25. On the 16-interval grid the code stops at t₈ ≈ 0.293. The two grids share their even points. The authors' evaluation script is not released, so this reading is consistent with the code but not confirmed.

### B5. Explain Learn-Then-Test simply. Why not just pick the threshold that looks best?

**Say this:** If you try many settings and pick the one that looks best, you will often pick one that only looked good by luck. Our simulation showed this: the "pick the best-looking one" approach broke the 5% promise in 58 of 100 repeats. Learn-Then-Test treats each setting like a suspect that must be proven safe. It tests them in a fixed order, from cautious to aggressive, and stops at the first one that fails. That keeps the overall chance of a wrong approval at 10%.

**If they push:** H_g: R(λ_g) > α, with binomial p-value p_g = P(Bin(n, α) ≤ failures). Fixed-sequence testing (Bauer 1991) controls the family-wise error at δ without monotonicity (Prop. 3.6). Appendix B: LTT met P(R ≤ α) ≥ 0.90 at every n (0.920–0.988), and the naive rule did not (0.399–0.469).

### B6. Why the exact binomial test instead of Hoeffding?

**Say this:** With rare failures, the exact test needs far fewer examples. With zero failures, certifying 5% at 90% confidence needs 45 speakers with the exact test and about 461 with Hoeffding's inequality.

**If they push:** (1 − α)ⁿ ≤ δ ⇒ n ≥ ln δ / ln(1 − α) = 44.9 (Eq. 2.6). That is also why the earlier 36-instance cross-validation could not certify anything: 0.95³⁶ = 0.158 > 0.10.

### B7. Is the guarantee for every sentence?

**Say this:** No. It's an average promise: across many new sentences like the calibration ones, at most 5% get worse. Any single sentence can still fail. And if new sentences are very different, for example a different kind of noise, the promise may not hold.

**If they push:** The guarantee is marginal and assumes the calibration and test inputs are i.i.d. (or exchangeable) draws from 𝒟 (Prop. 3.6, Remark). Robustness to distribution shift is tested by calibrating on clean prompts and testing on noisy ones (Objective 4).

### B8. Are your samples independent? You have clean and noisy versions of the same speaker.

**Say this:** That's why we count by speaker, not by recording. One utterance per speaker, and a speaker counts as a failure if either the clean or the noisy version fails. That gives one honest observation per person.

**If they push:** Ch. III Data and Splits: 500 speakers in speaker-disjoint splits of 100 development, 200 calibration and 200 test.

### B9. Does the stabilization ("drift") signal actually work?

**Say this:** Honestly, not as a stopping signal so far. On clean prompts it didn't separate good handovers from bad ones. On noisy prompts it does something surprising: noise makes it look *more* settled, because the model is getting less information, not because it's ready. So we treat it as a hypothesis to test, not an assumption. Used the other way round, as a warning sign when things are unusually still, it does help. That's the Trajectory Anomaly Score, one of our backup options in Appendix E.

**If they push:**
- Appendix D.5: AUC ≈ 0.47 for d₄ on clean prompts, though based on only 2–3 failures.
- Appendix E: median d₄ is 0.61 clean against 0.17 at 15 dB. The guidance gap falls from 0.236 to 0.116.
- EXP-012 report (development, 360 instances): SNR + drift as written in Def. 3.4 certified no threshold and fell back to the reference (0 failures at 21.00 passes). A low-drift veto gave 9/360 at 18.38. SNR + TAS gave 12/360 at 18.20, against SNR-only's 86/360 at 14.40.

**Don't say:** "Drift tells us when the speech is ready."

### B10. Then why not just use the noise level (SNR) alone?

**Say this:** That might turn out to be the best answer, and our design allows for it. SNR-only is one of the rules we compare against, with the same calibration. If it wins, we report that. The main contribution, adaptive handover with a guaranteed failure limit, holds whichever signal wins.

**If they push:** In the October 25 audit (offline, development), LTT on the 200 calibration speakers certified λ̂ = 1.2, the top of the grid, so the full rule reduced to SNR-only: 4/200 speaker failures at 17.04 passes on the test split. A separately calibrated SNR gate (θ̂ = 18.5 dB) gave 4/200 at 17.02. Source: `research/audits/2026-10-25/FINDINGS.md` and the EXP-011 pilot report.

### B11. Your SNR estimator: how reliable is it?

**Say this:** It's a simple loud-versus-quiet comparison. It separates our clean and 15 dB recordings perfectly. But moderate noise around 18–22 dB gets estimated above the 20 dB gate and slips through. Noise types like babble or echo may fool it too. That is a known weakness, and it's why we test those conditions.

**If they push:**
- Def. 3.3: loudest half of frames over quietest 10%, clipped to 0–50 dB.
- Appendix D.5: clean prompts estimate 23.3–50.0 dB and noisy ones 16.8–18.2 dB.
- Appendix E: SNR-only failed 86/360 because the 18–22 dB prompts pass the gate.

### B11a. What is Appendix E, then? Are TAS and SWC your method?

**Say this:** No. Our proposed method is SNR + drift, from Chapter 3. Appendix E holds two backups in case it doesn't do well in the full run. We prepared them because in development, SNR + drift was too cautious: it never certified a threshold and sent everything to step 8. And the SNR gate saves nothing on noisy voice samples. TAS is a stricter warning check, and SWC cleans the noise out of the voice sample so noisy inputs can hand over early too.

**If they push:** Appendix E. Development only, 360 instances: SNR + TAS 12/360 failures at 18.20 passes; SNR + SWC 20/360 at 13.00, or 11/360 at 16.98 with heavy-noise routing; SNR-only 86/360 at 14.40. Either backup is used only if SNR + drift falls short and the backup passes testing on new speakers with settings fixed in advance.

**Don't say:** "Our method is TAS" or "SWC is our contribution."

### B12. TAS was designed after you saw the data. Isn't that overfitting?

**Say this:** Yes, and we say so in the paper. The idea came from looking at these same cases, so those numbers are not proof. Before we adopt it, it has to be tested, with all settings fixed in advance, on speakers we haven't used.

**If they push:** Appendix E.4, Limitations. TAS is fitted on clean prompts only and uses no failure labels. It was not significantly better than a calibrated SNR gate at about 26 dB (7 failures at 18.71 passes; McNemar p = 0.23).

### B13. SWC is just spectral subtraction from 1979. What's novel?

**Say this:** The formula isn't new, and we cite it. What's new is where we apply it: cleaning the voice sample *inside* the generator before the models read it, with the strength set by the measured noise level, as a cheaper substitute for running the slow model longer.

**If they push:** Gain G = min{1, max{0.1, 1 − β(Ŝ)·N_b/m}}, with β ramping from 1 at ≤ 20 dB to 0 at ≥ 30 dB. On 300 noisy instances, step-4 failures fell from 126 to 21 and SIM rose by 0.126.

### B14. Your SWC results came from a different GPU. Does that matter?

**Say this:** Yes. The SWC runs used an A100 and were compared against references made on a T4. The random draws differ between the two GPUs, which adds disagreements and probably makes SWC look *worse* than it is. We'll redo it on one GPU before drawing conclusions.

**If they push:** Appendix E.4. The teacher reproduced exactly across GPUs, but the student's random draws did not. TAS and SNR-only used T4 outputs only and are unaffected.

### B15. How much of a "failure" is just randomness?

**Say this:** Good question. Even running the *same* setting twice with different random seeds can trip our failure rule. In a small pilot this happened for 4 of 10 speakers. That's why every comparison uses identical seeds, so the only thing that differs is the handover time. Measuring that randomness properly is part of our follow-up tests.

**If they push:** In the EXP-011 pilot report, step 8 with seed B "failed" against seed A for 4/10 speakers. This is a small sample, but it shows the margins (1 WER point, 0.02 SIM) are near the seed-noise level. The full seed-floor run is designed in EXP-011.

**Don't say:** "Every failure is caused by the early handover."

### B16. Is your test set really unseen?

**Say this:** Not yet, and we're upfront about it. Our splits were looked at during earlier analysis, so current numbers count as development evidence. A clean held-out test needs new data, such as LibriSpeech, or reusing the remaining speakers with that history stated. That decision comes before we lock the protocol.

**If they push:** See `docs/progress/STATUS.md` and the Appendix D.2 note. No Seed-TTS test-en speaker is unseen: the 166 outside the splits were used in an earlier run.

### B17. What is the "oracle", and why show it?

**Say this:** It's a cheating rule that looks at the answers: for each sentence, it picks the cheapest handover that didn't fail. It can't be used in practice, but it shows the best possible saving, which is 14.85 calls instead of 21. Every real method is judged against that ceiling.

**If they push:** Prop. 3.5 (oracle lower bound). In Appendix D, 666 of 1,000 instances could stop at step 4, 100 at step 5 and 55 at step 6, and 179 needed the reference, all of them noisy.

### B18. Why is the student's restart point fixed?

**Say this:** We tested moving it. It saved one call but caused more failures at every handover time, so we keep the released default. That way, the only thing we change is when the teacher stops.

**If they push:** Appendix D.3: restarting at j = 2 instead of j = 1 gave 26 vs 11 failures at step 4, 13 vs 8 at step 5, 9 vs 8 at step 6 and 14 vs 0 at step 8, out of 48 (CPU, lighter evaluators).

### B19. Which statistical tests do you use for comparisons?

**Say this:** Paired tests, because every method runs on the same sentences with the same seeds. To claim "no worse", we use a proper equivalence test, not just "no significant difference."

**If they push:**
- Wilcoxon signed-rank with Holm correction for WER, SIM, UTMOS and passes.
- Exact McNemar for paired failures.
- TOST for equivalence.
- Clopper–Pearson intervals for failure rates.
- Bootstrap CIs for the savings Δ and the oracle share η (Ch. III, Statistical Analysis).

### B20. Why not train a small model to predict the best switch?

**Say this:** Our scope is training-free: we change only how the released models are run. A learned predictor needs labeled data and becomes another model to validate. Any rule, learned or not, could still be wrapped in the same calibration. We keep it simple and auditable.

### B21. If the final rule ends up equal to SNR-only, what's left of your contribution?

**Say this:** The contribution is the framework: picking the handover per sentence, at zero extra cost, with a proven safe fallback and a statistical limit on added failures, tested properly on new speakers. Which signal drives it is an empirical question we answer honestly. "A simple noise check, with a guarantee, is enough" is a useful finding in itself.

**If they push:** Appendix D.6: "The contribution of the study … does not depend on which signal performs best."

---

# Part C: Lines to hold

- **Say "development evidence"** whenever you quote a preliminary number.
- **Say "within the sources we examined"**, never "first ever."
- **Separate passes from runtime.** Our savings are in network calls. Runtime savings are smaller and not yet measured for the policy.
- **The guarantee is an average**, not per utterance, and assumes new inputs resemble the calibration data.
- **We don't measure users or clinical outcomes.** AAC is motivation only.
- **Our method is SNR + drift.** TAS and SWC (Appendix E) are backups in case it falls short in the full run, not part of the proposal. Both need testing on new speakers with settings fixed in advance.
- **The manuscript is a working draft.** Don't call it adviser-approved.

# Part D: 30-second closing

> Teacher-guided TTS spends the same effort on every sentence. Our data show many sentences don't need it, but the right amount differs per sentence, and stopping early can hurt quality. We propose a rule that decides per sentence from signals the model already computes, never costs more than today, and is calibrated so that at most 5% of sentences get worse, with 90% confidence. We will test it on speakers it has never seen, and we will report the result either way.
