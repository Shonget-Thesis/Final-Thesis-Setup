# Preliminary tests: list and verdicts

Every preliminary test the team has run. All are **development evidence** (data already looked at), not the final held-out test. Details for each one are in `research/experiments/`.

**Verdicts:** ✅ supported · ❌ not supported · ⚠️ mixed / unclear · 📦 data or tooling only

## In the paper (Appendix D)

| # | Test | Question | Result | Verdict |
|---|---|---|---|---|
| EXP-001 | Fixed switching times (D.1) | Does one early step suit everyone? | Failures 25–33%, mostly noisy voices; per-sentence choice in hindsight = 14.85 of 21 calls | ✅ The problem is real |
| CPU (legacy) | Student restart level (D.2) | Can the student start later (j = 2)? | Departs from the reference more at every step (48 cases) | ❌ Keep j = 1 |
| EXP-009 | Our policy, cross-validated (D.3) | Does SNR + drift save calls safely? | 1 failure in 200; 10% fewer calls | ✅ Works on dev data |
| EXP-010 | Real timing on the T4 (D.3) | Do saved calls mean saved time? | Clean voices 3.29 s → 2.41 s (−27%); checks add no time | ✅ Yes |
| Dev diagnostic | Signal check (D.4) | Do SNR and drift predict failures? | SNR splits clean vs noisy perfectly; drift AUC ≈ 0.47 | ✅ SNR · ❌ drift |

## In the repo only (not in the paper)

| # | Test | Question | Result | Verdict |
|---|---|---|---|---|
| EXP-000 | Timing replay | Do the old tables add up? | Reproduces the October 5 audit's arithmetic | 📦 Bookkeeping |
| EXP-002 | T4 grid, 500 speakers | Base data for everything else | 4,000 scored outputs (Steps 4/5/6/8) | 📦 Data |
| EXP-003 | Earlier steps 2 and 3 | Can we hand over even earlier? | Step 3 as safe as Step 4 on clean voices (3/100); Step 2 worse (7/100) | ⚠️ Step 3 possible; not adopted |
| EXP-004 | Signal selection | Best gate? Best stop signal? Best calibration? | SNR is the best gate; drift no better than fixed Step 4 (AUC 0.53); LTT is the right calibration | ✅ SNR, LTT · ❌ drift |
| EXP-005 | Richer drift features (67) | Can a better drift rescue noisy voices? | Best AUC 0.69, barely above chance | ❌ Weak |
| EXP-006 | Echo and telephone voices | Does drift help where SNR is blind? | Drift changed no decision on any failure | ❌ Not supported |
| EXP-007 | Three scheduler revisions | Can a smarter drift rule beat SNR + fixed step? | No (McNemar p = 1.00) | ❌ |
| EXP-008 | Novel algorithms | Anything cheaper than SNR-only at the same risk? | No; student probe predicts better (AUC 0.71) but saves nothing | ❌ |
| Audit (Oct 25) | LTT recompute on T4 | What does real calibration pick? | λ̂ = 1.20, so our policy = SNR-only (4/200 failures, 17.04 calls) | ⚠️ Drift adds nothing here |
| EXP-011 | Follow-up pilot | Does the 20 dB gate catch all noise? | Misses 18–22 dB noise; low drift helps only as a noise warning | ⚠️ Gate needs fixing |
| EXP-012 | Proposed gates | Can a new gate beat a tuned SNR gate? | No; tuned SNR gate (26 dB) best: 7/360 failures | ❌ Raise the SNR gate instead |
| EXP-013 | Drift-extrapolated handover | Can we predict the Step 8 result at Step 4? | Clean failures 2 → 25 | ❌ Failed |
| EXP-014 | Three handover tweaks | Fix noisy voices at no cost? | None beat the tuned SNR gate; cDEH best (better voice match on noisy) | ❌ (cDEH promising) |
| EXP-015 | Noise-aware methods (A100) | Fix noisy voices for a little cost? | **TAS** 12/360 failures; **SWC** 20/360 at 13 calls; none beat the tuned SNR gate | ⚠️ Two candidates; needs T4 re-check |

## Overall verdict

- ✅ **The problem is real**, and per-sentence handover saves calls and time.
- ✅ **SNR is the signal that works.** Raising the gate (~26 dB) is the simplest fix.
- ❌ **Drift as a stopping signal is not supported** by any test so far.
- ⚠️ **TAS and SWC** might help noisy voices but need confirming on the same hardware (Appendix E, hidden).
- ⏳ **Not done yet:** a real held-out test with new speakers.
