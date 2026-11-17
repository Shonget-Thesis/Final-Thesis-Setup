"""A replacement for Chapter 3's drift: the invariant residual (screening only, on this computer's CPU; not evidence).

Chapter 3 (Definition 3.4, the endpoint-stabilization score):

    d_i  = || x_hat(t_i) - x_hat(t_{i-1}) ||_M / ( || x_hat(t_i) ||_M + eps )

asks whether the teacher's estimate has stopped changing. The problems measured since: noise makes the estimate stop
changing early (EXP-012), so d says "settled" exactly when the teacher copies the prompt's noise; d is the teacher's
local truncation error up to a factor (2026-11-16 adaptive-teacher log), a signal already used by adaptive solvers;
and its stopping job is now done by the commit point (t = 0.2929, EXP-033).

The changed term. The previous estimate is replaced by the estimate's condition-safe version, Q(x_hat, s0): in each
mel band the student draft's values placed in the teacher's time order (the quantile handover, the maximal invariant
of per-band increasing maps; 2026-11-15 theory note):

    r    = || x_hat - Q(x_hat, s0) ||_M / ( || x_hat ||_M + eps )          (the invariant residual)

r is the share of the teacher's estimate that lies outside its condition-safe part: per band, the teacher's levels
against the student's own at the same ranks (the per-band Wasserstein-2 distance between their value distributions).
- What the teacher copies and the mode-seeking student drops (spread-out background) raises r.
- What both copy (reverberation, band-limiting) cancels and does not raise r (the spread principle).
- The melody and timing live in the time order, which Q keeps, so they do not count.

The algorithm with the SNR (Chapter 3's structure, one threshold per signal): the teacher runs to t = 0.2929 on the
8-interval sway grid (10 passes); the student's one-pass draft s0 (1 pass); if S_hat >= theta and r <= lambda, hand
over x_hat (CPH8); otherwise hand over Q(x_hat, s0) with the draft's prompt frames (CP-QH); the student restarts at
0.2929 and 0.6173 (2 passes). 13 passes for every prompt.

Cases: the 12 shortest clean EXP-033 quality sentences (the 6 shortest = calibration half, the 7th-12th = test half)
x 7 prompt conditions (clean, noisy15, gauss25, pink25, hp25 [high-pass hiss], reverb09, telephone) x 2 seeds
(input_seed + 1000 + s). Per case: Definition 3.3 SNR, the spread-weighted SNR, Chapter 3's drift at the commit point
(estimates at steps 3 and 4 of the 8-interval grid), r, the earlier roughness share, and UTMOS of both payloads.
"QH needed" = UTMOS(CP-QH) - UTMOS(CPH8) > 0.30.

Pass rules for r, fixed before the run:
R1. On all cases, AUC(r) for "QH needed" >= 0.80 and >= drift's AUC + 0.10 (drift in its better direction).
R2. Among cases the SNR gate passes (Definition 3.3 >= 20 dB), AUC(r) >= 0.75.
R3. lambda is chosen on the calibration half to maximize the policy's mean UTMOS with at most 15% of clean cases
    routed to CP-QH; on the test half, SNR + r then gives mean UTMOS >= SNR-only (CP-QH below 20 dB, CPH8 above)
    + 0.05, with at most 15% of the test half's clean cases routed to CP-QH.
All three must hold. Drift gets the same calibration (both directions) for the SNR + drift comparison.

Usage: .venv/Scripts/python -I code/analysis/invariant_residual_screen.py
"""
import json
import statistics as st
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for p in ("code/colab", "code/prototype", "code/analysis"):
    sys.path.insert(0, str(ROOT / p))
import innovation_snr_screen as ins  # noqa: E402
import noise_color_screen as ncs  # noqa: E402
import noise_theory_screen as nts  # noqa: E402
import roughness_band_probe as rbp  # noqa: E402

OUT = ROOT / "research/experiments/EXP-033-cheaper-teacher-path/offline/invariant_residual_screen.json"
N_SENT, N_CAL, SEEDS, SEED_BASE = 12, 6, (0, 1), 1000
CONDS = ("clean", "noisy15", "gauss25", "pink25", "hp25", "reverb09", "telephone")
THETA = 20.0
NEEDED = 0.30
MAX_CLEAN_QH = 0.15
AUC_ALL, AUC_GAIN, AUC_OPEN, POLICY_GAIN = 0.80, 0.10, 0.75, 0.05


def invariant_residual(teacher_gen, student_gen):
    """r = ||x_hat - Q(x_hat, s0)|| / ||x_hat|| over (frames, bands) arrays of the generated frames."""
    import run_method_comparison as mc
    t = np.asarray(teacher_gen, dtype=np.float64)
    q = mc.quantile_handover(np.asarray(student_gen, dtype=np.float64), t)
    return float(np.linalg.norm(t - q) / (np.linalg.norm(t) + 1e-12))


def stabilization(curr, prev):
    """Chapter 3's d_i over the generated frames."""
    c, p = np.asarray(curr, dtype=np.float64), np.asarray(prev, dtype=np.float64)
    return float(np.linalg.norm(c - p) / (np.linalg.norm(c) + 1e-12))


def auc(pos, neg):
    """P(score of a positive > score of a negative), ties counted half (Mann-Whitney)."""
    if not pos or not neg:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def make_prompt(clean, sr, cond, seed, rf):
    if cond == "pink25":
        return ncs.add_at(clean, ncs.shaped_noise(len(clean), sr, seed, ncs.SHAPES["pink25"]), ncs.SNR_DB)
    if cond == "hp25":
        return ncs.add_at(clean, ncs.shaped_noise(len(clean), sr, seed, nts.SHAPES["hp25"]), ncs.SNR_DB)
    return rf.degrade(clean, sr, cond, seed)


def policy(cases, flag):
    """Mean UTMOS when flagged cases get CP-QH and the rest CPH8, and the share of clean cases flagged."""
    out = [c["utmos_CPQH"] if flag(c) else c["utmos_CPH8"] for c in cases]
    clean = [c for c in cases if c["condition"] == "clean"]
    return st.mean(out), (sum(bool(flag(c)) for c in clean) / len(clean) if clean else 0.0)


def calibrate(cases, key, direction):
    """The threshold on `key` (flag when key > lam, or key < lam for direction -1) that maximizes the SNR + key
    policy's mean UTMOS with at most MAX_CLEAN_QH of clean cases flagged; ties go to fewer flags."""
    vals = sorted({c[key] for c in cases})
    cands = ([float("inf")] + vals[::-1]) if direction > 0 else ([float("-inf")] + vals)
    best = None
    for lam in cands:
        flag = (lambda c, l=lam: c["snr"] < THETA or (c[key] > l if direction > 0 else c[key] < l))
        mean, clean_share = policy(cases, flag)
        if clean_share <= MAX_CLEAN_QH and (best is None or mean > best[1] + 1e-12):
            best = (lam, mean, clean_share)
    return best


def evaluate(cases):
    cal = [c for c in cases if c["half"] == "cal"]
    test = [c for c in cases if c["half"] == "test"]
    pos = [c for c in cases if c["needed"]]
    neg = [c for c in cases if not c["needed"]]
    open_pos = [c for c in pos if c["snr"] >= THETA]
    open_neg = [c for c in neg if c["snr"] >= THETA]
    a = {k: auc([c[k] for c in pos], [c[k] for c in neg]) for k in ("r", "drift", "rough_share")}
    a["drift_best"] = max(a["drift"], 1.0 - a["drift"]) if a["drift"] is not None else None
    a["snr_low"] = auc([-c["snr"] for c in pos], [-c["snr"] for c in neg])
    a["spread_snr_low"] = auc([-c["snr_sw"] for c in pos], [-c["snr_sw"] for c in neg])
    a_open = {k: auc([c[k] for c in open_pos], [c[k] for c in open_neg]) for k in ("r", "drift", "rough_share")}
    lam_r = calibrate(cal, "r", +1)
    snr_only = policy(test, lambda c: c["snr"] < THETA)
    snr_r = policy(test, lambda c: c["snr"] < THETA or c["r"] > lam_r[0])
    d_cal_plus = calibrate(cal, "drift", +1)
    d_cal_minus = calibrate(cal, "drift", -1)
    if d_cal_plus[1] >= d_cal_minus[1]:
        snr_d = policy(test, lambda c: c["snr"] < THETA or c["drift"] > d_cal_plus[0])
        drift_rule = {"direction": "flag high drift", "lambda": d_cal_plus[0]}
    else:
        snr_d = policy(test, lambda c: c["snr"] < THETA or c["drift"] < d_cal_minus[0])
        drift_rule = {"direction": "flag low drift", "lambda": d_cal_minus[0]}
    r_only = policy(test, lambda c: c["r"] > lam_r[0])
    rules = {"R1_detection": a["r"] is not None and a["r"] >= AUC_ALL and a["r"] >= a["drift_best"] + AUC_GAIN,
             "R2_added_to_snr": a_open["r"] is not None and a_open["r"] >= AUC_OPEN,
             "R3_policy_out_of_sample": snr_r[0] >= snr_only[0] + POLICY_GAIN and snr_r[1] <= MAX_CLEAN_QH}
    by_cond = {}
    for cond in CONDS:
        cs = [c for c in cases if c["condition"] == cond]
        by_cond[cond] = {"n": len(cs), "needed": sum(c["needed"] for c in cs),
                         "snr_median": float(np.median([c["snr"] for c in cs])),
                         "snr_sw_median": float(np.median([c["snr_sw"] for c in cs])),
                         "r_median": float(np.median([c["r"] for c in cs])),
                         "drift_median": float(np.median([c["drift"] for c in cs])),
                         "utmos_CPH8": st.mean(c["utmos_CPH8"] for c in cs),
                         "utmos_CPQH": st.mean(c["utmos_CPQH"] for c in cs),
                         "flagged_by_r_at_lambda": sum(c["r"] > lam_r[0] for c in cs)}
    return {"rules": {k: bool(v) for k, v in rules.items()}, "all_pass": bool(all(rules.values())),
            "n_cases": len(cases), "n_needed": len(pos), "auc_all": a, "auc_gate_open": a_open,
            "n_gate_open": {"needed": len(open_pos), "not_needed": len(open_neg)},
            "lambda_r": {"lambda": lam_r[0], "cal_mean_utmos": lam_r[1], "cal_clean_share": lam_r[2]},
            "drift_rule": drift_rule,
            "test_half": {"snr_only": snr_only, "snr_plus_r": snr_r, "snr_plus_drift": snr_d, "r_only": r_only,
                          "all_CPH8": policy(test, lambda c: False), "all_CPQH": policy(test, lambda c: True),
                          "oracle": (st.mean(max(c["utmos_CPH8"], c["utmos_CPQH"]) for c in test), None)},
            "by_condition": by_cond}


def main():
    import local_candidate_screen as ls
    import run_followup_tests as rf
    import run_method_comparison as mc
    import soundfile as sf
    r_ = ls.Runner()
    torch = r_.torch
    utmos = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True).eval()
    items = {it["utt_id"]: it for it in r_.en.development_examples()}
    t0 = time.time()
    cases = []
    total = N_SENT * len(CONDS) * len(SEEDS)
    with torch.inference_mode():
        for idx, row in enumerate(ncs.pick_cases(N_SENT)):
            it = items[row["utt_id"]]
            clean, sr = sf.read(str(r_.en.DATA_DIR / it["prompt_wav_rel"]), dtype="float32", always_2d=False)
            clean = (clean.mean(axis=1) if clean.ndim > 1 else clean).astype(np.float32)
            for cond in CONDS:
                prompt = make_prompt(clean, sr, cond, int(it["input_seed"]), rf)
                snr_sw = float(ins.snr_variants(prompt, sr)[1])
                for s in SEEDS:
                    ctx = r_.e.prepare(prompt, sr, it["prompt_text"], it["gt_text"], int(it["input_seed"]) + SEED_BASE + s)
                    plen = ctx["prompt_len"]
                    s0 = r_.student(ctx, ctx["step_cond"], [0.0])
                    ends = r_.teacher(ctx, ls.sway(8), 4, want=(3,))
                    t3, t4 = ends[3], ends[4]
                    g_t, g_s = t4[0, plen:, :].numpy(), s0[0, plen:, :].numpy()
                    outs = {"CPH8": r_.student(ctx, t4, ls.TAUS[1:]),
                            "CPQH": r_.student(ctx, mc.quantile_payload(s0, t4, plen), ls.TAUS[1:])}
                    rec = {"utt_id": row["utt_id"], "half": "cal" if idx < N_CAL else "test", "condition": cond,
                           "seed": s, "snr": float(ctx["prompt_snr"]), "snr_sw": snr_sw,
                           "drift": stabilization(g_t, t3[0, plen:, :].numpy()),
                           "r": invariant_residual(g_t, g_s),
                           "rough_share": float(np.mean(rbp.band_roughness(g_t) > rbp.band_roughness(g_s)))}
                    for a, x in outs.items():
                        audio = r_.e._decode(ctx, x).astype(np.float32)
                        rec[f"utmos_{a}"] = float(utmos(torch.from_numpy(audio).unsqueeze(0), 24000))
                    rec["needed"] = bool(rec["utmos_CPQH"] - rec["utmos_CPH8"] > NEEDED)
                    cases.append(rec)
                    print(f"[{len(cases)}/{total}] {row['utt_id'][:22]} {cond:9s} s{s} SNR {rec['snr']:5.1f} "
                          f"r {rec['r']:.3f} drift {rec['drift']:.3f} | UTMOS CPH8 {rec['utmos_CPH8']:.2f} "
                          f"CPQH {rec['utmos_CPQH']:.2f}{' NEEDED' if rec['needed'] else ''} ({time.time() - t0:.0f}s)",
                          flush=True)
    summary = evaluate(cases)
    OUT.write_text(json.dumps({"kind": "local CPU screen (screening only, not evidence)", "summary": summary,
                               "cases": cases, "seconds": round(time.time() - t0)}, indent=2, default=str) + "\n",
                   encoding="utf-8", newline="\n")
    print(json.dumps(summary, indent=2, default=str))
    print(f"[OK] wrote {OUT}")


if __name__ == "__main__":
    main()
