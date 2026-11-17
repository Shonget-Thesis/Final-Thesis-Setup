"""Local screen of a restart-masked adaptive teacher (screening only, on this computer's CPU; not evidence).

EXP-033 showed that a 3-step teacher to the commit point t_c = 0.2929 (CPH6, 10 passes) is fine or better for most
clean sentences but breaks badly for some (UTMOS -1.0 to -2.5 against W8R2), and that the breakage tracks how far the
coarse path's estimate is from the 16-interval Step 8 estimate. This screen asks whether a run-time error indicator,
computed from passes the teacher makes anyway, can find those sentences, and whether correcting only the flagged steps
repairs them for fewer passes than CPH8 (12).

The indicator (derived, not fitted). For one Euler step from t_a to t_b with step h, the endpoint estimates satisfy
x^_b - x^_a = (1 - t_b)(v_b - v_a) exactly, so the step's local error is e = (h/2)(v_b - v_a) = h / (2(1 - t_b)) *
(x^_b - x^_a). The student sees the handed-over estimate through its restart phi = tau x^ + (1 - tau) eps, so an error
counts in units of the restart noise: r = tau / (1 - tau) * rms(e) over the generated frames, at tau = t_c.
The correction: one Heun (trapezoid) correction of the flagged step, x_b = x_a + (h/2)(v_a + v_b), with one extra
guided evaluation at the corrected point (2 passes).

What runs, per sentence (6-interval sway grid to t_c, i.e. three coarse steps):
- every correction pattern (none, any subset, all three steps), sharing prefixes (15 guided evaluations);
- AB2: the known second-order multistep solver on the same grid, no extra evaluation (10 passes): the competitor;
- references: W8R2 (16-interval Euler to Step 8, 20 passes) and CPH8 (8-interval, 4 steps, 12 passes).
Each handed-over estimate goes to the student's levels 0.2929 and 0.6173 with the released per-level seeds; outputs
are scored with UTMOS (utmos22_strong) and the estimate's distance from the local 16-interval Step 8 estimate.

Sentences: the five clean EXP-033 cases where CPH6 broke worst on the T4 and the five where it did best (input
seeds; development speakers only).

Pass rules, fixed before the run:
1. Validation: Spearman(local CPH6 estimate distance, T4 est_dist_CPH6) >= 0.8 over the ten sentences.
2. Detection: the indicator on the uncorrected path (largest r over the three steps) separates the five broken from the
   five good sentences with AUC >= 0.8.
3. Algorithm: some threshold gives mean passes <= 11 with every broken sentence repaired (UTMOS >= its W8R2 - 0.3),
   and AB2 (10 passes) does not already repair them all. If AB2 does, the known solver is enough and the adaptive rule
   is not needed.

Usage: .venv/Scripts/python -I code/analysis/adaptive_teacher_screen.py
"""
import csv
import json
import math
import statistics as st
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for p in ("code/colab", "code/prototype"):
    sys.path.insert(0, str(ROOT / p))

EXP033_QUAL = ROOT / "research/experiments/EXP-033-cheaper-teacher-path/results/quality.csv"
OUT = ROOT / "research/experiments/EXP-033-cheaper-teacher-path/offline/adaptive_teacher_screen.json"
TAUS = (0.07612046748871326, 0.2928932188134524, 0.6173165676349102)
T_C = TAUS[1]
MASK = T_C / (1.0 - T_C)                 # tau / (1 - tau) at the restart
N_COARSE, K_COARSE = 6, 3                # sway(6)[3] = t_c
PASS_SPEARMAN, PASS_AUC, PASS_PASSES, REPAIR_MARGIN = 0.8, 0.8, 11.0, 0.3


def sway(n):
    return [1.0 - math.cos(math.pi / 2.0 * i / n) for i in range(n + 1)]


def pick_cases(n=5):
    rows = [r for r in csv.DictReader(open(EXP033_QUAL, encoding="utf-8")) if r["condition"] == "clean"]
    rows.sort(key=lambda r: float(r["utmos_CPH6"]) - float(r["utmos_W8R2"]))
    return ([(r, "broken") for r in rows[:n]] + [(r, "good") for r in rows[-n:]])


class Teacher:
    def __init__(self):
        import engine as en
        self.en, self.e = en, en.Engine()
        self.torch = self.e.torch
        self.evals = 0

    def vel(self, ctx, x, t):
        tt = self.torch.tensor(t, dtype=self.torch.float32)
        kw = dict(x=x, cond=ctx["step_cond"], text=ctx["text_ids"], time=tt, mask=None)
        vc = self.e.tts.teacher(drop_audio_cond=False, drop_text=False, **kw)
        vu = self.e.tts.teacher(drop_audio_cond=True, drop_text=True, **kw)
        self.evals += 1
        return vc + (vc - vu) * 2.0

    def euler_estimate(self, ctx, grid, k):
        x = ctx["y0"].clone()
        for i in range(k):
            x = x + (grid[i + 1] - grid[i]) * self.vel(ctx, x, grid[i])
        return x + (1.0 - grid[k]) * self.vel(ctx, x, grid[k])

    def student(self, ctx, xhat):
        torch = self.torch
        x = xhat.clone()
        for tau in TAUS[1:]:
            gen = torch.Generator(device="cpu")
            gen.manual_seed((int(ctx["seed"]) + int(tau * 100)) % (2 ** 31 - 1))
            eps = torch.randn(x.shape, dtype=torch.float32, generator=gen)
            tt = torch.tensor([tau], dtype=torch.float32)
            ts = tt.unsqueeze(-1).unsqueeze(-1)
            phi = (1.0 - ts) * eps + ts * x
            pred = self.e.tts.generator(x=phi, cond=ctx["cond"], text=ctx["text_ids"], time=tt,
                                        drop_audio_cond=False, drop_text=False)
            x = phi + (1.0 - ts) * pred
        return x


def rms_gen(x, plen):
    g = x[:, plen:, :]
    return float(g.pow(2).mean().sqrt())


def dist(a, ref, plen):
    return float((a[:, plen:, :] - ref[:, plen:, :]).norm() / ref[:, plen:, :].norm())


def correction_tree(T, ctx):
    """Every correction pattern on the coarse grid. Returns ({pattern: estimate}, {prefix: indicator r})."""
    g, plen = sway(N_COARSE), ctx["prompt_len"]
    leaves, ind = {}, {}

    def expand(x_a, v_a, i, pattern):
        t_a, t_b = g[i], g[i + 1]
        h = t_b - t_a
        x_e = x_a + h * v_a
        v_e = T.vel(ctx, x_e, t_b)
        xh_a, xh_e = x_a + (1.0 - t_a) * v_a, x_e + (1.0 - t_b) * v_e
        ind[pattern] = MASK * h / (2.0 * (1.0 - t_b)) * rms_gen(xh_e - xh_a, plen)
        x_h = x_a + (h / 2.0) * (v_a + v_e)
        v_h = T.vel(ctx, x_h, t_b)
        for tag, x_n, v_n in (("E", x_e, v_e), ("H", x_h, v_h)):
            if i + 1 == K_COARSE:
                leaves[pattern + tag] = x_n + (1.0 - t_b) * v_n
            else:
                expand(x_n, v_n, i + 1, pattern + tag)

    x0 = ctx["y0"].clone()
    expand(x0, T.vel(ctx, x0, g[0]), 0, "")
    return leaves, ind


def ab2_estimate(T, ctx):
    """Second-order Adams-Bashforth on the coarse grid (first step Euler); no extra evaluation."""
    g = sway(N_COARSE)
    x, v_prev, v = ctx["y0"].clone(), None, None
    v = T.vel(ctx, x, g[0])
    h_prev = None
    for i in range(K_COARSE):
        h = g[i + 1] - g[i]
        if v_prev is None:
            x = x + h * v
        else:
            w = h / (2.0 * h_prev)
            x = x + h * ((1.0 + w) * v - w * v_prev)
        v_prev, v, h_prev = v, T.vel(ctx, x, g[i + 1]), h
    return x + (1.0 - g[K_COARSE]) * v


def follow(ind, theta):
    """The pattern the adaptive rule takes at threshold theta (correct where r > theta)."""
    p = ""
    for _ in range(K_COARSE):
        p += "H" if ind[p] > theta else "E"
    return p


def auc(pos, neg):
    pairs = [(a > b) + 0.5 * (a == b) for a in pos for b in neg]
    return sum(pairs) / len(pairs)


def ranks(v):
    o = sorted(range(len(v)), key=lambda i: v[i])
    r = [0.0] * len(v)
    for p, i in enumerate(o):
        r[i] = p
    return r


def spearman(a, b):
    ra, rb = ranks(a), ranks(b)
    ma, mb = st.mean(ra), st.mean(rb)
    return sum((x - ma) * (y - mb) for x, y in zip(ra, rb)) / (
        sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb)) ** 0.5


def main():
    T = Teacher()
    torch = T.torch
    utmos = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True).eval()
    items = {it["utt_id"]: it for it in T.en.development_examples()}
    t0 = time.time()
    cases = []
    with torch.inference_mode():
        for row, group in pick_cases():
            it = items[row["utt_id"]]
            prompt, sr = T.en.load_example(it, "clean")
            ctx = T.e.prepare(prompt, sr, it["prompt_text"], it["gt_text"], int(it["input_seed"]))
            plen = ctx["prompt_len"]
            ref = T.euler_estimate(ctx, sway(16), 8)
            est = {"W8R2": ref, "CPH8": T.euler_estimate(ctx, sway(8), 4), "AB2": ab2_estimate(T, ctx)}
            leaves, ind = correction_tree(T, ctx)
            est.update({f"P_{p}": x for p, x in leaves.items()})
            scores = {}
            for name, x in est.items():
                audio = T.e._decode(ctx, T.student(ctx, x)).astype(np.float32)
                scores[name] = {"utmos": float(utmos(torch.from_numpy(audio).unsqueeze(0), 24000)),
                                "est_dist": dist(x, ref, plen)}
            case = {"utt_id": row["utt_id"], "group": group, "frames": ctx["frames"],
                    "t4": {"utmos_CPH6_minus_W8R2": float(row["utmos_CPH6"]) - float(row["utmos_W8R2"]),
                           "est_dist_CPH6": float(row["est_dist_CPH6"]), "est_dist_CPH8": float(row["est_dist_CPH8"])},
                    "indicator": ind, "scores": scores}
            cases.append(case)
            print(f"[{len(cases)}/10] {group:6s} {row['utt_id'][:24]}: r(E path) "
                  f"{[round(ind[p], 4) for p in ('', 'E', 'EE')]}, dist CPH6 {scores['P_EEE']['est_dist']:.3f} "
                  f"(T4 {case['t4']['est_dist_CPH6']:.3f}), UTMOS W8R2 {scores['W8R2']['utmos']:.2f} "
                  f"CPH6 {scores['P_EEE']['utmos']:.2f} AB2 {scores['AB2']['utmos']:.2f} HHH {scores['P_HHH']['utmos']:.2f} "
                  f"({time.time() - t0:.0f}s)", flush=True)

    broken = [c for c in cases if c["group"] == "broken"]
    good = [c for c in cases if c["group"] == "good"]
    maxr = lambda c: max(c["indicator"][p] for p in ("", "E", "EE"))   # noqa: E731
    valid = spearman([c["scores"]["P_EEE"]["est_dist"] for c in cases], [c["t4"]["est_dist_CPH6"] for c in cases])
    detect = auc([maxr(c) for c in broken], [maxr(c) for c in good])
    repaired = lambda c, name: c["scores"][name]["utmos"] >= c["scores"]["W8R2"]["utmos"] - REPAIR_MARGIN  # noqa: E731
    thetas = sorted({0.0, float("inf")} | {v for c in cases for v in c["indicator"].values()})
    curve = []
    for th in thetas:
        paths = [follow(c["indicator"], th) for c in cases]
        curve.append({"theta": th if th != float("inf") else "inf",
                      "mean_passes": st.mean(10 + 2 * p.count("H") for p in paths),
                      "mean_utmos": st.mean(c["scores"][f"P_{p}"]["utmos"] for c, p in zip(cases, paths)),
                      "mean_est_dist": st.mean(c["scores"][f"P_{p}"]["est_dist"] for c, p in zip(cases, paths)),
                      "broken_repaired": sum(repaired(c, f"P_{p}") for c, p in zip(cases, paths) if c["group"] == "broken"),
                      "good_kept": sum(repaired(c, f"P_{p}") for c, p in zip(cases, paths) if c["group"] == "good")})
    ok_curve = [x for x in curve if x["mean_passes"] <= PASS_PASSES and x["broken_repaired"] == len(broken)]
    ab2_all = all(repaired(c, "AB2") for c in broken)
    summary = {
        "validation_spearman_local_vs_t4_est_dist": valid, "pass_validation": valid >= PASS_SPEARMAN,
        "detection_auc": detect, "pass_detection": detect >= PASS_AUC,
        "ab2_repairs_all_broken": ab2_all,
        "pass_algorithm": bool(ok_curve) and not ab2_all,
        "means": {name: {"utmos": st.mean(c["scores"][name]["utmos"] for c in cases),
                         "est_dist": st.mean(c["scores"][name]["est_dist"] for c in cases),
                         "broken_repaired": sum(repaired(c, name) for c in broken)}
                  for name in ("W8R2", "CPH8", "AB2", "P_EEE", "P_HHH")},
        "passes": {"W8R2": 20, "CPH8": 12, "AB2": 10, "P_EEE": 10, "P_HHH": 16}}
    res = {"kind": "local CPU screen (screening only, not evidence)", "mask_factor": MASK, "t_c": T_C,
           "rules": {"spearman": PASS_SPEARMAN, "auc": PASS_AUC, "passes": PASS_PASSES, "repair_margin": REPAIR_MARGIN},
           "summary": summary, "threshold_curve": curve, "cases": cases, "seconds": round(time.time() - t0)}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(summary, indent=2))
    for x in curve:
        print(x)
    print(f"[OK] wrote {OUT}")


if __name__ == "__main__":
    main()
