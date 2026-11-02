"""EXP-017: one development run that rehearses Chapter 3 on two teacher grids (Colab Tesla T4).

The DMOSpeech 2 paper reports one teacher-guided setting, t_switch = 0.25 with 14 teacher and 2 student steps
(Li et al., Section 4.1, p. 9 of arXiv v1), and its Algorithm 2 runs the teacher to the last sway-grid point at or
below t_switch; fourteen steps at 0.25 match a 32-interval sway grid. The thesis runs the released sampler, whose
default 16-interval grid stops at the first point above s (0.25 -> Step 8, 21 forward passes). For every input this
run synthesizes and scores 13 outputs:

  g16_4 .. g16_8  16-interval grid, Steps 4, 5, 6 and 8 (s = 0.07, 0.10, 0.15, 0.25; Step 4 is the code default and
                  Step 8 the thesis reference); student restart j = 1 (three steps); 13 / 15 / 17 / 21 passes
  s007 .. s025    32-interval grid, the paper's stopping rule k*(s) = max{i : t_i <= s} for s = 0.07 .. 0.25
                  (k* = 7, 9, 11, 13, 14); student restart j = 1; 2(k* + 1) + 3 passes; s025 is this grid's reference
  paper           the paper's 14 + 2 with the released handover: endpoint at t_14 re-noised at the student's j = 2; 32 passes
  alg2            the paper's Algorithm 2 as written: the student continues from the teacher's state at s_1 = t_14 without
                  re-noising and steps again at s_2 = 0.6173 (the released student's last level; the paper does not
                  state s_2); 30 passes
  teacher32       teacher only: the 32-step guided teacher to t = 1 (F5-TTS's default step count); 64 passes
  student4        student only: DMOSpeech 2's four student steps from noise (the released code's student path); 4 passes

The teacher runs without its stochastic term (eta = 0), as in the thesis. Each input's prompt SNR, the drift on both
grids and the per-component times (preparation, one guided teacher evaluation, one student step, the vocoder) are
recorded. The analysis (CSV only) rehearses Objectives 1, 3 and 4 of Chapter 3 on both grids; see analyze().

Data: the EXP-012 Quick set (60 development speakers x clean, noisy15, Gaussian 18/20/22/25 dB = 360 cases).
Development evidence: the schedulers are replayed on recorded outputs; the held-out evaluation follows the proposal.

Subcommands: smoke (2 non-pilot development speakers x clean, gauss20, with reproduction checks), run, analyze.
"""
import argparse
import csv
import datetime
import hashlib
import json
import math
import os
import random
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_followup_tests as rf  # noqa: E402
import run_consistency_gate_prelim as cg  # noqa: E402
import run_endpoint_extrapolation_prelim as ex  # noqa: E402
import run_noise_guidance_prelim as ng  # noqa: E402

THESIS_ROOT = rf.THESIS_ROOT
N_SPEAKERS, CONDS, NOISY = ex.N_SPEAKERS, ex.CONDS, ex.NOISY
ALPHA, DELTA, FOLDS, SEED, THETA = ex.ALPHA, ex.DELTA, ex.FOLDS, ex.SEED, ex.THETA     # 0.05, 0.10, 5, 2026, 20 dB


def sway_grid(n):
    """Sway schedule (coefficient -1) with n intervals: t_i = 1 - cos(pi/2 * i/n)."""
    return [1.0 - math.cos(math.pi / 2.0 * i / n) for i in range(n + 1)]


def last_at_or_below(grid, s):
    """The paper's Algorithm 2: the teacher runs to the last grid point t_k <= t_switch."""
    return max(i for i, t in enumerate(grid) if t <= s)


def first_above(grid, s):
    """The released code: the teacher stops at the first grid point above the stopping argument."""
    return min(i for i, t in enumerate(grid) if t > s)


GRID32 = sway_grid(32)
SWITCH = (0.07, 0.10, 0.15, 0.20, 0.25)
NAME = {s: f"s{round(s * 100):03d}" for s in SWITCH}                 # s007 ... s025
K32 = {s: last_at_or_below(GRID32, s) for s in SWITCH}               # 7, 9, 11, 13, 14
CANDS32 = tuple(NAME[s] for s in SWITCH)                             # cheapest first; s025 is the reference
G16_STEPS = (4, 5, 6, 8)
TEACHER_ONLY_K = 31                                                  # x1hat(t_31) is the Euler step to t_32 = 1
REF32, REF16, PAPER, ALG2, TEACHER, STUDENT = "s025", "g16_8", "paper", "alg2", "teacher32", "student4"
ALG2_TIMES = (GRID32[K32[0.25]], 1.0 - math.cos(math.pi / 2.0 * 0.75))   # s_1 = t_14 = 0.2270; s_2 = 0.6173 (assumed)
OUTS = tuple(f"g16_{k}" for k in G16_STEPS) + CANDS32 + (PAPER, ALG2, TEACHER, STUDENT)
EVALS = {**{f"g16_{k}": k + 1 for k in G16_STEPS}, **{NAME[s]: K32[s] + 1 for s in SWITCH},
         PAPER: K32[0.25] + 1, ALG2: K32[0.25], TEACHER: TEACHER_ONLY_K + 1, STUDENT: 0}     # guided teacher evaluations
STUDENT_STEPS = {**{o: 3 for o in OUTS}, PAPER: 2, ALG2: 2, TEACHER: 0, STUDENT: 4}
COST = {o: 2 * EVALS[o] + STUDENT_STEPS[o] for o in OUTS}            # forward passes
LABEL = {**{f"g16_{k}": f"16-interval Step {k} (s = {s:.2f})" for k, s in zip(G16_STEPS, (0.07, 0.10, 0.15, 0.25))},
         **{NAME[s]: f"32-interval s = {s:.2f} ({K32[s]} teacher steps)" for s in SWITCH},
         PAPER: "Paper's 14 + 2, released handover (32-interval)",
         ALG2: "Paper's Algorithm 2 as written (32-interval)",
         TEACHER: "Teacher only, 32 steps (F5-TTS)", STUDENT: "Student only, 4 steps (DMOSpeech 2)"}
LABEL["g16_4"] += ", code default"
LABEL["g16_8"] += ", thesis reference"
LABEL[REF32] += ", reference on this grid"
METRICS = ("wer", "similarity", "utmos")
TIMING = ("t_prepare", "t_eval", "t_student", "t_vocoder", "audio_seconds", "peak_mem_mb")
FIELDS = ["utt_id", "speaker_id", "condition", "prompt_snr", "gpu", "gt_sr", "case_seconds", "d4_16", "d4_stored"] + \
         [f"d16_{i}" for i in range(1, 9)] + [f"d32_{i}" for i in range(1, K32[0.25] + 1)] + \
         [f"{m}_{o}" for o in OUTS for m in METRICS + ("hypothesis",)] + \
         [f"{m}_g16_{k}_stored" for k in G16_STEPS for m in ("wer", "similarity")] + list(TIMING) + ["utc"]


# ----------------------------------------------------------------------------- synthesis
def grid_teacher(pipe, ctx, grid, wanted):
    """Guided Euler teacher on `grid` with the released guidance (gamma = CFG_STRENGTH, no stochastic term).
    Returns ({i: x1hat(t_i)}, {i: d_i} for i = 1..max(wanted), {i: state x_{t_i}}) for i in wanted."""
    torch, sampler = pipe.torch, pipe.sampler
    kmax, plen = max(wanted), ctx["prompt_len"]
    x, prev, ends, drift, states = ctx["y0"].clone(), None, {}, {}, {}
    with torch.inference_mode():
        for i in range(kmax + 1):
            if i in wanted:
                states[i] = x
            t = torch.tensor(grid[i], device=pipe.device, dtype=torch.float32)
            kw = dict(x=x, cond=ctx["step_cond"], text=ctx["text_ids"], time=t, mask=None)
            vc = pipe.tts.teacher(drop_audio_cond=False, drop_text=False, **kw)
            vu = pipe.tts.teacher(drop_audio_cond=True, drop_text=True, **kw)
            v = vc + (vc - vu) * sampler.CFG_STRENGTH
            ep = x + (1.0 - grid[i]) * v
            if prev is not None:
                drift[i] = sampler.compute_stabilization_drift(ep, prev, plen)
            if i in wanted:
                ends[i] = ep
            prev = ep
            if i < kmax:
                x = x + (grid[i + 1] - grid[i]) * v
    return ends, drift, states


def student_from_state(pipe, ctx, state, times):
    """Algorithm 2's student (Li et al.): the first step reads the teacher's state at times[0] without re-noising;
    each later step re-noises the previous estimate to the next time, with the released student's per-level seeds."""
    torch, sampler = pipe.torch, pipe.sampler
    x, est = state, None
    with torch.inference_mode():
        for m, s in enumerate(times):
            tt = torch.tensor([s], device=pipe.device, dtype=torch.float32)
            ts = tt.unsqueeze(-1).unsqueeze(-1)
            if m > 0:
                gen = sampler.make_local_generator((int(ctx["seed"]) + int(s * 100)) % (2 ** 31 - 1), device=pipe.device)
                eps = torch.randn(est.shape, dtype=torch.float32, device=pipe.device, generator=gen)
                x = (1.0 - ts) * eps + ts * est
            pred = pipe.tts.generator(x=x, cond=ctx["cond"], text=ctx["text_ids"], time=tt, drop_audio_cond=False, drop_text=False)
            est = x + (1.0 - ts) * pred
    return est


def student(pipe, ctx, endpoint, taus):
    """The released student loop (same per-level seeds as run_student_completion) over the levels `taus`:
    STUDENT_TAUS for the default restart j = 1, STUDENT_TAUS[1:] for j = 2, [0] + STUDENT_TAUS for student only
    (level 0 is pure noise, as in the released _student_sampling)."""
    torch, sampler = pipe.torch, pipe.sampler
    x = endpoint.clone()
    with torch.inference_mode():
        for tau in taus:
            gen = sampler.make_local_generator((int(ctx["seed"]) + int(tau * 100)) % (2 ** 31 - 1), device=pipe.device)
            eps = torch.randn(x.shape, dtype=torch.float32, device=pipe.device, generator=gen)
            tt = torch.tensor([tau], device=pipe.device, dtype=torch.float32)
            ts = tt.unsqueeze(-1).unsqueeze(-1)
            phi = (1.0 - ts) * eps + ts * x
            pred = pipe.tts.generator(x=phi, cond=ctx["cond"], text=ctx["text_ids"], time=tt, drop_audio_cond=False, drop_text=False)
            x = phi + (1.0 - ts) * pred
    return x


def timed(pipe, fn, *args):
    """Runs fn(*args) between CUDA synchronizations; returns (result, seconds)."""
    if pipe.device.type == "cuda":
        pipe.torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = fn(*args)
    if pipe.device.type == "cuda":
        pipe.torch.cuda.synchronize()
    return out, time.perf_counter() - t0


def run_cases(pipe, todo_items, conds, out_dir, csv_name="grid32.csv", reproduce=False):
    path = Path(out_dir) / csv_name
    done = rf.done_keys(path, 1)
    stored = ex.stored_cases()
    todo = [(it, c) for it in todo_items for c in conds if (it["utt_id"], c) not in done]
    print(f"[grid32] {len(todo)} cases to run ({len(done)} already in {path.name}) on {pipe.gpu}", flush=True)
    repro = []
    taus = list(pipe.sampler.STUDENT_TAUS)
    for n, (item, cond) in enumerate(todo, 1):
        t0 = time.perf_counter()
        s = stored[(item["utt_id"], cond)]
        if pipe.device.type == "cuda":
            pipe.torch.cuda.reset_peak_memory_stats()
        ctx, t_prep = timed(pipe, pipe.prepare, item, cond, item["input_seed"], out_dir)
        plen = ctx["prompt_len"]
        E16, _ = pipe.teacher(ctx, tuple(range(9)))
        with pipe.torch.inference_mode():
            d16 = {i: pipe.sampler.compute_stabilization_drift(E16[i], E16[i - 1], plen) for i in range(1, 9)}
        (E32, d32, S32), t32 = timed(pipe, grid_teacher, pipe, ctx, GRID32, set(K32.values()) | {TEACHER_ONLY_K})
        starts = {**{f"g16_{k}": (E16[k], taus) for k in G16_STEPS}, **{NAME[sw]: (E32[k], taus) for sw, k in K32.items()},
                  PAPER: (E32[K32[0.25]], taus[1:]), STUDENT: (ctx["step_cond"], [0.0] + taus)}
        xs, t_stu, n_stu = {}, 0.0, 0
        for o, (x0, ts_) in starts.items():
            xs[o], dt = timed(pipe, student, pipe, ctx, x0, ts_)
            t_stu, n_stu = t_stu + dt, n_stu + len(ts_)
        xs[ALG2], dt = timed(pipe, student_from_state, pipe, ctx, S32[K32[0.25]], ALG2_TIMES)
        t_stu, n_stu = t_stu + dt, n_stu + len(ALG2_TIMES)
        xs[TEACHER] = E32[TEACHER_ONLY_K]
        audios, t_voc = {}, 0.0
        for o in OUTS:
            audios[o], dt = timed(pipe, ng.decode, pipe, ctx, xs[o])
            t_voc += dt
        peak = pipe.torch.cuda.max_memory_allocated() / 2 ** 20 if pipe.device.type == "cuda" else 0.0
        gt = pipe.ground_truth(item)
        row = {"utt_id": item["utt_id"], "speaker_id": item["speaker_id"], "condition": cond, "prompt_snr": ctx["prompt_snr"],
               "gpu": pipe.gpu, "gt_sr": gt[1], "d4_16": d16[4], "d4_stored": float(s[8]["d_4"]),
               **{f"d16_{i}": d for i, d in d16.items()}, **{f"d32_{i}": d32[i] for i in range(1, K32[0.25] + 1)},
               "t_prepare": round(t_prep, 4), "t_eval": round(t32 / (TEACHER_ONLY_K + 1), 5), "t_student": round(t_stu / n_stu, 5),
               "t_vocoder": round(t_voc / len(OUTS), 5), "audio_seconds": round(len(audios[REF16]) / 24000, 3), "peak_mem_mb": round(peak, 1)}
        for k in G16_STEPS:
            if s.get(k) is not None:
                m = ng.stored_metrics(s[k])
                row.update({f"wer_g16_{k}_stored": m["wer"], f"similarity_g16_{k}_stored": m["similarity"]})
        for o in OUTS:
            m = pipe.score(audios[o], item, gt)
            row.update({f"{k}_{o}": m[k] for k in METRICS + ("hypothesis",)})
        if reproduce:
            ends, _, _ = grid_teacher(pipe, ctx, pipe.sampler.SWAY_TEACHER_GRID, set(G16_STEPS))
            with pipe.torch.inference_mode():
                for k in G16_STEPS:
                    repro.append({"utt_id": item["utt_id"], "condition": cond, "step": k,
                                  "check": "grid_teacher on the released 16-interval grid vs the sampler's endpoint",
                                  "max_abs_diff": float((ends[k] - E16[k]).abs().max())})
        row["case_seconds"] = round(time.perf_counter() - t0, 2)
        row["utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        new = not path.exists()
        with open(path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            if new:
                w.writeheader()
            w.writerow(row)
            f.flush()
            os.fsync(f.fileno())
        print(f"[{n}/{len(todo)}] grid32: {item['utt_id']} {cond} ({row['case_seconds']:.0f} s)", flush=True)
    print(f"[OK] grid32 complete: {path}", flush=True)
    return repro


# ----------------------------------------------------------------------------- analysis: data
def thresholds():
    return json.loads((THESIS_ROOT / "code/scheduling/failure_config.json").read_text(encoding="utf-8"))


def failed(early, ref, th):
    """Definition 3.1: the early output is clearly worse than the reference output of the same input."""
    return early["wer"] > max(th["absolute_wer_ceiling"], ref["wer"] + th["relative_wer_margin"]) or \
        early["similarity"] < ref["similarity"] - th["similarity_margin"]


def _num(v):
    return None if v in (None, "") else float(v)


def build_table(rows):
    table = []
    for r in rows:
        m = {o: {k: float(r[f"{k}_{o}"]) for k in METRICS} for o in OUTS}
        stored = {k: {"wer": _num(r.get(f"wer_g16_{k}_stored")), "similarity": _num(r.get(f"similarity_g16_{k}_stored"))}
                  for k in G16_STEPS}
        table.append({"utt": r["utt_id"], "spk": r["speaker_id"], "cond": r["condition"], "snr": float(r["prompt_snr"]),
                      "gpu": r.get("gpu", ""), "gt_sr": r.get("gt_sr", ""), "secs": _num(r.get("case_seconds")) or 0.0,
                      "d4": float(r["d4_16"]), "d4_stored": float(r["d4_stored"]), "m": m, "stored": stored,
                      "d16": {i: float(r[f"d16_{i}"]) for i in range(1, 9) if _num(r.get(f"d16_{i}")) is not None},
                      "d32": {i: float(r[f"d32_{i}"]) for i in range(1, K32[0.25] + 1) if _num(r.get(f"d32_{i}")) is not None},
                      "time": {k: _num(r.get(k)) for k in TIMING}})
    return table


def group(cond):
    return "clean" if cond == "clean" else "noisy"


def key(r):
    return (r["utt"], r["cond"])


# Both grids, as Chapter 3 frames them: early candidates, the reference, the drift used by the policy, the two
# "fixed later step" controls, the fixed threshold grid for lambda and the drift profile length for TAS.
LAM16 = [round(0.20 + 0.05 * i, 2) for i in range(21)]                                  # Chapter 3: 0.20 -> 1.20
LAM32 = [round(0.01 * 120 ** (i / 40), 5) for i in range(41)]                           # 0.01 -> 1.20, geometric
GRIDS = {"g16": {"label": "16-interval grid (released sampler)", "cands": ("g16_4", "g16_5", "g16_6"), "ref": REF16,
                 "k": {"g16_4": 4, "g16_5": 5, "g16_6": 6}, "drift": "d16", "later": {"later5": "g16_5", "later6": "g16_6"},
                 "lam": LAM16, "tas_k": 4},
         "g32": {"label": "32-interval grid (paper's stopping rule)", "cands": CANDS32[:-1], "ref": REF32,
                 "k": {NAME[s]: K32[s] for s in SWITCH[:-1]}, "drift": "d32", "later": {"later5": "s010", "later6": "s015"},
                 "lam": LAM32, "tas_k": K32[0.07]}}
THETA_GRID = ex.GRIDS["theta"]                                                          # 40 -> 20 dB
THETA_DEV = [float(x) for x in range(20, 41)]                                           # Objective 1 gate scan
SCHEDULERS = {"snr20": "SNR-only, 20 dB gate",
              "snr_cal": "SNR-only rule, gate calibrated (λ = ∞)",
              "later5": "SNR gate (20 dB), fixed later step 5",
              "later6": "SNR gate (20 dB), fixed later step 6",
              "stab": "Stabilization only (θ = −∞)",
              "snr_drift": "Full policy (Definition 3.4), 20 dB gate",
              "snr_drift_dev": "Full policy, gate from the Objective 1 rule",
              "snr_tas": "SNR + TAS (candidate)",
              "oracle": "Oracle: cheapest safe candidate in hindsight (diagnostic)"}
EQUIV = {"wer": 0.01, "similarity": 0.02}            # TOST margins: the failure rule's WER and SIM margins
RESPLITS, SIZES, BOOT = 100, (50, 100, 200), 1000


def mark(table, G, th):
    """Failure indicators of every early candidate against the grid's reference."""
    for r in table:
        r.setdefault("fail", {})[G] = {**{o: failed(r["m"][o], r["m"][GRIDS[G]["ref"]], th) for o in GRIDS[G]["cands"]},
                                       GRIDS[G]["ref"]: False}


def clopper(k, n):
    try:
        return rf.clopper_pearson(k, n)
    except ImportError:
        return (None, None)


# ----------------------------------------------------------------------------- analysis: Objective 1
def oracle_pick(r, G):
    g = GRIDS[G]
    return next((o for o in g["cands"] if not r["fail"][G][o]), g["ref"])


def theta_rule(rows, G):
    """Chapter 3, Objective 1: the lowest prompt SNR above which the earliest candidate rarely fails (rate <= ALPHA)."""
    first = GRIDS[G]["cands"][0]
    for th in THETA_DEV:
        above = [r for r in rows if r["snr"] >= th]
        if above and sum(r["fail"][G][first] for r in above) / len(above) <= ALPHA:
            return th
    return None


def objective1(table, G):
    g = GRIDS[G]
    fixed = {}
    for o in g["cands"] + (g["ref"],):
        per = {c: sum(r["fail"][G][o] for r in table if r["cond"] == c) for c in CONDS}
        k, n = sum(per.values()), len(table)
        fixed[o] = {"failures": k, "n": n, "ci": clopper(k, n), "per_condition": per,
                    "clean": per.get("clean", 0), "noisy": sum(per[c] for c in NOISY)}
    picks = {key(r): oracle_pick(r, G) for r in table}
    order = g["cands"] + (g["ref"],)
    per_cond = {c: {o: sum(picks[key(r)] == o for r in table if r["cond"] == c) for o in order} for c in CONDS}
    n_c = {c: sum(r["cond"] == c for r in table) for c in CONDS}
    earlier = {c: (n_c[c] - per_cond[c][g["ref"]]) / n_c[c] if n_c[c] else None for c in CONDS}
    assoc = {}
    try:
        from scipy.stats import spearmanr
        idx = [order.index(picks[key(r)]) for r in table]
        for name, vals in [("snr", [r["snr"] for r in table])] + \
                [(f"drift at {o}", [r[g["drift"]].get(g["k"][o], float("nan")) for r in table]) for o in g["cands"]]:
            rho, p = spearmanr(vals, idx)
            assoc[name] = {"rho": None if rho != rho else float(rho), "p": None if p != p else float(p)}
    except ImportError:
        pass
    return {"fixed": fixed, "oracle_per_condition": per_cond, "oracle_earlier_share": earlier, "association": assoc,
            "theta_rule": theta_rule(table, G)}


def signal_plot(table, G, path):
    """Scatter of the prompt SNR against the drift at the earliest candidate, coloured by the oracle choice."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    g = GRIDS[G]
    first, order = g["cands"][0], g["cands"] + (g["ref"],)
    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=150)
    for o in order:
        pts = [(r["snr"], r[g["drift"]][g["k"][first]]) for r in table if oracle_pick(r, G) == o and g["k"][first] in r[g["drift"]]]
        if pts:
            ax.scatter(*zip(*pts), s=12, label=f"cheapest safe: {o}")
    ax.set_xlabel("Prompt SNR estimate (dB)")
    ax.set_ylabel(f"Drift at {first}")
    ax.set_yscale("log")
    ax.axvline(THETA, color="grey", lw=0.8, ls="--")
    ax.legend(fontsize=7)
    ax.set_title(g["label"], fontsize=9)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path.name


# ----------------------------------------------------------------------------- analysis: schedulers (Objectives 3-4)
def p_fixed(o):
    return lambda r: o


def p_snr(G, theta):
    g = GRIDS[G]
    return lambda r: g["cands"][0] if theta is not None and r["snr"] >= theta else g["ref"]


def p_later(G, o, theta=THETA):
    return lambda r: o if r["snr"] >= theta else GRIDS[G]["ref"]


def p_stab(G, lam, theta=None):
    """Definition 3.4: the first candidate whose drift is at most lam (and whose prompt passes the gate), else the reference."""
    g = GRIDS[G]

    def pol(r):
        if lam is None or (theta is not None and r["snr"] < theta):
            return g["ref"]
        return next((o for o in g["cands"] if r[g["drift"]][g["k"][o]] <= lam), g["ref"])
    return pol


def tas_fit(train, G):
    """One-sided squared Mahalanobis distance of log(d_1..d_k) from the clean training prompts (no failure labels)."""
    import numpy as np
    g = GRIDS[G]
    logs = lambda r: [math.log(max(r[g["drift"]][i], 1e-12)) for i in range(1, g["tas_k"] + 1)]
    X = np.array([logs(r) for r in train if r["cond"] == "clean"])
    mu = X.mean(0)
    Ci = np.linalg.inv(np.cov(X, rowvar=False) + 1e-3 * np.eye(X.shape[1]))

    def score(r):
        z = np.asarray(logs(r)) - mu
        return float(z @ Ci @ z) if z.mean() < 0 else 0.0
    return score


def p_tas(G, score, kappa):
    g = GRIDS[G]
    return lambda r: g["cands"][0] if kappa is not None and r["snr"] >= THETA and score(r) <= kappa else g["ref"]


def certified(train, pol, G):
    """Learn-Then-Test for one threshold: the binomial p-value of the training failures at risk ALPHA is at most DELTA."""
    return rf.binom_cdf(sum(int(r["fail"][G][pol(r)]) for r in train), len(train), ALPHA) <= DELTA


def fixed_sequence(train, grid, make, G):
    """Algorithm 3.4: tests thresholds from the most conservative; returns the last certified one (None if the first fails)."""
    chosen = None
    for p in grid:
        if not certified(train, make(p), G):
            break
        chosen = p
    return chosen


def fold_policy(name, train, G):
    g = GRIDS[G]
    if name == "snr20":
        return p_snr(G, THETA), THETA
    if name == "snr_cal":
        theta = fixed_sequence(train, THETA_GRID, lambda x: p_snr(G, x), G)
        return p_snr(G, theta), theta
    if name in g["later"]:
        return p_later(G, g["later"][name]), THETA
    if name in ("stab", "snr_drift", "snr_drift_dev"):
        gate = None if name == "stab" else (THETA if name == "snr_drift" else theta_rule(train, G))
        if name == "snr_drift_dev" and gate is None:
            return p_fixed(g["ref"]), (None, None)
        lam = fixed_sequence(train, g["lam"], lambda x: p_stab(G, x, gate), G)
        return p_stab(G, lam, gate), (lam if name != "snr_drift_dev" else (gate, lam))
    if name == "snr_tas":
        score = tas_fit(train, G)
        kappa = fixed_sequence(train, sorted({score(r) for r in train if r["snr"] >= THETA}), lambda x: p_tas(G, score, x), G)
        return p_tas(G, score, kappa), kappa
    raise ValueError(name)


def scheduler_names():
    names = [n for n in SCHEDULERS if n != "oracle"]
    try:
        import numpy  # noqa: F401  (TAS needs it)
    except ImportError:
        names.remove("snr_tas")
    return names


def folds_of(table, seed=SEED, k=FOLDS):
    spks = sorted({r["spk"] for r in table})
    random.Random(seed).shuffle(spks)
    return {s: i % k for i, s in enumerate(spks)}


def cross_validate(table, G, train_filter=None, test_filter=None):
    """Out-of-fold picks of every scheduler (train_filter / test_filter restrict the calibration and evaluation rows)."""
    fold = folds_of(table)
    picks, chosen = {}, {}
    for name in scheduler_names():
        pick, ch = {}, []
        for f in range(FOLDS):
            train = [r for r in table if fold[r["spk"]] != f and (train_filter is None or train_filter(r))]
            pol, p = fold_policy(name, train, G)
            ch.append(p)
            pick.update({key(r): pol(r) for r in table if fold[r["spk"]] == f and (test_filter is None or test_filter(r))})
        picks[name], chosen[name] = pick, ch
    return picks, chosen


def runtime(r, o):
    t = r["time"]
    if any(t.get(x) is None for x in ("t_prepare", "t_eval", "t_student", "t_vocoder", "audio_seconds")) or not t["audio_seconds"]:
        return None, None
    sec = t["t_prepare"] + EVALS[o] * t["t_eval"] + STUDENT_STEPS[o] * t["t_student"] + t["t_vocoder"]
    return sec, sec / t["audio_seconds"]


def summarize(table, pick, G, boot=True):
    """Failures (with an exact interval), forward passes, saving, share of the oracle saving (bootstrap over speakers),
    runtime and real-time factor, per-condition failures. Returns (summary, failure list in table order)."""
    g = GRIDS[G]
    rows = [r for r in table if key(r) in pick]
    ref_cost = COST[g["ref"]]
    fails = [bool(r["fail"][G][pick[key(r)]]) for r in rows]
    k, n = sum(fails), len(rows)
    per = {c: sum(f for f, r in zip(fails, rows) if r["cond"] == c) for c in CONDS}
    n_c = {c: sum(r["cond"] == c for r in rows) for c in CONDS}
    costs = [COST[pick[key(r)]] for r in rows]
    orcs = [COST[oracle_pick(r, G)] for r in rows]
    cost, orc = statistics.mean(costs), statistics.mean(orcs)
    eta = (ref_cost - cost) / (ref_cost - orc) if ref_cost > orc else None
    res = {"failures": k, "n": n, "rate": k / n, "ci": clopper(k, n), "mean_cost": cost, "saving": 1 - cost / ref_cost, "eta": eta,
           "per_condition": per, "worst_condition_rate": max((per[c] / n_c[c] for c in CONDS if n_c[c]), default=None),
           "picks": {o: sum(pick[key(r)] == o for r in rows) for o in sorted({pick[key(r)] for r in rows}, key=lambda x: COST[x])}}
    rts = [runtime(r, pick[key(r)]) for r in rows]
    if all(x[0] is not None for x in rts):
        ref_rt = statistics.mean(runtime(r, g["ref"])[0] for r in rows)
        res.update({"runtime": statistics.mean(x[0] for x in rts), "rtf": statistics.mean(x[1] for x in rts),
                    "runtime_saving": 1 - statistics.mean(x[0] for x in rts) / ref_rt})
    if boot:
        agg = {}
        for r, c, o in zip(rows, costs, orcs):                 # resample speakers, keeping all of a speaker's cases
            a = agg.setdefault(r["spk"], [0.0, 0.0, 0])
            a[0], a[1], a[2] = a[0] + c, a[1] + o, a[2] + 1
        spks, rng, sav, etas = sorted(agg), random.Random(SEED), [], []
        for _ in range(BOOT):
            sc = so = cnt = 0
            for s in rng.choices(spks, k=len(spks)):
                sc, so, cnt = sc + agg[s][0], so + agg[s][1], cnt + agg[s][2]
            c_, o_ = sc / cnt, so / cnt
            sav.append(1 - c_ / ref_cost)
            if ref_cost > o_:
                etas.append((ref_cost - c_) / (ref_cost - o_))
        q = lambda v: (sorted(v)[int(0.025 * (len(v) - 1))], sorted(v)[int(0.975 * (len(v) - 1))]) if v else (None, None)
        res["saving_ci"], res["eta_ci"] = q(sav), q(etas)
    return res, fails


def speaker_tests(table, pick, G):
    """Speaker-level paired differences (method - reference) in passes, WER, SIM and UTMOS: Wilcoxon signed-rank p and,
    for WER and SIM, TOST equivalence p with the EQUIV margins."""
    ref = GRIDS[G]["ref"]
    rows = [r for r in table if key(r) in pick]
    spks = sorted({r["spk"] for r in rows})
    out = {}
    for meas in ("passes",) + METRICS:
        diffs = []
        for s in spks:
            rs = [r for r in rows if r["spk"] == s]
            if meas == "passes":
                d = [COST[pick[key(r)]] - COST[ref] for r in rs]
            else:
                d = [r["m"][pick[key(r)]][meas] - r["m"][ref][meas] for r in rs]
            diffs.append(statistics.mean(d))
        v = {"n_speakers": len(diffs), "mean_diff": statistics.mean(diffs) if diffs else None}
        try:
            from scipy.stats import wilcoxon, ttest_1samp
            v["wilcoxon_p"] = float(wilcoxon(diffs).pvalue) if any(abs(x) > 1e-12 for x in diffs) else 1.0
            if meas in EQUIV and len(diffs) > 2:
                m = EQUIV[meas]
                if statistics.pstdev(diffs) < 1e-12:
                    v["tost_p"] = 0.0 if abs(statistics.mean(diffs)) < m else 1.0
                else:
                    lo = ttest_1samp(diffs, -m, alternative="greater").pvalue
                    hi = ttest_1samp(diffs, m, alternative="less").pvalue
                    v["tost_p"] = float(max(lo, hi))
        except (ImportError, ValueError):
            pass
        out[meas] = v
    return out


def holm_adjust(tests):
    """Holm correction across the compared methods, separately for each measure's Wilcoxon p-values."""
    for meas in ("passes",) + METRICS:
        names = [n for n in tests if tests[n][meas].get("wilcoxon_p") is not None]
        adj = rf.holm([tests[n][meas]["wilcoxon_p"] for n in names]) if names else []
        for n_, a in zip(names, adj):
            tests[n_][meas]["wilcoxon_p_holm"] = a


def random_rule(table, G, target_cost, draws=RESPLITS):
    """Cost-matched random rule: per input, the reference with probability p and otherwise a uniformly random early
    candidate, p set so that the expected cost equals target_cost; summarized over seeded draws."""
    g = GRIDS[G]
    early = statistics.mean(COST[o] for o in g["cands"])
    ref_cost = COST[g["ref"]]
    p = min(1.0, max(0.0, (target_cost - early) / (ref_cost - early))) if ref_cost > early else 1.0
    runs = []
    for d in range(draws):
        rng = random.Random(SEED + d)
        runs.append({key(r): (g["ref"] if rng.random() < p else rng.choice(g["cands"])) for r in table})
    return p, runs


def objective3(table, G):
    """Rehearsal of Objective 3 on the development speakers: random 80/20 speaker re-splits and calibration sizes."""
    spks = sorted({r["spk"] for r in table})
    out = {}
    for name in ("snr_drift", "snr_cal"):
        res = {}
        for size in (None,) + SIZES:
            risks, savings, certified_ = [], [], 0
            for rep in range(RESPLITS):
                rng = random.Random(SEED * 1000 + rep)
                cal = set(rng.sample(spks, int(round(0.8 * len(spks)))))
                train = [r for r in table if r["spk"] in cal]
                if size is not None:
                    train = rng.sample(train, min(size, len(train)))
                test = [r for r in table if r["spk"] not in cal]
                pol, p = fold_policy(name, train, G)
                if p is not None:
                    certified_ += 1
                risks.append(sum(r["fail"][G][pol(r)] for r in test) / len(test))
                savings.append(1 - statistics.mean(COST[pol(r)] for r in test) / COST[GRIDS[G]["ref"]])
            srt = sorted(risks)
            res["all" if size is None else str(size)] = {
                "certified_share": certified_ / RESPLITS, "mean_risk": statistics.mean(risks),
                "risk_p95": srt[int(0.95 * (len(srt) - 1))], "share_risk_above_alpha": sum(x > ALPHA for x in risks) / RESPLITS,
                "mean_saving": statistics.mean(savings)}
        out[name] = res
    return out


def objective4(table, G, out_dir=None):
    g = GRIDS[G]
    picks, chosen = cross_validate(table, G)
    methods, fails = {}, {}
    for o in (g["ref"],) + g["cands"]:
        methods[f"fixed {o}"], fails[f"fixed {o}"] = summarize(table, {key(r): o for r in table}, G)
    for name, pick in picks.items():
        methods[name], fails[name] = summarize(table, pick, G)
    p, runs = random_rule(table, G, methods["snr_drift"]["mean_cost"])
    rand = [summarize(table, run, G, boot=False) for run in runs]
    methods["random"] = dict(rand[0][0], p_reference=p, mean_failures_over_draws=statistics.mean(x[0]["failures"] for x in rand),
                             mean_cost_over_draws=statistics.mean(x[0]["mean_cost"] for x in rand))
    fails["random"] = rand[0][1]
    orc = {key(r): oracle_pick(r, G) for r in table}
    methods["oracle"], fails["oracle"] = summarize(table, orc, G)
    mcn = {}
    for a, b in (("snr_drift", "snr20"), ("snr_drift", "snr_cal"), ("snr_drift", "later5"), ("snr_drift", "later6"),
                 ("snr_drift", "random"), ("snr_drift", "stab"), ("snr_drift_dev", "snr_drift"), ("snr_tas", "snr_drift"),
                 ("snr_tas", "snr20")):
        if a in fails and b in fails:
            x, y, pv = rf.mcnemar_exact(fails[b], fails[a])
            mcn[f"{a} vs {b}"] = {"fixed": x, "new": y, "p": pv}
    tests = {n: speaker_tests(table, pk, G) for n, pk in
             [(f"fixed {o}", {key(r): o for r in table}) for o in g["cands"]] + list(picks.items()) + [("oracle", orc)]}
    holm_adjust(tests)
    clean_noisy, _ = cross_validate(table, G, train_filter=lambda r: r["cond"] == "clean", test_filter=lambda r: r["cond"] != "clean")
    robust = {n: summarize(table, clean_noisy[n], G, boot=False)[0] for n in ("snr_cal", "stab", "snr_drift") if n in clean_noisy}
    return {"methods": methods, "chosen": chosen, "mcnemar": mcn, "tests": tests, "clean_to_noisy": robust}


# ----------------------------------------------------------------------------- analysis: whole run
def quality(table):
    res = {}
    for o in OUTS:
        res[o] = {}
        for gname in ("clean", "noisy", "all"):
            rows = [r for r in table if gname == "all" or group(r["cond"]) == gname]
            res[o][gname] = {k: statistics.mean(r["m"][o][k] for r in rows) if rows else None for k in METRICS}
        rts = [runtime(r, o) for r in table]
        if rts and all(x[0] is not None for x in rts):
            res[o]["runtime"], res[o]["rtf"] = statistics.mean(x[0] for x in rts), statistics.mean(x[1] for x in rts)
    return res


def paired(table, a, b):
    """Mean differences a - b (SIM higher is better, WER lower is better), share of inputs with higher SIM, Wilcoxon p."""
    out = {}
    for gname, keep in (("clean", lambda c: c == "clean"), ("noisy", lambda c: c != "clean"), ("all", lambda c: True)):
        rows = [r for r in table if keep(r["cond"])]
        ds = [r["m"][a]["similarity"] - r["m"][b]["similarity"] for r in rows]
        dw = [r["m"][a]["wer"] - r["m"][b]["wer"] for r in rows]
        v = {"n": len(rows), "sim_mean_diff": statistics.mean(ds) if ds else None, "wer_mean_diff": statistics.mean(dw) if dw else None,
             "share_sim_higher": sum(d > 0 for d in ds) / len(ds) if ds else None}
        try:
            from scipy.stats import wilcoxon
            v["sim_wilcoxon_p"] = float(wilcoxon(ds).pvalue) if any(ds) else 1.0
            v["wer_wilcoxon_p"] = float(wilcoxon(dw).pvalue) if any(dw) else 1.0
        except (ImportError, ValueError):
            pass
        out[gname] = v
    return out


def reproduction(table, th):
    diffs = {}
    for k in G16_STEPS:
        for meas in ("wer", "similarity"):
            vals = [abs(r["m"][f"g16_{k}"][meas] - r["stored"][k][meas]) for r in table if r["stored"][k][meas] is not None]
            diffs[f"{meas}_g16_{k}"] = (max(vals), len(vals)) if vals else (None, 0)
    both = [r for r in table if r["stored"][4]["wer"] is not None and r["stored"][8]["wer"] is not None]
    same = sum(failed(r["m"]["g16_4"], r["m"]["g16_8"], th) == failed(r["stored"][4], r["stored"][8], th) for r in both)
    return {"max_abs_diff": diffs, "drift_d4_max_abs_diff": max((abs(r["d4"] - r["d4_stored"]) for r in table), default=None),
            "same_step4_failure_label": same, "n": len(both)}


def timing(table):
    t = {k: [r["time"][k] for r in table if r["time"].get(k) is not None] for k in TIMING}
    return {k: (statistics.mean(v) if v else None) for k, v in t.items()} | \
        {"peak_mem_mb_max": max(t["peak_mem_mb"]) if t["peak_mem_mb"] else None,
         "mean_case_seconds": statistics.mean(r["secs"] for r in table) if table else None}


def analyze(out, allow_partial=False):
    out = Path(out)
    prereg = json.loads((out / "preregistration.json").read_text(encoding="utf-8"))
    table = build_table(rf.read_csv(out / "grid32.csv", repair=True))
    expected = N_SPEAKERS * len(CONDS)
    if len(table) < expected and not allow_partial:
        raise SystemExit(f"{len(table)} of {expected} cases; finish the run or pass --allow_partial")
    th = thresholds()
    for G in GRIDS:
        mark(table, G, th)
    res = {"preregistration_sha256": prereg["sha256"], "cases": len(table), "expected_cases": expected, "partial": len(table) < expected,
           "gpus": sorted({r["gpu"] for r in table}), "gt_sample_rates": sorted({str(r["gt_sr"]) for r in table}),
           "cost": COST, "evals": EVALS, "student_steps": STUDENT_STEPS, "timing": timing(table), "quality": quality(table),
           "reproduction": reproduction(table, th),
           "fixed_vs_thesis_reference": {o: {c: sum(failed(r["m"][o], r["m"][REF16], th) for r in table if r["cond"] == c) for c in CONDS}
                                         for o in OUTS if o != REF16},
           "paired": {f"{a} vs {b}": paired(table, a, b) for a, b in
                      ((ALG2, REF16), (PAPER, REF16), (ALG2, PAPER), (REF32, REF16), (TEACHER, REF16), (STUDENT, REF16), ("g16_4", REF16))},
           "objective1": {}, "plots": {}}
    complete = len({r["spk"] for r in table}) >= FOLDS
    for G in GRIDS:
        res["objective1"][G] = objective1(table, G)
        res["plots"][G] = signal_plot(table, G, out / f"objective1_signals_{G}.png")
        drift_ok = all(set(GRIDS[G]["k"].values()) <= set(r[GRIDS[G]["drift"]]) for r in table) and \
            all(len(r[GRIDS[G]["drift"]]) >= GRIDS[G]["tas_k"] for r in table)
        if complete and drift_ok:
            res.setdefault("objective4", {})[G] = objective4(table, G)
            res.setdefault("objective3", {})[G] = objective3(table, G)
    (out / "grid32_results.json").write_text(json.dumps(res, indent=2, default=str) + "\n", encoding="utf-8")
    (out / "GRID32_REPORT.md").write_text(render(res), encoding="utf-8")
    print(f"Wrote {out / 'GRID32_REPORT.md'}", flush=True)
    return res


# ----------------------------------------------------------------------------- report
def _f(x, fmt):
    return "n/a" if x is None else format(x, fmt)


def _ci(ci, fmt=".1%"):
    return "n/a" if not ci or ci[0] is None else f"[{format(ci[0], fmt)}, {format(ci[1], fmt)}]"


def _thr(v):
    """A chosen threshold for the report: None, a number, or a (gate, lambda) pair, rounded."""
    if isinstance(v, (list, tuple)):
        return "(" + ", ".join(_thr(x) for x in v) + ")"
    return "None" if v is None else f"{v:.4g}"


def render(res):
    n = res["cases"]
    n_clean = n // len(CONDS)                                      # every speaker has every condition
    n_noisy = n - n_clean
    rep, tm = res["reproduction"], res["timing"]
    L = ["# EXP-017: Chapter 3 rehearsal on two teacher grids (development speakers)", "",
         f"{'**PARTIAL.** ' if res['partial'] else ''}{N_SPEAKERS} development speakers × {len(CONDS)} conditions ({', '.join(CONDS)}); "
         f"{n} of {res['expected_cases']} cases on {', '.join(res['gpus'])}; preregistration `{res['preregistration_sha256'][:16]}…`. "
         "Every number is computed from `grid32.csv` by `run_grid32_prelim.py analyze`. Development evidence: schedulers are replayed on "
         "recorded outputs and calibrated by cross-validation on the same speakers; the certified evaluation on held-out speakers follows the proposal.", "",
         "## Outputs", "", "| Output | Teacher evaluations | Student steps | Forward passes | Mean runtime (s) | RTF |", "|---|---:|---:|---:|---:|---:|"]
    for o in OUTS:
        q = res["quality"][o]
        L.append(f"| {LABEL[o]} | {EVALS[o]} | {STUDENT_STEPS[o]} | {COST[o]} | {_f(q.get('runtime'), '.2f')} | {_f(q.get('rtf'), '.3f')} |")
    L += ["", "Forward passes: two per guided teacher evaluation plus one per student step (Chapter 3's cost formula). Runtime is synthesis time "
          "(preparation, teacher, student, vocoder; not scoring), composed per input from the measured time of one guided evaluation, "
          "one student step and one vocoder call; RTF = runtime / audio length.", "",
          "## Checks", "",
          "- Fresh 16-interval outputs against the stored T4 labels (max |Δ|, cases compared): " +
          "; ".join(f"{k} {_f(v[0], '.4f')} ({v[1]})" for k, v in rep["max_abs_diff"].items()) +
          f". Same Step-4 failure label in {rep['same_step4_failure_label']} of {rep['n']} cases; max |Δd₄| {_f(rep['drift_d4_max_abs_diff'], '.2e')}.",
          f"- Ground-truth sample rates: {', '.join(res['gt_sample_rates'])}. Mean per case: {_f(tm['mean_case_seconds'], '.1f')} s "
          f"(preparation {_f(tm['t_prepare'], '.3f')} s, guided evaluation {_f(tm['t_eval'], '.4f')} s, student step {_f(tm['t_student'], '.4f')} s, "
          f"vocoder {_f(tm['t_vocoder'], '.4f')} s, audio {_f(tm['audio_seconds'], '.2f')} s); peak GPU memory {_f(tm['peak_mem_mb_max'], '.0f')} MB.", "",
          "## Quality of every output", "", "| Output | Prompts | WER | SIM | UTMOS |", "|---|---|---:|---:|---:|"]
    for o, q in res["quality"].items():
        for gname in ("clean", "noisy"):
            v = q[gname]
            if v["wer"] is not None:
                L.append(f"| {o} | {gname} | {v['wer']:.4f} | {v['similarity']:.4f} | {v['utmos']:.3f} |")
    L += ["", "Added failures (Definition 3.1) of each output against the thesis reference (16-interval Step 8), per condition:", "",
          "| Output | " + " | ".join(CONDS) + " | All |", "|---|" + "---:|" * (len(CONDS) + 1)]
    for o, per in res["fixed_vs_thesis_reference"].items():
        L.append(f"| {o} | " + " | ".join(str(per[c]) for c in CONDS) + f" | {sum(per.values())} |")
    for G, g in GRIDS.items():
        o1 = res["objective1"][G]
        L += ["", f"## Objective 1 on the {g['label']}", "",
              f"### Fixed switching times (Definition 3.1 against {g['ref']}; exact 95% intervals)", "",
              f"| Output | Passes | Clean failures (of {n_clean}) | Noisy failures (of {n_noisy}) | All | 95% interval |", "|---|---:|---:|---:|---:|---|"]
        for o, v in o1["fixed"].items():
            L.append(f"| {LABEL[o]} | {COST[o]} | {v['clean']} | {v['noisy']} | {v['failures']} ({v['failures'] / v['n']:.1%}) | {_ci(v['ci'])} |")
        L += ["", "Per condition: " + "; ".join(f"{o}: " + ", ".join(f"{c} {v['per_condition'][c]}" for c in CONDS) for o, v in o1["fixed"].items()), "",
              "### Cheapest safe candidate per input, in hindsight (the oracle)", "",
              "| Condition | " + " | ".join(g["cands"] + (g["ref"],)) + " | Earlier than the reference |", "|---|" + "---:|" * (len(g["cands"]) + 2)]
        for c, per in o1["oracle_per_condition"].items():
            L.append(f"| {c} | " + " | ".join(str(per[o]) for o in g["cands"] + (g["ref"],)) + f" | {_f(o1['oracle_earlier_share'][c], '.0%')} |")
        L += ["", "### How strongly the signals relate to the oracle choice (Spearman ρ with the oracle's candidate index)", ""]
        L += [f"- {name}: ρ = {_f(v['rho'], '+.3f')} (p {_f(v['p'], '.2g')})" for name, v in o1["association"].items()] or ["- n/a (scipy missing)"]
        L += ["", f"SNR gate by the Objective 1 rule (lowest SNR above which the earliest candidate fails at most {ALPHA:.0%}): "
              f"{_f(o1['theta_rule'], '.0f')} dB." + (f" Plot: `{res['plots'][G]}`." if res["plots"].get(G) else "")]
        if "objective4" not in res or G not in res["objective4"]:
            continue
        o4 = res["objective4"][G]
        L += ["", f"## Objective 4 on the {g['label']} (out of fold; Learn-Then-Test α = {ALPHA}, δ = {DELTA}; {FOLDS}-fold speaker cross-validation)", "",
              f"Saving Δ against {g['ref']} ({COST[g['ref']]} passes) and share of the oracle saving η, with bootstrap 95% intervals over speakers. "
              "Within limit = failure rate ≤ 5%.", "",
              "| Method | Failures | Rate [95% interval] | Within limit | Mean passes | Δ [95%] | η [95%] | Runtime (s) | RTF | Worst condition |",
              "|---|---:|---|---|---:|---|---|---:|---:|---:|"]
        for name, v in o4["methods"].items():
            label = SCHEDULERS.get(name, name)
            if name == "random":
                label = f"Cost-matched random rule (p_ref {v['p_reference']:.2f}; mean failures over draws {v['mean_failures_over_draws']:.1f})"
            L.append(f"| {label} | {v['failures']}/{v['n']} | {v['rate']:.1%} {_ci(v['ci'])} | {'yes' if v['rate'] <= ALPHA else 'no'} | "
                     f"{v['mean_cost']:.2f} | {v['saving']:.1%} {_ci(v.get('saving_ci'))} | {_f(v['eta'], '.0%')} {_ci(v.get('eta_ci'), '.0%')} | "
                     f"{_f(v.get('runtime'), '.2f')} | {_f(v.get('rtf'), '.3f')} | {_f(v['worst_condition_rate'], '.1%')} |")
        L += ["", "Choices per input: " + "; ".join(f"{name}: " + ", ".join(f"{o} {c}" for o, c in v["picks"].items()) for name, v in o4["methods"].items()), "",
              "Thresholds per fold (None = nothing certified; every input to the reference): " +
              "; ".join(f"{name}: [{', '.join(_thr(p) for p in ch)}]" for name, ch in o4["chosen"].items()), "",
              "η above 100% means a method is cheaper than the oracle, which is possible only by adding failures; η is meaningful "
              "for methods within the limit.", "",
              "### Testing the stabilization signal (exact McNemar on paired failures; fixed = failures of the second method avoided by the first)", "",
              "| Comparison | Fixed | New | p |", "|---|---:|---:|---:|"]
        L += [f"| {k} | {v['fixed']} | {v['new']} | {v['p']:.2g} |" for k, v in o4["mcnemar"].items()]
        L += ["", f"### Paired tests against {g['ref']} (speaker-level means; Wilcoxon with Holm correction; TOST margins WER ±{EQUIV['wer']}, SIM ±{EQUIV['similarity']})", "",
              "| Method | Δ passes | Δ WER | p (Holm) | TOST WER p | Δ SIM | p (Holm) | TOST SIM p | Δ UTMOS | p (Holm) |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for name, t in o4["tests"].items():
            L.append(f"| {SCHEDULERS.get(name, name)} | {_f(t['passes']['mean_diff'], '+.2f')} | {_f(t['wer']['mean_diff'], '+.4f')} | "
                     f"{_f(t['wer'].get('wilcoxon_p_holm'), '.2g')} | {_f(t['wer'].get('tost_p'), '.2g')} | {_f(t['similarity']['mean_diff'], '+.4f')} | "
                     f"{_f(t['similarity'].get('wilcoxon_p_holm'), '.2g')} | {_f(t['similarity'].get('tost_p'), '.2g')} | "
                     f"{_f(t['utmos']['mean_diff'], '+.3f')} | {_f(t['utmos'].get('wilcoxon_p_holm'), '.2g')} |")
        L += ["", "TOST p < 0.05 means equivalence within the margins is shown; a non-significant Wilcoxon p alone is not evidence of equal quality.", "",
              "### Robustness: calibrate on clean prompts, test on noisy prompts (speaker-disjoint folds)", "",
              "| Method | Failures | Rate [95% interval] | Mean passes | Δ |", "|---|---:|---|---:|---:|"]
        L += [f"| {SCHEDULERS[name]} | {v['failures']}/{v['n']} | {v['rate']:.1%} {_ci(v['ci'])} | {v['mean_cost']:.2f} | {v['saving']:.1%} |"
              for name, v in o4["clean_to_noisy"].items()]
        o3 = res["objective3"][G]
        L += ["", f"## Objective 3 rehearsal on the {g['label']} ({RESPLITS} random 80/20 speaker re-splits; calibration sizes in cases)", "",
              "| Method | Calibration set | Certified | Mean test risk | 95th percentile | Share of splits with risk > 5% | Mean saving |", "|---|---|---:|---:|---:|---:|---:|"]
        for name, by in o3.items():
            for size, v in by.items():
                L.append(f"| {SCHEDULERS[name]} | {'80% of speakers' if size == 'all' else f'{size} cases'} | {v['certified_share']:.0%} | "
                         f"{v['mean_risk']:.1%} | {v['risk_p95']:.1%} | {v['share_risk_above_alpha']:.0%} | {v['mean_saving']:.1%} |")
    L += ["", "## The paper's settings, teacher-only and student-only against the thesis reference (paired; a − b)", "",
          "| Comparison | Prompts | n | SIM mean diff | SIM higher | SIM p | WER mean diff | WER p |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for name, q in res["paired"].items():
        for gname, v in q.items():
            share = v["share_sim_higher"]
            L.append(f"| {name} | {gname} | {v['n']} | {_f(v['sim_mean_diff'], '+.4f')} | {_f(share, '.0%')} | "
                     f"{_f(v.get('sim_wilcoxon_p'), '.2g')} | {_f(v['wer_mean_diff'], '+.4f')} | {_f(v.get('wer_wilcoxon_p'), '.2g')} |")
    L += ["", "## Deviations from the Chapter 3 protocol", "",
          "- Development speakers only, six conditions per speaker; calibration by cross-validation and re-splits on the same 60 speakers, "
          "so the cases are not independent and the Learn-Then-Test guarantee is not certified here.",
          "- The 32-interval grid is inferred from the paper; its λ grid (0.01 → 1.20, geometric) replaces Chapter 3's 0.20 → 1.20, "
          "because the drift is smaller on a finer grid. Calibration sizes count cases, not speakers.",
          "- `alg2` assumes the student's second time (0.6173); runtime is composed from measured unit times, not timed end to end per method.",
          "- Teacher-only and student-only follow the released code's step counts with the thesis's deterministic teacher (eta = 0)."]
    return "\n".join(L) + "\n"


def write_prereg(out, args):
    path = Path(out) / "preregistration.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    pre = {"purpose": "One development run rehearsing Chapter 3 (Objectives 1, 3 and 4) on the released 16-interval grid and on a "
                      "32-interval grid with the paper's stopping rule, with the paper's 14 + 2 setting, teacher-only and student-only",
           "outputs": LABEL, "teacher_evaluations": EVALS, "student_steps": STUDENT_STEPS, "forward_passes": COST,
           "grid32": "t_i = 1 - cos(pi/2 * i/32), i = 0..32; teacher to k* = max{i : t_i <= s} (Li et al., Algorithm 2)",
           "switching_times": list(SWITCH), "k32": {NAME[s]: K32[s] for s in SWITCH},
           "handover": "teacher endpoint estimate re-noised at the released student's levels: j = 1 for every output except 'paper' "
                       "(j = 2), 'alg2' (from the teacher's state), 'teacher32' (no student) and 'student4' (four student steps from noise); "
                       "guidance gamma = 2, no stochastic term; same seeds",
           "alg2_times": list(ALG2_TIMES), "references": {"g16": REF16, "g32": REF32},
           "failure_rule": "Definition 3.1 with code/scheduling/failure_config.json",
           "objective1": "per candidate: WER, SIM, UTMOS, added-failure rate with Clopper-Pearson 95% intervals, per condition; oracle "
                         "(cheapest non-failing candidate) per condition and its share earlier than the reference; Spearman correlation "
                         "of the prompt SNR and of the drift at each candidate with the oracle's candidate index; SNR gate = lowest "
                         f"value in {THETA_DEV[0]}..{THETA_DEV[-1]} dB above which the earliest candidate fails at most {ALPHA:.0%}",
           "schedulers": SCHEDULERS, "lambda_grids": {"g16": LAM16, "g32": LAM32}, "theta_grid": THETA_GRID,
           "calibration": f"fixed-sequence Learn-Then-Test (Algorithm 3.4), pooled binomial p-value, alpha {ALPHA}, delta {DELTA}",
           "validation": f"{FOLDS}-fold speaker-grouped cross-validation (seed {SEED}); out of fold",
           "objective3": f"{RESPLITS} random 80/20 speaker re-splits; calibration sizes {list(SIZES)} cases; certified share, test risk, saving",
           "objective4": "every method of Table 3 (compared methods) plus SNR + TAS: failures with exact intervals, mean forward passes, "
                         f"saving and share of the oracle saving with {BOOT} speaker bootstrap resamples, runtime and real-time factor, "
                         "worst-condition rate; Wilcoxon signed-rank on speaker-level differences with Holm correction; exact McNemar "
                         f"(full policy against SNR-only, the fixed-later-step controls, stabilization only and the cost-matched random rule); "
                         f"TOST equivalence with margins {EQUIV}; robustness: calibrate on clean prompts, test on noisy prompts",
           "random_rule": "per input, the reference with probability p, else a uniformly random early candidate; p matches the full "
                          f"policy's mean cost; {RESPLITS} seeded draws",
           "tas": "one-sided squared Mahalanobis distance of log(d_1..d_k) from the clean training prompts; k = 4 (16-interval), 7 (32-interval)",
           "criteria": "none; descriptive development evidence",
           "n_speakers": N_SPEAKERS, "conditions": list(CONDS),
           "input_hashes": {str(p.relative_to(THESIS_ROOT)).replace("\\", "/"): rf.sha256_file(p)
                            for p in (rf.T4_GRID, cg.PILOT_GRID, ex.EXP012 / "labels_new.csv", Path(__file__))},
           "thesis_commit": rf.git_head(THESIS_ROOT), "dmo_commit": rf.git_head(args.dmo_root)}
    pre["sha256"] = hashlib.sha256(json.dumps(pre, sort_keys=True).encode("utf-8")).hexdigest()
    pre["created_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    path.write_text(json.dumps(pre, indent=2) + "\n", encoding="utf-8")
    print(f"[OK] preregistration written: {path}", flush=True)
    return pre


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["smoke", "run", "analyze"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--dmo_root", default="/content/Shonget-DMOSpeech2")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--allow_partial", action="store_true")
    ap.add_argument("--any_gpu", action="store_true", help="run on a GPU other than the Tesla T4 (the stored labels will not reproduce)")
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.cmd == "analyze":
        analyze(out, args.allow_partial)
        return
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    pipe = rf.Pipeline(args.dmo_root, args.device)
    if "T4" not in pipe.gpu and not args.any_gpu:
        raise SystemExit(f"This test runs on a Tesla T4 (found {pipe.gpu}): the student's random draws differ between GPU models, "
                         "so the stored T4 labels would not reproduce. Switch the runtime to T4, or pass --any_gpu.")
    if args.cmd == "smoke":
        out = out / "smoke"
        out.mkdir(exist_ok=True)
        its = [it for it in ex.items() if it["utt_id"] not in cg.pilot_utts()][:2]
        repro = run_cases(pipe, its, ("clean", "gauss20"), out, reproduce=True)
        (out / "reproduction.json").write_text(json.dumps(repro, indent=2) + "\n", encoding="utf-8")
        for r in repro:
            print(f"reproduction {r['utt_id'][-8:]} {r['condition']} Step {r['step']}: {r['check']}: max |diff| {r['max_abs_diff']:.2e}", flush=True)
        for r in rf.read_csv(out / "grid32.csv"):
            for k in G16_STEPS:
                if r.get(f"wer_g16_{k}_stored"):
                    print(f"stored check {r['utt_id'][-8:]} {r['condition']} Step {k}: WER {float(r[f'wer_g16_{k}']):.4f} "
                          f"(stored {float(r[f'wer_g16_{k}_stored']):.4f}), SIM {float(r[f'similarity_g16_{k}']):.4f} "
                          f"(stored {float(r[f'similarity_g16_{k}_stored']):.4f})", flush=True)
        print(f"Smoke run complete: {out}")
        return
    write_prereg(out, args)
    run_cases(pipe, ex.items(), CONDS, out)
    analyze(out)


if __name__ == "__main__":
    main()
