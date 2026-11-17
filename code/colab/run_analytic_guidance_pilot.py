"""Pilot of Analytic Unconditional Guidance (EXP-024): development speakers only, a screen before any frozen test.

Idea. Classifier-free guidance runs the teacher twice per step, v = v_c + w (v_c - v_u): v_c sees the text and the voice
prompt, v_u sees neither. In flow matching (x_t = (1 - t) eps + t x1) the best unconditional velocity is
v_u*(x, t) = (E[x1 | x_t = x] - x) / (1 - t). At t = 0 the state is pure noise, independent of x1, so E[x1 | x_0] is the
average mel mu and v_u*(x, 0) = mu - x exactly; for small t, E[x1 | x_t] = mu + O(t). Teacher-guided sampling hands
over to the student at Step 4 (t = 0.0761, the student's restart level) or Step 8 (t = 0.2929, the 0.25 reference), so
the teacher only runs while the signal is weak. Analytic Unconditional Guidance (AUG) replaces the network's
unconditional evaluation at Steps 0..K by the closed form

    v_u = (mu - x_i) / (1 - t_i),

and keeps the conditional evaluation and the guidance strength, so each of those steps costs one pass instead of two:
the Step-4 handover costs 13 - (K + 1) passes and the 0.25 reference 21 - (K + 1), for K <= 4. mu is one value per mel
band: the network's own unconditional estimate at t = 0, averaged over every frame of 20 calibration cases from speakers
the pilot does not use. Nothing is trained.

The pilot measures (1) how far the closed form is from the network's unconditional estimate at Steps 0-8, and at Step 0
how much of that gap is seed-specific (3 further noise draws per case); (2) how much AUG moves the teacher's handover
estimate at Steps 4 and 8, next to how much refining the grid from 16 to 32 intervals moves it; (3) the quality of AUG
with K = 2 and K = 4 at Steps 4 and 8 on 40 development speakers (clean prompts; noisy15 at Step 8 only), against the
stored outputs of the released teacher for the same cases and seeds (EXP-018); (4) pitch variety on 10 speakers x 20
seeds against the stored CVf0 of the same handovers (EXP-020); (5) T4 time of the teacher with and without AUG.
Development evidence only: these speakers designed earlier experiments.

Subcommands: freeze (write the preregistration once), smoke, run (resumable), analyze.
"""
import argparse
import hashlib
import json
import math
import random
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_followup_tests as rf  # noqa: E402
import run_consistency_gate_prelim as cg  # noqa: E402
import run_endpoint_extrapolation_prelim as ex  # noqa: E402
import run_noise_guidance_prelim as ng  # noqa: E402
import run_grid32_prelim as gl  # noqa: E402
import run_selective_handover_prelim as sh  # noqa: E402

THESIS_ROOT = rf.THESIS_ROOT
EXP_DIR = THESIS_ROOT / "research/experiments/EXP-024-analytic-unconditional-guidance-pilot"
PREREG_PATH = EXP_DIR / "preregistration.json"
POWER_JSON = EXP_DIR / "power_check.json"
STEPS_CSV = sh.STEPS_CSV
EXP020_DIV = THESIS_ROOT / "research/experiments/EXP-020-selective-handover-prelim/results/diversity.csv"
FAILURE_CONFIG = THESIS_ROOT / "code/scheduling/failure_config.json"
MANIFEST = rf.SPLITS_DIR / "development_manifest.json"
DMO_COMMIT = "8a7868fc3345702a586173447f4869205b501f26"
DEPENDENCIES = ("run_analytic_guidance_pilot.py", "run_selective_handover_prelim.py", "run_grid32_prelim.py",
                "run_noise_guidance_prelim.py", "run_followup_tests.py", "run_consistency_gate_prelim.py",
                "run_endpoint_extrapolation_prelim.py")

KS = (0, 2, 4)                        # AUG arms in the diagnostic: the closed form at Steps 0..K
Q_KS = (2, 4)                         # arms scored for quality and pitch variety
HANDOVER = (4, 8)                     # the Step-4 handover (13 passes) and the 0.25 reference (21 passes)
KMAX = max(HANDOVER)
GRID_K = {4: 8, 8: 16}                # the same sway time on the 32-interval grid
METRICS = ("wer", "similarity", "utmos")
PARTS = ("all", "slow")               # whole estimate; slow part (9-frame prosody band, what reaches the student)
N_PILOT, CONDS = 40, ("clean", "noisy15")
CAL_RANGE = (40, 60)                  # calibration speakers: Quick-set positions 41-60, disjoint from the pilot's 1-40
LEAK_DRAWS, LEAK_SEED_BASE = 3, 3000  # further noise draws at Step 0 (seed + 3000 + j)
DIV_N, DIV_SEEDS, DIV_SEED_BASE = 10, 20, 1000   # EXP-020's diversity seeds, so its stored CVf0 is comparable
TIMING_N, TIMING_REPEATS = 10, 3
ARMS = ("std", "K2", "K4")            # timing arms
BOOT_B, BOOT_SEED = 2000, 2026
TEACHER_TOLERANCE = 1e-4
CVF0_REPRO_TOLERANCE = 1e-6
Q_OUTS = {"clean": ("W4_K2", "W4_K4", "W8_K2", "W8_K4"), "noisy15": ("W8_K2", "W8_K4")}
DIV_OUTS = Q_OUTS["clean"]
# Bounds on the paired difference from the released teacher's output of the same case and seed (speaker bootstrap, 95%).
# Sized before the run by code/analysis/exp024_power_check.py: with 40 speakers a change as small as refining the grid
# passes the clean bounds every time; at Step 8 the clean WER bound +0.01 would fail it 67% of the time, +0.015 never.
BOUNDS = {("W4", "clean"): {"wer": 0.01, "similarity": 0.01, "utmos": 0.10},
          ("W8", "clean"): {"wer": 0.015, "similarity": 0.01, "utmos": 0.10},
          ("W8", "noisy15"): {"wer": 0.025, "similarity": 0.01, "utmos": 0.10}}
CVF0_BOUND = 0.01

FIELDS = ["utt_id", "speaker_id", "condition", "prompt_snr", "gpu", "prompt_len", "gen_len"] + \
         [f"e_{i}" for i in range(KMAX + 1)] + ["gap0_total", "gap0_seed", "gap0_position"] + \
         [f"y_{p}_{k}" for p in PARTS for k in HANDOVER] + \
         [f"d_{p}_K{K}_{k}" for p in PARTS for K in KS for k in HANDOVER] + \
         [f"{m}_{o}" for o in Q_OUTS["clean"] for m in METRICS + ("hypothesis",)] + ["case_seconds", "utc"]
SMOKE_FIELDS = ["teacher_loop_max_abs_diff", "aug_off_max_abs_diff"] + \
               [f"{m}_W{k}" for k in HANDOVER for m in METRICS + ("hypothesis",)]
DIV_FIELDS = ["utt_id", "speaker_id", "condition", "n_seeds"] + [f"cvf0_{o}" for o in DIV_OUTS] + \
             [f"voiced_{o}" for o in DIV_OUTS] + ["cvf0_W4", "voiced_W4", "case_seconds", "utc"]
TIMING_FIELDS = ["utt_id", "speaker_id", "condition", "gpu", "prompt_len", "gen_len", "repeats"] + \
                [f"t_{a}_{k}" for a in ARMS for k in HANDOVER] + ["t_student_decode", "case_seconds", "utc"]


# ------------------------------------------------------------------------------- design
def parse(name):
    """'W8_K4' -> (8, 4); 'W4' -> (4, -1), the released teacher."""
    head, _, tail = name.partition("_")
    return int(head[1:]), (int(tail[1:]) if tail else -1)


def cost(name):
    """Forward passes: one teacher evaluation per step at Steps 0..k, two with the network's unconditional branch, one
    where AUG uses the closed form (Steps 0..K), then the student's 3 passes."""
    k, K = parse(name)
    return 2 * (k + 1) - (min(K, k) + 1 if K >= 0 else 0) + 3


def pilot_items():
    return ex.items()[:N_PILOT]


def calibration_items():
    return ex.items()[CAL_RANGE[0]:CAL_RANGE[1]]


# ------------------------------------------------------------------------------- math
def _norm(x):
    return float((x * x).sum()) ** 0.5


def _maxabs(x):
    return float(abs(x).max())


def _gen(x, plen):
    return x[0, plen:, :]


def _band_sum(u):
    """Sum over frames of an array shaped (frames, bands), as a list of floats."""
    s = u.sum(dim=0) if hasattr(u, "clone") else u.sum(axis=0)
    s = s.double().cpu().numpy() if hasattr(s, "cpu") else s
    return [float(v) for v in s]


def as_mu(pipe, mu):
    """mu as an array shaped (1, 1, bands), broadcast over frames."""
    t = pipe.torch.tensor(mu, device=pipe.device, dtype=pipe.torch.float32)
    if hasattr(t, "view"):
        return t.view(1, 1, -1)
    import numpy as np
    return np.asarray(t, dtype=float).reshape(1, 1, -1)


def guided_loop(pipe, ctx, kmax, K=-1, mu=None, grid=None, keep=None, record=False):
    """The released guided Euler teacher (sampler.run_teacher_single_pass, CFG_STRENGTH) on `grid` (default: the released
    16-interval sway grid). At steps i <= K the unconditional evaluation is replaced by the closed form
    v_u = (mu - x_i) / (1 - t_i) (AUG); K = -1 is the released teacher. Returns ({i: T_i} for i in keep (default all),
    forward passes, {i: (T_c, T_u)} for every step whose unconditional branch the network evaluated, if record)."""
    torch, sampler = pipe.torch, pipe.sampler
    grid = sampler.SWAY_TEACHER_GRID if grid is None else grid
    keep = set(range(kmax + 1)) if keep is None else set(keep)
    x, ends, rec, passes = sh._clone(ctx["y0"]), {}, {}, 0
    with torch.inference_mode():
        for i in range(kmax + 1):
            t = torch.tensor(grid[i], device=pipe.device, dtype=torch.float32)
            kw = dict(x=x, cond=ctx["step_cond"], text=ctx["text_ids"], time=t, mask=None)
            vc = pipe.tts.teacher(drop_audio_cond=False, drop_text=False, **kw)
            passes += 1
            if i <= K:
                vu = (mu - x) / (1.0 - grid[i])
            else:
                vu = pipe.tts.teacher(drop_audio_cond=True, drop_text=True, **kw)
                passes += 1
                if record:
                    rec[i] = (x + (1.0 - grid[i]) * vc, x + (1.0 - grid[i]) * vu)
            v = vc + (vc - vu) * sampler.CFG_STRENGTH
            if i in keep:
                ends[i] = x + (1.0 - grid[i]) * v
            if i < kmax:
                x = x + (grid[i + 1] - grid[i]) * v
    return ends, passes, rec


def unconditional_endpoint(pipe, ctx, x):
    """The network's unconditional endpoint estimate at t = 0 for the state x: x + v_u(x, 0)."""
    torch = pipe.torch
    with torch.inference_mode():
        t = torch.tensor(0.0, device=pipe.device, dtype=torch.float32)
        return x + pipe.tts.teacher(x=x, cond=ctx["step_cond"], text=ctx["text_ids"], time=t, mask=None,
                                    drop_audio_cond=True, drop_text=True)


def closed_form_gap(mu, Tc, Tu, plen):
    """Generated frames: distance between the closed form and the network's unconditional estimate, in units of the
    guidance signal at that step (the distance between the conditional and unconditional estimates)."""
    gu = _gen(Tu, plen)
    return _norm(mu[0] - gu) / (_norm(_gen(Tc, plen) - gu) + 1e-8)


def step0_split(Tc0, Tu0, others, mu, plen):
    """Splits the Step-0 gap: the network's unconditional estimate for other noise draws of the same case (`others`)
    averages out what is seed-specific. Returns (total, seed-specific, position) gaps in units of the Step-0 guidance
    signal. With 3 draws the seed-specific part is overstated by about sqrt(1 + 1/3) (the average is itself noisy)."""
    mean = others[0]
    for o in others[1:]:
        mean = mean + o
    mean = mean * (1.0 / len(others))
    gu = _gen(Tu0, plen)
    sig = _norm(_gen(Tc0, plen) - gu) + 1e-8
    return (_norm(mu[0] - gu) / sig, _norm(gu - _gen(mean, plen)) / sig, _norm(_gen(mean, plen) - mu[0]) / sig)


def rel_change(a, b, plen, part="all"):
    """||a - b|| / ||b|| on generated frames; part 'slow' compares the 9-frame prosody bands (sh.decompose)."""
    ga, gb = _gen(a, plen), _gen(b, plen)
    if part == "slow":
        ga, gb = sh.decompose(ga)[1], sh.decompose(gb)[1]
    return _norm(ga - gb) / (_norm(gb) + 1e-8)


def extra_noise(pipe, ctx, j):
    torch, sampler = pipe.torch, pipe.sampler
    gen = sampler.make_local_generator((int(ctx["seed"]) + LEAK_SEED_BASE + j) % (2 ** 31 - 1), device=pipe.device)
    return torch.randn(ctx["y0"].shape, dtype=torch.float32, device=pipe.device, generator=gen)


# --------------------------------------------------------------------------------- runs
def calibrate(pipe, items, out_dir):
    """mu: per mel band, the network's unconditional endpoint estimate at t = 0 for each case's own initial noise,
    pooled over every frame of the calibration cases (clean prompts; the unconditional branch never sees the prompt).
    Written once to calibration.json and reused on resume."""
    path = Path(out_dir) / "calibration.json"
    if path.exists():
        cal = json.loads(path.read_text(encoding="utf-8"))
        if cal["cases"] == [it["utt_id"] for it in items]:
            print(f"[calibration] reusing {path.name} ({len(cal['cases'])} cases)", flush=True)
            return cal["mu"]
        raise SystemExit(f"{path} was written for other calibration cases; use a new --out folder")
    total, frames, per_case = None, 0, []
    for n, item in enumerate(items, 1):
        ctx = pipe.prepare(item, "clean", item["input_seed"], out_dir)
        u = unconditional_endpoint(pipe, ctx, ctx["y0"])[0]
        s, f = _band_sum(u), int(u.shape[0])
        total = s if total is None else [a + b for a, b in zip(total, s)]
        frames += f
        per_case.append({"utt_id": item["utt_id"], "frames": f, "band_mean": [v / f for v in s]})
        print(f"[{n}/{len(items)}] calibration: {item['utt_id']} ({f} frames)", flush=True)
    mu = [v / frames for v in total]
    spread = [(sum((a - b) ** 2 for a, b in zip(c["band_mean"], mu)) / len(mu)) ** 0.5 for c in per_case]
    cal = {"mu": mu, "cases": [it["utt_id"] for it in items], "frames": frames, "per_case": per_case,
           "case_rms_from_mu": {"median": sorted(spread)[len(spread) // 2], "max": max(spread)}, "gpu": pipe.gpu, "utc": sh._utc()}
    path.write_text(json.dumps(cal, indent=1) + "\n", encoding="utf-8")
    print(f"[OK] calibration: mu over {frames} frames of {len(items)} cases -> {path}", flush=True)
    return mu


def stored_w48():
    """(utt_id, condition) -> stored scores of the released teacher's Step-4 and Step-8 handovers (EXP-018)."""
    return {(r["utt_id"], r["condition"]): {f"W{k}": {x: float(r[f"{x}_g16_{k}"]) for x in METRICS} for k in HANDOVER}
            for r in rf.read_csv(STEPS_CSV)}


def run_cases(pipe, items, conds, mu, out_dir, csv_name="pilot.csv", smoke=False, stored=None):
    """Resumable, one row per (utterance, condition): diagnostics of the closed form and of AUG's handover estimates,
    and the scored AUG outputs. The smoke run also checks this runner's teacher against the released one and scores the
    released Step-4 and Step-8 handovers for reproduction of the stored labels."""
    path = Path(out_dir) / csv_name
    fields = FIELDS + (SMOKE_FIELDS if smoke else [])
    done = rf.done_keys(path, 1)
    todo = [(it, c) for it in items for c in conds if (it["utt_id"], c) not in done]
    print(f"[cases] {len(todo)} cases to run ({len(done)} already in {path.name}) on {pipe.gpu}", flush=True)
    taus, mu_t, repro, checks = list(pipe.sampler.STUDENT_TAUS), as_mu(pipe, mu), [], []
    for n, (item, cond) in enumerate(todo, 1):
        t0 = time.perf_counter()
        ctx = pipe.prepare(item, cond, item["input_seed"], out_dir)
        plen = ctx["prompt_len"]
        E, _, rec = guided_loop(pipe, ctx, KMAX, keep=HANDOVER, record=True)
        A = {K: guided_loop(pipe, ctx, KMAX, K=K, mu=mu_t, keep=HANDOVER)[0] for K in KS}
        G32 = guided_loop(pipe, ctx, GRID_K[KMAX], grid=gl.GRID32, keep=set(GRID_K.values()))[0]
        others = [unconditional_endpoint(pipe, ctx, extra_noise(pipe, ctx, j)) for j in range(LEAK_DRAWS)]
        row = {"utt_id": item["utt_id"], "speaker_id": item["speaker_id"], "condition": cond, "prompt_snr": ctx["prompt_snr"],
               "gpu": pipe.gpu, "prompt_len": plen, "gen_len": int(ctx["y0"].shape[1]) - plen}
        with pipe.torch.inference_mode():
            for i in range(KMAX + 1):
                row[f"e_{i}"] = closed_form_gap(mu_t, rec[i][0], rec[i][1], plen)
            row["gap0_total"], row["gap0_seed"], row["gap0_position"] = step0_split(rec[0][0], rec[0][1], others, mu_t, plen)
            for k in HANDOVER:
                for p in PARTS:
                    row[f"y_{p}_{k}"] = rel_change(G32[GRID_K[k]], E[k], plen, p)
                    for K in KS:
                        row[f"d_{p}_K{K}_{k}"] = rel_change(A[K][k], E[k], plen, p)
        del rec, G32, others
        gt = pipe.ground_truth(item)
        for o in Q_OUTS[cond]:
            k, K = parse(o)
            m = pipe.score(ng.decode(pipe, ctx, gl.student(pipe, ctx, A[K][k], taus)), item, gt)
            row.update({f"{x}_{o}": m[x] for x in METRICS + ("hypothesis",)})
        if smoke:
            released, _ = pipe.teacher(ctx, HANDOVER)
            off = guided_loop(pipe, ctx, KMAX, K=-1, mu=mu_t, keep=HANDOVER)[0]
            with pipe.torch.inference_mode():
                row["teacher_loop_max_abs_diff"] = max(_maxabs(released[k] - E[k]) for k in HANDOVER)
                row["aug_off_max_abs_diff"] = max(_maxabs(off[k] - E[k]) for k in HANDOVER)
            checks.append((row["teacher_loop_max_abs_diff"], row["aug_off_max_abs_diff"]))
            key = (item["utt_id"], cond)
            for k in HANDOVER:
                m = pipe.score(ng.decode(pipe, ctx, gl.student(pipe, ctx, E[k], taus)), item, gt)
                row.update({f"{x}_W{k}": m[x] for x in METRICS + ("hypothesis",)})
                if stored is not None and key in stored:
                    s = stored[key][f"W{k}"]
                    repro.append({"utt_id": item["utt_id"], "condition": cond, "output": f"W{k}", "wer": m["wer"],
                                  "wer_stored": s["wer"], "similarity": m["similarity"], "similarity_stored": s["similarity"]})
        row.update({"case_seconds": round(time.perf_counter() - t0, 2), "utc": sh._utc()})
        sh._append(path, fields, row)
        print(f"[{n}/{len(todo)}] cases: {item['utt_id']} {cond} ({row['case_seconds']:.0f} s)", flush=True)
    print(f"[OK] cases complete: {path}", flush=True)
    return repro, checks


def run_diversity(pipe, items, mu, out_dir, seeds=DIV_SEEDS, csv_name="diversity.csv", reproduce_first=True):
    """CVf0 across seeds of the four AUG handovers with EXP-020's seeds (input_seed + 1000 + m) and fixed prompt
    degradation, so EXP-020's stored CVf0 of the released Step-4 and Step-8 handovers is comparable. For the first
    speaker the released Step-4 handover is synthesized too, to check that its CVf0 reproduces EXP-020's."""
    path = Path(out_dir) / csv_name
    done = rf.done_keys(path, 1)
    order = [it["utt_id"] for it in items]
    todo = [it for it in items if (it["utt_id"], "clean") not in done]
    print(f"[diversity] {len(todo)} speakers x {seeds} seeds to run ({len(done)} already in {path.name})", flush=True)
    taus, mu_t = list(pipe.sampler.STUDENT_TAUS), as_mu(pipe, mu)
    for n, item in enumerate(todo, 1):
        t0 = time.perf_counter()
        reproduce = reproduce_first and order.index(item["utt_id"]) == 0
        tracks = {o: [] for o in DIV_OUTS + (("W4",) if reproduce else ())}
        for m in range(seeds):
            ctx = pipe.prepare(item, "clean", int(item["input_seed"]) + DIV_SEED_BASE + m, out_dir, noise_seed=item["input_seed"])
            A = {K: guided_loop(pipe, ctx, KMAX, K=K, mu=mu_t, keep=HANDOVER)[0] for K in Q_KS}
            for o in DIV_OUTS:
                k, K = parse(o)
                tracks[o].append(sh.f0_track(ng.decode(pipe, ctx, gl.student(pipe, ctx, A[K][k], taus))))
            if reproduce:
                E = guided_loop(pipe, ctx, 4, keep=(4,))[0]
                tracks["W4"].append(sh.f0_track(ng.decode(pipe, ctx, gl.student(pipe, ctx, E[4], taus))))
        row = {"utt_id": item["utt_id"], "speaker_id": item["speaker_id"], "condition": "clean", "n_seeds": seeds,
               "cvf0_W4": "", "voiced_W4": ""}
        for o in tracks:
            row[f"cvf0_{o}"], row[f"voiced_{o}"] = sh.cvf0(tracks[o])
        row.update({"case_seconds": round(time.perf_counter() - t0, 2), "utc": sh._utc()})
        sh._append(path, DIV_FIELDS, row)
        print(f"[{n}/{len(todo)}] diversity: {item['utt_id']} ({row['case_seconds']:.0f} s)", flush=True)
    print(f"[OK] diversity complete: {path}", flush=True)


def run_timing(pipe, items, mu, out_dir, repeats=TIMING_REPEATS, csv_name="timing.csv"):
    """T4 wall-clock of the teacher to Step 4 and to Step 8: released (std), AUG K = 2 and K = 4, between CUDA
    synchronizations, the order alternating across repeats after one warm-up case; medians per case. The student and
    vocoder are timed once per case (they do not depend on the arm)."""
    path = Path(out_dir) / csv_name
    done = rf.done_keys(path, 1)
    todo = [it for it in items if (it["utt_id"], "clean") not in done]
    print(f"[timing] {len(todo)} cases to time ({len(done)} already in {path.name})", flush=True)
    taus, mu_t = list(pipe.sampler.STUDENT_TAUS), as_mu(pipe, mu)
    if todo:
        warm = pipe.prepare(todo[0], "clean", todo[0]["input_seed"], out_dir)
        for K in (-1, 4):
            guided_loop(pipe, warm, KMAX, K=K, mu=mu_t, keep=(KMAX,))
    for n, item in enumerate(todo, 1):
        t0 = time.perf_counter()
        ctx = pipe.prepare(item, "clean", item["input_seed"], out_dir)
        times = {(a, k): [] for a in ARMS for k in HANDOVER}
        for r in range(repeats):
            for a in (ARMS if r % 2 == 0 else ARMS[::-1]):
                K = -1 if a == "std" else int(a[1:])
                for k in HANDOVER:
                    _, sec = gl.timed(pipe, lambda: guided_loop(pipe, ctx, k, K=K, mu=mu_t, keep=(k,)))
                    times[(a, k)].append(sec)
        E = guided_loop(pipe, ctx, 4, keep=(4,))[0]
        _, t_sd = gl.timed(pipe, lambda: ng.decode(pipe, ctx, gl.student(pipe, ctx, E[4], taus)))
        med = lambda v: sorted(v)[len(v) // 2]
        row = {"utt_id": item["utt_id"], "speaker_id": item["speaker_id"], "condition": "clean", "gpu": pipe.gpu,
               "prompt_len": ctx["prompt_len"], "gen_len": int(ctx["y0"].shape[1]) - ctx["prompt_len"], "repeats": repeats,
               **{f"t_{a}_{k}": round(med(times[(a, k)]), 5) for a in ARMS for k in HANDOVER},
               "t_student_decode": round(t_sd, 5), "case_seconds": round(time.perf_counter() - t0, 2), "utc": sh._utc()}
        sh._append(path, TIMING_FIELDS, row)
        print(f"[{n}/{len(todo)}] timing: {item['utt_id']} ({row['case_seconds']:.0f} s)", flush=True)
    print(f"[OK] timing complete: {path}", flush=True)


# ----------------------------------------------------------------------------- analysis
def load_stored():
    """(utt, cond) -> speaker and the stored W4 and W8 scores of the released teacher (EXP-018)."""
    return {(r["utt_id"], r["condition"]): {"speaker_id": r["speaker_id"],
                                            "m": {f"W{k}": {x: float(r[f"{x}_g16_{k}"]) for x in METRICS} for k in HANDOVER}}
            for r in rf.read_csv(STEPS_CSV)}


def stored_cvf0():
    """speaker -> stored clean-prompt CVf0 of the released Step-4 (plain_4) and Step-8 (plain_8) handovers, EXP-020."""
    return {r["speaker_id"]: {"W4": float(r["cvf0_plain_4"]), "W8": float(r["cvf0_plain_8"])}
            for r in rf.read_csv(EXP020_DIV) if r["condition"] == "clean"}


def _num(v):
    return None if v in (None, "") else float(v)


def build_table(stored, rows):
    table = []
    for r in rows:
        s = stored.get((r["utt_id"], r["condition"]))
        if s is None:
            continue
        m = dict(s["m"])
        for o in Q_OUTS[r["condition"]]:
            if _num(r.get(f"wer_{o}")) is not None:
                m[o] = {x: float(r[f"{x}_{o}"]) for x in METRICS}
        table.append({"utt": r["utt_id"], "spk": s["speaker_id"], "cond": r["condition"], "m": m, "row": r})
    return table


def boot(groups, B=BOOT_B, seed=BOOT_SEED):
    values = [sum(v) / len(v) for v in groups.values() if v]
    if not values:
        return {"mean": None, "lo": None, "hi": None, "n": 0}
    rng = random.Random(seed)
    boots = sorted(sum(rng.choice(values) for _ in values) / len(values) for _ in range(B))
    return {"mean": sum(values) / len(values), "lo": boots[int(0.025 * B)], "hi": boots[int(0.975 * B)], "n": len(values)}


def _capped(m, name, metric):
    v = m[name][metric]
    return min(1.0, v) if metric == "wer" else v


def paired(table, a, b, metric, cond):
    g = defaultdict(list)
    for r in table:
        if r["cond"] == cond and a in r["m"] and b in r["m"]:
            g[r["spk"]].append(_capped(r["m"], a, metric) - _capped(r["m"], b, metric))
    return boot(g)


def _median(values):
    v = sorted(x for x in values if x is not None and not math.isnan(x))
    return None if not v else (v[len(v) // 2] if len(v) % 2 else (v[len(v) // 2 - 1] + v[len(v) // 2]) / 2)


def quality_summary(table, th):
    out = {}
    for cond, outs in Q_OUTS.items():
        for o in outs:
            k, _ = parse(o)
            base = f"W{k}"
            q = {m: paired(table, o, base, m, cond) for m in METRICS}
            rows = [r for r in table if r["cond"] == cond and o in r["m"]]
            q["failures_vs_released"] = sum(bool(gl.failed(r["m"][o], r["m"][base], th)) for r in rows)
            q["failures_vs_reference"] = sum(bool(gl.failed(r["m"][o], r["m"]["W8"], th)) for r in rows)
            q["released_failures_vs_reference"] = sum(bool(gl.failed(r["m"][base], r["m"]["W8"], th)) for r in rows) if base != "W8" else 0
            q["n"] = len(rows)
            b = BOUNDS[(base, cond)]
            ok = q["wer"]["hi"] is not None and q["wer"]["hi"] <= b["wer"] and q["similarity"]["lo"] >= -b["similarity"] \
                and q["utmos"]["lo"] >= -b["utmos"]
            q["passes_bounds"] = None if q["wer"]["hi"] is None else bool(ok)
            q["bounds"] = b
            out[f"{o}|{cond}"] = q
    return out


def diagnostic_summary(rows):
    out = {}
    for cond in CONDS:
        rs = [r for r in rows if r["condition"] == cond]
        med = lambda f: _median([_num(r[f]) for r in rs])
        d = {"cases": len(rs), "closed_form_gap": {i: med(f"e_{i}") for i in range(KMAX + 1)},
             "step0": {p: med(f"gap0_{p}") for p in ("total", "seed", "position")}, "change": {}}
        for k in HANDOVER:
            for p in PARTS:
                y = med(f"y_{p}_{k}")
                d["change"][f"{p}_{k}"] = {"grid_32": y, **{f"K{K}": med(f"d_{p}_K{K}_{k}") for K in KS},
                                           **{f"K{K}_over_grid": _median([_num(r[f"d_{p}_K{K}_{k}"]) / max(_num(r[f"y_{p}_{k}"]), 1e-12)
                                                                         for r in rs]) for K in KS}}
        out[cond] = d
    return out


def diversity_summary(div_rows, stored):
    out = {}
    for o in DIV_OUTS:
        k, _ = parse(o)
        diffs, vals = {}, []
        for r in div_rows:
            s, x = r["speaker_id"], _num(r[f"cvf0_{o}"])
            if s not in stored or x is None or math.isnan(x):
                continue
            vals.append(x)
            diffs[s] = [x - stored[s][f"W{k}"]]
        out[o] = {"mean": (sum(vals) / len(vals)) if vals else None, "vs_released": boot(diffs)}
    rep = [(abs(_num(r["cvf0_W4"]) - stored[r["speaker_id"]]["W4"]) if r["speaker_id"] in stored else None)
           for r in div_rows if _num(r.get("cvf0_W4")) is not None]
    out["released_W4_reproduction_max_abs"] = max(rep) if rep else None
    return out


def timing_summary(t_rows):
    out = {"cases": len(t_rows)}
    if not t_rows:
        return out
    med = lambda f: _median([_num(r[f]) for r in t_rows])
    for k in HANDOVER:
        std = med(f"t_std_{k}")
        out[f"step{k}"] = {"std_s": std, **{f"{a}_s": med(f"t_{a}_{k}") for a in ARMS[1:]},
                           **{f"{a}_ratio": _median([_num(r[f"t_{a}_{k}"]) / _num(r[f"t_std_{k}"]) for r in t_rows]) for a in ARMS[1:]},
                           **{f"{a}_pass_ratio": (cost(f"W{k}_{a}") - 3) / (cost(f"W{k}") - 3) for a in ARMS[1:]}}
    out["student_decode_s"] = med("t_student_decode")
    return out


def analyze(out, allow_partial=False, prereg_path=PREREG_PATH):
    out = Path(out)
    pre = check_prereg(prereg_path)
    th = gl.thresholds()
    read = lambda name: rf.read_csv(out / name, repair=True) if (out / name).exists() else []
    rows, div_rows, t_rows = read("pilot.csv"), read("diversity.csv"), read("timing.csv")
    table = build_table(load_stored(), rows)
    expected = {"cases": N_PILOT * len(CONDS), "diversity_speakers": DIV_N, "timing_cases": TIMING_N}
    have = {"cases": len(table), "diversity_speakers": len(div_rows), "timing_cases": len(t_rows)}
    partial = any(have[k] < expected[k] for k in expected)
    if partial and not allow_partial:
        raise SystemExit(f"incomplete run {have} of {expected}; finish the run or pass --allow_partial")
    cal = json.loads((out / "calibration.json").read_text(encoding="utf-8")) if (out / "calibration.json").exists() else None
    res = {"prereg_sha256": pre["sha256"], "have": have, "expected": expected, "partial": partial,
           "passes": {o: cost(o) for o in ("W4", "W8") + tuple(f"W{k}_K{K}" for k in HANDOVER for K in KS)},
           "calibration": None if cal is None else {"cases": len(cal["cases"]), "frames": cal["frames"],
                                                     "case_rms_from_mu": cal["case_rms_from_mu"]},
           "quality": quality_summary(table, th), "diagnostic": diagnostic_summary(rows),
           "diversity": diversity_summary(div_rows, stored_cvf0()), "timing": timing_summary(t_rows)}
    complete_q, complete_d = have["cases"] >= expected["cases"], have["diversity_speakers"] >= expected["diversity_speakers"]
    crit = {}
    for K in Q_KS:
        qc = [res["quality"][f"W{k}_K{K}|clean"]["passes_bounds"] for k in HANDOVER]
        qn = res["quality"][f"W8_K{K}|noisy15"]["passes_bounds"]
        pv = [res["diversity"][f"W{k}_K{K}"]["vs_released"]["mean"] for k in HANDOVER]
        crit[f"K{K}"] = {"quality_clean": None if not complete_q or None in qc else all(qc),
                         "quality_noisy_reference": None if not complete_q or qn is None else qn,
                         "pitch_variety": None if not complete_d or None in pv else all(v >= -CVF0_BOUND for v in pv)}
    res["criteria"] = crit
    advancing = [K for K in Q_KS if crit[f"K{K}"]["quality_clean"] and crit[f"K{K}"]["pitch_variety"]]
    res["advance"] = bool(advancing)
    res["carried_K"] = max(advancing) if advancing else None
    res["applies_to_noisy_reference"] = None if res["carried_K"] is None else crit[f"K{res['carried_K']}"]["quality_noisy_reference"]
    (out / "aug_results.json").write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
    (out / "AUG_REPORT.md").write_text(render(res), encoding="utf-8")
    print(f"Wrote {out / 'AUG_REPORT.md'}", flush=True)
    return res


def _word(v):
    return "n/a" if v is None else ("PASS" if v else "FAIL")


def _ci(v, fmt="{:+.4f}"):
    return "n/a" if not v or v.get("mean") is None else f"{fmt.format(v['mean'])} [{fmt.format(v['lo'])}, {fmt.format(v['hi'])}]"


def _f(v, fmt="{:.3f}"):
    return "n/a" if v is None else fmt.format(v)


def render(res):
    c = res["criteria"]
    L = ["# Pilot: Analytic Unconditional Guidance (EXP-024)", "",
         f"{'**PARTIAL.** ' if res['partial'] else ''}Development speakers only; {res['have']} of {res['expected']}; "
         f"preregistration `{res['prereg_sha256'][:16]}…`. Every number is computed from the CSV files by "
         "`run_analytic_guidance_pilot.py analyze`. A pilot decides whether a frozen test on new speakers is worth running; "
         "it cannot support a claim.", "",
         "AUG replaces the teacher's unconditional network evaluation at Steps 0..K by the closed form "
         "v_u = (mu - x) / (1 - t), so each of those steps costs one pass instead of two.", "",
         "## Go / no-go", ""]
    for K in Q_KS:
        v = c[f"K{K}"]
        L.append(f"- K = {K} (Step 4: {res['passes'][f'W4_K{K}']} passes instead of 13; reference: {res['passes'][f'W8_K{K}']} instead of 21): "
                 f"clean quality **{_word(v['quality_clean'])}**, pitch variety **{_word(v['pitch_variety'])}**, "
                 f"noisy-prompt reference **{_word(v['quality_noisy_reference'])}**")
    L += [f"- **Advance to a frozen test on new speakers: {'YES, K = ' + str(res['carried_K']) if res['advance'] else 'NO'}**"
          + ("" if not res["advance"] else f"; also for noisy-prompt teacher runs: {_word(res['applies_to_noisy_reference'])}"), "",
          "## Quality against the released teacher's output of the same case and seed (speaker bootstrap, 95%)", "",
          "| Output | Prompts | Passes | WER | Similarity | UTMOS | Bounds (WER / SIM / UTMOS) | Within bounds | Failures vs released | Failures vs reference |",
          "|---|---|---:|---|---|---|---|---|---:|---:|"]
    for key, q in res["quality"].items():
        o, cond = key.split("|")
        b = q["bounds"]
        L.append(f"| {o} | {cond} | {res['passes'][o]} | {_ci(q['wer'])} | {_ci(q['similarity'])} | {_ci(q['utmos'], '{:+.3f}')} | "
                 f"+{b['wer']} / -{b['similarity']} / -{b['utmos']} | {_word(q['passes_bounds'])} | {q['failures_vs_released']}/{q['n']} | "
                 f"{q['failures_vs_reference']}/{q['n']} |")
    L += ["", "## Pitch variety, clean prompts (CVf0 minus EXP-020's stored CVf0 of the released handover at the same step)", "",
          "| Output | CVf0 | minus released |", "|---|---:|---|"]
    for o in DIV_OUTS:
        d = res["diversity"][o]
        L.append(f"| {o} | {_f(d['mean'], '{:.4f}')} | {_ci(d['vs_released'])} |")
    L += ["", f"Released Step-4 handover re-synthesized for the first speaker: CVf0 against EXP-020's stored value, max |difference| "
          f"{_f(res['diversity']['released_W4_reproduction_max_abs'], '{:.2e}')}.", "",
          "## Diagnostic: how far the closed form is from the network (medians)", "",
          "Gap: distance between the closed form and the network's unconditional estimate, in units of the guidance signal at that step.", "",
          "| Prompts | " + " | ".join(f"Step {i}" for i in range(KMAX + 1)) + " | Step 0: seed-specific | Step 0: position |",
          "|---|" + "---:|" * (KMAX + 3)]
    for cond, d in res["diagnostic"].items():
        L.append(f"| {cond} | " + " | ".join(_f(d['closed_form_gap'][i]) for i in range(KMAX + 1)) +
                 f" | {_f(d['step0']['seed'])} | {_f(d['step0']['position'])} |")
    L += ["", "Change of the teacher's handover estimate (relative, generated frames), next to refining the grid from 16 to 32 intervals:", "",
          "| Prompts | Estimate | Grid 16 -> 32 | " + " | ".join(f"AUG K = {K}" for K in KS) + " | " + " | ".join(f"K = {K} / grid" for K in KS) + " |",
          "|---|---|---:|" + "---:|" * (2 * len(KS))]
    for cond, d in res["diagnostic"].items():
        for key, v in d["change"].items():
            L.append(f"| {cond} | {key} | {_f(v['grid_32'], '{:.4f}')} | " + " | ".join(_f(v[f'K{K}'], '{:.4f}') for K in KS) + " | " +
                     " | ".join(_f(v[f'K{K}_over_grid'], '{:.2f}') for K in KS) + " |")
    t = res["timing"]
    L += ["", f"## T4 time of the teacher ({t['cases']} clean cases, medians)", ""]
    if t["cases"]:
        L += ["| Handover | Released (s) | K = 2 (s) | K = 4 (s) | K = 2 / released | K = 4 / released | Pass ratio K = 2 | Pass ratio K = 4 |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for k in HANDOVER:
            v = t[f"step{k}"]
            L.append(f"| Step {k} | {_f(v['std_s'], '{:.3f}')} | {_f(v['K2_s'], '{:.3f}')} | {_f(v['K4_s'], '{:.3f}')} | {_f(v['K2_ratio'], '{:.2f}')} | "
                     f"{_f(v['K4_ratio'], '{:.2f}')} | {_f(v['K2_pass_ratio'], '{:.2f}')} | {_f(v['K4_pass_ratio'], '{:.2f}')} |")
        L += ["", f"Student and vocoder, once per case: {_f(t['student_decode_s'], '{:.3f}')} s."]
    cal = res["calibration"]
    if cal:
        L += ["", f"Calibration: mu from {cal['frames']} frames of {cal['cases']} cases; per-case band means differ from mu by "
              f"{_f(cal['case_rms_from_mu']['median'])} (median rms), {_f(cal['case_rms_from_mu']['max'])} at most."]
    L += ["", "Development evidence only (speakers that designed earlier experiments), white-noise prompts. A claim needs a frozen test on new speakers."]
    return "\n".join(L) + "\n"


# ------------------------------------------------------------------------ preregistration
def input_hash(path):
    """SHA-256 with line endings normalized: text inputs are LF on Colab but may be CRLF in a Windows checkout."""
    return sh.sha256_text(path)


def code_hashes():
    return {f"code/colab/{n}": sh.sha256_text(Path(__file__).resolve().parent / n) for n in DEPENDENCIES}


def power_summary():
    p = json.loads(POWER_JSON.read_text(encoding="utf-8"))
    pick = lambda k, cond, n, wer: next(r for r in p["pass_rates"] if r["k"] == k and r["condition"] == cond and r["speakers"] == n
                                        and r["bounds"]["wer"] == wer)
    return {"script": "code/analysis/exp024_power_check.py", "proxy": p["proxy"],
            "clean_step4_40": pick(4, "clean", 40, 0.01)["all"], "clean_step4_20_utmos": pick(4, "clean", 20, 0.01)["utmos"],
            "clean_step8_40_wer_0.01": pick(7, "clean", 40, 0.01)["wer"], "clean_step8_40_wer_0.015": pick(7, "clean", 40, 0.015)["wer"],
            "noisy_step4_40": pick(4, "noisy15", 40, 0.025)["all"], "noisy_step8_40": pick(7, "noisy15", 40, 0.025)["all"],
            "noisy_step4_similarity_60": p["effects_60"]["k4_noisy15"]["similarity"]}


def build_prereg():
    pw = power_summary()
    return {"experiment": "EXP-024",
            "purpose": "Development pilot of Analytic Unconditional Guidance before any frozen test: inside the teacher-guided handover window, "
                       "can the unconditional half of classifier-free guidance be computed in closed form instead of by the network, with the "
                       "teacher's handovers keeping quality and pitch variety",
            "theory": "flow matching x_t = (1 - t) eps + t x1: the best unconditional velocity is (E[x1 | x_t] - x_t) / (1 - t); at t = 0 the state "
                      "is independent of x1, so E[x1 | x_0] = E[x1] = mu and v_u = mu - x exactly; for small t, E[x1 | x_t] = mu + O(t). The "
                      "teacher hands over at Step 4 (t = 0.0761) or Step 8 (t = 0.2929), so it only runs while the signal is weak.",
            "algorithm": {"teacher": "the released guided Euler teacher on the 16-interval sway grid (sampler.run_teacher_single_pass, "
                                     "CFG_STRENGTH), except that at Steps 0..K the unconditional evaluation is replaced by v_u = (mu - x_i) / (1 - t_i)",
                          "mu": f"per mel band, the network's own unconditional endpoint estimate at t = 0 (x + v_u(x, 0)) for each case's initial noise, "
                                f"pooled over every frame of the clean-prompt cases at Quick-set positions {CAL_RANGE[0] + 1}-{CAL_RANGE[1]} "
                                f"(speakers the pilot does not use), computed once at the start of the run (calibration.json)",
                          "student": "the released student loop at j = 1 from the teacher's endpoint estimate, unchanged",
                          "arms": {"diagnostic": list(KS), "quality_and_pitch": list(Q_KS)}},
            "passes": {o: cost(o) for o in ("W4", "W8") + tuple(f"W{k}_K{K}" for k in HANDOVER for K in KS)},
            "data": {"pilot": f"the first {N_PILOT} EXP-012 Quick-set development speakers x {list(CONDS)} ({N_PILOT * len(CONDS)} cases), "
                              "synthesis seed = input seed; quality is scored for W4_K and W8_K on clean prompts and W8_K on noisy15",
                     "calibration": f"Quick-set positions {CAL_RANGE[0] + 1}-{CAL_RANGE[1]}, clean prompts, input seed, teacher's unconditional branch at t = 0 only",
                     "diversity": f"the first {DIV_N} speakers x clean x {DIV_SEEDS} seeds (input_seed + {DIV_SEED_BASE} + m), the four AUG handovers; "
                                  "the released Step-4 handover is re-synthesized for the first speaker to check reproduction of EXP-020's stored CVf0",
                     "timing": f"the first {TIMING_N} speakers, clean, {TIMING_REPEATS} repeats in alternating order after one warm-up",
                     "reused": "the released teacher's Step-4 and Step-8 handovers (EXP-018, same cases and seeds) and their clean-prompt CVf0 "
                               "(EXP-020 plain_4 and plain_8, same speakers and seeds)"},
            "diagnostic": {"closed_form_gap": "per step 0-8 on the released trajectory: ||mu - T_u|| / ||T_c - T_u|| on generated frames",
                           "step0_split": f"{LEAK_DRAWS} further noise draws (seed + {LEAK_SEED_BASE} + j) of the same shape: seed-specific part "
                                          "||T_u - mean_j T_u^(j)|| and position part ||mean_j T_u^(j) - mu||, same units",
                           "estimate_change": "relative change of the Step-4 and Step-8 estimates (whole and 9-frame slow part) under AUG K = 0, 2, 4, "
                                              "next to the change from refining the grid to 32 intervals at the same sway time"},
            "hardware": "Colab Tesla T4 only; Shonget-DMOSpeech2 at the pinned commit", "dmo_commit": DMO_COMMIT,
            "smoke": f"must reproduce the stored W4 and W8 scores (EXP-018, tolerance {sh.REPRODUCTION_TOLERANCE}), match the released guided "
                     f"teacher with this runner's loop and with AUG switched off (K = -1), max abs difference <= {TEACHER_TOLERANCE} for both",
            "bounds": {f"{k}|{c}": v for (k, c), v in BOUNDS.items()},
            "criteria": {"quality_clean": "for each K in (2, 4): W4_K and W8_K against the released W4 and W8 of the same case and seed, "
                                          "clean prompts: capped-WER difference upper 95% bound <= the bound, similarity lower bound >= -0.01, "
                                          "UTMOS lower bound >= -0.10 (bounds above; speaker bootstrap)",
                         "pitch_variety": f"for each K in (2, 4): mean CVf0 of W4_K and of W8_K minus EXP-020's stored CVf0 of the released "
                                          f"handover at the same step, over the {DIV_N} diversity speakers, both >= -{CVF0_BOUND}",
                         "quality_noisy_reference": "for each K in (2, 4): W8_K against the released W8 on noisy15 with the bounds above (secondary)"},
            "decision_rule": "advance AUG to a frozen test on new speakers only if, for some K, quality_clean and pitch_variety both hold; the K "
                             "carried forward is the largest that does (the cheapest); quality_noisy_reference at that K decides whether the "
                             "frozen test also covers noisy-prompt teacher runs; otherwise report a negative pilot, with no retuning on these data",
            "power_check": {**pw, "summary": f"with 40 speakers the 16- vs 32-interval grid change passes the clean Step-4 bounds in "
                                             f"{pw['clean_step4_40']:.0%} of 300 subsets (UTMOS {pw['clean_step4_20_utmos']:.0%} with 20 speakers); "
                                             f"at Step 7 the clean WER bound +0.01 passes {pw['clean_step8_40_wer_0.01']:.0%}, +0.015 passes "
                                             f"{pw['clean_step8_40_wer_0.015']:.0%}; on noisy15 the same grid change passes all noisy bounds in "
                                             f"{pw['noisy_step4_40']:.0%} (Step 4) and {pw['noisy_step8_40']:.0%} (Step 7) of subsets, because it "
                                             "moves similarity itself, so noisy-prompt Step-4 handovers are not judged and the noisy-prompt "
                                             "reference criterion is secondary: a failure there does not show that AUG is worse than a grid change"},
            "descriptive": ["the closed-form gap per step and condition, and its Step-0 split", "estimate changes under K = 0, 2, 4 next to the grid change",
                            "Definition 3.1 failures against the released output of the same step and against the reference",
                            "the released W4's CVf0 reproduction for the first diversity speaker", "T4 time per arm and its ratio to the pass ratio",
                            "mu's spread across calibration cases"],
            "bootstrap": {"B": BOOT_B, "seed": BOOT_SEED, "unit": "speaker"},
            "evidence_status": "development pilot on speakers that designed earlier experiments; not evidence for a claim",
            "novelty": "family-level searches on 2026-11-14 found no method that keeps the conditional pass and computes classifier-free guidance's "
                       "unconditional term in closed form. Closest, all different: Lite-CFG (DC-DiT, fewer tokens in the unconditional pass, "
                       "trained multi-budget model), Path-Drop Guidance (SPRINT, unconditional pass skips middle blocks, trained architecture), "
                       "the guidance interval (guidance switched off at high noise), CFG-Zero* (zeroes the first steps), independent-condition "
                       "and time-step guidance (still a network pass), Gaussian analytical teleportation (Wang and Vastola, skips early steps "
                       "with the whole model), and analyses of CFG in Gaussian or linear settings (no sampling shortcut; the 2026 one's sampler "
                       "still runs the unconditional pass). The theory (Gaussian structure at high noise) is known; not a full literature review.",
            "adoption": "not adopted; changing the thesis method needs a dated decision record and adviser review",
            "code_hashes": code_hashes(),
            "input_hashes": {str(p.relative_to(THESIS_ROOT)).replace("\\", "/"): input_hash(p)
                             for p in (STEPS_CSV, EXP020_DIV, FAILURE_CONFIG, MANIFEST, POWER_JSON)},
            "thesis_commit": rf.git_head(THESIS_ROOT)}


def freeze(out=EXP_DIR):
    path = Path(out) / "preregistration.json"
    Path(out).mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise SystemExit(f"{path} exists; a preregistration is written once (delete it only if nothing was synthesized)")
    pre = build_prereg()
    pre["sha256"] = hashlib.sha256(json.dumps(pre, sort_keys=True).encode("utf-8")).hexdigest()
    pre["created_utc"] = sh._utc()
    path.write_text(json.dumps(pre, indent=2) + "\n", encoding="utf-8")
    print(f"[OK] preregistration written: {path} (sha256 {pre['sha256'][:16]}...)", flush=True)
    return pre


def check_prereg(prereg_path=PREREG_PATH):
    p = Path(prereg_path)
    if not p.exists():
        raise SystemExit(f"Missing {p}: run `freeze` and push the preregistration before any GPU run.")
    pre = json.loads(p.read_text(encoding="utf-8"))
    body = {k: v for k, v in pre.items() if k not in ("sha256", "created_utc")}
    if hashlib.sha256(json.dumps(body, sort_keys=True).encode("utf-8")).hexdigest() != pre["sha256"]:
        raise SystemExit("preregistration.json was edited after it was frozen")
    now = code_hashes()
    changed = [k for k, v in pre["code_hashes"].items() if now.get(k) != v]
    if changed:
        raise SystemExit(f"Code differs from the version frozen in preregistration.json: {changed}")
    moved = [k for k, v in pre["input_hashes"].items() if not (THESIS_ROOT / k).exists() or input_hash(THESIS_ROOT / k) != v]
    if moved:
        raise SystemExit(f"Input files are missing or differ from the frozen versions: {moved}")
    return pre


def check_smoke(rep):
    """The full run reuses stored labels only if the smoke test reproduced them and both teacher checks held."""
    sh.check_reproduction(rep["rows"])
    if rep["teacher_loop_max_abs_diff"] > TEACHER_TOLERANCE:
        raise SystemExit(f"This runner's teacher loop does not match the released teacher ({rep['teacher_loop_max_abs_diff']:.2e}); do not run.")
    if rep["aug_off_max_abs_diff"] > TEACHER_TOLERANCE:
        raise SystemExit(f"With AUG switched off the loop differs from the released loop ({rep['aug_off_max_abs_diff']:.2e}); do not run.")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["freeze", "smoke", "run", "analyze"])
    ap.add_argument("--out")
    ap.add_argument("--dmo_root", default="/content/Shonget-DMOSpeech2")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--allow_partial", action="store_true")
    args = ap.parse_args(argv)
    if args.cmd == "freeze":
        freeze(args.out or EXP_DIR)
        return
    if not args.out:
        ap.error("--out is required")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.cmd == "analyze":
        analyze(out, args.allow_partial)
        return
    check_prereg()
    head = rf.git_head(args.dmo_root)
    if head != DMO_COMMIT:
        raise SystemExit(f"Shonget-DMOSpeech2 is at {head}, not the pinned {DMO_COMMIT}.")
    pipe = rf.Pipeline(args.dmo_root, args.device)
    if "T4" not in pipe.gpu:
        raise SystemExit(f"EXP-024 reuses T4 labels and student noise depends on the GPU; found {pipe.gpu}.")
    if args.cmd == "smoke":
        sm = out / "smoke"
        sm.mkdir(exist_ok=True)
        mu = calibrate(pipe, calibration_items()[:2], sm)
        its = calibration_items()[-2:]
        repro, checks = run_cases(pipe, its, CONDS, mu, sm, smoke=True, stored=stored_w48())
        rep = {"rows": repro, "teacher_loop_max_abs_diff": max(c[0] for c in checks), "aug_off_max_abs_diff": max(c[1] for c in checks)}
        (sm / "reproduction.json").write_text(json.dumps(rep, indent=2) + "\n", encoding="utf-8")
        check_smoke(rep)
        run_diversity(pipe, pilot_items()[:1], mu, sm, seeds=2)
        run_timing(pipe, pilot_items()[:1], mu, sm, repeats=1)
        worst_wer, worst_sim = sh.check_reproduction(repro)
        print(f"Smoke run complete: {sm} (stored W4 and W8 reproduced: max |dWER| {worst_wer:.4f}, max |dSIM| {worst_sim:.4f}; "
              f"teacher loop matches the released teacher: max difference {rep['teacher_loop_max_abs_diff']:.2e}; "
              f"AUG switched off: max difference {rep['aug_off_max_abs_diff']:.2e})", flush=True)
        return
    smoke_file = out / "smoke" / "reproduction.json"
    if not smoke_file.exists():
        raise SystemExit("Run the smoke test first: the full run reuses stored labels only if they reproduce.")
    check_smoke(json.loads(smoke_file.read_text(encoding="utf-8")))
    shutil.copy2(PREREG_PATH, out / "preregistration.json")
    mu = calibrate(pipe, calibration_items(), out)
    run_cases(pipe, pilot_items(), CONDS, mu, out)
    run_diversity(pipe, pilot_items()[:DIV_N], mu, out)
    run_timing(pipe, pilot_items()[:TIMING_N], mu, out)
    analyze(out)


if __name__ == "__main__":
    main()
