"""Pilot of Decision Handover (EXP-023): development speakers only, a screen before any frozen test.

Decision Handover. In flow matching the guided teacher's first endpoint estimate T_0 (Step 0) is in theory the same for
every seed: the average output for this text and prompt, including whatever the prompt's recording conditions add.
Everything the teacher adds after that, D_k = T_k - T_0, is this seed's decision. The handover passes only that
decision, on top of the student's own one-pass draft S:

    x_k = S + slow(T_k - T_0)                                          (DH)

slow() is the prosody band of EXP-020's split (voice = per-band mean over time; slow = 9-frame moving average of the
rest; fine = the rest). The released student then finishes from x_k as in teacher-guided sampling. Anything the
teacher shares across seeds cancels in the difference. The family

    x_k(g) = voice(S) + fine(S) + slow(T_k) + g * (slow(S) - slow(T_0))

contains EXP-022's slow-part handover (g = 0) and DH (g = 1). DHc uses g = sqrt(1 - rho_k), the variance-preserving
completion, with rho_k the share of the reference's extra pitch variety over the student alone that the slow-part
handover reached at Step k in EXP-022. DHu is DH with the teacher deciding without guidance (conditional branch only,
one pass per step).

The pilot measures (1) whether T_0 is shared across seeds and the decision is seed-specific (4 seeds, teacher only),
(2) the quality of DH, DHc and DHu at Steps 2 and 3 on 20 development speakers x clean, white noise 20 dB and 15 dB,
scored against the stored 0.25 reference, and (3) their pitch variety on 10 speakers x 20 seeds. Stored outputs of
earlier experiments on the same cases and seeds are reused: the reference (EXP-018), student only (EXP-017),
Voice-Swap (EXP-020) and the slow-part handover (EXP-022). Development evidence only: these speakers designed EXP-020.

Two-Axis Handover, a second candidate (amendment of 2026-11-13, made before any synthesis). From the offline screen
(code/analysis/solution_screen.py, round 4): the prompt SNR decides what crosses (the whole estimate, Voice-Swap at
Step 4, or nothing, so the student works alone), and on the whole-estimate branch the teacher's trajectory action
A_3 = d_1 + d_2 + d_3 decides when (Step 3 if A_3 >= lambda_3, else Step 4). Its parameters are fixed at freeze time
from the stored data of the 60 Quick-set speakers. Its quality passed every rule on the stored outputs; its pitch
variety at Step 3 was never stored, so the diversity block also synthesizes the whole-estimate handovers at Steps 3
and 4 and records each seed's action (G3).

Agreement-Routed Handover, a third candidate (second amendment of 2026-11-13, also before any synthesis). From the
offline screen's round 5: what crosses is decided by whether the teacher and the student agree on the voice part (each
band's average over time, where any recording condition the teacher copies shows up), measured at Step 0 as the gap
between T_0 and the student's draft S; agreement -> the whole estimate at Step 4, a small disagreement -> Voice-Swap,
a large one -> the student alone. It needs no SNR estimator. The quality block already records that Step-0 gap (vg_g)
for every case, so G4 replays the rule out of fold on the pilot's own gaps with the stored outputs of the same cases
and seeds; no extra synthesis is needed.

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
EXP_DIR = THESIS_ROOT / "research/experiments/EXP-023-decision-handover-pilot"
PREREG_PATH = EXP_DIR / "preregistration.json"
_EXPS = THESIS_ROOT / "research/experiments"
EXP020_CSV = _EXPS / "EXP-020-selective-handover-prelim/results/selective.csv"
EXP020_DIV = _EXPS / "EXP-020-selective-handover-prelim/results/diversity.csv"
EXP022_CSV = _EXPS / "EXP-022-slow-part-handover-prelim/results/slowpart.csv"
EXP022_DIV = _EXPS / "EXP-022-slow-part-handover-prelim/results/diversity.csv"
STEPS_CSV, GRID32_CSV = sh.STEPS_CSV, sh.GRID32_CSV
FAILURE_CONFIG = THESIS_ROOT / "code/scheduling/failure_config.json"
DMO_COMMIT = "8a7868fc3345702a586173447f4869205b501f26"
DEPENDENCIES = ("run_decision_handover_pilot.py", "run_selective_handover_prelim.py", "run_grid32_prelim.py",
                "run_noise_guidance_prelim.py", "run_followup_tests.py", "run_consistency_gate_prelim.py",
                "run_endpoint_extrapolation_prelim.py")

STEPS = (2, 3)                        # handover steps
KMAX = max(STEPS)
SEED_KMAX = 4                         # teacher steps examined in the seed check
VARIANTS = ("DH", "DHc", "DHu")
OUTS = tuple(f"{v}_{k}" for v in VARIANTS for k in STEPS)
TAGS = {"g": True, "u": False}        # guided teacher (as released) and unguided (conditional branch only)
METRICS = ("wer", "similarity", "utmos")
N_SPEAKERS = 20
CONDS = ("clean", "gauss20", "noisy15")
DEGRADED = CONDS[1:]
SEED_N, SEED_CONDS, SEED_M, SEED_BASE = 10, ("clean", "noisy15"), 4, 2000
DIV_N, DIV_SEEDS, DIV_SEED_BASE = 10, 20, 1000      # the EXP-020/EXP-022 diversity seeds, so stored CVf0 is comparable
DIV_CONDS = ("clean",)
PARTS = ("all", "voice", "slow", "fine")
BOOT_B, BOOT_SEED = 2000, 2026
TEACHER_TOLERANCE = 1e-4
CRITERIA = {"shared_ratio": 0.25, "wer": 0.025, "similarity": 0.01, "cvf0": 0.01, "max_passes": 12, "ta_cvf0": 0.01}
# Two-Axis Handover (first amendment of 2026-11-13) and Agreement-Routed Handover (second amendment, same day), both
# before any synthesis
ORIGINAL_PREREG = "79752afbf4d5268847ec113ae4366d42c329711f2a4d01fa2dd7410d7309c688"
SECOND_PREREG = "03006f5c53a8ecf16b7cafc932e72bc88ef0cca02fd3edcccd40bfaf4ea2fbf0"
PREVIOUS_PREREG = "590404389927c4888c8059acf8a0548fb84d5cb562442eafb39d64ac1eb39b7c"
ARH_COVERAGE, ARH_FOLDS = 0.95, 5
ARH_OUT = {"whole": "W4", "vs": "STT_4", "student": "student4"}
ARH_COST = {"W4": 14, "STT_4": 14, "student4": 6}   # T_0 (2 passes) and the draft (1) come first, then the branch
SNR_COST = {"W4": 13, "STT_4": 14, "student4": 4}   # the SNR rule decides before any pass
# G4 judges the new signal: routing no worse than the SNR rule on the same cases and folds, cost and pitch variety.
# A misroute is a clean case sent to the student alone or a degraded case sent to the whole estimate.
ARH_RULES = {"misroute_slack": 1, "max_passes": 14.7, "cvf0": 0.01}
TA_OUTS = ("W3", "W4")                # whole-estimate handover at Steps 3 and 4 (plain teacher-guided sampling)
TA_COVERAGE, TA_BAND_DB = 0.95, 2.0   # SNR threshold keeps 95% of clean prompts on the whole branch; Voice-Swap band
TA_RULE = {"clean_wer": 0.01, "degraded_wer": 0.025, "similarity": 0.01, "utmos": 0.05}
DIV_KMAX = 4                          # the diversity block runs the guided teacher to Step 4 for W4
DIV_OUTS = OUTS + TA_OUTS + ("TA",)
SIG_FIELDS = [f"dn_{t}_{k}" for t in TAGS for k in range(1, KMAX + 1)] + \
             [f"dc_{t}_{k}" for t in TAGS for k in range(2, KMAX + 1)] + \
             [f"dr_{t}_{k}" for t in TAGS for k in range(1, KMAX + 1)] + [f"vg_{t}" for t in TAGS]
FIELDS = ["utt_id", "speaker_id", "condition", "prompt_snr", "gpu"] + SIG_FIELDS + \
         [f"{m}_{o}" for o in OUTS for m in METRICS + ("hypothesis",)] + ["case_seconds", "utc"]
SMOKE_FIELDS = [f"{m}_STS_3" for m in METRICS + ("hypothesis",)] + ["teacher_loop_max_abs_diff"]
SEED_FIELDS = ["utt_id", "speaker_id", "condition", "n_seeds", "len_min", "len_max"] + \
              [f"spread_{t}_{p}_{k}" for t in TAGS for p in PARTS for k in range(SEED_KMAX + 1)] + \
              [f"shared_{t}_{k}" for t in TAGS for k in range(1, SEED_KMAX + 1)] + ["case_seconds", "utc"]
DIV_FIELDS = ["utt_id", "speaker_id", "condition", "n_seeds", "prompt_snr"] + [f"cvf0_{o}" for o in DIV_OUTS] + \
             [f"voiced_{o}" for o in DIV_OUTS] + ["ta_branch", "ta_step3_share", "case_seconds", "utc"]


def cost(name):
    """Forward passes: 2 per guided teacher evaluation (1 unguided), 1 for the draft S, 3 student steps."""
    fixed = {"plain_8": 21, "STT_4": 14, "student4": 4}
    if name in fixed:
        return fixed[name]
    if name in TA_OUTS:
        return 2 * (int(name[1:]) + 1) + 3
    base, k = name.rsplit("_", 1)
    if base == "DHu":
        return (int(k) + 1) + 1 + 3
    if base in ("DH", "DHc", "STS"):
        return 2 * (int(k) + 1) + 1 + 3
    raise KeyError(name)


def committed_shares():
    """rho_k: the share of the reference's extra pitch variety over the student alone that the slow-part handover
    reached at Step k on clean prompts (EXP-022 STS_k and plain_8, EXP-020 student4; same speakers and seeds)."""
    a = {r["speaker_id"]: r for r in rf.read_csv(EXP022_DIV) if r["condition"] == "clean"}
    b = {r["speaker_id"]: r for r in rf.read_csv(EXP020_DIV) if r["condition"] == "clean"}
    spk = sorted(s for s in set(a) & set(b)
                 if not any(math.isnan(float(a[s][c])) for c in ["cvf0_plain_8"] + [f"cvf0_STS_{k}" for k in STEPS])
                 and not math.isnan(float(b[s]["cvf0_student4"])))
    mean = lambda rows, col: sum(float(r[col]) for r in rows) / len(rows)
    student, ref = mean([b[s] for s in spk], "cvf0_student4"), mean([a[s] for s in spk], "cvf0_plain_8")
    return {k: (mean([a[s] for s in spk], f"cvf0_STS_{k}") - student) / (ref - student) for k in STEPS}


def completion_gains(rho=None):
    """g_k = sqrt(1 - rho_k), clipped to [0, 1]."""
    rho = committed_shares() if rho is None else rho
    return {k: math.sqrt(1.0 - min(1.0, max(0.0, rho[k]))) for k in STEPS}


# --------------------------------------------------------------------------- Two-Axis Handover
def _steps_rows():
    return {(r["utt_id"], r["condition"]): r for r in rf.read_csv(STEPS_CSV)}


def _action3(r):
    """STAC's trajectory action at Step 3, A_3 = d_1 + d_2 + d_3, from a stored EXP-018 row."""
    return sum(float(r[f"d16_{j}"]) for j in range(1, 4))


def ta_branch(snr, theta, band=TA_BAND_DB):
    """What crosses: 'whole' at or above the SNR threshold, 'vs' (Voice-Swap) within `band` dB below it, else 'student'."""
    return "whole" if snr >= theta else ("vs" if snr >= theta - band else "student")


def ta_route(snr, a3, params):
    """The Two-Axis Handover's output for one input: W3, W4, STT_4 (Voice-Swap) or student4 (student alone)."""
    b = ta_branch(snr, params["snr_threshold"], params["band_db"])
    if b != "whole":
        return "STT_4" if b == "vs" else "student4"
    return "W3" if (params["lambda3"] is not None and a3 >= params["lambda3"]) else "W4"


def two_axis_params():
    """Frozen parameters from the stored EXP-018 data of the 60 Quick-set speakers (all 360 cases), with the offline
    screen's rules applied to all speakers at once:
    - snr_threshold: the prompt SNR that keeps 95% of the clean prompts on the whole-estimate branch;
    - lambda3: the most permissive Step-3 action threshold on that branch whose routing keeps, on means against the
      whole estimate at Step 8, capped-WER difference <= +0.01 on clean and <= +0.025 on degraded cases, similarity
      difference >= -0.01 and UTMOS difference >= -0.05 (scanned from the largest action down, as in the screen)."""
    rows = list(_steps_rows().values())
    clean = sorted((float(r["prompt_snr"]) for r in rows if r["condition"] == "clean"), reverse=True)
    theta = clean[min(len(clean) - 1, int(TA_COVERAGE * len(clean)))]
    whole = [r for r in rows if float(r["prompt_snr"]) >= theta]

    def delta(r, step, x):
        a, b = float(r[f"{x}_g16_{step}"]), float(r[f"{x}_g16_8"])
        return (min(1.0, a) - min(1.0, b)) if x == "wer" else a - b

    def ok(lam):
        step = lambda r: 3 if (lam is not None and _action3(r) >= lam) else 4
        for is_clean, tol in ((True, TA_RULE["clean_wer"]), (False, TA_RULE["degraded_wer"])):
            sub = [r for r in whole if (r["condition"] == "clean") == is_clean]
            if not sub:
                continue
            mean = lambda x: sum(delta(r, step(r), x) for r in sub) / len(sub)
            if mean("wer") > tol or mean("similarity") < -TA_RULE["similarity"] or mean("utmos") < -TA_RULE["utmos"]:
                return False
        return True

    vals = sorted({round(_action3(r), 5) for r in whole}, reverse=True)
    if len(vals) > 80:
        vals = [vals[i] for i in sorted({int(round(i * (len(vals) - 1) / 79)) for i in range(80)})]
    lam = None
    for v in vals:
        if not ok(v):
            break
        lam = v
    return {"snr_threshold": theta, "band_db": TA_BAND_DB, "lambda3": lam}


def two_axis_routes(params):
    """(utt, cond) -> the Two-Axis Handover's output on the stored cases, with its forward passes."""
    return {k: ta_route(float(r["prompt_snr"]), _action3(r), params) for k, r in _steps_rows().items()}


# --------------------------------------------------------------------------- Agreement-Routed Handover
def arh_cuts(clean_gaps):
    """(lo, hi): the whole-estimate branch keeps gaps up to the one that keeps 95% of the clean prompts; Voice-Swap up
    to the largest clean gap; the student alone beyond it."""
    g = sorted(clean_gaps)
    return g[min(len(g) - 1, int(ARH_COVERAGE * len(g)))], g[-1]


def arh_branch(gap, lo, hi):
    return "whole" if gap <= lo else ("vs" if gap <= hi else "student")


def arh_folds(speakers):
    order = sorted(speakers)
    random.Random(BOOT_SEED).shuffle(order)
    return {s: i % ARH_FOLDS for i, s in enumerate(order)}


# --------------------------------------------------------------------------- algorithm
def _norm(x):
    return float((x * x).sum()) ** 0.5


def _cos(a, b):
    return float((a * b).sum()) / (_norm(a) * _norm(b) + 1e-12)


def _maxabs(x):
    return float(abs(x).max())


def teacher_loop(pipe, ctx, kmax, guided):
    """Endpoint estimates T_0..T_kmax on the released 16-interval grid. guided=True repeats
    sampler.run_teacher_single_pass (classifier-free guidance, 2 passes per step); guided=False uses the conditional
    branch only (1 pass per step). Returns ({k: T_k}, forward passes)."""
    torch, sampler = pipe.torch, pipe.sampler
    grid = sampler.SWAY_TEACHER_GRID
    x, ends, passes = sh._clone(ctx["y0"]), {}, 0
    with torch.inference_mode():
        for i in range(kmax + 1):
            t = torch.tensor(grid[i], device=pipe.device, dtype=torch.float32)
            kw = dict(x=x, cond=ctx["step_cond"], text=ctx["text_ids"], time=t, mask=None)
            v = pipe.tts.teacher(drop_audio_cond=False, drop_text=False, **kw)
            passes += 1
            if guided:
                null = pipe.tts.teacher(drop_audio_cond=True, drop_text=True, **kw)
                passes += 1
                v = v + (v - null) * sampler.CFG_STRENGTH
            ends[i] = x + (1.0 - grid[i]) * v
            if i < kmax:
                x = x + (grid[i + 1] - grid[i]) * v
    return ends, passes


def decision_handover(T_k, T_0, S, plen, g=1.0, window=sh.SLOW_WINDOW):
    """x = voice(S) + fine(S) + slow(T_k) + g * (slow(S) - slow(T_0)) on generated frames; prompt frames from T_k.
    g = 1 gives S + slow(T_k - T_0), the teacher's decision on the student's sample; g = 0 gives EXP-022's STS."""
    tk, t0, s = T_k[0, plen:, :], T_0[0, plen:, :], S[0, plen:, :]
    v_s, sl_s, f_s = sh.decompose(s, window)
    sl_k, sl_0 = sh.decompose(tk, window)[1], sh.decompose(t0, window)[1]
    x = sh._clone(T_k)
    x[0, plen:, :] = v_s + f_s + sl_k + g * (sl_s - sl_0)
    return x


def handover_inputs(E, U, S, plen, gains):
    """{output: handover input}: DH and DHc from the guided trajectory E, DHu from the unguided trajectory U."""
    out = {}
    for k in STEPS:
        out[f"DH_{k}"] = decision_handover(E[k], E[0], S, plen, 1.0)
        out[f"DHc_{k}"] = decision_handover(E[k], E[0], S, plen, gains[k])
        out[f"DHu_{k}"] = decision_handover(U[k], U[0], S, plen, 1.0)
    return out


def decision_signals(T, S, plen, tag, kmax=KMAX):
    """Slow part, generated frames: decision size n_k = ||D_k||, turning c_k = cos(D_k, D_k-1), size relative to the
    draft's distance from T_0 r_k = ||D_k|| / ||slow(S) - slow(T_0)||, and the Step-0 voice gap between T_0 and S."""
    sl = {k: sh.decompose(T[k][0, plen:, :])[1] for k in range(kmax + 1)}
    D = {k: sl[k] - sl[0] for k in range(1, kmax + 1)}
    scale = _norm(sh.decompose(S[0, plen:, :])[1] - sl[0])
    out = {}
    for k in range(1, kmax + 1):
        out[f"dn_{tag}_{k}"] = _norm(D[k])
        out[f"dr_{tag}_{k}"] = _norm(D[k]) / (scale + 1e-8)
        if k >= 2:
            out[f"dc_{tag}_{k}"] = _cos(D[k], D[k - 1])
    out[f"vg_{tag}"] = sh.voice_rms(T[0], S, plen)
    return out


def seed_statistics(trajs, plen, tag, kmax=SEED_KMAX):
    """trajs: one {k: T_k} per seed. Relative spread across seeds of each part of T_k (mean pairwise distance over mean
    norm), and the shared share of the slow decision, ||mean D_k|| / mean ||D_k|| (1/sqrt(seeds) for independent
    zero-mean decisions, 1 for a change that every seed makes)."""
    lengths = [int(T[0].shape[1]) - plen for T in trajs]
    n, m = min(lengths), len(trajs)
    split = []
    for T in trajs:
        split.append({k: dict(zip(PARTS, (T[k][0, plen:, :],) + tuple(sh.decompose(T[k][0, plen:, :])))) for k in range(kmax + 1)})
    out = {"len_min": n, "len_max": max(lengths)}
    for p in PARTS:
        for k in range(kmax + 1):
            X = [split[j][k][p][:n] for j in range(m)]
            pairs = [_norm(X[a] - X[b]) for a in range(m) for b in range(a + 1, m)]
            out[f"spread_{tag}_{p}_{k}"] = (sum(pairs) / len(pairs)) / (sum(_norm(x) for x in X) / m + 1e-8)
    for k in range(1, kmax + 1):
        D = [split[j][k]["slow"][:n] - split[j][0]["slow"][:n] for j in range(m)]
        total = D[0]
        for d in D[1:]:
            total = total + d
        out[f"shared_{tag}_{k}"] = _norm(total * (1.0 / m)) / (sum(_norm(d) for d in D) / m + 1e-8)
    return out


# --------------------------------------------------------------------------------- runs
def stored_sts3():
    """(utt_id, condition) -> stored STS_3 scores (EXP-022), reproduced by the smoke test."""
    return {(r["utt_id"], r["condition"]): {"wer": float(r["wer_STS_3"]), "similarity": float(r["similarity_STS_3"])}
            for r in rf.read_csv(EXP022_CSV)}


def run_quality(pipe, items, conds, out_dir, csv_name="pilot.csv", smoke=False, stored=None):
    """Resumable, one row per (utterance, condition): decision signals of both trajectories and the six outputs. The
    smoke run also scores STS_3 (for reproduction) and checks teacher_loop against the released teacher."""
    path = Path(out_dir) / csv_name
    fields = FIELDS + (SMOKE_FIELDS if smoke else [])
    done = rf.done_keys(path, 1)
    todo = [(it, c) for it in items for c in conds if (it["utt_id"], c) not in done]
    print(f"[quality] {len(todo)} cases to run ({len(done)} already in {path.name}) on {pipe.gpu}", flush=True)
    taus, gains, repro, loop_diffs = list(pipe.sampler.STUDENT_TAUS), completion_gains(), [], []
    for n, (item, cond) in enumerate(todo, 1):
        t0 = time.perf_counter()
        ctx = pipe.prepare(item, cond, item["input_seed"], out_dir)
        plen = ctx["prompt_len"]
        E, _ = pipe.teacher(ctx, tuple(range(KMAX + 1)))
        U, _ = teacher_loop(pipe, ctx, KMAX, guided=False)
        S = sh.student_sample(pipe, ctx)
        with pipe.torch.inference_mode():
            sig = {**decision_signals(E, S, plen, "g"), **decision_signals(U, S, plen, "u")}
            inputs = handover_inputs(E, U, S, plen, gains)
            if smoke:
                inputs["STS_3"] = sh.selective_handover(E[3], S, plen, "STS")
        gt = pipe.ground_truth(item)
        row = {"utt_id": item["utt_id"], "speaker_id": item["speaker_id"], "condition": cond, "prompt_snr": ctx["prompt_snr"],
               "gpu": pipe.gpu, **sig}
        for o in OUTS + (("STS_3",) if smoke else ()):
            m = pipe.score(ng.decode(pipe, ctx, gl.student(pipe, ctx, inputs[o], taus)), item, gt)
            row.update({f"{k}_{o}": m[k] for k in METRICS + ("hypothesis",)})
        if smoke:
            G, _ = teacher_loop(pipe, ctx, KMAX, guided=True)
            with pipe.torch.inference_mode():
                row["teacher_loop_max_abs_diff"] = max(_maxabs(G[k] - E[k]) for k in range(KMAX + 1))
            loop_diffs.append(row["teacher_loop_max_abs_diff"])
            key = (item["utt_id"], cond)
            if stored is not None and key in stored:
                repro.append({"utt_id": item["utt_id"], "condition": cond, "output": "STS_3", "wer": row["wer_STS_3"],
                              "wer_stored": stored[key]["wer"], "similarity": row["similarity_STS_3"],
                              "similarity_stored": stored[key]["similarity"]})
        row.update({"case_seconds": round(time.perf_counter() - t0, 2), "utc": sh._utc()})
        sh._append(path, fields, row)
        print(f"[{n}/{len(todo)}] quality: {item['utt_id']} {cond} ({row['case_seconds']:.0f} s)", flush=True)
    print(f"[OK] quality complete: {path}", flush=True)
    return repro, loop_diffs


def run_seeds(pipe, items, conds, out_dir, seeds=SEED_M, csv_name="seeds.csv"):
    """Teacher only: both trajectories to Step 4 for several seeds per pair, prompt degradation fixed (noise seed =
    input seed, synthesis seed = input_seed + 2000 + m)."""
    path = Path(out_dir) / csv_name
    done = rf.done_keys(path, 1)
    todo = [(it, c) for c in conds for it in items if (it["utt_id"], c) not in done]
    print(f"[seeds] {len(todo)} pairs x {seeds} seeds to run ({len(done)} already in {path.name})", flush=True)
    for n, (item, cond) in enumerate(todo, 1):
        t0 = time.perf_counter()
        trajs, plens = {t: [] for t in TAGS}, set()
        for m in range(seeds):
            ctx = pipe.prepare(item, cond, int(item["input_seed"]) + SEED_BASE + m, out_dir, noise_seed=item["input_seed"])
            plens.add(ctx["prompt_len"])
            trajs["g"].append(pipe.teacher(ctx, tuple(range(SEED_KMAX + 1)))[0])
            trajs["u"].append(teacher_loop(pipe, ctx, SEED_KMAX, guided=False)[0])
        if len(plens) != 1:
            raise SystemExit(f"prompt length differs across seeds for {item['utt_id']} {cond}: {sorted(plens)}")
        plen = plens.pop()
        row = {"utt_id": item["utt_id"], "speaker_id": item["speaker_id"], "condition": cond, "n_seeds": seeds}
        with pipe.torch.inference_mode():
            for t in TAGS:
                row.update(seed_statistics(trajs[t], plen, t))
        row.update({"case_seconds": round(time.perf_counter() - t0, 2), "utc": sh._utc()})
        sh._append(path, SEED_FIELDS, row)
        print(f"[{n}/{len(todo)}] seeds: {item['utt_id']} {cond} ({row['case_seconds']:.0f} s)", flush=True)
    print(f"[OK] seeds complete: {path}", flush=True)


def run_diversity(pipe, items, conds, out_dir, seeds=DIV_SEEDS, csv_name="diversity.csv"):
    """CVf0 across seeds of the six outputs and of the Two-Axis Handover's whole-estimate handovers (W3, W4, and TA,
    its per-seed choice between them by the action A_3), with EXP-020/EXP-022's seeds (input_seed + 1000 + m) and
    fixed prompt degradation, so the stored CVf0 of the reference, STS, Voice-Swap and the student alone are
    comparable. A prompt whose SNR puts it on the Voice-Swap or student branch gets no TA value here; the analysis
    uses the stored CVf0 of that output instead."""
    path = Path(out_dir) / csv_name
    done = rf.done_keys(path, 1)
    todo = [(it, c) for c in conds for it in items if (it["utt_id"], c) not in done]
    print(f"[diversity] {len(todo)} pairs x {seeds} seeds to run ({len(done)} already in {path.name})", flush=True)
    taus, gains, ta = list(pipe.sampler.STUDENT_TAUS), completion_gains(), two_axis_params()
    for n, (item, cond) in enumerate(todo, 1):
        t0 = time.perf_counter()
        tracks = {o: [] for o in OUTS + TA_OUTS}
        chosen, step3, snr = [], 0, None
        for m in range(seeds):
            ctx = pipe.prepare(item, cond, int(item["input_seed"]) + DIV_SEED_BASE + m, out_dir, noise_seed=item["input_seed"])
            plen, snr = ctx["prompt_len"], float(ctx["prompt_snr"])
            E, _ = pipe.teacher(ctx, tuple(range(DIV_KMAX + 1)))
            U, _ = teacher_loop(pipe, ctx, KMAX, guided=False)
            S = sh.student_sample(pipe, ctx)
            with pipe.torch.inference_mode():
                inputs = handover_inputs(E, U, S, plen, gains)
                inputs.update({o: E[int(o[1:])] for o in TA_OUTS})
                # the same drift as the stored d16_k (EXP-017/EXP-018), so lambda3 applies unchanged
                a3 = sum(float(pipe.sampler.compute_stabilization_drift(E[i], E[i - 1], plen)) for i in range(1, 4))
            for o in OUTS + TA_OUTS:
                tracks[o].append(sh.f0_track(ng.decode(pipe, ctx, gl.student(pipe, ctx, inputs[o], taus))))
            pick = "W3" if (ta["lambda3"] is not None and a3 >= ta["lambda3"]) else "W4"
            step3 += pick == "W3"
            chosen.append(tracks[pick][-1])
        branch = ta_branch(snr, ta["snr_threshold"], ta["band_db"])
        row = {"utt_id": item["utt_id"], "speaker_id": item["speaker_id"], "condition": cond, "n_seeds": seeds, "prompt_snr": snr}
        for o in OUTS + TA_OUTS:
            row[f"cvf0_{o}"], row[f"voiced_{o}"] = sh.cvf0(tracks[o])
        row["cvf0_TA"], row["voiced_TA"] = sh.cvf0(chosen) if branch == "whole" else (float("nan"), float("nan"))
        row.update({"ta_branch": branch, "ta_step3_share": step3 / seeds,
                    "case_seconds": round(time.perf_counter() - t0, 2), "utc": sh._utc()})
        sh._append(path, DIV_FIELDS, row)
        print(f"[{n}/{len(todo)}] diversity: {item['utt_id']} {cond} ({row['case_seconds']:.0f} s)", flush=True)
    print(f"[OK] diversity complete: {path}", flush=True)


# ----------------------------------------------------------------------------- analysis
def load_stored():
    """(utt, cond) -> speaker and stored metrics: plain_8 (EXP-018), student4 (EXP-017), STT_4 (EXP-020), STS_2 and
    STS_3 (EXP-022)."""
    index = lambda p: {(r["utt_id"], r["condition"]): r for r in rf.read_csv(p)}
    steps, grid, vs, sts = index(STEPS_CSV), index(GRID32_CSV), index(EXP020_CSV), index(EXP022_CSV)
    out = {}
    for key, r in steps.items():
        if key in grid and key in vs and key in sts:
            m = {"plain_8": {x: float(r[f"{x}_g16_8"]) for x in METRICS},
                 "student4": {x: float(grid[key][f"{x}_student4"]) for x in METRICS},
                 "STT_4": {x: float(vs[key][f"{x}_STT_4"]) for x in METRICS}}
            m.update({f"STS_{k}": {x: float(sts[key][f"{x}_STS_{k}"]) for x in METRICS} for k in STEPS})
            out[key] = {"speaker_id": r["speaker_id"], "m": m}
    return out


def stored_cvf0():
    """speaker -> stored clean-prompt CVf0 of plain_8, STS_2 and STS_3 (EXP-022), and of student4, plain_4 (the
    whole-estimate handover at Step 4) and STT_4 (Voice-Swap) from EXP-020."""
    a = {r["speaker_id"]: r for r in rf.read_csv(EXP022_DIV) if r["condition"] == "clean"}
    b = {r["speaker_id"]: r for r in rf.read_csv(EXP020_DIV) if r["condition"] == "clean"}
    return {s: {**{o: float(a[s][f"cvf0_{o}"]) for o in ("plain_8",) + tuple(f"STS_{k}" for k in STEPS)},
                **{o: float(b[s][f"cvf0_{o}"]) for o in ("student4", "plain_4", "STT_4")}} for s in set(a) & set(b)}


def two_axis_pitch(div_rows, stored):
    """G3 data, clean prompts: the Two-Axis Handover's CVf0 minus the stored reference, per speaker. On the
    whole-estimate branch TA is its per-seed choice between W3 and W4; on the Voice-Swap or student branch it is that
    output's stored CVf0 (EXP-020, same speakers and seeds). Also W3 and W4 alone, and how closely W4 reproduces
    EXP-020's stored plain_4."""
    diffs = {"TA": {}, "W3": {}, "W4": {}}
    repro, step3, branches = [], [], defaultdict(int)
    for r in div_rows:
        s = r["speaker_id"]
        if s not in stored:
            continue
        b = r["ta_branch"]
        branches[b] += 1
        ta = float(r["cvf0_TA"]) if b == "whole" else stored[s]["STT_4" if b == "vs" else "student4"]
        for o, x in (("TA", ta), ("W3", float(r["cvf0_W3"])), ("W4", float(r["cvf0_W4"]))):
            if not math.isnan(x):
                diffs[o][s] = [x - stored[s]["plain_8"]]
        w4 = float(r["cvf0_W4"])
        if not math.isnan(w4) and not math.isnan(stored[s]["plain_4"]):
            repro.append(abs(w4 - stored[s]["plain_4"]))
        step3.append(float(r["ta_step3_share"]))
    out = {o: boot(g) for o, g in diffs.items()}
    out.update({"w4_vs_stored_plain_4_max_abs": max(repro) if repro else None,
                "step3_share_mean": (sum(step3) / len(step3)) if step3 else None, "branches": dict(branches)})
    return out


def two_axis_replay(params, th):
    """Descriptive: the Two-Axis Handover with the frozen parameters on all 360 stored cases. In-sample, because the
    same data fixed the parameters; the out-of-fold result is the offline screen's."""
    steps = _steps_rows()
    grid = {(r["utt_id"], r["condition"]): r for r in rf.read_csv(GRID32_CSV)}
    vs = {(r["utt_id"], r["condition"]): r for r in rf.read_csv(EXP020_CSV)}
    routes, table = two_axis_routes(params), []
    for key, r in steps.items():
        if key not in grid or key not in vs:
            continue
        o = routes[key]
        src, col = {"W3": (r, "g16_3"), "W4": (r, "g16_4"), "STT_4": (vs[key], "STT_4"), "student4": (grid[key], "student4")}[o]
        m = {"TA": {x: float(src[f"{x}_{col}"]) for x in METRICS}, "plain_8": {x: float(r[f"{x}_g16_8"]) for x in METRICS}}
        table.append({"spk": r["speaker_id"], "cond": r["condition"], "m": m, "out": o, "passes": cost(o),
                      "fail": bool(gl.failed(m["TA"], m["plain_8"], th))})
    degraded = tuple(sorted({t["cond"] for t in table} - {"clean"}))
    count = lambda cl: sum(t["fail"] for t in table if (t["cond"] == "clean") == cl)
    return {"cases": len(table), "mean_passes": sum(t["passes"] for t in table) / max(1, len(table)),
            "mean_passes_clean": sum(t["passes"] for t in table if t["cond"] == "clean") / max(1, sum(t["cond"] == "clean" for t in table)),
            "routes": {o: sum(t["out"] == o for t in table) for o in ("W3", "W4", "STT_4", "student4")},
            "clean": {x: paired(table, "TA", "plain_8", x, ("clean",)) for x in METRICS},
            "degraded": {x: paired(table, "TA", "plain_8", x, degraded) for x in METRICS},
            "failures_clean": count(True), "failures_degraded": count(False)}


def agreement_routed(rows, div_rows, stored_div, th):
    """G4 data. Out of fold over 5 folds of the pilot's speakers: the cuts come from the other folds' clean-prompt
    Step-0 voice gaps (vg_g, between T_0 and the draft S), each case goes to W4, STT_4 or student4 and is scored with
    the stored output of the same case and seed against the stored reference. Pitch variety: each clean diversity
    prompt follows its clean case's route; W4 uses this pilot's CVf0, STT_4 and student4 the stored CVf0 (EXP-020)."""
    steps = _steps_rows()
    grid = {(r["utt_id"], r["condition"]): r for r in rf.read_csv(GRID32_CSV)}
    vs = {(r["utt_id"], r["condition"]): r for r in rf.read_csv(EXP020_CSV)}
    fold = arh_folds({r["speaker_id"] for r in rows})
    table, route_of, cuts, snr_route = [], {}, {}, {}
    for f in range(ARH_FOLDS):
        train = [r for r in rows if r["condition"] == "clean" and fold[r["speaker_id"]] != f]
        if not train:
            continue
        lo, hi = arh_cuts([float(r["vg_g"]) for r in train])
        snrs = sorted((float(r["prompt_snr"]) for r in train), reverse=True)
        theta = snrs[min(len(snrs) - 1, int(TA_COVERAGE * len(snrs)))]
        cuts[f] = {"gap_whole": lo, "gap_student": hi, "snr_whole": theta, "snr_student": theta - TA_BAND_DB}
        for r in rows:
            key = (r["utt_id"], r["condition"])
            if fold[r["speaker_id"]] != f or key not in steps or key not in grid or key not in vs:
                continue
            snr_route[key] = ARH_OUT[ta_branch(float(r["prompt_snr"]), theta, TA_BAND_DB)]
            o = route_of[key] = ARH_OUT[arh_branch(float(r["vg_g"]), lo, hi)]
            src, col = {"W4": (steps[key], "g16_4"), "STT_4": (vs[key], "STT_4"), "student4": (grid[key], "student4")}[o]
            m = {"ARH": {x: float(src[f"{x}_{col}"]) for x in METRICS}, "plain_8": {x: float(steps[key][f"{x}_g16_8"]) for x in METRICS}}
            table.append({"spk": r["speaker_id"], "cond": r["condition"], "m": m, "out": o, "passes": ARH_COST[o],
                          "fail": bool(gl.failed(m["ARH"], m["plain_8"], th))})
    pitch = {}
    for d in div_rows:
        s, o = d["speaker_id"], route_of.get((d["utt_id"], "clean"))
        if s not in stored_div or o is None:
            continue
        x = float(d["cvf0_W4"]) if o == "W4" else stored_div[s][o]
        if not math.isnan(x):
            pitch[s] = [x - stored_div[s]["plain_8"]]
    clean = [t for t in table if t["cond"] == "clean"]
    misroute = lambda key, o: (key[1] == "clean" and o == "student4") or (key[1] != "clean" and o == "W4")
    return {"cases": len(table), "cuts": cuts,
            "mean_passes": (sum(t["passes"] for t in table) / len(table)) if table else None,
            "routes": {c: {o: sum(t["out"] == o and t["cond"] == c for t in table) for o in ARH_COST} for c in CONDS},
            "misroutes": sum(misroute(k, o) for k, o in route_of.items()),
            "snr_misroutes": sum(misroute(k, o) for k, o in snr_route.items()),
            "snr_mean_passes": (sum(SNR_COST[o] for o in snr_route.values()) / len(snr_route)) if snr_route else None,
            "clean": {x: paired(table, "ARH", "plain_8", x, ("clean",)) for x in METRICS},
            "degraded": {x: paired(table, "ARH", "plain_8", x, DEGRADED) for x in METRICS},
            "clean_failures": sum(t["fail"] for t in clean), "clean_cases": len(clean), "pitch": boot(pitch)}


def arh_passes(a):
    """G4: the Step-0 voice gap misroutes at most one case more than the SNR rule on the same cases and folds; at most
    14.7 mean passes; pitch variety kept. Absolute quality on 20 speakers is reported, not judged: even perfect routing
    of the stored outputs cannot meet a clean UTMOS bound of -0.10 on these speakers (checked before the run)."""
    r = ARH_RULES
    if a["mean_passes"] is None or a["pitch"]["mean"] is None or a["cases"] == 0:
        return None
    return bool(a["misroutes"] <= a["snr_misroutes"] + r["misroute_slack"] and a["mean_passes"] <= r["max_passes"]
                and a["pitch"]["mean"] >= -r["cvf0"])


def _capped(m, name, metric):
    v = m[name][metric]
    return min(1.0, v) if metric == "wer" else v


def build_table(stored, rows, th):
    table = []
    for r in rows:
        s = stored.get((r["utt_id"], r["condition"]))
        if s is None:
            continue
        m = {**s["m"], **{o: {x: float(r[f"{x}_{o}"]) for x in METRICS} for o in OUTS}}
        table.append({"utt": r["utt_id"], "spk": s["speaker_id"], "cond": r["condition"], "m": m,
                      "sig": {k: float(r[k]) for k in SIG_FIELDS},
                      "fail": {n: bool(gl.failed(v, m["plain_8"], th)) for n, v in m.items() if n != "plain_8"}})
    return table


def boot(groups, B=BOOT_B, seed=BOOT_SEED):
    values = [sum(v) / len(v) for v in groups.values() if v]
    if not values:
        return {"mean": None, "lo": None, "hi": None, "n": 0}
    rng = random.Random(seed)
    boots = sorted(sum(rng.choice(values) for _ in values) / len(values) for _ in range(B))
    return {"mean": sum(values) / len(values), "lo": boots[int(0.025 * B)], "hi": boots[int(0.975 * B)], "n": len(values)}


def paired(table, a, b, metric, conds=None):
    g = defaultdict(list)
    for r in table:
        if conds is None or r["cond"] in conds:
            g[r["spk"]].append(_capped(r["m"], a, metric) - _capped(r["m"], b, metric))
    return boot(g)


def _median(values):
    v = sorted(values)
    return None if not v else (v[len(v) // 2] if len(v) % 2 else (v[len(v) // 2 - 1] + v[len(v) // 2]) / 2)


def assumption_summary(seed_rows):
    """Per condition and trajectory: the median ratio of the Step-0 spread to the Step-3 spread of the slow part and of
    the whole estimate, and the mean shared share of the slow decision at Steps 1-4."""
    out = {}
    for c in SEED_CONDS:
        rows = [r for r in seed_rows if r["condition"] == c]
        out[c] = {}
        for t in TAGS:
            ratio = lambda p: _median([float(r[f"spread_{t}_{p}_0"]) / max(float(r[f"spread_{t}_{p}_3"]), 1e-12) for r in rows])
            out[c][t] = {"pairs": len(rows), "slow_ratio_0_to_3": ratio("slow"), "all_ratio_0_to_3": ratio("all"),
                         "voice_ratio_0_to_3": ratio("voice"),
                         "shared": {k: (sum(float(r[f"shared_{t}_{k}"]) for r in rows) / len(rows)) if rows else None
                                    for k in range(1, SEED_KMAX + 1)},
                         "independent_expectation": 1.0 / math.sqrt(SEED_M)}
    return out


def diversity_summary(div_rows, stored):
    """CVf0 of each output and its paired difference with the stored reference, STS at the same step and the student."""
    out = {}
    for o in OUTS:
        k = int(o.rsplit("_", 1)[1])
        diffs = {"vs_reference": {}, "vs_sts": {}, "vs_student": {}}
        vals = []
        for r in div_rows:
            s, x = r["speaker_id"], float(r[f"cvf0_{o}"])
            if s not in stored or math.isnan(x):
                continue
            vals.append(x)
            diffs["vs_reference"][s] = [x - stored[s]["plain_8"]]
            diffs["vs_sts"][s] = [x - stored[s][f"STS_{k}"]]
            diffs["vs_student"][s] = [x - stored[s]["student4"]]
        out[o] = {"mean": (sum(vals) / len(vals)) if vals else None, **{n: boot(g) for n, g in diffs.items()}}
    return out


def signal_summary(table):
    return {c: {f: (lambda v: sum(v) / len(v) if v else None)([r["sig"][f] for r in table if r["cond"] == c]) for f in SIG_FIELDS}
            for c in CONDS}


def analyze(out, allow_partial=False, prereg_path=PREREG_PATH):
    out = Path(out)
    pre = check_prereg(prereg_path)
    th = gl.thresholds()
    read = lambda name: rf.read_csv(out / name, repair=True) if (out / name).exists() else []
    rows, seed_rows, div_rows = read("pilot.csv"), read("seeds.csv"), read("diversity.csv")
    table = build_table(load_stored(), rows, th)
    expected = {"cases": N_SPEAKERS * len(CONDS), "seed_pairs": SEED_N * len(SEED_CONDS), "diversity_pairs": DIV_N * len(DIV_CONDS)}
    have = {"cases": len(table), "seed_pairs": len(seed_rows), "diversity_pairs": len(div_rows)}
    partial = any(have[k] < expected[k] for k in expected)
    if partial and not allow_partial:
        raise SystemExit(f"incomplete run {have} of {expected}; finish the run or pass --allow_partial")
    stored_div = stored_cvf0()
    res = {"prereg_sha256": pre["sha256"], "have": have, "expected": expected, "partial": partial,
           "passes": {o: cost(o) for o in OUTS + TA_OUTS + ("plain_8", "STT_4", "student4") + tuple(f"STS_{k}" for k in STEPS)},
           "completion": {"rho": committed_shares(), "gain": completion_gains()}}
    res["assumptions"] = assumption_summary(seed_rows)
    quality = {}
    for o in OUTS + ("STT_4", "student4") + tuple(f"STS_{k}" for k in STEPS):
        q = {"vs_reference": {m: paired(table, o, "plain_8", m) for m in METRICS},
             "vs_reference_clean": {m: paired(table, o, "plain_8", m, ("clean",)) for m in ("wer", "similarity")},
             "vs_reference_degraded": {m: paired(table, o, "plain_8", m, DEGRADED) for m in ("wer", "similarity")},
             "failures": sum(r["fail"][o] for r in table), "n": len(table)}
        if o in OUTS:
            sts = f"STS_{o.rsplit('_', 1)[1]}"
            q["vs_sts_degraded"] = {m: paired(table, o, sts, m, DEGRADED) for m in ("wer", "similarity")}
            q["vs_sts_clean"] = {m: paired(table, o, sts, m, ("clean",)) for m in ("wer", "similarity")}
        quality[o] = q
    res["quality"] = quality
    res["diversity"] = diversity_summary(div_rows, stored_div)
    res["signals"] = signal_summary(table)
    a = res["assumptions"]
    g1 = None
    if len(seed_rows) >= expected["seed_pairs"]:
        g1 = all(a[c]["g"]["slow_ratio_0_to_3"] is not None and a[c]["g"]["slow_ratio_0_to_3"] <= CRITERIA["shared_ratio"] for c in SEED_CONDS)
    g2 = {}
    complete_q, complete_d = have["cases"] >= expected["cases"], have["diversity_pairs"] >= expected["diversity_pairs"]
    for o in OUTS:
        q, d = quality[o], res["diversity"][o]["vs_reference"]
        if not (complete_q and complete_d) or q["vs_reference"]["wer"]["mean"] is None or d["mean"] is None:
            g2[o] = None
            continue
        g2[o] = bool(cost(o) <= CRITERIA["max_passes"] and q["vs_reference"]["wer"]["hi"] <= CRITERIA["wer"]
                     and q["vs_reference"]["similarity"]["lo"] >= -CRITERIA["similarity"]
                     and q["vs_sts_degraded"]["wer"]["mean"] <= 0.0 and d["mean"] >= -CRITERIA["cvf0"])
    qualifying = sorted((o for o, v in g2.items() if v), key=lambda o: (cost(o), quality[o]["vs_reference"]["wer"]["mean"]))
    ta_params = two_axis_params()
    res["two_axis"] = {"params": ta_params, "pitch": two_axis_pitch(div_rows, stored_div), "replay": two_axis_replay(ta_params, th)}
    ta_mean = res["two_axis"]["pitch"]["TA"]["mean"]
    g3 = None if not complete_d or ta_mean is None else bool(ta_mean >= -CRITERIA["ta_cvf0"])
    res["agreement_routed"] = agreement_routed(rows, div_rows, stored_div, th)
    g4 = arh_passes(res["agreement_routed"]) if (complete_q and complete_d) else None
    res["criteria"] = {"G1_first_estimate_shared": g1, "G2_variant": g2, "G3_two_axis_pitch_variety": g3,
                       "G4_agreement_routed": g4}
    res["advance"] = bool(g1) and bool(qualifying)
    res["carried_variant"] = qualifying[0] if res["advance"] else None
    res["advance_two_axis"] = bool(g3)
    res["advance_agreement_routed"] = bool(g4)
    (out / "pilot_results.json").write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
    (out / "PILOT_REPORT.md").write_text(render(res), encoding="utf-8")
    print(f"Wrote {out / 'PILOT_REPORT.md'}", flush=True)
    return res


def _word(v):
    return "n/a" if v is None else ("PASS" if v else "FAIL")


def _ci(v, fmt="{:+.4f}"):
    return "n/a" if not v or v.get("mean") is None else f"{fmt.format(v['mean'])} [{fmt.format(v['lo'])}, {fmt.format(v['hi'])}]"


def _f(v, fmt="{:.3f}"):
    return "n/a" if v is None else fmt.format(v)


def render(res):
    c = res["criteria"]
    L = ["# Pilot: Decision Handover (EXP-023)", "",
         f"{'**PARTIAL.** ' if res['partial'] else ''}Development speakers only; {res['have']} of {res['expected']}; "
         f"preregistration `{res['prereg_sha256'][:16]}…`. Every number is computed from the CSV files by "
         "`run_decision_handover_pilot.py analyze`. A pilot decides whether a frozen test on new speakers is worth running; "
         "it cannot support a claim.", "",
         "Decision Handover hands the student only the teacher's change since its first estimate, x_k = S + slow(T_k - T_0). "
         "DHc down-weights the student's own prosody by sqrt(1 - rho_k); DHu lets the teacher decide without guidance.", "",
         "## Go / no-go", "",
         f"- G1, the teacher's first estimate is shared across seeds: **{_word(c['G1_first_estimate_shared'])}**",
         "- G2, a variant keeps quality and pitch variety: " + ", ".join(f"{o} {_word(v)}" for o, v in c["G2_variant"].items()),
         f"- **Advance to a frozen test: {'YES, with ' + res['carried_variant'] if res['advance'] else 'NO'}**",
         f"- G3, the Two-Axis Handover keeps pitch variety: **{_word(c['G3_two_axis_pitch_variety'])}**",
         f"- **Two-Axis Handover advances to a frozen test on new speakers: {'YES' if res['advance_two_axis'] else 'NO'}**",
         f"- G4, the Agreement-Routed Handover keeps quality, cost and pitch variety: **{_word(c['G4_agreement_routed'])}**",
         f"- **Agreement-Routed Handover advances to a frozen test on new speakers: {'YES' if res['advance_agreement_routed'] else 'NO'}**", "",
         "## Assumptions (teacher only, " + f"{SEED_M} seeds per pair)", "",
         "| Condition | Trajectory | Spread at Step 0 / Step 3, slow part | Whole estimate | Voice part | Shared share of the decision, Steps 1-4 |",
         "|---|---|---:|---:|---:|---|"]
    for cond, v in res["assumptions"].items():
        for t, s in v.items():
            L.append(f"| {cond} | {'guided' if t == 'g' else 'unguided'} | {_f(s['slow_ratio_0_to_3'])} | {_f(s['all_ratio_0_to_3'])} | "
                     f"{_f(s['voice_ratio_0_to_3'])} | " + ", ".join(_f(s['shared'][k], '{:.2f}') for k in sorted(s['shared'])) +
                     f" (independent: {s['independent_expectation']:.2f}) |")
    L += ["", "## Quality against the 0.25 reference (speaker bootstrap, 95%)", "",
          "| Output | Passes | WER | Similarity | UTMOS | WER, degraded | WER vs STS, degraded | Failures |", "|---|---:|---|---|---|---|---|---:|"]
    for o, q in res["quality"].items():
        L.append(f"| {o} | {res['passes'][o]} | {_ci(q['vs_reference']['wer'])} | {_ci(q['vs_reference']['similarity'])} | "
                 f"{_ci(q['vs_reference']['utmos'], '{:+.3f}')} | {_ci(q['vs_reference_degraded']['wer'])} | "
                 f"{_ci(q['vs_sts_degraded']['wer']) if 'vs_sts_degraded' in q else '—'} | {q['failures']}/{q['n']} |")
    L += ["", "## Pitch variety, clean prompts (CVf0; stored reference, STS and student on the same speakers and seeds)", "",
          "| Output | CVf0 | minus reference | minus STS at the same step | minus student alone |", "|---|---:|---|---|---|"]
    for o, d in res["diversity"].items():
        L.append(f"| {o} | {_f(d['mean'], '{:.4f}')} | {_ci(d['vs_reference'])} | {_ci(d['vs_sts'])} | {_ci(d['vs_student'])} |")
    ta = res["two_axis"]
    p, pt, rp = ta["params"], ta["pitch"], ta["replay"]
    L += ["", "## Two-Axis Handover (second candidate)", "",
          f"Frozen parameters: SNR threshold {_f(p['snr_threshold'], '{:.2f}')} dB, Voice-Swap band {p['band_db']} dB, "
          f"Step-3 action threshold {_f(p['lambda3'], '{:.4f}')}.", "",
          "| Clean-prompt pitch variety (CVf0 minus the stored reference) | Mean [95% CI] | Speakers |", "|---|---|---:|"]
    for o, label in (("TA", "Two-Axis Handover (G3)"), ("W3", "whole estimate at Step 3"), ("W4", "whole estimate at Step 4")):
        L.append(f"| {label} | {_ci(pt[o])} | {pt[o]['n']} |")
    L += ["", f"Branches of the diversity prompts: {pt['branches']}; mean share of seeds handed over at Step 3: "
          f"{_f(pt['step3_share_mean'], '{:.2f}')}; W4 against EXP-020's stored plain_4, max |difference|: "
          f"{_f(pt['w4_vs_stored_plain_4_max_abs'], '{:.4f}')}.", "",
          f"In-sample replay on the {rp['cases']} stored cases (descriptive; the same data fixed the parameters): "
          f"{_f(rp['mean_passes'], '{:.2f}')} mean passes ({_f(rp['mean_passes_clean'], '{:.2f}')} on clean prompts), "
          f"routes {rp['routes']}; clean WER {_ci(rp['clean']['wer'])}, similarity {_ci(rp['clean']['similarity'])}, "
          f"UTMOS {_ci(rp['clean']['utmos'], '{:+.3f}')}; degraded WER {_ci(rp['degraded']['wer'])}, similarity "
          f"{_ci(rp['degraded']['similarity'])}; Definition 3.1 failures {rp['failures_clean']} clean, {rp['failures_degraded']} degraded."]
    ar = res["agreement_routed"]
    L += ["", "## Agreement-Routed Handover (third candidate)", "",
          "Out of fold over 5 speaker folds, on this pilot's Step-0 voice gaps (vg_g) and the stored outputs of the same cases; "
          f"{ar['cases']} cases, {_f(ar['mean_passes'], '{:.2f}')} mean passes; routes per condition {ar['routes']}.", "",
          f"Misroutes (clean case to the student alone, degraded case to the whole estimate): voice gap {ar['misroutes']}, "
          f"prompt SNR on the same cases and folds {ar['snr_misroutes']} (SNR rule {_f(ar['snr_mean_passes'], '{:.2f}')} mean passes). "
          "Absolute quality below is descriptive: 20 speakers cannot resolve it.", "",
          "| Against the stored reference | WER | Similarity | UTMOS |", "|---|---|---|---|",
          f"| clean | {_ci(ar['clean']['wer'])} | {_ci(ar['clean']['similarity'])} | {_ci(ar['clean']['utmos'], '{:+.3f}')} |",
          f"| degraded | {_ci(ar['degraded']['wer'])} | {_ci(ar['degraded']['similarity'])} | {_ci(ar['degraded']['utmos'], '{:+.3f}')} |", "",
          f"Definition 3.1 failures on clean prompts: {ar['clean_failures']} of {ar['clean_cases']}. Clean-prompt pitch variety "
          f"(CVf0 minus the stored reference): {_ci(ar['pitch'])} over {ar['pitch']['n']} speakers."]
    L += ["", "## Decision signals (means per condition)", "",
          "| Condition | " + " | ".join(SIG_FIELDS) + " |", "|---|" + "---:|" * len(SIG_FIELDS)]
    for cond, v in res["signals"].items():
        L.append(f"| {cond} | " + " | ".join(_f(v[f]) for f in SIG_FIELDS) + " |")
    L += ["", f"Completion weights from EXP-022: rho {res['completion']['rho']}, g {res['completion']['gain']}.", "",
          "Development evidence only (speakers that designed EXP-020), white-noise prompts. A claim needs a frozen test on new speakers."]
    return "\n".join(L) + "\n"


# ------------------------------------------------------------------------ preregistration
def input_hash(path):
    """SHA-256 with line endings normalized: text inputs are LF on Colab but may be CRLF in a Windows checkout."""
    return sh.sha256_text(path)


def code_hashes():
    return {f"code/colab/{n}": sh.sha256_text(Path(__file__).resolve().parent / n) for n in DEPENDENCIES}


def build_prereg():
    rho = committed_shares()
    ta = two_axis_params()
    routes = two_axis_routes(ta)
    return {"experiment": "EXP-023",
            "amendments": [
                {"date": "2026-11-13", "when": "before any synthesis", "replaces": ORIGINAL_PREREG,
                 "change": "adds the Two-Axis Handover as a second candidate: the diversity block also synthesizes the whole-estimate "
                           "handovers at Steps 3 and 4 (teacher run to Step 4) and records each seed's Step 1-3 drift; G3 and its "
                           "decision rule are added; nothing about Decision Handover (outputs, data, G1, G2) changes",
                 "reason": "requested by the team member: the offline screen (code/analysis/solution_screen.py, round 4) found it passed "
                           "every measured rule on the stored outputs, but its pitch variety at Step 3 was never stored"},
                {"date": "2026-11-13", "when": "before any synthesis", "replaces": SECOND_PREREG,
                 "change": "adds the Agreement-Routed Handover as a third candidate, analysis only: G4 replays it out of fold on the "
                           "Step-0 voice gaps the quality block already records (vg_g); no output, data or earlier rule changes",
                 "reason": "requested by the team member as the last candidate: the offline screen (round 5) found it passed every rule, "
                           "pitch variety included, with EXP-020's Step-4 gap standing in for the Step-0 gap the method uses"},
                {"date": "2026-11-13", "when": "before any synthesis", "replaces": PREVIOUS_PREREG,
                 "change": "G4 now judges the routing: misroutes by the Step-0 voice gap at most one more than by the prompt-SNR rule on "
                           "the same cases and folds, plus cost and pitch variety; absolute quality is reported, not judged",
                 "reason": "a check before any synthesis found that the previous G4 could not pass on these 20 speakers even with perfect "
                           "routing: the stored whole-estimate outputs at Step 4 give a clean UTMOS lower bound of -0.132 against -0.10, so "
                           "it would have tested the stored outputs on 20 speakers, not the new signal"}],
            "agreement_routed": {"what_crosses": "the Step-0 voice gap between T_0 and the draft S (sh.voice_rms, recorded as vg_g): up to "
                                                 "the cut that keeps 95% of clean prompts -> the whole estimate at Step 4; up to the largest "
                                                 "clean gap -> Voice-Swap at Step 4; beyond it -> the student alone",
                                 "cuts": f"out of fold over {ARH_FOLDS} folds of the pilot's speakers (random.Random({BOOT_SEED}) shuffle), from "
                                         "the other folds' clean-prompt gaps",
                                 "passes": ARH_COST,
                                 "scoring": "the stored outputs of the same cases and seeds (W4 EXP-018, STT_4 EXP-020, student4 EXP-017) against "
                                            "the stored reference; pitch variety: each clean diversity prompt follows its clean case's route, W4 "
                                            "from this pilot's CVf0, STT_4 and student4 from EXP-020's stored CVf0",
                                 "offline_screen": "round 5, with EXP-020's Step-4 gap as a stand-in: 7.60 mean passes out of fold, every rule passed, "
                                                   "clean CVf0 -0.0052 (docs/progress/log/2026-11-13-offline-solution-screen.md)",
                                 "literature": "targeted searches 2026-11-13 found no method that routes inputs by teacher-student disagreement on "
                                               "the voice or compares a teacher's first-step estimate with a student's output to decide what crosses"},
            "two_axis": {"what_crosses": "prompt SNR >= snr_threshold: the whole estimate; within band_db below it: Voice-Swap at Step 4 "
                                         "(EXP-020's STT); below that: nothing, the student alone",
                         "when": "on the whole-estimate branch: Step 3 if A_3 = d_1 + d_2 + d_3 >= lambda3 (d_k = the sampler's "
                                 "compute_stabilization_drift, as stored in EXP-018), else Step 4",
                         "params": ta,
                         "fit": "on the stored EXP-018 data of the 60 Quick-set speakers (all 360 cases): snr_threshold keeps 95% of clean prompts "
                                "on the whole branch; lambda3 is the most permissive action threshold whose routing keeps mean capped-WER "
                                "difference <= +0.01 (clean) and <= +0.025 (degraded), similarity >= -0.01 and UTMOS >= -0.05 against the whole "
                                "estimate at Step 8",
                         "passes": {"W3": cost("W3"), "W4": cost("W4"), "STT_4": cost("STT_4"), "student4": cost("student4")},
                         "stored_routes": {o: sum(v == o for v in routes.values()) for o in ("W3", "W4", "STT_4", "student4")},
                         "offline_screen": "out of fold over 5 speaker folds: 6.73 mean passes, every measured rule passed, pitch variety at "
                                           "Step 3 unmeasured (docs/progress/log/2026-11-13-offline-solution-screen.md)",
                         "literature": "targeted searches 2026-11-13 found no published policy choosing both when and what crosses; each part is known"},
            "purpose": "Development pilot of Decision Handover before any frozen test: is the guided teacher's first estimate shared across seeds, and does handing the student only the teacher's seed-specific change keep quality and pitch variety at Steps 2-3; for the Two-Axis Handover, does its choice of a Step-3 handover keep pitch variety; and for the Agreement-Routed Handover, does routing by the Step-0 teacher-student voice gap keep quality, cost and pitch variety",
            "algorithm": {"decision": "D_k = slow(T_k) - slow(T_0) on generated frames, with T_k the teacher's endpoint estimate at Step k and slow() EXP-020's 9-frame split (sh.decompose)",
                          "handover": "x_k(g) = voice(S) + fine(S) + slow(T_k) + g (slow(S) - slow(T_0)); g = 1 is S + D_k; g = 0 is EXP-022's STS; prompt frames from T_k",
                          "draft": "S = eps + g(eps, 0), one student pass from independent noise (seed + 1009), as in EXP-020",
                          "student": "the released student loop at j = 1 from x_k",
                          "unguided": "the teacher's conditional branch only, 1 pass per step, same grid and Euler update as sampler.run_teacher_single_pass"},
            "outputs": {"DH_k": "g = 1, guided teacher", "DHc_k": "g = sqrt(1 - rho_k), guided teacher", "DHu_k": "g = 1, unguided teacher"},
            "steps": list(STEPS),
            "passes": {n: cost(n) for n in OUTS + TA_OUTS + ("plain_8", "STT_4", "student4") + tuple(f"STS_{k}" for k in STEPS)},
            "completion": {"rho": rho, "gain": completion_gains(rho),
                           "source": "rho_k = (CVf0 STS_k - CVf0 student4) / (CVf0 plain_8 - CVf0 student4), clean prompts, EXP-022 and EXP-020 diversity files, matched speakers"},
            "data": {"quality": f"the first {N_SPEAKERS} EXP-012 Quick-set development speakers x {list(CONDS)} ({N_SPEAKERS * len(CONDS)} cases), synthesis seed = input seed",
                     "seeds": f"the first {SEED_N} speakers x {list(SEED_CONDS)} x {SEED_M} seeds (input_seed + {SEED_BASE} + m), teacher only, Steps 0-{SEED_KMAX}, prompt degradation fixed",
                     "diversity": f"the first {DIV_N} speakers x {list(DIV_CONDS)} x {DIV_SEEDS} seeds (input_seed + {DIV_SEED_BASE} + m), all six "
                                  f"outputs plus the whole-estimate handovers W3 and W4 and the Two-Axis Handover's per-seed choice between them",
                     "reused": "plain Step 8 (EXP-018), student only (EXP-017), Voice-Swap STT_4 (EXP-020), STS_2 and STS_3 (EXP-022) on the same cases and seeds; stored CVf0 of plain_8, STS_2, STS_3 (EXP-022) and student4 (EXP-020)"},
            "hardware": "Colab Tesla T4 only; Shonget-DMOSpeech2 at the pinned commit", "dmo_commit": DMO_COMMIT,
            "smoke": f"must reproduce the stored STS_3 scores (EXP-022, tolerance {sh.REPRODUCTION_TOLERANCE}) and match the released guided teacher with teacher_loop(guided=True) (max abs difference <= {TEACHER_TOLERANCE}) before the full run",
            "criteria": {"G1_first_estimate_shared": f"guided teacher, slow part: median over pairs of (spread across seeds at Step 0) / (spread at Step 3) <= {CRITERIA['shared_ratio']} on clean and on noisy15",
                         "G2_variant": f"for each output: passes <= {CRITERIA['max_passes']}; pooled capped-WER minus the reference, upper bound <= +{CRITERIA['wer']}; similarity minus the reference, lower bound >= -{CRITERIA['similarity']}; mean WER minus STS at the same step on degraded prompts <= 0; clean CVf0 minus the stored reference, mean >= -{CRITERIA['cvf0']}",
                         "G3_two_axis_pitch_variety": f"on the {DIV_N} clean diversity prompts: CVf0 of the Two-Axis Handover (its per-seed choice between W3 and W4 on the whole branch; the stored Voice-Swap or student-alone CVf0 on the other branches) minus the stored reference, mean >= -{CRITERIA['ta_cvf0']}",
                         "G4_agreement_routed": f"out of fold on the {N_SPEAKERS * len(CONDS)} quality cases with {ARH_FOLDS} speaker folds: misroutes by the Step-0 voice gap (a clean case sent to the student alone, or a degraded case sent to the whole estimate) at most {ARH_RULES['misroute_slack']} more than by the prompt-SNR rule (95% clean coverage, {TA_BAND_DB} dB Voice-Swap band) on the same cases and folds; mean passes <= {ARH_RULES['max_passes']}; clean CVf0 minus the stored reference, mean >= -{ARH_RULES['cvf0']}; absolute quality against the stored reference is reported, not judged"},
            "decision_rule": "advance Decision Handover to a frozen test on new speakers only if G1 holds and at least one output meets G2; the output carried forward is the cheapest that meets G2 (ties: lower pooled WER); separately, advance the Two-Axis Handover only if G3 holds, and the Agreement-Routed Handover only if G4 holds; otherwise report a negative pilot for that candidate, with no retuning on these data",
            "descriptive": ["spreads and shared shares for every part and step, both trajectories", "decision signals per condition (size, turning, size relative to the draft, Step-0 voice gap)",
                            "every output against the reference, STS at the same step, Voice-Swap and the student alone", "Definition 3.1 failures against the reference",
                            "the Two-Axis Handover's W3 and W4 pitch variety alone, its share of Step-3 handovers, W4 against EXP-020's stored plain_4, and its in-sample replay on the 360 stored cases",
                            "the Agreement-Routed Handover's routes per condition and its cuts per fold"],
            "bootstrap": {"B": BOOT_B, "seed": BOOT_SEED, "unit": "speaker"},
            "evidence_status": "development pilot on speakers that designed EXP-020 and EXP-022; not evidence for a claim",
            "novelty": "to our knowledge no published handover passes only the teacher's seed-specific change onto the student's own sample (targeted searches 2026-11-13; closest: DiFA, single-model deviation guidance; Gandikota and Bau, whole-state switch after the base model's first step; Distillation++, teacher refines student estimates); not a full literature review",
            "adoption": "not adopted; changing the thesis method needs a dated decision record and adviser review",
            "code_hashes": code_hashes(),
            "input_hashes": {str(p.relative_to(THESIS_ROOT)).replace("\\", "/"): input_hash(p)
                             for p in (STEPS_CSV, GRID32_CSV, EXP020_CSV, EXP020_DIV, EXP022_CSV, EXP022_DIV, FAILURE_CONFIG)},
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
        raise SystemExit(f"EXP-023 reuses T4 labels and student noise depends on the GPU; found {pipe.gpu}.")
    if args.cmd == "smoke":
        sm = out / "smoke"
        sm.mkdir(exist_ok=True)
        its = [it for it in ex.items() if it["utt_id"] not in cg.pilot_utts()][:2]
        repro, loop = run_quality(pipe, its, ("clean", "gauss20"), sm, smoke=True, stored=stored_sts3())
        (sm / "reproduction.json").write_text(json.dumps({"rows": repro, "teacher_loop_max_abs_diff": max(loop)}, indent=2) + "\n",
                                              encoding="utf-8")
        worst_wer, worst_sim = sh.check_reproduction(repro)
        if max(loop) > TEACHER_TOLERANCE:
            raise SystemExit(f"teacher_loop does not match the released teacher (max difference {max(loop):.2e}); do not run.")
        run_seeds(pipe, ex.items()[:1], ("clean",), sm, seeds=2)
        run_diversity(pipe, ex.items()[:1], ("clean",), sm, seeds=2)
        print(f"Smoke run complete: {sm} (stored STS_3 reproduced: max |dWER| {worst_wer:.4f}, max |dSIM| {worst_sim:.4f}; "
              f"teacher loop matches the released teacher: max difference {max(loop):.2e})", flush=True)
        return
    smoke_file = out / "smoke" / "reproduction.json"
    if not smoke_file.exists():
        raise SystemExit("Run the smoke test first: the full run reuses stored labels only if they reproduce.")
    rep = json.loads(smoke_file.read_text(encoding="utf-8"))
    sh.check_reproduction(rep["rows"])
    if rep["teacher_loop_max_abs_diff"] > TEACHER_TOLERANCE:
        raise SystemExit("The smoke test found that teacher_loop does not match the released teacher.")
    shutil.copy2(PREREG_PATH, out / "preregistration.json")
    run_quality(pipe, ex.items()[:N_SPEAKERS], CONDS, out)
    run_seeds(pipe, ex.items()[:SEED_N], SEED_CONDS, out)
    run_diversity(pipe, ex.items()[:DIV_N], DIV_CONDS, out)
    analyze(out)


if __name__ == "__main__":
    main()
