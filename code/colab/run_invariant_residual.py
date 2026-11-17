"""EXP-034 (EVAL-029): the invariant residual against SNR only and Chapter 3's drift, on the T4.

Chapter 3's method decides with two signals: the prompt SNR (Definition 3.3) and the endpoint-stabilization score
(drift, Definition 3.4), d_i = ||x_hat(t_i) - x_hat(t_{i-1})||_M / ||x_hat(t_i)||_M. Local CPU screens (EVAL-028;
docs/progress/log/2026-11-17-invariant-residual.md) proposed replacing drift by the invariant residual

    r = ||x_hat - Q(x_hat, s0)||_M / ||x_hat||_M,    Q(x_hat, s0)_b = sort(s0_b)[rank_t(x_hat_b)]   (per mel band)

the share of the teacher's estimate outside its condition-safe version (the quantile handover: the student draft's
values in the teacher's time order). This run asks which decision rule is better on new development speakers.

Every policy chooses, per prompt, between the same two payloads at the commit point t = 0.2929 (the teacher's 4 Euler
steps on the 8-interval sway grid; the student restarted at 0.2929 and 0.6173):
  CPH8   the teacher's estimate x_hat                                                      12 passes
  CPQH   its condition-safe version Q(x_hat, s0), with the draft's prompt frames          13 passes
Policies (thresholds fixed before this run; lambda values from the CPU screen):
  SNR_ONLY    CPQH if S_hat < 20 dB, else CPH8
  SNR_DRIFT   CPQH if S_hat < 20 dB or drift d_4 < LAMBDA_D (low drift flags noise, the screen's direction)
  SNR_R       CPQH if S_hat < 20 dB or r > LAMBDA_R          (the proposed method)
  R_ONLY      CPQH if r > LAMBDA_R
Reference arms on the same cases: STU (the released student, 4 passes), STUR2 (its draft restarted at 0.2929 and
0.6173, 3), W8 (Chapter 3's reference: 16-interval Step 8, student from 0.0761, 21), and in the variety stage P25 (the
paper's setting, 30).

Stages: quality on development positions 31-60 x 8 prompt conditions (240 cases; WER, speaker similarity, UTMOS);
variety on positions 41-60 x 12 seeds x {clean, gauss25}; timing and pass counts on positions 31-40 (clean). Positions
1-30 (EXP-031 to EXP-033, and the CPU screens that set the thresholds) are not used; 41-60 are new to every T4 run.

Subcommands: freeze, smoke, run, analyze.
"""
import argparse
import json
import math
import random
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_cheaper_teacher_path as ctp  # noqa: E402
import run_consistency_gate_prelim as cg  # noqa: E402
import run_damped_transmissibility_stac as dts  # noqa: E402
import run_followup_tests as rf  # noqa: E402
import run_grid32_prelim as gl  # noqa: E402
import run_method_comparison as mc  # noqa: E402
import run_noise_guidance_prelim as ng  # noqa: E402
import run_selective_handover_prelim as sh  # noqa: E402
import run_timing_only_handover as toh  # noqa: E402
import run_transmissibility_bridge as tb  # noqa: E402
import run_transmissibility_confirm as tc  # noqa: E402
import run_transmissibility_test as tt  # noqa: E402
import run_variety_per_pass as vp  # noqa: E402

THESIS_ROOT = rf.THESIS_ROOT
CODE_DIR = Path(__file__).resolve().parent
EXP_DIR = THESIS_ROOT / "research/experiments/EXP-034-invariant-residual"
PREREG_PATH = EXP_DIR / "preregistration.json"
EXP031_QUAL_CSV = THESIS_ROOT / "research/experiments/EXP-031-snr-toh-method-comparison/results/quality.csv"
EXP033_QUAL_CSV = THESIS_ROOT / "research/experiments/EXP-033-cheaper-teacher-path/results/quality.csv"
SCREEN_JSON = THESIS_ROOT / "research/experiments/EXP-033-cheaper-teacher-path/offline/invariant_residual_screen.json"

DEPENDENCIES = ("run_invariant_residual.py",) + tuple(ctp.DEPENDENCIES)

QUAL_POS, DIV_POS, TIME_POS = (31, 60), (41, 60), (31, 40)   # 1-based positions in cg.dev_items()
CAL_LAST = 45                                                 # secondary analysis: 31-45 calibrate, 46-60 test
CONDS = ("clean", "noisy15", "gauss25", "pink25", "hp25", "reverb09", "telephone", "babble15")
DIV_CONDS = ("clean", "gauss25")
DIV_SEEDS = 12
DIV_SEED_BASE = vp.DIV_SEED_BASE                              # 1000
SEED, BOOT_B = vp.SEED, vp.BOOT_B
THETA = 20.0
LAMBDA_R = 0.34004843191300677                                # invariant_residual_screen.json, calibration half
LAMBDA_D = 0.2251750830960417                                 # the same screen, drift flagged when below
NEEDED = 0.30                                                 # "CPQH needed": UTMOS(CPQH) - UTMOS(CPH8) > 0.30
MIN_VOICED = 0.30
K8 = 4                                                        # 8-interval sway grid, Step 4: t = 0.2929
T_COMMIT = gl.sway_grid(8)[K8]
SHAPED_SNR_DB = 25.0
SHAPES = {                                                    # identical to code/analysis noise screens
    "pink25": lambda f: 1.0 / np.sqrt(np.maximum(f, 20.0)),
    "hp25": lambda f: 1.0 / np.sqrt(1.0 + (2000.0 / np.maximum(f, 1.0)) ** 8),
}

A_ARM, B_ARM = "CPH8", "CPQH"
QUAL_ARMS = ("STU", "STUR2", "CPH8", "CPQH", "W8")
DIV_ARMS = ("STU", "STUR2", "CPH8", "CPQH", "P25")
POLICIES = ("SNR_ONLY", "SNR_DRIFT", "SNR_R", "R_ONLY")
TIME_ARMS = ("STU", "STUR2", "CPH8", "CPQH", "W8", "P25", "SNR_ONLY", "SNR_R")
TEACHER_PASSES = {"STU": 0, "STUR2": 0, "CPH8": 10, "CPQH": 10, "W8": 18, "P25": 28, "SNR_ONLY": 10, "SNR_R": 10}
STUDENT_PASSES = {"STU": 4, "STUR2": 3, "CPH8": 2, "CPQH": 3, "W8": 3, "P25": 2, "SNR_ONLY": 2, "SNR_R": 3}
COST = {a: TEACHER_PASSES[a] + STUDENT_PASSES[a] for a in TEACHER_PASSES}   # SNR_ONLY: clean prompts take CPH8

METRICS = vp.METRICS                                          # wer, similarity, utmos
VARIETY = vp.VARIETY                                          # cvf0, cvf0_aligned
DIV_MEASURES = ("cvf0", "voiced", "cvf0_aligned", "voiced_aligned", "timing_spread", "melody")
RUNAWAY_WER = vp.RUNAWAY_WER
WER_BAR, SIM_BAR = vp.WER_BAR, vp.SIM_BAR                     # +0.01 WER, -0.01 similarity
AUC_ALL, AUC_GAIN, MAX_WRONG_CLEAN, VARIETY_KEEP = 0.80, 0.10, 0.15, 0.90
GRID_TOL, SPLIT_TOL, REPRO_TOL = vp.GRID_TOL, vp.SPLIT_TOL, vp.REPRO_TOL

QUAL_FIELDS = (["utt_id", "speaker_id", "condition", "prompt_snr", "snr_sw", "r", "drift8", "rough_share"]
               + [f"d16_{j}" for j in range(4, 9)] + ["gpu", "frames", "seq_len"]
               + [f"choice_{p}" for p in POLICIES]
               + [f"{m}_{a}" for a in QUAL_ARMS for m in METRICS + ("hypothesis",)]
               + ["case_seconds", "utc"])
DIV_FIELDS = (["utt_id", "speaker_id", "condition", "n_seeds", "prompt_snr", "r_mean", "drift8_mean"]
              + [f"flagged_{p}" for p in POLICIES]
              + [f"{m}_{a}" for a in DIV_ARMS + POLICIES for m in DIV_MEASURES]
              + ["case_seconds", "utc"])
TIME_FIELDS = (["utt_id", "speaker_id", "condition", "prompt_snr", "gpu", "frames", "seq_len"]
               + [f"{m}_{a}" for a in TIME_ARMS for m in ("seconds", "teacher_passes", "student_passes")]
               + ["utc"])


def _items(pos):
    return cg.dev_items()[pos[0] - 1:pos[1]]


def qual_items():
    return _items(QUAL_POS)


def div_items():
    return _items(DIV_POS)


def time_items():
    return _items(TIME_POS)


# --------------------------------------------------------------------------- prompt conditions
def shaped_noise(n, sr, seed, weight):
    rng = np.random.RandomState(seed % (2 ** 31 - 1))
    spec = np.fft.rfft(rng.randn(n)) * weight(np.fft.rfftfreq(n, 1.0 / sr))
    return np.fft.irfft(spec, n)


def add_at(clean, noise, snr_db, eps=1e-12):
    """Scale `noise` to the total-power SNR of the experiments' add_gaussian, and add it."""
    sig = float(np.mean(clean.astype(np.float64) ** 2))
    noise = noise * np.sqrt((sig / (10.0 ** (snr_db / 10.0))) / (float(np.mean(noise ** 2)) + eps))
    return np.clip(clean + noise, -1.0, 1.0).astype(np.float32)


_RF_DEGRADE = getattr(rf, "_original_degrade", rf.degrade)   # the released function, even if already installed


def degrade(clean, sr, condition, seed, babble_track=None):
    """The experiments' degradations, plus pink noise and high-pass hiss at 25 dB (the local screens' noises)."""
    if condition in SHAPES:
        return add_at(np.asarray(clean, dtype=np.float32), shaped_noise(len(clean), sr, seed, SHAPES[condition]),
                      SHAPED_SNR_DB)
    return _RF_DEGRADE(clean, sr, condition, seed, babble_track)


def install_conditions():
    """Registers pink25 and hp25 with the shared pipeline, which looks up CONDITIONS and degrade at call time.
    Called by make_pipe() at run time only, so importing this module changes nothing."""
    rf._original_degrade = _RF_DEGRADE
    for name in SHAPES:
        rf.CONDITIONS.setdefault(name, ("shaped", SHAPED_SNR_DB))
    rf.degrade = degrade


# --------------------------------------------------------------------------- the signals
def invariant_residual(teacher_gen, student_gen):
    """r = ||x_hat - Q(x_hat, s0)|| / ||x_hat|| over (frames, bands) arrays of the generated frames."""
    t = np.asarray(teacher_gen, dtype=np.float64)
    q = mc.quantile_handover(np.asarray(student_gen, dtype=np.float64), t)
    return float(np.linalg.norm(t - q) / (np.linalg.norm(t) + 1e-12))


def band_roughness(x):
    """Per band: mean squared frame-to-frame change over the band's variance (von Neumann's ratio)."""
    x = np.asarray(x, dtype=np.float64)
    d = np.diff(x, axis=0)
    return np.mean(d * d, axis=0) / np.maximum(np.var(x, axis=0), 1e-12)


def rough_share(teacher_gen, student_gen):
    return float(np.mean(band_roughness(teacher_gen) > band_roughness(student_gen)))


def _frames(x, sr):
    fl, hop = round(0.025 * sr), round(0.010 * sr)
    n = 1 + (len(x) - fl) // hop
    return np.lib.stride_tricks.as_strided(x, shape=(n, fl), strides=(x.strides[0] * hop, x.strides[0])), fl


def spread_snr(x, sr, eps=1e-12, f_lo=60.0):
    """The spread-weighted SNR (EVAL-027): loud-half power over the geometric mean of the per-band 10th-percentile
    floor, 60 Hz to Nyquist, on Definition 3.3's frames."""
    x = np.ascontiguousarray(np.asarray(x, dtype=np.float64).squeeze())
    fr, fl = _frames(x, sr)
    spec = np.abs(np.fft.rfft(fr * np.hanning(fl), axis=1)) ** 2
    spec = spec[:, np.fft.rfftfreq(fl, 1.0 / sr) >= f_lo]
    energy = spec.mean(axis=1)
    p_speech = float(energy[energy >= np.median(energy)].mean())
    geo = float(np.exp(np.mean(np.log(np.percentile(spec, 10, axis=0) + eps))))
    return 10 * np.log10((p_speech + eps) / (geo + eps))


def choose(policy, snr, r, drift8, lam_r=LAMBDA_R, lam_d=LAMBDA_D, theta=THETA):
    """The payload a policy hands over: B_ARM (the condition-safe version) or A_ARM (the teacher's estimate)."""
    if policy == "SNR_ONLY":
        flag = snr < theta
    elif policy == "SNR_DRIFT":
        flag = snr < theta or drift8 < lam_d
    elif policy == "SNR_R":
        flag = snr < theta or r > lam_r
    elif policy == "R_ONLY":
        flag = r > lam_r
    else:
        raise ValueError(f"Unknown policy {policy!r}")
    return B_ARM if flag else A_ARM


# --------------------------------------------------------------------------- synthesis
def synthesize(pipe, ctx, arms):
    """All requested arms for one case from one 8-interval teacher run (Steps 3 and 4) and one student draft, plus a
    16-interval run for W8 and a 32-interval run for P25 when asked. Returns (outputs, parts)."""
    taus = list(pipe.sampler.STUDENT_TAUS)
    plen = ctx["prompt_len"]
    ends8, drift8, _ = gl.grid_teacher(pipe, ctx, gl.sway_grid(8), {K8 - 1, K8})
    t4 = ends8[K8]
    s0 = gl.student(pipe, ctx, ctx["step_cond"], [0.0])
    qh = mc.quantile_payload(s0, t4, plen)
    outs = {"CPH8": gl.student(pipe, ctx, t4, taus[1:]), "CPQH": gl.student(pipe, ctx, qh, taus[1:])}
    if "STU" in arms:
        outs["STU"] = gl.student(pipe, ctx, s0, taus)
    if "STUR2" in arms:
        outs["STUR2"] = gl.student(pipe, ctx, s0, taus[1:])
    drift16 = {}
    if "W8" in arms:
        e16, drift16, _ = gl.grid_teacher(pipe, ctx, tt.grid16(pipe), {8})
        outs["W8"] = gl.student(pipe, ctx, e16[8], taus)
    if "P25" in arms:
        _, _, s32 = gl.grid_teacher(pipe, ctx, gl.GRID32, {vp.K_PAPER})
        outs["P25"] = gl.student_from_state(pipe, ctx, s32[vp.K_PAPER], list(gl.ALG2_TIMES))
    g_t, g_s = toh._as_numpy(t4[0, plen:, :]), toh._as_numpy(s0[0, plen:, :])
    parts = {"t4": t4, "t3": ends8[K8 - 1], "s0": s0, "qh": qh, "drift8": float(drift8[K8]), "drift16": drift16,
             "r": invariant_residual(g_t, g_s), "rough_share": rough_share(g_t, g_s)}
    return {a: outs[a] for a in arms}, parts


def standalone(pipe, ctx, arm):
    """One arm or policy with exactly its own forward passes, for the pass count and the wall-clock time."""
    taus = list(pipe.sampler.STUDENT_TAUS)
    plen = ctx["prompt_len"]
    teacher = lambda: ctp.euler(pipe, ctx, gl.sway_grid(8), ctx["y0"].clone(), 0, K8)  # noqa: E731
    draft = lambda: gl.student(pipe, ctx, ctx["step_cond"], [0.0])                    # noqa: E731
    if arm == "STU":
        return gl.student(pipe, ctx, ctx["step_cond"], [0.0] + taus)
    if arm == "STUR2":
        return gl.student(pipe, ctx, draft(), taus[1:])
    if arm == "CPH8" or (arm == "SNR_ONLY" and float(ctx["prompt_snr"]) >= THETA):
        return gl.student(pipe, ctx, teacher(), taus[1:])
    if arm in ("CPQH", "SNR_ONLY"):
        t4 = teacher()
        return gl.student(pipe, ctx, mc.quantile_payload(draft(), t4, plen), taus[1:])
    if arm == "SNR_R":
        t4, s0 = teacher(), draft()
        r = invariant_residual(toh._as_numpy(t4[0, plen:, :]), toh._as_numpy(s0[0, plen:, :]))
        pick = choose("SNR_R", float(ctx["prompt_snr"]), r, 0.0)
        return gl.student(pipe, ctx, t4 if pick == A_ARM else mc.quantile_payload(s0, t4, plen), taus[1:])
    if arm == "W8":
        return gl.student(pipe, ctx, ctp.euler(pipe, ctx, tt.grid16(pipe), ctx["y0"].clone(), 0, 8), taus)
    if arm == "P25":
        return gl.student_from_state(pipe, ctx, ctp.teacher_state(pipe, ctx, gl.GRID32, vp.K_PAPER),
                                     list(gl.ALG2_TIMES))
    raise ValueError(f"Unknown arm {arm!r}")


def _num(v):
    return float("nan") if v is None else float(v)


def policy_passes(policy, arm):
    """Forward passes a policy spends on one prompt: the residual needs the draft, so SNR_R and R_ONLY always cost
    13; SNR_ONLY and SNR_DRIFT cost 12 when they hand over the teacher's estimate and 13 otherwise."""
    if policy in ("SNR_R", "R_ONLY"):
        return COST[B_ARM]
    return COST[A_ARM] if arm == A_ARM else COST[B_ARM]


def expected_passes(arm, snr):
    """(teacher, student) passes of standalone(arm) for a prompt with SNR `snr`."""
    if arm == "SNR_ONLY":
        return (TEACHER_PASSES[A_ARM], STUDENT_PASSES[A_ARM if snr >= THETA else B_ARM])
    return (TEACHER_PASSES[arm], STUDENT_PASSES[arm])


# --------------------------------------------------------------------------- runs
def run_quality(pipe, items, conds=CONDS, out_dir=None, csv_name="quality.csv"):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / csv_name
    done = rf.done_keys(path, 1)
    todo = [(it, c) for c in conds for it in items if (it["utt_id"], c) not in done]
    print(f"[quality] {len(todo)} cases to run ({len(done)} already in {path.name})", flush=True)
    for n_run, (item, cond) in enumerate(todo, 1):
        t0 = time.perf_counter()
        ctx = pipe.prepare(item, cond, int(item["input_seed"]), out_dir)
        outs, parts = synthesize(pipe, ctx, QUAL_ARMS)
        gt = pipe.ground_truth(item)
        snr = float(ctx["prompt_snr"])
        row = {"utt_id": item["utt_id"], "speaker_id": item["speaker_id"], "condition": cond, "prompt_snr": snr,
               "snr_sw": spread_snr(ctx["prompt"], ctx["sr"]), "r": parts["r"], "drift8": parts["drift8"],
               "rough_share": parts["rough_share"],
               **{f"d16_{j}": _num(parts["drift16"].get(j)) for j in range(4, 9)},
               "gpu": pipe.gpu, "frames": int(parts["s0"].shape[1] - ctx["prompt_len"]),
               "seq_len": int(parts["s0"].shape[1]),
               **{f"choice_{p}": choose(p, snr, parts["r"], parts["drift8"]) for p in POLICIES}}
        for arm in QUAL_ARMS:
            m = pipe.score(ng.decode(pipe, ctx, outs[arm]), item, gt)
            row.update({f"{k}_{arm}": m[k] for k in METRICS + ("hypothesis",)})
        row.update({"case_seconds": round(time.perf_counter() - t0, 2), "utc": tb._utc()})
        tb._append(path, QUAL_FIELDS, row)
        print(f"[{n_run}/{len(todo)}] quality: {item['utt_id']} {cond} SNR {snr:.1f} r {parts['r']:.3f} -> "
              f"{row['choice_SNR_R']} ({row['case_seconds']:.1f} s)", flush=True)
    print(f"[OK] quality complete: {path}", flush=True)
    return path


def run_diversity(pipe, items, out_dir, seeds=DIV_SEEDS, conds=DIV_CONDS, csv_name="diversity.csv"):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / csv_name
    done = rf.done_keys(path, 1)
    todo = [(it, c) for c in conds for it in items if (it["utt_id"], c) not in done]
    print(f"[diversity] {len(todo)} speaker-conditions x {seeds} seeds ({len(done)} already in {path.name})", flush=True)
    names = DIV_ARMS + POLICIES
    for n_run, (item, cond) in enumerate(todo, 1):
        t0 = time.perf_counter()
        tracks, mels, melody = ({a: [] for a in names} for _ in range(3))
        rs, ds, flags, snr = [], [], {p: [] for p in POLICIES}, None
        for s in range(seeds):
            seed = int(item["input_seed"]) + DIV_SEED_BASE + s
            ctx = pipe.prepare(item, cond, seed, out_dir, noise_seed=item["input_seed"])
            outs, parts = synthesize(pipe, ctx, DIV_ARMS)
            arm_tracks = dts._track_arms_fast(pipe, ctx, outs, DIV_ARMS)
            plen, snr = ctx["prompt_len"], float(ctx["prompt_snr"])
            arm_mels = {a: toh._as_numpy(outs[a][0, plen:, :]) for a in DIV_ARMS}
            pick = {p: choose(p, snr, parts["r"], parts["drift8"]) for p in POLICIES}
            for a in DIV_ARMS:
                tracks[a].append(arm_tracks[a])
                mels[a].append(arm_mels[a])
            for p in POLICIES:
                tracks[p].append(arm_tracks[pick[p]])
                mels[p].append(arm_mels[pick[p]])
                flags[p].append(pick[p] == B_ARM)
            for a in names:
                melody[a].append(ctp.melody_agreement(tracks[a][-1], arm_tracks["P25"]))
            rs.append(parts["r"])
            ds.append(parts["drift8"])
        row = {"utt_id": item["utt_id"], "speaker_id": item["speaker_id"], "condition": cond, "n_seeds": seeds,
               "prompt_snr": snr, "r_mean": ctp._mean(rs), "drift8_mean": ctp._mean(ds),
               **{f"flagged_{p}": sum(flags[p]) / len(flags[p]) for p in POLICIES}}
        for a in names:
            row[f"cvf0_{a}"], row[f"voiced_{a}"] = sh.cvf0(tracks[a])
            row[f"cvf0_aligned_{a}"], row[f"voiced_aligned_{a}"], row[f"timing_spread_{a}"] = \
                mc.aligned_cvf0(tracks[a], mels[a])
            row[f"melody_{a}"] = ctp._mean(melody[a])
        row.update({"case_seconds": round(time.perf_counter() - t0, 2), "utc": tb._utc()})
        tb._append(path, DIV_FIELDS, row)
        print(f"[{n_run}/{len(todo)}] diversity: {item['utt_id']} {cond} ({row['case_seconds']:.1f} s)", flush=True)
    print(f"[OK] diversity complete: {path}", flush=True)
    return path


def run_timing(pipe, items, out_dir, cond="clean", csv_name="timing.csv"):
    """Each arm and policy on its own, timed between GPU synchronizations, with its forward passes counted. One
    untimed warm-up; the order alternates between cases."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / csv_name
    done = rf.done_keys(path, 1)
    todo = [it for it in items if (it["utt_id"], cond) not in done]
    print(f"[timing] {len(todo)} cases to run ({len(done)} already in {path.name})", flush=True)

    def one(ctx, arm):
        return ng.decode(pipe, ctx, standalone(pipe, ctx, arm))

    with ctp.PassCounter(pipe) as counter:
        for n_run, item in enumerate(todo, 1):
            ctx = pipe.prepare(item, cond, int(item["input_seed"]), out_dir)
            if n_run == 1:
                one(ctx, "CPH8")                          # warm-up, not timed
            row = {"utt_id": item["utt_id"], "speaker_id": item["speaker_id"], "condition": cond,
                   "prompt_snr": float(ctx["prompt_snr"]), "gpu": pipe.gpu,
                   "frames": int(ctx["y0"].shape[1] - ctx["prompt_len"]), "seq_len": int(ctx["y0"].shape[1])}
            for arm in (TIME_ARMS if n_run % 2 else TIME_ARMS[::-1]):
                counter.reset()
                _, sec = gl.timed(pipe, one, ctx, arm)
                row.update({f"seconds_{arm}": round(sec, 4), f"teacher_passes_{arm}": counter.n["teacher"],
                            f"student_passes_{arm}": counter.n["student"]})
            row["utc"] = tb._utc()
            tb._append(path, TIME_FIELDS, row)
            print(f"[{n_run}/{len(todo)}] timing: {item['utt_id']}", flush=True)
    print(f"[OK] timing complete: {path}", flush=True)
    return path


# --------------------------------------------------------------------------- smoke test
def smoke(pipe, out_dir, prereg_path=PREREG_PATH):
    """Exact identities and pass counts on development speaker 1 (clean), reproductions of EXP-033 (CPH8, STU, W8;
    clean and noisy15) and EXP-031 (STU, W8; clean, noisy15, gauss25, reverb09, babble15) on the same speaker, then
    micro runs of the three stages."""
    check_prereg(prereg_path)
    sdir = Path(out_dir) / "smoke"
    sdir.mkdir(parents=True, exist_ok=True)
    taus = list(pipe.sampler.STUDENT_TAUS)
    item = cg.dev_items()[0]
    ctx = pipe.prepare(item, "clean", int(item["input_seed"]), sdir)
    plen = ctx["prompt_len"]
    outs, parts = synthesize(pipe, ctx, QUAL_ARMS + ("P25",))
    t4, s0 = parts["t4"], parts["s0"]
    qh_self = mc.quantile_payload(t4, t4, plen)
    rec = {"utt_id": item["utt_id"], "prompt_snr": float(ctx["prompt_snr"]), "r": parts["r"],
           "finite": all(bool(np.all(np.isfinite(toh._as_numpy(outs[a])))) for a in outs),
           "euler_vs_grid_teacher": tb._max_abs(ctp.euler(pipe, ctx, gl.sway_grid(8), ctx["y0"].clone(), 0, K8) - t4),
           "qh_of_itself": tb._max_abs(qh_self[:, plen:, :] - t4[:, plen:, :]),
           "r_of_itself": invariant_residual(toh._as_numpy(t4[0, plen:, :]), toh._as_numpy(t4[0, plen:, :])),
           "qh_prompt_frames_are_the_draft": tb._max_abs(parts["qh"][:, :plen, :] - s0[:, :plen, :]),
           "split_diff": tb._max_abs(gl.student(pipe, ctx, s0, taus) - gl.student(pipe, ctx, ctx["step_cond"], [0.0] + taus))}
    alone, counts = {}, {}
    with ctp.PassCounter(pipe) as counter:
        for arm in TIME_ARMS:
            counter.reset()
            out = standalone(pipe, ctx, arm)
            counts[arm] = dict(counter.n)
            ref = outs[arm] if arm in outs else outs[choose(arm, rec["prompt_snr"], rec["r"], parts["drift8"])]
            alone[arm] = tb._max_abs(out - ref)
    rec["standalone_diff"], rec["passes"] = alone, counts
    print(f"[smoke] {json.dumps(rec)}", flush=True)
    if not rec["finite"]:
        raise RuntimeError(f"Non-finite output: {rec}")
    if rec["euler_vs_grid_teacher"] > GRID_TOL or max(alone.values()) > GRID_TOL:
        raise RuntimeError(f"Teacher or stand-alone identity failed: {rec}")
    if max(rec["qh_of_itself"], rec["r_of_itself"], rec["qh_prompt_frames_are_the_draft"], rec["split_diff"]) > SPLIT_TOL:
        raise RuntimeError(f"Payload or student identity failed: {rec}")
    bad = {a: c for a, c in counts.items() if (c["teacher"], c["student"]) != expected_passes(a, rec["prompt_snr"])}
    if bad:
        raise RuntimeError(f"Forward-pass count differs from the cost table: {bad}")
    q = run_quality(pipe, cg.dev_items()[:1], conds=CONDS, out_dir=sdir, csv_name="quality.csv")
    qrows = rf.read_csv(q)
    repro033 = vp.quality_reproduction(qrows, vp._rows_by_key(EXP033_QUAL_CSV),
                                       {"CPH8": "CPH8", "STU": "STU", "W8": "W8"}, conds=("clean", "noisy15"))
    repro031 = vp.quality_reproduction(qrows, vp._rows_by_key(EXP031_QUAL_CSV), {"STU": "STU", "W8": "W8"},
                                       conds=("clean", "noisy15", "gauss25", "reverb09", "babble15"))
    print(f"[smoke] EXP-033 reproduction: {json.dumps(repro033)}", flush=True)
    print(f"[smoke] EXP-031 reproduction: {json.dumps(repro031)}", flush=True)
    for name, rep in (("EXP-033", repro033), ("EXP-031", repro031)):
        worst = max((v for e in rep.values() for k, v in e.items() if k.startswith("max_abs_")), default=None)
        if worst is None or worst > REPRO_TOL:
            raise RuntimeError(f"{name} reproduction failed (largest difference {worst}): {rep}")
    shaped = {c: next(float(r["prompt_snr"]) for r in qrows if r["condition"] == c) for c in SHAPES}
    run_diversity(pipe, div_items()[:1], sdir, seeds=2, csv_name="diversity.csv")
    run_timing(pipe, time_items()[:1], sdir, csv_name="timing.csv")
    summary = {"checks": rec, "exp033_reproduction": repro033, "exp031_reproduction": repro031,
               "shaped_prompt_snr": shaped, "utc": tb._utc()}
    (sdir / "smoke_check.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print("[OK] smoke passed", flush=True)
    return summary


# --------------------------------------------------------------------------- preregistration
ESTIMANDS = {
    "policy_quality": "Per case, a policy's outcome is the scores of the payload it chooses (CPH8 or CPQH; both are "
                      "synthesized for every case). Paired differences between policies are averaged per speaker over "
                      "the 8 conditions and bootstrapped over the 30 speakers (B = 2000, seed 2026).",
    "detection": "'CPQH needed' = UTMOS(CPQH) - UTMOS(CPH8) > 0.30. AUC of r (high), drift d_4 on the 8-interval grid "
                 "(better direction), the roughness share, Definition 3.3 (low) and the spread-weighted SNR (low), on "
                 "all 240 cases and on the cases with Definition 3.3 >= 20 dB.",
    "wrong_clean_flags": "Clean prompts routed to CPQH by a policy although CPQH is not needed, as a share of clean "
                         "cases. Routes to CPQH where it is needed (a teacher that breaks on a clean prompt) are "
                         "correct catches and do not count.",
    "variety": "Per condition (clean, gauss25), speaker means of CVf0 and aligned CVf0 for each arm and policy (a "
               "policy's 12 outputs are the payloads it chose per seed); the share of CPH8's clean gain over STU kept "
               "by each policy, and the share of P25's gain, with speaker-level bootstrap intervals.",
    "cost": "Forward passes counted by hooks and wall-clock seconds of each arm and policy run on its own, positions "
            "31-40, clean prompts.",
    "calibrated_secondary": "Secondary: lambda for r and for drift re-chosen on positions 31-45 (largest mean UTMOS "
                            "with at most 15% wrong clean flags; drift in its better direction) and applied to 46-60.",
    "spread_snr_gate": "Descriptive (EVAL-027): the policy CPQH below 40 dB of the spread-weighted SNR, else CPH8.",
}
LABEL_RULES = {
    "H1_snr_r_vs_snr_only": "SNR_R - SNR_ONLY: mean UTMOS difference > 0 with the bootstrap lower bound > 0, WER "
                            "difference upper bound <= +0.01 and similarity difference lower bound >= -0.01.",
    "H2_snr_r_vs_snr_drift": "SNR_R - SNR_DRIFT under the same three conditions.",
    "H3_detection": "AUC(r) >= 0.80 on all cases and >= drift's AUC (better direction) + 0.10.",
    "H4_wrong_clean_flags": "SNR_R's wrong clean flags <= 15% of clean cases.",
    "H5_clean_variety": "On clean prompts SNR_R keeps >= 90% of CPH8's gain over STU on both CVf0 and aligned CVf0 "
                        "(point estimates).",
    "verdict": "'snr_r_better' if H1-H5 all hold; 'snr_r_better_quality' if H1 holds and any of H2-H5 fails; "
               "'snr_only_not_beaten' otherwise; 'partial' if the quality or variety stage is incomplete.",
}
LOCAL_SCREEN = ("The thresholds come from local CPU screens on development positions 1-30 (the 12 shortest clean "
                "EXP-033 quality sentences): lambda_r and lambda_d from code/analysis/invariant_residual_screen.py "
                "(its calibration half; output hashed below). Those screens are development screening, not evidence; "
                "this run uses positions 31-60, and 41-60 have not been used in any earlier T4 quality run.")


def build_prereg():
    q = [{"utt_id": it["utt_id"], "speaker_id": it["speaker_id"]} for it in qual_items()]
    d = [{"utt_id": it["utt_id"], "speaker_id": it["speaker_id"]} for it in div_items()]
    t = [{"utt_id": it["utt_id"], "speaker_id": it["speaker_id"]} for it in time_items()]
    inputs = (EXP031_QUAL_CSV, EXP033_QUAL_CSV, SCREEN_JSON)
    payload = {
        "experiment": "EXP-034",
        "task_id": "EVAL-029",
        "title": "The invariant residual against SNR only and Chapter 3's drift: which decision rule is better",
        "framing": ("Development test on new development speakers. Every policy chooses per prompt between the "
                    "teacher's estimate at the commit point (CPH8) and its condition-safe version (CPQH); the "
                    "policies differ only in the decision signal. The invariant residual is the proposed replacement "
                    "for Chapter 3's drift. Not held-out evidence."),
        "commit_time": T_COMMIT,
        "student_levels_after_handover": list(gl.sway_grid(4)[2:4]),
        "qual_arms": list(QUAL_ARMS), "div_arms": list(DIV_ARMS), "policies": list(POLICIES),
        "time_arms": list(TIME_ARMS),
        "thresholds": {"theta_db": THETA, "lambda_r": LAMBDA_R, "lambda_d": LAMBDA_D,
                       "drift_direction": "flag when below", "needed_utmos": NEEDED, "min_voiced": MIN_VOICED},
        "costs": dict(COST), "teacher_passes": dict(TEACHER_PASSES), "student_passes": dict(STUDENT_PASSES),
        "conditions": list(CONDS), "div_conditions": list(DIV_CONDS),
        "shaped_noises": {"pink25": "1/sqrt(max(f, 20 Hz)), 25 dB total-power SNR",
                          "hp25": "1/sqrt(1 + (2000 Hz / f)^8), 25 dB total-power SNR"},
        "positions": {"quality": list(QUAL_POS), "diversity": list(DIV_POS), "timing": list(TIME_POS),
                      "calibration_last": CAL_LAST},
        "n_qual_cases": len(q) * len(CONDS), "qual_utts": q,
        "n_div_speakers": len(d), "div_seeds": DIV_SEEDS, "div_seed_base": DIV_SEED_BASE, "div_utts": d,
        "n_time_cases": len(t), "time_utts": t,
        "bars": {"wer": WER_BAR, "sim": SIM_BAR, "auc_all": AUC_ALL, "auc_gain_over_drift": AUC_GAIN,
                 "max_wrong_clean": MAX_WRONG_CLEAN, "variety_keep": VARIETY_KEEP, "runaway_wer": RUNAWAY_WER},
        "estimands": ESTIMANDS, "label_rules": LABEL_RULES, "local_screen": LOCAL_SCREEN,
        "code_hashes": {name: tb._sha256_file(CODE_DIR / name) for name in DEPENDENCIES},
        "input_hashes": {str(Path(p).relative_to(THESIS_ROOT).as_posix()): tb._sha256_file(p) for p in inputs},
        "seed": SEED, "boot_b": BOOT_B,
    }
    sha = tb._sha256_bytes(json.dumps(payload, sort_keys=True).encode("utf-8"))
    return {**payload, "sha256": sha, "created_utc": tb._utc()}


def freeze(prereg_path=PREREG_PATH, force=False):
    prereg_path = Path(prereg_path)
    if prereg_path.exists() and not force:
        raise RuntimeError(f"Preregistration already exists at {prereg_path}; pass --force to overwrite.")
    prereg_path.parent.mkdir(parents=True, exist_ok=True)
    doc = build_prereg()
    with open(prereg_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(doc, indent=2) + "\n")
    print(f"[OK] Froze {prereg_path} (sha256={doc['sha256'][:12]}...)")
    return doc


def check_prereg(prereg_path=PREREG_PATH):
    prereg_path = Path(prereg_path)
    if not prereg_path.exists():
        raise RuntimeError(f"Missing preregistration: {prereg_path}. Run `freeze` first.")
    doc = json.loads(prereg_path.read_text(encoding="utf-8"))
    payload = {k: v for k, v in doc.items() if k not in ("sha256", "created_utc")}
    if doc.get("sha256") != tb._sha256_bytes(json.dumps(payload, sort_keys=True).encode("utf-8")):
        raise RuntimeError("Preregistration SHA-256 mismatch.")
    for name, rec in doc.get("code_hashes", {}).items():
        actual = tb._sha256_file(CODE_DIR / name)
        if actual != rec:
            raise RuntimeError(f"Code hash mismatch for {name}: expected {rec[:12]}, got {actual[:12]}.")
    for rel, rec in doc.get("input_hashes", {}).items():
        actual = tb._sha256_file(THESIS_ROOT / rel)
        if actual != rec:
            raise RuntimeError(f"Input hash mismatch for {rel}: expected {rec[:12]}, got {actual[:12]}.")
    for key, items in (("qual_utts", qual_items()), ("div_utts", div_items()), ("time_utts", time_items())):
        if doc.get(key) != [{"utt_id": it["utt_id"], "speaker_id": it["speaker_id"]} for it in items]:
            raise RuntimeError(f"Speaker list {key} does not match preregistration.")
    return doc


# --------------------------------------------------------------------------- analysis
def _f(r, key):
    v = r.get(key, "")
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def auc(pos, neg):
    """P(a positive scores above a negative), ties counted half; None when either side is empty."""
    pos = [p for p in pos if p == p]
    neg = [n for n in neg if n == n]
    if not pos or not neg:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def needed(r):
    return _f(r, f"utmos_{B_ARM}") - _f(r, f"utmos_{A_ARM}") > NEEDED


def outcome(r, policy, m, lam_r=LAMBDA_R, lam_d=LAMBDA_D):
    arm = choose(policy, _f(r, "prompt_snr"), _f(r, "r"), _f(r, "drift8"), lam_r, lam_d)
    return _f(r, f"{m}_{arm}"), arm


def policy_value(r, policy, m, **kw):
    if policy in POLICIES:
        return outcome(r, policy, m, **kw)[0]
    if policy == "SW_SNR":
        return _f(r, f"{m}_{B_ARM if _f(r, 'snr_sw') < 40.0 else A_ARM}")
    if policy == "ORACLE":
        return _f(r, f"{m}_{max((A_ARM, B_ARM), key=lambda a: _f(r, f'utmos_{a}'))}")
    return _f(r, f"{m}_{policy}")


def wrong_clean(rows, policy, **kw):
    clean = [r for r in rows if r["condition"] == "clean"]
    if not clean:
        return None
    return sum(outcome(r, policy, "utmos", **kw)[1] == B_ARM and not needed(r) for r in clean) / len(clean)


def paired(rows, a, b, **kw):
    """a - b per case, averaged per speaker over conditions, bootstrapped over speakers."""
    out = {}
    for m in METRICS:
        per = {}
        for r in rows:
            per.setdefault(r["utt_id"], []).append(policy_value(r, a, m, **kw) - policy_value(r, b, m, **kw))
        out[m] = tc.boot_ci([ctp._mean(v) for v in per.values()], B=BOOT_B, seed=SEED)
    out["better"] = bool(out["utmos"]["mean"] > 0 and out["utmos"]["lo"] > 0 and out["wer"]["hi"] <= WER_BAR
                         and out["similarity"]["lo"] >= SIM_BAR)
    return out


def detection(rows):
    pos = [r for r in rows if needed(r)]
    neg = [r for r in rows if not needed(r)]

    def table(p, n):
        e = {"r": auc([_f(x, "r") for x in p], [_f(x, "r") for x in n]),
             "drift8": auc([_f(x, "drift8") for x in p], [_f(x, "drift8") for x in n]),
             "rough_share": auc([_f(x, "rough_share") for x in p], [_f(x, "rough_share") for x in n]),
             "snr_low": auc([-_f(x, "prompt_snr") for x in p], [-_f(x, "prompt_snr") for x in n]),
             "snr_sw_low": auc([-_f(x, "snr_sw") for x in p], [-_f(x, "snr_sw") for x in n])}
        e["drift8_better_direction"] = max(e["drift8"], 1 - e["drift8"]) if e["drift8"] is not None else None
        return e

    gate_open = [r for r in rows if _f(r, "prompt_snr") >= THETA]
    return {"n_needed": len(pos), "n_not_needed": len(neg), "all": table(pos, neg),
            "gate_open": table([r for r in gate_open if needed(r)], [r for r in gate_open if not needed(r)])}


def policy_tables(rows):
    names = POLICIES + ("SW_SNR", "ORACLE") + QUAL_ARMS
    out = {}
    for cond in CONDS + ("all",):
        rs = rows if cond == "all" else [r for r in rows if r["condition"] == cond]
        if not rs:
            continue
        e = {"n": len(rs)}
        for p in names:
            wer = [policy_value(r, p, "wer") for r in rs]
            e[p] = {"utmos": ctp._mean(policy_value(r, p, "utmos") for r in rs),
                    "similarity": ctp._mean(policy_value(r, p, "similarity") for r in rs),
                    "wer_mean": ctp._mean(wer), "wer_median": statistics.median(wer),
                    "runaway": sum(w > RUNAWAY_WER for w in wer)}
            if p in POLICIES:
                arms = [outcome(r, p, "utmos")[1] for r in rs]
                e[p]["flagged"] = sum(a == B_ARM for a in arms) / len(rs)
                e[p]["mean_passes"] = ctp._mean(policy_passes(p, a) for a in arms)
        e["needed"] = sum(needed(r) for r in rs)
        out[cond] = e
    return out


def calibrate(rows, key, direction):
    """Threshold on `key` for SNR + key: the largest mean UTMOS with wrong clean flags <= MAX_WRONG_CLEAN."""
    vals = sorted({_f(r, key) for r in rows if _f(r, key) == _f(r, key)})
    cands = ([float("inf")] + vals[::-1]) if direction > 0 else ([float("-inf")] + vals)
    clean = [r for r in rows if r["condition"] == "clean"]
    best, first = None, None
    for lam in cands:
        def flag(r, lam=lam):
            v = _f(r, key)
            return _f(r, "prompt_snr") < THETA or (v > lam if direction > 0 else v < lam)
        mean = ctp._mean(_f(r, f"utmos_{B_ARM if flag(r) else A_ARM}") for r in rows)
        wrong = sum(flag(r) and not needed(r) for r in clean) / len(clean) if clean else 0.0
        cand = {"lambda": lam, "mean": mean, "wrong_clean": wrong, "constraint_met": wrong <= MAX_WRONG_CLEAN}
        first = first or cand
        if cand["constraint_met"] and (best is None or mean > best["mean"] + 1e-12):
            best = cand
    return best or first        # no threshold meets the clean limit: the SNR-only rule, marked as not meeting it


def calibrated_secondary(rows):
    pos = {it["utt_id"]: i + 1 for i, it in enumerate(cg.dev_items())}
    cal = [r for r in rows if pos.get(r["utt_id"], 0) <= CAL_LAST]
    test = [r for r in rows if pos.get(r["utt_id"], 0) > CAL_LAST]
    if not cal or not test:
        return None
    lam_r = calibrate(cal, "r", +1)
    d_hi, d_lo = calibrate(cal, "drift8", +1), calibrate(cal, "drift8", -1)

    def mean_with(flag):
        return ctp._mean(_f(r, f"utmos_{B_ARM if flag(r) else A_ARM}") for r in test)

    out = {"n_cal": len(cal), "n_test": len(test), "lambda_r": lam_r, "drift_high": d_hi, "drift_low": d_lo,
           "test_utmos": {"SNR_ONLY": mean_with(lambda r: _f(r, "prompt_snr") < THETA),
                          "SNR_R": mean_with(lambda r: _f(r, "prompt_snr") < THETA or _f(r, "r") > lam_r["lambda"])}}
    dbest = d_hi if d_hi["mean"] >= d_lo["mean"] else d_lo
    sign = +1 if dbest is d_hi else -1
    out["test_utmos"]["SNR_DRIFT"] = mean_with(
        lambda r: _f(r, "prompt_snr") < THETA or (_f(r, "drift8") > dbest["lambda"] if sign > 0
                                                  else _f(r, "drift8") < dbest["lambda"]))
    out["test_utmos"]["ORACLE"] = ctp._mean(max(_f(r, f"utmos_{A_ARM}"), _f(r, f"utmos_{B_ARM}")) for r in test)
    return out


def variety(div_rows):
    out = {}
    for cond in DIV_CONDS:
        rows = [r for r in div_rows if r["condition"] == cond]
        if len(rows) < 2:
            continue
        names = DIV_ARMS + POLICIES
        e = {"n": len(rows),
             "means": {a: {m: ctp._mean(_f(r, f"{m}_{a}") for r in rows) for m in DIV_MEASURES} for a in names},
             "flagged": {p: ctp._mean(_f(r, f"flagged_{p}") for r in rows) for p in POLICIES},
             "r_mean": ctp._mean(_f(r, "r_mean") for r in rows)}
        if cond == "clean":
            for m in VARIETY:
                e[f"share_of_CPH8_gain_{m}"] = {p: vp.share_ci(rows, p, m, ref=A_ARM, base="STU")
                                                for p in POLICIES + (B_ARM, "STUR2")}
                e[f"share_of_P25_gain_{m}"] = {p: vp.share_ci(rows, p, m, ref="P25", base="STU")
                                               for p in POLICIES + (A_ARM, B_ARM, "STUR2")}
        else:
            e["voiced_at_least_min"] = {a: sum(_f(r, f"voiced_{a}") >= MIN_VOICED for r in rows) for a in names}
        out[cond] = e
    return out


def timing_table(time_rows):
    if not time_rows:
        return None
    ref = statistics.median(_f(r, "seconds_P25") for r in time_rows)
    out = {"cases": len(time_rows), "arms": {}}
    for a in TIME_ARMS:
        sec = statistics.median(_f(r, f"seconds_{a}") for r in time_rows)
        counted = {(int(_f(r, f"teacher_passes_{a}")), int(_f(r, f"student_passes_{a}"))) for r in time_rows}
        matches = all((int(_f(r, f"teacher_passes_{a}")), int(_f(r, f"student_passes_{a}")))
                      == expected_passes(a, _f(r, "prompt_snr")) for r in time_rows)
        out["arms"][a] = {"passes": COST[a], "median_seconds": sec, "ratio_to_P25": sec / ref if ref > 0 else None,
                          "counted_passes": sorted(list(c) for c in counted), "count_matches_cost": matches}
    return out


def verdict(h, complete):
    if not complete:
        return "partial"
    if all(h.values()):
        return "snr_r_better"
    return "snr_r_better_quality" if h["H1_snr_r_vs_snr_only"] else "snr_only_not_beaten"


def analyze(out_dir=None, prereg_path=PREREG_PATH, allow_partial=False):
    out_dir = Path(out_dir or EXP_DIR / "results")
    q_path, d_path, t_path = out_dir / "quality.csv", out_dir / "diversity.csv", out_dir / "timing.csv"
    if not q_path.exists():
        raise FileNotFoundError(f"Missing quality CSV: {q_path}")
    qrows = rf.read_csv(q_path, repair=True)
    drows = rf.read_csv(d_path, repair=True) if d_path.exists() else []
    trows = rf.read_csv(t_path, repair=True) if t_path.exists() else []
    n_q, n_d = len(qual_items()) * len(CONDS), len(div_items()) * len(DIV_CONDS)
    complete = len(qrows) >= n_q and len(drows) >= n_d
    if not allow_partial and not complete:
        raise RuntimeError(f"Incomplete: quality {len(qrows)}/{n_q}, diversity {len(drows)}/{n_d}")
    det = detection(qrows)
    comps = {"SNR_R_minus_SNR_ONLY": paired(qrows, "SNR_R", "SNR_ONLY"),
             "SNR_R_minus_SNR_DRIFT": paired(qrows, "SNR_R", "SNR_DRIFT"),
             "R_ONLY_minus_SNR_ONLY": paired(qrows, "R_ONLY", "SNR_ONLY"),
             "SNR_DRIFT_minus_SNR_ONLY": paired(qrows, "SNR_DRIFT", "SNR_ONLY"),
             "SNR_R_minus_W8": paired(qrows, "SNR_R", "W8")}
    var = variety(drows)
    keep = (var.get("clean", {}).get("share_of_CPH8_gain_cvf0", {}).get("SNR_R", {}).get("point"),
            var.get("clean", {}).get("share_of_CPH8_gain_cvf0_aligned", {}).get("SNR_R", {}).get("point"))
    a_all, a_drift = det["all"]["r"], det["all"]["drift8_better_direction"]
    wrong = wrong_clean(qrows, "SNR_R")
    h = {"H1_snr_r_vs_snr_only": comps["SNR_R_minus_SNR_ONLY"]["better"],
         "H2_snr_r_vs_snr_drift": comps["SNR_R_minus_SNR_DRIFT"]["better"],
         "H3_detection": a_all is not None and a_drift is not None and a_all >= AUC_ALL and a_all >= a_drift + AUC_GAIN,
         "H4_wrong_clean_flags": wrong is not None and wrong <= MAX_WRONG_CLEAN,
         "H5_clean_variety": all(k is not None and k >= VARIETY_KEEP for k in keep)}
    res = {"experiment": "EXP-034", "partial": not complete, "verdict": verdict(h, complete),
           "hypotheses": {k: bool(v) for k, v in h.items()},
           "n_quality_rows": len(qrows), "n_diversity_rows": len(drows), "n_timing_rows": len(trows),
           "thresholds": {"theta_db": THETA, "lambda_r": LAMBDA_R, "lambda_d": LAMBDA_D},
           "detection": det, "comparisons": comps,
           "wrong_clean_flags": {p: wrong_clean(qrows, p) for p in POLICIES},
           "policies": policy_tables(qrows), "calibrated_secondary": calibrated_secondary(qrows),
           "variety": var, "timing": timing_table(trows), "costs": dict(COST), "utc": tb._utc()}
    rjson, rmd = out_dir / "invariant_residual_results.json", out_dir / "INVARIANT_RESIDUAL_REPORT.md"
    with open(rjson, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(res, indent=2, default=str) + "\n")
    with open(rmd, "w", encoding="utf-8", newline="\n") as f:
        f.write(report(res) + "\n")
    entries = {}
    if Path(prereg_path).exists():
        entries["preregistration"] = {"path": str(prereg_path), "sha256": tb._sha256_file(prereg_path)}
    for p in (q_path, d_path, t_path, rjson, rmd):
        if p.exists():
            entries[p.name] = {"sha256": tb._sha256_file(p), "bytes": p.stat().st_size}
    with open(out_dir / "MANIFEST.json", "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps({"experiment": "EXP-034", "utc": tb._utc(), "files": entries}, indent=2) + "\n")
    print(f"[OK] Analysis complete: verdict={res['verdict']}, report={rmd}", flush=True)
    return res


def _fmt(v, d=3):
    return "n/a" if v is None or v != v else f"{v:.{d}f}"


def report(res):
    lines = ["# EXP-034: the invariant residual against SNR only and Chapter 3's drift", "",
             f"Rows: quality {res['n_quality_rows']} / {len(qual_items()) * len(CONDS)}, diversity "
             f"{res['n_diversity_rows']} / {len(div_items()) * len(DIV_CONDS)}, timing {res['n_timing_rows']} / "
             f"{len(time_items())} (partial={res['partial']}).", "",
             f"**Verdict (frozen rules): `{res['verdict']}`**", "", "| Hypothesis | Holds |", "| --- | --- |"]
    lines += [f"| {k} | {'yes' if v else 'no'} |" for k, v in res["hypotheses"].items()]
    lines += ["", "## Policies against each other (speaker-level bootstrap, 95% intervals)", "",
              "| Comparison | dUTMOS | dWER | dSIM | Better (frozen bars) |", "| --- | --- | --- | --- | --- |"]
    for name, c in res["comparisons"].items():
        lines.append(f"| {name} | {tc._fmt_ci(c['utmos'], 3)} | {tc._fmt_ci(c['wer'], 4)} | "
                     f"{tc._fmt_ci(c['similarity'], 4)} | {'yes' if c['better'] else 'no'} |")
    det = res["detection"]
    lines += ["", f"## Detection of 'CPQH needed' (n = {det['n_needed']} needed, {det['n_not_needed']} not)", "",
              "| Signal | AUC, all cases | AUC, SNR gate open |", "| --- | ---: | ---: |"]
    for k in ("r", "drift8_better_direction", "rough_share", "snr_low", "snr_sw_low"):
        lines.append(f"| {k} | {_fmt(det['all'][k])} | {_fmt(det['gate_open'][k])} |")
    lines += ["", "Wrong clean flags: " + ", ".join(f"{p} {_fmt(v)}" for p, v in res["wrong_clean_flags"].items()), ""]
    pol = res["policies"]
    names = POLICIES + ("SW_SNR", "ORACLE") + QUAL_ARMS
    lines += ["## Mean UTMOS by condition", "", "| Condition | n | " + " | ".join(names) + " |",
              "| --- | ---: |" + " ---: |" * len(names)]
    for cond, e in pol.items():
        lines.append(f"| {cond} | {e['n']} | " + " | ".join(_fmt(e[p]["utmos"]) for p in names) + " |")
    lines += ["", "## Mean WER by condition", "", "| Condition | " + " | ".join(names) + " |",
              "| --- |" + " ---: |" * len(names)]
    for cond, e in pol.items():
        lines.append(f"| {cond} | " + " | ".join(_fmt(e[p]["wer_mean"], 4) for p in names) + " |")
    lines += ["", "## Share of prompts sent to CPQH", "", "| Condition | " + " | ".join(POLICIES) + " |",
              "| --- |" + " ---: |" * len(POLICIES)]
    for cond, e in pol.items():
        lines.append(f"| {cond} | " + " | ".join(_fmt(e[p]["flagged"], 2) for p in POLICIES) + " |")
    for cond, e in (res["variety"] or {}).items():
        names_v = DIV_ARMS + POLICIES
        lines += ["", f"## Variety, `{cond}` (n = {e['n']} speakers x {DIV_SEEDS} seeds)", "",
                  "| Arm or policy | CVf0 | Aligned CVf0 | Voiced | Melody vs P25 |", "| --- | ---: | ---: | ---: | ---: |"]
        for a in names_v:
            mn = e["means"][a]
            lines.append(f"| {a} | {_fmt(mn['cvf0'], 4)} | {_fmt(mn['cvf0_aligned'], 4)} | {_fmt(mn['voiced'], 2)} | "
                         f"{_fmt(mn['melody'], 3)} |")
        if cond == "clean":
            for m in VARIETY:
                lines.append("")
                lines.append(f"Share of CPH8's gain over STU ({m}): " + ", ".join(
                    f"{p} {_fmt(v['point'], 2)}" for p, v in e[f"share_of_CPH8_gain_{m}"].items()))
    t = res["timing"]
    if t:
        lines += ["", f"## Cost (n = {t['cases']} clean cases, each on its own)", "",
                  "| Arm or policy | Passes | Counted = table | Median seconds | Ratio to P25 |",
                  "| --- | ---: | --- | ---: | ---: |"]
        for a, x in t["arms"].items():
            lines.append(f"| {a} | {x['passes']} | {'yes' if x['count_matches_cost'] else 'NO ' + str(x['counted_passes'])} | "
                         f"{_fmt(x['median_seconds'])} | {_fmt(x['ratio_to_P25'], 2)} |")
    lines += ["", "## Secondary: thresholds re-chosen on positions 31-45, applied to 46-60", "", "```",
              json.dumps(res["calibrated_secondary"], indent=2, default=str), "```"]
    return "\n".join(lines)


def make_pipe(args):
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    install_conditions()
    return rf.Pipeline(args.dmo_root, args.device)


def main():
    p = argparse.ArgumentParser(description="EXP-034: the invariant residual against SNR only and drift")
    sub = p.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("freeze")
    f.add_argument("--force", action="store_true")
    for name in ("smoke", "run"):
        s = sub.add_parser(name)
        s.add_argument("--dmo_root", required=True)
        s.add_argument("--out_dir", default=str(EXP_DIR / "results"))
        s.add_argument("--device", default="cuda")
        if name == "run":
            s.add_argument("--stage", choices=("all", "quality", "diversity", "timing"), default="all")
    a = sub.add_parser("analyze")
    a.add_argument("--out_dir", default=str(EXP_DIR / "results"))
    a.add_argument("--allow_partial", action="store_true")
    args = p.parse_args()
    if args.cmd == "freeze":
        freeze(force=args.force)
    elif args.cmd == "smoke":
        smoke(make_pipe(args), args.out_dir)
    elif args.cmd == "run":
        check_prereg()
        pipe = make_pipe(args)
        if args.stage in ("all", "quality"):
            run_quality(pipe, qual_items(), out_dir=args.out_dir)
        if args.stage in ("all", "diversity"):
            run_diversity(pipe, div_items(), out_dir=args.out_dir)
        if args.stage in ("all", "timing"):
            run_timing(pipe, time_items(), out_dir=args.out_dir)
    elif args.cmd == "analyze":
        analyze(args.out_dir, allow_partial=args.allow_partial)


if __name__ == "__main__":
    main()
