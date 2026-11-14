"""Colab T4 experiment for Transmissible Slow-Band Bridge (`TBRG`) & Diagnostics (EXP-028 / EVAL-021).

Implements a theory-guided, inference-time handover algorithm with placement novelty at tau_2 = 0.2929
and no fitted parameters, addressing the quality-diversity trade-off uncovered by EXP-027 on speakers 61-100:
  - 13-NFE `W4` preserves perceptual quality (UTMOS 3.759, dWER +0.0000 vs W8) because the student's
    tau_1 = 0.0761 step produces a denoised acoustic anchor S1, but compresses pitch variety
    (CVf0 0.0770 vs W8R2 0.0893 / T16 0.0911).
  - 12-NFE `W4R2` skips tau_1 and injects T4 at tau_2 = 0.2929, unlocking pitch variety
    (CVf0 0.0859, p = 2.94e-4 vs W4), but loses -0.373 UTMOS (3.404) and inflates slow-mel
    dispersion (1.0299) because raw T4 lacks tau_1 denoising on voice and fine bands.

Primary 13-NFE Inference-Time Handover Algorithm (`TBRG`):
  1. Run guided Euler teacher steps 0..4 (10 NFE) to obtain T4, drifts d_1..d_4, cumulative trajectory
     action A_4 = sum_{j=1}^4 d_j, and contraction rho_4 = d_4 / (d_3 + 1e-8). (Steps 5..6 are also
     evaluated on the shared trajectory to realize the 16-NFE `W6R2` baseline without extra runs.)
  2. Run the student's tau_1 = 0.0761 pass (1 NFE) from T4 to obtain denoised draft S1.
  3. Decompose generated frames plen: of T4 -> (v_T, s_T, f_T) and S1 -> (v_S, s_S, f_S) via
     sh.decompose(..., window=9), then inject Frobenius-matched teacher slow prosody at tau_2 (0 NFE):
       s_tilde_T = (||s_S||_F / (||s_T||_F + 1e-8)) * s_T
       X_TBRG[0, plen:, :] = v_S + s_tilde_T + f_S
  4. Apply standard DMOSpeech 2 flow-matching re-noising at tau_2 = 0.2929 and tau_3 = 0.6173 (2 NFE -> 13 NFE):
       phi_2 = (1 - tau_2) * eps_2 + tau_2 * X_TBRG,   S2 = G_theta(phi_2, tau_2)
       phi_3 = (1 - tau_3) * eps_3 + tau_3 * S2,       S3 = G_theta(phi_3, tau_3)

Ablations & Controls Evaluated on the Same Shared Trajectories:
  - `STS2` (13 NFE): Unnormalized slow-band bridge ablation (s_tilde_T = s_T).
  - `BLND` (13 NFE): Naive 50/50 linear blend 0.5 * S1 + 0.5 * T4 at tau_2 without band decomposition.
  - `SHUF` (13 NFE): Shuffled-speaker slow-band control (donor prompt slow band resampled and
    Frobenius-matched to ||s_S||_F) + per-seed `slow_corr` to verify causal pitch-contour tracking.
  - `W6R2` (16 NFE): Intermediate 6-step teacher (14 NFE) + [tau_2, tau_3] restart (2 NFE).
  - Prospective Unseen-Speaker Replication (`unseen_cal`): Synthesizes W4, W4R2, TBRG, BLND, SHUF,
    W6R2, W8R2, W8 jointly on completely unseen speakers from calibration_manifest.json (disjoint from
    development speakers 1-100), adding matched-seed vs shifted-seed f0 correlation vs W8R2.
  - `STAC-TBRG`: Kept strictly as a secondary exploratory diagnostic (5-fold speaker-CV routing
    between 13-NFE TBRG and stored 20-NFE W8R2).

Subcommands: freeze, smoke, run, analyze.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
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
import run_consistency_gate_prelim as cg  # noqa: E402
import run_followup_tests as rf  # noqa: E402
import run_grid32_prelim as gl  # noqa: E402
import run_noise_guidance_prelim as ng  # noqa: E402
import run_selective_handover_prelim as sh  # noqa: E402
import run_transmissibility_confirm as tc  # noqa: E402
import run_transmissibility_test as tt  # noqa: E402

THESIS_ROOT = rf.THESIS_ROOT
CODE_DIR = Path(__file__).resolve().parent
EXP_DIR = THESIS_ROOT / "research/experiments/EXP-028-transmissibility-bridge"
PREREG_PATH = EXP_DIR / "preregistration.json"
EXP027_RESULTS = THESIS_ROOT / "research/experiments/EXP-027-transmissibility-confirmation/results"
EXP027_DIV_CSV = EXP027_RESULTS / "diversity.csv"
EXP027_QUAL_CSV = EXP027_RESULTS / "quality.csv"

DEPENDENCIES = (
    "run_transmissibility_bridge.py",
    "run_transmissibility_confirm.py",
    "run_transmissibility_test.py",
    "run_selective_handover_prelim.py",
    "run_consistency_gate_prelim.py",
    "run_grid32_prelim.py",
    "run_noise_guidance_prelim.py",
    "run_followup_tests.py",
)

FRESH_N = 40
DIV_PAIRS = 20
DIV_SEEDS = 20
DIV_SEED_BASE = 1000
UNSEEN_QUAL_N = 12
UNSEEN_DIV_N = 8
UNSEEN_SEEDS = 10
FOLDS = 5
SEED = 2026
BOOT_B = 2000
METRICS = ("wer", "similarity", "utmos")
DRIFT_FIELDS = ("d_1", "d_2", "d_3", "d_4", "A_4", "rho_4")

NEW_ARMS = ("TBRG", "STS2", "BLND", "SHUF", "W6R2")
STORED_ARMS = ("student4", "W4", "W8", "W4R2", "W8R2", "T16")
ALL_ARMS = STORED_ARMS + NEW_ARMS
UNSEEN_ARMS = ("W4", "W4R2", "TBRG", "BLND", "SHUF", "W6R2", "W8R2", "W8")
COST = {
    "student4": 4,
    "W4R2": 12,
    "W4": 13,
    "TBRG": 13,
    "STS2": 13,
    "BLND": 13,
    "SHUF": 13,
    "W6R2": 16,
    "W8R2": 20,
    "W8": 21,
    "T16": 32,
}

REPRODUCTION_TOLERANCE = 0.01
SPLIT_STUDENT_TOLERANCE = 1e-5
ZERO_TOL = 1e-12
STAC_UTMOS_TOL = -0.10
STAC_WER_TOL = 0.002
WER_NONINF_MARGIN = 0.01
SIM_NONINF_MARGIN = -0.01
UTMOS_CONFIRM_MIN_MEAN = -0.08
CVF0_W8R2_NONINF_MARGIN = -0.008
UTMOS_W8R2_NONINF_MARGIN = -0.05

DIV_FIELDS = (
    ["utt_id", "speaker_id", "condition", "n_seeds"]
    + [f"{m}_{a}" for a in NEW_ARMS for m in ("cvf0", "voiced", "mel_spread", "slow_corr")]
    + ["case_seconds", "utc"]
)

QUAL_FIELDS = (
    ["utt_id", "speaker_id", "condition", "prompt_snr", "gpu"]
    + list(DRIFT_FIELDS)
    + [f"{m}_{a}" for a in NEW_ARMS for m in METRICS + ("hypothesis",)]
    + ["case_seconds", "utc"]
)

UNSEEN_DIV_FIELDS = (
    ["utt_id", "speaker_id", "condition", "n_seeds"]
    + [f"{m}_{a}" for a in UNSEEN_ARMS for m in ("cvf0", "voiced", "mel_spread", "slow_corr", "f0_corr_w8r2", "f0_shift_corr_w8r2")]
    + ["case_seconds", "utc"]
)

UNSEEN_QUAL_FIELDS = (
    ["utt_id", "speaker_id", "condition", "prompt_snr", "gpu"]
    + list(DRIFT_FIELDS)
    + [f"{m}_{a}" for a in UNSEEN_ARMS for m in METRICS + ("hypothesis",)]
    + ["case_seconds", "utc"]
)


def cost(arm):
    """Total neural function evaluations (NFE) for an arm."""
    return COST[arm]


def div_items(n=DIV_PAIRS):
    """Return the first n fresh speakers (indices 60..79) for paired diversity comparison vs EXP-027."""
    return tc.fresh_items()[:n]


def qual_items(n=FRESH_N):
    """Return all n fresh speakers (indices 60..99) matching EXP-027."""
    return tc.fresh_items(n)


def unseen_qual_items(n=UNSEEN_QUAL_N):
    """Return n calibration speakers strictly disjoint from all 100 development speakers (EXP-012..027)."""
    dev_spks = {it["speaker_id"] for it in cg.dev_items()}
    cal = sorted(rf.split_items("calibration"), key=lambda it: it["utt_id"])
    picked = [it for it in cal if it["speaker_id"] not in dev_spks][:n]
    if len(picked) != n:
        raise RuntimeError(f"Expected {n} unseen calibration speakers, got {len(picked)}")
    return picked


def unseen_div_items(n=UNSEEN_DIV_N):
    """Return the first n unseen calibration speakers for multi-seed prospective replication."""
    return unseen_qual_items(n)


def smoke_items():
    """Return the first 2 fresh speakers with locked EXP-027 stored references for reproduction check."""
    return tc.fresh_items()[:2]


def _utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path):
    return sh.sha256_text(path)


def _append(path, fields, row):
    new = not Path(path).exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new:
            w.writeheader()
        w.writerow(row)
        f.flush()
        os.fsync(f.fileno())


def _max_abs(arr):
    return float(abs(arr).max())


def _frobenius_norm(arr):
    return ((arr * arr).sum()) ** 0.5


# --------------------------------------------------------------------------- core operators
def teacher_steps(pipe, ctx, wanted=(4, 6)):
    """Run guided Euler teacher steps 0..max(wanted) and return ({step: ep}, step4_drift_metrics)."""
    torch, sampler = pipe.torch, pipe.sampler
    grid = tt.grid16(pipe)
    plen = ctx["prompt_len"]
    kmax = max(wanted)
    if kmax < 4:
        raise ValueError(f"teacher_steps requires max(wanted) >= 4 to compute drifts d_1..d_4; got {wanted}")
    wanted_set = set(wanted)
    x = sh._clone(ctx["y0"])
    prev = None
    drifts = {}
    ends = {}
    with torch.inference_mode():
        for i in range(kmax + 1):
            t = torch.tensor(grid[i], device=pipe.device, dtype=torch.float32)
            kw = dict(x=x, cond=ctx["step_cond"], text=ctx["text_ids"], time=t, mask=None)
            vc = pipe.tts.teacher(drop_audio_cond=False, drop_text=False, **kw)
            vu = pipe.tts.teacher(drop_audio_cond=True, drop_text=True, **kw)
            v = vc + (vc - vu) * sampler.CFG_STRENGTH
            ep = x + (1.0 - grid[i]) * v
            if prev is not None and i <= 4:
                drifts[i] = float(sampler.compute_stabilization_drift(ep, prev, plen))
            if i in wanted_set:
                ends[i] = ep
            prev = ep
            if i < kmax:
                x = x + (grid[i + 1] - grid[i]) * v
    d1, d2, d3, d4 = drifts[1], drifts[2], drifts[3], drifts[4]
    metrics = {
        "d_1": d1,
        "d_2": d2,
        "d_3": d3,
        "d_4": d4,
        "A_4": d1 + d2 + d3 + d4,
        "rho_4": d4 / (d3 + 1e-8),
    }
    return ends, metrics


def teacher_step4(pipe, ctx):
    """Run guided Euler teacher steps 0..4 (10 NFE) and return (T4, drift_metrics)."""
    ends, metrics = teacher_steps(pipe, ctx, wanted=(4,))
    return ends[4], metrics


def student_step_tau1(pipe, ctx, T4):
    """Run the student's tau_1 = 0.0761 step from T4 to obtain clean draft S1 (1 NFE)."""
    tau1 = float(pipe.sampler.STUDENT_TAUS[0])
    return gl.student(pipe, ctx, T4, [tau1])


def resample_time(x, target_len):
    """Linearly interpolate a 2D (frames, bands) torch tensor or numpy array along axis 0."""
    if target_len <= 0:
        raise ValueError(f"target_len must be positive, got {target_len}")
    src_len = x.shape[0]
    if src_len == target_len:
        return sh._clone(x)
    if target_len == 1:
        return sh._clone(x[:1])
    if hasattr(x, "clone"):
        import torch.nn.functional as F

        if src_len == 1:
            return x[:1].repeat(target_len, 1)
        return F.interpolate(x.T.unsqueeze(0), size=target_len, mode="linear", align_corners=True).squeeze(0).T
    import numpy as np

    if src_len == 1:
        return np.repeat(x[:1], target_len, axis=0)
    pos = np.linspace(0.0, src_len - 1, target_len)
    i0 = np.floor(pos).astype(int)
    i1 = np.minimum(i0 + 1, src_len - 1)
    w = (pos - i0)[:, None]
    return (1.0 - w) * x[i0] + w * x[i1]


def bridge_handover(T4, S1, plen, energy_match=True, window=sh.SLOW_WINDOW):
    """Inject teacher slow prosody (T4) into student draft S1 on generated frames."""
    _, s_t, _ = sh.decompose(T4[0, plen:, :], window)
    v_s, s_s, f_s = sh.decompose(S1[0, plen:, :], window)
    if energy_match:
        norm_s = _frobenius_norm(s_s)
        norm_t = _frobenius_norm(s_t)
        s_bridged = s_t * (norm_s / (norm_t + 1e-8))
    else:
        s_bridged = s_t
    out = sh._clone(S1)
    out[0, plen:, :] = v_s + s_bridged + f_s
    return out


def blend_handover(T4, S1, plen, alpha=0.5):
    """Simple fixed linear blend of S1 and T4 on generated frames at tau_2 without decomposition."""
    out = sh._clone(S1)
    out[0, plen:, :] = (1.0 - alpha) * S1[0, plen:, :] + alpha * T4[0, plen:, :]
    return out


def shuf_handover(donor_mel_2d, S1, plen, window=sh.SLOW_WINDOW):
    """Inject mismatched donor speaker slow band (resampled & Frobenius-matched to ||s_S||) onto S1."""
    if hasattr(S1, "device") and hasattr(donor_mel_2d, "to"):
        donor_mel_2d = donor_mel_2d.to(device=S1.device, dtype=S1.dtype)
    _, s_donor, _ = sh.decompose(donor_mel_2d, window)
    t_gen = S1.shape[1] - plen
    s_donor_res = resample_time(s_donor, t_gen)
    v_s, s_s, f_s = sh.decompose(S1[0, plen:, :], window)
    norm_s = _frobenius_norm(s_s)
    norm_d = _frobenius_norm(s_donor_res)
    s_matched = s_donor_res * (norm_s / (norm_d + 1e-8))
    out = sh._clone(S1)
    out[0, plen:, :] = v_s + s_matched + f_s
    return out


def slow_cosine(s_a, s_b):
    """Frobenius cosine correlation between two (frames, bands) slow-band arrays."""
    na = _frobenius_norm(s_a)
    nb = _frobenius_norm(s_b)
    if float(na) <= 1e-12 or float(nb) <= 1e-12:
        return 0.0
    return float((s_a * s_b).sum() / (na * nb))


def slow_band_corr(mel_a, mel_b, plen, window=sh.SLOW_WINDOW):
    """Cosine correlation between generated slow bands of two (1, frames, bands) mels."""
    _, s_a, _ = sh.decompose(mel_a[0, plen:, :], window)
    _, s_b, _ = sh.decompose(mel_b[0, plen:, :], window)
    return slow_cosine(s_a, s_b)


def paired_f0_corr(tracks_a, tracks_b, min_frames=5):
    """Mean Pearson correlation of overlapping voiced f0(t) contours across paired seed tracks."""
    import numpy as np

    corrs = []
    for ta, tb in zip(tracks_a, tracks_b):
        a = np.asarray(ta, dtype=float)
        b = np.asarray(tb, dtype=float)
        n = min(len(a), len(b))
        if n < min_frames:
            continue
        a, b = a[:n], b[:n]
        mask = np.isfinite(a) & np.isfinite(b)
        if int(mask.sum()) < min_frames:
            continue
        av = a[mask] - a[mask].mean()
        bv = b[mask] - b[mask].mean()
        da = float(np.sqrt((av * av).sum()))
        db = float(np.sqrt((bv * bv).sum()))
        if da > 1e-12 and db > 1e-12:
            corrs.append(float((av * bv).sum() / (da * db)))
    return float(np.mean(corrs)) if corrs else float("nan")


def bridge_outputs(pipe, ctx, T4, T6=None, donor_mel=None):
    """Synthesize all new EXP-028 candidate and control mels with explicit tau_2 re-noising."""
    taus_tail = list(pipe.sampler.STUDENT_TAUS)[1:]
    plen = ctx["prompt_len"]
    s1 = student_step_tau1(pipe, ctx, T4)
    x_tbrg = bridge_handover(T4, s1, plen, energy_match=True)
    x_sts2 = bridge_handover(T4, s1, plen, energy_match=False)
    x_blnd = blend_handover(T4, s1, plen, alpha=0.5)
    out = {
        "TBRG": gl.student(pipe, ctx, x_tbrg, taus_tail),
        "STS2": gl.student(pipe, ctx, x_sts2, taus_tail),
        "BLND": gl.student(pipe, ctx, x_blnd, taus_tail),
        "S1": s1,
    }
    if donor_mel is not None:
        x_shuf = shuf_handover(donor_mel, s1, plen)
        out["SHUF"] = gl.student(pipe, ctx, x_shuf, taus_tail)
    if T6 is not None:
        out["W6R2"] = gl.student(pipe, ctx, T6, taus_tail)
    return out


# --------------------------------------------------------------------------- synthesis helpers
def _donor_prompt_mel(pipe, items, idx, out_dir, cache):
    if len(items) < 2:
        raise ValueError("Need at least 2 speakers to assign a strictly disjoint SHUF donor.")
    donor = items[(idx + 1) % len(items)]
    du = donor["utt_id"]
    if du not in cache:
        d_ctx = pipe.prepare(donor, "clean", int(donor["input_seed"]), out_dir)
        cache[du] = sh._clone(d_ctx["step_cond"][0, :d_ctx["prompt_len"], :])
    return cache[du]


def _track_arms(pipe, ctx, outs, arms):
    audios = {arm: ng.decode(pipe, ctx, outs[arm]) for arm in arms}
    with ThreadPoolExecutor(max_workers=min(len(arms), 4)) as ex:
        futs = {arm: ex.submit(sh.f0_track, audios[arm]) for arm in arms}
        return {arm: futs[arm].result() for arm in arms}


# --------------------------------------------------------------------------- synthesis loops
def run_diversity(pipe, items, out_dir, seeds=DIV_SEEDS, csv_name="diversity.csv"):
    """Run multi-seed synthesis for TBRG, STS2, BLND, SHUF, W6R2 and log CVf0, spread & slow_corr."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / csv_name
    done = rf.done_keys(path, 1)
    todo = [(idx, it) for idx, it in enumerate(items) if (it["utt_id"], "clean") not in done]
    print(f"[diversity] {len(todo)} speakers x {seeds} seeds to run ({len(done)} already in {path.name})", flush=True)
    donor_cache = {}

    for run_idx, (idx, item) in enumerate(todo, 1):
        t0 = time.perf_counter()
        donor_mel = _donor_prompt_mel(pipe, items, idx, out_dir, donor_cache)
        tracks = {a: [] for a in NEW_ARMS}
        slow_mels = {a: [] for a in NEW_ARMS}
        slow_corrs = {a: [] for a in NEW_ARMS}
        for m in range(seeds):
            seed = int(item["input_seed"]) + DIV_SEED_BASE + m
            ctx = pipe.prepare(item, "clean", seed, out_dir, noise_seed=item["input_seed"])
            ends, _ = teacher_steps(pipe, ctx, wanted=(4, 6))
            t4, t6 = ends[4], ends[6]
            outs = bridge_outputs(pipe, ctx, t4, T6=t6, donor_mel=donor_mel)
            plen = ctx["prompt_len"]
            s_t4 = tc._slow_component(t4, plen)
            arm_tracks = _track_arms(pipe, ctx, outs, NEW_ARMS)
            for arm in NEW_ARMS:
                s_arm = tc._slow_component(outs[arm], plen)
                slow_mels[arm].append(s_arm)
                slow_corrs[arm].append(slow_cosine(s_arm, s_t4))
                tracks[arm].append(arm_tracks[arm])

        row = {"utt_id": item["utt_id"], "speaker_id": item["speaker_id"], "condition": "clean", "n_seeds": seeds}
        for arm in NEW_ARMS:
            row[f"cvf0_{arm}"], row[f"voiced_{arm}"] = sh.cvf0(tracks[arm])
            row[f"mel_spread_{arm}"] = tc.slow_mel_spread(slow_mels[arm])
            row[f"slow_corr_{arm}"] = statistics.mean(slow_corrs[arm])
        row.update({"case_seconds": round(time.perf_counter() - t0, 2), "utc": _utc()})
        _append(path, DIV_FIELDS, row)
        print(f"[{run_idx}/{len(todo)}] diversity: {item['utt_id']} ({row['case_seconds']:.1f} s)", flush=True)
    print(f"[OK] diversity complete: {path}", flush=True)
    return path


def run_quality(pipe, items, out_dir, csv_name="quality.csv"):
    """Synthesize and score single-seed quality plus Step-4 drift features on clean prompts."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / csv_name
    done = rf.done_keys(path, 1)
    todo = [(idx, it) for idx, it in enumerate(items) if (it["utt_id"], "clean") not in done]
    print(f"[quality] {len(todo)} speakers to run ({len(done)} already in {path.name})", flush=True)
    donor_cache = {}

    for run_idx, (idx, item) in enumerate(todo, 1):
        t0 = time.perf_counter()
        donor_mel = _donor_prompt_mel(pipe, items, idx, out_dir, donor_cache)
        ctx = pipe.prepare(item, "clean", int(item["input_seed"]), out_dir)
        ends, drift = teacher_steps(pipe, ctx, wanted=(4, 6))
        t4, t6 = ends[4], ends[6]
        outs = bridge_outputs(pipe, ctx, t4, T6=t6, donor_mel=donor_mel)
        gt = pipe.ground_truth(item)
        row = {
            "utt_id": item["utt_id"],
            "speaker_id": item["speaker_id"],
            "condition": "clean",
            "prompt_snr": ctx["prompt_snr"],
            "gpu": pipe.gpu,
            **drift,
        }
        for arm in NEW_ARMS:
            m = pipe.score(ng.decode(pipe, ctx, outs[arm]), item, gt)
            row.update({f"{k}_{arm}": m[k] for k in METRICS + ("hypothesis",)})
        row.update({"case_seconds": round(time.perf_counter() - t0, 2), "utc": _utc()})
        _append(path, QUAL_FIELDS, row)
        print(f"[{run_idx}/{len(todo)}] quality: {item['utt_id']} ({row['case_seconds']:.1f} s)", flush=True)
    print(f"[OK] quality complete: {path}", flush=True)
    return path


def run_unseen_replication(
    pipe,
    out_dir,
    div_items_list=None,
    qual_items_list=None,
    seeds=UNSEEN_SEEDS,
):
    """Run prospective replication on unseen calibration speakers disjoint from dev speakers 1-100."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if div_items_list is None:
        div_items_list = unseen_div_items()
    if qual_items_list is None:
        qual_items_list = unseen_qual_items()

    div_path = out_dir / "unseen_diversity.csv"
    qual_path = out_dir / "unseen_quality.csv"
    done_d = rf.done_keys(div_path, 1)
    todo_d = [(idx, it) for idx, it in enumerate(div_items_list) if (it["utt_id"], "clean") not in done_d]
    print(
        f"[unseen_diversity] {len(todo_d)} speakers x {seeds} seeds ({len(done_d)} already in {div_path.name})",
        flush=True,
    )
    donor_cache = {}
    taus = list(pipe.sampler.STUDENT_TAUS)

    for run_idx, (idx, item) in enumerate(todo_d, 1):
        t0 = time.perf_counter()
        donor_mel = _donor_prompt_mel(pipe, div_items_list, idx, out_dir, donor_cache)
        tracks = {a: [] for a in UNSEEN_ARMS}
        slow_mels = {a: [] for a in UNSEEN_ARMS}
        slow_corrs = {a: [] for a in UNSEEN_ARMS}
        for m in range(seeds):
            seed = int(item["input_seed"]) + DIV_SEED_BASE + m
            ctx = pipe.prepare(item, "clean", seed, out_dir, noise_seed=item["input_seed"])
            ends, _ = teacher_steps(pipe, ctx, wanted=(4, 6, 8))
            t4, t6, t8 = ends[4], ends[6], ends[8]
            outs = bridge_outputs(pipe, ctx, t4, T6=t6, donor_mel=donor_mel)
            outs["W4"] = gl.student(pipe, ctx, outs["S1"], taus[1:])
            outs["W4R2"] = gl.student(pipe, ctx, t4, taus[1:])
            outs["W8R2"] = gl.student(pipe, ctx, t8, taus[1:])
            outs["W8"] = gl.student(pipe, ctx, t8, taus)
            plen = ctx["prompt_len"]
            s_t4 = tc._slow_component(t4, plen)
            arm_tracks = _track_arms(pipe, ctx, outs, UNSEEN_ARMS)
            for arm in UNSEEN_ARMS:
                s_arm = tc._slow_component(outs[arm], plen)
                slow_mels[arm].append(s_arm)
                slow_corrs[arm].append(slow_cosine(s_arm, s_t4))
                tracks[arm].append(arm_tracks[arm])

        ref_tracks = tracks["W8R2"]
        shifted_ref = ref_tracks[1:] + ref_tracks[:1]
        row = {"utt_id": item["utt_id"], "speaker_id": item["speaker_id"], "condition": "clean", "n_seeds": seeds}
        for arm in UNSEEN_ARMS:
            row[f"cvf0_{arm}"], row[f"voiced_{arm}"] = sh.cvf0(tracks[arm])
            row[f"mel_spread_{arm}"] = tc.slow_mel_spread(slow_mels[arm])
            row[f"slow_corr_{arm}"] = statistics.mean(slow_corrs[arm])
            row[f"f0_corr_w8r2_{arm}"] = paired_f0_corr(tracks[arm], ref_tracks)
            row[f"f0_shift_corr_w8r2_{arm}"] = paired_f0_corr(tracks[arm], shifted_ref)
        row.update({"case_seconds": round(time.perf_counter() - t0, 2), "utc": _utc()})
        _append(div_path, UNSEEN_DIV_FIELDS, row)
        print(f"[{run_idx}/{len(todo_d)}] unseen_div: {item['utt_id']} ({row['case_seconds']:.1f} s)", flush=True)

    done_q = rf.done_keys(qual_path, 1)
    todo_q = [(idx, it) for idx, it in enumerate(qual_items_list) if (it["utt_id"], "clean") not in done_q]
    print(
        f"[unseen_quality] {len(todo_q)} speakers ({len(done_q)} already in {qual_path.name})",
        flush=True,
    )
    for run_idx, (idx, item) in enumerate(todo_q, 1):
        t0 = time.perf_counter()
        donor_mel = _donor_prompt_mel(pipe, qual_items_list, idx, out_dir, donor_cache)
        ctx = pipe.prepare(item, "clean", int(item["input_seed"]), out_dir)
        ends, drift = teacher_steps(pipe, ctx, wanted=(4, 6, 8))
        t4, t6, t8 = ends[4], ends[6], ends[8]
        outs = bridge_outputs(pipe, ctx, t4, T6=t6, donor_mel=donor_mel)
        outs["W4"] = gl.student(pipe, ctx, outs["S1"], taus[1:])
        outs["W4R2"] = gl.student(pipe, ctx, t4, taus[1:])
        outs["W8R2"] = gl.student(pipe, ctx, t8, taus[1:])
        outs["W8"] = gl.student(pipe, ctx, t8, taus)
        gt = pipe.ground_truth(item)
        row = {
            "utt_id": item["utt_id"],
            "speaker_id": item["speaker_id"],
            "condition": "clean",
            "prompt_snr": ctx["prompt_snr"],
            "gpu": pipe.gpu,
            **drift,
        }
        for arm in UNSEEN_ARMS:
            m = pipe.score(ng.decode(pipe, ctx, outs[arm]), item, gt)
            row.update({f"{k}_{arm}": m[k] for k in METRICS + ("hypothesis",)})
        row.update({"case_seconds": round(time.perf_counter() - t0, 2), "utc": _utc()})
        _append(qual_path, UNSEEN_QUAL_FIELDS, row)
        print(f"[{run_idx}/{len(todo_q)}] unseen_qual: {item['utt_id']} ({row['case_seconds']:.1f} s)", flush=True)

    print(f"[OK] unseen replication complete: {div_path}, {qual_path}", flush=True)
    return div_path, qual_path


def smoke(pipe, out_dir, prereg_path=PREREG_PATH, exp027_qual_csv=EXP027_QUAL_CSV):
    """Run 2-speaker verification: split-student identity, EXP-027 W4 reproduction, and smoke loops."""
    check_prereg(prereg_path)
    out_dir = Path(out_dir)
    sdir = out_dir / "smoke"
    sdir.mkdir(parents=True, exist_ok=True)
    items = smoke_items()
    stored_qual = {r["utt_id"]: r for r in rf.read_csv(exp027_qual_csv)}
    taus = list(pipe.sampler.STUDENT_TAUS)
    repro = []

    for item in items:
        ctx = pipe.prepare(item, "clean", int(item["input_seed"]), sdir)
        t4, _ = teacher_step4(pipe, ctx)
        s1 = student_step_tau1(pipe, ctx, t4)
        w4_split = gl.student(pipe, ctx, s1, taus[1:])
        w4_full = gl.student(pipe, ctx, t4, taus)
        split_diff = _max_abs(w4_split - w4_full)
        if split_diff > SPLIT_STUDENT_TOLERANCE:
            raise RuntimeError(
                f"Split-student identity failed on {item['utt_id']}: max_abs_diff={split_diff:.2e} "
                f"(tol {SPLIT_STUDENT_TOLERANCE:.2e})"
            )

        m = pipe.score(ng.decode(pipe, ctx, w4_split), item, pipe.ground_truth(item))
        ref = stored_qual[item["utt_id"]]
        d_wer = abs(m["wer"] - float(ref["wer_W4"]))
        d_sim = abs(m["similarity"] - float(ref["similarity_W4"]))
        if d_wer > REPRODUCTION_TOLERANCE or d_sim > REPRODUCTION_TOLERANCE:
            raise RuntimeError(
                f"EXP-027 W4 reproduction failed on {item['utt_id']}: "
                f"d_wer={d_wer:.4f}, d_sim={d_sim:.4f} (tol {REPRODUCTION_TOLERANCE})"
            )
        repro.append({
            "utt_id": item["utt_id"],
            "max_abs_split_diff": split_diff,
            "d_wer": d_wer,
            "d_sim": d_sim,
        })

    run_diversity(pipe, items, sdir, seeds=2, csv_name="diversity.csv")
    run_quality(pipe, items, sdir, csv_name="quality.csv")
    summary = {
        "repro": repro,
        "n_smoke_items": len(items),
        "max_abs_split_diff": max(r["max_abs_split_diff"] for r in repro),
        "max_d_wer": max(r["d_wer"] for r in repro),
        "max_d_sim": max(r["d_sim"] for r in repro),
        "utc": _utc(),
    }
    (sdir / "smoke_check.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(
        f"[OK] smoke passed (max split diff={summary['max_abs_split_diff']:.2e}, "
        f"max d_wer={summary['max_d_wer']:.4f}, max d_sim={summary['max_d_sim']:.4f})",
        flush=True,
    )
    return summary


# --------------------------------------------------------------------------- preregistration
def build_prereg(fresh=None, unseen=None, exp027_div_csv=EXP027_DIV_CSV, exp027_qual_csv=EXP027_QUAL_CSV):
    """Build deterministic preregistration payload with dependency, cohort, and EXP-027 hashes."""
    if fresh is None:
        fresh = qual_items()
    if unseen is None:
        unseen = unseen_qual_items()
    fresh_utts = [{"utt_id": it["utt_id"], "speaker_id": it["speaker_id"]} for it in fresh]
    unseen_utts = [{"utt_id": it["utt_id"], "speaker_id": it["speaker_id"]} for it in unseen]
    code_hashes = {name: _sha256_file(CODE_DIR / name) for name in DEPENDENCIES}
    input_hashes = {
        str(Path(exp027_div_csv).relative_to(THESIS_ROOT).as_posix()): _sha256_file(exp027_div_csv),
        str(Path(exp027_qual_csv).relative_to(THESIS_ROOT).as_posix()): _sha256_file(exp027_qual_csv),
    }
    fresh_sha = _sha256_bytes(json.dumps(fresh_utts, sort_keys=True).encode("utf-8"))
    unseen_sha = _sha256_bytes(json.dumps(unseen_utts, sort_keys=True).encode("utf-8"))
    payload = {
        "experiment": "EXP-028",
        "task_id": "EVAL-021",
        "title": "Transmissible Slow-Band Bridge (TBRG) with Causal & Cheap Controls on 61-100 + Unseen Calibration Cohort",
        "framing": "theory-guided inference-time handover algorithm; placement novelty at tau_2 = 0.2929; no fitted parameters",
        "tau2_renoising": "phi_2 = (1 - tau_2) * eps_2 + tau_2 * X_tau2^TBRG at tau_2 = 0.2929",
        "n_fresh_qual": len(fresh_utts),
        "n_fresh_div": DIV_PAIRS,
        "fresh_utts": fresh_utts,
        "fresh_utts_sha256": fresh_sha,
        "div_seeds": DIV_SEEDS,
        "div_seed_base": DIV_SEED_BASE,
        "n_unseen_qual": len(unseen_utts),
        "n_unseen_div": UNSEEN_DIV_N,
        "unseen_seeds": UNSEEN_SEEDS,
        "unseen_utts": unseen_utts,
        "unseen_utts_sha256": unseen_sha,
        "new_arms": list(NEW_ARMS),
        "stored_arms": list(STORED_ARMS),
        "unseen_arms": list(UNSEEN_ARMS),
        "costs": dict(COST),
        "cvf0_w8r2_noninf_margin": CVF0_W8R2_NONINF_MARGIN,
        "utmos_w8r2_noninf_margin": UTMOS_W8R2_NONINF_MARGIN,
        "utmos_confirm_min_mean": UTMOS_CONFIRM_MIN_MEAN,
        "wer_noninf_margin": WER_NONINF_MARGIN,
        "sim_noninf_margin": SIM_NONINF_MARGIN,
        "stac_folds": FOLDS,
        "stac_utmos_tol": STAC_UTMOS_TOL,
        "stac_wer_tol": STAC_WER_TOL,
        "boot_b": BOOT_B,
        "seed": SEED,
        "code_hashes": code_hashes,
        "input_hashes": input_hashes,
        "hypotheses": {
            "H1": (
                f"CVf0(TBRG) - CVf0(W4) > 0 (lower 95% CI > 0 & no-top2 > 0) AND "
                f"CVf0(TBRG) - CVf0(W8R2) >= {CVF0_W8R2_NONINF_MARGIN:+.3f} (lower 95% CI)"
            ),
            "H2": (
                f"UTMOS(TBRG) - UTMOS(W4R2) > 0 (lower 95% CI > 0 & no-top2 > 0) AND "
                f"UTMOS(TBRG) - UTMOS(W8R2) >= {UTMOS_W8R2_NONINF_MARGIN:+.2f} (lower 95% CI)"
            ),
            "H3": (
                f"TBRG preserves clean W8 quality (upper CI dWER <= +{WER_NONINF_MARGIN:.2f}, "
                f"lower CI dSIM >= {SIM_NONINF_MARGIN:+.2f}, mean dUTMOS >= {UTMOS_CONFIRM_MIN_MEAN:+.2f})"
            ),
            "H4": "mel_spread(TBRG) - mel_spread(W4R2) < 0 (upper 95% CI < 0; prevents artifact dispersion)",
            "H5": (
                "CVf0(TBRG) - CVf0(SHUF) > 0 (lower 95% CI > 0; causal tracking of matching slow band) AND "
                "TBRG beats naive linear blend BLND on UTMOS or CVf0 (lower 95% CI > 0)"
            ),
        },
        "verdict_rule": (
            "refuted if CVf0(TBRG-W4) upper CI < 0, CVf0(TBRG-W8R2) upper CI < -0.008, "
            "UTMOS(TBRG-W4R2) upper CI < 0, or TBRG fails W8 WER/SIM non-inferiority; "
            "confirmed if H1..H5 all clear frozen bounds (including W8R2 non-inferiority, mean dUTMOS >= -0.08, "
            "TBRG > SHUF on CVf0, and TBRG > BLND on UTMOS or CVf0); "
            "supported_direction_only if CVf0(TBRG-W4) lower CI > 0 and UTMOS(TBRG-W4R2) lower CI > 0 "
            "without clearing all stricter bars; mixed otherwise."
        ),
    }
    sha = _sha256_bytes(json.dumps(payload, sort_keys=True).encode("utf-8"))
    return {**payload, "sha256": sha, "created_utc": _utc()}


def freeze(prereg_path=PREREG_PATH, force=False):
    """Write preregistration.json unless it already exists (use --force to overwrite)."""
    prereg_path = Path(prereg_path)
    if prereg_path.exists() and not force:
        raise RuntimeError(f"Preregistration already exists at {prereg_path}; pass --force to overwrite.")
    prereg_path.parent.mkdir(parents=True, exist_ok=True)
    doc = build_prereg()
    prereg_path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    print(f"[OK] Froze {prereg_path} (sha256={doc['sha256'][:12]}...)")
    return doc


def check_prereg(prereg_path=PREREG_PATH):
    """Verify preregistration integrity, code hashes, locked EXP-027 input hashes, and both speaker manifests."""
    prereg_path = Path(prereg_path)
    if not prereg_path.exists():
        raise RuntimeError(f"Missing preregistration: {prereg_path}. Run `freeze` first.")
    doc = json.loads(prereg_path.read_text(encoding="utf-8"))
    payload = {k: v for k, v in doc.items() if k not in ("sha256", "created_utc")}
    expected_sha = _sha256_bytes(json.dumps(payload, sort_keys=True).encode("utf-8"))
    if doc.get("sha256") != expected_sha:
        raise RuntimeError("Preregistration SHA-256 mismatch (payload was modified after freezing).")
    for name, recorded in doc.get("code_hashes", {}).items():
        actual = _sha256_file(CODE_DIR / name)
        if actual != recorded:
            raise RuntimeError(f"Code hash mismatch for {name}: expected {recorded[:12]}, got {actual[:12]}.")
    for rel, recorded in doc.get("input_hashes", {}).items():
        inp = THESIS_ROOT / rel
        if not inp.exists():
            raise RuntimeError(f"Missing locked EXP-027 input file: {rel}")
        actual = _sha256_file(inp)
        if actual != recorded:
            raise RuntimeError(f"EXP-027 input hash mismatch for {rel}: expected {recorded[:12]}, got {actual[:12]}.")
    current_utts = [{"utt_id": it["utt_id"], "speaker_id": it["speaker_id"]} for it in qual_items()]
    if doc.get("fresh_utts") != current_utts:
        raise RuntimeError("Fresh speaker list does not match preregistration manifest.")
    current_unseen = [{"utt_id": it["utt_id"], "speaker_id": it["speaker_id"]} for it in unseen_qual_items()]
    if doc.get("unseen_utts") != current_unseen:
        raise RuntimeError("Unseen calibration speaker list does not match preregistration manifest.")
    return doc


# --------------------------------------------------------------------------- statistical analysis
def _summarize(vals, seed=SEED):
    return {
        "boot": tc.boot_ci(vals, B=BOOT_B, seed=seed),
        "sign": tc.sign_test(vals),
        "robust": tc.robust_stats(vals),
    }


def _paired_contrast(rows_a, rows_b, key_a, key_b):
    """Return paired differences row_a[key_a] - row_b[key_b] aligned on utt_id."""
    vals = []
    for u, ra in rows_a.items():
        rb = rows_b.get(u)
        if rb is None:
            continue
        try:
            va = float(ra[key_a])
            vb = float(rb[key_b])
        except (KeyError, TypeError, ValueError):
            continue
        if not math.isnan(va) and not math.isnan(vb):
            vals.append(va - vb)
    return vals


def _speaker_folds(qual_rows, n_folds=FOLDS, seed=SEED):
    spks = sorted({r["speaker_id"] for r in qual_rows})
    k = min(n_folds, len(spks))
    perm = list(spks)
    random.Random(seed).shuffle(perm)
    return {s: idx % k for idx, s in enumerate(perm)}, k


def _stac_policy_eval(qual_rows, div_map, stored_qual, stored_div, feature="relative_r4"):
    """5-fold speaker-CV exploratory routing between TBRG (13 NFE) and W8R2 (20 NFE)."""
    folds, k = _speaker_folds(qual_rows)
    if k < 2:
        return None

    records = []
    for r in qual_rows:
        u = r["utt_id"]
        sq = stored_qual.get(u)
        if sq is None:
            continue
        d4 = float(r["d_4"])
        a4 = float(r["A_4"])
        score = (d4 / (a4 + 1e-8)) if feature == "relative_r4" else a4
        records.append({
            "utt_id": u,
            "spk": r["speaker_id"],
            "score": score,
            "tbrg": {m: float(r[f"{m}_TBRG"]) for m in METRICS},
            "w8r2": {m: float(sq[f"{m}_W8R2"]) for m in METRICS},
            "w8": {m: float(sq[f"{m}_W8"]) for m in METRICS},
        })

    if not records:
        return None

    def route(rec, tau):
        if tau is None:
            return "W8R2"
        take_tbrg = (rec["score"] <= tau) if feature == "relative_r4" else (rec["score"] >= tau)
        return "TBRG" if take_tbrg else "W8R2"

    chosen_folds = []
    oof_pick = {}
    for f in range(k):
        train = [r for r in records if folds[r["spk"]] != f]
        test = [r for r in records if folds[r["spk"]] == f]
        uniq = sorted({r["score"] for r in train}, reverse=(feature != "relative_r4"))
        cands = [None] + uniq
        best_tau = None
        best_nfe = float("inf")
        for tau in cands:
            picks = [route(r, tau) for r in train]
            mean_dutmos = statistics.mean(
                (r["tbrg"]["utmos"] if p == "TBRG" else r["w8r2"]["utmos"]) - r["w8"]["utmos"]
                for r, p in zip(train, picks)
            )
            mean_dwer = statistics.mean(
                (r["tbrg"]["wer"] if p == "TBRG" else r["w8r2"]["wer"]) - r["w8"]["wer"]
                for r, p in zip(train, picks)
            )
            mean_nfe = statistics.mean(COST[p] for p in picks)
            if mean_dutmos >= STAC_UTMOS_TOL and mean_dwer <= STAC_WER_TOL and mean_nfe < best_nfe:
                best_nfe = mean_nfe
                best_tau = tau
        chosen_folds.append(best_tau)
        for r in test:
            oof_pick[r["utt_id"]] = route(r, best_tau)

    by_utt = {r["utt_id"]: r for r in records}
    costs = [COST[oof_pick[u]] for u in by_utt]
    n_tbrg = sum(1 for p in oof_pick.values() if p == "TBRG")
    n_w8r2 = len(oof_pick) - n_tbrg
    mean_cost = statistics.mean(costs)

    qual_deltas = {}
    for m in METRICS:
        vals = [
            (by_utt[u]["tbrg"][m] if oof_pick[u] == "TBRG" else by_utt[u]["w8r2"][m]) - by_utt[u]["w8"][m]
            for u in by_utt
        ]
        qual_deltas[f"d_{m}_vs_W8"] = tc.boot_ci(vals, B=BOOT_B, seed=SEED)

    div_utts = [u for u in div_map if u in stored_div and u in oof_pick]
    cv_vals, d_cv_vals = [], []
    for u in div_utts:
        val = float(div_map[u]["cvf0_TBRG"]) if oof_pick[u] == "TBRG" else float(stored_div[u]["cvf0_W8R2"])
        ref_w8 = float(stored_div[u]["cvf0_W8"])
        if not math.isnan(val):
            cv_vals.append(val)
            if not math.isnan(ref_w8):
                d_cv_vals.append(val - ref_w8)

    return {
        "feature": feature,
        "n_qual": len(by_utt),
        "n_div": len(cv_vals),
        "n_tbrg": n_tbrg,
        "n_w8r2": n_w8r2,
        "mean_cost": mean_cost,
        "saving_vs_W8": 1.0 - mean_cost / COST["W8"],
        "saving_vs_W8R2": 1.0 - mean_cost / COST["W8R2"],
        "saving_vs_T16": 1.0 - mean_cost / COST["T16"],
        "fold_thresholds": chosen_folds,
        "cvf0": tc.boot_ci(cv_vals, B=BOOT_B, seed=SEED),
        "d_cvf0_vs_W8": tc.boot_ci(d_cv_vals, B=BOOT_B, seed=SEED),
        **qual_deltas,
    }


def evaluate_stac_routing(qual_rows, div_map, stored_qual, stored_div):
    """Evaluate secondary 5-fold speaker-CV STAC routing between TBRG (13 NFE) and W8R2 (20 NFE)."""
    return {
        "relative_r4": _stac_policy_eval(qual_rows, div_map, stored_qual, stored_div, feature="relative_r4"),
        "cumulative_A4": _stac_policy_eval(qual_rows, div_map, stored_qual, stored_div, feature="cumulative_A4"),
    }


def determine_verdict(contrasts):
    """Apply the preregistered frozen H1-H5 verdict rule for EXP-028."""
    h1_cv_w4 = contrasts["H1_TBRG_minus_W4"]["cvf0"]
    h1_cv_w8r2 = contrasts["TBRG_vs_W8R2"]["cvf0"]
    h2_u_w4r2 = contrasts["H2_TBRG_minus_W4R2_utmos"]
    h2_u_w8r2 = contrasts["TBRG_vs_W8R2"]["utmos"]
    h3_w = contrasts["H3_TBRG_vs_W8_quality"]["wer"]
    h3_s = contrasts["H3_TBRG_vs_W8_quality"]["similarity"]
    h3_u = contrasts["H3_TBRG_vs_W8_quality"]["utmos"]
    h4_ms = contrasts["H4_TBRG_minus_W4R2_mel_spread"]
    h5_shuf_cv = contrasts["H5_TBRG_vs_SHUF"]["cvf0"]
    h5_blnd_u = contrasts["TBRG_vs_BLND"]["utmos"]
    h5_blnd_cv = contrasts["TBRG_vs_BLND"]["cvf0"]

    if (
        h1_cv_w4["boot"]["hi"] < -ZERO_TOL
        or h1_cv_w8r2["boot"]["hi"] < CVF0_W8R2_NONINF_MARGIN
        or h2_u_w4r2["boot"]["hi"] < -ZERO_TOL
        or h3_w["boot"]["lo"] > WER_NONINF_MARGIN
        or h3_s["boot"]["hi"] < SIM_NONINF_MARGIN
    ):
        return "refuted"

    h1_ok = (
        h1_cv_w4["boot"]["lo"] > ZERO_TOL
        and h1_cv_w4["robust"]["mean_without_top2_abs"] > ZERO_TOL
        and h1_cv_w8r2["boot"]["lo"] >= CVF0_W8R2_NONINF_MARGIN
    )
    h2_ok = (
        h2_u_w4r2["boot"]["lo"] > ZERO_TOL
        and h2_u_w4r2["robust"]["mean_without_top2_abs"] > ZERO_TOL
        and h2_u_w8r2["boot"]["lo"] >= UTMOS_W8R2_NONINF_MARGIN
    )
    h3_ok = (
        h3_w["boot"]["hi"] <= WER_NONINF_MARGIN
        and h3_s["boot"]["lo"] >= SIM_NONINF_MARGIN
        and h3_u["boot"]["mean"] >= UTMOS_CONFIRM_MIN_MEAN
    )
    h4_ok = h4_ms["boot"]["hi"] < -ZERO_TOL
    h5_ok = (
        h5_shuf_cv["boot"]["lo"] > ZERO_TOL
        and (h5_blnd_u["boot"]["lo"] > ZERO_TOL or h5_blnd_cv["boot"]["lo"] > ZERO_TOL)
    )

    if h1_ok and h2_ok and h3_ok and h4_ok and h5_ok:
        return "confirmed"
    if h1_cv_w4["boot"]["lo"] > ZERO_TOL and h2_u_w4r2["boot"]["lo"] > ZERO_TOL:
        return "supported_direction_only"
    return "mixed"


def _write_manifest(out_dir, paths, prereg_path):
    entries = {"preregistration": {"path": str(prereg_path), "sha256": _sha256_file(prereg_path)}}
    for p in paths:
        p = Path(p)
        if p.exists():
            entries[p.name] = {"sha256": _sha256_file(p), "bytes": p.stat().st_size}
    manifest = {"experiment": "EXP-028", "utc": _utc(), "files": entries}
    mpath = Path(out_dir) / "MANIFEST.json"
    mpath.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return mpath


def _arm_diversity_summary(div_map, stored_div):
    out = {}
    for arm in STORED_ARMS:
        rows = [stored_div[u] for u in div_map if u in stored_div]
        out[arm] = {
            "n": len(rows),
            "cvf0": tc.boot_ci([float(r[f"cvf0_{arm}"]) for r in rows], B=BOOT_B, seed=SEED),
            "voiced": tc.boot_ci([float(r[f"voiced_{arm}"]) for r in rows], B=BOOT_B, seed=SEED),
            "mel_spread": tc.boot_ci([float(r[f"mel_spread_{arm}"]) for r in rows], B=BOOT_B, seed=SEED),
            "slow_corr": None,
        }
    for arm in NEW_ARMS:
        rows = list(div_map.values())
        out[arm] = {
            "n": len(rows),
            "cvf0": tc.boot_ci([float(r[f"cvf0_{arm}"]) for r in rows], B=BOOT_B, seed=SEED),
            "voiced": tc.boot_ci([float(r[f"voiced_{arm}"]) for r in rows], B=BOOT_B, seed=SEED),
            "mel_spread": tc.boot_ci([float(r[f"mel_spread_{arm}"]) for r in rows], B=BOOT_B, seed=SEED),
            "slow_corr": tc.boot_ci([float(r[f"slow_corr_{arm}"]) for r in rows], B=BOOT_B, seed=SEED),
        }
    return out


def _arm_quality_summary(qual_map, stored_qual):
    out = {}
    ref_rows = [stored_qual[u] for u in qual_map if u in stored_qual]
    for arm in STORED_ARMS:
        entry = {m: tc.boot_ci([float(r[f"{m}_{arm}"]) for r in ref_rows], B=BOOT_B, seed=SEED) for m in METRICS}
        if arm != "W8":
            entry.update({
                f"d_{m}_vs_W8": tc.boot_ci(
                    [float(r[f"{m}_{arm}"]) - float(r[f"{m}_W8"]) for r in ref_rows], B=BOOT_B, seed=SEED
                )
                for m in METRICS
            })
        out[arm] = entry

    for arm in NEW_ARMS:
        entry = {
            m: tc.boot_ci([float(r[f"{m}_{arm}"]) for r in qual_map.values()], B=BOOT_B, seed=SEED) for m in METRICS
        }
        entry.update({
            f"d_{m}_vs_W8": tc.boot_ci(
                _paired_contrast(qual_map, stored_qual, f"{m}_{arm}", f"{m}_W8"), B=BOOT_B, seed=SEED
            )
            for m in METRICS
        })
        out[arm] = entry
    return out


def _summarize_unseen(div_rows, qual_rows):
    dmap = {r["utt_id"]: r for r in div_rows}
    qmap = {r["utt_id"]: r for r in qual_rows}
    div_summary = {}
    for arm in UNSEEN_ARMS:
        div_summary[arm] = {
            "n": len(div_rows),
            "cvf0": tc.boot_ci([float(r[f"cvf0_{arm}"]) for r in div_rows], B=BOOT_B, seed=SEED),
            "mel_spread": tc.boot_ci([float(r[f"mel_spread_{arm}"]) for r in div_rows], B=BOOT_B, seed=SEED),
            "slow_corr": tc.boot_ci([float(r[f"slow_corr_{arm}"]) for r in div_rows], B=BOOT_B, seed=SEED),
            "f0_corr_w8r2": tc.boot_ci(
                [float(r[f"f0_corr_w8r2_{arm}"]) for r in div_rows if not math.isnan(float(r[f"f0_corr_w8r2_{arm}"]))],
                B=BOOT_B,
                seed=SEED,
            ),
            "f0_shift_corr_w8r2": tc.boot_ci(
                [
                    float(r[f"f0_shift_corr_w8r2_{arm}"])
                    for r in div_rows
                    if not math.isnan(float(r[f"f0_shift_corr_w8r2_{arm}"]))
                ],
                B=BOOT_B,
                seed=SEED,
            ),
        }
    qual_summary = {}
    for arm in UNSEEN_ARMS:
        entry = {m: tc.boot_ci([float(r[f"{m}_{arm}"]) for r in qual_rows], B=BOOT_B, seed=SEED) for m in METRICS}
        if arm != "W8":
            entry.update({
                f"d_{m}_vs_W8": tc.boot_ci(
                    _paired_contrast(qmap, qmap, f"{m}_{arm}", f"{m}_W8"), B=BOOT_B, seed=SEED
                )
                for m in METRICS
            })
        qual_summary[arm] = entry

    contrasts = {
        "TBRG_minus_W4_cvf0": _summarize(_paired_contrast(dmap, dmap, "cvf0_TBRG", "cvf0_W4")),
        "TBRG_minus_W8R2_cvf0": _summarize(_paired_contrast(dmap, dmap, "cvf0_TBRG", "cvf0_W8R2")),
        "TBRG_minus_SHUF_cvf0": _summarize(_paired_contrast(dmap, dmap, "cvf0_TBRG", "cvf0_SHUF")),
        "TBRG_f0_corr_matched_minus_shifted": _summarize(
            _paired_contrast(dmap, dmap, "f0_corr_w8r2_TBRG", "f0_shift_corr_w8r2_TBRG")
        ),
        "TBRG_minus_W4R2_utmos": _summarize(_paired_contrast(qmap, qmap, "utmos_TBRG", "utmos_W4R2")),
        "TBRG_minus_W8R2_utmos": _summarize(_paired_contrast(qmap, qmap, "utmos_TBRG", "utmos_W8R2")),
        "TBRG_minus_BLND_utmos": _summarize(_paired_contrast(qmap, qmap, "utmos_TBRG", "utmos_BLND")),
        "TBRG_minus_W8_wer": _summarize(_paired_contrast(qmap, qmap, "wer_TBRG", "wer_W8")),
        "TBRG_minus_W8_similarity": _summarize(_paired_contrast(qmap, qmap, "similarity_TBRG", "similarity_W8")),
        "TBRG_minus_W8_utmos": _summarize(_paired_contrast(qmap, qmap, "utmos_TBRG", "utmos_W8")),
    }
    return {
        "n_diversity": len(div_rows),
        "n_quality": len(qual_rows),
        "arm_diversity": div_summary,
        "arm_quality": qual_summary,
        "contrasts": contrasts,
    }


def analyze(
    out_dir,
    allow_partial=False,
    prereg_path=PREREG_PATH,
    exp027_div_csv=EXP027_DIV_CSV,
    exp027_qual_csv=EXP027_QUAL_CSV,
):
    """Validate preregistration & completeness, merge EXP-027 references, analyze unseen cohort, and write report."""
    check_prereg(prereg_path)
    out_dir = Path(out_dir)
    div_path = out_dir / "diversity.csv"
    qual_path = out_dir / "quality.csv"
    if not div_path.exists():
        raise RuntimeError(f"Missing diversity CSV: {div_path}")
    if not qual_path.exists():
        raise RuntimeError(f"Missing quality CSV: {qual_path}")

    div_rows = rf.read_csv(div_path, repair=True)
    qual_rows = rf.read_csv(qual_path, repair=True)
    expected_div = {it["utt_id"] for it in div_items()}
    expected_qual = {it["utt_id"] for it in qual_items()}

    if not allow_partial and {r["utt_id"] for r in div_rows} != expected_div:
        raise RuntimeError(f"Expected {len(expected_div)} speakers in {div_path.name}, found {len(div_rows)}.")
    if not allow_partial and {r["utt_id"] for r in qual_rows} != expected_qual:
        raise RuntimeError(f"Expected {len(expected_qual)} speakers in {qual_path.name}, found {len(qual_rows)}.")
    if len(div_rows) < 2 or len(qual_rows) < 2:
        raise RuntimeError("Need at least 2 speakers in both diversity.csv and quality.csv to analyze.")

    div_map = {r["utt_id"]: r for r in div_rows}
    qual_map = {r["utt_id"]: r for r in qual_rows}
    stored_div = {r["utt_id"]: r for r in rf.read_csv(exp027_div_csv)}
    stored_qual = {r["utt_id"]: r for r in rf.read_csv(exp027_qual_csv)}

    contrasts = {
        "H1_TBRG_minus_W4": {
            "cvf0": _summarize(_paired_contrast(div_map, stored_div, "cvf0_TBRG", "cvf0_W4")),
            "mel_spread": _summarize(_paired_contrast(div_map, stored_div, "mel_spread_TBRG", "mel_spread_W4")),
        },
        "H2_TBRG_minus_W4R2_utmos": _summarize(
            _paired_contrast(qual_map, stored_qual, "utmos_TBRG", "utmos_W4R2")
        ),
        "H3_TBRG_vs_W8_quality": {
            m: _summarize(_paired_contrast(qual_map, stored_qual, f"{m}_TBRG", f"{m}_W8")) for m in METRICS
        },
        "H4_TBRG_minus_W4R2_mel_spread": _summarize(
            _paired_contrast(div_map, stored_div, "mel_spread_TBRG", "mel_spread_W4R2")
        ),
        "H5_TBRG_vs_SHUF": {
            "cvf0": _summarize(_paired_contrast(div_map, div_map, "cvf0_TBRG", "cvf0_SHUF")),
            "slow_corr": _summarize(_paired_contrast(div_map, div_map, "slow_corr_TBRG", "slow_corr_SHUF")),
            **{m: _summarize(_paired_contrast(qual_map, qual_map, f"{m}_TBRG", f"{m}_SHUF")) for m in METRICS},
        },
        "TBRG_vs_STS2": {
            "cvf0": _summarize(_paired_contrast(div_map, div_map, "cvf0_TBRG", "cvf0_STS2")),
            "mel_spread": _summarize(_paired_contrast(div_map, div_map, "mel_spread_TBRG", "mel_spread_STS2")),
            **{m: _summarize(_paired_contrast(qual_map, qual_map, f"{m}_TBRG", f"{m}_STS2")) for m in METRICS},
        },
        "TBRG_vs_BLND": {
            "cvf0": _summarize(_paired_contrast(div_map, div_map, "cvf0_TBRG", "cvf0_BLND")),
            "mel_spread": _summarize(_paired_contrast(div_map, div_map, "mel_spread_TBRG", "mel_spread_BLND")),
            "slow_corr": _summarize(_paired_contrast(div_map, div_map, "slow_corr_TBRG", "slow_corr_BLND")),
            **{m: _summarize(_paired_contrast(qual_map, qual_map, f"{m}_TBRG", f"{m}_BLND")) for m in METRICS},
        },
        "TBRG_vs_W6R2": {
            "cvf0": _summarize(_paired_contrast(div_map, div_map, "cvf0_TBRG", "cvf0_W6R2")),
            "mel_spread": _summarize(_paired_contrast(div_map, div_map, "mel_spread_TBRG", "mel_spread_W6R2")),
            **{m: _summarize(_paired_contrast(qual_map, qual_map, f"{m}_TBRG", f"{m}_W6R2")) for m in METRICS},
        },
        "TBRG_vs_W8R2": {
            "cvf0": _summarize(_paired_contrast(div_map, stored_div, "cvf0_TBRG", "cvf0_W8R2")),
            "mel_spread": _summarize(_paired_contrast(div_map, stored_div, "mel_spread_TBRG", "mel_spread_W8R2")),
            **{m: _summarize(_paired_contrast(qual_map, stored_qual, f"{m}_TBRG", f"{m}_W8R2")) for m in METRICS},
        },
        "STS2_minus_W4": {
            "cvf0": _summarize(_paired_contrast(div_map, stored_div, "cvf0_STS2", "cvf0_W4")),
            "mel_spread": _summarize(_paired_contrast(div_map, stored_div, "mel_spread_STS2", "mel_spread_W4")),
        },
    }

    stac_routing = evaluate_stac_routing(qual_rows, div_map, stored_qual, stored_div)
    arm_diversity = _arm_diversity_summary(div_map, stored_div)
    arm_quality = _arm_quality_summary(qual_map, stored_qual)
    verdict = determine_verdict(contrasts)

    udiv_path = out_dir / "unseen_diversity.csv"
    uqual_path = out_dir / "unseen_quality.csv"
    unseen_rep = None
    unseen_complete = True
    if udiv_path.exists() and uqual_path.exists():
        udiv_rows = rf.read_csv(udiv_path, repair=True)
        uqual_rows = rf.read_csv(uqual_path, repair=True)
        expected_udiv = {it["utt_id"] for it in unseen_div_items()}
        expected_uqual = {it["utt_id"] for it in unseen_qual_items()}
        unseen_complete = (
            {r["utt_id"] for r in udiv_rows} == expected_udiv
            and {r["utt_id"] for r in uqual_rows} == expected_uqual
        )
        if not allow_partial and not unseen_complete:
            raise RuntimeError(
                f"Incomplete unseen replication CSVs ({len(udiv_rows)}/{UNSEEN_DIV_N} div, "
                f"{len(uqual_rows)}/{UNSEEN_QUAL_N} qual)."
            )
        if len(udiv_rows) >= 2 and len(uqual_rows) >= 2:
            unseen_rep = _summarize_unseen(udiv_rows, uqual_rows)

    is_partial = allow_partial and (
        len(div_rows) < DIV_PAIRS or len(qual_rows) < FRESH_N or not unseen_complete
    )

    results = {
        "experiment": "EXP-028",
        "partial": is_partial,
        "n_diversity": len(div_rows),
        "n_quality": len(qual_rows),
        "verdict": verdict,
        "arm_diversity": arm_diversity,
        "arm_quality": arm_quality,
        "contrasts": contrasts,
        "unseen_replication": unseen_rep,
        "stac_routing_secondary": stac_routing,
        "utc": _utc(),
    }

    rjson = out_dir / "transmissibility_bridge_results.json"
    rmd = out_dir / "TRANSMISSIBILITY_BRIDGE_REPORT.md"
    rjson.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# EXP-028: Transmissible Slow-Band Bridge (`TBRG`) Report",
        "",
        f"- **Verdict:** `{verdict}` ({'PARTIAL' if is_partial else 'FULL'})",
        f"- **Speakers 61-100 (paired vs EXP-027):** diversity = {len(div_rows)}/{DIV_PAIRS} (x {DIV_SEEDS} seeds), "
        f"quality = {len(qual_rows)}/{FRESH_N}",
        "",
        "## Confirmatory Hypotheses (`TBRG`, 13 NFE, re-noised at tau_2 = 0.2929)",
        "",
        "| Contrast | Estimate (95% CI) | No-top2 mean | Sign (+/-, p) |",
        "| --- | ---: | ---: | ---: |",
    ]
    rows_md = [
        ("`H1: CVf0(TBRG - W4)`", contrasts["H1_TBRG_minus_W4"]["cvf0"], 4),
        ("`H1b: CVf0(TBRG - W8R2)` (NI >= -0.008)", contrasts["TBRG_vs_W8R2"]["cvf0"], 4),
        ("`H2: UTMOS(TBRG - W4R2)`", contrasts["H2_TBRG_minus_W4R2_utmos"], 3),
        ("`H2b: UTMOS(TBRG - W8R2)` (NI >= -0.05)", contrasts["TBRG_vs_W8R2"]["utmos"], 3),
        ("`H3: WER(TBRG - W8)`", contrasts["H3_TBRG_vs_W8_quality"]["wer"], 4),
        ("`H3: SIM(TBRG - W8)`", contrasts["H3_TBRG_vs_W8_quality"]["similarity"], 4),
        ("`H3: UTMOS(TBRG - W8)`", contrasts["H3_TBRG_vs_W8_quality"]["utmos"], 3),
        ("`H4: MelSpread(TBRG - W4R2)`", contrasts["H4_TBRG_minus_W4R2_mel_spread"], 4),
        ("`H5: CVf0(TBRG - SHUF)` (causal slow-band)", contrasts["H5_TBRG_vs_SHUF"]["cvf0"], 4),
        ("`H5b: slow_corr(TBRG - SHUF)`", contrasts["H5_TBRG_vs_SHUF"]["slow_corr"], 4),
        ("`H5c: UTMOS(TBRG - BLND)` (decomposition)", contrasts["TBRG_vs_BLND"]["utmos"], 3),
        ("`H5d: CVf0(TBRG - BLND)`", contrasts["TBRG_vs_BLND"]["cvf0"], 4),
    ]
    for label, c, digits in rows_md:
        b = c["boot"]
        lines.append(
            f"| {label} | {tc._fmt_ci(b, digits)} | {c['robust']['mean_without_top2_abs']:+.{digits}f} | "
            f"{c['sign']['pos']}/{c['sign']['neg']} (p={c['sign']['p']:.3g}) |"
        )

    lines.extend([
        "",
        "## Ablations & Intermediate-Budget Baselines",
        "",
        "| Contrast | Estimate (95% CI) | Sign (+/-, p) |",
        "| --- | ---: | ---: |",
        f"| `TBRG - STS2 (CVf0)` | {tc._fmt_ci(contrasts['TBRG_vs_STS2']['cvf0']['boot'], 4)} | "
        f"{contrasts['TBRG_vs_STS2']['cvf0']['sign']['pos']}/{contrasts['TBRG_vs_STS2']['cvf0']['sign']['neg']} |",
        f"| `TBRG - STS2 (UTMOS)` | {tc._fmt_ci(contrasts['TBRG_vs_STS2']['utmos']['boot'], 3)} | "
        f"{contrasts['TBRG_vs_STS2']['utmos']['sign']['pos']}/{contrasts['TBRG_vs_STS2']['utmos']['sign']['neg']} |",
        f"| `TBRG - W6R2 (CVf0)` | {tc._fmt_ci(contrasts['TBRG_vs_W6R2']['cvf0']['boot'], 4)} | "
        f"{contrasts['TBRG_vs_W6R2']['cvf0']['sign']['pos']}/{contrasts['TBRG_vs_W6R2']['cvf0']['sign']['neg']} |",
        f"| `TBRG - W6R2 (UTMOS)` | {tc._fmt_ci(contrasts['TBRG_vs_W6R2']['utmos']['boot'], 3)} | "
        f"{contrasts['TBRG_vs_W6R2']['utmos']['sign']['pos']}/{contrasts['TBRG_vs_W6R2']['utmos']['sign']['neg']} |",
        "",
        "## Side-by-Side Fresh-Speaker Summary (New EXP-028 + Stored EXP-027)",
        "",
        "| Arm | NFE | CVf0 (20 spk) | MelSpread | SlowCorr | WER vs W8 (40 spk) | SIM vs W8 | UTMOS vs W8 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for arm in ALL_ARMS:
        d = arm_diversity[arm]
        q = arm_quality[arm]
        sc_s = f"{d['slow_corr']['mean']:.3f}" if d["slow_corr"] is not None else "n/a"
        w_s = f"{q['wer']['mean']:.4f}" if arm == "W8" else tc._fmt_ci(q["d_wer_vs_W8"], 4)
        s_s = f"{q['similarity']['mean']:.4f}" if arm == "W8" else tc._fmt_ci(q["d_similarity_vs_W8"], 4)
        u_s = f"{q['utmos']['mean']:.3f}" if arm == "W8" else tc._fmt_ci(q["d_utmos_vs_W8"], 3)
        lines.append(
            f"| `{arm}` | {COST[arm]} | {d['cvf0']['mean']:.4f} | {d['mel_spread']['mean']:.4f} | {sc_s} | "
            f"{w_s} | {s_s} | {u_s} |"
        )

    if unseen_rep is not None:
        uc = unseen_rep["contrasts"]
        lines.extend([
            "",
            f"## Prospective Unseen-Speaker Replication Cohort (Calibration Split: "
            f"{unseen_rep['n_diversity']} x {UNSEEN_SEEDS} div, {unseen_rep['n_quality']} qual)",
            "",
            "| Unseen Arm | NFE | CVf0 | SlowCorr | f0 Corr vs W8R2 (matched / shifted) | WER vs W8 | SIM vs W8 | UTMOS vs W8 |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ])
        for arm in UNSEEN_ARMS:
            ud = unseen_rep["arm_diversity"][arm]
            uq = unseen_rep["arm_quality"][arm]
            fc_s = f"{ud['f0_corr_w8r2']['mean']:.3f} / {ud['f0_shift_corr_w8r2']['mean']:.3f}"
            w_s = f"{uq['wer']['mean']:.4f}" if arm == "W8" else tc._fmt_ci(uq["d_wer_vs_W8"], 4)
            s_s = f"{uq['similarity']['mean']:.4f}" if arm == "W8" else tc._fmt_ci(uq["d_similarity_vs_W8"], 4)
            u_s = f"{uq['utmos']['mean']:.3f}" if arm == "W8" else tc._fmt_ci(uq["d_utmos_vs_W8"], 3)
            lines.append(
                f"| `{arm}` | {COST[arm]} | {ud['cvf0']['mean']:.4f} | {ud['slow_corr']['mean']:.3f} | "
                f"{fc_s} | {w_s} | {s_s} | {u_s} |"
            )
        lines.extend([
            "",
            f"- **Unseen `TBRG - W4 (CVf0)`:** `{tc._fmt_ci(uc['TBRG_minus_W4_cvf0']['boot'], 4)}`",
            f"- **Unseen `TBRG - SHUF (CVf0)`:** `{tc._fmt_ci(uc['TBRG_minus_SHUF_cvf0']['boot'], 4)}`",
            f"- **Unseen `f0_corr(TBRG, W8R2)` matched minus shifted seed:** "
            f"`{tc._fmt_ci(uc['TBRG_f0_corr_matched_minus_shifted']['boot'], 4)}`",
            f"- **Unseen `TBRG - W4R2 (UTMOS)`:** `{tc._fmt_ci(uc['TBRG_minus_W4R2_utmos']['boot'], 3)}`",
            f"- **Unseen `TBRG - BLND (UTMOS)`:** `{tc._fmt_ci(uc['TBRG_minus_BLND_utmos']['boot'], 3)}`",
        ])

    rel = stac_routing.get("relative_r4")
    if rel is not None:
        lines.extend([
            "",
            "## Secondary Exploratory Diagnostic: 5-Fold CV Routing (`STAC-TBRG` between `TBRG` and `W8R2`)",
            "",
            f"- **Routed picks:** `{rel['n_tbrg']}` TBRG / `{rel['n_w8r2']}` W8R2 -> **`{rel['mean_cost']:.2f}` mean NFE** "
            f"(`{rel['saving_vs_W8'] * 100:.1f}%` saved vs `W8`, `{rel['saving_vs_W8R2'] * 100:.1f}%` vs `W8R2`, "
            f"`{rel['saving_vs_T16'] * 100:.1f}%` vs `T16`)",
            f"- **Out-of-fold quality vs `W8`:** dWER `{tc._fmt_ci(rel['d_wer_vs_W8'], 4)}`, "
            f"dSIM `{tc._fmt_ci(rel['d_similarity_vs_W8'], 4)}`, dUTMOS `{tc._fmt_ci(rel['d_utmos_vs_W8'], 3)}`",
            f"- **Out-of-fold diversity (`n={rel['n_div']}`):** CVf0 `{rel['cvf0']['mean']:.4f}` "
            f"(dCVf0 vs `W8`: `{tc._fmt_ci(rel['d_cvf0_vs_W8'], 4)}`)",
        ])

    rmd.write_text("\n".join(lines) + "\n", encoding="utf-8")
    manifest_files = [div_path, qual_path, rjson, rmd]
    for extra in (udiv_path, uqual_path):
        if extra.exists():
            manifest_files.append(extra)
    _write_manifest(out_dir, manifest_files, prereg_path)
    print(f"[OK] Analysis complete: verdict={verdict}, report={rmd}", flush=True)
    return results


def main():
    p = argparse.ArgumentParser(description="EXP-028 Transmissible Slow-Band Bridge (`TBRG`) runner")
    sub = p.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("freeze")
    f.add_argument("--force", action="store_true")

    s = sub.add_parser("smoke")
    s.add_argument("--dmo_root", required=True)
    s.add_argument("--out_dir", default=str(EXP_DIR / "results"))
    s.add_argument("--device", default="cuda")

    r = sub.add_parser("run")
    r.add_argument("--dmo_root", required=True)
    r.add_argument("--out_dir", default=str(EXP_DIR / "results"))
    r.add_argument("--device", default="cuda")
    r.add_argument("--stage", choices=("all", "both", "diversity", "quality", "unseen"), default="all")

    a = sub.add_parser("analyze")
    a.add_argument("--out_dir", default=str(EXP_DIR / "results"))
    a.add_argument("--allow_partial", action="store_true")

    args = p.parse_args()
    if args.cmd == "freeze":
        freeze(force=args.force)
    elif args.cmd == "smoke":
        smoke(rf.Pipeline(args.dmo_root, args.device), args.out_dir)
    elif args.cmd == "run":
        check_prereg()
        pipe = rf.Pipeline(args.dmo_root, args.device)
        if args.stage in ("all", "both", "diversity"):
            run_diversity(pipe, div_items(), args.out_dir)
        if args.stage in ("all", "both", "quality"):
            run_quality(pipe, qual_items(), args.out_dir)
        if args.stage in ("all", "unseen"):
            run_unseen_replication(pipe, args.out_dir)
    elif args.cmd == "analyze":
        analyze(args.out_dir, allow_partial=args.allow_partial)


if __name__ == "__main__":
    main()
