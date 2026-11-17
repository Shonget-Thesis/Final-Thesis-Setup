"""EXP-030 (EVAL-023): Timing-Only Handover (TOH) pilot.

DMOSpeech 2's teacher-guided sampling is kept: the guided teacher runs k sway steps, then the distilled student
finishes from its first restart level with fresh noise. What changes is what crosses the handover. In the released
handover (W_k) the student is re-noised from the teacher's endpoint estimate T_k, so every spectral property of T_k,
including the prompt's recording condition that the teacher copies, reaches the student. In TOH only the teacher's
timing crosses:

  1. S0 = the student's one-step draft from pure noise (level 0 of the released student loop, 1 pass).
  2. T_k = the guided teacher's endpoint estimate after k sway steps (2k + 2 passes).
  3. w = a monotone, slope-limited dynamic-time-warping map from T_k's frames to S0's frames, computed on band-group
     log energies z-scored over time (per-band offsets and scales do not affect it).
  4. S0 is resampled along w, and the student's remaining levels run from it exactly as in the student-alone loop
     (3 passes). Total: 2k + 6 passes.

With w = identity, TOH equals the student alone exactly. MIR3 (a mirrored copy of TOH3's map: the same amount and
kind of re-timing, placed where the teacher did not put it) is the control for the teacher's own timing.

Cohorts and seeds are EXP-029's, so W4 and W8 here repeat EXP-029's code path on the same cases; their scores are
compared with EXP-029's stored ones as a reproduction check. Development pilot only. Subcommands: freeze, smoke, run,
analyze.
"""
import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_consistency_gate_prelim as cg  # noqa: E402
import run_damped_transmissibility_stac as dts  # noqa: E402
import run_followup_tests as rf  # noqa: E402
import run_grid32_prelim as gl  # noqa: E402
import run_noise_guidance_prelim as ng  # noqa: E402
import run_selective_handover_prelim as sh  # noqa: E402
import run_transmissibility_bridge as tb  # noqa: E402
import run_transmissibility_confirm as tc  # noqa: E402

THESIS_ROOT = rf.THESIS_ROOT
CODE_DIR = Path(__file__).resolve().parent
EXP_DIR = THESIS_ROOT / "research/experiments/EXP-030-timing-only-handover"
PREREG_PATH = EXP_DIR / "preregistration.json"
EXP027_QUAL_CSV = tb.EXP027_QUAL_CSV
EXP029_RESULTS = THESIS_ROOT / "research/experiments/EXP-029-damped-transmissibility-stac/results"
EXP029_QUAL_CSV = EXP029_RESULTS / "quality.csv"
EXP029_DIV_CSV = EXP029_RESULTS / "diversity.csv"

DEPENDENCIES = ("run_timing_only_handover.py",) + tuple(dts.DEPENDENCIES)

DEV_OFFSET = 60
QUAL_N = 30
DIV_N = 20
DIV_SEEDS = 12
DIV_SEED_BASE = 1000
CONDS = ("clean", "gauss25", "noisy15", "telephone")
NOISE_CONDS = ("gauss25", "noisy15")
SEED = 2026
BOOT_B = 2000

METRICS = ("wer", "similarity", "utmos")
ARMS = ("STU", "W2", "W3", "W4", "TOH2", "TOH3", "MIR3", "W8")
TEACHER_STEPS = (2, 3, 4, 8)
COST = {"STU": 4, "W2": 9, "W3": 11, "W4": 13, "TOH2": 10, "TOH3": 12, "MIR3": 12, "W8": 21}
PRIMARY = ("TOH3", "W3")
SECONDARY = ("TOH2", "W2")

# Timing map (fixed before any synthesis).
N_GROUPS = 10          # band groups of the 100 mel bins
FEAT_SMOOTH = 3        # frames, moving average before z-scoring
BAND_FRAC = 0.10       # Sakoe-Chiba band as a share of the generated length
MIN_BAND = 8           # frames

# Bars (the clean ones are EXP-029's).
SPLIT_TOL = 1e-5
REPRO_TOL = 0.01
ZERO_TOL = 1e-12
CLEAN_UTMOS_BAR = -0.08
WER_BAR = 0.01
SIM_BAR = -0.01
VARIETY_NI_MARGIN = -0.010
NOISE_STU_MARGIN = -0.10
RUNAWAY_WER = 1.0

QUAL_FIELDS = (
    ["utt_id", "speaker_id", "condition", "prompt_snr", "gpu", "frames",
     "warp_rms_2", "warp_max_2", "warp_rms_3", "warp_max_3", "warp_band"]
    + [f"{m}_{a}" for a in ARMS for m in METRICS + ("hypothesis",)]
    + ["case_seconds", "utc"]
)
DIV_FIELDS = (
    ["utt_id", "speaker_id", "condition", "n_seeds", "warp_rms_2_mean", "warp_rms_3_mean"]
    + [f"{m}_{a}" for a in ARMS for m in ("cvf0", "voiced", "mel_spread")]
    + ["case_seconds", "utc"]
)


def qual_items(n=QUAL_N):
    """EXP-029's quality speakers (development 61-90), so W4 and W8 repeat EXP-029 on the same cases."""
    return cg.dev_items()[DEV_OFFSET:DEV_OFFSET + n]


def div_items(n=DIV_N):
    """EXP-029's diversity speakers (development 61-80) with its seeds."""
    return cg.dev_items()[DEV_OFFSET:DEV_OFFSET + n]


# --------------------------------------------------------------------------- the timing map
def timing_features(mel_gen, n_groups=N_GROUPS, smooth=FEAT_SMOOTH):
    """(frames, bands) log-mel -> (frames, n_groups): band-group means, smoothed over time, z-scored per group.

    Z-scoring over time removes any per-band constant offset or positive scale, so a recording condition that is
    stationary in log-mel does not move the alignment."""
    x = np.asarray(mel_gen, dtype=float)
    frames, bands = x.shape
    edges = np.linspace(0, bands, n_groups + 1).round().astype(int)
    g = np.stack([x[:, edges[i]:edges[i + 1]].mean(axis=1) for i in range(n_groups)], axis=1)
    if smooth > 1 and frames > 1:
        left, right = smooth // 2, smooth - 1 - smooth // 2
        kernel = np.ones(smooth) / smooth
        g = np.stack([np.convolve(np.pad(g[:, j], (left, right), mode="edge"), kernel, mode="valid")
                      for j in range(n_groups)], axis=1)
    g = g - g.mean(axis=0, keepdims=True)
    return g / np.maximum(g.std(axis=0, keepdims=True), 1e-6)


def band_width(frames, band_frac=BAND_FRAC, min_band=MIN_BAND):
    return max(int(min_band), int(round(band_frac * frames)))


def constrained_dtw(cost, band):
    """Minimum-cost path from (0, 0) to (n-1, m-1) using steps (1,1), (1,2) and (2,1), so every local slope is
    between 1/2 and 2, inside |j - i*(m-1)/(n-1)| <= band. The (1,2) and (2,1) steps add the cost of the cell they
    pass through. Returns the path as a list of (i, j) cells, intermediate cells included."""
    cost = np.asarray(cost, dtype=float)
    n, m = cost.shape
    if n < 2 or m < 2:
        raise ValueError("constrained_dtw needs at least 2 frames on each side")
    if not (0.5 * (n - 1) <= m - 1 <= 2.0 * (n - 1)):
        raise ValueError(f"lengths {n} and {m} cannot be joined with slopes between 1/2 and 2")
    inf = np.inf
    D = np.full((n, m), inf)
    P = np.full((n, m), -1, dtype=np.int8)
    D[0, 0] = cost[0, 0]
    scale = (m - 1) / (n - 1)
    for i in range(1, n):
        centre = i * scale
        lo, hi = max(1, int(math.ceil(centre - band))), min(m - 1, int(math.floor(centre + band)))
        if lo > hi:
            continue
        js = np.arange(lo, hi + 1)
        c_diag = D[i - 1, js - 1]
        c_12 = np.full(js.shape, inf)
        ok = js >= 2
        c_12[ok] = D[i - 1, js[ok] - 2] + cost[i, js[ok] - 1]
        c_21 = np.full(js.shape, inf)
        if i >= 2:
            c_21 = D[i - 2, js - 1] + cost[i - 1, js]
        stacked = np.stack([c_diag, c_12, c_21])
        best = np.argmin(stacked, axis=0)
        D[i, js] = stacked[best, np.arange(js.size)] + cost[i, js]
        P[i, js] = best
    if not np.isfinite(D[n - 1, m - 1]):
        raise RuntimeError("No admissible path inside the band")
    i, j = n - 1, m - 1
    path = [(i, j)]
    while (i, j) != (0, 0):
        step = P[i, j]
        if step == 0:
            i, j = i - 1, j - 1
        elif step == 1:
            path.append((i, j - 1))
            i, j = i - 1, j - 2
        elif step == 2:
            path.append((i - 1, j))
            i, j = i - 2, j - 1
        else:
            raise RuntimeError(f"Broken back-pointer at {(i, j)}")
        path.append((i, j))
    path.reverse()
    return path


def path_to_map(path, n, m):
    """Mean student frame per teacher frame: non-decreasing, steps between 0 and 2, w[0] = 0 and w[n-1] in
    [m - 1.5, m - 1]. The last frame is not pinned to m - 1: when the path ends with two (1,2) steps, pinning made a
    step of 2.5 (Amendment 1)."""
    sums, counts = np.zeros(n), np.zeros(n)
    for i, j in path:
        sums[i] += j
        counts[i] += 1
    if np.any(counts == 0):
        raise RuntimeError("Path leaves a teacher frame unmatched")
    return np.maximum.accumulate(sums / counts)


def timing_map(student_gen, teacher_gen, band_frac=BAND_FRAC, min_band=MIN_BAND):
    """Map w (one position in the student's frames per teacher frame) and its band, from (frames, bands) arrays."""
    fs, ft = timing_features(student_gen), timing_features(teacher_gen)
    n, m = ft.shape[0], fs.shape[0]
    cost = (ft ** 2).sum(1)[:, None] + (fs ** 2).sum(1)[None, :] - 2.0 * ft @ fs.T
    band = band_width(max(n, m), band_frac, min_band)
    return path_to_map(constrained_dtw(np.maximum(cost, 0.0), band), n, m), band


def mirror_map(w):
    """The map mirrored in time: w'(i) = (m-1) - w(n-1-i). Same displacement sizes and slopes, other places."""
    w = np.asarray(w, dtype=float)
    return (w[-1] + w[0]) - w[::-1]


def map_stats(w):
    d = np.asarray(w, dtype=float) - np.arange(len(w), dtype=float)
    return float(np.sqrt(np.mean(d ** 2))), float(np.max(np.abs(d)))


def warp_frames(x, w):
    """Linear interpolation of a (frames, bands) array (numpy or torch) at the positions w."""
    w = np.clip(np.asarray(w, dtype=float), 0.0, x.shape[0] - 1)
    lo = np.floor(w).astype(np.int64)
    hi = np.minimum(lo + 1, x.shape[0] - 1)
    frac = w - lo
    if hasattr(x, "clone"):
        import torch

        lo_t = torch.as_tensor(lo, device=x.device)
        hi_t = torch.as_tensor(hi, device=x.device)
        f_t = torch.as_tensor(frac, device=x.device, dtype=x.dtype).unsqueeze(-1)
        return x[lo_t] * (1.0 - f_t) + x[hi_t] * f_t
    return x[lo] * (1.0 - frac)[:, None] + x[hi] * frac[:, None]


def apply_timing_map(full, w, plen):
    """Resample the generated frames [plen:] of a (1, frames, bands) array along w; the prompt frames are kept."""
    out = full.clone() if hasattr(full, "clone") else np.array(full, copy=True)
    out[0, plen:, :] = warp_frames(full[0, plen:, :], w)
    return out


def _as_numpy(x):
    if hasattr(x, "detach"):
        return x.detach().float().cpu().numpy()
    return np.asarray(x, dtype=float)


def timing_handover(s0, t_k, plen):
    """TOH payload: the student's draft re-timed along the teacher's timing. Returns (payload, map, band)."""
    w, band = timing_map(_as_numpy(s0[0, plen:, :]), _as_numpy(t_k[0, plen:, :]))
    return apply_timing_map(s0, w, plen), w, band


# --------------------------------------------------------------------------- synthesis
def synthesize_arms(pipe, ctx, ends):
    """All arms from one teacher trajectory and one student draft; every arm uses the same restart noises."""
    taus = list(pipe.sampler.STUDENT_TAUS)
    plen = ctx["prompt_len"]
    s0 = gl.student(pipe, ctx, ctx["step_cond"], [0.0])
    outs = {"STU": gl.student(pipe, ctx, s0, taus)}
    for k in (2, 3, 4, 8):
        outs[f"W{k}"] = gl.student(pipe, ctx, ends[k], taus)
    x2, w2, band = timing_handover(s0, ends[2], plen)
    x3, w3, _ = timing_handover(s0, ends[3], plen)
    outs["TOH2"] = gl.student(pipe, ctx, x2, taus)
    outs["TOH3"] = gl.student(pipe, ctx, x3, taus)
    outs["MIR3"] = gl.student(pipe, ctx, apply_timing_map(s0, mirror_map(w3), plen), taus)
    r2, m2 = map_stats(w2)
    r3, m3 = map_stats(w3)
    stats = {"frames": len(w3), "warp_rms_2": r2, "warp_max_2": m2, "warp_rms_3": r3, "warp_max_3": m3, "warp_band": band}
    return outs, stats


def run_quality(pipe, items, conds=CONDS, out_dir=None, csv_name="quality.csv"):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / csv_name
    done = rf.done_keys(path, 1)
    todo = [(it, c) for it in items for c in conds if (it["utt_id"], c) not in done]
    print(f"[quality] {len(todo)} cases to run ({len(done)} already in {path.name})", flush=True)
    for n_run, (item, cond) in enumerate(todo, 1):
        t0 = time.perf_counter()
        ctx = pipe.prepare(item, cond, int(item["input_seed"]), out_dir)
        ends, _ = dts.teacher_trajectory(pipe, ctx, wanted=TEACHER_STEPS)
        outs, stats = synthesize_arms(pipe, ctx, ends)
        gt = pipe.ground_truth(item)
        row = {"utt_id": item["utt_id"], "speaker_id": item["speaker_id"], "condition": cond,
               "prompt_snr": ctx["prompt_snr"], "gpu": pipe.gpu, **stats}
        for arm in ARMS:
            m = pipe.score(ng.decode(pipe, ctx, outs[arm]), item, gt)
            row.update({f"{k}_{arm}": m[k] for k in METRICS + ("hypothesis",)})
        row.update({"case_seconds": round(time.perf_counter() - t0, 2), "utc": tb._utc()})
        tb._append(path, QUAL_FIELDS, row)
        print(f"[{n_run}/{len(todo)}] quality: {item['utt_id']} {cond} ({row['case_seconds']:.1f} s)", flush=True)
    print(f"[OK] quality complete: {path}", flush=True)
    return path


def run_diversity(pipe, items, out_dir, seeds=DIV_SEEDS, csv_name="diversity.csv"):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / csv_name
    done = rf.done_keys(path, 1)
    todo = [it for it in items if (it["utt_id"], "clean") not in done]
    print(f"[diversity] {len(todo)} speakers x {seeds} seeds ({len(done)} already in {path.name})", flush=True)
    for n_run, item in enumerate(todo, 1):
        t0 = time.perf_counter()
        tracks = {a: [] for a in ARMS}
        slow = {a: [] for a in ARMS}
        rms2, rms3 = [], []
        for s in range(seeds):
            seed = int(item["input_seed"]) + DIV_SEED_BASE + s
            ctx = pipe.prepare(item, "clean", seed, out_dir, noise_seed=item["input_seed"])
            ends, _ = dts.teacher_trajectory(pipe, ctx, wanted=TEACHER_STEPS)
            outs, stats = synthesize_arms(pipe, ctx, ends)
            rms2.append(stats["warp_rms_2"])
            rms3.append(stats["warp_rms_3"])
            arm_tracks = dts._track_arms_fast(pipe, ctx, outs, ARMS)
            for arm in ARMS:
                tracks[arm].append(arm_tracks[arm])
                slow[arm].append(tc._slow_component(outs[arm], ctx["prompt_len"]))
        row = {"utt_id": item["utt_id"], "speaker_id": item["speaker_id"], "condition": "clean", "n_seeds": seeds,
               "warp_rms_2_mean": statistics.mean(rms2), "warp_rms_3_mean": statistics.mean(rms3)}
        for arm in ARMS:
            row[f"cvf0_{arm}"], row[f"voiced_{arm}"] = sh.cvf0(tracks[arm])
            row[f"mel_spread_{arm}"] = tc.slow_mel_spread(slow[arm])
        row.update({"case_seconds": round(time.perf_counter() - t0, 2), "utc": tb._utc()})
        tb._append(path, DIV_FIELDS, row)
        print(f"[{n_run}/{len(todo)}] diversity: {item['utt_id']} ({row['case_seconds']:.1f} s)", flush=True)
    print(f"[OK] diversity complete: {path}", flush=True)
    return path


def smoke(pipe, out_dir, prereg_path=PREREG_PATH):
    """Identity checks (split student, identity map, TOH with the identity map = student alone), map checks on real
    estimates, EXP-027 W4 reproduction, then micro runs of both stages."""
    check_prereg(prereg_path)
    sdir = Path(out_dir) / "smoke"
    sdir.mkdir(parents=True, exist_ok=True)
    stored = {r["utt_id"]: r for r in rf.read_csv(EXP027_QUAL_CSV)}
    taus = list(pipe.sampler.STUDENT_TAUS)
    checks = []
    for item in tb.smoke_items():
        ctx = pipe.prepare(item, "clean", int(item["input_seed"]), sdir)
        ends, _ = dts.teacher_trajectory(pipe, ctx, wanted=TEACHER_STEPS)
        plen = ctx["prompt_len"]
        s0 = gl.student(pipe, ctx, ctx["step_cond"], [0.0])
        stu_split = gl.student(pipe, ctx, s0, taus)
        stu_full = gl.student(pipe, ctx, ctx["step_cond"], [0.0] + taus)
        n = s0.shape[1] - plen
        x_id = apply_timing_map(s0, np.arange(n, dtype=float), plen)
        toh_id = gl.student(pipe, ctx, x_id, taus)
        _, w3, band = timing_handover(s0, ends[3], plen)
        dw = np.diff(w3)
        map_ok = bool(w3[0] == 0.0 and n - 1.5 <= w3[-1] <= n - 1 and np.all(dw >= -1e-9) and np.all(dw <= 2.0 + 1e-9)
                      and np.max(np.abs(w3 - np.arange(n))) <= band + 1.0)
        rec = {"utt_id": item["utt_id"], "split_diff": tb._max_abs(stu_split - stu_full),
               "identity_map_diff": tb._max_abs(x_id - s0), "toh_identity_vs_student": tb._max_abs(toh_id - stu_full),
               "map_ok": map_ok, "warp_rms_3": map_stats(w3)[0], "band": band}
        m = pipe.score(ng.decode(pipe, ctx, gl.student(pipe, ctx, ends[4], taus)), item, pipe.ground_truth(item))
        ref = stored[item["utt_id"]]
        rec["d_wer"] = abs(m["wer"] - float(ref["wer_W4"]))
        rec["d_sim"] = abs(m["similarity"] - float(ref["similarity_W4"]))
        print(f"[smoke] {json.dumps(rec)}", flush=True)
        checks.append(rec)
        if max(rec["split_diff"], rec["identity_map_diff"], rec["toh_identity_vs_student"]) > SPLIT_TOL or not map_ok:
            raise RuntimeError(f"Identity or map check failed: {rec}")
        if rec["d_wer"] > REPRO_TOL or rec["d_sim"] > REPRO_TOL:
            raise RuntimeError(f"EXP-027 W4 reproduction failed: {rec}")
    run_quality(pipe, tb.smoke_items()[:1], conds=("clean", "telephone"), out_dir=sdir, csv_name="quality.csv")
    run_diversity(pipe, tb.smoke_items(), sdir, seeds=2, csv_name="diversity.csv")
    summary = {"checks": checks, "utc": tb._utc()}
    (sdir / "smoke_check.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"[OK] smoke passed ({summary})", flush=True)
    return summary


# --------------------------------------------------------------------------- preregistration
HYPOTHESES = {
    "H1a": "Clean pitch variety: CVf0(TOH3) - CVf0(STU), 95% bootstrap lower bound > 0 (the teacher's timing adds variety).",
    "H1b": "Clean pitch variety: CVf0(TOH3) - CVf0(W3), lower bound >= -0.010 (timing alone keeps W3's variety).",
    "H2a": "Noise blocked: per-speaker mean over gauss25 and noisy15 of UTMOS(TOH3) - UTMOS(W4), lower bound > 0.",
    "H2b": "As good as the student on noise: same pooling, UTMOS(TOH3) - UTMOS(STU), lower bound >= -0.10.",
    "H3": "Clean quality against W8: mean dUTMOS >= -0.08, dWER upper bound <= +0.01, dSIM lower bound >= -0.01.",
}
VERDICT_RULE = (
    "supported if H1a, H1b, H2a, H2b and H3 hold for TOH3; timing_does_not_carry_variety if H2a, H2b and H3 hold but "
    "H1a or H1b fails; not_supported if H2a, H2b or H3 fails; partial if a stage is missing. TOH2 is evaluated by the "
    "same rule against W2 as a secondary result and does not change the verdict. Telephone, MIR3, runaway counts "
    "(WER > 1) and the EXP-029 reproduction are exploratory or checks."
)


def build_prereg():
    q = [{"utt_id": it["utt_id"], "speaker_id": it["speaker_id"]} for it in qual_items()]
    d = [{"utt_id": it["utt_id"], "speaker_id": it["speaker_id"]} for it in div_items()]
    inputs = (EXP027_QUAL_CSV, EXP029_QUAL_CSV, EXP029_DIV_CSV)
    payload = {
        "experiment": "EXP-030",
        "task_id": "EVAL-023",
        "title": "Timing-Only Handover (TOH) pilot",
        "framing": ("Development pilot. DMOSpeech 2's teacher-guided handover is kept; only what crosses changes: the "
                    "teacher's timing, as a slope-limited time map applied to the student's own one-step draft, "
                    "instead of the teacher's spectral estimate. EXP-029's cohorts and seeds are reused."),
        "mechanism": {"teacher_steps": [2, 3], "n_groups": N_GROUPS, "feature_smooth": FEAT_SMOOTH,
                      "band_frac": BAND_FRAC, "min_band": MIN_BAND, "dtw_steps": [[1, 1], [1, 2], [2, 1]],
                      "map": "mean student frame per teacher frame, endpoints pinned, made non-decreasing"},
        "conditions": list(CONDS),
        "noise_conditions": list(NOISE_CONDS),
        "arms": list(ARMS),
        "costs": dict(COST),
        "n_qual_speakers": len(q),
        "n_qual_cases": len(q) * len(CONDS),
        "qual_utts": q,
        "n_div_speakers": len(d),
        "div_seeds": DIV_SEEDS,
        "div_seed_base": DIV_SEED_BASE,
        "div_utts": d,
        "bars": {"clean_utmos": CLEAN_UTMOS_BAR, "wer": WER_BAR, "sim": SIM_BAR,
                 "variety_ni_margin": VARIETY_NI_MARGIN, "noise_student_margin": NOISE_STU_MARGIN,
                 "runaway_wer": RUNAWAY_WER},
        "hypotheses": HYPOTHESES,
        "verdict_rule": VERDICT_RULE,
        "code_hashes": {name: tb._sha256_file(CODE_DIR / name) for name in DEPENDENCIES},
        "input_hashes": {str(Path(p).relative_to(THESIS_ROOT).as_posix()): tb._sha256_file(p) for p in inputs},
        "seed": SEED,
        "boot_b": BOOT_B,
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
    if doc.get("qual_utts") != [{"utt_id": it["utt_id"], "speaker_id": it["speaker_id"]} for it in qual_items()]:
        raise RuntimeError("Quality speaker list does not match preregistration.")
    if doc.get("div_utts") != [{"utt_id": it["utt_id"], "speaker_id": it["speaker_id"]} for it in div_items()]:
        raise RuntimeError("Diversity speaker list does not match preregistration.")
    return doc


# --------------------------------------------------------------------------- analysis
def _f(r, key):
    return float(r[key])


def _per_speaker(rows, conds, fn):
    """Per-speaker mean over the given conditions of fn(row); speakers without any such row are skipped."""
    acc = {}
    for r in rows:
        if r["condition"] in conds:
            acc.setdefault(r["speaker_id"], []).append(fn(r))
    return [statistics.mean(v) for _, v in sorted(acc.items())]


def condition_tables(qual_rows):
    out = {}
    for cond in sorted({r["condition"] for r in qual_rows}):
        sub = [r for r in qual_rows if r["condition"] == cond]
        arms = {}
        for arm in ARMS:
            e = {m: tc.boot_ci([_f(r, f"{m}_{arm}") for r in sub], B=BOOT_B, seed=SEED) for m in METRICS}
            e["median_wer"] = statistics.median(_f(r, f"wer_{arm}") for r in sub)
            e["runaway"] = sum(_f(r, f"wer_{arm}") > RUNAWAY_WER for r in sub)
            for ref in ("W8", "W4", "STU"):
                if arm != ref:
                    e[f"vs_{ref}"] = {m: tb._summarize([_f(r, f"{m}_{arm}") - _f(r, f"{m}_{ref}") for r in sub], seed=SEED)
                                      for m in METRICS}
            arms[arm] = e
        out[cond] = {"n": len(sub), "arms": arms}
    return out


def diversity_table(div_rows):
    if len(div_rows) < 2:
        return None
    arms = {a: {m: tc.boot_ci([_f(r, f"{m}_{a}") for r in div_rows], B=BOOT_B, seed=SEED)
                for m in ("cvf0", "voiced", "mel_spread")} for a in ARMS}
    pairs = (("TOH3", "STU"), ("TOH3", "W3"), ("TOH2", "STU"), ("TOH2", "W2"), ("MIR3", "TOH3"), ("MIR3", "STU"),
             ("W2", "STU"), ("W3", "STU"), ("W4", "STU"), ("W8", "W4"))
    contrasts = {f"{a}_minus_{b}": tb._summarize([_f(r, f"cvf0_{a}") - _f(r, f"cvf0_{b}") for r in div_rows], seed=SEED)
                 for a, b in pairs}
    return {"n": len(div_rows), "arms": arms, "cvf0_contrasts": contrasts,
            "warp_rms_2": tc.boot_ci([_f(r, "warp_rms_2_mean") for r in div_rows], B=BOOT_B, seed=SEED),
            "warp_rms_3": tc.boot_ci([_f(r, "warp_rms_3_mean") for r in div_rows], B=BOOT_B, seed=SEED)}


def evaluate(qual_rows, div_rows, arm, ref_w):
    """The preregistered hypotheses for one TOH arm; None where the needed stage is missing."""
    res = {"arm": arm, "variety_ref": ref_w}
    if len(div_rows) >= 2:
        g_stu = tc.boot_ci([_f(r, f"cvf0_{arm}") - _f(r, "cvf0_STU") for r in div_rows], B=BOOT_B, seed=SEED)
        g_w = tc.boot_ci([_f(r, f"cvf0_{arm}") - _f(r, f"cvf0_{ref_w}") for r in div_rows], B=BOOT_B, seed=SEED)
        res.update({"H1a": g_stu["lo"] > ZERO_TOL, "H1a_ci": g_stu, "H1b": g_w["lo"] >= VARIETY_NI_MARGIN, "H1b_ci": g_w})
    else:
        res.update({"H1a": None, "H1b": None})
    noise = [r for r in qual_rows if r["condition"] in NOISE_CONDS]
    clean = [r for r in qual_rows if r["condition"] == "clean"]
    if noise:
        v_w4 = tc.boot_ci(_per_speaker(noise, NOISE_CONDS, lambda r: _f(r, f"utmos_{arm}") - _f(r, "utmos_W4")),
                          B=BOOT_B, seed=SEED)
        v_stu = tc.boot_ci(_per_speaker(noise, NOISE_CONDS, lambda r: _f(r, f"utmos_{arm}") - _f(r, "utmos_STU")),
                           B=BOOT_B, seed=SEED)
        res.update({"H2a": v_w4["lo"] > ZERO_TOL, "H2a_ci": v_w4, "H2b": v_stu["lo"] >= NOISE_STU_MARGIN, "H2b_ci": v_stu})
    else:
        res.update({"H2a": None, "H2b": None})
    if clean:
        du = tc.boot_ci([_f(r, f"utmos_{arm}") - _f(r, "utmos_W8") for r in clean], B=BOOT_B, seed=SEED)
        dw = tc.boot_ci([_f(r, f"wer_{arm}") - _f(r, "wer_W8") for r in clean], B=BOOT_B, seed=SEED)
        ds = tc.boot_ci([_f(r, f"similarity_{arm}") - _f(r, "similarity_W8") for r in clean], B=BOOT_B, seed=SEED)
        res.update({"H3": du["mean"] >= CLEAN_UTMOS_BAR and dw["hi"] <= WER_BAR and ds["lo"] >= SIM_BAR,
                    "H3_utmos": du, "H3_wer": dw, "H3_sim": ds})
    else:
        res["H3"] = None
    return res


def verdict(h, complete=True):
    keys = ("H1a", "H1b", "H2a", "H2b", "H3")
    if not complete or any(h[k] is None for k in keys):
        return "partial"
    if not (h["H2a"] and h["H2b"] and h["H3"]):
        return "not_supported"
    if h["H1a"] and h["H1b"]:
        return "supported"
    return "timing_does_not_carry_variety"


def exp029_reproduction(qual_rows, div_rows):
    """Max absolute differences of W4 and W8 against EXP-029's stored scores on the shared cases."""
    out = {}
    if EXP029_QUAL_CSV.exists():
        stored = {(r["utt_id"], r["condition"]): r for r in rf.read_csv(EXP029_QUAL_CSV)}
        shared = [r for r in qual_rows if (r["utt_id"], r["condition"]) in stored]
        out["quality_cases"] = len(shared)
        for arm in ("W4", "W8"):
            for m in METRICS:
                diffs = [abs(_f(r, f"{m}_{arm}") - _f(stored[(r["utt_id"], r["condition"])], f"{m}_{arm}")) for r in shared]
                out[f"max_abs_{m}_{arm}"] = max(diffs) if diffs else None
    if EXP029_DIV_CSV.exists() and div_rows:
        stored = {r["utt_id"]: r for r in rf.read_csv(EXP029_DIV_CSV)}
        shared = [r for r in div_rows if r["utt_id"] in stored]
        out["diversity_speakers"] = len(shared)
        for arm in ("W4", "W8"):
            diffs = [abs(_f(r, f"cvf0_{arm}") - _f(stored[r["utt_id"]], f"cvf0_{arm}")) for r in shared]
            out[f"max_abs_cvf0_{arm}"] = max(diffs) if diffs else None
    return out


def analyze(out_dir=None, prereg_path=PREREG_PATH, allow_partial=False):
    out_dir = Path(out_dir or EXP_DIR / "results")
    q_path, d_path = out_dir / "quality.csv", out_dir / "diversity.csv"
    if not q_path.exists():
        raise FileNotFoundError(f"Missing quality CSV: {q_path}")
    qual_rows = rf.read_csv(q_path, repair=True)
    div_rows = rf.read_csv(d_path, repair=True) if d_path.exists() else []
    complete = len(qual_rows) >= QUAL_N * len(CONDS) and len(div_rows) >= DIV_N
    if not allow_partial and not complete:
        raise RuntimeError(f"Incomplete: quality {len(qual_rows)}/{QUAL_N * len(CONDS)}, diversity {len(div_rows)}/{DIV_N}")
    primary = evaluate(qual_rows, div_rows, *PRIMARY)
    secondary = evaluate(qual_rows, div_rows, *SECONDARY)
    results = {
        "experiment": "EXP-030",
        "partial": not complete,
        "verdict": verdict(primary, complete),
        "secondary_verdict_TOH2": verdict(secondary, complete),
        "n_quality_rows": len(qual_rows),
        "n_diversity_rows": len(div_rows),
        "primary": primary,
        "secondary": secondary,
        "condition_quality": condition_tables(qual_rows),
        "clean_diversity": diversity_table(div_rows),
        "exp029_reproduction": exp029_reproduction(qual_rows, div_rows),
        "utc": tb._utc(),
    }
    rjson = out_dir / "timing_only_handover_results.json"
    rmd = out_dir / "TIMING_ONLY_HANDOVER_REPORT.md"
    with open(rjson, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(results, indent=2) + "\n")
    with open(rmd, "w", encoding="utf-8", newline="\n") as f:
        f.write(report(results) + "\n")
    entries = {}
    if Path(prereg_path).exists():
        entries["preregistration"] = {"path": str(prereg_path), "sha256": tb._sha256_file(prereg_path)}
    for p in (q_path, d_path, rjson, rmd):
        if p.exists():
            entries[p.name] = {"sha256": tb._sha256_file(p), "bytes": p.stat().st_size}
    with open(out_dir / "MANIFEST.json", "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps({"experiment": "EXP-030", "utc": tb._utc(), "files": entries}, indent=2) + "\n")
    print(f"[OK] Analysis complete: verdict={results['verdict']}, report={rmd}", flush=True)
    return results


def _ci(b, d=3):
    return tc._fmt_ci(b, d) if b else "n/a"


def report(res):
    def yn(v):
        return "n/a" if v is None else ("yes" if v else "no")

    lines = ["# EXP-030: Timing-Only Handover (TOH) pilot", "",
             f"- **Verdict (TOH3, frozen rule):** `{res['verdict']}` (partial={res['partial']})",
             f"- **TOH2 under the same rule (secondary):** `{res['secondary_verdict_TOH2']}`",
             f"- **Rows:** quality {res['n_quality_rows']} / {QUAL_N * len(CONDS)}, diversity {res['n_diversity_rows']} / {DIV_N}",
             "", "## Hypotheses", "", "| | TOH3 (vs W3) | TOH2 (vs W2) |", "| --- | --- | --- |"]
    p, s = res["primary"], res["secondary"]
    rows = (("H1a CVf0 vs STU", "H1a", "H1a_ci", 4), ("H1b CVf0 vs W_k", "H1b", "H1b_ci", 4),
            ("H2a noise UTMOS vs W4", "H2a", "H2a_ci", 3), ("H2b noise UTMOS vs STU", "H2b", "H2b_ci", 3),
            ("H3 clean dUTMOS vs W8", "H3", "H3_utmos", 3))
    for label, key, ci, d in rows:
        lines.append(f"| {label} | {yn(p[key])}: {_ci(p.get(ci), d)} | {yn(s[key])}: {_ci(s.get(ci), d)} |")
    for cond, tab in res["condition_quality"].items():
        lines += ["", f"## Quality: `{cond}` (n={tab['n']})", "",
                  "| Arm | Passes | UTMOS | dUTMOS vs W8 | dUTMOS vs STU | WER mean / median | Runaway (WER > 1) | SIM |",
                  "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for arm in ARMS:
            e = tab["arms"][arm]
            vs8 = _ci(e["vs_W8"]["utmos"]["boot"]) if "vs_W8" in e else "ref"
            vss = _ci(e["vs_STU"]["utmos"]["boot"]) if "vs_STU" in e else "ref"
            lines.append(f"| `{arm}` | {COST[arm]} | {e['utmos']['mean']:.3f} | {vs8} | {vss} | "
                         f"{e['wer']['mean']:.4f} / {e['median_wer']:.4f} | {e['runaway']} | {e['similarity']['mean']:.4f} |")
    div = res["clean_diversity"]
    if div:
        lines += ["", f"## Clean pitch variety (n={div['n']} speakers x {DIV_SEEDS} seeds)", "",
                  "| Arm | CVf0 | voiced share | slow-mel spread |", "| --- | ---: | ---: | ---: |"]
        for arm in ARMS:
            a = div["arms"][arm]
            lines.append(f"| `{arm}` | {a['cvf0']['mean']:.4f} | {a['voiced']['mean']:.3f} | {a['mel_spread']['mean']:.4f} |")
        lines += ["", "| CVf0 contrast | Mean [95% CI] | Speakers above zero |", "| --- | --- | ---: |"]
        for k, v in div["cvf0_contrasts"].items():
            lines.append(f"| {k} | {_ci(v['boot'], 4)} | {v['sign']['pos']} / {v['sign']['n']} |")
        lines.append(f"\nMean map displacement (frames): TOH2 {div['warp_rms_2']['mean']:.2f}, TOH3 {div['warp_rms_3']['mean']:.2f}.")
    rep = res["exp029_reproduction"]
    if rep:
        lines += ["", "## Reproduction of EXP-029 (shared cases)", "", "```", json.dumps(rep, indent=2), "```"]
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description="EXP-030 Timing-Only Handover pilot")
    sub = p.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("freeze")
    f.add_argument("--force", action="store_true")
    for name in ("smoke", "run"):
        s = sub.add_parser(name)
        s.add_argument("--dmo_root", required=True)
        s.add_argument("--out_dir", default=str(EXP_DIR / "results"))
        s.add_argument("--device", default="cuda")
        if name == "run":
            s.add_argument("--stage", choices=("all", "quality", "diversity"), default="all")
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
        if args.stage in ("all", "quality"):
            run_quality(pipe, qual_items(), out_dir=args.out_dir)
        if args.stage in ("all", "diversity"):
            run_diversity(pipe, div_items(), out_dir=args.out_dir)
    elif args.cmd == "analyze":
        analyze(args.out_dir, allow_partial=args.allow_partial)


if __name__ == "__main__":
    main()
