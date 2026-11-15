"""EXP-031 (EVAL-024): three single-formula handovers against the previous handover methods, with a teacher-free control.

Amendment 1 (2026-11-15, before any synthesis): the primary methods are three handovers applied to every input, with no
routing. DMOSpeech 2's teacher runs 3 steps and the student finishes; only what crosses differs:
  - TOH3 (EXP-030): the teacher's timing, as a time map that re-times the student's own one-step draft S0;
  - QH3 (Quantile Handover): in every mel band, S0's values placed in the time order of the teacher's estimate, so the
    teacher's per-band pattern crosses but every value is the student's;
  - BTH3 (Blind-Teacher Handover): TOH's time map, from a teacher that never receives the prompt (target text only, over
    the target's frames, audio condition dropped in both guidance branches), so no property of the prompt can reach the
    teacher.
Each costs 12 passes; BTH3's teacher passes run on the target's frames only.

They are compared on the same cases with the previous methods, all with thresholds fixed by earlier work:
  - W4 (DMOSpeech 2's code default, 13 passes) and W8 (the 21-pass reference);
  - STAC (EXP-018): hand over at the first step k in {4, 5, 6} whose cumulative drift A_k reaches lambda_k, else 8;
  - CH3 (Chapter 3's stopping policy): SNR gate, then the first step k in {4, 5, 6} with d_k <= 0.35, else 8;
  - CH3_LTT: the same policy with the LTT-certified threshold 1.20.
RND, the teacher-free control, re-times S0 with a smooth random map of about the size of TOH3's maps in EXP-030
(4 passes). The SNR-routed versions (SNR_TOH, SNR_W4, SNR_RND) of the first design are kept as exploratory.

Development experiment on speakers that have not been through TOH (development 1-30; EXP-030 used 61-90). STAC's and
CH3's thresholds were set on development data that include these speakers, which favours them. Subcommands: freeze,
smoke, run, analyze.
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
import run_timing_only_handover as toh  # noqa: E402
import run_transmissibility_bridge as tb  # noqa: E402
import run_transmissibility_confirm as tc  # noqa: E402
import run_transmissibility_test as tt  # noqa: E402

THESIS_ROOT = rf.THESIS_ROOT
CODE_DIR = Path(__file__).resolve().parent
EXP_DIR = THESIS_ROOT / "research/experiments/EXP-031-snr-toh-method-comparison"
PREREG_PATH = EXP_DIR / "preregistration.json"
EXP027_QUAL_CSV = tb.EXP027_QUAL_CSV
EXP011_PILOT_CSV = cg.PILOT_GRID
EXP030_QUAL_CSV = THESIS_ROOT / "research/experiments/EXP-030-timing-only-handover/results/quality.csv"
FAILURE_CONFIG = THESIS_ROOT / "code/scheduling/failure_config.json"

DEPENDENCIES = ("run_method_comparison.py", "run_timing_only_handover.py") + tuple(dts.DEPENDENCIES)

QUAL_N = 30
DIV_N = 20
DIV_SEEDS = 12
DIV_SEED_BASE = 1000
CONDS = ("clean", "gauss25", "noisy15", "reverb03", "reverb09", "babble15")
DEGRADED = CONDS[1:]
HIDDEN = ("reverb03", "reverb09", "babble15")   # degradations other than white noise
SEED = 2026
BOOT_B = 2000

METRICS = ("wer", "similarity", "utmos")
CANDIDATES = ("TOH3", "QH3", "BTH3")
ARMS = ("STU", "W4", "W5", "W6", "W8") + CANDIDATES + ("RND",)
DIV_ARMS = ("STU", "W4", "W8") + CANDIDATES + ("RND",)
TEACHER_STEPS = (3, 4, 5, 6, 8)
HANDOVER_STEP = 3
COST = {"STU": 4, "W4": 13, "W5": 15, "W6": 17, "W8": 21, "TOH3": 12, "QH3": 12, "BTH3": 12, "RND": 4}

# Routing, all fixed by earlier work (copied here; the unit tests check them against their sources).
THETA = 20.0                       # Chapter 3's SNR gate (scheduling/sampler.py default; run_consistency_gate_prelim.THETA)
CH3_CANDIDATES = (4, 5, 6)         # scheduling/sampler.py CANDIDATE_STEPS
CH3_REFERENCE = 8                  # scheduling/sampler.py REFERENCE_STEP
CH3_LAMBDA = 0.35                  # Appendix D: speaker-grouped four-fold CV chose 0.35 in three of four folds
CH3_LAMBDA_LTT = 1.20              # Appendix D: fixed-sequence LTT on the 200 calibration speakers certified up to 1.20
STAC_LAMS = {4: 1.51279, 5: 1.64678, 6: 1.71607}   # run_swsr_handover_prelim.DEFAULT_STAC_LAMS (EXP-018)

PREVIOUS = ("W4", "W8", "STAC", "CH3", "CH3_LTT")
ROUTED = ("STAC", "CH3", "CH3_LTT", "SNR_W4", "SNR_TOH", "SNR_RND")
METHODS = ("STU", "RND", "W4", "W8") + CANDIDATES + ("STAC", "CH3", "CH3_LTT", "SNR_W4", "SNR_TOH", "SNR_RND")

# Teacher-free random map, sized from EXP-030's TOH3 maps on clean prompts (RMS displacement 0.0270 of the length on
# average, largest displacement 1.98 x RMS). In simulation over 250-750 frames these settings give 0.0261 and 1.90.
RND_RMS_FRAC = 0.029
RND_SIGMA_FRAC = 0.10
RND_SLOPE_DEV = 0.45
RND_SEED_OFFSET = 7919

# Bars (EXP-029's clean bars).
SPLIT_TOL = 1e-5
REPRO_TOL = 0.01
ZERO_TOL = 1e-12
CLEAN_UTMOS_BAR = -0.08
WER_BAR = 0.01
SIM_BAR = -0.01
VARIETY_MARGIN = 0.010
RUNAWAY_WER = 1.0

QUAL_FIELDS = (
    ["utt_id", "speaker_id", "condition", "prompt_snr", "gpu", "frames", "seq_len", "blind_seq_len", "warp_band",
     "toh_rms", "toh_max", "bth_rms", "bth_max", "qh_shift", "rnd_rms", "rnd_max", "rnd_shrink"]
    + [f"d_{j}" for j in range(1, 9)]
    + ["stac_step", "ch3_step", "ch3_ltt_step"]
    + [f"{m}_{a}" for a in ARMS for m in METRICS + ("hypothesis",)]
    + ["case_seconds", "utc"]
)
DIV_MEASURES = ("cvf0", "voiced", "mel_spread", "cvf0_aligned", "voiced_aligned", "timing_spread")
DIV_FIELDS = (
    ["utt_id", "speaker_id", "condition", "n_seeds", "toh_rms_mean", "bth_rms_mean", "rnd_rms_mean"]
    + [f"{m}_{a}" for a in DIV_ARMS for m in DIV_MEASURES]
    + ["case_seconds", "utc"]
)


def qual_items(n=QUAL_N):
    """Development speakers 1-30 by utterance ID. None has been through TOH (EXP-030 used 61-90). The first 20 are the
    EXP-011 pilot speakers with the same seeds, whose stored W4 and W8 scores give a reproduction check."""
    return cg.dev_items()[:n]


def div_items(n=DIV_N):
    """Development speakers 1-20, with seeds input_seed + 1000 + s as in EXP-029 and EXP-030."""
    return cg.dev_items()[:n]


# --------------------------------------------------------------------------- payloads
def random_map(n, seed, rms_frac=RND_RMS_FRAC, sigma_frac=RND_SIGMA_FRAC, slope_dev=RND_SLOPE_DEV, band=None):
    """A smooth random monotone time map with w[0] = 0, w[n-1] = n-1, local slopes in [1 - slope_dev, 1 + slope_dev]
    (inside TOH's [1/2, 2]) and |w - i| <= band. The displacement is Gaussian-smoothed white noise with both ends tied to
    zero, scaled to an RMS of rms_frac * n, then shrunk uniformly if it would break the slope or band limit. No teacher
    is involved. Returns (map, shrink factor)."""
    i = np.arange(n, dtype=float)
    band = toh.band_width(n) if band is None else band
    if n < 3:
        return i, 1.0
    rng = np.random.default_rng(int(seed) + RND_SEED_OFFSET)
    sig = max(1.0, sigma_frac * n)
    half = int(math.ceil(4 * sig))
    kernel = np.exp(-0.5 * (np.arange(-half, half + 1) / sig) ** 2)
    d = np.convolve(rng.standard_normal(n + 2 * half), kernel / kernel.sum(), mode="valid")
    d = d - (d[0] + (d[-1] - d[0]) * i / (n - 1))
    d[0] = d[-1] = 0.0
    rms = float(np.sqrt(np.mean(d ** 2)))
    if rms <= 0.0:
        return i, 0.0
    d *= rms_frac * n / rms
    step, peak = float(np.abs(np.diff(d)).max()), float(np.abs(d).max())
    shrink = min(1.0, slope_dev / step if step > 0 else 1.0, band / peak if peak > 0 else 1.0)
    return i + shrink * d, shrink


def quantile_handover(student_gen, teacher_gen):
    """Quantile Handover on (frames, bands) arrays (numpy or torch): in each band, the student's values sorted and
    placed at the teacher's time ranks. Every payload value is one of the student's values in the same band; the payload
    depends on the teacher only through each band's time order, so any per-band strictly increasing change of the
    teacher (a stationary channel or noise floor) leaves it unchanged."""
    if hasattr(student_gen, "clone"):
        import torch

        order = torch.sort(teacher_gen, dim=0, stable=True).indices
        values = torch.sort(student_gen, dim=0, stable=True).values
        ranks = torch.argsort(order, dim=0)          # rank of each frame in the teacher's band (inverse permutation)
        return torch.gather(values, 0, ranks)        # out-of-place: the student's tensors come from inference mode
    s = np.asarray(student_gen, dtype=float)
    order = np.argsort(np.asarray(teacher_gen, dtype=float), axis=0, kind="stable")
    out = np.empty_like(s)
    np.put_along_axis(out, order, np.sort(s, axis=0, kind="stable"), axis=0)
    return out


def quantile_payload(s0, t_k, plen):
    """QH payload on (1, frames, bands) arrays: the generated frames [plen:] re-ordered per band; prompt frames kept."""
    out = s0.clone() if hasattr(s0, "clone") else np.array(s0, copy=True)
    out[0, plen:, :] = quantile_handover(s0[0, plen:, :], t_k[0, plen:, :])
    return out


def resample_map(w, n):
    """A map over m teacher frames resampled to n teacher positions (used only if the blind teacher's length differs)."""
    w = np.asarray(w, dtype=float)
    if len(w) == n:
        return w
    return np.interp(np.linspace(0, len(w) - 1, n), np.arange(len(w), dtype=float), w)


def blind_context(pipe, item, n, seed):
    """Inputs for a teacher that never receives the prompt: the target text only, over the target's n frames. A
    two-hop silent placeholder satisfies the input routine; its condition is dropped in both guidance branches."""
    torch = pipe.torch
    audio = torch.zeros(1, 2 * pipe.tts.hop_length, device=pipe.device)
    text = pipe.convert_char_to_pinyin([item["gt_text"]])
    _, text_ids, step_cond, _, _, _ = pipe.tts._prepare_inputs(audio, text, int(n))
    m = int(step_cond.shape[1])
    gen = pipe.sampler.make_local_generator(int(seed))
    y0 = torch.randn(m, 100, dtype=torch.float32, generator=gen).unsqueeze(0).to(pipe.device)
    return {"text_ids": text_ids, "step_cond": step_cond, "y0": y0, "m": m}


def blind_teacher_endpoint(pipe, bctx, k=HANDOVER_STEP):
    """The released guided Euler teacher on the 16-interval grid, steps 0..k, with the audio condition dropped in both
    branches (text-only against unconditioned guidance). Returns the endpoint estimate at step k, shape (1, m, 100)."""
    torch, sampler = pipe.torch, pipe.sampler
    grid = tt.grid16(pipe)
    x = sh._clone(bctx["y0"])
    ep = None
    with torch.inference_mode():
        for i in range(k + 1):
            t = torch.tensor(grid[i], device=pipe.device, dtype=torch.float32)
            kw = dict(x=x, cond=bctx["step_cond"], text=bctx["text_ids"], time=t, mask=None)
            vc = pipe.tts.teacher(drop_audio_cond=True, drop_text=False, **kw)
            vu = pipe.tts.teacher(drop_audio_cond=True, drop_text=True, **kw)
            v = vc + (vc - vu) * sampler.CFG_STRENGTH
            ep = x + (1.0 - grid[i]) * v
            if i < k:
                x = x + (grid[i + 1] - grid[i]) * v
    return ep


def blind_payload(pipe, item, ctx, s0):
    """BTH payload: S0 re-timed along the time map from the blind teacher's estimate. Returns (payload, map, band, m)."""
    plen = ctx["prompt_len"]
    n = s0.shape[1] - plen
    bctx = blind_context(pipe, item, n, ctx["seed"])
    tb_ep = blind_teacher_endpoint(pipe, bctx)
    w, band = toh.timing_map(toh._as_numpy(s0[0, plen:, :]), toh._as_numpy(tb_ep[0]))
    w = resample_map(w, n)
    return toh.apply_timing_map(s0, w, plen), w, band, bctx["m"]


# --------------------------------------------------------------------------- routing
def ch3_step(drifts, snr, lam, theta=THETA, cands=CH3_CANDIDATES, ref=CH3_REFERENCE):
    """Chapter 3's stopping policy (Definition of the stabilization stopping policy; sampler.evaluate_handover_policy):
    the first candidate k with d_k <= lam if the prompt SNR passes the gate, else the reference step."""
    if snr < theta:
        return ref
    for k in cands:
        d = drifts.get(k)
        if d is not None and d <= lam:
            return k
    return ref


def stac_step(drifts, lams=None, cands=CH3_CANDIDATES, ref=CH3_REFERENCE):
    """STAC (run_swsr_handover_prelim.live_early_exit_teacher): the first candidate k whose cumulative drift
    A_k = d_1 + ... + d_k reaches lambda_k, else the reference step. No SNR."""
    lams = STAC_LAMS if lams is None else lams
    for k in cands:
        action = sum(drifts[j] for j in range(1, k + 1) if j in drifts)
        if k in lams and action >= lams[k]:
            return k
    return ref


def routed_arm(method, snr, drifts):
    """The arm whose output a method returns for a case with this prompt SNR and teacher drift trace."""
    if method in ARMS:
        return method
    if method == "STAC":
        return f"W{stac_step(drifts)}"
    if method == "CH3":
        return f"W{ch3_step(drifts, snr, CH3_LAMBDA)}"
    if method == "CH3_LTT":
        return f"W{ch3_step(drifts, snr, CH3_LAMBDA_LTT)}"
    upper = {"SNR_W4": "W4", "SNR_TOH": "TOH3", "SNR_RND": "RND"}[method]
    return upper if snr >= THETA else "STU"


# --------------------------------------------------------------------------- variety after alignment
def aligned_cvf0(tracks, mels):
    """CVf0 after aligning every seed to the first one in time, so timing differences between seeds no longer count.

    Each seed's generated mel is aligned to seed 0's with TOH's constrained DTW, and its f0 track (pYIN, same 256-sample
    hop as the mel) is read at the nearest aligned frame. Returns (CVf0, share of frames kept, timing spread), where the
    timing spread is the mean RMS displacement of those alignment maps in frames."""
    ref = np.asarray(mels[0], dtype=float)
    n = min(ref.shape[0], len(tracks[0]))
    out = [np.asarray(tracks[0], dtype=float)[:n]]
    spreads = []
    for track, mel in zip(tracks[1:], mels[1:]):
        w, _ = toh.timing_map(np.asarray(mel, dtype=float), ref)
        t = np.asarray(track, dtype=float)
        idx = np.clip(np.rint(w[:n]).astype(int), 0, len(t) - 1)
        out.append(t[idx])
        spreads.append(toh.map_stats(w)[0])
    cv, voiced = sh.cvf0(out)
    return cv, voiced, float(np.mean(spreads)) if spreads else 0.0


# --------------------------------------------------------------------------- synthesis
def synthesize_arms(pipe, ctx, ends, arms, item):
    """The requested arms from one teacher trajectory, one blind-teacher run and one student draft; every arm uses the
    same restart noises."""
    taus = list(pipe.sampler.STUDENT_TAUS)
    plen = ctx["prompt_len"]
    s0 = gl.student(pipe, ctx, ctx["step_cond"], [0.0])
    n = s0.shape[1] - plen
    outs = {}
    if "STU" in arms:
        outs["STU"] = gl.student(pipe, ctx, s0, taus)
    for k in (4, 5, 6, 8):
        if f"W{k}" in arms:
            outs[f"W{k}"] = gl.student(pipe, ctx, ends[k], taus)
    x3, w3, band = toh.timing_handover(s0, ends[HANDOVER_STEP], plen)
    outs["TOH3"] = gl.student(pipe, ctx, x3, taus)
    xq = quantile_payload(s0, ends[HANDOVER_STEP], plen)
    outs["QH3"] = gl.student(pipe, ctx, xq, taus)
    xb, wb, _, m = blind_payload(pipe, item, ctx, s0)
    outs["BTH3"] = gl.student(pipe, ctx, xb, taus)
    wr, shrink = random_map(n, ctx["seed"], band=band)
    outs["RND"] = gl.student(pipe, ctx, toh.apply_timing_map(s0, wr, plen), taus)
    r3, m3 = toh.map_stats(w3)
    rb, mb = toh.map_stats(wb)
    rr, mr = toh.map_stats(wr)
    qd = toh._as_numpy(xq[0, plen:, :]) - toh._as_numpy(ends[HANDOVER_STEP][0, plen:, :])
    stats = {"frames": n, "seq_len": int(s0.shape[1]), "blind_seq_len": m, "warp_band": band, "toh_rms": r3,
             "toh_max": m3, "bth_rms": rb, "bth_max": mb, "qh_shift": float(np.sqrt(np.mean(qd ** 2))),
             "rnd_rms": rr, "rnd_max": mr, "rnd_shrink": shrink}
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
        ends, tel = dts.teacher_trajectory(pipe, ctx, wanted=TEACHER_STEPS)
        outs, stats = synthesize_arms(pipe, ctx, ends, ARMS, item)
        drifts = {j: float(tel[f"d_{j}"]) for j in range(1, 9)}
        snr = float(ctx["prompt_snr"])
        gt = pipe.ground_truth(item)
        row = {"utt_id": item["utt_id"], "speaker_id": item["speaker_id"], "condition": cond, "prompt_snr": snr,
               "gpu": pipe.gpu, **stats, **{f"d_{j}": drifts[j] for j in range(1, 9)},
               "stac_step": stac_step(drifts), "ch3_step": ch3_step(drifts, snr, CH3_LAMBDA),
               "ch3_ltt_step": ch3_step(drifts, snr, CH3_LAMBDA_LTT)}
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
        tracks = {a: [] for a in DIV_ARMS}
        mels = {a: [] for a in DIV_ARMS}
        slow = {a: [] for a in DIV_ARMS}
        toh_rms, bth_rms, rnd_rms = [], [], []
        for s in range(seeds):
            seed = int(item["input_seed"]) + DIV_SEED_BASE + s
            ctx = pipe.prepare(item, "clean", seed, out_dir, noise_seed=item["input_seed"])
            ends, _ = dts.teacher_trajectory(pipe, ctx, wanted=(HANDOVER_STEP, 4, 8))
            outs, stats = synthesize_arms(pipe, ctx, ends, DIV_ARMS, item)
            toh_rms.append(stats["toh_rms"])
            bth_rms.append(stats["bth_rms"])
            rnd_rms.append(stats["rnd_rms"])
            arm_tracks = dts._track_arms_fast(pipe, ctx, outs, DIV_ARMS)
            plen = ctx["prompt_len"]
            for arm in DIV_ARMS:
                tracks[arm].append(arm_tracks[arm])
                mels[arm].append(toh._as_numpy(outs[arm][0, plen:, :]))
                slow[arm].append(tc._slow_component(outs[arm], plen))
        row = {"utt_id": item["utt_id"], "speaker_id": item["speaker_id"], "condition": "clean", "n_seeds": seeds,
               "toh_rms_mean": statistics.mean(toh_rms), "bth_rms_mean": statistics.mean(bth_rms),
               "rnd_rms_mean": statistics.mean(rnd_rms)}
        for arm in DIV_ARMS:
            row[f"cvf0_{arm}"], row[f"voiced_{arm}"] = sh.cvf0(tracks[arm])
            row[f"mel_spread_{arm}"] = tc.slow_mel_spread(slow[arm])
            (row[f"cvf0_aligned_{arm}"], row[f"voiced_aligned_{arm}"],
             row[f"timing_spread_{arm}"]) = aligned_cvf0(tracks[arm], mels[arm])
        row.update({"case_seconds": round(time.perf_counter() - t0, 2), "utc": tb._utc()})
        tb._append(path, DIV_FIELDS, row)
        print(f"[{n_run}/{len(todo)}] diversity: {item['utt_id']} ({row['case_seconds']:.1f} s)", flush=True)
    print(f"[OK] diversity complete: {path}", flush=True)
    return path


def random_map_ok(w, n, band):
    d = np.diff(w)
    return bool(w[0] == 0.0 and abs(w[-1] - (n - 1)) <= 1e-9 and np.all(d >= 1.0 - RND_SLOPE_DEV - 1e-9)
                and np.all(d <= 1.0 + RND_SLOPE_DEV + 1e-9) and np.max(np.abs(w - np.arange(n))) <= band + 1e-9)


def toh_map_ok(w, n, band):
    """The EXP-030 smoke test's check of a TOH-type map onto n student frames."""
    dw = np.diff(w)
    return bool(w[0] == 0.0 and n - 1.5 <= w[-1] <= n - 1 and np.all(dw >= -1e-9) and np.all(dw <= 2.0 + 1e-9)
                and np.max(np.abs(w - np.arange(len(w)) * (n - 1) / max(len(w) - 1, 1))) <= band + 1.0)


def smoke(pipe, out_dir, prereg_path=PREREG_PATH):
    """Identity checks (split student, identity map, TOH with the identity map = student alone, QH of the draft onto
    itself = the draft, QH values = the draft's values per band), checks of the TOH3, BTH3 and random maps and of the
    three policies on real teacher runs, blind-teacher determinism, EXP-027 W4 reproduction, then micro runs of both
    stages (the quality micro run includes babble15)."""
    check_prereg(prereg_path)
    sdir = Path(out_dir) / "smoke"
    sdir.mkdir(parents=True, exist_ok=True)
    stored = {r["utt_id"]: r for r in rf.read_csv(EXP027_QUAL_CSV)}
    taus = list(pipe.sampler.STUDENT_TAUS)
    torch = pipe.torch
    checks = []
    for item in tb.smoke_items():
        ctx = pipe.prepare(item, "clean", int(item["input_seed"]), sdir)
        ends, tel = dts.teacher_trajectory(pipe, ctx, wanted=TEACHER_STEPS)
        plen = ctx["prompt_len"]
        s0 = gl.student(pipe, ctx, ctx["step_cond"], [0.0])
        stu_split = gl.student(pipe, ctx, s0, taus)
        stu_full = gl.student(pipe, ctx, ctx["step_cond"], [0.0] + taus)
        n = s0.shape[1] - plen
        x_id = toh.apply_timing_map(s0, np.arange(n, dtype=float), plen)
        toh_id = gl.student(pipe, ctx, x_id, taus)
        _, w3, band = toh.timing_handover(s0, ends[HANDOVER_STEP], plen)
        map_ok = toh_map_ok(w3, n, band)
        wr, shrink = random_map(n, ctx["seed"], band=band)
        wr_again, _ = random_map(n, ctx["seed"], band=band)
        rnd_ok = random_map_ok(wr, n, band) and bool(np.array_equal(wr, wr_again))
        qh_self = quantile_payload(s0, s0, plen)
        xq = quantile_payload(s0, ends[HANDOVER_STEP], plen)
        qh_values = float((torch.sort(xq[0, plen:, :], dim=0).values - torch.sort(s0[0, plen:, :], dim=0).values)
                          .abs().max())
        xb, wb, bband, m = blind_payload(pipe, item, ctx, s0)
        xb_again, _, _, _ = blind_payload(pipe, item, ctx, s0)
        bth_ok = toh_map_ok(wb, n, bband) and bool(np.all(np.isfinite(toh._as_numpy(xb))))
        drifts = {j: float(tel[f"d_{j}"]) for j in range(1, 9)}
        snr = float(ctx["prompt_snr"])
        steps = {"stac": stac_step(drifts), "ch3": ch3_step(drifts, snr, CH3_LAMBDA),
                 "ch3_ltt": ch3_step(drifts, snr, CH3_LAMBDA_LTT)}
        policy_ok = (all(k in CH3_CANDIDATES + (CH3_REFERENCE,) for k in steps.values())
                     and ch3_step(drifts, snr, -1.0) == CH3_REFERENCE and ch3_step(drifts, -1.0, 99.0) == CH3_REFERENCE)
        rec = {"utt_id": item["utt_id"], "split_diff": tb._max_abs(stu_split - stu_full),
               "identity_map_diff": tb._max_abs(x_id - s0), "toh_identity_vs_student": tb._max_abs(toh_id - stu_full),
               "qh_self_diff": tb._max_abs(qh_self - s0), "qh_value_diff": qh_values,
               "blind_repeat_diff": tb._max_abs(xb - xb_again), "map_ok": map_ok, "bth_ok": bth_ok, "rnd_ok": rnd_ok,
               "policy_ok": policy_ok, "steps": steps, "prompt_snr": snr, "toh_rms": toh.map_stats(w3)[0],
               "bth_rms": toh.map_stats(wb)[0], "rnd_rms": toh.map_stats(wr)[0], "rnd_shrink": shrink, "band": band,
               "seq_len": int(s0.shape[1]), "blind_seq_len": m}
        mw = pipe.score(ng.decode(pipe, ctx, gl.student(pipe, ctx, ends[4], taus)), item, pipe.ground_truth(item))
        ref = stored[item["utt_id"]]
        rec["d_wer"] = abs(mw["wer"] - float(ref["wer_W4"]))
        rec["d_sim"] = abs(mw["similarity"] - float(ref["similarity_W4"]))
        print(f"[smoke] {json.dumps(rec)}", flush=True)
        checks.append(rec)
        if max(rec["split_diff"], rec["identity_map_diff"], rec["toh_identity_vs_student"], rec["qh_self_diff"],
               rec["qh_value_diff"], rec["blind_repeat_diff"]) > SPLIT_TOL:
            raise RuntimeError(f"Identity check failed: {rec}")
        if not (map_ok and bth_ok and rnd_ok and policy_ok):
            raise RuntimeError(f"Map or policy check failed: {rec}")
        if rec["d_wer"] > REPRO_TOL or rec["d_sim"] > REPRO_TOL:
            raise RuntimeError(f"EXP-027 W4 reproduction failed: {rec}")
    run_quality(pipe, tb.smoke_items()[:1], conds=("clean", "babble15"), out_dir=sdir, csv_name="quality.csv")
    run_diversity(pipe, tb.smoke_items(), sdir, seeds=2, csv_name="diversity.csv")
    summary = {"checks": checks, "demand_sha256": getattr(pipe, "babble_sha", None), "utc": tb._utc()}
    (sdir / "smoke_check.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"[OK] smoke passed ({summary})", flush=True)
    return summary


# --------------------------------------------------------------------------- preregistration
HYPOTHESES = {
    "T1": "For each candidate X in TOH3, QH3, BTH3: clean pitch variety CVf0(X) - CVf0(RND), 95% bootstrap lower "
          "bound > 0 (the teacher adds variety beyond a random re-timing of the same size).",
    "T2": "For each candidate X: clean UTMOS(X) - UTMOS(RND), lower bound > 0 (the teacher's contribution sounds more "
          "natural than random re-timing).",
    "C": "For each candidate X: per-speaker mean over reverb03, reverb09 and babble15 of UTMOS(X) - UTMOS(W4), lower "
         "bound > 0 (X keeps recording conditions other than white noise out better than DMOSpeech 2's handover).",
    "M": "Candidate X beats a previous method C if, against C, (a) clean prompts: mean dUTMOS >= -0.08, dWER upper bound "
         "<= +0.01 and dSIM lower bound >= -0.01, and (b) the per-speaker mean over the five degraded conditions: "
         "dUTMOS lower bound > 0, dWER upper bound <= +0.01 and dSIM lower bound >= -0.01. Previous methods: W4, W8, "
         "STAC, CH3 and CH3_LTT.",
}
VERDICT_RULE = (
    "Per candidate X in TOH3, QH3 and BTH3, three labels: teacher = teacher_needed if T1 or T2 holds, else "
    "teacher_not_shown_needed; method = beats_all_previous if M holds against all five previous methods, "
    "beats_some_previous if against at least one, else beats_no_previous; condition = condition_safety_shown if C "
    "holds, else condition_safety_not_shown. partial if a stage is missing. Candidates are not ranked by the frozen "
    "rule; no correction for the three candidates (development pilot). Exploratory: contrasts between candidates "
    "(including aligned CVf0 QH3 - TOH3 and reverb/babble UTMOS BTH3 - TOH3), the SNR-routed methods SNR_TOH, SNR_W4 "
    "and SNR_RND, the first design's S1 (SNR_TOH - SNR_W4 on gated-in reverb and babble cases), STU against the "
    "previous methods, per-condition tables, aligned CVf0 and timing spread, BTH3's teacher sequence length, the "
    "EXP-030 replication, Definition 3.1 failures against W8, the policies' step choices, runaway outputs (WER > 1) and "
    "the EXP-011 reproduction."
)
AMENDMENTS = [
    {"id": 1, "date": "2026-11-15", "before_synthesis": True, "first_freeze_sha256": "ec06ed0568b4",
     "change": "Added QH3 (Quantile Handover) and BTH3 (Blind-Teacher Handover) as arms in both stages; the primary "
               "methods are now the three single-formula handovers TOH3, QH3 and BTH3 applied to every input, each "
               "judged by T1, T2, C and M; SNR_TOH and the first design's S1 are exploratory."},
]


def build_prereg():
    q = [{"utt_id": it["utt_id"], "speaker_id": it["speaker_id"]} for it in qual_items()]
    d = [{"utt_id": it["utt_id"], "speaker_id": it["speaker_id"]} for it in div_items()]
    inputs = (EXP027_QUAL_CSV, EXP011_PILOT_CSV, EXP030_QUAL_CSV, FAILURE_CONFIG)
    payload = {
        "experiment": "EXP-031",
        "task_id": "EVAL-024",
        "title": "Three single-formula handovers (TOH, QH, BTH) against the previous handover methods, with a "
                 "teacher-free control",
        "framing": ("Development experiment. DMOSpeech 2's teacher runs 3 steps and the student finishes; the three "
                    "candidates differ only in what crosses (TOH: timing; QH: per-band time order with the student's "
                    "values; BTH: timing from a teacher that never receives the prompt) and are applied to every input "
                    "with no routing. Compared on the same cases with W4, W8, STAC, CH3 and CH3_LTT (thresholds fixed by "
                    "earlier work) and with a teacher-free random re-timing (RND). Speakers have not been through TOH."),
        "amendments": AMENDMENTS,
        "candidates": list(CANDIDATES),
        "handover_step": HANDOVER_STEP,
        "routing": {"theta_db": THETA, "ch3_candidates": list(CH3_CANDIDATES), "ch3_reference": CH3_REFERENCE,
                    "ch3_lambda": CH3_LAMBDA, "ch3_lambda_ltt": CH3_LAMBDA_LTT,
                    "stac_lambdas": {str(k): v for k, v in STAC_LAMS.items()}},
        "random_map": {"rms_frac": RND_RMS_FRAC, "sigma_frac": RND_SIGMA_FRAC, "slope_dev": RND_SLOPE_DEV,
                       "seed_offset": RND_SEED_OFFSET},
        "toh": {"n_groups": toh.N_GROUPS, "feature_smooth": toh.FEAT_SMOOTH, "band_frac": toh.BAND_FRAC,
                "min_band": toh.MIN_BAND},
        "qh": "per band, the student's draft values sorted and placed at the teacher's time ranks (stable sort)",
        "bth": ("teacher input = target text only over the target's frames, two-hop silent placeholder audio, audio "
                "condition dropped in both guidance branches, same seed as the case; TOH map onto the student's draft"),
        "conditions": list(CONDS),
        "degraded_conditions": list(DEGRADED),
        "hidden_conditions": list(HIDDEN),
        "arms": list(ARMS),
        "diversity_arms": list(DIV_ARMS),
        "methods": list(METHODS),
        "previous": list(PREVIOUS),
        "costs": dict(COST),
        "n_qual_speakers": len(q),
        "n_qual_cases": len(q) * len(CONDS),
        "qual_utts": q,
        "n_div_speakers": len(d),
        "div_seeds": DIV_SEEDS,
        "div_seed_base": DIV_SEED_BASE,
        "div_utts": d,
        "bars": {"clean_utmos": CLEAN_UTMOS_BAR, "wer": WER_BAR, "sim": SIM_BAR, "variety_margin": VARIETY_MARGIN,
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


def _boot(vals):
    return tc.boot_ci(list(vals), B=BOOT_B, seed=SEED)


def drifts_of(r):
    return {j: _f(r, f"d_{j}") for j in range(1, 9)}


def arm_for(method, r):
    return routed_arm(method, _f(r, "prompt_snr"), drifts_of(r))


def value(r, method, metric):
    return _f(r, f"{metric}_{arm_for(method, r)}")


def per_speaker(rows, fn):
    """Per-speaker mean of fn(row) over the given rows."""
    acc = {}
    for r in rows:
        acc.setdefault(r["speaker_id"], []).append(fn(r))
    return [statistics.mean(v) for _, v in sorted(acc.items())]


def compare(qual_rows, a, b):
    """Method a against method b under hypothesis M."""
    clean = [r for r in qual_rows if r["condition"] == "clean"]
    deg = [r for r in qual_rows if r["condition"] in DEGRADED]
    out = {"a": a, "b": b, "clean_ok": False, "degraded_ok": False}
    if clean:
        for m in METRICS:
            out[f"clean_{m}"] = _boot([value(r, a, m) - value(r, b, m) for r in clean])
        out["clean_ok"] = (out["clean_utmos"]["mean"] >= CLEAN_UTMOS_BAR and out["clean_wer"]["hi"] <= WER_BAR
                           and out["clean_similarity"]["lo"] >= SIM_BAR)
    if deg:
        for m in METRICS:
            out[f"degraded_{m}"] = _boot(per_speaker(deg, lambda r, m=m: value(r, a, m) - value(r, b, m)))
        out["degraded_ok"] = (out["degraded_utmos"]["lo"] > ZERO_TOL and out["degraded_wer"]["hi"] <= WER_BAR
                              and out["degraded_similarity"]["lo"] >= SIM_BAR)
        out["by_condition_utmos"] = {
            c: _boot([value(r, a, "utmos") - value(r, b, "utmos") for r in deg if r["condition"] == c])
            for c in DEGRADED if any(r["condition"] == c for r in deg)}
    out["beats"] = bool(out["clean_ok"] and out["degraded_ok"])
    out["mean_cost"] = {a: statistics.mean(COST[arm_for(a, r)] for r in qual_rows),
                        b: statistics.mean(COST[arm_for(b, r)] for r in qual_rows)}
    return out


def teacher_test(qual_rows, div_rows, arm):
    res = {"arm": arm, "T1": None, "T2": None}
    if len(div_rows) >= 2:
        t1 = _boot([_f(r, f"cvf0_{arm}") - _f(r, "cvf0_RND") for r in div_rows])
        res.update({"T1": t1["lo"] > ZERO_TOL, "T1_ci": t1,
                    "aligned_ci": _boot([_f(r, f"cvf0_aligned_{arm}") - _f(r, "cvf0_aligned_RND") for r in div_rows])})
    clean = [r for r in qual_rows if r["condition"] == "clean"]
    if clean:
        t2 = _boot([_f(r, f"utmos_{arm}") - _f(r, "utmos_RND") for r in clean])
        res.update({"T2": t2["lo"] > ZERO_TOL, "T2_ci": t2})
    if res["T1"] is not None and res["T2"] is not None:
        res["random_equivalent"] = bool(-VARIETY_MARGIN <= res["T1_ci"]["lo"] and res["T1_ci"]["hi"] <= VARIETY_MARGIN
                                        and CLEAN_UTMOS_BAR <= res["T2_ci"]["lo"]
                                        and res["T2_ci"]["hi"] <= -CLEAN_UTMOS_BAR)
    return res


def condition_safety(qual_rows, arm):
    rows = [r for r in qual_rows if r["condition"] in HIDDEN]
    per = per_speaker(rows, lambda r: _f(r, f"utmos_{arm}") - _f(r, "utmos_W4"))
    ci = _boot(per) if len(per) >= 2 else None
    by_cond = {c: _boot([_f(r, f"utmos_{arm}") - _f(r, "utmos_W4") for r in rows if r["condition"] == c])
               for c in HIDDEN if sum(r["condition"] == c for r in rows) >= 2}
    return {"arm": arm, "C": bool(ci and ci["lo"] > ZERO_TOL), "C_ci": ci, "n_speakers": len(per), "by_condition": by_cond}


def gated_safety_net(qual_rows):
    """The first design's S1 (exploratory): SNR_TOH - SNR_W4 on reverb and babble cases that pass the SNR gate."""
    gated = [r for r in qual_rows if r["condition"] in HIDDEN and _f(r, "prompt_snr") >= THETA]
    per = per_speaker(gated, lambda r: value(r, "SNR_TOH", "utmos") - value(r, "SNR_W4", "utmos"))
    return {"S1": bool(len(per) >= 2 and _boot(per)["lo"] > ZERO_TOL), "S1_ci": _boot(per) if len(per) >= 2 else None,
            "n_cases": len(gated), "n_speakers": len(per),
            "gated_in_by_condition": {c: sum(1 for r in gated if r["condition"] == c) for c in HIDDEN}}


def candidate_contrasts(qual_rows, div_rows):
    """Exploratory contrasts between the candidates."""
    out = {}
    hidden = [r for r in qual_rows if r["condition"] in HIDDEN]
    clean = [r for r in qual_rows if r["condition"] == "clean"]
    for a, b in (("QH3", "TOH3"), ("BTH3", "TOH3"), ("QH3", "BTH3")):
        e = {}
        if hidden:
            e["hidden_utmos"] = _boot(per_speaker(hidden, lambda r, a=a, b=b: _f(r, f"utmos_{a}") - _f(r, f"utmos_{b}")))
        if clean:
            e["clean_utmos"] = _boot(_f(r, f"utmos_{a}") - _f(r, f"utmos_{b}") for r in clean)
        if len(div_rows) >= 2:
            for m in ("cvf0", "cvf0_aligned"):
                e[m] = _boot(_f(r, f"{m}_{a}") - _f(r, f"{m}_{b}") for r in div_rows)
        out[f"{a}_minus_{b}"] = e
    if qual_rows:
        out["bth_teacher_sequence_share"] = statistics.mean(_f(r, "blind_seq_len") / _f(r, "seq_len") for r in qual_rows)
    return out


def method_tables(qual_rows):
    out = {}
    for cond in [c for c in CONDS if any(r["condition"] == c for r in qual_rows)] + ["all"]:
        rows = qual_rows if cond == "all" else [r for r in qual_rows if r["condition"] == cond]
        tab = {}
        for method in METHODS:
            arms = [arm_for(method, r) for r in rows]
            wer = [value(r, method, "wer") for r in rows]
            tab[method] = {"mean_cost": statistics.mean(COST[a] for a in arms),
                           "utmos": statistics.mean(value(r, method, "utmos") for r in rows),
                           "similarity": statistics.mean(value(r, method, "similarity") for r in rows),
                           "wer_mean": statistics.mean(wer), "wer_median": statistics.median(wer),
                           "runaway": sum(w > RUNAWAY_WER for w in wer),
                           "arms": {a: arms.count(a) for a in sorted(set(arms))}}
        out[cond] = {"n": len(rows), "methods": tab}
    return out


def diversity_table(div_rows):
    if len(div_rows) < 2:
        return None
    arms = {a: {m: _boot(_f(r, f"{m}_{a}") for r in div_rows) for m in DIV_MEASURES} for a in DIV_ARMS}
    pairs = (("TOH3", "RND"), ("QH3", "RND"), ("BTH3", "RND"), ("TOH3", "STU"), ("QH3", "STU"), ("BTH3", "STU"),
             ("RND", "STU"), ("W4", "STU"), ("QH3", "TOH3"), ("BTH3", "TOH3"), ("TOH3", "W4"), ("W8", "W4"))
    contrasts = {m: {f"{a}_minus_{b}": tb._summarize([_f(r, f"{m}_{a}") - _f(r, f"{m}_{b}") for r in div_rows], seed=SEED)
                     for a, b in pairs} for m in ("cvf0", "cvf0_aligned", "timing_spread")}
    return {"n": len(div_rows), "arms": arms, "contrasts": contrasts,
            "toh_rms": _boot(_f(r, "toh_rms_mean") for r in div_rows),
            "bth_rms": _boot(_f(r, "bth_rms_mean") for r in div_rows),
            "rnd_rms": _boot(_f(r, "rnd_rms_mean") for r in div_rows)}


def replication(qual_rows, div_rows):
    """EXP-030's frozen tests for TOH3, repeated on these speakers (exploratory here)."""
    out = {}
    if len(div_rows) >= 2:
        out["cvf0_toh3_minus_stu"] = _boot(_f(r, "cvf0_TOH3") - _f(r, "cvf0_STU") for r in div_rows)
    noise = [r for r in qual_rows if r["condition"] in ("gauss25", "noisy15")]
    if noise:
        out["noise_utmos_toh3_minus_w4"] = _boot(per_speaker(noise, lambda r: _f(r, "utmos_TOH3") - _f(r, "utmos_W4")))
        out["noise_utmos_toh3_minus_stu"] = _boot(per_speaker(noise, lambda r: _f(r, "utmos_TOH3") - _f(r, "utmos_STU")))
    clean = [r for r in qual_rows if r["condition"] == "clean"]
    if clean:
        out["clean_vs_w8"] = {m: _boot(_f(r, f"{m}_TOH3") - _f(r, f"{m}_W8") for r in clean) for m in METRICS}
    return out


def definition_3_1(qual_rows):
    """Failures against W8 under Definition 3.1 (the thesis's current yardstick), per method and condition."""
    cfg = json.loads(FAILURE_CONFIG.read_text(encoding="utf-8"))
    out = {}
    for cond in [c for c in CONDS if any(r["condition"] == c for r in qual_rows)]:
        rows = [r for r in qual_rows if r["condition"] == cond]
        out[cond] = {}
        for method in METHODS:
            n = 0
            for r in rows:
                w, s = value(r, method, "wer"), value(r, method, "similarity")
                n += (w > max(cfg["absolute_wer_ceiling"], _f(r, "wer_W8") + cfg["relative_wer_margin"])
                      or s < _f(r, "similarity_W8") - cfg["similarity_margin"])
            out[cond][method] = n
    return out


def policy_steps(qual_rows):
    out = {"mismatches_with_recorded": 0, "by_condition": {}}
    for r in qual_rows:
        snr, dr = _f(r, "prompt_snr"), drifts_of(r)
        rec = (stac_step(dr), ch3_step(dr, snr, CH3_LAMBDA), ch3_step(dr, snr, CH3_LAMBDA_LTT))
        if rec != (int(float(r["stac_step"])), int(float(r["ch3_step"])), int(float(r["ch3_ltt_step"]))):
            out["mismatches_with_recorded"] += 1
    for cond in [c for c in CONDS if any(r["condition"] == c for r in qual_rows)]:
        rows = [r for r in qual_rows if r["condition"] == cond]
        out["by_condition"][cond] = {
            "gated_in": sum(_f(r, "prompt_snr") >= THETA for r in rows),
            "prompt_snr_median": statistics.median(_f(r, "prompt_snr") for r in rows),
            **{m: {str(k): sum(arm_for(m, r) == f"W{k}" for r in rows) for k in CH3_CANDIDATES + (CH3_REFERENCE,)}
               for m in ("STAC", "CH3", "CH3_LTT")}}
    return out


def exp011_reproduction(qual_rows):
    """Prompt SNR and W4/W8 scores against the EXP-011 pilot's stored rows (same speakers 1-20 and seeds)."""
    if not EXP011_PILOT_CSV.exists():
        return None
    stored = {(r["utt_id"], r["condition"], int(r["teacher_step"])): r for r in rf.read_csv(EXP011_PILOT_CSV)
              if r["teacher_step"] in ("4", "8")}
    out = {}
    for cond in CONDS:
        shared = [r for r in qual_rows if r["condition"] == cond and (r["utt_id"], cond, 4) in stored
                  and (r["utt_id"], cond, 8) in stored]
        if not shared:
            continue
        e = {"cases": len(shared), "max_abs_prompt_snr": max(
            abs(_f(r, "prompt_snr") - float(stored[(r["utt_id"], cond, 4)]["prompt_snr"])) for r in shared)}
        for k in (4, 8):
            for m in METRICS:
                diffs = [abs(_f(r, f"{m}_W{k}") - float(stored[(r["utt_id"], cond, k)][m])) for r in shared]
                e[f"max_abs_{m}_W{k}"] = max(diffs)
                e[f"median_abs_{m}_W{k}"] = statistics.median(diffs)
        out[cond] = e
    return out


def verdicts(teacher, comparisons, safety, complete):
    out = {}
    for x in CANDIDATES:
        if not complete:
            out[x] = {"teacher": "partial", "method": "partial", "condition": "partial"}
            continue
        t = "teacher_needed" if (teacher[x]["T1"] or teacher[x]["T2"]) else "teacher_not_shown_needed"
        beaten = [c for c in PREVIOUS if comparisons[x][c]["beats"]]
        m = ("beats_all_previous" if len(beaten) == len(PREVIOUS) else
             "beats_some_previous" if beaten else "beats_no_previous")
        s = "condition_safety_shown" if safety[x]["C"] else "condition_safety_not_shown"
        out[x] = {"teacher": t, "method": m, "condition": s, "previous_beaten": beaten}
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
    teacher = {x: teacher_test(qual_rows, div_rows, x) for x in CANDIDATES}
    comparisons = {x: {c: compare(qual_rows, x, c) for c in PREVIOUS} for x in CANDIDATES}
    safety = {x: condition_safety(qual_rows, x) for x in CANDIDATES}
    exploratory = {f"{a}_vs_{c}": compare(qual_rows, a, c) for a in ("SNR_TOH", "SNR_W4", "SNR_RND", "STU")
                   for c in PREVIOUS}
    exploratory["SNR_TOH_vs_SNR_W4"] = compare(qual_rows, "SNR_TOH", "SNR_W4")
    results = {
        "experiment": "EXP-031",
        "partial": not complete,
        "verdicts": verdicts(teacher, comparisons, safety, complete),
        "n_quality_rows": len(qual_rows),
        "n_diversity_rows": len(div_rows),
        "teacher_test": teacher,
        "method_vs_previous": comparisons,
        "condition_safety": safety,
        "candidate_contrasts": candidate_contrasts(qual_rows, div_rows),
        "exploratory_comparisons": exploratory,
        "gated_safety_net_S1": gated_safety_net(qual_rows),
        "method_tables": method_tables(qual_rows),
        "clean_diversity": diversity_table(div_rows),
        "policy_steps": policy_steps(qual_rows),
        "exp030_replication": replication(qual_rows, div_rows),
        "definition_3_1_failures_vs_W8": definition_3_1(qual_rows),
        "exp011_reproduction": exp011_reproduction(qual_rows),
        "utc": tb._utc(),
    }
    rjson = out_dir / "method_comparison_results.json"
    rmd = out_dir / "METHOD_COMPARISON_REPORT.md"
    with open(rjson, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(results, indent=2) + "\n")
    with open(rmd, "w", encoding="utf-8", newline="\n") as f:
        f.write(report(results) + "\n")
    entries = {}
    if Path(prereg_path).exists():
        entries["preregistration"] = {"path": str(prereg_path), "sha256": tb._sha256_file(prereg_path)}
    for p in (q_path, d_path, rjson, rmd, out_dir / "demand_sha256.txt"):
        if p.exists():
            entries[p.name] = {"sha256": tb._sha256_file(p), "bytes": p.stat().st_size}
    with open(out_dir / "MANIFEST.json", "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps({"experiment": "EXP-031", "utc": tb._utc(), "files": entries}, indent=2) + "\n")
    print(f"[OK] Analysis complete: verdicts={results['verdicts']}, report={rmd}", flush=True)
    return results


def _ci(b, d=3):
    return tc._fmt_ci(b, d) if b else "n/a"


def _yn(v):
    return "n/a" if v is None else ("yes" if v else "no")


def report(res):
    lines = ["# EXP-031: three single-formula handovers against the previous handover methods", "",
             f"Rows: quality {res['n_quality_rows']} / {QUAL_N * len(CONDS)}, diversity {res['n_diversity_rows']} / "
             f"{DIV_N} (partial={res['partial']}).", "",
             "## Verdicts (frozen rule, per candidate)", "",
             "| Candidate | Teacher (T1, T2) | Method (M) | Previous methods beaten | Condition safety (C) |",
             "| --- | --- | --- | --- | --- |"]
    for x, v in res["verdicts"].items():
        lines.append(f"| `{x}` | `{v['teacher']}` | `{v['method']}` | {', '.join(v.get('previous_beaten', [])) or 'none'} "
                     f"| `{v['condition']}` |")
    lines += ["", "## Is the teacher needed? (each candidate against the random re-timing)", "",
              "| Candidate | T1: CVf0(X) - CVf0(RND) | T2: clean UTMOS(X) - UTMOS(RND) | Aligned CVf0 (expl.) | Within "
              "random's margins |", "| --- | --- | --- | --- | --- |"]
    for x, t in res["teacher_test"].items():
        lines.append(f"| `{x}` | {_ci(t.get('T1_ci'), 4)} ({_yn(t['T1'])}) | {_ci(t.get('T2_ci'))} ({_yn(t['T2'])}) | "
                     f"{_ci(t.get('aligned_ci'), 4)} | {_yn(t.get('random_equivalent'))} |")

    def comp_row(name, c):
        if "clean_utmos" not in c or "degraded_utmos" not in c:
            return f"| {name} | n/a | n/a | n/a | n/a | n/a | {_yn(c['beats'])} |"
        mc = c["mean_cost"]
        return (f"| {name} | {_ci(c['clean_utmos'])} | {c['clean_wer']['hi']:+.4f} / {c['clean_similarity']['lo']:+.4f} | "
                f"{_ci(c['degraded_utmos'])} | {c['degraded_wer']['hi']:+.4f} / {c['degraded_similarity']['lo']:+.4f} | "
                f"{mc[c['a']]:.2f} / {mc[c['b']]:.2f} | {_yn(c['beats'])} |")

    head = ["| Comparison | Clean dUTMOS | Clean dWER hi / dSIM lo | Degraded dUTMOS (per speaker) | Degraded dWER hi / "
            "dSIM lo | Mean passes | Beats |", "| --- | --- | --- | --- | --- | --- | --- |"]
    lines += ["", "## Each candidate against the previous methods (M)", ""] + head
    for x, comps in res["method_vs_previous"].items():
        for c, e in comps.items():
            lines.append(comp_row(f"{x} vs {c}", e))
    lines += ["", "## Condition safety on reverberation and babble (C: X - W4, per speaker)", "",
              "| Candidate | Pooled | reverb03 | reverb09 | babble15 | Holds |", "| --- | --- | --- | --- | --- | --- |"]
    for x, s in res["condition_safety"].items():
        bc = s["by_condition"]
        lines.append(f"| `{x}` | {_ci(s['C_ci'])} | {_ci(bc.get('reverb03'))} | {_ci(bc.get('reverb09'))} | "
                     f"{_ci(bc.get('babble15'))} | {_yn(s['C'])} |")
    lines += ["", "## Between the candidates (exploratory)", "", "```", json.dumps(res["candidate_contrasts"], indent=2),
              "```", "", "## SNR-routed methods and the student alone (exploratory, same criteria)", ""] + head
    for name, c in res["exploratory_comparisons"].items():
        lines.append(comp_row(name, c))
    s1 = res["gated_safety_net_S1"]
    lines.append(f"\nFirst design's S1 (SNR_TOH - SNR_W4 on gated-in reverb and babble): {_ci(s1['S1_ci'])} over "
                 f"{s1['n_cases']} cases from {s1['n_speakers']} speakers ({_yn(s1['S1'])}).")
    for cond, tab in res["method_tables"].items():
        lines += ["", f"## Methods: `{cond}` (n={tab['n']})", "",
                  "| Method | Mean passes | UTMOS | WER mean / median | Runaway (WER > 1) | SIM | Arms used |",
                  "| --- | ---: | ---: | ---: | ---: | ---: | --- |"]
        for method, e in tab["methods"].items():
            arms = ", ".join(f"{a} {k}" for a, k in e["arms"].items())
            lines.append(f"| `{method}` | {e['mean_cost']:.2f} | {e['utmos']:.3f} | {e['wer_mean']:.4f} / "
                         f"{e['wer_median']:.4f} | {e['runaway']} | {e['similarity']:.4f} | {arms} |")
    div = res["clean_diversity"]
    if div:
        lines += ["", f"## Clean pitch variety (n={div['n']} speakers x {DIV_SEEDS} seeds)", "",
                  "| Arm | CVf0 | Aligned CVf0 | Timing spread (frames) | voiced share | slow-mel spread |",
                  "| --- | ---: | ---: | ---: | ---: | ---: |"]
        for arm in DIV_ARMS:
            a = div["arms"][arm]
            lines.append(f"| `{arm}` | {a['cvf0']['mean']:.4f} | {a['cvf0_aligned']['mean']:.4f} | "
                         f"{a['timing_spread']['mean']:.2f} | {a['voiced']['mean']:.3f} | {a['mel_spread']['mean']:.4f} |")
        lines += ["", "| Contrast | CVf0 | Aligned CVf0 | Timing spread |", "| --- | --- | --- | --- |"]
        for k in div["contrasts"]["cvf0"]:
            lines.append(f"| {k} | {_ci(div['contrasts']['cvf0'][k]['boot'], 4)} | "
                         f"{_ci(div['contrasts']['cvf0_aligned'][k]['boot'], 4)} | "
                         f"{_ci(div['contrasts']['timing_spread'][k]['boot'], 2)} |")
        lines.append(f"\nMean map displacement (frames): TOH3 {div['toh_rms']['mean']:.2f}, BTH3 {div['bth_rms']['mean']:.2f}, "
                     f"RND {div['rnd_rms']['mean']:.2f}.")
    lines += ["", "## Policies' step choices", "", "```", json.dumps(res["policy_steps"], indent=2), "```",
              "", "## EXP-030 replication (exploratory)", "", "```", json.dumps(res["exp030_replication"], indent=2), "```",
              "", "## Definition 3.1 failures against W8 (current thesis yardstick)", "", "```",
              json.dumps(res["definition_3_1_failures_vs_W8"], indent=2), "```"]
    if res["exp011_reproduction"]:
        lines += ["", "## Reproduction of EXP-011 (speakers 1-20)", "", "```",
                  json.dumps(res["exp011_reproduction"], indent=2), "```"]
    return "\n".join(lines)


def make_pipe(args):
    if not args.demand_dir:
        raise SystemExit("--demand_dir is required: babble15 uses DEMAND PCAFETER/ch01.wav")
    pipe = rf.Pipeline(args.dmo_root, args.device)
    pipe.load_babble(args.demand_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "demand_sha256.txt", "w", encoding="utf-8", newline="\n") as f:
        f.write(f"{rf.DEMAND_FILE} {pipe.babble_sha}\n")
    return pipe


def main():
    p = argparse.ArgumentParser(description="EXP-031 single-formula handovers against the previous handover methods")
    sub = p.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("freeze")
    f.add_argument("--force", action="store_true")
    for name in ("smoke", "run"):
        s = sub.add_parser(name)
        s.add_argument("--dmo_root", required=True)
        s.add_argument("--demand_dir", required=True)
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
        smoke(make_pipe(args), args.out_dir)
    elif args.cmd == "run":
        check_prereg()
        pipe = make_pipe(args)
        if args.stage in ("all", "quality"):
            run_quality(pipe, qual_items(), out_dir=args.out_dir)
        if args.stage in ("all", "diversity"):
            run_diversity(pipe, div_items(), out_dir=args.out_dir)
    elif args.cmd == "analyze":
        analyze(args.out_dir, allow_partial=args.allow_partial)


if __name__ == "__main__":
    main()
