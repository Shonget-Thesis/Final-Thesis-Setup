"""EXP-036 (EVAL-031, Path B): Novel Inference-Time Mechanism Candidates (DSGH & CWPS) vs. SNR_R.

Evaluates two genuinely new inference-time mechanisms (no retraining, no binary 20 dB / 0.340 switch)
on 30 completely untouched calibration speakers (`unseen_cal[60:90]`, strictly disjoint from `dev[0:100]`,
EXP-028 `unseen_cal[0:30]`, and Path A EXP-035 `unseen_cal[30:60]`):

  - Candidate 1a (`DSGH_MICRO`, 13 passes): Continuous 1D Wasserstein-2 soft-shrinkage in spectral values
    paired with excess-roughness micro-rank damping toward student draft s0 (macro_weight = 0.0).
  - Candidate 1b (`DSGH_DUAL`, 13 passes): Same W2 soft-shrinkage with multiscale macro+micro rank
    coherence blending (macro_weight = 0.5) to test recovery on jittered noisy15 teacher ranks.
  - Candidate 2 (`CWPS`, 13 passes): Closed-Loop Wasserstein-Proximal Teacher Sampling executing
    s0 first (1 pass), injecting proximal W2 contraction inside Euler CFG steps t_1..t_3 (10 passes)
    to prevent static CFG amplification, and finishing via the dual-space readout + j=2 student (2 passes).

Baselines and references evaluated on identical seeds: STU (4), STUR2 (3), CPH8 (12), CPQH (13),
SNR_R (13, EXP-034 frozen policy), and P25 (30, diversity & timing).

Subcommands: freeze, smoke, run, analyze.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import handover_mechanisms as hm  # noqa: E402
import run_cheaper_teacher_path as ctp  # noqa: E402
import run_followup_tests as rf  # noqa: E402
import run_grid32_prelim as gl  # noqa: E402
import run_invariant_residual as ir  # noqa: E402
import run_method_comparison as mc  # noqa: E402
import run_noise_guidance_prelim as ng  # noqa: E402
import run_damped_transmissibility_stac as dts  # noqa: E402
import run_timing_only_handover as toh  # noqa: E402
import run_transmissibility_bridge as tb  # noqa: E402
import run_transmissibility_confirm as tc  # noqa: E402
import run_variety_per_pass as vp  # noqa: E402

THESIS_ROOT = ir.THESIS_ROOT
CODE_DIR = Path(__file__).resolve().parent
EXP_DIR = THESIS_ROOT / "research/experiments/EXP-036-dual-space-mechanisms"
PREREG_PATH = EXP_DIR / "preregistration.json"

DEPENDENCIES = ("run_dual_space_mechanisms.py", "handover_mechanisms.py") + tuple(ir.DEPENDENCIES)

QUAL_SLICE = (60, 90)
DIV_SLICE = (70, 90)
TIME_SLICE = (60, 70)
CONDS = ir.CONDS
DIV_CONDS = ir.DIV_CONDS
DIV_SEEDS = ir.DIV_SEEDS
DIV_SEED_BASE = ir.DIV_SEED_BASE
SEED, BOOT_B = ir.SEED, ir.BOOT_B

NOVEL_ARMS = ("DSGH_MICRO", "DSGH_DUAL", "CWPS")
QUAL_ARMS = ("STU", "STUR2", "CPH8", "CPQH") + NOVEL_ARMS
DIV_ARMS = ("STU", "STUR2", "CPH8", "CPQH", "P25") + NOVEL_ARMS
POLICIES = ("SNR_ONLY", "SNR_R")
TIME_ARMS = ("STU", "STUR2", "CPH8", "CPQH", "SNR_R") + NOVEL_ARMS + ("P25",)

TEACHER_PASSES = {
    "STU": 0, "STUR2": 0, "CPH8": 10, "CPQH": 10, "SNR_R": 10,
    "DSGH_MICRO": 10, "DSGH_DUAL": 10, "CWPS": 10, "P25": 28,
}
STUDENT_PASSES = {
    "STU": 4, "STUR2": 3, "CPH8": 2, "CPQH": 3, "SNR_R": 3,
    "DSGH_MICRO": 3, "DSGH_DUAL": 3, "CWPS": 3, "P25": 2,
}
COST = {a: TEACHER_PASSES[a] + STUDENT_PASSES[a] for a in TEACHER_PASSES}

QUAL_FIELDS = (
    ["utt_id", "speaker_id", "condition", "prompt_snr", "snr_sw", "r", "drift8", "rough_share", "gpu", "frames", "seq_len"]
    + [f"choice_{p}" for p in POLICIES]
    + [f"{m}_{a}" for a in QUAL_ARMS for m in ir.METRICS + ("hypothesis",)]
    + ["case_seconds", "utc"]
)
DIV_FIELDS = (
    ["utt_id", "speaker_id", "condition", "n_seeds", "prompt_snr", "r_mean"]
    + [f"flagged_{p}" for p in POLICIES]
    + [f"{m}_{a}" for a in DIV_ARMS + POLICIES for m in ir.DIV_MEASURES]
    + ["case_seconds", "utc"]
)
TIME_FIELDS = (
    ["utt_id", "speaker_id", "condition", "prompt_snr", "gpu", "frames", "seq_len"]
    + [f"{m}_{a}" for a in TIME_ARMS for m in ("seconds", "teacher_passes", "student_passes")]
    + ["utc"]
)


def qual_items():
    return hm.unseen_cal_slice(*QUAL_SLICE)


def div_items():
    return hm.unseen_cal_slice(*DIV_SLICE)


def time_items():
    return hm.unseen_cal_slice(*TIME_SLICE)


def expected_passes(arm):
    return (TEACHER_PASSES[arm], STUDENT_PASSES[arm])


# --------------------------------------------------------------------------- Synthesis
def synthesize(pipe, ctx, arms):
    """Synthesize Path B novel mechanisms and baseline/reference arms on identical seeds."""
    taus = list(pipe.sampler.STUDENT_TAUS)
    plen = ctx["prompt_len"]
    grid = gl.sway_grid(8)
    ends8, drift8, _ = gl.grid_teacher(pipe, ctx, grid, {ir.K8})
    t4 = ends8[ir.K8]
    s0 = gl.student(pipe, ctx, ctx["step_cond"], [0.0])
    outs = {
        "CPH8": gl.student(pipe, ctx, t4, taus[1:]),
        "CPQH": gl.student(pipe, ctx, mc.quantile_payload(s0, t4, plen), taus[1:]),
    }
    if "STU" in arms:
        outs["STU"] = gl.student(pipe, ctx, s0, taus)
    if "STUR2" in arms:
        outs["STUR2"] = gl.student(pipe, ctx, s0, taus[1:])
    if "DSGH_MICRO" in arms:
        outs["DSGH_MICRO"] = gl.student(pipe, ctx, hm.dsgh_payload(s0, t4, plen, macro_weight=0.0), taus[1:])
    if "DSGH_DUAL" in arms:
        outs["DSGH_DUAL"] = gl.student(pipe, ctx, hm.dsgh_payload(s0, t4, plen, macro_weight=0.5), taus[1:])
    if "CWPS" in arms:
        cw_payload = hm.cwps_teacher(pipe, ctx, s0, grid, k_end=ir.K8, macro_weight=0.5)
        outs["CWPS"] = gl.student(pipe, ctx, cw_payload, taus[1:])
    if "P25" in arms:
        _, _, s32 = gl.grid_teacher(pipe, ctx, gl.GRID32, {vp.K_PAPER})
        outs["P25"] = gl.student_from_state(pipe, ctx, s32[vp.K_PAPER], list(gl.ALG2_TIMES))
    g_t, g_s = toh._as_numpy(t4[0, plen:, :]), toh._as_numpy(s0[0, plen:, :])
    parts = {
        "t4": t4,
        "s0": s0,
        "drift8": float(drift8[ir.K8]),
        "r": ir.invariant_residual(g_t, g_s),
        "rough_share": ir.rough_share(g_t, g_s),
    }
    return {a: outs[a] for a in arms}, parts


def standalone(pipe, ctx, arm):
    """Run one Path B arm independently with forward-pass instrumentation."""
    taus = list(pipe.sampler.STUDENT_TAUS)
    plen = ctx["prompt_len"]
    grid = gl.sway_grid(8)
    teacher = lambda: ctp.euler(pipe, ctx, grid, ctx["y0"].clone(), 0, ir.K8)      # noqa: E731
    draft = lambda: gl.student(pipe, ctx, ctx["step_cond"], [0.0])                # noqa: E731
    if arm == "STU":
        return gl.student(pipe, ctx, ctx["step_cond"], [0.0] + taus)
    if arm == "STUR2":
        return gl.student(pipe, ctx, draft(), taus[1:])
    if arm == "CPH8":
        return gl.student(pipe, ctx, teacher(), taus[1:])
    if arm == "CPQH":
        return gl.student(pipe, ctx, mc.quantile_payload(draft(), teacher(), plen), taus[1:])
    if arm == "SNR_R":
        t4, s0 = teacher(), draft()
        r = ir.invariant_residual(toh._as_numpy(t4[0, plen:, :]), toh._as_numpy(s0[0, plen:, :]))
        pick = ir.choose("SNR_R", float(ctx["prompt_snr"]), r, 0.0)
        return gl.student(pipe, ctx, t4 if pick == "CPH8" else mc.quantile_payload(s0, t4, plen), taus[1:])
    if arm == "DSGH_MICRO":
        return gl.student(pipe, ctx, hm.dsgh_payload(draft(), teacher(), plen, macro_weight=0.0), taus[1:])
    if arm == "DSGH_DUAL":
        return gl.student(pipe, ctx, hm.dsgh_payload(draft(), teacher(), plen, macro_weight=0.5), taus[1:])
    if arm == "CWPS":
        s0 = draft()
        return gl.student(pipe, ctx, hm.cwps_teacher(pipe, ctx, s0, grid, k_end=ir.K8, macro_weight=0.5), taus[1:])
    if arm == "P25":
        return gl.student_from_state(pipe, ctx, ctp.teacher_state(pipe, ctx, gl.GRID32, vp.K_PAPER), list(gl.ALG2_TIMES))
    raise ValueError(f"Unknown arm {arm!r}")


# --------------------------------------------------------------------------- Stages
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
        row = {
            "utt_id": item["utt_id"],
            "speaker_id": item["speaker_id"],
            "condition": cond,
            "prompt_snr": snr,
            "snr_sw": ir.spread_snr(ctx["prompt"], ctx["sr"]),
            "r": parts["r"],
            "drift8": parts["drift8"],
            "rough_share": parts["rough_share"],
            "gpu": pipe.gpu,
            "frames": int(parts["s0"].shape[1] - ctx["prompt_len"]),
            "seq_len": int(parts["s0"].shape[1]),
            **{f"choice_{p}": ir.choose(p, snr, parts["r"], parts["drift8"]) for p in POLICIES},
        }
        for arm in QUAL_ARMS:
            m = pipe.score(ng.decode(pipe, ctx, outs[arm]), item, gt)
            row.update({f"{k}_{arm}": m[k] for k in ir.METRICS + ("hypothesis",)})
        row.update({"case_seconds": round(time.perf_counter() - t0, 2), "utc": tb._utc()})
        tb._append(path, QUAL_FIELDS, row)
        print(f"[{n_run}/{len(todo)}] quality: {item['utt_id']} {cond} | CPQH {row['utmos_CPQH']:.2f} "
              f"DSGH_D {row['utmos_DSGH_DUAL']:.2f} CWPS {row['utmos_CWPS']:.2f} ({row['case_seconds']:.1f}s)", flush=True)
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
        rs, flags, snr = [], {p: [] for p in POLICIES}, None
        for s in range(seeds):
            seed = int(item["input_seed"]) + DIV_SEED_BASE + s
            ctx = pipe.prepare(item, cond, seed, out_dir, noise_seed=item["input_seed"])
            outs, parts = synthesize(pipe, ctx, DIV_ARMS)
            arm_tracks = dts._track_arms_fast(pipe, ctx, outs, DIV_ARMS)
            plen, snr = ctx["prompt_len"], float(ctx["prompt_snr"])
            arm_mels = {a: toh._as_numpy(outs[a][0, plen:, :]) for a in DIV_ARMS}
            pick = {p: ir.choose(p, snr, parts["r"], parts["drift8"]) for p in POLICIES}
            for a in DIV_ARMS:
                tracks[a].append(arm_tracks[a])
                mels[a].append(arm_mels[a])
            for p in POLICIES:
                tracks[p].append(arm_tracks[pick[p]])
                mels[p].append(arm_mels[pick[p]])
                flags[p].append(pick[p] == "CPQH")
            for a in names:
                melody[a].append(ctp.melody_agreement(tracks[a][-1], arm_tracks["P25"]))
            rs.append(parts["r"])
        row = {
            "utt_id": item["utt_id"],
            "speaker_id": item["speaker_id"],
            "condition": cond,
            "n_seeds": seeds,
            "prompt_snr": snr,
            "r_mean": ctp._mean(rs),
            **{f"flagged_{p}": sum(flags[p]) / len(flags[p]) for p in POLICIES},
        }
        for a in names:
            row[f"cvf0_{a}"], row[f"voiced_{a}"] = mc.sh.cvf0(tracks[a])
            row[f"cvf0_aligned_{a}"], row[f"voiced_aligned_{a}"], row[f"timing_spread_{a}"] = mc.aligned_cvf0(
                tracks[a], mels[a]
            )
            row[f"melody_{a}"] = ctp._mean(melody[a])
        row.update({"case_seconds": round(time.perf_counter() - t0, 2), "utc": tb._utc()})
        tb._append(path, DIV_FIELDS, row)
        print(f"[{n_run}/{len(todo)}] diversity: {item['utt_id']} {cond} ({row['case_seconds']:.1f}s)", flush=True)
    print(f"[OK] diversity complete: {path}", flush=True)
    return path


def smoke(pipe, out_dir, prereg_path=PREREG_PATH):
    """Verify Path B mechanism identities, exact 13-pass hooks, and micro stage execution."""
    check_prereg(prereg_path)
    sdir = Path(out_dir) / "smoke"
    sdir.mkdir(parents=True, exist_ok=True)
    item = qual_items()[0]
    ctx = pipe.prepare(item, "clean", int(item["input_seed"]), sdir)
    plen = ctx["prompt_len"]
    outs, parts = synthesize(pipe, ctx, QUAL_ARMS + ("P25",))
    t4 = parts["t4"]
    dsgh_self = hm.dsgh_payload(t4, t4, plen, macro_weight=0.5)
    rec = {
        "utt_id": item["utt_id"],
        "finite": all(bool(np.all(np.isfinite(toh._as_numpy(outs[a])))) for a in outs),
        "dsgh_self_identity": tb._max_abs(dsgh_self[:, plen:, :] - t4[:, plen:, :]),
    }
    alone, counts = {}, {}
    with ctp.PassCounter(pipe) as counter:
        for arm in TIME_ARMS:
            counter.reset()
            out = standalone(pipe, ctx, arm)
            counts[arm] = dict(counter.n)
            ref = outs[arm] if arm in outs else outs[ir.choose("SNR_R", float(ctx["prompt_snr"]), parts["r"], parts["drift8"])]
            alone[arm] = tb._max_abs(out - ref)
    rec["standalone_diff"], rec["passes"] = alone, counts
    if not rec["finite"]:
        raise RuntimeError(f"Non-finite Path B smoke output: {rec}")
    if rec["dsgh_self_identity"] > ir.SPLIT_TOL:
        raise RuntimeError(f"DSGH self identity failed: {rec}")
    if max(alone.values()) > ir.GRID_TOL:
        raise RuntimeError(f"Standalone mismatch: {alone}")
    bad = {a: c for a, c in counts.items() if (c["teacher"], c["student"]) != expected_passes(a)}
    if bad:
        raise RuntimeError(f"Pass count mismatch: {bad}")
    run_quality(pipe, qual_items()[:1], conds=("clean", "noisy15"), out_dir=sdir, csv_name="quality.csv")
    run_diversity(pipe, div_items()[:1], sdir, seeds=2, csv_name="diversity.csv")
    hm.run_timing_arms(pipe, time_items()[:1], sdir, TIME_ARMS, standalone, TIME_FIELDS, csv_name="timing.csv")
    summary = {"checks": rec, "utc": tb._utc()}
    (sdir / "smoke_check.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print("[OK] Path B smoke passed", flush=True)
    return summary


# --------------------------------------------------------------------------- Preregistration & Analysis
ESTIMANDS = {
    "novel_vs_baselines": "Per-speaker paired comparison of DSGH_MICRO, DSGH_DUAL, and CWPS (all 13 passes, zero router "
                          "switches) against SNR_ONLY, SNR_R, CPQH, and STUR2 overall and on noisy15/clean.",
    "noisy15_recovery": "Direct UTMOS/WER test on noisy15 (targeting CPQH's 2.199 UTMOS / 0.1457 WER vs. STUR2's 2.876 / 0.0352).",
    "clean_prosody": "Retention of CPH8's clean raw and aligned CVf0 gain over STU without a heuristic gate.",
}
LABEL_RULES = {
    "M1_clean_prosody": "At least one novel candidate keeps >= 90% of CPH8's clean gain over STU (CVf0 & aligned CVf0) and clean dUTMOS >= -0.05 vs SNR_R.",
    "M2_noisy15_recovery": "That same candidate improves noisy15 UTMOS over SNR_R (lower 95% CI > 0) or reduces noisy15 WER without worsening SIM.",
    "M3_overall_competitiveness": "That candidate beats SNR_ONLY overall (lower CI > 0) and is non-inferior or better than SNR_R overall (dUTMOS lo >= -0.05).",
    "verdict": "'novel_mechanism_advances' if M1-M3 hold for any candidate; 'partial' if incomplete; otherwise 'snr_r_remains_strongest'.",
}


def build_prereg():
    q = [{"utt_id": it["utt_id"], "speaker_id": it["speaker_id"]} for it in qual_items()]
    d = [{"utt_id": it["utt_id"], "speaker_id": it["speaker_id"]} for it in div_items()]
    t = [{"utt_id": it["utt_id"], "speaker_id": it["speaker_id"]} for it in time_items()]
    inputs = (ir.PREREG_PATH, ir.SCREEN_JSON)
    return {
        "experiment": "EXP-036",
        "task_id": "EVAL-031",
        "title": "Path B: Continuous Dual-Space Handover (DSGH) & Closed-Loop Wasserstein-Proximal Sampling (CWPS)",
        "commit_time": ir.T_COMMIT,
        "novel_arms": list(NOVEL_ARMS),
        "qual_arms": list(QUAL_ARMS),
        "div_arms": list(DIV_ARMS),
        "policies": list(POLICIES),
        "time_arms": list(TIME_ARMS),
        "dsgh_anchors": {"r_clean": hm.CLEAN_R_MED, "r_hi": hm.LAMBDA_R, "macro_window": hm.MACRO_WINDOW},
        "costs": dict(COST),
        "teacher_passes": dict(TEACHER_PASSES),
        "student_passes": dict(STUDENT_PASSES),
        "conditions": list(CONDS),
        "div_conditions": list(DIV_CONDS),
        "cohort_slices": {"calibration_unseen_qual": list(QUAL_SLICE), "calibration_unseen_div": list(DIV_SLICE),
                          "calibration_unseen_time": list(TIME_SLICE)},
        "n_qual_cases": len(q) * len(CONDS),
        "qual_utts": q,
        "n_div_speakers": len(d),
        "div_seeds": DIV_SEEDS,
        "div_utts": d,
        "n_time_cases": len(t),
        "time_utts": t,
        "estimands": ESTIMANDS,
        "label_rules": LABEL_RULES,
        "code_hashes": {name: tb._sha256_file(CODE_DIR / name) for name in DEPENDENCIES if (CODE_DIR / name).exists()},
        "input_hashes": {str(Path(p).relative_to(THESIS_ROOT).as_posix()): tb._sha256_file(p) for p in inputs if Path(p).exists()},
        "seed": SEED,
        "boot_b": BOOT_B,
    }


def freeze(prereg_path=PREREG_PATH, force=False):
    doc = hm.write_prereg_doc(prereg_path, build_prereg(), force=force)
    print(f"[OK] Froze {prereg_path} (sha256={doc['sha256'][:12]}...)")
    return doc


def check_prereg(prereg_path=PREREG_PATH):
    return hm.verify_prereg_doc(
        prereg_path, CODE_DIR, THESIS_ROOT,
        [("qual_utts", qual_items()), ("div_utts", div_items()), ("time_utts", time_items())],
    )


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

    def val_fn(r, p, m):
        if p in POLICIES:
            return ir.outcome(r, p, m)[0]
        return ir._f(r, f"{m}_{p}")

    noisy15_q = [r for r in qrows if r["condition"] == "noisy15"]
    clean_q = [r for r in qrows if r["condition"] == "clean"]
    contrasts = {}
    for cand in NOVEL_ARMS:
        contrasts[f"{cand}_minus_SNR_R_all"] = hm.paired_speaker_contrast(qrows, val_fn, cand, "SNR_R")
        contrasts[f"{cand}_minus_SNR_ONLY_all"] = hm.paired_speaker_contrast(qrows, val_fn, cand, "SNR_ONLY")
        contrasts[f"{cand}_minus_CPQH_all"] = hm.paired_speaker_contrast(qrows, val_fn, cand, "CPQH")
        if noisy15_q:
            contrasts[f"{cand}_minus_SNR_R_noisy15"] = hm.paired_speaker_contrast(noisy15_q, val_fn, cand, "SNR_R")
        if clean_q:
            contrasts[f"{cand}_minus_SNR_R_clean"] = hm.paired_speaker_contrast(clean_q, val_fn, cand, "SNR_R")

    var = {}
    for cond in DIV_CONDS:
        rs = [r for r in drows if r["condition"] == cond]
        if len(rs) < 2:
            continue
        names = DIV_ARMS + POLICIES
        e = {"n": len(rs), "means": {a: {m: ctp._mean(ir._f(r, f"{m}_{a}") for r in rs) for m in ir.DIV_MEASURES} for a in names}}
        if cond == "clean":
            for m in ir.VARIETY:
                e[f"share_of_CPH8_gain_{m}"] = {p: vp.share_ci(rs, p, m, ref="CPH8", base="STU") for p in POLICIES + ("CPQH", "STUR2") + NOVEL_ARMS}
        else:
            e["voiced_at_least_min"] = {a: sum(ir._f(r, f"voiced_{a}") >= ir.MIN_VOICED for r in rs) for a in names}
        var[cond] = e

    cand_checks = {}
    clean_v = var.get("clean", {})
    for cand in NOVEL_ARMS:
        k_raw = clean_v.get("share_of_CPH8_gain_cvf0", {}).get(cand, {}).get("point")
        k_alg = clean_v.get("share_of_CPH8_gain_cvf0_aligned", {}).get(cand, {}).get("point")
        c_cl = contrasts.get(f"{cand}_minus_SNR_R_clean")
        c_n15 = contrasts.get(f"{cand}_minus_SNR_R_noisy15")
        c_all_r = contrasts[f"{cand}_minus_SNR_R_all"]
        c_all_s = contrasts[f"{cand}_minus_SNR_ONLY_all"]
        m1 = bool(
            k_raw is not None and k_alg is not None and k_raw >= ir.VARIETY_KEEP and k_alg >= ir.VARIETY_KEEP
            and c_cl is not None and c_cl["utmos"]["mean"] >= -0.05
        )
        m2 = bool(
            c_n15 is not None and (
                c_n15["utmos"]["lo"] > 0 or (c_n15["wer"]["hi"] < 0 and c_n15["similarity"]["lo"] >= ir.SIM_BAR)
            )
        )
        m3 = bool(c_all_s["better"] and c_all_r["non_inferior_utmos"])
        cand_checks[cand] = {"M1_clean_prosody": m1, "M2_noisy15_recovery": m2, "M3_overall_competitiveness": m3, "all_pass": m1 and m2 and m3}

    advances = any(v["all_pass"] for v in cand_checks.values())
    verdict = "partial" if not complete else ("novel_mechanism_advances" if advances else "snr_r_remains_strongest")
    cond_means = {}
    for cond in CONDS + ("all",):
        rs = qrows if cond == "all" else [r for r in qrows if r["condition"] == cond]
        if rs:
            cond_means[cond] = {
                a: {
                    "utmos": ctp._mean(val_fn(r, a, "utmos") for r in rs),
                    "wer": ctp._mean(val_fn(r, a, "wer") for r in rs),
                    "similarity": ctp._mean(val_fn(r, a, "similarity") for r in rs),
                }
                for a in QUAL_ARMS + POLICIES
            }

    res = {
        "experiment": "EXP-036",
        "partial": not complete,
        "verdict": verdict,
        "candidate_checks": cand_checks,
        "n_quality_rows": len(qrows),
        "n_diversity_rows": len(drows),
        "n_timing_rows": len(trows),
        "contrasts": contrasts,
        "condition_means": cond_means,
        "variety": var,
        "utc": tb._utc(),
    }
    rjson, rmd = out_dir / "dual_space_results.json", out_dir / "DUAL_SPACE_REPORT.md"
    rjson.write_text(json.dumps(res, indent=2, default=str) + "\n", encoding="utf-8")
    rmd.write_text(report(res) + "\n", encoding="utf-8")
    hm.write_results_manifest(out_dir, "EXP-036", prereg_path, (q_path, d_path, t_path, rjson, rmd, out_dir / "demand_sha256.txt"))
    print(f"[OK] Path B analysis complete: verdict={res['verdict']}, report={rmd}", flush=True)
    return res


def report(res):
    lines = [
        "# EXP-036 (Path B): Continuous Dual-Space & Closed-Loop Novel Mechanisms",
        "",
        f"Rows: quality {res['n_quality_rows']} / {len(qual_items()) * len(CONDS)}, diversity {res['n_diversity_rows']} / {len(div_items()) * len(DIV_CONDS)} (partial={res['partial']}).",
        "",
        f"**Verdict: `{res['verdict']}`**",
        "",
        "| Candidate | M1 clean prosody | M2 noisy15 recovery | M3 overall | All Pass |",
        "| --- | --- | --- | --- | --- |",
    ]
    for cand, v in res["candidate_checks"].items():
        lines.append(f"| {cand} | {v['M1_clean_prosody']} | {v['M2_noisy15_recovery']} | {v['M3_overall_competitiveness']} | {v['all_pass']} |")
    lines += ["", "## Paired Contrasts", "", "| Contrast | dUTMOS | dWER | dSIM | Better? |", "| --- | --- | --- | --- | --- |"]
    for name, c in res["contrasts"].items():
        lines.append(f"| {name} | {tc._fmt_ci(c['utmos'], 3)} | {tc._fmt_ci(c['wer'], 4)} | {tc._fmt_ci(c['similarity'], 4)} | {'yes' if c['better'] else 'no'} |")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description="EXP-036 (Path B): Novel Mechanism Candidates (DSGH & CWPS)")
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
            s.add_argument("--stage", choices=("all", "quality", "diversity", "timing"), default="all")
    a = sub.add_parser("analyze")
    a.add_argument("--out_dir", default=str(EXP_DIR / "results"))
    a.add_argument("--allow_partial", action="store_true")
    args = p.parse_args()
    if args.cmd == "freeze":
        freeze(force=args.force)
    elif args.cmd == "smoke":
        smoke(ir.make_pipe(args), args.out_dir)
    elif args.cmd == "run":
        check_prereg()
        pipe = ir.make_pipe(args)
        if args.stage in ("all", "quality"):
            run_quality(pipe, qual_items(), out_dir=args.out_dir)
        if args.stage in ("all", "diversity"):
            run_diversity(pipe, div_items(), out_dir=args.out_dir)
        if args.stage in ("all", "timing"):
            hm.run_timing_arms(pipe, time_items(), args.out_dir, TIME_ARMS, standalone, TIME_FIELDS)
    elif args.cmd == "analyze":
        analyze(args.out_dir, allow_partial=args.allow_partial)


if __name__ == "__main__":
    main()
