"""EXP-035 (EVAL-030, Path A): Condition-Invariance Ladder Ablation (G1..G3, FCH) & Unseen Confirmation of SNR_R.

Evaluates on 30 unseen calibration speakers (`unseen_cal[30:60]`, strictly disjoint from all 100 development
speakers and EXP-028's `unseen_cal[0:30]`):
  1. Direct commit-point (t = 0.2929, j = 2) invariance ladder ablation with draft prompt frames locked:
       Base (CPH8 / CPH8_DP) -> Floor-Canonical (FCH_DP) -> G1 Shift (G1_VSWAP) -> G2 Affine (G2_AFFINE) -> G3 Quantile (CPQH).
  2. Out-of-sample replication and 5-fold speaker-grouped calibration check of SNR_R vs. SNR_ONLY and SNR_DRIFT.

Subcommands: freeze, smoke, run, analyze.
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import handover_mechanisms as hm  # noqa: E402
import run_cheaper_teacher_path as ctp  # noqa: E402
import run_consistency_gate_prelim as cg  # noqa: E402
import run_damped_transmissibility_stac as dts  # noqa: E402
import run_followup_tests as rf  # noqa: E402
import run_grid32_prelim as gl  # noqa: E402
import run_invariant_residual as ir  # noqa: E402
import run_method_comparison as mc  # noqa: E402
import run_noise_guidance_prelim as ng  # noqa: E402
import run_timing_only_handover as toh  # noqa: E402
import run_transmissibility_bridge as tb  # noqa: E402
import run_transmissibility_confirm as tc  # noqa: E402
import run_transmissibility_test as tt  # noqa: E402
import run_variety_per_pass as vp  # noqa: E402

THESIS_ROOT = ir.THESIS_ROOT
CODE_DIR = Path(__file__).resolve().parent
EXP_DIR = THESIS_ROOT / "research/experiments/EXP-035-invariance-ladder-confirm"
PREREG_PATH = EXP_DIR / "preregistration.json"

DEPENDENCIES = ("run_invariance_ladder_confirm.py", "handover_mechanisms.py") + tuple(ir.DEPENDENCIES)

QUAL_SLICE = (30, 60)
DIV_SLICE = (40, 60)
TIME_SLICE = (30, 40)
CONDS = ir.CONDS
SPECTRAL_CONDS = ("gauss25", "pink25", "hp25")
DIV_CONDS = ir.DIV_CONDS
DIV_SEEDS = ir.DIV_SEEDS
DIV_SEED_BASE = ir.DIV_SEED_BASE
SEED, BOOT_B, FOLDS = ir.SEED, ir.BOOT_B, 5
THETA, LAMBDA_R, LAMBDA_D = ir.THETA, ir.LAMBDA_R, ir.LAMBDA_D

A_ARM, B_ARM = "CPH8", "CPQH"
QUAL_ARMS = ("STU", "STUR2", "CPH8", "CPH8_DP", "FCH_DP", "G1_VSWAP", "G2_AFFINE", "CPQH", "W8")
LADDER_ARMS = ("CPH8", "CPH8_DP", "FCH_DP", "G1_VSWAP", "G2_AFFINE", "CPQH")
DIV_ARMS = ("STU", "STUR2", "CPH8", "G1_VSWAP", "G2_AFFINE", "CPQH", "P25")
POLICIES = ir.POLICIES
TIME_ARMS = ("STU", "STUR2", "CPH8", "G1_VSWAP", "G2_AFFINE", "CPQH", "SNR_R", "W8", "P25")

TEACHER_PASSES = {
    "STU": 0, "STUR2": 0, "CPH8": 10, "CPH8_DP": 10, "FCH_DP": 10,
    "G1_VSWAP": 10, "G2_AFFINE": 10, "CPQH": 10, "SNR_R": 10, "W8": 18, "P25": 28,
}
STUDENT_PASSES = {
    "STU": 4, "STUR2": 3, "CPH8": 2, "CPH8_DP": 3, "FCH_DP": 3,
    "G1_VSWAP": 3, "G2_AFFINE": 3, "CPQH": 3, "SNR_R": 3, "W8": 3, "P25": 2,
}
COST = {a: TEACHER_PASSES[a] + STUDENT_PASSES[a] for a in TEACHER_PASSES}

PAYLOAD_BUILDERS = {
    "CPH8_DP": hm.cph8_dp_payload,
    "FCH_DP": hm.fch_payload,
    "G1_VSWAP": hm.g1_vswap_payload,
    "G2_AFFINE": hm.g2_affine_payload,
    "CPQH": mc.quantile_payload,
}

QUAL_FIELDS = (
    ["utt_id", "speaker_id", "condition", "prompt_snr", "snr_sw", "r", "drift8", "rough_share", "gpu", "frames", "seq_len"]
    + [f"choice_{p}" for p in POLICIES]
    + [f"{m}_{a}" for a in QUAL_ARMS for m in ir.METRICS + ("hypothesis",)]
    + ["case_seconds", "utc"]
)
DIV_FIELDS = (
    ["utt_id", "speaker_id", "condition", "n_seeds", "prompt_snr", "r_mean", "drift8_mean"]
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
    """Synthesize all requested Path A ladder and reference arms from shared trajectories."""
    taus = list(pipe.sampler.STUDENT_TAUS)
    plen = ctx["prompt_len"]
    ends8, drift8, _ = gl.grid_teacher(pipe, ctx, gl.sway_grid(8), {ir.K8 - 1, ir.K8})
    t4 = ends8[ir.K8]
    s0 = gl.student(pipe, ctx, ctx["step_cond"], [0.0])
    outs = {"CPH8": gl.student(pipe, ctx, t4, taus[1:])}
    for name, fn in PAYLOAD_BUILDERS.items():
        if name in arms:
            outs[name] = gl.student(pipe, ctx, fn(s0, t4, plen), taus[1:])
    if "STU" in arms:
        outs["STU"] = gl.student(pipe, ctx, s0, taus)
    if "STUR2" in arms:
        outs["STUR2"] = gl.student(pipe, ctx, s0, taus[1:])
    if "W8" in arms:
        e16, _, _ = gl.grid_teacher(pipe, ctx, tt.grid16(pipe), {8})
        outs["W8"] = gl.student(pipe, ctx, e16[8], taus)
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
    """Run a single Path A arm independently with its own forward passes for timing and hook counting."""
    taus = list(pipe.sampler.STUDENT_TAUS)
    plen = ctx["prompt_len"]
    teacher = lambda: ctp.euler(pipe, ctx, gl.sway_grid(8), ctx["y0"].clone(), 0, ir.K8)  # noqa: E731
    draft = lambda: gl.student(pipe, ctx, ctx["step_cond"], [0.0])                        # noqa: E731
    if arm == "STU":
        return gl.student(pipe, ctx, ctx["step_cond"], [0.0] + taus)
    if arm == "STUR2":
        return gl.student(pipe, ctx, draft(), taus[1:])
    if arm == "CPH8":
        return gl.student(pipe, ctx, teacher(), taus[1:])
    if arm in PAYLOAD_BUILDERS:
        return gl.student(pipe, ctx, PAYLOAD_BUILDERS[arm](draft(), teacher(), plen), taus[1:])
    if arm == "SNR_R":
        t4, s0 = teacher(), draft()
        r = ir.invariant_residual(toh._as_numpy(t4[0, plen:, :]), toh._as_numpy(s0[0, plen:, :]))
        pick = ir.choose("SNR_R", float(ctx["prompt_snr"]), r, 0.0)
        payload = PAYLOAD_BUILDERS["CPH8_DP" if pick == A_ARM else "CPQH"](s0, t4, plen)
        return gl.student(pipe, ctx, payload, taus[1:]) if pick == B_ARM else gl.student(pipe, ctx, t4, taus[1:])
    if arm == "W8":
        return gl.student(pipe, ctx, ctp.euler(pipe, ctx, tt.grid16(pipe), ctx["y0"].clone(), 0, 8), taus)
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
        print(f"[{n_run}/{len(todo)}] quality: {item['utt_id']} {cond} SNR {snr:.1f} r {parts['r']:.3f} -> "
              f"G2 {row['utmos_G2_AFFINE']:.2f} vs CPQH {row['utmos_CPQH']:.2f} ({row['case_seconds']:.1f}s)", flush=True)
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
            pick = {p: ir.choose(p, snr, parts["r"], parts["drift8"]) for p in POLICIES}
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
        row = {
            "utt_id": item["utt_id"],
            "speaker_id": item["speaker_id"],
            "condition": cond,
            "n_seeds": seeds,
            "prompt_snr": snr,
            "r_mean": ctp._mean(rs),
            "drift8_mean": ctp._mean(ds),
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
    """Verify Path A operator identities, pass counts, and micro runs on unseen speaker 1 + dev re-check."""
    check_prereg(prereg_path)
    sdir = Path(out_dir) / "smoke"
    sdir.mkdir(parents=True, exist_ok=True)
    item = qual_items()[0]
    ctx = pipe.prepare(item, "clean", int(item["input_seed"]), sdir)
    plen = ctx["prompt_len"]
    outs, parts = synthesize(pipe, ctx, QUAL_ARMS + ("P25",))
    t4, s0 = parts["t4"], parts["s0"]
    g1_self = hm.g1_vswap_payload(t4, t4 + 2.5, plen)
    g2_self = hm.g2_affine_payload(t4, 1.7 * t4 - 3.1, plen)
    t4_floor = (0.5 * pipe.torch.log(pipe.torch.exp(2.0 * t4.double()) + 1e-4)).to(dtype=t4.dtype)
    fch_floor_inv = hm.fch_payload(s0, t4_floor, plen)[:, plen:, :] - hm.fch_payload(s0, t4, plen)[:, plen:, :]
    rec = {
        "utt_id": item["utt_id"],
        "finite": all(bool(np.all(np.isfinite(toh._as_numpy(outs[a])))) for a in outs),
        "g1_shift_invariance": tb._max_abs(g1_self[:, plen:, :] - t4[:, plen:, :]),
        "g2_affine_invariance": tb._max_abs(g2_self[:, plen:, :] - t4[:, plen:, :]),
        "fch_floor_invariance": tb._max_abs(fch_floor_inv),
    }
    alone, counts = {}, {}
    with ctp.PassCounter(pipe) as counter:
        for arm in TIME_ARMS:
            counter.reset()
            out = standalone(pipe, ctx, arm)
            counts[arm] = dict(counter.n)
            ref = outs[arm] if arm in outs else outs[ir.choose(arm, float(ctx["prompt_snr"]), parts["r"], parts["drift8"])]
            alone[arm] = tb._max_abs(out - ref)
    rec["standalone_diff"], rec["passes"] = alone, counts
    if not rec["finite"]:
        raise RuntimeError(f"Non-finite Path A smoke output: {rec}")
    if max(rec["g1_shift_invariance"], rec["g2_affine_invariance"], rec["fch_floor_invariance"]) > ir.SPLIT_TOL:
        raise RuntimeError(f"Ladder identity failed: {rec}")
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
    print("[OK] Path A smoke passed", flush=True)
    return summary


# --------------------------------------------------------------------------- Preregistration & Analysis
ESTIMANDS = {
    "ladder_ablation": "Paired differences on the same unseen calibration cohort comparing Base (CPH8 / CPH8_DP), "
                       "FCH_DP, G1_VSWAP, G2_AFFINE, and G3 (CPQH) across all 8 conditions and on spectral noise.",
    "unseen_snr_r": "Replication of frozen EXP-034 SNR_R (theta=20 dB, lambda_r=0.340048) vs. SNR_ONLY and SNR_DRIFT "
                    "on speakers completely disjoint from dev[0:100] and EXP-028 unseen_cal[0:30].",
    "oof_ltt_check": "5-fold speaker-grouped out-of-fold threshold calibration selecting lambda_r and lambda_d subject to "
                     "<= 15% clean wrong flags on training speakers and scoring held folds.",
}
LABEL_RULES = {
    "H1_clean_prompt_control": "Clean CPH8_DP - CPH8 mean UTMOS >= -0.05 with lower 95% CI >= -0.08.",
    "H2_g3_beats_fch_and_g1": "Over all 240 cases, CPQH beats FCH_DP and G1_VSWAP in UTMOS (lower CI > 0) within WER/SIM bars.",
    "H3_g3_beats_g2_spectral": "On spectral noise (gauss25, pink25, hp25), CPQH beats G2_AFFINE in UTMOS (lower CI > 0) within WER/SIM bars.",
    "H4_snr_r_unseen_confirmed": "On unseen_cal[30:60], SNR_R beats SNR_ONLY and SNR_DRIFT (lower CI > 0), AUC(r) >= 0.80, clean FP <= 0.15.",
    "H5_clean_variety": "On clean prompts, SNR_R keeps >= 90% of CPH8's gain over STU on CVf0 and aligned CVf0.",
    "verdict": "'ladder_and_snr_r_confirmed' if H1-H5 hold; 'partial' if incomplete; otherwise 'partially_confirmed'.",
}


def build_prereg():
    q = [{"utt_id": it["utt_id"], "speaker_id": it["speaker_id"]} for it in qual_items()]
    d = [{"utt_id": it["utt_id"], "speaker_id": it["speaker_id"]} for it in div_items()]
    t = [{"utt_id": it["utt_id"], "speaker_id": it["speaker_id"]} for it in time_items()]
    inputs = (ir.PREREG_PATH, ir.SCREEN_JSON)
    payload = {
        "experiment": "EXP-035",
        "task_id": "EVAL-030",
        "title": "Path A: Invariance-Ladder Ablation (FCH, G1, G2, G3) and Disjoint Calibration Confirmation of SNR_R",
        "commit_time": ir.T_COMMIT,
        "qual_arms": list(QUAL_ARMS),
        "div_arms": list(DIV_ARMS),
        "policies": list(POLICIES),
        "time_arms": list(TIME_ARMS),
        "thresholds": {"theta_db": THETA, "lambda_r": LAMBDA_R, "lambda_d": LAMBDA_D},
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
    return payload


def freeze(prereg_path=PREREG_PATH, force=False):
    doc = hm.write_prereg_doc(prereg_path, build_prereg(), force=force)
    print(f"[OK] Froze {prereg_path} (sha256={doc['sha256'][:12]}...)")
    return doc


def check_prereg(prereg_path=PREREG_PATH):
    return hm.verify_prereg_doc(
        prereg_path, CODE_DIR, THESIS_ROOT,
        [("qual_utts", qual_items()), ("div_utts", div_items()), ("time_utts", time_items())],
    )


def oof_5fold_calibration(rows):
    """5-fold speaker-grouped CV calibrating lambda_r and lambda_d under <= 15% clean wrong flags."""
    spks = sorted({r["utt_id"] for r in rows})
    folds = [set(spks[i::FOLDS]) for i in range(FOLDS)]
    test_r, test_d, test_snr, chosen = [], [], [], []
    for k in range(FOLDS):
        test_set = folds[k]
        cal = [r for r in rows if r["utt_id"] not in test_set]
        tst = [r for r in rows if r["utt_id"] in test_set]
        lr = ir.calibrate(cal, "r", +1)
        dh, dl = ir.calibrate(cal, "drift8", +1), ir.calibrate(cal, "drift8", -1)
        db, sign = (dh, +1) if dh["mean"] >= dl["mean"] else (dl, -1)
        chosen.append({"fold": k, "lambda_r": lr["lambda"], "lambda_d": db["lambda"], "drift_sign": sign})
        for r in tst:
            test_snr.append(ir.outcome(r, "SNR_ONLY", "utmos")[0])
            fr = ir._f(r, "prompt_snr") < THETA or ir._f(r, "r") > lr["lambda"]
            test_r.append(ir._f(r, f"utmos_{B_ARM if fr else A_ARM}"))
            fd = ir._f(r, "prompt_snr") < THETA or (
                ir._f(r, "drift8") > db["lambda"] if sign > 0 else ir._f(r, "drift8") < db["lambda"]
            )
            test_d.append(ir._f(r, f"utmos_{B_ARM if fd else A_ARM}"))
    return {
        "n_speakers": len(spks),
        "oof_utmos": {"SNR_ONLY": ctp._mean(test_snr), "SNR_DRIFT": ctp._mean(test_d), "SNR_R": ctp._mean(test_r)},
        "folds": chosen,
    }


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
        return ir.policy_value(r, p, m)

    clean_q = [r for r in qrows if r["condition"] == "clean"]
    spec_q = [r for r in qrows if r["condition"] in SPECTRAL_CONDS]
    ladder = {
        "CPH8_DP_minus_CPH8_clean": hm.paired_speaker_contrast(clean_q, val_fn, "CPH8_DP", "CPH8") if clean_q else None,
        "FCH_DP_minus_CPH8_DP": hm.paired_speaker_contrast(qrows, val_fn, "FCH_DP", "CPH8_DP"),
        "G1_VSWAP_minus_FCH_DP": hm.paired_speaker_contrast(qrows, val_fn, "G1_VSWAP", "FCH_DP"),
        "G2_AFFINE_minus_G1_VSWAP": hm.paired_speaker_contrast(qrows, val_fn, "G2_AFFINE", "G1_VSWAP"),
        "CPQH_minus_FCH_DP": hm.paired_speaker_contrast(qrows, val_fn, "CPQH", "FCH_DP"),
        "CPQH_minus_G1_VSWAP": hm.paired_speaker_contrast(qrows, val_fn, "CPQH", "G1_VSWAP"),
        "CPQH_minus_G2_AFFINE_all": hm.paired_speaker_contrast(qrows, val_fn, "CPQH", "G2_AFFINE"),
        "CPQH_minus_G2_AFFINE_spectral": hm.paired_speaker_contrast(spec_q, val_fn, "CPQH", "G2_AFFINE") if spec_q else None,
    }
    policy_comps = {
        "SNR_R_minus_SNR_ONLY": hm.paired_speaker_contrast(qrows, val_fn, "SNR_R", "SNR_ONLY"),
        "SNR_R_minus_SNR_DRIFT": hm.paired_speaker_contrast(qrows, val_fn, "SNR_R", "SNR_DRIFT"),
        "R_ONLY_minus_SNR_ONLY": hm.paired_speaker_contrast(qrows, val_fn, "R_ONLY", "SNR_ONLY"),
        "SNR_R_minus_W8": hm.paired_speaker_contrast(qrows, val_fn, "SNR_R", "W8"),
    }
    det = ir.detection(qrows)
    wrong = ir.wrong_clean(qrows, "SNR_R")
    var = {}
    for cond in DIV_CONDS:
        rs = [r for r in drows if r["condition"] == cond]
        if len(rs) < 2:
            continue
        names = DIV_ARMS + POLICIES
        e = {"n": len(rs), "means": {a: {m: ctp._mean(ir._f(r, f"{m}_{a}") for r in rs) for m in ir.DIV_MEASURES} for a in names}}
        if cond == "clean":
            for m in ir.VARIETY:
                e[f"share_of_CPH8_gain_{m}"] = {p: vp.share_ci(rs, p, m, ref=A_ARM, base="STU") for p in POLICIES + (B_ARM, "G1_VSWAP", "G2_AFFINE", "STUR2")}
        else:
            e["voiced_at_least_min"] = {a: sum(ir._f(r, f"voiced_{a}") >= ir.MIN_VOICED for r in rs) for a in names}
        var[cond] = e

    keep = (
        var.get("clean", {}).get("share_of_CPH8_gain_cvf0", {}).get("SNR_R", {}).get("point"),
        var.get("clean", {}).get("share_of_CPH8_gain_cvf0_aligned", {}).get("SNR_R", {}).get("point"),
    )
    h1_c = ladder["CPH8_DP_minus_CPH8_clean"]
    h = {
        "H1_clean_prompt_control": bool(h1_c and h1_c["utmos"]["mean"] >= -0.05 and h1_c["utmos"]["lo"] >= -0.08),
        "H2_g3_beats_fch_and_g1": bool(ladder["CPQH_minus_FCH_DP"]["better"] and ladder["CPQH_minus_G1_VSWAP"]["better"]),
        "H3_g3_beats_g2_spectral": bool(ladder["CPQH_minus_G2_AFFINE_spectral"] and ladder["CPQH_minus_G2_AFFINE_spectral"]["better"]),
        "H4_snr_r_unseen_confirmed": bool(
            policy_comps["SNR_R_minus_SNR_ONLY"]["better"]
            and policy_comps["SNR_R_minus_SNR_DRIFT"]["better"]
            and det["all"]["r"] is not None
            and det["all"]["r"] >= ir.AUC_ALL
            and wrong is not None
            and wrong <= ir.MAX_WRONG_CLEAN
        ),
        "H5_clean_variety": all(k is not None and k >= ir.VARIETY_KEEP for k in keep),
    }
    v = "partial" if not complete else ("ladder_and_snr_r_confirmed" if all(h.values()) else "partially_confirmed")
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
        "experiment": "EXP-035",
        "partial": not complete,
        "verdict": v,
        "hypotheses": h,
        "n_quality_rows": len(qrows),
        "n_diversity_rows": len(drows),
        "n_timing_rows": len(trows),
        "ladder_contrasts": ladder,
        "policy_contrasts": policy_comps,
        "detection": det,
        "wrong_clean_snr_r": wrong,
        "oof_5fold": oof_5fold_calibration(qrows),
        "condition_means": cond_means,
        "variety": var,
        "utc": tb._utc(),
    }
    rjson, rmd = out_dir / "invariance_ladder_results.json", out_dir / "INVARIANCE_LADDER_REPORT.md"
    rjson.write_text(json.dumps(res, indent=2, default=str) + "\n", encoding="utf-8")
    rmd.write_text(report(res) + "\n", encoding="utf-8")
    hm.write_results_manifest(out_dir, "EXP-035", prereg_path, (q_path, d_path, t_path, rjson, rmd, out_dir / "demand_sha256.txt"))
    print(f"[OK] Path A analysis complete: verdict={res['verdict']}, report={rmd}", flush=True)
    return res


def report(res):
    lines = [
        "# EXP-035 (Path A): Condition-Invariance Ladder & Unseen Confirmation of SNR_R",
        "",
        f"Rows: quality {res['n_quality_rows']} / {len(qual_items()) * len(CONDS)}, diversity {res['n_diversity_rows']} / {len(div_items()) * len(DIV_CONDS)} (partial={res['partial']}).",
        "",
        f"**Verdict: `{res['verdict']}`**",
        "",
        "| Hypothesis | Holds |",
        "| --- | --- |",
    ]
    lines += [f"| {k} | {'yes' if v else 'no'} |" for k, v in res["hypotheses"].items()]
    lines += ["", "## Invariance Ladder & Policy Contrasts", "", "| Contrast | dUTMOS | dWER | dSIM | Better? |", "| --- | --- | --- | --- | --- |"]
    for group in (res["ladder_contrasts"], res["policy_contrasts"]):
        for name, c in group.items():
            if c is None:
                continue
            lines.append(f"| {name} | {tc._fmt_ci(c['utmos'], 3)} | {tc._fmt_ci(c['wer'], 4)} | {tc._fmt_ci(c['similarity'], 4)} | {'yes' if c['better'] else 'no'} |")
    lines += ["", f"Wrong clean flags (SNR_R): {ir._fmt(res['wrong_clean_snr_r'])} | AUC(r) all={ir._fmt(res['detection']['all']['r'])}, open={ir._fmt(res['detection']['gate_open']['r'])}", ""]
    lines += ["## Out-of-Fold 5-Fold Unseen Calibration UTMOS", "", json.dumps(res["oof_5fold"]["oof_utmos"], indent=2)]
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description="EXP-035 (Path A): Invariance Ladder & Unseen Confirmation")
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
