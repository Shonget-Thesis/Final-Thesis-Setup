"""Colab T4 experiment for Transmissible Slow-Band Bridge & STAC Handover (EXP-028 / EVAL-021).

Addresses the quality-diversity trade-off uncovered by EXP-027 on 40 fresh speakers (61-100):
  - 13-NFE `W4` keeps clean perceptual quality (UTMOS 3.759, dWER +0.0000 vs W8) because the
    tau_1 = 0.0761 student step produces a denoised acoustic draft S1, but compresses pitch variety
    (CVf0 0.0770 vs W8R2 0.0893 / T16 0.0911).
  - 12-NFE `W4R2` skips tau_1 and injects T4 directly at tau_2 = 0.2929, unlocking high diversity
    (CVf0 0.0859, p = 2.94e-4 vs W4), but loses -0.373 UTMOS (3.404) and inflates slow-mel
    dispersion (1.0299) because raw T4 lacks tau_1 denoising on voice and fine structure.

Proposed 13-NFE Transmissible Slow-Band Bridge (`TBRG`):
  1. Run guided Euler teacher steps 0..4 (10 NFE) to obtain T4 and step-wise drifts d_1..d_4,
     cumulative STAC trajectory action A_4 = sum_{j=1}^4 d_j, and contraction rho_4 = d_4 / d_3.
  2. Run the student's tau_1 = 0.0761 pass (1 NFE) from T4 to obtain clean acoustic draft S1.
  3. Decompose generated frames plen: of T4 -> (v_T, s_T, f_T) and S1 -> (v_S, s_S, f_S) via
     sh.decompose(..., window=9), then bridge energy-matched slow prosody at tau_2 (0 NFE):
       s_tilde_T = (||s_S||_F / (||s_T||_F + 1e-8)) * s_T
       X_TBRG[0, plen:, :] = v_S + s_tilde_T + f_S
  4. Finish student steps [tau_2, tau_3] = [0.2929, 0.6173] (2 NFE -> 13 NFE total).

Also evaluates unnormalized slow-band bridge ablation (`STS2`, 13 NFE, s_tilde_T = s_T) and 5-fold
out-of-fold speaker cross-validated STAC-TBRG adaptive routing between `TBRG` (13 NFE) and
EXP-027's stored `W8R2` (20 NFE) using Step-4 relative action velocity r_4 = d_4 / (A_4 + 1e-8).
Reuses locked EXP-027 fresh-speaker CSVs for student4, W4, W4R2, W8, W8R2, T16.

Subcommands: freeze, smoke, run, analyze.
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
    "run_grid32_prelim.py",
    "run_noise_guidance_prelim.py",
    "run_followup_tests.py",
)

FRESH_N = 40
DIV_PAIRS = 20
DIV_SEEDS = 20
DIV_SEED_BASE = 1000
FOLDS = 5
SEED = 2026
BOOT_B = 2000
METRICS = ("wer", "similarity", "utmos")
DRIFT_FIELDS = ("d_1", "d_2", "d_3", "d_4", "A_4", "rho_4")

NEW_ARMS = ("TBRG", "STS2")
STORED_ARMS = ("student4", "W4", "W8", "W4R2", "W8R2", "T16")
ALL_ARMS = STORED_ARMS + NEW_ARMS
COST = {
    "student4": 4,
    "W4R2": 12,
    "W4": 13,
    "TBRG": 13,
    "STS2": 13,
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
UTMOS_CONFIRM_MIN_MEAN = -0.12

DIV_FIELDS = (
    ["utt_id", "speaker_id", "condition", "n_seeds"]
    + [f"{m}_{a}" for a in NEW_ARMS for m in ("cvf0", "voiced", "mel_spread")]
    + ["case_seconds", "utc"]
)

QUAL_FIELDS = (
    ["utt_id", "speaker_id", "condition", "prompt_snr", "gpu"]
    + list(DRIFT_FIELDS)
    + [f"{m}_{a}" for a in NEW_ARMS for m in METRICS + ("hypothesis",)]
    + ["case_seconds", "utc"]
)


def cost(arm):
    """Total neural function evaluations (NFE) for an arm."""
    return COST[arm]


def div_items(n=DIV_PAIRS):
    """Return the first n fresh speakers (indices 60..79) for multi-seed diversity runs."""
    return tc.fresh_items()[:n]


def qual_items(n=FRESH_N):
    """Return all n fresh speakers (indices 60..99) matching EXP-027."""
    return tc.fresh_items(n)


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


# --------------------------------------------------------------------------- core operator
def teacher_step4(pipe, ctx):
    """Run guided Euler teacher steps 0..4 (10 NFE) and return (T4, drift_metrics)."""
    torch, sampler = pipe.torch, pipe.sampler
    grid = tt.grid16(pipe)
    plen = ctx["prompt_len"]
    x = sh._clone(ctx["y0"])
    prev = None
    drifts = {}
    ep = None
    with torch.inference_mode():
        for i in range(5):
            t = torch.tensor(grid[i], device=pipe.device, dtype=torch.float32)
            kw = dict(x=x, cond=ctx["step_cond"], text=ctx["text_ids"], time=t, mask=None)
            vc = pipe.tts.teacher(drop_audio_cond=False, drop_text=False, **kw)
            vu = pipe.tts.teacher(drop_audio_cond=True, drop_text=True, **kw)
            v = vc + (vc - vu) * sampler.CFG_STRENGTH
            ep = x + (1.0 - grid[i]) * v
            if prev is not None:
                drifts[i] = float(sampler.compute_stabilization_drift(ep, prev, plen))
            prev = ep
            if i < 4:
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
    return ep, metrics


def student_step_tau1(pipe, ctx, T4):
    """Run the student's tau_1 = 0.0761 step from T4 to obtain clean draft S1 (1 NFE)."""
    tau1 = float(pipe.sampler.STUDENT_TAUS[0])
    return gl.student(pipe, ctx, T4, [tau1])


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


def bridge_outputs(pipe, ctx, T4):
    """Synthesize TBRG (energy-matched) and STS2 (unnormalized) mels from T4 (1 shared tau_1 + 2x2 steps)."""
    taus_tail = list(pipe.sampler.STUDENT_TAUS)[1:]
    plen = ctx["prompt_len"]
    s1 = student_step_tau1(pipe, ctx, T4)
    x_tbrg = bridge_handover(T4, s1, plen, energy_match=True)
    x_sts2 = bridge_handover(T4, s1, plen, energy_match=False)
    return {
        "TBRG": gl.student(pipe, ctx, x_tbrg, taus_tail),
        "STS2": gl.student(pipe, ctx, x_sts2, taus_tail),
        "S1": s1,
    }


# --------------------------------------------------------------------------- synthesis loops
def run_diversity(pipe, items, out_dir, seeds=DIV_SEEDS, csv_name="diversity.csv"):
    """Run multi-seed synthesis for TBRG and STS2 and log CVf0, voiced share, and slow-mel spread."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / csv_name
    done = rf.done_keys(path, 1)
    todo = [it for it in items if (it["utt_id"], "clean") not in done]
    print(f"[diversity] {len(todo)} speakers x {seeds} seeds to run ({len(done)} already in {path.name})", flush=True)

    for idx, item in enumerate(todo, 1):
        t0 = time.perf_counter()
        tracks = {a: [] for a in NEW_ARMS}
        slow_mels = {a: [] for a in NEW_ARMS}
        for m in range(seeds):
            seed = int(item["input_seed"]) + DIV_SEED_BASE + m
            ctx = pipe.prepare(item, "clean", seed, out_dir, noise_seed=item["input_seed"])
            t4, _ = teacher_step4(pipe, ctx)
            outs = bridge_outputs(pipe, ctx, t4)
            plen = ctx["prompt_len"]
            for arm in NEW_ARMS:
                slow_mels[arm].append(tc._slow_component(outs[arm], plen))
                tracks[arm].append(sh.f0_track(ng.decode(pipe, ctx, outs[arm])))

        row = {"utt_id": item["utt_id"], "speaker_id": item["speaker_id"], "condition": "clean", "n_seeds": seeds}
        for arm in NEW_ARMS:
            row[f"cvf0_{arm}"], row[f"voiced_{arm}"] = sh.cvf0(tracks[arm])
            row[f"mel_spread_{arm}"] = tc.slow_mel_spread(slow_mels[arm])
        row.update({"case_seconds": round(time.perf_counter() - t0, 2), "utc": _utc()})
        _append(path, DIV_FIELDS, row)
        print(f"[{idx}/{len(todo)}] diversity: {item['utt_id']} ({row['case_seconds']:.1f} s)", flush=True)
    print(f"[OK] diversity complete: {path}", flush=True)
    return path


def run_quality(pipe, items, out_dir, csv_name="quality.csv"):
    """Synthesize and score single-seed quality plus Step-4 drift features on clean prompts."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / csv_name
    done = rf.done_keys(path, 1)
    todo = [it for it in items if (it["utt_id"], "clean") not in done]
    print(f"[quality] {len(todo)} speakers to run ({len(done)} already in {path.name})", flush=True)

    for idx, item in enumerate(todo, 1):
        t0 = time.perf_counter()
        ctx = pipe.prepare(item, "clean", int(item["input_seed"]), out_dir)
        t4, drift = teacher_step4(pipe, ctx)
        outs = bridge_outputs(pipe, ctx, t4)
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
        print(f"[{idx}/{len(todo)}] quality: {item['utt_id']} ({row['case_seconds']:.1f} s)", flush=True)
    print(f"[OK] quality complete: {path}", flush=True)
    return path


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
def build_prereg(fresh=None, exp027_div_csv=EXP027_DIV_CSV, exp027_qual_csv=EXP027_QUAL_CSV):
    """Build deterministic preregistration payload with dependency, fresh-manifest, and EXP-027 hashes."""
    if fresh is None:
        fresh = qual_items()
    fresh_utts = [{"utt_id": it["utt_id"], "speaker_id": it["speaker_id"]} for it in fresh]
    code_hashes = {name: _sha256_file(CODE_DIR / name) for name in DEPENDENCIES}
    input_hashes = {
        str(Path(exp027_div_csv).relative_to(THESIS_ROOT).as_posix()): _sha256_file(exp027_div_csv),
        str(Path(exp027_qual_csv).relative_to(THESIS_ROOT).as_posix()): _sha256_file(exp027_qual_csv),
    }
    fresh_sha = _sha256_bytes(json.dumps(fresh_utts, sort_keys=True).encode("utf-8"))
    payload = {
        "experiment": "EXP-028",
        "task_id": "EVAL-021",
        "title": "Transmissible Slow-Band Bridge (TBRG) and STAC Handover on 40 fresh speakers",
        "n_fresh_qual": len(fresh_utts),
        "n_fresh_div": DIV_PAIRS,
        "fresh_utts": fresh_utts,
        "fresh_utts_sha256": fresh_sha,
        "div_seeds": DIV_SEEDS,
        "div_seed_base": DIV_SEED_BASE,
        "new_arms": list(NEW_ARMS),
        "stored_arms": list(STORED_ARMS),
        "costs": dict(COST),
        "stac_folds": FOLDS,
        "stac_utmos_tol": STAC_UTMOS_TOL,
        "stac_wer_tol": STAC_WER_TOL,
        "boot_b": BOOT_B,
        "seed": SEED,
        "code_hashes": code_hashes,
        "input_hashes": input_hashes,
        "hypotheses": {
            "H1": "CVf0(TBRG) - CVf0(W4) > 0 (slow-band bridge unlocks transmissible pitch variety at 13 NFE)",
            "H2": "UTMOS(TBRG) - UTMOS(W4R2) > 0 (S1 voice/fine anchor + energy match recovers raw W4R2 UTMOS loss)",
            "H3": "TBRG preserves clean W8 WER/SIM (upper CI dWER <= +0.01, lower CI dSIM >= -0.01, mean dUTMOS >= -0.12)",
            "H4": "mel_spread(TBRG) - mel_spread(W4R2) < 0 (bridge prevents early re-noising artifact dispersion)",
        },
        "verdict_rule": (
            "refuted if H1 CVf0 95% upper CI < 0, H2 UTMOS(TBRG-W4R2) upper CI < 0, or TBRG fails W8 non-inferiority "
            "(dWER lower CI > +0.01 or dSIM upper CI < -0.01); "
            "confirmed if H1 CVf0 lower CI > 0 & no-top2 > 0, H2 UTMOS lower CI > 0 & no-top2 > 0, "
            "TBRG meets W8 WER/SIM non-inferiority & mean dUTMOS >= -0.12, and H4 mel_spread(TBRG-W4R2) upper CI < 0; "
            "supported_direction_only if H1 CVf0 lower CI > 0 and H2 UTMOS lower CI > 0 without full confirmation; "
            "mixed otherwise."
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
    """Verify preregistration integrity, code hashes, locked EXP-027 input hashes, and fresh speakers."""
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
    """5-fold speaker-CV routing between TBRG (13 NFE) and W8R2 (20 NFE)."""
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

    # For relative_r4 (smaller = more settled), TBRG if score <= tau (None = never -> pure W8R2).
    # For cumulative_A4 (larger = more action completed), TBRG if score >= tau (None = never).
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
    """Evaluate 5-fold speaker-CV STAC routing policies between TBRG (13 NFE) and W8R2 (20 NFE)."""
    return {
        "relative_r4": _stac_policy_eval(qual_rows, div_map, stored_qual, stored_div, feature="relative_r4"),
        "cumulative_A4": _stac_policy_eval(qual_rows, div_map, stored_qual, stored_div, feature="cumulative_A4"),
    }


def determine_verdict(contrasts):
    """Apply the preregistered frozen verdict rule for EXP-028."""
    h1_cv = contrasts["H1_TBRG_minus_W4"]["cvf0"]
    h2_u = contrasts["H2_TBRG_minus_W4R2_utmos"]
    h3_w = contrasts["H3_TBRG_vs_W8_quality"]["wer"]
    h3_s = contrasts["H3_TBRG_vs_W8_quality"]["similarity"]
    h3_u = contrasts["H3_TBRG_vs_W8_quality"]["utmos"]
    h4_ms = contrasts["H4_TBRG_minus_W4R2_mel_spread"]

    if (
        h1_cv["boot"]["hi"] < -ZERO_TOL
        or h2_u["boot"]["hi"] < -ZERO_TOL
        or h3_w["boot"]["lo"] > WER_NONINF_MARGIN
        or h3_s["boot"]["hi"] < SIM_NONINF_MARGIN
    ):
        return "refuted"

    h1_ok = h1_cv["boot"]["lo"] > ZERO_TOL
    h2_ok = h2_u["boot"]["lo"] > ZERO_TOL
    h1_rob = h1_cv["robust"]["mean_without_top2_abs"] > ZERO_TOL
    h2_rob = h2_u["robust"]["mean_without_top2_abs"] > ZERO_TOL
    h3_ok = (
        h3_w["boot"]["hi"] <= WER_NONINF_MARGIN
        and h3_s["boot"]["lo"] >= SIM_NONINF_MARGIN
        and h3_u["boot"]["mean"] >= UTMOS_CONFIRM_MIN_MEAN
    )
    h4_ok = h4_ms["boot"]["hi"] < -ZERO_TOL

    if h1_ok and h2_ok and h1_rob and h2_rob and h3_ok and h4_ok:
        return "confirmed"
    if h1_ok and h2_ok:
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
        }
    for arm in NEW_ARMS:
        rows = list(div_map.values())
        out[arm] = {
            "n": len(rows),
            "cvf0": tc.boot_ci([float(r[f"cvf0_{arm}"]) for r in rows], B=BOOT_B, seed=SEED),
            "voiced": tc.boot_ci([float(r[f"voiced_{arm}"]) for r in rows], B=BOOT_B, seed=SEED),
            "mel_spread": tc.boot_ci([float(r[f"mel_spread_{arm}"]) for r in rows], B=BOOT_B, seed=SEED),
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


def analyze(
    out_dir,
    allow_partial=False,
    prereg_path=PREREG_PATH,
    exp027_div_csv=EXP027_DIV_CSV,
    exp027_qual_csv=EXP027_QUAL_CSV,
):
    """Validate preregistration & completeness, merge EXP-027 references, compute CIs/STAC-CV, and write report."""
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
        "TBRG_vs_STS2": {
            "cvf0": _summarize(_paired_contrast(div_map, div_map, "cvf0_TBRG", "cvf0_STS2")),
            "mel_spread": _summarize(_paired_contrast(div_map, div_map, "mel_spread_TBRG", "mel_spread_STS2")),
            **{m: _summarize(_paired_contrast(qual_map, qual_map, f"{m}_TBRG", f"{m}_STS2")) for m in METRICS},
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
    is_partial = allow_partial and (len(div_rows) < DIV_PAIRS or len(qual_rows) < FRESH_N)

    results = {
        "experiment": "EXP-028",
        "partial": is_partial,
        "n_diversity": len(div_rows),
        "n_quality": len(qual_rows),
        "verdict": verdict,
        "arm_diversity": arm_diversity,
        "arm_quality": arm_quality,
        "contrasts": contrasts,
        "stac_routing": stac_routing,
        "utc": _utc(),
    }

    rjson = out_dir / "transmissibility_bridge_results.json"
    rmd = out_dir / "TRANSMISSIBILITY_BRIDGE_REPORT.md"
    rjson.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# EXP-028: Transmissible Slow-Band Bridge & STAC Handover Report",
        "",
        f"- **Verdict:** `{verdict}` ({'PARTIAL' if is_partial else 'FULL'})",
        f"- **Fresh speakers scored:** diversity = {len(div_rows)}/{DIV_PAIRS} (x {DIV_SEEDS} seeds), "
        f"quality = {len(qual_rows)}/{FRESH_N}",
        "",
        "## Confirmatory Hypotheses (`TBRG`, 13 NFE)",
        "",
        "| Hypothesis | Estimate (95% CI) | No-top2 mean | Sign (+/-, p) |",
        "| --- | ---: | ---: | ---: |",
    ]
    rows_md = [
        ("`H1: CVf0(TBRG - W4)`", contrasts["H1_TBRG_minus_W4"]["cvf0"], 4),
        ("`H2: UTMOS(TBRG - W4R2)`", contrasts["H2_TBRG_minus_W4R2_utmos"], 3),
        ("`H3: WER(TBRG - W8)`", contrasts["H3_TBRG_vs_W8_quality"]["wer"], 4),
        ("`H3: SIM(TBRG - W8)`", contrasts["H3_TBRG_vs_W8_quality"]["similarity"], 4),
        ("`H3: UTMOS(TBRG - W8)`", contrasts["H3_TBRG_vs_W8_quality"]["utmos"], 3),
        ("`H4: MelSpread(TBRG - W4R2)`", contrasts["H4_TBRG_minus_W4R2_mel_spread"], 4),
    ]
    for label, c, digits in rows_md:
        b = c["boot"]
        lines.append(
            f"| {label} | {tc._fmt_ci(b, digits)} | {c['robust']['mean_without_top2_abs']:+.{digits}f} | "
            f"{c['sign']['pos']}/{c['sign']['neg']} (p={c['sign']['p']:.3g}) |"
        )

    lines.extend([
        "",
        "## Ablations & Reference Comparisons",
        "",
        "| Contrast | Estimate (95% CI) | Sign (+/-, p) |",
        "| --- | ---: | ---: |",
        f"| `TBRG - STS2 (CVf0)` | {tc._fmt_ci(contrasts['TBRG_vs_STS2']['cvf0']['boot'], 4)} | "
        f"{contrasts['TBRG_vs_STS2']['cvf0']['sign']['pos']}/{contrasts['TBRG_vs_STS2']['cvf0']['sign']['neg']} |",
        f"| `TBRG - STS2 (UTMOS)` | {tc._fmt_ci(contrasts['TBRG_vs_STS2']['utmos']['boot'], 3)} | "
        f"{contrasts['TBRG_vs_STS2']['utmos']['sign']['pos']}/{contrasts['TBRG_vs_STS2']['utmos']['sign']['neg']} |",
        f"| `TBRG - W8R2 (CVf0)` | {tc._fmt_ci(contrasts['TBRG_vs_W8R2']['cvf0']['boot'], 4)} | "
        f"{contrasts['TBRG_vs_W8R2']['cvf0']['sign']['pos']}/{contrasts['TBRG_vs_W8R2']['cvf0']['sign']['neg']} |",
        f"| `TBRG - W8R2 (UTMOS)` | {tc._fmt_ci(contrasts['TBRG_vs_W8R2']['utmos']['boot'], 3)} | "
        f"{contrasts['TBRG_vs_W8R2']['utmos']['sign']['pos']}/"
        f"{contrasts['TBRG_vs_W8R2']['utmos']['sign']['neg']} |",
        "",
        "## Side-by-Side Fresh-Speaker Summary (New EXP-028 + Stored EXP-027)",
        "",
        "| Arm | NFE | CVf0 (20 spk) | MelSpread (20 spk) | WER vs W8 (40 spk) | SIM vs W8 | UTMOS vs W8 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for arm in ALL_ARMS:
        d = arm_diversity[arm]
        q = arm_quality[arm]
        w_s = f"{q['wer']['mean']:.4f}" if arm == "W8" else tc._fmt_ci(q["d_wer_vs_W8"], 4)
        s_s = f"{q['similarity']['mean']:.4f}" if arm == "W8" else tc._fmt_ci(q["d_similarity_vs_W8"], 4)
        u_s = f"{q['utmos']['mean']:.3f}" if arm == "W8" else tc._fmt_ci(q["d_utmos_vs_W8"], 3)
        lines.append(
            f"| `{arm}` | {COST[arm]} | {d['cvf0']['mean']:.4f} | {d['mel_spread']['mean']:.4f} | "
            f"{w_s} | {s_s} | {u_s} |"
        )

    rel = stac_routing.get("relative_r4")
    if rel is not None:
        lines.extend([
            "",
            "## Adaptive 5-Fold CV Routing (`STAC-TBRG` between 13-NFE `TBRG` and 20-NFE `W8R2`)",
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
    _write_manifest(out_dir, [div_path, qual_path, rjson, rmd], prereg_path)
    print(f"[OK] Analysis complete: verdict={verdict}, report={rmd}", flush=True)
    return results


def main():
    p = argparse.ArgumentParser(description="EXP-028 Transmissible Slow-Band Bridge & STAC Handover runner")
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
    r.add_argument("--stage", choices=("both", "diversity", "quality"), default="both")

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
        if args.stage in ("both", "diversity"):
            run_diversity(pipe, div_items(), args.out_dir)
        if args.stage in ("both", "quality"):
            run_quality(pipe, qual_items(), args.out_dir)
    elif args.cmd == "analyze":
        analyze(args.out_dir, allow_partial=args.allow_partial)


if __name__ == "__main__":
    main()
