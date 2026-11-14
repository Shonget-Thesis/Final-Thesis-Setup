"""Confirmatory Colab T4 test of the transmissibility theory on fresh speakers (EXP-027 / EVAL-020).

Addresses the four blockers left open by EXP-026 on 40 fresh development speakers (indices 60..99,
speakers 61-100, strictly disjoint from the 60 development speakers used in EXP-012 through EXP-026):

  1. No stored-data reuse: synthesizes all 6 arms (student4, W4, W8, W4R2, W8R2, T16) fresh
     from one shared 16-step teacher trajectory per (speaker, seed), plus clean quality on all 40.
  2. Discriminates the Phi((A - beta)/B) ceiling from a simpler linear-erasure view via the
     preregistered Step-by-Restart interaction contrast:
       H4 = (W8R2 - W4R2) - (W8 - W4) > 0
     At default restart tau_1 = 0.0761, Phi saturates so W8 - W4 is flat; at late restart
     tau_2 = 0.2929, enough teacher mass remains for the Step-8 vs Step-4 gap to appear.
  3. Eliminates pitch-tracker voiced-frame selection confounds with two controls in diversity.csv:
       - cvf0_common_<arm>: CVf0 restricted to the intersection of frames voiced (>= 50% seeds)
         across all 6 arms on the same utterance.
       - mel_spread_<arm>: pitch-tracker-free pairwise Frobenius spread of slow-mel frames
         extracted via sh.decompose(..., SLOW_WINDOW)[1].
  4. Tests outlier robustness (median, mean without top-2 |d| speakers, 10% trimmed mean, exact
     two-sided sign test) so one outlier speaker cannot carry the conclusion alone.

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
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_consistency_gate_prelim as cg  # noqa: E402
import run_followup_tests as rf  # noqa: E402
import run_grid32_prelim as gl  # noqa: E402
import run_noise_guidance_prelim as ng  # noqa: E402
import run_selective_handover_prelim as sh  # noqa: E402
import run_transmissibility_test as tt  # noqa: E402

THESIS_ROOT = rf.THESIS_ROOT
CODE_DIR = Path(__file__).resolve().parent
EXP_DIR = THESIS_ROOT / "research/experiments/EXP-027-transmissibility-confirmation"
PREREG_PATH = EXP_DIR / "preregistration.json"

DEPENDENCIES = (
    "run_transmissibility_confirm.py",
    "run_transmissibility_test.py",
    "run_selective_handover_prelim.py",
    "run_consistency_gate_prelim.py",
    "run_grid32_prelim.py",
    "run_noise_guidance_prelim.py",
    "run_followup_tests.py",
)

DEV_N = 60
FRESH_N = 40
DIV_PAIRS = 40
DIV_SEEDS = 20
DIV_SEED_BASE = 1000
SEED = 2026
BOOT_B = 2000
PLATEAU_TOL = 0.01
ZERO_TOL = 1e-12
REPRODUCTION_TOLERANCE = 0.01

ARMS = ("student4", "W4", "W8", "W4R2", "W8R2", "T16")
COST = {"student4": 4, "W4R2": 12, "W4": 13, "W8R2": 20, "W8": 21, "T16": 32}
METRICS = ("wer", "similarity", "utmos")

DIV_FIELDS = (
    ["utt_id", "speaker_id", "condition", "n_seeds"]
    + [f"cvf0_{a}" for a in ARMS]
    + [f"voiced_{a}" for a in ARMS]
    + [f"cvf0_common_{a}" for a in ARMS]
    + ["voiced_common"]
    + [f"mel_spread_{a}" for a in ARMS]
    + ["case_seconds", "utc"]
)

QUAL_FIELDS = (
    ["utt_id", "speaker_id", "condition", "prompt_snr", "gpu"]
    + [f"{m}_{a}" for a in ARMS for m in METRICS + ("hypothesis",)]
    + ["case_seconds", "utc"]
)


def cost(arm):
    """Total neural function evaluations (NFE) for an arm."""
    return COST[arm]


def fresh_items(n=FRESH_N):
    """Return the n fresh speakers immediately following the 60 dev speakers (indices 60..99)."""
    all_dev = cg.dev_items()
    seen = {it["utt_id"] for it in all_dev[:DEV_N]}
    fresh = all_dev[DEV_N:DEV_N + n]
    if len(fresh) != n or (seen & {it["utt_id"] for it in fresh}):
        raise RuntimeError(f"Expected {n} fresh speakers disjoint from first {DEV_N} dev speakers")
    return fresh


def _utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _append(path, fields, row):
    new = not Path(path).exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new:
            w.writeheader()
        w.writerow(row)
        f.flush()
        os.fsync(f.fileno())


def _sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path):
    return sh.sha256_text(path)


# --------------------------------------------------------------------------- diversity measures
def common_cvf0(tracks_by_arm, min_voiced=sh.MIN_VOICED):
    """CVf0 computed on the exact intersection of voiced frames across all arms.

    Each arm's per-frame mask requires >= max(2, ceil(min_voiced * n_seeds)) voiced seeds.
    Returns ({arm: cvf0}, share_of_frames_kept).
    """
    import numpy as np

    arms = tuple(tracks_by_arm)
    if not arms or any(len(tracks_by_arm[a]) < 2 for a in arms):
        return {a: float("nan") for a in arms}, 0.0
    n_frames = min(len(t) for a in arms for t in tracks_by_arm[a])
    if n_frames == 0:
        return {a: float("nan") for a in arms}, 0.0

    arrays, masks = {}, {}
    for a in arms:
        arr = np.stack([np.asarray(t[:n_frames], dtype=float) for t in tracks_by_arm[a]])
        req = max(2, math.ceil(min_voiced * arr.shape[0]))
        arrays[a] = arr
        masks[a] = (~np.isnan(arr)).sum(axis=0) >= req

    keep = np.logical_and.reduce(list(masks.values()))
    if not keep.any():
        return {a: float("nan") for a in arms}, 0.0

    out = {}
    for a in arms:
        sub = arrays[a][:, keep]
        out[a] = float(np.mean(np.nanstd(sub, axis=0, ddof=1) / np.nanmean(sub, axis=0)))
    return out, float(keep.mean())


def slow_mel_spread(mats):
    """Pitch-tracker-free pairwise Frobenius spread of slow-mel components: mean ||X_i - X_j|| / mean ||X_i||."""
    import numpy as np

    if len(mats) < 2:
        return float("nan")
    t_min = min(m.shape[0] for m in mats)
    if t_min == 0:
        return float("nan")
    arrs = [np.asarray(m[:t_min], dtype=float) for m in mats]
    norms = [float(np.linalg.norm(a)) for a in arrs]
    pairs = [float(np.linalg.norm(arrs[i] - arrs[j])) for i in range(len(arrs)) for j in range(i + 1, len(arrs))]
    return (sum(pairs) / len(pairs)) / (sum(norms) / len(norms) + 1e-12)


# --------------------------------------------------------------------------- synthesis loops
def outputs(pipe, ctx, ends, final):
    """Synthesize all 6 arm mel outputs from one teacher trajectory (ends 4, 8 and final 16)."""
    out = dict(tt.outputs(pipe, ctx, ends, final))
    taus = list(pipe.sampler.STUDENT_TAUS)
    init = ctx.get("y0", ctx.get("step_cond"))
    out["student4"] = gl.student(pipe, ctx, init, [0.0] + taus)
    return out


def _slow_component(mel, prompt_len):
    gen = mel[0, prompt_len:, :]
    _, slow, _ = sh.decompose(gen, sh.SLOW_WINDOW)
    if hasattr(slow, "detach"):
        return slow.detach().cpu().numpy()
    import numpy as np
    return np.asarray(slow, dtype=float)


def run_diversity(pipe, items, out_dir, seeds=DIV_SEEDS, csv_name="diversity.csv"):
    """Synthesize clean-prompt seed trajectories and log CVf0, common-frame CVf0, and slow-mel spread."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / csv_name
    done = rf.done_keys(path, 1)
    todo = [it for it in items if (it["utt_id"], "clean") not in done]
    print(f"[diversity] {len(todo)} speakers x {seeds} seeds to run ({len(done)} already in {path.name})", flush=True)

    for idx, item in enumerate(todo, 1):
        t0 = time.perf_counter()
        tracks = {a: [] for a in ARMS}
        slow_mels = {a: [] for a in ARMS}
        for m in range(seeds):
            seed = int(item["input_seed"]) + DIV_SEED_BASE + m
            ctx = pipe.prepare(item, "clean", seed, out_dir, noise_seed=item["input_seed"])
            ends, final = tt.teacher_run(pipe, ctx)
            outs = outputs(pipe, ctx, ends, final)
            plen = ctx["prompt_len"]
            for arm in ARMS:
                slow_mels[arm].append(_slow_component(outs[arm], plen))
                tracks[arm].append(sh.f0_track(ng.decode(pipe, ctx, outs[arm])))

        common_vals, common_share = common_cvf0(tracks)
        row = {"utt_id": item["utt_id"], "speaker_id": item["speaker_id"], "condition": "clean", "n_seeds": seeds,
               "voiced_common": common_share}
        for arm in ARMS:
            row[f"cvf0_{arm}"], row[f"voiced_{arm}"] = sh.cvf0(tracks[arm])
            row[f"cvf0_common_{arm}"] = common_vals[arm]
            row[f"mel_spread_{arm}"] = slow_mel_spread(slow_mels[arm])
        row.update({"case_seconds": round(time.perf_counter() - t0, 2), "utc": _utc()})
        _append(path, DIV_FIELDS, row)
        print(f"[{idx}/{len(todo)}] diversity: {item['utt_id']} ({row['case_seconds']:.1f} s)", flush=True)
    print(f"[OK] diversity complete: {path}", flush=True)
    return path


def run_quality(pipe, items, out_dir, csv_name="quality.csv"):
    """Score WER, speaker similarity, and UTMOS across all 6 arms on clean prompts (single trajectory each)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / csv_name
    done = rf.done_keys(path, 1)
    todo = [it for it in items if (it["utt_id"], "clean") not in done]
    print(f"[quality] {len(todo)} speakers to run ({len(done)} already in {path.name})", flush=True)

    for idx, item in enumerate(todo, 1):
        t0 = time.perf_counter()
        ctx = pipe.prepare(item, "clean", int(item["input_seed"]), out_dir)
        ends, final = tt.teacher_run(pipe, ctx)
        outs = outputs(pipe, ctx, ends, final)
        gt = pipe.ground_truth(item)
        row = {"utt_id": item["utt_id"], "speaker_id": item["speaker_id"], "condition": "clean",
               "prompt_snr": ctx["prompt_snr"], "gpu": pipe.gpu}
        for arm in ARMS:
            m = pipe.score(ng.decode(pipe, ctx, outs[arm]), item, gt)
            row.update({f"{k}_{arm}": m[k] for k in METRICS + ("hypothesis",)})
        row.update({"case_seconds": round(time.perf_counter() - t0, 2), "utc": _utc()})
        _append(path, QUAL_FIELDS, row)
        print(f"[{idx}/{len(todo)}] quality: {item['utt_id']} ({row['case_seconds']:.1f} s)", flush=True)
    print(f"[OK] quality complete: {path}", flush=True)
    return path


def smoke(pipe, out_dir, prereg_path=PREREG_PATH):
    """Run a 2-dev-speaker verification smoke test: W4/W8 reproduction, 2-seed diversity, and 6-arm quality."""
    check_prereg(prereg_path)
    sdir = Path(out_dir) / "smoke"
    sdir.mkdir(parents=True, exist_ok=True)
    items = tt.smoke_items()
    stored = tt.stored_w()
    repro = []
    for item in items:
        ctx = pipe.prepare(item, "clean", int(item["input_seed"]), sdir)
        ends, final = tt.teacher_run(pipe, ctx)
        outs = outputs(pipe, ctx, ends, final)
        gt = pipe.ground_truth(item)
        for arm in ("W4", "W8"):
            m = pipe.score(ng.decode(pipe, ctx, outs[arm]), item, gt)
            s = stored[(item["utt_id"], "clean")][arm]
            d_wer = abs(m["wer"] - s["wer"])
            d_sim = abs(m["similarity"] - s["similarity"])
            if d_wer > REPRODUCTION_TOLERANCE or d_sim > REPRODUCTION_TOLERANCE:
                raise RuntimeError(
                    f"Reproduction failed on {item['utt_id']} {arm}: "
                    f"d_wer={d_wer:.4f}, d_sim={d_sim:.4f} (tol {REPRODUCTION_TOLERANCE})"
                )
            repro.append({"utt_id": item["utt_id"], "arm": arm, "d_wer": d_wer, "d_sim": d_sim})

    run_diversity(pipe, items, sdir, seeds=2, csv_name="diversity.csv")
    run_quality(pipe, items, sdir, csv_name="quality.csv")
    summary = {"repro": repro, "n_smoke_items": len(items), "max_d_wer": max(r["d_wer"] for r in repro),
               "max_d_sim": max(r["d_sim"] for r in repro), "utc": _utc()}
    (sdir / "smoke_check.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"[OK] smoke passed (max d_wer={summary['max_d_wer']:.4f}, max d_sim={summary['max_d_sim']:.4f})", flush=True)
    return summary


# --------------------------------------------------------------------------- preregistration
def build_prereg(fresh=None):
    """Build deterministic preregistration payload and checksums before GPU synthesis."""
    if fresh is None:
        fresh = fresh_items()
    fresh_utts = [{"utt_id": it["utt_id"], "speaker_id": it["speaker_id"]} for it in fresh]
    code_hashes = {name: _sha256_file(CODE_DIR / name) for name in DEPENDENCIES}
    fresh_sha = _sha256_bytes(json.dumps(fresh_utts, sort_keys=True).encode("utf-8"))
    payload = {
        "experiment": "EXP-027",
        "task_id": "EVAL-020",
        "title": "Transmissibility confirmation and Step-by-Restart interaction on fresh speakers",
        "n_dev_excluded": DEV_N,
        "n_fresh": len(fresh_utts),
        "fresh_utts": fresh_utts,
        "fresh_utts_sha256": fresh_sha,
        "div_seeds": DIV_SEEDS,
        "div_seed_base": DIV_SEED_BASE,
        "arms": list(ARMS),
        "costs": dict(COST),
        "plateau_tol": PLATEAU_TOL,
        "boot_b": BOOT_B,
        "seed": SEED,
        "code_hashes": code_hashes,
        "hypotheses": {
            "H1": "T16 - W8 > 0 (pure teacher has higher pitch CVf0 than default Step-8 handover at tau_1 = 0.0761)",
            "H2": "W8R2 - W8 > 0 (late restart at tau_2 = 0.2929 from Step 8 recovers pitch CVf0 over default tau_1)",
            "H3": "|W8 - W4| < 0.01 (flat plateau at default restart tau_1 = 0.0761 on fresh data, no stored-data reuse)",
            "H4": "(W8R2 - W4R2) - (W8 - W4) > 0 (Step-by-Restart interaction predicted by Phi ceiling vs linear erasure)",
            "H5_exploratory": "Spearman rho(T16 - W8, W8R2 - W8) > 0 across fresh speakers",
        },
        "verdict_rule": (
            "refuted if 95% upper CI < 0 on any of H1, H2, H4 (in cvf0 or cvf0_common); "
            "confirmed if H1, H2, H4 have 95% lower CI > 0 in both cvf0 and cvf0_common, |W8 - W4| < 0.01 in cvf0, "
            "slow_mel_spread means of H1, H2, H4 are > 0, and mean_without_top2_abs of H1, H2, H4 are > 0; "
            "supported_direction_only if H1, H2 have 95% lower CI > 0 and |W8 - W4| < 0.01 in cvf0 without full confirmation; "
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
    """Verify preregistration integrity, code dependency SHA-256 hashes, and fresh speaker manifest."""
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
    current_utts = [{"utt_id": it["utt_id"], "speaker_id": it["speaker_id"]} for it in fresh_items()]
    if doc.get("fresh_utts") != current_utts:
        raise RuntimeError("Fresh speaker list does not match preregistration manifest.")
    return doc


# --------------------------------------------------------------------------- statistical analysis
def boot_ci(vals, B=BOOT_B, seed=SEED):
    """Bootstrap 95% confidence interval for the mean of paired per-speaker values."""
    clean = [float(v) for v in vals if not math.isnan(float(v))]
    n = len(clean)
    if n == 0:
        return {"n": 0, "mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "sd": float("nan")}
    mean = sum(clean) / n
    sd = (sum((v - mean) ** 2 for v in clean) / (n - 1)) ** 0.5 if n > 1 else 0.0
    rng = random.Random(seed)
    means = sorted(sum(clean[rng.randrange(n)] for _ in range(n)) / n for _ in range(B))
    return {"n": n, "mean": mean, "lo": means[int(0.025 * B)], "hi": means[min(B - 1, int(0.975 * B))], "sd": sd}


def sign_test(diffs):
    """Exact two-sided binomial sign test on non-zero differences."""
    nz = [d for d in diffs if not math.isnan(d) and abs(d) > ZERO_TOL]
    n = len(nz)
    pos = sum(1 for d in nz if d > ZERO_TOL)
    neg = n - pos
    if n == 0:
        return {"n": 0, "pos": 0, "neg": 0, "p": 1.0}
    k = min(pos, neg)
    p = min(1.0, 2.0 * sum(math.comb(n, i) for i in range(k + 1)) * (0.5 ** n))
    return {"n": n, "pos": pos, "neg": neg, "p": p}


def robust_stats(diffs):
    """Median, mean without top-2 |d| speakers, 10% trimmed mean, and max single-speaker share."""
    clean = [float(d) for d in diffs if not math.isnan(float(d))]
    n = len(clean)
    if n == 0:
        return {"median": float("nan"), "mean_without_top2_abs": float("nan"),
                "trimmed_mean_10": float("nan"), "max_abs_share": float("nan")}
    s = sorted(clean)
    mid = n // 2
    med = s[mid] if n % 2 else 0.5 * (s[mid - 1] + s[mid])
    wo2 = sorted(clean, key=abs)[:-2] if n > 2 else list(clean)
    trim_k = max(1, n // 10)
    trimmed = s[trim_k: n - trim_k] if n >= 10 else list(s)
    total = sum(clean)
    return {
        "median": med,
        "mean_without_top2_abs": sum(wo2) / len(wo2),
        "trimmed_mean_10": sum(trimmed) / len(trimmed),
        "max_abs_share": (max(abs(d) for d in clean) / abs(total)) if total != 0.0 else float("nan"),
    }


def _extract_contrast(rows, prefix, expr_fn):
    vals = []
    for r in rows:
        try:
            v = expr_fn(lambda arm: float(r[f"{prefix}_{arm}"]))
        except (KeyError, ValueError):
            continue
        if not math.isnan(v):
            vals.append(v)
    return vals


CONTRAST_FNS = {
    "H1_T16_minus_W8": lambda g: g("T16") - g("W8"),
    "H2_W8R2_minus_W8": lambda g: g("W8R2") - g("W8"),
    "H3_W8_minus_W4": lambda g: g("W8") - g("W4"),
    "H4_interaction": lambda g: (g("W8R2") - g("W4R2")) - (g("W8") - g("W4")),
    "W8R2_minus_W4R2": lambda g: g("W8R2") - g("W4R2"),
    "W4R2_minus_W4": lambda g: g("W4R2") - g("W4"),
    "W4_minus_student4": lambda g: g("W4") - g("student4"),
}


def _summarize_contrast(rows, prefix, name, seed=SEED):
    vals = _extract_contrast(rows, prefix, CONTRAST_FNS[name])
    return {"boot": boot_ci(vals, seed=seed), "sign": sign_test(vals), "robust": robust_stats(vals)}


def determine_verdict(by_metric):
    """Apply the preregistered frozen verdict rule across cvf0, cvf0_common, and mel_spread."""
    cv = by_metric["cvf0"]
    cc = by_metric["cvf0_common"]
    ms = by_metric["mel_spread"]

    h1_b, h2_b, h3_b, h4_b = (cv[k]["boot"] for k in ("H1_T16_minus_W8", "H2_W8R2_minus_W8", "H3_W8_minus_W4", "H4_interaction"))
    c1_b, c2_b, c4_b = (cc[k]["boot"] for k in ("H1_T16_minus_W8", "H2_W8R2_minus_W8", "H4_interaction"))

    if any(b["hi"] < -ZERO_TOL for b in (h1_b, h2_b, h4_b, c1_b, c2_b, c4_b) if not math.isnan(b["hi"])):
        return "refuted"

    h1_ok = h1_b["lo"] > ZERO_TOL
    h2_ok = h2_b["lo"] > ZERO_TOL
    h3_plateau = abs(h3_b["mean"]) < PLATEAU_TOL
    h4_ok = h4_b["lo"] > ZERO_TOL

    common_ok = c1_b["lo"] > ZERO_TOL and c2_b["lo"] > ZERO_TOL and c4_b["lo"] > ZERO_TOL
    mel_ok = all(ms[k]["boot"]["mean"] > ZERO_TOL for k in ("H1_T16_minus_W8", "H2_W8R2_minus_W8", "H4_interaction"))
    robust_ok = all(cv[k]["robust"]["mean_without_top2_abs"] > ZERO_TOL for k in ("H1_T16_minus_W8", "H2_W8R2_minus_W8", "H4_interaction"))

    if h1_ok and h2_ok and h3_plateau and h4_ok and common_ok and mel_ok and robust_ok:
        return "confirmed"
    if h1_ok and h2_ok and h3_plateau:
        return "supported_direction_only"
    return "mixed"


def _fmt_ci(b, digits=4):
    return f"{b['mean']:+.{digits}f} [{b['lo']:+.{digits}f}, {b['hi']:+.{digits}f}]"


def _write_manifest(out_dir, paths, prereg_path):
    entries = {"preregistration": {"path": str(prereg_path), "sha256": _sha256_file(prereg_path)}}
    for p in paths:
        p = Path(p)
        if p.exists():
            entries[p.name] = {"sha256": _sha256_file(p), "bytes": p.stat().st_size}
    manifest = {"experiment": "EXP-027", "utc": _utc(), "files": entries}
    mpath = Path(out_dir) / "MANIFEST.json"
    mpath.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return mpath


def analyze(out_dir, allow_partial=False, prereg_path=PREREG_PATH):
    """Validate preregistration and CSV completeness, compute statistics, and write JSON/MD/MANIFEST."""
    check_prereg(prereg_path)
    out_dir = Path(out_dir)
    div_path = out_dir / "diversity.csv"
    qual_path = out_dir / "quality.csv"
    if not div_path.exists():
        raise RuntimeError(f"Missing diversity CSV: {div_path}")

    div_rows = rf.read_csv(div_path, repair=True)
    expected_utts = {it["utt_id"] for it in fresh_items()}
    seen_div = {r["utt_id"] for r in div_rows}
    if not allow_partial and seen_div != expected_utts:
        raise RuntimeError(f"Expected {len(expected_utts)} fresh speakers in {div_path.name}, found {len(seen_div)}.")
    if len(div_rows) < 2:
        raise RuntimeError(f"Need at least 2 speakers in {div_path.name} to analyze, found {len(div_rows)}.")

    qual_rows = rf.read_csv(qual_path, repair=True) if qual_path.exists() else []
    if not allow_partial:
        if not qual_path.exists():
            raise RuntimeError(f"Missing quality CSV: {qual_path}")
        seen_qual = {r["utt_id"] for r in qual_rows}
        if seen_qual != expected_utts:
            raise RuntimeError(f"Expected {len(expected_utts)} fresh speakers in {qual_path.name}, found {len(seen_qual)}.")

    by_metric = {}
    for prefix in ("cvf0", "cvf0_common", "mel_spread"):
        by_metric[prefix] = {name: _summarize_contrast(div_rows, prefix, name) for name in CONTRAST_FNS}

    arm_means = {
        prefix: {arm: boot_ci([float(r[f"{prefix}_{arm}"]) for r in div_rows if f"{prefix}_{arm}" in r]) for arm in ARMS}
        for prefix in ("cvf0", "cvf0_common", "mel_spread")
    }
    voiced_means = {arm: boot_ci([float(r[f"voiced_{arm}"]) for r in div_rows]) for arm in ARMS}
    voiced_common = boot_ci([float(r["voiced_common"]) for r in div_rows])

    # Exploratory H5: speaker-level Spearman between T16-W8 loss and W8R2-W8 recovery
    paired_h5 = [
        (float(r["cvf0_T16"]) - float(r["cvf0_W8"]), float(r["cvf0_W8R2"]) - float(r["cvf0_W8"]))
        for r in div_rows
        if not math.isnan(float(r["cvf0_T16"])) and not math.isnan(float(r["cvf0_W8"])) and not math.isnan(float(r["cvf0_W8R2"]))
    ]
    xs, ys = [a for a, _ in paired_h5], [b for _, b in paired_h5]
    h5_rho = tt.spearman(xs, ys) if len(paired_h5) >= 3 and len(set(xs)) > 1 and len(set(ys)) > 1 else float("nan")

    quality = {}
    if qual_rows:
        for arm in ARMS:
            quality[arm] = {m: boot_ci([float(r[f"{m}_{arm}"]) for r in qual_rows]) for m in METRICS}
            if arm != "W8":
                quality[arm].update({
                    f"d_{m}_vs_W8": boot_ci([float(r[f"{m}_{arm}"]) - float(r[f"{m}_W8"]) for r in qual_rows])
                    for m in METRICS
                })

    verdict = determine_verdict(by_metric)
    results = {
        "experiment": "EXP-027",
        "partial": allow_partial and (len(div_rows) < FRESH_N or len(qual_rows) < FRESH_N),
        "n_diversity": len(div_rows),
        "n_quality": len(qual_rows),
        "verdict": verdict,
        "arm_means": arm_means,
        "voiced_means": voiced_means,
        "voiced_common": voiced_common,
        "contrasts": by_metric,
        "h5_spearman_rho": h5_rho,
        "quality": quality,
        "utc": _utc(),
    }

    rjson = out_dir / "transmissibility_confirm_results.json"
    rmd = out_dir / "TRANSMISSIBILITY_CONFIRM_REPORT.md"
    rjson.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# EXP-027: Transmissibility Confirmation Report",
        "",
        f"- **Verdict:** `{verdict}` ({'PARTIAL' if results['partial'] else 'FULL'})",
        f"- **Fresh speakers scored:** diversity = {len(div_rows)}/{FRESH_N}, quality = {len(qual_rows)}/{FRESH_N}",
        "",
        "## Arm Diversity Means (95% bootstrap CI)",
        "",
        "| Arm | Pass cost | CVf0 | Voiced share | Common-frame CVf0 | Slow-mel spread |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for arm in ARMS:
        lines.append(
            f"| `{arm}` | {COST[arm]} | {arm_means['cvf0'][arm]['mean']:.4f} | {voiced_means[arm]['mean']:.3f} | "
            f"{arm_means['cvf0_common'][arm]['mean']:.4f} | {arm_means['mel_spread'][arm]['mean']:.4f} |"
        )
    lines.extend([
        "",
        f"Common voiced frame share across all 6 arms: `{voiced_common['mean']:.3f}`.",
        "",
        "## Confirmatory Hypotheses (CVf0, Common-frame CVf0, Slow-mel Spread)",
        "",
        "| Contrast | CVf0 (95% CI) | No-top2 mean | Sign (+/-, p) | Common-frame CVf0 (95% CI) | Slow-mel spread (95% CI) |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ])
    for key in ("H1_T16_minus_W8", "H2_W8R2_minus_W8", "H3_W8_minus_W4", "H4_interaction"):
        c_cv, c_cc, c_ms = by_metric["cvf0"][key], by_metric["cvf0_common"][key], by_metric["mel_spread"][key]
        lines.append(
            f"| `{key}` | {_fmt_ci(c_cv['boot'])} | {c_cv['robust']['mean_without_top2_abs']:+.4f} | "
            f"{c_cv['sign']['pos']}/{c_cv['sign']['neg']} (p={c_cv['sign']['p']:.3g}) | "
            f"{_fmt_ci(c_cc['boot'])} | {_fmt_ci(c_ms['boot'])} |"
        )
    lines.extend(["", f"- Exploratory `H5` Spearman rho (`T16 - W8` vs `W8R2 - W8`): `{h5_rho:+.3f}`"])

    if quality:
        lines.extend([
            "",
            "## Clean-Prompt Quality on Fresh Speakers (vs W8)",
            "",
            "| Arm | Passes | WER (vs W8) | SIM (vs W8) | UTMOS (vs W8) |",
            "| --- | ---: | ---: | ---: | ---: |",
        ])
        for arm in ARMS:
            q = quality[arm]
            w_s = f"{q['wer']['mean']:.4f}" if arm == "W8" else _fmt_ci(q["d_wer_vs_W8"], 4)
            s_s = f"{q['similarity']['mean']:.4f}" if arm == "W8" else _fmt_ci(q["d_similarity_vs_W8"], 4)
            u_s = f"{q['utmos']['mean']:.3f}" if arm == "W8" else _fmt_ci(q["d_utmos_vs_W8"], 3)
            lines.append(f"| `{arm}` | {COST[arm]} | {w_s} | {s_s} | {u_s} |")

    rmd.write_text("\n".join(lines) + "\n", encoding="utf-8")
    _write_manifest(out_dir, [div_path, qual_path, rjson, rmd], prereg_path)
    print(f"[OK] Analysis complete: verdict={verdict}, report={rmd}", flush=True)
    return results


def main():
    p = argparse.ArgumentParser(description="EXP-027 Transmissibility confirmation runner")
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
        items = fresh_items()
        if args.stage in ("both", "diversity"):
            run_diversity(pipe, items, args.out_dir)
        if args.stage in ("both", "quality"):
            run_quality(pipe, items, args.out_dir)
    elif args.cmd == "analyze":
        analyze(args.out_dir, allow_partial=args.allow_partial)


if __name__ == "__main__":
    main()
