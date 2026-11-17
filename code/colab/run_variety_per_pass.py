"""EXP-032 (EVAL-025): how much prosodic variety each teacher pass buys, measured against the DMOSpeech 2 paper's
t_switch = 0.25, on clean and noisy prompts. A study for designing the next handover: no handover method is tested.

DMOSpeech 2 runs its teacher early for diversity (Li et al., arXiv 2507.14988, Sections 3.3 and 4.3.3). The distilled
student's outputs vary less across seeds ("mode shrinkage"), and the teacher's early steps restore prosodic variety. The
teacher's passes are therefore the price of that variety. On the same cases this run measures:

  P25   the paper's evaluated setting, its Algorithm 2 as written. The teacher runs 14 guided steps on the 32-interval
        sway grid (t_14 = 0.2270 <= 0.25); the student continues from the teacher's state and steps again at 0.6173 (the
        paper does not state the second time; EXP-017's `alg2`). 30 passes. This is the 100% reference.
  P25R  the same 14 teacher steps with the released code's handover: the endpoint is re-noised at the student's levels
        0.2929 and 0.6173 (EXP-017's `paper`). 32 passes.
  W1, W2, W3, W4, W8   the released sampler on the 16-interval grid: the teacher runs to Step k, then the student's three
        levels (2k + 5 passes). W8 is the thesis's version of 0.25 (21 passes); W4 is the released code's default (0.07).
  STU   the student alone (4 passes).
  TCH   the teacher alone, 32 guided steps (64 passes); quality stage only.

- Quality (UTMOS, WER, speaker similarity): development speakers 1-30 x clean and noisy15 (60 cases).
- Prosodic variety (CVf0, aligned CVf0, timing spread): speakers 1-20 x 12 seeds x clean and noisy15.
- Tensor bank: the student's draft, the teacher's estimates and four arms' outputs (float16, generated frames only) for
  speakers 1-5, so that new handover designs can be screened offline before any GPU run.

Subcommands: freeze, smoke, run, analyze.
"""
import argparse
import hashlib
import json
import random
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
import run_method_comparison as mc  # noqa: E402
import run_noise_guidance_prelim as ng  # noqa: E402
import run_selective_handover_prelim as sh  # noqa: E402
import run_timing_only_handover as toh  # noqa: E402
import run_transmissibility_bridge as tb  # noqa: E402
import run_transmissibility_confirm as tc  # noqa: E402
import run_transmissibility_test as tt  # noqa: E402

THESIS_ROOT = rf.THESIS_ROOT
CODE_DIR = Path(__file__).resolve().parent
EXP_DIR = THESIS_ROOT / "research/experiments/EXP-032-variety-per-pass"
PREREG_PATH = EXP_DIR / "preregistration.json"
EXP017_CSV = THESIS_ROOT / "research/experiments/EXP-017-grid32-prelim/results/grid32.csv"
EXP031_DIR = THESIS_ROOT / "research/experiments/EXP-031-snr-toh-method-comparison/results"
EXP031_QUAL_CSV = EXP031_DIR / "quality.csv"
EXP031_DIV_CSV = EXP031_DIR / "diversity.csv"
EXP027_QUAL_CSV = tb.EXP027_QUAL_CSV

DEPENDENCIES = ("run_variety_per_pass.py", "run_endpoint_extrapolation_prelim.py") + tuple(mc.DEPENDENCIES)

QUAL_N = 30
DIV_N = 20
DIV_SEEDS = 12
DIV_SEED_BASE = mc.DIV_SEED_BASE      # 1000, as in EXP-029 to EXP-031
CONDS = ("clean", "noisy15")
SEED = 2026
BOOT_B = 2000

G16 = (1, 2, 3, 4, 8)                 # teacher steps on the 16-interval grid
K_PAPER = gl.K32[0.25]                # 14: the last 32-interval point at or below 0.25 (Algorithm 2)
TCH_K = gl.TEACHER_ONLY_K             # 31: the teacher's endpoint at t_31 is its Euler step to t = 1
REFERENCE, BASE = "P25", "STU"
W_ARMS = tuple(f"W{k}" for k in G16)
ARMS = ("STU",) + W_ARMS + ("P25", "P25R", "TCH")
DIV_ARMS = ("STU",) + W_ARMS + ("P25", "P25R")
COST = {"STU": 4, **{f"W{k}": 2 * k + 5 for k in G16}, "P25": gl.COST[gl.ALG2], "P25R": gl.COST[gl.PAPER],
        "TCH": gl.COST[gl.TEACHER]}
EXP017_NAME = {"STU": gl.STUDENT, "W4": "g16_4", "W8": "g16_8", "P25": gl.ALG2, "P25R": gl.PAPER, "TCH": gl.TEACHER}

METRICS = ("wer", "similarity", "utmos")
VARIETY = ("cvf0", "cvf0_aligned")
DIV_MEASURES = mc.DIV_MEASURES       # cvf0, voiced, mel_spread, cvf0_aligned, voiced_aligned, timing_spread

# Bars: EXP-029's clean quality bars; a step "keeps the paper's variety" at 90% of P25's gain over the student (the
# paper's teacher-guided setting kept 89.1% of its teacher's CVf0).
SHARE_BAR = 0.90
CLEAN_UTMOS_BAR = -0.08
WER_BAR = 0.01
SIM_BAR = -0.01
RUNAWAY_WER = 1.0
ZERO_TOL = 1e-12
SPLIT_TOL = 1e-5
GRID_TOL = 1e-4
REPRO_TOL = 0.01

BANK_N = 5
BANK_STEPS = (1, 2, 3, 4, 8)
BANK_OUTS = ("STU", "W4", "W8", "P25")

QUAL_FIELDS = (["utt_id", "speaker_id", "condition", "prompt_snr", "gpu", "frames", "seq_len"]
               + [f"d_{j}" for j in range(1, 9)]
               + [f"{m}_{a}" for a in ARMS for m in METRICS + ("hypothesis",)]
               + ["case_seconds", "utc"])
DIV_FIELDS = (["utt_id", "speaker_id", "condition", "n_seeds"]
              + [f"{m}_{a}" for a in DIV_ARMS for m in DIV_MEASURES]
              + ["case_seconds", "utc"])


def qual_items(n=QUAL_N):
    """Development speakers 1-30: EXP-031's speakers and seeds (and EXP-017's), so both give reproduction checks."""
    return cg.dev_items()[:n]


def div_items(n=DIV_N):
    """Development speakers 1-20, with seeds input_seed + 1000 + s as in EXP-029 to EXP-031."""
    return cg.dev_items()[:n]


# --------------------------------------------------------------------------- synthesis
def synthesize(pipe, ctx, arms):
    """All requested arms from one 16-interval teacher run, one 32-interval teacher run and one student draft. Every
    arm uses the released student's per-level seeds. Returns (outputs, parts)."""
    taus = list(pipe.sampler.STUDENT_TAUS)
    ends, tel = dts.teacher_trajectory(pipe, ctx, wanted=G16)
    want32 = {K_PAPER} | ({TCH_K} if "TCH" in arms else set())
    e32, _, s32 = gl.grid_teacher(pipe, ctx, gl.GRID32, want32)
    s0 = gl.student(pipe, ctx, ctx["step_cond"], [0.0])
    outs = {"STU": gl.student(pipe, ctx, s0, taus)}
    for k in G16:
        outs[f"W{k}"] = gl.student(pipe, ctx, ends[k], taus)
    outs["P25"] = gl.student_from_state(pipe, ctx, s32[K_PAPER], list(gl.ALG2_TIMES))
    outs["P25R"] = gl.student(pipe, ctx, e32[K_PAPER], taus[1:])
    if "TCH" in arms:
        outs["TCH"] = e32[TCH_K]
    parts = {"ends": ends, "e32": e32, "s32": s32, "s0": s0, "tel": tel}
    return {a: outs[a] for a in arms}, parts


def bank_save(path, ctx, parts, outs, tracks):
    """One seed's tensors for offline screening, generated frames only, float16 (f0 tracks float32)."""
    plen = ctx["prompt_len"]

    def gen(x):
        return toh._as_numpy(x[0, plen:, :]).astype(np.float16)

    data = {"prompt_len": np.int64(plen), "prompt": toh._as_numpy(ctx["cond"][0, :plen, :]).astype(np.float16),
            "s0": gen(parts["s0"]), "T14_32": gen(parts["e32"][K_PAPER])}
    for k in BANK_STEPS:
        data[f"T{k}"] = gen(parts["ends"][k])
    for a in BANK_OUTS:
        data[f"out_{a}"] = gen(outs[a])
    for a, tr in tracks.items():
        data[f"f0_{a}"] = np.asarray(tr, dtype=np.float32)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **data)
    return path


def bank_path(out_dir, cond, utt_id, s):
    return Path(out_dir) / "bank" / cond / f"{utt_id}__s{s:02d}.npz"


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
        outs, parts = synthesize(pipe, ctx, ARMS)
        gt = pipe.ground_truth(item)
        s0 = parts["s0"]
        row = {"utt_id": item["utt_id"], "speaker_id": item["speaker_id"], "condition": cond,
               "prompt_snr": float(ctx["prompt_snr"]), "gpu": pipe.gpu, "frames": int(s0.shape[1] - ctx["prompt_len"]),
               "seq_len": int(s0.shape[1]), **{f"d_{j}": float(parts["tel"][f"d_{j}"]) for j in range(1, 9)}}
        for arm in ARMS:
            m = pipe.score(ng.decode(pipe, ctx, outs[arm]), item, gt)
            row.update({f"{k}_{arm}": m[k] for k in METRICS + ("hypothesis",)})
        row.update({"case_seconds": round(time.perf_counter() - t0, 2), "utc": tb._utc()})
        tb._append(path, QUAL_FIELDS, row)
        print(f"[{n_run}/{len(todo)}] quality: {item['utt_id']} {cond} ({row['case_seconds']:.1f} s)", flush=True)
    print(f"[OK] quality complete: {path}", flush=True)
    return path


def run_diversity(pipe, items, out_dir, seeds=DIV_SEEDS, conds=CONDS, csv_name="diversity.csv", bank_n=BANK_N):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / csv_name
    done = rf.done_keys(path, 1)
    bank_ids = {it["utt_id"] for it in items[:bank_n]}
    todo = [(it, c) for c in conds for it in items if (it["utt_id"], c) not in done]
    print(f"[diversity] {len(todo)} speaker-conditions x {seeds} seeds ({len(done)} already in {path.name})", flush=True)
    for n_run, (item, cond) in enumerate(todo, 1):
        t0 = time.perf_counter()
        tracks = {a: [] for a in DIV_ARMS}
        mels = {a: [] for a in DIV_ARMS}
        slow = {a: [] for a in DIV_ARMS}
        for s in range(seeds):
            seed = int(item["input_seed"]) + DIV_SEED_BASE + s
            ctx = pipe.prepare(item, cond, seed, out_dir, noise_seed=item["input_seed"])
            outs, parts = synthesize(pipe, ctx, DIV_ARMS)
            arm_tracks = dts._track_arms_fast(pipe, ctx, outs, DIV_ARMS)
            plen = ctx["prompt_len"]
            for arm in DIV_ARMS:
                tracks[arm].append(arm_tracks[arm])
                mels[arm].append(toh._as_numpy(outs[arm][0, plen:, :]))
                slow[arm].append(tc._slow_component(outs[arm], plen))
            if item["utt_id"] in bank_ids:
                bank_save(bank_path(out_dir, cond, item["utt_id"], s), ctx, parts, outs, arm_tracks)
        row = {"utt_id": item["utt_id"], "speaker_id": item["speaker_id"], "condition": cond, "n_seeds": seeds}
        for arm in DIV_ARMS:
            row[f"cvf0_{arm}"], row[f"voiced_{arm}"] = sh.cvf0(tracks[arm])
            row[f"mel_spread_{arm}"] = tc.slow_mel_spread(slow[arm])
            (row[f"cvf0_aligned_{arm}"], row[f"voiced_aligned_{arm}"],
             row[f"timing_spread_{arm}"]) = mc.aligned_cvf0(tracks[arm], mels[arm])
        row.update({"case_seconds": round(time.perf_counter() - t0, 2), "utc": tb._utc()})
        tb._append(path, DIV_FIELDS, row)
        print(f"[{n_run}/{len(todo)}] diversity: {item['utt_id']} {cond} ({row['case_seconds']:.1f} s)", flush=True)
    print(f"[OK] diversity complete: {path}", flush=True)
    return path


def bank_check(path):
    """Load one bank file and check its keys and shapes."""
    with np.load(path) as z:
        keys = set(z.files)
        need = {"prompt_len", "prompt", "s0", "T14_32"} | {f"T{k}" for k in BANK_STEPS} | {f"out_{a}" for a in BANK_OUTS}
        shapes = {k: list(z[k].shape) for k in sorted(need & keys) if k != "prompt_len"}
        frames = {v[0] for k, v in shapes.items() if k != "prompt"}
        ok = need <= keys and len(frames) == 1 and all(v[1:] == [100] for v in shapes.values())
        return {"path": str(path), "ok": bool(ok), "missing": sorted(need - keys), "shapes": shapes}


# --------------------------------------------------------------------------- smoke test
def _rows_by_key(path):
    return {(r["utt_id"], r["condition"]): r for r in rf.read_csv(path)} if Path(path).exists() else {}


def quality_reproduction(rows, stored, names, conds=CONDS):
    """Largest and median absolute differences of each arm's scores from stored rows of an earlier run."""
    out = {}
    for cond in conds:
        shared = [r for r in rows if r["condition"] == cond and (r["utt_id"], cond) in stored]
        if not shared:
            continue
        e = {"cases": len(shared), "max_abs_prompt_snr": max(
            abs(float(r["prompt_snr"]) - float(stored[(r["utt_id"], cond)]["prompt_snr"])) for r in shared)}
        for arm, old in names.items():
            for m in METRICS:
                key = f"{m}_{old}"
                if key not in stored[(shared[0]["utt_id"], cond)]:
                    continue
                d = [abs(float(r[f"{m}_{arm}"]) - float(stored[(r["utt_id"], cond)][key])) for r in shared]
                e[f"max_abs_{m}_{arm}"] = max(d)
                e[f"median_abs_{m}_{arm}"] = statistics.median(d)
        out[cond] = e
    return out


def smoke(pipe, out_dir, prereg_path=PREREG_PATH):
    """Exact identities (split student; P25's student repeated), the 32-interval teacher function against the 16-interval
    trajectory on the same grid, EXP-027's W4 reproduction, then micro runs: quality on development speaker 1 under both
    conditions (checked against EXP-031's stored rows and compared with EXP-017's) and diversity on one speaker with two
    seeds, including one tensor-bank file."""
    check_prereg(prereg_path)
    sdir = Path(out_dir) / "smoke"
    sdir.mkdir(parents=True, exist_ok=True)
    stored027 = {r["utt_id"]: r for r in rf.read_csv(EXP027_QUAL_CSV)}
    taus = list(pipe.sampler.STUDENT_TAUS)
    checks = []
    for item in tb.smoke_items():
        ctx = pipe.prepare(item, "clean", int(item["input_seed"]), sdir)
        outs, parts = synthesize(pipe, ctx, ARMS)
        full = gl.student(pipe, ctx, ctx["step_cond"], [0.0] + taus)
        p25_again = gl.student_from_state(pipe, ctx, parts["s32"][K_PAPER], list(gl.ALG2_TIMES))
        e16, _, _ = gl.grid_teacher(pipe, ctx, tt.grid16(pipe), set(G16))
        finite = all(bool(np.all(np.isfinite(toh._as_numpy(outs[a])))) for a in ARMS)
        rec = {"utt_id": item["utt_id"], "split_diff": tb._max_abs(outs["STU"] - full),
               "p25_repeat_diff": tb._max_abs(p25_again - outs["P25"]),
               "grid_teacher_vs_trajectory": max(tb._max_abs(e16[k] - parts["ends"][k]) for k in G16),
               "finite": finite, "prompt_snr": float(ctx["prompt_snr"]), "seq_len": int(parts["s0"].shape[1])}
        mw = pipe.score(ng.decode(pipe, ctx, outs["W4"]), item, pipe.ground_truth(item))
        ref = stored027[item["utt_id"]]
        rec["d_wer_W4"] = abs(mw["wer"] - float(ref["wer_W4"]))
        rec["d_sim_W4"] = abs(mw["similarity"] - float(ref["similarity_W4"]))
        print(f"[smoke] {json.dumps(rec)}", flush=True)
        checks.append(rec)
        if max(rec["split_diff"], rec["p25_repeat_diff"]) > SPLIT_TOL or rec["grid_teacher_vs_trajectory"] > GRID_TOL:
            raise RuntimeError(f"Identity check failed: {rec}")
        if not finite:
            raise RuntimeError(f"Non-finite output: {rec}")
        if rec["d_wer_W4"] > REPRO_TOL or rec["d_sim_W4"] > REPRO_TOL:
            raise RuntimeError(f"EXP-027 W4 reproduction failed: {rec}")
    q = run_quality(pipe, qual_items()[:1], conds=CONDS, out_dir=sdir, csv_name="quality.csv")
    qrows = rf.read_csv(q)
    repro031 = quality_reproduction(qrows, _rows_by_key(EXP031_QUAL_CSV), {"STU": "STU", "W4": "W4", "W8": "W8"})
    repro017 = quality_reproduction(qrows, _rows_by_key(EXP017_CSV), EXP017_NAME)
    worst031 = max((v for e in repro031.values() for k, v in e.items() if k.startswith("max_abs_")), default=None)
    print(f"[smoke] EXP-031 reproduction: {json.dumps(repro031)}", flush=True)
    print(f"[smoke] EXP-017 comparison (recorded, not a gate): {json.dumps(repro017)}", flush=True)
    if worst031 is None or worst031 > REPRO_TOL:
        raise RuntimeError(f"EXP-031 reproduction failed (largest difference {worst031}): {repro031}")
    run_diversity(pipe, tb.smoke_items()[:1], sdir, seeds=2, conds=CONDS, csv_name="diversity.csv", bank_n=1)
    bank = bank_check(bank_path(sdir, "noisy15", tb.smoke_items()[0]["utt_id"], 1))
    if not bank["ok"]:
        raise RuntimeError(f"Tensor bank check failed: {bank}")
    summary = {"checks": checks, "exp031_quality_reproduction": repro031, "exp017_quality_comparison": repro017,
               "bank_check": bank, "utc": tb._utc()}
    (sdir / "smoke_check.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print("[OK] smoke passed", flush=True)
    return summary


# --------------------------------------------------------------------------- preregistration
ESTIMANDS = {
    "variety_share": "For each arm A and condition c, with V = CVf0 or aligned CVf0 averaged over the 20 diversity "
                     "speakers: share_A = (V_A - V_STU) / (V_P25 - V_STU), the part of the paper setting's variety gain "
                     "over the student alone that A keeps; 95% speaker-level bootstrap interval of the ratio of means.",
    "variety_90": "L_A = V_A - 0.9 V_P25 - 0.1 V_STU per speaker; a 95% bootstrap lower bound above 0 confirms that A "
                  "keeps at least 90% of P25's gain.",
    "quality_clean": "Clean prompts, A - P25 per case: mean dUTMOS >= -0.08, dWER upper bound <= +0.01 and dSIM lower "
                     "bound >= -0.01 (EXP-029's clean bars).",
    "quality_noisy": "noisy15 prompts, A - STU per case under the same bars: whether a teacher step keeps the student "
                     "alone's quality when the prompt is noisy.",
    "variety_noise_change": "Per arm and speaker, V(noisy15) - V(clean): whether the variety survives a noisy prompt.",
    "mechanism": "P25 - P25R in both conditions (variety and quality): continuing from the teacher's state against "
                 "re-noising its endpoint.",
    "teacher_alone": "Clean prompts, TCH - P25 quality (the paper reports better WER and similarity for its "
                     "teacher-guided setting than for its teacher).",
}
LABEL_RULES = {
    "cheapest_step_point": "Clean prompts: the smallest k in (1, 2, 3, 4, 8) whose W_k keeps >= 90% of P25's gain on "
                           "both CVf0 and aligned CVf0 (point estimates) and passes the clean quality bars against "
                           "P25; 'none' if no k does.",
    "cheapest_step_confirmed": "The same with the 90% condition confirmed (L_A lower bound > 0 on both measures).",
    "w8_matches_paper": "W8 (the thesis's 0.25) meets the cheapest_step_point conditions.",
    "noisy_cheapest_step_point": "noisy15 prompts: the smallest k whose W_k keeps >= 90% of P25's gain on both "
                                 "measures (point estimates); variety only.",
    "noisy_quality_safe_arms": "noisy15 prompts: the teacher arms (W1-W8, P25, P25R) that pass the quality bars "
                               "against the student alone.",
    "partial": "Every label is 'partial' if either stage is incomplete.",
}


def build_prereg():
    q = [{"utt_id": it["utt_id"], "speaker_id": it["speaker_id"]} for it in qual_items()]
    d = [{"utt_id": it["utt_id"], "speaker_id": it["speaker_id"]} for it in div_items()]
    inputs = (EXP017_CSV, EXP031_QUAL_CSV, EXP031_DIV_CSV, EXP027_QUAL_CSV)
    payload = {
        "experiment": "EXP-032",
        "task_id": "EVAL-025",
        "title": "Variety per teacher pass against the DMOSpeech 2 paper's t_switch = 0.25, on clean and noisy prompts",
        "framing": ("Development study, no method tested. DMOSpeech 2 uses its teacher to restore the prosodic variety "
                    "the distilled student loses; this run measures how much of the paper setting's variety and quality "
                    "each amount of teacher work keeps, on clean and noisy15 prompts, to set the target for the next "
                    "handover design. The reference is the paper's evaluated setting (t_switch = 0.25, 14 teacher steps "
                    "on the 32-interval grid, Algorithm 2), not the released code's default."),
        "reference": {"arm": REFERENCE, "k_paper": K_PAPER, "grid": "32-interval sway grid",
                      "alg2_times": list(gl.ALG2_TIMES),
                      "note": "The paper does not state the student's second time; 0.6173 (the released student's last "
                              "level) is assumed, as in EXP-017."},
        "arms": list(ARMS),
        "diversity_arms": list(DIV_ARMS),
        "grid16_steps": list(G16),
        "costs": dict(COST),
        "conditions": list(CONDS),
        "n_qual_speakers": len(q),
        "n_qual_cases": len(q) * len(CONDS),
        "qual_utts": q,
        "n_div_speakers": len(d),
        "div_seeds": DIV_SEEDS,
        "div_seed_base": DIV_SEED_BASE,
        "div_utts": d,
        "variety_measures": list(VARIETY),
        "bank": {"n_speakers": BANK_N, "teacher_steps": list(BANK_STEPS), "outputs": list(BANK_OUTS),
                 "contents": "prompt mel, student draft S0, teacher estimates T_k (16-interval) and T_14 (32-interval), "
                             "outputs, f0 tracks; generated frames only, float16; stored with SHA-256 in bank_index.json"},
        "bars": {"share": SHARE_BAR, "clean_utmos": CLEAN_UTMOS_BAR, "wer": WER_BAR, "sim": SIM_BAR,
                 "runaway_wer": RUNAWAY_WER},
        "estimands": ESTIMANDS,
        "label_rules": LABEL_RULES,
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


def share_ci(rows, arm, measure, ref=REFERENCE, base=BASE):
    """Share of ref's gain over base kept by arm: ratio of speaker means, with a speaker-level percentile bootstrap.
    Resamples whose gain is not positive are counted and left out."""
    a = [_f(r, f"{measure}_{arm}") for r in rows]
    p = [_f(r, f"{measure}_{ref}") for r in rows]
    s = [_f(r, f"{measure}_{base}") for r in rows]
    n = len(rows)

    def ratio(idx):
        den = sum(p[i] - s[i] for i in idx) / len(idx)
        return (sum(a[i] - s[i] for i in idx) / len(idx)) / den if den > ZERO_TOL else None

    if n < 2:
        return {"n": n, "point": None, "lo": None, "hi": None, "undefined_resamples": None}
    rng = random.Random(SEED)
    vals, undefined = [], 0
    for _ in range(BOOT_B):
        v = ratio([rng.randrange(n) for _ in range(n)])
        if v is None:
            undefined += 1
        else:
            vals.append(v)
    vals.sort()
    lo = vals[int(0.025 * len(vals))] if vals else None
    hi = vals[min(len(vals) - 1, int(0.975 * len(vals)))] if vals else None
    return {"n": n, "point": ratio(range(n)), "lo": lo, "hi": hi, "undefined_resamples": undefined}


def variety_curve(div_rows):
    out = {}
    for cond in CONDS:
        rows = [r for r in div_rows if r["condition"] == cond]
        if len(rows) < 2:
            continue
        e = {"n": len(rows), "means": {a: {m: statistics.mean(_f(r, f"{m}_{a}") for r in rows) for m in DIV_MEASURES}
                                       for a in DIV_ARMS}}
        for m in VARIETY:
            e[m] = {"share": {a: share_ci(rows, a, m) for a in DIV_ARMS if a not in (REFERENCE, BASE)},
                    "L90": {a: _boot(_f(r, f"{m}_{a}") - SHARE_BAR * _f(r, f"{m}_{REFERENCE}")
                                     - (1 - SHARE_BAR) * _f(r, f"{m}_{BASE}") for r in rows)
                            for a in DIV_ARMS if a not in (REFERENCE, BASE)},
                    "minus_P25": {a: _boot(_f(r, f"{m}_{a}") - _f(r, f"{m}_{REFERENCE}") for r in rows)
                                  for a in DIV_ARMS if a != REFERENCE}}
        e["timing_spread_minus_P25"] = {a: _boot(_f(r, f"timing_spread_{a}") - _f(r, f"timing_spread_{REFERENCE}")
                                                 for r in rows) for a in DIV_ARMS if a != REFERENCE}
        out[cond] = e
    return out


def variety_noise_change(div_rows):
    by = {(r["utt_id"], r["condition"]): r for r in div_rows}
    ids = sorted({u for u, c in by if c == "clean" and (u, "noisy15") in by})
    if len(ids) < 2:
        return None
    return {a: {m: _boot(_f(by[(u, "noisy15")], f"{m}_{a}") - _f(by[(u, "clean")], f"{m}_{a}") for u in ids)
                for m in VARIETY + ("timing_spread",)} for a in DIV_ARMS}


def qdiff(rows, a, b):
    d = {m: _boot(_f(r, f"{m}_{a}") - _f(r, f"{m}_{b}") for r in rows) for m in METRICS}
    d["ok"] = bool(d["utmos"]["mean"] >= CLEAN_UTMOS_BAR and d["wer"]["hi"] <= WER_BAR
                   and d["similarity"]["lo"] >= SIM_BAR)
    return d


def quality_tables(qual_rows):
    out = {}
    for cond in CONDS:
        rows = [r for r in qual_rows if r["condition"] == cond]
        if len(rows) < 2:
            continue
        tab = {}
        for a in ARMS:
            wer = [_f(r, f"wer_{a}") for r in rows]
            tab[a] = {"passes": COST[a], "utmos": statistics.mean(_f(r, f"utmos_{a}") for r in rows),
                      "similarity": statistics.mean(_f(r, f"similarity_{a}") for r in rows),
                      "wer_mean": statistics.mean(wer), "wer_median": statistics.median(wer),
                      "runaway": sum(w > RUNAWAY_WER for w in wer)}
        out[cond] = {"n": len(rows), "arms": tab,
                     "vs_P25": {a: qdiff(rows, a, REFERENCE) for a in ARMS if a != REFERENCE},
                     "vs_STU": {a: qdiff(rows, a, BASE) for a in ARMS if a != BASE},
                     "prompt_snr_median": statistics.median(_f(r, "prompt_snr") for r in rows)}
    return out


def mechanism(qual_rows, div_rows):
    out = {}
    for cond in CONDS:
        q = [r for r in qual_rows if r["condition"] == cond]
        d = [r for r in div_rows if r["condition"] == cond]
        e = {}
        if len(q) >= 2:
            e["quality"] = {m: _boot(_f(r, f"{m}_P25") - _f(r, f"{m}_P25R") for r in q) for m in METRICS}
        if len(d) >= 2:
            e["variety"] = {m: _boot(_f(r, f"{m}_P25") - _f(r, f"{m}_P25R") for r in d)
                            for m in VARIETY + ("timing_spread",)}
        out[cond] = e
    return out


def _share_ok(curve, cond, arm, confirmed=False):
    for m in VARIETY:
        if confirmed:
            if curve[cond][m]["L90"][arm]["lo"] <= ZERO_TOL:
                return False
        else:
            pt = curve[cond][m]["share"][arm]["point"]
            if pt is None or pt < SHARE_BAR:
                return False
    return True


def labels(curve, quality, complete):
    keys = ("cheapest_step_point", "cheapest_step_confirmed", "w8_matches_paper", "noisy_cheapest_step_point",
            "noisy_quality_safe_arms")
    if not complete:
        return {k: "partial" for k in keys}

    def clean_ok(k, confirmed):
        return _share_ok(curve, "clean", f"W{k}", confirmed) and quality["clean"]["vs_P25"][f"W{k}"]["ok"]

    first = lambda ok: next((f"W{k}" for k in G16 if ok(k)), "none")  # noqa: E731
    return {"cheapest_step_point": first(lambda k: clean_ok(k, False)),
            "cheapest_step_confirmed": first(lambda k: clean_ok(k, True)),
            "w8_matches_paper": bool(clean_ok(8, False)),
            "noisy_cheapest_step_point": first(lambda k: _share_ok(curve, "noisy15", f"W{k}")),
            "noisy_quality_safe_arms": [a for a in W_ARMS + ("P25", "P25R") if quality["noisy15"]["vs_STU"][a]["ok"]]}


def diversity_reproduction(div_rows):
    """Clean variety of STU, W4 and W8 against EXP-031's stored rows (same speakers and seeds)."""
    stored = _rows_by_key(EXP031_DIV_CSV)
    shared = [r for r in div_rows if r["condition"] == "clean" and (r["utt_id"], "clean") in stored]
    if not shared:
        return None
    out = {"speakers": len(shared)}
    for a in ("STU", "W4", "W8"):
        for m in ("cvf0", "cvf0_aligned", "timing_spread"):
            out[f"max_abs_{m}_{a}"] = max(abs(_f(r, f"{m}_{a}") - _f(stored[(r["utt_id"], "clean")], f"{m}_{a}"))
                                          for r in shared)
    return out


def bank_index(out_dir):
    root = Path(out_dir) / "bank"
    files = sorted(root.rglob("*.npz")) if root.exists() else []
    entries = []
    for p in files:
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        entries.append({"path": p.relative_to(out_dir).as_posix(), "bytes": p.stat().st_size, "sha256": h.hexdigest()})
    return {"files": len(entries), "bytes": sum(e["bytes"] for e in entries), "entries": entries}


def analyze(out_dir=None, prereg_path=PREREG_PATH, allow_partial=False):
    out_dir = Path(out_dir or EXP_DIR / "results")
    q_path, d_path = out_dir / "quality.csv", out_dir / "diversity.csv"
    if not q_path.exists():
        raise FileNotFoundError(f"Missing quality CSV: {q_path}")
    qual_rows = rf.read_csv(q_path, repair=True)
    div_rows = rf.read_csv(d_path, repair=True) if d_path.exists() else []
    complete = len(qual_rows) >= QUAL_N * len(CONDS) and len(div_rows) >= DIV_N * len(CONDS)
    if not allow_partial and not complete:
        raise RuntimeError(f"Incomplete: quality {len(qual_rows)}/{QUAL_N * len(CONDS)}, "
                           f"diversity {len(div_rows)}/{DIV_N * len(CONDS)}")
    curve = variety_curve(div_rows)
    quality = quality_tables(qual_rows)
    results = {
        "experiment": "EXP-032",
        "partial": not complete,
        "labels": labels(curve, quality, complete),
        "n_quality_rows": len(qual_rows),
        "n_diversity_rows": len(div_rows),
        "costs": dict(COST),
        "variety_curve": curve,
        "quality": quality,
        "variety_noise_change": variety_noise_change(div_rows),
        "mechanism_P25_minus_P25R": mechanism(qual_rows, div_rows),
        "exp031_quality_reproduction": quality_reproduction(qual_rows, _rows_by_key(EXP031_QUAL_CSV),
                                                            {"STU": "STU", "W4": "W4", "W8": "W8"}),
        "exp031_diversity_reproduction": diversity_reproduction(div_rows),
        "exp017_quality_comparison": quality_reproduction(qual_rows, _rows_by_key(EXP017_CSV), EXP017_NAME),
        "utc": tb._utc(),
    }
    index = bank_index(out_dir)
    with open(out_dir / "bank_index.json", "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(index, indent=2) + "\n")
    rjson = out_dir / "variety_per_pass_results.json"
    rmd = out_dir / "VARIETY_PER_PASS_REPORT.md"
    with open(rjson, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(results, indent=2) + "\n")
    with open(rmd, "w", encoding="utf-8", newline="\n") as f:
        f.write(report(results, index) + "\n")
    entries = {}
    if Path(prereg_path).exists():
        entries["preregistration"] = {"path": str(prereg_path), "sha256": tb._sha256_file(prereg_path)}
    for p in (q_path, d_path, rjson, rmd, out_dir / "bank_index.json"):
        if p.exists():
            entries[p.name] = {"sha256": tb._sha256_file(p), "bytes": p.stat().st_size}
    with open(out_dir / "MANIFEST.json", "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps({"experiment": "EXP-032", "utc": tb._utc(), "files": entries}, indent=2) + "\n")
    print(f"[OK] Analysis complete: labels={results['labels']}, report={rmd}", flush=True)
    return results


def _ci(b, d=3):
    return tc._fmt_ci(b, d) if b else "n/a"


def _share(s):
    if not s or s.get("point") is None:
        return "n/a"
    lo = "n/a" if s["lo"] is None else f"{s['lo']:.0%}"
    hi = "n/a" if s["hi"] is None else f"{s['hi']:.0%}"
    return f"{s['point']:.0%} [{lo}, {hi}]"


def report(res, index=None):
    lines = ["# EXP-032: variety per teacher pass against the paper's t_switch = 0.25", "",
             f"Rows: quality {res['n_quality_rows']} / {QUAL_N * len(CONDS)}, diversity {res['n_diversity_rows']} / "
             f"{DIV_N * len(CONDS)} (partial={res['partial']}). Reference: `P25`, the paper's Algorithm 2 at 0.25 "
             f"({COST['P25']} passes).", "", "## Labels (frozen rules)", "", "```",
             json.dumps(res["labels"], indent=2), "```"]
    for cond, e in res["variety_curve"].items():
        lines += ["", f"## Variety, `{cond}` (n={e['n']} speakers x {DIV_SEEDS} seeds)", "",
                  "| Arm | Passes | CVf0 | Aligned CVf0 | Timing spread | Share of P25's gain: CVf0 | Aligned CVf0 | "
                  "90% confirmed (both) |", "| --- | ---: | ---: | ---: | ---: | --- | --- | --- |"]
        for a in DIV_ARMS:
            mn = e["means"][a]
            if a in (REFERENCE, BASE):
                sh_raw = sh_al = "100%" if a == REFERENCE else "0%"
                conf = ""
            else:
                sh_raw, sh_al = _share(e["cvf0"]["share"][a]), _share(e["cvf0_aligned"]["share"][a])
                conf = "yes" if all(e[m]["L90"][a]["lo"] > ZERO_TOL for m in VARIETY) else "no"
            lines.append(f"| `{a}` | {COST[a]} | {mn['cvf0']:.4f} | {mn['cvf0_aligned']:.4f} | "
                         f"{mn['timing_spread']:.2f} | {sh_raw} | {sh_al} | {conf} |")
    for cond, e in res["quality"].items():
        ref, label = ("vs_P25", "P25") if cond == "clean" else ("vs_STU", "STU")
        lines += ["", f"## Quality, `{cond}` (n={e['n']}; prompt SNR median {e['prompt_snr_median']:.1f} dB)", "",
                  f"| Arm | Passes | UTMOS | WER mean / median | Runaway | SIM | dUTMOS vs {label} | dWER hi / dSIM lo | "
                  "Bars |", "| --- | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |"]
        for a in ARMS:
            t = e["arms"][a]
            d = e[ref].get(a)
            cmp_ = (f"{_ci(d['utmos'])} | {d['wer']['hi']:+.4f} / {d['similarity']['lo']:+.4f} | "
                    f"{'pass' if d['ok'] else 'fail'}") if d else "(reference) | | "
            lines.append(f"| `{a}` | {t['passes']} | {t['utmos']:.3f} | {t['wer_mean']:.4f} / {t['wer_median']:.4f} | "
                         f"{t['runaway']} | {t['similarity']:.4f} | {cmp_} |")
    if res["variety_noise_change"]:
        lines += ["", "## Variety change from clean to noisy15 (same speakers)", "",
                  "| Arm | CVf0 | Aligned CVf0 | Timing spread |", "| --- | --- | --- | --- |"]
        for a, e in res["variety_noise_change"].items():
            lines.append(f"| `{a}` | {_ci(e['cvf0'], 4)} | {_ci(e['cvf0_aligned'], 4)} | {_ci(e['timing_spread'], 2)} |")
    lines += ["", "## Handover mechanism: P25 (continue from state) - P25R (re-noise endpoint)", "", "```",
              json.dumps(res["mechanism_P25_minus_P25R"], indent=2), "```",
              "", "## Reproductions", "", "EXP-031 quality (STU, W4, W8):", "", "```",
              json.dumps(res["exp031_quality_reproduction"], indent=2), "```", "", "EXP-031 clean variety:", "", "```",
              json.dumps(res["exp031_diversity_reproduction"], indent=2), "```", "",
              "EXP-017 quality (STU = student4, W4 = g16_4, W8 = g16_8, P25 = alg2, P25R = paper, TCH = teacher32):",
              "", "```", json.dumps(res["exp017_quality_comparison"], indent=2), "```"]
    if index is not None:
        lines += ["", f"Tensor bank: {index['files']} files, {index['bytes'] / 2 ** 20:.1f} MiB (hashes in "
                      "bank_index.json; the bank is not in the results zip)."]
    return "\n".join(lines)


def make_pipe(args):
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    return rf.Pipeline(args.dmo_root, args.device)


def main():
    p = argparse.ArgumentParser(description="EXP-032 variety per teacher pass against the paper's t_switch = 0.25")
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
