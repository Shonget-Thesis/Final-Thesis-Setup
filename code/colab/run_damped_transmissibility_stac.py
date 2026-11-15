"""EXP-029 (EVAL-022): Damped Slow-Band Transmissibility Bridge & Mixed-Condition STAC Diagnostic.

Resolves the two structural gaps revealed by EXP-028 (`TBRG` / `STAC-TBRG`) within the thesis's
`STAC (EXP-018) -> Transmissibility Theory (EXP-026/027) -> Upgraded STAC` research arc:

1. Clean Pareto Calibration across Energy-Preserving Slow-Band Damping (`TB_alpha`) and Step 5/6:
     X_TB(alpha) = v_S + s_hat(alpha) + f_S   on generated frames [plen:] (13 NFE)
   where s_tilde_T = (||s_S||_F / (||s_T||_F + eps)) * s_T and s_hat(alpha) interpolates
   (1 - alpha) * s_S + alpha * s_tilde_T renormalized to ||s_S||_F. This sweeps smoothly between
   exact identity W4 (alpha = 0) and EXP-028 TBRG (alpha = 1) at fixed 13 NFE, tested alongside
   13-NFE BLND, 14-NFE W5R2, 16-NFE W6R2, 20-NFE W8R2, and 21-NFE W8.

2. Degraded-Prompt Behavior & Upgraded Clean-Noisy STAC Routing:
   Neither EXP-026, EXP-027, nor EXP-028 evaluated tau_2 restarts (W5R2, W6R2, W8R2) or slow-band
   bridges on degraded prompts, even though EXP-018 STAC relies on mixed clean + noisy regimes.
   EXP-029 runs a paired 30-speaker development cohort (speakers 61-90, leaving all 170 unused
   calibration speakers and all 200 test speakers untouched) across clean, gauss18, and noisy15
   (90 cases), plus clean multi-seed pitch diversity (20 speakers x 12 seeds), logging zero-extra-NFE
   STAC (`A_k`, `gc_0`, `A_k / gc_0`) and post-S1 bridge discrepancy features (`slow_gap_4`,
   `voice_gap_4`, `fine_gap_4`) for 5-fold speaker-grouped CV adaptive routing.

Development diagnostic evidence only. Subcommands: freeze, smoke, run, analyze.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import json
import math
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
import run_transmissibility_bridge as tb  # noqa: E402
import run_transmissibility_confirm as tc  # noqa: E402
import run_transmissibility_test as tt  # noqa: E402

THESIS_ROOT = rf.THESIS_ROOT
CODE_DIR = Path(__file__).resolve().parent
EXP_DIR = THESIS_ROOT / "research/experiments/EXP-029-damped-transmissibility-stac"
PREREG_PATH = EXP_DIR / "preregistration.json"
EXP027_QUAL_CSV = tb.EXP027_QUAL_CSV

DEPENDENCIES = (
    "run_damped_transmissibility_stac.py",
    "run_transmissibility_bridge.py",
    "run_transmissibility_confirm.py",
    "run_transmissibility_test.py",
    "run_selective_handover_prelim.py",
    "run_consistency_gate_prelim.py",
    "run_grid32_prelim.py",
    "run_noise_guidance_prelim.py",
    "run_followup_tests.py",
)

DEV_OFFSET = 60
QUAL_N = 30
DIV_N = 20
DIV_SEEDS = 12
DIV_SEED_BASE = 1000
CONDS = ("clean", "gauss18", "noisy15")
FOLDS = 5
SEED = 2026
BOOT_B = 2000

METRICS = ("wer", "similarity", "utmos")
TB_ALPHAS = {"TB25": 0.25, "TB50": 0.50, "TB75": 0.75, "TBRG": 1.00}
ARMS = ("W4", "TB25", "TB50", "TB75", "TBRG", "BLND", "W5R2", "W6R2", "W8R2", "W8")
CHEAP_CANDS = ("W4", "TB25", "TB50", "TB75", "TBRG", "BLND", "W5R2", "W6R2")
COST = {
    "W4": 13,
    "TB25": 13,
    "TB50": 13,
    "TB75": 13,
    "TBRG": 13,
    "BLND": 13,
    "W5R2": 14,
    "W6R2": 16,
    "W8R2": 20,
    "W8": 21,
}

TELEMETRY_FIELDS = (
    "gc_0",
    "d_1",
    "d_2",
    "d_3",
    "d_4",
    "d_5",
    "d_6",
    "d_7",
    "d_8",
    "A_4",
    "A_5",
    "A_6",
    "A_8",
    "A_4_norm",
    "A_6_norm",
    "rho_4",
    "slow_cos_4",
    "slow_gap_4",
    "slow_scale_4",
    "voice_gap_4",
    "fine_gap_4",
    "anchor_gap_4",
)

DIV_FIELDS = (
    ["utt_id", "speaker_id", "condition", "n_seeds"]
    + [f"{m}_{a}" for a in ARMS for m in ("cvf0", "voiced", "mel_spread", "slow_corr", "f0_corr_w8r2", "f0_shift_corr_w8r2")]
    + ["case_seconds", "utc"]
)

QUAL_FIELDS = (
    ["utt_id", "speaker_id", "condition", "prompt_snr", "gpu"]
    + list(TELEMETRY_FIELDS)
    + [f"{m}_{a}" for a in ARMS for m in METRICS + ("hypothesis",)]
    + ["case_seconds", "utc"]
)

SPLIT_TOL = 1e-5
REPRO_TOL = 0.01
ZERO_TOL = 1e-12
CLEAN_UTMOS_BAR = -0.08
WER_BAR = 0.01
SIM_BAR = -0.01
ROUTE_CLEAN_UTMOS_TOL = -0.08
ROUTE_POOLED_WER_TOL = 0.005
ROUTE_POOLED_SIM_TOL = -0.005
ROUTE_MIN_EARLY_FRAC = 0.20
ROUTE_MIN_NFE_SAVING_VS_W8 = 0.10


def cost(arm):
    """Total neural function evaluations (NFE) for an arm."""
    return COST[arm]


def qual_items(n=QUAL_N):
    """First n speakers from dev 61-100 (disjoint from calibration & test splits)."""
    return cg.dev_items()[DEV_OFFSET:DEV_OFFSET + n]


def div_items(n=DIV_N):
    """First n speakers from dev 61-100 for paired clean pitch-diversity evaluation."""
    return cg.dev_items()[DEV_OFFSET:DEV_OFFSET + n]


def _ratio(num, den, eps=1e-8):
    return float(num) / (float(den) + eps)


# --------------------------------------------------------------------------- core operators
def damped_bridge_handover(T4, S1, plen, alpha, energy_match=True, window=sh.SLOW_WINDOW):
    """Inject alpha-interpolated teacher slow prosody onto student S1 with norm conservation."""
    if not (0.0 <= alpha <= 1.0):
        raise ValueError(f"alpha must lie in [0, 1], got {alpha}")
    if alpha == 0.0:
        return sh._clone(S1)
    _, s_t, _ = sh.decompose(T4[0, plen:, :], window)
    v_s, s_s, f_s = sh.decompose(S1[0, plen:, :], window)
    norm_s = tb._frobenius_norm(s_s)
    norm_t = tb._frobenius_norm(s_t)
    s_t_matched = s_t * (norm_s / (norm_t + 1e-8))
    if alpha == 1.0:
        s_out = s_t_matched if energy_match else s_t
    else:
        s_mix = (1.0 - alpha) * s_s + alpha * s_t_matched
        if energy_match:
            norm_mix = float(tb._frobenius_norm(s_mix))
            s_out = s_mix * (norm_s / (norm_mix + 1e-8)) if norm_mix > 1e-8 else s_mix
        else:
            s_out = s_mix
    out = sh._clone(S1)
    out[0, plen:, :] = v_s + s_out + f_s
    return out


def teacher_trajectory(pipe, ctx, wanted=(4, 5, 6, 8)):
    """Run guided Euler steps 0..max(wanted) and log zero-overhead drifts and initial gc_0."""
    torch, sampler = pipe.torch, pipe.sampler
    grid = tt.grid16(pipe)
    plen = ctx["prompt_len"]
    kmax = max(wanted)
    if kmax < 8:
        raise ValueError(f"Expected max(wanted) >= 8 to record d_1..d_8; got {wanted}")
    wanted_set = set(wanted)
    x = sh._clone(ctx["y0"])
    prev = None
    drifts = {}
    ends = {}
    gc_0 = 0.0
    with torch.inference_mode():
        for i in range(kmax + 1):
            t = torch.tensor(grid[i], device=pipe.device, dtype=torch.float32)
            kw = dict(x=x, cond=ctx["step_cond"], text=ctx["text_ids"], time=t, mask=None)
            vc = pipe.tts.teacher(drop_audio_cond=False, drop_text=False, **kw)
            vu = pipe.tts.teacher(drop_audio_cond=True, drop_text=True, **kw)
            diff = vc - vu
            if i == 0:
                gc_0 = _ratio(tb._frobenius_norm(diff[:, plen:, :]), tb._frobenius_norm(vc[:, plen:, :]))
            v = vc + diff * sampler.CFG_STRENGTH
            ep = x + (1.0 - grid[i]) * v
            if prev is not None:
                drifts[i] = float(sampler.compute_stabilization_drift(ep, prev, plen))
            if i in wanted_set:
                ends[i] = ep
            prev = ep
            if i < kmax:
                x = x + (grid[i + 1] - grid[i]) * v

    a4 = sum(drifts[j] for j in range(1, 5))
    a5 = a4 + drifts[5]
    a6 = a5 + drifts[6]
    a8 = a6 + drifts[7] + drifts[8]
    telemetry = {
        "gc_0": gc_0,
        **{f"d_{j}": drifts[j] for j in range(1, 9)},
        "A_4": a4,
        "A_5": a5,
        "A_6": a6,
        "A_8": a8,
        "A_4_norm": _ratio(a4, gc_0),
        "A_6_norm": _ratio(a6, gc_0),
        "rho_4": _ratio(drifts[4], drifts[3]),
    }
    return ends, telemetry


def handover_discrepancy_signals(T4, S1, plen, window=sh.SLOW_WINDOW):
    """Zero-extra-NFE post-S1 teacher-vs-student discrepancy signals across voice, slow & fine bands."""
    t_gen = T4[0, plen:, :]
    s_gen = S1[0, plen:, :]
    v_t, s_t, f_t = sh.decompose(t_gen, window)
    v_s, s_s, f_s = sh.decompose(s_gen, window)
    return {
        "slow_cos_4": tb.slow_cosine(s_t, s_s),
        "slow_gap_4": _ratio(tb._frobenius_norm(s_t - s_s), tb._frobenius_norm(s_s)),
        "slow_scale_4": _ratio(tb._frobenius_norm(s_t), tb._frobenius_norm(s_s)),
        "voice_gap_4": _ratio(tb._frobenius_norm(v_t - v_s), tb._frobenius_norm(v_s)),
        "fine_gap_4": _ratio(tb._frobenius_norm(f_t - f_s), tb._frobenius_norm(f_s)),
        "anchor_gap_4": _ratio(tb._frobenius_norm(t_gen - s_gen), tb._frobenius_norm(s_gen)),
    }


def synthesize_all_arms(pipe, ctx, ends):
    """Generate all EXP-029 candidate & reference mels on one shared teacher trajectory."""
    taus = list(pipe.sampler.STUDENT_TAUS)
    taus_tail = taus[1:]
    plen = ctx["prompt_len"]
    t4, t5, t6, t8 = ends[4], ends[5], ends[6], ends[8]
    s1 = tb.student_step_tau1(pipe, ctx, t4)
    outs = {
        "W4": gl.student(pipe, ctx, s1, taus_tail),
        "BLND": gl.student(pipe, ctx, tb.blend_handover(t4, s1, plen, alpha=0.5), taus_tail),
        "W5R2": gl.student(pipe, ctx, t5, taus_tail),
        "W6R2": gl.student(pipe, ctx, t6, taus_tail),
        "W8R2": gl.student(pipe, ctx, t8, taus_tail),
        "W8": gl.student(pipe, ctx, t8, taus),
    }
    for name, alpha in TB_ALPHAS.items():
        x_br = damped_bridge_handover(t4, s1, plen, alpha=alpha, energy_match=True)
        outs[name] = gl.student(pipe, ctx, x_br, taus_tail)
    return outs, s1


# --------------------------------------------------------------------------- synthesis loops
def run_quality(pipe, items, conds=CONDS, out_dir=None, csv_name="quality.csv"):
    """Synthesize and score paired mixed-condition quality and full handover telemetry."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / csv_name
    done = rf.done_keys(path, 1)
    todo = [(it, c) for it in items for c in conds if (it["utt_id"], c) not in done]
    print(f"[quality] {len(todo)} cases to run ({len(done)} already in {path.name})", flush=True)

    for run_idx, (item, cond) in enumerate(todo, 1):
        t0 = time.perf_counter()
        ctx = pipe.prepare(item, cond, int(item["input_seed"]), out_dir)
        ends, telem = teacher_trajectory(pipe, ctx, wanted=(4, 5, 6, 8))
        outs, s1 = synthesize_all_arms(pipe, ctx, ends)
        telem.update(handover_discrepancy_signals(ends[4], s1, ctx["prompt_len"]))
        gt = pipe.ground_truth(item)
        row = {
            "utt_id": item["utt_id"],
            "speaker_id": item["speaker_id"],
            "condition": cond,
            "prompt_snr": ctx["prompt_snr"],
            "gpu": pipe.gpu,
            **telem,
        }
        for arm in ARMS:
            m = pipe.score(ng.decode(pipe, ctx, outs[arm]), item, gt)
            row.update({f"{k}_{arm}": m[k] for k in METRICS + ("hypothesis",)})
        row.update({"case_seconds": round(time.perf_counter() - t0, 2), "utc": tb._utc()})
        tb._append(path, QUAL_FIELDS, row)
        print(f"[{run_idx}/{len(todo)}] quality: {item['utt_id']} {cond} ({row['case_seconds']:.1f} s)", flush=True)
    print(f"[OK] quality complete: {path}", flush=True)
    return path


def _track_arms_fast(pipe, ctx, outs, arms):
    audios = {arm: ng.decode(pipe, ctx, outs[arm]) for arm in arms}
    with ThreadPoolExecutor(max_workers=min(len(arms), 4)) as ex:
        futs = {arm: ex.submit(sh.f0_track, audios[arm]) for arm in arms}
        return {arm: futs[arm].result() for arm in arms}


def run_diversity(pipe, items, out_dir, seeds=DIV_SEEDS, csv_name="diversity.csv"):
    """Run paired multi-seed clean synthesis to trace CVf0, mel_spread & f0 alignment vs alpha/k."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / csv_name
    done = rf.done_keys(path, 1)
    todo = [it for it in items if (it["utt_id"], "clean") not in done]
    print(f"[diversity] {len(todo)} speakers x {seeds} seeds ({len(done)} already in {path.name})", flush=True)

    for run_idx, item in enumerate(todo, 1):
        t0 = time.perf_counter()
        tracks = {a: [] for a in ARMS}
        slow_mels = {a: [] for a in ARMS}
        slow_corrs = {a: [] for a in ARMS}
        for m in range(seeds):
            seed = int(item["input_seed"]) + DIV_SEED_BASE + m
            ctx = pipe.prepare(item, "clean", seed, out_dir, noise_seed=item["input_seed"])
            ends, _ = teacher_trajectory(pipe, ctx, wanted=(4, 5, 6, 8))
            outs, _ = synthesize_all_arms(pipe, ctx, ends)
            plen = ctx["prompt_len"]
            s_t4 = tc._slow_component(ends[4], plen)
            arm_tracks = _track_arms_fast(pipe, ctx, outs, ARMS)
            for arm in ARMS:
                s_arm = tc._slow_component(outs[arm], plen)
                slow_mels[arm].append(s_arm)
                slow_corrs[arm].append(tb.slow_cosine(s_arm, s_t4))
                tracks[arm].append(arm_tracks[arm])

        ref_tracks = tracks["W8R2"]
        shifted_ref = ref_tracks[1:] + ref_tracks[:1]
        row = {"utt_id": item["utt_id"], "speaker_id": item["speaker_id"], "condition": "clean", "n_seeds": seeds}
        for arm in ARMS:
            row[f"cvf0_{arm}"], row[f"voiced_{arm}"] = sh.cvf0(tracks[arm])
            row[f"mel_spread_{arm}"] = tc.slow_mel_spread(slow_mels[arm])
            row[f"slow_corr_{arm}"] = statistics.mean(slow_corrs[arm])
            row[f"f0_corr_w8r2_{arm}"] = tb.paired_f0_corr(tracks[arm], ref_tracks)
            row[f"f0_shift_corr_w8r2_{arm}"] = tb.paired_f0_corr(tracks[arm], shifted_ref)
        row.update({"case_seconds": round(time.perf_counter() - t0, 2), "utc": tb._utc()})
        tb._append(path, DIV_FIELDS, row)
        print(f"[{run_idx}/{len(todo)}] diversity: {item['utt_id']} ({row['case_seconds']:.1f} s)", flush=True)
    print(f"[OK] diversity complete: {path}", flush=True)
    return path


def smoke(pipe, out_dir, prereg_path=PREREG_PATH, exp027_qual_csv=EXP027_QUAL_CSV):
    """Run 2-speaker algebraic bridge identities, split-student check, EXP-027 W4 check, and micro runs."""
    check_prereg(prereg_path)
    out_dir = Path(out_dir)
    sdir = out_dir / "smoke"
    sdir.mkdir(parents=True, exist_ok=True)
    items = tb.smoke_items()
    stored = {r["utt_id"]: r for r in rf.read_csv(exp027_qual_csv)}
    taus = list(pipe.sampler.STUDENT_TAUS)
    repro = []

    for item in items:
        ctx = pipe.prepare(item, "clean", int(item["input_seed"]), sdir)
        ends, _ = teacher_trajectory(pipe, ctx, wanted=(4, 5, 6, 8))
        t4 = ends[4]
        s1 = tb.student_step_tau1(pipe, ctx, t4)
        plen = ctx["prompt_len"]

        diff_a0 = tb._max_abs(damped_bridge_handover(t4, s1, plen, alpha=0.0) - s1)
        diff_a1 = tb._max_abs(
            damped_bridge_handover(t4, s1, plen, alpha=1.0) - tb.bridge_handover(t4, s1, plen, True)
        )
        w4_split = gl.student(pipe, ctx, s1, taus[1:])
        w4_full = gl.student(pipe, ctx, t4, taus)
        split_diff = tb._max_abs(w4_split - w4_full)
        if max(diff_a0, diff_a1, split_diff) > SPLIT_TOL:
            raise RuntimeError(
                f"Algebraic identity failed on {item['utt_id']}: a0={diff_a0:.2e}, a1={diff_a1:.2e}, split={split_diff:.2e}"
            )

        m = pipe.score(ng.decode(pipe, ctx, w4_split), item, pipe.ground_truth(item))
        ref = stored[item["utt_id"]]
        d_wer = abs(m["wer"] - float(ref["wer_W4"]))
        d_sim = abs(m["similarity"] - float(ref["similarity_W4"]))
        if d_wer > REPRO_TOL or d_sim > REPRO_TOL:
            raise RuntimeError(f"EXP-027 W4 check failed on {item['utt_id']}: dWER={d_wer:.4f}, dSIM={d_sim:.4f}")
        repro.append({"utt_id": item["utt_id"], "diff_a0": diff_a0, "diff_a1": diff_a1, "split_diff": split_diff, "d_wer": d_wer, "d_sim": d_sim})

    run_quality(pipe, items[:1], conds=("clean", "gauss18"), out_dir=sdir, csv_name="quality.csv")
    run_diversity(pipe, items, sdir, seeds=2, csv_name="diversity.csv")
    summary = {
        "repro": repro,
        "max_diff_a0": max(r["diff_a0"] for r in repro),
        "max_diff_a1": max(r["diff_a1"] for r in repro),
        "max_split_diff": max(r["split_diff"] for r in repro),
        "utc": tb._utc(),
    }
    (sdir / "smoke_check.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"[OK] smoke passed ({summary})", flush=True)
    return summary


# --------------------------------------------------------------------------- preregistration
def build_prereg(qual_list=None, div_list=None, exp027_qual_csv=EXP027_QUAL_CSV):
    """Build deterministic preregistration payload before GPU synthesis."""
    if qual_list is None:
        qual_list = qual_items()
    if div_list is None:
        div_list = div_items()
    qual_utts = [{"utt_id": it["utt_id"], "speaker_id": it["speaker_id"]} for it in qual_list]
    div_utts = [{"utt_id": it["utt_id"], "speaker_id": it["speaker_id"]} for it in div_list]
    code_hashes = {name: tb._sha256_file(CODE_DIR / name) for name in DEPENDENCIES}
    input_hashes = {str(Path(exp027_qual_csv).relative_to(THESIS_ROOT).as_posix()): tb._sha256_file(exp027_qual_csv)}
    payload = {
        "experiment": "EXP-029",
        "task_id": "EVAL-022",
        "title": "Damped Slow-Band Transmissibility Bridge & Mixed-Condition STAC Diagnostic",
        "framing": (
            "Development diagnostic following EXP-028 Outcome 3: calibrates energy-preserving slow-band "
            "interpolation TB_alpha (alpha in {0.25, 0.50, 0.75, 1.00}) and Step 5/6 (W5R2/W6R2) on "
            "clean and degraded prompts (gauss18, noisy15), leaving all 170 remaining calibration speakers "
            "and all 200 test speakers untouched."
        ),
        "formula": "X_TB(alpha) = v_S + (||s_S|| / (||s_mix|| + eps)) * ((1 - alpha)*s_S + alpha*s_tilde_T) + f_S",
        "conditions": list(CONDS),
        "arms": list(ARMS),
        "tb_alphas": dict(TB_ALPHAS),
        "costs": dict(COST),
        "n_qual_speakers": len(qual_utts),
        "n_qual_cases": len(qual_utts) * len(CONDS),
        "qual_utts": qual_utts,
        "n_div_speakers": len(div_utts),
        "div_seeds": DIV_SEEDS,
        "div_seed_base": DIV_SEED_BASE,
        "div_utts": div_utts,
        "clean_utmos_bar": CLEAN_UTMOS_BAR,
        "wer_bar": WER_BAR,
        "sim_bar": SIM_BAR,
        "route_clean_utmos_tol": ROUTE_CLEAN_UTMOS_TOL,
        "route_pooled_wer_tol": ROUTE_POOLED_WER_TOL,
        "route_pooled_sim_tol": ROUTE_POOLED_SIM_TOL,
        "route_min_early_frac": ROUTE_MIN_EARLY_FRAC,
        "route_min_nfe_saving_vs_w8": ROUTE_MIN_NFE_SAVING_VS_W8,
        "code_hashes": code_hashes,
        "input_hashes": input_hashes,
        "seed": SEED,
        "boot_b": BOOT_B,
        "folds": FOLDS,
    }
    sha = tb._sha256_bytes(json.dumps(payload, sort_keys=True).encode("utf-8"))
    return {**payload, "sha256": sha, "created_utc": tb._utc()}


def freeze(prereg_path=PREREG_PATH, force=False):
    """Write preregistration.json once unless --force is given."""
    prereg_path = Path(prereg_path)
    if prereg_path.exists() and not force:
        raise RuntimeError(f"Preregistration already exists at {prereg_path}; pass --force to overwrite.")
    prereg_path.parent.mkdir(parents=True, exist_ok=True)
    doc = build_prereg()
    prereg_path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    print(f"[OK] Froze {prereg_path} (sha256={doc['sha256'][:12]}...)")
    return doc


def check_prereg(prereg_path=PREREG_PATH):
    """Verify preregistration integrity, script hashes, and manifests."""
    prereg_path = Path(prereg_path)
    if not prereg_path.exists():
        raise RuntimeError(f"Missing preregistration: {prereg_path}. Run `freeze` first.")
    doc = json.loads(prereg_path.read_text(encoding="utf-8"))
    payload = {k: v for k, v in doc.items() if k not in ("sha256", "created_utc")}
    expected = tb._sha256_bytes(json.dumps(payload, sort_keys=True).encode("utf-8"))
    if doc.get("sha256") != expected:
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


# --------------------------------------------------------------------------- statistical & STAC analysis
def _pearson(xs, ys):
    n = len(xs)
    if n < 3:
        return float("nan")
    mx = statistics.mean(xs)
    my = statistics.mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    return num / (dx * dy) if dx > 1e-12 and dy > 1e-12 else float("nan")


def _auc_binary(scores, pos_labels):
    """Exact rank-based ROC-AUC with positive class = True."""
    pos = [s for s, y in zip(scores, pos_labels) if y]
    neg = [s for s, y in zip(scores, pos_labels) if not y]
    if not pos or not neg:
        return float("nan")
    wins = 0.0
    for sp in pos:
        for sn in neg:
            wins += 1.0 if sp > sn else (0.5 if sp == sn else 0.0)
    return wins / (len(pos) * len(neg))


def _condition_quality_table(qual_rows):
    """Per-condition paired metrics vs W8 and vs W8R2."""
    by_cond = {}
    for cond in sorted({r["condition"] for r in qual_rows}):
        sub = [r for r in qual_rows if r["condition"] == cond]
        arms_out = {}
        for arm in ARMS:
            entry = {m: tc.boot_ci([float(r[f"{m}_{arm}"]) for r in sub], B=BOOT_B, seed=SEED) for m in METRICS}
            if arm != "W8":
                entry["vs_W8"] = {
                    m: tb._summarize([float(r[f"{m}_{arm}"]) - float(r[f"{m}_W8"]) for r in sub], seed=SEED)
                    for m in METRICS
                }
            if arm != "W8R2":
                entry["vs_W8R2"] = {
                    m: tb._summarize([float(r[f"{m}_{arm}"]) - float(r[f"{m}_W8R2"]) for r in sub], seed=SEED)
                    for m in METRICS
                }
            arms_out[arm] = entry
        by_cond[cond] = {"n": len(sub), "arms": arms_out}
    return by_cond


def _clean_diversity_table(div_rows):
    """Summary metrics and paired W4 / W8R2 diversity contrasts."""
    arms_out = {}
    for arm in ARMS:
        cv_vals = [float(r[f"cvf0_{arm}"]) for r in div_rows]
        entry = {
            "n": len(div_rows),
            "cvf0": tc.boot_ci(cv_vals, B=BOOT_B, seed=SEED),
            "voiced": tc.boot_ci([float(r[f"voiced_{arm}"]) for r in div_rows], B=BOOT_B, seed=SEED),
            "mel_spread": tc.boot_ci([float(r[f"mel_spread_{arm}"]) for r in div_rows], B=BOOT_B, seed=SEED),
            "slow_corr": tc.boot_ci([float(r[f"slow_corr_{arm}"]) for r in div_rows], B=BOOT_B, seed=SEED),
            "f0_corr_w8r2": tc.boot_ci([float(r[f"f0_corr_w8r2_{arm}"]) for r in div_rows], B=BOOT_B, seed=SEED),
            "f0_shift_corr_w8r2": tc.boot_ci([float(r[f"f0_shift_corr_w8r2_{arm}"]) for r in div_rows], B=BOOT_B, seed=SEED),
        }
        if arm != "W4":
            entry["cvf0_minus_W4"] = tb._summarize([float(r[f"cvf0_{arm}"]) - float(r["cvf0_W4"]) for r in div_rows], seed=SEED)
        if arm != "W8R2":
            entry["cvf0_minus_W8R2"] = tb._summarize([float(r[f"cvf0_{arm}"]) - float(r["cvf0_W8R2"]) for r in div_rows], seed=SEED)
        arms_out[arm] = entry
    return arms_out


def evaluate_mixed_stac_routing(qual_rows, cheap_arm, ref_arm="W8", feature="A_4"):
    """5-fold speaker-grouped CV routing between cheap_arm and ref_arm across all conditions."""
    folds, k = tb._speaker_folds(qual_rows, n_folds=FOLDS, seed=SEED)
    if k < 2:
        return None
    use_snr = feature == "prompt_snr"
    reverse_sort = True  # higher A_4 / A_4_norm / prompt_snr -> cleaner / safer

    records = []
    for r in qual_rows:
        score = float(r[feature])
        records.append({
            "utt_id": r["utt_id"],
            "spk": r["speaker_id"],
            "cond": r["condition"],
            "score": score,
            "cheap": {m: float(r[f"{m}_{cheap_arm}"]) for m in METRICS},
            "ref": {m: float(r[f"{m}_{ref_arm}"]) for m in METRICS},
        })

    def pick_arm(rec, thr):
        if thr is None:
            return ref_arm
        return cheap_arm if rec["score"] >= thr else ref_arm

    chosen_folds = []
    oof = []
    for f in range(k):
        train = [rec for rec in records if folds[rec["spk"]] != f]
        test = [rec for rec in records if folds[rec["spk"]] == f]
        u_scores = sorted({rec["score"] for rec in train}, reverse=reverse_sort)
        cands = (
            [None]
            + [0.5 * (u_scores[i] + u_scores[i + 1]) for i in range(len(u_scores) - 1)]
            + ([u_scores[-1] - 1e-6] if u_scores else [])
        )
        best_thr = None
        best_nfe = float("inf")
        for thr in cands:
            picks = [pick_arm(rec, thr) for rec in train]
            clean_diffs = [
                (rec["cheap"]["utmos"] if p == cheap_arm else rec["ref"]["utmos"]) - rec["ref"]["utmos"]
                for rec, p in zip(train, picks) if rec["cond"] == "clean"
            ]
            dwer = statistics.mean(
                (rec["cheap"]["wer"] if p == cheap_arm else rec["ref"]["wer"]) - rec["ref"]["wer"]
                for rec, p in zip(train, picks)
            )
            dsim = statistics.mean(
                (rec["cheap"]["similarity"] if p == cheap_arm else rec["ref"]["similarity"]) - rec["ref"]["similarity"]
                for rec, p in zip(train, picks)
            )
            mean_cu = statistics.mean(clean_diffs) if clean_diffs else 0.0
            mean_nfe = statistics.mean(COST[p] for p in picks)
            if mean_cu >= ROUTE_CLEAN_UTMOS_TOL and dwer <= ROUTE_POOLED_WER_TOL and dsim >= ROUTE_POOLED_SIM_TOL and mean_nfe < best_nfe:
                best_nfe = mean_nfe
                best_thr = thr
        chosen_folds.append(best_thr)
        for rec in test:
            oof.append((rec, pick_arm(rec, best_thr)))

    costs = [COST[p] for _, p in oof]
    early_frac = sum(1 for _, p in oof if p == cheap_arm) / len(oof)
    mean_cost = statistics.mean(costs)
    clean_pairs = [(rec, p) for rec, p in oof if rec["cond"] == "clean"]
    cond_rates = {
        c: sum(1 for rec, p in oof if rec["cond"] == c and p == cheap_arm) / max(1, sum(1 for rec, _ in oof if rec["cond"] == c))
        for c in sorted({r["condition"] for r in qual_rows})
    }
    return {
        "cheap_arm": cheap_arm,
        "ref_arm": ref_arm,
        "feature": feature,
        "use_snr": use_snr,
        "n_cases": len(oof),
        "early_frac": early_frac,
        "cond_early_frac": cond_rates,
        "mean_cost": mean_cost,
        "saving_vs_W8": 1.0 - mean_cost / COST["W8"],
        "saving_vs_W8R2": 1.0 - mean_cost / COST["W8R2"],
        "fold_thresholds": chosen_folds,
        "pooled_dwer_vs_ref": tc.boot_ci([(r["cheap"]["wer"] if p == cheap_arm else r["ref"]["wer"]) - r["ref"]["wer"] for r, p in oof], B=BOOT_B, seed=SEED),
        "pooled_dsim_vs_ref": tc.boot_ci([(r["cheap"]["similarity"] if p == cheap_arm else r["ref"]["similarity"]) - r["ref"]["similarity"] for r, p in oof], B=BOOT_B, seed=SEED),
        "clean_dutmos_vs_ref": tc.boot_ci([(r["cheap"]["utmos"] if p == cheap_arm else r["ref"]["utmos"]) - r["ref"]["utmos"] for r, p in clean_pairs], B=BOOT_B, seed=SEED) if clean_pairs else None,
    }


def determine_verdict(clean_qual, clean_div, routing_results):
    """Evaluate preregistered clean-Pareto and mixed-STAC criteria."""
    clean_arms = clean_qual["arms"]
    passed_clean_arms = []
    for arm in CHEAP_CANDS:
        if arm == "W4":
            continue
        q_w8 = clean_arms[arm]["vs_W8"]
        mean_du = q_w8["utmos"]["boot"]["mean"]
        hi_dw = q_w8["wer"]["boot"]["hi"]
        lo_ds = q_w8["similarity"]["boot"]["lo"]
        qual_ok = (mean_du >= CLEAN_UTMOS_BAR) and (hi_dw <= WER_BAR) and (lo_ds >= SIM_BAR)
        div_ok = True if clean_div is None else (clean_div[arm]["cvf0_minus_W4"]["boot"]["lo"] > ZERO_TOL)
        if qual_ok and div_ok:
            passed_clean_arms.append(arm)

    stac_winners = []
    for key, res in routing_results.items():
        if res is None or res["use_snr"]:
            continue
        if res["cheap_arm"] not in passed_clean_arms:
            continue
        cu = res["clean_dutmos_vs_ref"]["mean"] if res["clean_dutmos_vs_ref"] else 0.0
        dw = res["pooled_dwer_vs_ref"]["hi"]
        ds = res["pooled_dsim_vs_ref"]["lo"]
        if (
            res["early_frac"] >= ROUTE_MIN_EARLY_FRAC
            and res["saving_vs_W8"] >= ROUTE_MIN_NFE_SAVING_VS_W8
            and cu >= CLEAN_UTMOS_BAR
            and dw <= WER_BAR
            and ds >= SIM_BAR
        ):
            stac_winners.append(key)

    if passed_clean_arms and stac_winners:
        return "confirmed_pareto_and_stac", passed_clean_arms, stac_winners
    if passed_clean_arms:
        return "clean_pareto_found_no_stac_headroom", passed_clean_arms, stac_winners
    return "monotone_tradeoff_only", passed_clean_arms, stac_winners


def _signal_diagnostics(qual_rows):
    """Exploratory clean-vs-noisy separation AUCs and telemetry correlations."""
    clean_rows = [r for r in qual_rows if r["condition"] == "clean"]
    degraded_flags = [r["condition"] != "clean" for r in qual_rows]
    clean_flags = [r["condition"] == "clean" for r in qual_rows]
    feats = ("A_4", "A_4_norm", "gc_0", "prompt_snr", "slow_cos_4", "slow_gap_4", "voice_gap_4", "fine_gap_4", "anchor_gap_4")
    sep_auc = {}
    for f in feats:
        vals = [float(r[f]) for r in qual_rows]
        a_pos = _auc_binary(vals, clean_flags)
        sep_auc[f] = round(max(a_pos, 1.0 - a_pos), 4) if not math.isnan(a_pos) else None
    clean_corr = {}
    for arm in ("TB50", "TB75", "TBRG", "W5R2", "W6R2"):
        du = [float(r[f"utmos_{arm}"]) - float(r["utmos_W8"]) for r in clean_rows]
        clean_corr[arm] = {f: _pearson([float(r[f]) for r in clean_rows], du) for f in feats}
    return {"clean_vs_degraded_auc": sep_auc, "clean_dutmos_correlations": clean_corr, "n_degraded": sum(degraded_flags)}


def _alpha_sweep_summary(clean_qual, clean_div):
    """Track dUTMOS, CVf0, mel_spread, and matched f0 correlation along alpha = 0 -> 1."""
    ordered = (("W4", 0.0), ("TB25", 0.25), ("TB50", 0.50), ("TB75", 0.75), ("TBRG", 1.0))
    sweep = []
    for arm, alpha in ordered:
        q = clean_qual["arms"][arm]
        d_utmos = 0.0 if arm == "W8" else q["vs_W8"]["utmos"]["boot"]["mean"]
        entry = {
            "arm": arm,
            "alpha": alpha,
            "utmos_mean": q["utmos"]["mean"],
            "dutmos_vs_W8": d_utmos,
            "dwer_vs_W8": q["vs_W8"]["wer"]["boot"]["mean"],
            "dsim_vs_W8": q["vs_W8"]["similarity"]["boot"]["mean"],
        }
        if clean_div is not None and arm in clean_div:
            d = clean_div[arm]
            entry["cvf0_mean"] = d["cvf0"]["mean"]
            entry["dcvf0_vs_W4"] = 0.0 if arm == "W4" else d["cvf0_minus_W4"]["boot"]["mean"]
            entry["mel_spread"] = d["mel_spread"]["mean"]
            entry["f0_corr_w8r2"] = d["f0_corr_w8r2"]["mean"]
        sweep.append(entry)
    return sweep


def analyze(out_dir=None, prereg_path=PREREG_PATH, allow_partial=False):
    """Analyze quality & diversity CSVs and write JSON, Markdown report, and SHA-256 manifest."""
    out_dir = Path(out_dir or EXP_DIR / "results")
    qual_path = out_dir / "quality.csv"
    div_path = out_dir / "diversity.csv"
    if not qual_path.exists():
        raise FileNotFoundError(f"Missing quality CSV: {qual_path}")
    qual_rows = rf.read_csv(qual_path, repair=True)
    div_rows = rf.read_csv(div_path, repair=True) if div_path.exists() else []
    if not allow_partial:
        if len(qual_rows) < QUAL_N * len(CONDS):
            raise RuntimeError(f"Incomplete quality.csv: {len(qual_rows)} / {QUAL_N * len(CONDS)} rows")
        if len(div_rows) < DIV_N:
            raise RuntimeError(f"Incomplete diversity.csv: {len(div_rows)} / {DIV_N} rows")

    by_cond = _condition_quality_table(qual_rows)
    clean_div = _clean_diversity_table(div_rows) if len(div_rows) >= 2 else None

    routing = {}
    for ref in ("W8", "W8R2"):
        for arm in ("TB25", "TB50", "TB75", "TBRG", "BLND", "W5R2", "W6R2", "W4"):
            for feat in ("A_4", "A_4_norm", "prompt_snr"):
                key = f"{arm}_to_{ref}_via_{feat}"
                routing[key] = evaluate_mixed_stac_routing(qual_rows, arm, ref_arm=ref, feature=feat)

    verdict, passed_clean, stac_winners = determine_verdict(by_cond["clean"], clean_div, routing)
    alpha_sweep = _alpha_sweep_summary(by_cond["clean"], clean_div)
    diag = _signal_diagnostics(qual_rows)
    results = {
        "experiment": "EXP-029",
        "partial": len(qual_rows) < QUAL_N * len(CONDS) or len(div_rows) < DIV_N,
        "verdict": verdict,
        "passed_clean_arms": passed_clean,
        "stac_winners": stac_winners,
        "n_quality_rows": len(qual_rows),
        "n_diversity_rows": len(div_rows),
        "alpha_sweep": alpha_sweep,
        "condition_quality": by_cond,
        "clean_diversity": clean_div,
        "routing_5fold": routing,
        "signal_diagnostics": diag,
        "utc": tb._utc(),
    }

    rjson = out_dir / "damped_transmissibility_results.json"
    rmd = out_dir / "DAMPED_TRANSMISSIBILITY_REPORT.md"
    rjson.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# EXP-029: Damped Slow-Band Transmissibility Bridge & Mixed-Condition `STAC` Report",
        "",
        f"- **Verdict:** `{verdict}` (`partial={results['partial']}`)",
        f"- **Passed Clean Quality + Diversity Arms:** `{passed_clean}`",
        f"- **Winning `STAC` Mixed Routers (`non-SNR`):** `{stac_winners}`",
        f"- **Rows:** quality `{len(qual_rows)} / {QUAL_N * len(CONDS)}`, diversity `{len(div_rows)} / {DIV_N}`",
        "",
        "## 1. Clean Quality & Pitch Diversity (`STAC` Cheap Candidates)",
        "",
        "| Arm | NFE | dUTMOS vs W8 | dWER vs W8 | dSIM vs W8 | CVf0 (clean) | dCVf0 vs W4 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    c_arms = by_cond["clean"]["arms"]
    for arm in ARMS:
        u_s = f"{c_arms[arm]['utmos']['mean']:.3f}" if arm == "W8" else tc._fmt_ci(c_arms[arm]["vs_W8"]["utmos"]["boot"], 3)
        w_s = f"{c_arms[arm]['wer']['mean']:.4f}" if arm == "W8" else tc._fmt_ci(c_arms[arm]["vs_W8"]["wer"]["boot"], 4)
        s_s = f"{c_arms[arm]['similarity']['mean']:.4f}" if arm == "W8" else tc._fmt_ci(c_arms[arm]["vs_W8"]["similarity"]["boot"], 4)
        if clean_div is not None:
            cv_s = f"{clean_div[arm]['cvf0']['mean']:.4f}"
            dcv_s = "ref" if arm == "W4" else tc._fmt_ci(clean_div[arm]["cvf0_minus_W4"]["boot"], 4)
        else:
            cv_s, dcv_s = "n/a", "n/a"
        lines.append(f"| `{arm}` | {COST[arm]} | {u_s} | {w_s} | {s_s} | {cv_s} | {dcv_s} |")

    for cond in ("gauss18", "noisy15"):
        if cond not in by_cond:
            continue
        lines.extend(["", f"## 2. Degraded-Prompt Behavior (`{cond}`, n={by_cond[cond]['n']})", "",
                      "| Arm | NFE | WER (abs / dW8) | SIM (abs / dW8) | UTMOS |",
                      "| --- | ---: | ---: | ---: | ---: |"])
        sub = by_cond[cond]["arms"]
        for arm in ARMS:
            w_s = f"{sub[arm]['wer']['mean']:.4f}" if arm == "W8" else f"{sub[arm]['wer']['mean']:.4f} ({sub[arm]['vs_W8']['wer']['boot']['mean']:+.4f})"
            s_s = f"{sub[arm]['similarity']['mean']:.4f}" if arm == "W8" else f"{sub[arm]['similarity']['mean']:.4f} ({sub[arm]['vs_W8']['similarity']['boot']['mean']:+.4f})"
            lines.append(f"| `{arm}` | {COST[arm]} | {w_s} | {s_s} | {sub[arm]['utmos']['mean']:.3f} |")

    lines.extend(["", "## 3. Mixed-Cohort 5-Fold CV Routing Highlights (`ref = W8`)", "",
                  "| Router | Early % | Mean NFE (Saving vs W8) | Clean dUTMOS | Pooled dWER | Pooled dSIM |",
                  "| --- | ---: | ---: | ---: | ---: | ---: |"])
    for arm in passed_clean or ["TB50", "W5R2", "W6R2", "W4"]:
        for feat in ("A_4", "A_4_norm", "prompt_snr"):
            r = routing.get(f"{arm}_to_W8_via_{feat}")
            if not r:
                continue
            cu = tc._fmt_ci(r["clean_dutmos_vs_ref"], 3) if r["clean_dutmos_vs_ref"] else "n/a"
            lines.append(
                f"| `{arm}` via `{feat}` | {r['early_frac'] * 100:.1f}% | {r['mean_cost']:.2f} ({r['saving_vs_W8'] * 100:.1f}%) | "
                f"{cu} | {tc._fmt_ci(r['pooled_dwer_vs_ref'], 4)} | {tc._fmt_ci(r['pooled_dsim_vs_ref'], 4)} |"
            )

    rmd.write_text("\n".join(lines) + "\n", encoding="utf-8")
    manifest_paths = [p for p in (qual_path, div_path, rjson, rmd) if p.exists()]
    prereg_p = Path(prereg_path)
    entries = {}
    if prereg_p.exists():
        entries["preregistration"] = {"path": str(prereg_p), "sha256": tb._sha256_file(prereg_p)}
    for p in manifest_paths:
        entries[p.name] = {"sha256": tb._sha256_file(p), "bytes": p.stat().st_size}
    (out_dir / "MANIFEST.json").write_text(json.dumps({"experiment": "EXP-029", "utc": tb._utc(), "files": entries}, indent=2) + "\n", encoding="utf-8")
    print(f"[OK] Analysis complete: verdict={verdict}, report={rmd}", flush=True)
    return results


def main():
    p = argparse.ArgumentParser(description="EXP-029 Damped Transmissibility Bridge & Mixed-Condition STAC")
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
    r.add_argument("--stage", choices=("all", "quality", "diversity"), default="all")

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
