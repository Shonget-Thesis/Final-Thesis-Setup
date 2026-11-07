"""EXP-019: guidance-decomposed trajectory telemetry and a risk-controlled handover path (Colab Tesla T4).

Motivation. STAC (EXP-018) sums the endpoint drift d_j = ||x1hat(t_j) - x1hat(t_{j-1})||_M / ||x1hat(t_j)||_M. For the
guided Euler teacher (x_{j} = x_{j-1} + (t_j - t_{j-1}) v_{j-1}, x1hat(t_j) = x_j + (1 - t_j) v_j) the algebra is exact:

    Lemma 1   x1hat(t_j) - x1hat(t_{j-1}) = (1 - t_j) (v_j - v_{j-1}),         v_j = c_j + w g_j,   g_j = c_j - u_j

where c_j / u_j are the conditional / unconditional velocities the teacher already computes for classifier-free guidance
(w = CFG_STRENGTH). The endpoint change therefore splits exactly into a conditional part dc_j = (1 - t_j)(c_j - c_{j-1})
and a guidance part dg_j = w (1 - t_j)(g_j - g_{j-1}), and

    d_j^2 = p^c_j + p^g_j + p^x_j,   p^c = ||dc||^2 / ||x1hat||^2,  p^g = ||dg||^2 / ||x1hat||^2,  p^x = 2 <dc, dg> / ||x1hat||^2

(norms over the generated frames M). Every quantity recorded here reuses tensors of the existing teacher passes, so it costs
zero extra forward passes; the extra arithmetic is timed (sig_ms).

What this run tests (hypotheses and pass criteria are frozen in preregistration.json by `freeze` before any synthesis):
  H1  Lemma 1 holds on the T4 trajectories (relative residual <= 1e-3).
  H2  Which component (conditional or guidance) carries the noise-induced drop of the drift ("premature stabilization").
  H3  A guidance-aware signal, chosen on the development speakers from a fixed family of four, predicts which early
      handovers are safe better than STAC's action A_k on speakers not used by EXP-012..EXP-018.
  H4  The proposed rule (selected signal + Pareto marginal-efficiency path + split-path monotone conformal risk control),
      calibrated once on the development speakers, meets the quality criteria on those new speakers, and each of its three
      parts is necessary (ablations).
  H5  Zero-shot degraded prompts (reverberation, telephone band; babble with --demand_dir) on the new speakers.

Data. Development: the EXP-012 Quick set (first 60 development speakers x clean, noisy15, gauss18/20/22/25); its labels
are the committed EXP-018 steps1to8.csv, so `dev` runs the teacher only. Confirmation: development speakers 61..100 by
utterance ID (40 speakers), which no experiment from EXP-011 to EXP-018 used (they were synthesized only in EXP-001's
all1000 grid at Steps 4/5/6/8, clean and noisy15). Neither set is an independent held-out test of the thesis protocol.

Subcommands:
  freeze   write preregistration.json (no GPU). Run once, commit it, then run the GPU parts.
  smoke    2 development speakers x (clean, gauss20): telemetry with reproduction checks + one full label synthesis (timing)
  dev      telemetry for the 360 development cases (teacher only)
  fresh    the 40 new speakers x 6 conditions: telemetry + every 16-grid step synthesized and scored (+32-grid: --with_g32)
  ood      the 40 new speakers x reverb03, reverb09, telephone (+ babble15, babble20 with --demand_dir), 16-grid
  analyze  every reported number from the CSV files (no GPU)
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
import run_consistency_gate_prelim as cg  # noqa: E402
import run_grid32_prelim as g32  # noqa: E402
import run_swsr_handover_prelim as sw  # noqa: E402

THESIS_ROOT = rf.THESIS_ROOT
EXP_DIR = THESIS_ROOT / "research/experiments/EXP-019-guidance-decomposition-prelim"
PREREG_PATH = EXP_DIR / "preregistration.json"
STEPS1TO8_CSV = THESIS_ROOT / "research/experiments/EXP-018-swsr-handover-prelim/results/steps1to8.csv"
NO_STORED = THESIS_ROOT / "research/experiments/EXP-019-guidance-decomposition-prelim/__no_stored_labels__.csv"

DEV_N, FRESH_N = 60, 40
CONDS = tuple(cg.QUICK["conditions"])
OOD_CONDS = ("reverb03", "reverb09", "telephone")
BABBLE_CONDS = ("babble15", "babble20")
SEED, B_BOOT, ALPHA = 2026, 1000, 0.05
EPS = 1e-8
K16, K32 = 8, 15                      # telemetry runs the 16-grid to Step 8 and the 32-grid to Step 15
GRID16, GRID32 = sw.GRID16, sw.GRID32
EARLY_STEPS = (1, 2, 3, 4)             # H3: 16-grid steps where premature stabilization was observed
H2_STEPS = (2, 3, 4)
T_FLOOR = 0.019                       # EXP-018: candidate floor t_k >= 0.019 (16-grid Step 2, 32-grid Step 4)

# Multi-metric bounded loss (same margins as the EXP-018 hinge, plus clean-prompt naturalness)
LOSS = {"m_wer": 0.01, "b_wer": 0.10, "m_sim": 0.015, "b_sim": 0.025, "m_utmos": 0.15, "b_utmos": 0.50}
SAFE = {"wer": 0.01, "sim": 0.015, "utmos_clean": 0.15}   # hindsight-safe label for H3 (vs the grid reference)
PATH = {"n_q": 40, "r_max": 0.25, "eps": 1e-3, "path_share": 1.0 / 3.0}
H4 = {"utmos_clean_drop": 0.10, "saving_margin": 0.01}

SIGNAL_FAMILY = {
    "A_over_gc0": "guidance-normalized action: A_k / gc_0, with gc_0 = ||g_0||_M / ||c_0||_M at t_0 = 0",
    "guid_action": "guidance action: sum_{j<=k} sqrt(p^g_j) (cumulative guidance-induced endpoint change)",
    "cond_action": "conditional action: sum_{j<=k} sqrt(p^c_j) (cumulative conditional-velocity endpoint change)",
    "net_disp": "net endpoint displacement: D_k = ||x1hat(t_k) - x1hat(t_0)||_M / ||x1hat(t_k)||_M",
}
BASELINE_SIGNALS = {"A": "STAC action A_k = sum_{j<=k} d_j", "snr": "prompt SNR (dB), the SNR-only signal"}

# Grids: EXP-018's full 16-grid plus floor versions (t_k >= 0.019) used as the primary EXP-019 setting
sw.GRIDS_EXT.setdefault("g16_floor", {
    **sw.GRIDS_EXT["g16_all8"],
    "label": "16-interval grid, Steps 2..7 (t_k >= 0.019) vs Step 8",
    "cands": tuple(f"g16_{k}" for k in range(2, 8)),
    "k": {f"g16_{k}": k for k in range(2, 8)},
})
sw.GRIDS_EXT.setdefault("g32_floor14", {
    **sw.GRIDS_EXT["g32_all14"],
    "label": "32-interval grid, Steps 4..13 (t_k >= 0.019) vs Step 14",
    "cands": tuple(f"g32_{k}" for k in range(4, 14)),
    "k": {f"g32_{k}": k for k in range(4, 14)},
})
PRIMARY_GRID = "g16_floor"
SECONDARY_GRIDS = ("g16_all8", "g32_floor14")
BASELINES = ("stac", "crc_stac", "snr_casc", "snr_cal")
PROPOSED = "gpc"
ABLATIONS = {"gpc_A": "signal ablation: A_k instead of the selected signal",
             "gpc_greedy": "path ablation: stage-by-stage greedy path instead of the Pareto marginal-efficiency path",
             "gpc_nocrc": "calibration ablation: path built and stopped on all calibration data at mean loss <= alpha, no split"}


def tel_fields():
    f = ["utt_id", "speaker_id", "condition", "part", "prompt_snr", "gpu", "cfg_strength",
         "t_teacher16", "t_teacher32", "sig_ms16", "sig_ms32"]
    for p, K in (("g16", K16), ("g32", K32)):
        f += [f"{p}_gc_{i}" for i in range(K + 1)] + [f"{p}_gam_{i}" for i in range(K + 1)]
        for name in ("d", "pc", "pg", "px", "res", "D", "S", "cgd"):
            f += [f"{p}_{name}_{i}" for i in range(1, K + 1)]
        f += [f"{p}_R_{i}" for i in range(K)]
    return f + ["utc"]


TEL_FIELDS = tel_fields()


def sha256_text(path):
    """SHA-256 of a text file with normalized line endings (Windows and Colab checkouts agree)."""
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def fresh_items(n=FRESH_N):
    return cg.dev_items()[DEV_N:DEV_N + n]


# ============================================================================= GPU: telemetry teacher
def telemetry_teacher(pipe, ctx, grid, kmax):
    """Guided Euler teacher on `grid` to Step kmax (same operations as run_grid32_prelim.grid_teacher, so d_j reproduces
    the stored drift bit for bit). Returns ({i: x1hat(t_i)}, record, signal seconds). The record holds, per step, the
    guidance contrast gc, guidance share gam, drift d, the exact split p^c/p^g/p^x, the Lemma 1 residual, net displacement
    D, straightness S, cos(g, x1hat - x1hat_0) and the distance to the final endpoint R (an oracle quantity, analysis only)."""
    torch, sampler = pipe.torch, pipe.sampler
    w, plen = float(sampler.CFG_STRENGTH), ctx["prompt_len"]

    def gen(a):
        return a[:, plen:, :].double()

    def nrm(a):
        return float(torch.linalg.vector_norm(gen(a)))

    def dot(a, b):
        return float((gen(a) * gen(b)).sum())

    x, ends, rec, sig = ctx["y0"].clone(), {}, {}, 0.0
    prev_ep = prev_c = prev_g = prev_v = ep0 = None
    path_len = 0.0
    with torch.inference_mode():
        for i in range(kmax + 1):
            t = torch.tensor(grid[i], device=pipe.device, dtype=torch.float32)
            kw = dict(x=x, cond=ctx["step_cond"], text=ctx["text_ids"], time=t, mask=None)
            vc = pipe.tts.teacher(drop_audio_cond=False, drop_text=False, **kw)
            vu = pipe.tts.teacher(drop_audio_cond=True, drop_text=True, **kw)
            v = vc + (vc - vu) * sampler.CFG_STRENGTH
            ep = x + (1.0 - grid[i]) * v
            ends[i] = ep
            if prev_ep is not None:
                rec[f"d_{i}"] = float(sampler.compute_stabilization_drift(ep, prev_ep, plen))
            if pipe.device.type == "cuda":
                torch.cuda.synchronize()
            s0 = time.perf_counter()
            g = vc - vu
            ep_n, g_n = nrm(ep), nrm(g)
            rec[f"gc_{i}"] = g_n / (nrm(vc) + EPS)
            rec[f"gam_{i}"] = w * (1.0 - grid[i]) * g_n / (ep_n + EPS)
            if prev_ep is None:
                ep0 = ep
            else:
                h = 1.0 - grid[i]
                dc, dg, delta = h * (vc - prev_c), w * h * (g - prev_g), ep - prev_ep
                den = ep_n ** 2 + EPS
                rec[f"pc_{i}"] = nrm(dc) ** 2 / den
                rec[f"pg_{i}"] = nrm(dg) ** 2 / den
                rec[f"px_{i}"] = 2.0 * dot(dc, dg) / den
                delta_n = nrm(delta)
                rec[f"res_{i}"] = nrm(delta - h * (v - prev_v)) / (delta_n + EPS)
                path_len += delta_n
                disp = ep - ep0
                disp_n = nrm(disp)
                rec[f"D_{i}"] = disp_n / (ep_n + EPS)
                rec[f"S_{i}"] = disp_n / (path_len + EPS)
                rec[f"cgd_{i}"] = dot(g, disp) / (g_n * disp_n + EPS)
            if pipe.device.type == "cuda":
                torch.cuda.synchronize()
            sig += time.perf_counter() - s0
            prev_ep, prev_c, prev_g, prev_v = ep, vc, g, v
            if i < kmax:
                x = x + (grid[i + 1] - grid[i]) * v
        final_n = nrm(ends[kmax])
        for i in range(kmax):
            rec[f"R_{i}"] = nrm(ends[kmax] - ends[i]) / (final_n + EPS)
    return ends, rec, sig


def run_telemetry(pipe, items, conds, out_dir, part, csv_name="telemetry.csv"):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / csv_name
    done = rf.done_keys(path, 1)
    todo = [(it, c) for it in items for c in conds if (it["utt_id"], c) not in done]
    print(f"[telemetry:{part}] {len(todo)} cases to run ({len(done)} already in {path.name}) on {pipe.gpu}", flush=True)
    for n, (item, cond) in enumerate(todo, 1):
        ctx = pipe.prepare(item, cond, item["input_seed"], out_dir)
        row = {"utt_id": item["utt_id"], "speaker_id": item["speaker_id"], "condition": cond, "part": part,
               "prompt_snr": ctx["prompt_snr"], "gpu": pipe.gpu, "cfg_strength": float(pipe.sampler.CFG_STRENGTH)}
        for p, grid, K in (("g16", GRID16, K16), ("g32", GRID32, K32)):
            (_, rec, sig), dt = g32.timed(pipe, telemetry_teacher, pipe, ctx, grid, K)
            row.update({f"{p}_{k}": v for k, v in rec.items()})
            row[f"t_teacher{p[1:]}"] = round(dt, 4)
            row[f"sig_ms{p[1:]}"] = round(1000.0 * sig, 3)
        row["utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        new = not path.exists()
        with open(path, "a", newline="", encoding="utf-8") as f:
            wr = csv.DictWriter(f, fieldnames=TEL_FIELDS)
            if new:
                wr.writeheader()
            wr.writerow(row)
            f.flush()
            os.fsync(f.fileno())
        print(f"[{n}/{len(todo)}] telemetry {part}: {item['utt_id'][-8:]} {cond} (signals {row['sig_ms16']:.1f} + "
              f"{row['sig_ms32']:.1f} ms; max Lemma-1 residual "
              f"{max(rec.get(f'res_{i}', 0.0) for i in range(1, K32 + 1)):.1e})", flush=True)
    print(f"[OK] telemetry {part} complete: {path}", flush=True)
    return path


# ============================================================================= preregistration
def build_prereg():
    return {
        "experiment": "EXP-019",
        "purpose": "Mechanism (exact guidance decomposition of the endpoint drift) and confirmation of a guidance-aware, "
                   "risk-controlled handover rule on development speakers not used by EXP-012..EXP-018",
        "lemma1": "x1hat(t_j) - x1hat(t_{j-1}) = (1 - t_j)(v_j - v_{j-1}); v = c + w g; g = c - u; "
                  "d_j^2 = p^c_j + p^g_j + p^x_j (generated frames)",
        "data": {"dev": f"first {DEV_N} development speakers by utterance ID x {list(CONDS)}; labels: EXP-018 steps1to8.csv",
                 "fresh": f"development speakers {DEV_N + 1}..{DEV_N + FRESH_N} by utterance ID x {list(CONDS)}; "
                          "used before only in EXP-001 (all1000, Steps 4/5/6/8, clean and noisy15)",
                 "ood": f"the fresh speakers x {list(OOD_CONDS)} (+ {list(BABBLE_CONDS)} when a DEMAND track is given)",
                 "fresh_utt_ids": [it["utt_id"] for it in fresh_items()]},
        "grids": {"primary": PRIMARY_GRID, "secondary": list(SECONDARY_GRIDS), "t_floor": T_FLOOR,
                  "cands": {G: list(sw.GRIDS_EXT[G]["cands"]) for G in (PRIMARY_GRID,) + SECONDARY_GRIDS},
                  "refs": {G: sw.GRIDS_EXT[G]["ref"] for G in (PRIMARY_GRID,) + SECONDARY_GRIDS}},
        "loss": {**LOSS, "definition": "L = min(1, max(dWER_hinge, dSIM_hinge, dUTMOS_hinge [clean prompts only])) vs the "
                                       "grid reference, hinge(x) = max(0, (x - m) / b)"},
        "safe_label": {**SAFE, "definition": "dWER <= 0.01, dSIM >= -0.015 and (clean) dUTMOS >= -0.15 vs the reference"},
        "signal_family": SIGNAL_FAMILY, "baseline_signals": BASELINE_SIGNALS,
        "selection_rule": f"on dev only: the family member with the highest mean AUC over 16-grid steps {list(EARLY_STEPS)} "
                          "for the safe label (ties: family order)",
        "policies": {
            "gpc": "selected signal; candidate thresholds ordered by the Pareto marginal-efficiency path (each move lowers "
                   "mean NFE, chosen by NFE drop / (loss increase + eps)), built on 1/3 of the calibration speakers; the "
                   "stopping index chosen on the other 2/3 by conformal risk control on the speaker-level monotone "
                   f"envelope loss: largest t with (1 + sum_s max_{{t'<=t}} L_s(t')) / (n + 1) <= alpha = {ALPHA}",
            **ABLATIONS,
            "baselines": {b: sw.ALL_SCHEDULERS.get(b, b) for b in BASELINES},
            "fixed": "Step 7 and the reference",
        },
        "path": PATH,
        "hypotheses": {
            "H1": "max Lemma-1 residual <= 1e-3 over every telemetry row and step (pass/fail)",
            "H2": f"guidance share S_g = sum(dp^g + dp^x / 2) / sum(d(d^2)) of the noisy-minus-clean change at steps "
                  f"{list(H2_STEPS)} (paired by utterance), speaker bootstrap 95% CI: 'guidance-driven' if the lower bound "
                  "> 0.5, 'conditional-driven' if the upper bound < 0.5, else 'mixed' (descriptive, reported for dev and fresh)",
            "H3": f"on fresh: mean AUC over steps {list(EARLY_STEPS)} of the selected signal minus that of A_k, speaker "
                  f"bootstrap ({B_BOOT}) 95% CI lower bound > 0 (pass/fail); the same against prompt SNR is secondary",
            "H4": f"on fresh, {PRIMARY_GRID}, gpc calibrated once on dev: (a) TOST p < 0.05 on WER (1 pp) and SIM (0.02) vs "
                  f"the reference; (b) mean clean-prompt UTMOS drop <= {H4['utmos_clean_drop']}; (c) test risk (mean "
                  f"speaker-level loss) <= alpha; (d) NFE saving >= STAC's saving - {H4['saving_margin']:.0%} or STAC fails "
                  "(a) or (b). Necessity (descriptive): each ablation either saves at least 1 pp less or fails (a)-(c)",
            "H5": "on OOD fresh cases: gpc's mean dWER vs the reference has a speaker-bootstrap 95% upper bound <= +0.01 and "
                  "gpc saves at least as much as the Stage-Wise SNR Cascade (pass/fail)",
        },
        "bootstrap": {"B": B_BOOT, "seed": SEED, "unit": "speaker"},
        "evidence_status": "development evidence; not the thesis's frozen held-out evaluation (EVAL-001/EVAL-002)",
        "runner_sha256": sha256_text(Path(__file__)),
        "input_hashes": {"research/experiments/EXP-018-swsr-handover-prelim/results/steps1to8.csv": rf.sha256_file(STEPS1TO8_CSV)},
        "thesis_commit": rf.git_head(THESIS_ROOT),
    }


def freeze(out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "preregistration.json"
    if path.exists():
        raise SystemExit(f"{path} exists; a preregistration is written once (delete it only if nothing was synthesized)")
    pre = build_prereg()
    pre["sha256"] = hashlib.sha256(json.dumps(pre, sort_keys=True).encode("utf-8")).hexdigest()
    pre["created_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    path.write_text(json.dumps(pre, indent=2) + "\n", encoding="utf-8")
    print(f"[OK] preregistration written: {path} (sha256 {pre['sha256'][:16]}...)")
    return pre


def check_prereg(prereg_path=PREREG_PATH):
    p = Path(prereg_path)
    if not p.exists():
        raise SystemExit(f"Missing {p}: run `freeze` and commit the preregistration before any GPU run.")
    pre = json.loads(p.read_text(encoding="utf-8"))
    body = {k: v for k, v in pre.items() if k not in ("sha256", "created_utc")}
    if hashlib.sha256(json.dumps(body, sort_keys=True).encode("utf-8")).hexdigest() != pre["sha256"]:
        raise SystemExit("preregistration.json was edited after it was frozen")
    if pre["runner_sha256"] != sha256_text(Path(__file__)):
        raise SystemExit("This runner differs from the one frozen in preregistration.json")
    return pre


# ============================================================================= analysis: data
def _f(v):
    return None if v in (None, "") else float(v)


def load_telemetry(path):
    out = {}
    for r in rf.read_csv(path, repair=True):
        out[(r["utt_id"], r["condition"])] = {k: _f(v) for k, v in r.items()
                                              if k not in ("utt_id", "speaker_id", "condition", "part", "gpu", "utc")}
    return out


def join(labels_csv, tel):
    """EXP-018-format label rows joined with telemetry; rows without telemetry are dropped."""
    table = []
    for r in sw.build_steps1to8_table(rf.read_csv(labels_csv, repair=True)):
        t = tel.get((r["utt"], r["cond"]))
        if t is not None:
            r["tel"] = t
            table.append(r)
    return table


def prefix(G):
    return "g16" if G.startswith("g16") else "g32"


def signal(r, G, o, name):
    """Value of signal `name` at candidate output `o` of grid G for case r (zero extra NFE except 'snr')."""
    if name == "snr":
        return r["snr"]
    p, k, T = prefix(G), sw.GRIDS_EXT[G]["k"][o], r["tel"]
    if name == "A":
        return sum(T[f"{p}_d_{j}"] for j in range(1, k + 1))
    if name == "A_over_gc0":
        return sum(T[f"{p}_d_{j}"] for j in range(1, k + 1)) / (T[f"{p}_gc_0"] + EPS)
    if name == "guid_action":
        return sum(math.sqrt(max(T[f"{p}_pg_{j}"], 0.0)) for j in range(1, k + 1))
    if name == "cond_action":
        return sum(math.sqrt(max(T[f"{p}_pc_{j}"], 0.0)) for j in range(1, k + 1))
    if name == "net_disp":
        return T[f"{p}_D_{k}"]
    raise ValueError(name)


def loss(r, o, ref):
    m, f = r["m"], LOSS
    lw = max(0.0, (m[o]["wer"] - m[ref]["wer"] - f["m_wer"]) / f["b_wer"])
    ls = max(0.0, (m[ref]["similarity"] - m[o]["similarity"] - f["m_sim"]) / f["b_sim"])
    lu = max(0.0, (m[ref]["utmos"] - m[o]["utmos"] - f["m_utmos"]) / f["b_utmos"]) if r["cond"] == "clean" else 0.0
    return min(1.0, max(lw, ls, lu))


def safe(r, o, ref):
    m = r["m"]
    ok = m[o]["wer"] - m[ref]["wer"] <= SAFE["wer"] and m[o]["similarity"] - m[ref]["similarity"] >= -SAFE["sim"]
    if r["cond"] == "clean":
        ok = ok and m[o]["utmos"] - m[ref]["utmos"] >= -SAFE["utmos_clean"]
    return ok


# ============================================================================= analysis: statistics
def auc(scores, labels):
    """Mann-Whitney AUC with average ranks for ties (P(score_pos > score_neg) + 0.5 P(tie))."""
    n1 = sum(1 for y in labels if y)
    n0 = len(labels) - n1
    if n1 == 0 or n0 == 0:
        return None
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        for q in range(i, j + 1):
            ranks[order[q]] = (i + j) / 2.0 + 1.0
        i = j + 1
    r1 = sum(ranks[i] for i, y in enumerate(labels) if y)
    return (r1 - n1 * (n1 + 1) / 2.0) / (n1 * n0)


def mean_early_auc(rows, name, G="g16_all8", steps=EARLY_STEPS):
    ref = sw.GRIDS_EXT[G]["ref"]
    vals = []
    for k in steps:
        o = f"g16_{k}"
        a = auc([signal(r, G, o, name) for r in rows], [safe(r, o, ref) for r in rows])
        if a is not None:
            vals.append(a)
    return statistics.mean(vals) if vals else None


def by_speaker(rows):
    out = {}
    for r in rows:
        out.setdefault(r["spk"], []).append(r)
    return out


def boot_ci(rows, stat, B=B_BOOT, seed=SEED):
    """Speaker-cluster bootstrap percentile 95% interval of stat(rows)."""
    groups = list(by_speaker(rows).values())
    rng = random.Random(seed)
    vals = []
    for _ in range(B):
        sample = [r for _ in groups for r in groups[rng.randrange(len(groups))]]
        v = stat(sample)
        if v is not None:
            vals.append(v)
    vals.sort()
    if not vals:
        return None, None
    return vals[int(0.025 * (len(vals) - 1))], vals[int(0.975 * (len(vals) - 1))]


def guidance_share(rows, steps=H2_STEPS):
    """S_g of the noisy-minus-clean change of d^2 (paired by utterance), pooled over noisy conditions and steps."""
    by = {(r["utt"], r["cond"]): r for r in rows}
    num = den = 0.0
    for (u, c), r in by.items():
        if c == "clean" or (u, "clean") not in by:
            continue
        T, C = r["tel"], by[(u, "clean")]["tel"]
        for j in steps:
            dpc = T[f"g16_pc_{j}"] - C[f"g16_pc_{j}"]
            dpg = T[f"g16_pg_{j}"] - C[f"g16_pg_{j}"]
            dpx = T[f"g16_px_{j}"] - C[f"g16_px_{j}"]
            num += dpg + 0.5 * dpx
            den += dpc + dpg + dpx
    return num / den if abs(den) > 1e-12 else None


def mechanism(rows):
    res = {"per_condition": {}, "S_g": guidance_share(rows)}
    res["S_g_ci"] = boot_ci(rows, guidance_share)
    lo, hi = res["S_g_ci"]
    res["verdict"] = ("guidance-driven" if lo is not None and lo > 0.5 else
                      "conditional-driven" if hi is not None and hi < 0.5 else "mixed")
    conds = sorted({r["cond"] for r in rows}, key=lambda c: (c != "clean", c))
    for c in conds:
        sub = [r for r in rows if r["cond"] == c]
        T = [r["tel"] for r in sub]
        res["per_condition"][c] = {
            "n": len(sub),
            "gc0": statistics.mean(t["g16_gc_0"] for t in T),
            "gc4": statistics.mean(t["g16_gc_4"] for t in T),
            **{f"{q}_{j}": statistics.mean(t[f"g16_{q}_{j}"] for t in T) for q in ("d", "pc", "pg", "px") for j in H2_STEPS},
            "D4": statistics.mean(t["g16_D_4"] for t in T),
            "S4": statistics.mean(t["g16_S_4"] for t in T),
        }
    return res


def identity_check(rows):
    res_max = max(r["tel"][f"{p}_res_{i}"] for r in rows for p, K in (("g16", K16), ("g32", K32))
                  for i in range(1, K + 1) if r["tel"].get(f"{p}_res_{i}") is not None)
    split_max = 0.0
    for r in rows:
        for p, K in (("g16", K16), ("g32", K32)):
            for i in range(1, K + 1):
                T = r["tel"]
                if T.get(f"{p}_d_{i}") is None:
                    continue
                d2 = T[f"{p}_d_{i}"] ** 2
                split_max = max(split_max, abs(d2 - (T[f"{p}_pc_{i}"] + T[f"{p}_pg_{i}"] + T[f"{p}_px_{i}"])) / (d2 + EPS))
    return {"max_residual": res_max, "max_split_rel_error": split_max, "pass": res_max <= 1e-3}


def reproduction(rows):
    """Telemetry drift vs the label file's drift (same T4 operations: expected exactly 0)."""
    m16 = max((abs(r["tel"][f"g16_d_{k}"] - r["d16"][k]) for r in rows for k in sw.ALL_G16_STEPS), default=None)
    m32 = max((abs(r["tel"][f"g32_d_{k}"] - r["d32"][k]) for r in rows if r["d32"] for k in sw.ALL_G32_STEPS), default=None)
    return {"max_abs_diff_d16": m16, "max_abs_diff_d32": m32}


# ============================================================================= analysis: decision rules
def _compile(rows, G, name):
    g = sw.GRIDS_EXT[G]
    cands, ref = list(g["cands"]), g["ref"]
    S = [[signal(r, G, o, name) for o in cands] for r in rows]
    outs = cands + [ref]
    L = [[loss(r, o, ref) for o in outs] for r in rows]
    C = [sw.COST_ALL[o] for o in outs]
    return cands, ref, S, L, C


def _ladders(S, M, n_q):
    lad = []
    for m in range(M):
        vals = sorted({round(s[m], 6) for s in S}, reverse=True)
        if len(vals) > n_q:
            idxs = sorted({int(round(i * (len(vals) - 1) / (n_q - 1))) for i in range(n_q)})
            vals = [vals[i] for i in idxs]
        lad.append(vals)
    return lad


def _picks(S, lams):
    M = len(lams)
    return [next((m for m in range(M) if lams[m] is not None and s[m] >= lams[m]), M) for s in S]


def _cost_risk(S, L, C, lams):
    p = _picks(S, lams)
    return statistics.mean(C[i] for i in p), statistics.mean(L[n][i] for n, i in enumerate(p))


def build_path(rows, G, name, mode="pareto", n_q=PATH["n_q"], r_max=PATH["r_max"], eps=PATH["eps"]):
    """Ordered list of threshold vectors from 'always the reference' to the most aggressive admissible rule.
    pareto: each move advances the one stage whose next threshold lowers mean NFE with the best NFE drop per unit of
            mean-loss increase; greedy: stages are advanced one after another from the earliest (STAC's order)."""
    cands, ref, S, L, C = _compile(rows, G, name)
    M = len(cands)
    lad = _ladders(S, M, n_q)
    idx = [-1] * M

    def lams_of(ix):
        return [lad[m][ix[m]] if ix[m] >= 0 else None for m in range(M)]

    path = [lams_of(idx)]
    c, rk = _cost_risk(S, L, C, lams_of(idx))
    if mode == "pareto":
        while True:
            best = None
            for m in range(M):
                j = idx[m] + 1
                while j < len(lad[m]):
                    trial = list(idx)
                    trial[m] = j
                    c2, r2 = _cost_risk(S, L, C, lams_of(trial))
                    if c2 < c - 1e-9:
                        if r2 <= r_max:
                            eff = (c - c2) / (max(r2 - rk, 0.0) + eps)
                            if best is None or eff > best[0] + 1e-12:
                                best = (eff, m, j, c2, r2)
                        break
                    j += 1
            if best is None:
                break
            _, m, j, c, rk = best
            idx[m] = j
            path.append(lams_of(idx))
    elif mode == "greedy":
        for m in range(M):
            j = idx[m] + 1
            while j < len(lad[m]):
                trial = list(idx)
                trial[m] = j
                c2, r2 = _cost_risk(S, L, C, lams_of(trial))
                if r2 > r_max:
                    break
                if c2 < c - 1e-9:
                    idx, c, rk = trial, c2, r2
                    path.append(lams_of(idx))
                j += 1
    else:
        raise ValueError(mode)
    return path


def make_policy(G, name, lams):
    g = sw.GRIDS_EXT[G]
    cands, ref = list(g["cands"]), g["ref"]

    def pol(r):
        for m, o in enumerate(cands):
            if lams[m] is not None and signal(r, G, o, name) >= lams[m]:
                return o
        return ref
    return pol


def crc_index(rows, G, name, path, alpha=ALPHA):
    """Largest path index t with (1 + sum_s Lmono_s(t)) / (n + 1) <= alpha, where Lmono_s(t) = max_{t' <= t} L_s(t') is the
    speaker-level monotone envelope (Angelopoulos et al., 2024, conformal risk control with bound B = 1)."""
    ref = sw.GRIDS_EXT[G]["ref"]
    spk = by_speaker(rows)
    n = len(spk)
    env = {s: 0.0 for s in spk}
    t_hat, trace = 0, []
    for t, lams in enumerate(path):
        pol = make_policy(G, name, lams)
        for s, rs in spk.items():
            env[s] = max(env[s], statistics.mean(loss(r, pol(r), ref) for r in rs))
        bound = (1.0 + sum(env.values())) / (n + 1.0)
        trace.append(bound)
        if bound <= alpha:
            t_hat = t
        else:
            break
    return t_hat, trace


def split_speakers(rows, share=PATH["path_share"]):
    spks = sorted({r["spk"] for r in rows})
    design = {s for i, s in enumerate(spks) if i % round(1.0 / share) == 0}
    return [r for r in rows if r["spk"] in design], [r for r in rows if r["spk"] not in design]


def fit_gpc(train, G, name, mode="pareto", crc=True):
    """Proposed rule (and its ablations). Returns (policy, info)."""
    if crc:
        design, cal = split_speakers(train)
        path = build_path(design, G, name, mode=mode)
        t_hat, trace = crc_index(cal, G, name, path)
    else:
        path = build_path(train, G, name, mode=mode)
        ref = sw.GRIDS_EXT[G]["ref"]
        t_hat, trace = 0, []
        for t, lams in enumerate(path):
            pol = make_policy(G, name, lams)
            risk = statistics.mean(loss(r, pol(r), ref) for r in train)
            trace.append(risk)
            if risk <= ALPHA:
                t_hat = t
    lams = path[t_hat]
    return make_policy(G, name, lams), {"signal": name, "mode": mode, "crc": crc, "path_len": len(path), "t_hat": t_hat,
                                        "lams": dict(zip(sw.GRIDS_EXT[G]["cands"], lams)), "bound_trace_tail": trace[-3:]}


def fit_named(name, train, G, sel):
    if name == PROPOSED:
        return fit_gpc(train, G, sel)
    if name == "gpc_A":
        return fit_gpc(train, G, "A")
    if name == "gpc_greedy":
        return fit_gpc(train, G, sel, mode="greedy")
    if name == "gpc_nocrc":
        return fit_gpc(train, G, sel, crc=False)
    pol, p = sw.fold_policy_direct(name, train, G)
    return pol, {"threshold": p}


def test_risk(rows, pick, G):
    ref = sw.GRIDS_EXT[G]["ref"]

    def stat(rs):
        spk = by_speaker(rs)
        return statistics.mean(statistics.mean(loss(r, pick[g32.key(r)], ref) for r in v) for v in spk.values())
    return stat(rows), boot_ci(rows, stat)


def evaluate_policies(dev, test, G, sel, names):
    out = {}
    g = sw.GRIDS_EXT[G]
    fixed = {f"fixed {g['ref']}": g["ref"], f"fixed {g['cands'][-1]}": g["cands"][-1]}
    for label, o in fixed.items():
        pick = {g32.key(r): o for r in test}
        out[label] = {**sw.summarize_direct(test, pick, G), "risk": test_risk(test, pick, G)}
    for name in names:
        t0 = time.perf_counter()
        pol, info = fit_named(name, dev, G, sel)
        pick = {g32.key(r): pol(r) for r in test}
        out[name] = {**sw.summarize_direct(test, pick, G), "risk": test_risk(test, pick, G), "fit": info,
                     "fit_seconds": round(time.perf_counter() - t0, 2)}
    return out


def h4_verdict(res):
    p = res[PROPOSED]
    stac = res.get("stac")

    def quality_ok(v):
        a = (v.get("wer_tost_p") is not None and v["wer_tost_p"] < 0.05 and
             v.get("similarity_tost_p") is not None and v["similarity_tost_p"] < 0.05)
        ref_u = res[next(k for k in res if k.startswith("fixed ") and k.endswith(sw.GRIDS_EXT[PRIMARY_GRID]["ref"]))]
        b = (ref_u["utmos_clean"] - v["utmos_clean"]) <= H4["utmos_clean_drop"]
        return a, b
    a, b = quality_ok(p)
    c = p["risk"][0] <= ALPHA
    if stac is not None:
        sa, sb = quality_ok(stac)
        d = p["saving"] >= stac["saving"] - H4["saving_margin"] or not (sa and sb)
    else:
        d = None
    necessity = {}
    for ab in ABLATIONS:
        if ab in res:
            aa, bb = quality_ok(res[ab])
            cc = res[ab]["risk"][0] <= ALPHA
            necessity[ab] = {"saving": res[ab]["saving"], "quality_ok": aa and bb and cc,
                             "needed": (res[ab]["saving"] <= p["saving"] - 0.01) or not (aa and bb and cc)}
    return {"a_tost": a, "b_utmos": b, "c_risk": c, "d_vs_stac": d, "pass": bool(a and b and c and (d is not False)),
            "necessity": necessity}


def ood_summary(dev, ood, G, sel):
    res = {}
    for name in (PROPOSED, "stac", "snr_casc"):
        pol, info = fit_named(name, dev, G, sel)
        pick = {g32.key(r): pol(r) for r in ood}
        s = sw.summarize_direct(ood, pick, G)
        ref = sw.GRIDS_EXT[G]["ref"]

        def dwer(rs, pick=pick):
            return statistics.mean(r["m"][pick[g32.key(r)]]["wer"] - r["m"][ref]["wer"] for r in rs)
        per = {}
        for c in sorted({r["cond"] for r in ood}):
            sub = [r for r in ood if r["cond"] == c]
            per[c] = {"n": len(sub), "nfe": statistics.mean(sw.COST_ALL[pick[g32.key(r)]] for r in sub),
                      "d_wer": dwer(sub), "snr": statistics.mean(r["snr"] for r in sub)}
        res[name] = {"saving": s["saving"], "mean_cost": s["mean_cost"], "d_wer": dwer(ood), "d_wer_ci": boot_ci(ood, dwer),
                     "per_condition": per}
    p, snr = res[PROPOSED], res["snr_casc"]
    res["verdict"] = {"pass": (p["d_wer_ci"][1] is not None and p["d_wer_ci"][1] <= 0.01 and p["saving"] >= snr["saving"])}
    return res


def dev_cv(dev, G, sel, names):
    fold = g32.folds_of(dev)
    out = {}
    for name in names:
        pick = {}
        for f in range(sw.FOLDS):
            train = [r for r in dev if fold[r["spk"]] != f]
            pol, _ = fit_named(name, train, G, sel)
            pick.update({g32.key(r): pol(r) for r in dev if fold[r["spk"]] == f})
        out[name] = {**sw.summarize_direct(dev, pick, G), "risk": test_risk(dev, pick, G)}
    return out


# ============================================================================= analysis: driver and report
def analyze(out, prereg_path=PREREG_PATH, labels_dev=STEPS1TO8_CSV, with_dev_cv=True):
    out = Path(out)
    pre = json.loads(Path(prereg_path).read_text(encoding="utf-8")) if Path(prereg_path).exists() else None
    res = {"prereg_sha256": pre["sha256"] if pre else None, "parts": {}}
    tel_dev = out / "telemetry_dev.csv"
    if not tel_dev.exists():
        raise SystemExit(f"Missing {tel_dev}: run the `dev` part first.")
    dev = join(labels_dev, load_telemetry(tel_dev))
    res["parts"]["dev"] = {"n": len(dev), "reproduction": reproduction(dev), "identity": identity_check(dev),
                           "mechanism": mechanism(dev),
                           "signal_ms": statistics.mean(r["tel"]["sig_ms16"] for r in dev),
                           "teacher_s16": statistics.mean(r["tel"]["t_teacher16"] for r in dev)}
    aucs = {name: mean_early_auc(dev, name) for name in list(SIGNAL_FAMILY) + list(BASELINE_SIGNALS)}
    sel = max(SIGNAL_FAMILY, key=lambda n: (aucs[n] if aucs[n] is not None else -1.0, -list(SIGNAL_FAMILY).index(n)))
    res["selection"] = {"dev_mean_auc": aucs, "selected": sel}
    names = (PROPOSED,) + tuple(ABLATIONS) + BASELINES
    if with_dev_cv:
        res["dev_cv"] = {PRIMARY_GRID: dev_cv(dev, PRIMARY_GRID, sel, (PROPOSED,) + tuple(ABLATIONS) + ("stac", "snr_casc"))}

    fresh_csv, tel_fresh = out / "fresh_steps.csv", out / "telemetry_fresh.csv"
    if fresh_csv.exists() and tel_fresh.exists():
        fresh = join(fresh_csv, load_telemetry(tel_fresh))
        part = {"n": len(fresh), "speakers": len({r["spk"] for r in fresh}), "reproduction": reproduction(fresh),
                "identity": identity_check(fresh), "mechanism": mechanism(fresh)}
        h3 = {name: mean_early_auc(fresh, name) for name in (sel, "A", "snr")}

        def diff(rs, other):
            a, b = mean_early_auc(rs, sel), mean_early_auc(rs, other)
            return None if a is None or b is None else a - b
        h3["diff_vs_A"] = diff(fresh, "A")
        h3["diff_vs_A_ci"] = boot_ci(fresh, lambda rs: diff(rs, "A"))
        h3["diff_vs_snr"] = diff(fresh, "snr")
        h3["diff_vs_snr_ci"] = boot_ci(fresh, lambda rs: diff(rs, "snr"))
        h3["pass"] = h3["diff_vs_A_ci"][0] is not None and h3["diff_vs_A_ci"][0] > 0
        part["H3"] = h3
        part["policies"] = {PRIMARY_GRID: evaluate_policies(dev, fresh, PRIMARY_GRID, sel, names)}
        part["H4"] = h4_verdict(part["policies"][PRIMARY_GRID])
        part["policies"]["g16_all8"] = evaluate_policies(dev, fresh, "g16_all8", sel, (PROPOSED, "stac", "snr_casc"))
        if all(r["has_g32_all"] for r in fresh) and fresh:
            part["policies"]["g32_floor14"] = evaluate_policies(dev, fresh, "g32_floor14", sel, (PROPOSED, "stac", "snr_casc"))
        res["parts"]["fresh"] = part

    ood_csv, tel_ood = out / "ood_steps.csv", out / "telemetry_ood.csv"
    if ood_csv.exists() and tel_ood.exists():
        ood = join(ood_csv, load_telemetry(tel_ood))
        res["parts"]["ood"] = {"n": len(ood), "reproduction": reproduction(ood), "identity": identity_check(ood),
                               "H5": ood_summary(dev, ood, PRIMARY_GRID, sel)}

    (out / "guidance_results.json").write_text(json.dumps(res, indent=2, default=str) + "\n", encoding="utf-8")
    (out / "GUIDANCE_WORK_REPORT.md").write_text(render(res), encoding="utf-8")
    print(f"[OK] Wrote {out / 'GUIDANCE_WORK_REPORT.md'} and {out / 'guidance_results.json'}", flush=True)
    return res


def _p(x, fmt=".3f"):
    return "n/a" if x is None else format(x, fmt)


def _ci(ci, fmt=".3f"):
    return "n/a" if not ci or ci[0] is None else f"[{format(ci[0], fmt)}, {format(ci[1], fmt)}]"


def _policy_table(pol):
    L = ["| Policy | Mean NFE (saving) | E2E time (speedup) | WER (dWER) | Capped WER | SIM (dSIM) | UTMOS clean | "
         "TOST p WER / SIM | Test risk [95% CI] | Picks |",
         "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for name, v in pol.items():
        label = {PROPOSED: "**GPC (proposed)**", **{k: f"Ablation `{k}`" for k in ABLATIONS}}.get(name, sw.ALL_SCHEDULERS.get(name, name))
        rt = f"{v['runtime']:.2f}s ({v.get('speedup', 1.0):.2f}x)" if v.get("runtime") else "n/a"
        picks = ", ".join(f"{k}:{c}" for k, c in v["picks"].items())
        L.append(f"| {label} | {v['mean_cost']:.2f} ({v['saving']:.1%}) | {rt} | {v['wer_all']:.4f} ({v['d_wer_all']:+.4f}) | "
                 f"{_p(v.get('wer_capped_all'), '.4f')} | {v['sim_all']:.4f} ({v['d_sim_all']:+.4f}) | {_p(v.get('utmos_clean'))} | "
                 f"{_p(v.get('wer_tost_p'), '.2g')} / {_p(v.get('similarity_tost_p'), '.2g')} | "
                 f"{_p(v['risk'][0], '.4f')} {_ci(v['risk'][1], '.4f')} | `{picks}` |")
    return L


def render(res):
    L = ["# EXP-019: Guidance-Decomposed Trajectory Telemetry and Risk-Controlled Handover", "",
         f"Preregistration sha256 `{res.get('prereg_sha256')}`. Every number is computed from the CSV files by "
         "`run_guidance_work_prelim.py analyze`. Development evidence only.", ""]
    for part in ("dev", "fresh", "ood"):
        P = res["parts"].get(part)
        if not P:
            continue
        L += [f"## Part `{part}` ({P['n']} cases)", ""]
        rep, idt = P["reproduction"], P["identity"]
        L += [f"- Reproduction of the stored drift: max |diff| d16 = {_p(rep['max_abs_diff_d16'], '.2e')}, "
              f"d32 = {_p(rep['max_abs_diff_d32'], '.2e')}",
              f"- **H1 (Lemma 1)**: max relative residual {_p(idt['max_residual'], '.2e')}, max split error "
              f"{_p(idt['max_split_rel_error'], '.2e')} -> **{'PASS' if idt['pass'] else 'FAIL'}**"]
        if part == "dev":
            L.append(f"- Signal arithmetic: {P['signal_ms']:.2f} ms per 16-grid trajectory (all telemetry; teacher "
                     f"{P['teacher_s16']:.2f} s), zero extra forward passes")
        L.append("")
        if "mechanism" in P:
            M = P["mechanism"]
            L += [f"**H2 (mechanism)**: guidance share of the noisy-minus-clean change in d^2 at steps {list(H2_STEPS)}: "
                  f"S_g = {_p(M['S_g'])} {_ci(M['S_g_ci'])} -> **{M['verdict']}**", "",
                  "| Condition | n | gc_0 | gc_4 | " + " | ".join(f"d_{j} | p^c_{j} | p^g_{j} | p^x_{j}" for j in H2_STEPS) +
                  " | D_4 | S_4 |",
                  "|---|---:|---:|---:|" + "---:|" * (4 * len(H2_STEPS)) + "---:|---:|"]
            for c, v in M["per_condition"].items():
                L.append(f"| {c} | {v['n']} | {v['gc0']:.3f} | {v['gc4']:.3f} | " +
                         " | ".join(f"{v[f'd_{j}']:.3f} | {v[f'pc_{j}']:.4f} | {v[f'pg_{j}']:.4f} | {v[f'px_{j}']:+.4f}"
                                    for j in H2_STEPS) + f" | {v['D4']:.3f} | {v['S4']:.3f} |")
            L.append("")
        if part == "dev":
            S = res["selection"]
            L += ["**Signal selection (dev only)**: mean AUC over 16-grid steps 1-4 for the safe label", "",
                  "| Signal | Mean AUC |", "|---|---:|"]
            for n, a in S["dev_mean_auc"].items():
                L.append(f"| `{n}` {'**(selected)**' if n == S['selected'] else ''} | {_p(a)} |")
            L.append("")
            if res.get("dev_cv"):
                L += [f"**Development 5-fold speaker CV ({PRIMARY_GRID})**", ""] + _policy_table(res["dev_cv"][PRIMARY_GRID]) + [""]
        if part == "fresh":
            h3 = P["H3"]
            L += [f"**H3 (prediction on new speakers)**: mean AUC steps 1-4: selected `{res['selection']['selected']}` "
                  f"{_p(h3[res['selection']['selected']])}, A_k {_p(h3['A'])}, SNR {_p(h3['snr'])}; difference vs A_k "
                  f"{_p(h3['diff_vs_A'], '+.3f')} {_ci(h3['diff_vs_A_ci'], '+.3f')}, vs SNR {_p(h3['diff_vs_snr'], '+.3f')} "
                  f"{_ci(h3['diff_vs_snr_ci'], '+.3f')} -> **{'PASS' if h3['pass'] else 'FAIL'}**", ""]
            for G, pol in P["policies"].items():
                L += [f"**Policies calibrated once on dev, evaluated on the new speakers ({G}: {sw.GRIDS_EXT[G]['label']})**", ""]
                L += _policy_table(pol) + [""]
            h4 = P["H4"]
            L += [f"**H4**: (a) TOST {h4['a_tost']}, (b) clean UTMOS {h4['b_utmos']}, (c) risk {h4['c_risk']}, (d) vs STAC "
                  f"{h4['d_vs_stac']} -> **{'PASS' if h4['pass'] else 'FAIL'}**", "",
                  "| Ablation | Saving | Meets (a)-(c) | Component needed |", "|---|---:|---|---|"]
            for ab, v in h4["necessity"].items():
                L.append(f"| `{ab}` ({ABLATIONS[ab]}) | {v['saving']:.1%} | {v['quality_ok']} | {v['needed']} |")
            L.append("")
        if part == "ood":
            H5 = P["H5"]
            L += ["**H5 (zero-shot degraded prompts, new speakers)**", "",
                  "| Policy | Saving | dWER [95% CI] | " + " | ".join(f"{c} NFE / dWER" for c in H5[PROPOSED]["per_condition"]) + " |",
                  "|---|---:|---:|" + "---:|" * len(H5[PROPOSED]["per_condition"])]
            for name in (PROPOSED, "stac", "snr_casc"):
                v = H5[name]
                L.append(f"| {name} | {v['saving']:.1%} | {v['d_wer']:+.4f} {_ci(v['d_wer_ci'], '+.4f')} | " +
                         " | ".join(f"{q['nfe']:.2f} / {q['d_wer']:+.4f}" for q in v["per_condition"].values()) + " |")
            L += ["", f"-> **{'PASS' if H5['verdict']['pass'] else 'FAIL'}**", ""]
    L += ["Calibration used the development speakers only; the new speakers were synthesized after the preregistration. "
          "They are not the thesis's frozen held-out evaluation (EVAL-001/EVAL-002)."]
    return "\n".join(L) + "\n"


# ============================================================================= main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["freeze", "smoke", "dev", "fresh", "ood", "analyze"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--prereg", default=str(PREREG_PATH))
    ap.add_argument("--dmo_root", default="/content/Shonget-DMOSpeech2")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n_fresh", type=int, default=FRESH_N)
    ap.add_argument("--with_g32", action="store_true", help="fresh: also synthesize and score the 32-grid Steps 1..15")
    ap.add_argument("--demand_dir", default=None, help="ood: DEMAND directory with PCAFETER/ch01.wav (adds babble15/20)")
    ap.add_argument("--no_dev_cv", action="store_true")
    ap.add_argument("--any_gpu", action="store_true")
    args = ap.parse_args(argv)
    out = Path(args.out)
    if args.cmd == "freeze":
        freeze(out)
        return
    if args.cmd == "analyze":
        analyze(out, args.prereg, with_dev_cv=not args.no_dev_cv)
        return
    pre = check_prereg(args.prereg)
    out.mkdir(parents=True, exist_ok=True)
    (out / "preregistration.json").write_text(json.dumps(pre, indent=2) + "\n", encoding="utf-8")
    sw._ensure_colab_py313_compat()
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    pipe = rf.Pipeline(args.dmo_root, args.device)
    if "T4" not in pipe.gpu and not args.any_gpu:
        raise SystemExit(f"Expected a Tesla T4 (found {pipe.gpu}): the stored EXP-018 labels and drift would not reproduce.")
    if args.cmd == "smoke":
        sm = out / "smoke"
        its = [it for it in cg.dev_items(DEV_N) if it["utt_id"] not in cg.pilot_utts()][:2]
        run_telemetry(pipe, its, ("clean", "gauss20"), sm, "smoke", "telemetry_dev.csv")
        stored = {(r["utt_id"], r["condition"]): r for r in rf.read_csv(STEPS1TO8_CSV)}
        for r in rf.read_csv(sm / "telemetry_dev.csv"):
            s = stored[(r["utt_id"], r["condition"])]
            m16 = max(abs(float(r[f"g16_d_{k}"]) - float(s[f"d16_{k}"])) for k in sw.ALL_G16_STEPS)
            m32 = max(abs(float(r[f"g32_d_{k}"]) - float(s[f"d32_{k}"])) for k in sw.ALL_G32_STEPS)
            res = max(float(r[f"g32_res_{i}"]) for i in range(1, K32 + 1))
            print(f"reproduction {r['utt_id'][-8:]} {r['condition']}: max |d16 - stored| {m16:.2e}, max |d32 - stored| "
                  f"{m32:.2e}; Lemma-1 residual {res:.2e}; signals {r['sig_ms16']} ms (16-grid)", flush=True)
        sw.run_steps1to8_cases(pipe, its[:1], ("clean",), sm, NO_STORED, include_g32_all=args.with_g32, csv_name="fresh_steps.csv")
        t = rf.read_csv(sm / "fresh_steps.csv")[0]
        per_case = float(t["case_seconds"]) + 25 * 0.3
        print(f"Smoke OK. One fully synthesized case took {float(t['case_seconds']):.0f} s -> fresh part "
              f"~{per_case * args.n_fresh * len(CONDS) / 3600:.1f} h, ood part ~{per_case * args.n_fresh * len(OOD_CONDS) / 3600:.1f} h "
              f"(+ dev telemetry ~{360 * 12 / 3600:.1f} h)", flush=True)
        return
    if args.cmd == "dev":
        run_telemetry(pipe, cg.dev_items(DEV_N), CONDS, out, "dev", "telemetry_dev.csv")
        return
    items = fresh_items(args.n_fresh)
    if args.cmd == "fresh":
        run_telemetry(pipe, items, CONDS, out, "fresh", "telemetry_fresh.csv")
        sw.run_steps1to8_cases(pipe, items, CONDS, out, NO_STORED, include_g32_all=args.with_g32, csv_name="fresh_steps.csv")
        return
    if args.cmd == "ood":
        conds = OOD_CONDS
        if args.demand_dir:
            pipe.load_babble(args.demand_dir)
            conds = OOD_CONDS + BABBLE_CONDS
        run_telemetry(pipe, items, conds, out, "ood", "telemetry_ood.csv")
        sw.run_steps1to8_cases(pipe, items, conds, out, NO_STORED, include_g32_all=False, csv_name="ood_steps.csv")
        return


if __name__ == "__main__":
    main()

