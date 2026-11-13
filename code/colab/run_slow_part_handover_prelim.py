"""Development test of Slow-Part Handover (EXP-022).

Slow-Part Handover: the guided teacher runs one step at a time. After each step k its endpoint estimate T_k is split, on
generated frames, into voice (each band's average over time), slow (a 9-frame moving average of the rest: timing and
pitch movement) and fine (what is left), exactly as in EXP-020. From k = 2 on, the teacher stops at the first step where
its slow part has settled,

    d_slow_k = ||slow(T_k) - slow(T_k-1)|| / ||slow(T_k)|| <= lambda,

and hands over only that part: x = voice(S) + slow(T_k) + fine(S), with S a one-pass student draft from independent
noise. The released student then finishes from x as usual. If no step up to 7 qualifies, the teacher continues to
Step 8 and hands over T_8 whole, exactly as the 0.25 reference does. Passes: 2(k + 1) + 1 + 3 = 2k + 6 at step k
(10 at Step 2, 14 at Step 4), 21 for the fallback. No model is trained; lambda is the only calibrated number.

This run synthesizes the handover at every Step 2..7 for the 60 development speakers x 6 prompt conditions (360 cases),
records the voice, slow, fine and whole-estimate changes at every teacher step (no extra passes), and measures pitch
variety at Steps 2, 3 and 4. The stop rule is then evaluated offline with 5-fold speaker cross-validation for any
lambda. Development evidence only: these speakers designed EXP-020.

Subcommands: freeze (write the preregistration once), smoke, run (resumable), analyze.
"""
import argparse
import hashlib
import json
import math
import random
import shutil
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_followup_tests as rf  # noqa: E402
import run_consistency_gate_prelim as cg  # noqa: E402
import run_endpoint_extrapolation_prelim as ex  # noqa: E402
import run_noise_guidance_prelim as ng  # noqa: E402
import run_grid32_prelim as gl  # noqa: E402
import run_selective_handover_prelim as sh  # noqa: E402

THESIS_ROOT = rf.THESIS_ROOT
EXP_DIR = THESIS_ROOT / "research/experiments/EXP-022-slow-part-handover-prelim"
PREREG_PATH = EXP_DIR / "preregistration.json"
EXP020_CSV = THESIS_ROOT / "research/experiments/EXP-020-selective-handover-prelim/results/selective.csv"
STEPS_CSV, GRID32_CSV = sh.STEPS_CSV, sh.GRID32_CSV
DMO_COMMIT = "8a7868fc3345702a586173447f4869205b501f26"
DEPENDENCIES = ("run_slow_part_handover_prelim.py", "run_selective_handover_prelim.py", "run_grid32_prelim.py",
                "run_noise_guidance_prelim.py", "run_followup_tests.py", "run_consistency_gate_prelim.py",
                "run_endpoint_extrapolation_prelim.py")

CODE = "STS"                          # (voice, slow, fine) = (student, teacher, student)
STEPS = (2, 3, 4, 5, 6, 7)            # candidate handover steps; Step 8 is the fallback
REF_STEP = 8
SIGNAL_STEPS = tuple(range(1, REF_STEP + 1))
PARTS = ("voice", "slow", "fine", "full")
OUTS = tuple(f"STS_{k}" for k in STEPS)
METRICS = ("wer", "similarity", "utmos")
N_SPEAKERS, CONDS, NOISY = ex.N_SPEAKERS, ex.CONDS, ex.NOISY
FOLDS, SEED = ex.FOLDS, ex.SEED
DIV_N, DIV_SEEDS, DIV_SEED_BASE = 20, 20, 1000
DIV_CONDS = ("clean", "noisy15")
EARLY = (2, 3)
DIV_OUTS = ("STS_2", "STS_3", "STS_4", "plain_8")
BOOT_B = 2000
CALIBRATION = {"wer": 0.01, "similarity": 0.01}     # training-fold mean change against Step 8 allowed by the stop rule
MARGINS = {"wer": 0.025, "similarity": 0.01, "cvf0": 0.02}
Q_GRID = [i / 40 for i in range(1, 40)]
REPRODUCTION_TOLERANCE, DRIFT_TOLERANCE = sh.REPRODUCTION_TOLERANCE, 1e-3
FIELDS = ["utt_id", "speaker_id", "condition", "prompt_snr", "gpu"] + [f"d_{p}_{k}" for p in PARTS for k in SIGNAL_STEPS] + \
         [f"{m}_{o}" for o in OUTS for m in METRICS + ("hypothesis",)] + ["case_seconds", "utc"]
DIV_FIELDS = ["utt_id", "speaker_id", "condition", "n_seeds"] + [f"cvf0_{o}" for o in DIV_OUTS] + \
             [f"voiced_{o}" for o in DIV_OUTS] + ["case_seconds", "utc"]


def cost(name):
    """Forward passes: 2 per guided teacher evaluation, 1 for the draft, 1 per student step."""
    if name == "student4":
        return 4
    if name == "STT_4":
        return 14
    base, k = name.rsplit("_", 1)
    return 2 * (int(k) + 1) + 3 + (0 if base == "plain" else 1)


# --------------------------------------------------------------------------- signals
def _norm(x):
    return float((x * x).sum()) ** 0.5


def part_signals(E, plen):
    """Relative change of each part, and of the whole estimate, between consecutive teacher steps (generated frames).
    d_full_k is the sampler's own stabilization drift."""
    split = {k: sh.decompose(E[k][0, plen:, :]) for k in range(0, REF_STEP + 1)}
    out = {}
    for k in SIGNAL_STEPS:
        for i, p in enumerate(("voice", "slow", "fine")):
            a, b = split[k][i], split[k - 1][i]
            out[f"d_{p}_{k}"] = _norm(a - b) / (_norm(a) + 1e-8)
        a, b = E[k][0, plen:, :], E[k - 1][0, plen:, :]
        out[f"d_full_{k}"] = _norm(a - b) / (_norm(a) + 1e-8)
    return out


# --------------------------------------------------------------------------------- runs
def stored_reproduction():
    """(utt_id, condition) -> stored STS_4 scores (EXP-020) and teacher drift d_1..d_8 (EXP-018) for the smoke test."""
    steps = {(r["utt_id"], r["condition"]): r for r in rf.read_csv(STEPS_CSV)}
    out = {}
    for r in rf.read_csv(EXP020_CSV):
        key = (r["utt_id"], r["condition"])
        if key in steps:
            out[key] = {"STS_4": {"wer": float(r["wer_STS_4"]), "similarity": float(r["similarity_STS_4"])},
                        "drift": {k: float(steps[key][f"d16_{k}"]) for k in SIGNAL_STEPS}}
    return out


def run_quality(pipe, items, conds, out_dir, csv_name="slowpart.csv", stored=None):
    """Resumable, one row per (utterance, condition): the part signals at every teacher step and the handover output
    at every Step 2..7. Returns reproduction rows and drift differences when stored values are given."""
    path = Path(out_dir) / csv_name
    done = rf.done_keys(path, 1)
    todo = [(it, c) for it in items for c in conds if (it["utt_id"], c) not in done]
    print(f"[quality] {len(todo)} cases to run ({len(done)} already in {path.name}) on {pipe.gpu}", flush=True)
    taus, repro, drift_diff = list(pipe.sampler.STUDENT_TAUS), [], []
    for n, (item, cond) in enumerate(todo, 1):
        t0 = time.perf_counter()
        ctx = pipe.prepare(item, cond, item["input_seed"], out_dir)
        plen = ctx["prompt_len"]
        E, _ = pipe.teacher(ctx, tuple(range(REF_STEP + 1)))
        S = sh.student_sample(pipe, ctx)
        with pipe.torch.inference_mode():
            sig = part_signals(E, plen)
            inputs = {f"STS_{k}": sh.selective_handover(E[k], S, plen, CODE) for k in STEPS}
        gt = pipe.ground_truth(item)
        row = {"utt_id": item["utt_id"], "speaker_id": item["speaker_id"], "condition": cond, "prompt_snr": ctx["prompt_snr"],
               "gpu": pipe.gpu, **sig}
        for o in OUTS:
            m = pipe.score(ng.decode(pipe, ctx, gl.student(pipe, ctx, inputs[o], taus)), item, gt)
            row.update({f"{k}_{o}": m[k] for k in METRICS + ("hypothesis",)})
        if stored is not None and (item["utt_id"], cond) in stored:
            s = stored[(item["utt_id"], cond)]
            repro.append({"utt_id": item["utt_id"], "condition": cond, "output": "STS_4", "wer": row["wer_STS_4"],
                          "wer_stored": s["STS_4"]["wer"], "similarity": row["similarity_STS_4"], "similarity_stored": s["STS_4"]["similarity"]})
            drift_diff.append(max(abs(sig[f"d_full_{k}"] - s["drift"][k]) for k in SIGNAL_STEPS))
        row.update({"case_seconds": round(time.perf_counter() - t0, 2), "utc": sh._utc()})
        sh._append(path, FIELDS, row)
        print(f"[{n}/{len(todo)}] quality: {item['utt_id']} {cond} ({row['case_seconds']:.0f} s)", flush=True)
    print(f"[OK] quality complete: {path}", flush=True)
    return repro, drift_diff


def run_diversity(pipe, items, conds, out_dir, seeds=DIV_SEEDS, csv_name="diversity.csv"):
    """CVf0 across seeds of the handover at Steps 2, 3 and 4 and of the reference, as in EXP-020 (prompt degradation
    fixed per pair, synthesis seed = input_seed + 1000 + m)."""
    path = Path(out_dir) / csv_name
    done = rf.done_keys(path, 1)
    todo = [(it, c) for c in conds for it in items if (it["utt_id"], c) not in done]
    print(f"[diversity] {len(todo)} pairs x {seeds} seeds to run ({len(done)} already in {path.name})", flush=True)
    taus = list(pipe.sampler.STUDENT_TAUS)
    for n, (item, cond) in enumerate(todo, 1):
        t0 = time.perf_counter()
        tracks = {o: [] for o in DIV_OUTS}
        for m in range(seeds):
            ctx = pipe.prepare(item, cond, int(item["input_seed"]) + DIV_SEED_BASE + m, out_dir, noise_seed=item["input_seed"])
            plen = ctx["prompt_len"]
            E, _ = pipe.teacher(ctx, (2, 3, 4, REF_STEP))
            S = sh.student_sample(pipe, ctx)
            with pipe.torch.inference_mode():
                ins = {f"STS_{k}": sh.selective_handover(E[k], S, plen, CODE) for k in (2, 3, 4)}
            xs = {**{o: gl.student(pipe, ctx, h, taus) for o, h in ins.items()}, "plain_8": gl.student(pipe, ctx, E[REF_STEP], taus)}
            for o in DIV_OUTS:
                tracks[o].append(sh.f0_track(ng.decode(pipe, ctx, xs[o])))
        row = {"utt_id": item["utt_id"], "speaker_id": item["speaker_id"], "condition": cond, "n_seeds": seeds}
        for o in DIV_OUTS:
            row[f"cvf0_{o}"], row[f"voiced_{o}"] = sh.cvf0(tracks[o])
        row.update({"case_seconds": round(time.perf_counter() - t0, 2), "utc": sh._utc()})
        sh._append(path, DIV_FIELDS, row)
        print(f"[{n}/{len(todo)}] diversity: {item['utt_id']} {cond} ({row['case_seconds']:.0f} s)", flush=True)
    print(f"[OK] diversity complete: {path}", flush=True)


# ----------------------------------------------------------------------------- analysis
def load_stored():
    """(utt, cond) -> plain Steps 1..8 (EXP-018), student only (EXP-017) and Voice-Swap STT_4 (EXP-020)."""
    steps = {(r["utt_id"], r["condition"]): r for r in rf.read_csv(STEPS_CSV)}
    grid = {(r["utt_id"], r["condition"]): r for r in rf.read_csv(GRID32_CSV)}
    vs = {(r["utt_id"], r["condition"]): r for r in rf.read_csv(EXP020_CSV)}
    out = {}
    for key, r in steps.items():
        if key in grid and key in vs:
            m = {f"plain_{k}": {x: float(r[f"{x}_g16_{k}"]) for x in METRICS} for k in range(1, REF_STEP + 1)}
            m["student4"] = {x: float(grid[key][f"{x}_student4"]) for x in METRICS}
            m["STT_4"] = {x: float(vs[key][f"{x}_STT_4"]) for x in METRICS}
            out[key] = {"speaker_id": r["speaker_id"], "prompt_snr": float(r["prompt_snr"]), "m": m}
    return out


def build_table(stored, rows, th):
    table = []
    for r in rows:
        s = stored.get((r["utt_id"], r["condition"]))
        if s is None:
            continue
        m = {**s["m"], **{o: {x: float(r[f"{x}_{o}"]) for x in METRICS} for o in OUTS}}
        sig = {p: {k: float(r[f"d_{p}_{k}"]) for k in SIGNAL_STEPS} for p in PARTS}
        table.append({"utt": r["utt_id"], "spk": s["speaker_id"], "cond": r["condition"], "snr": s["prompt_snr"], "m": m, "sig": sig,
                      "fail": {n: bool(gl.failed(v, m["plain_8"], th)) for n, v in m.items()}})
    return table


def stop_rule(signal, lam, steps):
    """The first candidate step whose part change is at most lam; otherwise the Step-8 reference."""
    return lambda r: next((f"STS_{k}" for k in steps if r["sig"][signal][k] <= lam), "plain_8")


def _capped(r, name, metric):
    v = r["m"][name][metric]
    return min(1.0, v) if metric == "wer" else v


def within_calibration(train, pol):
    """Training-fold mean change against Step 8: WER at most +0.01, similarity at least -0.01."""
    picks = [(r, pol(r)) for r in train]
    dw = sum(_capped(r, k, "wer") - _capped(r, "plain_8", "wer") for r, k in picks) / len(picks)
    ds = sum(_capped(r, k, "similarity") - _capped(r, "plain_8", "similarity") for r, k in picks) / len(picks)
    return dw <= CALIBRATION["wer"] and ds >= -CALIBRATION["similarity"]


def lam_grid(train, signal, steps):
    values = sorted(r["sig"][signal][k] for r in train for k in steps)
    return [0.0] + [values[min(len(values) - 1, int(q * len(values)))] for q in Q_GRID]


def calibrate(train, signal, steps):
    """Fixed sequence from conservative (lambda 0: everything goes to Step 8) to aggressive; the last lambda before the
    first violation of the training constraint is used."""
    chosen = 0.0
    for lam in lam_grid(train, signal, steps):
        if not within_calibration(train, stop_rule(signal, lam, steps)):
            break
        chosen = lam
    return chosen


def cross_validate(table, signal, steps):
    fold = gl.folds_of(table)
    routed, lams = [], []
    for f in range(FOLDS):
        train = [r for r in table if fold[r["spk"]] != f]
        lam = calibrate(train, signal, steps)
        lams.append(lam)
        pol = stop_rule(signal, lam, steps)
        routed += [(r, pol(r)) for r in table if fold[r["spk"]] == f]
    return routed, lams


def boot(groups, B=BOOT_B, seed=SEED):
    values = [sum(v) / len(v) for v in groups.values() if v]
    if not values:
        return {"mean": None, "lo": None, "hi": None, "n": 0}
    rng = random.Random(seed)
    boots = sorted(sum(rng.choice(values) for _ in values) / len(values) for _ in range(B))
    return {"mean": sum(values) / len(values), "lo": boots[int(0.025 * B)], "hi": boots[int(0.975 * B)], "n": len(values)}


def paired(routed, metric, against="plain_8", conds=None):
    g = defaultdict(list)
    for r, k in routed:
        if conds is None or r["cond"] in conds:
            g[r["spk"]].append(_capped(r, k, metric) - _capped(r, against, metric))
    return boot(g)


def summarize(routed):
    n = len(routed)
    by = lambda conds: [x for x in routed if x[0]["cond"] in conds]
    return {"mean_passes": sum(cost(k) for _, k in routed) / n if n else None,
            "mean_passes_clean": (lambda s: sum(cost(k) for _, k in s) / len(s) if s else None)(by(("clean",))),
            "mean_passes_noisy": (lambda s: sum(cost(k) for _, k in s) / len(s) if s else None)(by(NOISY)),
            "stops": dict(sorted(Counter(k for _, k in routed).items())),
            "stops_by_condition": {c: dict(sorted(Counter(k for r, k in routed if r["cond"] == c).items())) for c in CONDS},
            "failures": sum(r["fail"][k] for r, k in routed), "n": n,
            "vs_reference": {m: paired(routed, m) for m in METRICS},
            "vs_reference_clean": {m: paired(routed, m, conds=("clean",)) for m in ("wer", "similarity")},
            "vs_reference_noisy": {m: paired(routed, m, conds=NOISY) for m in ("wer", "similarity")},
            "vs_voice_swap": {m: paired(routed, m, against="STT_4") for m in ("wer", "similarity")}}


def cv_diff(div, cond, a, b):
    g = {}
    for r in div:
        if r["condition"] == cond:
            x, y = float(r[f"cvf0_{a}"]), float(r[f"cvf0_{b}"])
            if not (math.isnan(x) or math.isnan(y)):
                g[r["speaker_id"]] = [x - y]
    return boot(g)


def early_steps_check(div):
    """{k: CVf0 of STS_k minus the reference per diversity condition, and whether k keeps pitch variety in every one}."""
    out = {}
    for k in EARLY:
        diffs = {c: cv_diff(div, c, f"STS_{k}", "plain_8") for c in DIV_CONDS}
        out[k] = {"diffs": diffs, "keeps": all(d["lo"] is not None and d["lo"] >= -MARGINS["cvf0"] for d in diffs.values())}
    return out


def analyze(out, allow_partial=False, prereg_path=PREREG_PATH):
    out = Path(out)
    pre = check_prereg(prereg_path)
    th = gl.thresholds()
    rows = rf.read_csv(out / "slowpart.csv", repair=True) if (out / "slowpart.csv").exists() else []
    div = rf.read_csv(out / "diversity.csv", repair=True) if (out / "diversity.csv").exists() else []
    table = build_table(load_stored(), rows, th)
    expected, expected_div = N_SPEAKERS * len(CONDS), DIV_N * len(DIV_CONDS)
    partial = len(table) < expected or len(div) < expected_div
    if partial and not allow_partial:
        raise SystemExit(f"{len(table)} of {expected} cases, {len(div)} of {expected_div} diversity pairs; finish the run or pass --allow_partial")
    complete_div = len(div) >= expected_div
    early = early_steps_check(div) if div else {k: {"diffs": {}, "keeps": None} for k in EARLY}
    if not complete_div:
        for k in EARLY:
            early[k]["keeps"] = None          # never drop a step on incomplete pitch-variety data
    steps = tuple(k for k in STEPS if k not in EARLY or early[k]["keeps"] is not False)
    res = {"prereg_sha256": pre["sha256"], "cases": len(table), "expected_cases": expected, "diversity_pairs": len(div),
           "expected_diversity_pairs": expected_div, "partial": partial, "candidate_steps": list(steps), "early_steps": early,
           "passes": {n: cost(n) for n in ("plain_8", "STT_4", "student4") + OUTS}}
    rules = {}
    if len({r["spk"] for r in table}) >= FOLDS:
        for signal in ("slow", "full", "voice"):
            routed, lams = cross_validate(table, signal, steps)
            rules[signal] = {"lambdas": lams, **summarize(routed)}
    res["rules"] = rules
    res["fixed"] = {n: summarize([(r, n) for r in table]) for n in ("plain_8", "STT_4") + OUTS} if table else {}
    res["cvf0_means"] = {c: {o: (lambda v: sum(v) / len(v) if v else None)([float(r[f"cvf0_{o}"]) for r in div if r["condition"] == c and not math.isnan(float(r[f"cvf0_{o}"]))])
                             for o in DIV_OUTS} for c in DIV_CONDS}
    res["signal_means"] = {c: {p: {k: (lambda v: sum(v) / len(v) if v else None)([r["sig"][p][k] for r in table if r["cond"] == c]) for k in SIGNAL_STEPS}
                               for p in PARTS} for c in CONDS}
    slow, full = rules.get("slow"), rules.get("full")
    c = {"C1_cheaper_than_voice_swap": None if not slow else slow["mean_passes"] < 14.0,
         "C2_quality": None if not slow else (slow["vs_reference"]["wer"]["hi"] <= MARGINS["wer"] and slow["vs_reference"]["similarity"]["lo"] >= -MARGINS["similarity"]),
         "C3_pitch_variety_at_step_3": early[3]["keeps"] if complete_div else None,
         "C4_slow_part_beats_whole_estimate": None if not (slow and full) else slow["mean_passes"] < full["mean_passes"]}
    res["criteria"], res["advance"] = c, all(v is True for v in c.values())
    (out / "slowpart_results.json").write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
    (out / "SLOW_PART_REPORT.md").write_text(render(res), encoding="utf-8")
    print(f"Wrote {out / 'SLOW_PART_REPORT.md'}", flush=True)
    return res


def _word(v):
    return "n/a" if v is None else ("PASS" if v else "FAIL")


def _ci(v, fmt="{:+.4f}"):
    return "n/a" if not v or v.get("mean") is None else f"{fmt.format(v['mean'])} [{fmt.format(v['lo'])}, {fmt.format(v['hi'])}]"


def _f(v, fmt="{:.2f}"):
    return "n/a" if v is None else fmt.format(v)


def render(res):
    c = res["criteria"]
    L = ["# Preliminary test: Slow-Part Handover (EXP-022)", "",
         f"{'**PARTIAL.** ' if res['partial'] else ''}Development speakers only; {res['cases']} of {res['expected_cases']} cases, "
         f"{res['diversity_pairs']} of {res['expected_diversity_pairs']} diversity pairs; preregistration `{res['prereg_sha256'][:16]}…`. "
         "Every number is computed from the CSV files by `run_slow_part_handover_prelim.py analyze`.", "",
         "The teacher stops at the first step k >= 2 where its slow part has settled (d_slow_k <= lambda) and hands over only that part; "
         "the student's one-pass draft supplies the voice part and the fine detail. Otherwise Step 8 as today. "
         f"Candidate steps after the pitch-variety check: {res['candidate_steps']}.", "",
         "## Preregistered criteria", "", "| " + " | ".join(c) + " | Advance |", "|" + "---|" * (len(c) + 1),
         "| " + " | ".join(_word(v) for v in c.values()) + f" | **{'YES' if res['advance'] else 'NO'}** |", ""]
    if res["rules"]:
        L += ["## Stop rules, out of fold (5-fold speaker CV; lambda chosen on training folds)", "",
              "| Signal | Mean passes | Clean | Noisy | Stops | WER vs Step 8 | Similarity vs Step 8 | WER vs Voice-Swap | Failures |",
              "|---|---:|---:|---:|---|---|---|---|---:|"]
        for name, v in res["rules"].items():
            L.append(f"| {name} | {_f(v['mean_passes'])} | {_f(v['mean_passes_clean'])} | {_f(v['mean_passes_noisy'])} | {v['stops']} | "
                     f"{_ci(v['vs_reference']['wer'])} | {_ci(v['vs_reference']['similarity'])} | {_ci(v['vs_voice_swap']['wer'])} | {v['failures']}/{v['n']} |")
        L += ["", "slow is the proposed rule; full uses the change of the whole estimate (the thesis's drift) and voice the change of the voice part, both calibrated the same way.", ""]
    if res["fixed"]:
        L += ["## Fixed handovers (descriptive)", "", "| Output | Passes | WER vs Step 8 | Similarity vs Step 8 | UTMOS vs Step 8 | Failures |", "|---|---:|---|---|---|---:|"]
        for name, v in res["fixed"].items():
            L.append(f"| {name} | {cost(name)} | {_ci(v['vs_reference']['wer'])} | {_ci(v['vs_reference']['similarity'])} | {_ci(v['vs_reference']['utmos'], '{:+.3f}')} | {v['failures']}/{v['n']} |")
    L += ["", "## Pitch variety (CVf0)", "", "| Condition | " + " | ".join(DIV_OUTS) + " |", "|---|" + "---:|" * len(DIV_OUTS)]
    L += [f"| {c_} | " + " | ".join(_f(res["cvf0_means"][c_][o], "{:.4f}") for o in DIV_OUTS) + " |" for c_ in DIV_CONDS]
    for k, v in res["early_steps"].items():
        L.append(f"\nStep {k} minus Step 8: " + "; ".join(f"{c_} {_ci(d)}" for c_, d in v["diffs"].items()) + f" (keeps pitch variety: {_word(v['keeps'])})")
    L += ["", "## Mean part change per teacher step (descriptive)", ""]
    for p in PARTS:
        L += [f"**{p}**", "", "| Condition | " + " | ".join(f"k={k}" for k in SIGNAL_STEPS) + " |", "|---|" + "---:|" * len(SIGNAL_STEPS)]
        L += [f"| {c_} | " + " | ".join(_f(res["signal_means"][c_][p][k], "{:.3f}") for k in SIGNAL_STEPS) + " |" for c_ in CONDS]
        L.append("")
    L += ["Development evidence only (speakers that designed EXP-020), white-noise prompts. A claim needs a frozen test on new speakers."]
    return "\n".join(L) + "\n"


# ------------------------------------------------------------------------ preregistration
def input_hash(path):
    """SHA-256 with line endings normalized: text inputs are LF on Colab but may be CRLF in a Windows checkout."""
    return sh.sha256_text(path)


def code_hashes():
    return {f"code/colab/{n}": sh.sha256_text(Path(__file__).resolve().parent / n) for n in DEPENDENCIES}


def build_prereg():
    return {"experiment": "EXP-022",
            "purpose": "Development test of Slow-Part Handover: can the teacher hand over before Step 4 by passing only the part of its estimate the student uses, and stopping when that part has settled",
            "algorithm": {"split": "on generated frames: voice = per-band mean over time; slow = centred 9-frame moving average of the remainder, edge padded; fine = the rest (sh.decompose, as in EXP-020)",
                          "signal": "d_slow_k = ||slow(T_k) - slow(T_k-1)|| / ||slow(T_k)||, no extra passes",
                          "stop_rule": f"the first k in {list(STEPS)} with d_slow_k <= lambda hands over x = voice(S) + slow(T_k) + fine(S) (code STS); otherwise Step 8 whole, as the 0.25 reference",
                          "draft": "S = eps + g(eps, 0), one student pass from independent noise (seed + 1009), made only at handover",
                          "student": "the released student loop at j = 1 from x"},
            "passes": {n: cost(n) for n in ("plain_8", "STT_4", "student4") + OUTS},
            "data": {"quality": f"the EXP-012 Quick set: first {N_SPEAKERS} development speakers x {list(CONDS)} (360 cases)",
                     "diversity": f"the first {DIV_N} development speakers x {list(DIV_CONDS)}, {DIV_SEEDS} seeds (input_seed + {DIV_SEED_BASE} + m), outputs {list(DIV_OUTS)}",
                     "reused": "plain Steps 1-8 (EXP-018), student only (EXP-017) and Voice-Swap STT_4 (EXP-020), same seeds and T4"},
            "hardware": "Colab Tesla T4 only; Shonget-DMOSpeech2 at the pinned commit", "dmo_commit": DMO_COMMIT,
            "calibration": {"folds": FOLDS, "seed": SEED, "grid": "lambda 0 then the 1/40 ... 39/40 quantiles of the training folds' signal values over the candidate steps, ascending",
                            "constraint": f"training-fold mean of (output minus Step 8): capped WER at most +{CALIBRATION['wer']} and similarity at least -{CALIBRATION['similarity']}",
                            "procedure": "fixed sequence from conservative to aggressive; the last lambda before the first violation"},
            "candidate_steps": f"Steps 4-7 always; Step 2 or 3 only if its CVf0 minus Step 8 has lower bound >= -{MARGINS['cvf0']} on both {list(DIV_CONDS)}; decided before the stop rules are calibrated",
            "criteria": {"C1_cheaper_than_voice_swap": "out-of-fold mean passes of the slow-part rule below 14 (fixed Voice-Swap)",
                         "C2_quality": f"out-of-fold slow-part rule minus Step 8, speaker bootstrap: capped WER upper bound <= +{MARGINS['wer']} and similarity lower bound >= -{MARGINS['similarity']}, pooled over all 360 cases",
                         "C3_pitch_variety_at_step_3": f"CVf0 of the handover at Step 3 minus Step 8: lower bound >= -{MARGINS['cvf0']} on clean and noisy15",
                         "C4_slow_part_beats_whole_estimate": "out-of-fold mean passes of the slow-part rule below those of the same rule on the whole estimate's change (the thesis's drift), calibrated identically"},
            "decision_rule": "Slow-Part Handover advances to a frozen test on new speakers only if C1 to C4 all pass; otherwise it is reported as a negative result with no retuning on these data",
            "descriptive": ["the voice-part rule", "fixed handovers at every Step 2..7", "stop-step distributions per condition", "mean part changes per step and condition", "Definition 3.1 failures against Step 8"],
            "bootstrap": {"B": BOOT_B, "seed": SEED, "unit": "speaker"},
            "evidence_status": "development evidence on the speakers that designed EXP-020; not the thesis's frozen held-out evaluation",
            "novelty": "to our knowledge no published handover passes only part of the teacher's estimate or stops on the change of that part (targeted searches); not a full literature review",
            "adoption": "not adopted; changing the thesis method needs a dated decision record and adviser review",
            "code_hashes": code_hashes(),
            "input_hashes": {str(p.relative_to(THESIS_ROOT)).replace("\\", "/"): input_hash(p)
                             for p in (STEPS_CSV, GRID32_CSV, EXP020_CSV, THESIS_ROOT / "code/scheduling/failure_config.json")},
            "thesis_commit": rf.git_head(THESIS_ROOT)}


def freeze(out=EXP_DIR):
    path = Path(out) / "preregistration.json"
    Path(out).mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise SystemExit(f"{path} exists; a preregistration is written once (delete it only if nothing was synthesized)")
    pre = build_prereg()
    pre["sha256"] = hashlib.sha256(json.dumps(pre, sort_keys=True).encode("utf-8")).hexdigest()
    pre["created_utc"] = sh._utc()
    path.write_text(json.dumps(pre, indent=2) + "\n", encoding="utf-8")
    print(f"[OK] preregistration written: {path} (sha256 {pre['sha256'][:16]}...)", flush=True)
    return pre


def check_prereg(prereg_path=PREREG_PATH):
    p = Path(prereg_path)
    if not p.exists():
        raise SystemExit(f"Missing {p}: run `freeze` and push the preregistration before any GPU run.")
    pre = json.loads(p.read_text(encoding="utf-8"))
    body = {k: v for k, v in pre.items() if k not in ("sha256", "created_utc")}
    if hashlib.sha256(json.dumps(body, sort_keys=True).encode("utf-8")).hexdigest() != pre["sha256"]:
        raise SystemExit("preregistration.json was edited after it was frozen")
    now = code_hashes()
    changed = [k for k, v in pre["code_hashes"].items() if now.get(k) != v]
    if changed:
        raise SystemExit(f"Code differs from the version frozen in preregistration.json: {changed}")
    moved = [k for k, v in pre["input_hashes"].items() if not (THESIS_ROOT / k).exists() or input_hash(THESIS_ROOT / k) != v]
    if moved:
        raise SystemExit(f"Input files are missing or differ from the frozen versions: {moved}")
    return pre


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["freeze", "smoke", "run", "analyze"])
    ap.add_argument("--out")
    ap.add_argument("--dmo_root", default="/content/Shonget-DMOSpeech2")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--allow_partial", action="store_true")
    args = ap.parse_args(argv)
    if args.cmd == "freeze":
        freeze(args.out or EXP_DIR)
        return
    if not args.out:
        ap.error("--out is required")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.cmd == "analyze":
        analyze(out, args.allow_partial)
        return
    check_prereg()
    head = rf.git_head(args.dmo_root)
    if head != DMO_COMMIT:
        raise SystemExit(f"Shonget-DMOSpeech2 is at {head}, not the pinned {DMO_COMMIT}.")
    pipe = rf.Pipeline(args.dmo_root, args.device)
    if "T4" not in pipe.gpu:
        raise SystemExit(f"EXP-022 reuses T4 labels and student noise depends on the GPU; found {pipe.gpu}.")
    if args.cmd == "smoke":
        sm = out / "smoke"
        sm.mkdir(exist_ok=True)
        its = [it for it in ex.items() if it["utt_id"] not in cg.pilot_utts()][:2]
        repro, drift = run_quality(pipe, its, ("clean", "gauss20"), sm, stored=stored_reproduction())
        (sm / "reproduction.json").write_text(json.dumps({"rows": repro, "max_drift_diff": max(drift)}, indent=2) + "\n", encoding="utf-8")
        worst_wer, worst_sim = sh.check_reproduction(repro)
        if max(drift) > DRIFT_TOLERANCE:
            raise SystemExit(f"Stored teacher drift not reproduced (max difference {max(drift):.2e}); do not run.")
        run_diversity(pipe, ex.items()[:1], ("clean",), sm, seeds=2)
        print(f"Smoke run complete: {sm} (stored STS_4 reproduced: max |dWER| {worst_wer:.4f}, max |dSIM| {worst_sim:.4f}; "
              f"stored drift reproduced: max difference {max(drift):.2e})", flush=True)
        return
    smoke_file = out / "smoke" / "reproduction.json"
    if not smoke_file.exists():
        raise SystemExit("Run the smoke test first: the full run reuses stored labels only if they reproduce.")
    rep = json.loads(smoke_file.read_text(encoding="utf-8"))
    sh.check_reproduction(rep["rows"])
    if rep["max_drift_diff"] > DRIFT_TOLERANCE:
        raise SystemExit("The smoke test did not reproduce the stored teacher drift.")
    shutil.copy2(PREREG_PATH, out / "preregistration.json")
    run_quality(pipe, ex.items(), CONDS, out)
    run_diversity(pipe, ex.items()[:DIV_N], DIV_CONDS, out)
    analyze(out)


if __name__ == "__main__":
    main()
