"""Preliminary Colab T4 test of the proposed gates against SNR-only and SNR + drift.

Development speakers only (100), eleven prompt conditions. Compared policies (candidates Steps
4/5/6, reference Step 8, SNR gate 20 dB):

  reference       Step 8 for every input
  snr_only        Step 4 if prompt SNR >= 20 dB, else Step 8 (the current gate)
  snr_drift       Definition 3.4: first candidate with d_k <= lam (the proposal as written)
  low_drift_veto  SNR gate, then fall back if d_4 < lam_low (two-sided stabilization window)
  sdc_gate        SNR-Drift Consistency: residual r = log d_4 - (b0 + b1 * SNR), fitted on clean
                  training prompts; fall back if r < -kappa (trajectory inconsistent with SNR)
  guidance_gate   guidance contrast g_4 = ||v_c - v_u|| / ||v_c|| over generated frames; fall back
                  on the side oriented on training folds
  sdc_guidance    fall back if either the SDC or the guidance gate fires

Every gate's parameter is chosen on training folds (5-fold speaker-grouped cross-validation) by
fixed-sequence Learn-Then-Test from conservative to aggressive, either pooled over conditions or
stratified (a parameter is certified only if every condition's binomial p-value <= delta:
an intersection-union test). Out-of-fold failures and NFE are reported per condition.

Subcommands:
  smoke     2 development speakers, 3 conditions, into OUT/smoke/ (labels + signals)
  run       write preregistration.json (once), synthesize the missing labels, record the signals
            for every case, then analyze. Resumable.
  analyze   compute every number from the CSV files; nothing is typed in
"""
import argparse
import datetime
import hashlib
import json
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_followup_tests as rf  # noqa: E402

THESIS_ROOT = rf.THESIS_ROOT
PILOT_GRID = THESIS_ROOT / "research/experiments/EXP-011-followup-t4/results/pilot/pilot_grid.csv"
CONDS = tuple(rf.CONDITIONS)                      # 11 conditions
NEW_CONDS = tuple(c for c in CONDS if c not in rf.GRID_CONDITIONS)
THETA, ALPHA, DELTA, FOLDS, SEED = 20.0, 0.05, 0.10, 5, 2026
CANDS = (4, 5, 6)
SIG_FIELDS = ["utt_id", "speaker_id", "condition", "prompt_snr"] + [f"d_{i}" for i in range(1, 7)] + \
             [f"g_{i}" for i in range(0, 7)] + ["utc"]
GRIDS = {
    "snr_drift": [round(0.20 + 0.05 * i, 2) for i in range(21)],          # lam, conservative -> aggressive
    "low_drift_veto": [round(0.60 - 0.05 * i, 2) for i in range(12)],     # lam_low 0.60 -> 0.05
    "sdc_gate": [round(0.0 + 0.1 * i, 1) for i in range(16)],             # kappa 0.0 -> 1.5
    "guidance_gate": [round(1.0 - 0.1 * i, 1) for i in range(11)],          # q 1.0 -> 0.0 (see make_policy)
}
GRIDS["sdc_guidance"] = GRIDS["sdc_gate"]


# ------------------------------------------------------------------------------- data
QUICK = {"n_speakers": 60, "conditions": ("clean", "noisy15", "gauss18", "gauss20", "gauss22", "gauss25")}


def dev_items(n=None):
    """Development speakers by utterance ID; the first 20 are the EXP-011 pilot speakers."""
    return sorted(rf.split_items("development"), key=lambda it: it["utt_id"])[:n]


def pilot_utts():
    return {it["utt_id"] for it in rf.pilot_items(20)}


def load_labels(out):
    """(utt_id, condition) -> {step: row}: T4 grid (clean/noisy15), pilot (20 speakers x 11), new runs."""
    dev = {it["utt_id"] for it in dev_items()}
    cases = {k: v for k, v in rf.index_cases(rf.read_csv(rf.T4_GRID)).items() if k[0] in dev}
    for k, v in rf.index_cases(rf.read_csv(PILOT_GRID)).items():
        cases.setdefault(k, v)
    new = Path(out) / "labels_new.csv"
    if new.exists():
        for k, v in rf.index_cases(rf.read_csv(new, repair=True)).items():
            cases.setdefault(k, v)
    return cases


# ---------------------------------------------------------------------------- signals
def record_signals(pipe, items, conds, out_dir, csv_name="signals.csv"):
    """Teacher-only pass (steps 0..6) for every case: SNR, d_1..d_6 (sampler's own function) and the
    guidance contrast g_i = ||v_c - v_u||_M / ||v_c||_M over generated frames. No student, no scoring."""
    import csv
    torch, sampler = pipe.torch, pipe.sampler
    path = Path(out_dir) / csv_name
    rf.repair_csv(path)
    done = set()
    if path.exists():
        with open(path, encoding="utf-8") as f:
            done = {(r["utt_id"], r["condition"]) for r in csv.DictReader(f)}
    todo = [(it, c) for it in items for c in conds if (it["utt_id"], c) not in done]
    print(f"[signals] {len(todo)} cases to run ({len(done)} done) on {pipe.gpu}", flush=True)
    grid = sampler.SWAY_TEACHER_GRID
    for n, (item, cond) in enumerate(todo, 1):
        ctx = pipe.prepare(item, cond, item["input_seed"], out_dir)
        state, prev, plen = ctx["y0"].clone(), None, ctx["prompt_len"]
        row = {"utt_id": item["utt_id"], "speaker_id": item["speaker_id"], "condition": cond, "prompt_snr": ctx["prompt_snr"]}
        with torch.inference_mode():
            for i in range(7):
                t = torch.tensor(grid[i], device=pipe.device, dtype=torch.float32)
                kw = dict(x=state, cond=ctx["step_cond"], text=ctx["text_ids"], time=t, mask=None)
                vc = pipe.tts.teacher(drop_audio_cond=False, drop_text=False, **kw)
                vu = pipe.tts.teacher(drop_audio_cond=True, drop_text=True, **kw)
                v = vc + (vc - vu) * sampler.CFG_STRENGTH
                ep = state + (1.0 - grid[i]) * v
                row[f"g_{i}"] = float(torch.norm((vc - vu)[:, plen:, :]) / (torch.norm(vc[:, plen:, :]) + 1e-8))
                if prev is not None:
                    row[f"d_{i}"] = sampler.compute_stabilization_drift(ep, prev, plen)
                prev = ep
                state = state + (grid[i + 1] - grid[i]) * v
        row["utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        new = not path.exists()
        with open(path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=SIG_FIELDS)
            if new:
                w.writeheader()
            w.writerow(row)
            f.flush()
        print(f"[{n}/{len(todo)}] signals: {item['utt_id']} {cond}", flush=True)
    print(f"[OK] signals complete: {path}", flush=True)


# --------------------------------------------------------------------------- analysis
def build_table(cases, signals):
    """One record per (utt, condition) with labels and signals; cases missing either are skipped."""
    table = []
    for (u, c), steps in cases.items():
        s = signals.get((u, c))
        if s is None:
            continue
        rec = {"utt": u, "spk": steps[8]["speaker_id"], "cond": c, "snr": float(s["prompt_snr"]),
               "fail": {k: rf.is_true(steps[k]["is_failure"]) for k in rf.STEPS},
               "d": {k: float(s[f"d_{k}"]) for k in range(1, 7)}, "g": {k: float(s[f"g_{k}"]) for k in range(0, 7)},
               "d_stored": {k: float(steps[8][f"d_{k}"]) for k in CANDS}}
        table.append(rec)
    return table


def fit_sdc(train):
    """Least-squares fit of log d_4 on prompt SNR over clean training prompts."""
    xs = [r["snr"] for r in train if r["cond"] == "clean"]
    ys = [math.log(r["d"][4]) for r in train if r["cond"] == "clean"]
    n, mx, my = len(xs), sum(xs) / len(xs), sum(ys) / len(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    b1 = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx if sxx > 0 else 0.0
    return my - b1 * mx, b1


def orient_guidance(train):
    """+1 if low g_4 marks gate-passed Step-4 failures on training data (fall back below), else -1."""
    pas = [r for r in train if r["snr"] >= THETA]
    fails = [r["g"][4] for r in pas if r["fail"][4]]
    oks = [r["g"][4] for r in pas if not r["fail"][4]]
    if not fails or not oks:
        return 1
    return 1 if sum(fails) / len(fails) < sum(oks) / len(oks) else -1


def make_policy(name, param, model):
    """Return f(record) -> chosen step."""
    b0, b1, orient, g_scale = model["b0"], model["b1"], model["orient"], model["g_scale"]

    def sdc_fires(r):
        return math.log(r["d"][4]) - (b0 + b1 * r["snr"]) < -param

    def guid_fires(r, q):
        """g_4 scaled by the clean training median; q = 1 fires on about half the clean prompts, q = 0 never.
        Low g_4 marks failures when orient > 0, high g_4 when orient < 0 (set on training folds)."""
        if q is None:
            return False
        v = r["g"][4] / g_scale
        return v < q if orient > 0 else v > 2.0 - q

    def pol(r):
        if name == "reference" or r["snr"] < THETA:
            return 8
        if name == "snr_only":
            return 4
        if name == "snr_drift":
            return next((k for k in CANDS if r["d"][k] <= param), 8)
        if name == "low_drift_veto":
            return 8 if r["d"][4] < param else 4
        if name == "sdc_gate":
            return 8 if sdc_fires(r) else 4
        if name == "guidance_gate":
            return 8 if guid_fires(r, param) else 4
        if name == "sdc_guidance":
            return 8 if (sdc_fires(r) or guid_fires(r, model["guid_q"])) else 4
        raise ValueError(name)
    return pol


def fit_model(train):
    b0, b1 = fit_sdc(train)
    clean_g = sorted(r["g"][4] for r in train if r["cond"] == "clean")
    return {"b0": b0, "b1": b1, "orient": orient_guidance(train),
            "g_scale": clean_g[len(clean_g) // 2] if clean_g else 1.0, "guid_q": None}


def speaker_failures(records, pol):
    """Per condition: speaker-level failure count (one utterance per speaker), cases, NFE list."""
    by = {}
    for r in records:
        k = pol(r)
        d = by.setdefault(r["cond"], {"fails": 0, "n": 0, "nfe": []})
        d["fails"] += int(r["fail"][k])
        d["n"] += 1
        d["nfe"].append(rf.COST[k])
    return by


def ltt_select(train, name, model, stratified):
    """Fixed-sequence LTT over GRIDS[name] (conservative -> aggressive); returns the certified parameter."""
    chosen = None
    for p in GRIDS[name]:
        by = speaker_failures(train, make_policy(name, p, model))
        if stratified:
            pval = max(rf.binom_cdf(d["fails"], d["n"], ALPHA) for d in by.values())
        else:
            k, n = sum(d["fails"] for d in by.values()), sum(d["n"] for d in by.values())
            pval = rf.binom_cdf(k, n, ALPHA)
        if pval > DELTA:
            break
        chosen = p
    return chosen


def cross_validate(table):
    spks = sorted({r["spk"] for r in table})
    random.Random(SEED).shuffle(spks)
    fold_of = {s: i % FOLDS for i, s in enumerate(spks)}
    names = ["reference", "snr_only", "snr_drift", "low_drift_veto", "sdc_gate", "guidance_gate", "sdc_guidance"]
    results, chosen = {}, {}
    for strat in (False, True):
        tag = "stratified" if strat else "pooled"
        oof = {n: [] for n in names}
        chosen[tag] = {n: [] for n in names}
        for f in range(FOLDS):
            train = [r for r in table if fold_of[r["spk"]] != f]
            test = [r for r in table if fold_of[r["spk"]] == f]
            model = fit_model(train)
            model["guid_q"] = ltt_select(train, "guidance_gate", model, strat)   # None: guidance part never fires
            for n in names:
                p = ltt_select(train, n, model, strat) if n in GRIDS else None
                chosen[tag][n].append(p)
                fallback = n in GRIDS and p is None
                pol = make_policy("reference" if fallback else n, p, model)
                oof[n] += [(r, pol(r)) for r in test]
        results[tag] = {}
        for n in names:
            per = {}
            for c in CONDS:
                rows = [(r, k) for r, k in oof[n] if r["cond"] == c]
                if rows:
                    fails = [int(r["fail"][k]) for r, k in rows]
                    per[c] = {"n": len(rows), "failures": sum(fails), "mean_nfe": sum(rf.COST[k] for _, k in rows) / len(rows),
                              "fails_vec": {r["utt"]: f for (r, _), f in zip(rows, fails)}}
            tot_n = sum(d["n"] for d in per.values())
            results[tag][n] = {"per_condition": per, "failures": sum(d["failures"] for d in per.values()), "n": tot_n,
                               "mean_nfe": sum(d["mean_nfe"] * d["n"] for d in per.values()) / tot_n,
                               "worst_condition_risk": max(d["failures"] / d["n"] for d in per.values())}
    return results, chosen


def discrimination(table):
    """AUC of each signal for Step-4 failure among gate-passed cases (higher score = more likely to fail)."""
    def auc(pos, neg):
        if not pos or not neg:
            return None
        return sum((p > q) + 0.5 * (p == q) for p in pos for q in neg) / (len(pos) * len(neg))
    passed = [r for r in table if r["snr"] >= THETA]
    b0, b1 = fit_sdc(table)
    score = {"low SNR (-S)": lambda r: -r["snr"], "low drift (-d_4)": lambda r: -r["d"][4],
             "SDC residual (-r)": lambda r: -(math.log(r["d"][4]) - (b0 + b1 * r["snr"])),
             "low guidance (-g_4)": lambda r: -r["g"][4], "high guidance (+g_4)": lambda r: r["g"][4]}
    out = {}
    for name, s in score.items():
        pos = [s(r) for r in passed if r["fail"][4]]
        neg = [s(r) for r in passed if not r["fail"][4]]
        out[name] = {"auc": auc(pos, neg), "failures": len(pos), "passes": len(neg)}
    return out


def mcnemar_vs(results, tag, a, b):
    out = {}
    for c in CONDS:
        pa, pb = results[tag][a]["per_condition"].get(c), results[tag][b]["per_condition"].get(c)
        if pa and pb:
            utts = sorted(pa["fails_vec"])
            x, y, p = rf.mcnemar_exact([pa["fails_vec"][u] for u in utts], [pb["fails_vec"][u] for u in utts])
            out[c] = {"only_" + a: x, "only_" + b: y, "p": p}
    ps = [v["p"] for v in out.values()]
    for c, adj in zip(out, rf.holm(ps)):
        out[c]["p_holm"] = adj
    return out


def analyze(out, allow_partial=False):
    global CONDS
    out = Path(out)
    prereg = json.loads((out / "preregistration.json").read_text(encoding="utf-8"))
    CONDS = tuple(prereg["conditions"])
    utts = {it["utt_id"] for it in dev_items(prereg["n_speakers"])}
    cases = {k: v for k, v in load_labels(out).items() if k[0] in utts and k[1] in CONDS}
    sig = {(r["utt_id"], r["condition"]): r for r in rf.read_csv(out / "signals.csv", repair=True)}
    table = build_table(cases, sig)
    expected = len(utts) * len(CONDS)
    res = {"preregistration_sha256": prereg["sha256"], "n_speakers": prereg["n_speakers"], "conditions": list(CONDS),
           "cases": len(table), "expected_cases": expected,
           "partial": len(table) < expected}
    if res["partial"] and not allow_partial:
        raise SystemExit(f"{len(table)}/{expected} cases have labels and signals; finish the run or pass --allow_partial.")
    diffs = [abs(r["d"][k] - r["d_stored"][k]) for r in table for k in CANDS]
    res["drift_reproduction_max_abs_diff"] = max(diffs) if diffs else None
    res["discrimination_gate_passed_step4"] = discrimination(table)
    results, chosen = cross_validate(table)
    res["chosen_parameters_per_fold"] = chosen
    res["out_of_fold"] = {tag: {n: {k: v for k, v in d.items() if k != "per_condition"} | {
        "per_condition": {c: {kk: vv for kk, vv in pc.items() if kk != "fails_vec"} for c, pc in d["per_condition"].items()}}
        for n, d in r.items()} for tag, r in results.items()}
    res["mcnemar"] = {tag: {n: mcnemar_vs(results, tag, "snr_only", n) for n in ("snr_drift", "low_drift_veto", "sdc_gate", "guidance_gate", "sdc_guidance")}
                      for tag in results}
    (out / "consistency_results.json").write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
    (out / "CONSISTENCY_GATE_REPORT.md").write_text(render(res), encoding="utf-8")
    print(f"Wrote {out / 'CONSISTENCY_GATE_REPORT.md'}")
    return res


def render(res):
    pct = lambda x: f"{100 * x:.1f}%"
    L = ["# Preliminary test: proposed gates vs SNR-only and SNR + drift", "",
         f"Development speakers only ({res['n_speakers']} speakers x {len(res['conditions'])} conditions: {', '.join(res['conditions'])}); {res['cases']} of {res['expected_cases']} cases; preregistration `{res['preregistration_sha256'][:16]}…`. "
         "Every number is computed from the CSV files by `run_consistency_gate_prelim.py analyze`."
         + (" **PARTIAL.**" if res["partial"] else ""), "",
         f"Drift reproduction (teacher-only pass vs stored labels): max |Δd| = {res['drift_reproduction_max_abs_diff']}.", "",
         "## Which signal flags gate-passed Step-4 failures? (AUC, all conditions pooled)", "",
         "| Signal | AUC | Failures / passes |", "|---|---:|---|"]
    for name, d in res["discrimination_gate_passed_step4"].items():
        a = "—" if d["auc"] is None else f"{d['auc']:.3f}"
        L.append(f"| {name} | {a} | {d['failures']} / {d['passes']} |")
    for tag in ("pooled", "stratified"):
        L += ["", f"## Out-of-fold results, {tag} calibration (5-fold speaker CV, α = 0.05, δ = 0.10)", "",
              "| Policy | Failures (all) | Mean NFE | Worst-condition risk | " + " | ".join(CONDS) + " |",
              "|---|---:|---:|---:|" + "---:|" * len(CONDS)]
        for n, d in res["out_of_fold"][tag].items():
            cells = [f"{d['per_condition'][c]['failures']}" if c in d["per_condition"] else "—" for c in CONDS]
            L.append(f"| {n} | {d['failures']}/{d['n']} | {d['mean_nfe']:.2f} | {pct(d['worst_condition_risk'])} | " + " | ".join(cells) + " |")
        L += ["", "Chosen parameter per fold: " + "; ".join(f"{n}: {v}" for n, v in res["chosen_parameters_per_fold"][tag].items() if any(x is not None for x in v))]
        L += ["", "McNemar vs SNR-only (per condition, Holm-adjusted p; only conditions with discordant pairs):", ""]
        for n, per in res["mcnemar"][tag].items():
            parts = [f"{c}: {v['only_snr_only']} vs {v['only_' + n]} (p {v['p_holm']:.3f})" for c, v in per.items() if v["only_snr_only"] + v["only_" + n] > 0]
            L.append(f"- {n}: " + ("; ".join(parts) if parts else "no discordant pairs"))
    L += ["", "Development evidence only (100 speakers); the parameters were chosen on training folds. "
          "A claim needs a held-out test with the protocol frozen first."]
    return "\n".join(L) + "\n"


# ------------------------------------------------------------------------------- main
def write_prereg(out, args):
    path = Path(out) / "preregistration.json"
    n, conds = args.n_speakers, list(args.conditions)
    if path.exists():
        pre = json.loads(path.read_text(encoding="utf-8"))
        if pre["n_speakers"] != n or pre["conditions"] != conds:
            raise SystemExit(f"{path} was written for {pre['n_speakers']} speakers x {pre['conditions']}; "
                             "use a different --out folder for another size.")
        return pre
    pre = {"purpose": "Preliminary development test of the proposed gates (SDC, low-drift veto, guidance contrast, stratified LTT) vs SNR-only and SNR + drift",
           "n_speakers": n, "speakers": f"first {n} development speakers by utterance ID (includes the 20 EXP-011 pilot speakers)",
           "conditions": conds, "theta_db": THETA, "alpha": ALPHA, "delta": DELTA,
           "folds": FOLDS, "seed": SEED, "grids": GRIDS, "candidates": list(CANDS), "reference_step": 8,
           "policies": ["reference", "snr_only", "snr_drift", "low_drift_veto", "sdc_gate", "guidance_gate", "sdc_guidance"],
           "calibration": ["pooled fixed-sequence LTT", "stratified fixed-sequence LTT (intersection-union over conditions)"],
           "reused_labels": ["research/evidence/benchmarks/gpu_grid_all1000.csv (clean, noisy15)",
                             "research/experiments/EXP-011-followup-t4/results/pilot/pilot_grid.csv (20 pilot speakers, all conditions)"],
           "input_hashes": {str(p.relative_to(THESIS_ROOT)).replace("\\", "/"): rf.sha256_file(p) for p in (rf.T4_GRID, PILOT_GRID, Path(__file__))},
           "thesis_commit": rf.git_head(THESIS_ROOT), "dmo_commit": rf.git_head(args.dmo_root)}
    pre["sha256"] = hashlib.sha256(json.dumps(pre, sort_keys=True).encode("utf-8")).hexdigest()
    pre["created_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    path.write_text(json.dumps(pre, indent=2) + "\n", encoding="utf-8")
    print(f"[OK] preregistration written: {path}", flush=True)
    return pre


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["smoke", "run", "analyze"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--dmo_root", default="/content/Shonget-DMOSpeech2")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--demand_dir", default=None)
    ap.add_argument("--allow_partial", action="store_true")
    ap.add_argument("--n_speakers", type=int, default=100, help="run: first N development speakers (at least 20)")
    ap.add_argument("--conditions", default=",".join(CONDS), help="run: comma-separated prompt conditions")
    ap.add_argument("--quick", action="store_true", help="run: 60 speakers x clean, noisy15 and Gaussian 18/20/22/25 dB")
    args = ap.parse_args(argv)
    if args.quick:
        args.n_speakers, args.conditions = QUICK["n_speakers"], ",".join(QUICK["conditions"])
    args.conditions = [c for c in args.conditions.split(",") if c]
    unknown = [c for c in args.conditions if c not in CONDS]
    if unknown or args.n_speakers < 20:
        raise SystemExit(f"unknown conditions {unknown} or fewer than 20 speakers")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.cmd == "analyze":
        analyze(out, args.allow_partial)
        return
    if args.cmd == "smoke":
        out = out / "smoke"
        out.mkdir(exist_ok=True)
        pipe = rf.Pipeline(args.dmo_root, args.device)
        if args.demand_dir:
            pipe.load_babble(args.demand_dir)
        items = [it for it in dev_items() if it["utt_id"] not in pilot_utts()][:2]
        rf.run_grid(pipe, items, ("gauss20", "telephone"), "consistency_smoke", out, "labels_new.csv")
        record_signals(pipe, items, ("clean", "gauss20", "telephone"), out)
        print(f"Smoke run complete: {out}")
        return
    write_prereg(out, args)
    pipe = rf.Pipeline(args.dmo_root, args.device)
    if any(rf.CONDITIONS[c][0] == "babble" for c in args.conditions):
        pipe.load_babble(args.demand_dir or "")
    items = dev_items(args.n_speakers)
    rest = [it for it in items if it["utt_id"] not in pilot_utts()]
    new_conds = [c for c in args.conditions if c not in rf.GRID_CONDITIONS]
    rf.run_grid(pipe, rest, new_conds, "consistency_prelim", out, "labels_new.csv")
    record_signals(pipe, items, args.conditions, out)
    analyze(out)


if __name__ == "__main__":
    main()
