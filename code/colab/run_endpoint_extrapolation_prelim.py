"""Preliminary Colab T4 test of Drift-Extrapolated Handover (DEH), the proposed handover operator.

The student always restarts at tau = 0.0761, which equals the teacher's Step-4 time, so a Step-4
and a Step-8 handover differ only in the endpoint estimate handed over. DEH predicts the Step-8
endpoint from Steps 2-4 at no extra network cost and hands that over at Step 4 (13 NFE):

  Delta_k = x1hat(t_k) - x1hat(t_{k-1})                       (the change that drift d_k measures)
  rho_k   = min(||Delta_k||_M / ||Delta_{k-1}||_M, rho_max)   (measured contraction rate)
  T_DEH   = x1hat(t_k) + beta * S(rho_k, m) * Delta_k,  S = rho (1 - rho^m) / (1 - rho),  m = H - k

on generated frames only (prompt frames are left unchanged); k = 4, H = 8, beta = 1 (DEH) or
0.5 (DEH-half). Scored against the stored Step-8 reference with the frozen failure rule.

Data: the EXP-012 Quick set (60 development speakers x clean, noisy15, Gaussian 18/20/22/25 dB),
whose Step-4 and Step-8 labels are committed; only the DEH outputs are synthesized and scored.

Subcommands:
  smoke     2 non-pilot development speakers x clean and gauss20 into OUT/smoke/, plus a
            reproduction check of the stored Step-4 and Step-8 scores
  run       write preregistration.json (once), run every case (resumable), then analyze
  analyze   compute every number from the CSV files; nothing is typed in
"""
import argparse
import csv
import datetime
import hashlib
import json
import random
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_followup_tests as rf  # noqa: E402
import run_consistency_gate_prelim as cg  # noqa: E402

THESIS_ROOT = rf.THESIS_ROOT
EXP012 = THESIS_ROOT / "research/experiments/EXP-012-consistency-gate-prelim/results"
N_SPEAKERS, CONDS = cg.QUICK["n_speakers"], tuple(cg.QUICK["conditions"])
NOISY = tuple(c for c in CONDS if c != "clean")
K, H, RHO_MAX = 4, 8, 0.95
VARIANTS = {"deh": 1.0, "half": 0.5}                    # scored: name -> beta
COST = {4: 13, "deh": 13, "half": 13, 8: 21}
ALPHA, DELTA, FOLDS, SEED, THETA = 0.05, 0.10, 5, 2026, 20.0
GRIDS = {"theta": [40.0 - i for i in range(21)],                       # dB, conservative -> aggressive
         "lam_low": [round(0.60 - 0.05 * i, 2) for i in range(12)]}    # low-drift veto, 0.60 -> 0.05
FIELDS = ["utt_id", "speaker_id", "condition", "prompt_snr"] + [f"d_{i}" for i in range(1, 9)] + \
         ["rho", "S", "cos_d4_d3", "cos_d4_rest", "gain_deh", "gain_half", "gain_h6", "gain_lin"] + \
         [f"{m}_{v}" for v in VARIANTS for m in ("wer", "similarity", "utmos", "is_failure", "fail_wer", "fail_sim", "hypothesis")] + \
         ["wer_ref", "similarity_ref", "d4_stored", "utc"]


# --------------------------------------------------------------------------- operator
def deh_factor(rho, m):
    """S = rho + rho^2 + ... + rho^m: the predicted remaining change in units of Delta_k."""
    return float(m) if rho >= 1.0 else rho * (1.0 - rho ** m) / (1.0 - rho)


def _norm(x):
    return float((x * x).sum()) ** 0.5


def _cos(a, b):
    na, nb = _norm(a), _norm(b)
    return float((a * b).sum()) / (na * nb) if na > 0 and nb > 0 else 0.0


def extrapolate(E, k, horizon, beta, plen, rho_max=RHO_MAX):
    """DEH endpoint from the endpoint estimates E[k-2..k]; works on torch tensors or numpy arrays."""
    dk, dk1 = E[k] - E[k - 1], E[k - 1] - E[k - 2]
    rho = min(_norm(dk[:, plen:, :]) / (_norm(dk1[:, plen:, :]) + 1e-8), rho_max)
    S = deh_factor(rho, horizon - k)
    T = E[k].clone() if hasattr(E[k], "clone") else E[k].copy()
    T[:, plen:, :] = T[:, plen:, :] + beta * S * dk[:, plen:, :]
    return T, rho, S


def diagnostics(E, plen, grid):
    """How much closer each candidate endpoint is to the true later endpoint (1 = exact, 0 = no better
    than the plain Step-4 endpoint, < 0 = worse), plus the direction coherence of the trajectory."""
    g = lambda x: x[:, plen:, :]
    gain = lambda T, target: 1.0 - _norm(g(T - E[target])) / (_norm(g(E[K] - E[target])) + 1e-8)
    T, rho, S = extrapolate(E, K, H, 1.0, plen)
    Th, _, _ = extrapolate(E, K, H, 0.5, plen)
    T6, _, _ = extrapolate(E, K, 6, 1.0, plen)
    lin = E[K] + (E[K] - E[K - 1]) * ((grid[H] - grid[K]) / (grid[K] - grid[K - 1]))
    lin[:, :plen, :] = E[K][:, :plen, :]
    return {"rho": rho, "S": S, "cos_d4_d3": _cos(g(E[K] - E[K - 1]), g(E[K - 1] - E[K - 2])),
            "cos_d4_rest": _cos(g(E[K] - E[K - 1]), g(E[H] - E[K])),
            "gain_deh": gain(T, H), "gain_half": gain(Th, H), "gain_h6": gain(T6, 6), "gain_lin": gain(lin, H)}


# ------------------------------------------------------------------------------- data
def items():
    return cg.dev_items(N_SPEAKERS)


def stored_cases():
    """(utt, cond) -> {step: row} from the committed T4 grid, EXP-011 pilot and EXP-012 labels."""
    utts = {it["utt_id"] for it in items()}
    return {k: v for k, v in cg.load_labels(EXP012).items() if k[0] in utts and k[1] in CONDS}


def run_cases(pipe, todo_items, conds, out_dir, csv_name="extrap.csv", reproduce=False):
    path = Path(out_dir) / csv_name
    done = rf.done_keys(path, 1)
    stored = stored_cases()
    todo = [(it, c) for it in todo_items for c in conds if (it["utt_id"], c) not in done]
    print(f"[extrap] {len(todo)} cases to run ({len(done)} already in {path.name}) on {pipe.gpu}", flush=True)
    repro = []
    for n, (item, cond) in enumerate(todo, 1):
        ref = stored[(item["utt_id"], cond)][8]
        ctx = pipe.prepare(item, cond, item["input_seed"], out_dir)
        E, drift = pipe.teacher(ctx, tuple(range(H + 1)))
        plen = ctx["prompt_len"]
        with pipe.torch.inference_mode():
            diag = diagnostics(E, plen, pipe.sampler.SWAY_TEACHER_GRID)
            handover = {v: extrapolate(E, K, H, beta, plen)[0] for v, beta in VARIANTS.items()}
        row = {"utt_id": item["utt_id"], "speaker_id": item["speaker_id"], "condition": cond, "prompt_snr": ctx["prompt_snr"],
               **{f"d_{i}": drift[i] for i in range(1, H + 1)}, **diag,
               "wer_ref": float(ref["wer_ref"]), "similarity_ref": float(ref["similarity_ref"]), "d4_stored": float(ref["d_4"])}
        gt = pipe.ground_truth(item)
        for v in VARIANTS:
            m = pipe.score(pipe.student(ctx, handover[v]), item, gt)
            fail, diag = pipe.check_failure(m["wer"], row["wer_ref"], m["similarity"], row["similarity_ref"], config=pipe.cfg)
            row.update({f"wer_{v}": m["wer"], f"similarity_{v}": m["similarity"], f"utmos_{v}": m["utmos"],
                        f"is_failure_{v}": bool(fail), f"fail_wer_{v}": bool(diag["wer_failed"]),
                        f"fail_sim_{v}": bool(diag["sim_failed"]), f"hypothesis_{v}": m["hypothesis"]})
        if reproduce:
            for k in (4, 8):
                m = pipe.score(pipe.student(ctx, E[k]), item, gt)
                s = stored[(item["utt_id"], cond)][k]
                repro.append({"utt_id": item["utt_id"], "condition": cond, "step": k,
                              "wer": m["wer"], "wer_stored": float(s["wer"]),
                              "similarity": m["similarity"], "similarity_stored": float(s["similarity"])})
        row["utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        new = not path.exists()
        with open(path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            if new:
                w.writeheader()
            w.writerow(row)
            f.flush()
        print(f"[{n}/{len(todo)}] extrap: {item['utt_id']} {cond}", flush=True)
    print(f"[OK] extrap complete: {path}", flush=True)
    return repro


# --------------------------------------------------------------------------- analysis
def build_table(stored, rows):
    table = []
    for r in rows:
        s = stored.get((r["utt_id"], r["condition"]))
        if s is None:
            continue
        table.append({"utt": r["utt_id"], "spk": r["speaker_id"], "cond": r["condition"], "snr": float(r["prompt_snr"]),
                      "d4": float(r["d_4"]), "d4_stored": float(r["d4_stored"]),
                      "fail": {4: rf.is_true(s[4]["is_failure"]), 8: False,
                               **{v: rf.is_true(r[f"is_failure_{v}"]) for v in VARIANTS}},
                      **{f: float(r[f]) for f in ("rho", "S", "cos_d4_d3", "cos_d4_rest", "gain_deh", "gain_half", "gain_h6", "gain_lin")}})
    return table


def policy(name, p):
    """f(record) -> step; p is the calibrated parameter (or None for parameterless policies)."""
    return {
        "reference": lambda r: 8,
        "snr_only": lambda r: 4 if r["snr"] >= THETA else 8,
        "snr_theta": lambda r: 4 if r["snr"] >= p else 8,
        "drift_window": lambda r: 4 if (r["snr"] >= THETA and r["d4"] >= p) else 8,
        "deh_snr_only": lambda r: "deh" if r["snr"] >= THETA else 8,
        "deh_theta": lambda r: "deh" if r["snr"] >= p else 8,
        "deh_window": lambda r: "deh" if (r["snr"] >= THETA and r["d4"] >= p) else 8,
        "deh_all": lambda r: "deh",
        "half_theta": lambda r: "half" if r["snr"] >= p else 8,
        "half_all": lambda r: "half",
    }[name]


PARAM = {"snr_theta": "theta", "drift_window": "lam_low", "deh_theta": "theta", "deh_window": "lam_low", "half_theta": "theta"}
POLICIES = ["reference", "snr_only", "snr_theta", "drift_window", "deh_snr_only", "deh_theta", "deh_window", "deh_all",
            "half_theta", "half_all"]


def ltt_select(train, name, stratified):
    chosen = None
    for p in GRIDS[PARAM[name]]:
        pol, by = policy(name, p), {}
        for r in train:
            d = by.setdefault(r["cond"], [0, 0])
            d[0] += int(r["fail"][pol(r)])
            d[1] += 1
        if stratified:
            pval = max(rf.binom_cdf(k, n, ALPHA) for k, n in by.values())
        else:
            pval = rf.binom_cdf(sum(k for k, _ in by.values()), sum(n for _, n in by.values()), ALPHA)
        if pval > DELTA:
            break
        chosen = p
    return chosen


def cross_validate(table):
    spks = sorted({r["spk"] for r in table})
    random.Random(SEED).shuffle(spks)
    fold = {s: i % FOLDS for i, s in enumerate(spks)}
    res, chosen = {}, {}
    for tag, strat in (("pooled", False), ("stratified", True)):
        res[tag], chosen[tag] = {}, {}
        for name in POLICIES:
            oof, ch = [], []
            for f in range(FOLDS):
                train = [r for r in table if fold[r["spk"]] != f]
                test = [r for r in table if fold[r["spk"]] == f]
                p = ltt_select(train, name, strat) if name in PARAM else None
                ch.append(p)
                pol = policy("reference", None) if (name in PARAM and p is None) else policy(name, p)
                oof += [(r, pol(r)) for r in test]
            per = {}
            for c in CONDS:
                rows = [(r, k) for r, k in oof if r["cond"] == c]
                if rows:
                    per[c] = {"n": len(rows), "failures": sum(int(r["fail"][k]) for r, k in rows),
                              "mean_nfe": sum(COST[k] for _, k in rows) / len(rows)}
            res[tag][name] = {"failures": sum(d["failures"] for d in per.values()), "n": len(oof),
                              "mean_nfe": sum(COST[k] for _, k in oof) / len(oof),
                              "worst_condition_risk": max(d["failures"] / d["n"] for d in per.values()), "per_condition": per}
            chosen[tag][name] = ch
    return res, chosen


def paired(table, a, b, conds):
    rows = sorted((r for r in table if r["cond"] in conds), key=lambda r: (r["utt"], r["cond"]))
    x, y, p = rf.mcnemar_exact([r["fail"][a] for r in rows], [r["fail"][b] for r in rows])
    return {"n": len(rows), "failures_" + str(a): sum(r["fail"][a] for r in rows),
            "failures_" + str(b): sum(r["fail"][b] for r in rows), "only_" + str(a): x, "only_" + str(b): y, "p": p}


def analyze(out, allow_partial=False):
    out = Path(out)
    prereg = json.loads((out / "preregistration.json").read_text(encoding="utf-8"))
    stored = stored_cases()
    table = build_table(stored, rf.read_csv(out / "extrap.csv", repair=True))
    expected = N_SPEAKERS * len(CONDS)
    if len(table) < expected and not allow_partial:
        raise SystemExit(f"{len(table)} of {expected} cases; finish the run or pass --allow_partial")
    res = {"preregistration_sha256": prereg["sha256"], "cases": len(table), "expected_cases": expected,
           "partial": len(table) < expected,
           "drift_reproduction_max_abs_diff": max((abs(r["d4"] - r["d4_stored"]) for r in table), default=None)}
    res["step4_failures"] = {}
    for c in CONDS:
        rows = [r for r in table if r["cond"] == c]
        res["step4_failures"][c] = {"n": len(rows), **{str(k): sum(r["fail"][k] for r in rows) for k in (4, "deh", "half")}}
    tests = {f"{v} vs Step 4, {grp}": paired(table, 4, v, conds)
             for v in VARIANTS for grp, conds in (("noisy (primary)", NOISY), ("clean", ("clean",)))}
    tests.update({f"deh vs Step 4, {c}": paired(table, 4, "deh", (c,)) for c in NOISY})
    res["paired_tests"] = tests
    res["diagnostics"] = {}
    for c in CONDS:
        rows = [r for r in table if r["cond"] == c]
        if rows:
            res["diagnostics"][c] = {f: statistics.median(r[f] for r in rows)
                                     for f in ("rho", "S", "cos_d4_d3", "cos_d4_rest", "gain_deh", "gain_half", "gain_h6", "gain_lin")}
            res["diagnostics"][c]["share_gain_deh_positive"] = sum(r["gain_deh"] > 0 for r in rows) / len(rows)
    if len({r["spk"] for r in table}) >= FOLDS:
        res["cross_validation"], res["chosen"] = cross_validate(table)
    prim = tests["deh vs Step 4, noisy (primary)"]
    clean = tests["deh vs Step 4, clean"]
    res["criteria"] = {
        "C1 noisy Step-4 failures drop by at least 25% with exact McNemar p < 0.05":
            prim["failures_deh"] <= 0.75 * prim["failures_4"] and prim["p"] < 0.05 and prim["failures_deh"] < prim["failures_4"],
        "C2 clean Step-4 failures rise by at most 1": clean["failures_deh"] <= clean["failures_4"] + 1}
    if "cross_validation" in res:
        a, b = res["cross_validation"]["pooled"]["deh_theta"], res["cross_validation"]["pooled"]["snr_theta"]
        res["criteria"]["C3 pooled deh_theta: lower mean NFE than snr_theta with no more failures"] = \
            a["mean_nfe"] < b["mean_nfe"] and a["failures"] <= b["failures"]
    (out / "extrap_results.json").write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
    (out / "EXTRAPOLATION_REPORT.md").write_text(render(res), encoding="utf-8")
    print(f"Wrote {out / 'EXTRAPOLATION_REPORT.md'}", flush=True)
    return res


def render(res):
    f3 = lambda x: "n/a" if x is None else f"{x:.3f}"
    L = ["# Preliminary test: Drift-Extrapolated Handover (DEH)", "",
         f"{'**PARTIAL.** ' if res['partial'] else ''}Development speakers only ({N_SPEAKERS} speakers x {len(CONDS)} conditions: "
         f"{', '.join(CONDS)}); {res['cases']} of {res['expected_cases']} cases; preregistration `{res['preregistration_sha256'][:16]}…`. "
         "Every number is computed from the CSV files by `run_endpoint_extrapolation_prelim.py analyze`.", "",
         f"Drift reproduction (this teacher pass vs stored labels): max |Δd₄| = {f3(res['drift_reproduction_max_abs_diff'])}.", "",
         "## Preregistered criteria", ""]
    L += [f"- {'PASS' if ok else 'FAIL'}: {name}" for name, ok in res["criteria"].items()]
    L += ["", "## Step-4 failures: plain endpoint vs DEH (13 NFE each)", "",
          "| Condition | n | Step 4 | DEH | DEH-half |", "|---|---:|---:|---:|---:|"]
    for c, d in res["step4_failures"].items():
        L.append(f"| {c} | {d['n']} | {d['4']} | {d['deh']} | {d['half']} |")
    L += ["", "Exact McNemar (two-sided), unadjusted; the noisy pooled test is the primary one:", ""]
    for name, t in res["paired_tests"].items():
        keys = [k for k in t if k.startswith("failures_")]
        L.append(f"- {name}: {t[keys[0]]} → {t[keys[1]]} failures of {t['n']} (p {t['p']:.4f})")
    L += ["", "## Endpoint diagnostics (medians; no scoring needed)", "",
          "Gain = 1 − ‖T − x̂₁(t_H)‖ / ‖x̂₁(t₄) − x̂₁(t_H)‖: 1 means the extrapolated endpoint equals the later one, 0 no better than Step 4.", "",
          "| Condition | ρ | S | cos(Δ₄, Δ₃) | cos(Δ₄, rest) | gain DEH | gain half | gain to Step 6 | gain linear-in-t | share gain DEH > 0 |",
          "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for c, d in res["diagnostics"].items():
        L.append(f"| {c} | {d['rho']:.2f} | {d['S']:.2f} | {d['cos_d4_d3']:.2f} | {d['cos_d4_rest']:.2f} | {d['gain_deh']:.2f} | "
                 f"{d['gain_half']:.2f} | {d['gain_h6']:.2f} | {d['gain_lin']:.2f} | {d['share_gain_deh_positive']:.0%} |")
    for tag in ("pooled", "stratified"):
        if "cross_validation" not in res:
            break
        L += ["", f"## Out-of-fold policies, {tag} calibration (5-fold speaker CV, α = {ALPHA}, δ = {DELTA})", "",
              "| Policy | Failures | Mean NFE | Worst-condition risk | " + " | ".join(CONDS) + " |",
              "|---|---:|---:|---:|" + "---:|" * len(CONDS)]
        for name, d in res["cross_validation"][tag].items():
            cells = " | ".join(f"{d['per_condition'][c]['failures']}" if c in d["per_condition"] else "–" for c in CONDS)
            L.append(f"| {name} | {d['failures']}/{d['n']} | {d['mean_nfe']:.2f} | {d['worst_condition_risk']:.1%} | {cells} |")
        L += ["", "Chosen parameter per fold: " + "; ".join(f"{n}: {v}" for n, v in res["chosen"][tag].items() if n in PARAM)]
    L += ["", "Policies: Step 4 = plain endpoint; deh/half = DEH endpoint handed over at Step 4 (13 NFE); "
          "theta = calibrated SNR gate; window = SNR ≥ 20 dB and d₄ ≥ λ_low; *_all = no gate. Parameterless policies are reported as is.",
          "", "Development evidence only; a claim needs a held-out test with the protocol frozen first."]
    return "\n".join(L) + "\n"


# ------------------------------------------------------------------- preregistration
def write_prereg(out, args):
    path = Path(out) / "preregistration.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    labels = EXP012 / "labels_new.csv"
    pre = {"purpose": "Preliminary development test of Drift-Extrapolated Handover (DEH) vs the plain Step-4 handover, SNR-only, a calibrated SNR gate and the drift window",
           "operator": "T = x1hat(t_k) + beta * S(rho, m) * Delta_k on generated frames; rho = min(||Delta_k|| / ||Delta_{k-1}||, rho_max); S = rho (1 - rho^m) / (1 - rho)",
           "k": K, "horizon": H, "rho_max": RHO_MAX, "variants_beta": VARIANTS,
           "n_speakers": N_SPEAKERS, "speakers": f"first {N_SPEAKERS} development speakers by utterance ID (the EXP-012 Quick set)",
           "conditions": list(CONDS), "theta_db": THETA, "alpha": ALPHA, "delta": DELTA, "folds": FOLDS, "seed": SEED, "grids": GRIDS,
           "policies": POLICIES,
           "primary_test": "deh vs plain Step 4 on the noisy conditions pooled (exact McNemar, two-sided)",
           "criteria": ["C1 noisy Step-4 failures drop by at least 25% with p < 0.05",
                        "C2 clean Step-4 failures rise by at most 1",
                        "C3 pooled out-of-fold deh_theta: lower mean NFE than snr_theta with no more failures"],
           "reused_labels": ["research/evidence/benchmarks/gpu_grid_all1000.csv", "research/experiments/EXP-011-followup-t4/results/pilot/pilot_grid.csv",
                             "research/experiments/EXP-012-consistency-gate-prelim/results/labels_new.csv"],
           "input_hashes": {str(p.relative_to(THESIS_ROOT)).replace("\\", "/"): rf.sha256_file(p)
                            for p in (rf.T4_GRID, cg.PILOT_GRID, labels, Path(__file__))},
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
    ap.add_argument("--allow_partial", action="store_true")
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.cmd == "analyze":
        analyze(out, args.allow_partial)
        return
    pipe = rf.Pipeline(args.dmo_root, args.device)
    if args.cmd == "smoke":
        out = out / "smoke"
        out.mkdir(exist_ok=True)
        smoke_items = [it for it in items() if it["utt_id"] not in cg.pilot_utts()][:2]
        repro = run_cases(pipe, smoke_items, ("clean", "gauss20"), out, reproduce=True)
        (out / "reproduction.json").write_text(json.dumps(repro, indent=2) + "\n", encoding="utf-8")
        for r in repro:
            print(f"reproduction {r['utt_id'][-8:]} {r['condition']} Step {r['step']}: WER {r['wer']:.4f} (stored {r['wer_stored']:.4f}), "
                  f"SIM {r['similarity']:.4f} (stored {r['similarity_stored']:.4f})", flush=True)
        print(f"Smoke run complete: {out}")
        return
    write_prereg(out, args)
    run_cases(pipe, items(), CONDS, out)
    analyze(out)


if __name__ == "__main__":
    main()
