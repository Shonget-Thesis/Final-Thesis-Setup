"""Preliminary Colab T4 test of three zero-cost handover operators at Step 4 (EXP-014).

EXP-013 showed that 96 of 126 noisy Step-4 failures are speaker-similarity (SIM) failures and that
the Step-8 endpoint cannot be predicted by extrapolation. The operators below change what the
student receives at the Step-4 handover (13 NFE), without extra network evaluations:

  tcr   Teacher-Coupled Restart. The student's first input is the teacher's own state x(t_4)
        instead of the endpoint re-noised with fresh noise. Because tau_1 = t_4 = 0.0761,
        x(t_4) = (1 - tau) eps_hat + tau x1hat(t_4) with the teacher's implied noise eps_hat:
        the student continues the teacher's trajectory.
  cdeh  Coherence-gated DEH. T = x1hat(t_4) + S(rho, 4) Delta_4 if cos(Delta_4, Delta_3) >= 0.2,
        else the plain endpoint (EXP-013 operator, applied only to coherent trajectories).
  sea   Speech-Envelope Anchoring. Per mel band b, shift the generated frames of the endpoint by
        alpha (mu_prompt[b] - mu_gen[b]), where both means use only the louder half of the frames
        (speech, not noise floor); alpha = 0.5.

Data: the EXP-012 Quick set (60 development speakers x 6 conditions); the plain Step-4 and Step-8
labels are committed, so only the three operator outputs are synthesized and scored.

Subcommands: smoke (2 speakers x clean, gauss20, plus a reproduction check of the stored Step-4
and Step-8 scores through this runner's own student loop), run, analyze.
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
import run_endpoint_extrapolation_prelim as ex  # noqa: E402

THESIS_ROOT = rf.THESIS_ROOT
N_SPEAKERS, CONDS, NOISY = ex.N_SPEAKERS, ex.CONDS, ex.NOISY
K, COH_MIN, SEA_ALPHA = 4, 0.2, 0.5
OPS = ("tcr", "cdeh", "sea")
COST = {4: 13, 8: 21, **{o: 13 for o in OPS}}
ALPHA, DELTA, FOLDS, SEED, THETA = ex.ALPHA, ex.DELTA, ex.FOLDS, ex.SEED, ex.THETA
GRIDS = {"theta": ex.GRIDS["theta"]}
FIELDS = ["utt_id", "speaker_id", "condition", "prompt_snr", "d_4", "d4_stored", "cos_d4_d3", "cdeh_applied", "sea_shift"] + \
         [f"{m}_{o}" for o in OPS for m in ("wer", "similarity", "utmos", "is_failure", "fail_wer", "fail_sim", "hypothesis")] + \
         ["wer_ref", "similarity_ref", "utc"]


# --------------------------------------------------------------------------- operators
def teacher_state(E, y0, grid, k):
    """Recover the teacher state x(t_k) from the endpoints: E_i = x_i + (1 - t_i) v_i, x_{i+1} = x_i + dt v_i."""
    x = y0
    for i in range(k):
        v = (E[i] - x) / (1.0 - grid[i])
        x = x + (grid[i + 1] - grid[i]) * v
    return x


def cdeh(E, plen):
    g = lambda a: a[:, plen:, :]
    c = ex._cos(g(E[K] - E[K - 1]), g(E[K - 1] - E[K - 2]))
    if c < COH_MIN:
        return (E[K].clone() if hasattr(E[K], "clone") else E[K].copy()), c, False
    return ex.extrapolate(E, K, 8, 1.0, plen)[0], c, True


def _loud_mean(x):
    """Per-band mean over the louder half of the frames; x is (frames, bands)."""
    e = x.mean(axis=1) if not hasattr(x, "clone") else x.mean(dim=1)
    order = sorted(range(len(e)), key=lambda i: float(e[i]), reverse=True)[: max(1, len(e) // 2)]
    return x[order].mean(axis=0) if not hasattr(x, "clone") else x[order].mean(dim=0)


def sea(E4, plen, alpha=SEA_ALPHA):
    T = E4.clone() if hasattr(E4, "clone") else E4.copy()
    shift = alpha * (_loud_mean(E4[0, :plen, :]) - _loud_mean(E4[0, plen:, :]))
    T[0, plen:, :] = T[0, plen:, :] + shift
    return T, float((shift * shift).mean()) ** 0.5


def student(pipe, ctx, endpoint, phi0=None):
    """The sampler's run_student_completion, with an optional first input phi0 (coupled restart)."""
    torch, sampler = pipe.torch, pipe.sampler
    x = endpoint.clone()
    with torch.inference_mode():
        for j, tau in enumerate(sampler.STUDENT_TAUS):
            gen = sampler.make_local_generator((int(ctx["seed"]) + int(tau * 100)) % (2 ** 31 - 1), device=pipe.device)
            eps = torch.randn(x.shape, dtype=torch.float32, device=pipe.device, generator=gen)
            tt = torch.tensor([tau], device=pipe.device, dtype=torch.float32)
            ts = tt.unsqueeze(-1).unsqueeze(-1)
            phi = phi0 if (j == 0 and phi0 is not None) else (1.0 - ts) * eps + ts * x
            pred = pipe.tts.generator(x=phi, cond=ctx["cond"], text=ctx["text_ids"], time=tt, drop_audio_cond=False, drop_text=False)
            x = phi + (1.0 - ts) * pred
        mel = x.permute(0, 2, 1) * getattr(pipe.tts, "scale", 1.0)
        return pipe.tts.vocos.decode(mel[..., ctx["prompt_len"]:]).cpu().numpy().squeeze()


# -------------------------------------------------------------------------------- runs
def run_cases(pipe, todo_items, conds, out_dir, csv_name="ops.csv", reproduce=False):
    path = Path(out_dir) / csv_name
    done = rf.done_keys(path, 1)
    stored = ex.stored_cases()
    todo = [(it, c) for it in todo_items for c in conds if (it["utt_id"], c) not in done]
    print(f"[ops] {len(todo)} cases to run ({len(done)} already in {path.name}) on {pipe.gpu}", flush=True)
    grid, repro = pipe.sampler.SWAY_TEACHER_GRID, []
    for n, (item, cond) in enumerate(todo, 1):
        ref = stored[(item["utt_id"], cond)][8]
        ctx = pipe.prepare(item, cond, item["input_seed"], out_dir)
        E, drift = pipe.teacher(ctx, tuple(range(K + 1)))
        plen = ctx["prompt_len"]
        with pipe.torch.inference_mode():
            x4 = teacher_state(E, ctx["y0"], grid, K)
            T_cdeh, coh, applied = cdeh(E, plen)
            T_sea, shift = sea(E[K], plen)
        inputs = {"tcr": (E[K], x4), "cdeh": (T_cdeh, None), "sea": (T_sea, None)}
        row = {"utt_id": item["utt_id"], "speaker_id": item["speaker_id"], "condition": cond, "prompt_snr": ctx["prompt_snr"],
               "d_4": drift[4], "d4_stored": float(ref["d_4"]), "cos_d4_d3": coh, "cdeh_applied": applied, "sea_shift": shift,
               "wer_ref": float(ref["wer_ref"]), "similarity_ref": float(ref["similarity_ref"])}
        gt = pipe.ground_truth(item)
        for o, (T, phi0) in inputs.items():
            m = pipe.score(student(pipe, ctx, T, phi0), item, gt)
            fail, diag = pipe.check_failure(m["wer"], row["wer_ref"], m["similarity"], row["similarity_ref"], config=pipe.cfg)
            row.update({f"wer_{o}": m["wer"], f"similarity_{o}": m["similarity"], f"utmos_{o}": m["utmos"], f"is_failure_{o}": bool(fail),
                        f"fail_wer_{o}": bool(diag["wer_failed"]), f"fail_sim_{o}": bool(diag["sim_failed"]), f"hypothesis_{o}": m["hypothesis"]})
        if reproduce:
            E8, _ = pipe.teacher(ctx, (8,))
            for k, T in ((4, E[K]), (8, E8[8])):
                m = pipe.score(student(pipe, ctx, T), item, gt)
                s = stored[(item["utt_id"], cond)][k]
                repro.append({"utt_id": item["utt_id"], "condition": cond, "step": k, "wer": m["wer"], "wer_stored": float(s["wer"]),
                              "similarity": m["similarity"], "similarity_stored": float(s["similarity"])})
        row["utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        new = not path.exists()
        with open(path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            if new:
                w.writeheader()
            w.writerow(row)
            f.flush()
        print(f"[{n}/{len(todo)}] ops: {item['utt_id']} {cond}", flush=True)
    print(f"[OK] ops complete: {path}", flush=True)
    return repro


# ----------------------------------------------------------------------------- analysis
def build_table(stored, rows):
    table = []
    for r in rows:
        s = stored.get((r["utt_id"], r["condition"]))
        if s is not None:
            table.append({"utt": r["utt_id"], "spk": r["speaker_id"], "cond": r["condition"], "snr": float(r["prompt_snr"]),
                          "d4": float(r["d_4"]), "d4_stored": float(r["d4_stored"]),
                          "fail": {4: rf.is_true(s[4]["is_failure"]), 8: False, **{o: rf.is_true(r[f"is_failure_{o}"]) for o in OPS}},
                          "sim": {4: float(s[4]["similarity"]), **{o: float(r[f"similarity_{o}"]) for o in OPS}}})
    return table


def policy(name, p):
    if name == "reference":
        return lambda r: 8
    if name == "snr_only":
        return lambda r: 4 if r["snr"] >= THETA else 8
    if name == "snr_theta":
        return lambda r: 4 if r["snr"] >= p else 8
    op, kind = name.split("_", 1)
    return (lambda r: op if r["snr"] >= p else 8) if kind == "theta" else (lambda r: op)


POLICIES = ["reference", "snr_only", "snr_theta"] + [f"{o}_{k}" for o in OPS for k in ("theta", "all")]
PARAM = {n: "theta" for n in POLICIES if n.endswith("_theta")}


def ltt_select(train, name, stratified):
    chosen = None
    for p in GRIDS[PARAM[name]]:
        pol, by = policy(name, p), {}
        for r in train:
            d = by.setdefault(r["cond"], [0, 0])
            d[0] += int(r["fail"][pol(r)])
            d[1] += 1
        ks = [rf.binom_cdf(k, n, ALPHA) for k, n in by.values()] if stratified else \
             [rf.binom_cdf(sum(k for k, _ in by.values()), sum(n for _, n in by.values()), ALPHA)]
        if max(ks) > DELTA:
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
                p = ltt_select(train, name, strat) if name in PARAM else None
                ch.append(p)
                pol = policy("reference", None) if (name in PARAM and p is None) else policy(name, p)
                oof += [(r, pol(r)) for r in table if fold[r["spk"]] == f]
            per = {c: sum(int(r["fail"][k]) for r, k in oof if r["cond"] == c) for c in CONDS}
            res[tag][name] = {"failures": sum(per.values()), "n": len(oof), "mean_nfe": sum(COST[k] for _, k in oof) / len(oof),
                              "worst_condition_risk": max(per[c] / max(1, sum(r["cond"] == c for r, _ in oof)) for c in CONDS),
                              "per_condition": per}
            chosen[tag][name] = ch
    return res, chosen


def analyze(out, allow_partial=False):
    out = Path(out)
    prereg = json.loads((out / "preregistration.json").read_text(encoding="utf-8"))
    table = build_table(ex.stored_cases(), rf.read_csv(out / "ops.csv", repair=True))
    expected = N_SPEAKERS * len(CONDS)
    if len(table) < expected and not allow_partial:
        raise SystemExit(f"{len(table)} of {expected} cases; finish the run or pass --allow_partial")
    res = {"preregistration_sha256": prereg["sha256"], "cases": len(table), "expected_cases": expected, "partial": len(table) < expected,
           "drift_reproduction_max_abs_diff": max((abs(r["d4"] - r["d4_stored"]) for r in table), default=None),
           "step4_failures": {c: {"n": sum(r["cond"] == c for r in table),
                                  **{str(k): sum(r["fail"][k] for r in table if r["cond"] == c) for k in (4,) + OPS}} for c in CONDS},
           "median_sim_change": {o: {c: statistics.median([r["sim"][o] - r["sim"][4] for r in table if r["cond"] == c] or [0.0])
                                     for c in CONDS} for o in OPS},
           "paired_tests": {f"{o} vs Step 4, {g}": ex.paired(table, 4, o, cs) for o in OPS
                            for g, cs in (("noisy (primary)", NOISY), ("clean", ("clean",)))}}
    if len({r["spk"] for r in table}) >= FOLDS:
        res["cross_validation"], res["chosen"] = cross_validate(table)
    res["criteria"] = {}
    for o in OPS:
        a, b = res["paired_tests"][f"{o} vs Step 4, noisy (primary)"], res["paired_tests"][f"{o} vs Step 4, clean"]
        crit = {"C1": a[f"failures_{o}"] <= 0.75 * a["failures_4"] and a["p"] < 0.05 and a[f"failures_{o}"] < a["failures_4"],
                "C2": b[f"failures_{o}"] <= b["failures_4"] + 1}
        if "cross_validation" in res:
            x, y = res["cross_validation"]["pooled"][f"{o}_theta"], res["cross_validation"]["pooled"]["snr_theta"]
            crit["C3"] = x["mean_nfe"] < y["mean_nfe"] and x["failures"] <= y["failures"]
        res["criteria"][o] = crit
    (out / "ops_results.json").write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
    (out / "HANDOVER_OPERATORS_REPORT.md").write_text(render(res), encoding="utf-8")
    print(f"Wrote {out / 'HANDOVER_OPERATORS_REPORT.md'}", flush=True)
    return res


def render(res):
    names = {"tcr": "Teacher-Coupled Restart", "cdeh": "Coherence-gated DEH", "sea": "Speech-Envelope Anchoring"}
    d = res["drift_reproduction_max_abs_diff"]
    L = ["# Preliminary test: zero-cost handover operators at Step 4", "",
         f"{'**PARTIAL.** ' if res['partial'] else ''}Development speakers only ({N_SPEAKERS} speakers x {len(CONDS)} conditions: {', '.join(CONDS)}); "
         f"{res['cases']} of {res['expected_cases']} cases; preregistration `{res['preregistration_sha256'][:16]}…`. "
         "Every number is computed from the CSV files by `run_handover_operators_prelim.py analyze`.", "",
         f"Drift reproduction: max |Δd₄| = {'n/a' if d is None else f'{d:.3f}'}.", "",
         "## Preregistered criteria (C1 noisy failures −25% with p < 0.05; C2 clean +1 at most; C3 calibrated gate beats the calibrated SNR gate)", "",
         "| Operator | C1 | C2 | C3 |", "|---|---|---|---|"]
    for o, c in res["criteria"].items():
        L.append(f"| {names[o]} (`{o}`) | " + " | ".join("PASS" if c.get(k) else ("FAIL" if k in c else "n/a") for k in ("C1", "C2", "C3")) + " |")
    L += ["", "## Step-4 failures (13 NFE each)", "", "| Condition | n | plain | " + " | ".join(OPS) + " |", "|---|---:|---:|" + "---:|" * len(OPS)]
    for c, v in res["step4_failures"].items():
        L.append(f"| {c} | {v['n']} | {v['4']} | " + " | ".join(str(v[o]) for o in OPS) + " |")
    L += ["", "Median SIM change vs the plain Step-4 output: " + "; ".join(
        f"{o}: " + ", ".join(f"{c} {v:+.4f}" for c, v in res["median_sim_change"][o].items()) for o in OPS), "",
        "Exact McNemar (two-sided), unadjusted:", ""]
    for name, t in res["paired_tests"].items():
        ks = [k for k in t if k.startswith("failures_")]
        L.append(f"- {name}: {t[ks[0]]} → {t[ks[1]]} of {t['n']} (p {t['p']:.4f})")
    for tag in ("pooled", "stratified"):
        if "cross_validation" not in res:
            break
        L += ["", f"## Out-of-fold policies, {tag} calibration (5-fold speaker CV, α = {ALPHA}, δ = {DELTA})", "",
              "| Policy | Failures | Mean NFE | Worst-condition risk | " + " | ".join(CONDS) + " |", "|---|---:|---:|---:|" + "---:|" * len(CONDS)]
        for n, v in res["cross_validation"][tag].items():
            L.append(f"| {n} | {v['failures']}/{v['n']} | {v['mean_nfe']:.2f} | {v['worst_condition_risk']:.1%} | "
                     + " | ".join(str(v["per_condition"][c]) for c in CONDS) + " |")
        L += ["", "Chosen θ per fold: " + "; ".join(f"{n}: {v}" for n, v in res["chosen"][tag].items() if n in PARAM)]
    L += ["", "*_theta = the operator's output for inputs with SNR ≥ θ (calibrated), Step 8 otherwise; *_all = the operator for every input.",
          "", "Development evidence only; a claim needs a held-out test with the protocol frozen first."]
    return "\n".join(L) + "\n"


def write_prereg(out, args):
    path = Path(out) / "preregistration.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    pre = {"purpose": "Preliminary development test of three zero-cost Step-4 handover operators: Teacher-Coupled Restart, coherence-gated DEH, Speech-Envelope Anchoring",
           "operators": {"tcr": "first student input = teacher state x(t_4) (tau_1 = t_4)",
                         "cdeh": f"x1hat(t_4) + S(rho, 4) Delta_4 if cos(Delta_4, Delta_3) >= {COH_MIN}, else x1hat(t_4)",
                         "sea": f"generated frames + {SEA_ALPHA} * (mu_prompt - mu_gen), per band, means over the louder half of the frames"},
           "n_speakers": N_SPEAKERS, "conditions": list(CONDS), "theta_db": THETA, "alpha": ALPHA, "delta": DELTA, "folds": FOLDS,
           "seed": SEED, "grids": GRIDS, "policies": POLICIES,
           "primary_test": "each operator vs plain Step 4 on the noisy conditions pooled (exact McNemar, two-sided)",
           "criteria": ["C1 noisy Step-4 failures drop by at least 25% with p < 0.05", "C2 clean Step-4 failures rise by at most 1",
                        "C3 pooled out-of-fold <op>_theta: lower mean NFE than snr_theta with no more failures"],
           "selection_rule": "the operator passing the most criteria; ties broken by the lower pooled <op>_theta mean NFE",
           "input_hashes": {str(p.relative_to(THESIS_ROOT)).replace("\\", "/"): rf.sha256_file(p)
                            for p in (rf.T4_GRID, cg.PILOT_GRID, ex.EXP012 / "labels_new.csv", Path(__file__))},
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
        its = [it for it in ex.items() if it["utt_id"] not in cg.pilot_utts()][:2]
        repro = run_cases(pipe, its, ("clean", "gauss20"), out, reproduce=True)
        (out / "reproduction.json").write_text(json.dumps(repro, indent=2) + "\n", encoding="utf-8")
        for r in repro:
            print(f"reproduction {r['utt_id'][-8:]} {r['condition']} Step {r['step']}: WER {r['wer']:.4f} (stored {r['wer_stored']:.4f}), "
                  f"SIM {r['similarity']:.4f} (stored {r['similarity_stored']:.4f})", flush=True)
        print(f"Smoke run complete: {out}")
        return
    write_prereg(out, args)
    run_cases(pipe, ex.items(), CONDS, out)
    analyze(out)


if __name__ == "__main__":
    main()
