"""Preliminary Colab T4 test of Self-Noise Negative Guidance (SNNG) and the antithetic student pair (EXP-015).

EXP-013 and EXP-014 showed that most noisy Step-4 failures are speaker-similarity failures: the
prompt's noise leaks into the generated voice. Two new methods are tested at the Step-4 handover:

  snng   Self-Noise Negative Guidance. A noise-only prompt is built from the prompt's own quietest
         frames (the 20% lowest-energy frames, tiled over the prompt length). Each teacher step adds
         a third pass with that prompt and steers away from it:
             v = v_c + gamma (v_c - v_u) + eta (v_c - v_noise),   gamma = 2 (unchanged CFG)
         The teacher runs to Step 4 with this velocity; the student completes as usual.
         Cost: 3 passes per teacher step, so Step 4 costs 3*5 + 3 = 18 NFE. eta = 0.5 and 1.0.
  anti   Antithetic student pair. From the plain Step-4 endpoint, the student runs once with its
         usual noise eps and once with -eps; the output closer to the teacher's endpoint (generated
         frames) is kept. Cost 13 + 3 = 16 NFE.

Data: the EXP-012 Quick set (60 development speakers x 6 conditions); the plain Step-4 and Step-8
labels are committed, so only the new outputs are synthesized and scored.

Subcommands: smoke (2 speakers x clean, gauss20; reproduction checks: SNNG with eta = 0 equals the
plain teacher endpoint, and the stored Step-4 and Step-8 scores are reproduced), run, analyze.
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
K, NOISE_FRAC = 4, 0.2
ETAS = {"snng05": 0.5, "snng10": 1.0}
OUTS = tuple(ETAS) + ("anti",)
COST = {4: 13, 8: 21, "snng05": 18, "snng10": 18, "anti": 16}
ALPHA, DELTA, FOLDS, SEED, THETA = ex.ALPHA, ex.DELTA, ex.FOLDS, ex.SEED, ex.THETA
THETA_GRID = ex.GRIDS["theta"]                                   # 40 -> 20 dB
LOW_GRID = [float(x) for x in range(40, 9, -1)]                  # theta_lo candidates, 40 -> 10 dB
FIELDS = ["utt_id", "speaker_id", "condition", "prompt_snr", "d_4", "d4_stored", "noise_frames", "anti_pick"] + \
         [f"{m}_{o}" for o in OUTS for m in ("wer", "similarity", "utmos", "is_failure", "fail_wer", "fail_sim", "hypothesis")] + \
         ["wer_ref", "similarity_ref", "utc"]


# --------------------------------------------------------------------------- operators
def noise_prompt(step_cond, plen, frac=NOISE_FRAC):
    """Copy of step_cond whose prompt frames are the prompt's quietest frames, tiled; works on torch or numpy."""
    p = step_cond[0, :plen, :]
    energy = [float(p[i].mean()) for i in range(plen)]
    n = max(1, int(frac * plen))
    quiet = sorted(range(plen), key=lambda i: energy[i])[:n]
    nc = step_cond.clone() if hasattr(step_cond, "clone") else step_cond.copy()
    nc[0, :plen, :] = p[[quiet[i % n] for i in range(plen)]]
    return nc, n


def snng_endpoints(pipe, ctx, etas, k=K):
    """Teacher Euler pass to step k with negative guidance from the noise-only prompt; returns {eta: x1hat(t_k)}."""
    torch, sampler = pipe.torch, pipe.sampler
    grid = sampler.SWAY_TEACHER_GRID
    ncond, n = noise_prompt(ctx["step_cond"], ctx["prompt_len"])
    out = {}
    with torch.inference_mode():
        for eta in etas:
            x = ctx["y0"].clone()
            for i in range(k + 1):
                t = torch.tensor(grid[i], device=pipe.device, dtype=torch.float32)
                kw = dict(x=x, text=ctx["text_ids"], time=t, mask=None)
                vc = pipe.tts.teacher(cond=ctx["step_cond"], drop_audio_cond=False, drop_text=False, **kw)
                vu = pipe.tts.teacher(cond=ctx["step_cond"], drop_audio_cond=True, drop_text=True, **kw)
                v = vc + (vc - vu) * sampler.CFG_STRENGTH
                if eta:
                    vn = pipe.tts.teacher(cond=ncond, drop_audio_cond=False, drop_text=False, **kw)
                    v = v + eta * (vc - vn)
                ep = x + (1.0 - grid[i]) * v
                if i < k:
                    x = x + (grid[i + 1] - grid[i]) * v
            out[eta] = ep
    return out, n


def student_x(pipe, ctx, endpoint, sign=1.0):
    """run_student_completion's loop (same seeds), with the student noise multiplied by sign; returns the final mel state."""
    torch, sampler = pipe.torch, pipe.sampler
    x = endpoint.clone()
    with torch.inference_mode():
        for tau in sampler.STUDENT_TAUS:
            gen = sampler.make_local_generator((int(ctx["seed"]) + int(tau * 100)) % (2 ** 31 - 1), device=pipe.device)
            eps = sign * torch.randn(x.shape, dtype=torch.float32, device=pipe.device, generator=gen)
            tt = torch.tensor([tau], device=pipe.device, dtype=torch.float32)
            ts = tt.unsqueeze(-1).unsqueeze(-1)
            phi = (1.0 - ts) * eps + ts * x
            pred = pipe.tts.generator(x=phi, cond=ctx["cond"], text=ctx["text_ids"], time=tt, drop_audio_cond=False, drop_text=False)
            x = phi + (1.0 - ts) * pred
    return x


def decode(pipe, ctx, x):
    with pipe.torch.inference_mode():
        mel = x.permute(0, 2, 1) * getattr(pipe.tts, "scale", 1.0)
        return pipe.tts.vocos.decode(mel[..., ctx["prompt_len"]:]).cpu().numpy().squeeze()


def antithetic_pick(xp, xm, anchor, plen):
    """+1 if the eps output is at least as close to the teacher endpoint (generated frames) as the -eps output, else -1."""
    g = lambda a: a[:, plen:, :]
    return 1 if ex._norm(g(xp - anchor)) <= ex._norm(g(xm - anchor)) else -1


# -------------------------------------------------------------------------------- runs
def run_cases(pipe, todo_items, conds, out_dir, csv_name="snng.csv", reproduce=False):
    path = Path(out_dir) / csv_name
    done = rf.done_keys(path, 1)
    stored = ex.stored_cases()
    todo = [(it, c) for it in todo_items for c in conds if (it["utt_id"], c) not in done]
    print(f"[snng] {len(todo)} cases to run ({len(done)} already in {path.name}) on {pipe.gpu}", flush=True)
    repro = []
    for n, (item, cond) in enumerate(todo, 1):
        s = stored[(item["utt_id"], cond)]
        ref = s[8]
        ctx = pipe.prepare(item, cond, item["input_seed"], out_dir)
        E, drift = pipe.teacher(ctx, tuple(range(K + 1)))
        plen = ctx["prompt_len"]
        ends, nframes = snng_endpoints(pipe, ctx, (0.0,) + tuple(ETAS.values()) if reproduce else tuple(ETAS.values()))
        xp, xm = student_x(pipe, ctx, E[K], 1.0), student_x(pipe, ctx, E[K], -1.0)
        with pipe.torch.inference_mode():
            pick = antithetic_pick(xp, xm, E[K], plen)
        row = {"utt_id": item["utt_id"], "speaker_id": item["speaker_id"], "condition": cond, "prompt_snr": ctx["prompt_snr"],
               "d_4": drift[4], "d4_stored": float(ref["d_4"]), "noise_frames": nframes, "anti_pick": pick,
               "wer_ref": float(ref["wer_ref"]), "similarity_ref": float(ref["similarity_ref"])}
        gt = pipe.ground_truth(item)
        audios = {o: decode(pipe, ctx, student_x(pipe, ctx, ends[eta])) for o, eta in ETAS.items()}
        if pick < 0:
            audios["anti"] = decode(pipe, ctx, xm)
        for o in OUTS:
            if o == "anti" and pick > 0:                    # the eps output is the stored plain Step-4 output
                m = {"wer": float(s[4]["wer"]), "similarity": float(s[4]["similarity"]), "utmos": float(s[4]["utmos"]),
                     "hypothesis": s[4]["hypothesis"]}
            else:
                m = pipe.score(audios[o], item, gt)
            fail, diag = pipe.check_failure(m["wer"], row["wer_ref"], m["similarity"], row["similarity_ref"], config=pipe.cfg)
            row.update({f"wer_{o}": m["wer"], f"similarity_{o}": m["similarity"], f"utmos_{o}": m["utmos"], f"is_failure_{o}": bool(fail),
                        f"fail_wer_{o}": bool(diag["wer_failed"]), f"fail_sim_{o}": bool(diag["sim_failed"]), f"hypothesis_{o}": m["hypothesis"]})
        if reproduce:
            with pipe.torch.inference_mode():
                diff = float((ends[0.0] - E[K]).abs().max())
            repro.append({"utt_id": item["utt_id"], "condition": cond, "check": "snng eta=0 endpoint vs plain teacher endpoint", "max_abs_diff": diff})
            E8, _ = pipe.teacher(ctx, (8,))
            for k, T in ((4, E[K]), (8, E8[8])):
                m = pipe.score(decode(pipe, ctx, student_x(pipe, ctx, T)), item, gt)
                repro.append({"utt_id": item["utt_id"], "condition": cond, "step": k, "wer": m["wer"], "wer_stored": float(s[k]["wer"]),
                              "similarity": m["similarity"], "similarity_stored": float(s[k]["similarity"])})
        row["utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        new = not path.exists()
        with open(path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            if new:
                w.writeheader()
            w.writerow(row)
            f.flush()
        print(f"[{n}/{len(todo)}] snng: {item['utt_id']} {cond}", flush=True)
    print(f"[OK] snng complete: {path}", flush=True)
    return repro


# ----------------------------------------------------------------------------- analysis
def build_table(stored, rows):
    table = []
    for r in rows:
        s = stored.get((r["utt_id"], r["condition"]))
        if s is not None:
            table.append({"utt": r["utt_id"], "spk": r["speaker_id"], "cond": r["condition"], "snr": float(r["prompt_snr"]),
                          "d4": float(r["d_4"]), "d4_stored": float(r["d4_stored"]),
                          "fail": {4: rf.is_true(s[4]["is_failure"]), 8: False, **{o: rf.is_true(r[f"is_failure_{o}"]) for o in OUTS}},
                          "sim": {4: float(s[4]["similarity"]), 8: float(s[8]["similarity"]), **{o: float(r[f"similarity_{o}"]) for o in OUTS}},
                          "wer": {4: float(s[4]["wer"]), 8: float(s[8]["wer"]), **{o: float(r[f"wer_{o}"]) for o in OUTS}}})
    return table


def route(hi, lo, o):
    """Plain Step 4 if SNR >= hi; the new output o if lo <= SNR < hi; Step 8 otherwise (None = never)."""
    hi = float("inf") if hi is None else hi
    lo = float("inf") if lo is None else lo
    return lambda r: 4 if r["snr"] >= hi else (o if r["snr"] >= lo else 8)


def certify(train, pol, stratified):
    by = {}
    for r in train:
        d = by.setdefault(r["cond"], [0, 0])
        d[0] += int(r["fail"][pol(r)])
        d[1] += 1
    ps = [rf.binom_cdf(k, n, ALPHA) for k, n in by.values()] if stratified else \
         [rf.binom_cdf(sum(k for k, _ in by.values()), sum(n for _, n in by.values()), ALPHA)]
    return max(ps) <= DELTA


def fixed_sequence(train, grid, make, stratified):
    chosen = None
    for p in grid:
        if not certify(train, make(p), stratified):
            break
        chosen = p
    return chosen


POLICIES = ["reference", "snr_only", "snr_theta"] + [f"{o}_{k}" for o in OUTS for k in ("route", "noisy", "all")]


def fold_policy(name, train, stratified):
    """Returns (policy, chosen parameters) fitted on the training fold."""
    if name == "reference":
        return (lambda r: 8), None
    if name == "snr_only":
        return (lambda r: 4 if r["snr"] >= THETA else 8), None
    hi = fixed_sequence(train, THETA_GRID, lambda p: route(p, None, None), stratified)
    if name == "snr_theta":
        return route(hi, None, None), hi
    o, kind = name.rsplit("_", 1)
    if kind == "all":
        return (lambda r: o), None
    if kind == "noisy":
        return route(hi, float("-inf"), o), (hi, "all below")
    top = float("inf") if hi is None else hi
    lo = fixed_sequence(train, [x for x in LOW_GRID if x < top], lambda p: route(hi, p, o), stratified)
    return route(hi, lo, o), (hi, lo)


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
                pol, p = fold_policy(name, [r for r in table if fold[r["spk"]] != f], strat)
                ch.append(p)
                oof += [(r, pol(r)) for r in table if fold[r["spk"]] == f]
            per = {c: sum(int(r["fail"][k]) for r, k in oof if r["cond"] == c) for c in CONDS}
            res[tag][name] = {"failures": sum(per.values()), "n": len(oof), "mean_nfe": sum(COST[k] for _, k in oof) / len(oof),
                              "worst_condition_risk": max(per[c] / max(1, sum(r["cond"] == c for r, _ in oof)) for c in CONDS),
                              "per_condition": per}
            chosen[tag][name] = ch
    return res, chosen


def quality_vs_reference(table, o, conds):
    """Paired mean differences of the new output minus Step 8 (SIM higher is better, WER lower is better)."""
    rows = [r for r in table if r["cond"] in conds]
    ds = [r["sim"][o] - r["sim"][8] for r in rows]
    dw = [r["wer"][o] - r["wer"][8] for r in rows]
    out = {"n": len(rows), "sim_mean_diff": statistics.mean(ds) if ds else None, "wer_mean_diff": statistics.mean(dw) if dw else None,
           "share_sim_better": sum(d > 0 for d in ds) / len(ds) if ds else None}
    try:
        from scipy.stats import wilcoxon
        out["sim_wilcoxon_p"] = float(wilcoxon(ds).pvalue) if any(ds) else 1.0
        out["wer_wilcoxon_p"] = float(wilcoxon(dw).pvalue) if any(dw) else 1.0
    except ImportError:
        pass
    return out


def analyze(out, allow_partial=False):
    out = Path(out)
    prereg = json.loads((out / "preregistration.json").read_text(encoding="utf-8"))
    rows = rf.read_csv(out / "snng.csv", repair=True)
    table = build_table(ex.stored_cases(), rows)
    expected = N_SPEAKERS * len(CONDS)
    if len(table) < expected and not allow_partial:
        raise SystemExit(f"{len(table)} of {expected} cases; finish the run or pass --allow_partial")
    res = {"preregistration_sha256": prereg["sha256"], "cases": len(table), "expected_cases": expected, "partial": len(table) < expected,
           "drift_reproduction_max_abs_diff": max((abs(r["d4"] - r["d4_stored"]) for r in table), default=None),
           "antithetic_minus_eps_chosen": sum(int(float(r["anti_pick"])) < 0 for r in rows),
           "step4_failures": {c: {"n": sum(r["cond"] == c for r in table),
                                  **{str(k): sum(r["fail"][k] for r in table if r["cond"] == c) for k in (4,) + OUTS}} for c in CONDS},
           "paired_tests": {f"{o} vs Step 4, {g}": ex.paired(table, 4, o, cs) for o in OUTS
                            for g, cs in (("noisy (primary)", NOISY), ("clean", ("clean",)))},
           "quality_vs_step8": {o: {"noisy": quality_vs_reference(table, o, NOISY), "clean": quality_vs_reference(table, o, ("clean",))}
                                for o in OUTS}}
    if len({r["spk"] for r in table}) >= FOLDS:
        res["cross_validation"], res["chosen"] = cross_validate(table)
    res["criteria"] = {}
    for o in OUTS:
        a, b = res["paired_tests"][f"{o} vs Step 4, noisy (primary)"], res["paired_tests"][f"{o} vs Step 4, clean"]
        crit = {"C1": a[f"failures_{o}"] <= 0.75 * a["failures_4"] and a["p"] < 0.05 and a[f"failures_{o}"] < a["failures_4"],
                "C2": b[f"failures_{o}"] <= b["failures_4"] + 1}
        if "cross_validation" in res:
            x, y = res["cross_validation"]["pooled"][f"{o}_route"], res["cross_validation"]["pooled"]["snr_theta"]
            crit["C3"] = x["mean_nfe"] < y["mean_nfe"] and x["failures"] <= y["failures"]
        res["criteria"][o] = crit
    (out / "snng_results.json").write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
    (out / "NOISE_GUIDANCE_REPORT.md").write_text(render(res), encoding="utf-8")
    print(f"Wrote {out / 'NOISE_GUIDANCE_REPORT.md'}", flush=True)
    return res


def render(res):
    names = {"snng05": "SNNG η = 0.5 (18 NFE)", "snng10": "SNNG η = 1.0 (18 NFE)", "anti": "Antithetic pair (16 NFE)"}
    d = res["drift_reproduction_max_abs_diff"]
    f4 = lambda x: "n/a" if x is None else f"{x:+.4f}"
    L = ["# Preliminary test: Self-Noise Negative Guidance and the antithetic student pair", "",
         f"{'**PARTIAL.** ' if res['partial'] else ''}Development speakers only ({N_SPEAKERS} speakers x {len(CONDS)} conditions: {', '.join(CONDS)}); "
         f"{res['cases']} of {res['expected_cases']} cases; preregistration `{res['preregistration_sha256'][:16]}…`. "
         "Every number is computed from the CSV files by `run_noise_guidance_prelim.py analyze`.", "",
         f"Drift reproduction: max |Δd₄| = {'n/a' if d is None else f'{d:.3f}'}. Antithetic pair chose −ε in {res['antithetic_minus_eps_chosen']} cases.", "",
         "## Preregistered criteria (C1 noisy failures −25% with p < 0.05; C2 clean +1 at most; C3 the calibrated route beats the calibrated SNR gate)", "",
         "| Method | C1 | C2 | C3 |", "|---|---|---|---|"]
    for o, c in res["criteria"].items():
        L.append(f"| {names[o]} | " + " | ".join("PASS" if c.get(k) else ("FAIL" if k in c else "n/a") for k in ("C1", "C2", "C3")) + " |")
    L += ["", "## Failures when every input uses the method at Step 4", "", "| Condition | n | plain Step 4 | " + " | ".join(OUTS) + " |",
          "|---|---:|---:|" + "---:|" * len(OUTS)]
    for c, v in res["step4_failures"].items():
        L.append(f"| {c} | {v['n']} | {v['4']} | " + " | ".join(str(v[o]) for o in OUTS) + " |")
    L += ["", "Exact McNemar (two-sided), unadjusted:", ""]
    for name, t in res["paired_tests"].items():
        ks = [k for k in t if k.startswith("failures_")]
        L.append(f"- {name}: {t[ks[0]]} → {t[ks[1]]} of {t['n']} (p {t['p']:.4f})")
    L += ["", "## Average quality vs the Step-8 reference (paired; SIM higher is better, WER lower is better)", "",
          "| Method | Prompts | n | SIM mean diff | SIM better | SIM p | WER mean diff | WER p |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for o, q in res["quality_vs_step8"].items():
        for g, v in q.items():
            p1, p2, share = v.get("sim_wilcoxon_p"), v.get("wer_wilcoxon_p"), v["share_sim_better"]
            L.append(f"| {o} | {g} | {v['n']} | {f4(v['sim_mean_diff'])} | {'n/a' if share is None else f'{share:.0%}'} | "
                     f"{'n/a' if p1 is None else f'{p1:.2g}'} | {f4(v['wer_mean_diff'])} | {'n/a' if p2 is None else f'{p2:.2g}'} |")
    for tag in ("pooled", "stratified"):
        if "cross_validation" not in res:
            break
        L += ["", f"## Out-of-fold policies, {tag} calibration (5-fold speaker CV, α = {ALPHA}, δ = {DELTA})", "",
              "| Policy | Failures | Mean NFE | Worst-condition risk | " + " | ".join(CONDS) + " |", "|---|---:|---:|---:|" + "---:|" * len(CONDS)]
        for n, v in res["cross_validation"][tag].items():
            L.append(f"| {n} | {v['failures']}/{v['n']} | {v['mean_nfe']:.2f} | {v['worst_condition_risk']:.1%} | "
                     + " | ".join(str(v["per_condition"][c]) for c in CONDS) + " |")
        L += ["", "Chosen per fold: " + "; ".join(f"{n}: {v}" for n, v in res["chosen"][tag].items() if v and any(x is not None for x in v))]
    L += ["", "Policies: snr_theta = plain Step 4 if SNR ≥ θ (calibrated), else Step 8. <m>_route = plain Step 4 if SNR ≥ θ_hi (calibrated as snr_theta), "
          "the method if θ_lo ≤ SNR < θ_hi (θ_lo calibrated next on the same fold), else Step 8. <m>_noisy = the method for every input below θ_hi. "
          "<m>_all = the method for every input.", "", "Development evidence only; a claim needs a held-out test with the protocol frozen first."]
    return "\n".join(L) + "\n"


def write_prereg(out, args):
    path = Path(out) / "preregistration.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    pre = {"purpose": "Preliminary development test of Self-Noise Negative Guidance (SNNG) and the antithetic student pair at the Step-4 handover",
           "methods": {"snng": f"v = v_c + gamma (v_c - v_u) + eta (v_c - v_noise); noise-only prompt = the {int(NOISE_FRAC * 100)}% quietest prompt frames tiled; "
                               "teacher to Step 4 with this velocity, then the usual student", "etas": ETAS,
                       "anti": "student with eps and with -eps from the plain Step-4 endpoint; keep the output closer to the endpoint on generated frames"},
           "nfe": {str(k): v for k, v in COST.items()},
           "n_speakers": N_SPEAKERS, "conditions": list(CONDS), "theta_db": THETA, "alpha": ALPHA, "delta": DELTA, "folds": FOLDS, "seed": SEED,
           "theta_grid": THETA_GRID, "theta_lo_grid": LOW_GRID, "policies": POLICIES,
           "primary_test": "each method vs plain Step 4 on the noisy conditions pooled (exact McNemar, two-sided)",
           "criteria": ["C1 noisy Step-4 failures drop by at least 25% with p < 0.05", "C2 clean Step-4 failures rise by at most 1",
                        "C3 pooled out-of-fold <m>_route: lower mean NFE than snr_theta with no more failures"],
           "secondary": "average SIM and WER vs Step 8, paired Wilcoxon, noisy and clean (descriptive)",
           "selection_rule": "the method passing the most criteria; ties broken by the lower pooled <m>_route mean NFE",
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
            if "check" in r:
                print(f"reproduction {r['utt_id'][-8:]} {r['condition']}: {r['check']}: max |diff| {r['max_abs_diff']:.2e}", flush=True)
            else:
                print(f"reproduction {r['utt_id'][-8:]} {r['condition']} Step {r['step']}: WER {r['wer']:.4f} (stored {r['wer_stored']:.4f}), "
                      f"SIM {r['similarity']:.4f} (stored {r['similarity_stored']:.4f})", flush=True)
        print(f"Smoke run complete: {out}")
        return
    write_prereg(out, args)
    run_cases(pipe, ex.items(), CONDS, out)
    analyze(out)


if __name__ == "__main__":
    main()
