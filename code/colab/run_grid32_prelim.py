"""EXP-017: preliminary test of the switching time on a 32-interval teacher grid (Colab Tesla T4).

The DMOSpeech 2 paper reports one teacher-guided setting, t_switch = 0.25 with 14 teacher and 2 student steps
(Li et al., Section 4.1, p. 9 of arXiv v1), and its Algorithm 2 runs the teacher to the last sway-grid point at or
below t_switch. Fourteen teacher steps at 0.25 match a 32-interval sway grid. The thesis runs the released sampler,
whose default 16-interval grid stops at the first point above s (0.25 -> Step 8, 21 forward passes). This test changes
the grid and the stopping rule and keeps the rest (guidance, seeds, the student and its restart level):

  grid    t_i = 1 - cos(pi/2 * i/32), i = 0..32
  stop    k*(s) = max{i : t_i <= s} for s in 0.07, 0.10, 0.15, 0.20, 0.25  ->  k* = 7, 9, 11, 13, 14
  output  teacher endpoint x1hat(t_k*) -> student at the released default restart (j = 1, three steps);
          2(k* + 1) + 3 forward passes; the reference on this grid is s = 0.25 (k* = 14, 33 passes)
  paper   the paper's 14 + 2 setting on this grid: k* = 14, student restart j = 2 (two steps), 2 * 15 + 2 = 32 passes
  g16     the thesis's 16-interval Step 4 (s = 0.07, the code default, 13 passes) and Step 8 (s = 0.25, the thesis
          reference, 21 passes), synthesized and scored in the same run and checked against the stored T4 labels

Data: the EXP-012 Quick set (60 development speakers x clean, noisy15, Gaussian 18/20/22/25 dB = 360 cases).
Descriptive development evidence; nothing here is a decision criterion.

Subcommands: smoke (2 non-pilot development speakers x clean, gauss20, with reproduction checks), run, analyze.
"""
import argparse
import csv
import datetime
import hashlib
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_followup_tests as rf  # noqa: E402
import run_consistency_gate_prelim as cg  # noqa: E402
import run_endpoint_extrapolation_prelim as ex  # noqa: E402
import run_noise_guidance_prelim as ng  # noqa: E402

THESIS_ROOT = rf.THESIS_ROOT
N_SPEAKERS, CONDS, NOISY = ex.N_SPEAKERS, ex.CONDS, ex.NOISY


def sway_grid(n):
    """Sway schedule (coefficient -1) with n intervals: t_i = 1 - cos(pi/2 * i/n)."""
    return [1.0 - math.cos(math.pi / 2.0 * i / n) for i in range(n + 1)]


def last_at_or_below(grid, s):
    """The paper's Algorithm 2: the teacher runs to the last grid point t_k <= t_switch."""
    return max(i for i, t in enumerate(grid) if t <= s)


def first_above(grid, s):
    """The released code: the teacher stops at the first grid point above the stopping argument."""
    return min(i for i, t in enumerate(grid) if t > s)


GRID32 = sway_grid(32)
SWITCH = (0.07, 0.10, 0.15, 0.20, 0.25)
NAME = {s: f"s{round(s * 100):03d}" for s in SWITCH}                 # s007 ... s025
K32 = {s: last_at_or_below(GRID32, s) for s in SWITCH}               # 7, 9, 11, 13, 14
CANDS32 = tuple(NAME[s] for s in SWITCH)                             # cheapest first; s025 is the reference
REF32, REF16, PAPER = "s025", "g16_8", "paper"
OUTS = ("g16_4", "g16_8") + CANDS32 + (PAPER,)
COST = {"g16_4": 13, "g16_8": 21, **{NAME[s]: 2 * (K32[s] + 1) + 3 for s in SWITCH}, PAPER: 2 * (K32[0.25] + 1) + 2}
LABEL = {"g16_4": "16-interval Step 4 (s = 0.07, code default)", "g16_8": "16-interval Step 8 (s = 0.25, thesis reference)",
         **{NAME[s]: f"32-interval s = {s:.2f} ({K32[s]} teacher steps)" for s in SWITCH},
         PAPER: "32-interval s = 0.25, two student steps (paper's 14 + 2)"}
METRICS = ("wer", "similarity", "utmos")
FIELDS = ["utt_id", "speaker_id", "condition", "prompt_snr", "gpu", "gt_sr", "case_seconds", "d4_16", "d4_stored"] + \
         [f"d32_{i}" for i in range(1, K32[0.25] + 1)] + \
         [f"{m}_{o}" for o in OUTS for m in METRICS + ("hypothesis",)] + \
         [f"{m}_{o}_stored" for o in ("g16_4", "g16_8") for m in ("wer", "similarity")] + ["utc"]


# ----------------------------------------------------------------------------- synthesis
def grid_teacher(pipe, ctx, grid, wanted):
    """Guided Euler teacher on `grid` with the released guidance (gamma = CFG_STRENGTH, no stochastic term).
    Returns ({i: x1hat(t_i)} for i in wanted, {i: d_i} for i = 1..max(wanted))."""
    torch, sampler = pipe.torch, pipe.sampler
    kmax, plen = max(wanted), ctx["prompt_len"]
    x, prev, ends, drift = ctx["y0"].clone(), None, {}, {}
    with torch.inference_mode():
        for i in range(kmax + 1):
            t = torch.tensor(grid[i], device=pipe.device, dtype=torch.float32)
            kw = dict(x=x, cond=ctx["step_cond"], text=ctx["text_ids"], time=t, mask=None)
            vc = pipe.tts.teacher(drop_audio_cond=False, drop_text=False, **kw)
            vu = pipe.tts.teacher(drop_audio_cond=True, drop_text=True, **kw)
            v = vc + (vc - vu) * sampler.CFG_STRENGTH
            ep = x + (1.0 - grid[i]) * v
            if prev is not None:
                drift[i] = sampler.compute_stabilization_drift(ep, prev, plen)
            if i in wanted:
                ends[i] = ep
            prev = ep
            if i < kmax:
                x = x + (grid[i + 1] - grid[i]) * v
    return ends, drift


def student(pipe, ctx, endpoint, taus):
    """The released student loop (same per-level seeds as run_student_completion) over the restart levels `taus`:
    STUDENT_TAUS for the default restart j = 1, STUDENT_TAUS[1:] for j = 2 (two student steps)."""
    torch, sampler = pipe.torch, pipe.sampler
    x = endpoint.clone()
    with torch.inference_mode():
        for tau in taus:
            gen = sampler.make_local_generator((int(ctx["seed"]) + int(tau * 100)) % (2 ** 31 - 1), device=pipe.device)
            eps = torch.randn(x.shape, dtype=torch.float32, device=pipe.device, generator=gen)
            tt = torch.tensor([tau], device=pipe.device, dtype=torch.float32)
            ts = tt.unsqueeze(-1).unsqueeze(-1)
            phi = (1.0 - ts) * eps + ts * x
            pred = pipe.tts.generator(x=phi, cond=ctx["cond"], text=ctx["text_ids"], time=tt, drop_audio_cond=False, drop_text=False)
            x = phi + (1.0 - ts) * pred
    return x


def run_cases(pipe, todo_items, conds, out_dir, csv_name="grid32.csv", reproduce=False):
    path = Path(out_dir) / csv_name
    done = rf.done_keys(path, 1)
    stored = ex.stored_cases()
    todo = [(it, c) for it in todo_items for c in conds if (it["utt_id"], c) not in done]
    print(f"[grid32] {len(todo)} cases to run ({len(done)} already in {path.name}) on {pipe.gpu}", flush=True)
    repro = []
    taus = list(pipe.sampler.STUDENT_TAUS)
    for n, (item, cond) in enumerate(todo, 1):
        t0 = time.perf_counter()
        s = stored[(item["utt_id"], cond)]
        ctx = pipe.prepare(item, cond, item["input_seed"], out_dir)
        E16, d16 = pipe.teacher(ctx, tuple(range(9)))
        E32, d32 = grid_teacher(pipe, ctx, GRID32, set(K32.values()))
        xs = {"g16_4": student(pipe, ctx, E16[4], taus), "g16_8": student(pipe, ctx, E16[8], taus),
              **{NAME[sw]: student(pipe, ctx, E32[k], taus) for sw, k in K32.items()},
              PAPER: student(pipe, ctx, E32[K32[0.25]], taus[1:])}
        gt = pipe.ground_truth(item)
        row = {"utt_id": item["utt_id"], "speaker_id": item["speaker_id"], "condition": cond, "prompt_snr": ctx["prompt_snr"],
               "gpu": pipe.gpu, "gt_sr": gt[1], "d4_16": d16[4], "d4_stored": float(s[8]["d_4"]),
               **{f"d32_{i}": d for i, d in d32.items()}}
        for k in (4, 8):
            m = ng.stored_metrics(s[k])
            row.update({f"wer_g16_{k}_stored": m["wer"], f"similarity_g16_{k}_stored": m["similarity"]})
        for o in OUTS:
            m = pipe.score(ng.decode(pipe, ctx, xs[o]), item, gt)
            row.update({f"{k}_{o}": m[k] for k in METRICS + ("hypothesis",)})
        if reproduce:
            ends, _ = grid_teacher(pipe, ctx, pipe.sampler.SWAY_TEACHER_GRID, {4, 8})
            with pipe.torch.inference_mode():
                for k in (4, 8):
                    repro.append({"utt_id": item["utt_id"], "condition": cond, "step": k,
                                  "check": "grid_teacher on the released 16-interval grid vs the sampler's endpoint",
                                  "max_abs_diff": float((ends[k] - E16[k]).abs().max())})
        row["case_seconds"] = round(time.perf_counter() - t0, 2)
        row["utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        new = not path.exists()
        with open(path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            if new:
                w.writeheader()
            w.writerow(row)
            f.flush()
            os.fsync(f.fileno())
        print(f"[{n}/{len(todo)}] grid32: {item['utt_id']} {cond} ({row['case_seconds']:.0f} s)", flush=True)
    print(f"[OK] grid32 complete: {path}", flush=True)
    return repro


# ----------------------------------------------------------------------------- analysis (CSV only)
def thresholds():
    return json.loads((THESIS_ROOT / "code/scheduling/failure_config.json").read_text(encoding="utf-8"))


def failed(early, ref, th):
    """Definition 3.1: the early output is clearly worse than the reference output of the same input."""
    return early["wer"] > max(th["absolute_wer_ceiling"], ref["wer"] + th["relative_wer_margin"]) or \
        early["similarity"] < ref["similarity"] - th["similarity_margin"]


def build_table(rows):
    table = []
    for r in rows:
        m = {o: {k: float(r[f"{k}_{o}"]) for k in METRICS} for o in OUTS}
        m.update({f"{o}_stored": {"wer": float(r[f"wer_{o}_stored"]), "similarity": float(r[f"similarity_{o}_stored"])}
                  for o in ("g16_4", "g16_8")})
        table.append({"utt": r["utt_id"], "spk": r["speaker_id"], "cond": r["condition"], "snr": float(r["prompt_snr"]),
                      "gpu": r.get("gpu", ""), "gt_sr": r.get("gt_sr", ""), "secs": float(r["case_seconds"] or 0),
                      "d4": float(r["d4_16"]), "d4_stored": float(r["d4_stored"]), "m": m})
    return table


def group(cond):
    return "clean" if cond == "clean" else "noisy"


def failure_counts(table, early, ref, th):
    per = {c: 0 for c in CONDS}
    for r in table:
        per[r["cond"]] += int(failed(r["m"][early], r["m"][ref], th))
    return {"per_condition": per, "clean": per["clean"], "noisy": sum(per[c] for c in NOISY), "total": sum(per.values())}


def cheapest_safe(table, cands, ref, th):
    """Per input, the cheapest candidate that does not fail against `ref` (the reference itself never fails)."""
    choice = {}
    for r in table:
        pick = next((o for o in cands if o != ref and not failed(r["m"][o], r["m"][ref], th)), ref)
        choice[(r["utt"], r["cond"])] = pick
    counts = {o: sum(v == o for v in choice.values()) for o in cands}
    mean = sum(COST[v] for v in choice.values()) / len(choice) if choice else None
    return {"counts": counts, "mean_cost": mean, "reference_cost": COST[ref],
            "saving": None if mean is None else 1 - mean / COST[ref]}


def paired(table, a, b):
    """Mean differences a - b (SIM higher is better, WER lower is better), share of inputs with higher SIM, Wilcoxon p."""
    out = {}
    for g, keep in (("clean", lambda c: c == "clean"), ("noisy", lambda c: c != "clean"), ("all", lambda c: True)):
        rows = [r for r in table if keep(r["cond"])]
        ds = [r["m"][a]["similarity"] - r["m"][b]["similarity"] for r in rows]
        dw = [r["m"][a]["wer"] - r["m"][b]["wer"] for r in rows]
        v = {"n": len(rows), "sim_mean_diff": statistics.mean(ds) if ds else None, "wer_mean_diff": statistics.mean(dw) if dw else None,
             "share_sim_higher": sum(d > 0 for d in ds) / len(ds) if ds else None}
        try:
            from scipy.stats import wilcoxon
            v["sim_wilcoxon_p"] = float(wilcoxon(ds).pvalue) if any(ds) else 1.0
            v["wer_wilcoxon_p"] = float(wilcoxon(dw).pvalue) if any(dw) else 1.0
        except (ImportError, ValueError):
            pass
        out[g] = v
    return out


def quality(table):
    res = {}
    for o in OUTS:
        res[o] = {}
        for g in ("clean", "noisy", "all"):
            rows = [r for r in table if g == "all" or group(r["cond"]) == g]
            res[o][g] = {k: statistics.mean(r["m"][o][k] for r in rows) if rows else None for k in METRICS}
    return res


def reproduction(table, th):
    diffs = {f"{k}_{o}": max((abs(r["m"][o][k] - r["m"][f"{o}_stored"][k]) for r in table), default=None)
             for o in ("g16_4", "g16_8") for k in ("wer", "similarity")}
    same = sum(failed(r["m"]["g16_4"], r["m"]["g16_8"], th) == failed(r["m"]["g16_4_stored"], r["m"]["g16_8_stored"], th) for r in table)
    return {"max_abs_diff": diffs, "drift_d4_max_abs_diff": max((abs(r["d4"] - r["d4_stored"]) for r in table), default=None),
            "same_step4_failure_label": same, "n": len(table)}


def analyze(out, allow_partial=False):
    out = Path(out)
    prereg = json.loads((out / "preregistration.json").read_text(encoding="utf-8"))
    table = build_table(rf.read_csv(out / "grid32.csv", repair=True))
    expected = N_SPEAKERS * len(CONDS)
    if len(table) < expected and not allow_partial:
        raise SystemExit(f"{len(table)} of {expected} cases; finish the run or pass --allow_partial")
    th = thresholds()
    res = {"preregistration_sha256": prereg["sha256"], "cases": len(table), "expected_cases": expected, "partial": len(table) < expected,
           "gpus": sorted({r["gpu"] for r in table}), "gt_sample_rates": sorted({str(r["gt_sr"]) for r in table}),
           "mean_case_seconds": statistics.mean(r["secs"] for r in table) if table else None,
           "grid32": {"k": {NAME[s]: K32[s] for s in SWITCH}, "t": {NAME[s]: GRID32[K32[s]] for s in SWITCH}}, "cost": COST,
           "fixed_vs_ref32": {o: failure_counts(table, o, REF32, th) for o in CANDS32[:-1] + (PAPER,)},
           "fixed_vs_ref16": {o: failure_counts(table, o, REF16, th) for o in ("g16_4", REF32, PAPER)},
           "cheapest_safe_32": cheapest_safe(table, CANDS32, REF32, th),
           "cheapest_safe_16": cheapest_safe(table, ("g16_4", "g16_8"), REF16, th),
           "quality": quality(table),
           "paired": {f"{a} vs {b}": paired(table, a, b) for a, b in ((PAPER, REF16), (REF32, REF16), (PAPER, REF32), ("g16_4", REF16))},
           "reproduction": reproduction(table, th)}
    (out / "grid32_results.json").write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
    (out / "GRID32_REPORT.md").write_text(render(res), encoding="utf-8")
    print(f"Wrote {out / 'GRID32_REPORT.md'}", flush=True)
    return res


def render(res):
    pct = lambda k, n: f"{k} ({k / n:.1%})" if n else str(k)
    f4 = lambda x: "n/a" if x is None else f"{x:+.4f}"
    n_clean = res["cases"] // len(CONDS) if res["cases"] else 0
    n_noisy = res["cases"] - n_clean
    rep = res["reproduction"]
    secs = "n/a" if res["mean_case_seconds"] is None else f"{res['mean_case_seconds']:.1f} s"
    L = ["# Preliminary test: the switching time on a 32-interval teacher grid (EXP-017)", "",
         f"{'**PARTIAL.** ' if res['partial'] else ''}Development speakers only ({N_SPEAKERS} speakers x {len(CONDS)} conditions: {', '.join(CONDS)}); "
         f"{res['cases']} of {res['expected_cases']} cases on {', '.join(res['gpus'])}; preregistration `{res['preregistration_sha256'][:16]}…`. "
         "Every number is computed from `grid32.csv` by `run_grid32_prelim.py analyze`. Descriptive development evidence.", "",
         "## Settings", "",
         "| Output | Teacher grid and stop | Teacher steps | Student steps | Forward passes |", "|---|---|---:|---:|---:|",
         f"| {LABEL['g16_4']} | 16 intervals, first point above 0.07 | 4 | 3 | {COST['g16_4']} |",
         f"| {LABEL['g16_8']} | 16 intervals, first point above 0.25 | 8 | 3 | {COST['g16_8']} |"]
    for s in SWITCH:
        o = NAME[s]
        L.append(f"| {LABEL[o]} | 32 intervals, last point ≤ {s:.2f} (t = {res['grid32']['t'][o]:.4f}) | {K32[s]} | 3 | {COST[o]} |")
    L += [f"| {LABEL[PAPER]} | 32 intervals, last point ≤ 0.25 | {K32[0.25]} | 2 | {COST[PAPER]} |", "",
          "Teacher steps are Euler steps; the endpoint estimate adds one more guided evaluation, so the cost is 2(k + 1) + student steps.", "",
          "## Reproduction checks", "",
          f"- Fresh 16-interval Step 4 and Step 8 against the stored T4 labels: max |ΔWER| {rep['max_abs_diff']['wer_g16_4']}, {rep['max_abs_diff']['wer_g16_8']}; "
          f"max |ΔSIM| {rep['max_abs_diff']['similarity_g16_4']}, {rep['max_abs_diff']['similarity_g16_8']}; same Step-4 failure label in "
          f"{rep['same_step4_failure_label']} of {rep['n']} cases; max |Δd₄| {rep['drift_d4_max_abs_diff']}.",
          f"- Ground-truth sample rates: {', '.join(res['gt_sample_rates'])}. Mean time per case: {secs}.", "",
          "## Fixed switching times on the 32-interval grid (Definition 3.1 against s = 0.25 on the same grid)", "",
          f"| Output | Passes | Clean failures (of {n_clean}) | Noisy failures (of {n_noisy}) | All |", "|---|---:|---:|---:|---:|"]
    for o, v in res["fixed_vs_ref32"].items():
        L.append(f"| {LABEL[o]} | {COST[o]} | {pct(v['clean'], n_clean)} | {pct(v['noisy'], n_noisy)} | {pct(v['total'], res['cases'])} |")
    L += ["", "Per condition:", "", "| Output | " + " | ".join(CONDS) + " |", "|---|" + "---:|" * len(CONDS)]
    for o, v in res["fixed_vs_ref32"].items():
        L.append(f"| {o} | " + " | ".join(str(v["per_condition"][c]) for c in CONDS) + " |")
    L += ["", "## Against the thesis reference (16-interval Step 8)", "",
          f"| Output | Passes | Clean failures (of {n_clean}) | Noisy failures (of {n_noisy}) | All |", "|---|---:|---:|---:|---:|"]
    for o, v in res["fixed_vs_ref16"].items():
        L.append(f"| {LABEL[o]} | {COST[o]} | {pct(v['clean'], n_clean)} | {pct(v['noisy'], n_noisy)} | {pct(v['total'], res['cases'])} |")
    L += ["", "## Cheapest safe switching time per input (in hindsight)", ""]
    for name, key in (("32-interval grid", "cheapest_safe_32"), ("16-interval grid (Steps 4 and 8 only)", "cheapest_safe_16")):
        v = res[key]
        L.append(f"- {name}: " + ", ".join(f"{o} {k}" for o, k in v["counts"].items()) +
                 f"; mean {v['mean_cost']:.2f} passes against {v['reference_cost']} ({-v['saving']:+.1%} passes)." if v["mean_cost"] is not None else f"- {name}: no cases")
    L += ["", "## Average quality of each output", "", "| Output | Prompts | WER | SIM | UTMOS |", "|---|---|---:|---:|---:|"]
    for o, q in res["quality"].items():
        for g in ("clean", "noisy"):
            v = q[g]
            if v["wer"] is not None:
                L.append(f"| {o} | {g} | {v['wer']:.4f} | {v['similarity']:.4f} | {v['utmos']:.3f} |")
    L += ["", "## Paired comparisons (a − b; SIM higher is better, WER lower is better)", "",
          "| Comparison | Prompts | n | SIM mean diff | SIM higher | SIM p | WER mean diff | WER p |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for name, q in res["paired"].items():
        for g, v in q.items():
            p1, p2, share = v.get("sim_wilcoxon_p"), v.get("wer_wilcoxon_p"), v["share_sim_higher"]
            L.append(f"| {name} | {g} | {v['n']} | {f4(v['sim_mean_diff'])} | {'n/a' if share is None else f'{share:.0%}'} | "
                     f"{'n/a' if p1 is None else f'{p1:.2g}'} | {f4(v['wer_mean_diff'])} | {'n/a' if p2 is None else f'{p2:.2g}'} |")
    L += ["", "Development evidence only (60 development speakers, white Gaussian noise). The 32-interval grid and the stopping rule "
          "follow the paper's Algorithm 2; the handover keeps the released mechanism (endpoint estimate re-noised at the student's "
          "restart level), because the paper does not state the student's later times."]
    return "\n".join(L) + "\n"


def write_prereg(out, args):
    path = Path(out) / "preregistration.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    pre = {"purpose": "Preliminary development test of the switching time on a 32-interval teacher grid with the paper's stopping rule, "
                      "and of the paper's 14 + 2 setting, against the thesis's 16-interval reference",
           "grid32": "t_i = 1 - cos(pi/2 * i/32), i = 0..32; teacher to k* = max{i : t_i <= s} (Li et al., Algorithm 2)",
           "switching_times": list(SWITCH), "k32": {NAME[s]: K32[s] for s in SWITCH}, "outputs": LABEL, "forward_passes": COST,
           "handover": "teacher endpoint estimate at t_k* re-noised at the released student's restart levels: j = 1 (three steps) for every "
                       "output except 'paper', which uses j = 2 (two steps); guidance gamma = 2, no stochastic term; same seeds",
           "references": {"within_grid32": REF32, "thesis": REF16},
           "failure_rule": "Definition 3.1 with code/scheduling/failure_config.json",
           "questions": ["Q1: on the 32-interval grid, do fixed early switching times fail mainly on noisy prompts and rarely on clean ones, "
                         "as on the 16-interval grid (Appendix D)? Failures per output and condition against s = 0.25 on the same grid.",
                         "Q2: does the cheapest safe switching time vary between inputs on this grid? Hindsight distribution and mean cost "
                         "against 33 passes.",
                         "Q3: quality of the paper's 14 + 2 setting and of the code default (16-interval Step 4) against the thesis "
                         "reference (16-interval Step 8): mean WER, SIM and UTMOS, paired Wilcoxon tests and added failures."],
           "criteria": "none; descriptive development evidence",
           "n_speakers": N_SPEAKERS, "conditions": list(CONDS),
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
    ap.add_argument("--any_gpu", action="store_true", help="run on a GPU other than the Tesla T4 (the stored labels will not reproduce)")
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.cmd == "analyze":
        analyze(out, args.allow_partial)
        return
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    pipe = rf.Pipeline(args.dmo_root, args.device)
    if "T4" not in pipe.gpu and not args.any_gpu:
        raise SystemExit(f"This test runs on a Tesla T4 (found {pipe.gpu}): the student's random draws differ between GPU models, "
                         "so the stored T4 labels would not reproduce. Switch the runtime to T4, or pass --any_gpu.")
    if args.cmd == "smoke":
        out = out / "smoke"
        out.mkdir(exist_ok=True)
        its = [it for it in ex.items() if it["utt_id"] not in cg.pilot_utts()][:2]
        repro = run_cases(pipe, its, ("clean", "gauss20"), out, reproduce=True)
        (out / "reproduction.json").write_text(json.dumps(repro, indent=2) + "\n", encoding="utf-8")
        for r in repro:
            print(f"reproduction {r['utt_id'][-8:]} {r['condition']} Step {r['step']}: {r['check']}: max |diff| {r['max_abs_diff']:.2e}", flush=True)
        for r in rf.read_csv(out / "grid32.csv"):
            for k in (4, 8):
                print(f"stored check {r['utt_id'][-8:]} {r['condition']} Step {k}: WER {float(r[f'wer_g16_{k}']):.4f} "
                      f"(stored {float(r[f'wer_g16_{k}_stored']):.4f}), SIM {float(r[f'similarity_g16_{k}']):.4f} "
                      f"(stored {float(r[f'similarity_g16_{k}_stored']):.4f})", flush=True)
        print(f"Smoke run complete: {out}")
        return
    write_prereg(out, args)
    run_cases(pipe, ex.items(), CONDS, out)
    analyze(out)


if __name__ == "__main__":
    main()
