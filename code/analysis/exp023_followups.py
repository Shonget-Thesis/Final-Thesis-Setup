"""Pre-specified exploratory follow-ups to EXP-023, written on 2026-11-13 BEFORE the pilot ran (no pilot data existed).

They use only measurements the pilot records anyway and the stored outputs of the same cases and seeds, so they need
no synthesis. They are NOT part of EXP-023's frozen go/no-go rules: a "promising" result here leads only to a frozen
test on new speakers, never to a claim. Every rule below is fixed in this file before the data exist.

1. Guidance-Amplification Routing (GAR), from classifier-free guidance theory. The guided velocity is
   v_c + w (v_c - v_null) with w = 2, so whatever the prompt imposes on the conditional branch, copied recording
   conditions included, is amplified in the guided estimate, while plain teacher-student mismatch is not. The pilot
   records the Step-0 voice gap to the student's draft for the guided teacher (vg_g, used by ARH) and for the
   conditional branch alone (vg_u). GAR-u routes on vg_u (guidance's exaggeration removed; it needs only the
   conditional pass and the draft first, so the student branch costs 5); GAR-ratio routes on vg_g / vg_u.
2. Collapse-Gated Handover (CGH), from the mode-seeking theory of the student's distillation (round 6 of the offline
   screen, negative with the crude stand-in d16_1). On ARH's teacher branches, the teacher's slow-part decision after
   Step 1 relative to the student's own distance from T_0 (dr_g_1, the theory's quantity) gates the teacher: small
   -> the student has not collapsed here, so it finishes alone (8 passes).
3. Decision Commitment Handover (candidate 4 in the EXP-023 README), replayed: dr_g_1 <= l0 -> the student alone
   (8 passes); otherwise dc_g_2 >= lc -> DH_2 (10); otherwise DH_3 (12).
4. Prosody-Agreement Safeguard (PAS), from alignment theory: in infilling TTS the timing of the words is fixed by the
   text and prompt and so already present in the teacher's average T_0, while finer prosody varies by seed. A draft
   whose slow part is far from slow(T_0) has likely misaligned the words, the student's failure mode. Gap at Step 0,
   free: ||slow(S) - slow(T_0)|| = dn_g_1 / dr_g_1, as an RMS over generated frames x 100 bands (frames from EXP-017's
   stored audio_seconds x 24000 / 256). On ARH's student branch, a gap above the cut that keeps 95% of the training
   clean prompts sends the case to Voice-Swap (14 passes) instead of the student alone.
5. Guidance-Choice Handover (GCH), from early-guidance theory: guidance at the first steps pushes past the conditional
   mean, cuts diversity and amplifies what the prompt imposes. At Step 0 (conditional pass, null pass, draft: 3 passes)
   the ratio vg_g / vg_u >= lam -> the unguided teacher decides (DHu_3, 9 passes with the null pass), else the guided
   one (DH_3, 12). lam, out of fold: the lowest (most DHu) threshold whose training cases keep the quality rule on
   means and whose training clean diversity prompts keep the pitch rule.

All thresholds are fitted out of fold over EXP-023's 5 speaker folds (run_decision_handover_pilot.arh_folds).
Routing candidates are compared with the prompt-SNR rule and with ARH on the same cases and folds; a misroute is a
clean case sent to the student alone or a degraded case sent to the whole estimate.
Promising: routing candidates misroute at most 1 more than the SNR rule and no more than ARH, cost at most 14.7 mean
passes and keep clean CVf0 within -0.01 of the stored reference; DCH keeps the screen's quality rule on means
(capped WER <= +0.01 clean, <= +0.025 degraded; similarity >= -0.01; UTMOS >= -0.05), at most 12 mean passes and the
same pitch rule. PAS: fewer Definition 3.1 failures on degraded cases than ARH, degraded mean capped WER no higher than
ARH's, at most 1 mean pass more than ARH, and the pitch rule. GCH: the quality rule on means and the pitch rule, fewer
than 12 mean passes, and DHu_3 alone failing the quality or pitch rule (otherwise the fixed cheaper choice suffices).
(PAS and GCH were added the same day, also before the run.)

Run after the pilot: python code/analysis/exp023_followups.py --out <results folder>
"""
import argparse
import itertools
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "colab"))
import run_decision_handover_pilot as dh  # noqa: E402

GARU_COST = {"W4": 14, "STT_4": 14, "student4": 5}
DCH_COST = {"student4": 8, "DH_2": 10, "DH_3": 12}
GCH_COST = {"DHu_3": 9, "DH_3": 12}
FRAMES_PER_SECOND = 24000 / 256
CGH_STUDENT_COST = 8
RULE = {"misroute_slack": 1, "max_passes": 14.7, "dch_max_passes": 12.0, "cvf0": 0.01,
        "clean_wer": 0.01, "degraded_wer": 0.025, "similarity": 0.01, "utmos": 0.05}


def _stored():
    steps = dh._steps_rows()
    grid = {(r["utt_id"], r["condition"]): r for r in dh.rf.read_csv(dh.GRID32_CSV)}
    vs = {(r["utt_id"], r["condition"]): r for r in dh.rf.read_csv(dh.EXP020_CSV)}
    return steps, grid, vs


def metrics(key, o, rows_by_key, stored):
    """Scores of output o on case key: stored W4/STT_4/student4/reference; DH outputs from the pilot's own rows."""
    steps, grid, vs = stored
    if o.startswith("DH"):
        return {x: float(rows_by_key[key][f"{x}_{o}"]) for x in dh.METRICS}
    src, col = {"W4": (steps, "g16_4"), "STT_4": (vs, "STT_4"), "student4": (grid, "student4"), "plain_8": (steps, "g16_8")}[o]
    return {x: float(src[key][f"{x}_{col}"]) for x in dh.METRICS}


def misroute(key, o):
    return (key[1] == "clean" and o == "student4") or (key[1] != "clean" and o == "W4")


def _delta(m, ref, x):
    return (min(1.0, m[x]) - min(1.0, ref[x])) if x == "wer" else m[x] - ref[x]


def quality_ok(route_of, keys, rows_by_key, stored):
    """The screen's rule on means, against the stored reference."""
    for is_clean, tol in ((True, RULE["clean_wer"]), (False, RULE["degraded_wer"])):
        sub = [k for k in keys if (k[1] == "clean") == is_clean]
        if not sub:
            continue
        mean = lambda x: sum(_delta(metrics(k, route_of[k], rows_by_key, stored), metrics(k, "plain_8", rows_by_key, stored), x)
                             for k in sub) / len(sub)
        if mean("wer") > tol or mean("similarity") < -RULE["similarity"] or mean("utmos") < -RULE["utmos"]:
            return False
    return True


def pitch_diffs(route_of, div_rows, stored_div, speakers=None):
    """Per clean diversity prompt: CVf0 of the output its clean case is routed to, minus the stored reference."""
    out = {}
    for d in div_rows:
        s, o = d["speaker_id"], route_of.get((d["utt_id"], "clean"))
        if s not in stored_div or o is None or (speakers is not None and s not in speakers):
            continue
        x = float(d[f"cvf0_{o}"]) if (o == "W4" or o.startswith("DH")) else stored_div[s][o]
        if not math.isnan(x):
            out[s] = [x - stored_div[s]["plain_8"]]
    return out


def gap_routes(rows, fold, signal):
    """ARH's cut rule (95% clean coverage, largest clean value) applied to `signal`, out of fold."""
    route_of = {}
    for f in range(dh.ARH_FOLDS):
        train = [signal(r) for r in rows if r["condition"] == "clean" and fold[r["speaker_id"]] != f]
        if not train:
            continue
        lo, hi = dh.arh_cuts(train)
        for r in rows:
            if fold[r["speaker_id"]] == f:
                route_of[(r["utt_id"], r["condition"])] = dh.ARH_OUT[dh.arh_branch(signal(r), lo, hi)]
    return route_of


def snr_routes(rows, fold):
    route_of = {}
    for f in range(dh.ARH_FOLDS):
        snrs = sorted((float(r["prompt_snr"]) for r in rows if r["condition"] == "clean" and fold[r["speaker_id"]] != f), reverse=True)
        if not snrs:
            continue
        theta = snrs[min(len(snrs) - 1, int(dh.TA_COVERAGE * len(snrs)))]
        for r in rows:
            if fold[r["speaker_id"]] == f:
                route_of[(r["utt_id"], r["condition"])] = dh.ARH_OUT[dh.ta_branch(float(r["prompt_snr"]), theta, dh.TA_BAND_DB)]
    return route_of


def summary(route_of, passes_of, rows_by_key, stored, div_rows, stored_div):
    keys = [k for k in route_of if k in rows_by_key]
    th = dh.gl.thresholds()
    fail = lambda k: bool(dh.gl.failed(metrics(k, route_of[k], rows_by_key, stored), metrics(k, "plain_8", rows_by_key, stored), th))
    table = [{"spk": rows_by_key[k]["speaker_id"], "cond": k[1],
              "m": {"X": metrics(k, route_of[k], rows_by_key, stored), "plain_8": metrics(k, "plain_8", rows_by_key, stored)}} for k in keys]
    return {"cases": len(keys), "mean_passes": (sum(passes_of[k] for k in keys) / len(keys)) if keys else None,
            "misroutes": sum(misroute(k, route_of[k]) for k in keys),
            "routes": {c: {o: sum(route_of[k] == o for k in keys if k[1] == c) for o in sorted(set(route_of.values()))} for c in dh.CONDS},
            "clean": {x: dh.paired(table, "X", "plain_8", x, ("clean",)) for x in dh.METRICS},
            "degraded": {x: dh.paired(table, "X", "plain_8", x, dh.DEGRADED) for x in dh.METRICS},
            "pitch": dh.boot(pitch_diffs(route_of, div_rows, stored_div)), "quality_rule_on_means": quality_ok(route_of, keys, rows_by_key, stored),
            "failures_clean": sum(fail(k) for k in keys if k[1] == "clean"),
            "failures_degraded": sum(fail(k) for k in keys if k[1] != "clean"),
            "degraded_wer_mean": (lambda v: sum(v) / len(v) if v else None)(
                [min(1.0, metrics(k, route_of[k], rows_by_key, stored)["wer"]) for k in keys if k[1] != "clean"])}


def cgh_routes(rows, fold, rows_by_key, stored, div_rows, stored_div):
    route_of, passes_of = {}, {}
    for f in range(dh.ARH_FOLDS):
        train_rows = [r for r in rows if fold[r["speaker_id"]] != f]
        clean = [float(r["vg_g"]) for r in train_rows if r["condition"] == "clean"]
        if not clean:
            continue
        lo, hi = dh.arh_cuts(clean)
        arh = lambda r: dh.ARH_OUT[dh.arh_branch(float(r["vg_g"]), lo, hi)]
        train_spk = {r["speaker_id"] for r in train_rows}

        def route_with(lam):
            return {(r["utt_id"], r["condition"]): ("student4" if (arh(r) != "student4" and float(r["dr_g_1"]) <= lam) else arh(r))
                    for r in rows}

        def ok(lam):
            ro = route_with(lam)
            keys = [(r["utt_id"], r["condition"]) for r in train_rows]
            if not quality_ok(ro, keys, rows_by_key, stored):
                return False
            p = dh.boot(pitch_diffs(ro, div_rows, stored_div, train_spk))
            return p["mean"] is not None and p["mean"] >= -RULE["cvf0"]

        lam = -1.0
        for v in sorted({float(r["dr_g_1"]) for r in train_rows if arh(r) != "student4"}):
            if not ok(v):
                break
            lam = v
        ro = route_with(lam)
        for r in rows:
            if fold[r["speaker_id"]] == f:
                k = (r["utt_id"], r["condition"])
                route_of[k] = ro[k]
                gated = arh(r) != "student4" and ro[k] == "student4"
                passes_of[k] = CGH_STUDENT_COST if gated else dh.ARH_COST[ro[k]]
    return route_of, passes_of


def _quantiles(vals, n=20):
    v = sorted(vals)
    return sorted({v[min(len(v) - 1, int(i * len(v) / n))] for i in range(n + 1)}) if v else []


def dch_routes(rows, fold, rows_by_key, stored, div_rows, stored_div):
    route_of = {}
    for f in range(dh.ARH_FOLDS):
        train_rows = [r for r in rows if fold[r["speaker_id"]] != f]
        if not train_rows:
            continue
        train_spk, keys = {r["speaker_id"] for r in train_rows}, [(r["utt_id"], r["condition"]) for r in train_rows]

        def route_with(l0, lc):
            return {(r["utt_id"], r["condition"]): ("student4" if float(r["dr_g_1"]) <= l0 else ("DH_2" if float(r["dc_g_2"]) >= lc else "DH_3"))
                    for r in rows}

        best = None
        for l0, lc in itertools.product([-1.0] + _quantiles([float(r["dr_g_1"]) for r in train_rows]),
                                        [2.0] + _quantiles([float(r["dc_g_2"]) for r in train_rows])):
            ro = route_with(l0, lc)
            if not quality_ok(ro, keys, rows_by_key, stored):
                continue
            p = dh.boot(pitch_diffs(ro, div_rows, stored_div, train_spk))
            if p["mean"] is None or p["mean"] < -RULE["cvf0"]:
                continue
            c = sum(DCH_COST[ro[k]] for k in keys) / len(keys)
            if best is None or c < best[0]:
                best = (c, l0, lc)
        ro = route_with(best[1], best[2]) if best else {(r["utt_id"], r["condition"]): "DH_3" for r in rows}
        route_of.update({(r["utt_id"], r["condition"]): ro[(r["utt_id"], r["condition"])] for r in rows if fold[r["speaker_id"]] == f})
    return route_of


def prosody_gap(r, stored):
    """RMS of slow(S) - slow(T_0) over the generated frames x 100 bands, from the pilot's dn_g_1 / dr_g_1."""
    dn, dr = float(r["dn_g_1"]), float(r["dr_g_1"])
    grid = stored[1].get((r["utt_id"], r["condition"]))
    if dr <= 0 or grid is None:
        return float("nan")
    frames = max(1, round(float(grid["audio_seconds"]) * FRAMES_PER_SECOND))
    return (dn / dr) / math.sqrt(frames * 100)


def pas_routes(rows, fold, stored):
    arh = gap_routes(rows, fold, lambda r: float(r["vg_g"]))
    route_of, passes_of = {}, {}
    for f in range(dh.ARH_FOLDS):
        clean = [prosody_gap(r, stored) for r in rows if r["condition"] == "clean" and fold[r["speaker_id"]] != f]
        clean = [g for g in clean if not math.isnan(g)]
        if not clean:
            continue
        cut = dh.arh_cuts(clean)[0]
        for r in rows:
            if fold[r["speaker_id"]] != f:
                continue
            k = (r["utt_id"], r["condition"])
            g = prosody_gap(r, stored)
            o = "STT_4" if (arh[k] == "student4" and not math.isnan(g) and g > cut) else arh[k]
            route_of[k], passes_of[k] = o, dh.ARH_COST[o]
    return route_of, passes_of


def gch_routes(rows, fold, rows_by_key, stored, div_rows, stored_div):
    ratio = lambda r: float(r["vg_g"]) / max(float(r["vg_u"]), 1e-9)
    route_of = {}
    for f in range(dh.ARH_FOLDS):
        train_rows = [r for r in rows if fold[r["speaker_id"]] != f]
        if not train_rows:
            continue
        train_spk, keys = {r["speaker_id"] for r in train_rows}, [(r["utt_id"], r["condition"]) for r in train_rows]

        def route_with(lam):
            return {(r["utt_id"], r["condition"]): ("DHu_3" if ratio(r) >= lam else "DH_3") for r in rows}

        def ok(lam):
            ro = route_with(lam)
            if not quality_ok(ro, keys, rows_by_key, stored):
                return False
            pv = dh.boot(pitch_diffs(ro, div_rows, stored_div, train_spk))
            return pv["mean"] is not None and pv["mean"] >= -RULE["cvf0"]

        lam = float("inf")
        for v in sorted({ratio(r) for r in train_rows}, reverse=True):
            if not ok(v):
                break
            lam = v
        ro = route_with(lam)
        route_of.update({(r["utt_id"], r["condition"]): ro[(r["utt_id"], r["condition"])] for r in rows if fold[r["speaker_id"]] == f})
    return route_of


def analyze(out):
    out = Path(out)
    rows = dh.rf.read_csv(out / "pilot.csv", repair=True)
    div_rows = dh.rf.read_csv(out / "diversity.csv", repair=True) if (out / "diversity.csv").exists() else []
    stored, stored_div = _stored(), dh.stored_cvf0()
    rows = [r for r in rows if (r["utt_id"], r["condition"]) in stored[0]]
    rows_by_key = {(r["utt_id"], r["condition"]): r for r in rows}
    fold = dh.arh_folds({r["speaker_id"] for r in rows})
    res = {}
    snr = snr_routes(rows, fold)
    res["SNR rule"] = summary(snr, {k: dh.SNR_COST[o] for k, o in snr.items()}, rows_by_key, stored, div_rows, stored_div)
    arh = gap_routes(rows, fold, lambda r: float(r["vg_g"]))
    res["ARH (vg_g)"] = summary(arh, {k: dh.ARH_COST[o] for k, o in arh.items()}, rows_by_key, stored, div_rows, stored_div)
    garu = gap_routes(rows, fold, lambda r: float(r["vg_u"]))
    res["GAR-u (vg_u)"] = summary(garu, {k: GARU_COST[o] for k, o in garu.items()}, rows_by_key, stored, div_rows, stored_div)
    garr = gap_routes(rows, fold, lambda r: float(r["vg_g"]) / max(float(r["vg_u"]), 1e-9))
    res["GAR-ratio (vg_g / vg_u)"] = summary(garr, {k: dh.ARH_COST[o] for k, o in garr.items()}, rows_by_key, stored, div_rows, stored_div)
    cgh, cgh_passes = cgh_routes(rows, fold, rows_by_key, stored, div_rows, stored_div)
    res["CGH (ARH + dr_g_1 gate)"] = summary(cgh, cgh_passes, rows_by_key, stored, div_rows, stored_div)
    dch = dch_routes(rows, fold, rows_by_key, stored, div_rows, stored_div)
    res["DCH replay"] = summary(dch, {k: DCH_COST[o] for k, o in dch.items()}, rows_by_key, stored, div_rows, stored_div)
    pas, pas_passes = pas_routes(rows, fold, stored)
    res["PAS (ARH + prosody-agreement safeguard)"] = summary(pas, pas_passes, rows_by_key, stored, div_rows, stored_div)
    for o, c in (("DH_3", 12), ("DHu_3", 8)):
        fixed = {(r["utt_id"], r["condition"]): o for r in rows}
        res[f"{o} alone"] = summary(fixed, {k: c for k in fixed}, rows_by_key, stored, div_rows, stored_div)
    gch = gch_routes(rows, fold, rows_by_key, stored, div_rows, stored_div)
    res["GCH (guided or unguided decision)"] = summary(gch, {k: GCH_COST[o] for k, o in gch.items()}, rows_by_key, stored, div_rows, stored_div)
    snr_m, arh_m = res["SNR rule"]["misroutes"], res["ARH (vg_g)"]["misroutes"]
    for name in ("GAR-u (vg_u)", "GAR-ratio (vg_g / vg_u)", "CGH (ARH + dr_g_1 gate)"):
        r = res[name]
        r["promising"] = bool(r["mean_passes"] is not None and r["pitch"]["mean"] is not None and r["misroutes"] <= snr_m + RULE["misroute_slack"]
                              and r["misroutes"] <= arh_m and r["mean_passes"] <= RULE["max_passes"] and r["pitch"]["mean"] >= -RULE["cvf0"])
    d = res["DCH replay"]
    d["promising"] = bool(d["mean_passes"] is not None and d["pitch"]["mean"] is not None and d["quality_rule_on_means"]
                          and d["mean_passes"] <= RULE["dch_max_passes"] and d["pitch"]["mean"] >= -RULE["cvf0"])
    a, pz = res["ARH (vg_g)"], res["PAS (ARH + prosody-agreement safeguard)"]
    pz["promising"] = bool(pz["mean_passes"] is not None and pz["pitch"]["mean"] is not None and a["degraded_wer_mean"] is not None
                           and pz["failures_degraded"] < a["failures_degraded"] and pz["degraded_wer_mean"] <= a["degraded_wer_mean"]
                           and pz["mean_passes"] <= a["mean_passes"] + 1.0 and pz["pitch"]["mean"] >= -RULE["cvf0"])
    meets = lambda r: bool(r["quality_rule_on_means"] and r["pitch"]["mean"] is not None and r["pitch"]["mean"] >= -RULE["cvf0"])
    g = res["GCH (guided or unguided decision)"]
    g["promising"] = bool(g["mean_passes"] is not None and meets(g) and g["mean_passes"] < 12.0 and not meets(res["DHu_3 alone"]))
    (out / "followups_results.json").write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
    (out / "FOLLOWUPS_REPORT.md").write_text(render(res), encoding="utf-8")
    return res


def render(res):
    ci = lambda v: dh._ci(v)
    L = ["# EXP-023 pre-specified follow-ups (exploratory)", "",
         "Rules fixed in `code/analysis/exp023_followups.py` before the pilot ran. Out of fold over the pilot's 5 speaker folds; "
         "development speakers only. A promising result leads to a frozen test on new speakers, never to a claim.", "",
         "| Rule | Mean passes | Misroutes | Failures clean / degraded | Clean WER | Clean UTMOS | Degraded WER | Clean CVf0 vs reference | Promising |",
         "|---|---:|---:|---|---|---|---|---|---|"]
    for name, r in res.items():
        L.append(f"| {name} | {dh._f(r['mean_passes'], '{:.2f}')} | {r['misroutes']} | {r['failures_clean']} / {r['failures_degraded']} | {ci(r['clean']['wer'])} | "
                 f"{dh._ci(r['clean']['utmos'], '{:+.3f}')} | {ci(r['degraded']['wer'])} | {ci(r['pitch'])} | "
                 f"{'—' if 'promising' not in r else ('yes' if r['promising'] else 'no')} |")
    L += ["", "Routes per condition:", ""] + [f"- {name}: {r['routes']}" for name, r in res.items()]
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="EXP-023 results folder (pilot.csv, diversity.csv)")
    res = analyze(ap.parse_args(argv).out)
    print(render(res))


if __name__ == "__main__":
    main()
