"""Descriptive diagnostics for EXP-032, not part of its preregistration.

These were chosen after the results were seen. They describe the results and cannot change the frozen labels.
Committed files are only read. The runner (code/colab/run_variety_per_pass.py) is imported read-only for its
bootstrap and share functions, so every interval here is computed exactly as the frozen analysis computes them.

- noisy15 measurability: the share of frames pYIN voices in at least half of the 12 seeds, per arm, and how many of
  the 20 speakers have a defined CVf0. An undefined CVf0 makes the speaker mean undefined, which is why the frozen
  noisy15 label is "none".
- The plateau: consecutive differences between the released-grid arms (W2 - W1, W3 - W2, W4 - W3, W8 - W4) and
  W8 - W3 on CVf0 and aligned CVf0, with speaker-level bootstrap intervals.
- The restart level: P25R - W8 on variety and clean quality. Both re-noise a teacher estimate; P25R's first student
  level is 0.2929 (after 14 steps on the 32-interval grid, t = 0.2270), W8's is 0.0761 (after 8 steps on the
  16-interval grid, t = 0.2929). The two arms also differ in grid, so this contrast does not isolate the level.
- Per-speaker direction: how many of the 20 speakers have P25 above each arm, per measure.
- Variety and quality side by side on clean prompts, per arm.
- EXP-031's arms against P25: EXP-031 used the same 20 diversity speakers, the same 12 seeds and the same 30 quality
  cases, and EXP-032 reproduces its STU, W4 and W8 rows (largest CVf0 difference 3.2e-6, UTMOS 4.3e-4). Joining the
  stored rows gives the share of P25's gain kept by TOH3, QH3, BTH3 and RND, and their clean quality against P25.
- noisy15 timing spread: the one variety measure that stays defined; its share of P25's gain per arm. It is computed
  from mel alignment and does not separate speech timing from copied noise.
- EXP-026's restart arms against P25: EXP-026 ran W4R2 and W8R2 (student restarted at 0.2929 from Step 4 or 8, 12 and
  20 passes) and T16 (the 16-step teacher alone) on 30 of the same clean quality cases. Its W8 reference is EXP-018's
  stored score, which matches EXP-032's W8 on those cases (checked here), so the quality differences are paired. Its
  variety used the same 20 speakers but 20 seeds instead of 12, so only the arm means are set side by side; they are
  not paired.

Usage: python code/analysis/exp032_diagnostics.py [results dir]
"""
import csv
import json
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code" / "colab"))
import run_variety_per_pass as v  # noqa: E402

EXP = ROOT / "research/experiments/EXP-032-variety-per-pass"
RES = Path(sys.argv[1]) if len(sys.argv) > 1 else EXP / "results"
EXP031 = ROOT / "research/experiments/EXP-031-snr-toh-method-comparison/results"
EXP026 = ROOT / "research/experiments/EXP-026-transmissibility-test/results"
EXP018_STEPS = ROOT / "research/experiments/EXP-018-swsr-handover-prelim/results/steps1to8.csv"
EXP026_ARMS = {"W4R2": 12, "W8R2": 20, "T16": 32}
DIV_ARMS = v.DIV_ARMS
EXP031_ARMS = ("TOH3", "QH3", "BTH3", "RND")
EXP031_COST = {"TOH3": 12, "QH3": 12, "BTH3": 12, "RND": 4}
VARIETY = ("cvf0", "cvf0_aligned")
METRICS = ("utmos", "wer", "similarity")


def read(path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def num(r, key):
    x = r.get(key, "")
    try:
        return float(x)
    except ValueError:
        return float("nan")


def defined(x):
    return x == x


def boot(vals):
    vals = list(vals)
    out = v._boot(vals)
    out["positive"] = sum(x > 0 for x in vals)
    return out


def share(rows, arm, measure):
    s = v.share_ci(rows, arm, measure)
    return {k: s[k] for k in ("n", "point", "lo", "hi", "undefined_resamples")}


def main():
    div = read(RES / "diversity.csv")
    qual = read(RES / "quality.csv")
    by_cond = {c: [r for r in div if r["condition"] == c] for c in v.CONDS}
    q_by_cond = {c: [r for r in qual if r["condition"] == c] for c in v.CONDS}
    out = {"experiment": "EXP-032", "kind": "post hoc descriptive diagnostics (not preregistered)"}

    # noisy15 measurability
    meas = {}
    for c, rows in by_cond.items():
        meas[c] = {a: {"voiced_median": st.median(num(r, f"voiced_{a}") for r in rows),
                       "voiced_max": max(num(r, f"voiced_{a}") for r in rows),
                       "cvf0_defined": sum(defined(num(r, f"cvf0_{a}")) for r in rows),
                       "cvf0_aligned_defined": sum(defined(num(r, f"cvf0_aligned_{a}")) for r in rows),
                       "speakers": len(rows)} for a in DIV_ARMS}
    out["measurability"] = meas

    clean = by_cond["clean"]
    # plateau along the released grid
    pairs = (("W2", "W1"), ("W3", "W2"), ("W4", "W3"), ("W8", "W4"), ("W8", "W3"), ("W8", "W2"))
    out["plateau_clean"] = {f"{a}-{b}": {m: boot(num(r, f"{m}_{a}") - num(r, f"{m}_{b}") for r in clean)
                                         for m in VARIETY + ("timing_spread",)} for a, b in pairs}

    # restart level contrast (P25R restarts at 0.2929, W8 at 0.0761)
    restart = {"variety_clean": {m: boot(num(r, f"{m}_P25R") - num(r, f"{m}_W8") for r in clean)
                                 for m in VARIETY + ("timing_spread",)},
               "quality": {c: {m: boot(num(r, f"{m}_P25R") - num(r, f"{m}_W8") for r in rows) for m in METRICS}
                           for c, rows in q_by_cond.items()}}
    out["P25R_minus_W8"] = restart

    # per-speaker direction against P25
    out["speakers_P25_above_clean"] = {a: {m: sum(num(r, f"{m}_P25") > num(r, f"{m}_{a}") for r in clean)
                                           for m in VARIETY} for a in DIV_ARMS if a != "P25"}

    # variety and quality side by side (clean)
    qrows = q_by_cond["clean"]
    side = {}
    for a in v.ARMS:
        e = {"passes": v.COST[a], "utmos": st.mean(num(r, f"utmos_{a}") for r in qrows),
             "similarity": st.mean(num(r, f"similarity_{a}") for r in qrows),
             "wer_mean": st.mean(num(r, f"wer_{a}") for r in qrows)}
        if a in DIV_ARMS:
            for m in VARIETY + ("timing_spread",):
                e[m] = st.mean(num(r, f"{m}_{a}") for r in clean)
        side[a] = e
    out["clean_variety_and_quality"] = side

    # EXP-031 arms against P25 (same speakers, seeds and cases)
    d31 = {r["utt_id"]: r for r in read(EXP031 / "diversity.csv") if r["condition"] == "clean"}
    q31 = {r["utt_id"]: r for r in read(EXP031 / "quality.csv") if r["condition"] == "clean"}
    joined = []
    for r in clean:
        s = d31.get(r["utt_id"])
        if s is None:
            continue
        row = dict(r)
        for a in EXP031_ARMS:
            for m in VARIETY + ("timing_spread",):
                row[f"{m}_{a}"] = s[f"{m}_{a}"]
        joined.append(row)
    repro = {a: {m: max(abs(num(r, f"{m}_{a}") - num(d31[r["utt_id"]], f"{m}_{a}")) for r in joined)
                 for m in VARIETY} for a in ("STU", "W4", "W8")}
    qj = [(r, q31[r["utt_id"]]) for r in qrows if r["utt_id"] in q31]
    exp031 = {"speakers": len(joined), "quality_cases": len(qj), "reproduction_max_abs": repro, "arms": {}}
    for a in EXP031_ARMS:
        e = {"passes": EXP031_COST[a],
             "share_of_P25_gain": {m: share(joined, a, m) for m in VARIETY},
             "mean": {m: st.mean(num(r, f"{m}_{a}") for r in joined) for m in VARIETY + ("timing_spread",)},
             "clean_quality_minus_P25": {m: boot(num(s, f"{m}_{a}") - num(r, f"{m}_P25") for r, s in qj)
                                         for m in METRICS},
             "clean_utmos": st.mean(num(s, f"utmos_{a}") for _, s in qj)}
        exp031["arms"][a] = e
    out["exp031_arms_vs_P25"] = exp031

    # noisy15 timing spread, the variety measure that stays defined
    noisy = by_cond["noisy15"]
    out["timing_spread_share_of_P25_gain"] = {
        c: {a: share(by_cond[c], a, "timing_spread") for a in DIV_ARMS if a not in ("P25", "STU")}
        for c in v.CONDS}
    out["noisy15_timing_spread_mean"] = {a: st.mean(num(r, f"timing_spread_{a}") for r in noisy) for a in DIV_ARMS}

    # EXP-026's restart arms (student restarted at 0.2929) against P25, on the same clean cases
    q26 = {r["utt_id"]: r for r in read(EXP026 / "quality.csv")}
    s18 = {r["utt_id"]: r for r in read(EXP018_STEPS) if r["condition"] == "clean"}
    shared = [(r, q26[r["utt_id"]], s18[r["utt_id"]]) for r in qrows if r["utt_id"] in q26 and r["utt_id"] in s18]
    e26 = {"quality_cases": len(shared),
           "scoring_check_W8_EXP018_vs_EXP032_max_abs": {
               m: max(abs(num(s, f"{m}_g16_8") - num(r, f"{m}_W8")) for r, _, s in shared) for m in METRICS},
           "arms": {}}
    d26 = {r["utt_id"]: r for r in read(EXP026 / "diversity.csv")}
    same_spk = [r for r in clean if r["utt_id"] in d26]
    e26["diversity_speakers_shared"] = len(same_spk)
    e26["diversity_seeds"] = {"EXP-026": int(num(next(iter(d26.values())), "n_seeds")), "EXP-032": int(num(clean[0], "n_seeds"))}
    e26["cvf0_means_not_paired"] = {
        "EXP-026": {a: st.mean(num(d26[r["utt_id"]], f"cvf0_{a}") for r in same_spk) for a in ("W4", "W8") + tuple(EXP026_ARMS)},
        "EXP-032": {a: st.mean(num(r, f"cvf0_{a}") for r in same_spk) for a in ("STU", "W4", "W8", "P25", "P25R")}}
    for a, passes in EXP026_ARMS.items():
        e26["arms"][a] = {"passes": passes,
                          "minus_P25": {m: boot(num(q, f"{m}_{a}") - num(r, f"{m}_P25") for r, q, _ in shared) for m in METRICS},
                          "minus_W8": {m: boot(num(q, f"{m}_{a}") - num(r, f"{m}_W8") for r, q, _ in shared) for m in METRICS},
                          "mean": {m: st.mean(num(q, f"{m}_{a}") for _, q, _ in shared) for m in METRICS}}
    out["exp026_restart_arms_vs_P25"] = e26

    path = RES / "diagnostics.json"
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(out, indent=2) + "\n")
    print(f"[OK] wrote {path}")
    return out


if __name__ == "__main__":
    main()
