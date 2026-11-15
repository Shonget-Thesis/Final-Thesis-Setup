"""Descriptive diagnostics for EXP-031, not part of its preregistration.

These were chosen after the results were seen. They describe the results and cannot change the frozen verdicts.
Committed files are only read. The runner (code/colab/run_method_comparison.py) is imported read-only so that the
teacher-free controls are judged by exactly the frozen M and C functions; nothing in it is called that synthesizes.

- Teacher-free controls under the frozen rule: the student alone (STU, 4 passes) and the random re-timing (RND, 4 passes)
  judged by M against the five previous methods and by C against W4, as the candidates were.
- What each candidate adds over the student alone: X - STU on clean UTMOS, and per speaker over the five degraded and
  the three hidden conditions (UTMOS, WER, similarity).
- What each candidate adds over the random re-timing beyond T1 and T2: X - RND on WER, similarity and UTMOS, per
  speaker over the degraded conditions and on noisy15 alone.
- Per-condition UTMOS against W4 for every single-formula arm, to show where the degraded margin in M comes from.
- QH3's clean WER: the cases behind the upper bound that failed M.
- BTH3's inputs across conditions: the blind teacher receives the target text and the target length but not the prompt.
  The length is set by the pipeline's duration estimate, which reads the prompt, and the map is aligned to the student's
  draft, which hears it. Reported as the spread of the target length, the blind teacher's length and the map sizes across
  the six conditions per speaker, beside TOH3's.
- Definition 3.1 failures against W8 split into WER and similarity failures
  (code/scheduling/failure_config.json: WER > max(0.125, WER_W8 + 0.01) or SIM < SIM_W8 - 0.02).
- Aligned pitch variety and map size: Spearman correlation across the 20 diversity speakers between each arm's mean map
  displacement and its aligned-CVf0 gain over the student alone, and the share of W4's aligned gain each arm recovers.
- BTH3's compute: the blind teacher's sequence share per case and the implied full-length pass equivalents (8 teacher
  passes at that share plus 4 student passes). This is arithmetic on recorded lengths, not a measured time.

Usage: python code/analysis/exp031_diagnostics.py [results dir]
"""
import csv
import json
import math
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code" / "colab"))
import run_method_comparison as mc  # noqa: E402

EXP = ROOT / "research/experiments/EXP-031-snr-toh-method-comparison"
RES = Path(sys.argv[1]) if len(sys.argv) > 1 else EXP / "results"
FAIL_CFG = ROOT / "code/scheduling/failure_config.json"
SINGLE = ("STU", "RND", "TOH3", "QH3", "BTH3")
METRICS = ("utmos", "wer", "similarity")
TEACHER_PASSES_K3 = 8      # steps 0..3, conditioned and unconditioned branch
STUDENT_PASSES = 4         # the draft S0 and the three restart levels


def read(path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def f(r, key):
    return float(r[key])


def ranks(v):
    order = sorted(range(len(v)), key=lambda i: v[i])
    r = [0.0] * len(v)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2.0
        i = j + 1
    return r


def spearman(x, y):
    a, b = ranks(x), ranks(y)
    ma, mb = st.mean(a), st.mean(b)
    den = math.sqrt(sum((p - ma) ** 2 for p in a) * sum((q - mb) ** 2 for q in b))
    return sum((p - ma) * (q - mb) for p, q in zip(a, b)) / den if den else None


def boot(vals):
    out = mc._boot(list(vals))
    out["positive"] = sum(v > 0 for v in vals)
    return out


def brief(cmp):
    keep = ("clean_utmos", "clean_wer", "clean_similarity", "degraded_utmos", "degraded_wer", "degraded_similarity")
    return {"beats": cmp["beats"], "clean_ok": cmp["clean_ok"], "degraded_ok": cmp["degraded_ok"],
            **{k: {x: cmp[k][x] for x in ("mean", "lo", "hi")} for k in keep}}


def main():
    qual = read(RES / "quality.csv")
    div = read(RES / "diversity.csv")
    cfg = json.loads(FAIL_CFG.read_text(encoding="utf-8"))
    results = json.loads((RES / "method_comparison_results.json").read_text(encoding="utf-8"))
    out = {"note": "Descriptive diagnostics chosen after the results were seen; not preregistered and not a verdict.",
           "frozen_verdicts": results["verdicts"], "quality_rows": len(qual), "diversity_rows": len(div)}

    # Teacher-free controls under the frozen M and C functions.
    controls = {}
    for arm in ("STU", "RND"):
        cmps = {prev: brief(mc.compare(qual, arm, prev)) for prev in mc.PREVIOUS}
        cs = mc.condition_safety(qual, arm)
        controls[arm] = {"M": cmps, "previous_beaten": [p for p, c in cmps.items() if c["beats"]],
                         "C": cs["C"], "C_ci": cs["C_ci"], "C_by_condition": cs["by_condition"]}
    out["teacher_free_controls_frozen_rule"] = controls

    # What each arm adds over the student alone.
    clean = [r for r in qual if r["condition"] == "clean"]
    deg = [r for r in qual if r["condition"] in mc.DEGRADED]
    hid = [r for r in qual if r["condition"] in mc.HIDDEN]
    over_stu = {}
    for arm in ("RND", "TOH3", "QH3", "BTH3"):
        e = {}
        for m in METRICS:
            d = (lambda r, m=m, a=arm: f(r, f"{m}_{a}") - f(r, f"{m}_STU"))
            e[f"clean_{m}"] = boot([d(r) for r in clean])
            e[f"degraded_{m}_per_speaker"] = boot(mc.per_speaker(deg, d))
            e[f"hidden_{m}_per_speaker"] = boot(mc.per_speaker(hid, d))
        over_stu[arm] = e
    out["minus_STU"] = over_stu

    # What the teacher adds over random re-timing beyond T1 and T2: intelligibility and similarity.
    over_rnd = {}
    for arm in ("TOH3", "QH3", "BTH3"):
        e = {}
        for m in ("wer", "similarity", "utmos"):
            d = (lambda r, m=m, a=arm: f(r, f"{m}_{a}") - f(r, f"{m}_RND"))
            e[f"degraded_{m}_per_speaker"] = boot(mc.per_speaker(deg, d))
            e[f"noisy15_{m}"] = boot([d(r) for r in qual if r["condition"] == "noisy15"])
        over_rnd[arm] = e
    out["minus_RND"] = over_rnd

    # Per-condition UTMOS against W4.
    per_cond = {}
    for cond in mc.CONDS:
        sub = [r for r in qual if r["condition"] == cond]
        per_cond[cond] = {a: boot([f(r, f"utmos_{a}") - f(r, "utmos_W4") for r in sub]) for a in SINGLE}
    out["utmos_minus_W4_by_condition"] = per_cond

    # QH3's clean WER against each previous method's clean WER.
    qh_cases = []
    for r in clean:
        row = {"utt_id": r["utt_id"], "wer_QH3": f(r, "wer_QH3"), "hypothesis_QH3": r["hypothesis_QH3"]}
        for prev in mc.PREVIOUS:
            row[f"wer_{prev}"] = mc.value(r, prev, "wer")
        row["hypothesis_W4"] = r["hypothesis_W4"]
        if any(abs(row["wer_QH3"] - row[f"wer_{p}"]) > 1e-12 for p in mc.PREVIOUS):
            qh_cases.append(row)
    out["qh3_clean_wer_cases_differing_from_previous"] = qh_cases

    # BTH3's map across conditions.
    by_spk = {}
    for r in qual:
        by_spk.setdefault(r["speaker_id"], []).append(r)
    spread = {"bth_rms": [], "bth_max": [], "blind_seq_len": [], "toh_rms": [], "toh_max": [], "frames": []}
    for rows in by_spk.values():
        for k in spread:
            v = [f(r, k) for r in rows]
            spread[k].append(max(v) - min(v))
    out["map_spread_across_conditions_per_speaker"] = {
        k: {"max": max(v), "mean": st.mean(v), "speakers_with_zero_spread": sum(x <= 1e-9 for x in v), "n": len(v)}
        for k, v in spread.items()}
    out["target_length_vs_clean"] = {
        "speakers_with_same_length_as_clean": {
            c: sum(1 for rows in by_spk.values() for r in rows if r["condition"] == c
                   and f(r, "frames") == next(f(x, "frames") for x in rows if x["condition"] == "clean"))
            for c in mc.CONDS},
        "prompt_frames_changed_across_conditions": sum(
            1 for rows in by_spk.values() if len({f(r, "seq_len") - f(r, "frames") for r in rows}) > 1),
        "blind_length_equals_target_length": sum(f(r, "blind_seq_len") == f(r, "frames") for r in qual)}

    # Definition 3.1 split into WER and similarity failures.
    d31 = {}
    for cond in mc.CONDS:
        sub = [r for r in qual if r["condition"] == cond]
        d31[cond] = {}
        for a in ("STU", "RND", "W4", "TOH3", "QH3", "BTH3"):
            fw = fs = both = 0
            for r in sub:
                w, wr = f(r, f"wer_{a}"), f(r, "wer_W8")
                s, sr = f(r, f"similarity_{a}"), f(r, "similarity_W8")
                xw = w > max(cfg["absolute_wer_ceiling"], wr + cfg["relative_wer_margin"])
                xs = s < sr - cfg["similarity_margin"]
                fw += xw
                fs += xs
                both += xw or xs
            d31[cond][a] = {"failures": both, "wer_failures": fw, "sim_failures": fs}
    out["definition_3_1_split_vs_W8"] = {"rule": "WER > max(0.125, WER_W8 + 0.01) or SIM < SIM_W8 - 0.02",
                                         "by_condition": d31}
    out["bth3_minus_toh3_similarity_by_condition"] = {
        cond: boot([f(r, "similarity_BTH3") - f(r, "similarity_TOH3") for r in qual if r["condition"] == cond])
        for cond in mc.CONDS}

    # Aligned pitch variety and map size.
    gains = {}
    for arm, disp in (("TOH3", "toh_rms_mean"), ("BTH3", "bth_rms_mean"), ("RND", "rnd_rms_mean")):
        x = [f(r, disp) for r in div]
        gains[f"{arm}_aligned_gain_vs_{disp}"] = spearman(x, [f(r, f"cvf0_aligned_{arm}") - f(r, "cvf0_aligned_STU")
                                                             for r in div])
        gains[f"{arm}_raw_gain_vs_{disp}"] = spearman(x, [f(r, f"cvf0_{arm}") - f(r, "cvf0_STU") for r in div])
    gains["BTH3_minus_TOH3_aligned_vs_rms_difference"] = spearman(
        [f(r, "bth_rms_mean") - f(r, "toh_rms_mean") for r in div],
        [f(r, "cvf0_aligned_BTH3") - f(r, "cvf0_aligned_TOH3") for r in div])
    out["variety_vs_map_size_spearman_20_speakers"] = gains
    mean = {a: st.mean(f(r, f"cvf0_aligned_{a}") for r in div) for a in ("STU", "W4", "W8", "TOH3", "QH3", "BTH3", "RND")}
    gap = mean["W4"] - mean["STU"]
    out["aligned_cvf0_share_of_W4_gain_point_estimates"] = {
        "means": mean, "share": {a: (mean[a] - mean["STU"]) / gap for a in ("TOH3", "QH3", "BTH3", "RND", "W8")}}
    out["mean_map_rms_frames_diversity"] = {k: st.mean(f(r, k) for r in div)
                                            for k in ("toh_rms_mean", "bth_rms_mean", "rnd_rms_mean")}

    # BTH3's compute share.
    share = [f(r, "blind_seq_len") / f(r, "seq_len") for r in qual]
    out["bth3_compute"] = {
        "teacher_sequence_share": {"mean": st.mean(share), "min": min(share), "max": max(share)},
        "full_length_pass_equivalents_if_cost_linear_in_length": STUDENT_PASSES + TEACHER_PASSES_K3 * st.mean(share),
        "full_length_pass_equivalents_if_cost_quadratic_in_length":
            STUDENT_PASSES + TEACHER_PASSES_K3 * st.mean(s * s for s in share),
        "nominal_passes": mc.COST["BTH3"],
        "caveat": "Arithmetic on recorded sequence lengths; no per-arm wall-clock time was recorded."}

    path = RES / "diagnostics.json"
    with open(path, "w", encoding="utf-8", newline="\n") as fh:  # LF on every platform: results/ is stored as -text
        fh.write(json.dumps(out, indent=2) + "\n")
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
