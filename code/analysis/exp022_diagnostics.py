"""Descriptive diagnostics for EXP-022, not part of its preregistration.

Part 1 reads the raw CSV files without importing the runner:
- which outputs have a trackable pitch, which shows why C3 could not be computed;
- the reference's quality per prompt condition;
- the Definition 3.1 failure count of every fixed handover, as a cross-check of the report.

Part 2 imports the frozen runner and reuses its calibration and cross-validation unchanged:
- whether the quality limit ever bound;
- what the stop rules would have done had Step 3, or Steps 2 and 3, been candidates;
- a random routing with the same stops per condition, which asks whether the signal picks the right inputs.

Part 3 applies EXP-021's amended P3 (pitch variety judged on clean prompts) to EXP-020's clean-prompt pitch data. Those
speakers designed Voice-Swap, so it only shows that the amended rule can be computed.

These were chosen after the results were seen. They explain the verdict and cannot change it, and the Part 2 and Part 3
numbers cannot support a claim. Committed files are only read (no CSV repair).

Usage: python code/analysis/exp022_diagnostics.py [results dir]
"""
import csv
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "research/experiments"
RESULTS = EXP / "EXP-022-slow-part-handover-prelim/results"
CONDS = ["clean", "noisy15", "gauss18", "gauss20", "gauss22", "gauss25"]
NOISY = CONDS[1:]
METRICS = ("wer", "similarity", "utmos")
FIXED = ["STT_4"] + [f"STS_{k}" for k in range(2, 8)]
DIV_OUTS = ["STS_2", "STS_3", "STS_4", "plain_8"]
EXPLORATORY_STEPS = ((3, 4, 5, 6, 7), (2, 3, 4, 5, 6, 7))
BOOT_B, SEED = 2000, 2026


def read(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def mean(values):
    values = list(values)
    return sum(values) / len(values) if values else float("nan")


def failed(early, ref, th):
    """Definition 3.1, written out again rather than imported."""
    return early["wer"] > max(th["absolute_wer_ceiling"], ref["wer"] + th["relative_wer_margin"]) or \
        early["similarity"] < ref["similarity"] - th["similarity_margin"]


def part1(results):
    div = read(results / "diversity.csv")
    print("== 1a. Pitch tracking (diversity.csv): pairs with a CVf0 value, and mean share of frames voiced in >= 10 of 20 seeds ==")
    for c in ("clean", "noisy15"):
        sub = [r for r in div if r["condition"] == c]
        print(f"{c:8} " + "; ".join(f"{o} {sum(not math.isnan(float(r[f'cvf0_{o}'])) for r in sub)}/{len(sub)} pairs, "
                                    f"{mean(float(r[f'voiced_{o}']) for r in sub):.3f}" for o in DIV_OUTS))

    th = json.loads((ROOT / "code/scheduling/failure_config.json").read_text(encoding="utf-8"))
    steps = {(r["utt_id"], r["condition"]): r for r in read(EXP / "EXP-018-swsr-handover-prelim/results/steps1to8.csv")}
    vs = {(r["utt_id"], r["condition"]): r for r in read(EXP / "EXP-020-selective-handover-prelim/results/selective.csv")}
    cases = []
    for r in read(results / "slowpart.csv"):
        key = (r["utt_id"], r["condition"])
        m = {"plain_8": {x: float(steps[key][f"{x}_g16_8"]) for x in METRICS},
             "STT_4": {x: float(vs[key][f"{x}_STT_4"]) for x in METRICS}}
        m.update({o: {x: float(r[f"{x}_{o}"]) for x in METRICS} for o in FIXED if o != "STT_4"})
        cases.append((r["condition"], m))

    print("\n== 1b. Mean quality per condition: the reference (plain_8) against the Step-4 and Step-3 handovers ==")
    print(f"{'condition':9} {'UTMOS p8':>8} {'UTMOS STS_4':>11} {'SIM p8':>7} {'SIM STS_4':>9} {'WER p8':>7} {'WER STS_3':>9} {'WER STS_4':>9}")
    for c in CONDS:
        g = lambda o, x: mean(min(1.0, m[o][x]) if x == "wer" else m[o][x] for c_, m in cases if c_ == c)
        print(f"{c:9} {g('plain_8', 'utmos'):8.3f} {g('STS_4', 'utmos'):11.3f} {g('plain_8', 'similarity'):7.3f} {g('STS_4', 'similarity'):9.3f} "
              f"{g('plain_8', 'wer'):7.4f} {g('STS_3', 'wer'):9.4f} {g('STS_4', 'wer'):9.4f}")

    print("\n== 1c. Definition 3.1 failures against plain_8, recounted without the runner (all / clean / noisy) ==")
    for o in FIXED:
        f = [(c, failed(m[o], m["plain_8"], th)) for c, m in cases]
        print(f"{o}: {sum(x for _, x in f)}/{len(f)} / {sum(x for c, x in f if c == 'clean')} / {sum(x for c, x in f if c != 'clean')}")


def part2(results):
    sys.path.insert(0, str(ROOT / "code/colab"))
    import run_grid32_prelim as gl  # noqa: E402
    import run_slow_part_handover_prelim as sp  # noqa: E402
    table = sp.build_table(sp.load_stored(), read(results / "slowpart.csv"), gl.thresholds())
    fold = gl.folds_of(table)

    print("\n== 2a. Frozen candidate steps 4-7: first grid position where the training-fold quality limit fails (None = never) ==")
    frozen = (4, 5, 6, 7)
    for signal in ("slow", "full", "voice"):
        out = []
        for f in range(sp.FOLDS):
            train = [r for r in table if fold[r["spk"]] != f]
            grid = sp.lam_grid(train, signal, frozen)
            out.append(next((i for i, lam in enumerate(grid) if not sp.within_calibration(train, sp.stop_rule(signal, lam, frozen))), None))
        print(f"{signal:5} per fold: {out} (grid positions 0..{len(grid) - 1})")

    ci = lambda s: f"{s['mean']:+.4f} [{s['lo']:+.4f}, {s['hi']:+.4f}]"
    for steps in EXPLORATORY_STEPS:
        print(f"\n== 2b. EXPLORATORY: candidate steps {list(steps)}, same calibration and 5-fold speaker CV ==")
        for signal in ("slow", "full", "voice"):
            routed, _ = sp.cross_validate(table, signal, steps)
            s = sp.summarize(routed)
            print(f"{signal:5} passes {s['mean_passes']:.2f} (clean {s['mean_passes_clean']:.2f}, noisy {s['mean_passes_noisy']:.2f}); "
                  f"WER vs Step 8 {ci(s['vs_reference']['wer'])}; similarity vs Step 8 {ci(s['vs_reference']['similarity'])}; "
                  f"WER vs Voice-Swap {ci(s['vs_voice_swap']['wer'])}, noisy only {ci(sp.paired(routed, 'wer', against='STT_4', conds=sp.NOISY))}; "
                  f"failures {s['failures']}/{s['n']}")
            print(f"      stops clean {dict(sorted(Counter(k for r, k in routed if r['cond'] == 'clean').items()))}, "
                  f"noisy {dict(sorted(Counter(k for r, k in routed if r['cond'] != 'clean').items()))}")

    print("\n== 2c. EXPLORATORY: the rule against random routing with the same stops per condition "
          f"(mean capped-WER change vs Step 8 and failures; {BOOT_B} shuffles, 95% range) ==")
    for steps in EXPLORATORY_STEPS:
        for signal in ("slow", "full"):
            routed, _ = sp.cross_validate(table, signal, steps)
            dwer = lambda pairs: mean(sp._capped(r, k, "wer") - sp._capped(r, "plain_8", "wer") for r, k in pairs)
            by = defaultdict(list)
            for r, k in routed:
                by[r["cond"]].append((r, k))
            rng, w, fl = random.Random(SEED), [], []
            for _ in range(BOOT_B):
                pairs = []
                for lst in by.values():
                    ks = [k for _, k in lst]
                    rng.shuffle(ks)
                    pairs += [(r, k) for (r, _), k in zip(lst, ks)]
                w.append(dwer(pairs))
                fl.append(sum(r["fail"][k] for r, k in pairs))
            w.sort()
            fl.sort()
            lo, hi = int(0.025 * BOOT_B), int(0.975 * BOOT_B)
            print(f"steps {list(steps)} {signal:5}: rule WER {dwer(routed):+.4f}, random {mean(w):+.4f} [{w[lo]:+.4f}, {w[hi]:+.4f}]; "
                  f"rule failures {sum(r['fail'][k] for r, k in routed)}, random {mean(fl):.1f} [{fl[lo]}, {fl[hi]}]")

    print("\n== 2d. Fixed handovers against Voice-Swap (STT_4, 14 passes), all 360 cases ==")
    for o in ("STS_2", "STS_3", "STS_4"):
        s = sp.summarize([(r, o) for r in table])
        print(f"{o} ({sp.cost(o)} passes): WER vs Voice-Swap {ci(s['vs_voice_swap']['wer'])}, similarity vs Voice-Swap {ci(s['vs_voice_swap']['similarity'])}")


def part3():
    if str(ROOT / "code/colab") not in sys.path:
        sys.path.insert(0, str(ROOT / "code/colab"))
    import run_voice_swap_confirm as vc  # noqa: E402
    div = [{"speaker_id": r["speaker_id"], "condition": r["condition"], "cvf0_ref": r["cvf0_plain_8"],
            "cvf0_vs": r["cvf0_STT_4"], "cvf0_student": r["cvf0_student4"]}
           for r in read(EXP / "EXP-020-selective-handover-prelim/results/diversity.csv")]
    print(f"\n== 3. EXP-021's amended P3 on EXP-020's development pitch data (judged on {list(vc.P3_CONDS)}; "
          "ref = plain_8, vs = STT_4, student = student4) ==")
    ci = lambda s: "n/a" if s["mean"] is None else f"{s['mean']:+.4f} [{s['lo']:+.4f}, {s['hi']:+.4f}]"
    for c in vc.P3_CONDS:
        a, b = vc.cv_diff(div, c, "vs", "ref"), vc.cv_diff(div, c, "vs", "student")
        passes = a["lo"] is not None and b["lo"] is not None and a["lo"] >= -vc.MARGINS["cvf0"] and b["lo"] > 0
        print(f"{c}: vs minus ref {ci(a)}, n = {a['n']} (needs lower bound >= -{vc.MARGINS['cvf0']}); "
              f"vs minus student {ci(b)}, n = {b['n']} (needs lower bound > 0); computable and passing: {passes}")


def main():
    results = Path(sys.argv[1]) if len(sys.argv) > 1 else RESULTS
    part1(results)
    part2(results)
    part3()


if __name__ == "__main__":
    main()
