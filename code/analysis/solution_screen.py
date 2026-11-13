"""Fast offline screen of handover solutions on stored T4 outputs (EXPLORATORY, development data).

Written 2026-11-13 at the requester's request ("test different solutions and cases here, fast; if it passes
well, test it on Colab"). Round 2 (marked) was added the same day, after round 1 had been seen, on the same data; round 3
(marked) after round 2. Round 3 replays the decision logic of candidate 4 with stored stand-in payloads. Round 4 (marked) tests
the Two-Axis Handover after a literature check found no published policy choosing both when and what crosses.
Round 5 (marked) tests the Agreement-Routed Handover after a literature check found no method that compares the
teacher's first estimate with the student's draft to decide what crosses. Round 6 (marked) tests Variety-Value
Routing, derived from the mode-seeking theory of the student's distillation, after a literature check. Round 7
(marked) tests Commitment-Timed ARH, derived from speciation theory. Every solution below only routes among outputs that were already synthesized and
scored on the same 360 cases (EXP-012 Quick set: 60 development speakers x clean, gauss25, gauss22, gauss20,
gauss18, noisy15). Decision Handover cannot be screened here: no teacher estimates were stored.

Sources: EXP-018 (whole estimate at Steps 1-8 and drift d16_k), EXP-017 (student alone), EXP-020 (Voice-Swap
STT_k, prompt SNR, pitch variety), EXP-022 (slow part STS_k, pitch variety). The 60 speakers designed those
experiments, so a pass here only earns a Colab test on new speakers; it is not evidence for a claim.

Pass rules, fixed before this screen was run (reference = whole estimate at Step 8, 21 passes):
  clean (n=60):     capped-WER difference upper 95% bound <= +0.01, similarity lower bound >= -0.01,
                    UTMOS lower bound >= -0.10, Definition 3.1 failures <= 3
  degraded (n=300): capped-WER upper bound <= +0.025, similarity lower bound >= -0.01, UTMOS lower bound >= -0.10
  cost:             mean passes <= 14.7 (at least 30% saved)
  pitch variety:    clean CVf0 minus the reference, mean >= -0.01 where every routed output has stored CVf0;
                    otherwise "unmeasured", which a Colab test must settle.
Calibration (thresholds) is out of fold over 5 speaker folds. SNR policies use the earlier replays' rule (the
95th percentile of the training folds' clean SNR); STAC-type and drift thresholds keep the training folds' mean
capped-WER difference <= +0.01 and similarity difference >= -0.01 on clean and on degraded cases, per condition at
twice those margins, and no single case's capped WER more than 0.25 above the reference (as EXP-018's STAC fit).
Bootstrap: over speakers, B = 2000, seed 2026, the same resamples for every solution.

Run: python code/analysis/solution_screen.py [thesis_root]
"""
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

R = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[2]
E = R / "research/experiments"
CONDS = ("clean", "gauss25", "gauss22", "gauss20", "gauss18", "noisy15")
FOLDS, BOOT_B, BOOT_SEED = 5, 2000, 2026
RULES = {"clean_wer": 0.01, "deg_wer": 0.025, "sim": 0.01, "utmos": 0.10, "clean_fails": 3, "max_passes": 14.7,
         "cvf0": 0.01}
CAL = {"wer": 0.01, "sim": 0.01, "case_wer": 0.25}


def rd(p):
    return {(r["utt_id"], r["condition"]): r for r in csv.DictReader(open(p, newline="", encoding="utf-8"))}


S18 = rd(E / "EXP-018-swsr-handover-prelim/results/steps1to8.csv")
S17 = rd(E / "EXP-017-grid32-prelim/results/grid32.csv")
S20 = rd(E / "EXP-020-selective-handover-prelim/results/selective.csv")
S22 = rd(E / "EXP-022-slow-part-handover-prelim/results/slowpart.csv")
TH = json.loads((R / "code/scheduling/failure_config.json").read_text(encoding="utf-8"))
KEYS = sorted(k for k in S18 if k in S17 and k in S20 and k in S22)
assert len(KEYS) == 360, len(KEYS)
SPK = {k: S20[k]["speaker_id"] for k in KEYS}
SNR = {k: float(S20[k]["prompt_snr"]) for k in KEYS}
DRIFT = {k: {j: float(S18[k][f"d16_{j}"]) for j in range(1, 9)} for k in KEYS}
SPEAKERS = sorted(set(SPK.values()))


def metric(k, o, x):
    if o == "student":
        return float(S17[k][f"{x}_student4"])
    if o.startswith("W"):
        return float(S18[k][f"{x}_g16_{o[1:]}"])
    if o.startswith("STS_"):
        return float(S22[k][f"{x}_{o}"])
    if o.startswith("STT_") or o in PART_SWAPS:
        return float(S20[k][f"{x}_{o}"])
    raise KeyError(o)


def cost(o):
    if o == "student":
        return 4
    if o.startswith("W"):
        return 2 * int(o[1:]) + 5
    return 2 * int(o.split("_")[1]) + 6


REF = "W8"
PART_SWAPS = ("TST_4", "TTS_4", "SST_4", "TSS_4")  # EXP-020: voice, slow, fine from Teacher or Student
cap = lambda v: min(1.0, v)
DELTA = {x: {} for x in ("wer", "similarity", "utmos")}


def delta(k, o, x):
    d = DELTA[x].get((k, o))
    if d is None:
        a, b = metric(k, o, x), metric(k, REF, x)
        d = (cap(a) - cap(b)) if x == "wer" else a - b
        DELTA[x][(k, o)] = d
    return d


def fails(k, o):
    if o == REF:
        return False
    return (metric(k, o, "wer") > max(TH["absolute_wer_ceiling"], metric(k, REF, "wer") + TH["relative_wer_margin"])
            or metric(k, o, "similarity") < metric(k, REF, "similarity") - TH["similarity_margin"])


# --------------------------------------------------------------------------- pitch variety (clean prompts only)
def _cvf0(p, prefix_map):
    out = defaultdict(dict)
    for r in csv.DictReader(open(p, newline="", encoding="utf-8")):
        if r["condition"] != "clean":
            continue
        for col, name in prefix_map.items():
            if r.get(col) not in (None, ""):
                out[r["utt_id"]][name] = float(r[col])
    return out


CV = defaultdict(dict)
for src, cols in ((E / "EXP-020-selective-handover-prelim/results/diversity.csv",
                   {"cvf0_student4": "student", "cvf0_plain_4": "W4", "cvf0_plain_8": "W8", "cvf0_STT_4": "STT_4",
                    "cvf0_TST_4": "TST_4", "cvf0_TTS_4": "TTS_4", "cvf0_SST_4": "SST_4", "cvf0_TSS_4": "TSS_4"}),
                  (E / "EXP-022-slow-part-handover-prelim/results/diversity.csv",
                   {"cvf0_STS_2": "STS_2", "cvf0_STS_3": "STS_3", "cvf0_STS_4": "STS_4", "cvf0_plain_8": "W8_22"})):
    for utt, vals in _cvf0(src, cols).items():
        CV[utt].update(vals)


def pitch_variety(route):
    """Mean over the stored clean diversity utterances of (CVf0 of the routed output - CVf0 of the reference)."""
    diffs = []
    for utt, vals in CV.items():
        o = route.get((utt, "clean"))
        if o is None or "W8" not in vals:
            continue
        if o not in vals:
            return None, len(diffs)
        diffs.append(vals[o] - vals["W8"])
    return (sum(diffs) / len(diffs) if diffs else None), len(diffs)


# --------------------------------------------------------------------------- folds and bootstrap
rng = random.Random(BOOT_SEED)
order = list(SPEAKERS)
random.Random(2026).shuffle(order)
FOLD = {sp: i % FOLDS for i, sp in enumerate(order)}
SPK_INDEX = {sp: i for i, sp in enumerate(SPEAKERS)}
RESAMPLES = np.array([[rng.randrange(len(SPEAKERS)) for _ in SPEAKERS] for _ in range(BOOT_B)])


def boot(route, sel, x):
    per = defaultdict(list)
    for k in KEYS:
        if sel(k):
            per[SPK[k]].append(delta(k, route[k], x))
    vals = np.full(len(SPEAKERS), np.nan)
    for sp, v in per.items():
        vals[SPK_INDEX[sp]] = sum(v) / len(v)
    means = np.nanmean(vals[RESAMPLES], axis=1)
    return float(np.nanmean(vals)), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


# --------------------------------------------------------------------------- calibration helpers
def snr_threshold(train_keys):
    tr = sorted((SNR[k] for k in train_keys if k[1] == "clean"), reverse=True)
    return tr[min(len(tr) - 1, int(0.95 * len(tr)))]


def quality_ok(route, train_keys, case_guard=True):
    groups = (("clean",), CONDS[1:])
    for conds in groups:
        sub = [k for k in train_keys if k[1] in conds]
        if not sub:
            continue
        if sum(delta(k, route(k), "wer") for k in sub) / len(sub) > CAL["wer"]:
            return False
        if sum(delta(k, route(k), "similarity") for k in sub) / len(sub) < -CAL["sim"]:
            return False
    for c in CONDS:
        sub = [k for k in train_keys if k[1] == c]
        if sub and (sum(delta(k, route(k), "wer") for k in sub) / len(sub) > 2 * CAL["wer"]
                    or sum(delta(k, route(k), "similarity") for k in sub) / len(sub) < -2 * CAL["sim"]):
            return False
    return (not case_guard) or all(delta(k, route(k), "wer") <= CAL["case_wer"] for k in train_keys)


def action(k, step):
    return sum(DRIFT[k][j] for j in range(1, step + 1))


def fit_stac(train_keys, steps=tuple(range(1, 8)), fallback=REF):
    lams = {}

    def route_with(cur):
        def route(k):
            for s in steps:
                if cur.get(s) is not None and action(k, s) >= cur[s]:
                    return f"W{s}"
            return fallback
        return route

    for s in steps:
        vals = sorted({round(action(k, s), 5) for k in train_keys}, reverse=True)
        if len(vals) > 80:
            vals = [vals[i] for i in sorted({int(round(i * (len(vals) - 1) / 79)) for i in range(80)})]
        best = None
        for v in vals:
            trial = dict(lams); trial[s] = v
            if not quality_ok(route_with(trial), train_keys):
                break
            best = v
        lams[s] = best
    return route_with(lams)


def fit_snr_drift(train_keys, theta, steps=(4, 5, 6)):
    """Chapter 3's Definition 3.4: the first candidate step with d_k <= lambda and SNR >= theta, else the reference."""
    def route_with(lam):
        return lambda k: next((f"W{s}" for s in steps if lam is not None and DRIFT[k][s] <= lam and SNR[k] >= theta), REF)
    grid = sorted({round(DRIFT[k][s], 5) for k in train_keys for s in steps})
    if len(grid) > 80:
        grid = [grid[i] for i in sorted({int(round(i * (len(grid) - 1) / 79)) for i in range(80)})]
    chosen = None
    for lam in grid:
        if not quality_ok(route_with(lam), train_keys):
            break
        chosen = lam
    return route_with(chosen)


def out_of_fold(make):
    """make(train_keys) -> route(k); returns {case: output} with every case routed by a fit on the other folds."""
    route = {}
    for f in range(FOLDS):
        train = [k for k in KEYS if FOLD[SPK[k]] != f]
        r = make(train)
        route.update({k: r(k) for k in KEYS if FOLD[SPK[k]] == f})
    return route


def fixed(o):
    return {k: o for k in KEYS}


def snr_route(clean_out, noisy_out):
    def make(train):
        th = snr_threshold(train)
        return lambda k: clean_out if SNR[k] >= th else noisy_out
    return out_of_fold(make)


def snr_student_stac():
    def make(train):
        th = snr_threshold(train)
        clean_like = [k for k in train if SNR[k] >= th]
        stac = fit_stac(clean_like)
        return lambda k: stac(k) if SNR[k] >= th else "student"
    return out_of_fold(make)


SOLUTIONS = [
    ("reference, Step 8 whole", "known", fixed(REF)),
    ("whole estimate, Step 3", "known", fixed("W3")),
    ("whole estimate, Step 4 (code default)", "known", fixed("W4")),
    ("slow part, Step 3 (EXP-022)", "ours, earlier", fixed("STS_3")),
    ("Voice-Swap, Step 4 (EXP-020)", "ours, earlier", fixed("STT_4")),
    ("student alone", "known", fixed("student")),
    ("SNR only: clean-like Step 4, else reference", "known", snr_route("W4", REF)),
    ("SNR + drift, Definition 3.4", "Chapter 3", out_of_fold(lambda tr: fit_snr_drift(tr, snr_threshold(tr)))),
    ("STAC, Steps 1-7", "Appendix E", out_of_fold(fit_stac)),
    ("SNR -> student, clean-like whole Step 2", "known", snr_route("W2", "student")),
    ("SNR -> student, clean-like whole Step 3", "known", snr_route("W3", "student")),
    ("SNR -> student, clean-like whole Step 4", "known", snr_route("W4", "student")),
    ("SNR -> student, clean-like slow part Step 2", "ours, earlier", snr_route("STS_2", "student")),
    ("SNR -> student, clean-like slow part Step 3", "ours, earlier", snr_route("STS_3", "student")),
    ("SNR -> student, clean-like Voice-Swap Step 4", "ours, earlier", snr_route("STT_4", "student")),
    ("SNR -> student, STAC on clean-like", "known parts", snr_student_stac()),
]


# --------------------------------------------------------------------------- round 2 (designed after round 1)
CANDIDATES = [f"W{k}" for k in range(1, 9)] + [f"STS_{k}" for k in range(2, 8)] + [f"STT_{k}" for k in range(4, 8)] \
             + list(PART_SWAPS) + ["student"]


def cvf0_mean(o):
    d = [v[o] - v["W8"] for v in CV.values() if o in v and "W8" in v]
    return sum(d) / len(d) if d else None


def strict_route(top):
    """Student only below the lowest clean-prompt SNR of the training folds."""
    def make(train):
        th = min(SNR[k] for k in train if k[1] == "clean")
        return lambda k: top if SNR[k] >= th else "student"
    return out_of_fold(make)


def three_way(margin_db, top="W4", middle="STT_4"):
    """Clean-like prompts to `top`; prompts within `margin_db` below the SNR threshold to `middle`; the rest to the
    student."""
    def make(train):
        hi = snr_threshold(train)
        return lambda k: top if SNR[k] >= hi else (middle if SNR[k] >= hi - margin_db else "student")
    return out_of_fold(make)


def drift_protected(top="W4"):
    """Student only if the prompt SNR is low AND the teacher's first step barely moved (d16_1 below the training
    folds' 5th percentile on clean prompts). The check costs teacher Steps 0-1 (4 passes) for low-SNR prompts."""
    route, extra = {}, {}
    for f in range(FOLDS):
        train = [k for k in KEYS if FOLD[SPK[k]] != f]
        th = snr_threshold(train)
        d1 = sorted(DRIFT[k][1] for k in train if k[1] == "clean")
        lam = d1[int(0.05 * len(d1))]
        for k in KEYS:
            if FOLD[SPK[k]] != f:
                continue
            if SNR[k] >= th or DRIFT[k][1] >= lam:
                route[k] = top          # for low-SNR prompts the teacher already ran Steps 0-1 inside `top`
            else:
                route[k], extra[k] = "student", 4
    return route, extra


def stac_over(prefix, steps):
    """STAC's stage-wise cumulative drift, handing over a part-wise payload instead of the whole estimate."""
    def make(train):
        lams = {}

        def route_with(cur):
            return lambda k: next((f"{prefix}{s}" for s in steps if cur.get(s) is not None and action(k, s) >= cur[s]), REF)

        for s in steps:
            vals = sorted({round(action(k, s), 5) for k in train}, reverse=True)
            if len(vals) > 80:
                vals = [vals[i] for i in sorted({int(round(i * (len(vals) - 1) / 79)) for i in range(80)})]
            best = None
            for v in vals:
                trial = dict(lams); trial[s] = v
                if not quality_ok(route_with(trial), train):
                    break
                best = v
            lams[s] = best
        return route_with(lams)
    return out_of_fold(make)


def snr_table(n_bins=6):
    """Learned SNR lookup table: per SNR band (training-fold quantiles), the cheapest stored output whose training
    cases meet mean capped-WER difference <= +0.01, similarity >= -0.01 and UTMOS >= -0.05; bands that are mostly
    clean also need stored pitch variety within -0.01 of the reference."""
    def make(train):
        snrs = sorted(SNR[k] for k in train)
        edges = [snrs[int(i * len(snrs) / n_bins)] for i in range(1, n_bins)]
        band = lambda k: sum(SNR[k] >= e for e in edges)
        choice = {}
        for b in range(n_bins):
            sub = [k for k in train if band(k) == b]
            pick = REF
            for o in sorted(CANDIDATES, key=cost):
                if not sub:
                    break
                mean = lambda x: sum(delta(k, o, x) for k in sub) / len(sub)
                if mean("wer") > 0.01 or mean("similarity") < -0.01 or mean("utmos") < -0.05:
                    continue
                if sum(k[1] == "clean" for k in sub) >= len(sub) / 2:
                    cv = cvf0_mean(o)
                    if cv is None or cv < -RULES["cvf0"]:
                        continue
                pick = o
                break
            choice[b] = pick
        return lambda k: choice[band(k)]
    return out_of_fold(make)


def hindsight():
    """Not a solution: per case, the cheapest stored output with capped WER no worse than the reference and
    similarity at most 0.01 lower. Shows the headroom only."""
    out = {}
    for k in KEYS:
        ok = [o for o in CANDIDATES if delta(k, o, "wer") <= 0 and delta(k, o, "similarity") >= -0.01]
        out[k] = min(ok, key=cost) if ok else REF
    return out


ROUND2 = [
    ("payload routing at Step 4: whole / Voice-Swap", "ours, earlier", snr_route("W4", "STT_4")),
    ("strict SNR -> student, clean-like whole Step 4", "known", strict_route("W4")),
    ("strict SNR -> student, clean-like Voice-Swap Step 4", "ours, earlier", strict_route("STT_4")),
    ("strict SNR -> student, clean-like slow part Step 3", "ours, earlier", strict_route("STS_3")),
    ("three-way SNR, 2 dB band: whole 4 / Voice-Swap 4 / student", "new combination", three_way(2.0)),
    ("three-way SNR, 4 dB band: whole 4 / Voice-Swap 4 / student", "new combination", three_way(4.0)),
    ("three-way SNR, 4 dB band: Voice-Swap 4 / Voice-Swap 4 / student", "new combination", three_way(4.0, top="STT_4")),
    ("drift-protected SNR -> student, else whole Step 4", "new combination", drift_protected("W4")),
    ("drift-protected SNR -> student, else Voice-Swap Step 4", "new combination", drift_protected("STT_4")),
    ("STAC over Voice-Swap, Steps 4-7", "new combination", stac_over("STT_", range(4, 8))),
    ("STAC over slow part, Steps 2-7", "new combination", stac_over("STS_", range(2, 8))),
    ("learned SNR table, 6 bands, cheapest safe output", "known", snr_table(6)),
] + [(f"part swap {o}, Step 4", "ours, earlier", fixed(o)) for o in PART_SWAPS] \
  + [(f"SNR -> student, clean-like part swap {o}", "ours, earlier", snr_route(o, "student")) for o in PART_SWAPS]


# --------------------------------------------------------------------------- round 3 (designed after rounds 1-2)
# d16_1 = ||T_1 - T_0|| / ||T_1|| is exactly the size of the teacher's decision after Step 1 (whole band, relative to
# T_1), so candidate 4's "empty decision" test can be replayed. Its Step-2 "settled" test (the decision's direction)
# needs the vectors, so the contraction d_2 / d_1 stands in for it. Round-3 fits drop EXP-018's single-case guard
# (no case more than 0.25 capped WER above the reference), which the pass rules do not have; with it, no round-3
# rule found any feasible threshold (first run, 2026-11-13). This change was made after that run. Stored outputs stand in for its payload: the slow
# part STS_k (the g = 0 member of the Decision Handover family), Voice-Swap or the whole estimate.
import itertools


def _quantiles(vals, n=20):
    v = sorted(vals)
    return sorted({v[min(len(v) - 1, int(i * len(v) / n))] for i in range(n + 1)})


def fit_grid(train, make, grids):
    """make(*params) -> (route_fn, extra_fn). Keeps the parameters with the lowest training mean passes among those
    that meet the training quality rule (quality_ok)."""
    best = None
    for params in itertools.product(*grids):
        route, extra = make(*params)
        if not quality_ok(route, train, case_guard=False):
            continue
        c = sum((21 if route(k) == REF else cost(route(k))) + extra(k) for k in train) / len(train)
        if best is None or c < best[0]:
            best = (c, params)
    return None if best is None else best[1]


def out_of_fold_grid(make, grid_fn):
    route, extra = {}, {}
    for f in range(FOLDS):
        train = [k for k in KEYS if FOLD[SPK[k]] != f]
        params = fit_grid(train, make, grid_fn(train))
        r, e = make(*params) if params is not None else ((lambda k: REF), (lambda k: 0))
        for k in KEYS:
            if FOLD[SPK[k]] == f:
                route[k], extra[k] = r(k), e(k)
    return route, extra


def contraction(k, step):
    return DRIFT[k][step] / max(DRIFT[k][step - 1], 1e-9)


def commitment_logic(early, late, snr_strict=False):
    """Candidate 4's stop rule: an empty Step-1 decision (d_1 < l0) ends the teacher and the student finishes alone
    (teacher Steps 0-1 cost 4 passes); a settling decision at Step 2 (d_2 / d_1 <= lc) hands over `early`; otherwise
    `late`. With snr_strict, prompts below the training folds' lowest clean SNR go to the student first (4 passes)."""
    def make_factory(train):
        th = min(SNR[k] for k in train if k[1] == "clean") if snr_strict else None

        def make(l0, lc):
            def route(k):
                if th is not None and SNR[k] < th:
                    return "student"
                if DRIFT[k][1] < l0:
                    return "student"
                return early if contraction(k, 2) <= lc else late

            def extra(k):
                if th is not None and SNR[k] < th:
                    return 0
                return 4 if DRIFT[k][1] < l0 else 0
            return route, extra
        return make

    route, extra = {}, {}
    for f in range(FOLDS):
        train = [k for k in KEYS if FOLD[SPK[k]] != f]
        make = make_factory(train)
        grids = ([0.0] + _quantiles([DRIFT[k][1] for k in train], 20), [-1.0] + _quantiles([contraction(k, 2) for k in train], 20))
        params = fit_grid(train, make, grids)
        r, e = make(*params) if params is not None else ((lambda k: REF), (lambda k: 0))
        for k in KEYS:
            if FOLD[SPK[k]] == f:
                route[k], extra[k] = r(k), e(k)
    return route, extra


def tail(k, step):
    """Geometric-tail estimate of the teacher's remaining relative endpoint change after `step`: if the drift contracts
    by r = d_k / d_k-1 < 1 per step, the rest sums to d_k * r / (1 - r)."""
    r = contraction(k, step)
    return float("inf") if r >= 1 else DRIFT[k][step] * r / (1 - r)


def tail_stop(prefix, steps, empty_gate=False):
    """Hand over `prefix`k at the first candidate step whose geometric tail is <= lam; otherwise the reference.
    With empty_gate, an empty Step-1 decision (d_1 < l0) sends the input to the student (4 extra passes)."""
    def make_factory(train):
        def make(lam, l0=0.0):
            def route(k):
                if empty_gate and DRIFT[k][1] < l0:
                    return "student"
                return next((f"{prefix}{st}" for st in steps if tail(k, st) <= lam), REF)
            return route, (lambda k: 4 if empty_gate and DRIFT[k][1] < l0 else 0)
        return make

    def grids(train):
        tails = [tail(k, st) for k in train for st in steps if tail(k, st) != float("inf")]
        g = [_quantiles(tails, 30)]
        if empty_gate:
            g.append([0.0] + _quantiles([DRIFT[k][1] for k in train], 20))
        return g

    route, extra = {}, {}
    for f in range(FOLDS):
        train = [k for k in KEYS if FOLD[SPK[k]] != f]
        make = make_factory(train)
        params = fit_grid(train, make, grids(train))
        r, e = make(*params) if params is not None else ((lambda k: REF), (lambda k: 0))
        for k in KEYS:
            if FOLD[SPK[k]] == f:
                route[k], extra[k] = r(k), e(k)
    return route, extra


ROUND3 = [
    ("commitment logic: empty -> student, settled -> slow part 2, else slow part 3", "candidate 4 logic, stand-in payload", commitment_logic("STS_2", "STS_3")),
    ("commitment logic: empty -> student, settled -> whole 2, else whole 3", "candidate 4 logic, stand-in payload", commitment_logic("W2", "W3")),
    ("commitment logic: empty -> student, settled -> slow part 3, else Voice-Swap 4", "candidate 4 logic, stand-in payload", commitment_logic("STS_3", "STT_4")),
    ("commitment logic + strict SNR first, slow part 2 / slow part 3", "candidate 4 logic, stand-in payload", commitment_logic("STS_2", "STS_3", snr_strict=True)),
    ("commitment logic + strict SNR first, slow part 3 / Voice-Swap 4", "candidate 4 logic, stand-in payload", commitment_logic("STS_3", "STT_4", snr_strict=True)),
    ("geometric-tail stop, whole estimate Steps 2-7", "new formula", tail_stop("W", range(2, 8))),
    ("geometric-tail stop, slow part Steps 2-7", "new formula", tail_stop("STS_", range(2, 8))),
    ("geometric-tail stop, Voice-Swap Steps 4-7", "new formula", tail_stop("STT_", range(4, 8))),
    ("empty -> student, then geometric-tail stop over Voice-Swap 4-7", "new formula", tail_stop("STT_", range(4, 8), empty_gate=True)),
    ("empty -> student, then geometric-tail stop over slow part 2-7", "new formula", tail_stop("STS_", range(2, 8), empty_gate=True)),
]


# --------------------------------------------------------------------------- round 4: Two-Axis Handover
# Axis 1, what crosses, by prompt SNR (0 passes): clean-like -> whole estimate; within `margin_db` below the SNR
# threshold -> Voice-Swap at Step 4; below that -> nothing (student alone, 4 passes).
# Axis 2, when, by STAC's trajectory action (0 passes), on the whole-estimate branch only: hand over at the earliest
# early step whose action reaches its stage threshold, else at Step 4.
# Calibration rule (fixed before the first run; mirrors the pass rules on means): on the branch's training cases,
# mean capped-WER difference <= +0.01 on clean and <= +0.025 on degraded cases, mean similarity difference >= -0.01
# and mean UTMOS difference >= -0.05; optionally EXP-018's single-case guard.


def axis_rule_ok(route, keys, case_guard=False):
    for conds, wer_tol in ((("clean",), 0.01), (CONDS[1:], 0.025)):
        sub = [k for k in keys if k[1] in conds]
        if not sub:
            continue
        mean = lambda x: sum(delta(k, route(k), x) for k in sub) / len(sub)
        if mean("wer") > wer_tol or mean("similarity") < -0.01 or mean("utmos") < -0.05:
            return False
    return (not case_guard) or all(delta(k, route(k), "wer") <= CAL["case_wer"] for k in keys)


def fit_stage_exits(keys, early_steps, fallback, case_guard=False):
    """STAC-style stage thresholds on A_k for the early steps, fitted in order; each threshold is the most permissive
    one that keeps the rule on `keys`."""
    lams = {}

    def route_with(cur):
        return lambda k: next((f"W{st}" for st in early_steps if cur.get(st) is not None and action(k, st) >= cur[st]), fallback)

    for st in early_steps:
        vals = sorted({round(action(k, st), 5) for k in keys}, reverse=True)
        if len(vals) > 80:
            vals = [vals[i] for i in sorted({int(round(i * (len(vals) - 1) / 79)) for i in range(80)})]
        best = None
        for v in vals:
            trial = dict(lams); trial[st] = v
            if not axis_rule_ok(route_with(trial), keys, case_guard):
                break
            best = v
        lams[st] = best
    return route_with(lams)


def two_axis(margin_db=2.0, early_steps=(2, 3), case_guard=False):
    def make(train):
        hi = snr_threshold(train)
        branch = lambda k: "whole" if SNR[k] >= hi else ("vs" if SNR[k] >= hi - margin_db else "student")
        whole = fit_stage_exits([k for k in train if branch(k) == "whole"], early_steps, "W4", case_guard)

        def route(k):
            b = branch(k)
            return "student" if b == "student" else ("STT_4" if b == "vs" else whole(k))
        return route
    return out_of_fold(make)


ROUND4 = [
    ("TWO-AXIS: SNR -> what (whole/Voice-Swap/none), action -> when (Step 2/3, else 4)", "candidate 5", two_axis()),
    ("TWO-AXIS, early exit at Step 3 only", "candidate 5 variant", two_axis(early_steps=(3,))),
    ("TWO-AXIS, 4 dB band", "candidate 5 variant", two_axis(margin_db=4.0)),
    ("TWO-AXIS, with EXP-018's single-case guard", "candidate 5 variant", two_axis(case_guard=True)),
]


# --------------------------------------------------------------------------- round 5: Agreement-Routed Handover
# What crosses is decided by whether the teacher and the student agree on the voice part (each band's average over
# time), which is where a copied recording condition shows up; when, by the Step-3 action as in the Two-Axis Handover.
# The method measures the gap at Step 0 (T_0 against the student's draft S: 2 teacher passes + 1 draft pass); only
# EXP-020's Step-4 gap (voice_rms) is stored, so it stands in for the Step-0 gap here, while the passes charged are the
# Step-0 method's: whole estimate +1 (the draft), Voice-Swap 14 (draft reused), student alone 4 + 2 (T_0).
# Cuts, out of fold: whole estimate up to the gap that keeps 95% of training clean prompts; Voice-Swap up to the largest
# training clean gap; the student alone beyond it. Fixed before this round was run.
VOICE = {k: float(S20[k]["voice_rms"]) for k in KEYS}


def agreement_routed(early_steps=(3,), middle=True):
    route, extra = {}, {}
    for f in range(FOLDS):
        train = [k for k in KEYS if FOLD[SPK[k]] != f]
        gaps = sorted(VOICE[k] for k in train if k[1] == "clean")
        lo = gaps[min(len(gaps) - 1, int(0.95 * len(gaps)))]
        hi = gaps[-1] if middle else lo
        branch = lambda k: "whole" if VOICE[k] <= lo else ("vs" if VOICE[k] <= hi else "student")
        whole = fit_stage_exits([k for k in train if branch(k) == "whole"], early_steps, "W4")
        for k in KEYS:
            if FOLD[SPK[k]] != f:
                continue
            b = branch(k)
            if b == "student":
                route[k], extra[k] = "student", 2
            elif b == "vs":
                route[k], extra[k] = "STT_4", 0
            else:
                route[k], extra[k] = whole(k), 1
    return route, extra


ROUND5 = [
    ("AGREEMENT-ROUTED: voice gap -> what, action -> when (Step 3, else 4)", "candidate 6", agreement_routed()),
    ("AGREEMENT-ROUTED, no Voice-Swap band", "candidate 6 variant", agreement_routed(middle=False)),
    ("AGREEMENT-ROUTED, early exit at Steps 2 or 3", "candidate 6 variant", agreement_routed(early_steps=(2, 3))),
]


# --------------------------------------------------------------------------- round 6: Variety-Value Routing
# Theory: the student is distilled by distribution matching, which is mode-seeking, so it loses pitch variety only
# where the teacher's output varies; the teacher's decision after Step 1, d16_1 = ||T_1 - T_0|| / ||T_1||, estimates
# that for this input (Spearman +0.42 with the stored variety gap on the 20 clean diversity prompts, checked first).
# Rule, on top of the Agreement-Routed Handover: a large Step-0 voice gap -> the student alone (6 passes); otherwise
# the teacher runs Step 1, and if d16_1 < lam_v it stops and the student finishes from its draft (8 passes); otherwise
# ARH's handover at Step 4 (14). lam_v, out of fold: the largest threshold, scanned upward, whose training cases keep
# the axis rule on means and whose training clean diversity prompts keep stored CVf0 within -0.01 of the reference.
def variety_value(margin_cvf0=0.01):
    route, extra = {}, {}
    for f in range(FOLDS):
        train = [k for k in KEYS if FOLD[SPK[k]] != f]
        gaps = sorted(VOICE[k] for k in train if k[1] == "clean")
        lo, hi = gaps[min(len(gaps) - 1, int(0.95 * len(gaps)))], gaps[-1]
        arh = lambda k: "student" if VOICE[k] > hi else ("STT_4" if VOICE[k] > lo else "W4")

        def make(lam):
            def r(k):
                a = arh(k)
                if a == "student":
                    return "student", 2
                if DRIFT[k][1] < lam:
                    return "student", 4             # T_0 + draft + teacher Step 1, then the student finishes
                return a, (1 if a == "W4" else 0)
            return r

        def ok(lam):
            r = make(lam)
            if not axis_rule_ok(lambda k: r(k)[0], train):
                return False
            diffs = []
            for utt, vals in CV.items():
                key = (utt, "clean")
                if key not in SPK or FOLD[SPK[key]] == f or "W8" not in vals:
                    continue
                o = r(key)[0]
                if o not in vals:
                    return False
                diffs.append(vals[o] - vals["W8"])
            return bool(diffs) and sum(diffs) / len(diffs) >= -margin_cvf0

        cands = sorted({round(DRIFT[k][1], 5) for k in train if arh(k) != "student"})
        lam = 0.0
        for v in cands:
            nxt = v + 1e-9                          # route every case with d16_1 <= v to the student
            if not ok(nxt):
                break
            lam = nxt
        r = make(lam)
        for k in KEYS:
            if FOLD[SPK[k]] == f:
                route[k], extra[k] = r(k)
    return route, extra


ROUND6 = [
    ("VARIETY-VALUE: ARH + stop after Step 1 when the teacher's decision is small", "candidate 7", variety_value()),
]


# --------------------------------------------------------------------------- round 7: Commitment-Timed ARH
# Theory: a trajectory commits to its sample around the peak of its rate of change (speciation); every stored drift
# profile peaks at Step 2, and a fast collapse after it (r3 = d_3 / d_2 small) means the commitment is done. Drift
# misled on noisy prompts (EXP-012); ARH removes those at Step 0, so on its agreement branch the commitment is
# readable: r3 <= lam_r -> hand over at Step 3, else the whole estimate at Step 4 (ARH). Step-3 payload: the slow part
# (restart-survival theory: the restart keeps 7.6% of the estimate, so mainly its slow part reaches the student; its
# pitch variety is stored), or the whole estimate. The draft already exists, so the slow part at Step 3 costs 12.
# lam_r, out of fold: the most permissive threshold (scanned upward) that keeps the axis rule on the branch's cases.
def commitment_timed(step3_out="STS_3", clean_only=False):
    route, extra = {}, {}
    r3 = lambda k: DRIFT[k][3] / max(DRIFT[k][2], 1e-9)
    for f in range(FOLDS):
        train = [k for k in KEYS if FOLD[SPK[k]] != f]
        gaps = sorted(VOICE[k] for k in train if k[1] == "clean")
        lo, hi = gaps[min(len(gaps) - 1, int(0.95 * len(gaps)))], gaps[-1]
        arh = lambda k: "student" if VOICE[k] > hi else ("STT_4" if VOICE[k] > lo else "W4")
        agree = [k for k in train if arh(k) == "W4" and (k[1] == "clean" or not clean_only)]

        def route_with(lam):
            return lambda k: (step3_out if (lam is not None and r3(k) <= lam) else "W4")

        lam = None
        for v in sorted({round(r3(k), 5) for k in agree}):
            if not axis_rule_ok(route_with(v), agree):
                break
            lam = v
        whole = route_with(lam)
        for k in KEYS:
            if FOLD[SPK[k]] != f:
                continue
            a = arh(k)
            if a == "student":
                route[k], extra[k] = "student", 2
            elif a == "STT_4":
                route[k], extra[k] = "STT_4", 0
            else:
                o = whole(k)
                route[k], extra[k] = o, (0 if o.startswith("STS_") else 1)
    return route, extra


ROUND7 = [
    ("COMMITMENT-TIMED ARH: committed by Step 3 -> slow part 3, else whole 4", "candidate 8", commitment_timed("STS_3")),
    ("COMMITMENT-TIMED ARH, whole estimate at Step 3", "candidate 8 variant", commitment_timed("W3")),
    # added after the first run: the fit above stops at the first failing threshold, and the few degraded prompts on
    # the agreement branch collapse fastest, so they blocked every exit; this variant fits on the branch's clean cases
    ("COMMITMENT-TIMED ARH, fitted on clean cases: slow part 3 / whole 4", "candidate 8 variant (post hoc)", commitment_timed("STS_3", True)),
    ("COMMITMENT-TIMED ARH, fitted on clean cases: whole 3 / whole 4", "candidate 8 variant (post hoc)", commitment_timed("W3", True)),
]


def evaluate(route, extra=None):
    """route: {case: output}; extra: optional {case: passes spent on checks before routing}."""
    clean = lambda k: k[1] == "clean"
    deg = lambda k: k[1] != "clean"
    spent = {k: (21 if route[k] == REF else cost(route[k])) + (extra or {}).get(k, 0) for k in KEYS}
    res = {"passes": sum(spent.values()) / len(KEYS),
           "passes_clean": sum(spent[k] for k in KEYS if clean(k)) / 60,
           "passes_deg": sum(spent[k] for k in KEYS if deg(k)) / 300}
    for g, sel in (("clean", clean), ("deg", deg)):
        for x in ("wer", "similarity", "utmos"):
            res[f"{g}_{x}"] = boot(route, sel, x)
        res[f"{g}_fails"] = sum(fails(k, route[k]) for k in KEYS if sel(k))
    res["cvf0"], res["cvf0_n"] = pitch_variety(route)
    res["by_cond"] = {c: {x: sum(delta(k, route[k], x) for k in KEYS if k[1] == c) / 60 for x in ("wer", "similarity", "utmos")}
                      for c in CONDS}
    res["share_student"] = {c: sum(route[k] == "student" for k in KEYS if k[1] == c) for c in CONDS}
    checks = {
        "clean quality": res["clean_wer"][2] <= RULES["clean_wer"] and res["clean_similarity"][1] >= -RULES["sim"]
                         and res["clean_utmos"][1] >= -RULES["utmos"],
        "clean failures": res["clean_fails"] <= RULES["clean_fails"],
        "degraded quality": res["deg_wer"][2] <= RULES["deg_wer"] and res["deg_similarity"][1] >= -RULES["sim"]
                            and res["deg_utmos"][1] >= -RULES["utmos"],
        "30% saving": res["passes"] <= RULES["max_passes"],
    }
    if res["cvf0"] is not None:
        checks["pitch variety"] = res["cvf0"] >= -RULES["cvf0"]
    res["checks"] = checks
    res["verdict"] = ("PASS" if all(checks.values()) and res["cvf0"] is not None
                      else "PASS, pitch unmeasured" if all(checks.values()) else "fail")
    return res


def fmt(t):
    return f"{t[0]:+.3f} [{t[1]:+.3f},{t[2]:+.3f}]"


def main():
    rows = [(name, kind, evaluate(route)) for name, kind, route in SOLUTIONS]
    for name, kind, sol in ROUND2:
        route, extra = sol if isinstance(sol, tuple) else (sol, None)
        rows.append((name + "  [round 2]", kind, evaluate(route, extra)))
    for name, kind, (route, extra) in ROUND3:
        rows.append((name + "  [round 3]", kind, evaluate(route, extra)))
    for name, kind, route in ROUND4:
        rows.append((name + "  [round 4]", kind, evaluate(route)))
    for name, kind, (route, extra) in ROUND5:
        rows.append((name + "  [round 5]", kind, evaluate(route, extra)))
    for name, kind, (route, extra) in ROUND6:
        rows.append((name + "  [round 6]", kind, evaluate(route, extra)))
    for name, kind, (route, extra) in ROUND7:
        rows.append((name + "  [round 7]", kind, evaluate(route, extra)))
    rows.append(("HINDSIGHT best case per input (not a solution)", "upper bound", evaluate(hindsight())))
    print("EXPLORATORY: 60 development speakers that designed these methods; white noise only.\n")
    print(f"{'solution':72} {'passes':>6} {'cl/deg':>11}  {'clean WER':>22} {'clean SIM':>22} {'cl.fail':>7}  "
          f"{'degraded WER':>22} {'degraded SIM':>22} {'degraded UTMOS':>22}  {'CVf0':>7}  verdict")
    for name, kind, r in sorted(rows, key=lambda t: t[2]["passes"]):
        cv = "  n/a  " if r["cvf0"] is None else f"{r['cvf0']:+.4f}"
        print(f"{name:72} {r['passes']:6.2f} {r['passes_clean']:5.1f}/{r['passes_deg']:<5.1f}  {fmt(r['clean_wer']):>22} {fmt(r['clean_similarity']):>22} "
              f"{r['clean_fails']:4d}/60  {fmt(r['deg_wer']):>22} {fmt(r['deg_similarity']):>22} "
              f"{fmt(r['deg_utmos']):>22}  {cv}  {r['verdict']}"
              + ("" if r["verdict"] != "fail" else "  (" + ", ".join(c for c, ok in r["checks"].items() if not ok) + ")"))
    print("\nPer prompt condition, capped-WER / similarity / UTMOS minus the reference, for solutions that pass:")
    for name, kind, r in rows:
        if r["verdict"] == "fail":
            continue
        print(f"  {name} ({kind})")
        for c in CONDS:
            b = r["by_cond"][c]
            print(f"    {c:8} WER {b['wer']:+.4f}  SIM {b['similarity']:+.4f}  UTMOS {b['utmos']:+.3f}"
                  f"  sent to the student {r['share_student'][c]}/60")
    return rows


if __name__ == "__main__":
    main()
