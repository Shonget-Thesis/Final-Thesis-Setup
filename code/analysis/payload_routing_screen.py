"""EXPLORATORY offline screen, run 2026-11-13 after the requester asked whether SNR routing can be made novel (development data: 60 inspected speakers x 6 conditions, white noise only).
Question: does routing WHAT CROSSES (payload) at a fixed early step beat routing WHEN (step) or teacher-vs-student?
SNR routing is out of fold: 5-fold speaker CV, lambda = 95th percentile of the training folds' clean prompt SNR.
Costs: whole estimate at Step k (g16_k) 2k+5; slow part STS_k and Voice-Swap STT_k 2k+6 (one draft pass); student 4;
reference g16_8 21. Failures: Definition 3.1 against g16_8. Bootstrap over speakers, B=2000, seed 2026, capped WER."""
import csv, json, random, sys
from collections import defaultdict
from pathlib import Path

R = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[2]
E = R / "research/experiments"
rd = lambda p: {(r["utt_id"], r["condition"]): r for r in csv.DictReader(open(p, newline="", encoding="utf-8"))}
g17 = rd(E / "EXP-017-grid32-prelim/results/grid32.csv")
g18 = rd(E / "EXP-018-swsr-handover-prelim/results/steps1to8.csv")
g20 = rd(E / "EXP-020-selective-handover-prelim/results/selective.csv")
g22 = rd(E / "EXP-022-slow-part-handover-prelim/results/slowpart.csv")
th = json.loads((R / "code/scheduling/failure_config.json").read_text(encoding="utf-8"))
keys = sorted(k for k in g18 if k in g17 and k in g20 and k in g22)
assert len(keys) == 360, len(keys)


def m(k, o, x):
    if o == "student": return float(g17[k][f"{x}_student4"])
    if o.startswith("W"): return float(g18[k][f"{x}_g16_{o[1:]}"])        # whole estimate at Step k
    if o.startswith("STS_"): return float(g22[k][f"{x}_{o}"])             # slow part (EXP-022)
    if o.startswith("STT_"): return float(g20[k][f"{x}_{o}"])             # Voice-Swap (EXP-020)
    raise KeyError(o)


def cost(o):
    if o == "student": return 4
    if o.startswith("W"): k = int(o[1:]); return 2 * k + 5
    k = int(o.split("_")[1]); return 2 * k + 6


cap = lambda x: min(1.0, x)
REF = "W8"


def fail(k, o):
    if o == REF: return False
    return (m(k, o, "wer") > max(th["absolute_wer_ceiling"], m(k, REF, "wer") + th["relative_wer_margin"])
            or m(k, o, "similarity") < m(k, REF, "similarity") - th["similarity_margin"])


snr = {k: float(g20[k]["prompt_snr"]) for k in keys}
spk = {k: g20[k]["speaker_id"] for k in keys}
spks = sorted(set(spk.values())); random.Random(2026).shuffle(spks); fold = {sp: i % 5 for i, sp in enumerate(spks)}
cln = [k for k in keys if k[1] == "clean"]
clean_like = {}
for f in range(5):
    tr = sorted((snr[k] for k in cln if fold[spk[k]] != f), reverse=True)
    lam = tr[min(len(tr) - 1, int(0.95 * len(tr)))]
    clean_like.update({k: snr[k] >= lam for k in keys if fold[spk[k]] == f})


def boot(pairs, x, B=2000, seed=2026):
    gr = defaultdict(list)
    for k, o in pairs:
        a, b = m(k, o, x), m(k, REF, x)
        gr[spk[k]].append((cap(a) - cap(b)) if x == "wer" else a - b)
    vals = [sum(z) / len(z) for z in gr.values()]; rng = random.Random(seed)
    bs = sorted(sum(rng.choice(vals) for _ in vals) / len(vals) for _ in range(B))
    return f"{sum(vals) / len(vals):+.4f} [{bs[int(.025 * B)]:+.4f}, {bs[int(.975 * B)]:+.4f}]"


def policy(on_clean, on_noisy):
    return [(k, on_clean if clean_like[k] else on_noisy) for k in keys]


rows = []
for k in (2, 3, 4):
    rows += [(f"route WHEN      Step {k} / ref", policy(f"W{k}", REF)),
             (f"teacher-or-not  Step {k} / student", policy(f"W{k}", "student")),
             (f"route WHAT      Step {k}: whole / slow part", policy(f"W{k}", f"STS_{k}"))]
    if k == 4:
        rows.append((f"route WHAT      Step {k}: whole / Voice-Swap", policy(f"W{k}", "STT_4")))
rows += [("fixed           slow part Step 2 for all", [(k, "STS_2") for k in keys]),
         ("fixed           student for all", [(k, "student") for k in keys])]

print(f"clean-like routing: clean {sum(clean_like[k] for k in cln)}/60 kept on the clean branch; "
      f"degraded {sum(clean_like[k] for k in keys if k[1] != 'clean')}/300 on the clean branch")
print("degraded prompts only (n=300): metric minus the reference (g16_8), mean [95% CI]; failures = Definition 3.1")
for name, P in rows:
    passes = sum(cost(o) if o != REF else 21 for _, o in P) / len(P)
    D = [(k, o) for k, o in P if k[1] != "clean"]; C = [(k, o) for k, o in P if k[1] == "clean"]
    print(f"{name:44} passes {passes:5.2f} | degraded WER {boot(D, 'wer')} SIM {boot(D, 'similarity')} "
          f"UTMOS {boot(D, 'utmos')} fails {sum(fail(k, o) for k, o in D):3d}/300 | clean fails {sum(fail(k, o) for k, o in C)}/60")
