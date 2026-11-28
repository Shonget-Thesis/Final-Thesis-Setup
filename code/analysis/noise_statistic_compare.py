"""Which prompt statistic best predicts where the teacher hurts? (no synthesis; stored scores only)

After the noise-theory screen, the supported explanation is masking: the teacher's extra harm over the student comes
from copied noise in the frequency bands where speech is weak. Four statistics are compared, all from the same 25 ms
frames (10 ms hop), with S(f) the mean power of the loudest half of the frames and N(f) the 10th-percentile floor, per
frequency bin from 60 Hz to the Nyquist frequency:

  snr        Definition 3.3 (broadband loudest half over quietest 10%)
  inn        10 log10( mean_f S(f) / exp(mean_f log N(f)) )          the innovation SNR of the first screen
  band_db    mean_f 10 log10( S(f) / N(f) )                           log-average of the per-band SNR
  unmasked   -10 log10( mean_f N(f) / S(f) )                          mean per-band noise-to-speech ratio, in dB

`unmasked` is the one the masking explanation derives directly: noise is exposed in a band in proportion to N(f)/S(f),
and the mean over bands is dominated by the bands where speech is weakest.

Data:
- 270 stored T4 cases (EXP-031: speakers 1-30 x clean, gauss25, noisy15, reverb03, reverb09; EXP-030: speakers 61-90
  x clean, gauss25, noisy15, telephone); harm = UTMOS(student alone) - UTMOS(W8) from the T4 scores.
- 48 local CPU cases (six development sentences x clean, white25, pink25, ssn25, lp25, tone25, hp25, bp25); harm from
  the noise-colour and noise-theory screens (screening only).

Reported: AUC for harmful cases (harm > 0.3; lower statistic = harmful) on each set; Spearman over the eight local
condition means; and how many harmful and harmless cases each statistic's best single threshold misroutes on the
stored set. Exploratory: the choice of statistic for a confirmation run is made from these numbers.

Usage: .venv/Scripts/python -I code/analysis/noise_statistic_compare.py
"""
import csv
import json
import statistics as st
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for p in ("code/colab", "code/scheduling/prototype", "code/analysis"):
    sys.path.insert(0, str(ROOT / p))
import innovation_snr_screen as ins  # noqa: E402
import noise_color_screen as ncs  # noqa: E402
import noise_theory_screen as nts  # noqa: E402

OFF = ROOT / "research/experiments/EXP-033-cheaper-teacher-path/offline"
OUT = OFF / "noise_statistic_compare.json"
STATS = ("snr", "inn", "band_db", "unmasked")
HARM = 0.3


def statistics(x, sr, eps=1e-12):
    import snr_drift as sd
    x = np.ascontiguousarray(np.asarray(x, dtype=np.float64).squeeze())
    fr, fl = ins.frames_of(x, sr)
    spec = np.abs(np.fft.rfft(fr * np.hanning(fl), axis=1)) ** 2
    spec = spec[:, np.fft.rfftfreq(fl, 1.0 / sr) >= ins.F_LO]
    energy = spec.mean(axis=1)
    s = spec[energy >= np.median(energy)].mean(axis=0)
    n = np.percentile(spec, 10, axis=0)
    return {"snr": float(sd.estimate_prompt_snr(x.astype(np.float32), sample_rate=sr)),
            "inn": float(10 * np.log10((s.mean() + eps) / (np.exp(np.mean(np.log(n + eps))) + eps))),
            "band_db": float(np.mean(10 * np.log10((s + eps) / (n + eps)))),
            "unmasked": float(-10 * np.log10(np.mean((n + eps) / (s + eps))))}


def auc(cases, key):
    pos = [c[key] for c in cases if c["harm"] > HARM]
    neg = [c[key] for c in cases if c["harm"] <= HARM]
    return sum((a < b) + 0.5 * (a == b) for a in pos for b in neg) / (len(pos) * len(neg))


def best_threshold_errors(cases, key):
    """The single threshold with the fewest misroutes (harmful above it, harmless below it)."""
    vals = sorted({c[key] for c in cases})
    best = None
    for th in [vals[0] - 1] + [(a + b) / 2 for a, b in zip(vals, vals[1:])] + [vals[-1] + 1]:
        missed = sum(c["harm"] > HARM and c[key] >= th for c in cases)
        wasted = sum(c["harm"] <= HARM and c[key] < th for c in cases)
        if best is None or missed + wasted < best["missed"] + best["wasted"]:
            best = {"threshold": th, "missed": missed, "wasted": wasted}
    return best


def spearman(a, b):
    return ncs.spearman(a, b)


def main():
    import engine as en
    import run_followup_tests as rf
    import soundfile as sf
    items = {it["utt_id"]: it for it in rf.split_items("development")}

    def clean_prompt(utt):
        it = items[utt]
        audio, sr = sf.read(str(en.DATA_DIR / it["prompt_wav_rel"]), dtype="float32", always_2d=False)
        return (audio.mean(axis=1) if audio.ndim > 1 else audio).astype(np.float32), sr, it

    stored = []
    for exp, path in ins.SOURCES.items():
        for r in csv.DictReader(open(path, encoding="utf-8")):
            if rf.CONDITIONS[r["condition"]][0] == "babble":
                continue
            clean, sr, it = clean_prompt(r["utt_id"])
            prompt = rf.degrade(clean, sr, r["condition"], it["input_seed"])
            stored.append({"set": exp, "utt_id": r["utt_id"], "condition": r["condition"], **statistics(prompt, sr),
                           "harm": float(r["utmos_STU"]) - float(r["utmos_W8"])})
    assert max(abs(c["snr"] - float(r)) for c, r in zip(stored, [x["snr_stored"] for x in json.loads(
        (OFF / "innovation_snr_screen.json").read_text(encoding="utf-8"))["cases"]])) < 0.01

    local = []
    for src, maker in (("noise_color_screen.json", ncs.make_prompt), ("noise_theory_screen.json", nts.make_prompt)):
        for c in json.loads((OFF / src).read_text(encoding="utf-8"))["cases"]:
            if src == "noise_theory_screen.json" and c["condition"] == "white25":
                continue                                  # the same prompts and scores as in the colour screen
            clean, sr, it = clean_prompt(c["utt_id"])
            prompt = maker(clean, sr, c["condition"], int(it["input_seed"]), rf)
            local.append({"utt_id": c["utt_id"], "condition": c["condition"], **statistics(prompt, sr), "harm": c["harm_W8"]})

    conds = sorted({c["condition"] for c in local})
    means = {k: [st.mean(c[k] for c in local if c["condition"] == cond) for cond in conds] for k in STATS + ("harm",)}
    res = {"stored_cases": len(stored), "local_cases": len(local),
           "auc_stored": {k: auc(stored, k) for k in STATS},
           "auc_local": {k: auc(local, k) for k in STATS},
           "spearman_local_condition_means_vs_harm": {k: spearman(means[k], means["harm"]) for k in STATS},
           "best_threshold_stored": {k: best_threshold_errors(stored, k) for k in STATS},
           "best_threshold_local": {k: best_threshold_errors(local, k) for k in STATS},
           "local_condition_means": {cond: {k: means[k][i] for k in STATS + ("harm",)} for i, cond in enumerate(conds)},
           "stored_condition_means": {cond: {k: st.mean(c[k] for c in stored if c["condition"] == cond) for k in STATS + ("harm",)}
                                      for cond in sorted({c["condition"] for c in stored})}}
    OUT.write_text(json.dumps({**res, "stored": stored, "local": local}, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({k: v for k, v in res.items() if not k.endswith("condition_means")}, indent=2))
    for cond, e in res["local_condition_means"].items():
        print(f"  local {cond:8s} " + " ".join(f"{k} {e[k]:6.1f}" for k in STATS) + f" | harm {e['harm']:+.2f}")
    for cond, e in res["stored_condition_means"].items():
        print(f"  T4    {cond:9s} " + " ".join(f"{k} {e[k]:6.1f}" for k in STATS) + f" | harm {e['harm']:+.2f}")
    print(f"[OK] wrote {OUT}")


if __name__ == "__main__":
    main()
