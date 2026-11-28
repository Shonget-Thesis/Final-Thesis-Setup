"""Local screen that tries to break the innovation SNR (screening only, on this computer's CPU; not evidence).

The innovation SNR (innovation_snr_screen.py) passed an offline screen in which the only harmful backgrounds were white
noise. Its noise term, the geometric mean of the per-band floor, is dominated by the emptiest frequency regions, so it
reads any background that leaves part of the spectrum empty as clean. Everyday stationary noise (fans, cars, air
conditioning) is low-pass. If the teacher copies such noise much more than the student does, the innovation SNR sends
those prompts to the teacher and the gate fails where it matters. This screen tests that on purpose.

Two readings of the earlier result make different predictions for coloured stationary noise of equal total power:
- innovation reading: the student drops only the unpredictable part of a background; a coloured (predictable)
  background is kept by both models, so the teacher's extra harm shrinks as the innovation SNR rises;
- additive-floor reading: the student drops any stationary additive floor and the teacher copies it, whatever its
  colour; the harm then follows the noise power, and the innovation SNR misses coloured noise.

Conditions (each added to the clean prompt at 25 dB total-power SNR, the level where white noise harms most; seeded by
the utterance's input seed):
  clean, white25 (the experiments' gauss25), pink25 (1/f), ssn25 (speech-shaped: flat to 500 Hz, then -9 dB/octave),
  lp25 (low-pass at 1 kHz, fan- or car-like), tone25 (three steady tones: 500, 1200, 2500 Hz).
Arms: the student alone (4 passes), W8 (21 passes) and CPH8 (12 passes). Harm = UTMOS(student) - UTMOS(arm).
Sentences: the six shortest clean EXP-033 quality cases (development speakers).

Pass rules, fixed before the run:
1. Validation: mean harm of W8 on white25 >= +0.5 and |mean harm on clean| <= 0.3 (the T4 direction is reproduced).
2. No miss: no coloured condition both reads clean on the innovation SNR (median >= 40 dB, the gate found offline) and
   has mean W8 harm >= half of white25's.
3. Ordering: across the five noisy conditions, Spearman(mean innovation SNR, mean W8 harm) <= -0.6.
If rule 2 fails, the innovation SNR as defined is rejected for coloured stationary noise, whatever the other rules say.

Usage: .venv/Scripts/python -I code/analysis/noise_color_screen.py
"""
import csv
import json
import statistics as st
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for p in ("code/colab", "code/scheduling/prototype", "code/analysis"):
    sys.path.insert(0, str(ROOT / p))
import innovation_snr_screen as ins  # noqa: E402

EXP033_QUAL = ROOT / "research/experiments/EXP-033-cheaper-teacher-path/results/quality.csv"
OUT = ROOT / "research/experiments/EXP-033-cheaper-teacher-path/offline/noise_color_screen.json"
SNR_DB = 25.0
N_CASES = 6
GATE_INN = 40.0
CONDS = ("clean", "white25", "pink25", "ssn25", "lp25", "tone25")
COLOURED = ("pink25", "ssn25", "lp25", "tone25")
TONES_HZ = (500.0, 1200.0, 2500.0)
PASS_WHITE, PASS_CLEAN, MISS_SHARE, PASS_RHO = 0.5, 0.3, 0.5, -0.6

SHAPES = {
    "pink25": lambda f: 1.0 / np.sqrt(np.maximum(f, 20.0)),
    "ssn25": lambda f: np.where(f < 60.0, 0.0, np.where(f <= 500.0, 1.0, (np.maximum(f, 1.0) / 500.0) ** -1.5)),
    "lp25": lambda f: 1.0 / np.sqrt(1.0 + (f / 1000.0) ** 8),
}


def shaped_noise(n, sr, seed, weight):
    rng = np.random.RandomState(seed % (2 ** 31 - 1))
    spec = np.fft.rfft(rng.randn(n)) * weight(np.fft.rfftfreq(n, 1.0 / sr))
    return np.fft.irfft(spec, n)


def tone_noise(n, sr, seed):
    rng = np.random.RandomState(seed % (2 ** 31 - 1))
    t = np.arange(n) / sr
    return sum(np.sin(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi)) for f in TONES_HZ)


def add_at(clean, noise, snr_db, eps=1e-12):
    """Scale `noise` to the total-power SNR the experiments' add_gaussian uses, and add it."""
    sig = float(np.mean(clean.astype(np.float64) ** 2))
    noise = noise * np.sqrt((sig / (10.0 ** (snr_db / 10.0))) / (float(np.mean(noise ** 2)) + eps))
    return np.clip(clean + noise, -1.0, 1.0).astype(np.float32)


def make_prompt(clean, sr, cond, seed, rf):
    if cond == "clean":
        return clean
    if cond == "white25":
        return rf.degrade(clean, sr, "gauss25", seed)
    if cond == "tone25":
        return add_at(clean, tone_noise(len(clean), sr, seed), SNR_DB)
    return add_at(clean, shaped_noise(len(clean), sr, seed, SHAPES[cond]), SNR_DB)


def band_snr_db(x, sr, eps=1e-12):
    """Mean over frequency bins (60 Hz - 8 kHz) of the band SNR in dB: loud-frame power over the 10th-percentile floor."""
    x = np.ascontiguousarray(np.asarray(x, dtype=np.float64).squeeze())
    fr, fl = ins.frames_of(x, sr)
    spec = np.abs(np.fft.rfft(fr * np.hanning(fl), axis=1)) ** 2
    freqs = np.fft.rfftfreq(fl, 1.0 / sr)
    spec = spec[:, (freqs >= ins.F_LO) & (freqs <= 8000.0)]
    energy = spec.mean(axis=1)
    loud = energy >= np.median(energy)
    return float(np.mean(10 * np.log10((spec[loud].mean(axis=0) + eps) / (np.percentile(spec, 10, axis=0) + eps))))


def spearman(a, b):
    def ranks(v):
        o = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        for p, i in enumerate(o):
            r[i] = p
        return r
    ra, rb = ranks(a), ranks(b)
    ma, mb = st.mean(ra), st.mean(rb)
    den = (sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb)) ** 0.5
    return sum((x - ma) * (y - mb) for x, y in zip(ra, rb)) / den if den else 0.0


def pick_cases(n=N_CASES):
    rows = [r for r in csv.DictReader(open(EXP033_QUAL, encoding="utf-8")) if r["condition"] == "clean"]
    return sorted(rows, key=lambda r: int(r["seq_len"]))[:n]


def main():
    import local_candidate_screen as ls
    import run_followup_tests as rf
    import snr_drift as sd
    import soundfile as sf
    r = ls.Runner()
    torch = r.torch
    utmos = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True).eval()
    items = {it["utt_id"]: it for it in r.en.development_examples()}
    t0 = time.time()
    cases = []
    with torch.inference_mode():
        for row in pick_cases():
            it = items[row["utt_id"]]
            clean, sr = sf.read(str(r.en.DATA_DIR / it["prompt_wav_rel"]), dtype="float32", always_2d=False)
            if clean.ndim > 1:
                clean = clean.mean(axis=1)
            for cond in CONDS:
                prompt = make_prompt(np.asarray(clean, dtype=np.float32), sr, cond, int(it["input_seed"]), rf)
                s_floor, s_inn, flat = ins.snr_variants(prompt, sr)
                ctx = r.e.prepare(prompt, sr, it["prompt_text"], it["gt_text"], int(it["input_seed"]))
                outs = {"STU": r.student(ctx, ctx["step_cond"], [0.0] + list(ls.TAUS)),
                        "W8": r.student(ctx, r.teacher(ctx, ls.sway(16), 8)[8], ls.TAUS),
                        "CPH8": r.student(ctx, r.teacher(ctx, ls.sway(8), 4)[4], ls.TAUS[1:])}
                score = {a: float(utmos(torch.from_numpy(r.e._decode(ctx, x).astype(np.float32)).unsqueeze(0), 24000))
                         for a, x in outs.items()}
                case = {"utt_id": row["utt_id"], "condition": cond, "snr": float(sd.estimate_prompt_snr(prompt, sample_rate=sr)),
                        "snr_floor": float(s_floor), "snr_inn": float(s_inn), "flatness": float(flat),
                        "band_snr_db": band_snr_db(prompt, sr), **{f"utmos_{a}": v for a, v in score.items()},
                        "harm_W8": score["STU"] - score["W8"], "harm_CPH8": score["STU"] - score["CPH8"]}
                cases.append(case)
                print(f"[{len(cases)}/{N_CASES * len(CONDS)}] {row['utt_id'][:22]} {cond:8s} SNR {case['snr']:5.1f} "
                      f"inn {case['snr_inn']:5.1f} flat {case['flatness']:.3f} | UTMOS STU {score['STU']:.2f} "
                      f"W8 {score['W8']:.2f} CPH8 {score['CPH8']:.2f} | harm {case['harm_W8']:+.2f} ({time.time() - t0:.0f}s)",
                      flush=True)
    per = {}
    for cond in CONDS:
        cs = [c for c in cases if c["condition"] == cond]
        per[cond] = {"n": len(cs), **{k: st.mean(c[k] for c in cs) for k in (
            "snr", "snr_floor", "snr_inn", "flatness", "band_snr_db", "utmos_STU", "utmos_W8", "utmos_CPH8",
            "harm_W8", "harm_CPH8")}, "snr_inn_median": st.median(c["snr_inn"] for c in cs),
            "harm_W8_positive": sum(c["harm_W8"] > 0 for c in cs)}
    white = per["white25"]["harm_W8"]
    misses = [c for c in COLOURED if per[c]["snr_inn_median"] >= GATE_INN and per[c]["harm_W8"] >= MISS_SHARE * white]
    noisy = [c for c in CONDS if c != "clean"]
    rho = {k: spearman([per[c][k] for c in noisy], [per[c]["harm_W8"] for c in noisy])
           for k in ("snr", "snr_floor", "snr_inn", "band_snr_db")}
    summary = {"pass_validation": bool(white >= PASS_WHITE and abs(per["clean"]["harm_W8"]) <= PASS_CLEAN),
               "white25_harm_W8": white, "clean_harm_W8": per["clean"]["harm_W8"],
               "misses": misses, "pass_no_miss": not misses,
               "spearman_statistic_vs_harm_over_noisy_conditions": rho, "pass_ordering": bool(rho["snr_inn"] <= PASS_RHO),
               "per_condition": per}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"kind": "local CPU screen (screening only, not evidence)",
                               "rules": {"white": PASS_WHITE, "clean": PASS_CLEAN, "miss_share": MISS_SHARE,
                                         "gate_inn": GATE_INN, "rho": PASS_RHO},
                               "summary": summary, "cases": cases, "seconds": round(time.time() - t0)}, indent=2) + "\n",
                   encoding="utf-8", newline="\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "per_condition"}, indent=2))
    for cond in CONDS:
        e = per[cond]
        print(f"{cond:8s} SNR {e['snr']:5.1f} | inn {e['snr_inn']:5.1f} | band {e['band_snr_db']:5.1f} | flat {e['flatness']:.3f} | "
              f"UTMOS STU {e['utmos_STU']:.2f} W8 {e['utmos_W8']:.2f} CPH8 {e['utmos_CPH8']:.2f} | harm W8 {e['harm_W8']:+.2f} "
              f"({e['harm_W8_positive']}/{e['n']}) CPH8 {e['harm_CPH8']:+.2f}")
    print(f"[OK] wrote {OUT}")


if __name__ == "__main__":
    main()
