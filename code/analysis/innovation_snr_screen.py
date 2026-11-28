"""Offline screen of an innovation SNR for the handover gate (no synthesis; stored T4 scores only).

The flaw it targets (stored EXP-018, EXP-030, EXP-031 data). Chapter 3's prompt SNR compares the loudest half of the
frames with the quietest 10% (Definition 3.3). It reads white noise at 25 dB as 27 dB (gate open) although the teacher
then loses 1.26 UTMOS against the student alone, and it reads cafeteria babble at 15 dB and heavy reverberation as
18-19 dB (gate mostly closed) although the teacher is no worse than the student there. The SNR measures how loud the
background is, but the teacher's harm depends on what kind of background it is.

The changed term. A mode-seeking distilled student can drop the part of a prompt's background it cannot predict, and
keeps the structured part, as the teacher does. The power of the unpredictable part of a stationary background is, by
the Kolmogorov-Szego formula, the geometric mean of its power spectrum (the one-step prediction-error variance). So the
noise term of the SNR changes from the broadband energy of the quietest frames to the geometric mean, over frequency,
of the per-band background floor:

    SNR      = 10 log10( P_speech / mean_t-quietest-10% E(t) )                      (Definition 3.3)
    SNR_inn  = 10 log10( P_speech / exp( mean_f log N(f) ) )                         (this screen)

with N(f) the 10th percentile over frames of the power in frequency bin f (a per-band floor, as in minimum-statistics
noise estimation), and P_speech the mean band-limited power of the loudest half of the frames. For a white background
the two agree; for a speech-shaped or decaying background (babble, reverberation, band-limiting) SNR_inn is higher.
An ablation keeps the per-band floor but averages it arithmetically (SNR_floor), to separate the two changes.

Cases: EXP-031 (development speakers 1-30 x clean, gauss25, noisy15, reverb03, reverb09, babble15) and EXP-030
(development speakers 61-90 x clean, gauss25, noisy15, telephone), each case's degraded prompt rebuilt with the
experiments' own degradation code and seed. Harm = UTMOS(student alone) - UTMOS(W8) on the same case, from the stored
T4 scores. A case is harmful when harm > 0.3.

Pass rules, fixed before the run:
1. Validation: Definition 3.3 recomputed on the rebuilt prompts equals the stored prompt_snr (largest difference
   <= 0.01 dB), so the prompts are the ones the T4 runs used.
2. Detection: AUC(SNR_inn) >= AUC(SNR) + 0.05 for harmful cases (a lower statistic means harmful).
3. Routing: sending to the student the same number of cases as the 20 dB gate does, the lowest by each statistic,
   SNR_inn's mean UTMOS (student for routed cases, W8 otherwise) is >= SNR's + 0.05.

Usage: .venv/Scripts/python -I code/analysis/innovation_snr_screen.py [--demand_dir research/data/demand]
"""
import argparse
import csv
import hashlib
import json
import statistics as st
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for p in ("code/colab", "code/scheduling/prototype"):
    sys.path.insert(0, str(ROOT / p))

EXPS = ROOT / "research/experiments"
SOURCES = {"EXP-031": EXPS / "EXP-031-snr-toh-method-comparison/results/quality.csv",
           "EXP-030": EXPS / "EXP-030-timing-only-handover/results/quality.csv"}
DEMAND_SHA = "854dd0e5ef334a66abdd02d6d2b3e9d01b685df033d0acd1cd06e471c062550b"
OUT = EXPS / "EXP-033-cheaper-teacher-path/offline/innovation_snr_screen.json"
HARM = 0.3
GATE = 20.0
F_LO = 60.0
PASS_VALID, PASS_AUC, PASS_UTMOS = 0.01, 0.05, 0.05


def frames_of(x, sr):
    fl, hop = round(0.025 * sr), round(0.010 * sr)
    n = 1 + (len(x) - fl) // hop
    return np.lib.stride_tricks.as_strided(x, shape=(n, fl), strides=(x.strides[0] * hop, x.strides[0])), fl


def snr_variants(x, sr, eps=1e-12):
    """(SNR_floor, SNR_inn, floor flatness) on the band 60 Hz - sr/2 (same frames as Definition 3.3)."""
    x = np.ascontiguousarray(np.asarray(x, dtype=np.float64).squeeze())
    fr, fl = frames_of(x, sr)
    spec = np.abs(np.fft.rfft(fr * np.hanning(fl), axis=1)) ** 2
    freqs = np.fft.rfftfreq(fl, 1.0 / sr)
    spec = spec[:, freqs >= F_LO]
    energy = spec.mean(axis=1)
    loud = energy >= np.median(energy)
    p_speech = float(energy[loud].mean())
    floor = np.percentile(spec, 10, axis=0)
    arith = float(floor.mean())
    geo = float(np.exp(np.mean(np.log(floor + eps))))
    return (10 * np.log10((p_speech + eps) / (arith + eps)), 10 * np.log10((p_speech + eps) / (geo + eps)),
            geo / (arith + eps))


def auc(pos, neg):
    """P(statistic of a harmful case < statistic of a harmless case): lower means harmful."""
    pairs = [(a < b) + 0.5 * (a == b) for a in pos for b in neg]
    return sum(pairs) / len(pairs)


def route(cases, key, k):
    """Send the k cases with the lowest statistic to the student; mean UTMOS of the routed outputs."""
    order = sorted(range(len(cases)), key=lambda i: cases[i][key])
    to_student = set(order[:k])
    return st.mean(c["utmos_STU"] if i in to_student else c["utmos_W8"] for i, c in enumerate(cases)), to_student


def load_babble(demand_dir):
    import soundfile as sf
    from math import gcd
    from scipy.signal import resample_poly
    path = Path(demand_dir) / "PCAFETER" / "ch01.wav"
    if not path.exists():
        return None, None
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    if sha != DEMAND_SHA:
        raise SystemExit(f"DEMAND ch01.wav hash {sha[:12]} differs from the experiments' {DEMAND_SHA[:12]}")
    track, tsr = sf.read(str(path), dtype="float32")
    g = gcd(24000, tsr)
    return resample_poly(track, 24000 // g, tsr // g).astype(np.float32), sha


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demand_dir", default=str(ROOT / "research/data/demand"))
    a = ap.parse_args()
    import engine as en
    import run_followup_tests as rf
    import snr_drift as sd
    import soundfile as sf
    babble, demand_sha = load_babble(a.demand_dir)
    items = {it["utt_id"]: it for it in rf.split_items("development")}
    cases, skipped = [], []
    for exp, path in SOURCES.items():
        for r in csv.DictReader(open(path, encoding="utf-8")):
            it = items[r["utt_id"]]
            cond = r["condition"]
            if rf.CONDITIONS[cond][0] == "babble" and babble is None:
                skipped.append((exp, r["utt_id"], cond))
                continue
            audio, sr = sf.read(str(en.DATA_DIR / it["prompt_wav_rel"]), dtype="float32", always_2d=False)
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            prompt = rf.degrade(np.asarray(audio, dtype=np.float32), sr, cond, it["input_seed"], babble_track=babble)
            snr = sd.estimate_prompt_snr(prompt, sample_rate=sr)
            s_floor, s_inn, flat = snr_variants(prompt, sr)
            cases.append({"exp": exp, "utt_id": r["utt_id"], "condition": cond, "snr": snr, "snr_stored": float(r["prompt_snr"]),
                          "snr_floor": float(s_floor), "snr_inn": float(s_inn), "floor_flatness": float(flat),
                          "utmos_STU": float(r["utmos_STU"]), "utmos_W8": float(r["utmos_W8"]),
                          "harm": float(r["utmos_STU"]) - float(r["utmos_W8"])})
    valid = max(abs(c["snr"] - c["snr_stored"]) for c in cases)
    pos = [c for c in cases if c["harm"] > HARM]
    neg = [c for c in cases if c["harm"] <= HARM]
    aucs = {k: auc([c[k] for c in pos], [c[k] for c in neg]) for k in ("snr", "snr_floor", "snr_inn")}
    k_gate = sum(c["snr"] < GATE for c in cases)
    routed = {k: route(cases, k, k_gate)[0] for k in ("snr", "snr_floor", "snr_inn")}
    oracle = st.mean(max(c["utmos_STU"], c["utmos_W8"]) for c in cases)
    curves = {}
    for key in ("snr", "snr_inn"):
        curves[key] = [{"k_to_student": k, "mean_utmos": route(cases, key, k)[0]} for k in range(0, len(cases) + 1, 5)]
    _, to_inn = route(cases, "snr_inn", k_gate)
    _, to_snr = route(cases, "snr", k_gate)
    per_cond = {}
    for cond in sorted({c["condition"] for c in cases}):
        cs = [i for i, c in enumerate(cases) if c["condition"] == cond]
        per_cond[cond] = {"n": len(cs), "harm_mean": st.mean(cases[i]["harm"] for i in cs),
                          "snr_median": st.median(cases[i]["snr"] for i in cs),
                          "snr_inn_median": st.median(cases[i]["snr_inn"] for i in cs),
                          "flatness_median": st.median(cases[i]["floor_flatness"] for i in cs),
                          "to_student_by_snr_gate": sum(cases[i]["snr"] < GATE for i in cs),
                          "to_student_by_snr_inn_equal_k": sum(i in to_inn for i in cs),
                          "to_student_by_snr_equal_k": sum(i in to_snr for i in cs)}
    summary = {"cases": len(cases), "skipped": len(skipped), "demand_sha": demand_sha, "harmful": len(pos),
               "validation_max_abs_snr_diff": valid, "pass_validation": bool(valid <= PASS_VALID),
               "auc": aucs, "pass_detection": bool(aucs["snr_inn"] >= aucs["snr"] + PASS_AUC),
               "k_routed_like_the_20db_gate": k_gate, "mean_utmos_routed": routed, "mean_utmos_oracle": oracle,
               "mean_utmos_all_W8": st.mean(c["utmos_W8"] for c in cases),
               "mean_utmos_all_STU": st.mean(c["utmos_STU"] for c in cases),
               "pass_routing": bool(routed["snr_inn"] >= routed["snr"] + PASS_UTMOS),
               "per_condition": per_cond}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"kind": "offline screen (no synthesis; stored T4 scores)", "rules": {
        "harm": HARM, "gate": GATE, "valid": PASS_VALID, "auc": PASS_AUC, "utmos": PASS_UTMOS},
        "summary": summary, "curves": curves, "cases": cases, "skipped": skipped}, indent=2) + "\n",
        encoding="utf-8", newline="\n")
    print(json.dumps(summary, indent=2))
    print(f"[OK] wrote {OUT}")


if __name__ == "__main__":
    main()
