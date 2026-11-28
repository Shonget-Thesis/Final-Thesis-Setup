"""Local screen that tells two explanations of the innovation SNR apart (screening only, CPU; not evidence).

The noise-colour screen showed that, at equal total power (25 dB), the teacher's extra harm over the student alone falls
in the order white > pink > speech-shaped > low-pass ~ tones, the order of the innovation SNR. Two explanations predict
that same order:

- Predictability (Kolmogorov-Szego) theory: the mode-seeking student drops the unpredictable part of a background and
  keeps the predictable part, as the teacher does. A background whose spectrum is empty over a whole band is perfectly
  predictable in theory, so high-pass noise should do LITTLE harm (like low-pass noise).
- Masking theory: copied noise hurts where speech is weak, i.e. at high frequencies; the innovation SNR works because
  its log-average over linearly spaced bins is dominated by those bins. High-pass noise puts all its power where speech
  is weak, so it should do AS MUCH harm as white noise, or more.

Conditions (added to the clean prompt at 25 dB total-power SNR, seeded by the utterance's input seed):
  white25 (the experiments' gauss25, the reference), hp25 (high-pass: empty below 2 kHz),
  bp25 (band-pass 1-4 kHz, the speech-formant region).
Arms: the student alone (4 passes) and W8 (21 passes). Harm = UTMOS(student) - UTMOS(W8).
Sentences: the same six shortest clean EXP-033 quality cases as the noise-colour screen.

Predictions, fixed before the run:
- predictability theory holds if mean harm(hp25) <= 0.33 x mean harm(white25);
- masking theory holds if mean harm(hp25) >= 0.67 x mean harm(white25);
- anything in between: neither explanation is supported on its own.
Gate check, reported either way: does the innovation SNR read hp25 below 40 dB (the gate found offline)?

Usage: .venv/Scripts/python -I code/analysis/noise_theory_screen.py
"""
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
import noise_color_screen as ncs  # noqa: E402

OUT = ROOT / "research/experiments/EXP-033-cheaper-teacher-path/offline/noise_theory_screen.json"
CONDS = ("white25", "hp25", "bp25")
SHAPES = {
    "hp25": lambda f: 1.0 / np.sqrt(1.0 + (2000.0 / np.maximum(f, 1.0)) ** 8),
    "bp25": lambda f: 1.0 / np.sqrt(1.0 + (1000.0 / np.maximum(f, 1.0)) ** 8) / np.sqrt(1.0 + (f / 4000.0) ** 8),
}
PREDICTABILITY_MAX, MASKING_MIN = 0.33, 0.67


def make_prompt(clean, sr, cond, seed, rf):
    if cond == "white25":
        return rf.degrade(clean, sr, "gauss25", seed)
    return ncs.add_at(clean, ncs.shaped_noise(len(clean), sr, seed, SHAPES[cond]), ncs.SNR_DB)


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
        for row in ncs.pick_cases():
            it = items[row["utt_id"]]
            clean, sr = sf.read(str(r.en.DATA_DIR / it["prompt_wav_rel"]), dtype="float32", always_2d=False)
            if clean.ndim > 1:
                clean = clean.mean(axis=1)
            for cond in CONDS:
                prompt = make_prompt(np.asarray(clean, dtype=np.float32), sr, cond, int(it["input_seed"]), rf)
                s_floor, s_inn, flat = ins.snr_variants(prompt, sr)
                ctx = r.e.prepare(prompt, sr, it["prompt_text"], it["gt_text"], int(it["input_seed"]))
                outs = {"STU": r.student(ctx, ctx["step_cond"], [0.0] + list(ls.TAUS)),
                        "W8": r.student(ctx, r.teacher(ctx, ls.sway(16), 8)[8], ls.TAUS)}
                score = {a: float(utmos(torch.from_numpy(r.e._decode(ctx, x).astype(np.float32)).unsqueeze(0), 24000))
                         for a, x in outs.items()}
                case = {"utt_id": row["utt_id"], "condition": cond, "snr": float(sd.estimate_prompt_snr(prompt, sample_rate=sr)),
                        "snr_inn": float(s_inn), "flatness": float(flat), "band_snr_db": ncs.band_snr_db(prompt, sr),
                        **{f"utmos_{a}": v for a, v in score.items()}, "harm_W8": score["STU"] - score["W8"]}
                cases.append(case)
                print(f"[{len(cases)}/{len(CONDS) * ncs.N_CASES}] {row['utt_id'][:22]} {cond:7s} SNR {case['snr']:5.1f} "
                      f"inn {case['snr_inn']:5.1f} | UTMOS STU {score['STU']:.2f} W8 {score['W8']:.2f} | harm "
                      f"{case['harm_W8']:+.2f} ({time.time() - t0:.0f}s)", flush=True)
    per = {c: {k: st.mean(x[k] for x in cases if x["condition"] == c)
               for k in ("snr", "snr_inn", "flatness", "band_snr_db", "utmos_STU", "utmos_W8", "harm_W8")} for c in CONDS}
    ratio = per["hp25"]["harm_W8"] / per["white25"]["harm_W8"] if per["white25"]["harm_W8"] > 0 else None
    verdict = ("predictability" if ratio is not None and ratio <= PREDICTABILITY_MAX else
               "masking" if ratio is not None and ratio >= MASKING_MIN else "neither")
    summary = {"hp25_over_white25_harm": ratio, "supported_explanation": verdict,
               "hp25_inn_below_40": sum(x["snr_inn"] < 40.0 for x in cases if x["condition"] == "hp25"),
               "bp25_inn_below_40": sum(x["snr_inn"] < 40.0 for x in cases if x["condition"] == "bp25"),
               "per_condition": per}
    OUT.write_text(json.dumps({"kind": "local CPU screen (screening only, not evidence)",
                               "rules": {"predictability_max": PREDICTABILITY_MAX, "masking_min": MASKING_MIN},
                               "summary": summary, "cases": cases, "seconds": round(time.time() - t0)}, indent=2) + "\n",
                   encoding="utf-8", newline="\n")
    print(json.dumps(summary, indent=2))
    print(f"[OK] wrote {OUT}")


if __name__ == "__main__":
    main()
