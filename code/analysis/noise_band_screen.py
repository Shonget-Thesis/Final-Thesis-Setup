"""Local screen: narrow high-band noise, the case that separates 'spread' from 'masking' (screening only; CPU).

After the noise-colour, noise-theory and statistic-comparison screens, the statistic that best predicts the teacher's
extra harm is the innovation SNR, which equals the per-band floor SNR plus a penalty for how spread out (spectrally
flat) the background floor is. Two explanations remain:
- spread: the mode-seeking student drops spread-out, high-entropy background texture and keeps narrow or structured
  backgrounds, while the teacher copies both; harm follows the floor's level and spread;
- masking: copied noise hurts where speech is weak (high frequencies), whatever its spread.

Noise confined to a narrow high band (6-8 kHz) at 25 dB total-power SNR separates them. It is narrow (low spread), so
the innovation SNR reads it as clean and the spread explanation predicts little harm; it sits where speech is weak, so
the masking explanation predicts harm comparable to white noise, and then the gate would miss it.

Condition: nb68 (band 6-8 kHz). Reference: white25 per sentence, taken from noise_color_screen.json (same sentences,
same prompts and seeds, deterministic). Arms: the student alone and W8. Harm = UTMOS(student) - UTMOS(W8).

Predictions, fixed before the run:
- spread holds if mean harm(nb68) <= 0.33 x mean harm(white25);
- masking holds if mean harm(nb68) >= 0.67 x mean harm(white25) (and the gate misses: innovation SNR >= 40 dB);
- anything in between: neither on its own.

Usage: .venv/Scripts/python -I code/analysis/noise_band_screen.py
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

OFF = ROOT / "research/experiments/EXP-033-cheaper-teacher-path/offline"
OUT = OFF / "noise_band_screen.json"
SPREAD_MAX, MASKING_MIN = 0.33, 0.67


def band_shape(f):
    return 1.0 / np.sqrt(1.0 + (6000.0 / np.maximum(f, 1.0)) ** 16) / np.sqrt(1.0 + (f / 8000.0) ** 16)


def main():
    import local_candidate_screen as ls
    import snr_drift as sd
    import soundfile as sf
    white = {c["utt_id"]: c for c in json.loads((OFF / "noise_color_screen.json").read_text(encoding="utf-8"))["cases"]
             if c["condition"] == "white25"}
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
            clean = (clean.mean(axis=1) if clean.ndim > 1 else clean).astype(np.float32)
            prompt = ncs.add_at(clean, ncs.shaped_noise(len(clean), sr, int(it["input_seed"]), band_shape), ncs.SNR_DB)
            _, s_inn, flat = ins.snr_variants(prompt, sr)
            ctx = r.e.prepare(prompt, sr, it["prompt_text"], it["gt_text"], int(it["input_seed"]))
            outs = {"STU": r.student(ctx, ctx["step_cond"], [0.0] + list(ls.TAUS)),
                    "W8": r.student(ctx, r.teacher(ctx, ls.sway(16), 8)[8], ls.TAUS)}
            score = {a: float(utmos(torch.from_numpy(r.e._decode(ctx, x).astype(np.float32)).unsqueeze(0), 24000))
                     for a, x in outs.items()}
            case = {"utt_id": row["utt_id"], "condition": "nb68", "snr": float(sd.estimate_prompt_snr(prompt, sample_rate=sr)),
                    "snr_inn": float(s_inn), "flatness": float(flat), **{f"utmos_{a}": v for a, v in score.items()},
                    "harm_W8": score["STU"] - score["W8"], "white25_harm_W8": white[row["utt_id"]]["harm_W8"]}
            cases.append(case)
            print(f"[{len(cases)}/{ncs.N_CASES}] {row['utt_id'][:22]} nb68 SNR {case['snr']:5.1f} inn {s_inn:5.1f} | "
                  f"UTMOS STU {score['STU']:.2f} W8 {score['W8']:.2f} | harm {case['harm_W8']:+.2f} "
                  f"(white25 {case['white25_harm_W8']:+.2f}) ({time.time() - t0:.0f}s)", flush=True)
    h, w = st.mean(c["harm_W8"] for c in cases), st.mean(c["white25_harm_W8"] for c in cases)
    ratio = h / w if w > 0 else None
    verdict = "spread" if ratio is not None and ratio <= SPREAD_MAX else "masking" if ratio is not None and ratio >= MASKING_MIN else "neither"
    summary = {"nb68_harm": h, "white25_harm": w, "ratio": ratio, "supported_explanation": verdict,
               "nb68_inn_median": st.median(c["snr_inn"] for c in cases),
               "nb68_inn_at_or_above_40": sum(c["snr_inn"] >= 40.0 for c in cases)}
    OUT.write_text(json.dumps({"kind": "local CPU screen (screening only, not evidence)",
                               "rules": {"spread_max": SPREAD_MAX, "masking_min": MASKING_MIN},
                               "summary": summary, "cases": cases, "seconds": round(time.time() - t0)}, indent=2) + "\n",
                   encoding="utf-8", newline="\n")
    print(json.dumps(summary, indent=2))
    print(f"[OK] wrote {OUT}")


if __name__ == "__main__":
    main()
