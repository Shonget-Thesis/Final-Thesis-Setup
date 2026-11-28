"""Exploratory probe (design input, not a screen): does the teacher's excess temporal roughness over the student's draft,
per mel band, separate clean from noisy15 prompts? (CPU; no synthesis beyond the teacher's estimate and the draft.)

The control screen (cpqh_controls_screen.py) found the teacher's estimate at t = 0.2929 rougher than the student's
draft on noisy15 prompts (0.14 against 0.07, averaged over bands) and smoother on clean prompts (0.06 against 0.10).
This probe measures the per-band ratio rho_T,b / rho_S,b on the same six sentences, two seeds each, to decide whether a
per-band rule can tell copied background from prosody without a prompt-level gate.

Usage: .venv/Scripts/python -I code/analysis/roughness_band_probe.py
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for p in ("code/colab", "code/scheduling/prototype", "code/analysis"):
    sys.path.insert(0, str(ROOT / p))
import noise_color_screen as ncs  # noqa: E402

OUT = ROOT / "research/experiments/EXP-033-cheaper-teacher-path/offline/roughness_band_probe.json"
N_SENT, SEEDS, SEED_BASE = 6, (0, 1), 1000


def band_roughness(x):
    """Per band: mean squared frame-to-frame change over the band's variance (x: frames x bands)."""
    x = np.asarray(x, dtype=np.float64)
    d = np.diff(x, axis=0)
    return np.mean(d * d, axis=0) / np.maximum(np.var(x, axis=0), 1e-12)


def main():
    import local_candidate_screen as ls
    import run_followup_tests as rf
    import soundfile as sf
    r = ls.Runner()
    torch = r.torch
    items = {it["utt_id"]: it for it in r.en.development_examples()}
    ratios = {"clean": [], "noisy15": []}
    with torch.inference_mode():
        for row in ncs.pick_cases(N_SENT):
            it = items[row["utt_id"]]
            clean, sr = sf.read(str(r.en.DATA_DIR / it["prompt_wav_rel"]), dtype="float32", always_2d=False)
            clean = (clean.mean(axis=1) if clean.ndim > 1 else clean).astype(np.float32)
            for cond in ratios:
                prompt = rf.degrade(clean, sr, cond, int(it["input_seed"]))
                for s in SEEDS:
                    ctx = r.e.prepare(prompt, sr, it["prompt_text"], it["gt_text"], int(it["input_seed"]) + SEED_BASE + s)
                    plen = ctx["prompt_len"]
                    s0 = r.student(ctx, ctx["step_cond"], [0.0])[0, plen:, :].numpy()
                    t8 = r.teacher(ctx, ls.sway(8), 4)[4][0, plen:, :].numpy()
                    ratios[cond].append((band_roughness(t8) / np.maximum(band_roughness(s0), 1e-12)).tolist())
                    print(f"{row['utt_id'][:22]} {cond:7s} s{s} median ratio {np.median(ratios[cond][-1]):.2f}", flush=True)
    summary = {}
    for cond, rs in ratios.items():
        a = np.asarray(rs)                                   # cases x bands
        summary[cond] = {"cases": int(a.shape[0]),
                         "median_ratio": float(np.median(a)),
                         "share_of_bands_ratio_above_1": float(np.mean(a > 1.0)),
                         "share_of_bands_ratio_above_1.5": float(np.mean(a > 1.5)),
                         "per_case_share_above_1": [float(np.mean(x > 1.0)) for x in a],
                         "band_median_ratio": np.median(a, axis=0).round(3).tolist()}
    OUT.write_text(json.dumps({"kind": "exploratory probe (design input; not a screen)", "summary": summary,
                               "ratios": ratios}, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({c: {k: v for k, v in s.items() if k != "band_median_ratio"} for c, s in summary.items()}, indent=2))
    print(f"[OK] wrote {OUT}")


if __name__ == "__main__":
    main()
