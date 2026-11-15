"""Toy check of the handover invariance ladder on synthetic speech-like log-mel. These are not model outputs.

A teacher rendering and a student rendering of 'the same sentence' differ in timing and pitch movement. The teacher
copies white noise (additive in power, with a random texture). Payloads:
  W   the teacher's estimate as is (the released handover)
  C1  per-band mean swap (Voice-Swap, EXP-020)
  C2  per-band affine: z-score, then the student's per-band mean and sd
  C3  quantile handover: the student's per-band values, arranged in the teacher's per-band time order
  TOH the student's draft re-timed to the teacher (EXP-030)
  STU the student's draft
Measures:
  floor   per-band 10th-percentile level minus the clean teacher's, dB (0 = no level leak)
  pattern correlation of the low-band (harmonic) pattern with the clean teacher over speech frames (teacher pitch kept)
Theory note: docs/progress/log/2026-11-15-handover-invariance-theory.md. Usage: python code/analysis/handover_ladder_toy.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "colab"))
import run_timing_only_handover as toh  # noqa: E402

SR, NFFT, B = 24000, 1024, 100
FREQS = np.linspace(0, SR / 2, NFFT // 2 + 1)


def _mel(f):
    return 2595 * np.log10(1 + f / 700)


EDGES = np.interp(np.linspace(_mel(0), _mel(SR / 2), B + 2), _mel(FREQS), FREQS)
FB = np.zeros((B, len(FREQS)))
for b in range(B):
    lo, c, hi = EDGES[b], EDGES[b + 1], EDGES[b + 2]
    FB[b] = np.clip(np.minimum((FREQS - lo) / (c - lo + 1e-9), (hi - FREQS) / (hi - c + 1e-9)), 0, None)


def render(rng, n, bounds, f0_base, f0_dev):
    """Power mel of a harmonic source with moving formants; silence floor 1e-5."""
    P = np.full((n, len(FREQS)), 1e-5)
    t = np.arange(n)
    f0 = f0_base * (1 + f0_dev * np.sin(2 * np.pi * t / rng.uniform(60, 120) + rng.uniform(0, 6)))
    speech = np.zeros(n, bool)
    for a, z in bounds:
        speech[a:z] = True
        formants = np.array([rng.uniform(500, 800), rng.uniform(1100, 1900), rng.uniform(2300, 3000)])
        env = sum(np.exp(-0.5 * ((FREQS - fk) / 120) ** 2) for fk in formants) + 0.05
        for i in range(a, z):
            amp = np.sin(np.pi * (i - a) / max(z - a, 1)) ** 0.5
            for k in range(1, int(5000 / f0[i])):
                P[i] += amp * env * np.exp(-0.5 * ((FREQS - k * f0[i]) / 25) ** 2)
    return P @ FB.T, speech


def syllables(rng, n, k):
    cuts = np.sort(rng.choice(np.arange(20, n - 20), 2 * k, replace=False))
    return list(zip(cuts[::2], cuts[1::2]))


def quantile_handover(T, S):
    """Per band: the student's sorted values placed at the teacher's time ranks."""
    out = np.empty_like(S)
    for b in range(T.shape[1]):
        out[np.argsort(T[:, b], kind="stable"), b] = np.sort(S[:, b])
    return out


def main():
    db = 10 / np.log(10)
    for snr_db in (25.0, 15.0):
        res = {k: {"floor": [], "pattern": []} for k in ("W", "C1", "C2", "C3", "TOH", "STU")}
        for trial in range(12):
            rng = np.random.default_rng(trial)
            n = 300
            Pt, speech = render(rng, n, syllables(rng, n, 5), 140, 0.08)
            Ps, _ = render(rng, n, syllables(rng, n, 5), 140, 0.02)
            mean_noise = Pt.sum() / n / 10 ** (snr_db / 10) * FB.sum(1) / FB.sum()
            shape = np.maximum(FB.sum(1), 1)
            N = rng.gamma(shape=shape, scale=mean_noise / shape, size=(n, B))
            Tc, T, S = np.log(Pt), np.log(Pt + N), np.log(Ps)
            payloads = {"W": T, "C1": T - T.mean(0) + S.mean(0),
                        "C2": (T - T.mean(0)) / (T.std(0) + 1e-9) * S.std(0) + S.mean(0),
                        "C3": quantile_handover(T, S), "TOH": toh.warp_frames(S, toh.timing_map(S, T)[0]), "STU": S}
            low = slice(5, 45)
            for k, X in payloads.items():
                res[k]["floor"].append(db * float(np.mean(np.quantile(X, 0.1, 0) - np.quantile(Tc, 0.1, 0))))
                a, c = X[speech][:, low], Tc[speech][:, low]
                a, c = a - a.mean(1, keepdims=True), c - c.mean(1, keepdims=True)
                res[k]["pattern"].append(float(np.mean((a * c).sum(1) / np.sqrt((a * a).sum(1) * (c * c).sum(1)))))
        print(f"--- prompt SNR {snr_db:.0f} dB (toy, 12 trials)")
        for k, v in res.items():
            print(f"{k:4s} floor {np.mean(v['floor']):+6.1f} dB   teacher pattern {np.mean(v['pattern']):.3f}")


if __name__ == "__main__":
    main()
