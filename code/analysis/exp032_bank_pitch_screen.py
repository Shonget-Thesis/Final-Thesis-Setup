"""Offline screen on EXP-032's tensor bank: when is the pitch decided, and what does each restart keep?

Post hoc and exploratory; not part of EXP-032's preregistration. No DMOSpeech 2 network runs. The bank's float16 mels
(generated frames, speakers 1-5, 12 seeds, clean and noisy15) are vocoded with Vocos on this computer's CPU and tracked
with the experiments' own pYIN settings (run_selective_handover_prelim.f0_track). The stored T4 f0 tracks of the outputs
are used as the reference; the CPU re-tracking of the same outputs is reported as a check of this pipeline.

Mels in the bank (per seed):
- `s0`: the student's one-step draft from pure noise;
- `T1`, `T2`, `T3`, `T4`, `T8`: the teacher's endpoint estimates on the 16-interval grid (t = 0.0048 ... 0.2929);
- `T14_32`: the teacher's estimate at the paper's Step 14 on the 32-interval grid (t = 0.2270);
- `out_STU`, `out_W4`, `out_W8`, `out_P25`: four arms' outputs.

Measures:
1. Variety already present in each estimate: CVf0 and aligned CVf0 across the 12 seeds (the experiments' functions),
   and the voiced share, per speaker.
2. Melody agreement within a seed: for two f0 tracks of the same seed, on frames voiced in both, the share of frames
   within 100 cents (one semitone) and the correlation of log f0. Pairs:
   - each estimate against `P25`'s final melody (when is the paper setting's melody decided?);
   - the handed-over estimate against the output of its own restart: `T4` -> `W4` and `T8` -> `W8` (restart 0.0761),
     `T14_32` -> `P25` (continuation at 0.2270) and `T14_32` -> `P25R` (re-noised at 0.2929);
   - baselines: `STU` and `W8` against `P25` (same seed), and `P25` against `P25` of the next seed (different seeds,
     same speaker and text), which is what "no shared melody" looks like.
3. Mel distance: ||X - out_P25|| / ||out_P25|| per seed over generated frames.

Usage (.venv with torch, vocos and librosa):
    .venv/Scripts/python code/analysis/exp032_bank_pitch_screen.py <bank dir containing bank/> [--conds clean,noisy15]
"""
import argparse
import hashlib
import json
import math
import statistics as st
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code" / "colab"))
import run_method_comparison as mc  # noqa: E402
import run_selective_handover_prelim as sh  # noqa: E402

EXP = ROOT / "research/experiments/EXP-032-variety-per-pass"
OUT = EXP / "offline" / "bank_pitch_screen.json"
MEL_SCALE = 2.901921510696411       # checkpoint['scale'] of model_85000.pt; the pipeline decodes x * scale
ESTIMATES = ("s0", "T1", "T2", "T3", "T4", "T8", "T14_32")
OUTPUTS = ("STU", "W4", "W8", "P25")
TEACHER_TIME = {**{f"T{k}": 1.0 - math.cos(math.pi / 2.0 * k / 16) for k in (1, 2, 3, 4, 8)},
                "T14_32": 1.0 - math.cos(math.pi / 2.0 * 14 / 32)}   # sway grids, t_i = 1 - cos(pi/2 * i/N)
SEMITONE = 100.0


def _f0(audio):
    return np.asarray(sh.f0_track(audio), dtype=np.float32)


def load_vocoder():
    import torch
    from huggingface_hub import hf_hub_download
    from vocos import Vocos
    voc = Vocos.from_hparams(hf_hub_download("charactr/vocos-mel-24khz", "config.yaml"))
    voc.load_state_dict(torch.load(hf_hub_download("charactr/vocos-mel-24khz", "pytorch_model.bin"),
                                   map_location="cpu", weights_only=True))
    return voc.eval()


def decode(voc, mel):
    import torch
    with torch.inference_mode():
        x = torch.from_numpy(np.asarray(mel, dtype=np.float32) * MEL_SCALE).T.unsqueeze(0)
        return voc.decode(x).squeeze().numpy().astype(np.float32)


def agreement(a, b):
    """Melody agreement of two f0 tracks of one seed (frame by frame, frames voiced in both)."""
    n = min(len(a), len(b))
    a, b = np.asarray(a[:n], dtype=float), np.asarray(b[:n], dtype=float)
    both = ~np.isnan(a) & ~np.isnan(b)
    either = ~np.isnan(a) | ~np.isnan(b)
    out = {"both_voiced": float(both.sum() / max(1, either.sum()))}
    if both.sum() < 5:
        return {**out, "within_semitone": None, "logf0_corr": None, "median_cents": None}
    cents = 1200.0 * np.abs(np.log2(a[both] / b[both]))
    la, lb = np.log(a[both]), np.log(b[both])
    corr = float(np.corrcoef(la, lb)[0, 1]) if la.std() > 0 and lb.std() > 0 else None
    return {**out, "within_semitone": float(np.mean(cents <= SEMITONE)), "logf0_corr": corr,
            "median_cents": float(np.median(cents))}


def mean_of(vals):
    vals = [v for v in vals if v is not None and v == v]
    return {"mean": st.mean(vals) if vals else None, "n": len(vals)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("bank_root", help="directory that contains bank/<condition>/*.npz")
    p.add_argument("--conds", default="clean,noisy15")
    p.add_argument("--workers", type=int, default=10)
    args = p.parse_args()
    root = Path(args.bank_root)
    index = json.loads((EXP / "results" / "bank_index.json").read_text(encoding="utf-8"))
    for e in index["entries"]:  # every bank file must match the hash index committed with the results
        f = root / e["path"]
        if hashlib.sha256(f.read_bytes()).hexdigest() != e["sha256"]:
            raise SystemExit(f"hash mismatch: {e['path']}")
    voc = load_vocoder()
    t0 = time.time()
    result = {"kind": "post hoc offline screen on EXP-032's tensor bank (CPU vocoding and pYIN; no TTS network)",
              "bank_files_verified": len(index["entries"]), "mel_scale": MEL_SCALE, "conditions": {}}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for cond in args.conds.split(","):
            files = sorted((root / "bank" / cond).glob("*.npz"))
            speakers = sorted({f.name.split("__s")[0] for f in files})
            per_speaker, pairs, mel_dist, check = {}, {}, {}, {}
            for spk in speakers:
                seeds = sorted(f for f in files if f.name.startswith(spk + "__s"))
                data = [np.load(f, allow_pickle=False) for f in seeds]
                mels = {k: [d[k].astype(np.float32) for d in data] for k in ESTIMATES + tuple(f"out_{a}" for a in OUTPUTS)}
                jobs = {k: [pool.submit(_f0, decode(voc, m)) for m in ms] for k, ms in mels.items()}
                f0 = {k: [j.result() for j in js] for k, js in jobs.items()}
                stored = {a: [d[f"f0_{a}"] for d in data] for a in OUTPUTS}
                e = {}
                for k in ESTIMATES + tuple(f"out_{a}" for a in OUTPUTS):
                    cv, voiced = sh.cvf0(f0[k])
                    cva, _, spread = mc.aligned_cvf0(f0[k], mels[k])
                    e[k] = {"cvf0": cv, "voiced": voiced, "cvf0_aligned": cva, "timing_spread": spread}
                for a in OUTPUTS:  # the stored T4 tracks, for the check
                    cv, voiced = sh.cvf0(stored[a])
                    cva, _, spread = mc.aligned_cvf0(stored[a], mels[f"out_{a}"])
                    e[f"stored_{a}"] = {"cvf0": cv, "voiced": voiced, "cvf0_aligned": cva, "timing_spread": spread}
                per_speaker[spk] = e
                n = len(data)
                ref = stored["P25"]
                for k in ESTIMATES:
                    pairs.setdefault(f"{k}~P25", []).extend(agreement(f0[k][s], ref[s]) for s in range(n))
                pairs.setdefault("T8~W8", []).extend(agreement(f0["T8"][s], stored["W8"][s]) for s in range(n))
                pairs.setdefault("T14_32~P25R", []).extend(agreement(f0["T14_32"][s], d["f0_P25R"])
                                                           for s, d in enumerate(data))
                pairs.setdefault("T4~W4", []).extend(agreement(f0["T4"][s], stored["W4"][s]) for s in range(n))
                pairs.setdefault("s0~STU", []).extend(agreement(f0["s0"][s], stored["STU"][s]) for s in range(n))
                for a in ("STU", "W4", "W8"):
                    pairs.setdefault(f"{a}~P25", []).extend(agreement(stored[a][s], ref[s]) for s in range(n))
                pairs.setdefault("P25~P25_other_seed", []).extend(agreement(ref[s], ref[(s + 1) % n]) for s in range(n))
                for a in OUTPUTS:
                    check.setdefault(a, []).extend(agreement(f0[f"out_{a}"][s], stored[a][s]) for s in range(n))
                for k in ESTIMATES + ("out_STU", "out_W4", "out_W8"):
                    mel_dist.setdefault(k, []).extend(
                        float(np.linalg.norm(mels[k][s] - mels["out_P25"][s]) / np.linalg.norm(mels["out_P25"][s]))
                        for s in range(n))
                print(f"[{cond}] {spk}: done ({time.time() - t0:.0f}s)", flush=True)
            summary = {}
            for key in ESTIMATES + tuple(f"out_{a}" for a in OUTPUTS) + tuple(f"stored_{a}" for a in OUTPUTS):
                summary[key] = {m: mean_of(per_speaker[s][key][m] for s in per_speaker)
                                for m in ("cvf0", "cvf0_aligned", "voiced", "timing_spread")}
            agree = {k: {m: mean_of(x[m] for x in v) for m in ("within_semitone", "logf0_corr", "median_cents",
                                                              "both_voiced")} for k, v in pairs.items()}
            result["conditions"][cond] = {
                "speakers": len(per_speaker), "seeds": n,
                "variety_mean_over_speakers": summary,
                "melody_agreement_same_seed": agree,
                "cpu_retrack_vs_stored_T4": {a: {m: mean_of(x[m] for x in v) for m in ("within_semitone", "both_voiced")}
                                             for a, v in check.items()},
                "mel_distance_to_P25_output": {k: mean_of(v) for k, v in mel_dist.items()},
                "per_speaker": per_speaker}
    result["seconds"] = round(time.time() - t0, 1)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"[OK] wrote {OUT}")


if __name__ == "__main__":
    main()
