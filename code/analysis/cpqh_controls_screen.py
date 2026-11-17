"""Controls for CP-QH's noisy-prompt cleaning (screening only, on this computer's CPU; not evidence).

In both CP-QH screens (cpqh_screen.py, floor_canonical_screen.py) CP-QH was much cleaner than the student alone on
noisy15 prompts (UTMOS 2.74 and 2.84 against 1.68 and 2.14) although every payload value is the student draft's. Two
differences could explain it:
- the restart levels: the student alone restarts at 0.0761, 0.2929 and 0.6173; CP-QH only at 0.2929 and 0.6173;
- the time order: CP-QH arranges the draft's values in the teacher's order (the teacher's estimate at t = 0.2929).

Arms (same seeds and per-level student noises as the earlier screens):
  STU     the student alone: draft, then 0.0761, 0.2929, 0.6173 (4 passes). Reproduces the earlier screens.
  STUR2   the draft restarted at 0.2929 and 0.6173 only (3 passes): CP-QH with the teacher's order replaced by the
          draft's own order (QH of the draft with itself is the draft).
  SQH     the draft's values in the time order of the draft smoothed over 9 frames (about 96 ms) per band, then
          0.2929 and 0.6173 (3 passes): any smooth order, no teacher.
  CPQH    as before (13 passes). Reproduces the earlier screens.
Cases: the six sentences of the two earlier screens (the six shortest clean EXP-033 quality cases), clean and noisy15,
seeds input_seed + 1000 + s, s = 0..5.

Questions, with the reading fixed before the run (differences in mean noisy15 UTMOS over all 36 noisy runs):
Q1. Does the teacher's order matter? Yes if UTMOS(CPQH) - UTMOS(STUR2) >= 0.30; the restart levels explain the
    cleaning if the difference is below 0.10 (in between: partly).
Q2. Does any smooth order do it? Yes if UTMOS(SQH) >= UTMOS(CPQH) - 0.10; no if UTMOS(CPQH) - UTMOS(SQH) >= 0.30.
Also reported: pitch variety of each arm (clean and noisy15), the voiced share, and diagnostics of the teacher's
estimate at t = 0.2929 in the upper bands (60-99): its mean power over its 10th percentile (how much of the
copied noise lies above the floor that FCH matched) and the temporal roughness (mean squared frame-to-frame change
over the variance) of the teacher's estimate, the draft and the smoothed draft.

Usage: .venv/Scripts/python -I code/analysis/cpqh_controls_screen.py
"""
import json
import statistics as st
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for p in ("code/colab", "code/prototype", "code/analysis"):
    sys.path.insert(0, str(ROOT / p))
import cpqh_screen as cq  # noqa: E402
import noise_color_screen as ncs  # noqa: E402

OUT = ROOT / "research/experiments/EXP-033-cheaper-teacher-path/offline/cpqh_controls_screen.json"
EARLIER = (ROOT / "research/experiments/EXP-033-cheaper-teacher-path/offline/cpqh_screen.json",
           ROOT / "research/experiments/EXP-033-cheaper-teacher-path/offline/floor_canonical_screen.json")
N_SENT, N_SEEDS, SEED_BASE = 6, 6, 1000
CONDS = ("clean", "noisy15")
ARMS = ("STU", "STUR2", "SQH", "CPQH")
SMOOTH_FRAMES = 9
UPPER_BANDS = slice(60, None)
MATTERS, NO_EFFECT, SAME = 0.30, 0.10, 0.10


def smooth_time(x, frames=SMOOTH_FRAMES):
    """Moving average over `frames` frames along time (axis 0) of a (frames, bands) array, edges padded by reflection."""
    x = np.asarray(x, dtype=np.float64)
    half = frames // 2
    padded = np.pad(x, ((half, half), (0, 0)), mode="reflect")
    c = np.cumsum(np.vstack([np.zeros((1, x.shape[1])), padded]), axis=0)
    return (c[frames:] - c[:-frames]) / frames


def roughness(x):
    """Mean squared frame-to-frame change over the variance, averaged over bands (2 for white noise, near 0 if smooth)."""
    x = np.asarray(x, dtype=np.float64)
    d = np.diff(x, axis=0)
    return float(np.mean(np.mean(d * d, axis=0) / np.maximum(np.var(x, axis=0), 1e-12)))


def above_floor_db(x):
    """Upper-band mean power over the 10th-percentile power, in dB (x is natural-log magnitude, P = exp(2x))."""
    p = np.exp(2.0 * np.asarray(x, dtype=np.float64))
    return float(np.median(10 * np.log10(p.mean(axis=0) / np.percentile(p, 10, axis=0))))


def earlier_utmos():
    out = {}
    for path in EARLIER:
        if path.exists():
            for r in json.loads(path.read_text(encoding="utf-8"))["runs"]:
                out[(r["utt_id"], r["condition"], r["seed"])] = {a: r[f"utmos_{a}"] for a in ("STU", "CPQH")}
    return out


def main():
    import local_candidate_screen as ls
    import run_followup_tests as rf
    import run_method_comparison as mc
    import run_selective_handover_prelim as sh
    import soundfile as sf
    from concurrent.futures import ProcessPoolExecutor
    r = ls.Runner()
    torch = r.torch
    utmos = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True).eval()
    items = {it["utt_id"]: it for it in r.en.development_examples()}
    before = earlier_utmos()
    t0 = time.time()
    per_sentence, runs = [], []
    with ProcessPoolExecutor(max_workers=6) as pool, torch.inference_mode():
        for row in ncs.pick_cases(N_SENT):
            it = items[row["utt_id"]]
            clean, sr = sf.read(str(r.en.DATA_DIR / it["prompt_wav_rel"]), dtype="float32", always_2d=False)
            clean = (clean.mean(axis=1) if clean.ndim > 1 else clean).astype(np.float32)
            for cond in CONDS:
                prompt = rf.degrade(clean, sr, cond, int(it["input_seed"]))
                tracks, mels = {a: [] for a in ARMS}, {a: [] for a in ARMS}
                for s in range(N_SEEDS):
                    ctx = r.e.prepare(prompt, sr, it["prompt_text"], it["gt_text"], int(it["input_seed"]) + SEED_BASE + s)
                    plen = ctx["prompt_len"]
                    s0 = r.student(ctx, ctx["step_cond"], [0.0])
                    t8 = r.teacher(ctx, ls.sway(8), 4)[4]
                    g_s0 = s0[0, plen:, :].numpy()
                    g_sm = smooth_time(g_s0)
                    sqh = s0.clone()
                    sqh[0, plen:, :] = torch.from_numpy(mc.quantile_handover(g_s0, g_sm).astype(np.float32))
                    outs = {"STU": r.student(ctx, s0, ls.TAUS), "STUR2": r.student(ctx, s0, ls.TAUS[1:]),
                            "SQH": r.student(ctx, sqh, ls.TAUS[1:]),
                            "CPQH": r.student(ctx, mc.quantile_payload(s0, t8, plen), ls.TAUS[1:])}
                    audio = {a: r.e._decode(ctx, x).astype(np.float32) for a, x in outs.items()}
                    jobs = {a: pool.submit(sh.f0_track, audio[a]) for a in ARMS}
                    g_t = t8[0, plen:, :].numpy()
                    rec = {"utt_id": row["utt_id"], "condition": cond, "seed": s,
                           "teacher_above_floor_db": above_floor_db(g_t[:, UPPER_BANDS]),
                           "draft_above_floor_db": above_floor_db(g_s0[:, UPPER_BANDS]),
                           "roughness_teacher": roughness(g_t), "roughness_draft": roughness(g_s0),
                           "roughness_smoothed_draft": roughness(g_sm)}
                    for a in ARMS:
                        tracks[a].append(np.asarray(jobs[a].result(), dtype=float))
                        mels[a].append(outs[a][0, plen:, :].numpy().astype(np.float32))
                        rec[f"utmos_{a}"] = float(utmos(torch.from_numpy(audio[a]).unsqueeze(0), 24000))
                    old = before.get((row["utt_id"], cond, s))
                    rec["repro_max_abs"] = (max(abs(rec[f"utmos_{a}"] - old[a]) for a in ("STU", "CPQH"))
                                            if old else None)
                    runs.append(rec)
                    print(f"[{len(runs)}/{N_SENT * len(CONDS) * N_SEEDS}] {row['utt_id'][:22]} {cond:7s} s{s} UTMOS "
                          + " ".join(f"{a} {rec[f'utmos_{a}']:.2f}" for a in ARMS)
                          + f" | rough T {rec['roughness_teacher']:.2f} S {rec['roughness_draft']:.2f}"
                          + f" | repro {rec['repro_max_abs']} ({time.time() - t0:.0f}s)", flush=True)
                e = {"utt_id": row["utt_id"], "condition": cond}
                for a in ARMS:
                    e[f"cvf0_{a}"], e[f"voiced_{a}"] = sh.cvf0(tracks[a])
                    e[f"cvf0_aligned_{a}"], _, e[f"timing_{a}"] = mc.aligned_cvf0(tracks[a], mels[a])
                per_sentence.append(e)

    def utm(cond, a):
        return st.mean(x[f"utmos_{a}"] for x in runs if x["condition"] == cond)

    q1 = utm("noisy15", "CPQH") - utm("noisy15", "STUR2")
    q2 = utm("noisy15", "CPQH") - utm("noisy15", "SQH")
    summary = {
        "Q1_teacher_order": {"cpqh_minus_stur2": q1,
                             "reading": "matters" if q1 >= MATTERS else ("restart levels" if q1 < NO_EFFECT else "partly")},
        "Q2_any_smooth_order": {"cpqh_minus_sqh": q2,
                                "reading": "yes" if q2 <= SAME else ("no" if q2 >= MATTERS else "partly")},
        "utmos": {c: {a: utm(c, a) for a in ARMS} for c in CONDS},
        "below_2": {c: {a: sum(x[f"utmos_{a}"] < 2.0 for x in runs if x["condition"] == c) for a in ARMS} for c in CONDS},
        "cvf0_mean": {c: {a: {m: cq.mean_defined([e[f"{m}_{a}"] for e in per_sentence if e["condition"] == c])
                              for m in ("cvf0", "cvf0_aligned")} for a in ARMS} for c in CONDS},
        "voiced_mean": {c: {a: cq.mean_defined([e[f"voiced_{a}"] for e in per_sentence if e["condition"] == c])
                            for a in ARMS} for c in CONDS},
        "diagnostics": {c: {k: cq.mean_defined([x[k] for x in runs if x["condition"] == c])
                            for k in ("teacher_above_floor_db", "draft_above_floor_db", "roughness_teacher",
                                      "roughness_draft", "roughness_smoothed_draft")} for c in CONDS},
        "reproduction_max_abs_utmos": max((x["repro_max_abs"] for x in runs if x["repro_max_abs"] is not None),
                                          default=None)}
    OUT.write_text(json.dumps({"kind": "local CPU screen (screening only, not evidence)", "summary": summary,
                               "per_sentence": per_sentence, "runs": runs, "seconds": round(time.time() - t0)},
                              indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(summary, indent=2))
    print(f"[OK] wrote {OUT}")


if __name__ == "__main__":
    main()
