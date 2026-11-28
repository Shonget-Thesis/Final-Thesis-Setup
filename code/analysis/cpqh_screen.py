"""Local screen of the Commit-Point Quantile Handover (CP-QH) (screening only, on this computer's CPU; not evidence).

The open problem (TTS-specific; see docs/progress/log/2026-11-17-literature-map.md): with a noisy voice prompt, the
teacher's prosodic variety arrives together with the prompt's noise, which the teacher copies; the student alone is
cleaner but flat. Every published remedy retrains the model or cleans the prompt first.

The candidate. The team's Quantile Handover (QH; 2026-11-15 theory note, Theorem 2) is the most informative handover
payload that is invariant to any per-band strictly increasing change of the teacher's estimate, such as a stationary
noise floor: in each mel band it places the student draft's own values in the teacher's time order. In EXP-031 it was
as clean as the student alone on white-noise prompts, but it ran at Step 3 with the student restarted at 0.0761,
which erases the teacher's melody (EXP-032 bank screen), and it kept little of the paper setting's pitch variety.
CP-QH moves it to the commit point found since: the teacher reaches t = 0.2929 in four steps (CPH8), QH replaces its
values by the student's, and the student restarts at 0.2929, the level at which the melody survives.

Arms (same seeds and student noises): STU (student alone, 4 passes), CPH8 (12), CP-QH (13: CPH8's teacher, the
student's one-pass draft for the values, the student's levels 0.2929 and 0.6173).
Conditions: clean and noisy15 (the experiments' white noise at 15 dB). Sentences: the three shortest clean EXP-033
quality cases; seeds: input_seed + 1000 + s, s = 0..5 (the experiments' diversity seeds).
Measures: UTMOS per output; CVf0 and aligned CVf0 across the six seeds per sentence (the experiments' functions);
melody agreement of CP-QH with CPH8 per seed (share of frames within a semitone).

Pass rules, fixed before the run:
1. noisy15 cleanliness: mean UTMOS(CP-QH) >= mean UTMOS(STU) - 0.10.
2. noisy15 variety: mean CVf0(CP-QH) > mean CVf0(STU) over the sentences where both are defined, and the same for
   aligned CVf0 (the teacher's melody gets through where the student alone has none).
3. clean variety: CP-QH keeps >= 75% of CPH8's CVf0 gain over STU (mean over sentences) on both measures.
4. clean quality: mean UTMOS(CP-QH) >= mean UTMOS(CPH8) - 0.10.
All four must hold to recommend a T4 test. Six seeds on three sentences make this a coarse filter.

Usage: .venv/Scripts/python -I code/analysis/cpqh_screen.py
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
import noise_color_screen as ncs  # noqa: E402

OUT = ROOT / "research/experiments/EXP-033-cheaper-teacher-path/offline/cpqh_screen.json"
N_SENT, N_SEEDS, SEED_BASE = 3, 6, 1000
CONDS = ("clean", "noisy15")
ARMS = ("STU", "CPH8", "CPQH")
NOISY_UTMOS_MARGIN, CLEAN_SHARE, CLEAN_UTMOS_MARGIN = 0.10, 0.75, 0.10


def melody_agreement(a, b):
    n = min(len(a), len(b))
    a, b = np.asarray(a[:n], dtype=float), np.asarray(b[:n], dtype=float)
    both = ~np.isnan(a) & ~np.isnan(b)
    if both.sum() < 5:
        return None
    return float(np.mean(1200 * np.abs(np.log2(a[both] / b[both])) <= 100))


def mean_defined(vals):
    vals = [v for v in vals if v is not None and v == v]
    return st.mean(vals) if vals else None


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
                    outs = {"STU": r.student(ctx, s0, ls.TAUS), "CPH8": r.student(ctx, t8, ls.TAUS[1:]),
                            "CPQH": r.student(ctx, mc.quantile_payload(s0, t8, plen), ls.TAUS[1:])}
                    audio = {a: r.e._decode(ctx, x).astype(np.float32) for a, x in outs.items()}
                    jobs = {a: pool.submit(sh.f0_track, audio[a]) for a in ARMS}
                    rec = {"utt_id": row["utt_id"], "condition": cond, "seed": s}
                    for a in ARMS:
                        tracks[a].append(np.asarray(jobs[a].result(), dtype=float))
                        mels[a].append(outs[a][0, plen:, :].numpy().astype(np.float32))
                        rec[f"utmos_{a}"] = float(utmos(torch.from_numpy(audio[a]).unsqueeze(0), 24000))
                    rec["melody_CPQH_vs_CPH8"] = melody_agreement(tracks["CPQH"][-1], tracks["CPH8"][-1])
                    rec["melody_STU_vs_CPH8"] = melody_agreement(tracks["STU"][-1], tracks["CPH8"][-1])
                    runs.append(rec)
                    print(f"[{len(runs)}/{N_SENT * len(CONDS) * N_SEEDS}] {row['utt_id'][:22]} {cond:7s} s{s} UTMOS "
                          + " ".join(f"{a} {rec[f'utmos_{a}']:.2f}" for a in ARMS)
                          + f" | melody QH~CPH8 {rec['melody_CPQH_vs_CPH8']} ({time.time() - t0:.0f}s)", flush=True)
                e = {"utt_id": row["utt_id"], "condition": cond}
                for a in ARMS:
                    e[f"cvf0_{a}"], e[f"voiced_{a}"] = sh.cvf0(tracks[a])
                    e[f"cvf0_aligned_{a}"], _, e[f"timing_{a}"] = mc.aligned_cvf0(tracks[a], mels[a])
                per_sentence.append(e)

    def utm(cond, a):
        return st.mean(x[f"utmos_{a}"] for x in runs if x["condition"] == cond)

    def cv(cond, a, m):
        return [e[f"{m}_{a}"] for e in per_sentence if e["condition"] == cond]

    noisy_pairs = {m: [(q, s) for q, s in zip(cv("noisy15", "CPQH", m), cv("noisy15", "STU", m)) if q == q and s == s]
                   for m in ("cvf0", "cvf0_aligned")}
    clean_share = {}
    for m in ("cvf0", "cvf0_aligned"):
        q, c, s = (mean_defined(cv("clean", a, m)) for a in ("CPQH", "CPH8", "STU"))
        clean_share[m] = (q - s) / (c - s) if None not in (q, c, s) and c - s > 1e-9 else None
    rules = {
        "noisy_cleanliness": utm("noisy15", "CPQH") >= utm("noisy15", "STU") - NOISY_UTMOS_MARGIN,
        "noisy_variety": all(p and st.mean(q for q, _ in p) > st.mean(s for _, s in p) for p in noisy_pairs.values()),
        "clean_variety": all(v is not None and v >= CLEAN_SHARE for v in clean_share.values()),
        "clean_quality": utm("clean", "CPQH") >= utm("clean", "CPH8") - CLEAN_UTMOS_MARGIN}
    summary = {"rules": {k: bool(v) for k, v in rules.items()}, "all_pass": bool(all(rules.values())),
               "utmos": {c: {a: utm(c, a) for a in ARMS} for c in CONDS},
               "noisy_variety_pairs": {m: {"n": len(p), "CPQH": mean_defined([q for q, _ in p]),
                                           "STU": mean_defined([s for _, s in p])} for m, p in noisy_pairs.items()},
               "cvf0_mean": {c: {a: {m: mean_defined(cv(c, a, m)) for m in ("cvf0", "cvf0_aligned")} for a in ARMS} for c in CONDS},
               "clean_share_of_CPH8_gain": clean_share,
               "melody_vs_CPH8": {c: {a: mean_defined([x[f"melody_{a}_vs_CPH8"] for x in runs if x["condition"] == c])
                                      for a in ("CPQH", "STU")} for c in CONDS}}
    OUT.write_text(json.dumps({"kind": "local CPU screen (screening only, not evidence)", "summary": summary,
                               "per_sentence": per_sentence, "runs": runs, "seconds": round(time.time() - t0)},
                              indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(summary, indent=2))
    print(f"[OK] wrote {OUT}")


if __name__ == "__main__":
    main()
