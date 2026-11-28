"""Local screen of the Floor-Canonical Handover (FCH), with a replication of CP-QH (screening only, CPU; not evidence).

Why. The CP-QH screen (cpqh_screen.py, 2026-11-17) failed one pre-set rule: on clean prompts it kept 54% of CPH8's
CVf0 gain over the student alone (34% aligned), against a 75% bar. The diagnosis: QH is invariant to every per-band
strictly increasing map, so it may take only the time order from the teacher; every value, and with it each band's
share of loud frames, is the student's, which carries the student's narrower pitch range.

The changed term. A stationary additive background does much less than an arbitrary increasing map: in each mel band
it adds a constant in power, P -> P + n_b (log-mel here is the natural log of the mel magnitude, so P = exp(2x)). The
maximal invariant of that group is the band's power relative to its own floor, P - q10(P). Its canonical form, with
the floor fixed by the student's draft s0, is the Floor-Canonical Handover:

    FCH(x_hat, s0)_b(t) = 0.5 * log( max( P_T(t) - q10(P_T) + q10(P_S),  q01(P_S) ) )

with P_T = exp(2 x_hat_b), P_S = exp(2 s0_b) over the generated frames, q10/q01 the 10th/1st percentiles over time.
It keeps the teacher's values above its floor (melody and range) and replaces only the floor with the student's; if
the teacher's and the student's floors agree, it returns the teacher's estimate (CPH8). It runs at the commit point
like CP-QH: the teacher to t = 0.2929 (10 passes), the student's draft (1), restarts at 0.2929 and 0.6173 (2): 13.

Arms: STU, CPH8, CPQH (replication) and FCH, same seeds and student noises. Conditions: clean and noisy15. Sentences:
the 4th to 6th shortest clean EXP-033 quality cases (not used in the CP-QH screen); seeds input_seed + 1000 + s,
s = 0..5. Measures as in cpqh_screen.py.

Pass rules for FCH, fixed before the run (the same four as CP-QH's):
1. noisy15 cleanliness: mean UTMOS(FCH) >= mean UTMOS(STU) - 0.10.
2. noisy15 variety: mean CVf0(FCH) > mean CVf0(STU) over sentences where both are defined, and the same for aligned.
3. clean variety: FCH keeps >= 75% of CPH8's gain over STU on both measures.
4. clean quality: mean UTMOS(FCH) >= mean UTMOS(CPH8) - 0.10.
All four must hold to recommend a T4 test. CP-QH's verdict stays the first screen's (fail); the same rules are
reported for it here only as a replication on new sentences.

Usage: .venv/Scripts/python -I code/analysis/floor_canonical_screen.py
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
import cpqh_screen as cq  # noqa: E402
import noise_color_screen as ncs  # noqa: E402

OUT = ROOT / "research/experiments/EXP-033-cheaper-teacher-path/offline/floor_canonical_screen.json"
FIRST, N_SENT, N_SEEDS, SEED_BASE = 3, 3, 6, 1000
CONDS = ("clean", "noisy15")
ARMS = ("STU", "CPH8", "CPQH", "FCH")
CANDIDATES = ("FCH", "CPQH")
FLOOR_Q, CLAMP_Q = 10.0, 1.0


def floor_canonical(student_gen, teacher_gen, floor_q=FLOOR_Q, clamp_q=CLAMP_Q):
    """FCH on (frames, bands) natural-log mel magnitudes (numpy or torch): per band, the teacher's power relative to
    its own floor (q10 over time) plus the student's floor, clamped at the student's 1st-percentile power."""
    is_torch = hasattr(teacher_gen, "clone")
    t = teacher_gen.detach().cpu().numpy() if is_torch else np.asarray(teacher_gen)
    s = student_gen.detach().cpu().numpy() if is_torch else np.asarray(student_gen)
    pt, ps = np.exp(2.0 * t.astype(np.float64)), np.exp(2.0 * s.astype(np.float64))
    shifted = pt - np.percentile(pt, floor_q, axis=0) + np.percentile(ps, floor_q, axis=0)
    out = 0.5 * np.log(np.maximum(shifted, np.percentile(ps, clamp_q, axis=0)))
    if is_torch:
        import torch
        return torch.from_numpy(out.astype(np.float32)).to(dtype=teacher_gen.dtype, device=teacher_gen.device)
    return out.astype(t.dtype)


def floor_payload(s0, t_k, plen):
    """FCH payload on (1, frames, bands): the generated frames [plen:] canonicalized; the prompt frames are the
    teacher's, as in CPH8 (the student's restart keeps the prompt region from its condition)."""
    out = t_k.clone() if hasattr(t_k, "clone") else np.array(t_k, copy=True)
    out[0, plen:, :] = floor_canonical(s0[0, plen:, :], t_k[0, plen:, :])
    return out


def rules_for(arm, runs, per_sentence):
    def utm(cond, a):
        return st.mean(x[f"utmos_{a}"] for x in runs if x["condition"] == cond)

    def cv(cond, a, m):
        return [e[f"{m}_{a}"] for e in per_sentence if e["condition"] == cond]

    pairs = {m: [(q, s) for q, s in zip(cv("noisy15", arm, m), cv("noisy15", "STU", m))
                 if q is not None and s is not None and q == q and s == s] for m in ("cvf0", "cvf0_aligned")}
    share = {}
    for m in ("cvf0", "cvf0_aligned"):
        q, c, s = (cq.mean_defined(cv("clean", a, m)) for a in (arm, "CPH8", "STU"))
        share[m] = (q - s) / (c - s) if None not in (q, c, s) and c - s > 1e-9 else None
    rules = {"noisy_cleanliness": utm("noisy15", arm) >= utm("noisy15", "STU") - cq.NOISY_UTMOS_MARGIN,
             "noisy_variety": all(p and st.mean(q for q, _ in p) > st.mean(s for _, s in p) for p in pairs.values()),
             "clean_variety": all(v is not None and v >= cq.CLEAN_SHARE for v in share.values()),
             "clean_quality": utm("clean", arm) >= utm("clean", "CPH8") - cq.CLEAN_UTMOS_MARGIN}
    return {"rules": {k: bool(v) for k, v in rules.items()}, "all_pass": bool(all(rules.values())),
            "clean_share_of_CPH8_gain": share,
            "noisy_variety_pairs": {m: {"n": len(p), arm: cq.mean_defined([q for q, _ in p]),
                                        "STU": cq.mean_defined([s for _, s in p])} for m, p in pairs.items()}}


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
    per_sentence, runs, floors = [], [], []
    with ProcessPoolExecutor(max_workers=6) as pool, torch.inference_mode():
        for row in ncs.pick_cases(FIRST + N_SENT)[FIRST:]:
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
                    fch = floor_payload(s0, t8, plen)
                    pt = np.exp(2.0 * t8[0, plen:, :].numpy().astype(np.float64))
                    ps = np.exp(2.0 * s0[0, plen:, :].numpy().astype(np.float64))
                    excess = 10 * np.log10(np.percentile(pt, FLOOR_Q, axis=0) / np.percentile(ps, FLOOR_Q, axis=0))
                    floors.append({"utt_id": row["utt_id"], "condition": cond, "seed": s,
                                   "teacher_floor_excess_db_median": float(np.median(excess)),
                                   "teacher_floor_excess_db_high_bands": float(np.median(excess[60:])),
                                   "payload_change_rms": float(torch.sqrt(torch.mean((fch - t8)[0, plen:, :] ** 2)))})
                    outs = {"STU": r.student(ctx, s0, ls.TAUS), "CPH8": r.student(ctx, t8, ls.TAUS[1:]),
                            "CPQH": r.student(ctx, mc.quantile_payload(s0, t8, plen), ls.TAUS[1:]),
                            "FCH": r.student(ctx, fch, ls.TAUS[1:])}
                    audio = {a: r.e._decode(ctx, x).astype(np.float32) for a, x in outs.items()}
                    jobs = {a: pool.submit(sh.f0_track, audio[a]) for a in ARMS}
                    rec = {"utt_id": row["utt_id"], "condition": cond, "seed": s}
                    for a in ARMS:
                        tracks[a].append(np.asarray(jobs[a].result(), dtype=float))
                        mels[a].append(outs[a][0, plen:, :].numpy().astype(np.float32))
                        rec[f"utmos_{a}"] = float(utmos(torch.from_numpy(audio[a]).unsqueeze(0), 24000))
                    for a in ("CPQH", "FCH", "STU"):
                        rec[f"melody_{a}_vs_CPH8"] = cq.melody_agreement(tracks[a][-1], tracks["CPH8"][-1])
                    runs.append(rec)
                    print(f"[{len(runs)}/{N_SENT * len(CONDS) * N_SEEDS}] {row['utt_id'][:22]} {cond:7s} s{s} UTMOS "
                          + " ".join(f"{a} {rec[f'utmos_{a}']:.2f}" for a in ARMS)
                          + f" | floor excess {floors[-1]['teacher_floor_excess_db_median']:+.1f} dB"
                          + f" | melody FCH~CPH8 {rec['melody_FCH_vs_CPH8']} ({time.time() - t0:.0f}s)", flush=True)
                e = {"utt_id": row["utt_id"], "condition": cond}
                for a in ARMS:
                    e[f"cvf0_{a}"], e[f"voiced_{a}"] = sh.cvf0(tracks[a])
                    e[f"cvf0_aligned_{a}"], _, e[f"timing_{a}"] = mc.aligned_cvf0(tracks[a], mels[a])
                per_sentence.append(e)

    summary = {a: rules_for(a, runs, per_sentence) for a in CANDIDATES}
    summary["utmos"] = {c: {a: st.mean(x[f"utmos_{a}"] for x in runs if x["condition"] == c) for a in ARMS} for c in CONDS}
    summary["cvf0_mean"] = {c: {a: {m: cq.mean_defined([e[f"{m}_{a}"] for e in per_sentence if e["condition"] == c])
                                    for m in ("cvf0", "cvf0_aligned")} for a in ARMS} for c in CONDS}
    summary["voiced_mean"] = {c: {a: cq.mean_defined([e[f"voiced_{a}"] for e in per_sentence if e["condition"] == c])
                                  for a in ARMS} for c in CONDS}
    summary["melody_vs_CPH8"] = {c: {a: cq.mean_defined([x[f"melody_{a}_vs_CPH8"] for x in runs if x["condition"] == c])
                                     for a in ("CPQH", "FCH", "STU")} for c in CONDS}
    summary["teacher_floor_excess_db"] = {c: {k: cq.mean_defined([f[k] for f in floors if f["condition"] == c])
                                              for k in ("teacher_floor_excess_db_median",
                                                        "teacher_floor_excess_db_high_bands", "payload_change_rms")}
                                          for c in CONDS}
    OUT.write_text(json.dumps({"kind": "local CPU screen (screening only, not evidence)", "summary": summary,
                               "per_sentence": per_sentence, "runs": runs, "floors": floors,
                               "seconds": round(time.time() - t0)}, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(summary, indent=2))
    print(f"[OK] wrote {OUT}")


if __name__ == "__main__":
    main()
