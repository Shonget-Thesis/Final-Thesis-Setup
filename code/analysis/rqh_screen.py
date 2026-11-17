"""Local screen of the Roughness-Selected Quantile Handover (RQH) (screening only, CPU; not evidence).

What the earlier screens found (docs/progress/log/2026-11-17-commit-point-quantile-handover.md):
- CPH8 (the teacher's estimate at t = 0.2929 handed over) carries the teacher's prosody on clean prompts but copies a
  noisy prompt's background: on noisy15 the teacher's estimate is rougher in time than the student's draft.
- CP-QH (the draft's values in the teacher's time order) is clean on noisy15 and keeps the teacher's timing, but on
  clean prompts it gives up part of the teacher's pitch range.
- The student's first restart level (0.0761) is where the student alone copies the noise; every arm here restarts
  the student at 0.2929 and 0.6173 only (except STU, the released student path).
- Exploratory probe (roughness_band_probe.py, the same six sentences): the teacher's estimate is rougher than the
  draft in 1.8% of bands on clean prompts and in 85% on noisy15 prompts.

The candidate. Per mel band b, over the generated frames, with rho the mean squared frame-to-frame change over the
band's variance:

    RQH(x_hat, s0)_b = QH(x_hat, s0)_b   if rho(x_hat_b) > rho(s0_b)      (the teacher's band carries texture the
                       x_hat_b           otherwise                          student's draft does not)

The student's draft is the clean reference; no prompt statistic, no fitted threshold. It reduces to CPH8 when no band
is rougher than the draft and to CP-QH when all are. Prompt frames are the teacher's, as in CPH8. 13 passes.

Arms (same seeds and per-level student noises): STU (4), STUR2 (the draft restarted at 0.2929 and 0.6173, 3),
CPH8 (12), CPQH (13), RQH (13). Conditions: clean and noisy15. Sentences: the 7th to 12th shortest clean EXP-033
quality cases (not used in any earlier screen); seeds input_seed + 1000 + s, s = 0..5.

Pass rules for RQH, fixed before the run:
1. noisy15 quality: mean UTMOS(RQH) >= mean UTMOS(CPQH) - 0.10, and at most 10% of its noisy15 outputs below 2.0.
2. noisy15 variety: over sentences where RQH and STUR2 both voice >= 30% of frames, mean CVf0(RQH) > mean
   CVf0(STUR2), and the same for aligned CVf0 (the teacher's prosody gets through the clean path).
3. clean variety: RQH keeps >= 75% of CPH8's gain over STU on both measures.
4. clean quality: mean UTMOS(RQH) >= mean UTMOS(CPH8) - 0.10.
All four must hold to recommend a T4 test. The other arms' numbers are reported as replications.

Usage: .venv/Scripts/python -I code/analysis/rqh_screen.py
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
import roughness_band_probe as rbp  # noqa: E402

OUT = ROOT / "research/experiments/EXP-033-cheaper-teacher-path/offline/rqh_screen.json"
FIRST, N_SENT, N_SEEDS, SEED_BASE = 6, 6, 6, 1000
CONDS = ("clean", "noisy15")
ARMS = ("STU", "STUR2", "CPH8", "CPQH", "RQH")
MIN_VOICED, MAX_BELOW_2 = 0.30, 0.10


def roughness_selected(student_gen, teacher_gen):
    """RQH on (frames, bands) arrays (numpy or torch): QH in the bands where the teacher is rougher than the draft,
    the teacher's values elsewhere. Returns (payload, mask of QH bands)."""
    import run_method_comparison as mc
    is_torch = hasattr(teacher_gen, "clone")
    t = teacher_gen.detach().cpu().numpy() if is_torch else np.asarray(teacher_gen)
    s = student_gen.detach().cpu().numpy() if is_torch else np.asarray(student_gen)
    rough = rbp.band_roughness(t) > rbp.band_roughness(s)
    out = np.where(rough[None, :], mc.quantile_handover(s, t), t)
    if is_torch:
        import torch
        return torch.from_numpy(out.astype(np.float32)).to(dtype=teacher_gen.dtype, device=teacher_gen.device), rough
    return out.astype(t.dtype), rough


def rqh_payload(s0, t_k, plen):
    """RQH payload on (1, frames, bands): the generated frames [plen:] selected per band; prompt frames the teacher's."""
    out = t_k.clone() if hasattr(t_k, "clone") else np.array(t_k, copy=True)
    out[0, plen:, :], rough = roughness_selected(s0[0, plen:, :], t_k[0, plen:, :])
    return out, rough


def evaluate(runs, per_sentence):
    def utm(cond, a):
        return st.mean(x[f"utmos_{a}"] for x in runs if x["condition"] == cond)

    def cv(cond, a, m):
        return [e[f"{m}_{a}"] for e in per_sentence if e["condition"] == cond]

    noisy = [x for x in runs if x["condition"] == "noisy15"]
    below = sum(x["utmos_RQH"] < 2.0 for x in noisy) / max(len(noisy), 1)
    pairs = {}
    for m in ("cvf0", "cvf0_aligned"):
        pairs[m] = [(e[f"{m}_RQH"], e[f"{m}_STUR2"]) for e in per_sentence if e["condition"] == "noisy15"
                    and (e["voiced_RQH"] or 0) >= MIN_VOICED and (e["voiced_STUR2"] or 0) >= MIN_VOICED
                    and e[f"{m}_RQH"] == e[f"{m}_RQH"] and e[f"{m}_STUR2"] == e[f"{m}_STUR2"]]
    share = {}
    for m in ("cvf0", "cvf0_aligned"):
        q, c, s = (cq.mean_defined(cv("clean", a, m)) for a in ("RQH", "CPH8", "STU"))
        share[m] = (q - s) / (c - s) if None not in (q, c, s) and c - s > 1e-9 else None
    rules = {"noisy_quality": utm("noisy15", "RQH") >= utm("noisy15", "CPQH") - 0.10 and below <= MAX_BELOW_2,
             "noisy_variety": all(p and st.mean(q for q, _ in p) > st.mean(s for _, s in p) for p in pairs.values()),
             "clean_variety": all(v is not None and v >= cq.CLEAN_SHARE for v in share.values()),
             "clean_quality": utm("clean", "RQH") >= utm("clean", "CPH8") - cq.CLEAN_UTMOS_MARGIN}
    return {"rules": {k: bool(v) for k, v in rules.items()}, "all_pass": bool(all(rules.values())),
            "noisy_share_below_2": below, "clean_share_of_CPH8_gain": share,
            "noisy_variety_pairs": {m: {"n": len(p), "RQH": cq.mean_defined([q for q, _ in p]),
                                        "STUR2": cq.mean_defined([s for _, s in p])} for m, p in pairs.items()},
            "utmos": {c: {a: utm(c, a) for a in ARMS} for c in CONDS},
            "below_2": {c: {a: sum(x[f"utmos_{a}"] < 2.0 for x in runs if x["condition"] == c) for a in ARMS}
                        for c in CONDS},
            "cvf0_mean": {c: {a: {m: cq.mean_defined(cv(c, a, m)) for m in ("cvf0", "cvf0_aligned")} for a in ARMS}
                          for c in CONDS},
            "voiced_mean": {c: {a: cq.mean_defined(cv(c, a, "voiced")) for a in ARMS} for c in CONDS},
            "qh_band_share": {c: cq.mean_defined([x["qh_band_share"] for x in runs if x["condition"] == c])
                              for c in CONDS}}


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
                    rqh, rough = rqh_payload(s0, t8, plen)
                    outs = {"STU": r.student(ctx, s0, ls.TAUS), "STUR2": r.student(ctx, s0, ls.TAUS[1:]),
                            "CPH8": r.student(ctx, t8, ls.TAUS[1:]),
                            "CPQH": r.student(ctx, mc.quantile_payload(s0, t8, plen), ls.TAUS[1:]),
                            "RQH": r.student(ctx, rqh, ls.TAUS[1:])}
                    audio = {a: r.e._decode(ctx, x).astype(np.float32) for a, x in outs.items()}
                    jobs = {a: pool.submit(sh.f0_track, audio[a]) for a in ARMS}
                    rec = {"utt_id": row["utt_id"], "condition": cond, "seed": s, "qh_band_share": float(rough.mean())}
                    for a in ARMS:
                        tracks[a].append(np.asarray(jobs[a].result(), dtype=float))
                        mels[a].append(outs[a][0, plen:, :].numpy().astype(np.float32))
                        rec[f"utmos_{a}"] = float(utmos(torch.from_numpy(audio[a]).unsqueeze(0), 24000))
                    for a in ("RQH", "CPQH", "STU"):
                        rec[f"melody_{a}_vs_CPH8"] = cq.melody_agreement(tracks[a][-1], tracks["CPH8"][-1])
                    runs.append(rec)
                    print(f"[{len(runs)}/{N_SENT * len(CONDS) * N_SEEDS}] {row['utt_id'][:22]} {cond:7s} s{s} UTMOS "
                          + " ".join(f"{a} {rec[f'utmos_{a}']:.2f}" for a in ARMS)
                          + f" | QH bands {rec['qh_band_share']:.2f} ({time.time() - t0:.0f}s)", flush=True)
                e = {"utt_id": row["utt_id"], "condition": cond}
                for a in ARMS:
                    e[f"cvf0_{a}"], e[f"voiced_{a}"] = sh.cvf0(tracks[a])
                    e[f"cvf0_aligned_{a}"], _, e[f"timing_{a}"] = mc.aligned_cvf0(tracks[a], mels[a])
                per_sentence.append(e)

    summary = evaluate(runs, per_sentence)
    summary["melody_vs_CPH8"] = {c: {a: cq.mean_defined([x[f"melody_{a}_vs_CPH8"] for x in runs
                                                          if x["condition"] == c]) for a in ("RQH", "CPQH", "STU")}
                                 for c in CONDS}
    OUT.write_text(json.dumps({"kind": "local CPU screen (screening only, not evidence)", "summary": summary,
                               "per_sentence": per_sentence, "runs": runs, "seconds": round(time.time() - t0)},
                              indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(summary, indent=2))
    print(f"[OK] wrote {OUT}")


if __name__ == "__main__":
    main()
