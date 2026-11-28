"""Control: does the payload's prompt region carry the copied background? (screening only, CPU; not evidence)

Every handover payload covers the whole sequence: the prompt frames [:plen] and the generated frames [plen:]. The
student is re-noised from all of it. The earlier screens mixed two sources without testing the difference:
- CPH8, FCH and RQH hand over the teacher's estimate in the prompt frames;
- CP-QH (run_method_comparison.quantile_payload) and the student-only arms hand over the student's draft there.
In the RQH screen (rqh_screen.py), RQH applied QH to 73-99% of the bands on one noisy15 sentence yet scored far below
CP-QH (for example 1.43 against 2.43 with 99% of bands), which points at the prompt frames.

Arms (the student restarted at 0.2929 and 0.6173 in all; same seeds and student noises):
  CPH8      teacher's generated frames, teacher's prompt frames (the released payload at the commit point)  12 passes
  CPH8SP    teacher's generated frames, the draft's prompt frames                                            13
  CPQH      QH generated frames, the draft's prompt frames                                                   13
  CPQHTP    QH generated frames, the teacher's prompt frames                                                 13
  STUR2     the draft everywhere                                                                              3
Cases: the six shortest clean EXP-033 quality cases (screens 1 and 2), clean and noisy15, seeds input_seed + 1000 + s,
s = 0..5 (72 runs).

Readings fixed before the run (mean noisy15 UTMOS over the 36 noisy runs):
Q3. The teacher's prompt frames carry its copying: yes if UTMOS(CPH8SP) - UTMOS(CPH8) >= 0.50; no if below 0.10.
Q4. The draft's prompt frames are part of CP-QH's cleanliness: yes if UTMOS(CPQH) - UTMOS(CPQHTP) >= 0.30; no if
    below 0.10.
If Q3 is yes, CPH8SP becomes a candidate single handover, judged by the four rules of the earlier screens against
CPH8 and the student alone (reported here as a first look; a candidate verdict needs new sentences).

Usage: .venv/Scripts/python -I code/analysis/prompt_region_control.py
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

OUT = ROOT / "research/experiments/EXP-033-cheaper-teacher-path/offline/prompt_region_control.json"
N_SENT, N_SEEDS, SEED_BASE = 6, 6, 1000
CONDS = ("clean", "noisy15")
ARMS = ("CPH8", "CPH8SP", "CPQH", "CPQHTP", "STUR2")
YES3, YES4, NO = 0.50, 0.30, 0.10


def with_prompt(payload, source, plen):
    """`payload` with its prompt frames [:plen] taken from `source` (both (1, frames, bands))."""
    out = payload.clone() if hasattr(payload, "clone") else np.array(payload, copy=True)
    out[0, :plen, :] = source[0, :plen, :]
    return out


def reading(diff, yes):
    return "yes" if diff >= yes else ("no" if diff < NO else "partly")


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
                    qh = mc.quantile_payload(s0, t8, plen)
                    payloads = {"CPH8": t8, "CPH8SP": with_prompt(t8, s0, plen), "CPQH": qh,
                                "CPQHTP": with_prompt(qh, t8, plen), "STUR2": s0}
                    outs = {a: r.student(ctx, p, ls.TAUS[1:]) for a, p in payloads.items()}
                    audio = {a: r.e._decode(ctx, x).astype(np.float32) for a, x in outs.items()}
                    jobs = {a: pool.submit(sh.f0_track, audio[a]) for a in ARMS}
                    rec = {"utt_id": row["utt_id"], "condition": cond, "seed": s,
                           "prompt_rms_teacher_minus_draft": float(torch.sqrt(torch.mean((t8 - s0)[0, :plen, :] ** 2)))}
                    for a in ARMS:
                        tracks[a].append(np.asarray(jobs[a].result(), dtype=float))
                        mels[a].append(outs[a][0, plen:, :].numpy().astype(np.float32))
                        rec[f"utmos_{a}"] = float(utmos(torch.from_numpy(audio[a]).unsqueeze(0), 24000))
                    rec["melody_CPH8SP_vs_CPH8"] = cq.melody_agreement(tracks["CPH8SP"][-1], tracks["CPH8"][-1])
                    runs.append(rec)
                    print(f"[{len(runs)}/{N_SENT * len(CONDS) * N_SEEDS}] {row['utt_id'][:22]} {cond:7s} s{s} UTMOS "
                          + " ".join(f"{a} {rec[f'utmos_{a}']:.2f}" for a in ARMS)
                          + f" | prompt diff {rec['prompt_rms_teacher_minus_draft']:.2f} ({time.time() - t0:.0f}s)",
                          flush=True)
                e = {"utt_id": row["utt_id"], "condition": cond}
                for a in ARMS:
                    e[f"cvf0_{a}"], e[f"voiced_{a}"] = sh.cvf0(tracks[a])
                    e[f"cvf0_aligned_{a}"], _, e[f"timing_{a}"] = mc.aligned_cvf0(tracks[a], mels[a])
                per_sentence.append(e)

    def utm(cond, a):
        return st.mean(x[f"utmos_{a}"] for x in runs if x["condition"] == cond)

    q3 = utm("noisy15", "CPH8SP") - utm("noisy15", "CPH8")
    q4 = utm("noisy15", "CPQH") - utm("noisy15", "CPQHTP")
    summary = {
        "Q3_teacher_prompt_frames_carry_copying": {"cph8sp_minus_cph8": q3, "reading": reading(q3, YES3)},
        "Q4_draft_prompt_frames_in_cpqh": {"cpqh_minus_cpqhtp": q4, "reading": reading(q4, YES4)},
        "utmos": {c: {a: utm(c, a) for a in ARMS} for c in CONDS},
        "below_2": {c: {a: sum(x[f"utmos_{a}"] < 2.0 for x in runs if x["condition"] == c) for a in ARMS} for c in CONDS},
        "cvf0_mean": {c: {a: {m: cq.mean_defined([e[f"{m}_{a}"] for e in per_sentence if e["condition"] == c])
                              for m in ("cvf0", "cvf0_aligned")} for a in ARMS} for c in CONDS},
        "voiced_mean": {c: {a: cq.mean_defined([e[f"voiced_{a}"] for e in per_sentence if e["condition"] == c])
                            for a in ARMS} for c in CONDS},
        "melody_CPH8SP_vs_CPH8": {c: cq.mean_defined([x["melody_CPH8SP_vs_CPH8"] for x in runs if x["condition"] == c])
                                  for c in CONDS},
        "prompt_rms_teacher_minus_draft": {c: cq.mean_defined([x["prompt_rms_teacher_minus_draft"] for x in runs
                                                               if x["condition"] == c]) for c in CONDS}}
    OUT.write_text(json.dumps({"kind": "local CPU screen (screening only, not evidence)", "summary": summary,
                               "per_sentence": per_sentence, "runs": runs, "seconds": round(time.time() - t0)},
                              indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(summary, indent=2))
    print(f"[OK] wrote {OUT}")


if __name__ == "__main__":
    main()
