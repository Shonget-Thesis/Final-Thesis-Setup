"""Quick local screen of handover candidates that aim at the paper's 0.25 variety and quality for fewer passes.

SCREENING ONLY, on this computer's CPU: it decides which candidates deserve a Colab T4 run and is not evidence.
Development speakers only (three of EXP-032's tensor-bank speakers, the first six diversity seeds, clean prompts), so
every output can be compared with the paper setting's stored T4 melody for the same seed (`f0_P25` in the bank).

Reference: `W8R2` (teacher to Step 8 of the 16-interval grid, student restarted at 0.2929; 20 passes), which matches the
paper's 14 + 2 setting (30 passes) in quality and roughly in variety (EXP-026 joined to EXP-032).

Known settings, for calibration of this screen:
- `STU` (4), `W8` (21, j = 1), `W4R2` (12; known to lose UTMOS), `W6R2` (16; known to lose some variety).

Candidates (all finish with the student's two steps at 0.2929 and 0.6173):
- `CPH8`, `CPH6` (12, 10 passes): the guided teacher reaches the same time t = 0.2929 on an 8- or 6-interval sway grid
  (4 or 3 Euler steps instead of 8). Known lever (fewer solver steps); new only as a handover rule.
- `GI` (15): 16-interval grid, guidance on Steps 0-3 only, conditional branch alone on Steps 4-8 (guidance interval).
- `SST` (13): student-seeded teacher. The student's one-pass draft S0 replaces teacher Steps 0-3: the teacher starts at
  Step 4 from (1 - t4) * y0 + t4 * S0 and runs Steps 4-8 guided.
- `CWH4` (13): Commitment-Weighted Handover at Step 4. kappa = 1 / (1 + (D / rms(D))^2) per time-frequency cell, with
  D = T4 - T3 (the teacher's own change); the student's first-level draft S1 (from T4) fills what is undecided:
  x = kappa * T4 + (1 - kappa) * S1. No fitted parameter.

Pass rule, fixed before the first run (all against `W8R2` on the same cases):
- melody: share of frames within a semitone of the paper's melody (same seed) >= W8R2's - 0.10;
- variety: CVf0 (mean over speakers) >= W8 + 0.75 * (W8R2 - W8);
- quality: mean UTMOS >= W8R2's - 0.10.
With 3 speakers x 6 seeds these are coarse filters; a pass means "worth a T4 run", nothing more.

Usage: .venv/Scripts/python -I code/analysis/local_candidate_screen.py <bank dir> [--speakers 3] [--seeds 6]
"""
import argparse
import json
import math
import statistics as st
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for p in ("code/colab", "code/scheduling/prototype", "code/analysis"):
    sys.path.insert(0, str(ROOT / p))
import exp032_bank_pitch_screen as bs  # noqa: E402
import run_selective_handover_prelim as sh  # noqa: E402

OUT = ROOT / "research/experiments/EXP-032-variety-per-pass/offline/local_candidate_screen.json"
SPEAKER_ORDER = (4, 3, 1, 0, 2)          # bank speakers, shortest first (indices into the sorted utt ids)
TAUS = (0.07612046748871326, 0.2928932188134524, 0.6173165676349102)
PASSES = {"STU": 4, "W8": 21, "W8R2": 20, "W6R2": 16, "W4R2": 12, "CPH8": 12, "CPH6": 10, "GI": 15, "SST": 13, "CWH4": 13}
KNOWN = ("STU", "W8", "W4R2", "W6R2", "W8R2")
CANDIDATES = ("CPH8", "CPH6", "GI", "SST", "CWH4")
DIV_SEED_BASE = 1000
MELODY_MARGIN, VARIETY_SHARE, UTMOS_MARGIN = 0.10, 0.75, 0.10


def sway(n):
    return [1.0 - math.cos(math.pi / 2.0 * i / n) for i in range(n + 1)]


def _f0(audio):
    return np.asarray(sh.f0_track(audio), dtype=np.float32)


class Runner:
    def __init__(self):
        import engine as en
        self.en = en
        self.e = en.Engine()
        self.torch = self.e.torch
        self.passes = 0

    def vel(self, ctx, x, t, guided=True):
        tt = self.torch.tensor(t, dtype=self.torch.float32)
        kw = dict(x=x, cond=ctx["step_cond"], text=ctx["text_ids"], time=tt, mask=None)
        vc = self.e.tts.teacher(drop_audio_cond=False, drop_text=False, **kw)
        self.passes += 1
        if not guided:
            return vc
        vu = self.e.tts.teacher(drop_audio_cond=True, drop_text=True, **kw)
        self.passes += 1
        return vc + (vc - vu) * 2.0

    def teacher(self, ctx, grid, last, x=None, first=0, guided_until=None, want=()):
        """Euler on `grid` from step `first` to `last`; returns {step: endpoint estimate} for `want` and `last`."""
        x = ctx["y0"].clone() if x is None else x
        ends = {}
        for i in range(first, last + 1):
            v = self.vel(ctx, x, grid[i], guided=guided_until is None or i <= guided_until)
            if i in want or i == last:
                ends[i] = x + (1.0 - grid[i]) * v
            if i < last:
                x = x + (grid[i + 1] - grid[i]) * v
        return ends

    def student(self, ctx, endpoint, taus):
        torch = self.torch
        x = endpoint.clone()
        for tau in taus:
            gen = torch.Generator(device="cpu")
            gen.manual_seed((int(ctx["seed"]) + int(tau * 100)) % (2 ** 31 - 1))
            eps = torch.randn(x.shape, dtype=torch.float32, generator=gen)
            tt = torch.tensor([tau], dtype=torch.float32)
            ts = tt.unsqueeze(-1).unsqueeze(-1)
            phi = (1.0 - ts) * eps + ts * x
            pred = self.e.tts.generator(x=phi, cond=ctx["cond"], text=ctx["text_ids"], time=tt,
                                        drop_audio_cond=False, drop_text=False)
            self.passes += 1
            x = phi + (1.0 - ts) * pred
        return x

    def arms(self, ctx):
        torch = self.torch
        plen = ctx["prompt_len"]
        g16 = sway(16)
        out = {}
        e16 = self.teacher(ctx, g16, 8, want=(3, 4, 6, 8))
        s0 = self.student(ctx, ctx["step_cond"], [0.0])
        out["STU"] = self.student(ctx, s0, TAUS)
        out["W8"] = self.student(ctx, e16[8], TAUS)
        out["W8R2"] = self.student(ctx, e16[8], TAUS[1:])
        out["W6R2"] = self.student(ctx, e16[6], TAUS[1:])
        out["W4R2"] = self.student(ctx, e16[4], TAUS[1:])
        out["CPH8"] = self.student(ctx, self.teacher(ctx, sway(8), 4)[4], TAUS[1:])
        out["CPH6"] = self.student(ctx, self.teacher(ctx, sway(6), 3)[3], TAUS[1:])
        # GI: the shared guided trajectory up to Step 3, then the conditional branch alone
        x = ctx["y0"].clone()
        for i in range(4):
            x = x + (g16[i + 1] - g16[i]) * self.vel(ctx, x, g16[i])
        out["GI"] = self.student(ctx, self.teacher(ctx, g16, 8, x=x, first=4, guided_until=3)[8], TAUS[1:])
        # SST: the student's draft stands in for teacher Steps 0-3
        x4 = (1.0 - g16[4]) * ctx["y0"] + g16[4] * s0
        out["SST"] = self.student(ctx, self.teacher(ctx, g16, 8, x=x4, first=4)[8], TAUS[1:])
        # CWH4: commitment-weighted mix of T4 and the student's first-level draft from T4
        s1 = self.student(ctx, e16[4], TAUS[:1])
        d = (e16[4] - e16[3])[:, plen:, :]
        kappa = 1.0 / (1.0 + (d / d.pow(2).mean().sqrt()) ** 2)
        x = e16[4].clone()
        x[:, plen:, :] = kappa * e16[4][:, plen:, :] + (1.0 - kappa) * s1[:, plen:, :]
        out["CWH4"] = self.student(ctx, x, TAUS[1:])
        info = {"kappa_mean": float(kappa.mean())}
        self.last_t8 = e16[8][0, plen:, :].numpy().astype(np.float32)
        mels = {a: v[0, plen:, :].numpy().astype(np.float32) for a, v in out.items()}
        audio = {a: self.e._decode(ctx, v).astype(np.float32) for a, v in out.items()}
        return mels, audio, info


def main():
    p = argparse.ArgumentParser()
    p.add_argument("bank_root")
    p.add_argument("--speakers", type=int, default=3)
    p.add_argument("--seeds", type=int, default=6)
    p.add_argument("--no-utmos", action="store_true")
    a = p.parse_args()
    bank = Path(a.bank_root) / "bank" / "clean"
    utts = sorted({f.name.split("__s")[0] for f in bank.glob("*.npz")})
    utts = [utts[i] for i in SPEAKER_ORDER[:a.speakers]]
    r = Runner()
    torch = r.torch
    items = {it["utt_id"]: it for it in r.en.development_examples()}
    utmos = None
    if not a.no_utmos:
        utmos = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True).eval()
    arms = KNOWN + CANDIDATES
    per_speaker, cases, check = {}, [], []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=8) as pool, torch.inference_mode():
        for u in utts:
            it = items[u]
            prompt, sr = r.en.load_example(it, "clean")
            tracks = {x: [] for x in arms}
            for s in range(a.seeds):
                d = np.load(bank / f"{u}__s{s:02d}.npz", allow_pickle=False)
                ctx = r.e.prepare(prompt, sr, it["prompt_text"], it["gt_text"], int(it["input_seed"]) + DIV_SEED_BASE + s)
                if ctx["frames"] != d["out_W8"].shape[0]:
                    raise SystemExit(f"length mismatch for {u} seed {s}: {ctx['frames']} vs {d['out_W8'].shape[0]}")
                mels, audio, info = r.arms(ctx)
                # this CPU run against the T4 tensors of the same seed
                # (the teacher's Step 8 estimate; student noise is drawn on the GPU there, so outputs cannot be equal)
                ref = d["T8"].astype(np.float32)
                check.append(float(np.linalg.norm(r.last_t8 - ref) / np.linalg.norm(ref)))
                jobs = {x: pool.submit(_f0, audio[x]) for x in arms}
                row = {"utt_id": u, "seed": s, **info}
                for x in arms:
                    f0 = jobs[x].result()
                    tracks[x].append(f0)
                    row[f"melody_{x}"] = bs.agreement(f0, d["f0_P25"])["within_semitone"]
                    if utmos is not None:
                        row[f"utmos_{x}"] = float(utmos(torch.from_numpy(audio[x]).unsqueeze(0), 24000))
                cases.append(row)
                print(f"[{u[:24]} s{s}] {time.time() - t0:.0f}s, teacher T8 vs T4 bank {check[-1]:.4f}, "
                      f"kappa {info['kappa_mean']:.2f}", flush=True)
            per_speaker[u] = {x: sh.cvf0(tracks[x])[0] for x in arms}

    def mean(vals):
        vals = [v for v in vals if v is not None and v == v]
        return st.mean(vals) if vals else None

    summary = {x: {"passes": PASSES[x], "saved_vs_paper_30": 1 - PASSES[x] / 30,
                   "cvf0": mean(per_speaker[u][x] for u in utts),
                   "melody_vs_paper": mean(c[f"melody_{x}"] for c in cases),
                   "utmos": mean(c.get(f"utmos_{x}") for c in cases)} for x in arms}
    ref, w8 = summary["W8R2"], summary["W8"]
    for x in arms:
        s = summary[x]
        s["pass_melody"] = s["melody_vs_paper"] >= ref["melody_vs_paper"] - MELODY_MARGIN
        s["pass_variety"] = s["cvf0"] >= w8["cvf0"] + VARIETY_SHARE * (ref["cvf0"] - w8["cvf0"])
        s["pass_quality"] = None if s["utmos"] is None else s["utmos"] >= ref["utmos"] - UTMOS_MARGIN
        s["pass"] = bool(s["pass_melody"] and s["pass_variety"] and s["pass_quality"] is not False)
    res = {"kind": "local CPU screen (screening only, not evidence)", "speakers": utts, "seeds": a.seeds,
           "rule": {"melody_margin": MELODY_MARGIN, "variety_share": VARIETY_SHARE, "utmos_margin": UTMOS_MARGIN},
           "cpu_teacher_T8_vs_T4_bank_rel_mel_distance": {"mean": st.mean(check), "max": max(check)},
           "summary": summary, "per_speaker_cvf0": per_speaker, "cases": cases, "seconds": round(time.time() - t0)}
    OUT.write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"[OK] wrote {OUT}")
    for x in arms:
        s = summary[x]
        print(f"{x:5s} {s['passes']:>2} passes | CVf0 {s['cvf0']:.4f} | melody {s['melody_vs_paper']:.3f} | "
              f"UTMOS {s['utmos'] if s['utmos'] is None else round(s['utmos'], 3)} | pass {s['pass']}")


if __name__ == "__main__":
    main()
