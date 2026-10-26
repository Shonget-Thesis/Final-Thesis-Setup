"""Preliminary Colab T4 test of Self-Noise Negative Guidance (SNNG) and the antithetic student pair (EXP-015).

EXP-013 and EXP-014 showed that most noisy Step-4 failures are speaker-similarity failures: the
prompt's noise leaks into the generated voice. Two new methods are tested at the Step-4 handover:

  snng   Self-Noise Negative Guidance. A noise-only prompt is built from the prompt's own quietest
         frames (the 20% lowest-energy frames, tiled over the prompt length). Each teacher step adds
         a third pass with that prompt and steers away from it:
             v = v_c + gamma (v_c - v_u) + eta (v_c - v_noise),   gamma = 2 (unchanged CFG)
         The teacher runs to Step 4 with this velocity; the student completes as usual.
         Cost: 3 passes per teacher step, so Step 4 costs 3*5 + 3 = 18 NFE. eta = 0.5 and 1.0.
  anti   Antithetic student pair. From the plain Step-4 endpoint, the student runs once with its
         usual noise eps and once with -eps; the output closer to the teacher's endpoint (generated
         frames) is kept. Cost 13 + 3 = 16 NFE.
  scg    SNR-Compensated Guidance. Prompt noise shrinks the guidance gap g = v_c - v_u (EXP-012:
         median ||g_4||/||v_c|| 0.236 clean, 0.116 at 15 dB). The CFG scale is raised by the factor the
         prompt SNR predicts: gamma(S) = 2 * clip(G_CLEAN / max(A + B*S, G_FLOOR), 1, r_max), with the
         line A + B*S fitted to g_4 against SNR on the EXP-012 development signals (prompts < 30 dB).
         r_max = 2.5 (scg, full) or 1.5 (scg15, capped). No extra pass: Step 4 costs 13 NFE.
  swc    SNR-Wiener Conditioning. The prompt's noise is removed from the conditioning before the teacher
         and the student see it. Per mel band, the noise floor N_b is the mean linear magnitude of the
         prompt's quietest 20% of frames; every prompt frame gets the gain
             G = clip(1 - beta(S) * N_b / M, G_MIN, 1)
         with beta(S) = beta_max * clip((30 - S) / (30 - 20), 0, 1): no cleaning at 30 dB or above, full
         strength at 20 dB or below. beta_max = 1.0 (swc) or 0.5 (swc05). No extra pass: 13 NFE.
  piec   Prompt-Informed Environment Cancellation. Zero-shot TTS copies the prompt's acoustic environment into
         the output, so the environment is estimated from the prompt itself (per mel band, the mean linear
         magnitude of the prompt's quietest 20% of frames) and cancelled from the GENERATED frames before the
         vocoder: G = clip(1 - beta(S) * N_b / M, 0.1, 1), beta(S) as for SWC. piec4: plain Step 4 output;
         swcpiec4: SWC Step-4 output (clean conditioning, then cancel the residual). No network cost: 13 NFE.
  enh8, piec8  Controls, not candidates: the same cleaning applied to the full Step-8 output. They separate the
         gain from cleaning from the safety of the early handover within the cleaned system.

Data: the EXP-012 Quick set (60 development speakers x 6 conditions); the plain Step-4 and Step-8
labels are committed, so only the new outputs are synthesized and scored.

Subcommands: smoke (2 speakers x clean, gauss20; reproduction checks: SNNG with eta = 0 equals the
plain teacher endpoint, and the stored Step-4 and Step-8 scores are reproduced), run, analyze.
"""
import argparse
import csv
import datetime
import hashlib
import json
import math
import random
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_followup_tests as rf  # noqa: E402
import run_consistency_gate_prelim as cg  # noqa: E402
import run_endpoint_extrapolation_prelim as ex  # noqa: E402

THESIS_ROOT = rf.THESIS_ROOT
N_SPEAKERS, CONDS, NOISY = ex.N_SPEAKERS, ex.CONDS, ex.NOISY
K, NOISE_FRAC = 4, 0.2
ETAS = {"snng05": 0.5, "snng10": 1.0}
SCG_FIT = {"A": -0.0263, "B": 0.0078, "G_CLEAN": 0.236, "G_FLOOR": 0.02}   # EXP-012 signals.csv, prompts < 30 dB (R^2 0.60)
SCG_RMAX = {"scg": 2.5, "scg15": 1.5}
SWC_SNR, SWC_GMIN = (30.0, 20.0), 0.1          # no cleaning at >= 30 dB, full strength at <= 20 dB; gain floor
SWC_BMAX = {"swc": 1.0, "swc05": 0.5}
ENH = {"enh4": ("full", 4), "enh6": ("full", 6), "enhs4": ("snr", 4), "enh8": ("full", 8)}   # wet mix, handover step
PIEC = ("piec4", "swcpiec4", "piec8")
CONTROLS = ("enh8", "piec8")                    # cleaned Step-8 outputs: references for the within-system check
OUTS = tuple(ETAS) + ("anti",) + tuple(SCG_RMAX) + tuple(SWC_BMAX) + tuple(ENH) + PIEC
COST = {4: 13, 8: 21, "snng05": 18, "snng10": 18, "anti": 16, "scg": 13, "scg15": 13, "swc": 13, "swc05": 13,
        "enh4": 13, "enh6": 17, "enhs4": 13, "enh8": 21, "piec4": 13, "swcpiec4": 13, "piec8": 21}
ALPHA, DELTA, FOLDS, SEED, THETA = ex.ALPHA, ex.DELTA, ex.FOLDS, ex.SEED, ex.THETA
THETA_GRID = ex.GRIDS["theta"]                                   # 40 -> 20 dB
LOW_GRID = [float(x) for x in range(40, 9, -1)]                  # theta_lo candidates, 40 -> 10 dB
HEAVY_GRID = [float(x) for x in range(10, 41)]                   # theta_e candidates, 10 -> 40 dB (method below theta_e)
FIELDS = ["utt_id", "speaker_id", "condition", "prompt_snr", "d_4", "d4_stored", "noise_frames", "anti_pick"] + \
         [f"gamma_{o}" for o in SCG_RMAX] + [f"beta_{o}" for o in SWC_BMAX] + ["wet_enhs4", "snr_after_enh", "snr_after_enhs", "enh_ms", "beta_piec"] + \
         [f"{m}_{o}" for o in OUTS for m in ("wer", "similarity", "utmos", "is_failure", "fail_wer", "fail_sim", "hypothesis")] + \
         ["wer_ref", "similarity_ref", "utc"]


# --------------------------------------------------------------------------- operators
def noise_prompt(step_cond, plen, frac=NOISE_FRAC):
    """Copy of step_cond whose prompt frames are the prompt's quietest frames, tiled; works on torch or numpy."""
    p = step_cond[0, :plen, :]
    energy = [float(p[i].mean()) for i in range(plen)]
    n = max(1, int(frac * plen))
    quiet = sorted(range(plen), key=lambda i: energy[i])[:n]
    nc = step_cond.clone() if hasattr(step_cond, "clone") else step_cond.copy()
    nc[0, :plen, :] = p[[quiet[i % n] for i in range(plen)]]
    return nc, n


def scg_gamma(snr, r_max, base=2.0, fit=SCG_FIT):
    """SNR-compensated CFG scale: base * clip(G_CLEAN / max(A + B*snr, G_FLOOR), 1, r_max)."""
    g_hat = max(fit["A"] + fit["B"] * snr, fit["G_FLOOR"])
    return base * min(max(fit["G_CLEAN"] / g_hat, 1.0), r_max)


def swc_beta(snr, beta_max, band=SWC_SNR):
    """Cleaning strength: 0 at or above band[0] dB, beta_max at or below band[1] dB, linear in between."""
    return beta_max * min(max((band[0] - snr) / (band[0] - band[1]), 0.0), 1.0)


def swc_clean(cond, plen, beta, scale=1.0, frac=NOISE_FRAC, gmin=SWC_GMIN):
    """Spectral-subtraction gain on the prompt frames of a (log-mel / scale) conditioning tensor; torch or numpy.
    Noise per band = mean linear magnitude of the quietest frames; G = clip(1 - beta * N / M, gmin, 1)."""
    out = cond.clone() if hasattr(cond, "clone") else cond.copy()
    if beta <= 0:
        return out
    noise, exp_, log_ = _prompt_noise_floor(cond, plen, scale, frac)
    out[0, :plen, :] = _subtract(exp_(cond[0, :plen, :] * scale), noise, beta, gmin, log_) / scale
    return out


def _prompt_noise_floor(cond, plen, scale, frac):
    """Mean linear magnitude per band of the quietest prompt frames, and the matching exp/log (torch or numpy)."""
    is_torch = hasattr(cond, "clone")
    exp_, log_ = ((lambda a: a.exp()), (lambda a: a.log())) if is_torch else (__import__("numpy").exp, __import__("numpy").log)
    p = cond[0, :plen, :]
    energy = [float(p[i].mean()) for i in range(plen)]
    quiet = sorted(range(plen), key=lambda i: energy[i])[:max(1, int(frac * plen))]
    return exp_(p * scale)[quiet].mean(0), exp_, log_


def _subtract(lin, noise, beta, gmin, log_):
    gain = 1.0 - beta * noise / lin
    gain = gain.clamp(gmin, 1.0) if hasattr(gain, "clamp") else gain.clip(gmin, 1.0)
    return log_(lin * gain)


def piec_cancel(x, cond, plen, beta, scale=1.0, frac=NOISE_FRAC, gmin=SWC_GMIN):
    """Prompt-Informed Environment Cancellation on a generated mel state x (log-mel / scale; torch or numpy):
    the environment floor is estimated from the quietest prompt frames of `cond` and cancelled from x's generated
    frames only; prompt frames are left unchanged."""
    out = x.clone() if hasattr(x, "clone") else x.copy()
    if beta <= 0:
        return out
    noise, exp_, log_ = _prompt_noise_floor(cond, plen, scale, frac)
    out[0, plen:, :] = _subtract(exp_(x[0, plen:, :] * scale), noise, beta, gmin, log_) / scale
    return out


def guided_endpoints(pipe, ctx, specs, k=K):
    """Teacher Euler passes to step k, one per spec {name: (gamma, eta, step_cond or None)}:
    v = v_c + gamma (v_c - v_u) + eta (v_c - v_noise), the noise-only prompt used only when eta > 0, and the
    spec's own conditioning (for SWC) in place of ctx["step_cond"] when given.
    Returns ({name: x1hat(t_k)}, number of noise frames)."""
    torch = pipe.torch
    grid = pipe.sampler.SWAY_TEACHER_GRID
    ncond, n = noise_prompt(ctx["step_cond"], ctx["prompt_len"])
    out = {}
    with torch.inference_mode():
        for name, (gamma, eta, sc) in specs.items():
            sc = ctx["step_cond"] if sc is None else sc
            x = ctx["y0"].clone()
            for i in range(k + 1):
                t = torch.tensor(grid[i], device=pipe.device, dtype=torch.float32)
                kw = dict(x=x, text=ctx["text_ids"], time=t, mask=None)
                vc = pipe.tts.teacher(cond=sc, drop_audio_cond=False, drop_text=False, **kw)
                vu = pipe.tts.teacher(cond=sc, drop_audio_cond=True, drop_text=True, **kw)
                v = vc + (vc - vu) * gamma
                if eta:
                    vn = pipe.tts.teacher(cond=ncond, drop_audio_cond=False, drop_text=False, **kw)
                    v = v + eta * (vc - vn)
                ep = x + (1.0 - grid[i]) * v
                if i < k:
                    x = x + (grid[i + 1] - grid[i]) * v
            out[name] = ep
    return out, n


def student_x(pipe, ctx, endpoint, sign=1.0):
    """run_student_completion's loop (same seeds), with the student noise multiplied by sign; returns the final mel state."""
    torch, sampler = pipe.torch, pipe.sampler
    x = endpoint.clone()
    with torch.inference_mode():
        for tau in sampler.STUDENT_TAUS:
            gen = sampler.make_local_generator((int(ctx["seed"]) + int(tau * 100)) % (2 ** 31 - 1), device=pipe.device)
            eps = sign * torch.randn(x.shape, dtype=torch.float32, device=pipe.device, generator=gen)
            tt = torch.tensor([tau], device=pipe.device, dtype=torch.float32)
            ts = tt.unsqueeze(-1).unsqueeze(-1)
            phi = (1.0 - ts) * eps + ts * x
            pred = pipe.tts.generator(x=phi, cond=ctx["cond"], text=ctx["text_ids"], time=tt, drop_audio_cond=False, drop_text=False)
            x = phi + (1.0 - ts) * pred
    return x


def decode(pipe, ctx, x):
    with pipe.torch.inference_mode():
        mel = x.permute(0, 2, 1) * getattr(pipe.tts, "scale", 1.0)
        return pipe.tts.vocos.decode(mel[..., ctx["prompt_len"]:]).cpu().numpy().squeeze()


class Enhancer:
    """Pretrained DNS64 Demucs speech enhancer (Defossez et al., 2020; pip package `denoiser`), 16 kHz.
    __call__(wav, sr, wet) returns wet * enhanced + (1 - wet) * wav at the input sample rate."""

    def __init__(self, torch, torchaudio, device):
        from denoiser import pretrained
        self.torch, self.F, self.device = torch, torchaudio.functional, device
        self.model = pretrained.dns64().to(device).eval()
        self.sr = self.model.sample_rate

    def __call__(self, wav, sr, wet=1.0):
        if wet <= 0:
            return wav
        torch = self.torch
        x = torch.from_numpy(wav).float().to(self.device)[None, None]
        x16 = self.F.resample(x, sr, self.sr) if sr != self.sr else x
        with torch.inference_mode():
            y = self.model(x16)
        y = self.F.resample(y, self.sr, sr) if sr != self.sr else y
        y = y[..., :x.shape[-1]]
        y = torch.nn.functional.pad(y, (0, x.shape[-1] - y.shape[-1]))
        return (wet * y + (1.0 - wet) * x)[0, 0].cpu().numpy().astype(wav.dtype)


def enh_wet(snr, mode):
    """Wet mix of the enhanced prompt: 1 for "full"; for "snr", 0 at 30 dB or above and 1 at 20 dB or below."""
    return 1.0 if mode == "full" else swc_beta(snr, 1.0)


def antithetic_pick(xp, xm, anchor, plen):
    """+1 if the eps output is at least as close to the teacher endpoint (generated frames) as the -eps output, else -1."""
    g = lambda a: a[:, plen:, :]
    return 1 if ex._norm(g(xp - anchor)) <= ex._norm(g(xm - anchor)) else -1


# -------------------------------------------------------------------------------- runs
def run_cases(pipe, todo_items, conds, out_dir, enhancer, csv_name="snng.csv", reproduce=False):
    path = Path(out_dir) / csv_name
    done = rf.done_keys(path, 1)
    stored = ex.stored_cases()
    todo = [(it, c) for it in todo_items for c in conds if (it["utt_id"], c) not in done]
    print(f"[snng] {len(todo)} cases to run ({len(done)} already in {path.name}) on {pipe.gpu}", flush=True)
    repro = []
    for n, (item, cond) in enumerate(todo, 1):
        s = stored[(item["utt_id"], cond)]
        ref = s[8]
        ctx = pipe.prepare(item, cond, item["input_seed"], out_dir)
        E, drift = pipe.teacher(ctx, tuple(range(9)))          # one pass to Step 8: Steps 0-4 as before, Step 8 for the PIEC control
        plen = ctx["prompt_len"]
        cfg = pipe.sampler.CFG_STRENGTH
        gammas = {o: scg_gamma(ctx["prompt_snr"], r, base=cfg) for o, r in SCG_RMAX.items()}
        betas = {o: swc_beta(ctx["prompt_snr"], b) for o, b in SWC_BMAX.items()}
        scale = getattr(pipe.tts, "scale", 1.0)
        with pipe.torch.inference_mode():
            clean = {o: (swc_clean(ctx["step_cond"], plen, b, scale), swc_clean(ctx["cond"], plen, b, scale)) for o, b in betas.items()}
        specs = {**{o: (cfg, eta, None) for o, eta in ETAS.items()}, **{o: (g, 0.0, None) for o, g in gammas.items()},
                 **{o: (cfg, 0.0, clean[o][0]) for o in SWC_BMAX}}
        if reproduce:
            specs["plain"] = (cfg, 0.0, None)
        ends, nframes = guided_endpoints(pipe, ctx, specs)
        xp, xm = student_x(pipe, ctx, E[K], 1.0), student_x(pipe, ctx, E[K], -1.0)
        with pipe.torch.inference_mode():
            pick = antithetic_pick(xp, xm, E[K], plen)
        row = {"utt_id": item["utt_id"], "speaker_id": item["speaker_id"], "condition": cond, "prompt_snr": ctx["prompt_snr"],
               "d_4": drift[4], "d4_stored": float(ref["d_4"]), "noise_frames": nframes, "anti_pick": pick,
               **{f"gamma_{o}": g for o, g in gammas.items()}, **{f"beta_{o}": b for o, b in betas.items()},
               "wer_ref": float(ref["wer_ref"]), "similarity_ref": float(ref["similarity_ref"])}
        gt = pipe.ground_truth(item)
        audios = {o: decode(pipe, ctx, student_x(pipe, ctx, ends[o])) for o in tuple(ETAS) + tuple(SCG_RMAX)}
        xs = {o: student_x(pipe, dict(ctx, cond=clean[o][1]), ends[o]) for o in SWC_BMAX}
        audios.update({o: decode(pipe, ctx, x) for o, x in xs.items()})
        if pick < 0:
            audios["anti"] = decode(pipe, ctx, xm)
        timing = {}

        def enhance(wet):
            def transform(wav, sr):
                t0 = time.perf_counter()
                out = enhancer(wav, sr, wet)
                timing.setdefault("ms", (time.perf_counter() - t0) * 1000.0)
                return out
            return transform

        snr_of = lambda c: pipe.sampler.estimate_prompt_snr(c["prompt"], sample_rate=c["sr"])
        ctx_e = pipe.prepare(item, cond, item["input_seed"], out_dir, prompt_transform=enhance(1.0))
        Ee, _ = pipe.teacher(ctx_e, (4, 6, 8))
        audios.update({o: decode(pipe, ctx_e, student_x(pipe, ctx_e, Ee[step])) for o, (mode, step) in ENH.items() if mode == "full"})
        wet = enh_wet(ctx["prompt_snr"], "snr")
        row.update({"wet_enhs4": wet, "snr_after_enh": snr_of(ctx_e)})
        if wet > 0:
            ctx_s = pipe.prepare(item, cond, item["input_seed"], out_dir, prompt_transform=enhance(wet))
            Es, _ = pipe.teacher(ctx_s, (4,))
            audios["enhs4"] = decode(pipe, ctx_s, student_x(pipe, ctx_s, Es[4]))
            row["snr_after_enhs"] = snr_of(ctx_s)
        row["enh_ms"] = timing.get("ms")
        b_p = swc_beta(ctx["prompt_snr"], 1.0)
        row["beta_piec"] = b_p
        if b_p > 0:
            src = {"piec4": xp, "swcpiec4": xs["swc"], "piec8": student_x(pipe, ctx, E[8])}
            with pipe.torch.inference_mode():
                audios.update({o: decode(pipe, ctx, piec_cancel(x, ctx["cond"], plen, b_p, scale)) for o, x in src.items()})
        same = {}                                           # outputs identical to one already scored or stored
        if pick > 0:
            same["anti"] = 4
        if wet <= 0:
            same["enhs4"] = 4
        if b_p <= 0:
            same.update({"piec4": 4, "swcpiec4": "swc", "piec8": 8})
        scored = {}
        for o in OUTS:
            k = same.get(o)
            if k in (4, 8):
                m = {"wer": float(s[k]["wer"]), "similarity": float(s[k]["similarity"]), "utmos": float(s[k]["utmos"]),
                     "hypothesis": s[k]["hypothesis"]}
            elif k is not None:
                m = scored[k]
            else:
                m = pipe.score(audios[o], item, gt)
            scored[o] = m
            fail, diag = pipe.check_failure(m["wer"], row["wer_ref"], m["similarity"], row["similarity_ref"], config=pipe.cfg)
            row.update({f"wer_{o}": m["wer"], f"similarity_{o}": m["similarity"], f"utmos_{o}": m["utmos"], f"is_failure_{o}": bool(fail),
                        f"fail_wer_{o}": bool(diag["wer_failed"]), f"fail_sim_{o}": bool(diag["sim_failed"]), f"hypothesis_{o}": m["hypothesis"]})
        if reproduce:
            with pipe.torch.inference_mode():
                diff = float((ends["plain"] - E[K]).abs().max())
            repro.append({"utt_id": item["utt_id"], "condition": cond, "check": "guided loop (gamma=2, eta=0) vs plain teacher endpoint", "max_abs_diff": diff})
            for k, T in ((4, E[K]), (8, E[8])):
                m = pipe.score(decode(pipe, ctx, student_x(pipe, ctx, T)), item, gt)
                repro.append({"utt_id": item["utt_id"], "condition": cond, "step": k, "wer": m["wer"], "wer_stored": float(s[k]["wer"]),
                              "similarity": m["similarity"], "similarity_stored": float(s[k]["similarity"])})
        row["utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        new = not path.exists()
        with open(path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            if new:
                w.writeheader()
            w.writerow(row)
            f.flush()
        print(f"[{n}/{len(todo)}] snng: {item['utt_id']} {cond}", flush=True)
    print(f"[OK] snng complete: {path}", flush=True)
    return repro


# ----------------------------------------------------------------------------- analysis
def build_table(stored, rows, signals=None):
    """signals: (utt, cond) -> EXP-012 teacher-pass row, for the drift profile d_1..d_6 used by TAS."""
    table = []
    for r in rows:
        s = stored.get((r["utt_id"], r["condition"]))
        if s is not None:
            g = (signals or {}).get((r["utt_id"], r["condition"]))
            table.append({"utt": r["utt_id"], "spk": r["speaker_id"], "cond": r["condition"], "snr": float(r["prompt_snr"]),
                          "d4": float(r["d_4"]), "d4_stored": float(r["d4_stored"]),
                          "ld": [math.log(float(g[f"d_{i}"])) for i in range(1, 7)] if g else None,
                          "fail": {4: rf.is_true(s[4]["is_failure"]), 8: False, **{o: rf.is_true(r[f"is_failure_{o}"]) for o in OUTS}},
                          "sim": {4: float(s[4]["similarity"]), 8: float(s[8]["similarity"]), **{o: float(r[f"similarity_{o}"]) for o in OUTS}},
                          "wer": {4: float(s[4]["wer"]), 8: float(s[8]["wer"]), **{o: float(r[f"wer_{o}"]) for o in OUTS}}})
    return table


def route(hi, lo, o):
    """Plain Step 4 if SNR >= hi; the new output o if lo <= SNR < hi; Step 8 otherwise (None = never)."""
    hi = float("inf") if hi is None else hi
    lo = float("inf") if lo is None else lo
    return lambda r: 4 if r["snr"] >= hi else (o if r["snr"] >= lo else 8)


def certify(train, pol, stratified):
    by = {}
    for r in train:
        d = by.setdefault(r["cond"], [0, 0])
        d[0] += int(r["fail"][pol(r)])
        d[1] += 1
    ps = [rf.binom_cdf(k, n, ALPHA) for k, n in by.values()] if stratified else \
         [rf.binom_cdf(sum(k for k, _ in by.values()), sum(n for _, n in by.values()), ALPHA)]
    return max(ps) <= DELTA


def fixed_sequence(train, grid, make, stratified):
    chosen = None
    for p in grid:
        if not certify(train, make(p), stratified):
            break
        chosen = p
    return chosen


def heavy(hi, e, o):
    """Plain Step 4 if SNR >= hi; the method o if SNR < e (heavy noise); Step 8 in between (None = never)."""
    hi = float("inf") if hi is None else hi
    e = float("-inf") if e is None else e
    return lambda r: 4 if r["snr"] >= hi else (o if r["snr"] < e else 8)


def tas_fit(train):
    """Trajectory Anomaly Score, fitted on the clean-prompt training cases only (no failure labels): squared Mahalanobis
    distance of the drift profile log(d_1..d_6) from the clean mean, counted only when the profile lies below that mean
    (the trajectory is too still: premature stabilization); 0 otherwise."""
    import numpy as np
    X = np.array([r["ld"] for r in train if r["cond"] == "clean" and r["ld"] is not None])
    mu = X.mean(0)
    Ci = np.linalg.inv(np.cov(X, rowvar=False) + 1e-3 * np.eye(X.shape[1]))

    def score(r):
        if r["ld"] is None:
            return float("inf")
        z = np.asarray(r["ld"]) - mu
        return float(z @ Ci @ z) if z.mean() < 0 else 0.0
    return score


def gated(score, tau, o):
    """The SNR-only gate plus a detector: output o if SNR >= 20 dB and score <= tau; Step 8 otherwise (None = never)."""
    return lambda r: o if (tau is not None and r["snr"] >= THETA and score(r) <= tau) else 8


POLICIES = ["reference", "snr_only", "snr_theta", "d4veto", "tas"] + \
           [f"{o}_{k}" for o in OUTS for k in ("route", "heavy", "tas", "noisy", "all")]


def fold_policy(name, train, stratified):
    """Returns (policy, chosen parameters) fitted on the training fold."""
    if name == "reference":
        return (lambda r: 8), None
    if name == "snr_only":
        return (lambda r: 4 if r["snr"] >= THETA else 8), None
    if name in ("d4veto", "tas") or name.endswith("_tas"):
        early = 4 if name in ("d4veto", "tas") else name.rsplit("_", 1)[0]
        score = (lambda r: -r["d4"]) if name == "d4veto" else tas_fit(train)
        grid = sorted({score(r) for r in train if r["snr"] >= THETA})              # conservative -> aggressive
        tau = fixed_sequence(train, grid, lambda t: gated(score, t, early), stratified)
        return gated(score, tau, early), tau
    hi = fixed_sequence(train, THETA_GRID, lambda p: route(p, None, None), stratified)
    if name == "snr_theta":
        return route(hi, None, None), hi
    o, kind = name.rsplit("_", 1)
    if kind == "all":
        return (lambda r: o), None
    if kind == "noisy":
        return route(hi, float("-inf"), o), (hi, "all below")
    top = float("inf") if hi is None else hi
    if kind == "heavy":
        e = fixed_sequence(train, [x for x in HEAVY_GRID if x <= top], lambda p: heavy(hi, p, o), stratified)
        return heavy(hi, e, o), (hi, e)
    lo = fixed_sequence(train, [x for x in LOW_GRID if x < top], lambda p: route(hi, p, o), stratified)
    return route(hi, lo, o), (hi, lo)


def cross_validate(table):
    spks = sorted({r["spk"] for r in table})
    random.Random(SEED).shuffle(spks)
    fold = {s: i % FOLDS for i, s in enumerate(spks)}
    res, chosen = {}, {}
    for tag, strat in (("pooled", False), ("stratified", True)):
        res[tag], chosen[tag], vecs = {}, {}, {}
        for name in POLICIES:
            oof, ch = [], []
            for f in range(FOLDS):
                pol, p = fold_policy(name, [r for r in table if fold[r["spk"]] != f], strat)
                ch.append(p)
                oof += [(r, pol(r)) for r in table if fold[r["spk"]] == f]
            per = {c: sum(int(r["fail"][k]) for r, k in oof if r["cond"] == c) for c in CONDS}
            res[tag][name] = {"failures": sum(per.values()), "n": len(oof), "mean_nfe": sum(COST[k] for _, k in oof) / len(oof),
                              "worst_condition_risk": max(per[c] / max(1, sum(r["cond"] == c for r, _ in oof)) for c in CONDS),
                              "per_condition": per}
            chosen[tag][name] = ch
            failed = {id(r): bool(r["fail"][k]) for r, k in oof}
            vecs[name] = [failed[id(r)] for r in table]
        for name in POLICIES:                     # paired against SNR-only on the same cases
            x, y, pv = rf.mcnemar_exact(vecs["snr_only"], vecs[name])
            res[tag][name]["vs_snr_only"] = {"fixed": x, "broken": y, "p": pv}
    return res, chosen


def quality_vs_reference(table, o, conds):
    """Paired mean differences of the new output minus Step 8 (SIM higher is better, WER lower is better)."""
    rows = [r for r in table if r["cond"] in conds]
    ds = [r["sim"][o] - r["sim"][8] for r in rows]
    dw = [r["wer"][o] - r["wer"][8] for r in rows]
    out = {"n": len(rows), "sim_mean_diff": statistics.mean(ds) if ds else None, "wer_mean_diff": statistics.mean(dw) if dw else None,
           "share_sim_better": sum(d > 0 for d in ds) / len(ds) if ds else None}
    try:
        from scipy.stats import wilcoxon
        out["sim_wilcoxon_p"] = float(wilcoxon(ds).pvalue) if any(ds) else 1.0
        out["wer_wilcoxon_p"] = float(wilcoxon(dw).pvalue) if any(dw) else 1.0
    except ImportError:
        pass
    return out


def within(table, early, late):
    """Failures of the early output judged against the cleaned Step-8 output (Definition 3.1 thresholds), per condition."""
    th = json.loads((THESIS_ROOT / "code/scheduling/failure_config.json").read_text(encoding="utf-8"))
    bad = lambda r: r["wer"][early] > max(th["absolute_wer_ceiling"], r["wer"][late] + th["relative_wer_margin"]) or \
        r["sim"][early] < r["sim"][late] - th["similarity_margin"]
    return {c: sum(bad(r) for r in table if r["cond"] == c) for c in CONDS}


def analyze(out, allow_partial=False):
    out = Path(out)
    prereg = json.loads((out / "preregistration.json").read_text(encoding="utf-8"))
    rows = rf.read_csv(out / "snng.csv", repair=True)
    signals = {(r["utt_id"], r["condition"]): r for r in rf.read_csv(ex.EXP012 / "signals.csv")}
    table = build_table(ex.stored_cases(), rows, signals)
    expected = N_SPEAKERS * len(CONDS)
    if len(table) < expected and not allow_partial:
        raise SystemExit(f"{len(table)} of {expected} cases; finish the run or pass --allow_partial")
    res = {"preregistration_sha256": prereg["sha256"], "cases": len(table), "expected_cases": expected, "partial": len(table) < expected,
           "drift_reproduction_max_abs_diff": max((abs(r["d4"] - r["d4_stored"]) for r in table), default=None),
           "antithetic_minus_eps_chosen": sum(int(float(r["anti_pick"])) < 0 for r in rows),
           "step4_failures": {c: {"n": sum(r["cond"] == c for r in table),
                                  **{str(k): sum(r["fail"][k] for r in table if r["cond"] == c) for k in (4,) + OUTS}} for c in CONDS},
           "paired_tests": {f"{o} vs Step 4, {g}": ex.paired(table, 4, o, cs) for o in OUTS
                            for g, cs in (("noisy (primary)", NOISY), ("clean", ("clean",)))},
           "quality_vs_step8": {o: {"noisy": quality_vs_reference(table, o, NOISY), "clean": quality_vs_reference(table, o, ("clean",))}
                                for o in OUTS}}
    if len({r["spk"] for r in table}) >= FOLDS:
        res["cross_validation"], res["chosen"] = cross_validate(table)
    res["within_cleaned_system"] = {f"{a} judged against {b}": within(table, a, b) for a, b in (("enh4", "enh8"), ("piec4", "piec8"))}
    res["criteria"] = {}
    for o in OUTS:
        a, b = res["paired_tests"][f"{o} vs Step 4, noisy (primary)"], res["paired_tests"][f"{o} vs Step 4, clean"]
        crit = {"C1": a[f"failures_{o}"] <= 0.75 * a["failures_4"] and a["p"] < 0.05 and a[f"failures_{o}"] < a["failures_4"],
                "C2": b[f"failures_{o}"] <= b["failures_4"] + 1}
        if "cross_validation" in res:
            y = res["cross_validation"]["pooled"]["snr_theta"]
            crit["C3"] = any(x["mean_nfe"] < y["mean_nfe"] and x["failures"] <= y["failures"]
                             for x in (res["cross_validation"]["pooled"][f"{o}_{k}"] for k in ("route", "heavy")))
        res["criteria"][o] = crit
    (out / "snng_results.json").write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
    (out / "NOISE_GUIDANCE_REPORT.md").write_text(render(res), encoding="utf-8")
    print(f"Wrote {out / 'NOISE_GUIDANCE_REPORT.md'}", flush=True)
    return res


def render(res):
    names = {"snng05": "SNNG η = 0.5 (18 NFE)", "snng10": "SNNG η = 1.0 (18 NFE)", "anti": "Antithetic pair (16 NFE)",
             "scg": "SNR-Compensated Guidance, r_max 2.5 (13 NFE)", "scg15": "SNR-Compensated Guidance, r_max 1.5 (13 NFE)",
             "swc": "SNR-Wiener Conditioning, β_max 1.0 (13 NFE)", "swc05": "SNR-Wiener Conditioning, β_max 0.5 (13 NFE)",
             "enh4": "Enhanced prompt, Step 4 (13 NFE)", "enh6": "Enhanced prompt, Step 6 (17 NFE)",
             "enhs4": "SNR-scheduled enhanced prompt, Step 4 (13 NFE)", "enh8": "Enhanced prompt, Step 8 (control, 21 NFE)",
             "piec4": "PIEC on Step 4 (13 NFE)", "swcpiec4": "SWC + PIEC, Step 4 (13 NFE)", "piec8": "PIEC on Step 8 (control, 21 NFE)"}
    d = res["drift_reproduction_max_abs_diff"]
    f4 = lambda x: "n/a" if x is None else f"{x:+.4f}"
    L = ["# Preliminary test: noise-aware handover methods (PIEC, prompt enhancement, SWC, SCG, SNNG, antithetic pair)", "",
         f"{'**PARTIAL.** ' if res['partial'] else ''}Development speakers only ({N_SPEAKERS} speakers x {len(CONDS)} conditions: {', '.join(CONDS)}); "
         f"{res['cases']} of {res['expected_cases']} cases; preregistration `{res['preregistration_sha256'][:16]}…`. "
         "Every number is computed from the CSV files by `run_noise_guidance_prelim.py analyze`.", "",
         f"Drift reproduction: max |Δd₄| = {'n/a' if d is None else f'{d:.3f}'}. Antithetic pair chose −ε in {res['antithetic_minus_eps_chosen']} cases.", "",
         "## Preregistered criteria (C1 noisy failures −25% with p < 0.05; C2 clean +1 at most; C3 the calibrated route beats the calibrated SNR gate)", "",
         "| Method | C1 | C2 | C3 |", "|---|---|---|---|"]
    for o, c in res["criteria"].items():
        L.append(f"| {names[o]} | " + " | ".join("PASS" if c.get(k) else ("FAIL" if k in c else "n/a") for k in ("C1", "C2", "C3")) + " |")
    L += ["", "## Failures when every input uses the method at Step 4", "", "| Condition | n | plain Step 4 | " + " | ".join(OUTS) + " |",
          "|---|---:|---:|" + "---:|" * len(OUTS)]
    for c, v in res["step4_failures"].items():
        L.append(f"| {c} | {v['n']} | {v['4']} | " + " | ".join(str(v[o]) for o in OUTS) + " |")
    L += ["", "Exact McNemar (two-sided), unadjusted:", ""]
    for name, t in res["paired_tests"].items():
        ks = [k for k in t if k.startswith("failures_")]
        L.append(f"- {name}: {t[ks[0]]} → {t[ks[1]]} of {t['n']} (p {t['p']:.4f})")
    if "cross_validation" in res:
        L += ["", "## Against SNR-only (pooled calibration, out of fold; exact McNemar on the same cases)", "",
              "SNR-only = Step 4 if SNR ≥ 20 dB, else Step 8. Within limit = out-of-fold risk ≤ 5%.", "",
              "| Policy | Failures | Risk | Within limit | Mean NFE | Fixed vs SNR-only | Broken | p |", "|---|---:|---:|---|---:|---:|---:|---:|"]
        for n, v in res["cross_validation"]["pooled"].items():
            c = v["vs_snr_only"]
            L.append(f"| {n} | {v['failures']}/{v['n']} | {v['failures'] / v['n']:.1%} | {'yes' if v['failures'] / v['n'] <= ALPHA else 'no'} | "
                     f"{v['mean_nfe']:.2f} | {c['fixed']} | {c['broken']} | {c['p']:.2g} |")
    L += ["", "## Early handover within the cleaned system", "",
          "The early cleaned output judged against the cleaned Step-8 output (Definition 3.1 thresholds). Low counts mean the early "
          "handover is safe within the cleaned system, not only better than the uncleaned reference.", ""]
    for name, per in res["within_cleaned_system"].items():
        L.append(f"- {name}: {sum(per.values())} failures ({', '.join(f'{c} {v}' for c, v in per.items())})")
    L += ["", "## Average quality vs the Step-8 reference (paired; SIM higher is better, WER lower is better)", "",
          "| Method | Prompts | n | SIM mean diff | SIM better | SIM p | WER mean diff | WER p |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for o, q in res["quality_vs_step8"].items():
        for g, v in q.items():
            p1, p2, share = v.get("sim_wilcoxon_p"), v.get("wer_wilcoxon_p"), v["share_sim_better"]
            L.append(f"| {o} | {g} | {v['n']} | {f4(v['sim_mean_diff'])} | {'n/a' if share is None else f'{share:.0%}'} | "
                     f"{'n/a' if p1 is None else f'{p1:.2g}'} | {f4(v['wer_mean_diff'])} | {'n/a' if p2 is None else f'{p2:.2g}'} |")
    for tag in ("pooled", "stratified"):
        if "cross_validation" not in res:
            break
        L += ["", f"## Out-of-fold policies, {tag} calibration (5-fold speaker CV, α = {ALPHA}, δ = {DELTA})", "",
              "| Policy | Failures | Mean NFE | Worst-condition risk | " + " | ".join(CONDS) + " |", "|---|---:|---:|---:|" + "---:|" * len(CONDS)]
        for n, v in res["cross_validation"][tag].items():
            L.append(f"| {n} | {v['failures']}/{v['n']} | {v['mean_nfe']:.2f} | {v['worst_condition_risk']:.1%} | "
                     + " | ".join(str(v["per_condition"][c]) for c in CONDS) + " |")
        L += ["", "Chosen per fold: " + "; ".join(f"{n}: {v}" for n, v in res["chosen"][tag].items() if v and any(x is not None for x in v))]
    L += ["", "Policies: snr_theta = plain Step 4 if SNR ≥ θ (calibrated), else Step 8. <m>_route = plain Step 4 if SNR ≥ θ_hi (calibrated as snr_theta), "
          "the method if θ_lo ≤ SNR < θ_hi (θ_lo calibrated next on the same fold), else Step 8. <m>_heavy = plain Step 4 if SNR ≥ θ_hi, "
          "the method if SNR < θ_e (heavy noise; θ_e calibrated next), else Step 8. d4veto / tas = plain Step 4 if SNR ≥ 20 dB and the "
          "detector (−d₄, or the Trajectory Anomaly Score of log d₁..d₆ fitted on clean training prompts) is below its certified threshold, "
          "else Step 8; <m>_tas = the same gate with the method at the handover. <m>_noisy = the method for every input below θ_hi. "
          "<m>_all = the method for every input.", "", "Development evidence only; a claim needs a held-out test with the protocol frozen first."]
    return "\n".join(L) + "\n"


def write_prereg(out, args):
    path = Path(out) / "preregistration.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    pre = {"purpose": "Preliminary development test of Self-Noise Negative Guidance (SNNG), SNR-Compensated Guidance (SCG), SNR-Wiener Conditioning (SWC) and the antithetic student pair at the Step-4 handover",
           "methods": {"snng": f"v = v_c + gamma (v_c - v_u) + eta (v_c - v_noise); noise-only prompt = the {int(NOISE_FRAC * 100)}% quietest prompt frames tiled; "
                               "teacher to Step 4 with this velocity, then the usual student", "etas": ETAS,
                       "anti": "student with eps and with -eps from the plain Step-4 endpoint; keep the output closer to the endpoint on generated frames",
                       "scg": "teacher to Step 4 with CFG scale gamma(S) = 2 * clip(G_CLEAN / max(A + B*S, G_FLOOR), 1, r_max); "
                              "A, B fitted to g_4 against prompt SNR on the EXP-012 development signals (prompts < 30 dB, the same 60 speakers "
                              "evaluated here; the fit uses no failure labels)", "scg_fit": SCG_FIT, "scg_rmax": SCG_RMAX,
                       "swc": "prompt frames of the teacher's and the student's conditioning multiplied (linear mel magnitude) by "
                              "G = clip(1 - beta(S) N_b / M, G_MIN, 1), N_b = mean of the quietest 20% prompt frames per band; "
                              "beta(S) = beta_max * clip((30 - S) / 10, 0, 1)", "swc_beta_max": SWC_BMAX, "swc_gain_floor": SWC_GMIN,
                       "enh": "prompt cleaned by the pretrained DNS64 Demucs enhancer (pip denoiser 0.1.5) before conditioning, mixed wet*enhanced + (1-wet)*noisy; enh4/enh6: wet 1 at Steps 4/6; enhs4: wet = clip((30 - S)/10, 0, 1) at Step 4; the router SNR is that of the unenhanced prompt; enhancer time is recorded (enh_ms) and not counted in NFE", "enh_outputs": ENH,
                       "piec": "generated frames of the final mel multiplied (linear magnitude) by G = clip(1 - beta(S) N_b / M, 0.1, 1), "
                               "N_b = mean of the quietest 20% prompt frames per band of the noisy conditioning; beta(S) = clip((30 - S) / 10, 0, 1); "
                               "piec4 on the plain Step-4 output, swcpiec4 on the SWC Step-4 output",
                       "controls": "enh8 and piec8 are cleaned Step-8 outputs, used to judge enh4 and piec4 within the cleaned system; not candidates",
                       "tas": "Trajectory Anomaly Score: squared Mahalanobis distance of log(d_1..d_6) from the clean-prompt training mean (one-sided: "
                              "only below the mean); hand over iff SNR >= 20 dB and score <= tau, tau certified by LTT; drift profile from the "
                              "committed EXP-012 signals of the same cases"},
           "nfe": {str(k): v for k, v in COST.items()},
           "n_speakers": N_SPEAKERS, "conditions": list(CONDS), "theta_db": THETA, "alpha": ALPHA, "delta": DELTA, "folds": FOLDS, "seed": SEED,
           "theta_grid": THETA_GRID, "theta_lo_grid": LOW_GRID, "policies": POLICIES,
           "primary_test": "each method vs plain Step 4 on the noisy conditions pooled (exact McNemar, two-sided)",
           "criteria": ["C1 noisy Step-4 failures drop by at least 25% with p < 0.05", "C2 clean Step-4 failures rise by at most 1",
                        "C3 pooled out-of-fold <m>_route or <m>_heavy: lower mean NFE than snr_theta with no more failures"],
           "secondary": "average SIM and WER vs Step 8, paired Wilcoxon, noisy and clean (descriptive)",
           "selection_rule": "among candidates (controls enh8 and piec8 excluded), the method passing the most criteria; ties broken by the lower pooled mean NFE of <m>_route or <m>_heavy",
           "input_hashes": {str(p.relative_to(THESIS_ROOT)).replace("\\", "/"): rf.sha256_file(p)
                            for p in (rf.T4_GRID, cg.PILOT_GRID, ex.EXP012 / "labels_new.csv", Path(__file__))},
           "thesis_commit": rf.git_head(THESIS_ROOT), "dmo_commit": rf.git_head(args.dmo_root)}
    pre["sha256"] = hashlib.sha256(json.dumps(pre, sort_keys=True).encode("utf-8")).hexdigest()
    pre["created_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    path.write_text(json.dumps(pre, indent=2) + "\n", encoding="utf-8")
    print(f"[OK] preregistration written: {path}", flush=True)
    return pre


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["smoke", "run", "analyze"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--dmo_root", default="/content/Shonget-DMOSpeech2")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--allow_partial", action="store_true")
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.cmd == "analyze":
        analyze(out, args.allow_partial)
        return
    import torch
    # The stored references were computed on a Tesla T4 in FP32. On Ampere or newer GPUs (A100, L4, G4), keep full FP32
    # instead of TF32 so a faster GPU changes speed, not the comparison; the smoke test reports any remaining difference.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    pipe = rf.Pipeline(args.dmo_root, args.device)
    enhancer = Enhancer(pipe.torch, pipe.torchaudio, pipe.device)
    if args.cmd == "smoke":
        out = out / "smoke"
        out.mkdir(exist_ok=True)
        its = [it for it in ex.items() if it["utt_id"] not in cg.pilot_utts()][:2]
        repro = run_cases(pipe, its, ("clean", "gauss20"), out, enhancer, reproduce=True)
        (out / "reproduction.json").write_text(json.dumps(repro, indent=2) + "\n", encoding="utf-8")
        for r in repro:
            if "check" in r:
                print(f"reproduction {r['utt_id'][-8:]} {r['condition']}: {r['check']}: max |diff| {r['max_abs_diff']:.2e}", flush=True)
            else:
                print(f"reproduction {r['utt_id'][-8:]} {r['condition']} Step {r['step']}: WER {r['wer']:.4f} (stored {r['wer_stored']:.4f}), "
                      f"SIM {r['similarity']:.4f} (stored {r['similarity_stored']:.4f})", flush=True)
        print(f"Smoke run complete: {out}")
        return
    write_prereg(out, args)
    run_cases(pipe, ex.items(), CONDS, out, enhancer)
    analyze(out)


if __name__ == "__main__":
    main()
