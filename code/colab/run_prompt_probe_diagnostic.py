"""Diagnostic of the Prompt-Probe Handover (EXP-025): development speakers only, before any pilot or frozen test.

Idea. In prompt-infilling TTS part of the input's answer is known: the voice prompt. Hide the last second of speech in
the prompt from the teacher (its text stays, the canvas length stays; that second simply joins the part the teacher must
generate). At every step the teacher's estimate of the hidden second can then be compared with the real audio: a
ground-truth measurement of the teacher's current error for this speaker, this sentence and this step, at no extra
forward pass. The handover uses it to correct the teacher's estimate before the student finishes: a small correction
fitted on the hidden second (estimate -> real audio) is applied to the target. For an ideal teacher the fitted
correction is the identity (a posterior mean is already calibrated), so whatever it corrects is the network's own bias
at that step, measured where the answer is known. Two corrections are tried:

    A  per mel band, y = a_b x + c_b, with the slope shrunk halfway to 1 (ridge, lambda = 1)
    B  per mel band the voice-part offset (band average over time), and one gain on the 9-frame slow part

The student keeps the full prompt. The diagnostic asks two questions on 40 development speakers (clean prompts):
(1) does a correction fitted on the first half of the hidden second reduce the error on the second half (transfer
within known audio), and does it move the target estimate toward the teacher's own later estimate (Step 8); (2) does the
corrected handover at Step 2 or 3 (9 or 11 passes) keep the quality of the released Step-4 handover (13 passes), and is
the correction what helps (corrected against plain, both with the second hidden)? Stored scores of the released teacher
for the same cases and seeds are reused (EXP-018). Development evidence only.

Subcommands: freeze (write the preregistration once), smoke, run (resumable), analyze.
"""
import argparse
import hashlib
import json
import math
import random
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_followup_tests as rf  # noqa: E402
import run_consistency_gate_prelim as cg  # noqa: E402
import run_endpoint_extrapolation_prelim as ex  # noqa: E402
import run_noise_guidance_prelim as ng  # noqa: E402
import run_grid32_prelim as gl  # noqa: E402
import run_selective_handover_prelim as sh  # noqa: E402

THESIS_ROOT = rf.THESIS_ROOT
EXP_DIR = THESIS_ROOT / "research/experiments/EXP-025-prompt-probe-handover-diagnostic"
PREREG_PATH = EXP_DIR / "preregistration.json"
STEPS_CSV = sh.STEPS_CSV
FAILURE_CONFIG = THESIS_ROOT / "code/scheduling/failure_config.json"
MANIFEST = rf.SPLITS_DIR / "development_manifest.json"
DMO_COMMIT = "8a7868fc3345702a586173447f4869205b501f26"
DEPENDENCIES = ("run_prompt_probe_diagnostic.py", "run_selective_handover_prelim.py", "run_grid32_prelim.py",
                "run_noise_guidance_prelim.py", "run_followup_tests.py", "run_consistency_gate_prelim.py",
                "run_endpoint_extrapolation_prelim.py")

N_CASES, COND = 40, "clean"
FRAMES_PER_SECOND = 24000 / 256
PROBE_FRAMES = round(1.0 * FRAMES_PER_SECOND)  # the hidden second: 94 frames
MIN_KEPT_SHARE = 1.0 / 3.0                     # at least a third of the prompt stays visible to the teacher
RIDGE = 1.0                                    # correction A: slope shrunk halfway to 1
KMAX = 8
STEPS = (2, 3)                                 # corrected handovers: 9 and 11 passes
CHECK_STEPS = (2, 3, 4)
VARIANTS = ("A", "B")
OUTS = ("P2", "P3", "P4", "A2", "A3", "B2", "B3")   # P = plain (second hidden, no correction)
METRICS = ("wer", "similarity", "utmos")
BOOT_B, BOOT_SEED = 2000, 2026
TEACHER_TOLERANCE = 1e-4
BOUNDS = {"wer": 0.01, "similarity": 0.01, "utmos": 0.10}   # against the released Step-4 handover, clean prompts
TRANSFER_MAX = 0.9                                          # median error ratio on the unseen half at Step 3

FIELDS = ["utt_id", "speaker_id", "condition", "prompt_snr", "gpu", "prompt_len", "gen_len", "probe_start", "probe_end"] + \
         [f"p_{k}" for k in range(KMAX + 1)] + \
         [f"r{v}_{k}" for v in VARIANTS for k in CHECK_STEPS] + [f"m{v}_{k}" for v in VARIANTS for k in CHECK_STEPS] + \
         [f"slopeA_{k}" for k in CHECK_STEPS] + [f"gainB_{k}" for k in CHECK_STEPS] + \
         [f"{m}_{o}" for o in OUTS for m in METRICS + ("hypothesis",)] + ["case_seconds", "utc"]
SMOKE_FIELDS = ["teacher_loop_max_abs_diff", "probe_outside_max_abs"] + [f"{m}_W{k}" for k in (3, 4) for m in METRICS + ("hypothesis",)]


def cost(name):
    """Forward passes: 2 per guided teacher evaluation at Steps 0..k, then the student's 3. Hiding the second and
    fitting the correction cost no forward pass."""
    return 2 * (int(name[1:]) + 1) + 3


def items():
    return ex.items()[:N_CASES]


def smoke_items():
    return ex.items()[58:60]


# ------------------------------------------------------------------------------- math
def _norm(x):
    return float((x * x).sum()) ** 0.5


def _maxabs(x):
    return float(abs(x).max())


def _mean_frames(x):
    return x.mean(dim=0, keepdim=True) if hasattr(x, "clone") else x.mean(axis=0, keepdims=True)


def _sum_frames(x):
    return x.sum(dim=0, keepdim=True) if hasattr(x, "clone") else x.sum(axis=0, keepdims=True)


def probe_span(cond, plen, frames=PROBE_FRAMES):
    """The hidden second: `frames` frames ending at the prompt's last frame louder than the prompt's median frame (so
    trailing silence is not the probe). Returns (start, end) or None if less than a third of the prompt would stay."""
    e = cond[0, :plen, :].mean(-1)
    vals = [float(v) for v in (e.float().cpu().tolist() if hasattr(e, "cpu") else e)]
    med = sorted(vals)[len(vals) // 2]
    loud = [i for i, v in enumerate(vals) if v > med]
    if not loud:
        return None
    end = loud[-1] + 1
    start = end - frames
    if start < math.ceil(MIN_KEPT_SHARE * plen):
        return None
    return start, end


def hide(step_cond, span):
    """The teacher's conditioning with the probe frames masked (zeros, as the masked target region)."""
    sc = sh._clone(step_cond)
    sc[:, span[0]:span[1], :] = 0.0
    return sc


def fit_affine(x, y, ridge=RIDGE):
    """Per band y = a x + c on frames (rows), slope shrunk toward 1: a = (Sxy + ridge Sxx) / ((1 + ridge) Sxx)."""
    mx, my = _mean_frames(x), _mean_frames(y)
    dx, dy = x - mx, y - my
    sxx, sxy = _sum_frames(dx * dx), _sum_frames(dx * dy)
    a = (sxy + ridge * sxx) / ((1.0 + ridge) * sxx + 1e-8)
    return a, my - a * mx


def apply_affine(fit, x):
    a, c = fit
    return a * x + c


def fit_parts(x, y):
    """Voice-part offset per band and one gain on the slow part (sh.decompose on the probe frames)."""
    vx, sx, _ = sh.decompose(x)
    vy, sy, _ = sh.decompose(y)
    g = float((sx * sy).sum()) / (float((sx * sx).sum()) + 1e-8)
    return vy - vx, g


def apply_parts(fit, x):
    off, g = fit
    v, s, f = sh.decompose(x)
    return v + off + g * s + f


FIT = {"A": (fit_affine, apply_affine), "B": (fit_parts, apply_parts)}


def corrected(T, span, truth, plen, variant):
    """The handover input: the teacher's estimate with the target frames corrected by the fit on the hidden second."""
    fit_fn, apply_fn = FIT[variant]
    fit = fit_fn(T[0, span[0]:span[1], :], truth)
    x = sh._clone(T)
    x[0, plen:, :] = apply_fn(fit, T[0, plen:, :])
    return x, fit


def transfer_ratio(T, span, truth, variant):
    """Fit on the first half of the hidden second, measure on the second half: corrected error / plain error."""
    mid = (span[1] - span[0]) // 2
    est = T[0, span[0]:span[1], :]
    fit_fn, apply_fn = FIT[variant]
    fit = fit_fn(est[:mid], truth[:mid])
    plain = _norm(est[mid:] - truth[mid:])
    return _norm(apply_fn(fit, est[mid:]) - truth[mid:]) / (plain + 1e-8)


def toward_later(C, T, T_late, plen):
    """Slow part, target frames: distance of the corrected estimate to the teacher's Step-8 estimate, over the
    plain estimate's distance (below 1: the correction moves the estimate where the teacher was heading)."""
    s = lambda x: sh.decompose(x[0, plen:, :])[1]
    return _norm(s(C) - s(T_late)) / (_norm(s(T) - s(T_late)) + 1e-8)


def teacher_trajectory(pipe, ctx, kmax, step_cond):
    """The released guided Euler teacher (sampler.run_teacher_single_pass) with a given conditioning: {i: T_i}."""
    torch, sampler = pipe.torch, pipe.sampler
    grid = sampler.SWAY_TEACHER_GRID
    x, ends = sh._clone(ctx["y0"]), {}
    with torch.inference_mode():
        for i in range(kmax + 1):
            t = torch.tensor(grid[i], device=pipe.device, dtype=torch.float32)
            kw = dict(x=x, cond=step_cond, text=ctx["text_ids"], time=t, mask=None)
            vc = pipe.tts.teacher(drop_audio_cond=False, drop_text=False, **kw)
            vu = pipe.tts.teacher(drop_audio_cond=True, drop_text=True, **kw)
            v = vc + (vc - vu) * sampler.CFG_STRENGTH
            ends[i] = x + (1.0 - grid[i]) * v
            if i < kmax:
                x = x + (grid[i + 1] - grid[i]) * v
    return ends


# --------------------------------------------------------------------------------- runs
def stored_w():
    """(utt_id, condition) -> stored scores of the released teacher's handovers at Steps 2, 3, 4 and 8 (EXP-018)."""
    return {(r["utt_id"], r["condition"]): {f"W{k}": {x: float(r[f"{x}_g16_{k}"]) for x in METRICS} for k in (2, 3, 4, 8)}
            for r in rf.read_csv(STEPS_CSV)}


def run_cases(pipe, todo_items, out_dir, csv_name="probe.csv", smoke=False, stored=None):
    path = Path(out_dir) / csv_name
    fields = FIELDS + (SMOKE_FIELDS if smoke else [])
    done = rf.done_keys(path, 1)
    todo = [it for it in todo_items if (it["utt_id"], COND) not in done]
    print(f"[probe] {len(todo)} cases to run ({len(done)} already in {path.name}) on {pipe.gpu}", flush=True)
    taus, repro, checks = list(pipe.sampler.STUDENT_TAUS), [], []
    for n, item in enumerate(todo, 1):
        t0 = time.perf_counter()
        ctx = pipe.prepare(item, COND, item["input_seed"], out_dir)
        plen = ctx["prompt_len"]
        span = probe_span(ctx["cond"], plen)
        if span is None:
            raise SystemExit(f"{item['utt_id']}: prompt too short for a {PROBE_FRAMES}-frame probe")
        truth = ctx["cond"][0, span[0]:span[1], :]
        E = teacher_trajectory(pipe, ctx, KMAX, hide(ctx["step_cond"], span))
        row = {"utt_id": item["utt_id"], "speaker_id": item["speaker_id"], "condition": COND, "prompt_snr": ctx["prompt_snr"],
               "gpu": pipe.gpu, "prompt_len": plen, "gen_len": int(ctx["y0"].shape[1]) - plen, "probe_start": span[0], "probe_end": span[1]}
        inputs = {f"P{k}": E[k] for k in (2, 3, 4)}
        with pipe.torch.inference_mode():
            for k in range(KMAX + 1):
                row[f"p_{k}"] = _norm(E[k][0, span[0]:span[1], :] - truth) / (_norm(truth) + 1e-8)
            for k in CHECK_STEPS:
                for v in VARIANTS:
                    row[f"r{v}_{k}"] = transfer_ratio(E[k], span, truth, v)
                    C, fit = corrected(E[k], span, truth, plen, v)
                    row[f"m{v}_{k}"] = toward_later(C, E[k], E[KMAX], plen)
                    if v == "A":
                        row[f"slopeA_{k}"] = float(abs(fit[0] - 1.0).mean())
                    else:
                        row[f"gainB_{k}"] = fit[1]
                    if k in STEPS:
                        inputs[f"{v}{k}"] = C
        gt = pipe.ground_truth(item)
        for o in OUTS:
            m = pipe.score(ng.decode(pipe, ctx, gl.student(pipe, ctx, inputs[o], taus)), item, gt)
            row.update({f"{x}_{o}": m[x] for x in METRICS + ("hypothesis",)})
        if smoke:
            released, _ = pipe.teacher(ctx, (3, 4))
            own = teacher_trajectory(pipe, ctx, 4, ctx["step_cond"])
            hidden = hide(ctx["step_cond"], span)
            with pipe.torch.inference_mode():
                row["teacher_loop_max_abs_diff"] = max(_maxabs(released[k] - own[k]) for k in (3, 4))
                outside = hidden - ctx["step_cond"]
                outside[:, span[0]:span[1], :] = 0.0
                row["probe_outside_max_abs"] = _maxabs(outside)
            checks.append((row["teacher_loop_max_abs_diff"], row["probe_outside_max_abs"]))
            for k in (3, 4):
                m = pipe.score(ng.decode(pipe, ctx, gl.student(pipe, ctx, own[k], taus)), item, gt)
                row.update({f"{x}_W{k}": m[x] for x in METRICS + ("hypothesis",)})
                key = (item["utt_id"], COND)
                if stored is not None and key in stored:
                    s = stored[key][f"W{k}"]
                    repro.append({"utt_id": item["utt_id"], "condition": COND, "output": f"W{k}", "wer": m["wer"], "wer_stored": s["wer"],
                                  "similarity": m["similarity"], "similarity_stored": s["similarity"]})
        row.update({"case_seconds": round(time.perf_counter() - t0, 2), "utc": sh._utc()})
        sh._append(path, fields, row)
        print(f"[{n}/{len(todo)}] probe: {item['utt_id']} ({row['case_seconds']:.0f} s)", flush=True)
    print(f"[OK] probe complete: {path}", flush=True)
    return repro, checks


# ----------------------------------------------------------------------------- analysis
def boot(groups, B=BOOT_B, seed=BOOT_SEED):
    values = [sum(v) / len(v) for v in groups.values() if v]
    if not values:
        return {"mean": None, "lo": None, "hi": None, "n": 0}
    rng = random.Random(seed)
    boots = sorted(sum(rng.choice(values) for _ in values) / len(values) for _ in range(B))
    return {"mean": sum(values) / len(values), "lo": boots[int(0.025 * B)], "hi": boots[int(0.975 * B)], "n": len(values)}


def _num(v):
    return None if v in (None, "") else float(v)


def _median(values):
    v = sorted(x for x in values if x is not None and not math.isnan(x))
    return None if not v else (v[len(v) // 2] if len(v) % 2 else (v[len(v) // 2 - 1] + v[len(v) // 2]) / 2)


def build_table(rows, stored):
    table = []
    for r in rows:
        s = stored.get((r["utt_id"], r["condition"]))
        if s is None:
            continue
        m = dict(s)
        m.update({o: {x: float(r[f"{x}_{o}"]) for x in METRICS} for o in OUTS})
        table.append({"spk": r["speaker_id"], "m": m, "row": r})
    return table


def paired(table, a, b, metric):
    g = defaultdict(list)
    for r in table:
        x, y = r["m"][a][metric], r["m"][b][metric]
        g[r["spk"]].append((min(1.0, x) - min(1.0, y)) if metric == "wer" else x - y)
    return boot(g)


def analyze(out, allow_partial=False, prereg_path=PREREG_PATH):
    out = Path(out)
    pre = check_prereg(prereg_path)
    th = gl.thresholds()
    rows = rf.read_csv(out / "probe.csv", repair=True) if (out / "probe.csv").exists() else []
    table = build_table(rows, stored_w())
    partial = len(table) < N_CASES
    if partial and not allow_partial:
        raise SystemExit(f"incomplete run: {len(table)} of {N_CASES} cases; finish the run or pass --allow_partial")
    med = lambda f: _median([_num(r[f]) for r in rows])
    res = {"prereg_sha256": pre["sha256"], "cases": len(table), "expected": N_CASES, "partial": partial,
           "passes": {o: cost(o) for o in OUTS + ("W2", "W3", "W4", "W8")},
           "probe_error": {k: med(f"p_{k}") for k in range(KMAX + 1)},
           "transfer": {v: {k: med(f"r{v}_{k}") for k in CHECK_STEPS} for v in VARIANTS},
           "toward_step8": {v: {k: med(f"m{v}_{k}") for k in CHECK_STEPS} for v in VARIANTS},
           "correction_size": {"A_mean_abs_slope_minus_1": {k: med(f"slopeA_{k}") for k in CHECK_STEPS},
                               "B_slow_gain": {k: med(f"gainB_{k}") for k in CHECK_STEPS}}}
    q = {}
    for o in OUTS:
        k = int(o[1:])
        q[o] = {"vs_W4": {x: paired(table, o, "W4", x) for x in METRICS}, "vs_released_same_step": {x: paired(table, o, f"W{k}", x) for x in METRICS},
                "failures_vs_W4": sum(bool(gl.failed(r["m"][o], r["m"]["W4"], th)) for r in table),
                "failures_vs_reference": sum(bool(gl.failed(r["m"][o], r["m"]["W8"], th)) for r in table)}
        if o[0] in VARIANTS:
            q[o]["vs_plain"] = {x: paired(table, o, f"P{k}", x) for x in METRICS}
    res["quality"] = q
    crit = {}
    for v in VARIANTS:
        tr = res["transfer"][v][3]
        g1 = None if partial or tr is None else bool(tr <= TRANSFER_MAX)
        g2 = {}
        for k in STEPS:
            o, w = f"{v}{k}", q[f"{v}{k}"]
            if partial or w["vs_W4"]["wer"]["hi"] is None:
                g2[k] = None
                continue
            g2[k] = bool(w["vs_W4"]["wer"]["hi"] <= BOUNDS["wer"] and w["vs_W4"]["similarity"]["lo"] >= -BOUNDS["similarity"]
                         and w["vs_W4"]["utmos"]["lo"] >= -BOUNDS["utmos"] and w["vs_plain"]["utmos"]["mean"] > 0.0)
        crit[v] = {"G1_transfer": g1, "G2_quality": g2}
    res["criteria"] = crit
    passing = [(k, v) for v in VARIANTS for k in STEPS if crit[v]["G1_transfer"] and crit[v]["G2_quality"][k]]
    passing.sort(key=lambda kv: (kv[0], -q[f"{kv[1]}{kv[0]}"]["vs_plain"]["utmos"]["mean"]))
    res["advance"] = bool(passing)
    res["carried"] = f"{passing[0][1]}{passing[0][0]}" if passing else None
    (out / "probe_results.json").write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
    (out / "PROBE_REPORT.md").write_text(render(res), encoding="utf-8")
    print(f"Wrote {out / 'PROBE_REPORT.md'}", flush=True)
    return res


def _word(v):
    return "n/a" if v is None else ("PASS" if v else "FAIL")


def _ci(v, fmt="{:+.4f}"):
    return "n/a" if not v or v.get("mean") is None else f"{fmt.format(v['mean'])} [{fmt.format(v['lo'])}, {fmt.format(v['hi'])}]"


def _f(v, fmt="{:.3f}"):
    return "n/a" if v is None else fmt.format(v)


def render(res):
    c = res["criteria"]
    L = ["# Diagnostic: Prompt-Probe Handover (EXP-025)", "",
         f"{'**PARTIAL.** ' if res['partial'] else ''}Development speakers only, clean prompts; {res['cases']} of {res['expected']} cases; "
         f"preregistration `{res['prereg_sha256'][:16]}…`. Every number is computed from probe.csv by `run_prompt_probe_diagnostic.py analyze`. "
         "A diagnostic decides whether a pilot is worth building; it cannot support a claim.", "",
         "The teacher does not see the prompt's last second of speech; its estimate of that second, compared with the real audio, "
         "fits a correction that is applied to the target before the student finishes. P = second hidden, no correction; "
         "A = per-band affine correction; B = voice offset and slow-part gain.", "", "## Go / no-go", ""]
    for v in VARIANTS:
        L.append(f"- Variant {v}: G1 transfer **{_word(c[v]['G1_transfer'])}**; G2 quality " +
                 ", ".join(f"Step {k} **{_word(g)}**" for k, g in c[v]["G2_quality"].items()))
    L += [f"- **Build a pilot: {'YES, with ' + res['carried'] if res['advance'] else 'NO'}**", "",
          "## Does the correction transfer? (medians)", "",
          "| Variant | Error ratio on the unseen half, Steps 2 / 3 / 4 | Distance to the Step-8 estimate, corrected / plain, Steps 2 / 3 / 4 |",
          "|---|---|---|"]
    for v in VARIANTS:
        L.append(f"| {v} | " + " / ".join(_f(res['transfer'][v][k], '{:.3f}') for k in CHECK_STEPS) + " | " +
                 " / ".join(_f(res['toward_step8'][v][k], '{:.3f}') for k in CHECK_STEPS) + " |")
    L += ["", "Probe error by step (relative to the real audio, median): " +
          ", ".join(f"{k}: {_f(v, '{:.3f}')}" for k, v in res["probe_error"].items()) + ".", "",
          f"Correction size (median): A |slope - 1| " + ", ".join(f"Step {k} {_f(v)}" for k, v in res['correction_size']['A_mean_abs_slope_minus_1'].items()) +
          "; B slow gain " + ", ".join(f"Step {k} {_f(v)}" for k, v in res['correction_size']['B_slow_gain'].items()) + ".", "",
          "## Quality (speaker bootstrap, 95%)", "",
          "| Output | Passes | vs released Step 4: WER | Similarity | UTMOS | vs plain at the same step: UTMOS | vs released same step: UTMOS | Failures vs Step 4 |",
          "|---|---:|---|---|---|---|---|---:|"]
    for o, w in res["quality"].items():
        L.append(f"| {o} | {res['passes'][o]} | {_ci(w['vs_W4']['wer'])} | {_ci(w['vs_W4']['similarity'])} | {_ci(w['vs_W4']['utmos'], '{:+.3f}')} | "
                 f"{_ci(w['vs_plain']['utmos'], '{:+.3f}') if 'vs_plain' in w else '—'} | {_ci(w['vs_released_same_step']['utmos'], '{:+.3f}')} | "
                 f"{w['failures_vs_W4']}/{res['cases']} |")
    L += ["", "Development evidence only (speakers that designed earlier experiments), clean prompts. Pitch variety and noisy prompts are not measured here."]
    return "\n".join(L) + "\n"


# ------------------------------------------------------------------------ preregistration
def code_hashes():
    return {f"code/colab/{n}": sh.sha256_text(Path(__file__).resolve().parent / n) for n in DEPENDENCIES}


def build_prereg():
    return {"experiment": "EXP-025",
            "purpose": "Diagnostic of the Prompt-Probe Handover before any pilot: does a correction fitted on a hidden second of the voice prompt "
                       "(teacher estimate -> real audio) transfer to unseen audio, and does it let the teacher hand over at Step 2 or 3 with the "
                       "quality of the released Step-4 handover",
            "theory": "for an ideal teacher the endpoint estimate is the posterior mean, which is calibrated, so the fitted correction is the identity; "
                      "a non-identity fit measures the network's own bias at that step where the answer is known",
            "algorithm": {"probe": f"{PROBE_FRAMES} frames ending at the prompt's last frame whose band-averaged log-mel exceeds the prompt's median frame; "
                                   "masked in the teacher's conditioning only (zeros, like the target region); text and canvas length unchanged; "
                                   f"at least {MIN_KEPT_SHARE:.2f} of the prompt stays visible",
                          "A": f"per mel band y = a x + c over the probe frames, slope shrunk to (Sxy + {RIDGE} Sxx) / ((1 + {RIDGE}) Sxx)",
                          "B": "per band the voice-part offset (band average over the probe frames) and one gain on the 9-frame slow part (sh.decompose)",
                          "handover": "the corrected estimate on the target frames, the teacher's estimate elsewhere; the student keeps the full prompt "
                                      "and runs unchanged (released student loop at j = 1)",
                          "teacher": "the released guided Euler teacher on the 16-interval grid with the probe masked"},
            "passes": {o: cost(o) for o in OUTS + ("W2", "W3", "W4", "W8")},
            "data": f"the first {N_CASES} EXP-012 Quick-set development speakers, clean prompts, synthesis seed = input seed; stored released "
                    "handovers at Steps 2, 3, 4 and 8 (EXP-018) for the same cases and seeds",
            "smoke": f"2 cases (Quick-set positions 59-60): this runner's teacher loop matches the released teacher (max abs <= {TEACHER_TOLERANCE}), "
                     f"the hidden conditioning differs only inside the probe, and the stored W3 and W4 scores are reproduced (tolerance {sh.REPRODUCTION_TOLERANCE})",
            "criteria": {"G1_transfer": f"per variant: median over cases of (corrected / plain error) on the second half of the probe when the "
                                        f"correction is fitted on the first half, at Step 3, <= {TRANSFER_MAX}",
                         "G2_quality": f"per variant and Step k in {list(STEPS)}: the corrected handover against the released Step-4 handover of the "
                                       f"same case and seed: capped-WER difference upper 95% bound <= +{BOUNDS['wer']}, similarity lower bound >= "
                                       f"-{BOUNDS['similarity']}, UTMOS lower bound >= -{BOUNDS['utmos']}; and its UTMOS minus the plain hidden-probe "
                                       "handover at the same step, mean > 0 (the correction is what helps)"},
            "decision_rule": "build a pilot (pitch variety, noisy prompts, an adaptive handover from the probe error) only if some variant meets G1 and "
                             "G2 at some step; carry the earliest step, ties broken by the larger UTMOS gain over plain; otherwise report a negative "
                             "diagnostic with no retuning on these data",
            "power": "with 40 speakers the clean Step-4 bounds pass a change as small as refining the teacher's grid in 100% of 300 subsets "
                     "(EXP-024's power check, code/analysis/exp024_power_check.py)",
            "descriptive": ["the probe error at Steps 0-8", "the correction's size per step", "the corrected estimate's distance to the teacher's Step-8 estimate",
                            "every output against the released handover at the same step and against the reference; Definition 3.1 failures"],
            "bootstrap": {"B": BOOT_B, "seed": BOOT_SEED, "unit": "speaker"},
            "novelty": "searches on 2026-11-14 (papers and practitioner tools) found no method that hides part of the conditioning to create a "
                       "known-answer probe of the generator at inference and uses it to correct the handed-over estimate. Closest: CoPaint (known "
                       "region as a sampling constraint), TD-Paint (known-region error as a diagnostic), Vox-Evaluator (masks generated speech), "
                       "VoiceStar (masks prompts in training), NoiseShift and cache calibration (corrections fitted offline). Not a full review.",
            "evidence_status": "development diagnostic on speakers that designed earlier experiments; not evidence for a claim",
            "adoption": "not adopted; changing the thesis method needs a dated decision record and adviser review",
            "hardware": "Colab Tesla T4 only; Shonget-DMOSpeech2 at the pinned commit", "dmo_commit": DMO_COMMIT,
            "code_hashes": code_hashes(),
            "input_hashes": {str(p.relative_to(THESIS_ROOT)).replace("\\", "/"): sh.sha256_text(p) for p in (STEPS_CSV, FAILURE_CONFIG, MANIFEST)},
            "thesis_commit": rf.git_head(THESIS_ROOT)}


def freeze(out=EXP_DIR):
    path = Path(out) / "preregistration.json"
    Path(out).mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise SystemExit(f"{path} exists; a preregistration is written once (delete it only if nothing was synthesized)")
    pre = build_prereg()
    pre["sha256"] = hashlib.sha256(json.dumps(pre, sort_keys=True).encode("utf-8")).hexdigest()
    pre["created_utc"] = sh._utc()
    path.write_text(json.dumps(pre, indent=2) + "\n", encoding="utf-8")
    print(f"[OK] preregistration written: {path} (sha256 {pre['sha256'][:16]}...)", flush=True)
    return pre


def check_prereg(prereg_path=PREREG_PATH):
    p = Path(prereg_path)
    if not p.exists():
        raise SystemExit(f"Missing {p}: run `freeze` and push the preregistration before any GPU run.")
    pre = json.loads(p.read_text(encoding="utf-8"))
    body = {k: v for k, v in pre.items() if k not in ("sha256", "created_utc")}
    if hashlib.sha256(json.dumps(body, sort_keys=True).encode("utf-8")).hexdigest() != pre["sha256"]:
        raise SystemExit("preregistration.json was edited after it was frozen")
    changed = [k for k, v in pre["code_hashes"].items() if code_hashes().get(k) != v]
    if changed:
        raise SystemExit(f"Code differs from the version frozen in preregistration.json: {changed}")
    moved = [k for k, v in pre["input_hashes"].items() if not (THESIS_ROOT / k).exists() or sh.sha256_text(THESIS_ROOT / k) != v]
    if moved:
        raise SystemExit(f"Input files are missing or differ from the frozen versions: {moved}")
    return pre


def check_smoke(rep):
    sh.check_reproduction(rep["rows"])
    if rep["teacher_loop_max_abs_diff"] > TEACHER_TOLERANCE:
        raise SystemExit(f"This runner's teacher loop does not match the released teacher ({rep['teacher_loop_max_abs_diff']:.2e}); do not run.")
    if rep["probe_outside_max_abs"] != 0.0:
        raise SystemExit("Hiding the probe changed the conditioning outside the probe; do not run.")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["freeze", "smoke", "run", "analyze"])
    ap.add_argument("--out")
    ap.add_argument("--dmo_root", default="/content/Shonget-DMOSpeech2")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--allow_partial", action="store_true")
    args = ap.parse_args(argv)
    if args.cmd == "freeze":
        freeze(args.out or EXP_DIR)
        return
    if not args.out:
        ap.error("--out is required")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.cmd == "analyze":
        analyze(out, args.allow_partial)
        return
    check_prereg()
    head = rf.git_head(args.dmo_root)
    if head != DMO_COMMIT:
        raise SystemExit(f"Shonget-DMOSpeech2 is at {head}, not the pinned {DMO_COMMIT}.")
    pipe = rf.Pipeline(args.dmo_root, args.device)
    if "T4" not in pipe.gpu:
        raise SystemExit(f"EXP-025 reuses T4 labels and student noise depends on the GPU; found {pipe.gpu}.")
    if args.cmd == "smoke":
        sm = out / "smoke"
        sm.mkdir(exist_ok=True)
        repro, checks = run_cases(pipe, smoke_items(), sm, smoke=True, stored=stored_w())
        rep = {"rows": repro, "teacher_loop_max_abs_diff": max(c[0] for c in checks), "probe_outside_max_abs": max(c[1] for c in checks)}
        (sm / "reproduction.json").write_text(json.dumps(rep, indent=2) + "\n", encoding="utf-8")
        check_smoke(rep)
        worst_wer, worst_sim = sh.check_reproduction(repro)
        print(f"Smoke run complete: {sm} (stored W3 and W4 reproduced: max |dWER| {worst_wer:.4f}, max |dSIM| {worst_sim:.4f}; "
              f"teacher loop matches the released teacher: max difference {rep['teacher_loop_max_abs_diff']:.2e})", flush=True)
        return
    smoke_file = out / "smoke" / "reproduction.json"
    if not smoke_file.exists():
        raise SystemExit("Run the smoke test first: the run reuses stored labels only if they reproduce.")
    check_smoke(json.loads(smoke_file.read_text(encoding="utf-8")))
    shutil.copy2(PREREG_PATH, out / "preregistration.json")
    run_cases(pipe, items(), out)
    analyze(out)


if __name__ == "__main__":
    main()
