"""Frozen confirmation of Voice-Swap Handover on fresh speakers (EXP-021).

Voice-Swap Handover is the STT combination of EXP-020's Selective Handover. The guided teacher runs to Step 4; one
student pass from independent noise gives a draft S; on generated frames the teacher's voice part (each band's average
over time) is replaced by the draft's, keeping the teacher's slow and fine parts; the released student loop then runs
from that input as usual. 14 forward passes against 21 for the 0.25 reference (Step 8). No model is trained.

EXP-020 found, after its results were seen, that Voice-Swap at Step 4 matched the reference on its 60 development
speakers. This runner tests that claim, frozen in advance, on development speakers 61-100, which were never used to
design Voice-Swap, under clean, white-noise, recorded cafeteria babble, reverberation and telephone-band prompts, and
measures the time it saves on the Colab T4. Every Voice-Swap operation is imported unchanged from
run_selective_handover_prelim.py.

Outputs per case: ref and ref2 (Step 8 with two student noise draws), vs and vs2 (Voice-Swap with two draft and student
noise draws), plain4 (today's Step-4 handover) and student (student only). The two draws per method halve the sampling
noise in the paired comparison; ref2 against ref also measures the failure rule's own noise floor.

Subcommands: freeze (write the preregistration once), smoke (development speakers only), run (resumable), analyze.
"""
import argparse
import datetime
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
EXP_DIR = THESIS_ROOT / "research/experiments/EXP-021-voice-swap-confirmation"
PREREG_PATH = EXP_DIR / "preregistration.json"
EXP020_CSV = THESIS_ROOT / "research/experiments/EXP-020-selective-handover-prelim/results/selective.csv"
STEPS_CSV, GRID32_CSV = sh.STEPS_CSV, sh.GRID32_CSV
DMO_COMMIT = "8a7868fc3345702a586173447f4869205b501f26"
DEPENDENCIES = ("run_voice_swap_confirm.py", "run_selective_handover_prelim.py", "run_grid32_prelim.py",
                "run_noise_guidance_prelim.py", "run_followup_tests.py", "run_consistency_gate_prelim.py",
                "run_endpoint_extrapolation_prelim.py")

DEV_N, FRESH_N = 60, 40               # development speakers 1-60 designed Voice-Swap; 61-100 are fresh
CONDS = ("clean", "noisy15", "gauss20", "babble15", "reverb09", "telephone")
DEGRADED = CONDS[1:]
K, REF_STEP, CODE = 4, 8, "STT"
OUTS = ("ref", "ref2", "vs", "vs2", "plain4", "student")
VS_PAIR, REF_PAIR = ("vs", "vs2"), ("ref", "ref2")
PASSES = {"ref": 21, "ref2": 21, "vs": 14, "vs2": 14, "plain4": 13, "student": 4}
REPLICATE_OFFSET = 7919               # second noise draw; differs from the student levels (+0, 7, 29, 61) and the draft (+1009)
METRICS = ("wer", "similarity", "utmos")
DIV_N, DIV_SEEDS, DIV_SEED_BASE = 30, 20, 1000
DIV_CONDS = ("clean", "noisy15")
DIV_OUTS = ("ref", "vs", "student")
BOOT_B, SEED = 2000, 2026
MARGINS = {"wer_clean": 0.01, "wer_degraded": 0.025, "similarity": 0.01, "cvf0": 0.02, "time_ratio": 0.80}
REPRODUCTION_TOLERANCE = sh.REPRODUCTION_TOLERANCE
FIELDS = ["utt_id", "speaker_id", "condition", "prompt_snr", "gpu", "order", "t_ref", "t_vs", "audio_seconds",
          "e4_max_abs_diff", "voice_rms"] + [f"{m}_{o}" for o in OUTS for m in METRICS + ("hypothesis",)] + ["case_seconds", "utc"]
DIV_FIELDS = ["utt_id", "speaker_id", "condition", "n_seeds"] + [f"cvf0_{o}" for o in DIV_OUTS] + \
             [f"voiced_{o}" for o in DIV_OUTS] + ["case_seconds", "utc"]


def fresh_items():
    """Development speakers 61-100 by utterance ID: never used to design Voice-Swap (EXP-012 to EXP-020 used 1-60)."""
    items = cg.dev_items()[DEV_N:DEV_N + FRESH_N]
    designed = {it["utt_id"] for it in cg.dev_items(DEV_N)}
    if len(items) != FRESH_N or designed & {it["utt_id"] for it in items}:
        raise SystemExit("The fresh speaker list is not 40 speakers disjoint from the 60 design speakers.")
    return items


# --------------------------------------------------------------------------- algorithm
def voice_swap_input(pipe, ctx, teacher_endpoint):
    """EXP-020's STT input: the draft's voice part, the teacher's slow and fine parts (prompt frames unchanged)."""
    S = sh.student_sample(pipe, ctx)
    with pipe.torch.inference_mode():
        return sh.selective_handover(teacher_endpoint, S, ctx["prompt_len"], CODE), S


def _max_abs_diff(a, b):
    return float(abs(a - b).max())


def generate(pipe, ctx, vs_first=False):
    """Every output of one case. The reference and Voice-Swap paths are timed separately, end to end from the start of
    the teacher to decoded audio, with CUDA synchronized; the order alternates between cases."""
    taus = list(pipe.sampler.STUDENT_TAUS)

    def ref_path():
        E, _ = pipe.teacher(ctx, (K, REF_STEP))
        x = gl.student(pipe, ctx, E[REF_STEP], taus)
        return E, ng.decode(pipe, ctx, x)

    def vs_path():
        E, _ = pipe.teacher(ctx, (K,))
        h, S = voice_swap_input(pipe, ctx, E[K])
        x = gl.student(pipe, ctx, h, taus)
        return E, S, ng.decode(pipe, ctx, x)

    if vs_first:
        (Ev, S, a_vs), t_vs = gl.timed(pipe, vs_path)
        (Er, a_ref), t_ref = gl.timed(pipe, ref_path)
    else:
        (Er, a_ref), t_ref = gl.timed(pipe, ref_path)
        (Ev, S, a_vs), t_vs = gl.timed(pipe, vs_path)
    ctx2 = {**ctx, "seed": int(ctx["seed"]) + REPLICATE_OFFSET}
    h2, _ = voice_swap_input(pipe, ctx2, Ev[K])
    audio = {"ref": a_ref, "vs": a_vs,
             "ref2": ng.decode(pipe, ctx, gl.student(pipe, ctx2, Er[REF_STEP], taus)),
             "vs2": ng.decode(pipe, ctx, gl.student(pipe, ctx2, h2, taus)),
             "plain4": ng.decode(pipe, ctx, gl.student(pipe, ctx, Er[K], taus)),
             "student": ng.decode(pipe, ctx, gl.student(pipe, ctx, ctx["step_cond"], [0.0] + taus))}
    with pipe.torch.inference_mode():
        info = {"t_ref": t_ref, "t_vs": t_vs, "e4_max_abs_diff": _max_abs_diff(Er[K], Ev[K]),
                "voice_rms": sh.voice_rms(Ev[K], S, ctx["prompt_len"])}
    return audio, info


# --------------------------------------------------------------------------------- runs
def stored_labels():
    """(utt_id, condition) -> stored WER and similarity of ref (EXP-018 Step 8), plain4 (EXP-018 Step 4), vs (EXP-020
    STT_4) and student (EXP-017) for the development cases, used only by the smoke test's reproduction check."""
    steps = {(r["utt_id"], r["condition"]): r for r in rf.read_csv(STEPS_CSV)}
    grid = {(r["utt_id"], r["condition"]): r for r in rf.read_csv(GRID32_CSV)}
    out = {}
    for r in rf.read_csv(EXP020_CSV):
        key = (r["utt_id"], r["condition"])
        if key in steps and key in grid:
            pick = lambda src, suffix: {"wer": float(src[f"wer_{suffix}"]), "similarity": float(src[f"similarity_{suffix}"])}
            out[key] = {"ref": pick(steps[key], "g16_8"), "plain4": pick(steps[key], "g16_4"), "vs": pick(r, "STT_4"),
                        "student": pick(grid[key], "student4")}
    return out


def run_quality(pipe, items, conds, out_dir, csv_name="confirm.csv", stored=None):
    """Resumable: one row per (utterance, condition), flushed to disk after every case. One untimed warm-up case runs
    first so that cuDNN autotuning does not enter the timings. Returns reproduction rows when stored labels are given."""
    path = Path(out_dir) / csv_name
    done = rf.done_keys(path, 1)
    todo = [(it, c) for it in items for c in conds if (it["utt_id"], c) not in done]
    print(f"[quality] {len(todo)} cases to run ({len(done)} already in {path.name}) on {pipe.gpu}", flush=True)
    repro = []
    if todo:
        item, cond = todo[0]
        generate(pipe, pipe.prepare(item, cond, item["input_seed"], out_dir))
        print("[quality] warm-up case done (not recorded)", flush=True)
    for n, (item, cond) in enumerate(todo, 1):
        t0 = time.perf_counter()
        ctx = pipe.prepare(item, cond, item["input_seed"], out_dir)
        vs_first = n % 2 == 0
        audio, info = generate(pipe, ctx, vs_first)
        gt = pipe.ground_truth(item)
        row = {"utt_id": item["utt_id"], "speaker_id": item["speaker_id"], "condition": cond, "prompt_snr": ctx["prompt_snr"],
               "gpu": pipe.gpu, "order": "vs_first" if vs_first else "ref_first", "audio_seconds": len(audio["ref"]) / 24000.0, **info}
        for o in OUTS:
            m = pipe.score(audio[o], item, gt)
            row.update({f"{k}_{o}": m[k] for k in METRICS + ("hypothesis",)})
        if stored is not None and (item["utt_id"], cond) in stored:
            for o, s in stored[(item["utt_id"], cond)].items():
                repro.append({"utt_id": item["utt_id"], "condition": cond, "output": o, "wer": row[f"wer_{o}"], "wer_stored": s["wer"],
                              "similarity": row[f"similarity_{o}"], "similarity_stored": s["similarity"]})
        row.update({"case_seconds": round(time.perf_counter() - t0, 2), "utc": sh._utc()})
        sh._append(path, FIELDS, row)
        print(f"[{n}/{len(todo)}] quality: {item['utt_id']} {cond} ({row['case_seconds']:.0f} s)", flush=True)
    print(f"[OK] quality complete: {path}", flush=True)
    return repro


def run_diversity(pipe, items, conds, out_dir, seeds=DIV_SEEDS, csv_name="diversity.csv"):
    """CVf0 across seeds for ref, vs and student, as in EXP-020: the prompt degradation is fixed per pair
    (noise seed = input seed) and the synthesis seed is input_seed + 1000 + m."""
    path = Path(out_dir) / csv_name
    done = rf.done_keys(path, 1)
    todo = [(it, c) for c in conds for it in items if (it["utt_id"], c) not in done]
    print(f"[diversity] {len(todo)} pairs x {seeds} seeds to run ({len(done)} already in {path.name})", flush=True)
    taus = list(pipe.sampler.STUDENT_TAUS)
    for n, (item, cond) in enumerate(todo, 1):
        t0 = time.perf_counter()
        tracks = {o: [] for o in DIV_OUTS}
        for m in range(seeds):
            ctx = pipe.prepare(item, cond, int(item["input_seed"]) + DIV_SEED_BASE + m, out_dir, noise_seed=item["input_seed"])
            E, _ = pipe.teacher(ctx, (K, REF_STEP))
            h, _ = voice_swap_input(pipe, ctx, E[K])
            xs = {"ref": gl.student(pipe, ctx, E[REF_STEP], taus), "vs": gl.student(pipe, ctx, h, taus),
                  "student": gl.student(pipe, ctx, ctx["step_cond"], [0.0] + taus)}
            for o in DIV_OUTS:
                tracks[o].append(sh.f0_track(ng.decode(pipe, ctx, xs[o])))
        row = {"utt_id": item["utt_id"], "speaker_id": item["speaker_id"], "condition": cond, "n_seeds": seeds}
        for o in DIV_OUTS:
            row[f"cvf0_{o}"], row[f"voiced_{o}"] = sh.cvf0(tracks[o])
        row.update({"case_seconds": round(time.perf_counter() - t0, 2), "utc": sh._utc()})
        sh._append(path, DIV_FIELDS, row)
        print(f"[{n}/{len(todo)}] diversity: {item['utt_id']} {cond} ({row['case_seconds']:.0f} s)", flush=True)
    print(f"[OK] diversity complete: {path}", flush=True)


# ----------------------------------------------------------------------------- analysis
def boot(groups, B=BOOT_B, seed=SEED):
    """Mean of per-speaker means and its percentile 95% interval, resampling speakers (the EXP-020 procedure)."""
    values = [sum(v) / len(v) for v in groups.values() if v]
    if not values:
        return {"mean": None, "lo": None, "hi": None, "n": 0}
    rng = random.Random(seed)
    boots = sorted(sum(rng.choice(values) for _ in values) / len(values) for _ in range(B))
    return {"mean": sum(values) / len(values), "lo": boots[int(0.025 * B)], "hi": boots[int(0.975 * B)], "n": len(values)}


def value(r, o, m):
    v = float(r[f"{m}_{o}"])
    return min(1.0, v) if m == "wer" else v


def paired(rows, conds, m, a=VS_PAIR, b=REF_PAIR):
    """Per case: mean over the draws of a minus mean over the draws of b (WER capped at 1); speaker bootstrap."""
    g = defaultdict(list)
    for r in rows:
        if r["condition"] in conds:
            g[r["speaker_id"]].append(sum(value(r, o, m) for o in a) / len(a) - sum(value(r, o, m) for o in b) / len(b))
    return boot(g)


def time_ratio(rows):
    g = defaultdict(list)
    for r in rows:
        g[r["speaker_id"]].append(float(r["t_vs"]) / float(r["t_ref"]))
    return boot(g)


def cv_diff(div, cond, a, b):
    g = {}
    for r in div:
        if r["condition"] == cond:
            x, y = float(r[f"cvf0_{a}"]), float(r[f"cvf0_{b}"])
            if not (math.isnan(x) or math.isnan(y)):
                g[r["speaker_id"]] = [x - y]
    return boot(g)


def failures(rows, th):
    """Definition 3.1 against ref, per output and condition (descriptive; ref2 against ref is the noise floor)."""
    out = {}
    for o in OUTS[1:]:
        out[o] = {c: sum(bool(gl.failed({m: float(r[f"{m}_{o}"]) for m in ("wer", "similarity")},
                                        {m: float(r[f"{m}_ref"]) for m in ("wer", "similarity")}, th))
                         for r in rows if r["condition"] == c) for c in CONDS}
    return out


def evaluate(rows, div, complete_div):
    clean = [r for r in rows if r["condition"] == "clean"]
    degraded = [r for r in rows if r["condition"] in DEGRADED]
    d = {"wer_clean": paired(clean, ("clean",), "wer"), "wer_degraded": paired(degraded, DEGRADED, "wer"),
         "similarity_clean": paired(clean, ("clean",), "similarity"), "similarity_degraded": paired(degraded, DEGRADED, "similarity"),
         "time_ratio": time_ratio(rows) if rows else boot({}),
         "cvf0": {c: {"vs_minus_ref": cv_diff(div, c, "vs", "ref"), "vs_minus_student": cv_diff(div, c, "vs", "student")} for c in DIV_CONDS}}

    def ok(stat, test):
        return None if stat["mean"] is None else bool(test(stat))
    p1 = [ok(d["wer_clean"], lambda s: s["hi"] <= MARGINS["wer_clean"]), ok(d["wer_degraded"], lambda s: s["hi"] <= MARGINS["wer_degraded"])]
    p2 = [ok(d["similarity_clean"], lambda s: s["lo"] >= -MARGINS["similarity"]), ok(d["similarity_degraded"], lambda s: s["lo"] >= -MARGINS["similarity"])]
    p3 = [ok(d["cvf0"][c]["vs_minus_ref"], lambda s: s["lo"] >= -MARGINS["cvf0"]) for c in DIV_CONDS] + \
         [ok(d["cvf0"][c]["vs_minus_student"], lambda s: s["lo"] > 0) for c in DIV_CONDS]
    both = lambda parts: None if any(p is None for p in parts) else all(parts)
    c = {"P1_word_errors": both(p1), "P2_voice_similarity": both(p2),
         "P3_pitch_variety": both(p3) if complete_div else None,
         "P4_time": ok(d["time_ratio"], lambda s: s["hi"] <= MARGINS["time_ratio"])}
    return {"criteria": c, "confirmed": all(v is True for v in c.values()), "measured": d}


def _avg(values):
    values = list(values)
    return sum(values) / len(values) if values else None


def analyze(out, allow_partial=False, prereg_path=PREREG_PATH):
    out = Path(out)
    pre = check_prereg(prereg_path)
    th = gl.thresholds()
    rows = rf.read_csv(out / "confirm.csv", repair=True) if (out / "confirm.csv").exists() else []
    div = rf.read_csv(out / "diversity.csv", repair=True) if (out / "diversity.csv").exists() else []
    expected, expected_div = FRESH_N * len(CONDS), DIV_N * len(DIV_CONDS)
    partial = len(rows) < expected or len(div) < expected_div
    if partial and not allow_partial:
        raise SystemExit(f"{len(rows)} of {expected} cases, {len(div)} of {expected_div} diversity pairs; finish the run or pass --allow_partial")
    res = {"prereg_sha256": pre["sha256"], "cases": len(rows), "expected_cases": expected, "diversity_pairs": len(div),
           "expected_diversity_pairs": expected_div, "partial": partial, "passes": PASSES, "margins": MARGINS,
           "primary": evaluate(rows, div, len(div) >= expected_div),
           "means": {c: {o: {m: _avg(value(r, o, m) for r in rows if r["condition"] == c) for m in METRICS} for o in OUTS} for c in CONDS},
           "per_condition": {c: {m: paired(rows, (c,), m) for m in METRICS} for c in CONDS},
           "vs_minus_student": {g: {m: paired(rows, conds, m, VS_PAIR, ("student",)) for m in METRICS}
                                for g, conds in (("clean", ("clean",)), ("degraded", DEGRADED))},
           "failures": failures(rows, th) if rows else {},
           "mcnemar_vs_against_plain4_degraded": None,
           "cvf0_means": {c: {o: _avg(float(r[f"cvf0_{o}"]) for r in div if r["condition"] == c and not math.isnan(float(r[f"cvf0_{o}"])))
                              for o in DIV_OUTS} for c in DIV_CONDS},
           "timing": {"t_ref_mean": _avg(float(r["t_ref"]) for r in rows), "t_vs_mean": _avg(float(r["t_vs"]) for r in rows),
                      "rtf_ref": _avg(float(r["t_ref"]) / float(r["audio_seconds"]) for r in rows),
                      "rtf_vs": _avg(float(r["t_vs"]) / float(r["audio_seconds"]) for r in rows)} if rows else {},
           "checks": {"max_e4_abs_diff": max((float(r["e4_max_abs_diff"]) for r in rows), default=None),
                      "gpus": sorted({r["gpu"] for r in rows}),
                      "voice_rms": {c: _avg(float(r["voice_rms"]) for r in rows if r["condition"] == c) for c in CONDS}}}
    if rows:
        deg = sorted((r for r in rows if r["condition"] in DEGRADED), key=lambda r: (r["utt_id"], r["condition"]))
        fail = lambda r, o: bool(gl.failed({m: float(r[f"{m}_{o}"]) for m in ("wer", "similarity")},
                                           {m: float(r[f"{m}_ref"]) for m in ("wer", "similarity")}, th))
        a, b, p = rf.mcnemar_exact([fail(r, "plain4") for r in deg], [fail(r, "vs") for r in deg])
        res["mcnemar_vs_against_plain4_degraded"] = {"only_plain4": a, "only_vs": b, "p": p}
    run_info = out / "run_info.json"
    res["run_info"] = json.loads(run_info.read_text(encoding="utf-8")) if run_info.exists() else None
    (out / "confirm_results.json").write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
    (out / "CONFIRMATION_REPORT.md").write_text(render(res), encoding="utf-8")
    print(f"Wrote {out / 'CONFIRMATION_REPORT.md'}", flush=True)
    return res


def _word(v):
    return "n/a" if v is None else ("PASS" if v else "FAIL")


def _ci(v, fmt="{:+.4f}"):
    return "n/a" if not v or v.get("mean") is None else f"{fmt.format(v['mean'])} [{fmt.format(v['lo'])}, {fmt.format(v['hi'])}]"


def render(res):
    p = res["primary"]
    d = p["measured"]
    L = ["# Confirmation test: Voice-Swap Handover (EXP-021)", "",
         f"{'**PARTIAL.** ' if res['partial'] else ''}Fresh development speakers 61-100; {res['cases']} of {res['expected_cases']} cases, "
         f"{res['diversity_pairs']} of {res['expected_diversity_pairs']} diversity pairs; preregistration `{res['prereg_sha256'][:16]}…`. "
         "Every number is computed from the CSV files by `run_voice_swap_confirm.py analyze`.", "",
         "Voice-Swap (vs) hands over at Step 4 after taking the voice part from a one-pass student draft (14 passes); the reference "
         "(ref) is the 0.25 switch, Step 8 (21 passes). vs and ref each average two noise draws.", "",
         "## Preregistered criteria", "",
         "| P1 word errors | P2 voice similarity | P3 pitch variety | P4 time | Confirmed |", "|---|---|---|---|---|",
         "| " + " | ".join(_word(v) for v in p["criteria"].values()) + f" | **{'YES' if p['confirmed'] else 'NO'}** |", "",
         "| Quantity (vs minus ref unless stated) | Mean [95% speaker CI] | Requirement |", "|---|---|---|",
         f"| WER, clean | {_ci(d['wer_clean'])} | upper bound <= +{MARGINS['wer_clean']} |",
         f"| WER, degraded prompts pooled | {_ci(d['wer_degraded'])} | upper bound <= +{MARGINS['wer_degraded']} |",
         f"| Similarity, clean | {_ci(d['similarity_clean'])} | lower bound >= -{MARGINS['similarity']} |",
         f"| Similarity, degraded prompts pooled | {_ci(d['similarity_degraded'])} | lower bound >= -{MARGINS['similarity']} |"]
    for c in DIV_CONDS:
        L += [f"| CVf0 vs minus ref, {c} | {_ci(d['cvf0'][c]['vs_minus_ref'])} | lower bound >= -{MARGINS['cvf0']} |",
              f"| CVf0 vs minus student, {c} | {_ci(d['cvf0'][c]['vs_minus_student'])} | lower bound > 0 |"]
    L += [f"| Time ratio t_vs / t_ref | {_ci(d['time_ratio'], '{:.3f}')} | upper bound <= {MARGINS['time_ratio']} |", ""]
    if res["timing"]:
        t = res["timing"]
        L += [f"Mean sampling time (teacher, student and vocoder): ref {t['t_ref_mean']:.3f} s, vs {t['t_vs_mean']:.3f} s; "
              f"real-time factor ref {t['rtf_ref']:.3f}, vs {t['rtf_vs']:.3f}.", ""]
    L += ["## Per condition (descriptive): vs minus ref, mean [95% speaker CI]", "",
          "| Condition | WER | Similarity | UTMOS |", "|---|---|---|---|"]
    L += [f"| {c} | {_ci(res['per_condition'][c]['wer'])} | {_ci(res['per_condition'][c]['similarity'])} | {_ci(res['per_condition'][c]['utmos'], '{:+.3f}')} |" for c in CONDS]
    for key, title, fmt in (("wer", "Mean WER (capped at 1 per utterance)", "{:.3f}"), ("similarity", "Mean speaker similarity", "{:.3f}"), ("utmos", "Mean UTMOS", "{:.2f}")):
        L += ["", f"## {title}", "", "| Output | Passes | " + " | ".join(CONDS) + " |", "|---|---:|" + "---:|" * len(CONDS)]
        for o in OUTS:
            L.append(f"| {o} | {PASSES[o]} | " + " | ".join(fmt.format(res["means"][c][o][key]) if res["means"][c][o][key] is not None else "n/a" for c in CONDS) + " |")
    if res["failures"]:
        L += ["", "## Added failures against ref (Definition 3.1, descriptive)", "",
              "ref2 is the reference with another student noise draw: its row is the failure rule's own noise floor.", "",
              "| Output | " + " | ".join(CONDS) + " |", "|---|" + "---:|" * len(CONDS)]
        L += [f"| {o} | " + " | ".join(str(res["failures"][o][c]) for c in CONDS) + " |" for o in OUTS[1:]]
        m = res["mcnemar_vs_against_plain4_degraded"]
        L += ["", f"Degraded prompts, exact McNemar of plain4 against vs: only plain4 fails {m['only_plain4']}, only vs fails {m['only_vs']}, p = {m['p']:.3g}."]
    L += ["", "## Voice-Swap against the student alone (descriptive): vs minus student", "", "| Prompts | WER | Similarity | UTMOS |", "|---|---|---|---|"]
    L += [f"| {g} | {_ci(v['wer'])} | {_ci(v['similarity'])} | {_ci(v['utmos'], '{:+.3f}')} |" for g, v in res["vs_minus_student"].items()]
    if any(v is not None for c in DIV_CONDS for v in res["cvf0_means"][c].values()):
        L += ["", "## Pitch variety (CVf0 means)", "", "| Condition | " + " | ".join(DIV_OUTS) + " |", "|---|" + "---:|" * len(DIV_OUTS)]
        L += [f"| {c} | " + " | ".join(f"{res['cvf0_means'][c][o]:.4f}" if res["cvf0_means"][c][o] is not None else "n/a" for o in DIV_OUTS) + " |" for c in DIV_CONDS]
    ck = res["checks"]
    L += ["", "## Checks", "",
          f"- Largest difference between the Step-4 teacher estimates of the two timed paths: {ck['max_e4_abs_diff']} (0 means the paths are identical).",
          f"- GPUs: {ck['gpus']}. DEMAND babble SHA-256: {(res['run_info'] or {}).get('babble_sha256', 'n/a')}.", "",
          "Speakers 61-100 were not used to design Voice-Swap, but they are development speakers of this project (EXP-001/002 "
          "synthesized them for the problem grid). This is not the thesis's frozen held-out evaluation."]
    return "\n".join(L) + "\n"


# ------------------------------------------------------------------------ preregistration
def input_hash(path):
    """SHA-256 with line endings normalized: text inputs are LF on Colab but may be CRLF in a Windows checkout."""
    return sh.sha256_text(path)


def code_hashes():
    return {f"code/colab/{n}": sh.sha256_text(Path(__file__).resolve().parent / n) for n in DEPENDENCIES}


def build_prereg():
    items = fresh_items()
    return {"experiment": "EXP-021",
            "purpose": "Frozen confirmation, on speakers never used to design it, that Voice-Swap Handover at Step 4 matches the 0.25 reference (Step 8) at two-thirds of its forward passes",
            "claim_under_test": "Voice-Swap at Step 4 (14 passes) is not worse than the reference at Step 8 (21 passes) beyond the margins below on word errors and speaker similarity, keeps the teacher's pitch variety, and takes at most 80% of the reference's sampling time on a T4",
            "origin": "EXP-020 (preregistration 7afeef06...) did not advance Selective Handover's adaptive ladder; the comparison of fixed Voice-Swap with the reference was computed after its results were seen, so it is tested here before any claim",
            "algorithm": {"draft": "S = eps + g(eps, 0), one student pass, eps from seed + 1009 (sh.student_sample)",
                          "voice_swap": "on generated frames, the teacher's Step-4 endpoint estimate keeps its slow and fine parts and takes the draft's voice part (per-band mean over time) (sh.selective_handover, code STT, slow window 9 frames); prompt frames unchanged",
                          "handover": "the released student loop at j = 1 (taus 0.0761, 0.2929, 0.6173) runs from that input",
                          "source": "imported unchanged from code/colab/run_selective_handover_prelim.py"},
            "outputs": {"ref": "Step 8 (the 0.25 switch), student seeds as released", "ref2": f"Step 8, student seeds + {REPLICATE_OFFSET}",
                        "vs": "Voice-Swap at Step 4", "vs2": f"Voice-Swap at Step 4, draft and student seeds + {REPLICATE_OFFSET}",
                        "plain4": "today's Step-4 handover (descriptive)", "student": "student only (descriptive)"},
            "passes": PASSES,
            "data": {"quality": f"development speakers {DEV_N + 1}-{DEV_N + FRESH_N} by utterance ID (40 speakers) x {list(CONDS)}",
                     "fresh_utt_ids": [it["utt_id"] for it in items],
                     "diversity": f"the first {DIV_N} fresh speakers x {list(DIV_CONDS)}, {DIV_SEEDS} seeds (input_seed + {DIV_SEED_BASE} + m), prompt degradation fixed per pair",
                     "conditions": {c: rf.CONDITIONS[c] for c in CONDS},
                     "babble": {"file": rf.DEMAND_FILE, "source": rf.DEMAND_URL, "hash": "recorded at run time in run_info.json; the full run refuses a file that differs from the smoke test's"},
                     "history": "speakers 61-100 were synthesized for the EXP-001/EXP-002 problem grid (clean and noisy15, Steps 4/5/6/8) but never used to design Voice-Swap or any EXP-012 to EXP-020 method; EXP-019 also names them as its fresh set"},
            "timing": "the reference path (teacher to Step 8, student, vocoder) and the Voice-Swap path (teacher to Step 4, draft, swap, student, vocoder) are each run and timed separately with CUDA synchronized; the order alternates between cases; one untimed warm-up case runs first; prompt preparation and scoring are excluded",
            "hardware": "Colab Tesla T4 only, enforced by the runner; Shonget-DMOSpeech2 at the pinned commit",
            "dmo_commit": DMO_COMMIT,
            "metrics": "WER (Whisper via the released metric suite, capped at 1 per utterance in all comparisons), speaker similarity to the ground-truth recording, UTMOS",
            "diversity_measure": {"name": "CVf0", "definition": "as in EXP-020: librosa pYIN 65-500 Hz, frame 1024, hop 256 at 24 kHz; per frame std/mean (ddof 1) across seeds over frames voiced in at least half of the seeds; averaged over frames"},
            "criteria": {"P1_word_errors": f"per case, the mean capped WER of vs and vs2 minus that of ref and ref2; speaker-bootstrap 95% CI upper bound <= +{MARGINS['wer_clean']} on clean prompts and <= +{MARGINS['wer_degraded']} on the five degraded conditions pooled",
                         "P2_voice_similarity": f"the same paired difference in similarity: lower bound >= -{MARGINS['similarity']} on clean prompts and on the degraded conditions pooled",
                         "P3_pitch_variety": f"for clean and noisy15 separately: CVf0 of vs minus ref lower bound >= -{MARGINS['cvf0']}, and vs minus student lower bound > 0",
                         "P4_time": f"per case t_vs / t_ref: speaker-bootstrap upper bound <= {MARGINS['time_ratio']}"},
            "decision_rule": "Voice-Swap Handover is confirmed on these speakers only if P1 to P4 all pass; otherwise it is reported as a negative result, with no retuning and no change of margins on these data",
            "descriptive": ["per-condition differences with CIs", "means of every output", "Definition 3.1 failures of every output against ref, with ref2 as the noise floor",
                            "exact McNemar of plain4 against vs on degraded prompts", "vs against student only", "real-time factors",
                            "the largest difference between the two timed paths' Step-4 estimates (expected 0)"],
            "bootstrap": {"B": BOOT_B, "seed": SEED, "unit": "speaker", "interval": "percentile, sorted index int(0.025 B) and int(0.975 B)"},
            "design_notes": "Margins were sized before any fresh data from the EXP-020 development data by resampling 40-speaker studies: a method equal to EXP-020's estimate passes the degraded WER margin +0.025 with probability about 0.8 using single draws (the two draws used here should raise it), passes +0.01 or +0.015 with probability 0.2-0.4, passes the clean margins with probability about 1, and passes the pitch-variety margin with probability about 0.76 at 20 pairs (30 are used here). The expected time ratio from EXP-018's T4 timings is about 0.71",
            "evidence_status": "confirmation on development speakers not used to design the method; not the thesis's frozen held-out evaluation (EVAL-001/EVAL-002); white-noise, recorded babble, synthetic reverberation and telephone-band prompts",
            "novelty": "not tested by this experiment; to our knowledge no published handover selects which parts of the teacher's estimate cross to the student (targeted searches, 2026-11-08 to 2026-11-13); a full literature review is outstanding",
            "adoption": "not adopted; changing the thesis method needs a dated decision record and adviser review",
            "code_hashes": code_hashes(),
            "input_hashes": {str(p.relative_to(THESIS_ROOT)).replace("\\", "/"): input_hash(p)
                             for p in (STEPS_CSV, GRID32_CSV, EXP020_CSV, rf.SPLITS_DIR / "development_manifest.json",
                                       THESIS_ROOT / "code/scheduling/failure_config.json")},
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
    now = code_hashes()
    changed = [k for k, v in pre["code_hashes"].items() if now.get(k) != v]
    if changed:
        raise SystemExit(f"Code differs from the version frozen in preregistration.json: {changed}")
    moved = [k for k, v in pre["input_hashes"].items() if not (THESIS_ROOT / k).exists() or input_hash(THESIS_ROOT / k) != v]
    if moved:
        raise SystemExit(f"Input files are missing or differ from the frozen versions: {moved}")
    return pre


def run_info(pipe, pre):
    return {"gpu": pipe.gpu, "dmo_commit": rf.git_head(pipe_root(pipe)), "thesis_commit": rf.git_head(THESIS_ROOT),
            "babble_sha256": getattr(pipe, "babble_sha", None), "prereg_sha256": pre["sha256"],
            "utc": sh._utc()}


def pipe_root(pipe):
    return getattr(pipe, "dmo_root", None) or "."


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["freeze", "smoke", "run", "analyze"])
    ap.add_argument("--out")
    ap.add_argument("--dmo_root", default="/content/Shonget-DMOSpeech2")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--demand_dir", help="folder holding PCAFETER/ch01.wav (DEMAND cafeteria babble, 16 kHz)")
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
    pre = check_prereg()
    if not args.demand_dir:
        ap.error("--demand_dir is required: the babble15 condition uses the DEMAND cafeteria recording")
    head = rf.git_head(args.dmo_root)
    if head != DMO_COMMIT:
        raise SystemExit(f"Shonget-DMOSpeech2 is at {head}, not the pinned {DMO_COMMIT}.")
    pipe = rf.Pipeline(args.dmo_root, args.device)
    pipe.dmo_root = args.dmo_root
    if "T4" not in pipe.gpu:
        raise SystemExit(f"EXP-021 is defined on the Colab Tesla T4 (timing and student noise depend on the GPU); found {pipe.gpu}.")
    pipe.load_babble(args.demand_dir)
    if args.cmd == "smoke":
        sm = out / "smoke"
        sm.mkdir(exist_ok=True)
        dev2 = [it for it in ex.items() if it["utt_id"] not in cg.pilot_utts()][:2]
        repro = run_quality(pipe, dev2, ("clean", "gauss20"), sm, stored=stored_labels())
        (sm / "reproduction.json").write_text(json.dumps(repro, indent=2) + "\n", encoding="utf-8")
        worst_wer, worst_sim = sh.check_reproduction(repro)
        run_quality(pipe, dev2[:1], ("babble15", "reverb09", "telephone"), sm)
        run_diversity(pipe, ex.items()[:1], ("clean",), sm, seeds=2)
        (sm / "run_info.json").write_text(json.dumps(run_info(pipe, pre), indent=2) + "\n", encoding="utf-8")
        print(f"Smoke run complete: {sm} (stored labels reproduced for ref, plain4, vs and student: max |dWER| {worst_wer:.4f}, "
              f"max |dSIM| {worst_sim:.4f}; development speakers only)", flush=True)
        return
    smoke = out / "smoke"
    if not (smoke / "reproduction.json").exists() or not (smoke / "run_info.json").exists():
        raise SystemExit("Run the smoke test first: the full run starts only after the stored labels reproduce.")
    sh.check_reproduction(json.loads((smoke / "reproduction.json").read_text(encoding="utf-8")))
    smoke_info = json.loads((smoke / "run_info.json").read_text(encoding="utf-8"))
    if smoke_info.get("babble_sha256") != pipe.babble_sha:
        raise SystemExit("The DEMAND babble file differs from the one used in the smoke test.")
    shutil.copy2(PREREG_PATH, out / "preregistration.json")
    info_path = out / "run_info.json"
    if not info_path.exists():
        info_path.write_text(json.dumps(run_info(pipe, pre), indent=2) + "\n", encoding="utf-8")
    items = fresh_items()
    run_quality(pipe, items, CONDS, out)
    run_diversity(pipe, items[:DIV_N], DIV_CONDS, out)
    analyze(out)


if __name__ == "__main__":
    main()
