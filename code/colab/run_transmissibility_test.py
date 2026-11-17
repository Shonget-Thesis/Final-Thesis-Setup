"""Test of the transmissibility theory of teacher-to-student handover (EXP-026): development speakers only.

Theory (docs/progress/log/2026-11-14-exp026-transmissibility-test.md). Along one prosodic direction with two modes at +/-m,
the teacher's belief in the dominant mode after Step k has log-odds l_k = log(pi/(1-pi)) + 2 m t x_t / (1-t)^2 and its
estimate is m tanh(l_k/2). The handover passes y = tau * estimate + (1 - tau) * fresh noise; a mode-seeking student with
prior log-odds beta (> log(pi/(1-pi))) keeps the dominant mode iff beta + A tanh(l_k/2) + B eps' > 0, with
A = 2 m^2 tau^2 / (1-tau)^2 and B = 2 m tau / (1-tau). A minority mode the teacher has fully committed to survives with
probability R* = Phi((A - beta) / B): a ceiling that more teacher steps cannot raise, set by the restart level tau and the
student's bias. Predictions tested here, on clean prompts, with every arm sharing the same teacher trajectory per seed:

  H1  the 0.25 reference (Step 8, student restarted at tau1 = 0.0761) has less pitch variety than the pure teacher run to
      the end on the same trajectory (T16);
  H2  restarting the student at tau2 = 0.2929 from the same Step-8 estimate (W8R2) keeps more pitch variety than the
      reference, and lands closer to the pure teacher;
  H3  at tau1, more teacher steps do not add variety once the ceiling is reached: Step 4 and Step 8 handovers agree.

Pitch variety is CVf0 across 20 seeds (EXP-020's seeds) on 20 development speakers; quality of the new arms is scored on
40 speakers against the stored reference (EXP-018). Development evidence only; a test of a theory, not of a method.

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
EXP_DIR = THESIS_ROOT / "research/experiments/EXP-026-transmissibility-test"
PREREG_PATH = EXP_DIR / "preregistration.json"
STEPS_CSV = sh.STEPS_CSV
EXP020_DIV = THESIS_ROOT / "research/experiments/EXP-020-selective-handover-prelim/results/diversity.csv"
FAILURE_CONFIG = THESIS_ROOT / "code/scheduling/failure_config.json"
MANIFEST = rf.SPLITS_DIR / "development_manifest.json"
DMO_COMMIT = "8a7868fc3345702a586173447f4869205b501f26"
DEPENDENCIES = ("run_transmissibility_test.py", "run_selective_handover_prelim.py", "run_grid32_prelim.py",
                "run_noise_guidance_prelim.py", "run_followup_tests.py", "run_consistency_gate_prelim.py",
                "run_endpoint_extrapolation_prelim.py")

COND = "clean"
TEACHER_END = 16                       # the teacher run to t = 1 on the released 16-interval grid
ARMS = ("W4", "W8", "W4R2", "W8R2", "T16")
NEW_ARMS = ("W4R2", "W8R2", "T16")     # not stored before; scored for quality
DIV_N, DIV_SEEDS, DIV_SEED_BASE = 20, 20, 1000
Q_N = 40
METRICS = ("wer", "similarity", "utmos")
BOOT_B, BOOT_SEED = 2000, 2026
TEACHER_TOLERANCE = 1e-4
SATURATION = 0.005                     # H3: |CVf0(W8) - CVf0(W4)| at most this
DIV_FIELDS = ["utt_id", "speaker_id", "condition", "n_seeds"] + [f"cvf0_{a}" for a in ARMS] + [f"voiced_{a}" for a in ARMS] + \
             ["case_seconds", "utc"]
Q_FIELDS = ["utt_id", "speaker_id", "condition", "prompt_snr", "gpu"] + [f"{m}_{a}" for a in NEW_ARMS for m in METRICS + ("hypothesis",)] + \
           ["case_seconds", "utc"]
SMOKE_FIELDS = ["teacher_loop_max_abs_diff"] + [f"{m}_{a}" for a in ("W4", "W8") for m in METRICS + ("hypothesis",)]


def cost(arm):
    """Forward passes: 2 per guided teacher evaluation, then the student's passes (3 from tau1, 2 from tau2); the pure
    teacher evaluates Steps 0-15 and ends at t = 1 with no student."""
    if arm == "T16":
        return 2 * TEACHER_END
    k = int(arm[1])
    return 2 * (k + 1) + (2 if arm.endswith("R2") else 3)


def div_items():
    return ex.items()[:DIV_N]


def q_items():
    return ex.items()[:Q_N]


def smoke_items():
    return ex.items()[58:60]


def _maxabs(x):
    return float(abs(x).max())


def grid16(pipe):
    g = list(pipe.sampler.SWAY_TEACHER_GRID)
    full = gl.sway_grid(TEACHER_END)
    return g + full[len(g):] if len(g) < TEACHER_END + 1 else g


def teacher_run(pipe, ctx):
    """The released guided Euler teacher on the 16-interval grid, run to t = 1. Returns ({4: T_4, 8: T_8}, the final
    state at t = 1, which is the pure teacher's sample)."""
    torch, sampler = pipe.torch, pipe.sampler
    grid = grid16(pipe)
    x, ends = sh._clone(ctx["y0"]), {}
    with torch.inference_mode():
        for i in range(TEACHER_END):
            t = torch.tensor(grid[i], device=pipe.device, dtype=torch.float32)
            kw = dict(x=x, cond=ctx["step_cond"], text=ctx["text_ids"], time=t, mask=None)
            vc = pipe.tts.teacher(drop_audio_cond=False, drop_text=False, **kw)
            vu = pipe.tts.teacher(drop_audio_cond=True, drop_text=True, **kw)
            v = vc + (vc - vu) * sampler.CFG_STRENGTH
            if i in (4, 8):
                ends[i] = x + (1.0 - grid[i]) * v
            x = x + (grid[i + 1] - grid[i]) * v
    return ends, x


def outputs(pipe, ctx, ends, final):
    """Mel outputs of the five arms from one teacher trajectory."""
    taus = list(pipe.sampler.STUDENT_TAUS)
    return {"W4": gl.student(pipe, ctx, ends[4], taus), "W8": gl.student(pipe, ctx, ends[8], taus),
            "W4R2": gl.student(pipe, ctx, ends[4], taus[1:]), "W8R2": gl.student(pipe, ctx, ends[8], taus[1:]), "T16": final}


# --------------------------------------------------------------------------------- runs
def run_diversity(pipe, todo_items, out_dir, seeds=DIV_SEEDS, csv_name="diversity.csv"):
    path = Path(out_dir) / csv_name
    done = rf.done_keys(path, 1)
    todo = [it for it in todo_items if (it["utt_id"], COND) not in done]
    print(f"[diversity] {len(todo)} speakers x {seeds} seeds to run ({len(done)} already in {path.name}) on {pipe.gpu}", flush=True)
    for n, item in enumerate(todo, 1):
        t0 = time.perf_counter()
        tracks = {a: [] for a in ARMS}
        for m in range(seeds):
            ctx = pipe.prepare(item, COND, int(item["input_seed"]) + DIV_SEED_BASE + m, out_dir, noise_seed=item["input_seed"])
            ends, final = teacher_run(pipe, ctx)
            for a, mel in outputs(pipe, ctx, ends, final).items():
                tracks[a].append(sh.f0_track(ng.decode(pipe, ctx, mel)))
        row = {"utt_id": item["utt_id"], "speaker_id": item["speaker_id"], "condition": COND, "n_seeds": seeds}
        for a in ARMS:
            row[f"cvf0_{a}"], row[f"voiced_{a}"] = sh.cvf0(tracks[a])
        row.update({"case_seconds": round(time.perf_counter() - t0, 2), "utc": sh._utc()})
        sh._append(path, DIV_FIELDS, row)
        print(f"[{n}/{len(todo)}] diversity: {item['utt_id']} ({row['case_seconds']:.0f} s)", flush=True)
    print(f"[OK] diversity complete: {path}", flush=True)


def stored_w():
    return {(r["utt_id"], r["condition"]): {f"W{k}": {x: float(r[f"{x}_g16_{k}"]) for x in METRICS} for k in (4, 8)}
            for r in rf.read_csv(STEPS_CSV)}


def run_quality(pipe, todo_items, out_dir, csv_name="quality.csv", smoke=False, stored=None):
    path = Path(out_dir) / csv_name
    fields = Q_FIELDS + (SMOKE_FIELDS if smoke else [])
    done = rf.done_keys(path, 1)
    todo = [it for it in todo_items if (it["utt_id"], COND) not in done]
    print(f"[quality] {len(todo)} cases to run ({len(done)} already in {path.name})", flush=True)
    repro, diffs = [], []
    for n, item in enumerate(todo, 1):
        t0 = time.perf_counter()
        ctx = pipe.prepare(item, COND, item["input_seed"], out_dir)
        ends, final = teacher_run(pipe, ctx)
        mels = outputs(pipe, ctx, ends, final)
        gt = pipe.ground_truth(item)
        row = {"utt_id": item["utt_id"], "speaker_id": item["speaker_id"], "condition": COND, "prompt_snr": ctx["prompt_snr"], "gpu": pipe.gpu}
        for a in NEW_ARMS + (("W4", "W8") if smoke else ()):
            m = pipe.score(ng.decode(pipe, ctx, mels[a]), item, gt)
            row.update({f"{x}_{a}": m[x] for x in METRICS + ("hypothesis",)})
        if smoke:
            released, _ = pipe.teacher(ctx, (4, 8))
            with pipe.torch.inference_mode():
                row["teacher_loop_max_abs_diff"] = max(_maxabs(released[k] - ends[k]) for k in (4, 8))
            diffs.append(row["teacher_loop_max_abs_diff"])
            key = (item["utt_id"], COND)
            if stored is not None and key in stored:
                for a in ("W4", "W8"):
                    s = stored[key][a]
                    repro.append({"utt_id": item["utt_id"], "condition": COND, "output": a, "wer": row[f"wer_{a}"], "wer_stored": s["wer"],
                                  "similarity": row[f"similarity_{a}"], "similarity_stored": s["similarity"]})
        row.update({"case_seconds": round(time.perf_counter() - t0, 2), "utc": sh._utc()})
        sh._append(path, fields, row)
        print(f"[{n}/{len(todo)}] quality: {item['utt_id']} ({row['case_seconds']:.0f} s)", flush=True)
    print(f"[OK] quality complete: {path}", flush=True)
    return repro, diffs


# ----------------------------------------------------------------------------- analysis
def boot(groups, B=BOOT_B, seed=BOOT_SEED):
    values = [sum(v) / len(v) for v in groups.values() if v]
    if not values:
        return {"mean": None, "lo": None, "hi": None, "n": 0}
    rng = random.Random(seed)
    boots = sorted(sum(rng.choice(values) for _ in values) / len(values) for _ in range(B))
    return {"mean": sum(values) / len(values), "lo": boots[int(0.025 * B)], "hi": boots[int(0.975 * B)], "n": len(values)}


def _num(v):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(x) else x


def stored_cvf0():
    """speaker -> EXP-020's stored clean CVf0 of the student alone and of the released Step-4 and Step-8 handovers."""
    return {r["speaker_id"]: {"student4": _num(r["cvf0_student4"]), "W4": _num(r["cvf0_plain_4"]), "W8": _num(r["cvf0_plain_8"])}
            for r in rf.read_csv(EXP020_DIV) if r["condition"] == "clean"}


def spearman(a, b):
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        for pos, i in enumerate(order):
            r[i] = float(pos)
        return r
    if len(a) < 3:
        return None
    ra, rb = ranks(a), ranks(b)
    ma, mb = sum(ra) / len(ra), sum(rb) / len(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = math.sqrt(sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb))
    return num / den if den else None


def diversity_summary(div_rows, stored):
    cv = {r["speaker_id"]: {a: _num(r[f"cvf0_{a}"]) for a in ARMS} for r in div_rows}
    pairs = {"T16_minus_W8": ("T16", "W8"), "W8R2_minus_W8": ("W8R2", "W8"), "W8_minus_W4": ("W8", "W4"),
             "W4R2_minus_W4": ("W4R2", "W4"), "T16_minus_W8R2": ("T16", "W8R2")}
    out = {"mean_cvf0": {a: (lambda v: sum(v) / len(v) if v else None)([c[a] for c in cv.values() if c[a] is not None]) for a in ARMS}}
    for name, (a, b) in pairs.items():
        out[name] = boot({s: [c[a] - c[b]] for s, c in cv.items() if c[a] is not None and c[b] is not None})
    gap_ref = [abs(c["T16"] - c["W8"]) for c in cv.values() if c["T16"] is not None and c["W8"] is not None]
    gap_r2 = [abs(c["T16"] - c["W8R2"]) for c in cv.values() if c["T16"] is not None and c["W8R2"] is not None]
    out["mean_abs_gap_to_teacher"] = {"W8": sum(gap_ref) / len(gap_ref) if gap_ref else None,
                                      "W8R2": sum(gap_r2) / len(gap_r2) if gap_r2 else None}
    rep = [(abs(c[a] - stored[s][a]), s, a) for s, c in cv.items() if s in stored for a in ("W4", "W8")
           if c[a] is not None and stored[s][a] is not None]
    out["reproduction_max_abs"] = max(r[0] for r in rep) if rep else None
    both = [(c["T16"] - c["W8"], c["T16"] - stored[s]["student4"]) for s, c in cv.items()
            if s in stored and None not in (c["T16"], c["W8"], stored[s]["student4"])]
    out["H4_spearman_gap_vs_student_deficit"] = spearman([x for x, _ in both], [y for _, y in both]) if both else None
    return out


def quality_summary(q_rows, stored):
    out = {}
    for a in NEW_ARMS:
        g = {m: defaultdict(list) for m in METRICS}
        for r in q_rows:
            s = stored.get((r["utt_id"], r["condition"]))
            if s is None or _num(r.get(f"wer_{a}")) is None:
                continue
            for m in METRICS:
                x, y = float(r[f"{m}_{a}"]), s["W8"][m]
                g[m][r["speaker_id"]].append((min(1.0, x) - min(1.0, y)) if m == "wer" else x - y)
        out[a] = {m: boot(g[m]) for m in METRICS}
    return out


def analyze(out, allow_partial=False, prereg_path=PREREG_PATH):
    out = Path(out)
    pre = check_prereg(prereg_path)
    read = lambda name: rf.read_csv(out / name, repair=True) if (out / name).exists() else []
    div_rows, q_rows = read("diversity.csv"), read("quality.csv")
    have = {"diversity_speakers": len(div_rows), "quality_cases": len(q_rows)}
    expected = {"diversity_speakers": DIV_N, "quality_cases": Q_N}
    partial = any(have[k] < expected[k] for k in expected)
    if partial and not allow_partial:
        raise SystemExit(f"incomplete run {have} of {expected}; finish the run or pass --allow_partial")
    d = diversity_summary(div_rows, stored_cvf0())
    res = {"prereg_sha256": pre["sha256"], "have": have, "expected": expected, "partial": partial,
           "passes": {a: cost(a) for a in ARMS}, "diversity": d, "quality": quality_summary(q_rows, stored_w())}
    complete = not partial
    h1 = None if not complete or d["T16_minus_W8"]["lo"] is None else bool(d["T16_minus_W8"]["lo"] > 0.0)
    g = d["mean_abs_gap_to_teacher"]
    h2 = None if not complete or d["W8R2_minus_W8"]["lo"] is None or None in g.values() else \
        bool(d["W8R2_minus_W8"]["lo"] > 0.0 and g["W8R2"] < g["W8"])
    h3 = None if not complete or d["W8_minus_W4"]["mean"] is None else bool(abs(d["W8_minus_W4"]["mean"]) <= SATURATION)
    res["hypotheses"] = {"H1_reference_below_teacher": h1, "H2_later_restart_recovers": h2, "H3_saturation_at_tau1": h3}
    res["verdict"] = None if not complete else ("supported" if h1 and h2 else ("refuted" if h1 is False else "mixed"))
    (out / "transmissibility_results.json").write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
    (out / "TRANSMISSIBILITY_REPORT.md").write_text(render(res), encoding="utf-8")
    print(f"Wrote {out / 'TRANSMISSIBILITY_REPORT.md'}", flush=True)
    return res


def _word(v):
    return "n/a" if v is None else ("YES" if v else "NO")


def _ci(v, fmt="{:+.4f}"):
    return "n/a" if not v or v.get("mean") is None else f"{fmt.format(v['mean'])} [{fmt.format(v['lo'])}, {fmt.format(v['hi'])}]"


def _f(v, fmt="{:.4f}"):
    return "n/a" if v is None else fmt.format(v)


def render(res):
    d, h = res["diversity"], res["hypotheses"]
    L = ["# Test: transmissibility theory of teacher-to-student handover (EXP-026)", "",
         f"{'**PARTIAL.** ' if res['partial'] else ''}Development speakers only, clean prompts; {res['have']} of {res['expected']}; "
         f"preregistration `{res['prereg_sha256'][:16]}…`. Every number is computed from the CSV files by `run_transmissibility_test.py analyze`. "
         "This tests a theory's predictions; it is not evidence for a method.", "",
         "Arms share one teacher trajectory per seed: W4 / W8 = student restarted at tau = 0.0761 from Step 4 / Step 8 (W8 is the 0.25 "
         "reference); W4R2 / W8R2 = restarted at tau = 0.2929; T16 = the teacher alone run to the end.", "",
         f"## Verdict: **{res['verdict'] or 'n/a'}**", "",
         f"- H1, the reference has less pitch variety than the pure teacher: **{_word(h['H1_reference_below_teacher'])}** "
         f"(T16 - W8 {_ci(d['T16_minus_W8'])})",
         f"- H2, restarting at tau = 0.29 recovers variety: **{_word(h['H2_later_restart_recovers'])}** (W8R2 - W8 {_ci(d['W8R2_minus_W8'])}; "
         f"mean |gap to the teacher|: W8 {_f(d['mean_abs_gap_to_teacher']['W8'])}, W8R2 {_f(d['mean_abs_gap_to_teacher']['W8R2'])})",
         f"- H3, at tau = 0.0761 Step 4 and Step 8 agree (saturation): **{_word(h['H3_saturation_at_tau1'])}** (W8 - W4 {_ci(d['W8_minus_W4'])})",
         f"- H4 (exploratory), speakers whose student loses more variety also lose more at the reference: Spearman "
         f"{_f(d['H4_spearman_gap_vs_student_deficit'], '{:+.2f}')}", "",
         "## Pitch variety (CVf0, mean over speakers)", "", "| Arm | Passes | CVf0 |", "|---|---:|---:|"]
    for a in ARMS:
        L.append(f"| {a} | {res['passes'][a]} | {_f(d['mean_cvf0'][a])} |")
    L += ["", f"W4R2 - W4: {_ci(d['W4R2_minus_W4'])}; T16 - W8R2: {_ci(d['T16_minus_W8R2'])}. Reproduction of EXP-020's stored W4 and W8 "
          f"CVf0, max |difference|: {_f(d['reproduction_max_abs'], '{:.2e}')}.", "",
          "## Quality of the new arms against the stored reference (40 speakers, speaker bootstrap, 95%)", "",
          "| Arm | Passes | WER | Similarity | UTMOS |", "|---|---:|---|---|---|"]
    for a, q in res["quality"].items():
        L.append(f"| {a} | {res['passes'][a]} | {_ci(q['wer'])} | {_ci(q['similarity'])} | {_ci(q['utmos'], '{:+.3f}')} |")
    L += ["", "Development evidence only (speakers that designed earlier experiments), clean prompts."]
    return "\n".join(L) + "\n"


# ------------------------------------------------------------------------ preregistration
def code_hashes():
    return {f"code/colab/{n}": sh.sha256_text(Path(__file__).resolve().parent / n) for n in DEPENDENCIES}


def build_prereg():
    return {"experiment": "EXP-026",
            "purpose": "Test the transmissibility theory of teacher-to-student handover: a mode-seeking student restarted at level tau "
                       "keeps a committed teacher's minority prosodic mode with probability at most Phi((A - beta) / B), a ceiling that "
                       "more teacher steps cannot raise and that a later restart level raises",
            "theory": "two prosodic modes at +/-m with dominant weight pi; teacher log-odds l_k = log(pi/(1-pi)) + 2 m t x_t/(1-t)^2, "
                      "estimate m tanh(l_k/2); handover y = tau * estimate + (1-tau) * fresh noise; student keeps the dominant mode iff "
                      "beta + A tanh(l_k/2) + B eps' > 0 with A = 2 m^2 tau^2/(1-tau)^2, B = 2 m tau/(1-tau); retention ceiling "
                      "R* = Phi((A - beta)/B)",
            "arms": {"W4": "student restarted at tau1 = 0.0761 from the Step-4 estimate (13 passes)",
                     "W8": "student restarted at tau1 from the Step-8 estimate: the 0.25 reference (21)",
                     "W4R2": "student restarted at tau2 = 0.2929 from the Step-4 estimate (12)",
                     "W8R2": "student restarted at tau2 from the Step-8 estimate (20)",
                     "T16": "the guided teacher alone, run to t = 1 on the released 16-interval grid (32)",
                     "shared": "all five arms come from one teacher trajectory per seed; the student loop is the released one "
                               "(STUDENT_TAUS for tau1, STUDENT_TAUS[1:] for tau2)"},
            "passes": {a: cost(a) for a in ARMS},
            "data": {"diversity": f"the first {DIV_N} EXP-012 Quick-set development speakers, clean prompts, {DIV_SEEDS} seeds "
                                  f"(input_seed + {DIV_SEED_BASE} + m), EXP-020's seeds; CVf0 across seeds of each arm",
                     "quality": f"the first {Q_N} speakers, clean, synthesis seed = input seed; W4R2, W8R2 and T16 scored against the stored "
                                "reference (EXP-018)"},
            "hypotheses": {"H1": "CVf0(T16) - CVf0(W8), speaker bootstrap 95% lower bound > 0",
                           "H2": "CVf0(W8R2) - CVf0(W8) lower bound > 0, and the mean absolute CVf0 gap to T16 is smaller for W8R2 than for W8",
                           "H3": f"|mean CVf0(W8) - CVf0(W4)| <= {SATURATION}",
                           "H4_exploratory": "Spearman correlation across speakers between CVf0(T16) - CVf0(W8) and CVf0(T16) - stored "
                                             "CVf0(student alone): positive under the theory (a more biased student loses more)"},
            "verdict_rule": "supported if H1 and H2 hold; refuted if H1 fails (the reference keeps the pure teacher's variety, so there is "
                            "no ceiling to explain); mixed otherwise. Supported: design an algorithm aimed at the ceiling. Refuted: drop "
                            "the theory's central prediction. No retuning on these data",
            "power": "on EXP-020's 20 clean speakers the speaker-bootstrap 95% half-width of a CVf0 difference was 0.0016 between two "
                     "handovers from the same teacher trajectory (plain_8 - plain_4), 0.0083 for Voice-Swap against the reference and 0.0215 "
                     "for the student alone against the reference; smaller differences may go undetected",
            "smoke": f"2 cases (Quick-set positions 59-60): the teacher loop matches the released teacher at Steps 4 and 8 (max abs <= "
                     f"{TEACHER_TOLERANCE}) and the stored W4 and W8 scores are reproduced (tolerance {sh.REPRODUCTION_TOLERANCE}); in the full "
                     "run, the released W4 and W8 CVf0 are compared with EXP-020's stored values",
            "bootstrap": {"B": BOOT_B, "seed": BOOT_SEED, "unit": "speaker"},
            "novelty": "the theory is the candidate contribution: no published two-model analysis or retention ceiling for a biased student "
                       "after re-noising was found (searches 2026-11-14; closest: speciation theory for one model, empirical first-step "
                       "diversity papers, reflection coupling in speculative sampling); not certified",
            "evidence_status": "development test of a theory on speakers that designed earlier experiments; not evidence for a method",
            "adoption": "not adopted; changing the thesis method needs a dated decision record and adviser review",
            "hardware": "Colab Tesla T4 only; Shonget-DMOSpeech2 at the pinned commit", "dmo_commit": DMO_COMMIT,
            "code_hashes": code_hashes(),
            "input_hashes": {str(p.relative_to(THESIS_ROOT)).replace("\\", "/"): sh.sha256_text(p)
                             for p in (STEPS_CSV, EXP020_DIV, FAILURE_CONFIG, MANIFEST)},
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
        raise SystemExit(f"EXP-026 reuses T4 labels and student noise depends on the GPU; found {pipe.gpu}.")
    if args.cmd == "smoke":
        sm = out / "smoke"
        sm.mkdir(exist_ok=True)
        repro, diffs = run_quality(pipe, smoke_items(), sm, smoke=True, stored=stored_w())
        rep = {"rows": repro, "teacher_loop_max_abs_diff": max(diffs)}
        (sm / "reproduction.json").write_text(json.dumps(rep, indent=2) + "\n", encoding="utf-8")
        check_smoke(rep)
        run_diversity(pipe, div_items()[:1], sm, seeds=2)
        worst_wer, worst_sim = sh.check_reproduction(repro)
        print(f"Smoke run complete: {sm} (stored W4 and W8 reproduced: max |dWER| {worst_wer:.4f}, max |dSIM| {worst_sim:.4f}; "
              f"teacher loop matches the released teacher: max difference {rep['teacher_loop_max_abs_diff']:.2e})", flush=True)
        return
    smoke_file = out / "smoke" / "reproduction.json"
    if not smoke_file.exists():
        raise SystemExit("Run the smoke test first: the run reuses stored labels only if they reproduce.")
    check_smoke(json.loads(smoke_file.read_text(encoding="utf-8")))
    shutil.copy2(PREREG_PATH, out / "preregistration.json")
    run_diversity(pipe, div_items(), out)
    run_quality(pipe, q_items(), out)
    analyze(out)


if __name__ == "__main__":
    main()
