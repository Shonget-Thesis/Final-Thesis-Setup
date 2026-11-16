"""EXP-033 (EVAL-026): cheaper teacher paths to the DMOSpeech 2 paper's t_switch = 0.25.

Which handover keeps the paper setting's prosodic variety and quality for the fewest forward passes?

EXP-032 showed that the paper's evaluated setting (`P25`, 30 passes) has far more pitch variety than the thesis's
Step 8 with the student restarted at 0.0761 (`W8`, 21 passes), and its tensor bank showed why: the teacher decides the
melody between Steps 3 and 8, and the restart at 0.0761 discards most of it. Earlier experiments (EXP-026 to EXP-029) had
already run the restart at 0.2929; joined to EXP-032, `W8R2` (20 passes) matches `P25` in quality and roughly in variety.
Stopping the teacher earlier under that restart only trades variety for passes (EXP-029). What has never been run is
making the teacher's own path to t = 0.2929 cheaper. This run tests five ways, each against the paper's setting on the
same cases:

  P25    the paper's Algorithm 2 at 0.25 (14 teacher steps on the 32-interval grid, 2 student steps). 30 passes.
         The 100% reference, as in EXP-032.
  W8R2   16-interval grid to Step 8 (t = 0.2929), student restarted at 0.2929 and 0.6173. 20 passes.
  W8     the same teacher, student restarted at 0.0761 (the thesis's fixed j = 1). 21 passes.
  W6R2, W4R2   the teacher stopped earlier (t = 0.1685, 0.0761) under the 0.2929 restart. 16 and 12 passes.
  STU    the student alone. 4 passes.

  CPH12, CPH8, CPH6   Commit-Point Handover: the guided teacher reaches the same time t = 0.2929 on a 12-, 8- or
         6-interval sway grid (6, 4 or 3 Euler steps instead of 8), then the 0.2929 restart. N + 4 passes: 16, 12, 10.
         This is the released sampler with fewer teacher steps; no new operator.
  GI     guidance interval: 16-interval grid, the released guidance on Steps 0-3 and the conditional branch alone on
         Steps 4-8, then the 0.2929 restart. 15 passes.
  SST    student-seeded teacher: the student's one-pass draft S0 stands in for teacher Steps 0-3. The teacher starts at
         Step 4 from (1 - t_4) y0 + t_4 S0 and runs Steps 4-8 guided, then the 0.2929 restart. 13 passes.
  CWH4   Commitment-Weighted Handover at Step 4: per time-frequency cell, kappa = 1 / (1 + (D / rms(D))^2) with
         D = T_4 - T_3, the teacher's own last change; the student's first-level draft S1 (from T_4) fills what the teacher
         has not decided, x = kappa T_4 + (1 - kappa) S1, then the student's levels 0.2929 and 0.6173. 13 passes.

- Quality (UTMOS, WER, speaker similarity): development speakers 1-30 x clean and noisy15 (60 cases), all arms.
- Prosodic variety (CVf0, aligned CVf0, timing spread; melody agreement with P25 per seed): speakers 1-20 x 12 seeds,
  clean prompts only (EXP-032: pitch cannot be tracked in the teacher arms' noisy15 outputs).
- Cost: forward passes counted by hooks on the teacher and student networks, and wall-clock time of each arm run on its
  own, on development speakers 1-10 (clean).

Subcommands: freeze, smoke, run, analyze.
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_consistency_gate_prelim as cg  # noqa: E402
import run_damped_transmissibility_stac as dts  # noqa: E402
import run_followup_tests as rf  # noqa: E402
import run_grid32_prelim as gl  # noqa: E402
import run_method_comparison as mc  # noqa: E402
import run_noise_guidance_prelim as ng  # noqa: E402
import run_selective_handover_prelim as sh  # noqa: E402
import run_timing_only_handover as toh  # noqa: E402
import run_transmissibility_bridge as tb  # noqa: E402
import run_transmissibility_confirm as tc  # noqa: E402
import run_transmissibility_test as tt  # noqa: E402
import run_variety_per_pass as vp  # noqa: E402

THESIS_ROOT = rf.THESIS_ROOT
CODE_DIR = Path(__file__).resolve().parent
EXP_DIR = THESIS_ROOT / "research/experiments/EXP-033-cheaper-teacher-path"
PREREG_PATH = EXP_DIR / "preregistration.json"
EXP032_DIR = THESIS_ROOT / "research/experiments/EXP-032-variety-per-pass/results"
EXP032_QUAL_CSV = EXP032_DIR / "quality.csv"
EXP032_DIV_CSV = EXP032_DIR / "diversity.csv"
EXP026_QUAL_CSV = THESIS_ROOT / "research/experiments/EXP-026-transmissibility-test/results/quality.csv"
EXP027_QUAL_CSV = tb.EXP027_QUAL_CSV

DEPENDENCIES = ("run_cheaper_teacher_path.py",) + tuple(vp.DEPENDENCIES)

QUAL_N = 30
DIV_N = 20
DIV_SEEDS = 12
DIV_SEED_BASE = vp.DIV_SEED_BASE      # 1000, as in EXP-029 to EXP-032
TIME_N = 10
CONDS = ("clean", "noisy15")          # quality stage
DIV_COND = "clean"                    # variety stage
SEED = vp.SEED
BOOT_B = vp.BOOT_B
THETA = mc.THETA                      # 20 dB, Chapter 3's SNR gate (descriptive routing only)

K_PAPER = vp.K_PAPER                  # 14
K_COMMIT = 8                          # 16-interval Step 8: t = 0.2929, the student's second level
T_COMMIT = gl.sway_grid(16)[K_COMMIT]
CPH_GRIDS = (12, 8, 6)                # teacher grids whose midpoint is t = 0.2929
GI_LAST_GUIDED = 3                    # guidance on Steps 0-3
SST_FIRST = 4                         # the teacher starts at Step 4
CWH_STEP = 4
R2_STEPS = (8, 6, 4)
REFERENCE, BASE = "P25", "STU"
KNOWN = ("STU", "W8", "W8R2", "W6R2", "W4R2", "P25")
CANDIDATES = tuple(f"CPH{n}" for n in CPH_GRIDS) + ("GI", "SST", "CWH4")
ARMS = KNOWN + CANDIDATES
TEST_ARMS = ("W8", "W8R2", "W6R2", "W4R2") + CANDIDATES      # every arm judged against P25
EST_ARMS = tuple(f"CPH{n}" for n in CPH_GRIDS) + ("GI", "SST")  # arms that hand over an estimate at t = 0.2929

TEACHER_PASSES = {"STU": 0, "W8": 18, "W8R2": 18, "W6R2": 14, "W4R2": 10, "P25": 28,
                  **{f"CPH{n}": n + 2 for n in CPH_GRIDS}, "GI": 2 * (GI_LAST_GUIDED + 1) + (K_COMMIT - GI_LAST_GUIDED),
                  "SST": 2 * (K_COMMIT - SST_FIRST + 1), "CWH4": 2 * (CWH_STEP + 1)}
STUDENT_PASSES = {"STU": 4, "W8": 3, "W8R2": 2, "W6R2": 2, "W4R2": 2, "P25": 2,
                  **{f"CPH{n}": 2 for n in CPH_GRIDS}, "GI": 2, "SST": 3, "CWH4": 3}
COST = {a: TEACHER_PASSES[a] + STUDENT_PASSES[a] for a in ARMS}

METRICS = vp.METRICS
VARIETY = vp.VARIETY
DIV_MEASURES = vp.DIV_MEASURES + ("melody",)   # melody: frames within a semitone of P25's melody, same seed
SEMITONE_CENTS = 100.0
MIN_SHARED_VOICED = 5

SHARE_BAR = vp.SHARE_BAR              # 0.90, EXP-032's bar
ZERO_TOL = vp.ZERO_TOL
SPLIT_TOL = vp.SPLIT_TOL              # 1e-5
GRID_TOL = vp.GRID_TOL                # 1e-4
REPRO_TOL = vp.REPRO_TOL              # 0.01
RUNAWAY_WER = vp.RUNAWAY_WER

QUAL_FIELDS = (["utt_id", "speaker_id", "condition", "prompt_snr", "gpu", "frames", "seq_len"]
               + [f"d_{j}" for j in range(1, 9)] + ["kappa_mean"] + [f"est_dist_{a}" for a in EST_ARMS]
               + [f"{m}_{a}" for a in ARMS for m in METRICS + ("hypothesis",)]
               + ["case_seconds", "utc"])
DIV_FIELDS = (["utt_id", "speaker_id", "condition", "n_seeds", "kappa_mean"] + [f"est_dist_{a}" for a in EST_ARMS]
              + [f"{m}_{a}" for a in ARMS for m in DIV_MEASURES]
              + ["case_seconds", "utc"])
TIME_FIELDS = (["utt_id", "speaker_id", "condition", "gpu", "frames", "seq_len"]
               + [f"{m}_{a}" for a in ARMS for m in ("seconds", "teacher_passes", "student_passes")]
               + ["utc"])


def qual_items(n=QUAL_N):
    """Development speakers 1-30: EXP-032's speakers and seeds (and EXP-026's), so both give reproduction checks."""
    return cg.dev_items()[:n]


def div_items(n=DIV_N):
    """Development speakers 1-20, with seeds input_seed + 1000 + s as in EXP-029 to EXP-032."""
    return cg.dev_items()[:n]


def time_items(n=TIME_N):
    return cg.dev_items()[:n]


# --------------------------------------------------------------------------- handover arithmetic (arrays or tensors)
def commitment(curr, prev, plen):
    """kappa = 1 / (1 + (D / rms(D))^2) over the generated frames, with D = curr - prev: near 1 where the teacher's
    estimate has stopped changing, near 0 where it changed more than is typical for this input. No fitted parameter."""
    d = (curr - prev)[:, plen:, :]
    rms = max(float((d * d).mean()) ** 0.5, 1e-12)
    return 1.0 / (1.0 + (d / rms) ** 2)


def cwh_mix(teacher_est, student_draft, kappa, plen):
    """x = kappa T + (1 - kappa) S on the generated frames; the prompt frames stay the teacher's."""
    x = sh._clone(teacher_est)
    x[:, plen:, :] = kappa * teacher_est[:, plen:, :] + (1.0 - kappa) * student_draft[:, plen:, :]
    return x


def seeded_state(y0, draft, t):
    """The flow-matching state at time t that has `draft` as its data and y0 as its noise: (1 - t) y0 + t draft."""
    return (1.0 - t) * y0 + t * draft


def rel_dist(a, b, plen):
    """||a - b|| / ||b|| over the generated frames."""
    x, y = toh._as_numpy(a[0, plen:, :]), toh._as_numpy(b[0, plen:, :])
    return float(np.linalg.norm(x - y) / max(float(np.linalg.norm(y)), 1e-12))


def melody_agreement(track, ref):
    """Share of frames within one semitone of `ref`, over the frames both tracks voice; None below five shared frames."""
    n = min(len(track), len(ref))
    a, b = np.asarray(track[:n], dtype=float), np.asarray(ref[:n], dtype=float)
    both = ~np.isnan(a) & ~np.isnan(b)
    if int(both.sum()) < MIN_SHARED_VOICED:
        return None
    cents = 1200.0 * np.abs(np.log2(a[both] / b[both]))
    return float(np.mean(cents <= SEMITONE_CENTS))


# --------------------------------------------------------------------------- teacher loops
def euler(pipe, ctx, grid, x, first, last, guided_until=None):
    """Euler teacher from the state x at grid[first] to the endpoint estimate at grid[last]. Steps i <= guided_until
    use the released guidance (every step when guided_until is None); later steps use the conditional branch alone.
    The arithmetic is run_grid32_prelim.grid_teacher's."""
    torch, sampler = pipe.torch, pipe.sampler
    with torch.inference_mode():
        for i in range(first, last + 1):
            t = torch.tensor(grid[i], device=pipe.device, dtype=torch.float32)
            kw = dict(x=x, cond=ctx["step_cond"], text=ctx["text_ids"], time=t, mask=None)
            v = pipe.tts.teacher(drop_audio_cond=False, drop_text=False, **kw)
            if guided_until is None or i <= guided_until:
                vu = pipe.tts.teacher(drop_audio_cond=True, drop_text=True, **kw)
                v = v + (v - vu) * sampler.CFG_STRENGTH
            if i == last:
                return x + (1.0 - grid[i]) * v
            x = x + (grid[i + 1] - grid[i]) * v


def teacher_state(pipe, ctx, grid, k):
    """The guided teacher's state at grid[k] after k Euler steps (k evaluations; no endpoint estimate)."""
    torch, sampler = pipe.torch, pipe.sampler
    x = ctx["y0"].clone()
    with torch.inference_mode():
        for i in range(k):
            t = torch.tensor(grid[i], device=pipe.device, dtype=torch.float32)
            kw = dict(x=x, cond=ctx["step_cond"], text=ctx["text_ids"], time=t, mask=None)
            vc = pipe.tts.teacher(drop_audio_cond=False, drop_text=False, **kw)
            vu = pipe.tts.teacher(drop_audio_cond=True, drop_text=True, **kw)
            x = x + (grid[i + 1] - grid[i]) * (vc + (vc - vu) * sampler.CFG_STRENGTH)
    return x


# --------------------------------------------------------------------------- synthesis
def synthesize(pipe, ctx, arms=ARMS):
    """Every requested arm for one case, sharing what can be shared: one 16-interval teacher run (Steps 3, 4, 6, 8 and
    the state at Step 4), one 32-interval run for P25, one run per coarse grid, the conditional-only continuation for
    GI, the seeded continuation for SST and one student draft. Every arm uses the released student's per-level seeds.
    Returns (outputs, parts)."""
    taus = list(pipe.sampler.STUDENT_TAUS)
    plen = ctx["prompt_len"]
    g16 = tt.grid16(pipe)
    e16, drift, s16 = gl.grid_teacher(pipe, ctx, g16, {CWH_STEP - 1, CWH_STEP, 6, K_COMMIT})
    s0 = gl.student(pipe, ctx, ctx["step_cond"], [0.0])
    outs = {"STU": gl.student(pipe, ctx, s0, taus), "W8": gl.student(pipe, ctx, e16[K_COMMIT], taus)}
    for k in R2_STEPS:
        outs[f"W{k}R2"] = gl.student(pipe, ctx, e16[k], taus[1:])
    if "P25" in arms:
        _, _, s32 = gl.grid_teacher(pipe, ctx, gl.GRID32, {K_PAPER})
        outs["P25"] = gl.student_from_state(pipe, ctx, s32[K_PAPER], list(gl.ALG2_TIMES))
    est = {}
    for n in CPH_GRIDS:
        ends, _, _ = gl.grid_teacher(pipe, ctx, gl.sway_grid(n), {n // 2})
        est[f"CPH{n}"] = ends[n // 2]
    est["GI"] = euler(pipe, ctx, g16, s16[SST_FIRST], GI_LAST_GUIDED + 1, K_COMMIT, guided_until=GI_LAST_GUIDED)
    est["SST"] = euler(pipe, ctx, g16, seeded_state(ctx["y0"], s0, g16[SST_FIRST]), SST_FIRST, K_COMMIT)
    for a, e in est.items():
        outs[a] = gl.student(pipe, ctx, e, taus[1:])
    s1 = gl.student(pipe, ctx, e16[CWH_STEP], taus[:1])
    kappa = commitment(e16[CWH_STEP], e16[CWH_STEP - 1], plen)
    outs["CWH4"] = gl.student(pipe, ctx, cwh_mix(e16[CWH_STEP], s1, kappa, plen), taus[1:])
    parts = {"e16": e16, "s16": s16, "drift": drift, "s0": s0, "s1": s1, "est": est,
             "kappa_mean": float(kappa.mean()),
             "est_dist": {a: rel_dist(e, e16[K_COMMIT], plen) for a, e in est.items()}}
    return {a: outs[a] for a in arms}, parts


def standalone(pipe, ctx, arm):
    """One arm with exactly its own forward passes (nothing shared), for the pass count and the wall-clock time."""
    taus = list(pipe.sampler.STUDENT_TAUS)
    plen = ctx["prompt_len"]
    g16 = tt.grid16(pipe)
    if arm == "STU":
        return gl.student(pipe, ctx, ctx["step_cond"], [0.0] + taus)
    if arm == "W8":
        return gl.student(pipe, ctx, euler(pipe, ctx, g16, ctx["y0"].clone(), 0, K_COMMIT), taus)
    if arm in tuple(f"W{k}R2" for k in R2_STEPS):
        return gl.student(pipe, ctx, euler(pipe, ctx, g16, ctx["y0"].clone(), 0, int(arm[1])), taus[1:])
    if arm == "P25":
        return gl.student_from_state(pipe, ctx, teacher_state(pipe, ctx, gl.GRID32, K_PAPER), list(gl.ALG2_TIMES))
    if arm.startswith("CPH"):
        n = int(arm[3:])
        return gl.student(pipe, ctx, euler(pipe, ctx, gl.sway_grid(n), ctx["y0"].clone(), 0, n // 2), taus[1:])
    if arm == "GI":
        return gl.student(pipe, ctx, euler(pipe, ctx, g16, ctx["y0"].clone(), 0, K_COMMIT,
                                           guided_until=GI_LAST_GUIDED), taus[1:])
    if arm == "SST":
        s0 = gl.student(pipe, ctx, ctx["step_cond"], [0.0])
        return gl.student(pipe, ctx, euler(pipe, ctx, g16, seeded_state(ctx["y0"], s0, g16[SST_FIRST]),
                                           SST_FIRST, K_COMMIT), taus[1:])
    if arm == "CWH4":
        ends, _, _ = gl.grid_teacher(pipe, ctx, g16, {CWH_STEP - 1, CWH_STEP})
        s1 = gl.student(pipe, ctx, ends[CWH_STEP], taus[:1])
        kappa = commitment(ends[CWH_STEP], ends[CWH_STEP - 1], plen)
        return gl.student(pipe, ctx, cwh_mix(ends[CWH_STEP], s1, kappa, plen), taus[1:])
    raise ValueError(f"Unknown arm {arm!r}")


class PassCounter:
    """Counts forward passes of the teacher and student networks with forward hooks."""

    def __init__(self, pipe):
        self.n = {"teacher": 0, "student": 0}
        self._hooks = [pipe.tts.teacher.register_forward_hook(lambda *a: self._add("teacher")),
                       pipe.tts.generator.register_forward_hook(lambda *a: self._add("student"))]

    def _add(self, key):
        self.n[key] += 1

    def reset(self):
        self.n = {"teacher": 0, "student": 0}

    def close(self):
        for h in self._hooks:
            h.remove()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


# --------------------------------------------------------------------------- runs
def run_quality(pipe, items, conds=CONDS, out_dir=None, csv_name="quality.csv"):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / csv_name
    done = rf.done_keys(path, 1)
    todo = [(it, c) for c in conds for it in items if (it["utt_id"], c) not in done]
    print(f"[quality] {len(todo)} cases to run ({len(done)} already in {path.name})", flush=True)
    for n_run, (item, cond) in enumerate(todo, 1):
        t0 = time.perf_counter()
        ctx = pipe.prepare(item, cond, int(item["input_seed"]), out_dir)
        outs, parts = synthesize(pipe, ctx, ARMS)
        gt = pipe.ground_truth(item)
        s0 = parts["s0"]
        row = {"utt_id": item["utt_id"], "speaker_id": item["speaker_id"], "condition": cond,
               "prompt_snr": float(ctx["prompt_snr"]), "gpu": pipe.gpu, "frames": int(s0.shape[1] - ctx["prompt_len"]),
               "seq_len": int(s0.shape[1]), **{f"d_{j}": float(parts["drift"][j]) for j in range(1, 9)},
               "kappa_mean": parts["kappa_mean"], **{f"est_dist_{a}": parts["est_dist"][a] for a in EST_ARMS}}
        for arm in ARMS:
            m = pipe.score(ng.decode(pipe, ctx, outs[arm]), item, gt)
            row.update({f"{k}_{arm}": m[k] for k in METRICS + ("hypothesis",)})
        row.update({"case_seconds": round(time.perf_counter() - t0, 2), "utc": tb._utc()})
        tb._append(path, QUAL_FIELDS, row)
        print(f"[{n_run}/{len(todo)}] quality: {item['utt_id']} {cond} ({row['case_seconds']:.1f} s)", flush=True)
    print(f"[OK] quality complete: {path}", flush=True)
    return path


def _mean(vals):
    vals = [v for v in vals if v is not None and v == v]
    return statistics.mean(vals) if vals else float("nan")


def run_diversity(pipe, items, out_dir, seeds=DIV_SEEDS, cond=DIV_COND, csv_name="diversity.csv"):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / csv_name
    done = rf.done_keys(path, 1)
    todo = [it for it in items if (it["utt_id"], cond) not in done]
    print(f"[diversity] {len(todo)} speakers x {seeds} seeds, {cond} ({len(done)} already in {path.name})", flush=True)
    for n_run, item in enumerate(todo, 1):
        t0 = time.perf_counter()
        tracks = {a: [] for a in ARMS}
        mels = {a: [] for a in ARMS}
        slow = {a: [] for a in ARMS}
        melody = {a: [] for a in ARMS}
        kappas, dists = [], {a: [] for a in EST_ARMS}
        for s in range(seeds):
            seed = int(item["input_seed"]) + DIV_SEED_BASE + s
            ctx = pipe.prepare(item, cond, seed, out_dir, noise_seed=item["input_seed"])
            outs, parts = synthesize(pipe, ctx, ARMS)
            arm_tracks = dts._track_arms_fast(pipe, ctx, outs, ARMS)
            plen = ctx["prompt_len"]
            for arm in ARMS:
                tracks[arm].append(arm_tracks[arm])
                mels[arm].append(toh._as_numpy(outs[arm][0, plen:, :]))
                slow[arm].append(tc._slow_component(outs[arm], plen))
                melody[arm].append(melody_agreement(arm_tracks[arm], arm_tracks[REFERENCE]))
            kappas.append(parts["kappa_mean"])
            for a in EST_ARMS:
                dists[a].append(parts["est_dist"][a])
        row = {"utt_id": item["utt_id"], "speaker_id": item["speaker_id"], "condition": cond, "n_seeds": seeds,
               "kappa_mean": _mean(kappas), **{f"est_dist_{a}": _mean(dists[a]) for a in EST_ARMS}}
        for arm in ARMS:
            row[f"cvf0_{arm}"], row[f"voiced_{arm}"] = sh.cvf0(tracks[arm])
            row[f"mel_spread_{arm}"] = tc.slow_mel_spread(slow[arm])
            (row[f"cvf0_aligned_{arm}"], row[f"voiced_aligned_{arm}"],
             row[f"timing_spread_{arm}"]) = mc.aligned_cvf0(tracks[arm], mels[arm])
            row[f"melody_{arm}"] = _mean(melody[arm])
        row.update({"case_seconds": round(time.perf_counter() - t0, 2), "utc": tb._utc()})
        tb._append(path, DIV_FIELDS, row)
        print(f"[{n_run}/{len(todo)}] diversity: {item['utt_id']} {cond} ({row['case_seconds']:.1f} s)", flush=True)
    print(f"[OK] diversity complete: {path}", flush=True)
    return path


def run_timing(pipe, items, out_dir, cond="clean", csv_name="timing.csv"):
    """Each arm on its own, timed between GPU synchronizations (teacher, student and vocoder; the shared preparation is
    left out), with its forward passes counted. One untimed warm-up; the arm order alternates between cases."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / csv_name
    done = rf.done_keys(path, 1)
    todo = [it for it in items if (it["utt_id"], cond) not in done]
    print(f"[timing] {len(todo)} cases to run ({len(done)} already in {path.name})", flush=True)

    def one(ctx, arm):
        return ng.decode(pipe, ctx, standalone(pipe, ctx, arm))

    with PassCounter(pipe) as counter:
        for n_run, item in enumerate(todo, 1):
            ctx = pipe.prepare(item, cond, int(item["input_seed"]), out_dir)
            if n_run == 1:
                one(ctx, "W8R2")                       # warm-up, not timed
            row = {"utt_id": item["utt_id"], "speaker_id": item["speaker_id"], "condition": cond, "gpu": pipe.gpu,
                   "frames": int(ctx["y0"].shape[1] - ctx["prompt_len"]), "seq_len": int(ctx["y0"].shape[1])}
            for arm in (ARMS if n_run % 2 else ARMS[::-1]):
                counter.reset()
                _, sec = gl.timed(pipe, one, ctx, arm)
                row.update({f"seconds_{arm}": round(sec, 4), f"teacher_passes_{arm}": counter.n["teacher"],
                            f"student_passes_{arm}": counter.n["student"]})
            row["utc"] = tb._utc()
            tb._append(path, TIME_FIELDS, row)
            print(f"[{n_run}/{len(todo)}] timing: {item['utt_id']}", flush=True)
    print(f"[OK] timing complete: {path}", flush=True)
    return path


# --------------------------------------------------------------------------- smoke test
def _rows_by_key(path):
    return vp._rows_by_key(path)


def smoke(pipe, out_dir, prereg_path=PREREG_PATH):
    """Exact identities, the pass count of every arm, EXP-027's W4 reproduction, then micro runs of all three stages:
    quality on development speaker 1 under both conditions (STU, W8 and P25 checked against EXP-032's stored rows;
    W8R2 and W4R2 compared with EXP-026's), variety on one speaker with two seeds, and timing on one case."""
    check_prereg(prereg_path)
    sdir = Path(out_dir) / "smoke"
    sdir.mkdir(parents=True, exist_ok=True)
    stored027 = {r["utt_id"]: r for r in rf.read_csv(EXP027_QUAL_CSV)}
    taus = list(pipe.sampler.STUDENT_TAUS)
    g16 = tt.grid16(pipe)
    checks = []
    for item in tb.smoke_items():
        ctx = pipe.prepare(item, "clean", int(item["input_seed"]), sdir)
        outs, parts = synthesize(pipe, ctx, ARMS)
        e16, s16 = parts["e16"], parts["s16"]
        traj, _ = dts.teacher_trajectory(pipe, ctx, wanted=(CWH_STEP, K_COMMIT))
        finite = all(bool(np.all(np.isfinite(toh._as_numpy(outs[a])))) for a in ARMS)
        rec = {"utt_id": item["utt_id"], "finite": finite, "prompt_snr": float(ctx["prompt_snr"]),
               "seq_len": int(parts["s0"].shape[1]), "kappa_mean": parts["kappa_mean"], "est_dist": parts["est_dist"],
               # the 16-interval run against the trajectory the earlier experiments use
               "grid_teacher_vs_trajectory": max(tb._max_abs(e16[k] - traj[k]) for k in (CWH_STEP, K_COMMIT)),
               # the generic teacher loop: from noise, and continued from the state at Step 4, reaches the same estimate
               "euler_vs_grid_teacher": tb._max_abs(euler(pipe, ctx, g16, ctx["y0"].clone(), 0, K_COMMIT) - e16[K_COMMIT]),
               "continued_vs_grid_teacher": tb._max_abs(euler(pipe, ctx, g16, s16[SST_FIRST], SST_FIRST, K_COMMIT)
                                                        - e16[K_COMMIT]),
               # guidance on every step is the released teacher
               "gi_all_guided_vs_grid_teacher": tb._max_abs(
                   euler(pipe, ctx, g16, ctx["y0"].clone(), 0, K_COMMIT, guided_until=K_COMMIT) - e16[K_COMMIT]),
               # the student's first level, then the other two, is the released three-level student
               "split_diff": tb._max_abs(gl.student(pipe, ctx, parts["s1"], taus[1:])
                                         - gl.student(pipe, ctx, e16[CWH_STEP], taus)),
               # kappa = 1 hands over the teacher's estimate unchanged
               "cwh_kappa_one": tb._max_abs(cwh_mix(e16[CWH_STEP], parts["s1"], 1.0, ctx["prompt_len"]) - e16[CWH_STEP])}
        counts, alone = {}, {}
        with PassCounter(pipe) as counter:
            for arm in ARMS:
                counter.reset()
                alone[arm] = tb._max_abs(standalone(pipe, ctx, arm) - outs[arm])
                counts[arm] = dict(counter.n)
        rec["standalone_diff"] = alone
        rec["passes"] = counts
        mw = pipe.score(ng.decode(pipe, ctx, gl.student(pipe, ctx, e16[CWH_STEP], taus)), item, pipe.ground_truth(item))
        ref = stored027[item["utt_id"]]
        rec["d_wer_W4"] = abs(mw["wer"] - float(ref["wer_W4"]))
        rec["d_sim_W4"] = abs(mw["similarity"] - float(ref["similarity_W4"]))
        print(f"[smoke] {json.dumps(rec)}", flush=True)
        checks.append(rec)
        if not finite:
            raise RuntimeError(f"Non-finite output: {rec}")
        if max(rec["grid_teacher_vs_trajectory"], rec["euler_vs_grid_teacher"], rec["continued_vs_grid_teacher"],
               rec["gi_all_guided_vs_grid_teacher"]) > GRID_TOL:
            raise RuntimeError(f"Teacher identity check failed: {rec}")
        if max(rec["split_diff"], rec["cwh_kappa_one"]) > SPLIT_TOL:
            raise RuntimeError(f"Student identity check failed: {rec}")
        if max(alone.values()) > GRID_TOL:
            raise RuntimeError(f"An arm run on its own differs from the shared run: {alone}")
        bad = {a: c for a, c in counts.items()
               if (c["teacher"], c["student"]) != (TEACHER_PASSES[a], STUDENT_PASSES[a])}
        if bad:
            raise RuntimeError(f"Forward-pass count differs from the cost table: {bad}")
        if rec["d_wer_W4"] > REPRO_TOL or rec["d_sim_W4"] > REPRO_TOL:
            raise RuntimeError(f"EXP-027 W4 reproduction failed: {rec}")
    q = run_quality(pipe, qual_items()[:1], conds=CONDS, out_dir=sdir, csv_name="quality.csv")
    qrows = rf.read_csv(q)
    repro032 = vp.quality_reproduction(qrows, _rows_by_key(EXP032_QUAL_CSV), {"STU": "STU", "W8": "W8", "P25": "P25"})
    repro026 = vp.quality_reproduction(qrows, _rows_by_key(EXP026_QUAL_CSV), {"W8R2": "W8R2", "W4R2": "W4R2"},
                                       conds=("clean",))
    worst032 = max((v for e in repro032.values() for k, v in e.items() if k.startswith("max_abs_")), default=None)
    print(f"[smoke] EXP-032 reproduction: {json.dumps(repro032)}", flush=True)
    print(f"[smoke] EXP-026 comparison (recorded, not a gate): {json.dumps(repro026)}", flush=True)
    if worst032 is None or worst032 > REPRO_TOL:
        raise RuntimeError(f"EXP-032 reproduction failed (largest difference {worst032}): {repro032}")
    run_diversity(pipe, div_items()[:1], sdir, seeds=2, csv_name="diversity.csv")
    run_timing(pipe, time_items()[:1], sdir, csv_name="timing.csv")
    summary = {"checks": checks, "exp032_quality_reproduction": repro032, "exp026_quality_comparison": repro026,
               "utc": tb._utc()}
    (sdir / "smoke_check.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print("[OK] smoke passed", flush=True)
    return summary


# --------------------------------------------------------------------------- preregistration
ESTIMANDS = {
    "variety_share": "Clean prompts. For each arm A, with V = CVf0 or aligned CVf0 averaged over the 20 diversity "
                     "speakers: share_A = (V_A - V_STU) / (V_P25 - V_STU), the part of the paper setting's variety gain "
                     "over the student alone that A keeps; 95% speaker-level bootstrap interval of the ratio of means "
                     "(EXP-032's estimand; P25 and STU are synthesized again in this run).",
    "variety_90": "L_A = V_A - 0.9 V_P25 - 0.1 V_STU per speaker; a 95% bootstrap lower bound above 0 confirms that A "
                  "keeps at least 90% of P25's gain.",
    "quality_clean": "Clean prompts, A - P25 per case: mean dUTMOS >= -0.08, dWER upper bound <= +0.01 and dSIM lower "
                     "bound >= -0.01 (EXP-032's bars, from EXP-029).",
    "quality_noisy": "noisy15 prompts, A - STU per case under the same bars, and the number of runaway outputs "
                     "(WER > 1).",
    "cost": "Forward passes (teacher and student, counted by hooks on both networks in the timing stage) and wall-clock "
            "seconds of each arm run on its own on development speakers 1-10, clean prompts.",
    "melody": "Descriptive. Per seed, the share of frames within one semitone of P25's pitch track over the frames "
              "both voice, averaged over the 12 seeds per speaker.",
    "equal_cost": "Descriptive. CPH12 - W6R2 (16 passes each) and CPH8 - W4R2 (12 passes each) on variety and clean "
                  "quality: reaching t = 0.2929 in larger steps against stopping earlier at the same cost.",
    "estimate_distance": "Descriptive. ||E_A - E_8|| / ||E_8|| over generated frames, the distance of an arm's "
                         "handed-over estimate from the 16-interval Step 8 estimate, for the CPH arms, GI and SST.",
    "snr_routing": "Descriptive, no synthesis. With Chapter 3's 20 dB gate, prompts below the gate go to the student "
                   "alone (4 passes) and the rest to arm A; mean passes and mean quality over the 60 quality cases.",
}
LABEL_RULES = {
    "arm_labels": "For each arm in (W8, W8R2, W6R2, W4R2, CPH12, CPH8, CPH6, GI, SST, CWH4): 'matches_paper' if it "
                  "keeps >= 90% of P25's gain on both CVf0 and aligned CVf0 (point estimates) and passes the clean "
                  "quality bars against P25; 'matches_paper_confirmed' if in addition the 90% is confirmed on both "
                  "measures; otherwise 'quality_short' (variety kept, bars failed), 'variety_short' (bars passed, "
                  "variety not kept) or 'both_short'.",
    "cheapest_match_point": "The arm with the fewest forward passes whose label is 'matches_paper' or "
                            "'matches_paper_confirmed'; 'none' if no arm qualifies. Ties go to the arm listed first.",
    "cheapest_match_confirmed": "The same over arms labelled 'matches_paper_confirmed'.",
    "w8r2_matches_paper": "W8R2's label is 'matches_paper' or 'matches_paper_confirmed' (the post hoc reading of "
                          "EXP-026 joined to EXP-032, tested here on one run).",
    "noisy_quality_safe_arms": "noisy15 prompts: the arms other than STU that pass the quality bars against the "
                               "student alone.",
    "partial": "Every label is 'partial' if the quality or the variety stage is incomplete.",
}
LOCAL_SCREEN = ("A local CPU screen of the five candidates (code/analysis/local_candidate_screen.py; three development "
                "speakers x six seeds, clean prompts) was started on 2026-11-16 before this freeze. Only its two-seed "
                "trial output on one speaker had been seen when the arms, estimands and label rules were fixed; the "
                "rules are EXP-032's and were not adjusted to it. All five candidates are run whatever the screen "
                "shows.")


def build_prereg():
    q = [{"utt_id": it["utt_id"], "speaker_id": it["speaker_id"]} for it in qual_items()]
    d = [{"utt_id": it["utt_id"], "speaker_id": it["speaker_id"]} for it in div_items()]
    t = [{"utt_id": it["utt_id"], "speaker_id": it["speaker_id"]} for it in time_items()]
    inputs = (EXP032_QUAL_CSV, EXP032_DIV_CSV, EXP026_QUAL_CSV, EXP027_QUAL_CSV)
    payload = {
        "experiment": "EXP-033",
        "task_id": "EVAL-026",
        "title": "Cheaper teacher paths to the DMOSpeech 2 paper's t_switch = 0.25: variety, quality and passes",
        "framing": ("Development test of handover candidates. The target is the paper's evaluated setting (t_switch = "
                    "0.25, Algorithm 2, 30 passes): keep its prosodic variety and quality on clean prompts for fewer "
                    "forward passes. Five candidates make the teacher's path to t = 0.2929 cheaper (fewer and larger "
                    "steps, guidance on the early steps only, a student draft in place of the first steps, or a "
                    "commitment-weighted handover at Step 4); none is claimed to be novel before a literature audit. "
                    "Development speakers only; not held-out evidence."),
        "reference": {"arm": REFERENCE, "k_paper": K_PAPER, "grid": "32-interval sway grid",
                      "alg2_times": list(gl.ALG2_TIMES)},
        "arms": list(ARMS),
        "known_arms": list(KNOWN),
        "candidates": list(CANDIDATES),
        "test_arms": list(TEST_ARMS),
        "design": {"commit_time": T_COMMIT, "cph_grids": list(CPH_GRIDS), "gi_last_guided_step": GI_LAST_GUIDED,
                   "sst_first_teacher_step": SST_FIRST, "cwh_step": CWH_STEP,
                   "cwh_kappa": "1 / (1 + (D / rms(D))^2), D = T_4 - T_3 over generated frames",
                   "student_levels_after_handover": list(gl.sway_grid(4)[2:4])},
        "costs": dict(COST),
        "teacher_passes": dict(TEACHER_PASSES),
        "student_passes": dict(STUDENT_PASSES),
        "conditions": list(CONDS),
        "diversity_condition": DIV_COND,
        "n_qual_speakers": len(q),
        "n_qual_cases": len(q) * len(CONDS),
        "qual_utts": q,
        "n_div_speakers": len(d),
        "div_seeds": DIV_SEEDS,
        "div_seed_base": DIV_SEED_BASE,
        "div_utts": d,
        "n_time_cases": len(t),
        "time_utts": t,
        "variety_measures": list(VARIETY),
        "bars": {"share": SHARE_BAR, "clean_utmos": vp.CLEAN_UTMOS_BAR, "wer": vp.WER_BAR, "sim": vp.SIM_BAR,
                 "runaway_wer": RUNAWAY_WER, "snr_gate_db": THETA},
        "estimands": ESTIMANDS,
        "label_rules": LABEL_RULES,
        "local_screen": LOCAL_SCREEN,
        "code_hashes": {name: tb._sha256_file(CODE_DIR / name) for name in DEPENDENCIES},
        "input_hashes": {str(Path(p).relative_to(THESIS_ROOT).as_posix()): tb._sha256_file(p) for p in inputs},
        "seed": SEED,
        "boot_b": BOOT_B,
    }
    sha = tb._sha256_bytes(json.dumps(payload, sort_keys=True).encode("utf-8"))
    return {**payload, "sha256": sha, "created_utc": tb._utc()}


def freeze(prereg_path=PREREG_PATH, force=False):
    prereg_path = Path(prereg_path)
    if prereg_path.exists() and not force:
        raise RuntimeError(f"Preregistration already exists at {prereg_path}; pass --force to overwrite.")
    prereg_path.parent.mkdir(parents=True, exist_ok=True)
    doc = build_prereg()
    with open(prereg_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(doc, indent=2) + "\n")
    print(f"[OK] Froze {prereg_path} (sha256={doc['sha256'][:12]}...)")
    return doc


def check_prereg(prereg_path=PREREG_PATH):
    prereg_path = Path(prereg_path)
    if not prereg_path.exists():
        raise RuntimeError(f"Missing preregistration: {prereg_path}. Run `freeze` first.")
    doc = json.loads(prereg_path.read_text(encoding="utf-8"))
    payload = {k: v for k, v in doc.items() if k not in ("sha256", "created_utc")}
    if doc.get("sha256") != tb._sha256_bytes(json.dumps(payload, sort_keys=True).encode("utf-8")):
        raise RuntimeError("Preregistration SHA-256 mismatch.")
    for name, rec in doc.get("code_hashes", {}).items():
        actual = tb._sha256_file(CODE_DIR / name)
        if actual != rec:
            raise RuntimeError(f"Code hash mismatch for {name}: expected {rec[:12]}, got {actual[:12]}.")
    for rel, rec in doc.get("input_hashes", {}).items():
        actual = tb._sha256_file(THESIS_ROOT / rel)
        if actual != rec:
            raise RuntimeError(f"Input hash mismatch for {rel}: expected {rec[:12]}, got {actual[:12]}.")
    for key, items in (("qual_utts", qual_items()), ("div_utts", div_items()), ("time_utts", time_items())):
        if doc.get(key) != [{"utt_id": it["utt_id"], "speaker_id": it["speaker_id"]} for it in items]:
            raise RuntimeError(f"Speaker list {key} does not match preregistration.")
    return doc


# --------------------------------------------------------------------------- analysis
def _f(r, key):
    return float(r[key])


def _boot(vals):
    return vp._boot(vals)


def variety_curve(div_rows):
    """Clean-prompt variety per arm against P25 and STU (EXP-032's share and 90% contrast)."""
    rows = [r for r in div_rows if r["condition"] == DIV_COND]
    if len(rows) < 2:
        return None
    e = {"n": len(rows), "means": {a: {m: _mean(_f(r, f"{m}_{a}") for r in rows) for m in DIV_MEASURES} for a in ARMS}}
    others = [a for a in ARMS if a not in (REFERENCE, BASE)]
    for m in VARIETY:
        e[m] = {"share": {a: vp.share_ci(rows, a, m, ref=REFERENCE, base=BASE) for a in others},
                "L90": {a: _boot(_f(r, f"{m}_{a}") - SHARE_BAR * _f(r, f"{m}_{REFERENCE}")
                                 - (1 - SHARE_BAR) * _f(r, f"{m}_{BASE}") for r in rows) for a in others},
                "minus_P25": {a: _boot(_f(r, f"{m}_{a}") - _f(r, f"{m}_{REFERENCE}") for r in rows)
                              for a in ARMS if a != REFERENCE}}
    e["melody_minus_W8R2"] = {a: _boot(_f(r, f"melody_{a}") - _f(r, "melody_W8R2") for r in rows)
                              for a in ARMS if a not in (REFERENCE, "W8R2")}
    e["est_dist"] = {a: _mean(_f(r, f"est_dist_{a}") for r in rows) for a in EST_ARMS}
    e["kappa_mean"] = _mean(_f(r, "kappa_mean") for r in rows)
    return e


def quality_tables(qual_rows):
    out = {}
    for cond in CONDS:
        rows = [r for r in qual_rows if r["condition"] == cond]
        if len(rows) < 2:
            continue
        tab = {}
        for a in ARMS:
            wer = [_f(r, f"wer_{a}") for r in rows]
            tab[a] = {"passes": COST[a], "utmos": statistics.mean(_f(r, f"utmos_{a}") for r in rows),
                      "similarity": statistics.mean(_f(r, f"similarity_{a}") for r in rows),
                      "wer_mean": statistics.mean(wer), "wer_median": statistics.median(wer),
                      "runaway": sum(w > RUNAWAY_WER for w in wer)}
        out[cond] = {"n": len(rows), "arms": tab,
                     "vs_P25": {a: vp.qdiff(rows, a, REFERENCE) for a in ARMS if a != REFERENCE},
                     "vs_STU": {a: vp.qdiff(rows, a, BASE) for a in ARMS if a != BASE},
                     "prompt_snr_median": statistics.median(_f(r, "prompt_snr") for r in rows),
                     "est_dist": {a: _mean(_f(r, f"est_dist_{a}") for r in rows) for a in EST_ARMS}}
    return out


def arm_verdict(curve, quality, arm):
    point = all(curve[m]["share"][arm]["point"] is not None and curve[m]["share"][arm]["point"] >= SHARE_BAR
                for m in VARIETY)
    confirmed = point and all(curve[m]["L90"][arm]["lo"] > ZERO_TOL for m in VARIETY)
    ok = bool(quality["clean"]["vs_P25"][arm]["ok"])
    if point and ok:
        label = "matches_paper_confirmed" if confirmed else "matches_paper"
    elif point:
        label = "quality_short"
    elif ok:
        label = "variety_short"
    else:
        label = "both_short"
    return {"variety_point": bool(point), "variety_confirmed": bool(confirmed), "quality": ok, "label": label,
            "passes": COST[arm], "saved_vs_P25": 1.0 - COST[arm] / COST[REFERENCE]}


def labels(curve, quality, complete):
    keys = ("arm_labels", "cheapest_match_point", "cheapest_match_confirmed", "w8r2_matches_paper",
            "noisy_quality_safe_arms")
    if not complete:
        return {k: "partial" for k in keys}, None
    verdicts = {a: arm_verdict(curve, quality, a) for a in TEST_ARMS}

    def cheapest(names):
        ok = [a for a in TEST_ARMS if verdicts[a]["label"] in names]
        return min(ok, key=lambda a: (COST[a], TEST_ARMS.index(a))) if ok else "none"

    lab = {"arm_labels": {a: v["label"] for a, v in verdicts.items()},
           "cheapest_match_point": cheapest(("matches_paper", "matches_paper_confirmed")),
           "cheapest_match_confirmed": cheapest(("matches_paper_confirmed",)),
           "w8r2_matches_paper": verdicts["W8R2"]["label"] in ("matches_paper", "matches_paper_confirmed"),
           "noisy_quality_safe_arms": [a for a in ARMS if a != BASE and quality["noisy15"]["vs_STU"][a]["ok"]]}
    return lab, verdicts


def equal_cost(qual_rows, div_rows):
    q = [r for r in qual_rows if r["condition"] == "clean"]
    d = [r for r in div_rows if r["condition"] == DIV_COND]
    out = {}
    for a, b in (("CPH12", "W6R2"), ("CPH8", "W4R2")):
        e = {"passes": COST[a]}
        if len(q) >= 2:
            e["quality_clean"] = {m: _boot(_f(r, f"{m}_{a}") - _f(r, f"{m}_{b}") for r in q) for m in METRICS}
        if len(d) >= 2:
            e["variety"] = {m: _boot(_f(r, f"{m}_{a}") - _f(r, f"{m}_{b}") for r in d)
                            for m in VARIETY + ("timing_spread", "melody")}
        out[f"{a}_minus_{b}"] = e
    return out


def snr_routing(qual_rows):
    """Chapter 3's 20 dB gate with the student alone below it and arm A above it: passes and quality (no synthesis)."""
    out = {"theta_db": THETA,
           "below_gate": {c: sum(_f(r, "prompt_snr") < THETA for r in qual_rows if r["condition"] == c) for c in CONDS},
           "cases": {c: sum(r["condition"] == c for r in qual_rows) for c in CONDS}, "arms": {}}
    for a in ARMS:
        if a == BASE:
            continue
        e = {"mean_passes": _mean(COST[BASE] if _f(r, "prompt_snr") < THETA else COST[a] for r in qual_rows)}
        for c in CONDS:
            rows = [r for r in qual_rows if r["condition"] == c]
            if rows:
                e[c] = {m: _mean(_f(r, f"{m}_{BASE if _f(r, 'prompt_snr') < THETA else a}") for r in rows)
                        for m in METRICS}
                e[c]["passes"] = _mean(COST[BASE] if _f(r, "prompt_snr") < THETA else COST[a] for r in rows)
        out["arms"][a] = e
    return out


def timing_table(time_rows):
    if not time_rows:
        return None
    ref = statistics.median(_f(r, f"seconds_{REFERENCE}") for r in time_rows)
    out = {"cases": len(time_rows), "arms": {}}
    for a in ARMS:
        sec = statistics.median(_f(r, f"seconds_{a}") for r in time_rows)
        counted = {(int(_f(r, f"teacher_passes_{a}")), int(_f(r, f"student_passes_{a}"))) for r in time_rows}
        out["arms"][a] = {"passes": COST[a], "median_seconds": sec, "ratio_to_P25": sec / ref if ref > 0 else None,
                          "counted_passes": sorted(list(c) for c in counted),
                          "count_matches_cost": counted == {(TEACHER_PASSES[a], STUDENT_PASSES[a])}}
    return out


def diversity_reproduction(div_rows):
    """Clean variety of STU, W8 and P25 against EXP-032's stored rows (same speakers and seeds)."""
    stored = _rows_by_key(EXP032_DIV_CSV)
    shared = [r for r in div_rows if r["condition"] == DIV_COND and (r["utt_id"], DIV_COND) in stored]
    if not shared:
        return None
    out = {"speakers": len(shared)}
    for a in ("STU", "W8", "P25"):
        for m in ("cvf0", "cvf0_aligned", "timing_spread"):
            out[f"max_abs_{m}_{a}"] = max(abs(_f(r, f"{m}_{a}") - _f(stored[(r["utt_id"], DIV_COND)], f"{m}_{a}"))
                                          for r in shared)
    return out


def analyze(out_dir=None, prereg_path=PREREG_PATH, allow_partial=False):
    out_dir = Path(out_dir or EXP_DIR / "results")
    q_path, d_path, t_path = out_dir / "quality.csv", out_dir / "diversity.csv", out_dir / "timing.csv"
    if not q_path.exists():
        raise FileNotFoundError(f"Missing quality CSV: {q_path}")
    qual_rows = rf.read_csv(q_path, repair=True)
    div_rows = rf.read_csv(d_path, repair=True) if d_path.exists() else []
    time_rows = rf.read_csv(t_path, repair=True) if t_path.exists() else []
    complete = len(qual_rows) >= QUAL_N * len(CONDS) and len(div_rows) >= DIV_N
    if not allow_partial and not complete:
        raise RuntimeError(f"Incomplete: quality {len(qual_rows)}/{QUAL_N * len(CONDS)}, diversity {len(div_rows)}/{DIV_N}")
    curve = variety_curve(div_rows)
    quality = quality_tables(qual_rows)
    lab, verdicts = labels(curve, quality, complete)
    results = {
        "experiment": "EXP-033",
        "partial": not complete,
        "labels": lab,
        "arm_verdicts": verdicts,
        "n_quality_rows": len(qual_rows),
        "n_diversity_rows": len(div_rows),
        "n_timing_rows": len(time_rows),
        "costs": dict(COST),
        "teacher_passes": dict(TEACHER_PASSES),
        "student_passes": dict(STUDENT_PASSES),
        "variety": curve,
        "quality": quality,
        "equal_cost": equal_cost(qual_rows, div_rows),
        "snr_routing": snr_routing(qual_rows),
        "timing": timing_table(time_rows),
        "exp032_quality_reproduction": vp.quality_reproduction(qual_rows, _rows_by_key(EXP032_QUAL_CSV),
                                                               {"STU": "STU", "W8": "W8", "P25": "P25"}),
        "exp032_diversity_reproduction": diversity_reproduction(div_rows),
        "exp026_quality_comparison": vp.quality_reproduction(qual_rows, _rows_by_key(EXP026_QUAL_CSV),
                                                             {"W8R2": "W8R2", "W4R2": "W4R2"}, conds=("clean",)),
        "utc": tb._utc(),
    }
    rjson = out_dir / "cheaper_teacher_path_results.json"
    rmd = out_dir / "CHEAPER_TEACHER_PATH_REPORT.md"
    with open(rjson, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(results, indent=2) + "\n")
    with open(rmd, "w", encoding="utf-8", newline="\n") as f:
        f.write(report(results) + "\n")
    entries = {}
    if Path(prereg_path).exists():
        entries["preregistration"] = {"path": str(prereg_path), "sha256": tb._sha256_file(prereg_path)}
    for p in (q_path, d_path, t_path, rjson, rmd):
        if p.exists():
            entries[p.name] = {"sha256": tb._sha256_file(p), "bytes": p.stat().st_size}
    with open(out_dir / "MANIFEST.json", "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps({"experiment": "EXP-033", "utc": tb._utc(), "files": entries}, indent=2) + "\n")
    print(f"[OK] Analysis complete: labels={results['labels']}, report={rmd}", flush=True)
    return results


def _ci(b, d=3):
    return tc._fmt_ci(b, d) if b else "n/a"


def report(res):
    lines = ["# EXP-033: cheaper teacher paths to the paper's t_switch = 0.25", "",
             f"Rows: quality {res['n_quality_rows']} / {QUAL_N * len(CONDS)}, diversity {res['n_diversity_rows']} / "
             f"{DIV_N}, timing {res['n_timing_rows']} / {TIME_N} (partial={res['partial']}). Reference: `P25`, the "
             f"paper's Algorithm 2 at 0.25 ({COST['P25']} passes).", "", "## Labels (frozen rules)", "", "```",
             json.dumps(res["labels"], indent=2), "```"]
    e, v = res["variety"], res["arm_verdicts"]
    if e:
        lines += ["", f"## Variety, `{DIV_COND}` (n={e['n']} speakers x {DIV_SEEDS} seeds)", "",
                  "| Arm | Passes | Saved vs P25 | CVf0 | Aligned CVf0 | Timing spread | Melody vs P25 | "
                  "Share of P25's gain: CVf0 | Aligned CVf0 | 90% confirmed (both) |",
                  "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |"]
        for a in ARMS:
            mn = e["means"][a]
            if a in (REFERENCE, BASE):
                sh_raw = sh_al = "100%" if a == REFERENCE else "0%"
                conf = ""
            else:
                sh_raw, sh_al = vp._share(e["cvf0"]["share"][a]), vp._share(e["cvf0_aligned"]["share"][a])
                conf = "yes" if all(e[m]["L90"][a]["lo"] > ZERO_TOL for m in VARIETY) else "no"
            lines.append(f"| `{a}` | {COST[a]} | {1 - COST[a] / COST[REFERENCE]:.0%} | {mn['cvf0']:.4f} | "
                         f"{mn['cvf0_aligned']:.4f} | {mn['timing_spread']:.2f} | {mn['melody']:.3f} | {sh_raw} | "
                         f"{sh_al} | {conf} |")
        lines += ["", "Handed-over estimate's distance from the 16-interval Step 8 estimate: "
                  + ", ".join(f"`{a}` {d:.3f}" for a, d in e["est_dist"].items())
                  + f". CWH4's mean kappa: {e['kappa_mean']:.3f}."]
    for cond, q in res["quality"].items():
        ref, label = ("vs_P25", "P25") if cond == "clean" else ("vs_STU", "STU")
        lines += ["", f"## Quality, `{cond}` (n={q['n']}; prompt SNR median {q['prompt_snr_median']:.1f} dB)", "",
                  f"| Arm | Passes | UTMOS | WER mean / median | Runaway | SIM | dUTMOS vs {label} | dWER hi / dSIM lo | "
                  "Bars |", "| --- | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |"]
        for a in ARMS:
            t = q["arms"][a]
            d = q[ref].get(a)
            cmp_ = (f"{_ci(d['utmos'])} | {d['wer']['hi']:+.4f} / {d['similarity']['lo']:+.4f} | "
                    f"{'pass' if d['ok'] else 'fail'}") if d else "(reference) | | "
            lines.append(f"| `{a}` | {t['passes']} | {t['utmos']:.3f} | {t['wer_mean']:.4f} / {t['wer_median']:.4f} | "
                         f"{t['runaway']} | {t['similarity']:.4f} | {cmp_} |")
    if v:
        lines += ["", "## Verdict per arm (clean prompts, against P25)", "",
                  "| Arm | Passes | Saved vs P25 | Variety >= 90% (point) | Confirmed | Quality bars | Label |",
                  "| --- | ---: | ---: | --- | --- | --- | --- |"]
        for a in TEST_ARMS:
            x = v[a]
            lines.append(f"| `{a}` | {x['passes']} | {x['saved_vs_P25']:.0%} | {'yes' if x['variety_point'] else 'no'} | "
                         f"{'yes' if x['variety_confirmed'] else 'no'} | {'pass' if x['quality'] else 'fail'} | "
                         f"`{x['label']}` |")
    t = res["timing"]
    if t:
        lines += ["", f"## Cost on the T4 (n={t['cases']} clean cases, each arm on its own)", "",
                  "| Arm | Passes (teacher + student) | Counted = table | Median seconds | Ratio to P25 |",
                  "| --- | --- | --- | ---: | ---: |"]
        for a in ARMS:
            x = t["arms"][a]
            lines.append(f"| `{a}` | {COST[a]} ({TEACHER_PASSES[a]} + {STUDENT_PASSES[a]}) | "
                         f"{'yes' if x['count_matches_cost'] else 'NO: ' + str(x['counted_passes'])} | "
                         f"{x['median_seconds']:.3f} | {x['ratio_to_P25']:.2f} |")
    r = res["snr_routing"]
    lines += ["", f"## SNR routing (descriptive): below {r['theta_db']:.0f} dB the student alone, otherwise the arm", "",
              f"Prompts below the gate: " + ", ".join(f"`{c}` {r['below_gate'][c]} of {r['cases'][c]}" for c in CONDS)
              + ".", "", "| Arm above the gate | Mean passes (all cases) | clean UTMOS | noisy15 UTMOS | noisy15 WER |",
              "| --- | ---: | ---: | ---: | ---: |"]
    for a, x in r["arms"].items():
        if "clean" in x and "noisy15" in x:
            lines.append(f"| `{a}` | {x['mean_passes']:.2f} | {x['clean']['utmos']:.3f} | {x['noisy15']['utmos']:.3f} | "
                         f"{x['noisy15']['wer']:.4f} |")
    lines += ["", "## Equal-cost contrasts (descriptive)", "", "```", json.dumps(res["equal_cost"], indent=2), "```",
              "", "## Reproductions", "", "EXP-032 quality (STU, W8, P25):", "", "```",
              json.dumps(res["exp032_quality_reproduction"], indent=2), "```", "", "EXP-032 clean variety:", "", "```",
              json.dumps(res["exp032_diversity_reproduction"], indent=2), "```", "",
              "EXP-026 quality (W8R2, W4R2; recorded, not a gate):", "", "```",
              json.dumps(res["exp026_quality_comparison"], indent=2), "```"]
    return "\n".join(lines)


def make_pipe(args):
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    return rf.Pipeline(args.dmo_root, args.device)


def main():
    p = argparse.ArgumentParser(description="EXP-033 cheaper teacher paths to the paper's t_switch = 0.25")
    sub = p.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("freeze")
    f.add_argument("--force", action="store_true")
    for name in ("smoke", "run"):
        s = sub.add_parser(name)
        s.add_argument("--dmo_root", required=True)
        s.add_argument("--out_dir", default=str(EXP_DIR / "results"))
        s.add_argument("--device", default="cuda")
        if name == "run":
            s.add_argument("--stage", choices=("all", "quality", "diversity", "timing"), default="all")
    a = sub.add_parser("analyze")
    a.add_argument("--out_dir", default=str(EXP_DIR / "results"))
    a.add_argument("--allow_partial", action="store_true")
    args = p.parse_args()
    if args.cmd == "freeze":
        freeze(force=args.force)
    elif args.cmd == "smoke":
        smoke(make_pipe(args), args.out_dir)
    elif args.cmd == "run":
        check_prereg()
        pipe = make_pipe(args)
        if args.stage in ("all", "quality"):
            run_quality(pipe, qual_items(), out_dir=args.out_dir)
        if args.stage in ("all", "diversity"):
            run_diversity(pipe, div_items(), out_dir=args.out_dir)
        if args.stage in ("all", "timing"):
            run_timing(pipe, time_items(), out_dir=args.out_dir)
    elif args.cmd == "analyze":
        analyze(args.out_dir, allow_partial=args.allow_partial)


if __name__ == "__main__":
    main()
