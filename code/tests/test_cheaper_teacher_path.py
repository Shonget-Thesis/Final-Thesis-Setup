"""Unit tests for EXP-033 (cheaper teacher paths to the paper's t_switch = 0.25). They run without PyTorch or the TTS
models: the teacher and the student are replaced by small deterministic functions on NumPy arrays."""
import csv
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code/colab"))
sys.path.insert(0, str(ROOT / "code/analysis"))
import run_cheaper_teacher_path as ct  # noqa: E402
import run_followup_tests as rf  # noqa: E402
import run_grid32_prelim as gl  # noqa: E402
import run_variety_per_pass as vp  # noqa: E402

TAUS = [1.0 - math.cos(math.pi / 2.0 * s) for s in (0.25, 0.50, 0.75)]


class Arr(np.ndarray):
    """A NumPy array with the one tensor method the samplers call."""

    def clone(self):
        return self.copy()


def arr(a):
    return np.asarray(a, dtype=float).view(Arr)


class DesignTests(unittest.TestCase):
    def test_costs(self):
        self.assertEqual(ct.COST, {"STU": 4, "W8": 21, "W8R2": 20, "W6R2": 16, "W4R2": 12, "P25": 30,
                                   "CPH12": 16, "CPH8": 12, "CPH6": 10, "GI": 15, "SST": 13, "CWH4": 13})
        for n in ct.CPH_GRIDS:
            self.assertEqual(ct.COST[f"CPH{n}"], n + 4)
        for k in ct.R2_STEPS:
            self.assertEqual(ct.COST[f"W{k}R2"], 2 * (k + 1) + 2)
        self.assertEqual(ct.COST["P25"], vp.COST["P25"])
        self.assertEqual(ct.COST["W8"], vp.COST["W8"])
        self.assertEqual(ct.COST["STU"], vp.COST["STU"])

    def test_every_coarse_grid_reaches_the_same_time(self):
        self.assertAlmostEqual(ct.T_COMMIT, 0.2929, places=4)
        self.assertAlmostEqual(ct.T_COMMIT, TAUS[1], places=12)          # the student's second level
        for n in ct.CPH_GRIDS:
            self.assertEqual(n % 2, 0)
            self.assertAlmostEqual(gl.sway_grid(n)[n // 2], ct.T_COMMIT, places=12)
            self.assertEqual(gl.first_above(gl.sway_grid(n), 0.25), n // 2)  # the released stopping rule at 0.25
        self.assertAlmostEqual(gl.sway_grid(16)[ct.SST_FIRST], TAUS[0], places=12)

    def test_arms_and_dependencies(self):
        self.assertEqual(set(ct.ARMS), set(ct.KNOWN) | set(ct.CANDIDATES))
        self.assertEqual(set(ct.TEST_ARMS), set(ct.ARMS) - {"P25", "STU"})
        self.assertTrue(set(ct.EST_ARMS) <= set(ct.CANDIDATES))
        self.assertEqual((ct.REFERENCE, ct.BASE), ("P25", "STU"))
        for name in ct.DEPENDENCIES:
            self.assertTrue((ct.CODE_DIR / name).exists(), name)
        for name in ("run_cheaper_teacher_path.py", "run_variety_per_pass.py", "run_grid32_prelim.py"):
            self.assertIn(name, ct.DEPENDENCIES)
        self.assertEqual(ct.THETA, 20.0)

    def test_cohorts_match_exp032_and_exp026(self):
        q = [it["utt_id"] for it in ct.qual_items()]
        self.assertEqual(q, [it["utt_id"] for it in vp.qual_items()])
        self.assertEqual([it["utt_id"] for it in ct.div_items()], [it["utt_id"] for it in vp.div_items()])
        self.assertEqual([it["utt_id"] for it in ct.time_items()], q[:10])
        e32 = {(r["utt_id"], r["condition"]) for r in rf.read_csv(ct.EXP032_QUAL_CSV)}
        e26 = {r["utt_id"] for r in rf.read_csv(ct.EXP026_QUAL_CSV)}
        for u in q:
            self.assertIn(u, e26)
            for c in ct.CONDS:
                self.assertIn((u, c), e32)
        d32 = {r["utt_id"] for r in rf.read_csv(ct.EXP032_DIV_CSV) if r["condition"] == ct.DIV_COND}
        self.assertEqual(set(q[:20]), d32)
        for c in ct.CONDS:
            self.assertIn(c, rf.CONDITIONS)


class ArithmeticTests(unittest.TestCase):
    def test_commitment(self):
        prev = np.zeros((1, 5, 4))
        curr = np.zeros((1, 5, 4))
        curr[0, 2:, 0] = 1.0                       # only band 0 of the generated frames changed
        k = ct.commitment(curr, prev, 2)
        self.assertEqual(k.shape, (1, 3, 4))
        rms = math.sqrt(3 / 12)
        np.testing.assert_allclose(k[0, :, 1:], 1.0)                      # unchanged cells are fully committed
        np.testing.assert_allclose(k[0, :, 0], 1.0 / (1.0 + (1.0 / rms) ** 2))
        d = np.full((1, 6, 3), 0.7)
        np.testing.assert_allclose(ct.commitment(d, np.zeros_like(d), 0), 0.5)   # |D| = rms(D) gives one half
        np.testing.assert_allclose(ct.commitment(d, d, 0), 1.0)                  # no change at all

    def test_commitment_ignores_the_scale_of_the_change(self):
        rng = np.random.default_rng(0)
        a, b = rng.normal(size=(1, 9, 4)), rng.normal(size=(1, 9, 4))
        np.testing.assert_allclose(ct.commitment(a, b, 3), ct.commitment(b + 10.0 * (a - b), b, 3))

    def test_mix(self):
        rng = np.random.default_rng(1)
        t, s = arr(rng.normal(size=(1, 7, 4))), arr(rng.normal(size=(1, 7, 4)))
        np.testing.assert_array_equal(ct.cwh_mix(t, s, 1.0, 2), t)        # kappa = 1: the teacher's estimate
        x0 = ct.cwh_mix(t, s, 0.0, 2)
        np.testing.assert_array_equal(x0[:, 2:], s[:, 2:])                # kappa = 0: the student's draft ...
        np.testing.assert_array_equal(x0[:, :2], t[:, :2])                # ... and the prompt frames stay the teacher's
        k = ct.commitment(t, s, 2)
        x = ct.cwh_mix(t, s, k, 2)
        np.testing.assert_allclose(x[:, 2:], k * t[:, 2:] + (1 - k) * s[:, 2:])
        self.assertIsNot(x, t)

    def test_seeded_state_and_distance(self):
        y0, s0 = np.full((1, 4, 2), 2.0), np.full((1, 4, 2), -1.0)
        np.testing.assert_allclose(ct.seeded_state(y0, s0, 0.25), 0.75 * 2.0 + 0.25 * -1.0)
        np.testing.assert_allclose(ct.seeded_state(y0, s0, 0.0), y0)
        a, b = np.ones((1, 5, 3)), np.ones((1, 5, 3))
        self.assertEqual(ct.rel_dist(a, b, 2), 0.0)
        a[0, :2] = 99.0                                                    # prompt frames are left out
        self.assertEqual(ct.rel_dist(a, b, 2), 0.0)
        self.assertAlmostEqual(ct.rel_dist(2 * b, b, 2), 1.0)

    def test_melody_agreement_matches_the_bank_screen(self):
        import exp032_bank_pitch_screen as bs
        t = np.linspace(0, 1, 120)
        ref = 140.0 * 2 ** (np.sin(2 * np.pi * 2 * t) / 6)
        near, far = ref * 2 ** (0.5 / 12), ref * 2 ** (2 / 12)
        mixed = np.where(t < 0.5, near, far)
        mixed[:10] = np.nan
        for track in (ref, near, far, mixed):
            self.assertAlmostEqual(ct.melody_agreement(track, ref), bs.agreement(track, ref)["within_semitone"])
        self.assertEqual(ct.melody_agreement(ref, ref), 1.0)
        self.assertEqual(ct.melody_agreement(far, ref), 0.0)
        self.assertIsNone(ct.melody_agreement(np.full(120, np.nan), ref))


class _NoGrad:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _Torch:
    float32 = "float32"

    @staticmethod
    def inference_mode():
        return _NoGrad()

    @staticmethod
    def tensor(v, device=None, dtype=None):
        return float(v)


class _Sampler:
    STUDENT_TAUS = TAUS
    SWAY_TEACHER_GRID = gl.sway_grid(16)
    CFG_STRENGTH = 2.0

    @staticmethod
    def compute_stabilization_drift(curr, prev, plen):
        return float(np.linalg.norm(curr[:, plen:] - prev[:, plen:]) / (np.linalg.norm(curr[:, plen:]) + 1e-8))


class _Teacher:
    """A deterministic velocity field: the conditional and unconditional branches differ, so guidance matters."""

    def __init__(self, counter):
        self.counter = counter

    def __call__(self, x, cond, text, time, mask, drop_audio_cond, drop_text):
        self.counter["teacher"] += 1
        pull = 0.0 if drop_audio_cond else 0.3 * cond
        return arr(np.sin(3.0 * x) * (1.0 - time) + pull - 0.5 * x)


class _Tts:
    def __init__(self, counter):
        self.teacher = _Teacher(counter)


class _Pipe:
    torch = _Torch()
    sampler = _Sampler()
    device = "cpu"

    def __init__(self, counter):
        self.tts = _Tts(counter)


class SamplerTests(unittest.TestCase):
    """The real teacher loops and the arm wiring, with a NumPy teacher and student in place of the networks."""

    def setUp(self):
        self.counter = {"teacher": 0, "student": 0}
        self.pipe = _Pipe(self.counter)
        rng = np.random.default_rng(3)
        n, self.plen = 14, 5
        self.ctx = {"prompt_len": self.plen, "y0": arr(rng.normal(size=(1, n, 6))),
                    "step_cond": arr(rng.normal(size=(1, n, 6))), "text_ids": None, "cond": None}
        self.saved = (ct.gl.student, ct.gl.student_from_state)
        counter = self.counter

        def student(pipe, ctx, endpoint, taus):
            x = endpoint
            for tau in taus:
                counter["student"] += 1
                x = arr(0.5 * x + tau + 0.1 * np.cos(x))
            return x

        def student_from_state(pipe, ctx, state, times):
            counter["student"] += len(times)
            return arr(0.25 * state + sum(times))

        ct.gl.student, ct.gl.student_from_state = student, student_from_state

    def tearDown(self):
        ct.gl.student, ct.gl.student_from_state = self.saved

    def reset(self):
        self.counter["teacher"] = self.counter["student"] = 0

    def test_teacher_loops_agree_with_the_released_loop(self):
        g16 = gl.sway_grid(16)
        ends, _, states = gl.grid_teacher(self.pipe, self.ctx, g16, {4, 8})
        y0 = self.ctx["y0"]
        np.testing.assert_allclose(ct.euler(self.pipe, self.ctx, g16, y0.clone(), 0, 8), ends[8], rtol=0, atol=1e-12)
        np.testing.assert_allclose(ct.euler(self.pipe, self.ctx, g16, states[4], 4, 8), ends[8], rtol=0, atol=1e-12)
        np.testing.assert_allclose(ct.euler(self.pipe, self.ctx, g16, y0.clone(), 0, 8, guided_until=8), ends[8],
                                   rtol=0, atol=1e-12)
        np.testing.assert_allclose(ct.teacher_state(self.pipe, self.ctx, g16, 4), states[4], rtol=0, atol=1e-12)
        unguided = ct.euler(self.pipe, self.ctx, g16, y0.clone(), 0, 8, guided_until=3)
        self.assertGreater(float(np.abs(unguided - ends[8]).max()), 1e-3)   # dropping guidance changes the estimate

    def test_each_arm_alone_equals_the_shared_run_and_costs_its_passes(self):
        outs, parts = ct.synthesize(self.pipe, self.ctx, ct.ARMS)
        self.assertEqual(tuple(outs), ct.ARMS)
        self.assertEqual((self.counter["teacher"], self.counter["student"]), (95, 28))   # the shared run's own cost
        for arm in ct.ARMS:
            self.reset()
            alone = ct.standalone(self.pipe, self.ctx, arm)
            np.testing.assert_allclose(alone, outs[arm], rtol=0, atol=1e-10, err_msg=arm)
            self.assertEqual((self.counter["teacher"], self.counter["student"]),
                             (ct.TEACHER_PASSES[arm], ct.STUDENT_PASSES[arm]), arm)
        self.assertEqual(set(parts["est_dist"]), set(ct.EST_ARMS))
        self.assertTrue(0.0 < parts["kappa_mean"] < 1.0)
        with self.assertRaises(ValueError):
            ct.standalone(self.pipe, self.ctx, "W5")

    def test_which_estimate_feeds_which_arm(self):
        g16 = gl.sway_grid(16)
        e16, _, s16 = gl.grid_teacher(self.pipe, self.ctx, g16, {3, 4, 6, 8})
        outs, parts = ct.synthesize(self.pipe, self.ctx, ct.ARMS)
        stu, taus, c = ct.gl.student, TAUS, self.ctx
        np.testing.assert_allclose(outs["W8"], stu(None, c, e16[8], taus))
        for k in ct.R2_STEPS:
            np.testing.assert_allclose(outs[f"W{k}R2"], stu(None, c, e16[k], taus[1:]))
        s0 = stu(None, c, c["step_cond"], [0.0])
        np.testing.assert_allclose(parts["s0"], s0)
        np.testing.assert_allclose(outs["STU"], stu(None, c, s0, taus))
        s32 = ct.teacher_state(self.pipe, c, gl.GRID32, 14)
        np.testing.assert_allclose(outs["P25"], ct.gl.student_from_state(None, c, s32, list(gl.ALG2_TIMES)))
        for n in ct.CPH_GRIDS:
            ends, _, _ = gl.grid_teacher(self.pipe, c, gl.sway_grid(n), {n // 2})
            np.testing.assert_allclose(outs[f"CPH{n}"], stu(None, c, ends[n // 2], taus[1:]))
            self.assertGreater(parts["est_dist"][f"CPH{n}"], 0.0)
        gi = ct.euler(self.pipe, c, g16, s16[4], 4, 8, guided_until=3)
        np.testing.assert_allclose(outs["GI"], stu(None, c, gi, taus[1:]))
        x4 = (1.0 - g16[4]) * c["y0"] + g16[4] * s0
        np.testing.assert_allclose(outs["SST"], stu(None, c, ct.euler(self.pipe, c, g16, x4, 4, 8), taus[1:]))
        s1 = stu(None, c, e16[4], taus[:1])
        kappa = ct.commitment(e16[4], e16[3], self.plen)
        np.testing.assert_allclose(outs["CWH4"], stu(None, c, ct.cwh_mix(e16[4], s1, kappa, self.plen), taus[1:]))
        np.testing.assert_allclose(stu(None, c, s1, taus[1:]), stu(None, c, e16[4], taus))   # the split student
        self.assertAlmostEqual(parts["kappa_mean"], float(kappa.mean()))

    def test_arms_without_the_paper_setting_skip_its_teacher(self):
        arms = tuple(a for a in ct.ARMS if a != "P25")
        outs, _ = ct.synthesize(self.pipe, self.ctx, arms)
        self.assertEqual(tuple(outs), arms)
        self.assertEqual(self.counter["teacher"], 95 - 30)


def div_row(it, shares, rng, noise=0.0005):
    row = {"utt_id": it["utt_id"], "speaker_id": it["speaker_id"], "condition": ct.DIV_COND, "n_seeds": 12,
           "kappa_mean": 0.66, **{f"est_dist_{a}": 0.1 for a in ct.EST_ARMS}, "case_seconds": 1.0, "utc": "x"}
    for a in ct.ARMS:
        s = shares.get(a, 1.0 if a == "P25" else 0.0)
        row.update({f"cvf0_{a}": 0.05 + 0.04 * s + float(rng.normal(0.0, noise)),
                    f"cvf0_aligned_{a}": 0.04 + 0.02 * s + float(rng.normal(0.0, noise)),
                    f"voiced_{a}": 0.6, f"voiced_aligned_{a}": 0.6, f"mel_spread_{a}": 0.3,
                    f"timing_spread_{a}": 4.0 + 6.0 * s, f"melody_{a}": 0.4 + 0.4 * s if a != "P25" else 1.0})
    return row


PROFILES = {
    # share of P25's gain per arm; clean UTMOS offset from P25 per arm
    "cph8": ({"W8": 0.5, "W8R2": 1.0, "W6R2": 0.8, "W4R2": 0.7, "CPH12": 1.0, "CPH8": 0.97, "CPH6": 0.6, "GI": 0.95,
              "SST": 0.5, "CWH4": 0.85}, {"W4R2": -0.3, "CPH6": -0.3, "W8": 0.15}),
    "nothing": ({a: 0.5 for a in ct.TEST_ARMS} | {"W8R2": 0.8}, {}),
    "quality_gate": ({"W8": 0.5, "W8R2": 1.0, "W6R2": 0.5, "W4R2": 0.5, "CPH12": 1.0, "CPH8": 1.0, "CPH6": 0.5,
                      "GI": 1.0, "SST": 0.5, "CWH4": 0.5}, {"CPH8": -0.3}),
}


def write_synthetic(tmp, profile, with_div=True, with_timing=True):
    shares, offsets = PROFILES[profile]
    rng = np.random.default_rng(7)
    with open(Path(tmp) / "quality.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=ct.QUAL_FIELDS)
        w.writeheader()
        for it in ct.qual_items():
            for cond in ct.CONDS:
                row = {"utt_id": it["utt_id"], "speaker_id": it["speaker_id"], "condition": cond,
                       "prompt_snr": 35.0 if cond == "clean" else 18.0, "gpu": "UnitTest", "frames": 400,
                       "seq_len": 800, **{f"d_{j}": 0.3 for j in range(1, 9)}, "kappa_mean": 0.66,
                       **{f"est_dist_{a}": 0.1 for a in ct.EST_ARMS}, "case_seconds": 1.0, "utc": "x"}
                for a in ct.ARMS:
                    if cond == "clean":
                        utm = 3.7 + offsets.get(a, 0.0)
                    else:
                        utm = 3.3 if a == "STU" else 1.9      # every teacher arm copies the noise
                    row.update({f"wer_{a}": 0.0, f"similarity_{a}": 0.93,
                                f"utmos_{a}": utm + float(rng.normal(0.0, 0.01)), f"hypothesis_{a}": "x"})
                w.writerow(row)
    if with_div:
        with open(Path(tmp) / "diversity.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=ct.DIV_FIELDS)
            w.writeheader()
            for it in ct.div_items():
                w.writerow(div_row(it, shares, rng))
    if with_timing:
        with open(Path(tmp) / "timing.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=ct.TIME_FIELDS)
            w.writeheader()
            for it in ct.time_items():
                row = {"utt_id": it["utt_id"], "speaker_id": it["speaker_id"], "condition": "clean", "gpu": "UnitTest",
                       "frames": 400, "seq_len": 800, "utc": "x"}
                for a in ct.ARMS:
                    row.update({f"seconds_{a}": 0.01 * ct.COST[a], f"teacher_passes_{a}": ct.TEACHER_PASSES[a],
                                f"student_passes_{a}": ct.STUDENT_PASSES[a]})
                w.writerow(row)


class AnalysisTests(unittest.TestCase):
    def _run(self, profile, with_div=True, allow_partial=False, with_timing=True):
        with tempfile.TemporaryDirectory() as tmp:
            prereg = Path(tmp) / "preregistration.json"
            prereg.write_text(json.dumps(ct.build_prereg(), indent=2) + "\n", encoding="utf-8")
            write_synthetic(tmp, profile, with_div, with_timing)
            res = ct.analyze(tmp, prereg_path=prereg, allow_partial=allow_partial)
            for name in ("CHEAPER_TEACHER_PATH_REPORT.md", "MANIFEST.json", "cheaper_teacher_path_results.json"):
                self.assertTrue((Path(tmp) / name).exists(), name)
            self.assertIn("clean", res["exp032_quality_reproduction"])
            self.assertIn("Verdict per arm" if with_div else "Labels", (Path(tmp) / "CHEAPER_TEACHER_PATH_REPORT.md")
                          .read_text(encoding="utf-8"))
            return res

    def test_labels(self):
        res = self._run("cph8")
        lab = res["labels"]
        self.assertEqual(lab["arm_labels"], {
            "W8": "variety_short", "W8R2": "matches_paper_confirmed", "W6R2": "variety_short", "W4R2": "both_short",
            "CPH12": "matches_paper_confirmed", "CPH8": "matches_paper_confirmed", "CPH6": "both_short",
            "GI": "matches_paper_confirmed", "SST": "variety_short", "CWH4": "variety_short"})
        self.assertEqual((lab["cheapest_match_point"], lab["cheapest_match_confirmed"]), ("CPH8", "CPH8"))
        self.assertTrue(lab["w8r2_matches_paper"])
        self.assertEqual(lab["noisy_quality_safe_arms"], [])
        v = res["arm_verdicts"]["CPH8"]
        self.assertEqual((v["passes"], round(v["saved_vs_P25"], 2)), (12, 0.6))
        self.assertAlmostEqual(res["variety"]["cvf0"]["share"]["CPH8"]["point"], 0.97, delta=0.03)

    def test_no_match_and_quality_gate(self):
        lab = self._run("nothing")["labels"]
        self.assertEqual((lab["cheapest_match_point"], lab["cheapest_match_confirmed"]), ("none", "none"))
        self.assertFalse(lab["w8r2_matches_paper"])
        lab = self._run("quality_gate")["labels"]
        self.assertEqual(lab["arm_labels"]["CPH8"], "quality_short")
        self.assertEqual(lab["cheapest_match_point"], "GI")                # 15 passes; CPH12 (16) and W8R2 (20) also match

    def test_cost_routing_and_contrasts(self):
        res = self._run("cph8")
        t = res["timing"]
        self.assertEqual(t["cases"], 10)
        for a in ct.ARMS:
            self.assertTrue(t["arms"][a]["count_matches_cost"], a)
            self.assertAlmostEqual(t["arms"][a]["ratio_to_P25"], ct.COST[a] / 30)
        r = res["snr_routing"]
        self.assertEqual(r["below_gate"], {"clean": 0, "noisy15": 30})
        self.assertAlmostEqual(r["arms"]["CPH8"]["mean_passes"], (12 + 4) / 2)
        self.assertAlmostEqual(r["arms"]["CPH8"]["noisy15"]["utmos"], 3.3, delta=0.02)   # the student alone's score
        self.assertAlmostEqual(r["arms"]["CPH8"]["clean"]["passes"], 12)
        self.assertNotIn("STU", r["arms"])
        e = res["equal_cost"]["CPH8_minus_W4R2"]
        self.assertEqual(e["passes"], 12)
        self.assertGreater(e["quality_clean"]["utmos"]["lo"], 0.0)
        self.assertGreater(e["variety"]["cvf0"]["lo"], 0.0)
        self.assertIn("CPH12_minus_W6R2", res["equal_cost"])
        self.assertIsNone(self._run("cph8", with_timing=False)["timing"])

    def test_a_wrong_pass_count_is_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_synthetic(tmp, "cph8")
            rows = rf.read_csv(Path(tmp) / "timing.csv")
            rows[0]["teacher_passes_CPH8"] = 12
            self.assertFalse(ct.timing_table(rows)["arms"]["CPH8"]["count_matches_cost"])
            self.assertTrue(ct.timing_table(rows)["arms"]["CPH6"]["count_matches_cost"])

    def test_partial(self):
        res = self._run("cph8", with_div=False, allow_partial=True)
        self.assertEqual(set(res["labels"].values()), {"partial"})
        self.assertIsNone(res["arm_verdicts"])
        with self.assertRaises(RuntimeError):
            self._run("cph8", with_div=False, allow_partial=False)


class LocalScreenTests(unittest.TestCase):
    def test_the_local_screen_uses_the_runners_costs_levels_and_seeds(self):
        import local_candidate_screen as ls
        for arm, passes in ls.PASSES.items():
            self.assertEqual(passes, ct.COST[arm], arm)
        self.assertTrue(set(ls.CANDIDATES) <= set(ct.CANDIDATES))
        np.testing.assert_allclose(ls.TAUS, TAUS, rtol=0, atol=1e-15)
        self.assertEqual(ls.DIV_SEED_BASE, ct.DIV_SEED_BASE)
        self.assertEqual(ls.sway(16), gl.sway_grid(16))


class PreregistrationAndNotebookTests(unittest.TestCase):
    def test_tampering_is_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "preregistration.json"
            doc = ct.build_prereg()
            p.write_text(json.dumps(doc), encoding="utf-8")
            ct.check_prereg(p)
            doc["bars"]["share"] = 0.8
            p.write_text(json.dumps(doc), encoding="utf-8")
            with self.assertRaises(RuntimeError):
                ct.check_prereg(p)

    def test_committed_preregistration_and_notebook(self):
        if ct.PREREG_PATH.exists():
            doc = ct.check_prereg()
            self.assertEqual((doc["experiment"], doc["task_id"]), ("EXP-033", "EVAL-026"))
            self.assertEqual(doc["reference"]["arm"], "P25")
            self.assertEqual(doc["conditions"], ["clean", "noisy15"])
            self.assertEqual(doc["candidates"], list(ct.CANDIDATES))
            self.assertEqual(doc["costs"], ct.COST)
            self.assertIn("local", doc["local_screen"])
        nb_path = ROOT / "code/colab/DMOSpeech2_EXP033_Cheaper_Teacher_Path.ipynb"
        nb = json.loads(nb_path.read_text(encoding="utf-8"))
        code = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
        for i, src in enumerate(code):
            compile("\n".join(l for l in src.splitlines() if not l.lstrip().startswith(("!", "%"))), f"cell{i}", "exec")
        joined = "\n".join(code)
        self.assertIn("run_cheaper_teacher_path.py", joined)
        self.assertIn("EXP-033-cheaper-teacher-path", joined)
        self.assertIn("DMOSpeech2_EXP033_Cheaper_Teacher_Path_Results.zip", joined)
        self.assertIn("exp033_cheaper_teacher_path", joined)
        self.assertIn("timing.csv", joined)
        self.assertNotIn("EXP-032", joined)
        self.assertNotIn("exp032", joined)
        self.assertNotIn("Tensor_Bank", joined)
        self.assertIn('STAGE = "all"', joined)
        md = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "markdown")
        for word in ("0.25", "noisy15", "CPH8", "CWH4", "EVAL-026"):
            self.assertIn(word, md)


if __name__ == "__main__":
    unittest.main()
