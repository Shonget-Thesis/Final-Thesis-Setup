import csv
import hashlib
import importlib.util
import json
import math
import random
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code/colab"))
import run_analytic_guidance_pilot as ag  # noqa: E402

HAVE_NP = importlib.util.find_spec("numpy") is not None
GRID16 = [1.0 - math.cos(math.pi / 2.0 * i / 16.0) for i in range(17)]


class DesignTests(unittest.TestCase):
    def test_forward_passes(self):
        self.assertEqual((ag.cost("W4"), ag.cost("W8")), (13, 21))
        self.assertEqual({K: ag.cost(f"W4_K{K}") for K in ag.KS}, {0: 12, 2: 10, 4: 8})
        self.assertEqual({K: ag.cost(f"W8_K{K}") for K in ag.KS}, {0: 20, 2: 18, 4: 16})
        self.assertEqual(ag.parse("W8_K4"), (8, 4))
        self.assertEqual(ag.parse("W4"), (4, -1))

    def test_pilot_and_calibration_speakers_are_disjoint(self):
        pilot, cal = ag.pilot_items(), ag.calibration_items()
        self.assertEqual((len(pilot), len(cal)), (40, 20))
        self.assertFalse({it["speaker_id"] for it in pilot} & {it["speaker_id"] for it in cal})
        self.assertEqual([it["utt_id"] for it in pilot[:ag.DIV_N]], [it["utt_id"] for it in ag.ex.items()[:ag.DIV_N]])

    def test_every_pilot_case_has_stored_labels(self):
        stored, w48 = ag.load_stored(), ag.stored_w48()
        cases = [(it["utt_id"], c) for it in ag.pilot_items() for c in ag.CONDS]
        self.assertEqual(len(cases), 80)
        self.assertTrue(all(k in stored and k in w48 for k in cases))
        self.assertEqual(sorted(stored[cases[0]]["m"]), ["W4", "W8"])
        smoke = [(it["utt_id"], c) for it in ag.calibration_items()[-2:] for c in ag.CONDS]
        self.assertTrue(all(k in w48 for k in smoke))

    def test_diversity_reuses_the_exp020_seeds_and_speakers(self):
        self.assertEqual(ag.DIV_SEED_BASE, ag.sh.DIV_SEED_BASE)
        cv = ag.stored_cvf0()
        self.assertTrue(all(it["speaker_id"] in cv for it in ag.pilot_items()[:ag.DIV_N]))
        self.assertEqual(sorted(next(iter(cv.values()))), ["W4", "W8"])

    def test_extra_noise_seeds_stay_clear_of_the_case_and_student_streams(self):
        extra = {ag.LEAK_SEED_BASE + j for j in range(ag.LEAK_DRAWS)}
        used = {0, ag.sh.STUDENT_SAMPLE_OFFSET} | {int(t * 100) for t in (0.0761, 0.2929, 0.6173)} | \
               {ag.DIV_SEED_BASE + m for m in range(ag.DIV_SEEDS)}
        self.assertFalse(extra & used)

    def test_the_power_check_supports_the_bounds(self):
        p = ag.power_summary()
        self.assertGreaterEqual(p["clean_step4_40"], 0.95)            # 40 speakers resolve the clean Step-4 bounds
        self.assertLess(p["clean_step4_20_utmos"], 0.8)                # 20 would not
        self.assertGreaterEqual(p["clean_step8_40_wer_0.015"], 0.95)
        self.assertLess(p["clean_step8_40_wer_0.01"], 0.5)             # why the Step-8 clean WER bound is +0.015
        self.assertLess(p["noisy_step4_40"], 0.1)                      # why noisy Step-4 handovers are not judged
        self.assertEqual(ag.BOUNDS[("W8", "clean")]["wer"], 0.015)
        self.assertEqual(ag.BOUNDS[("W4", "clean")]["wer"], 0.01)

    def test_the_32_interval_grid_lands_on_the_handover_times(self):
        for k, k32 in ag.GRID_K.items():
            self.assertAlmostEqual(ag.gl.GRID32[k32], GRID16[k], places=12)


class _NoContext:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _FakeTTS:
    """A linear stand-in for the teacher; with exact_mu set, the unconditional branch is exactly the closed form."""

    def __init__(self, np, exact_mu=None):
        self.np, self.calls, self.exact_mu = np, [], exact_mu

    def teacher(self, x, cond, text, time, mask, drop_audio_cond, drop_text):
        self.calls.append("uncond" if drop_audio_cond else "cond")
        if drop_audio_cond:
            if self.exact_mu is not None:
                return (self.exact_mu - x) / (1.0 - time)
            return -0.4 * x + 0.05 * time + 0.01 * self.np.sin(x)
        return -0.5 * x + 0.1 * cond + 0.2 * time


def reference_single_pass(tts, y0, cond, grid, kmax, cfg):
    """sampler.run_teacher_single_pass written out again, for comparison."""
    state, ends = y0.copy(), {}
    for i in range(kmax + 1):
        t = grid[i]
        pred = tts.teacher(x=state, cond=cond, text=None, time=t, mask=None, drop_audio_cond=False, drop_text=False)
        null = tts.teacher(x=state, cond=cond, text=None, time=t, mask=None, drop_audio_cond=True, drop_text=True)
        v = pred + (pred - null) * cfg
        ends[i] = state + (1.0 - t) * v
        if i < kmax:
            state = state + (grid[i + 1] - t) * v
    return ends


class _FakePipe:
    gpu = "Tesla T4 (fake)"

    def __init__(self, exact_mu=None):
        import numpy as np
        self.np = np
        self.torch = type("T", (), {"inference_mode": staticmethod(_NoContext), "tensor": staticmethod(lambda v, device=None, dtype=None: v),
                                    "float32": None})
        self.sampler = type("S", (), {"STUDENT_TAUS": [0.0761, 0.2929, 0.6173], "SWAY_TEACHER_GRID": GRID16, "CFG_STRENGTH": 2.0})
        self.device = type("D", (), {"type": "cpu"})()
        self.tts = _FakeTTS(np, exact_mu)
        self.teacher_calls = []

    def prepare(self, item, cond, seed, out_dir, noise_seed=None):
        rng = self.np.random.default_rng(int(seed))
        return {"prompt_len": 10, "prompt_snr": 30.0, "seed": seed, "step_cond": self.np.ones((1, 50, 100)) * (2.0 if cond == "clean" else 1.0),
                "text_ids": None, "y0": rng.normal(size=(1, 50, 100))}

    def teacher(self, ctx, steps):
        self.teacher_calls.append(tuple(steps))
        return reference_single_pass(self.tts, ctx["y0"], ctx["step_cond"], GRID16, max(steps), 2.0), {}

    def ground_truth(self, item):
        return self.np.zeros(10), 24000

    def score(self, audio, item, gt):
        return {"wer": 0.1, "similarity": 0.8, "utmos": 3.0, "hypothesis": "text"}


@unittest.skipUnless(HAVE_NP, "numpy is needed")
class FormulaTests(unittest.TestCase):
    def setUp(self):
        import numpy as np
        self.np, self.plen = np, 12
        rng = np.random.default_rng(0)
        self.Tc, self.Tu = rng.normal(size=(1, 60, 100)), rng.normal(size=(1, 60, 100))

    def test_the_closed_form_gap_is_in_units_of_the_guidance_signal(self):
        np, p = self.np, self.plen
        mu = self.Tu[0, p:].mean(axis=0).reshape(1, 1, -1)
        e = ag.closed_form_gap(mu, self.Tc, self.Tu, p)
        expected = np.linalg.norm(mu[0] - self.Tu[0, p:]) / np.linalg.norm(self.Tc[0, p:] - self.Tu[0, p:])
        self.assertAlmostEqual(e, float(expected), places=7)          # the runner adds 1e-8 to the denominator
        flat = np.broadcast_to(mu, self.Tu.shape).copy()
        self.assertAlmostEqual(ag.closed_form_gap(mu, self.Tc, flat, p), 0.0, places=12)

    def test_the_step0_split_separates_seed_and_position(self):
        np, p = self.np, self.plen
        profile = np.linspace(-1.0, 1.0, 60)[None, :, None] * np.ones((1, 60, 100))
        mu = np.zeros((1, 1, 100))
        same = [profile.copy() for _ in range(3)]
        total, seed, position = ag.step0_split(self.Tc, profile, same, mu, p)
        self.assertAlmostEqual(seed, 0.0, places=12)                   # every draw agrees: nothing seed-specific
        self.assertAlmostEqual(position, total, places=10)             # the whole gap is the position profile
        flat = np.zeros((1, 60, 100))
        total, seed, position = ag.step0_split(self.Tc, self.Tu, [flat, flat, flat], mu, p)
        self.assertAlmostEqual(position, 0.0, places=12)
        self.assertAlmostEqual(seed, total, places=10)                 # the draws average to mu: the gap is seed-specific

    def test_the_slow_part_change_ignores_the_band_means(self):
        np, p = self.np, self.plen
        b = self.Tu
        a = b + np.arange(100.0)[None, None, :]                         # a change only in each band's average over time
        self.assertGreater(ag.rel_change(a, b, p), 0.1)
        self.assertAlmostEqual(ag.rel_change(a, b, p, "slow"), 0.0, places=10)

    def test_mu_broadcasts_over_frames(self):
        pipe = _FakePipe()
        mu = ag.as_mu(pipe, list(range(100)))
        self.assertEqual(mu.shape, (1, 1, 100))
        self.assertEqual((mu - self.Tu).shape, self.Tu.shape)


@unittest.skipUnless(HAVE_NP, "numpy is needed")
class LoopTests(unittest.TestCase):
    def test_aug_switched_off_is_the_released_teacher(self):
        pipe = _FakePipe()
        ctx = pipe.prepare({}, "clean", 7, None)
        ends, passes, rec = ag.guided_loop(pipe, ctx, 8, record=True)
        ref = reference_single_pass(pipe.tts, ctx["y0"], ctx["step_cond"], GRID16, 8, 2.0)
        self.assertEqual(passes, 18)
        for k in range(9):
            pipe.np.testing.assert_allclose(ends[k], ref[k], atol=1e-12)
        self.assertEqual(sorted(rec), list(range(9)))
        t, x = GRID16[0], ctx["y0"]
        vc = pipe.tts.teacher(x=x, cond=ctx["step_cond"], text=None, time=t, mask=None, drop_audio_cond=False, drop_text=False)
        vu = pipe.tts.teacher(x=x, cond=ctx["step_cond"], text=None, time=t, mask=None, drop_audio_cond=True, drop_text=True)
        pipe.np.testing.assert_allclose(rec[0][0], x + vc, atol=1e-12)
        pipe.np.testing.assert_allclose(rec[0][1], x + vu, atol=1e-12)

    def test_aug_skips_the_unconditional_evaluations_up_to_step_k(self):
        pipe = _FakePipe()
        ctx = pipe.prepare({}, "clean", 7, None)
        mu = pipe.np.zeros((1, 1, 100))
        for K in ag.KS:
            pipe.tts.calls.clear()
            _, passes, rec = ag.guided_loop(pipe, ctx, 8, K=K, mu=mu, record=True)
            self.assertEqual(passes, ag.cost(f"W8_K{K}") - 3)
            self.assertEqual(pipe.tts.calls.count("uncond"), 8 - K)
            self.assertEqual(sorted(rec), list(range(K + 1, 9)))

    def test_an_unconditional_branch_equal_to_the_closed_form_makes_aug_exact(self):
        import numpy as np
        mu = np.random.default_rng(5).normal(size=(1, 1, 100))
        pipe = _FakePipe(exact_mu=mu)
        ctx = pipe.prepare({}, "clean", 11, None)
        released = ag.guided_loop(pipe, ctx, 8)[0]
        for K in ag.KS:
            aug = ag.guided_loop(pipe, ctx, 8, K=K, mu=mu)[0]
            for k in (4, 8):
                np.testing.assert_allclose(aug[k], released[k], atol=1e-10)
        other = ag.guided_loop(pipe, ctx, 8, K=4, mu=mu + 0.5)[0]       # a wrong mu does move the estimate
        self.assertGreater(ag.rel_change(other[8], released[8], ctx["prompt_len"]), 1e-3)

    def test_keep_limits_the_stored_estimates(self):
        pipe = _FakePipe()
        ctx = pipe.prepare({}, "clean", 3, None)
        ends = ag.guided_loop(pipe, ctx, 16, grid=ag.gl.GRID32, keep=(8, 16))[0]
        self.assertEqual(sorted(ends), [8, 16])


@unittest.skipUnless(HAVE_NP, "numpy is needed")
class RunLoopTests(unittest.TestCase):
    def setUp(self):
        import numpy as np
        self.saved = (ag.gl.student, ag.ng.decode, ag.sh.f0_track, ag.extra_noise)
        self.pipe = _FakePipe()
        self.student_calls = 0

        def fake_student(pipe, ctx, x, taus):
            self.student_calls += 1
            return x
        ag.gl.student = fake_student
        ag.ng.decode = lambda pipe, ctx, x: x[0, ctx["prompt_len"]:, 0]
        ag.sh.f0_track = lambda audio: np.abs(audio) * 10 + 100
        ag.extra_noise = lambda pipe, ctx, j: np.random.default_rng(int(ctx["seed"]) + ag.LEAK_SEED_BASE + j).normal(size=ctx["y0"].shape)

    def tearDown(self):
        ag.gl.student, ag.ng.decode, ag.sh.f0_track, ag.extra_noise = self.saved

    def test_calibration_pools_every_frame_and_is_reused(self):
        items = ag.calibration_items()[:2]
        with tempfile.TemporaryDirectory() as tmp:
            mu = ag.calibrate(self.pipe, items, tmp)
            cal = json.loads((Path(tmp) / "calibration.json").read_text(encoding="utf-8"))
            calls = len(self.pipe.tts.calls)
            self.assertEqual(ag.calibrate(self.pipe, items, tmp), mu)
            self.assertEqual(len(self.pipe.tts.calls), calls)
            with self.assertRaises(SystemExit):
                ag.calibrate(self.pipe, ag.calibration_items()[2:4], tmp)
        self.assertEqual(len(mu), 100)
        self.assertEqual((cal["frames"], cal["cases"]), (100, [it["utt_id"] for it in items]))
        u = [self.pipe.np.mean([pc["band_mean"][b] for pc in cal["per_case"]]) for b in range(100)]
        self.pipe.np.testing.assert_allclose(mu, u, atol=1e-12)          # equal frame counts: the mean of the case means

    def test_the_case_loop_records_diagnostics_and_outputs_and_resumes(self):
        item = ag.pilot_items()[0]
        mu = [0.0] * 100
        with tempfile.TemporaryDirectory() as tmp:
            ag.run_cases(self.pipe, [item], ag.CONDS, mu, tmp)
            with open(Path(tmp) / "pilot.csv", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            calls = self.student_calls
            ag.run_cases(self.pipe, [item], ag.CONDS, mu, tmp)
            self.assertEqual(self.student_calls, calls)
        self.assertEqual(len(rows), 2)
        self.assertEqual(list(rows[0]), ag.FIELDS)
        self.assertEqual(calls, len(ag.Q_OUTS["clean"]) + len(ag.Q_OUTS["noisy15"]))
        clean, noisy = rows
        self.assertTrue(all(clean[k] != "" for k in ag.FIELDS))
        self.assertTrue(all(noisy[f"wer_{o}"] == "" for o in ("W4_K2", "W4_K4")))
        self.assertTrue(all(noisy[f"wer_{o}"] != "" for o in ag.Q_OUTS["noisy15"]))
        for k in ag.HANDOVER:
            self.assertGreater(float(clean[f"d_all_K4_{k}"]), 0.0)
            self.assertGreater(float(clean[f"y_all_{k}"]), 0.0)
        self.assertGreater(float(clean["gap0_seed"]), 0.0)

    def test_the_smoke_case_loop_checks_the_teacher_and_reproduces_w4_and_w8(self):
        item = ag.pilot_items()[0]
        stored = {(item["utt_id"], "clean"): {"W4": {"wer": 0.1, "similarity": 0.8}, "W8": {"wer": 0.1, "similarity": 0.8}}}
        with tempfile.TemporaryDirectory() as tmp:
            repro, checks = ag.run_cases(self.pipe, [item], ("clean",), [0.0] * 100, tmp, smoke=True, stored=stored)
            with open(Path(tmp) / "pilot.csv", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
        self.assertEqual(list(rows[0]), ag.FIELDS + ag.SMOKE_FIELDS)
        self.assertEqual(len(repro), 2)
        self.assertEqual(ag.sh.check_reproduction(repro), (0.0, 0.0))
        self.assertEqual(checks, [(0.0, 0.0)])
        self.assertIn(ag.HANDOVER, self.pipe.teacher_calls)
        ag.check_smoke({"rows": repro, "teacher_loop_max_abs_diff": 0.0, "aug_off_max_abs_diff": 0.0})
        with self.assertRaises(SystemExit):
            ag.check_smoke({"rows": repro, "teacher_loop_max_abs_diff": 0.0, "aug_off_max_abs_diff": 1e-3})

    def test_the_diversity_loop_checks_the_released_handover_for_the_first_speaker_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            ag.run_diversity(self.pipe, ag.pilot_items()[:2], [0.0] * 100, tmp, seeds=3)
            with open(Path(tmp) / "diversity.csv", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 2)
        self.assertEqual(list(rows[0]), ag.DIV_FIELDS)
        self.assertEqual(self.student_calls, 2 * 3 * len(ag.DIV_OUTS) + 3)
        self.assertFalse(math.isnan(float(rows[0]["cvf0_W4"])))
        self.assertEqual(rows[1]["cvf0_W4"], "")
        for o in ag.DIV_OUTS:
            self.assertFalse(math.isnan(float(rows[1][f"cvf0_{o}"])), o)

    def test_the_timing_loop(self):
        with tempfile.TemporaryDirectory() as tmp:
            ag.run_timing(self.pipe, ag.pilot_items()[:1], [0.0] * 100, tmp, repeats=2)
            with open(Path(tmp) / "timing.csv", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 1)
        self.assertEqual(list(rows[0]), ag.TIMING_FIELDS)
        self.assertTrue(all(float(rows[0][f"t_{a}_{k}"]) >= 0.0 for a in ag.ARMS for k in ag.HANDOVER))


class AnalyzeTests(unittest.TestCase):
    """Synthetic AUG outputs on the committed pilot cases: by default every output scores exactly as the released
    teacher's output of the same case and step, and its CVf0 equals the stored one."""

    def _write(self, out, wer_add=None, sim_add=None, cvf0_add=None, with_div=True, with_timing=True, n_pilot=None):
        stored = ag.load_stored()
        cv = ag.stored_cvf0()
        wer_add, sim_add, cvf0_add = wer_add or {}, sim_add or {}, cvf0_add or {}
        rng = random.Random(9)
        items = ag.pilot_items()[:n_pilot]
        with open(out / "pilot.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=ag.FIELDS)
            w.writeheader()
            for it in items:
                for c in ag.CONDS:
                    s = stored[(it["utt_id"], c)]
                    row = {"utt_id": it["utt_id"], "speaker_id": s["speaker_id"], "condition": c, "prompt_snr": 30.0, "gpu": "Tesla T4",
                           "prompt_len": 400, "gen_len": 400, "case_seconds": 1, "utc": ""}
                    row.update({f: rng.uniform(0.05, 0.5) for f in ag.FIELDS if f.startswith(("e_", "gap0_", "y_", "d_"))})
                    for o in ag.Q_OUTS[c]:
                        k, _ = ag.parse(o)
                        base = s["m"][f"W{k}"]
                        row.update({f"wer_{o}": base["wer"] + wer_add.get((o, c), 0.0),
                                    f"similarity_{o}": base["similarity"] + sim_add.get((o, c), 0.0),
                                    f"utmos_{o}": base["utmos"], f"hypothesis_{o}": ""})
                    w.writerow(row)
        if with_div:
            with open(out / "diversity.csv", "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=ag.DIV_FIELDS)
                w.writeheader()
                for n, it in enumerate(ag.pilot_items()[:ag.DIV_N]):
                    row = {"utt_id": it["utt_id"], "speaker_id": it["speaker_id"], "condition": "clean", "n_seeds": 20, "case_seconds": 1, "utc": "",
                           "cvf0_W4": cv[it["speaker_id"]]["W4"] if n == 0 else "", "voiced_W4": 0.6 if n == 0 else ""}
                    for o in ag.DIV_OUTS:
                        k, _ = ag.parse(o)
                        row[f"cvf0_{o}"] = cv[it["speaker_id"]][f"W{k}"] + cvf0_add.get(o, 0.0)
                        row[f"voiced_{o}"] = 0.6
                    w.writerow(row)
        if with_timing:
            with open(out / "timing.csv", "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=ag.TIMING_FIELDS)
                w.writeheader()
                for it in ag.pilot_items()[:ag.TIMING_N]:
                    w.writerow({"utt_id": it["utt_id"], "speaker_id": it["speaker_id"], "condition": "clean", "gpu": "Tesla T4", "prompt_len": 400,
                                "gen_len": 400, "repeats": 3, "t_std_4": 1.0, "t_K2_4": 0.72, "t_K4_4": 0.52, "t_std_8": 1.8, "t_K2_8": 1.52,
                                "t_K4_8": 1.32, "t_student_decode": 0.4, "case_seconds": 1, "utc": ""})
        (out / "calibration.json").write_text(json.dumps({"mu": [0.0] * 100, "cases": ["a"] * 20, "frames": 20000,
                                                          "case_rms_from_mu": {"median": 0.1, "max": 0.3}}), encoding="utf-8")

    def test_outputs_like_the_released_teacher_carry_the_cheapest_k(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            self._write(out)
            res = ag.analyze(out)
            report = (out / "AUG_REPORT.md").read_text(encoding="utf-8")
        self.assertEqual(res["have"], res["expected"])
        self.assertTrue(all(v["quality_clean"] and v["pitch_variety"] and v["quality_noisy_reference"] for v in res["criteria"].values()))
        self.assertTrue(res["advance"])
        self.assertEqual(res["carried_K"], 4)
        self.assertTrue(res["applies_to_noisy_reference"])
        self.assertEqual(res["diversity"]["released_W4_reproduction_max_abs"], 0.0)
        self.assertAlmostEqual(res["timing"]["step4"]["K4_ratio"], 0.52)
        self.assertAlmostEqual(res["timing"]["step4"]["K4_pass_ratio"], 0.5)
        self.assertEqual(res["quality"]["W8_K4|clean"]["failures_vs_released"], 0)
        self.assertIn("Analytic Unconditional Guidance", report)
        self.assertIn("Advance to a frozen test on new speakers: YES, K = 4", report)

    def test_more_word_errors_at_step8_stop_k4_and_carry_k2(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            self._write(out, wer_add={("W8_K4", "clean"): 0.05})
            res = ag.analyze(out)
        self.assertFalse(res["criteria"]["K4"]["quality_clean"])
        self.assertTrue(res["criteria"]["K2"]["quality_clean"])
        self.assertEqual(res["carried_K"], 2)

    def test_lost_pitch_variety_stops_a_k(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            self._write(out, cvf0_add={"W4_K4": -0.03})
            res = ag.analyze(out)
        self.assertFalse(res["criteria"]["K4"]["pitch_variety"])
        self.assertEqual(res["carried_K"], 2)

    def test_the_noisy_reference_criterion_is_secondary(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            self._write(out, sim_add={("W8_K4", "noisy15"): -0.05})
            res = ag.analyze(out)
        self.assertFalse(res["criteria"]["K4"]["quality_noisy_reference"])
        self.assertTrue(res["advance"])
        self.assertEqual(res["carried_K"], 4)
        self.assertFalse(res["applies_to_noisy_reference"])

    def test_nothing_advances_when_both_ks_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            self._write(out, wer_add={("W4_K2", "clean"): 0.05, ("W4_K4", "clean"): 0.05})
            res = ag.analyze(out)
        self.assertFalse(res["advance"])
        self.assertIsNone(res["carried_K"])
        self.assertIsNone(res["applies_to_noisy_reference"])

    def test_partial_runs_need_the_flag_and_cannot_advance(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            self._write(out, with_div=False, n_pilot=10)
            with self.assertRaises(SystemExit):
                ag.analyze(out)
            res = ag.analyze(out, allow_partial=True)
        self.assertTrue(res["partial"])
        self.assertTrue(all(v["quality_clean"] is None and v["pitch_variety"] is None for v in res["criteria"].values()))
        self.assertFalse(res["advance"])


class PreregistrationTests(unittest.TestCase):
    def test_committed_preregistration_matches_this_code(self):
        pre = ag.check_prereg()
        self.assertEqual(pre["experiment"], "EXP-024")
        self.assertEqual(pre["code_hashes"], ag.code_hashes())
        self.assertEqual(pre["passes"], {o: ag.cost(o) for o in ("W4", "W8") + tuple(f"W{k}_K{K}" for k in ag.HANDOVER for K in ag.KS)})
        self.assertEqual(pre["bounds"], {f"{k}|{c}": v for (k, c), v in ag.BOUNDS.items()})
        self.assertEqual(pre["power_check"]["script"], "code/analysis/exp024_power_check.py")
        self.assertIn("research/experiments/EXP-024-analytic-unconditional-guidance-pilot/power_check.json", pre["input_hashes"])

    def test_an_edited_preregistration_is_rejected(self):
        pre = json.loads(ag.PREREG_PATH.read_text(encoding="utf-8"))
        pre["decision_rule"] = "anything passes"
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "preregistration.json"
            p.write_text(json.dumps(pre), encoding="utf-8")
            with self.assertRaises(SystemExit):
                ag.check_prereg(p)

    def test_changed_code_is_rejected(self):
        pre = json.loads(ag.PREREG_PATH.read_text(encoding="utf-8"))
        pre["code_hashes"]["code/colab/run_analytic_guidance_pilot.py"] = "0" * 64
        body = {k: v for k, v in pre.items() if k not in ("sha256", "created_utc")}
        pre["sha256"] = hashlib.sha256(json.dumps(body, sort_keys=True).encode("utf-8")).hexdigest()
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "preregistration.json"
            p.write_text(json.dumps(pre), encoding="utf-8")
            with self.assertRaises(SystemExit):
                ag.check_prereg(p)


if __name__ == "__main__":
    unittest.main()
