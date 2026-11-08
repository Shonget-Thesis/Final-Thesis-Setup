import csv
import importlib.util
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code/colab"))
import run_selective_handover_prelim as sh  # noqa: E402

HAVE_NP = importlib.util.find_spec("numpy") is not None
HAVE_SCI = HAVE_NP and importlib.util.find_spec("scipy") is not None


class CostTests(unittest.TestCase):
    def test_forward_passes(self):
        self.assertEqual(sh.cost("student4"), 4)
        self.assertEqual(sh.cost("plain_4"), 13)
        self.assertEqual(sh.cost("plain_8"), 21)
        self.assertEqual(sh.cost("STT_4"), 14)
        self.assertEqual(sh.cost("TSS_4"), 14)
        self.assertEqual(sh.cost("STT_5"), 16)
        self.assertEqual(sh.cost("STT_6"), 18)
        self.assertEqual(sh.cost("STT_7"), 20)

    def test_draft_noise_is_separate_from_the_student_loop(self):
        self.assertNotIn(sh.STUDENT_SAMPLE_OFFSET, [0, 7, 29, 61])

    def test_the_eight_combinations_are_complete(self):
        self.assertEqual(len(sh.ALL_CODES), 8)
        self.assertEqual(set(sh.ALL_CODES), set(sh.COMBOS) | {"TTT", "SSS"})
        self.assertEqual(sh.combo_output("TTT"), "plain_4")
        self.assertEqual(sh.combo_output("SSS"), "student4")
        self.assertEqual(sh.combo_output("STT"), "STT_4")


@unittest.skipUnless(HAVE_NP, "numpy is needed")
class AlgorithmTests(unittest.TestCase):
    def _arrays(self, seed=3):
        import numpy as np
        r = np.random.default_rng(seed)
        return r.normal(size=(1, 64, 100)), r.normal(loc=2.0, size=(1, 64, 100))

    def test_the_three_parts_add_back_to_the_signal(self):
        import numpy as np
        x = np.random.default_rng(1).normal(size=(40, 12))
        voice, slow, fine = sh.decompose(x)
        np.testing.assert_allclose(voice + slow + fine, x, atol=1e-10)

    def test_voice_is_the_band_average_and_a_constant_has_no_other_part(self):
        import numpy as np
        x = np.ones((30, 5)) * np.arange(5.0)
        voice, slow, fine = sh.decompose(x)
        np.testing.assert_allclose(voice[0], np.arange(5.0), atol=1e-10)
        np.testing.assert_allclose(slow, 0.0, atol=1e-10)
        np.testing.assert_allclose(fine, 0.0, atol=1e-10)

    def test_a_slow_ramp_is_slow_and_alternation_is_fine(self):
        import numpy as np
        n = 101
        ramp = np.linspace(-1.0, 1.0, n)[:, None]
        _, slow, fine = sh.decompose(ramp)
        middle = range(sh.SLOW_WINDOW, n - sh.SLOW_WINDOW)
        self.assertLess(max(abs(fine[i, 0]) for i in middle), 1e-9)
        alt = ((-1.0) ** np.arange(n))[:, None]
        _, slow, fine = sh.decompose(alt)
        self.assertLess(max(abs(slow[i, 0]) for i in middle), 0.15)
        self.assertGreater(min(abs(fine[i, 0]) for i in middle), 0.85)

    def test_smoothing_keeps_the_shape(self):
        import numpy as np
        x = np.random.default_rng(2).normal(size=(17, 3))
        self.assertEqual(sh.smooth_time(x).shape, x.shape)

    def test_all_teacher_and_all_student_give_those_signals(self):
        import numpy as np
        T, S = self._arrays()
        np.testing.assert_allclose(sh.selective_handover(T, S, 16, "TTT")[0, 16:], T[0, 16:], atol=1e-10)
        np.testing.assert_allclose(sh.selective_handover(T, S, 16, "SSS")[0, 16:], S[0, 16:], atol=1e-10)

    def test_voice_swap_takes_the_student_averages_and_keeps_the_teacher_time_variation(self):
        import numpy as np
        T, S = self._arrays()
        x = sh.selective_handover(T, S, 16, "STT")
        np.testing.assert_allclose(x[0, 16:].mean(axis=0), S[0, 16:].mean(axis=0), atol=1e-10)
        np.testing.assert_allclose(x[0, 16:] - x[0, 16:].mean(axis=0), T[0, 16:] - T[0, 16:].mean(axis=0), atol=1e-10)

    def test_each_letter_selects_its_own_part(self):
        import numpy as np
        T, S = self._arrays()
        pt, ps = sh.decompose(T[0, 16:]), sh.decompose(S[0, 16:])
        for code in sh.ALL_CODES:
            x = sh.selective_handover(T, S, 16, code)[0, 16:]
            expected = sum((ps[i] if code[i] == "S" else pt[i]) for i in range(3))
            np.testing.assert_allclose(x, expected, atol=1e-10)

    def test_prompt_frames_and_inputs_are_left_unchanged(self):
        import numpy as np
        T, S = self._arrays()
        t0, s0 = T.copy(), S.copy()
        for code in sh.COMBOS:
            np.testing.assert_array_equal(sh.selective_handover(T, S, 16, code)[0, :16], T[0, :16])
        np.testing.assert_array_equal(T, t0)
        np.testing.assert_array_equal(S, s0)

    def test_handover_inputs_names(self):
        import numpy as np
        T, S = self._arrays()
        E = {4: T, 5: T + 1.0, 8: T + 2.0}
        out = sh.handover_inputs(E, S, 16, ("STT_4", "STT_5", "TSS_4", "plain_8"))
        self.assertEqual(set(out), {"STT_4", "STT_5", "TSS_4", "plain_8"})
        self.assertIs(out["plain_8"], E[8])
        np.testing.assert_allclose(out["STT_5"], sh.selective_handover(E[5], S, 16, "STT"))


class LadderTests(unittest.TestCase):
    def test_rungs_go_from_cheap_to_expensive_as_snr_falls(self):
        pol = sh.policy("ladder", (30, 2))
        self.assertEqual(pol({"snr": 31.0}), "plain_4")
        self.assertEqual(pol({"snr": 28.0}), "STT_4")
        self.assertEqual(pol({"snr": 26.5}), "STT_5")
        self.assertEqual(pol({"snr": 24.0}), "STT_6")
        self.assertEqual(pol({"snr": 22.0}), "STT_7")
        self.assertEqual(pol({"snr": 21.9}), "plain_8")

    def test_ladder_costs_rise_along_the_rungs(self):
        costs = [sh.cost(n) for n in sh.LADDER]
        self.assertEqual(costs, sorted(costs))

    def _cases(self, fail):
        return [{"snr": 35.0 if i % 2 else 15.0,
                 "fail": {**{n: fail for n in sh.LADDER}, "plain_4": False, "plain_8": False}} for i in range(300)]

    @unittest.skipUnless(HAVE_SCI, "scipy is needed")
    def test_calibration_uses_the_ladder_when_it_is_safe(self):
        base, gap = sh.select_ladder(self._cases(False), "ladder")
        self.assertIn(gap, sh.LADDER_GAPS)
        self.assertLessEqual(base, 15 + gap * 4)        # the 15 dB inputs reach a rung

    @unittest.skipUnless(HAVE_SCI, "scipy is needed")
    def test_calibration_refuses_a_ladder_that_always_fails(self):
        cases = [{"snr": 15.0, "fail": {**{n: True for n in sh.LADDER}, "plain_8": False}} for _ in range(300)]
        p = sh.select_ladder(cases, "ladder")
        self.assertTrue(p is None or sh.policy("ladder", p)(cases[0]) == "plain_8")


@unittest.skipUnless(HAVE_NP, "numpy is needed")
class DiversityTests(unittest.TestCase):
    def test_cvf0_on_known_tracks(self):
        import numpy as np
        a = [np.array([100.0, 200.0, np.nan]), np.array([110.0, 220.0, np.nan]), np.array([90.0, 180.0, 300.0])]
        value, kept = sh.cvf0(a)
        expected = float(np.mean([np.std([100, 110, 90], ddof=1) / 100.0, np.std([200, 220, 180], ddof=1) / 200.0]))
        self.assertAlmostEqual(value, expected)
        self.assertAlmostEqual(kept, 2 / 3)

    def test_silence_returns_nan(self):
        import numpy as np
        value, kept = sh.cvf0([np.array([np.nan]), np.array([np.nan])])
        self.assertTrue(math.isnan(value))
        self.assertEqual(kept, 0.0)


class PreregistrationTests(unittest.TestCase):
    def test_committed_preregistration_matches_this_runner(self):
        pre = sh.check_prereg()
        self.assertEqual(pre["experiment"], "EXP-020")
        self.assertEqual(pre["runner_sha256"], sh.sha256_text(Path(sh.__file__)))

    def test_an_edited_preregistration_is_rejected(self):
        pre = json.loads(sh.PREREG_PATH.read_text(encoding="utf-8"))
        pre["decision_rule"] = "anything passes"
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "preregistration.json"
            p.write_text(json.dumps(pre), encoding="utf-8")
            with self.assertRaises(SystemExit):
                sh.check_prereg(p)

    def test_reproduction_gate(self):
        good = [{"wer": 0.1, "wer_stored": 0.1, "similarity": 0.8, "similarity_stored": 0.8}]
        self.assertEqual(sh.check_reproduction(good), (0.0, 0.0))
        bad = [{"wer": 0.2, "wer_stored": 0.1, "similarity": 0.8, "similarity_stored": 0.8}]
        with self.assertRaises(SystemExit):
            sh.check_reproduction(bad)


class _NoContext:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _FakePipe:
    """The parts of rf.Pipeline that the run loops touch, on numpy arrays."""
    gpu = "Tesla T4 (fake)"

    def __init__(self):
        import numpy as np
        self.np = np
        self.rng = np.random.default_rng(9)
        self.torch = type("T", (), {"inference_mode": staticmethod(_NoContext)})
        self.sampler = type("S", (), {"STUDENT_TAUS": [0.0761, 0.2929, 0.6173]})
        self.prepared = 0

    def prepare(self, item, cond, seed, out_dir, noise_seed=None):
        self.prepared += 1
        return {"prompt_len": 10, "prompt_snr": 30.0, "seed": seed, "step_cond": self.np.zeros((1, 50, 100))}

    def teacher(self, ctx, steps):
        return {i: self.rng.normal(size=(1, 50, 100)) for i in steps}, {}

    def ground_truth(self, item):
        return self.np.zeros(10), 24000

    def score(self, audio, item, gt):
        return {"wer": 0.1, "similarity": 0.8, "utmos": 3.0, "hypothesis": "text"}


@unittest.skipUnless(HAVE_NP, "numpy is needed")
class RunLoopTests(unittest.TestCase):
    """Exercises the run loops with a fake model, since torch is not installed for the local checks."""

    def setUp(self):
        import numpy as np
        self.saved = (sh.student_sample, sh.gl.student, sh.ng.decode, sh.f0_track)
        self.pipe = _FakePipe()
        self.student_calls = []
        sh.student_sample = lambda pipe, ctx: pipe.rng.normal(size=(1, 50, 100))

        def fake_student(pipe, ctx, x, taus):
            self.student_calls.append(list(taus))
            return x
        sh.gl.student = fake_student
        sh.ng.decode = lambda pipe, ctx, x: x[0, ctx["prompt_len"]:, 0]
        sh.f0_track = lambda audio: np.abs(audio) * 10 + 100

    def tearDown(self):
        sh.student_sample, sh.gl.student, sh.ng.decode, sh.f0_track = self.saved

    def test_quality_loop_writes_every_column_reproduces_and_resumes(self):
        item = sh.ex.items()[0]
        with tempfile.TemporaryDirectory() as tmp:
            repro = sh.run_quality(self.pipe, [item], ("clean", "gauss20"), tmp, reproduce=True)
            with open(Path(tmp) / "selective.csv", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            calls, prepared = len(self.student_calls), self.pipe.prepared
            sh.run_quality(self.pipe, [item], ("clean", "gauss20"), tmp)
            self.assertEqual(self.pipe.prepared, prepared)         # finished cases are not run again
            self.assertEqual(len(self.student_calls), calls)
        self.assertEqual(len(rows), 2)
        self.assertEqual(set(rows[0]), set(sh.FIELDS))
        self.assertTrue(all(rows[0][k] != "" for k in sh.FIELDS))
        self.assertEqual(sorted({r["step"] for r in repro}), [4, 8])

    def test_diversity_loop_uses_student_only_and_every_output(self):
        item = sh.ex.items()[0]
        with tempfile.TemporaryDirectory() as tmp:
            sh.run_diversity(self.pipe, [item], tmp, seeds=3)
            with open(Path(tmp) / "diversity.csv", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 1)
        self.assertEqual(set(rows[0]), set(sh.DIV_FIELDS))
        self.assertEqual(sum(c == [0.0, 0.0761, 0.2929, 0.6173] for c in self.student_calls), 3)
        self.assertEqual(len(self.student_calls), 3 * len(sh.DIV_OUTS))
        self.assertFalse(math.isnan(float(rows[0]["cvf0_STT_4"])))


@unittest.skipUnless(HAVE_SCI, "scipy is needed")
class AnalyzeTests(unittest.TestCase):
    CV = {"student4": 0.46, "plain_4": 0.52, "plain_8": 0.59, "STT_4": 0.58, "TST_4": 0.50, "TTS_4": 0.51, "SST_4": 0.49, "STS_4": 0.55, "TSS_4": 0.48}

    def _write(self, out, like, cvf0):
        stored = sh.load_stored()
        with open(out / "selective.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=sh.FIELDS)
            w.writeheader()
            for (u, c), s in stored.items():
                row = {k: 0.5 for k in sh.FIELDS}
                row.update({"utt_id": u, "speaker_id": s["speaker_id"], "condition": c, "prompt_snr": s["prompt_snr"], "gpu": "Tesla T4", "utc": ""})
                for o in sh.OUTS:
                    source = like(o, s["m"])
                    row.update({f"{m}_{o}": source[m] for m in sh.METRICS})
                    row[f"hypothesis_{o}"] = ""
                w.writerow(row)
        with open(out / "diversity.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=sh.DIV_FIELDS)
            w.writeheader()
            for i, it in enumerate(sh.ex.items()[:sh.DIV_PAIRS]):
                jitter = 0.002 * (i % 3)
                row = {"utt_id": it["utt_id"], "speaker_id": it["speaker_id"], "condition": "clean", "n_seeds": 20, "case_seconds": 1, "utc": ""}
                row.update({f"cvf0_{o}": cvf0[o] + jitter for o in sh.DIV_OUTS})
                row.update({f"voiced_{o}": 0.8 for o in sh.DIV_OUTS})
                w.writerow(row)

    def test_a_primary_combination_as_good_as_the_reference_advances(self):
        def like(o, m):
            return m["plain_8"] if o.startswith("STT") else m["plain_4"]
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            self._write(out, like, self.CV)
            res = sh.analyze(out)
            report = (out / "SELECTIVE_HANDOVER_REPORT.md").read_text(encoding="utf-8")
        self.assertEqual(res["cases"], 360)
        self.assertFalse(res["partial"])
        self.assertTrue(all(res["primary"]["criteria"].values()), res["primary"]["criteria"])
        self.assertTrue(res["primary"]["advance"])
        self.assertIn("Selective Handover", report)
        self.assertIn("The map", report)
        self.assertGreater(res["effect_map"]["voice"]["similarity_noisy"]["mean"], 0)
        self.assertEqual(res["cross_validation"]["STT4_all"]["mean_nfe"], 14.0)
        self.assertEqual(res["cross_validation"]["STT4_all"]["failures"], 0)
        self.assertEqual(res["cross_validation"]["reference"]["failures"], 0)

    def test_a_combination_equal_to_todays_handover_does_not_advance(self):
        def like(o, m):
            return m["plain_" + o.rsplit("_", 1)[1]]
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            self._write(out, like, {**self.CV, "STT_4": 0.52})
            res = sh.analyze(out)
        self.assertFalse(res["primary"]["criteria"]["C1_noisy_failures"])
        self.assertFalse(res["primary"]["advance"])

    def test_a_partial_run_needs_the_flag_and_leaves_diversity_open(self):
        def like(o, m):
            return m["plain_8"]
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            self._write(out, like, self.CV)
            (out / "diversity.csv").unlink()
            with self.assertRaises(SystemExit):
                sh.analyze(out)
            res = sh.analyze(out, allow_partial=True)
        self.assertTrue(res["partial"])
        self.assertIsNone(res["primary"]["criteria"]["C4_diversity"])
        self.assertFalse(res["primary"]["advance"])


if __name__ == "__main__":
    unittest.main()
