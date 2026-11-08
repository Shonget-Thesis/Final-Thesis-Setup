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
import run_voice_swap_prelim as vs  # noqa: E402

HAVE_NP = importlib.util.find_spec("numpy") is not None
HAVE_SCI = HAVE_NP and importlib.util.find_spec("scipy") is not None


class CostTests(unittest.TestCase):
    def test_forward_passes(self):
        self.assertEqual(vs.cost("student4"), 4)
        self.assertEqual(vs.cost("plain_4"), 13)
        self.assertEqual(vs.cost("plain_8"), 21)
        self.assertEqual(vs.cost("vswap_4"), 14)
        self.assertEqual(vs.cost("rswap_4"), 14)

    def test_student_draft_noise_is_separate_from_the_student_loop(self):
        self.assertNotIn(vs.STUDENT_SAMPLE_OFFSET, [0, 7, 29, 61])


@unittest.skipUnless(HAVE_NP, "numpy is needed")
class AlgorithmTests(unittest.TestCase):
    def _arrays(self, seed=3):
        import numpy as np
        r = np.random.default_rng(seed)
        return r.normal(size=(1, 64, 100)), r.normal(loc=2.0, size=(1, 64, 100))

    def test_swap_takes_the_student_averages_and_keeps_the_teacher_time_variation(self):
        import numpy as np
        T, S = self._arrays()
        x = vs.voice_swap(T, S, 16)
        np.testing.assert_allclose(x[0, 16:].mean(axis=0), S[0, 16:].mean(axis=0), atol=1e-10)
        np.testing.assert_allclose(x[0, 16:] - x[0, 16:].mean(axis=0), T[0, 16:] - T[0, 16:].mean(axis=0), atol=1e-10)

    def test_reverse_control_takes_the_teacher_averages_and_the_student_time_variation(self):
        import numpy as np
        T, S = self._arrays()
        x = vs.reverse_swap(T, S, 16)
        np.testing.assert_allclose(x[0, 16:].mean(axis=0), T[0, 16:].mean(axis=0), atol=1e-10)
        np.testing.assert_allclose(x[0, 16:] - x[0, 16:].mean(axis=0), S[0, 16:] - S[0, 16:].mean(axis=0), atol=1e-10)

    def test_swapping_a_signal_with_itself_changes_nothing(self):
        import numpy as np
        T, _ = self._arrays()
        np.testing.assert_allclose(vs.voice_swap(T, T, 16), T, atol=1e-10)
        np.testing.assert_allclose(vs.reverse_swap(T, T, 16), T, atol=1e-10)
        self.assertEqual(vs.swap_rms(T, T, 16), 0.0)

    def test_prompt_frames_and_inputs_are_left_unchanged(self):
        import numpy as np
        T, S = self._arrays()
        t0, s0 = T.copy(), S.copy()
        for x in (vs.voice_swap(T, S, 16), vs.reverse_swap(T, S, 16)):
            np.testing.assert_array_equal(x[0, :16], T[0, :16])
        np.testing.assert_array_equal(T, t0)
        np.testing.assert_array_equal(S, s0)

    def test_swap_rms_measures_the_change_in_the_band_averages(self):
        import numpy as np
        T, S = np.zeros((1, 20, 4)), np.ones((1, 20, 4)) * 3.0
        self.assertAlmostEqual(vs.swap_rms(T, S, 5), 3.0)

    def test_handover_inputs_names(self):
        import numpy as np
        T, S = self._arrays()
        E = {4: T, 8: T + 1.0}
        out = vs.handover_inputs(E, S, 16, ("vswap_4", "rswap_4", "plain_8"))
        self.assertEqual(set(out), {"vswap_4", "rswap_4", "plain_8"})
        self.assertIs(out["plain_8"], E[8])
        np.testing.assert_allclose(out["vswap_4"], vs.voice_swap(T, S, 16))


class RoutingTests(unittest.TestCase):
    def test_three_way_route(self):
        pol = vs.policy("route3", (30, 20))
        self.assertEqual(pol({"snr": 35.0}), "plain_4")
        self.assertEqual(pol({"snr": 30.0}), "plain_4")
        self.assertEqual(pol({"snr": 25.0}), "vswap_4")
        self.assertEqual(pol({"snr": 20.0}), "vswap_4")
        self.assertEqual(pol({"snr": 19.9}), "plain_8")

    def _cases(self, fail_vswap):
        return [{"snr": 35.0 if i % 2 else 15.0, "fail": {"plain_4": bool(i % 2 == 0), "vswap_4": fail_vswap, "plain_8": False}} for i in range(300)]

    @unittest.skipUnless(HAVE_SCI, "scipy is needed")
    def test_calibration_uses_the_swap_when_it_is_safe(self):
        hi, lo = vs.select_route(self._cases(False))
        self.assertLessEqual(lo, 15)                   # every 15 dB input is routed to the safe swap
        self.assertIn(hi, vs.THETA_HI_CHAINS)

    @unittest.skipUnless(HAVE_SCI, "scipy is needed")
    def test_calibration_refuses_a_swap_that_always_fails(self):
        p = vs.select_route(self._cases(True))
        self.assertTrue(p is None or p[1] > 15.0)      # the 15 dB inputs are never routed to a failing swap


@unittest.skipUnless(HAVE_NP, "numpy is needed")
class DiversityTests(unittest.TestCase):
    def test_cvf0_on_known_tracks(self):
        import numpy as np
        a = [np.array([100.0, 200.0, np.nan]), np.array([110.0, 220.0, np.nan]), np.array([90.0, 180.0, 300.0])]
        value, kept = vs.cvf0(a)
        expected = float(np.mean([np.std([100, 110, 90], ddof=1) / 100.0, np.std([200, 220, 180], ddof=1) / 200.0]))
        self.assertAlmostEqual(value, expected)
        self.assertAlmostEqual(kept, 2 / 3)

    def test_tracks_are_cut_to_the_shortest_and_silence_returns_nan(self):
        import numpy as np
        value, kept = vs.cvf0([np.array([100.0, 100.0, 100.0]), np.array([100.0, 100.0])])
        self.assertEqual(value, 0.0)
        self.assertEqual(kept, 1.0)
        value, kept = vs.cvf0([np.array([np.nan]), np.array([np.nan])])
        self.assertTrue(math.isnan(value))
        self.assertEqual(kept, 0.0)


class PreregistrationTests(unittest.TestCase):
    def test_committed_preregistration_matches_this_runner(self):
        pre = vs.check_prereg()
        self.assertEqual(pre["experiment"], "EXP-020")
        self.assertEqual(pre["runner_sha256"], vs.sha256_text(Path(vs.__file__)))

    def test_an_edited_preregistration_is_rejected(self):
        pre = json.loads(vs.PREREG_PATH.read_text(encoding="utf-8"))
        pre["decision_rule"] = "anything passes"
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "preregistration.json"
            p.write_text(json.dumps(pre), encoding="utf-8")
            with self.assertRaises(SystemExit):
                vs.check_prereg(p)

    def test_reproduction_gate(self):
        good = [{"wer": 0.1, "wer_stored": 0.1, "similarity": 0.8, "similarity_stored": 0.8}]
        self.assertEqual(vs.check_reproduction(good), (0.0, 0.0))
        bad = [{"wer": 0.2, "wer_stored": 0.1, "similarity": 0.8, "similarity_stored": 0.8}]
        with self.assertRaises(SystemExit):
            vs.check_reproduction(bad)


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
        self.saved = (vs.student_sample, vs.gl.student, vs.ng.decode, vs.f0_track)
        self.pipe = _FakePipe()
        self.student_calls = []
        vs.student_sample = lambda pipe, ctx: pipe.rng.normal(size=(1, 50, 100))

        def fake_student(pipe, ctx, x, taus):
            self.student_calls.append(list(taus))
            return x
        vs.gl.student = fake_student
        vs.ng.decode = lambda pipe, ctx, x: x[0, ctx["prompt_len"]:, 0]
        vs.f0_track = lambda audio: np.abs(audio) * 10 + 100

    def tearDown(self):
        vs.student_sample, vs.gl.student, vs.ng.decode, vs.f0_track = self.saved

    def test_quality_loop_writes_every_column_reproduces_and_resumes(self):
        item = vs.ex.items()[0]
        with tempfile.TemporaryDirectory() as tmp:
            repro = vs.run_quality(self.pipe, [item], ("clean", "gauss20"), tmp, reproduce=True)
            with open(Path(tmp) / "swap.csv", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            calls, prepared = len(self.student_calls), self.pipe.prepared
            vs.run_quality(self.pipe, [item], ("clean", "gauss20"), tmp)
            self.assertEqual(self.pipe.prepared, prepared)         # finished cases are not run again
            self.assertEqual(len(self.student_calls), calls)
        self.assertEqual(len(rows), 2)
        self.assertEqual(set(rows[0]), set(vs.FIELDS))
        self.assertTrue(all(rows[0][k] != "" for k in vs.FIELDS))
        self.assertEqual(sorted({r["step"] for r in repro}), [4, 8])
        self.assertEqual(len(repro), 4)

    def test_diversity_loop_uses_student_only_and_every_output(self):
        item = vs.ex.items()[0]
        with tempfile.TemporaryDirectory() as tmp:
            vs.run_diversity(self.pipe, [item], tmp, seeds=3)
            with open(Path(tmp) / "diversity.csv", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            prepared = self.pipe.prepared
            vs.run_diversity(self.pipe, [item], tmp, seeds=3)
        self.assertEqual(prepared, 3)
        self.assertEqual(self.pipe.prepared, 3)
        self.assertEqual(len(rows), 1)
        self.assertEqual(set(rows[0]), set(vs.DIV_FIELDS))
        self.assertEqual(sum(c == [0.0, 0.0761, 0.2929, 0.6173] for c in self.student_calls), 3)   # student only, once per seed
        self.assertEqual(len(self.student_calls), 3 * len(vs.DIV_OUTS))
        self.assertFalse(math.isnan(float(rows[0]["cvf0_vswap_4"])))


@unittest.skipUnless(HAVE_SCI, "scipy is needed")
class AnalyzeTests(unittest.TestCase):
    CV = {"student4": 0.46, "plain_4": 0.52, "plain_8": 0.59, "vswap_4": 0.58, "rswap_4": 0.48}

    def _write(self, out, like, cvf0):
        stored = vs.load_stored()
        with open(out / "swap.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=vs.FIELDS)
            w.writeheader()
            for (u, c), s in stored.items():
                row = {k: 0.5 for k in vs.FIELDS}
                row.update({"utt_id": u, "speaker_id": s["speaker_id"], "condition": c, "prompt_snr": s["prompt_snr"], "gpu": "Tesla T4", "utc": ""})
                for o in vs.OUTS:
                    source = like(o, s["m"])
                    row.update({f"{m}_{o}": source[m] for m in vs.METRICS})
                    row[f"hypothesis_{o}"] = ""
                w.writerow(row)
        with open(out / "diversity.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=vs.DIV_FIELDS)
            w.writeheader()
            for i, it in enumerate(vs.ex.items()[:vs.DIV_PAIRS]):
                jitter = 0.002 * (i % 3)
                row = {"utt_id": it["utt_id"], "speaker_id": it["speaker_id"], "condition": "clean", "n_seeds": 20, "case_seconds": 1, "utc": ""}
                row.update({f"cvf0_{o}": cvf0[o] + jitter for o in vs.DIV_OUTS})
                row.update({f"voiced_{o}": 0.8 for o in vs.DIV_OUTS})
                w.writerow(row)

    def test_a_swap_as_good_as_the_reference_advances(self):
        def like(o, m):
            return m["plain_8"] if o == "vswap_4" else m["plain_4"]
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            self._write(out, like, self.CV)
            res = vs.analyze(out)
            report = (out / "VOICE_SWAP_REPORT.md").read_text(encoding="utf-8")
        self.assertEqual(res["cases"], 360)
        self.assertFalse(res["partial"])
        self.assertTrue(all(res["primary"]["criteria"].values()), res["primary"]["criteria"])
        self.assertTrue(res["primary"]["advance"])
        self.assertIn("Voice-Swap Handover", report)
        self.assertEqual(res["cross_validation"]["vswap_all"]["mean_nfe"], 14.0)
        self.assertEqual(res["cross_validation"]["vswap_all"]["failures"], 0)
        self.assertEqual(res["cross_validation"]["reference"]["failures"], 0)

    def test_a_swap_equal_to_todays_handover_does_not_advance(self):
        def like(o, m):
            return m["plain_4"]
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            self._write(out, like, {**self.CV, "vswap_4": 0.52})
            res = vs.analyze(out)
        c = res["primary"]["criteria"]
        self.assertFalse(c["C1_noisy_failures"])
        self.assertFalse(c["C3_voice_is_the_cause"])
        self.assertFalse(res["primary"]["advance"])

    def test_a_partial_run_needs_the_flag_and_leaves_diversity_open(self):
        def like(o, m):
            return m["plain_8"]
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            self._write(out, like, self.CV)
            (out / "diversity.csv").unlink()
            with self.assertRaises(SystemExit):
                vs.analyze(out)
            res = vs.analyze(out, allow_partial=True)
        self.assertTrue(res["partial"])
        self.assertIsNone(res["primary"]["criteria"]["C4_diversity"])
        self.assertFalse(res["primary"]["advance"])


if __name__ == "__main__":
    unittest.main()
