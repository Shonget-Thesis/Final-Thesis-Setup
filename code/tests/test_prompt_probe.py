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
import run_prompt_probe_diagnostic as pp  # noqa: E402

HAVE_NP = importlib.util.find_spec("numpy") is not None
GRID16 = [1.0 - math.cos(math.pi / 2.0 * i / 16.0) for i in range(17)]
PLEN, LENGTH = 200, 300


class DesignTests(unittest.TestCase):
    def test_forward_passes(self):
        self.assertEqual({o: pp.cost(o) for o in pp.OUTS}, {"P2": 9, "P3": 11, "P4": 13, "A2": 9, "A3": 11, "B2": 9, "B3": 11})
        self.assertEqual((pp.cost("W4"), pp.cost("W8")), (13, 21))

    def test_cases_have_stored_labels_and_smoke_cases_are_separate(self):
        stored = pp.stored_w()
        self.assertEqual(len(pp.items()), 40)
        self.assertTrue(all((it["utt_id"], "clean") in stored for it in pp.items() + pp.smoke_items()))
        self.assertFalse({it["utt_id"] for it in pp.items()} & {it["utt_id"] for it in pp.smoke_items()})
        self.assertEqual(sorted(stored[(pp.items()[0]["utt_id"], "clean")]), ["W2", "W3", "W4", "W8"])

    def test_the_probe_is_one_second(self):
        self.assertEqual(pp.PROBE_FRAMES, 94)


@unittest.skipUnless(HAVE_NP, "numpy is needed")
class FormulaTests(unittest.TestCase):
    def setUp(self):
        import numpy as np
        self.np = np
        self.rng = np.random.default_rng(0)

    def test_the_affine_slope_is_shrunk_halfway_to_one(self):
        x = self.rng.normal(size=(94, 100))
        a, c = pp.fit_affine(x, 2.0 * x + 1.0)
        self.np.testing.assert_allclose(a, 1.5, atol=1e-6)
        a0, c0 = pp.fit_affine(x, 2.0 * x + 1.0, ridge=0.0)
        self.np.testing.assert_allclose(a0, 2.0, atol=1e-6)
        self.np.testing.assert_allclose(c0, 1.0, atol=1e-6)

    def test_a_calibrated_estimate_gets_the_identity(self):
        # An estimate equal to the truth on the probe leaves the target unchanged: the correction only removes bias.
        T = self.rng.normal(size=(1, LENGTH, 100))
        span = (100, 194)
        for v in pp.VARIANTS:
            C, _ = pp.corrected(T, span, T[0, span[0]:span[1], :].copy(), PLEN, v)
            self.np.testing.assert_allclose(C, T, atol=1e-6)

    def test_a_band_offset_seen_on_the_probe_is_removed_from_the_target(self):
        truth_full = self.rng.normal(size=(1, LENGTH, 100))
        bias = self.np.linspace(-1.0, 1.0, 100)[None, None, :]
        T = truth_full + bias
        span = (100, 194)
        truth = truth_full[0, span[0]:span[1], :]
        for v in pp.VARIANTS:
            C, _ = pp.corrected(T, span, truth, PLEN, v)
            self.np.testing.assert_allclose(C[0, PLEN:], truth_full[0, PLEN:], atol=1e-6)
            self.assertLess(pp.transfer_ratio(T, span, truth, v), 1e-3)

    def test_without_bias_the_transfer_ratio_is_about_one(self):
        truth = self.rng.normal(size=(94, 100))
        T = self.np.zeros((1, LENGTH, 100))
        T[0, 100:194] = truth + 0.5 * self.rng.normal(size=(94, 100))      # noise only, no shared bias
        r = pp.transfer_ratio(T, (100, 194), truth, "B")
        self.assertGreater(r, 0.95)

    def test_toward_later(self):
        T, L = self.rng.normal(size=(1, LENGTH, 100)), self.rng.normal(size=(1, LENGTH, 100))
        self.assertAlmostEqual(pp.toward_later(L, T, L, PLEN), 0.0, places=9)
        self.assertAlmostEqual(pp.toward_later(T, T, L, PLEN), 1.0, places=6)

    def test_the_probe_skips_trailing_silence_and_needs_enough_prompt(self):
        cond = self.np.zeros((1, LENGTH, 100))
        cond[0, :180] = 1.0 + 0.01 * self.rng.normal(size=(180, 100))
        cond[0, 180:PLEN] = -3.0
        start, end = pp.probe_span(cond, PLEN)
        self.assertEqual(end - start, pp.PROBE_FRAMES)
        self.assertLessEqual(end, 180)
        self.assertGreater(end, 150)
        self.assertIsNone(pp.probe_span(cond, 120))                          # too little prompt would stay visible

    def test_hiding_masks_only_the_probe(self):
        sc = self.rng.normal(size=(1, LENGTH, 100))
        h = pp.hide(sc, (100, 194))
        self.assertTrue((h[:, 100:194] == 0).all())
        self.np.testing.assert_array_equal(h[:, :100], sc[:, :100])
        self.np.testing.assert_array_equal(h[:, 194:], sc[:, 194:])


class _NoContext:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _FakeTTS:
    def __init__(self, np):
        self.np = np

    def teacher(self, x, cond, text, time, mask, drop_audio_cond, drop_text):
        if drop_audio_cond:
            return -0.4 * x + 0.05 * time
        return -0.5 * x + 0.1 * cond + 0.2 * time


def reference_single_pass(tts, y0, cond, kmax, cfg=2.0):
    state, ends = y0.copy(), {}
    for i in range(kmax + 1):
        t = GRID16[i]
        pred = tts.teacher(x=state, cond=cond, text=None, time=t, mask=None, drop_audio_cond=False, drop_text=False)
        null = tts.teacher(x=state, cond=cond, text=None, time=t, mask=None, drop_audio_cond=True, drop_text=True)
        v = pred + (pred - null) * cfg
        ends[i] = state + (1.0 - t) * v
        if i < kmax:
            state = state + (GRID16[i + 1] - t) * v
    return ends


class _FakePipe:
    gpu = "Tesla T4 (fake)"

    def __init__(self):
        import numpy as np
        self.np = np
        self.torch = type("T", (), {"inference_mode": staticmethod(_NoContext), "tensor": staticmethod(lambda v, device=None, dtype=None: v),
                                    "float32": None})
        self.sampler = type("S", (), {"STUDENT_TAUS": [0.0761, 0.2929, 0.6173], "SWAY_TEACHER_GRID": GRID16, "CFG_STRENGTH": 2.0})
        self.device = type("D", (), {"type": "cpu"})()
        self.tts = _FakeTTS(np)

    def prepare(self, item, cond, seed, out_dir, noise_seed=None):
        rng = self.np.random.default_rng(int(seed))
        c = self.np.zeros((1, LENGTH, 100))
        c[0, :180] = 1.0 + 0.3 * rng.normal(size=(180, 100))
        c[0, 180:PLEN] = -3.0
        return {"prompt_len": PLEN, "prompt_snr": 40.0, "seed": seed, "cond": c, "step_cond": c.copy(), "text_ids": None,
                "y0": rng.normal(size=(1, LENGTH, 100))}

    def teacher(self, ctx, steps):
        return reference_single_pass(self.tts, ctx["y0"], ctx["step_cond"], max(steps)), {}

    def ground_truth(self, item):
        return self.np.zeros(10), 24000

    def score(self, audio, item, gt):
        return {"wer": 0.1, "similarity": 0.8, "utmos": 3.0, "hypothesis": "text"}


@unittest.skipUnless(HAVE_NP, "numpy is needed")
class LoopTests(unittest.TestCase):
    def setUp(self):
        self.saved = (pp.gl.student, pp.ng.decode)
        self.calls = 0

        def fake_student(pipe, ctx, x, taus):
            self.calls += 1
            return x
        pp.gl.student = fake_student
        pp.ng.decode = lambda pipe, ctx, x: x[0, ctx["prompt_len"]:, 0]
        self.pipe = _FakePipe()

    def tearDown(self):
        pp.gl.student, pp.ng.decode = self.saved

    def test_the_teacher_trajectory_is_the_released_teacher_with_the_given_conditioning(self):
        ctx = self.pipe.prepare({}, "clean", 3, None)
        hidden = pp.hide(ctx["step_cond"], (90, 184))
        own = pp.teacher_trajectory(self.pipe, ctx, 8, hidden)
        ref = reference_single_pass(self.pipe.tts, ctx["y0"], hidden, 8)
        for k in range(9):
            self.pipe.np.testing.assert_allclose(own[k], ref[k], atol=1e-12)

    def test_the_case_loop_records_everything_and_resumes(self):
        with tempfile.TemporaryDirectory() as tmp:
            pp.run_cases(self.pipe, pp.items()[:2], tmp)
            with open(Path(tmp) / "probe.csv", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            calls = self.calls
            pp.run_cases(self.pipe, pp.items()[:2], tmp)
            self.assertEqual(self.calls, calls)
        self.assertEqual(len(rows), 2)
        self.assertEqual(list(rows[0]), pp.FIELDS)
        self.assertTrue(all(rows[0][k] != "" for k in pp.FIELDS))
        self.assertEqual(calls, 2 * len(pp.OUTS))
        self.assertEqual(int(rows[0]["probe_end"]) - int(rows[0]["probe_start"]), pp.PROBE_FRAMES)

    def test_the_smoke_loop_checks_the_teacher_the_mask_and_the_stored_scores(self):
        item = pp.smoke_items()[0]
        stored = {(item["utt_id"], "clean"): {f"W{k}": {"wer": 0.1, "similarity": 0.8, "utmos": 3.0} for k in (2, 3, 4, 8)}}
        with tempfile.TemporaryDirectory() as tmp:
            repro, checks = pp.run_cases(self.pipe, [item], tmp, smoke=True, stored=stored)
            with open(Path(tmp) / "probe.csv", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
        self.assertEqual(list(rows[0]), pp.FIELDS + pp.SMOKE_FIELDS)
        self.assertEqual(checks, [(0.0, 0.0)])
        self.assertEqual(len(repro), 2)
        pp.check_smoke({"rows": repro, "teacher_loop_max_abs_diff": 0.0, "probe_outside_max_abs": 0.0})
        with self.assertRaises(SystemExit):
            pp.check_smoke({"rows": repro, "teacher_loop_max_abs_diff": 0.0, "probe_outside_max_abs": 1.0})


class AnalyzeTests(unittest.TestCase):
    """Synthetic outputs on the committed cases: the corrected handovers score as the released Step-4 handover plus
    corrected_utmos; the plain hidden-probe handovers lose plain_utmos."""

    def _write(self, out, transfer=0.7, corrected_utmos=0.0, plain_utmos=-0.2, n=None):
        stored = pp.stored_w()
        rng = random.Random(4)
        with open(out / "probe.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=pp.FIELDS)
            w.writeheader()
            for it in pp.items()[:n]:
                s = stored[(it["utt_id"], "clean")]["W4"]
                row = {"utt_id": it["utt_id"], "speaker_id": it["speaker_id"], "condition": "clean", "prompt_snr": 40.0, "gpu": "Tesla T4",
                       "prompt_len": 400, "gen_len": 400, "probe_start": 200, "probe_end": 294, "case_seconds": 1, "utc": ""}
                row.update({f: rng.uniform(0.1, 0.9) for f in pp.FIELDS if f.startswith(("p_", "m", "slopeA_", "gainB_"))})
                row.update({f"r{v}_{k}": transfer for v in pp.VARIANTS for k in pp.CHECK_STEPS})
                for o in pp.OUTS:
                    du = plain_utmos if o.startswith("P") else corrected_utmos
                    row.update({f"wer_{o}": s["wer"], f"similarity_{o}": s["similarity"], f"utmos_{o}": s["utmos"] + du, f"hypothesis_{o}": ""})
                w.writerow(row)

    def test_a_correction_that_transfers_and_restores_quality_carries_the_earliest_step(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            self._write(out)
            res = pp.analyze(out)
            report = (out / "PROBE_REPORT.md").read_text(encoding="utf-8")
        self.assertTrue(res["criteria"]["A"]["G1_transfer"])
        self.assertTrue(res["criteria"]["A"]["G2_quality"][2])
        self.assertTrue(res["advance"])
        self.assertEqual(res["carried"], "A2")
        self.assertAlmostEqual(res["quality"]["A2"]["vs_plain"]["utmos"]["mean"], 0.2, places=9)
        self.assertIn("Prompt-Probe Handover", report)

    def test_no_transfer_stops_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            self._write(out, transfer=1.0)
            res = pp.analyze(out)
        self.assertFalse(res["criteria"]["A"]["G1_transfer"])
        self.assertFalse(res["advance"])

    def test_a_correction_that_does_not_help_stops_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            self._write(out, corrected_utmos=-0.2, plain_utmos=-0.2)
            res = pp.analyze(out)
        self.assertFalse(any(res["criteria"]["A"]["G2_quality"].values()))
        self.assertFalse(res["advance"])

    def test_partial_runs_need_the_flag(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            self._write(out, n=20)
            with self.assertRaises(SystemExit):
                pp.analyze(out)
            res = pp.analyze(out, allow_partial=True)
        self.assertTrue(res["partial"])
        self.assertFalse(res["advance"])


class PreregistrationTests(unittest.TestCase):
    def test_committed_preregistration_matches_this_code(self):
        pre = pp.check_prereg()
        self.assertEqual(pre["experiment"], "EXP-025")
        self.assertEqual(pre["code_hashes"], pp.code_hashes())
        self.assertEqual(pre["passes"], {o: pp.cost(o) for o in pp.OUTS + ("W2", "W3", "W4", "W8")})

    def test_an_edited_preregistration_is_rejected(self):
        pre = json.loads(pp.PREREG_PATH.read_text(encoding="utf-8"))
        pre["decision_rule"] = "anything passes"
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "preregistration.json"
            p.write_text(json.dumps(pre), encoding="utf-8")
            with self.assertRaises(SystemExit):
                pp.check_prereg(p)

    def test_changed_code_is_rejected(self):
        pre = json.loads(pp.PREREG_PATH.read_text(encoding="utf-8"))
        pre["code_hashes"]["code/colab/run_prompt_probe_diagnostic.py"] = "0" * 64
        body = {k: v for k, v in pre.items() if k not in ("sha256", "created_utc")}
        pre["sha256"] = hashlib.sha256(json.dumps(body, sort_keys=True).encode("utf-8")).hexdigest()
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "preregistration.json"
            p.write_text(json.dumps(pre), encoding="utf-8")
            with self.assertRaises(SystemExit):
                pp.check_prereg(p)


if __name__ == "__main__":
    unittest.main()
