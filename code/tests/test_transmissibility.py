import csv
import hashlib
import importlib.util
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code/colab"))
import run_transmissibility_test as tt  # noqa: E402

HAVE_NP = importlib.util.find_spec("numpy") is not None
GRID16 = [1.0 - math.cos(math.pi / 2.0 * i / 16.0) for i in range(17)]


class DesignTests(unittest.TestCase):
    def test_forward_passes(self):
        self.assertEqual({a: tt.cost(a) for a in tt.ARMS}, {"W4": 13, "W8": 21, "W4R2": 12, "W8R2": 20, "T16": 32})

    def test_speakers_and_stored_data(self):
        cv = tt.stored_cvf0()
        self.assertEqual(len(tt.div_items()), 20)
        self.assertTrue(all(it["speaker_id"] in cv for it in tt.div_items()))
        stored = tt.stored_w()
        self.assertTrue(all((it["utt_id"], "clean") in stored for it in tt.q_items() + tt.smoke_items()))
        self.assertFalse({it["utt_id"] for it in tt.q_items()} & {it["utt_id"] for it in tt.smoke_items()})
        self.assertEqual(tt.DIV_SEED_BASE, tt.sh.DIV_SEED_BASE)

    def test_spearman(self):
        self.assertAlmostEqual(tt.spearman([1, 2, 3, 4], [10, 20, 30, 40]), 1.0)
        self.assertAlmostEqual(tt.spearman([1, 2, 3, 4], [4, 3, 2, 1]), -1.0)
        self.assertIsNone(tt.spearman([1, 2], [1, 2]))


class _NoContext:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _FakeTTS:
    def teacher(self, x, cond, text, time, mask, drop_audio_cond, drop_text):
        return (-0.4 * x + 0.05 * time) if drop_audio_cond else (-0.5 * x + 0.1 * cond + 0.2 * time)


def reference_loop(tts, y0, cond, kmax, cfg=2.0):
    state, ends = y0.copy(), {}
    for i in range(kmax + 1):
        t = GRID16[i]
        pred = tts.teacher(x=state, cond=cond, text=None, time=t, mask=None, drop_audio_cond=False, drop_text=False)
        null = tts.teacher(x=state, cond=cond, text=None, time=t, mask=None, drop_audio_cond=True, drop_text=True)
        v = pred + (pred - null) * cfg
        ends[i] = state + (1.0 - t) * v
        state = state + (GRID16[i + 1] - t) * v
    return ends, state


class _FakePipe:
    gpu = "Tesla T4 (fake)"

    def __init__(self):
        import numpy as np
        self.np = np
        self.torch = type("T", (), {"inference_mode": staticmethod(_NoContext), "tensor": staticmethod(lambda v, device=None, dtype=None: v),
                                    "float32": None})
        self.sampler = type("S", (), {"STUDENT_TAUS": [0.0761, 0.2929, 0.6173], "SWAY_TEACHER_GRID": GRID16, "CFG_STRENGTH": 2.0})
        self.device = type("D", (), {"type": "cpu"})()
        self.tts = _FakeTTS()

    def prepare(self, item, cond, seed, out_dir, noise_seed=None):
        rng = self.np.random.default_rng(int(seed))
        return {"prompt_len": 10, "prompt_snr": 40.0, "seed": seed, "cond": self.np.ones((1, 60, 100)), "step_cond": self.np.ones((1, 60, 100)),
                "text_ids": None, "y0": rng.normal(size=(1, 60, 100))}

    def teacher(self, ctx, steps):
        ends, _ = reference_loop(self.tts, ctx["y0"], ctx["step_cond"], max(steps))
        return ends, {}

    def ground_truth(self, item):
        return self.np.zeros(10), 24000

    def score(self, audio, item, gt):
        return {"wer": 0.1, "similarity": 0.8, "utmos": 3.0, "hypothesis": "text"}


@unittest.skipUnless(HAVE_NP, "numpy is needed")
class LoopTests(unittest.TestCase):
    def setUp(self):
        import numpy as np
        self.saved = (tt.gl.student, tt.ng.decode, tt.sh.f0_track)
        self.student_taus = []

        def fake_student(pipe, ctx, x, taus):
            self.student_taus.append(tuple(taus))
            return x + 0.01 * len(taus)
        tt.gl.student = fake_student
        tt.ng.decode = lambda pipe, ctx, x: x[0, ctx["prompt_len"]:, 0]
        tt.sh.f0_track = lambda audio: np.abs(audio) * 10 + 100
        self.pipe = _FakePipe()

    def tearDown(self):
        tt.gl.student, tt.ng.decode, tt.sh.f0_track = self.saved

    def test_the_teacher_runs_to_the_end_and_matches_the_released_steps(self):
        ctx = self.pipe.prepare({}, "clean", 5, None)
        ends, final = tt.teacher_run(self.pipe, ctx)
        ref, state = reference_loop(self.pipe.tts, ctx["y0"], ctx["step_cond"], 15)
        for k in (4, 8):
            self.pipe.np.testing.assert_allclose(ends[k], ref[k], atol=1e-12)
        self.pipe.np.testing.assert_allclose(final, state, atol=1e-12)

    def test_the_arms_use_the_two_restart_levels(self):
        ctx = self.pipe.prepare({}, "clean", 5, None)
        ends, final = tt.teacher_run(self.pipe, ctx)
        out = tt.outputs(self.pipe, ctx, ends, final)
        self.assertEqual(sorted(out), sorted(tt.ARMS))
        self.assertEqual(sorted(self.student_taus), sorted([(0.0761, 0.2929, 0.6173)] * 2 + [(0.2929, 0.6173)] * 2))
        self.pipe.np.testing.assert_array_equal(out["T16"], final)

    def test_the_diversity_loop(self):
        with tempfile.TemporaryDirectory() as tmp:
            tt.run_diversity(self.pipe, tt.div_items()[:1], tmp, seeds=3)
            with open(Path(tmp) / "diversity.csv", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
        self.assertEqual(list(rows[0]), tt.DIV_FIELDS)
        self.assertTrue(all(not math.isnan(float(rows[0][f"cvf0_{a}"])) for a in tt.ARMS))

    def test_the_smoke_quality_loop_checks_the_teacher_and_reproduces_w4_w8(self):
        item = tt.smoke_items()[0]
        stored = {(item["utt_id"], "clean"): {f"W{k}": {"wer": 0.1, "similarity": 0.8, "utmos": 3.0} for k in (4, 8)}}
        with tempfile.TemporaryDirectory() as tmp:
            repro, diffs = tt.run_quality(self.pipe, [item], tmp, smoke=True, stored=stored)
            with open(Path(tmp) / "quality.csv", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
        self.assertEqual(list(rows[0]), tt.Q_FIELDS + tt.SMOKE_FIELDS)
        self.assertEqual(diffs, [0.0])
        self.assertEqual(len(repro), 2)
        tt.check_smoke({"rows": repro, "teacher_loop_max_abs_diff": 0.0})
        with self.assertRaises(SystemExit):
            tt.check_smoke({"rows": repro, "teacher_loop_max_abs_diff": 1.0})


class AnalyzeTests(unittest.TestCase):
    """Synthetic CVf0 built from EXP-020's stored values for the same speakers."""

    def _write(self, out, t16_add=0.02, r2_add=0.015, w4_add=0.0, n_div=None):
        cv = tt.stored_cvf0()
        stored = tt.stored_w()
        with open(out / "diversity.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=tt.DIV_FIELDS)
            w.writeheader()
            for i, it in enumerate(tt.div_items()[:n_div]):
                s = cv[it["speaker_id"]]
                base = s["W8"]
                row = {"utt_id": it["utt_id"], "speaker_id": it["speaker_id"], "condition": "clean", "n_seeds": 20, "case_seconds": 1, "utc": "",
                       "cvf0_W8": base, "cvf0_W4": s["W4"] + w4_add, "cvf0_W8R2": base + r2_add + 0.001 * (i % 3),
                       "cvf0_W4R2": s["W4"] + r2_add, "cvf0_T16": base + t16_add + 0.001 * (i % 2)}
                row.update({f"voiced_{a}": 0.6 for a in tt.ARMS})
                w.writerow(row)
        with open(out / "quality.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=tt.Q_FIELDS)
            w.writeheader()
            for it in tt.q_items():
                ref = stored[(it["utt_id"], "clean")]["W8"]
                row = {"utt_id": it["utt_id"], "speaker_id": it["speaker_id"], "condition": "clean", "prompt_snr": 40.0, "gpu": "Tesla T4",
                       "case_seconds": 1, "utc": ""}
                for a in tt.NEW_ARMS:
                    row.update({f"wer_{a}": ref["wer"], f"similarity_{a}": ref["similarity"], f"utmos_{a}": ref["utmos"], f"hypothesis_{a}": ""})
                w.writerow(row)

    def test_a_ceiling_that_a_later_restart_lifts_supports_the_theory(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            self._write(out)
            res = tt.analyze(out)
            report = (out / "TRANSMISSIBILITY_REPORT.md").read_text(encoding="utf-8")
        self.assertEqual(res["verdict"], "supported")
        self.assertTrue(all(res["hypotheses"].values()))
        self.assertEqual(res["diversity"]["reproduction_max_abs"], 0.0)
        self.assertAlmostEqual(res["quality"]["W8R2"]["wer"]["mean"], 0.0, places=12)
        self.assertIn("transmissibility", report)

    def test_no_gap_to_the_teacher_refutes_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            self._write(out, t16_add=-0.01)
            res = tt.analyze(out)
        self.assertFalse(res["hypotheses"]["H1_reference_below_teacher"])
        self.assertEqual(res["verdict"], "refuted")

    def test_a_gap_that_the_later_restart_does_not_close_is_mixed(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            self._write(out, r2_add=-0.005)
            res = tt.analyze(out)
        self.assertFalse(res["hypotheses"]["H2_later_restart_recovers"])
        self.assertEqual(res["verdict"], "mixed")

    def test_partial_runs_need_the_flag(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            self._write(out, n_div=10)
            with self.assertRaises(SystemExit):
                tt.analyze(out)
            res = tt.analyze(out, allow_partial=True)
        self.assertTrue(res["partial"])
        self.assertIsNone(res["verdict"])


class PreregistrationTests(unittest.TestCase):
    def test_committed_preregistration_matches_this_code(self):
        pre = tt.check_prereg()
        self.assertEqual(pre["experiment"], "EXP-026")
        self.assertEqual(pre["code_hashes"], tt.code_hashes())
        self.assertEqual(pre["passes"], {a: tt.cost(a) for a in tt.ARMS})

    def test_an_edited_preregistration_is_rejected(self):
        pre = json.loads(tt.PREREG_PATH.read_text(encoding="utf-8"))
        pre["verdict_rule"] = "always supported"
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "preregistration.json"
            p.write_text(json.dumps(pre), encoding="utf-8")
            with self.assertRaises(SystemExit):
                tt.check_prereg(p)

    def test_changed_code_is_rejected(self):
        pre = json.loads(tt.PREREG_PATH.read_text(encoding="utf-8"))
        pre["code_hashes"]["code/colab/run_transmissibility_test.py"] = "0" * 64
        body = {k: v for k, v in pre.items() if k not in ("sha256", "created_utc")}
        pre["sha256"] = hashlib.sha256(json.dumps(body, sort_keys=True).encode("utf-8")).hexdigest()
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "preregistration.json"
            p.write_text(json.dumps(pre), encoding="utf-8")
            with self.assertRaises(SystemExit):
                tt.check_prereg(p)


if __name__ == "__main__":
    unittest.main()
