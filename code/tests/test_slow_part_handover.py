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
import run_slow_part_handover_prelim as sp  # noqa: E402

HAVE_NP = importlib.util.find_spec("numpy") is not None


class DesignTests(unittest.TestCase):
    def test_forward_passes(self):
        self.assertEqual([sp.cost(f"STS_{k}") for k in sp.STEPS], [10, 12, 14, 16, 18, 20])
        self.assertEqual(sp.cost("plain_8"), 21)
        self.assertEqual(sp.cost("STT_4"), 14)
        self.assertEqual(sp.cost("STS_4"), sp.sh.cost("STS_4"))

    def test_the_handover_takes_the_voice_and_fine_parts_from_the_draft(self):
        self.assertEqual(sp.CODE, "STS")
        self.assertIn(sp.CODE, sp.sh.COMBOS)

    def test_the_stop_rule_picks_the_first_settled_step(self):
        r = {"sig": {"slow": {1: 0.9, 2: 0.5, 3: 0.2, 4: 0.1, 5: 0.05, 6: 0.04, 7: 0.03, 8: 0.02}}}
        self.assertEqual(sp.stop_rule("slow", 0.3, sp.STEPS)(r), "STS_3")
        self.assertEqual(sp.stop_rule("slow", 1.0, sp.STEPS)(r), "STS_2")
        self.assertEqual(sp.stop_rule("slow", 0.01, sp.STEPS)(r), "plain_8")
        self.assertEqual(sp.stop_rule("slow", 0.3, (4, 5, 6, 7))(r), "STS_4")

    def test_stored_reproduction_targets(self):
        stored = sp.stored_reproduction()
        self.assertEqual(len(stored), 360)
        one = next(iter(stored.values()))
        self.assertEqual(sorted(one["drift"]), list(sp.SIGNAL_STEPS))
        self.assertIn("STS_4", one)


@unittest.skipUnless(HAVE_NP, "numpy is needed")
class SignalTests(unittest.TestCase):
    def test_whole_estimate_change_is_the_samplers_drift(self):
        import numpy as np
        rng = np.random.default_rng(0)
        E = {k: rng.normal(size=(1, 40, 100)) for k in range(9)}
        sig = sp.part_signals(E, 10)
        for k in sp.SIGNAL_STEPS:
            a, b = E[k][0, 10:], E[k - 1][0, 10:]
            self.assertAlmostEqual(sig[f"d_full_{k}"], float(np.linalg.norm(a - b) / (np.linalg.norm(a) + 1e-8)), places=10)

    def test_a_change_only_in_the_band_averages_moves_only_the_voice_part(self):
        import numpy as np
        base = np.random.default_rng(1).normal(size=(1, 40, 100))
        E = {k: base + 0.1 * k * np.arange(100.0)[None, None, :] for k in range(9)}
        sig = sp.part_signals(E, 10)
        self.assertGreater(sig["d_voice_3"], 0.01)
        self.assertLess(sig["d_slow_3"], 1e-10)
        self.assertLess(sig["d_fine_3"], 1e-10)


class CalibrationTests(unittest.TestCase):
    def _table(self):
        """Clean cases whose slow part settles by Step 2 at no cost; noisy cases that lose 0.1 WER before Step 5."""
        rng = random.Random(3)
        table = []
        for i in range(60):
            for cond in ("clean", "noisy15"):
                settle = 2 if cond == "clean" else 5
                sig = {p: {k: (0.05 if k >= settle else 0.5) + rng.uniform(0, 0.01) for k in sp.SIGNAL_STEPS} for p in sp.PARTS}
                m = {"plain_8": {"wer": 0.05, "similarity": 0.8, "utmos": 3.0}, "STT_4": {"wer": 0.05, "similarity": 0.8, "utmos": 3.0}}
                for k in sp.STEPS:
                    bad = cond == "noisy15" and k < 5
                    m[f"STS_{k}"] = {"wer": 0.15 if bad else 0.05, "similarity": 0.8, "utmos": 3.0}
                table.append({"utt": f"u{i}", "spk": f"s{i}", "cond": cond, "snr": 40.0, "m": m, "sig": sig,
                              "fail": {n: n.startswith("STS") and cond == "noisy15" and int(n[-1]) < 5 for n in m}})
        return table

    def test_calibration_stops_clean_inputs_early_and_waits_on_noisy_ones(self):
        table = self._table()
        lam = sp.calibrate(table, "slow", sp.STEPS)
        pol = sp.stop_rule("slow", lam, sp.STEPS)
        self.assertEqual({pol(r) for r in table if r["cond"] == "clean"}, {"STS_2"})
        noisy = [pol(r) for r in table if r["cond"] == "noisy15"]
        # The constraint is on the mean: a few noisy inputs may stop early while the average WER rise stays <= 0.01.
        self.assertGreaterEqual(noisy.count("STS_5"), 0.85 * len(noisy))
        self.assertLessEqual(sum(n != "STS_5" for n in noisy) * 0.1 / len(table), sp.CALIBRATION["wer"] + 1e-12)
        self.assertTrue(sp.within_calibration(table, pol))

    def test_lambda_zero_sends_everything_to_step_8(self):
        table = self._table()
        self.assertEqual({sp.stop_rule("slow", 0.0, sp.STEPS)(r) for r in table}, {"plain_8"})

    def test_cross_validation_routes_every_case_once(self):
        table = self._table()
        routed, lams = sp.cross_validate(table, "slow", sp.STEPS)
        self.assertEqual(len(routed), len(table))
        self.assertEqual(len(lams), sp.FOLDS)
        s = sp.summarize(routed)
        self.assertAlmostEqual(s["mean_passes"], (10 + 16) / 2, delta=0.5)
        self.assertEqual(s["mean_passes_clean"], 10.0)


class _NoContext:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _FakePipe:
    gpu = "Tesla T4 (fake)"

    def __init__(self):
        import numpy as np
        self.np = np
        self.torch = type("T", (), {"inference_mode": staticmethod(_NoContext)})
        self.sampler = type("S", (), {"STUDENT_TAUS": [0.0761, 0.2929, 0.6173]})
        self.device = type("D", (), {"type": "cpu"})()
        self.teacher_calls = []

    def prepare(self, item, cond, seed, out_dir, noise_seed=None):
        return {"prompt_len": 10, "prompt_snr": 30.0, "seed": seed, "step_cond": self.np.zeros((1, 50, 100))}

    def teacher(self, ctx, steps):
        self.teacher_calls.append(tuple(steps))
        return {i: self.np.random.default_rng(int(ctx["seed"]) * 10 + i).normal(size=(1, 50, 100)) for i in steps}, {}

    def ground_truth(self, item):
        return self.np.zeros(10), 24000

    def score(self, audio, item, gt):
        return {"wer": 0.1, "similarity": 0.8, "utmos": 3.0, "hypothesis": "text"}


@unittest.skipUnless(HAVE_NP, "numpy is needed")
class RunLoopTests(unittest.TestCase):
    def setUp(self):
        import numpy as np
        self.saved = (sp.sh.student_sample, sp.gl.student, sp.ng.decode, sp.sh.f0_track)
        self.pipe = _FakePipe()
        self.student_calls = 0
        sp.sh.student_sample = lambda pipe, ctx: np.random.default_rng(int(ctx["seed"]) + 1009).normal(size=(1, 50, 100))

        def fake_student(pipe, ctx, x, taus):
            self.student_calls += 1
            return x
        sp.gl.student = fake_student
        sp.ng.decode = lambda pipe, ctx, x: x[0, ctx["prompt_len"]:, 0]
        sp.sh.f0_track = lambda audio: np.abs(audio) * 10 + 100

    def tearDown(self):
        sp.sh.student_sample, sp.gl.student, sp.ng.decode, sp.sh.f0_track = self.saved

    def test_quality_loop_records_every_signal_and_output_and_resumes(self):
        item = sp.ex.items()[0]
        stored = {(item["utt_id"], "clean"): {"STS_4": {"wer": 0.1, "similarity": 0.8}, "drift": {k: 0.0 for k in sp.SIGNAL_STEPS}}}
        with tempfile.TemporaryDirectory() as tmp:
            repro, drift = sp.run_quality(self.pipe, [item], ("clean", "gauss20"), tmp, stored=stored)
            with open(Path(tmp) / "slowpart.csv", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            calls = self.student_calls
            sp.run_quality(self.pipe, [item], ("clean", "gauss20"), tmp)
            self.assertEqual(self.student_calls, calls)
        self.assertEqual(len(rows), 2)
        self.assertEqual(set(rows[0]), set(sp.FIELDS))
        self.assertTrue(all(rows[0][k] != "" for k in sp.FIELDS))
        self.assertEqual(calls, 2 * len(sp.STEPS))
        self.assertIn(tuple(range(9)), self.pipe.teacher_calls)
        self.assertEqual(len(repro), 1)
        self.assertEqual(sp.sh.check_reproduction(repro), (0.0, 0.0))
        self.assertEqual(len(drift), 1)
        self.assertGreater(drift[0], 0.0)          # the fake teacher's drift is not the stored zeros

    def test_diversity_loop(self):
        with tempfile.TemporaryDirectory() as tmp:
            sp.run_diversity(self.pipe, sp.ex.items()[:1], sp.DIV_CONDS, tmp, seeds=3)
            with open(Path(tmp) / "diversity.csv", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 2)
        self.assertEqual(set(rows[0]), set(sp.DIV_FIELDS))
        self.assertEqual(self.student_calls, 2 * 3 * len(sp.DIV_OUTS))
        self.assertFalse(math.isnan(float(rows[0]["cvf0_STS_2"])))


class AnalyzeTests(unittest.TestCase):
    """Synthetic outputs on the committed development cases: clean inputs settle and are safe at Step 2, noisy ones at
    Step 4 (so the mean is below Voice-Swap's 14 passes on this noisy-heavy set); the whole-estimate change settles
    late everywhere."""

    def _write(self, out, early_cvf0=0.093, slow_like_full=False, with_div=True):
        stored = sp.load_stored()
        rng = random.Random(5)
        with open(out / "slowpart.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=sp.FIELDS)
            w.writeheader()
            for (u, c), s in stored.items():
                ref = s["m"]["plain_8"]
                settle = 2 if c == "clean" else 4
                row = {"utt_id": u, "speaker_id": s["speaker_id"], "condition": c, "prompt_snr": s["prompt_snr"], "gpu": "Tesla T4",
                       "case_seconds": 1, "utc": ""}
                for k in sp.SIGNAL_STEPS:
                    slow = (0.05 if k >= settle else 0.5) + rng.uniform(0, 0.01)
                    full = (0.05 if k >= 7 else 0.5) + rng.uniform(0, 0.01)
                    row.update({f"d_slow_{k}": full if slow_like_full else slow, f"d_full_{k}": full,
                                f"d_voice_{k}": 0.2, f"d_fine_{k}": 0.3})
                for k in sp.STEPS:
                    safe = k >= settle
                    row.update({f"wer_STS_{k}": ref["wer"] + (0.0 if safe else 0.2), f"similarity_STS_{k}": ref["similarity"],
                                f"utmos_STS_{k}": ref["utmos"], f"hypothesis_STS_{k}": ""})
                w.writerow(row)
        if with_div:
            with open(out / "diversity.csv", "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=sp.DIV_FIELDS)
                w.writeheader()
                for c in sp.DIV_CONDS:
                    for it in sp.ex.items()[:sp.DIV_N]:
                        j = rng.uniform(-0.003, 0.003)
                        row = {"utt_id": it["utt_id"], "speaker_id": it["speaker_id"], "condition": c, "n_seeds": 20, "case_seconds": 1, "utc": ""}
                        row.update({"cvf0_STS_2": early_cvf0 + j, "cvf0_STS_3": early_cvf0 + j, "cvf0_STS_4": 0.094 + j, "cvf0_plain_8": 0.095 + j})
                        row.update({f"voiced_{o}": 0.8 for o in sp.DIV_OUTS})
                        w.writerow(row)

    def test_a_signal_that_finds_early_safe_inputs_advances(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            self._write(out)
            res = sp.analyze(out)
            report = (out / "SLOW_PART_REPORT.md").read_text(encoding="utf-8")
        self.assertEqual(res["cases"], 360)
        self.assertTrue(all(res["criteria"].values()), res["criteria"])
        self.assertTrue(res["advance"])
        self.assertLess(res["rules"]["slow"]["mean_passes"], res["rules"]["full"]["mean_passes"])
        self.assertEqual(res["candidate_steps"], list(sp.STEPS))
        self.assertIn("Slow-Part Handover", report)

    def test_early_steps_that_lose_pitch_variety_are_dropped_and_c3_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            self._write(out, early_cvf0=0.050)
            res = sp.analyze(out)
        self.assertEqual(res["candidate_steps"], [4, 5, 6, 7])
        self.assertFalse(res["criteria"]["C3_pitch_variety_at_step_3"])
        self.assertFalse(res["advance"])

    def test_no_gain_over_the_whole_estimate_fails_c4(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            self._write(out, slow_like_full=True)
            res = sp.analyze(out)
        self.assertFalse(res["criteria"]["C4_slow_part_beats_whole_estimate"])
        self.assertFalse(res["advance"])

    def test_partial_runs_need_the_flag_and_keep_every_step(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            self._write(out, early_cvf0=0.050, with_div=False)
            with self.assertRaises(SystemExit):
                sp.analyze(out)
            res = sp.analyze(out, allow_partial=True)
        self.assertTrue(res["partial"])
        self.assertEqual(res["candidate_steps"], list(sp.STEPS))
        self.assertIsNone(res["criteria"]["C3_pitch_variety_at_step_3"])
        self.assertFalse(res["advance"])


class PreregistrationTests(unittest.TestCase):
    def test_committed_preregistration_matches_this_code(self):
        pre = sp.check_prereg()
        self.assertEqual(pre["experiment"], "EXP-022")
        self.assertEqual(pre["code_hashes"], sp.code_hashes())

    def test_an_edited_preregistration_is_rejected(self):
        pre = json.loads(sp.PREREG_PATH.read_text(encoding="utf-8"))
        pre["decision_rule"] = "anything passes"
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "preregistration.json"
            p.write_text(json.dumps(pre), encoding="utf-8")
            with self.assertRaises(SystemExit):
                sp.check_prereg(p)

    def test_input_hashes_do_not_depend_on_line_endings(self):
        # A Windows checkout may hold CRLF where Colab holds LF; the frozen hashes must match on both.
        with tempfile.TemporaryDirectory() as tmp:
            a, b = Path(tmp) / "a.json", Path(tmp) / "b.json"
            a.write_bytes(b'{\n  "x": 1\n}\n')
            b.write_bytes(b'{\r\n  "x": 1\r\n}\r\n')
            self.assertEqual(sp.input_hash(a), sp.input_hash(b))

    def test_changed_code_is_rejected(self):
        pre = json.loads(sp.PREREG_PATH.read_text(encoding="utf-8"))
        pre["code_hashes"]["code/colab/run_slow_part_handover_prelim.py"] = "0" * 64
        body = {k: v for k, v in pre.items() if k not in ("sha256", "created_utc")}
        pre["sha256"] = hashlib.sha256(json.dumps(body, sort_keys=True).encode("utf-8")).hexdigest()
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "preregistration.json"
            p.write_text(json.dumps(pre), encoding="utf-8")
            with self.assertRaises(SystemExit):
                sp.check_prereg(p)


if __name__ == "__main__":
    unittest.main()
