"""EXP-034 runner (code/colab/run_invariant_residual.py): thresholds, prompt noises, signals, policies, the
preregistration and the analysis on synthetic results. No model is loaded."""
import csv
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for p in ("code/colab", "code/analysis"):
    sys.path.insert(0, str(ROOT / p))
import run_invariant_residual as ir  # noqa: E402
import run_followup_tests as rf  # noqa: E402
import innovation_snr_screen as ins  # noqa: E402
import invariant_residual_screen as irs  # noqa: E402
import noise_color_screen as ncs  # noqa: E402
import noise_theory_screen as nts  # noqa: E402

SR = 24000


def speech_like(seconds=2.0, seed=0):
    t = np.arange(int(seconds * SR)) / SR
    gate = (np.sin(2 * np.pi * 1.5 * t) > 0).astype(float)
    rng = np.random.default_rng(seed)
    return (0.3 * sum(np.sin(2 * np.pi * 140 * h * t) / h for h in range(1, 6)) * gate
            + 0.001 * rng.normal(size=t.size)).astype(np.float32)


class ThresholdAndNoiseTests(unittest.TestCase):
    def test_thresholds_come_from_the_screen(self):
        s = json.loads(ir.SCREEN_JSON.read_text(encoding="utf-8"))["summary"]
        self.assertEqual(ir.LAMBDA_R, s["lambda_r"]["lambda"])
        self.assertEqual(ir.LAMBDA_D, s["drift_rule"]["lambda"])
        self.assertEqual(s["drift_rule"]["direction"], "flag low drift")

    def test_shaped_noises_match_the_screens(self):
        x = speech_like()
        np.testing.assert_array_equal(ir.degrade(x, SR, "pink25", 7),
                                      ncs.add_at(x, ncs.shaped_noise(len(x), SR, 7, ncs.SHAPES["pink25"]), 25.0))
        np.testing.assert_array_equal(ir.degrade(x, SR, "hp25", 7),
                                      ncs.add_at(x, ncs.shaped_noise(len(x), SR, 7, nts.SHAPES["hp25"]), 25.0))

    def test_released_conditions_pass_through(self):
        x = speech_like()
        for cond in ("clean", "noisy15", "gauss25"):
            np.testing.assert_array_equal(ir.degrade(x, SR, cond, 3), rf.degrade(x, SR, cond, 3))

    def test_install_registers_and_is_reversible(self):
        saved_conditions, saved_degrade = dict(rf.CONDITIONS), rf.degrade
        try:
            ir.install_conditions()
            self.assertIn("pink25", rf.CONDITIONS)
            self.assertIs(rf.degrade, ir.degrade)
            ir.install_conditions()                        # twice is harmless: no recursion
            np.testing.assert_array_equal(rf.degrade(speech_like(), SR, "clean", 1), speech_like())
        finally:
            rf.CONDITIONS.clear()
            rf.CONDITIONS.update(saved_conditions)
            rf.degrade = saved_degrade


class SignalTests(unittest.TestCase):
    def test_residual_and_spread_snr_match_the_screens(self):
        rng = np.random.default_rng(2)
        t, s = rng.normal(size=(120, 8)) - 5, rng.normal(size=(120, 8)) - 5.5
        self.assertAlmostEqual(ir.invariant_residual(t, s), irs.invariant_residual(t, s), places=12)
        x = ncs.add_at(speech_like(), np.random.default_rng(4).normal(size=2 * SR), 20.0)
        self.assertAlmostEqual(ir.spread_snr(x, SR), ins.snr_variants(x, SR)[1], places=9)

    def test_policies(self):
        self.assertEqual(ir.choose("SNR_ONLY", 15.0, 0.1, 0.5), ir.B_ARM)
        self.assertEqual(ir.choose("SNR_ONLY", 30.0, 0.9, 0.0), ir.A_ARM)
        self.assertEqual(ir.choose("SNR_R", 30.0, 0.9, 0.5), ir.B_ARM)
        self.assertEqual(ir.choose("SNR_R", 30.0, 0.1, 0.0), ir.A_ARM)
        self.assertEqual(ir.choose("SNR_DRIFT", 30.0, 0.9, 0.1), ir.B_ARM)       # low drift flags
        self.assertEqual(ir.choose("SNR_DRIFT", 30.0, 0.9, 0.5), ir.A_ARM)
        self.assertEqual(ir.choose("R_ONLY", 10.0, 0.1, 0.0), ir.A_ARM)          # ignores the SNR
        with self.assertRaises(ValueError):
            ir.choose("NOPE", 30.0, 0.1, 0.1)

    def test_pass_accounting(self):
        self.assertEqual(ir.COST["CPH8"], 12)
        self.assertEqual(ir.COST["CPQH"], 13)
        self.assertEqual(ir.COST["P25"], 30)
        self.assertEqual(ir.COST["W8"], 21)
        self.assertEqual(ir.policy_passes("SNR_R", ir.A_ARM), 13)
        self.assertEqual(ir.policy_passes("SNR_ONLY", ir.A_ARM), 12)
        self.assertEqual(ir.expected_passes("SNR_ONLY", 30.0), (10, 2))
        self.assertEqual(ir.expected_passes("SNR_ONLY", 10.0), (10, 3))

    def test_cohorts_are_disjoint_from_the_screens(self):
        screened = {r["utt_id"] for r in ncs.pick_cases(12)}
        self.assertFalse(screened & {it["utt_id"] for it in ir.qual_items()})
        self.assertEqual(len(ir.qual_items()), 30)
        self.assertEqual(len(ir.div_items()), 20)


class PreregistrationTests(unittest.TestCase):
    def test_freeze_check_and_tamper(self):
        tmp = Path(tempfile.mkdtemp())
        try:
            path = tmp / "prereg.json"
            ir.freeze(prereg_path=path)
            self.assertEqual(ir.check_prereg(path)["experiment"], "EXP-034")
            doc = json.loads(path.read_text(encoding="utf-8"))
            doc["thresholds"]["lambda_r"] = 0.5
            path.write_text(json.dumps(doc), encoding="utf-8")
            with self.assertRaises(RuntimeError):
                ir.check_prereg(path)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


def write_rows(path, fields, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


class AnalysisTests(unittest.TestCase):
    def test_analysis_on_synthetic_results(self):
        rng = np.random.default_rng(0)
        q = []
        for it in ir.qual_items():
            for cond in ir.CONDS:
                hiss = cond in ("noisy15", "gauss25", "pink25", "hp25")
                row = {"utt_id": it["utt_id"], "speaker_id": it["speaker_id"], "condition": cond,
                       "prompt_snr": 17.0 if cond == "noisy15" else 27.0 if hiss else 40.0, "snr_sw": 30.0 if hiss else 60.0,
                       "r": (0.55 if hiss else 0.22) + 0.02 * rng.random(), "drift8": 0.25 + 0.05 * rng.random(),
                       "rough_share": 0.5}
                for a in ir.QUAL_ARMS:
                    good = (a == "CPQH") or (not hiss and a in ("CPH8", "W8", "STU", "STUR2"))
                    row.update({f"utmos_{a}": (3.8 if good else 1.4) + 0.05 * rng.random(), f"wer_{a}": 0.05,
                                f"similarity_{a}": 0.6})
                q.append(row)
        d = []
        for it in ir.div_items():
            for cond in ir.DIV_CONDS:
                row = {"utt_id": it["utt_id"], "speaker_id": it["speaker_id"], "condition": cond}
                for a in ir.DIV_ARMS + ir.POLICIES:
                    base = {"STU": 0.04, "STUR2": 0.04, "CPH8": 0.09, "CPQH": 0.07, "P25": 0.095}.get(a, 0.089)
                    for m in ir.DIV_MEASURES:
                        row[f"{m}_{a}"] = base + 0.002 * rng.random()
                d.append(row)
        t = [{"utt_id": "x", "speaker_id": "x", "condition": "clean", "prompt_snr": 40.0,
              **{f"seconds_{a}": 1.0 + ir.COST[a] / 10 for a in ir.TIME_ARMS},
              **{f"teacher_passes_{a}": ir.TEACHER_PASSES[a] for a in ir.TIME_ARMS},
              **{f"student_passes_{a}": ir.STUDENT_PASSES[a] for a in ir.TIME_ARMS}}]
        tmp = Path(tempfile.mkdtemp())
        try:
            write_rows(tmp / "quality.csv", ir.QUAL_FIELDS, q)
            write_rows(tmp / "diversity.csv", ir.DIV_FIELDS, d)
            write_rows(tmp / "timing.csv", ir.TIME_FIELDS, t)
            res = ir.analyze(tmp, prereg_path=tmp / "missing.json")
            self.assertFalse(res["partial"])
            self.assertTrue(res["hypotheses"]["H1_snr_r_vs_snr_only"])
            self.assertTrue(res["hypotheses"]["H3_detection"])
            self.assertTrue(res["timing"]["arms"]["SNR_R"]["count_matches_cost"])
            self.assertTrue((tmp / "INVARIANT_RESIDUAL_REPORT.md").exists())
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
