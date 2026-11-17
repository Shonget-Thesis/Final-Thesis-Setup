"""Checks of the paper-anchor re-screen's helpers and of the stored cohorts it reads (committed CSVs only)."""
import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code" / "analysis"))
import paper_anchor_rescreen as pr  # noqa: E402


class HelperTest(unittest.TestCase):
    def test_mean_skips_undefined(self):
        self.assertEqual(pr.mean([1.0, float("nan"), 3.0]), 2.0)
        self.assertTrue(math.isnan(pr.mean([float("nan")])))

    def test_spearman(self):
        self.assertAlmostEqual(pr.spearman([1, 2, 3, 4], [10, 20, 30, 40]), 1.0)
        self.assertAlmostEqual(pr.spearman([1, 2, 3, 4], [4, 3, 2, 1]), -1.0)
        self.assertAlmostEqual(pr.spearman([1, 2, 2, 3], [1, 2, 2, 3]), 1.0)   # ties get average ranks

    def test_passes_follow_the_cost_formula(self):
        for k in (4, 8):
            self.assertEqual(pr.PASSES[f"W{k}"], 2 * (k + 1) + 3)   # j = 1: three student steps
        for k in (4, 5, 6, 8):
            self.assertEqual(pr.PASSES[f"W{k}R2"], 2 * (k + 1) + 2)  # j = 2: two student steps


class CohortTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.res = pr.main()

    def test_anchor_present_and_shares_defined(self):
        for name, c in self.res["cohorts"].items():
            self.assertIn("W8R2", c["arms"], name)
            for a, e in c["arms"].items():
                if a == "W4":
                    self.assertAlmostEqual(e["share_of_W8R2_gain_over_W4"], 0.0, places=12)

    def test_w8r2_above_w8_in_every_cohort(self):
        for name, c in self.res["cohorts"].items():
            if "W8" in c["arms"]:
                self.assertLess(c["arms"]["W8"]["cvf0_minus_W8R2"]["hi"], 0.0, name)


if __name__ == "__main__":
    unittest.main()
