"""Checks of the adaptive-teacher screen's helpers (no model needed)."""
import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code" / "analysis"))
import adaptive_teacher_screen as ats  # noqa: E402


class HelperTests(unittest.TestCase):
    def test_grid_reaches_the_commit_point(self):
        self.assertAlmostEqual(ats.sway(ats.N_COARSE)[ats.K_COARSE], ats.T_C, places=12)
        self.assertAlmostEqual(ats.T_C, 1 - math.cos(math.pi / 4), places=12)
        self.assertAlmostEqual(ats.MASK, ats.T_C / (1 - ats.T_C), places=12)

    def test_threshold_rule_follows_the_tree(self):
        ind = {"": 0.010, "E": 0.020, "H": 0.005, "EE": 0.030, "EH": 0.001, "HE": 0.002, "HH": 0.040}
        self.assertEqual(ats.follow(ind, float("inf")), "EEE")
        self.assertEqual(ats.follow(ind, 0.0), "HHH")
        # '' 0.010 <= 0.015 -> E; 'E' 0.020 > 0.015 -> H; 'EH' 0.001 <= 0.015 -> E
        self.assertEqual(ats.follow(ind, 0.015), "EHE")
        self.assertEqual(ats.follow({**ind, "EH": 0.050}, 0.015), "EHH")

    def test_auc_and_spearman(self):
        self.assertEqual(ats.auc([3, 4], [1, 2]), 1.0)
        self.assertEqual(ats.auc([1, 2], [3, 4]), 0.0)
        self.assertEqual(ats.auc([1], [1]), 0.5)
        self.assertAlmostEqual(ats.spearman([1, 2, 3], [10, 20, 30]), 1.0)

    def test_case_selection(self):
        cases = ats.pick_cases()
        self.assertEqual([g for _, g in cases], ["broken"] * 5 + ["good"] * 5)
        drop = [float(r["utmos_CPH6"]) - float(r["utmos_W8R2"]) for r, _ in cases]
        self.assertEqual(drop, sorted(drop))
        self.assertTrue(all(r["condition"] == "clean" for r, _ in cases))


if __name__ == "__main__":
    unittest.main()
