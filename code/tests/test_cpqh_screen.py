"""Helpers of the Commit-Point Quantile Handover screen (the QH operator itself is tested in test_method_comparison)."""
import math
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code" / "analysis"))
import cpqh_screen as cs  # noqa: E402


class MelodyAgreementTests(unittest.TestCase):
    def test_identical_and_octave_apart(self):
        f0 = np.linspace(100, 200, 50)
        self.assertEqual(cs.melody_agreement(f0, f0), 1.0)
        self.assertEqual(cs.melody_agreement(f0, 2 * f0), 0.0)

    def test_semitone_tolerance(self):
        f0 = np.full(20, 150.0)
        self.assertEqual(cs.melody_agreement(f0, f0 * 2 ** (90 / 1200)), 1.0)    # 90 cents: within a semitone
        self.assertEqual(cs.melody_agreement(f0, f0 * 2 ** (110 / 1200)), 0.0)   # 110 cents: outside

    def test_unvoiced_frames_and_lengths(self):
        a = np.array([100.0, np.nan, 100.0, 100.0, 100.0, 100.0, 100.0])
        b = np.array([100.0, 100.0, np.nan, 200.0, 100.0, 100.0, 100.0, 100.0, 100.0])   # longer: truncated to 7
        self.assertAlmostEqual(cs.melody_agreement(a, b), 4 / 5)                        # 5 frames voiced in both
        self.assertIsNone(cs.melody_agreement(a[:4], b[:4]))                             # under 5 shared frames


class MeanDefinedTests(unittest.TestCase):
    def test_skips_none_and_nan(self):
        self.assertEqual(cs.mean_defined([1.0, None, float("nan"), 3.0]), 2.0)
        self.assertIsNone(cs.mean_defined([None, math.nan]))


if __name__ == "__main__":
    unittest.main()
