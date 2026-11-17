"""Checks of the melody-agreement measure used by the EXP-032 tensor-bank screen (no bank or audio needed)."""
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code" / "analysis"))
import exp032_bank_pitch_screen as bs  # noqa: E402


class AgreementTest(unittest.TestCase):
    def setUp(self):
        t = np.linspace(0, 1, 200)
        self.f0 = 150.0 * 2 ** (np.sin(2 * np.pi * 2 * t) / 6)     # a melody moving +-2 semitones around 150 Hz

    def test_identical_tracks_agree_fully(self):
        a = bs.agreement(self.f0, self.f0)
        self.assertEqual(a["within_semitone"], 1.0)
        self.assertAlmostEqual(a["median_cents"], 0.0)
        self.assertAlmostEqual(a["logf0_corr"], 1.0)
        self.assertEqual(a["both_voiced"], 1.0)

    def test_two_semitone_shift_is_outside_but_correlated(self):
        a = bs.agreement(self.f0, self.f0 * 2 ** (2 / 12))
        self.assertEqual(a["within_semitone"], 0.0)
        self.assertAlmostEqual(a["median_cents"], 200.0, places=6)
        self.assertAlmostEqual(a["logf0_corr"], 1.0)               # same melody shape, transposed

    def test_unvoiced_frames_are_left_out(self):
        b = self.f0.copy()
        b[:100] = np.nan
        a = bs.agreement(self.f0, b)
        self.assertAlmostEqual(a["both_voiced"], 0.5)
        self.assertEqual(a["within_semitone"], 1.0)

    def test_too_few_shared_voiced_frames_gives_none(self):
        b = np.full_like(self.f0, np.nan)
        b[:3] = self.f0[:3]
        a = bs.agreement(self.f0, b)
        self.assertIsNone(a["within_semitone"])
        self.assertIsNone(a["median_cents"])

    def test_tracks_are_cut_to_the_shorter(self):
        a = bs.agreement(self.f0, self.f0[:150])
        self.assertEqual(a["within_semitone"], 1.0)

    def test_teacher_times_match_the_sway_grid(self):
        import math
        grid16 = [1 - math.cos(math.pi / 2 * i / 16) for i in range(17)]
        for k in (1, 2, 3, 4, 8):
            self.assertAlmostEqual(bs.TEACHER_TIME[f"T{k}"], grid16[k], places=12)
        self.assertAlmostEqual(bs.TEACHER_TIME["T14_32"], 0.2270, places=4)            # the paper's Step 14
        self.assertAlmostEqual(bs.TEACHER_TIME["T8"], 0.2929, places=4)


if __name__ == "__main__":
    unittest.main()
