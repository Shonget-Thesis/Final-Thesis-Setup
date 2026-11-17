"""Helpers of the CP-QH control screen: time smoothing, roughness and the above-floor statistic."""
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code" / "analysis"))
import cpqh_controls_screen as cc  # noqa: E402


class ControlHelperTests(unittest.TestCase):
    def test_smoothing_keeps_shape_constants_and_ramps(self):
        x = np.tile(np.arange(50, dtype=float)[:, None], (1, 3))
        y = cc.smooth_time(x)
        self.assertEqual(y.shape, x.shape)
        h = cc.SMOOTH_FRAMES // 2
        np.testing.assert_allclose(y[h:-h], x[h:-h])                     # a ramp is unchanged away from the edges
        np.testing.assert_allclose(cc.smooth_time(np.full((20, 2), 3.0)), 3.0)

    def test_roughness_separates_noise_from_smooth_signals(self):
        rng = np.random.default_rng(0)
        self.assertAlmostEqual(cc.roughness(rng.normal(size=(20000, 2))), 2.0, delta=0.05)
        t = np.arange(2000)[:, None]
        self.assertLess(cc.roughness(np.sin(2 * np.pi * t / 200.0)), 0.01)
        self.assertLess(cc.roughness(cc.smooth_time(rng.normal(size=(4000, 2)))), 0.5)

    def test_above_floor_is_zero_for_a_constant_band(self):
        self.assertAlmostEqual(cc.above_floor_db(np.full((100, 4), -3.0)), 0.0, places=9)
        x = np.zeros((100, 1))
        x[50:] = np.log(10.0) / 2                                         # half the frames at 10 times the power
        self.assertAlmostEqual(cc.above_floor_db(x), 10 * np.log10(5.5), places=6)


if __name__ == "__main__":
    unittest.main()
