"""Properties of the invariant residual and the screen's helpers (code/analysis/invariant_residual_screen.py)."""
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for p in ("code/colab", "code/analysis"):
    sys.path.insert(0, str(ROOT / p))
import invariant_residual_screen as irs  # noqa: E402


def log_mel(seed=0, frames=200, bands=5):
    rng = np.random.default_rng(seed)
    t = np.arange(frames)[:, None]
    return -6.0 + 3.0 * (np.sin(0.05 * t + np.arange(bands)[None, :]) > 0) + 0.2 * rng.normal(size=(frames, bands))


class InvariantResidualTests(unittest.TestCase):
    def test_zero_when_the_teacher_has_the_students_values(self):
        x = log_mel()
        self.assertAlmostEqual(irs.invariant_residual(x, x), 0.0, places=12)

    def test_timing_differences_do_not_count(self):
        """The student's values in any time order: same per-band values, so r = 0 (melody and timing are free)."""
        x = log_mel()
        rng = np.random.default_rng(1)
        shuffled = np.stack([rng.permutation(x[:, b]) for b in range(x.shape[1])], axis=1)
        self.assertAlmostEqual(irs.invariant_residual(x, shuffled), 0.0, places=12)

    def test_an_added_floor_raises_the_residual(self):
        x = log_mel()
        floors = [irs.invariant_residual(0.5 * np.log(np.exp(2 * x) + n), x) for n in (1e-6, 1e-5, 1e-4)]
        self.assertTrue(floors[0] < floors[1] < floors[2])

    def test_stabilization_score(self):
        x = log_mel()
        self.assertEqual(irs.stabilization(x, x), 0.0)
        self.assertAlmostEqual(irs.stabilization(x, 0.9 * x), 0.1, places=9)


class HelperTests(unittest.TestCase):
    def test_auc(self):
        self.assertEqual(irs.auc([3, 4], [1, 2]), 1.0)
        self.assertEqual(irs.auc([1], [1]), 0.5)
        self.assertIsNone(irs.auc([], [1]))

    def test_calibration_respects_the_clean_budget(self):
        cases = [{"condition": "clean", "snr": 40.0, "r": 0.10, "utmos_CPH8": 4.0, "utmos_CPQH": 3.9},
                 {"condition": "clean", "snr": 40.0, "r": 0.12, "utmos_CPH8": 4.0, "utmos_CPQH": 3.9},
                 {"condition": "gauss25", "snr": 27.0, "r": 0.30, "utmos_CPH8": 1.3, "utmos_CPQH": 3.0},
                 {"condition": "noisy15", "snr": 17.0, "r": 0.40, "utmos_CPH8": 1.3, "utmos_CPQH": 2.8}]
        lam, mean, clean_share = irs.calibrate(cases, "r", +1)
        self.assertTrue(0.12 <= lam < 0.30)            # flags the gauss25 case, no clean case
        self.assertEqual(clean_share, 0.0)
        self.assertAlmostEqual(mean, (4.0 + 4.0 + 3.0 + 2.8) / 4)


if __name__ == "__main__":
    unittest.main()
