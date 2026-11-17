"""Checks of the innovation-SNR screen's estimator and helpers on synthetic signals (no data or model needed)."""
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code" / "analysis"))
import innovation_snr_screen as ins  # noqa: E402

SR = 24000


def speechlike(seconds=3.0, seed=0):
    """Voiced bursts (harmonic tone, 160 Hz) separated by pauses: the loud half of the frames carries the 'speech'."""
    t = np.arange(int(seconds * SR)) / SR
    tone = sum(np.sin(2 * np.pi * 160 * h * t) / h for h in range(1, 8))
    gate = (np.sin(2 * np.pi * 1.5 * t) > 0).astype(float)
    return (0.3 * tone * gate).astype(np.float64)


def white(n, level, seed=1):
    return level * np.random.default_rng(seed).normal(size=n)


def colored(n, level, seed=2):
    """Strongly low-pass background (a stand-in for speech-shaped or decaying backgrounds)."""
    w = np.random.default_rng(seed).normal(size=n)
    y = np.zeros(n)
    for i in range(1, n):
        y[i] = 0.995 * y[i - 1] + w[i]
    return level * y / y.std()


class EstimatorTests(unittest.TestCase):
    def test_white_background_is_flat_and_both_terms_agree(self):
        x = speechlike()
        s_floor, s_inn, flat = ins.snr_variants(x + white(len(x), 0.01), SR)
        self.assertGreater(flat, 0.4)
        self.assertLess(abs(s_inn - s_floor), 4.0)          # geometric ~ arithmetic for a flat floor

    def test_structured_background_reads_higher_than_its_power_suggests(self):
        x = speechlike()
        s_floor_w, s_inn_w, flat_w = ins.snr_variants(x + white(len(x), 0.01), SR)
        s_floor_c, s_inn_c, flat_c = ins.snr_variants(x + colored(len(x), 0.01), SR)
        self.assertLess(flat_c, flat_w)
        self.assertGreater(s_inn_c - s_floor_c, s_inn_w - s_floor_w + 5.0)
        self.assertGreater(s_inn_c, s_inn_w)                  # same background power, judged far less harmful

    def test_louder_white_noise_lowers_the_innovation_snr(self):
        x = speechlike()
        quiet = ins.snr_variants(x + white(len(x), 0.003), SR)[1]
        loud = ins.snr_variants(x + white(len(x), 0.03), SR)[1]
        self.assertAlmostEqual(quiet - loud, 20.0, delta=3.0)   # 10x amplitude = 20 dB


class HelperTests(unittest.TestCase):
    def test_auc_orientation_lower_is_harmful(self):
        self.assertEqual(ins.auc([1.0, 2.0], [3.0, 4.0]), 1.0)
        self.assertEqual(ins.auc([3.0, 4.0], [1.0, 2.0]), 0.0)

    def test_route_sends_the_lowest_to_the_student(self):
        cases = [{"s": 5.0, "utmos_STU": 3.0, "utmos_W8": 1.0}, {"s": 50.0, "utmos_STU": 1.0, "utmos_W8": 4.0},
                 {"s": 10.0, "utmos_STU": 2.0, "utmos_W8": 2.5}]
        mean, chosen = ins.route(cases, "s", 1)
        self.assertEqual(chosen, {0})
        self.assertAlmostEqual(mean, (3.0 + 4.0 + 2.5) / 3)
        self.assertAlmostEqual(ins.route(cases, "s", 0)[0], (1.0 + 4.0 + 2.5) / 3)


if __name__ == "__main__":
    unittest.main()
