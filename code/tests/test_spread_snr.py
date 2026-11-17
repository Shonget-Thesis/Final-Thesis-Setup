"""Properties of the spread-weighted SNR (the innovation-SNR formula) and the band screen's noise shape."""
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code" / "analysis"))
import innovation_snr_screen as ins  # noqa: E402
import noise_band_screen as nbs  # noqa: E402
import noise_color_screen as ncs  # noqa: E402

SR = 24000


def voiced(seconds=3.0):
    t = np.arange(int(seconds * SR)) / SR
    gate = (np.sin(2 * np.pi * 1.5 * t) > 0).astype(float)
    return (0.3 * sum(np.sin(2 * np.pi * 150 * h * t) / h for h in range(1, 8)) * gate).astype(np.float32)


class SpreadSNRProperties(unittest.TestCase):
    def test_never_below_the_floor_snr(self):
        x = voiced()
        for shape in (ncs.SHAPES["pink25"], ncs.SHAPES["lp25"], nbs.band_shape):
            prompt = ncs.add_at(x, ncs.shaped_noise(len(x), SR, 4, shape), 25.0)
            s_floor, s_sw, sfm = ins.snr_variants(prompt, SR)
            self.assertGreaterEqual(s_sw + 1e-9, s_floor)
            self.assertLessEqual(sfm, 1.0 + 1e-9)
            self.assertAlmostEqual(s_sw - s_floor, -10 * np.log10(sfm), places=3)   # up to the estimator's eps

    def test_background_gain_shifts_the_reading_by_20_log10(self):
        x = voiced()
        noise = np.random.default_rng(5).normal(size=len(x))
        quiet = ins.snr_variants(np.clip(x + 0.002 * noise, -1, 1), SR)[1]
        loud = ins.snr_variants(np.clip(x + 0.02 * noise, -1, 1), SR)[1]
        self.assertAlmostEqual(quiet - loud, 20.0, delta=1.5)

    def test_spread_noise_reads_lower_than_narrow_noise_of_equal_power(self):
        x = voiced()
        broad = ncs.add_at(x, np.random.default_rng(6).normal(size=len(x)), 25.0)
        narrow = ncs.add_at(x, ncs.shaped_noise(len(x), SR, 6, nbs.band_shape), 25.0)
        self.assertLess(ins.snr_variants(broad, SR)[1], ins.snr_variants(narrow, SR)[1] - 5.0)


class BandShapeTests(unittest.TestCase):
    def test_band_noise_is_narrow_and_high(self):
        n = 4 * SR
        y = ncs.shaped_noise(n, SR, 1, nbs.band_shape)
        p = np.abs(np.fft.rfft(y)) ** 2
        f = np.fft.rfftfreq(n, 1.0 / SR)
        share = lambda lo, hi: p[(f >= lo) & (f <= hi)].sum() / p.sum()   # noqa: E731
        self.assertGreater(share(6000, 8000), 0.65)      # half-power edges at 6 and 8 kHz, eighth-order skirts
        self.assertGreater(share(5000, 9000), 0.95)
        self.assertLess(share(0, 4000), 0.001)


if __name__ == "__main__":
    unittest.main()
