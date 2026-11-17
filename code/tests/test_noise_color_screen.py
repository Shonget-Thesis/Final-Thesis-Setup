"""Checks of the noise-colour screen's signal helpers (no data or model needed)."""
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code" / "analysis"))
import noise_color_screen as ncs  # noqa: E402

SR = 24000


class HelperTests(unittest.TestCase):
    def test_add_at_sets_the_total_power_snr(self):
        rng = np.random.default_rng(0)
        clean = (0.2 * rng.normal(size=SR)).astype(np.float32)
        for cond in ("pink25", "ssn25", "lp25"):
            noise = ncs.shaped_noise(len(clean), SR, 7, ncs.SHAPES[cond])
            mixed = ncs.add_at(clean, noise, 25.0)
            added = mixed.astype(np.float64) - clean
            snr = 10 * np.log10(np.mean(clean.astype(np.float64) ** 2) / np.mean(added ** 2))
            self.assertAlmostEqual(snr, 25.0, delta=0.05, msg=cond)
        tones = ncs.tone_noise(len(clean), SR, 3)
        self.assertEqual(len(tones), len(clean))

    def test_shaped_noise_has_the_intended_colour(self):
        n = 4 * SR
        white = np.random.default_rng(1).normal(size=n)
        lp = ncs.shaped_noise(n, SR, 1, ncs.SHAPES["lp25"])
        pink = ncs.shaped_noise(n, SR, 1, ncs.SHAPES["pink25"])
        f = np.fft.rfftfreq(n, 1.0 / SR)
        def share_above_4k(x):
            p = np.abs(np.fft.rfft(x)) ** 2
            return p[f > 4000].sum() / p.sum()
        self.assertGreater(share_above_4k(white), 0.6)
        self.assertLess(share_above_4k(lp), 0.01)
        self.assertLess(share_above_4k(pink), share_above_4k(white))

    def test_innovation_snr_orders_the_colours_as_designed(self):
        rng = np.random.default_rng(2)
        t = np.arange(3 * SR) / SR
        gate = (np.sin(2 * np.pi * 1.5 * t) > 0).astype(float)
        clean = (0.3 * sum(np.sin(2 * np.pi * 160 * h * t) / h for h in range(1, 8)) * gate).astype(np.float32)
        readings = {}
        for cond in ("pink25", "ssn25", "lp25"):
            prompt = ncs.add_at(clean, ncs.shaped_noise(len(clean), SR, 5, ncs.SHAPES[cond]), 25.0)
            readings[cond] = ncs.ins.snr_variants(prompt, SR)[1]
        white = ncs.add_at(clean, rng.normal(size=len(clean)), 25.0)
        readings["white25"] = ncs.ins.snr_variants(white, SR)[1]
        self.assertLess(readings["white25"], readings["pink25"])
        self.assertLess(readings["pink25"], readings["ssn25"])
        self.assertLess(readings["ssn25"], readings["lp25"])

    def test_case_selection_is_the_shortest_clean_cases(self):
        rows = ncs.pick_cases()
        self.assertEqual(len(rows), ncs.N_CASES)
        self.assertEqual([int(r["seq_len"]) for r in rows], sorted(int(r["seq_len"]) for r in rows))


if __name__ == "__main__":
    unittest.main()
