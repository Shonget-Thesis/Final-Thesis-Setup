"""Properties of the Floor-Canonical Handover (FCH) payload in code/analysis/floor_canonical_screen.py."""
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code" / "analysis"))
import floor_canonical_screen as fc  # noqa: E402


def log_mel(rng, frames=240, bands=6):
    """A log-mel-like (natural log of magnitude) array: a loud harmonic-like pattern over a quiet floor."""
    t = np.arange(frames)[:, None]
    b = np.arange(bands)[None, :]
    level = -6.0 + 4.0 * (np.sin(0.07 * t + 0.9 * b) > 0.3) + 0.3 * rng.normal(size=(frames, bands))
    return level.astype(np.float64)


def add_power(x, n):
    """A stationary additive background: n (per band) added in power, P = exp(2x)."""
    return 0.5 * np.log(np.exp(2.0 * x) + n)


class FloorCanonicalTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(3)
        self.T, self.S = log_mel(rng), log_mel(rng) - 0.5

    def test_invariant_to_an_added_floor(self):
        n = np.array([1e-4, 1e-3, 3e-3, 1e-2, 2e-2, 5e-2])
        np.testing.assert_allclose(fc.floor_canonical(self.S, add_power(self.T, n)),
                                   fc.floor_canonical(self.S, self.T), atol=1e-9)

    def test_floor_matches_the_students(self):
        S = self.S.copy()
        S[:3] -= 8.0                       # a few very quiet frames keep the clamp far below the 10th percentile
        out_power = np.exp(2.0 * fc.floor_canonical(S, self.T))
        np.testing.assert_allclose(np.percentile(out_power, fc.FLOOR_Q, axis=0),
                                   np.percentile(np.exp(2.0 * S), fc.FLOOR_Q, axis=0), rtol=1e-9)

    def test_returns_the_teacher_when_floors_agree(self):
        S = self.T.copy()
        quietest = np.argsort(S, axis=0)[:5]    # q01 interpolates the 3rd and 4th quietest; q10 sits near the 24th
        np.put_along_axis(S, quietest, np.take_along_axis(S, quietest, axis=0) - 8.0, axis=0)   # same q10, lower clamp
        self.assertTrue(np.allclose(np.percentile(np.exp(2 * S), fc.FLOOR_Q, axis=0),
                                    np.percentile(np.exp(2 * self.T), fc.FLOOR_Q, axis=0)))
        np.testing.assert_allclose(fc.floor_canonical(S, self.T), self.T, atol=1e-9)

    def test_torch_matches_numpy_and_payload_keeps_prompt(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch not installed")
        t = torch.tensor(self.T, dtype=torch.float32)
        s = torch.tensor(self.S, dtype=torch.float32)
        np.testing.assert_allclose(fc.floor_canonical(s, t).numpy(),
                                   fc.floor_canonical(self.S.astype(np.float32), self.T.astype(np.float32)), atol=1e-5)
        out = fc.floor_payload(s.unsqueeze(0), t.unsqueeze(0), 40)
        torch.testing.assert_close(out[0, :40], t[:40])
        torch.testing.assert_close(out[0, 40:], fc.floor_canonical(s[40:], t[40:]))


if __name__ == "__main__":
    unittest.main()
