"""The prompt-frame swap used by code/analysis/prompt_region_control.py."""
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code" / "analysis"))
import prompt_region_control as prc  # noqa: E402


class PromptSwapTests(unittest.TestCase):
    def test_numpy_swap_replaces_only_prompt_frames(self):
        payload, source = np.zeros((1, 10, 3)), np.ones((1, 10, 3))
        out = prc.with_prompt(payload, source, 4)
        np.testing.assert_array_equal(out[0, :4], 1.0)
        np.testing.assert_array_equal(out[0, 4:], 0.0)
        np.testing.assert_array_equal(payload, 0.0)                 # the input is not modified

    def test_torch_swap(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch not installed")
        payload, source = torch.zeros(1, 10, 3), torch.ones(1, 10, 3)
        out = prc.with_prompt(payload, source, 6)
        self.assertTrue(bool((out[0, :6] == 1).all()) and bool((out[0, 6:] == 0).all()))
        self.assertTrue(bool((payload == 0).all()))

    def test_readings(self):
        self.assertEqual(prc.reading(0.6, prc.YES3), "yes")
        self.assertEqual(prc.reading(0.05, prc.YES3), "no")
        self.assertEqual(prc.reading(0.2, prc.YES3), "partly")


if __name__ == "__main__":
    unittest.main()
