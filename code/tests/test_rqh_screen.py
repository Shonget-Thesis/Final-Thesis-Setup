"""The Roughness-Selected Quantile Handover (RQH) payload in code/analysis/rqh_screen.py."""
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for p in ("code/colab", "code/analysis"):
    sys.path.insert(0, str(ROOT / p))
import rqh_screen as rq  # noqa: E402
import run_method_comparison as mc  # noqa: E402


def smooth_and_rough(frames=200, bands=4, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(frames)[:, None]
    smooth = np.sin(2 * np.pi * t / 80.0 + np.arange(bands)[None, :]) + 0.01 * rng.normal(size=(frames, bands))
    rough = smooth + 1.5 * rng.normal(size=(frames, bands))
    return smooth, rough


class RoughnessSelectedTests(unittest.TestCase):
    def test_smooth_teacher_is_handed_over_unchanged(self):
        smooth, rough = smooth_and_rough()
        out, mask = rq.roughness_selected(rough, smooth)        # student draft rough, teacher smooth
        self.assertFalse(mask.any())
        np.testing.assert_array_equal(out, smooth)

    def test_rough_teacher_gives_the_quantile_handover(self):
        smooth, rough = smooth_and_rough()
        out, mask = rq.roughness_selected(smooth, rough)        # teacher rough, draft smooth
        self.assertTrue(mask.all())
        np.testing.assert_allclose(out, mc.quantile_handover(smooth, rough))

    def test_mixed_bands_are_selected_one_by_one(self):
        smooth, rough = smooth_and_rough()
        teacher = smooth.copy()
        teacher[:, 1] = rough[:, 1]
        out, mask = rq.roughness_selected(smooth, teacher)
        self.assertEqual(mask.tolist(), [False, True, False, False])
        np.testing.assert_array_equal(out[:, [0, 2, 3]], teacher[:, [0, 2, 3]])
        np.testing.assert_allclose(out[:, 1], mc.quantile_handover(smooth, teacher)[:, 1])

    def test_torch_payload_keeps_the_teachers_prompt(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch not installed")
        smooth, rough = smooth_and_rough()
        s0 = torch.tensor(smooth, dtype=torch.float32).unsqueeze(0)
        t8 = torch.tensor(rough, dtype=torch.float32).unsqueeze(0)
        out, mask = rq.rqh_payload(s0, t8, 30)
        torch.testing.assert_close(out[0, :30], t8[0, :30])
        np.testing.assert_allclose(out[0, 30:].numpy(), mc.quantile_handover(smooth[30:], rough[30:]), atol=1e-6)


if __name__ == "__main__":
    unittest.main()
