import csv
import importlib.util
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code/colab"))
import run_guidance_work_prelim as gwp  # noqa: E402
import run_swsr_handover_prelim as swsr  # noqa: E402
import test_swsr_handover as tswsr  # noqa: E402

HAVE_SCI = importlib.util.find_spec("scipy") is not None and importlib.util.find_spec("numpy") is not None


class GuidanceWorkFormulaTests(unittest.TestCase):
    def test_preregistration_verification(self):
        pre = gwp.check_prereg()
        self.assertEqual(pre["experiment"], "EXP-019")
        self.assertIn("H1", pre["hypotheses"])
        self.assertIn("H2", pre["hypotheses"])
        self.assertIn("H3", pre["hypotheses"])
        self.assertIn("H4", pre["hypotheses"])
        self.assertIn("H5", pre["hypotheses"])

    def test_lemma1_exact_algebraic_decomposition(self):
        """Validates that delta_ep = (1 - t_j)(v_j - v_{j-1}) and d_j^2 = pc + pg + px exactly."""
        import numpy as np
        rng = np.random.RandomState(42)
        # Random velocities over frames M
        c_prev, u_prev = rng.randn(10, 80), rng.randn(10, 80)
        c_curr, u_curr = rng.randn(10, 80), rng.randn(10, 80)
        w = 2.0
        g_prev = c_prev - u_prev
        g_curr = c_curr - u_curr
        v_prev = c_prev + w * g_prev
        v_curr = c_curr + w * g_curr

        t_prev, t_curr = 0.05, 0.10
        # Euler update: x_curr = x_prev + (t_curr - t_prev) * v_prev
        x_prev = rng.randn(10, 80)
        x_curr = x_prev + (t_curr - t_prev) * v_prev

        # Projected endpoints
        ep_prev = x_prev + (1.0 - t_prev) * v_prev
        ep_curr = x_curr + (1.0 - t_curr) * v_curr

        delta_ep = ep_curr - ep_prev
        # Lemma 1: delta_ep == (1 - t_curr) * (v_curr - v_prev)
        lemma1_rhs = (1.0 - t_curr) * (v_curr - v_prev)
        np.testing.assert_allclose(delta_ep, lemma1_rhs, rtol=1e-12, atol=1e-12)

        # Decomposed parts
        h = 1.0 - t_curr
        dc = h * (c_curr - c_prev)
        dg = w * h * (g_curr - g_prev)
        np.testing.assert_allclose(delta_ep, dc + dg, rtol=1e-12, atol=1e-12)

        norm_ep = np.linalg.norm(ep_curr)
        den = norm_ep ** 2
        d2 = (np.linalg.norm(delta_ep) / norm_ep) ** 2
        pc = np.linalg.norm(dc) ** 2 / den
        pg = np.linalg.norm(dg) ** 2 / den
        px = 2.0 * np.sum(dc * dg) / den
        self.assertAlmostEqual(d2, pc + pg + px, places=10)

    def test_auc_calculation(self):
        # Perfect separation
        scores = [1.0, 2.0, 3.0, 4.0]
        labels = [False, False, True, True]
        self.assertAlmostEqual(gwp.auc(scores, labels), 1.0)

        # Inverted separation
        labels_inv = [True, True, False, False]
        self.assertAlmostEqual(gwp.auc(scores, labels_inv), 0.0)

        # Tie handling
        scores_tie = [1.0, 2.0, 2.0, 3.0]
        labels_tie = [False, False, True, True]
        self.assertAlmostEqual(gwp.auc(scores_tie, labels_tie), 0.875)

    def test_pareto_path_monotonicity(self):
        """Checks that the Pareto marginal-efficiency path produces non-increasing NFE on training data."""
        s18_rows = tswsr.make_steps1to8_rows(8)
        table = swsr.build_steps1to8_table(s18_rows)
        # Mock telemetry
        for r in table:
            r["tel"] = {
                f"g16_d_{j}": r["d16"][j] for j in swsr.ALL_G16_STEPS
            }
            r["tel"]["g16_gc_0"] = 0.5
        path = gwp.build_path(table, "g16_floor", "A", mode="pareto", n_q=20)
        self.assertGreater(len(path), 1)
        prev_cost = float("inf")
        cands, ref, S, L, C = gwp._compile(table, "g16_floor", "A")
        for lams in path:
            cost, _ = gwp._cost_risk(S, L, C, lams)
            self.assertLessEqual(cost, prev_cost + 1e-9)
            prev_cost = cost


if __name__ == "__main__":
    unittest.main()

