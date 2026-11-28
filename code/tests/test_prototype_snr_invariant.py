"""Tests for the SNR + Invariant Residual (SNR_R) prototype module, engine helpers, server status, and UI."""
import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code/scheduling/prototype"))
sys.path.insert(0, str(ROOT / "code/colab"))

import engine as eng  # noqa: E402
import run_grid32_prelim as gl  # noqa: E402
import run_invariant_residual as ir  # noqa: E402
import run_method_comparison as mc  # noqa: E402
import snr_invariant as si  # noqa: E402


def _speech_like(rng, n, sr):
    t = np.arange(n) / sr
    env = (np.sin(2 * np.pi * 2.3 * t) > 0).astype(np.float32) * 0.9 + 0.05
    return (0.3 * env * np.sin(2 * np.pi * 180 * t) * (1 + 0.3 * rng.normal(size=n))).astype(np.float32)


class ConstantsTests(unittest.TestCase):
    def test_matches_exp034_constants_and_costs(self):
        self.assertEqual(len(si.TEACHER_GRID), 9)
        for a, b in zip(si.TEACHER_GRID, gl.sway_grid(8)):
            self.assertAlmostEqual(a, b)
        self.assertAlmostEqual(si.TEACHER_GRID[si.COMMIT_STEP], ir.T_COMMIT)
        self.assertEqual(si.TEACHER_PASSES, 10)
        self.assertEqual(si.DRAFT_PASSES, 1)
        self.assertEqual(si.STUDENT_PASSES, 2)
        self.assertEqual(si.METHOD_PASSES, 13)
        self.assertEqual(si.CPH8_PASSES, 12)
        self.assertEqual(si.W8_REFERENCE_PASSES, 21)
        self.assertEqual(si.P25_PAPER_PASSES, 30)
        self.assertEqual(si.SNR_GATE_DB, ir.THETA)
        self.assertEqual(si.EMPIRICAL_DEV_LAMBDA_R, ir.LAMBDA_R)
        self.assertEqual(si.DEFAULT_LAMBDA_R, si.LTT_15_LAMBDA_R)


class OperatorTests(unittest.TestCase):
    def test_quantile_and_residual_match_experiment_runners_numpy(self):
        rng = np.random.default_rng(11)
        s0 = rng.normal(size=(1, 70, 100)).astype(np.float32)
        t4 = rng.normal(loc=0.4, scale=1.8, size=(1, 70, 100)).astype(np.float32)
        plen = 25

        q_proto = si.quantile_handover(s0[0, plen:, :], t4[0, plen:, :])
        q_exp = mc.quantile_handover(s0[0, plen:, :], t4[0, plen:, :])
        np.testing.assert_allclose(q_proto, q_exp)

        p_proto = si.quantile_payload(s0, t4, plen)
        p_exp = mc.quantile_payload(s0, t4, plen)
        np.testing.assert_allclose(p_proto, p_exp)
        np.testing.assert_array_equal(p_proto[:, :plen, :], s0[:, :plen, :])

        r_proto = si.invariant_residual(t4[0, plen:, :], s0[0, plen:, :])
        r_exp = ir.invariant_residual(t4[0, plen:, :], s0[0, plen:, :])
        self.assertAlmostEqual(r_proto, r_exp, places=12)
        self.assertEqual(si.invariant_residual(s0[0, plen:, :], s0[0, plen:, :]), 0.0)

        bins = si.band_residuals(t4[0, plen:, :], s0[0, plen:, :], n_bins=8)
        self.assertEqual(len(bins), 8)
        self.assertTrue(all(v >= 0.0 for v in bins))

    def test_monotone_invariance_of_quantile_handover(self):
        rng = np.random.default_rng(12)
        s0 = rng.normal(size=(45, 20))
        t4 = rng.normal(size=(45, 20))
        q_base = si.quantile_handover(s0, t4)
        # Positive affine transform per band preserves ranks -> identical Q(t4, s0)
        scale = rng.uniform(0.5, 3.0, size=(1, 20))
        shift = rng.uniform(-5.0, 5.0, size=(1, 20))
        q_aff = si.quantile_handover(s0, t4 * scale + shift)
        np.testing.assert_allclose(q_base, q_aff)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch not installed")
    def test_torch_matches_experiment_runners(self):
        import torch

        g = torch.Generator().manual_seed(7)
        s0 = torch.randn(1, 65, 100, generator=g)
        t4 = torch.randn(1, 65, 100, generator=g) * 1.5 + 0.2
        plen = 20
        torch.testing.assert_close(
            si.quantile_handover(s0[0, plen:, :], t4[0, plen:, :]),
            mc.quantile_handover(s0[0, plen:, :], t4[0, plen:, :]),
        )
        torch.testing.assert_close(si.quantile_payload(s0, t4, plen), mc.quantile_payload(s0, t4, plen))
        self.assertAlmostEqual(
            si.invariant_residual(t4[0, plen:, :], s0[0, plen:, :]),
            ir.invariant_residual(t4[0, plen:, :].numpy(), s0[0, plen:, :].numpy()),
            places=6,
        )


class RuleTests(unittest.TestCase):
    def test_decide_matches_ir_choose_snr_r_and_snr_only(self):
        rng = np.random.default_rng(19)
        for _ in range(2000):
            snr = float(rng.uniform(5.0, 45.0))
            r = float(rng.uniform(0.05, 0.95))
            lam = float(rng.choice([0.25, si.DEFAULT_LAMBDA_R, 0.5, 10.0]))
            self.assertEqual(si.decide(snr, r, lam_r=lam), ir.choose("SNR_R", snr, r, 0.0, lam_r=lam))
        for _ in range(500):
            snr = float(rng.uniform(5.0, 45.0))
            r = float(rng.uniform(0.05, 0.95))
            self.assertEqual(
                si.decide(snr, r, lam_r=si.LAMBDA_R_PRESETS["snr_only"]["value"]),
                ir.choose("SNR_ONLY", snr, r, 0.0),
            )

    def test_evaluate_triggers_and_savings(self):
        clean = si.evaluate(35.0, 0.22, r_trace={0: 0.4, 4: 0.22})
        self.assertEqual((clean["choice"], clean["flagged"], clean["trigger"]), ("CPH8", False, "none"))
        self.assertTrue(clean["gate_open"])
        self.assertEqual(clean["passes"], 13)
        self.assertAlmostEqual(clean["saving"], 1.0 - 13.0 / 21.0)
        self.assertAlmostEqual(clean["saving_vs_paper"], 1.0 - 13.0 / 30.0)

        spec = si.evaluate(27.0, 0.65)
        self.assertEqual((spec["choice"], spec["flagged"], spec["trigger"]), ("CPQH", True, "residual"))

        low_snr = si.evaluate(16.0, 0.20)
        self.assertEqual((low_snr["choice"], low_snr["flagged"], low_snr["trigger"]), ("CPQH", True, "snr"))
        self.assertFalse(low_snr["gate_open"])

        both = si.evaluate(16.0, 0.65)
        self.assertEqual((both["choice"], both["flagged"], both["trigger"]), ("CPQH", True, "both"))

        self.assertIn("risk_certificate", clean)
        self.assertIn("LTT", clean["risk_certificate"])

        custom = si.evaluate(35.0, 0.22, lam_r=0.45)
        self.assertIn("Custom", custom["risk_certificate"])

    def test_binomial_cdf_and_ltt_calibration(self):
        self.assertAlmostEqual(si._binom_cdf(0, 30, 0.15), 0.85 ** 30, places=12)
        self.assertAlmostEqual(si._binom_cdf(1, 30, 0.15), 0.85 ** 30 + 30 * 0.15 * (0.85 ** 29), places=12)

        synth = [
            {"condition": "clean", "prompt_snr": 35.0, "r": 0.10 + 0.01 * i, "utmos_CPH8": 3.8, "utmos_CPQH": 3.8}
            for i in range(30)
        ] + [
            {"condition": "gauss25", "prompt_snr": 27.0, "r": 0.65, "utmos_CPH8": 1.5, "utmos_CPQH": 3.7}
            for _ in range(10)
        ]
        # Clean rows span r in [0.10..0.39]. For alpha=0.15, delta=0.10 on n=30, max k=1 -> lambda_r = 0.38.
        res = si.calibrate_ltt(synth, alpha=0.15, delta=0.10)
        self.assertIsNotNone(res)
        self.assertEqual(res["k_err"], 1)
        self.assertAlmostEqual(res["lambda_r"], 0.38, places=6)
        self.assertLessEqual(res["p_value"], 0.10)

        p34 = ROOT / "research/experiments/EXP-034-invariant-residual/results/quality.csv"
        if p34.exists():
            import csv
            with open(p34, encoding="utf-8", newline="") as f:
                rows34 = list(csv.DictReader(f))
            r15 = si.calibrate_ltt(rows34, alpha=0.15, delta=0.10)
            r10 = si.calibrate_ltt(rows34, alpha=0.10, delta=0.10)
            self.assertAlmostEqual(r15["lambda_r"], si.LTT_15_LAMBDA_R, places=12)
            self.assertEqual(r15["k_err"], 1)
            self.assertLessEqual(r15["p_value"], 0.10)
            self.assertAlmostEqual(r10["lambda_r"], si.LTT_10_LAMBDA_R, places=12)
            self.assertEqual(r10["k_err"], 0)
            self.assertLessEqual(r10["p_value"], 0.10)


class EngineAndInterfaceTests(unittest.TestCase):
    def test_offline_degradations_match_exp034(self):
        rng = np.random.default_rng(23)
        x = _speech_like(rng, 24000 * 3, 24000)
        for cond in eng.OFFLINE_CONDITIONS:
            np.testing.assert_array_equal(eng.degrade_prompt(x, 24000, cond, 17), ir.degrade(x, 24000, cond, 17))
        self.assertIn("pink25", eng.OFFLINE_CONDITIONS)
        self.assertIn("hp25", eng.OFFLINE_CONDITIONS)
        with self.assertRaises(ValueError):
            eng.degrade_prompt(x, 24000, "babble15", 17)

    @unittest.skipUnless(
        importlib.util.find_spec("uvicorn") and importlib.util.find_spec("fastapi"),
        "uvicorn/fastapi not installed",
    )
    def test_server_status_and_examples(self):
        import server

        st = server.status()
        self.assertEqual(st["method"], "snr_invariant")
        self.assertEqual(st["passes"], 13)
        self.assertEqual(st["reference_passes"], 21)
        self.assertEqual(st["paper_passes"], 30)
        self.assertAlmostEqual(st["default_lambda_r"], si.DEFAULT_LAMBDA_R)
        self.assertIn("pink25", st["conditions"])
        self.assertIn("hp25", st["conditions"])
        if (ROOT / "research/experiments/EXP-034-invariant-residual/results/quality.csv").exists():
            rec = server._recorded()
            self.assertEqual(len(rec), 30)
            self.assertAlmostEqual(st["risk_calibration"]["ltt_15"]["lambda_r"], si.LTT_15_LAMBDA_R, places=12)
            self.assertAlmostEqual(st["risk_calibration"]["ltt_10"]["lambda_r"], si.LTT_10_LAMBDA_R, places=12)

    def test_ui_tokens_present(self):
        html = (ROOT / "code/scheduling/prototype/web/index.html").read_text(encoding="utf-8")
        for token in ("SNR + Invariant Residual", "Learn-Then-Test", "CPH8", "CPQH", "EXP-034", "lam_r"):
            self.assertIn(token, html)


if __name__ == "__main__":
    unittest.main()
