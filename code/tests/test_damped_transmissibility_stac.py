"""Unit tests for EXP-029 (Damped Slow-Band Transmissibility Bridge & Mixed-Condition STAC)."""
import ast
import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "colab"))
import run_damped_transmissibility_stac as dts  # noqa: E402
import run_selective_handover_prelim as sh  # noqa: E402
import run_transmissibility_bridge as tb  # noqa: E402


class DampedBridgeMathTests(unittest.TestCase):
    def test_alpha_endpoints_and_energy_conservation(self):
        rng = np.random.default_rng(2029)
        plen, frames, bands = 6, 42, 12
        t = np.linspace(0.0, 2.0 * np.pi, frames - plen)[:, None]
        T4 = rng.normal(0.0, 0.25, size=(1, frames, bands))
        S1 = rng.normal(0.0, 0.25, size=(1, frames, bands))
        T4[0, plen:, :] += 2.8 * np.sin(t)
        S1[0, plen:, :] += 0.9 * np.cos(t)

        # Endpoint identities: alpha=0 == S1 (W4), alpha=1 == EXP-028 TBRG
        out0 = dts.damped_bridge_handover(T4, S1, plen, alpha=0.0, energy_match=True)
        out1 = dts.damped_bridge_handover(T4, S1, plen, alpha=1.0, energy_match=True)
        ref1 = tb.bridge_handover(T4, S1, plen, energy_match=True)
        np.testing.assert_allclose(out0, S1, atol=1e-12)
        np.testing.assert_allclose(out1, ref1, atol=1e-12)

        # Constant slow-band Frobenius norm across intermediate alphas + untouched prompt/v/f
        v_s, s_s, f_s = sh.decompose(S1[0, plen:, :], sh.SLOW_WINDOW)
        norm_s = float(np.linalg.norm(s_s))
        _, s_t, _ = sh.decompose(T4[0, plen:, :], sh.SLOW_WINDOW)
        cos_prev = 1.0
        for alpha in (0.25, 0.50, 0.75):
            out = dts.damped_bridge_handover(T4, S1, plen, alpha=alpha, energy_match=True)
            np.testing.assert_allclose(out[:, :plen, :], S1[:, :plen, :], atol=1e-12)
            s_comp = out[0, plen:, :] - (v_s + f_s)
            np.testing.assert_allclose(float(np.linalg.norm(s_comp)), norm_s, rtol=1e-6, atol=1e-6)
            cos_t = tb.slow_cosine(s_comp, s_t)
            self.assertLessEqual(tb.slow_cosine(s_comp, s_s), 1.0 + 1e-9)
            # Slow band rotates monotonically toward T4
            if alpha > 0.25:
                self.assertGreater(cos_t, cos_prev)
            cos_prev = cos_t

        with self.assertRaises(ValueError):
            dts.damped_bridge_handover(T4, S1, plen, alpha=1.1)

    def test_handover_discrepancy_signals_zero_on_identical(self):
        rng = np.random.default_rng(9)
        plen, frames, bands = 5, 35, 8
        T4 = rng.normal(size=(1, frames, bands))
        sigs = dts.handover_discrepancy_signals(T4, T4, plen)
        self.assertAlmostEqual(sigs["slow_cos_4"], 1.0, places=6)
        self.assertAlmostEqual(sigs["slow_gap_4"], 0.0, places=6)
        self.assertAlmostEqual(sigs["voice_gap_4"], 0.0, places=6)
        self.assertAlmostEqual(sigs["fine_gap_4"], 0.0, places=6)
        self.assertAlmostEqual(sigs["anchor_gap_4"], 0.0, places=6)

        S1 = T4 * 0.5 + 1.0
        sigs_diff = dts.handover_discrepancy_signals(T4, S1, plen)
        self.assertGreater(sigs_diff["anchor_gap_4"], 0.1)
        self.assertGreater(sigs_diff["voice_gap_4"], 0.1)


class CohortCostsAndAnalysisTests(unittest.TestCase):
    def test_cohort_disjoint_from_cal_and_test_and_costs(self):
        self.assertEqual(len(dts.qual_items()), 30)
        self.assertEqual(len(dts.div_items()), 20)
        self.assertEqual(dts.COST["W4"], 13)
        self.assertEqual(dts.COST["TB50"], 13)
        self.assertEqual(dts.COST["W5R2"], 14)
        self.assertEqual(dts.COST["W6R2"], 16)
        self.assertEqual(dts.COST["W8R2"], 20)
        self.assertEqual(dts.COST["W8"], 21)

        root = Path(__file__).resolve().parents[2]
        cal = json.loads((root / "research/protocols/splits/calibration_manifest.json").read_text(encoding="utf-8"))
        test = json.loads((root / "research/protocols/splits/test_manifest.json").read_text(encoding="utf-8"))
        q_spks = {it["speaker_id"] for it in dts.qual_items()}
        self.assertFalse(q_spks & {r["speaker_id"] for r in cal})
        self.assertFalse(q_spks & {r["speaker_id"] for r in test})

    def test_synthetic_mixed_stac_routing_and_verdict(self):
        rng = np.random.default_rng(77)
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            prereg = dts.build_prereg()
            p_path = tmp_dir / "preregistration.json"
            p_path.write_text(json.dumps(prereg, indent=2) + "\n", encoding="utf-8")

            q_path = tmp_dir / "quality.csv"
            d_path = tmp_dir / "diversity.csv"
            items = dts.qual_items()[:15]

            with open(q_path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=dts.QUAL_FIELDS)
                w.writeheader()
                for it in items:
                    for cond in dts.CONDS:
                        is_clean = cond == "clean"
                        a4 = 2.6 + float(rng.normal(0.0, 0.1)) if is_clean else 1.1 + float(rng.normal(0.0, 0.1))
                        gc0 = 0.25 if is_clean else 0.08
                        row = {
                            "utt_id": it["utt_id"],
                            "speaker_id": it["speaker_id"],
                            "condition": cond,
                            "prompt_snr": 36.0 if is_clean else 16.0,
                            "gpu": "UnitTest",
                            "gc_0": gc0,
                            **{f"d_{j}": a4 / 4.0 for j in range(1, 9)},
                            "A_4": a4,
                            "A_5": a4 * 1.2,
                            "A_6": a4 * 1.3,
                            "A_8": a4 * 1.6,
                            "A_4_norm": a4 / gc0,
                            "A_6_norm": a4 * 1.3 / gc0,
                            "rho_4": 0.9,
                            "slow_cos_4": 0.92 if is_clean else 0.70,
                            "slow_gap_4": 0.25 if is_clean else 0.70,
                            "slow_scale_4": 1.1,
                            "voice_gap_4": 0.15 if is_clean else 0.60,
                            "fine_gap_4": 0.20,
                            "anchor_gap_4": 0.30,
                            "case_seconds": 1.0,
                            "utc": "2026-11-15T00:00:00+00:00",
                        }
                        for arm in dts.ARMS:
                            # Make TB50 strong on clean (UTMOS preserved) and risky on noisy (so STAC routes!)
                            if arm == "TB50":
                                utm = 3.92 if is_clean else 1.25
                                wer = 0.005 if is_clean else 0.45
                                sim = 0.94 if is_clean else 0.68
                            elif arm == "TBRG":
                                utm = 3.74 if is_clean else 1.25
                                wer = 0.005 if is_clean else 0.45
                                sim = 0.94 if is_clean else 0.68
                            elif arm in ("W5R2", "W6R2"):
                                utm = 3.90 if is_clean else 1.25
                                wer = 0.005 if is_clean else 0.18
                                sim = 0.94 if is_clean else 0.72
                            else:
                                utm = 3.95 if is_clean else 1.25
                                wer = 0.005 if is_clean else 0.08
                                sim = 0.94 if is_clean else 0.75
                            row[f"wer_{arm}"] = wer
                            row[f"similarity_{arm}"] = sim
                            row[f"utmos_{arm}"] = utm
                            row[f"hypothesis_{arm}"] = "ok"
                        w.writerow(row)

            with open(d_path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=dts.DIV_FIELDS)
                w.writeheader()
                for it in items[:10]:
                    row = {"utt_id": it["utt_id"], "speaker_id": it["speaker_id"], "condition": "clean", "n_seeds": 12}
                    for arm in dts.ARMS:
                        row[f"cvf0_{arm}"] = 0.074 if arm == "W4" else 0.088
                        row[f"voiced_{arm}"] = 0.60
                        row[f"mel_spread_{arm}"] = 0.96
                        row[f"slow_corr_{arm}"] = 0.90
                        row[f"f0_corr_w8r2_{arm}"] = 0.92
                        row[f"f0_shift_corr_w8r2_{arm}"] = 0.50
                    row.update({"case_seconds": 1.0, "utc": "2026-11-15T00:00:00+00:00"})
                    w.writerow(row)

            res = dts.analyze(out_dir=tmp_dir, prereg_path=p_path, allow_partial=True)
            self.assertEqual(res["verdict"], "confirmed_pareto_and_stac")
            self.assertIn("TB50", res["passed_clean_arms"])
            self.assertIn("TB50_to_W8_via_A_4", res["stac_winners"])
            tb50_route = res["routing_5fold"]["TB50_to_W8_via_A_4"]
            self.assertAlmostEqual(tb50_route["cond_early_frac"]["clean"], 1.0, places=2)
            self.assertLess(tb50_route["cond_early_frac"]["noisy15"], 0.05)
            self.assertTrue((tmp_dir / "MANIFEST.json").exists())

    def test_preregistration_hash_and_notebook_valid(self):
        root = Path(__file__).resolve().parents[2]
        prereg_path = root / "research/experiments/EXP-029-damped-transmissibility-stac/preregistration.json"
        if prereg_path.exists():
            doc = dts.check_prereg(prereg_path)
            self.assertEqual(doc["experiment"], "EXP-029")
            self.assertEqual(doc["task_id"], "EVAL-022")

        runner_src = (root / "code/colab/run_damped_transmissibility_stac.py").read_text(encoding="utf-8")
        ast.parse(runner_src)
        nb = json.loads((root / "code/colab/DMOSpeech2_EXP029_Damped_Transmissibility_STAC.ipynb").read_text(encoding="utf-8"))
        self.assertGreaterEqual(len(nb["cells"]), 6)


if __name__ == "__main__":
    unittest.main()
