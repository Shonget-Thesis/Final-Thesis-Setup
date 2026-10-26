import csv
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code/colab"))
import run_noise_guidance_prelim as ng  # noqa: E402

HAVE_NP = importlib.util.find_spec("numpy") is not None
HAVE_SCI = importlib.util.find_spec("scipy") is not None


@unittest.skipUnless(HAVE_NP, "numpy is needed")
class OperatorTests(unittest.TestCase):
    def test_noise_prompt_tiles_the_quietest_frames_and_keeps_the_rest(self):
        import numpy as np
        sc = np.zeros((1, 12, 3))
        energies = [5, 4, -9, 6, -8, 7, 3, 2, 8, 1]           # prompt frames 0-9; frames 10-11 are generated (zeros)
        for i, e in enumerate(energies):
            sc[0, i, :] = e
        nc, n = ng.noise_prompt(sc, 10)
        self.assertEqual(n, 2)                                 # 20% of 10 frames
        self.assertEqual(sorted(set(nc[0, :10, 0])), [-9.0, -8.0])
        np.testing.assert_array_equal(nc[0, 10:], sc[0, 10:])
        self.assertEqual(sc[0, 0, 0], 5)                       # input untouched

    def test_antithetic_pick_prefers_the_output_nearer_the_endpoint(self):
        import numpy as np
        anchor = np.zeros((1, 6, 2))
        near, far = anchor + 0.1, anchor + 1.0
        self.assertEqual(ng.antithetic_pick(near, far, anchor, 2), 1)
        self.assertEqual(ng.antithetic_pick(far, near, anchor, 2), -1)

    def test_scg_gamma_raises_guidance_as_snr_falls_within_the_cap(self):
        self.assertEqual(ng.scg_gamma(40.0, 2.5), 2.0)                     # predicted gap above clean level: no change
        g = ng.SCG_FIT
        self.assertAlmostEqual(ng.scg_gamma(20.0, 2.5), 2.0 * g["G_CLEAN"] / (g["A"] + g["B"] * 20.0))
        self.assertEqual(ng.scg_gamma(20.0, 1.5), 3.0)                     # capped at 1.5 x 2
        self.assertEqual(ng.scg_gamma(0.0, 2.5), 5.0)                      # floor on the predicted gap, then the cap
        self.assertLess(ng.scg_gamma(25.0, 2.5), ng.scg_gamma(20.0, 2.5))

    def test_swc_strength_follows_the_snr_band(self):
        self.assertEqual(ng.swc_beta(35.0, 1.0), 0.0)
        self.assertEqual(ng.swc_beta(25.0, 1.0), 0.5)
        self.assertEqual(ng.swc_beta(15.0, 0.5), 0.5)

    def test_swc_quiets_noise_frames_keeps_speech_and_generated_frames(self):
        import numpy as np
        noise, speech = np.log(0.1), np.log(10.0)              # linear magnitude 0.1 (noise floor) vs 10 (speech)
        cond = np.full((1, 12, 2), noise)
        cond[0, 2:8, :] = speech                                 # prompt frames 0-9: 4 noise, 6 speech; 10-11 generated
        cond[0, 10:, :] = 7.0
        out = ng.swc_clean(cond, 10, beta=1.0)
        np.testing.assert_allclose(np.exp(out[0, 0]), 0.1 * ng.SWC_GMIN)      # noise frame: gain at the floor
        np.testing.assert_allclose(np.exp(out[0, 2]), 10.0 * (1 - 0.1 / 10.0))  # speech frame: 1% reduction
        np.testing.assert_array_equal(out[0, 10:], cond[0, 10:])
        np.testing.assert_array_equal(ng.swc_clean(cond, 10, beta=0.0), cond)
        self.assertEqual(cond[0, 0, 0], noise)                  # input untouched

    def test_heavy_route_and_enhancement_schedule(self):
        pol = ng.heavy(26.0, 22.0, "enh4")
        self.assertEqual(pol({"snr": 30.0}), 4)            # clean: plain Step 4
        self.assertEqual(pol({"snr": 24.0}), 8)            # mild noise: full teacher
        self.assertEqual(pol({"snr": 18.0}), "enh4")       # heavy noise: enhanced prompt, Step 4
        self.assertEqual(ng.heavy(None, None, "enh4")({"snr": 50.0}), 8)
        self.assertEqual(ng.enh_wet(45.0, "full"), 1.0)
        self.assertEqual(ng.enh_wet(35.0, "snr"), 0.0)
        self.assertEqual(ng.enh_wet(18.0, "snr"), 1.0)

    def test_route_sends_inputs_by_snr_band(self):
        pol = ng.route(30.0, 22.0, "snng10")
        self.assertEqual(pol({"snr": 35.0}), 4)
        self.assertEqual(pol({"snr": 25.0}), "snng10")
        self.assertEqual(pol({"snr": 18.0}), 8)
        self.assertEqual(ng.route(None, None, "anti")({"snr": 99.0}), 8)


@unittest.skipUnless(HAVE_SCI, "scipy is needed")
class AnalyzeTests(unittest.TestCase):
    def test_analyze_on_committed_labels_with_synthetic_outcomes(self):
        stored = ng.ex.stored_cases()
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            ng.write_prereg(out, type("A", (), {"dmo_root": tmp}))
            with open(out / "snng.csv", "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=ng.FIELDS)
                w.writeheader()
                for (u, c), s in stored.items():
                    f4 = ng.rf.is_true(s[4]["is_failure"])
                    row = {k: "" for k in ng.FIELDS}
                    row.update({"utt_id": u, "speaker_id": s[8]["speaker_id"], "condition": c, "prompt_snr": s[8]["prompt_snr"],
                                "d_4": s[8]["d_4"], "d4_stored": s[8]["d_4"], "noise_frames": 1, "anti_pick": 1})
                    for o in ng.OUTS:
                        # synthetic: snng10 removes every noisy failure; the others equal plain Step 4
                        row[f"is_failure_{o}"] = f4 and (o != "snng10" or c == "clean")
                        row[f"similarity_{o}"] = s[4]["similarity"]
                        row[f"wer_{o}"] = s[4]["wer"]
                    w.writerow(row)
            res = ng.analyze(out)
            report = (out / "NOISE_GUIDANCE_REPORT.md").read_text(encoding="utf-8")
        self.assertEqual(res["cases"], 360)
        self.assertEqual(res["drift_reproduction_max_abs_diff"], 0.0)
        self.assertTrue(res["criteria"]["snng10"]["C1"] and res["criteria"]["snng10"]["C2"])
        self.assertFalse(res["criteria"]["anti"]["C1"])
        # noisy inputs below the calibrated SNR gate go to a method without noisy failures at 18 instead of 21 NFE
        noisy, base = res["cross_validation"]["pooled"]["snng10_noisy"], res["cross_validation"]["pooled"]["snr_theta"]
        for c in ("noisy15", "gauss18", "gauss20"):
            self.assertEqual(noisy["per_condition"][c], 0)
        self.assertLess(noisy["mean_nfe"], base["mean_nfe"])
        self.assertIn("noise-aware handover methods", report)
        self.assertIn("SNR-Wiener Conditioning", report)
        self.assertIn("enh4_heavy", report)


if __name__ == "__main__":
    unittest.main()
