import csv
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code/colab"))
import run_endpoint_extrapolation_prelim as ex  # noqa: E402

HAVE_NP = importlib.util.find_spec("numpy") is not None
HAVE_SCI = importlib.util.find_spec("scipy") is not None


class OperatorTests(unittest.TestCase):
    def test_deh_factor_is_the_geometric_tail(self):
        self.assertAlmostEqual(ex.deh_factor(0.5, 4), 0.5 + 0.25 + 0.125 + 0.0625)
        self.assertEqual(ex.deh_factor(1.0, 4), 4.0)

    @unittest.skipUnless(HAVE_NP, "numpy is needed")
    def test_exact_on_a_geometric_trajectory_and_prompt_untouched(self):
        import numpy as np
        rng = np.random.default_rng(0)
        A, B, r, plen = rng.normal(size=(1, 30, 4)), rng.normal(size=(1, 30, 4)), 0.8, 10
        E = {i: A + B * r ** i for i in range(9)}
        T, rho, S = ex.extrapolate(E, 4, 8, 1.0, plen)
        self.assertAlmostEqual(rho, r, places=6)                       # up to the 1e-8 guard in the ratio
        np.testing.assert_allclose(T[:, plen:, :], E[8][:, plen:, :], atol=1e-6)
        np.testing.assert_array_equal(T[:, :plen, :], E[4][:, :plen, :])
        d = ex.diagnostics(E, plen, [i / 16 for i in range(17)])
        self.assertAlmostEqual(d["gain_deh"], 1.0, places=6)
        self.assertAlmostEqual(d["gain_h6"], 1.0, places=6)
        self.assertAlmostEqual(d["cos_d4_rest"], 1.0, places=9)


@unittest.skipUnless(HAVE_SCI, "scipy is needed")
class AnalyzeTests(unittest.TestCase):
    def test_analyze_on_committed_labels_with_synthetic_deh_outcomes(self):
        stored = ex.stored_cases()
        self.assertEqual(len(stored), 360)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            ex.write_prereg(out, type("A", (), {"dmo_root": tmp}))
            with open(out / "extrap.csv", "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=ex.FIELDS)
                w.writeheader()
                for (u, c), s in stored.items():
                    fail4 = ex.rf.is_true(s[4]["is_failure"])
                    row = {k: 0.5 for k in ex.FIELDS}
                    row.update({"utt_id": u, "speaker_id": s[8]["speaker_id"], "condition": c, "prompt_snr": s[8]["prompt_snr"],
                                "d_4": s[8]["d_4"], "d4_stored": s[8]["d_4"], "utc": "", "hypothesis_deh": "", "hypothesis_half": "",
                                # synthetic outcome: DEH keeps clean failures and removes noisy ones
                                "is_failure_deh": fail4 and c == "clean", "is_failure_half": fail4})
                    w.writerow(row)
            res = ex.analyze(out)
            report = (out / "EXTRAPOLATION_REPORT.md").read_text(encoding="utf-8")
        self.assertEqual(res["drift_reproduction_max_abs_diff"], 0.0)
        self.assertEqual(res["cases"], 360)
        noisy = res["paired_tests"]["deh vs Step 4, noisy (primary)"]
        self.assertEqual(noisy["failures_deh"], 0)
        self.assertGreater(noisy["failures_4"], 0)
        self.assertTrue(all(res["criteria"][k] for k in res["criteria"] if k.startswith(("C1", "C2"))))
        # deh_all with every noisy failure removed: only the clean Step-4 failures remain
        clean4 = res["step4_failures"]["clean"]["4"]
        self.assertEqual(res["cross_validation"]["pooled"]["deh_all"]["failures"], clean4)
        self.assertEqual(res["cross_validation"]["pooled"]["deh_all"]["mean_nfe"], 13.0)
        self.assertIn("## Preregistered criteria", report)


if __name__ == "__main__":
    unittest.main()
