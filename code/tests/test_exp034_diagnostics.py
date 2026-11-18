"""Unit and imported-result checks for EXP-034 post-hoc diagnostics."""
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code" / "analysis"))
sys.path.insert(0, str(ROOT / "code" / "colab"))
import exp034_diagnostics as dg  # noqa: E402
import run_invariant_residual as ir  # noqa: E402


class HelperTests(unittest.TestCase):
    def test_verify_choices_detects_formula_match_and_mismatch(self):
        good = {"prompt_snr": 25.0, "r": 0.40, "drift8": 0.10,
                "choice_SNR_ONLY": "CPH8", "choice_SNR_DRIFT": "CPQH",
                "choice_SNR_R": "CPQH", "choice_R_ONLY": "CPQH"}
        bad = dict(good, choice_SNR_R="CPH8")
        self.assertEqual(dg.verify_choices([good]), {"cases": 1, "mismatches": 0})
        self.assertEqual(dg.verify_choices([bad]), {"cases": 1, "mismatches": 1})


@unittest.skipUnless((dg.RES / "quality.csv").exists(), "EXP-034 results not imported")
class ImportedResultsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.diag = dg.main()
        cls.res = json.loads((dg.RES / "invariant_residual_results.json").read_text(encoding="utf-8"))

    def test_preregistration_and_choices_verified(self):
        ir.check_prereg()
        self.assertEqual(self.diag["choice_verification"], {"cases": 240, "mismatches": 0})
        self.assertTrue(self.diag["demand_matches_exp031"])

    def test_verdict_and_fresh_cohort(self):
        self.assertEqual(self.res["verdict"], "snr_r_better")
        self.assertTrue(all(self.res["hypotheses"].values()))
        fresh = self.diag["cohort_breakdown"]["pos_41_60_fresh"]
        self.assertEqual(fresh["cases"], 160)
        self.assertEqual(fresh["speakers"], 20)
        self.assertTrue(fresh["SNR_R_minus_SNR_ONLY"]["better"])
        self.assertTrue(fresh["SNR_R_minus_SNR_DRIFT"]["better"])

    def test_clean_flags_and_runaways(self):
        self.assertEqual(len(self.diag["clean_flagged_cases"]), 2)
        self.assertEqual(self.diag["runaways_by_condition"]["noisy15"]["CPH8"], 6)
        self.assertEqual(self.diag["runaways_by_condition"]["noisy15"]["CPQH"], 1)
        self.assertEqual(self.diag["runaways_by_condition"]["noisy15"]["STUR2"], 0)


if __name__ == "__main__":
    unittest.main()
