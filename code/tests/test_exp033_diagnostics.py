"""Checks of the EXP-033 diagnostics' policy helper and, when the results are present, of what it reads."""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code" / "analysis"))
sys.path.insert(0, str(ROOT / "code" / "colab"))
import exp033_diagnostics as dg  # noqa: E402
import run_cheaper_teacher_path as ct  # noqa: E402
import run_method_comparison as mc  # noqa: E402


def row(cond, snr, utmos):
    r = {"utt_id": f"u_{cond}", "condition": cond, "prompt_snr": snr}
    for a, u in utmos.items():
        r.update({f"utmos_{a}": u, f"wer_{a}": 0.0, f"similarity_{a}": 0.9})
    return r


class PolicyTests(unittest.TestCase):
    def setUp(self):
        self.rows = [row("clean", 35.0, {"STU": 3.9, "CPH8": 3.7}), row("noisy15", 18.0, {"STU": 2.0, "CPH8": 1.3})]
        self.read = lambda r, a, m: dg.num(r, f"{m}_{a}")

    def test_snr_routing_sends_noisy_prompts_to_the_student(self):
        def pick(r):
            return ("STU", ct.COST["STU"]) if dg.num(r, "prompt_snr") < ct.THETA else ("CPH8", ct.COST["CPH8"])
        p = dg.policy(self.rows, pick, self.read)
        self.assertEqual(p["mean_passes"], (12 + 4) / 2)
        self.assertEqual((p["clean"]["passes"], p["clean"]["utmos"], p["clean"]["choices"]), (12, 3.7, {"CPH8": 1}))
        self.assertEqual((p["noisy15"]["passes"], p["noisy15"]["utmos"], p["noisy15"]["choices"]), (4, 2.0, {"STU": 1}))

    def test_chapter_3_sends_noisy_prompts_to_step_8(self):
        self.assertEqual(mc.ch3_step({4: 0.1, 5: 0.1, 6: 0.1}, 18.0, mc.CH3_LAMBDA), 8)     # gate closed
        self.assertEqual(mc.ch3_step({4: 0.5, 5: 0.3, 6: 0.1}, 35.0, mc.CH3_LAMBDA), 5)     # first d_k <= 0.35
        self.assertEqual(2 * 8 + 5, ct.COST["W8"])

    def test_missing_values_are_not_numbers(self):
        self.assertNotEqual(dg.num({}, "x"), dg.num({}, "x"))            # NaN
        self.assertEqual(dg.num({"x": "1.5"}, "x"), 1.5)


@unittest.skipUnless((dg.EXP / "results" / "quality.csv").exists(), "EXP-033 results not imported")
class ImportedResultsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.res = dg.main()

    def test_the_join_to_exp031_is_exact_enough(self):
        j = self.res["policies_on_the_60_cases"]["join_check"]
        self.assertEqual(j["cases"], 60)
        self.assertLess(j["max_abs_d4_d5_d6"], 1e-4)
        self.assertLess(max(j["max_abs_W8"].values()), 1e-2)

    def test_counts_and_costs(self):
        p = self.res["policies_on_the_60_cases"]
        self.assertEqual(p["fixed_P25"]["mean_passes"], 30)
        self.assertEqual(p["fixed_W8"]["mean_passes"], 21)
        self.assertEqual(sum(p["CH3_lambda_0.35"]["noisy15"]["choices"].values()), 30)
        self.assertLessEqual(p["SNR_then_CPH8"]["mean_passes"], ct.COST["CPH8"])
        self.assertEqual(set(self.res["noisy15"]["runaway"]), set(ct.ARMS))
        self.assertEqual(self.res["timing"]["cases"], 10)


if __name__ == "__main__":
    unittest.main()
