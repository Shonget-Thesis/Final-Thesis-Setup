import csv
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code/colab"))
import run_grid32_prelim as g  # noqa: E402


class GridTests(unittest.TestCase):
    def test_grid32_shares_the_released_16_interval_points(self):
        g16 = g.sway_grid(16)
        self.assertEqual(len(g.GRID32), 33)
        self.assertEqual(g.GRID32[0], 0.0)
        self.assertAlmostEqual(g.GRID32[-1], 1.0)
        self.assertTrue(all(a < b for a, b in zip(g.GRID32, g.GRID32[1:])))
        for i, t in enumerate(g16):
            self.assertAlmostEqual(g.GRID32[2 * i], t)

    def test_paper_rule_gives_fourteen_teacher_steps_at_0_25(self):
        self.assertEqual(g.K32, {0.07: 7, 0.10: 9, 0.15: 11, 0.20: 13, 0.25: 14})
        for s, k in g.K32.items():
            self.assertLessEqual(g.GRID32[k], s)
            self.assertGreater(g.GRID32[k + 1], s)

    def test_released_rule_on_16_intervals_gives_steps_4_and_8(self):
        g16 = g.sway_grid(16)
        self.assertEqual(g.first_above(g16, 0.07), 4)
        self.assertEqual(g.first_above(g16, 0.25), 8)
        self.assertAlmostEqual(g16[4], 1 - math.cos(math.pi / 8))

    def test_costs_count_two_passes_per_guided_evaluation(self):
        self.assertEqual(g.COST, {"g16_4": 13, "g16_8": 21, "s007": 19, "s010": 23, "s015": 27, "s020": 31, "s025": 33, "paper": 32})
        self.assertEqual(g.CANDS32[-1], g.REF32)


def _m(wer, sim):
    return {"wer": wer, "similarity": sim, "utmos": 3.5}


class RuleTests(unittest.TestCase):
    th = {"absolute_wer_ceiling": 0.125, "relative_wer_margin": 0.01, "similarity_margin": 0.02}

    def test_failure_rule_matches_definition_3_1(self):
        ref = _m(0.03, 0.70)
        self.assertFalse(g.failed(_m(0.045, 0.69), ref, self.th))     # WER limit is max(12.5%, 4%) = 12.5%
        self.assertTrue(g.failed(_m(0.045, 0.67), ref, self.th))      # SIM limit is 0.68
        self.assertTrue(g.failed(_m(0.20, 0.70), _m(0.153, 0.70), self.th))

    def test_cheapest_safe_picks_the_first_candidate_that_passes(self):
        rows = []
        for i, sims in enumerate(([0.70] * 5, [0.60, 0.70, 0.70, 0.70, 0.70], [0.60] * 4 + [0.70])):
            m = {o: _m(0.0, s) for o, s in zip(g.CANDS32, sims)}
            rows.append({"utt": f"u{i}", "cond": "clean", "m": m})
        v = g.cheapest_safe(rows, g.CANDS32, g.REF32, self.th)
        self.assertEqual(v["counts"]["s007"], 1)
        self.assertEqual(v["counts"]["s010"], 1)
        self.assertEqual(v["counts"]["s025"], 1)
        self.assertAlmostEqual(v["mean_cost"], (19 + 23 + 33) / 3)


class AnalysisTests(unittest.TestCase):
    def test_analyze_writes_report_and_counts_failures_from_the_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            (out / "preregistration.json").write_text(json.dumps({"sha256": "0" * 64}), encoding="utf-8")
            rows = []
            for spk in range(2):
                for c in g.CONDS:
                    noisy = c != "clean"
                    row = {"utt_id": f"u{spk}", "speaker_id": f"s{spk}", "condition": c, "prompt_snr": 18.0 if noisy else 40.0,
                           "gpu": "Tesla T4", "gt_sr": 16000, "case_seconds": 30.0, "d4_16": 0.5, "d4_stored": 0.5, "utc": ""}
                    for o in g.OUTS:
                        early = o in ("g16_4", "s007") and noisy
                        row.update({f"wer_{o}": 0.02, f"similarity_{o}": 0.60 if early else 0.70, f"utmos_{o}": 3.5, f"hypothesis_{o}": "x"})
                    for o in ("g16_4", "g16_8"):
                        row.update({f"wer_{o}_stored": row[f"wer_{o}"], f"similarity_{o}_stored": row[f"similarity_{o}"]})
                    rows.append(row)
            with open(out / "grid32.csv", "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=g.FIELDS)
                w.writeheader()
                w.writerows(rows)
            res = g.analyze(out, allow_partial=True)
            self.assertTrue(res["partial"])
            self.assertEqual(res["fixed_vs_ref32"]["s007"]["noisy"], 10)
            self.assertEqual(res["fixed_vs_ref32"]["s007"]["clean"], 0)
            self.assertEqual(res["fixed_vs_ref32"]["s010"]["total"], 0)
            self.assertEqual(res["fixed_vs_ref16"]["g16_4"]["noisy"], 10)
            self.assertEqual(res["reproduction"]["same_step4_failure_label"], 12)
            self.assertEqual(res["reproduction"]["max_abs_diff"]["similarity_g16_4"], 0.0)
            self.assertEqual(res["cheapest_safe_32"]["counts"]["s010"], 10)
            report = (out / "GRID32_REPORT.md").read_text(encoding="utf-8")
            self.assertIn("32-interval grid", report)
            self.assertIn("PARTIAL", report)


if __name__ == "__main__":
    unittest.main()
