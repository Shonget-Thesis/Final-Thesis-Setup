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
import run_grid32_prelim as g  # noqa: E402

HAVE_SCI = importlib.util.find_spec("scipy") is not None and importlib.util.find_spec("numpy") is not None
SNR = {"clean": 40.0, "noisy15": 17.0, "gauss18": 21.0, "gauss20": 22.0, "gauss22": 23.0, "gauss25": 24.0}


def make_rows(n_speakers, with_drift=True):
    """Premature stabilization by construction: noisy prompts read 21-24 dB except noisy15 (17 dB), have lower drift than
    clean ones, and fail at every early candidate of both grids; clean prompts never fail."""
    early = set(g.GRIDS["g16"]["cands"]) | set(g.GRIDS["g32"]["cands"])
    rows = []
    for spk in range(n_speakers):
        for c in g.CONDS:
            noisy = c != "clean"
            row = {"utt_id": f"u{spk}", "speaker_id": f"s{spk}", "condition": c, "prompt_snr": SNR[c], "gpu": "Tesla T4",
                   "gt_sr": 16000, "case_seconds": 45.0, "d4_16": 0.5, "d4_stored": 0.5, "utc": "",
                   "t_prepare": 1.0, "t_eval": 0.05, "t_student": 0.04, "t_vocoder": 0.1, "audio_seconds": 4.0, "peak_mem_mb": 5000.0}
            if with_drift:
                row.update({f"d16_{i}": 0.1 if noisy else 0.5 for i in range(1, 9)})
                row.update({f"d32_{i}": 0.05 if noisy else 0.25 for i in range(1, 15)})
            for o in g.OUTS:
                bad = noisy and o in early
                row.update({f"wer_{o}": 0.02, f"similarity_{o}": 0.60 if bad else 0.70, f"utmos_{o}": 3.5, f"hypothesis_{o}": "x"})
            for k in (4, 8):
                row.update({f"wer_g16_{k}_stored": row[f"wer_g16_{k}"], f"similarity_g16_{k}_stored": row[f"similarity_g16_{k}"]})
            rows.append(row)
    return rows


def run_analysis(tmp, rows):
    out = Path(tmp)
    (out / "preregistration.json").write_text(json.dumps({"sha256": "0" * 64}), encoding="utf-8")
    with open(out / "grid32.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=g.FIELDS)
        w.writeheader()
        w.writerows(rows)
    return g.analyze(out, allow_partial=True), (out / "GRID32_REPORT.md").read_text(encoding="utf-8")


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

    def test_released_rule_on_16_intervals_gives_steps_4_to_8(self):
        g16 = g.sway_grid(16)
        self.assertEqual([g.first_above(g16, s) for s in (0.07, 0.10, 0.15, 0.25)], [4, 5, 6, 8])

    def test_costs_count_two_passes_per_guided_evaluation(self):
        self.assertEqual(g.COST, {"g16_4": 13, "g16_5": 15, "g16_6": 17, "g16_8": 21, "s007": 19, "s010": 23, "s015": 27,
                                  "s020": 31, "s025": 33, "paper": 32, "alg2": 30, "teacher32": 64, "student4": 4})

    def test_algorithm_2_student_starts_at_t14_then_the_last_student_level(self):
        s1, s2 = g.ALG2_TIMES
        self.assertAlmostEqual(s1, 1 - math.cos(14 * math.pi / 64))
        self.assertAlmostEqual(s2, 1 - math.cos(math.pi / 2 * 0.75))

    def test_threshold_grids_are_fixed_and_ordered_conservative_first(self):
        self.assertEqual(g.LAM16[0], 0.20)
        self.assertEqual(g.LAM16[-1], 1.20)
        self.assertAlmostEqual(g.LAM32[0], 0.01)
        self.assertAlmostEqual(g.LAM32[-1], 1.20)
        self.assertTrue(all(a < b for a, b in zip(g.LAM32, g.LAM32[1:])))


class RuleTests(unittest.TestCase):
    th = {"absolute_wer_ceiling": 0.125, "relative_wer_margin": 0.01, "similarity_margin": 0.02}

    def test_failure_rule_matches_definition_3_1(self):
        m = lambda w, s: {"wer": w, "similarity": s, "utmos": 3.5}
        self.assertFalse(g.failed(m(0.045, 0.69), m(0.03, 0.70), self.th))    # WER limit max(12.5%, 4%) = 12.5%
        self.assertTrue(g.failed(m(0.045, 0.67), m(0.03, 0.70), self.th))     # SIM limit 0.68
        self.assertTrue(g.failed(m(0.20, 0.70), m(0.153, 0.70), self.th))

    def test_runtime_composes_measured_unit_times(self):
        r = {"time": {"t_prepare": 1.0, "t_eval": 0.05, "t_student": 0.04, "t_vocoder": 0.1, "audio_seconds": 4.0}}
        sec, rtf = g.runtime(r, "g16_8")                                   # 9 guided evaluations, 3 student steps
        self.assertAlmostEqual(sec, 1.0 + 9 * 0.05 + 3 * 0.04 + 0.1)
        self.assertAlmostEqual(rtf, sec / 4.0)
        self.assertEqual(g.runtime({"time": {"t_prepare": None}}, "g16_8"), (None, None))


class AnalysisTests(unittest.TestCase):
    def test_partial_run_without_enough_speakers_skips_the_replay(self):
        with tempfile.TemporaryDirectory() as tmp:
            res, report = run_analysis(tmp, make_rows(2))
            self.assertTrue(res["partial"])
            self.assertNotIn("objective4", res)
            self.assertEqual(res["objective1"]["g16"]["fixed"]["g16_4"]["noisy"], 10)
            self.assertEqual(res["objective1"]["g32"]["fixed"]["s007"]["clean"], 0)
            self.assertEqual(res["reproduction"]["same_step4_failure_label"], 12)
            self.assertIn("PARTIAL", report)

    @unittest.skipUnless(HAVE_SCI, "scipy and numpy are needed")
    def test_full_rehearsal_on_premature_stabilization(self):
        with tempfile.TemporaryDirectory() as tmp:
            res, report = run_analysis(tmp, make_rows(10))
        for G in ("g16", "g32"):
            o1, o4, o3 = res["objective1"][G], res["objective4"][G], res["objective3"][G]
            first = g.GRIDS[G]["cands"][0]
            self.assertEqual(o1["theta_rule"], 25.0)                          # above the noisy prompts' 21-24 dB
            self.assertEqual(o1["oracle_per_condition"]["clean"][first], 10)
            self.assertEqual(o1["oracle_earlier_share"]["gauss18"], 0.0)
            self.assertLess(o1["association"]["snr"]["rho"], 0)                # higher SNR -> earlier oracle choice
            m = o4["methods"]
            self.assertEqual(m["snr20"]["failures"], 40)                      # 18-25 dB noise passes the 20 dB gate and fails
            self.assertEqual(m["later5"]["failures"], 40)
            self.assertEqual(m["snr_cal"]["failures"], 0)
            self.assertEqual(m["stab"]["failures"], 0)                        # low drift marks the failing prompts: nothing certified
            self.assertEqual(m["stab"]["mean_cost"], g.COST[g.GRIDS[G]["ref"]])
            self.assertEqual(m["snr_drift"]["failures"], 0)
            self.assertEqual(m["snr_drift_dev"]["failures"], 0)
            self.assertEqual(m["snr_drift_dev"]["picks"][first], 10)          # gate 25 dB, then every clean prompt goes early
            self.assertEqual(m["snr_tas"]["picks"][first], 10)
            self.assertEqual(m["oracle"]["failures"], 0)
            self.assertAlmostEqual(m["oracle"]["eta"], 1.0)
            self.assertAlmostEqual(m["random"]["mean_cost_over_draws"], m["snr_drift"]["mean_cost"], delta=0.5)
            self.assertEqual(o4["mcnemar"]["snr_drift vs snr20"]["fixed"], 40)
            self.assertIn("wilcoxon_p_holm", o4["tests"]["snr20"]["similarity"])
            self.assertIn("tost_p", o4["tests"]["snr20"]["wer"])
            self.assertEqual(set(o4["clean_to_noisy"]), {"snr_cal", "stab", "snr_drift"})
            self.assertEqual(set(o3["snr_drift"]), {"all", "50", "100", "200"})
            self.assertEqual(o3["snr_cal"]["all"]["mean_risk"], 0.0)
        self.assertIn("Testing the stabilization signal", report)
        self.assertIn("Objective 3 rehearsal", report)
        self.assertIn("Full policy (Definition 3.4)", report)


if __name__ == "__main__":
    unittest.main()
