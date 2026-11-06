import csv
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code/colab"))
import run_grid32_prelim as g32  # noqa: E402
import run_swsr_handover_prelim as swsr  # noqa: E402
import test_grid32 as tg32  # noqa: E402

HAVE_SCI = importlib.util.find_spec("scipy") is not None and importlib.util.find_spec("numpy") is not None


def make_steps1to8_rows(n_speakers):
    """Synthetic premature-stabilization dataset for both 16-grid Steps 1..8 and 32-grid Steps 1..15."""
    rows = []
    for spk in range(n_speakers):
        for c in g32.CONDS:
            noisy = c != "clean"
            row = {
                "utt_id": f"u{spk}",
                "speaker_id": f"s{spk}",
                "condition": c,
                "prompt_snr": tg32.SNR[c],
                "gpu": "Tesla T4",
                "has_g32_all": True,
                "case_seconds": 35.0,
                "t_prepare": 0.08,
                "t_eval": 0.30,
                "t_student": 0.15,
                "t_vocoder": 0.01,
                "audio_seconds": 4.8,
                "peak_mem_mb": 5000.0,
                "utc": "",
            }
            for k in swsr.ALL_G16_STEPS:
                row[f"d16_{k}"] = 0.08 if noisy else 0.55
                bad = (k <= 3) or (noisy and k <= 6)
                row[f"wer_g16_{k}"] = 0.15 if bad else 0.02
                row[f"similarity_g16_{k}"] = 0.60 if bad else 0.72
                row[f"utmos_g16_{k}"] = 3.6
            for i in swsr.ALL_G32_STEPS:
                row[f"d32_{i}"] = 0.04 if noisy else 0.28
                bad = (i <= 6) or (noisy and i <= 13)
                row[f"wer_g32_{i}"] = 0.15 if bad else 0.02
                row[f"similarity_g32_{i}"] = 0.60 if bad else 0.72
                row[f"utmos_g32_{i}"] = 3.6
            rows.append(row)
    return rows


class SWSRFormulaTests(unittest.TestCase):
    def test_sway_step_sizes_sum_to_sway_times(self):
        for k in range(1, 17):
            self.assertAlmostEqual(sum(swsr.DELTA_T16[j] for j in range(1, k + 1)), swsr.GRID16[k])
        for k in range(1, 33):
            self.assertAlmostEqual(sum(swsr.DELTA_T32[j] for j in range(1, k + 1)), swsr.GRID32[k])

    def test_step_checkpoints_around_025_on_both_grids(self):
        self.assertLessEqual(swsr.GRID16[7], 0.25)
        self.assertGreater(swsr.GRID16[8], 0.25)
        self.assertEqual(g32.last_at_or_below(swsr.GRID16, 0.25), 7)
        self.assertEqual(g32.first_above(swsr.GRID16, 0.25), 8)
        self.assertLessEqual(swsr.GRID32[14], 0.25)
        self.assertGreater(swsr.GRID32[15], 0.25)
        self.assertEqual(g32.last_at_or_below(swsr.GRID32, 0.25), 14)
        self.assertEqual(g32.first_above(swsr.GRID32, 0.25), 15)

    def test_readiness_scores_match_definitions_on_both_grids(self):
        r = {
            "snr": 30.0,
            "d16": {k: 0.50 for k in swsr.ALL_G16_STEPS},
            "d32": {i: 0.25 for i in swsr.ALL_G32_STEPS},
        }
        self.assertAlmostEqual(
            swsr.readiness_score(r, "g16_all8", "g16_3", "swsr"),
            30.0 * (swsr.GRID16[3] - swsr.GRID16[2]) * 0.50,
        )
        self.assertAlmostEqual(
            swsr.readiness_score(r, "g32_all14", "g32_7", "swsr"),
            30.0 * (swsr.GRID32[7] - swsr.GRID32[6]) * 0.25,
        )

    @unittest.skipUnless(HAVE_SCI, "scipy and numpy are needed")
    def test_direct_wer_sim_comparison_without_failure_rule(self):
        rows = tg32.make_rows(10)
        s18_rows = make_steps1to8_rows(10)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            csv_path = out / "grid32.csv"
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=g32.FIELDS)
                w.writeheader()
                w.writerows(rows)
            s18_path = out / "steps1to8.csv"
            with open(s18_path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=swsr.STEPS_CSV_FIELDS)
                w.writeheader()
                w.writerows(s18_rows)
            res = swsr.analyze(out, exp017_csv=csv_path, all1000_csv=out / "missing.csv", steps1to8_csv=s18_path)
            report = (out / "SWSR_HANDOVER_REPORT.md").read_text(encoding="utf-8")

        for G in ("g16", "g32", "g16_all8", "g16_le025", "g32_all14", "g32_all15"):
            self.assertIn(G, res["exp017"])
            m = res["exp017"][G]["methods"]
            ref_cost = swsr.COST_ALL[swsr.GRIDS_EXT[G]["ref"]]
            for variant in ("stac", "crc_stac", "swsr", "stsr", "csdr"):
                self.assertLessEqual(m[variant]["d_wer_all"], swsr.WER_CAL_TOL)
                self.assertGreaterEqual(m[variant]["d_sim_all"], -swsr.SIM_CAL_TOL)
            for variant in ("swsr", "stsr", "csdr"):
                self.assertLess(m[variant]["mean_cost"], ref_cost)
            if G in ("g16_all8", "g16_le025", "g32_all14", "g32_all15"):
                self.assertLess(m["stac"]["mean_cost"], ref_cost)
        self.assertIn("Direct Quality & Efficiency Comparison (No Binary Failure Rule)", report)


if __name__ == "__main__":
    unittest.main()
