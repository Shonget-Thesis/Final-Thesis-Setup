"""Guard evidence arithmetic, units, boundary rules, and matching integrity."""
import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
from replay_timing import CONFIG, EVIDENCE, passes, read, replay, run


class TimingReplayTests(unittest.TestCase):
    def test_matches_independently_recorded_audit(self):
        result = run()
        self.assertEqual(result["instances"], 48)
        expected = {"t=0.05": (7, 26, 27), "t=0.10": (9, 13, 14),
                    "t=0.15": (11, 5, 7), "t=0.25": (13, 0, 4), "t=0.40": (16, 10, 12)}
        for label, values in expected.items():
            row = result["policies"][label]
            self.assertEqual((row["recorded_mean_nfe"], row["relative_failures"], row["strict_failures"]), values)
        snr = result["policies"]["snr_only"]
        self.assertAlmostEqual(snr["recorded_mean_nfe"], 11.458333333333334)
        self.assertEqual(snr["relative_failures"], 0)
        self.assertEqual(snr["strict_failure_ids"], [23, 24, 32, 44])
        self.assertEqual(result["policies"]["calibrated_hybrid"]["exact_fixed_record_matches_wer_sim_latency"], 48)

    def test_absolute_and_relative_rules_differ(self):
        config = read(CONFIG)
        reference = {"wer": 20, "similarity": 0.8}
        boundary = {"wer": 21, "similarity": 0.8 - 0.02}
        self.assertTrue(passes(reference, boundary, config))
        self.assertFalse(passes(reference, boundary, config, strict=True))
        self.assertFalse(passes(reference, {**boundary, "wer": 21.01}, config))
        self.assertFalse(passes(reference, {**boundary, "similarity": 0.77}, config))

    def test_join_rejects_missing_or_duplicate_records(self):
        fixed = read(EVIDENCE / "fixed_handover_timing_proof.json")["instances"]
        adaptive = read(EVIDENCE / "calibrated_hybrid_scheduler_telemetry.json")["instances"]
        config = read(CONFIG)
        with self.assertRaises(ValueError):
            replay(fixed, adaptive[:-1], config)
        with self.assertRaises(ValueError):
            replay(fixed + [fixed[0]], adaptive, config)

    def test_snr_threshold_selects_high_branch_at_equality(self):
        fixed = [{"utt_id": "u", "condition": "c", "policies": {
            "t=0.25": {"nfe": 13, "wer": 0, "similarity": 1, "latency_sec": 2},
            "t=0.15": {"nfe": 11, "wer": 0, "similarity": 1, "latency_sec": 1}}}]
        adaptive = [{"utt_id": "u", "condition": "c", "instance_id": 1, "prompt_snr_db": 18,
                     "calibrated_hybrid": {"chosen_policy": "t=0.15", "effective_nfe": 11,
                                           "wer": 0, "similarity": 1, "latency_sec": 1}}]
        self.assertEqual(replay(fixed, adaptive, read(CONFIG))["policies"]["snr_only"]["recorded_mean_nfe"], 11)
        low = copy.deepcopy(adaptive)
        low[0]["prompt_snr_db"] = 17.999
        self.assertEqual(replay(fixed, low, read(CONFIG))["policies"]["snr_only"]["recorded_mean_nfe"], 13)


if __name__ == "__main__":
    unittest.main()
