import csv
import importlib.util
import random
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code/colab"))
import run_consistency_gate_prelim as cg  # noqa: E402

HAVE_SCI = importlib.util.find_spec("scipy") is not None


class PolicyTests(unittest.TestCase):
    def rec(self, snr, d4, g4, cond="clean"):
        return {"snr": snr, "cond": cond, "d": {k: d4 for k in range(1, 7)}, "g": {k: g4 for k in range(0, 7)}}

    def test_gates_fall_back_as_defined(self):
        model = {"b0": 0.0, "b1": 0.0, "orient": 1, "g_scale": 1.0, "guid_q": None}
        self.assertEqual(cg.make_policy("snr_only", None, model)(self.rec(15.0, 0.6, 1.0)), 8)
        self.assertEqual(cg.make_policy("low_drift_veto", 0.30, model)(self.rec(25.0, 0.2, 1.0)), 8)
        self.assertEqual(cg.make_policy("low_drift_veto", 0.30, model)(self.rec(25.0, 0.6, 1.0)), 4)
        # SDC: residual log(0.2) - 0 < -kappa for kappa 1.0 -> fall back; d_4 = 0.6 -> keep
        self.assertEqual(cg.make_policy("sdc_gate", 1.0, model)(self.rec(25.0, 0.2, 1.0)), 8)
        self.assertEqual(cg.make_policy("sdc_gate", 1.0, model)(self.rec(25.0, 0.6, 1.0)), 4)
        self.assertEqual(cg.make_policy("guidance_gate", 0.5, model)(self.rec(25.0, 0.6, 0.4)), 8)
        self.assertEqual(cg.make_policy("sdc_guidance", 1.0, model)(self.rec(25.0, 0.6, 0.1)), 4)   # guid_q None never fires

    def test_sdc_fit_recovers_a_line(self):
        train = [{"cond": "clean", "snr": s, "d": {4: 2.718281828 ** (0.5 + 0.1 * s)}} for s in (20, 30, 40)]
        b0, b1 = cg.fit_sdc(train)
        self.assertAlmostEqual(b0, 0.5, places=6)
        self.assertAlmostEqual(b1, 0.1, places=6)

    def test_stratified_ltt_needs_every_condition(self):
        # Two conditions, 80 speakers each; one condition has 3 failures at Step 4 -> stratified rejects, pooled certifies.
        train = []
        for c, nfail in (("clean", 0), ("gauss20", 3)):
            for i in range(80):
                train.append({"cond": c, "snr": 30.0, "d": {k: 0.6 for k in range(1, 7)}, "g": {k: 1.0 for k in range(0, 7)},
                              "fail": {4: i < nfail, 5: False, 6: False, 8: False}})
        model = {"b0": 0.0, "b1": 0.0, "orient": 1, "g_scale": 1.0, "guid_q": None}
        self.assertIsNotNone(cg.ltt_select(train, "low_drift_veto", model, stratified=False))
        self.assertIsNone(cg.ltt_select(train, "low_drift_veto", model, stratified=True))


@unittest.skipUnless(HAVE_SCI, "scipy is needed")
class AnalyzeTests(unittest.TestCase):
    def test_analyze_on_committed_labels_with_synthetic_guidance(self):
        cases = cg.load_labels(Path(tempfile.gettempdir()) / "no_such_dir")
        rng = random.Random(0)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            cg.write_prereg(out, type("A", (), {"dmo_root": tmp, "n_speakers": 100, "conditions": list(cg.CONDS)}))
            with open(out / "signals.csv", "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=cg.SIG_FIELDS)
                w.writeheader()
                for (u, c), steps in cases.items():
                    r8 = steps[8]
                    row = {"utt_id": u, "speaker_id": r8["speaker_id"], "condition": c, "prompt_snr": r8["prompt_snr"], "utc": ""}
                    row.update({f"d_{k}": r8[f"d_{k}"] if k in (4, 5, 6) else 1.0 for k in range(1, 7)})
                    row.update({f"g_{k}": rng.uniform(0.5, 1.5) for k in range(0, 7)})
                    w.writerow(row)
            res = cg.analyze(out, allow_partial=True)
            report = (out / "CONSISTENCY_GATE_REPORT.md").read_text(encoding="utf-8")
        self.assertEqual(res["drift_reproduction_max_abs_diff"], 0.0)
        pooled = res["out_of_fold"]["pooled"]
        self.assertEqual(pooled["reference"]["failures"], 0)
        # SNR-only has no parameter: its out-of-fold failures equal a direct count.
        table = cg.build_table(cases, {(u, c): {"prompt_snr": s[8]["prompt_snr"], **{f"d_{k}": s[8][f"d_{k}"] if k in (4, 5, 6) else 1.0 for k in range(1, 7)},
                                                **{f"g_{k}": 1.0 for k in range(0, 7)}} for (u, c), s in cases.items()})
        direct = sum(r["fail"][4] for r in table if r["snr"] >= cg.THETA)
        self.assertEqual(pooled["snr_only"]["failures"], direct)
        self.assertIn("## Out-of-fold results, stratified calibration", report)
        self.assertTrue(res["partial"])

    def test_quick_mode_restricts_speakers_and_conditions(self):
        cases = cg.load_labels(Path(tempfile.gettempdir()) / "no_such_dir")
        try:
            with tempfile.TemporaryDirectory() as tmp:
                out = Path(tmp)
                quick = list(cg.QUICK["conditions"])
                cg.write_prereg(out, type("A", (), {"dmo_root": tmp, "n_speakers": 60, "conditions": quick}))
                with self.assertRaises(SystemExit):        # a restart with another size is refused
                    cg.write_prereg(out, type("A", (), {"dmo_root": tmp, "n_speakers": 100, "conditions": quick}))
                with open(out / "signals.csv", "w", newline="", encoding="utf-8") as f:
                    w = csv.DictWriter(f, fieldnames=cg.SIG_FIELDS)
                    w.writeheader()
                    for (u, c), steps in cases.items():
                        r8 = steps[8]
                        row = {"utt_id": u, "speaker_id": r8["speaker_id"], "condition": c, "prompt_snr": r8["prompt_snr"], "utc": ""}
                        row.update({f"d_{k}": r8[f"d_{k}"] if k in (4, 5, 6) else 1.0 for k in range(1, 7)})
                        row.update({f"g_{k}": 1.0 for k in range(0, 7)})
                        w.writerow(row)
                res = cg.analyze(out, allow_partial=True)
        finally:
            cg.CONDS = tuple(cg.rf.CONDITIONS)              # restore the module default for other tests
        self.assertEqual(res["conditions"], quick)
        self.assertEqual(res["expected_cases"], 60 * 6)
        self.assertLessEqual(res["cases"], 360)


if __name__ == "__main__":
    unittest.main()
