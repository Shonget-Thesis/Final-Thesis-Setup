import csv
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code/colab"))
import run_handover_operators_prelim as op  # noqa: E402

HAVE_NP = importlib.util.find_spec("numpy") is not None
HAVE_SCI = importlib.util.find_spec("scipy") is not None


@unittest.skipUnless(HAVE_NP, "numpy is needed")
class OperatorTests(unittest.TestCase):
    def test_teacher_state_is_recovered_from_the_endpoints(self):
        import numpy as np
        rng = np.random.default_rng(1)
        grid = [1 - np.cos(np.pi / 2 * i / 16) for i in range(17)]
        y0 = rng.normal(size=(1, 20, 5))
        x, E, states = y0, {}, {0: y0}
        for i in range(5):                                   # Euler pass with arbitrary velocities
            v = rng.normal(size=x.shape)
            E[i] = x + (1 - grid[i]) * v
            x = x + (grid[i + 1] - grid[i]) * v
            states[i + 1] = x
        np.testing.assert_allclose(op.teacher_state(E, y0, grid, 4), states[4], atol=1e-10)

    def test_cdeh_applies_only_to_coherent_trajectories(self):
        import numpy as np
        rng = np.random.default_rng(2)
        A, B = rng.normal(size=(1, 30, 4)), rng.normal(size=(1, 30, 4))
        coherent = {i: A + B * 0.8 ** i for i in range(9)}
        T, c, applied = op.cdeh(coherent, 10)
        self.assertTrue(applied)
        self.assertAlmostEqual(c, 1.0, places=9)
        osc = {i: A + B * (-0.8) ** i for i in range(9)}
        T, c, applied = op.cdeh(osc, 10)
        self.assertFalse(applied)
        np.testing.assert_array_equal(T, osc[4])

    def test_sea_moves_generated_band_means_halfway_and_keeps_the_prompt(self):
        import numpy as np
        E4 = np.zeros((1, 8, 3))
        E4[0, :4, :] = [[2, 0, 0], [2, 0, 0], [-9, -9, -9], [-9, -9, -9]]   # prompt: 2 loud frames, 2 noise frames
        E4[0, 4:, :] = [[1, 1, 1], [1, 1, 1], [-5, -5, -5], [-5, -5, -5]]
        T, rms = op.sea(E4, 4)
        np.testing.assert_array_equal(T[0, :4], E4[0, :4])
        np.testing.assert_allclose(T[0, 4], [1.5, 0.5, 0.5])          # loud means: prompt (2,0,0), generated (1,1,1)
        self.assertGreater(rms, 0)


@unittest.skipUnless(HAVE_SCI, "scipy is needed")
class AnalyzeTests(unittest.TestCase):
    def test_analyze_on_committed_labels_with_synthetic_outcomes(self):
        stored = op.ex.stored_cases()
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            op.write_prereg(out, type("A", (), {"dmo_root": tmp}))
            with open(out / "ops.csv", "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=op.FIELDS)
                w.writeheader()
                for (u, c), s in stored.items():
                    f4 = op.rf.is_true(s[4]["is_failure"])
                    row = {k: 0.5 for k in op.FIELDS}
                    row.update({"utt_id": u, "speaker_id": s[8]["speaker_id"], "condition": c, "prompt_snr": s[8]["prompt_snr"],
                                "d_4": s[8]["d_4"], "d4_stored": s[8]["d_4"], "utc": "",
                                **{f"hypothesis_{o}": "" for o in op.OPS},
                                "is_failure_tcr": f4 and c == "clean", "is_failure_cdeh": f4, "is_failure_sea": True})
                    w.writerow(row)
            res = op.analyze(out)
            report = (out / "HANDOVER_OPERATORS_REPORT.md").read_text(encoding="utf-8")
        self.assertEqual(res["cases"], 360)
        self.assertEqual(res["drift_reproduction_max_abs_diff"], 0.0)
        self.assertTrue(res["criteria"]["tcr"]["C1"] and res["criteria"]["tcr"]["C2"])
        self.assertFalse(res["criteria"]["cdeh"]["C1"])
        self.assertFalse(res["criteria"]["sea"]["C2"])
        self.assertEqual(res["cross_validation"]["pooled"]["tcr_all"]["mean_nfe"], 13.0)
        self.assertIn("Teacher-Coupled Restart", report)


if __name__ == "__main__":
    unittest.main()
