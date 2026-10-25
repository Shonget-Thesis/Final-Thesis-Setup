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
import run_followup_tests as ft  # noqa: E402

HAVE_SCI = all(importlib.util.find_spec(m) for m in ("numpy", "scipy"))


def write_meta(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join("|".join(r) for r in rows) + "\n", encoding="utf-8")


class SelectionTests(unittest.TestCase):
    def test_parse_meta_and_select_one_per_unused_speaker(self):
        with tempfile.TemporaryDirectory() as tmp:
            meta = Path(tmp) / "en/meta.lst"
            write_meta(meta, [("spkA-u1", "p", "prompt-wavs/spkA.wav", "t1"), ("spkA-u2", "p", "prompt-wavs/spkA.wav", "t2"),
                              ("spkB-u1", "p", "prompt-wavs/spkB.wav", "t3"), ("used-u1", "p", "prompt-wavs/used.wav", "t4")])
            entries = ft.parse_meta_lst(meta)
            fresh = ft.select_fresh(entries, {"used"})
        self.assertEqual([e["speaker_id"] for e in fresh], ["spkA", "spkB"])
        self.assertEqual(fresh, ft.select_fresh(list(reversed(entries)), {"used"}))  # order-independent
        self.assertEqual(entries[0]["gt_wav_rel"], "wavs/spkA-u1.wav")

    def test_fresh_speakers_are_outside_every_split(self):
        used = ft.used_speakers()
        self.assertEqual(len(used), 500)

    def test_input_seed_matches_committed_manifest(self):
        row = json.loads((ROOT / "research/protocols/splits/test_manifest.json").read_text(encoding="utf-8"))[0]
        self.assertEqual(ft.derive_input_seed(42, row["utt_id"]), row["input_seed"])

    def test_split_items_use_relative_dataset_paths(self):
        items = ft.split_items("test")
        self.assertEqual(len(items), 200)
        self.assertTrue(items[0]["prompt_wav_rel"].startswith("prompt-wavs/"))


class PolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        split_of = {}
        for name in ("calibration", "test"):
            for r in json.loads((ft.SPLITS_DIR / f"{name}_manifest.json").read_text(encoding="utf-8")):
                split_of[r["utt_id"]] = name
        cls.grid = ft.index_cases(ft.read_csv(ft.T4_GRID))
        cls.calib = sorted({u for (u, _) in cls.grid if split_of.get(u) == "calibration"})
        cls.test = sorted({u for (u, _) in cls.grid if split_of.get(u) == "test"})

    def test_ltt_on_t4_calibration_matches_audit(self):
        lam, table = ft.ltt(self.grid, self.calib, ft.GRID_CONDITIONS)
        self.assertEqual(lam, 1.2)
        self.assertEqual(len(table), 21)
        self.assertEqual(next(t for t in table if t["lambda"] == 0.5)["failures"], 2)

    def test_test_split_outcomes_match_audit(self):
        n, k, _, nfe = ft.evaluate(self.grid, self.test, ft.GRID_CONDITIONS, "snr_only", 0)
        self.assertEqual((n, k, round(nfe, 2)), (200, 4, 17.04))
        n, k, _, nfe = ft.evaluate(self.grid, self.test, ft.GRID_CONDITIONS, "snr_drift_050", 0)
        self.assertEqual((k, round(nfe, 2)), (3, 18.15))
        self.assertEqual(ft.evaluate(self.grid, self.test, ft.GRID_CONDITIONS, "reference", 0)[1], 0)

    def test_binomial_and_holm(self):
        self.assertAlmostEqual(ft.binom_cdf(1, 100, 0.05), 0.0371, places=4)
        self.assertEqual(ft.holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])


class ResumeTests(unittest.TestCase):
    def test_partial_line_and_partial_case_are_rerun(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "grid.csv"
            full = [{"utt_id": "u1", "condition": "clean", "teacher_step": k} for k in ft.STEPS]
            ft.append_rows(path, full)
            ft.append_rows(path, [{"utt_id": "u2", "condition": "clean", "teacher_step": 4}])
            with open(path, "a", encoding="utf-8") as f:
                f.write("test,u2,spk,clean,42,1,5")      # disconnect mid-line
            self.assertEqual(ft.done_keys(path, len(ft.STEPS)), {("u1", "clean")})
            self.assertTrue(path.read_bytes().endswith(b"\n"))
            self.assertEqual(len(ft.read_csv(path)), 5)


class PreregistrationTests(unittest.TestCase):
    def test_freeze_once_and_detect_edits(self):
        with tempfile.TemporaryDirectory() as tmp:
            dmo, out = Path(tmp) / "dmo", Path(tmp) / "out"
            write_meta(dmo / "data/seed-tts-eval/en/meta.lst", [("freshA-u1", "p", "prompt-wavs/freshA.wav", "t")])
            args = type("A", (), {"out": str(out), "dmo_root": str(dmo)})
            ft.cmd_freeze(args)
            prereg = ft.load_prereg(out)
            self.assertEqual(prereg["n_fresh_speakers"], 1)
            with self.assertRaises(SystemExit):
                ft.cmd_freeze(args)
            prereg["theta_db"] = 18.0
            (out / "preregistration.json").write_text(json.dumps(prereg), encoding="utf-8")
            with self.assertRaises(SystemExit):
                ft.load_prereg(out)


@unittest.skipUnless(HAVE_SCI, "numpy and scipy are needed")
class DegradationTests(unittest.TestCase):
    def setUp(self):
        import numpy as np
        self.np = np
        t = np.arange(24000) / 24000.0
        self.x = (0.1 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)

    def snr(self, clean, noisy):
        np = self.np
        return 10 * np.log10(np.mean(clean.astype(float) ** 2) / np.mean((noisy - clean).astype(float) ** 2))

    def test_gaussian_and_babble_hit_target_snr(self):
        np = self.np
        self.assertAlmostEqual(self.snr(self.x, ft.add_gaussian(self.x, 18.0, 3)), 18.0, places=1)
        track = np.random.RandomState(1).randn(60000).astype(np.float32)
        self.assertAlmostEqual(self.snr(self.x, ft.add_babble(self.x, track, 15.0, 3)), 15.0, places=1)
        self.assertTrue(np.array_equal(ft.add_gaussian(self.x, 18.0, 3), ft.add_gaussian(self.x, 18.0, 3)))

    def test_reverb_keeps_length_and_loudness(self):
        y = ft.add_reverb(self.x, 24000, 0.9, 5)
        self.assertEqual(len(y), len(self.x))
        self.assertAlmostEqual(ft._rms(y), ft._rms(self.x), places=3)
        self.assertGreater(float(self.np.max(self.np.abs(y - self.x))), 0.01)

    def test_telephone_removes_high_band(self):
        np = self.np
        t = np.arange(24000) / 24000.0
        hi = (0.1 * np.sin(2 * np.pi * 6000 * t)).astype(np.float32)
        mixed = self.x * 0 + (0.1 * np.sin(2 * np.pi * 1000 * t)).astype(np.float32) + hi
        y = ft.telephone_band(mixed, 24000)
        spec = np.abs(np.fft.rfft(y))
        self.assertLess(spec[6000], 0.05 * spec[1000])
        self.assertEqual(len(y), len(mixed))

    def test_clopper_pearson_and_mcnemar(self):
        lo, hi = ft.clopper_pearson(0, 45)
        self.assertEqual(lo, 0.0)
        self.assertAlmostEqual(hi, 1 - 0.025 ** (1 / 45), places=6)
        self.assertEqual(ft.mcnemar_exact([1, 1, 0, 0], [0, 0, 0, 1]), (2, 1, 1.0))


@unittest.skipUnless(HAVE_SCI, "numpy and scipy are needed")
class PilotTests(unittest.TestCase):
    def test_pilot_uses_development_speakers_and_reports_reproduction(self):
        dev = ft.pilot_items(3)
        dev_ids = {r["utt_id"] for r in json.loads((ft.SPLITS_DIR / "development_manifest.json").read_text(encoding="utf-8"))}
        self.assertTrue(all(it["utt_id"] in dev_ids for it in dev))
        grid = ft.index_cases(ft.read_csv(ft.T4_GRID))
        with tempfile.TemporaryDirectory() as tmp:
            pdir = Path(tmp) / "pilot"
            pdir.mkdir()
            rows, t = [], 0
            for it in dev:
                for c in ("clean", "noisy15", "reverb09"):
                    src = grid[(it["utt_id"], "clean" if c == "reverb09" else c)]
                    t += 30
                    for k in ft.STEPS:
                        r = {key: src[k].get(key, "") for key in ft.ROW_FIELDS}
                        r.update(condition=c, gt_sr=24000, utc=f"2026-10-26T00:{t // 60:02d}:{t % 60:02d}+00:00")
                        rows.append(r)
            ft.append_rows(pdir / "pilot_grid.csv", rows)
            ft.append_rows(pdir / "pilot_seed_floor.csv", [{"test": "seed_floor_B", "utt_id": dev[0]["utt_id"], "condition": "clean", "is_failure": False}])
            res = ft.pilot_report(tmp)
        rep = res["reproduction_vs_t4_grid"]
        self.assertEqual(rep["cases"], 6)
        self.assertEqual(rep["same_failure_label"], rep["rows"])
        self.assertEqual(rep["max_abs_diff_similarity"], 0.0)
        self.assertEqual(res["conditions"]["reverb09"]["n"], 3)
        self.assertEqual(res["seconds_per_case_median"], 30.0)
        self.assertEqual(res["seed_floor"], {"speakers": 1, "failures": 0})


@unittest.skipUnless(HAVE_SCI, "numpy and scipy are needed")
class AnalyzeTests(unittest.TestCase):
    def test_analyze_computes_report_from_csv(self):
        grid = ft.index_cases(ft.read_csv(ft.T4_GRID))
        test = ft.split_items("test")[:5]
        with tempfile.TemporaryDirectory() as tmp:
            dmo, out = Path(tmp) / "dmo", Path(tmp) / "out"
            write_meta(dmo / "data/seed-tts-eval/en/meta.lst",
                       [(f"fresh{i}-u", "p", f"prompt-wavs/fresh{i}.wav", "t") for i in range(5)])
            ft.cmd_freeze(type("A", (), {"out": str(out), "dmo_root": str(dmo)}))
            fresh = ft.fresh_items(out)
            rows = []
            for f, src in zip(fresh, test):
                for c in ft.GRID_CONDITIONS:
                    for k in ft.STEPS:
                        r = dict(grid[(src["utt_id"], c)][k])
                        r["utt_id"], r["speaker_id"] = f["utt_id"], f["speaker_id"]
                        rows.append({key: r.get(key, "") for key in ft.ROW_FIELDS})
            with open(out / "fresh_grid.csv", "w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=ft.ROW_FIELDS)
                w.writeheader()
                w.writerows(rows)
            res = ft.analyze(out)
            expected = ft.evaluate(grid, [t["utt_id"] for t in test], ft.GRID_CONDITIONS, "snr_only", 0)
            self.assertEqual(res["H1_heldout"]["snr_only"]["failures"], expected[1])
            self.assertEqual(res["H2_drift_value"], "not run")
            self.assertIn("## H1.", (out / "FOLLOWUP_REPORT.md").read_text(encoding="utf-8"))
            self.assertTrue(math.isclose(res["H1_heldout"]["reference"]["mean_nfe"], 21.0))


if __name__ == "__main__":
    unittest.main()
