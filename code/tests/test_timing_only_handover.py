"""Unit tests for EXP-030 (Timing-Only Handover pilot). They run without PyTorch or the TTS models."""
import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code/colab"))
import run_damped_transmissibility_stac as dts  # noqa: E402
import run_timing_only_handover as toh  # noqa: E402


def smooth_signal(rng, frames, groups=10, pad=40):
    base = np.cumsum(rng.normal(size=(frames + 2 * pad, groups)), axis=0)
    base = np.stack([np.convolve(base[:, j], np.ones(9) / 9, mode="same") for j in range(groups)], axis=1)
    return base[pad:pad + frames]


def mel_from_groups(groups_signal, bands=100):
    return np.repeat(groups_signal, bands // groups_signal.shape[1], axis=1)


class TimingMapTests(unittest.TestCase):
    def test_features_ignore_stationary_offsets_and_scales(self):
        rng = np.random.default_rng(1)
        mel = rng.normal(size=(120, 100))
        shifted = 2.5 * mel + np.linspace(-4.0, 3.0, 100)[None, :]
        np.testing.assert_allclose(toh.timing_features(mel), toh.timing_features(shifted), atol=1e-9)
        f = toh.timing_features(mel)
        self.assertEqual(f.shape, (120, toh.N_GROUPS))
        np.testing.assert_allclose(f.mean(axis=0), 0.0, atol=1e-9)
        np.testing.assert_allclose(f.std(axis=0), 1.0, atol=1e-9)

    def test_identical_inputs_give_the_identity_map(self):
        rng = np.random.default_rng(2)
        mel = mel_from_groups(smooth_signal(rng, 250))
        w, band = toh.timing_map(mel, mel)
        np.testing.assert_array_equal(w, np.arange(250, dtype=float))
        self.assertEqual(band, toh.band_width(250))

    def test_map_is_monotone_slope_limited_and_inside_the_band(self):
        rng = np.random.default_rng(3)
        for n in (40, 180, 600):
            w, band = toh.timing_map(rng.normal(size=(n, 100)), rng.normal(size=(n, 100)))
            self.assert_smoke_map_check(w, band, n)

    def assert_smoke_map_check(self, w, band, n):
        """The smoke test's map check (run_timing_only_handover.smoke)."""
        d = np.diff(w)
        self.assertEqual(w[0], 0.0)
        self.assertTrue(n - 1.5 <= w[-1] <= n - 1)
        self.assertTrue(np.all(d >= -1e-9))
        self.assertTrue(np.all(d <= 2.0 + 1e-9), f"largest step {d.max()}")
        self.assertLessEqual(float(np.max(np.abs(w - np.arange(n)))), band + 1.0)

    def test_structured_retimings_pass_the_smoke_map_check(self):
        """Regression for Amendment 1: pinning the last frame gave a final step of 2.5 on retimed, speech-like
        inputs (18 of 400 in a local search), which failed the first Colab smoke test."""
        rng = np.random.default_rng(0)
        for _ in range(60):
            n = int(rng.integers(150, 900))
            base = np.cumsum(rng.normal(size=(n + 80, 10)), 0)[40:40 + n]
            student = np.repeat(base, 10, 1)
            shift = rng.uniform(2, 15) * np.sin(np.pi * np.arange(n) / (n - 1) * rng.integers(1, 4))
            teacher = toh.warp_frames(student, np.arange(n) + shift) + 0.3 * rng.normal(size=student.shape)
            w, band = toh.timing_map(student, teacher)
            self.assert_smoke_map_check(w, band, n)

    def test_recovers_a_known_retiming_despite_a_recording_condition(self):
        rng = np.random.default_rng(4)
        n = 300
        student = mel_from_groups(smooth_signal(rng, n)) + 0.05 * rng.normal(size=(n, 100))
        truth = np.arange(n) + 8.0 * np.sin(np.pi * np.arange(n) / (n - 1))
        teacher = toh.warp_frames(student, truth) + np.linspace(-2.0, 3.0, 100)[None, :]
        w, _ = toh.timing_map(student, teacher)
        self.assertLess(float(np.sqrt(np.mean((w - truth) ** 2))), 1.0)

    def test_condition_cannot_cross(self):
        """A stationary condition added to the teacher leaves the payload unchanged, and the payload only
        re-times the student's own frames."""
        rng = np.random.default_rng(5)
        plen, n = 7, 160
        s0 = rng.normal(size=(1, plen + n, 100))
        s0[0, plen:, :] = mel_from_groups(smooth_signal(rng, n))
        t_k = rng.normal(size=(1, plen + n, 100))
        t_k[0, plen:, :] = toh.warp_frames(s0[0, plen:, :], np.arange(n) + 5.0 * np.sin(np.pi * np.arange(n) / (n - 1)))
        noisy = t_k.copy()
        noisy[0, plen:, :] = 1.7 * noisy[0, plen:, :] + np.linspace(1.0, 6.0, 100)[None, :]
        a, w_a, _ = toh.timing_handover(s0, t_k, plen)
        b, w_b, _ = toh.timing_handover(s0, noisy, plen)
        np.testing.assert_allclose(w_a, w_b, atol=1e-9)
        np.testing.assert_allclose(a, b, atol=1e-12)
        np.testing.assert_array_equal(a[:, :plen, :], s0[:, :plen, :])
        gen, src = a[0, plen:, :], s0[0, plen:, :]
        self.assertTrue(np.all(gen >= src.min(axis=0) - 1e-12) and np.all(gen <= src.max(axis=0) + 1e-12))

    def test_identity_map_is_exact_and_mirror_keeps_statistics(self):
        rng = np.random.default_rng(6)
        plen, n = 5, 90
        x = rng.normal(size=(1, plen + n, 100))
        np.testing.assert_array_equal(toh.apply_timing_map(x, np.arange(n, dtype=float), plen), x)
        w, _ = toh.timing_map(rng.normal(size=(n, 100)), rng.normal(size=(n, 100)))
        m = toh.mirror_map(w)
        self.assertEqual(m[0], 0.0)
        self.assertEqual(m[-1], w[-1])
        self.assertTrue(np.all(np.diff(m) >= -1e-12))
        self.assertAlmostEqual(toh.map_stats(m)[0], toh.map_stats(w)[0], places=9)
        self.assertAlmostEqual(toh.map_stats(m)[1], toh.map_stats(w)[1], places=9)
        np.testing.assert_allclose(toh.mirror_map(m), w, atol=1e-12)

    def test_dtw_rejects_impossible_lengths(self):
        with self.assertRaises(ValueError):
            toh.constrained_dtw(np.zeros((10, 30)), band=50)


class DesignTests(unittest.TestCase):
    def test_costs_and_cohorts(self):
        self.assertEqual(toh.COST, {"STU": 4, "W2": 9, "W3": 11, "W4": 13, "TOH2": 10, "TOH3": 12, "MIR3": 12, "W8": 21})
        for k in (2, 3, 4, 8):
            self.assertEqual(toh.COST[f"W{k}"], 2 * k + 5)
        for k in (2, 3):
            self.assertEqual(toh.COST[f"TOH{k}"], 2 * k + 6)
        self.assertEqual([it["utt_id"] for it in toh.qual_items()], [it["utt_id"] for it in dts.qual_items()])
        self.assertEqual([it["utt_id"] for it in toh.div_items()], [it["utt_id"] for it in dts.div_items()])
        cal = json.loads((ROOT / "research/protocols/splits/calibration_manifest.json").read_text(encoding="utf-8"))
        test = json.loads((ROOT / "research/protocols/splits/test_manifest.json").read_text(encoding="utf-8"))
        spk = {it["speaker_id"] for it in toh.qual_items()}
        self.assertFalse(spk & {r["speaker_id"] for r in cal})
        self.assertFalse(spk & {r["speaker_id"] for r in test})


def write_synthetic(tmp, profile, with_div=True):
    """Quality and diversity CSVs where TOH3 behaves as `profile` says."""
    rng = np.random.default_rng(11)
    q = Path(tmp) / "quality.csv"
    with open(q, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=toh.QUAL_FIELDS)
        w.writeheader()
        for it in toh.qual_items():
            for cond in toh.CONDS:
                clean = cond == "clean"
                row = {"utt_id": it["utt_id"], "speaker_id": it["speaker_id"], "condition": cond, "prompt_snr": 35.0,
                       "gpu": "UnitTest", "frames": 400, "warp_rms_2": 3.0, "warp_max_2": 8.0, "warp_rms_3": 4.0,
                       "warp_max_3": 9.0, "warp_band": 40, "case_seconds": 1.0, "utc": "2026-11-15T00:00:00+00:00"}
                for arm in toh.ARMS:
                    utm = 3.9 if clean else (3.3 if arm in ("STU", "TOH2", "TOH3", "MIR3") else 1.9)
                    if arm == "TOH3" and not clean and profile == "noise_crosses":
                        utm = 1.9
                    if arm == "TOH3" and clean and profile == "clean_loss":
                        utm = 3.6
                    row.update({f"wer_{arm}": 0.0, f"similarity_{arm}": 0.93,
                                f"utmos_{arm}": utm + float(rng.normal(0.0, 0.02)), f"hypothesis_{arm}": "x"})
                w.writerow(row)
    if with_div:
        d = Path(tmp) / "diversity.csv"
        with open(d, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=toh.DIV_FIELDS)
            w.writeheader()
            for it in toh.div_items():
                row = {"utt_id": it["utt_id"], "speaker_id": it["speaker_id"], "condition": "clean", "n_seeds": 12,
                       "warp_rms_2_mean": 3.0, "warp_rms_3_mean": 4.0, "case_seconds": 1.0, "utc": "x"}
                for arm in toh.ARMS:
                    cv = 0.05 if arm in ("STU", "MIR3") else 0.09
                    if arm == "TOH3" and profile == "no_variety":
                        cv = 0.05
                    row.update({f"cvf0_{arm}": cv + float(rng.normal(0.0, 0.001)), f"voiced_{arm}": 0.6,
                                f"mel_spread_{arm}": 0.3})
                w.writerow(row)


class AnalysisTests(unittest.TestCase):
    def _run(self, profile, with_div=True, allow_partial=False):
        with tempfile.TemporaryDirectory() as tmp:
            prereg = Path(tmp) / "preregistration.json"
            prereg.write_text(json.dumps(toh.build_prereg(), indent=2) + "\n", encoding="utf-8")
            write_synthetic(tmp, profile, with_div)
            res = toh.analyze(tmp, prereg_path=prereg, allow_partial=allow_partial)
            self.assertTrue((Path(tmp) / "TIMING_ONLY_HANDOVER_REPORT.md").exists())
            self.assertTrue((Path(tmp) / "MANIFEST.json").exists())
            return res

    def test_verdicts(self):
        self.assertEqual(self._run("works")["verdict"], "supported")
        self.assertEqual(self._run("no_variety")["verdict"], "timing_does_not_carry_variety")
        self.assertEqual(self._run("noise_crosses")["verdict"], "not_supported")
        self.assertEqual(self._run("clean_loss")["verdict"], "not_supported")
        self.assertEqual(self._run("works", with_div=False, allow_partial=True)["verdict"], "partial")
        with self.assertRaises(RuntimeError):
            self._run("works", with_div=False, allow_partial=False)

    def test_secondary_arm_does_not_change_the_verdict(self):
        res = self._run("no_variety")
        self.assertEqual(res["secondary_verdict_TOH2"], "supported")
        self.assertEqual(res["verdict"], "timing_does_not_carry_variety")


class PreregistrationAndNotebookTests(unittest.TestCase):
    def test_tampering_is_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "preregistration.json"
            doc = toh.build_prereg()
            p.write_text(json.dumps(doc), encoding="utf-8")
            toh.check_prereg(p)
            doc["bars"]["clean_utmos"] = -0.5
            p.write_text(json.dumps(doc), encoding="utf-8")
            with self.assertRaises(RuntimeError):
                toh.check_prereg(p)

    def test_committed_preregistration_and_notebook(self):
        if toh.PREREG_PATH.exists():
            doc = toh.check_prereg()
            self.assertEqual(doc["experiment"], "EXP-030")
            self.assertEqual(doc["task_id"], "EVAL-023")
        nb_path = ROOT / "code/colab/DMOSpeech2_EXP030_Timing_Only_Handover.ipynb"
        nb = json.loads(nb_path.read_text(encoding="utf-8"))
        code = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
        for i, src in enumerate(code):
            compile("\n".join(l for l in src.splitlines() if not l.lstrip().startswith(("!", "%"))), f"cell{i}", "exec")
        joined = "\n".join(code)
        self.assertIn("run_timing_only_handover.py", joined)
        self.assertIn("EXP-030-timing-only-handover", joined)
        self.assertNotIn("EXP-029", joined)
        self.assertIn('STAGE = "all"', joined)


if __name__ == "__main__":
    unittest.main()
