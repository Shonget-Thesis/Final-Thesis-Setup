"""Unit tests for EXP-027 (transmissibility confirmation runner and notebook)."""
import ast
import csv
import json
import math
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "colab"))
import run_consistency_gate_prelim as cg  # noqa: E402
import run_transmissibility_confirm as tc  # noqa: E402
import run_transmissibility_test as tt  # noqa: E402


class DesignTests(unittest.TestCase):
    def test_fresh_speakers_disjoint_from_dev(self):
        dev60 = {it["utt_id"] for it in cg.dev_items()[:tc.DEV_N]}
        fresh = tc.fresh_items()
        fresh_ids = {it["utt_id"] for it in fresh}
        self.assertEqual(len(dev60), 60)
        self.assertEqual(len(fresh), 40)
        self.assertEqual(len(fresh_ids), 40)
        self.assertFalse(dev60 & fresh_ids)

    def test_arms_and_pass_costs(self):
        self.assertEqual(tc.ARMS, ("student4", "W4", "W8", "W4R2", "W8R2", "T16"))
        self.assertEqual(tc.cost("student4"), 4)
        for arm in ("W4", "W8", "W4R2", "W8R2", "T16"):
            self.assertEqual(tc.cost(arm), tt.cost(arm))


class ConfoundControlTests(unittest.TestCase):
    def test_common_cvf0_eliminates_frame_dropout_bias(self):
        # Frame 0..3: all voiced across arms, same tight CVf0 across arms.
        # Frame 4..7: dropped (NaN) in W8, voiced with huge variance ONLY in T16 -> unshared confound.
        tracks = {"W8": [], "T16": []}
        for s in range(6):
            base = np.full(8, 120.0 + (2.0 if s % 2 else -2.0))
            t_w8 = base.copy()
            t_w8[4:] = np.nan
            t_t16 = base.copy()
            t_t16[4:] = 100.0 + s * 30.0
            tracks["W8"].append(t_w8)
            tracks["T16"].append(t_t16)

        common, share = tc.common_cvf0(tracks)
        self.assertAlmostEqual(share, 0.5, places=6)
        # Restricted to common voiced frames 0..3, T16 and W8 must match exactly.
        self.assertAlmostEqual(common["W8"], common["T16"], places=6)

        # In contrast, T16's unshared CVf0 is inflated by frames 4..7.
        cv_t16, share_t16 = tc.sh.cvf0(tracks["T16"])
        self.assertGreater(cv_t16, common["T16"] * 2.0)
        self.assertAlmostEqual(share_t16, 1.0, places=6)

    def test_common_cvf0_returns_nan_when_no_shared_frames(self):
        t1 = [np.array([100.0, np.nan]), np.array([110.0, np.nan])]
        t2 = [np.array([np.nan, 100.0]), np.array([np.nan, 110.0])]
        common, share = tc.common_cvf0({"W8": t1, "T16": t2})
        self.assertEqual(share, 0.0)
        self.assertTrue(math.isnan(common["W8"]))
        self.assertTrue(math.isnan(common["T16"]))

    def test_slow_mel_spread_monotonic_in_dispersion(self):
        rng = np.random.default_rng(7)
        base = np.ones((16, 8), dtype=float)
        identical = [base.copy() for _ in range(4)]
        spread_small = [base + 0.05 * rng.standard_normal((16, 8)) for _ in range(4)]
        spread_large = [base + 0.40 * rng.standard_normal((16, 8)) for _ in range(4)]
        self.assertEqual(tc.slow_mel_spread(identical), 0.0)
        self.assertLess(tc.slow_mel_spread(spread_small), tc.slow_mel_spread(spread_large))


class _FakeSampler:
    STUDENT_TAUS = [0.0761, 0.2929, 0.6173]


class _FakePipe:
    gpu = "Fake-T4"
    sampler = _FakeSampler()

    def prepare(self, item, cond, seed, out_dir, noise_seed=None):
        rng = np.random.default_rng(int(seed))
        return {
            "seed": seed,
            "prompt_len": 4,
            "prompt_snr": 30.0,
            "y0": rng.standard_normal((1, 20, 8)),
            "step_cond": rng.standard_normal((1, 20, 8)),
        }

    def ground_truth(self, item):
        return item["gt_text"]

    def score(self, audio, item, gt):
        arm = audio[0]
        if isinstance(arm, str) and arm in ("W4", "W8"):
            s = tt.stored_w()[(item["utt_id"], "clean")][arm]
            return {"wer": s["wer"], "similarity": s["similarity"], "utmos": 3.9, "hypothesis": "ok"}
        return {"wer": 0.05, "similarity": 0.60, "utmos": 3.7, "hypothesis": "ok"}


def _fake_outputs(pipe, ctx, ends, final):
    rng = np.random.default_rng(int(ctx["seed"]))
    out = {}
    scales = {"student4": 0.02, "W4": 0.04, "W8": 0.05, "W4R2": 0.08, "W8R2": 0.14, "T16": 0.16}
    for arm, sc in scales.items():
        # Include a smooth time-varying slow trajectory scaled per arm across seeds.
        t = np.linspace(0.0, 2.0 * np.pi, 20)[None, :, None]
        mel = np.sin(t + rng.normal(0.0, sc)) * np.ones((1, 20, 8))
        out[arm] = mel
    return out


class LoopTests(unittest.TestCase):
    def test_outputs_passes_zero_tau_for_student4(self):
        pipe = _FakePipe()
        ctx = pipe.prepare({"utt_id": "u0"}, "clean", 42, ".")
        tt_ret = {"W4": "w4", "W8": "w8", "W4R2": "w4r2", "W8R2": "w8r2", "T16": "t16"}
        with mock.patch.object(tt, "outputs", return_value=tt_ret) as m_tt, \
                mock.patch.object(tc.gl, "student", return_value="s4") as m_st:
            got = tc.outputs(pipe, ctx, {}, None)
        m_tt.assert_called_once()
        m_st.assert_called_once()
        self.assertEqual(m_st.call_args.args[3], [0.0, 0.0761, 0.2929, 0.6173])
        self.assertEqual(set(got), set(tc.ARMS))

    def test_diversity_quality_and_smoke_loops(self):
        pipe = _FakePipe()
        items = tt.smoke_items()

        def fake_f0(audio):
            v = float(audio[0])
            return np.array([110.0 + 30.0 * v, 125.0 + 40.0 * v, 140.0 + 50.0 * v, 155.0 + 60.0 * v])

        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            with mock.patch.object(tt, "teacher_run", return_value=({}, None)), \
                    mock.patch.object(tc, "outputs", side_effect=_fake_outputs), \
                    mock.patch.object(tc.ng, "decode", side_effect=lambda _p, _c, m: m), \
                    mock.patch.object(tc.sh, "f0_track", side_effect=lambda m: fake_f0([float(m[0, -1, 0])])):

                div_csv = tc.run_diversity(pipe, items, out_dir, seeds=3)
                with open(div_csv, encoding="utf-8") as f:
                    rows = list(csv.DictReader(f))
                self.assertEqual(len(rows), 2)
                self.assertTrue(set(tc.DIV_FIELDS).issubset(rows[0].keys()))

                # Second call resumes without duplicating existing rows.
                tc.run_diversity(pipe, items, out_dir, seeds=3)
                with open(div_csv, encoding="utf-8") as f:
                    self.assertEqual(len(list(csv.DictReader(f))), 2)

                qual_csv = tc.run_quality(pipe, items, out_dir)
                with open(qual_csv, encoding="utf-8") as f:
                    qrows = list(csv.DictReader(f))
                self.assertEqual(len(qrows), 2)
                self.assertTrue(set(tc.QUAL_FIELDS).issubset(qrows[0].keys()))

                # Reproduction matches stored W4 and W8 -> smoke passes.
                with mock.patch.object(tc, "check_prereg"), \
                        mock.patch.object(tc, "outputs", side_effect=lambda _p, c, e, f: {a: a for a in tc.ARMS}), \
                        mock.patch.object(tc.ng, "decode", side_effect=lambda _p, _c, m: [m]), \
                        mock.patch.object(tc, "_slow_component", return_value=np.ones((16, 8))), \
                        mock.patch.object(tc.sh, "f0_track", return_value=np.array([120.0, 130.0, 140.0])):
                    summary = tc.smoke(pipe, out_dir)
                    self.assertLessEqual(summary["max_d_wer"], tc.REPRODUCTION_TOLERANCE)
                    self.assertTrue((out_dir / "smoke/smoke_check.json").exists())

                # Reproduction drift > tolerance must fail smoke.
                with mock.patch.object(tc, "check_prereg"), \
                        mock.patch.object(tc, "outputs", side_effect=lambda _p, c, e, f: {a: a for a in tc.ARMS}), \
                        mock.patch.object(tc.ng, "decode", side_effect=lambda _p, _c, m: [m]), \
                        mock.patch.object(pipe, "score", return_value={"wer": 0.99, "similarity": 0.1, "utmos": 2.0, "hypothesis": "x"}):
                    with self.assertRaises(RuntimeError):
                        tc.smoke(pipe, out_dir)


def _write_synthetic_csvs(out_dir, div_vals_fn, n=tc.FRESH_N):
    items = tc.fresh_items()[:n]
    div_path = Path(out_dir) / "diversity.csv"
    qual_path = Path(out_dir) / "quality.csv"
    with open(div_path, "w", newline="", encoding="utf-8") as fd, open(qual_path, "w", newline="", encoding="utf-8") as fq:
        dw = csv.DictWriter(fd, fieldnames=tc.DIV_FIELDS)
        qw = csv.DictWriter(fq, fieldnames=tc.QUAL_FIELDS)
        dw.writeheader()
        qw.writeheader()
        for idx, it in enumerate(items):
            drow = {"utt_id": it["utt_id"], "speaker_id": it["speaker_id"], "condition": "clean",
                    "n_seeds": 20, "voiced_common": 0.55, "case_seconds": 1.0, "utc": "2026-11-14T00:00:00Z"}
            drow.update(div_vals_fn(idx))
            dw.writerow(drow)

            qrow = {"utt_id": it["utt_id"], "speaker_id": it["speaker_id"], "condition": "clean",
                    "prompt_snr": 30.0, "gpu": "Fake-T4", "case_seconds": 1.0, "utc": "2026-11-14T00:00:00Z"}
            for arm in tc.ARMS:
                qrow.update({f"wer_{arm}": 0.05, f"similarity_{arm}": 0.60, f"utmos_{arm}": 3.8, f"hypothesis_{arm}": "ok"})
            qw.writerow(qrow)


class AnalyzeTests(unittest.TestCase):
    def _base_row(self, cvf0_vals, common_vals=None, mel_vals=None):
        common_vals = cvf0_vals if common_vals is None else common_vals
        mel_vals = cvf0_vals if mel_vals is None else mel_vals
        row = {}
        for arm in tc.ARMS:
            row[f"cvf0_{arm}"] = cvf0_vals[arm]
            row[f"voiced_{arm}"] = 0.60
            row[f"cvf0_common_{arm}"] = common_vals[arm]
            row[f"mel_spread_{arm}"] = mel_vals[arm]
        return row

    def test_confirmed_verdict_and_outputs(self):
        def gen(i):
            jitter = 0.002 * ((i % 5) - 2)
            vals = {"student4": 0.075, "W4": 0.095 + jitter, "W8": 0.096 + jitter,
                    "W4R2": 0.104 + jitter, "W8R2": 0.120 + jitter, "T16": 0.122 + jitter}
            return self._base_row(vals)

        with tempfile.TemporaryDirectory() as tmp:
            _write_synthetic_csvs(tmp, gen)
            res = tc.analyze(tmp)
            self.assertEqual(res["verdict"], "confirmed")
            self.assertFalse(res["partial"])
            self.assertEqual(res["n_diversity"], 40)
            self.assertTrue((Path(tmp) / "MANIFEST.json").exists())
            self.assertTrue((Path(tmp) / "TRANSMISSIBILITY_CONFIRM_REPORT.md").exists())
            self.assertTrue((Path(tmp) / "transmissibility_confirm_results.json").exists())

    def test_supported_direction_only_when_confound_or_interaction_fails(self):
        # Case A: H1 & H2 & H3 hold on cvf0, but H4 interaction is 0 -> linear view, not confirmed.
        def no_interaction(i):
            jitter = 0.001 * ((i % 4) - 1.5)
            vals = {"student4": 0.075, "W4": 0.095 + jitter, "W8": 0.096 + jitter,
                    "W4R2": 0.117 + jitter, "W8R2": 0.118 + jitter, "T16": 0.120 + jitter}
            return self._base_row(vals)

        with tempfile.TemporaryDirectory() as tmp:
            _write_synthetic_csvs(tmp, no_interaction)
            self.assertEqual(tc.analyze(tmp)["verdict"], "supported_direction_only")

        # Case B: H1, H2, H3, H4 hold on raw cvf0, but common_cvf0 interaction vanishes -> tracker confound caught.
        def common_drops(i):
            jitter = 0.001 * ((i % 4) - 1.5)
            raw = {"student4": 0.075, "W4": 0.095 + jitter, "W8": 0.096 + jitter,
                   "W4R2": 0.104 + jitter, "W8R2": 0.120 + jitter, "T16": 0.122 + jitter}
            common = {"student4": 0.075, "W4": 0.095, "W8": 0.095, "W4R2": 0.095, "W8R2": 0.095, "T16": 0.095}
            return self._base_row(raw, common_vals=common)

        with tempfile.TemporaryDirectory() as tmp:
            _write_synthetic_csvs(tmp, common_drops)
            self.assertEqual(tc.analyze(tmp)["verdict"], "supported_direction_only")

    def test_refuted_verdict(self):
        def reversed_h1(i):
            jitter = 0.001 * ((i % 4) - 1.5)
            vals = {"student4": 0.075, "W4": 0.110 + jitter, "W8": 0.112 + jitter,
                    "W4R2": 0.112 + jitter, "W8R2": 0.114 + jitter, "T16": 0.080 + jitter}
            return self._base_row(vals)

        with tempfile.TemporaryDirectory() as tmp:
            _write_synthetic_csvs(tmp, reversed_h1)
            self.assertEqual(tc.analyze(tmp)["verdict"], "refuted")

    def test_mixed_verdict_and_partial_handling(self):
        def h3_not_flat(i):
            jitter = 0.001 * ((i % 4) - 1.5)
            vals = {"student4": 0.075, "W4": 0.080 + jitter, "W8": 0.100 + jitter,
                    "W4R2": 0.105 + jitter, "W8R2": 0.125 + jitter, "T16": 0.130 + jitter}
            return self._base_row(vals)

        with tempfile.TemporaryDirectory() as tmp:
            _write_synthetic_csvs(tmp, h3_not_flat, n=10)
            with self.assertRaises(RuntimeError):
                tc.analyze(tmp, allow_partial=False)
            res = tc.analyze(tmp, allow_partial=True)
            self.assertTrue(res["partial"])
            self.assertEqual(res["n_diversity"], 10)
            self.assertEqual(res["verdict"], "mixed")


class PreregistrationTests(unittest.TestCase):
    def test_repo_preregistration_valid(self):
        doc = tc.check_prereg()
        self.assertEqual(doc["experiment"], "EXP-027")
        self.assertEqual(doc["n_fresh"], 40)

    def test_tampered_preregistration_and_code_hash_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            copy_path = Path(tmp) / "preregistration.json"
            shutil.copy2(tc.PREREG_PATH, copy_path)
            tc.check_prereg(copy_path)

            # Tamper payload -> sha256 mismatch
            data = json.loads(copy_path.read_text(encoding="utf-8"))
            data["n_fresh"] = 99
            copy_path.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaises(RuntimeError):
                tc.check_prereg(copy_path)

            # Tamper recorded code hash with recomputed payload sha -> code hash mismatch
            shutil.copy2(tc.PREREG_PATH, copy_path)
            data = json.loads(copy_path.read_text(encoding="utf-8"))
            payload = {k: v for k, v in data.items() if k not in ("sha256", "created_utc")}
            payload["code_hashes"]["run_transmissibility_confirm.py"] = "0" * 64
            payload_sha = tc._sha256_bytes(json.dumps(payload, sort_keys=True).encode("utf-8"))
            copy_path.write_text(json.dumps({**payload, "sha256": payload_sha, "created_utc": data["created_utc"]}),
                                 encoding="utf-8")
            with self.assertRaises(RuntimeError):
                tc.check_prereg(copy_path)


class NotebookTests(unittest.TestCase):
    def test_notebook_structure_and_python_syntax(self):
        nb_path = tc.THESIS_ROOT / "code/colab/DMOSpeech2_EXP027_Transmissibility_Confirmation.ipynb"
        nb = json.loads(nb_path.read_text(encoding="utf-8"))
        self.assertEqual(len(nb["cells"]), 17)
        all_src = "\n".join("".join(c["source"]) for c in nb["cells"])
        self.assertIn("rhnl-3", all_src)
        self.assertIn("run_transmissibility_confirm.py", all_src)
        for cell in nb["cells"]:
            if cell["cell_type"] == "code":
                ast.parse("".join(cell["source"]))


if __name__ == "__main__":
    unittest.main()
