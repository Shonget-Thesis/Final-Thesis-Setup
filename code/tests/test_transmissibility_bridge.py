"""Unit tests for EXP-028 (Transmissible Slow-Band Bridge & STAC Handover runner and notebook)."""
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
import run_selective_handover_prelim as sh  # noqa: E402
import run_transmissibility_bridge as tb  # noqa: E402
import run_transmissibility_confirm as tc  # noqa: E402


class BridgeMathTests(unittest.TestCase):
    def test_bridge_handover_prompt_direction_and_norm(self):
        rng = np.random.default_rng(123)
        plen, frames, bands = 6, 40, 10
        t = np.linspace(0.0, 2.0 * np.pi, frames - plen)[:, None]

        # Build zero-mean slow oscillation on T4 (large amplitude) and S1 (smaller amplitude)
        T4 = rng.normal(0.0, 0.2, size=(1, frames, bands))
        S1 = rng.normal(0.0, 0.2, size=(1, frames, bands))
        T4[0, plen:, :] += 3.5 * np.sin(t)
        S1[0, plen:, :] += 0.7 * np.cos(t) + 0.1 * rng.normal(size=(frames - plen, bands))

        _, s_t, _ = sh.decompose(T4[0, plen:, :], sh.SLOW_WINDOW)
        v_s, s_s, f_s = sh.decompose(S1[0, plen:, :], sh.SLOW_WINDOW)
        norm_s = float(np.linalg.norm(s_s))
        norm_t = float(np.linalg.norm(s_t))
        self.assertGreater(norm_t, norm_s * 2.0)

        out_matched = tb.bridge_handover(T4, S1, plen, energy_match=True)
        out_raw = tb.bridge_handover(T4, S1, plen, energy_match=False)

        # Prompt frames untouched
        np.testing.assert_allclose(out_matched[:, :plen, :], S1[:, :plen, :], atol=1e-12)
        np.testing.assert_allclose(out_raw[:, :plen, :], S1[:, :plen, :], atol=1e-12)

        # Exact bridged slow component added on top of S1's voice + fine
        comp_matched = out_matched[0, plen:, :] - (v_s + f_s)
        comp_raw = out_raw[0, plen:, :] - (v_s + f_s)
        np.testing.assert_allclose(np.linalg.norm(comp_matched), norm_s, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(np.linalg.norm(comp_raw), norm_t, rtol=1e-6, atol=1e-6)

        # Cosine similarity of slow direction == +1.0
        cos_m = float((comp_matched * s_t).sum()) / (np.linalg.norm(comp_matched) * norm_t)
        cos_r = float((comp_raw * s_t).sum()) / (np.linalg.norm(comp_raw) * norm_t)
        self.assertAlmostEqual(cos_m, 1.0, places=6)
        self.assertAlmostEqual(cos_r, 1.0, places=6)

        # Bridging preserves S1's voice and fine components identically
        np.testing.assert_allclose(out_matched[0, plen:, :] - comp_matched, v_s + f_s, atol=1e-12)
        np.testing.assert_allclose(out_raw[0, plen:, :] - s_t, v_s + f_s, atol=1e-12)

        # Re-decomposing the output aligns slow movement with s_t (+0.98+) and fine detail with f_s (+0.90+)
        _, s_out, f_out = sh.decompose(out_matched[0, plen:, :], sh.SLOW_WINDOW)
        self.assertGreater(float((s_out * s_t).sum()) / (np.linalg.norm(s_out) * norm_t), 0.98)
        self.assertGreater(float((f_out * f_s).sum()) / (np.linalg.norm(f_out) * np.linalg.norm(f_s)), 0.90)

    def test_costs_and_fresh_speaker_alignment(self):
        self.assertEqual(tb.cost("student4"), 4)
        self.assertEqual(tb.cost("W4R2"), 12)
        self.assertEqual(tb.cost("W4"), 13)
        self.assertEqual(tb.cost("TBRG"), 13)
        self.assertEqual(tb.cost("STS2"), 13)
        self.assertEqual(tb.cost("W8R2"), 20)
        self.assertEqual(tb.cost("W8"), 21)
        self.assertEqual(tb.cost("T16"), 32)

        exp27_fresh = tc.fresh_items()
        self.assertEqual([it["utt_id"] for it in tb.qual_items()], [it["utt_id"] for it in exp27_fresh])
        self.assertEqual([it["utt_id"] for it in tb.div_items()], [it["utt_id"] for it in exp27_fresh[:20]])
        self.assertEqual([it["utt_id"] for it in tb.smoke_items()], [it["utt_id"] for it in exp27_fresh[:2]])


class _FakeSampler:
    STUDENT_TAUS = [0.0761, 0.2929, 0.6173]

    @staticmethod
    def compute_stabilization_drift(ep, prev, plen):
        diff = ep[:, plen:, :] - prev[:, plen:, :]
        return float(np.linalg.norm(diff) / (np.linalg.norm(ep[:, plen:, :]) + 1e-8))


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
        stored = {r["utt_id"]: r for r in tb.rf.read_csv(tb.EXP027_QUAL_CSV)}
        if item["utt_id"] in stored:
            r = stored[item["utt_id"]]
            return {
                "wer": float(r["wer_W4"]),
                "similarity": float(r["similarity_W4"]),
                "utmos": float(r["utmos_W4"]),
                "hypothesis": "ok",
            }
        return {"wer": 0.02, "similarity": 0.92, "utmos": 3.8, "hypothesis": "ok"}


class LoopTests(unittest.TestCase):
    def test_diversity_quality_and_smoke_loops(self):
        pipe = _FakePipe()
        items = tb.smoke_items()

        def fake_teacher(_pipe, ctx):
            rng = np.random.default_rng(int(ctx["seed"]))
            t4 = rng.normal(0.0, 0.5, size=(1, 20, 8))
            drift = {"d_1": 0.4, "d_2": 0.2, "d_3": 0.1, "d_4": 0.05, "A_4": 0.75, "rho_4": 0.5}
            return t4, drift

        def fake_student(_pipe, ctx, state, taus):
            # Deterministic function of state + taus so split S1 -> taus[1:] equals full T4 -> taus
            acc = np.array(state, dtype=float, copy=True)
            for tau in taus:
                acc = 0.5 * acc + float(tau)
            return acc

        def fake_f0(mel):
            v = float(np.mean(mel[0, 4:, :]))
            return np.array([120.0 + 20.0 * v, 130.0 + 25.0 * v, 140.0 + 30.0 * v])

        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            with mock.patch.object(tb, "teacher_step4", side_effect=fake_teacher), \
                    mock.patch.object(tb.gl, "student", side_effect=fake_student), \
                    mock.patch.object(tb.ng, "decode", side_effect=lambda _p, _c, m: m), \
                    mock.patch.object(tb.sh, "f0_track", side_effect=fake_f0):

                div_csv = tb.run_diversity(pipe, items, out_dir, seeds=3)
                with open(div_csv, encoding="utf-8") as f:
                    rows = list(csv.DictReader(f))
                self.assertEqual(len(rows), 2)
                self.assertTrue(set(tb.DIV_FIELDS).issubset(rows[0].keys()))

                # Resumes cleanly
                tb.run_diversity(pipe, items, out_dir, seeds=3)
                with open(div_csv, encoding="utf-8") as f:
                    self.assertEqual(len(list(csv.DictReader(f))), 2)

                qual_csv = tb.run_quality(pipe, items, out_dir)
                with open(qual_csv, encoding="utf-8") as f:
                    qrows = list(csv.DictReader(f))
                self.assertEqual(len(qrows), 2)
                self.assertTrue(set(tb.QUAL_FIELDS).issubset(qrows[0].keys()))

                summary = tb.smoke(pipe, out_dir)
                self.assertEqual(summary["max_abs_split_diff"], 0.0)
                self.assertLessEqual(summary["max_d_wer"], tb.REPRODUCTION_TOLERANCE)
                self.assertLessEqual(summary["max_d_sim"], tb.REPRODUCTION_TOLERANCE)
                self.assertTrue((out_dir / "smoke/smoke_check.json").exists())

                # Drift in stored reproduction raises
                with mock.patch.object(pipe, "score", return_value={"wer": 0.99, "similarity": 0.1, "utmos": 1.0, "hypothesis": "x"}):
                    with self.assertRaises(RuntimeError):
                        tb.smoke(pipe, out_dir)

            # Split-student mismatch raises
            with mock.patch.object(tb, "teacher_step4", side_effect=fake_teacher), \
                    mock.patch.object(tb.gl, "student", side_effect=[np.zeros((1, 20, 8)), np.zeros((1, 20, 8)), np.ones((1, 20, 8))]):
                with self.assertRaises(RuntimeError):
                    tb.smoke(pipe, out_dir)


def _write_synthetic_dataset(out_dir, tbrg_fn, sts2_fn=None, n_div=tb.DIV_PAIRS, n_qual=tb.FRESH_N):
    """Write synthetic EXP-028 CSVs plus matching EXP-027 CSV copies for fast isolated tests."""
    out_dir = Path(out_dir)
    all_items = tb.qual_items(tb.FRESH_N)
    e27_div = out_dir / "exp027_div.csv"
    e27_qual = out_dir / "exp027_qual.csv"
    div_path = out_dir / "diversity.csv"
    qual_path = out_dir / "quality.csv"

    e27_div_fields = tc.DIV_FIELDS
    e27_qual_fields = tc.QUAL_FIELDS

    with open(e27_div, "w", newline="", encoding="utf-8") as fd27, \
            open(e27_qual, "w", newline="", encoding="utf-8") as fq27, \
            open(div_path, "w", newline="", encoding="utf-8") as fd28, \
            open(qual_path, "w", newline="", encoding="utf-8") as fq28:

        dw27, qw27 = csv.DictWriter(fd27, fieldnames=e27_div_fields), csv.DictWriter(fq27, fieldnames=e27_qual_fields)
        dw28, qw28 = csv.DictWriter(fd28, fieldnames=tb.DIV_FIELDS), csv.DictWriter(fq28, fieldnames=tb.QUAL_FIELDS)
        dw27.writeheader()
        qw27.writeheader()
        dw28.writeheader()
        qw28.writeheader()

        for i, it in enumerate(all_items):
            spec = tbrg_fn(i)
            s2 = sts2_fn(i) if sts2_fn else {
                "cvf0": spec["cvf0"] - 0.002,
                "mel_spread": spec["mel_spread"] + 0.01,
                "wer": spec["wer"],
                "similarity": spec["similarity"],
                "utmos": spec["utmos"] - 0.08,
            }

            d27 = {"utt_id": it["utt_id"], "speaker_id": it["speaker_id"], "condition": "clean",
                   "n_seeds": 20, "voiced_common": 0.50, "case_seconds": 1.0, "utc": "2026-11-14T00:00:00Z"}
            base_cv = {"student4": 0.050, "W4": 0.077, "W8": 0.077, "W4R2": 0.086, "W8R2": 0.089, "T16": 0.091}
            base_ms = {"student4": 0.530, "W4": 0.944, "W8": 0.934, "W4R2": 1.030, "W8R2": 0.935, "T16": 0.924}
            for arm in tc.ARMS:
                d27[f"cvf0_{arm}"] = base_cv[arm] + spec.get("jitter", 0.0)
                d27[f"voiced_{arm}"] = 0.60
                d27[f"cvf0_common_{arm}"] = d27[f"cvf0_{arm}"]
                d27[f"mel_spread_{arm}"] = base_ms[arm] + spec.get("jitter", 0.0)
            dw27.writerow(d27)

            if i < n_div:
                dw28.writerow({
                    "utt_id": it["utt_id"], "speaker_id": it["speaker_id"], "condition": "clean", "n_seeds": 20,
                    "cvf0_TBRG": spec["cvf0"], "voiced_TBRG": 0.60, "mel_spread_TBRG": spec["mel_spread"],
                    "cvf0_STS2": s2["cvf0"], "voiced_STS2": 0.60, "mel_spread_STS2": s2["mel_spread"],
                    "case_seconds": 1.0, "utc": "2026-11-14T00:00:00Z",
                })

            q27 = {"utt_id": it["utt_id"], "speaker_id": it["speaker_id"], "condition": "clean",
                   "prompt_snr": 30.0, "gpu": "Fake-T4", "case_seconds": 1.0, "utc": "2026-11-14T00:00:00Z"}
            base_u = {"student4": 3.74, "W4": 3.76, "W8": 3.78, "W4R2": 3.40, "W8R2": 3.67, "T16": 3.23}
            for arm in tc.ARMS:
                q27.update({f"wer_{arm}": 0.016, f"similarity_{arm}": 0.932, f"utmos_{arm}": base_u[arm], f"hypothesis_{arm}": "ok"})
            qw27.writerow(q27)

            if i < n_qual:
                r4 = spec.get("r4", 0.05 + 0.02 * (i % 5))
                a4 = spec.get("a4", 0.60 + 0.05 * (i % 5))
                d4 = r4 * a4
                qw28.writerow({
                    "utt_id": it["utt_id"], "speaker_id": it["speaker_id"], "condition": "clean",
                    "prompt_snr": 30.0, "gpu": "Fake-T4",
                    "d_1": 0.30 * a4, "d_2": 0.25 * a4, "d_3": (0.45 * a4) - d4, "d_4": d4, "A_4": a4,
                    "rho_4": d4 / max(1e-8, 0.45 * a4 - d4),
                    "wer_TBRG": spec["wer"], "similarity_TBRG": spec["similarity"], "utmos_TBRG": spec["utmos"], "hypothesis_TBRG": "ok",
                    "wer_STS2": s2["wer"], "similarity_STS2": s2["similarity"], "utmos_STS2": s2["utmos"], "hypothesis_STS2": "ok",
                    "case_seconds": 1.0, "utc": "2026-11-14T00:00:00Z",
                })
    return e27_div, e27_qual


class AnalyzeAndStacTests(unittest.TestCase):
    def test_confirmed_verdict_and_stac_routing(self):
        def spec(i):
            j = 0.001 * ((i % 4) - 1.5)
            # Half of trajectories (low r4) settle cleanly with high TBRG UTMOS (3.76);
            # unsettled half (high r4) drops UTMOS to 3.45 under TBRG so STAC routes to W8R2 (3.67).
            r4 = 0.04 if i % 2 == 0 else 0.18
            utmos = 3.76 if i % 2 == 0 else 3.66
            return {"cvf0": 0.088 + j, "mel_spread": 0.940 + j, "wer": 0.016, "similarity": 0.932, "utmos": utmos, "r4": r4, "jitter": j}

        with tempfile.TemporaryDirectory() as tmp:
            e27_d, e27_q = _write_synthetic_dataset(tmp, spec)
            res = tb.analyze(tmp, exp027_div_csv=e27_d, exp027_qual_csv=e27_q)
            self.assertEqual(res["verdict"], "confirmed")
            self.assertFalse(res["partial"])
            self.assertEqual(res["n_diversity"], 20)
            self.assertEqual(res["n_quality"], 40)
            self.assertTrue((Path(tmp) / "TRANSMISSIBILITY_BRIDGE_REPORT.md").exists())
            self.assertTrue((Path(tmp) / "transmissibility_bridge_results.json").exists())
            self.assertTrue((Path(tmp) / "MANIFEST.json").exists())
            self.assertLessEqual(res["stac_routing"]["relative_r4"]["mean_cost"], 20.0)
            self.assertGreaterEqual(res["stac_routing"]["relative_r4"]["mean_cost"], 13.0)

        # Mixed trajectories where settled subset (r4=0.04) has UTMOS 3.77 and unsettled (r4=0.22) has UTMOS 3.40 -> split routing
        def split_spec(i):
            j = 0.001 * ((i % 4) - 1.5)
            settled = (i % 2 == 0)
            return {
                "cvf0": 0.088 + j,
                "mel_spread": 0.940 + j,
                "wer": 0.016,
                "similarity": 0.932,
                "utmos": 3.77 if settled else 3.40,
                "r4": 0.04 if settled else 0.22,
                "jitter": j,
            }

        with tempfile.TemporaryDirectory() as tmp:
            e27_d, e27_q = _write_synthetic_dataset(tmp, split_spec)
            q_rows = tb.rf.read_csv(Path(tmp) / "quality.csv")
            d_map = {r["utt_id"]: r for r in tb.rf.read_csv(Path(tmp) / "diversity.csv")}
            sq = {r["utt_id"]: r for r in tb.rf.read_csv(e27_q)}
            sd = {r["utt_id"]: r for r in tb.rf.read_csv(e27_d)}
            rout = tb.evaluate_stac_routing(q_rows, d_map, sq, sd)["relative_r4"]
            self.assertGreater(rout["n_tbrg"], 0)
            self.assertGreater(rout["n_w8r2"], 0)
            self.assertGreater(rout["mean_cost"], 13.0)
            self.assertLess(rout["mean_cost"], 20.0)

    def test_supported_mixed_refuted_and_partial_verdicts(self):
        # supported_direction_only: H1 & H2 pass, but UTMOS mean d vs W8 < -0.12
        def sup(i):
            j = 0.001 * ((i % 4) - 1.5)
            return {"cvf0": 0.088 + j, "mel_spread": 0.940 + j, "wer": 0.016, "similarity": 0.932, "utmos": 3.55, "jitter": j}

        with tempfile.TemporaryDirectory() as tmp:
            e27_d, e27_q = _write_synthetic_dataset(tmp, sup)
            self.assertEqual(tb.analyze(tmp, exp027_div_csv=e27_d, exp027_qual_csv=e27_q)["verdict"], "supported_direction_only")

        # refuted: H1 CVf0 lower than W4
        def ref(i):
            j = 0.001 * ((i % 4) - 1.5)
            return {"cvf0": 0.060 + j, "mel_spread": 0.940 + j, "wer": 0.016, "similarity": 0.932, "utmos": 3.74, "jitter": j}

        with tempfile.TemporaryDirectory() as tmp:
            e27_d, e27_q = _write_synthetic_dataset(tmp, ref)
            self.assertEqual(tb.analyze(tmp, exp027_div_csv=e27_d, exp027_qual_csv=e27_q)["verdict"], "refuted")

        # mixed + partial
        def mix(i):
            j = 0.001 * ((i % 4) - 1.5)
            return {"cvf0": 0.077 + (0.005 if i % 2 else -0.005), "mel_spread": 0.940, "wer": 0.016, "similarity": 0.932, "utmos": 3.72, "jitter": j}

        with tempfile.TemporaryDirectory() as tmp:
            e27_d, e27_q = _write_synthetic_dataset(tmp, mix, n_div=6, n_qual=10)
            with self.assertRaises(RuntimeError):
                tb.analyze(tmp, allow_partial=False, exp027_div_csv=e27_d, exp027_qual_csv=e27_q)
            res = tb.analyze(tmp, allow_partial=True, exp027_div_csv=e27_d, exp027_qual_csv=e27_q)
            self.assertTrue(res["partial"])
            self.assertEqual(res["verdict"], "mixed")


class PreregistrationTests(unittest.TestCase):
    def test_repo_preregistration_valid(self):
        doc = tb.check_prereg()
        self.assertEqual(doc["experiment"], "EXP-028")
        self.assertEqual(doc["task_id"], "EVAL-021")
        self.assertEqual(doc["n_fresh_div"], 20)
        self.assertEqual(doc["n_fresh_qual"], 40)

    def test_tampered_preregistration_code_or_input_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            copy_path = Path(tmp) / "preregistration.json"
            shutil.copy2(tb.PREREG_PATH, copy_path)
            tb.check_prereg(copy_path)

            # Tamper payload -> sha256 mismatch
            data = json.loads(copy_path.read_text(encoding="utf-8"))
            data["n_fresh_qual"] = 99
            copy_path.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaises(RuntimeError):
                tb.check_prereg(copy_path)

            # Tamper code hash with recomputed payload sha -> code hash mismatch
            shutil.copy2(tb.PREREG_PATH, copy_path)
            data = json.loads(copy_path.read_text(encoding="utf-8"))
            payload = {k: v for k, v in data.items() if k not in ("sha256", "created_utc")}
            payload["code_hashes"]["run_transmissibility_bridge.py"] = "0" * 64
            payload_sha = tb._sha256_bytes(json.dumps(payload, sort_keys=True).encode("utf-8"))
            copy_path.write_text(json.dumps({**payload, "sha256": payload_sha, "created_utc": data["created_utc"]}),
                                 encoding="utf-8")
            with self.assertRaises(RuntimeError):
                tb.check_prereg(copy_path)

            # Tamper EXP-027 input hash with recomputed payload sha -> input hash mismatch
            shutil.copy2(tb.PREREG_PATH, copy_path)
            data = json.loads(copy_path.read_text(encoding="utf-8"))
            payload = {k: v for k, v in data.items() if k not in ("sha256", "created_utc")}
            k0 = next(iter(payload["input_hashes"]))
            payload["input_hashes"][k0] = "f" * 64
            payload_sha = tb._sha256_bytes(json.dumps(payload, sort_keys=True).encode("utf-8"))
            copy_path.write_text(json.dumps({**payload, "sha256": payload_sha, "created_utc": data["created_utc"]}),
                                 encoding="utf-8")
            with self.assertRaises(RuntimeError):
                tb.check_prereg(copy_path)


class NotebookTests(unittest.TestCase):
    def test_notebook_structure_and_python_syntax(self):
        nb_path = tb.THESIS_ROOT / "code/colab/DMOSpeech2_EXP028_Transmissibility_Bridge.ipynb"
        nb = json.loads(nb_path.read_text(encoding="utf-8"))
        self.assertEqual(len(nb["cells"]), 17)
        all_src = "\n".join("".join(c["source"]) for c in nb["cells"])
        self.assertIn("rhnl-3", all_src)
        self.assertIn("run_transmissibility_bridge.py", all_src)
        for cell in nb["cells"]:
            if cell["cell_type"] == "code":
                ast.parse("".join(cell["source"]))


if __name__ == "__main__":
    unittest.main()
