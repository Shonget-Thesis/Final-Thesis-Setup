"""Unit tests for combined EXP-035 + EXP-036 (Path A + Path B) operators, disjoint cohort, notebook, and runner."""
import csv
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code/colab"))
import handover_mechanisms as hm  # noqa: E402
import run_consistency_gate_prelim as cg  # noqa: E402
import run_exp035_036_combined as comb  # noqa: E402
import run_invariant_residual as ir  # noqa: E402
import run_method_comparison as mc  # noqa: E402
import run_transmissibility_bridge as tb  # noqa: E402


def _mel(seed=0, frames=160, bands=12):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 3 * np.pi, frames)[:, None]
    macro = -5.5 + 2.2 * np.sin(t + np.arange(bands)[None, :] * 0.25)
    return (macro + 0.15 * rng.normal(size=(frames, bands))).astype(np.float32)


def _write_rows(path, fields, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


class OperatorMathTests(unittest.TestCase):
    def test_ladder_invariance_laws_and_prompt_frames(self):
        rng = np.random.default_rng(1)
        plen = 20
        s_full = _mel(seed=10, frames=100, bands=8)[None, :, :]
        t_full = s_full.copy()
        t_gen = t_full[0, plen:, :]
        s_gen = s_full[0, plen:, :]

        shifts = rng.uniform(-2.0, 2.0, size=(1, t_gen.shape[1]))
        scales = rng.uniform(0.5, 2.5, size=(1, t_gen.shape[1]))

        np.testing.assert_allclose(hm.voice_swap_gen(s_gen, s_gen + shifts), s_gen, atol=1e-5)
        np.testing.assert_allclose(hm.voice_swap_gen(s_gen, t_gen + shifts), hm.voice_swap_gen(s_gen, t_gen), atol=1e-5)
        np.testing.assert_allclose(hm.affine_gen(s_gen, scales * s_gen + shifts), s_gen, atol=1e-5)
        np.testing.assert_allclose(hm.affine_gen(s_gen, scales * t_gen + shifts), hm.affine_gen(s_gen, t_gen), atol=1e-5)

        t_zero = t_gen.copy()
        t_zero[:, 0] = -4.0
        g2_z = hm.affine_gen(s_gen, t_zero)
        self.assertTrue(np.all(np.isfinite(g2_z)))
        np.testing.assert_allclose(g2_z[:, 0], s_gen[:, 0].mean(), atol=1e-5)

        t_mono = 0.5 * np.log(np.exp(2.0 * t_gen.astype(np.float64)) + 1e-4)
        np.testing.assert_allclose(mc.quantile_handover(s_gen, t_mono), mc.quantile_handover(s_gen, t_gen), atol=1e-6)
        np.testing.assert_allclose(hm.floor_canonical_gen(s_gen, t_mono), hm.floor_canonical_gen(s_gen, t_gen), atol=1e-5)
        expected_fch_self = 0.5 * np.log(np.maximum(np.exp(2.0 * s_gen.astype(np.float64)),
                                                    np.percentile(np.exp(2.0 * s_gen.astype(np.float64)), 1.0, axis=0)))
        np.testing.assert_allclose(hm.floor_canonical_gen(s_gen, s_gen), expected_fch_self, atol=1e-5)

        t_dirty_prompt = t_full.copy()
        t_dirty_prompt[:, :plen, :] += 5.0
        for builder in (hm.cph8_dp_payload, hm.fch_payload, hm.g1_vswap_payload, hm.g2_affine_payload):
            p = builder(s_full, t_dirty_prompt, plen)
            np.testing.assert_array_equal(p[:, :plen, :], s_full[:, :plen, :])

    def test_dual_space_geodesic_and_rank_coherence_identities(self):
        s_gen = _mel(seed=20, frames=120, bands=10)
        np.testing.assert_allclose(hm.dual_space_geodesic_gen(s_gen, s_gen, macro_weight=0.0), s_gen, atol=1e-5)
        np.testing.assert_allclose(hm.dual_space_geodesic_gen(s_gen, s_gen, macro_weight=0.5), s_gen, atol=1e-5)
        np.testing.assert_allclose(hm.proximal_endpoint_gen(s_gen, s_gen, step_scale=1.0), s_gen, atol=1e-5)

        t_clean = 0.98 * s_gen + 0.02 * (-5.0)
        _, _, _, alpha_cl, _, r_glob_cl = hm.geodesic_shrink_weights(s_gen, t_clean)
        self.assertLess(r_glob_cl, hm.CLEAN_R_MED)
        self.assertTrue(np.all(alpha_cl == 0.0))

        t_noisy = 0.5 * np.log(np.exp(2.0 * s_gen.astype(np.float64)) + 0.08).astype(np.float32)
        out_noisy = hm.dual_space_geodesic_gen(s_gen, t_noisy, macro_weight=0.5)
        self.assertLess(ir.invariant_residual(out_noisy, s_gen), 1e-4)

        rng = np.random.default_rng(99)
        t_jitter = s_gen + 1.5 * rng.normal(size=s_gen.shape).astype(np.float32)
        ranks_micro, eta = hm.multiscale_rank_blend(t_jitter, s_gen, np.ones(s_gen.shape[1]), macro_weight=0.0)
        self.assertTrue(float(eta.mean()) > 0.4)
        self.assertEqual(ranks_micro.shape, s_gen.shape)

    def test_pytorch_parity_and_cwps_mock_loop(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch not installed")
        rng = np.random.default_rng(42)
        plen = 16
        s_np = _mel(seed=3, frames=90, bands=10)[None, :, :]
        t_np = (s_np + 0.4 * rng.normal(size=s_np.shape)).astype(np.float32)
        s_t, t_t = torch.from_numpy(s_np), torch.from_numpy(t_np)
        s_g_np, t_g_np = s_np[0, plen:, :], t_np[0, plen:, :]
        s_g_t, t_g_t = s_t[0, plen:, :], t_t[0, plen:, :]

        for fn_name in ("floor_canonical_gen", "voice_swap_gen", "affine_gen", "proximal_endpoint_gen"):
            fn = getattr(hm, fn_name)
            np.testing.assert_allclose(fn(s_g_t, t_g_t).numpy(), fn(s_g_np, t_g_np), atol=1e-4)
        for mw in (0.0, 0.5):
            np.testing.assert_allclose(
                hm.dual_space_geodesic_gen(s_g_t, t_g_t, macro_weight=mw).numpy(),
                hm.dual_space_geodesic_gen(s_g_np, t_g_np, macro_weight=mw),
                atol=1e-4,
            )

        torch_mod = torch

        class _DummySampler:
            CFG_STRENGTH = 2.0

        class _DummyTTS:
            def teacher(self, x, cond, text, time, mask, drop_audio_cond, drop_text):
                return (cond - x) * (0.2 if drop_audio_cond else 0.5)

        class _DummyPipe:
            torch = torch_mod
            device = "cpu"
            sampler = _DummySampler()
            tts = _DummyTTS()

        ctx = {
            "prompt_len": plen,
            "y0": torch.randn_like(s_t),
            "step_cond": s_t.clone(),
            "text_ids": torch.zeros((1, 5), dtype=torch.long),
        }
        out_cwps = hm.cwps_teacher(_DummyPipe(), ctx, s_t, hm.gl.sway_grid(8), k_end=4, macro_weight=0.5)
        self.assertEqual(out_cwps.shape, s_t.shape)
        self.assertTrue(bool(torch.all(torch.isfinite(out_cwps))))
        self.assertEqual(float((out_cwps[:, :plen, :] - s_t[:, :plen, :]).abs().max()), 0.0)

    def test_warm_metrics_runs_once_and_restores_threads(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch not installed")
        calls = []

        class _DummyASR:
            def transcribe(self, audio, sample_rate):
                calls.append(("asr", len(audio), sample_rate, torch.get_num_threads()))
                return "ok"

        class _DummyMetrics:
            asr = _DummyASR()

            def compute_similarity(self, a, b, sample_rate, gt_sr):
                calls.append(("sim", len(a), sample_rate))
                return 0.9

            def compute_utmos(self, a, sample_rate):
                calls.append(("utmos", len(a), sample_rate))
                return 3.5

        class _WarmPipe:
            pass

        p = _WarmPipe()
        p.torch = torch
        p.metrics = _DummyMetrics()
        orig_threads = torch.get_num_threads()
        comb._warm_metrics(p)
        comb._warm_metrics(p)  # second call must be a no-op
        self.assertEqual(len(calls), 3)
        self.assertEqual(calls[0][3], 1)
        self.assertEqual(torch.get_num_threads(), orig_threads)


class CohortAndNotebookTests(unittest.TestCase):
    def test_strict_speaker_disjointness_and_pass_counts(self):
        dev = {it["speaker_id"] for it in cg.dev_items()}
        exp028 = {it["speaker_id"] for it in tb.unseen_qual_items(30)}
        combined = {it["speaker_id"] for it in comb.qual_items()}
        self.assertEqual(len(combined), 30)
        self.assertEqual(len(comb.div_items()), 20)
        self.assertEqual(len(comb.time_items()), 10)
        self.assertFalse(dev & combined)
        self.assertFalse(exp028 & combined)
        self.assertEqual(comb.COST["STU"], 4)
        self.assertEqual(comb.COST["STUR2"], 3)
        self.assertEqual(comb.COST["CPH8"], 12)
        for arm in ("CPQH", "SNR_R", "DSGH_MICRO", "DSGH_DUAL", "CWPS", "FCH_DP", "G1_VSWAP", "G2_AFFINE"):
            self.assertEqual(comb.COST[arm], 13)

    def test_combined_notebook_valid_json_and_clean_paths(self):
        nb_path = ROOT / "code/colab/DMOSpeech2_EXP035_EXP036_AllInOne.ipynb"
        nb = json.loads(nb_path.read_text(encoding="utf-8"))
        self.assertEqual(nb["nbformat"], 4)
        text = "\n".join("".join(c["source"]) for c in nb["cells"])
        self.assertIn('THESIS_BRANCH = "rhnl-3"', text)
        self.assertIn("run_exp035_036_combined.py", text)
        self.assertIn("EXP-035-036-all-in-one", text)
        self.assertNotIn("EXP-034-invariant-residual", text)
        self.assertNotIn("run_invariant_residual.py", text)


class SyntheticRunnerTests(unittest.TestCase):
    def test_combined_prereg_and_analysis(self):
        doc = comb.check_prereg()
        self.assertEqual(doc["experiment"], "EXP-035+EXP-036")
        tmp = Path(tempfile.mkdtemp())
        try:
            rng = np.random.default_rng(7)
            q = []
            for it in comb.qual_items():
                for cond in comb.CONDS:
                    hiss = cond in ("noisy15", "gauss25", "pink25", "hp25")
                    row = {
                        "utt_id": it["utt_id"], "speaker_id": it["speaker_id"], "condition": cond,
                        "prompt_snr": 17.0 if cond == "noisy15" else 27.0 if hiss else 40.0,
                        "snr_sw": 30.0 if hiss else 60.0,
                        "r": (0.55 if hiss else 0.22) + 0.02 * rng.random(),
                        "drift8": 0.25 + 0.05 * rng.random(),
                        "rough_share": 0.5,
                    }
                    if cond == "noisy15":
                        scores = {"STU": 1.6, "STUR2": 2.9, "CPH8": 1.3, "CPH8_DP": 1.4, "FCH_DP": 1.5,
                                  "G1_VSWAP": 2.1, "G2_AFFINE": 3.0, "CPQH": 2.2, "W8": 1.3,
                                  "DSGH_MICRO": 2.5, "DSGH_DUAL": 2.8, "CWPS": 2.85}
                    elif hiss:
                        scores = {"STU": 3.6, "STUR2": 3.85, "CPH8": 1.9, "CPH8_DP": 2.0, "FCH_DP": 2.1,
                                  "G1_VSWAP": 2.5, "G2_AFFINE": 3.2, "CPQH": 3.75, "W8": 2.0,
                                  "DSGH_MICRO": 3.74, "DSGH_DUAL": 3.76, "CWPS": 3.80}
                    else:
                        scores = {a: 3.9 for a in comb.QUAL_ARMS}
                    for a in comb.QUAL_ARMS:
                        row.update({f"utmos_{a}": scores[a] + 0.03 * rng.random(), f"wer_{a}": 0.04, f"similarity_{a}": 0.88})
                    q.append(row)
            d = []
            for it in comb.div_items():
                for cond in comb.DIV_CONDS:
                    row = {"utt_id": it["utt_id"], "speaker_id": it["speaker_id"], "condition": cond}
                    for a in comb.DIV_ARMS + comb.POLICIES:
                        base = {"STU": 0.04, "STUR2": 0.04, "CPH8": 0.09, "G1_VSWAP": 0.08, "G2_AFFINE": 0.075,
                                "CPQH": 0.075, "DSGH_MICRO": 0.088, "DSGH_DUAL": 0.088, "CWPS": 0.087, "P25": 0.095}.get(a, 0.089)
                        for m in ir.DIV_MEASURES:
                            row[f"{m}_{a}"] = base + 0.002 * rng.random()
                    d.append(row)
            _write_rows(tmp / "quality.csv", comb.QUAL_FIELDS, q)
            _write_rows(tmp / "diversity.csv", comb.DIV_FIELDS, d)
            res = comb.analyze(tmp)
            self.assertFalse(res["partial"])
            self.assertEqual(res["path_a_verdict"], "ladder_and_snr_r_confirmed")
            self.assertEqual(res["path_b_verdict"], "novel_mechanism_advances")
            for fn in ("PATH_A_INVARIANCE_LADDER_REPORT.md", "PATH_B_DUAL_SPACE_REPORT.md", "EXP035_036_COMBINED_REPORT.md"):
                self.assertTrue((tmp / fn).exists())
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
