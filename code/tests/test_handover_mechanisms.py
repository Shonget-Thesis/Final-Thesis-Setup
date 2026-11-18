"""Unit tests for Path A (EXP-035) and Path B (EXP-036) operators, disjoint cohorts, notebooks, and runners."""
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
import run_dual_space_mechanisms as dsm  # noqa: E402
import run_invariance_ladder_confirm as ilc  # noqa: E402
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

        # G1 shift invariance
        np.testing.assert_allclose(hm.voice_swap_gen(s_gen, s_gen + shifts), s_gen, atol=1e-5)
        np.testing.assert_allclose(hm.voice_swap_gen(s_gen, t_gen + shifts), hm.voice_swap_gen(s_gen, t_gen), atol=1e-5)

        # G2 positive affine invariance
        np.testing.assert_allclose(hm.affine_gen(s_gen, scales * s_gen + shifts), s_gen, atol=1e-5)
        np.testing.assert_allclose(hm.affine_gen(s_gen, scales * t_gen + shifts), hm.affine_gen(s_gen, t_gen), atol=1e-5)

        # G2 handles constant zero-variance band safely
        t_zero = t_gen.copy()
        t_zero[:, 0] = -4.0
        g2_z = hm.affine_gen(s_gen, t_zero)
        self.assertTrue(np.all(np.isfinite(g2_z)))
        np.testing.assert_allclose(g2_z[:, 0], s_gen[:, 0].mean(), atol=1e-5)

        # G3 monotone invariance and FCH additive floor invariance + clamp identity
        t_mono = 0.5 * np.log(np.exp(2.0 * t_gen.astype(np.float64)) + 1e-4)
        np.testing.assert_allclose(mc.quantile_handover(s_gen, t_mono), mc.quantile_handover(s_gen, t_gen), atol=1e-6)
        np.testing.assert_allclose(hm.floor_canonical_gen(s_gen, t_mono), hm.floor_canonical_gen(s_gen, t_gen), atol=1e-5)
        expected_fch_self = 0.5 * np.log(np.maximum(np.exp(2.0 * s_gen.astype(np.float64)),
                                                    np.percentile(np.exp(2.0 * s_gen.astype(np.float64)), 1.0, axis=0)))
        np.testing.assert_allclose(hm.floor_canonical_gen(s_gen, s_gen), expected_fch_self, atol=1e-5)

        # Prompt frames always match student draft s0
        t_dirty_prompt = t_full.copy()
        t_dirty_prompt[:, :plen, :] += 5.0
        for builder in (hm.cph8_dp_payload, hm.fch_payload, hm.g1_vswap_payload, hm.g2_affine_payload):
            p = builder(s_full, t_dirty_prompt, plen)
            np.testing.assert_array_equal(p[:, :plen, :], s_full[:, :plen, :])

    def test_dual_space_geodesic_and_rank_coherence_identities(self):
        s_gen = _mel(seed=20, frames=120, bands=10)
        # Identity when T == S
        np.testing.assert_allclose(hm.dual_space_geodesic_gen(s_gen, s_gen, macro_weight=0.0), s_gen, atol=1e-5)
        np.testing.assert_allclose(hm.dual_space_geodesic_gen(s_gen, s_gen, macro_weight=0.5), s_gen, atol=1e-5)
        np.testing.assert_allclose(hm.proximal_endpoint_gen(s_gen, s_gen, step_scale=1.0), s_gen, atol=1e-5)

        # Small perturbation below clean floor leaves alpha == 0 and smooth teacher intact
        t_clean = 0.98 * s_gen + 0.02 * (-5.0)
        _, _, _, alpha_cl, r_band_cl, r_glob_cl = hm.geodesic_shrink_weights(s_gen, t_clean)
        self.assertLess(r_glob_cl, hm.CLEAN_R_MED)
        self.assertTrue(np.all(alpha_cl == 0.0))

        # Severe stationary noise floor saturates alpha -> 1 and maps onto clean s_gen levels
        t_noisy = 0.5 * np.log(np.exp(2.0 * s_gen.astype(np.float64)) + 0.08).astype(np.float32)
        out_noisy = hm.dual_space_geodesic_gen(s_gen, t_noisy, macro_weight=0.5)
        self.assertLess(ir.invariant_residual(out_noisy, s_gen), 1e-4)

        # High-frequency frame-rank jitter increases rough weight eta and pulls noisy output closer to S
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

        # Mock pipeline verifying cwps_teacher returns exact shape, clean prompt frames, and finite values
        class _DummySampler:
            CFG_STRENGTH = 2.0

        class _DummyTTS:
            def teacher(self, x, cond, text, time, mask, drop_audio_cond, drop_text):
                return (cond - x) * (0.2 if drop_audio_cond else 0.5)

        torch_mod = torch

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
        grid = hm.gl.sway_grid(8)
        out_cwps = hm.cwps_teacher(_DummyPipe(), ctx, s_t, grid, k_end=4, macro_weight=0.5)
        self.assertEqual(out_cwps.shape, s_t.shape)
        self.assertTrue(bool(torch.all(torch.isfinite(out_cwps))))
        self.assertEqual(float((out_cwps[:, :plen, :] - s_t[:, :plen, :]).abs().max()), 0.0)


class CohortAndNotebookTests(unittest.TestCase):
    def test_strict_speaker_disjointness_across_experiments(self):
        dev = {it["speaker_id"] for it in cg.dev_items()}
        exp028 = {it["speaker_id"] for it in tb.unseen_qual_items(30)}
        exp035 = {it["speaker_id"] for it in ilc.qual_items()}
        exp036 = {it["speaker_id"] for it in dsm.qual_items()}
        self.assertEqual(len(exp035), 30)
        self.assertEqual(len(exp036), 30)
        self.assertEqual(len(ilc.div_items()), 20)
        self.assertEqual(len(dsm.div_items()), 20)
        self.assertEqual(len(ilc.time_items()), 10)
        self.assertEqual(len(dsm.time_items()), 10)
        self.assertFalse(dev & exp035)
        self.assertFalse(dev & exp036)
        self.assertFalse(exp028 & exp035)
        self.assertFalse(exp028 & exp036)
        self.assertFalse(exp035 & exp036)

    def test_notebooks_valid_json_and_clean_paths(self):
        checks = [
            (ROOT / "code/colab/DMOSpeech2_EXP035_PathA_Ladder_Unseen.ipynb",
             "run_invariance_ladder_confirm.py", "EXP-035-invariance-ladder-confirm"),
            (ROOT / "code/colab/DMOSpeech2_EXP036_PathB_Novel_Mechanisms.ipynb",
             "run_dual_space_mechanisms.py", "EXP-036-dual-space-mechanisms"),
        ]
        for nb_path, runner_name, exp_dir_name in checks:
            nb = json.loads(nb_path.read_text(encoding="utf-8"))
            self.assertEqual(nb["nbformat"], 4)
            text = "\n".join("".join(c["source"]) for c in nb["cells"])
            self.assertIn('THESIS_BRANCH = "rhnl-3"', text)
            self.assertIn(runner_name, text)
            self.assertIn(exp_dir_name, text)
            self.assertNotIn("EXP-034-invariant-residual", text)
            self.assertNotIn("run_invariant_residual.py", text)


class SyntheticRunnerTests(unittest.TestCase):
    def test_exp035_prereg_and_analysis(self):
        doc = ilc.check_prereg()
        self.assertEqual(doc["experiment"], "EXP-035")
        tmp = Path(tempfile.mkdtemp())
        try:
            rng = np.random.default_rng(7)
            q = []
            for it in ilc.qual_items():
                for cond in ilc.CONDS:
                    hiss = cond in ("noisy15", "gauss25", "pink25", "hp25")
                    row = {
                        "utt_id": it["utt_id"], "speaker_id": it["speaker_id"], "condition": cond,
                        "prompt_snr": 17.0 if cond == "noisy15" else 27.0 if hiss else 40.0,
                        "snr_sw": 30.0 if hiss else 60.0,
                        "r": (0.55 if hiss else 0.22) + 0.02 * rng.random(),
                        "drift8": 0.25 + 0.05 * rng.random(),
                        "rough_share": 0.5,
                    }
                    scores = {"STU": 2.9, "STUR2": 3.1, "CPQH": 3.85}
                    if not hiss:
                        scores.update({"CPH8": 3.88, "CPH8_DP": 3.89, "FCH_DP": 3.86, "G1_VSWAP": 3.87, "G2_AFFINE": 3.88, "W8": 3.9})
                    else:
                        scores.update({"CPH8": 1.3, "CPH8_DP": 1.4, "FCH_DP": 1.5, "G1_VSWAP": 2.2, "G2_AFFINE": 3.2, "W8": 1.4})
                    for a in ilc.QUAL_ARMS:
                        row.update({f"utmos_{a}": scores[a] + 0.03 * rng.random(), f"wer_{a}": 0.04, f"similarity_{a}": 0.88})
                    q.append(row)
            d = []
            for it in ilc.div_items():
                for cond in ilc.DIV_CONDS:
                    row = {"utt_id": it["utt_id"], "speaker_id": it["speaker_id"], "condition": cond}
                    for a in ilc.DIV_ARMS + ilc.POLICIES:
                        base = {"STU": 0.04, "STUR2": 0.04, "CPH8": 0.09, "G1_VSWAP": 0.08, "G2_AFFINE": 0.075, "CPQH": 0.07, "P25": 0.095}.get(a, 0.089)
                        for m in ir.DIV_MEASURES:
                            row[f"{m}_{a}"] = base + 0.002 * rng.random()
                    d.append(row)
            _write_rows(tmp / "quality.csv", ilc.QUAL_FIELDS, q)
            _write_rows(tmp / "diversity.csv", ilc.DIV_FIELDS, d)
            res = ilc.analyze(tmp)
            self.assertFalse(res["partial"])
            self.assertEqual(res["verdict"], "ladder_and_snr_r_confirmed")
            self.assertTrue((tmp / "INVARIANCE_LADDER_REPORT.md").exists())
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_exp036_prereg_and_analysis(self):
        doc = dsm.check_prereg()
        self.assertEqual(doc["experiment"], "EXP-036")
        tmp = Path(tempfile.mkdtemp())
        try:
            rng = np.random.default_rng(11)
            q = []
            for it in dsm.qual_items():
                for cond in dsm.CONDS:
                    hiss = cond in ("noisy15", "gauss25", "pink25", "hp25")
                    row = {
                        "utt_id": it["utt_id"], "speaker_id": it["speaker_id"], "condition": cond,
                        "prompt_snr": 17.0 if cond == "noisy15" else 27.0 if hiss else 40.0,
                        "snr_sw": 30.0 if hiss else 60.0,
                        "r": (0.55 if hiss else 0.22) + 0.02 * rng.random(),
                        "drift8": 0.25 + 0.05 * rng.random(),
                        "rough_share": 0.5,
                    }
                    for a in dsm.QUAL_ARMS:
                        if cond == "noisy15":
                            utm = {"STU": 1.6, "STUR2": 2.9, "CPH8": 1.3, "CPQH": 2.2, "DSGH_MICRO": 2.4, "DSGH_DUAL": 2.75, "CWPS": 2.8}[a]
                        elif hiss:
                            utm = {"STU": 3.6, "STUR2": 3.8, "CPH8": 1.9, "CPQH": 3.7, "DSGH_MICRO": 3.7, "DSGH_DUAL": 3.72, "CWPS": 3.78}[a]
                        else:
                            utm = 3.9
                        row.update({f"utmos_{a}": utm + 0.03 * rng.random(), f"wer_{a}": 0.04, f"similarity_{a}": 0.88})
                    q.append(row)
            d = []
            for it in dsm.div_items():
                for cond in dsm.DIV_CONDS:
                    row = {"utt_id": it["utt_id"], "speaker_id": it["speaker_id"], "condition": cond}
                    for a in dsm.DIV_ARMS + dsm.POLICIES:
                        base = {"STU": 0.04, "STUR2": 0.04, "CPH8": 0.09, "CPQH": 0.075, "DSGH_MICRO": 0.088, "DSGH_DUAL": 0.088, "CWPS": 0.087, "P25": 0.095}.get(a, 0.088)
                        for m in ir.DIV_MEASURES:
                            row[f"{m}_{a}"] = base + 0.002 * rng.random()
                    d.append(row)
            _write_rows(tmp / "quality.csv", dsm.QUAL_FIELDS, q)
            _write_rows(tmp / "diversity.csv", dsm.DIV_FIELDS, d)
            res = dsm.analyze(tmp)
            self.assertFalse(res["partial"])
            self.assertEqual(res["verdict"], "novel_mechanism_advances")
            self.assertTrue((tmp / "DUAL_SPACE_REPORT.md").exists())
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
