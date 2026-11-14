"""Unit tests for EXP-028 (Transmissible Slow-Band Bridge `TBRG`, controls & unseen cohort)."""
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
import run_selective_handover_prelim as sh  # noqa: E402
import run_transmissibility_bridge as tb  # noqa: E402
import run_transmissibility_confirm as tc  # noqa: E402


class BridgeMathTests(unittest.TestCase):
    def test_bridge_handover_prompt_direction_and_norm(self):
        rng = np.random.default_rng(123)
        plen, frames, bands = 6, 40, 10
        t = np.linspace(0.0, 2.0 * np.pi, frames - plen)[:, None]

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

        np.testing.assert_allclose(out_matched[:, :plen, :], S1[:, :plen, :], atol=1e-12)
        np.testing.assert_allclose(out_raw[:, :plen, :], S1[:, :plen, :], atol=1e-12)

        comp_matched = out_matched[0, plen:, :] - (v_s + f_s)
        comp_raw = out_raw[0, plen:, :] - (v_s + f_s)
        np.testing.assert_allclose(np.linalg.norm(comp_matched), norm_s, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(np.linalg.norm(comp_raw), norm_t, rtol=1e-6, atol=1e-6)

        cos_m = float((comp_matched * s_t).sum()) / (np.linalg.norm(comp_matched) * norm_t)
        cos_r = float((comp_raw * s_t).sum()) / (np.linalg.norm(comp_raw) * norm_t)
        self.assertAlmostEqual(cos_m, 1.0, places=6)
        self.assertAlmostEqual(cos_r, 1.0, places=6)

        np.testing.assert_allclose(out_matched[0, plen:, :] - comp_matched, v_s + f_s, atol=1e-12)
        np.testing.assert_allclose(out_raw[0, plen:, :] - s_t, v_s + f_s, atol=1e-12)

        _, s_out, f_out = sh.decompose(out_matched[0, plen:, :], sh.SLOW_WINDOW)
        self.assertGreater(float((s_out * s_t).sum()) / (np.linalg.norm(s_out) * norm_t), 0.98)
        self.assertGreater(float((f_out * f_s).sum()) / (np.linalg.norm(f_out) * np.linalg.norm(f_s)), 0.90)

    def test_blend_handover_interpolation_and_prompt_preservation(self):
        rng = np.random.default_rng(7)
        plen, frames, bands = 5, 25, 6
        T4 = rng.normal(size=(1, frames, bands))
        S1 = rng.normal(size=(1, frames, bands))

        b0 = tb.blend_handover(T4, S1, plen, alpha=0.0)
        b1 = tb.blend_handover(T4, S1, plen, alpha=1.0)
        b5 = tb.blend_handover(T4, S1, plen, alpha=0.5)

        np.testing.assert_allclose(b0, S1, atol=1e-12)
        np.testing.assert_allclose(b1[:, :plen, :], S1[:, :plen, :], atol=1e-12)
        np.testing.assert_allclose(b1[:, plen:, :], T4[:, plen:, :], atol=1e-12)
        np.testing.assert_allclose(b5[:, :plen, :], S1[:, :plen, :], atol=1e-12)
        np.testing.assert_allclose(b5[:, plen:, :], 0.5 * (S1[:, plen:, :] + T4[:, plen:, :]), atol=1e-12)

    def test_resample_time_and_shuf_handover_controls(self):
        x = np.arange(20, dtype=float).reshape(10, 2)
        up = tb.resample_time(x, 19)
        down = tb.resample_time(x, 4)
        one_t = tb.resample_time(x, 1)
        one_s = tb.resample_time(x[:1], 5)
        self.assertEqual(up.shape, (19, 2))
        self.assertEqual(down.shape, (4, 2))
        self.assertEqual(one_t.shape, (1, 2))
        np.testing.assert_allclose(up[0], x[0])
        np.testing.assert_allclose(up[-1], x[-1])
        np.testing.assert_allclose(down[-1], x[-1])
        np.testing.assert_allclose(one_s, np.repeat(x[:1], 5, axis=0))
        with self.assertRaises(ValueError):
            tb.resample_time(x, 0)

        # Orthogonal slow waves between target T4 (sin) and donor (cos)
        plen, frames, bands = 6, 46, 8
        t = np.linspace(0.0, 2.0 * np.pi, frames - plen, endpoint=False)[:, None]
        donor_t = np.linspace(0.0, 2.0 * np.pi, 28, endpoint=False)[:, None]
        T4 = np.zeros((1, frames, bands))
        S1 = np.zeros((1, frames, bands))
        T4[0, plen:, :] = 2.5 * np.sin(t)
        S1[0, plen:, :] = 0.8 * np.sin(2.0 * t) + 0.1 * np.random.default_rng(1).normal(size=(frames - plen, bands))
        donor_mel = 3.0 * np.cos(donor_t) + np.ones((28, bands))

        _, s_t, _ = sh.decompose(T4[0, plen:, :], sh.SLOW_WINDOW)
        v_s, s_s, f_s = sh.decompose(S1[0, plen:, :], sh.SLOW_WINDOW)
        out_shuf = tb.shuf_handover(donor_mel, S1, plen)
        out_tbrg = tb.bridge_handover(T4, S1, plen, energy_match=True)

        np.testing.assert_allclose(out_shuf[:, :plen, :], S1[:, :plen, :], atol=1e-12)
        comp_shuf = out_shuf[0, plen:, :] - (v_s + f_s)
        np.testing.assert_allclose(np.linalg.norm(comp_shuf), np.linalg.norm(s_s), rtol=1e-6, atol=1e-6)

        c_tbrg = tb.slow_band_corr(out_tbrg, T4, plen)
        c_shuf = tb.slow_band_corr(out_shuf, T4, plen)
        self.assertGreater(c_tbrg, 0.95)
        self.assertLess(abs(c_shuf), 0.25)

    def test_paired_f0_corr(self):
        a = [np.array([100.0, 110.0, np.nan, 130.0, 140.0, 150.0, 160.0])]
        b = [np.array([200.0, 220.0, 230.0, 260.0, 280.0, 300.0, 320.0])]
        c = [np.array([320.0, 300.0, np.nan, 260.0, 240.0, 220.0, 200.0])]
        self.assertAlmostEqual(tb.paired_f0_corr(a, b), 1.0, places=6)
        self.assertAlmostEqual(tb.paired_f0_corr(a, c), -1.0, places=6)
        short = [np.array([100.0, 110.0, 120.0])]
        self.assertTrue(math.isnan(tb.paired_f0_corr(short, short)))

    def test_costs_and_speaker_disjointness(self):
        expected_costs = {
            "student4": 4, "W4R2": 12, "W4": 13, "TBRG": 13, "STS2": 13,
            "BLND": 13, "SHUF": 13, "W6R2": 16, "W8R2": 20, "W8": 21, "T16": 32,
        }
        for arm, val in expected_costs.items():
            self.assertEqual(tb.cost(arm), val)

        exp27_fresh = tc.fresh_items()
        self.assertEqual([it["utt_id"] for it in tb.qual_items()], [it["utt_id"] for it in exp27_fresh])
        self.assertEqual([it["utt_id"] for it in tb.div_items()], [it["utt_id"] for it in exp27_fresh[:20]])
        self.assertEqual([it["utt_id"] for it in tb.smoke_items()], [it["utt_id"] for it in exp27_fresh[:2]])

        dev_spks = {it["speaker_id"] for it in cg.dev_items()}
        unseen_q = tb.unseen_qual_items()
        unseen_d = tb.unseen_div_items()
        self.assertEqual(len(dev_spks), 100)
        self.assertEqual(len(unseen_q), 30)
        self.assertEqual(len(unseen_d), 20)
        self.assertEqual(len(dev_spks & {it["speaker_id"] for it in unseen_q}), 0)
        self.assertEqual([it["utt_id"] for it in unseen_d], [it["utt_id"] for it in unseen_q[:20]])


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
    def test_diversity_quality_unseen_and_smoke_loops(self):
        pipe = _FakePipe()
        items = tb.smoke_items()
        unseen_sub = tb.unseen_qual_items(3)

        def fake_teacher(_pipe, ctx, wanted=(4, 6)):
            rng = np.random.default_rng(int(ctx["seed"]))
            ends = {k: rng.normal(0.0, 0.5, size=(1, 20, 8)) for k in wanted}
            drift = {"d_1": 0.4, "d_2": 0.2, "d_3": 0.1, "d_4": 0.05, "A_4": 0.75, "rho_4": 0.5}
            return ends, drift

        def fake_student(_pipe, ctx, state, taus):
            acc = np.array(state, dtype=float, copy=True)
            for tau in taus:
                acc = 0.5 * acc + float(tau)
            return acc

        def fake_f0(mel):
            base = np.mean(mel[0, 4:, :], axis=-1)
            return 120.0 + 20.0 * base[:6]

        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            with mock.patch.object(tb, "teacher_steps", side_effect=fake_teacher), \
                    mock.patch.object(tb.gl, "student", side_effect=fake_student), \
                    mock.patch.object(tb.ng, "decode", side_effect=lambda _p, _c, m: m), \
                    mock.patch.object(tb.sh, "f0_track", side_effect=fake_f0):

                div_csv = tb.run_diversity(pipe, items, out_dir, seeds=3)
                with open(div_csv, encoding="utf-8") as f:
                    rows = list(csv.DictReader(f))
                self.assertEqual(len(rows), 2)
                self.assertTrue(set(tb.DIV_FIELDS).issubset(rows[0].keys()))

                tb.run_diversity(pipe, items, out_dir, seeds=3)
                with open(div_csv, encoding="utf-8") as f:
                    self.assertEqual(len(list(csv.DictReader(f))), 2)

                qual_csv = tb.run_quality(pipe, items, out_dir)
                with open(qual_csv, encoding="utf-8") as f:
                    qrows = list(csv.DictReader(f))
                self.assertEqual(len(qrows), 2)
                self.assertTrue(set(tb.QUAL_FIELDS).issubset(qrows[0].keys()))

                udiv_csv, uqual_csv = tb.run_unseen_replication(
                    pipe, out_dir, div_items_list=unseen_sub[:2], qual_items_list=unseen_sub, seeds=3
                )
                with open(udiv_csv, encoding="utf-8") as f:
                    ud_rows = list(csv.DictReader(f))
                with open(uqual_csv, encoding="utf-8") as f:
                    uq_rows = list(csv.DictReader(f))
                self.assertEqual(len(ud_rows), 2)
                self.assertEqual(len(uq_rows), 3)
                self.assertTrue(set(tb.UNSEEN_DIV_FIELDS).issubset(ud_rows[0].keys()))
                self.assertTrue(set(tb.UNSEEN_QUAL_FIELDS).issubset(uq_rows[0].keys()))

                summary = tb.smoke(pipe, out_dir)
                self.assertEqual(summary["max_abs_split_diff"], 0.0)
                self.assertLessEqual(summary["max_d_wer"], tb.REPRODUCTION_TOLERANCE)
                self.assertLessEqual(summary["max_d_sim"], tb.REPRODUCTION_TOLERANCE)
                self.assertTrue((out_dir / "smoke/smoke_check.json").exists())

                with mock.patch.object(pipe, "score", return_value={"wer": 0.99, "similarity": 0.1, "utmos": 1.0, "hypothesis": "x"}):
                    with self.assertRaises(RuntimeError):
                        tb.smoke(pipe, out_dir)

            with mock.patch.object(tb, "teacher_steps", side_effect=fake_teacher), \
                    mock.patch.object(tb.gl, "student", side_effect=[np.zeros((1, 20, 8)), np.zeros((1, 20, 8)), np.ones((1, 20, 8))]):
                with self.assertRaises(RuntimeError):
                    tb.smoke(pipe, out_dir)


def _write_synthetic_dataset(
    out_dir,
    tbrg_fn,
    sts2_fn=None,
    blnd_fn=None,
    shuf_fn=None,
    w6r2_fn=None,
    n_div=tb.DIV_PAIRS,
    n_qual=tb.FRESH_N,
    with_unseen=True,
    n_udiv=tb.UNSEEN_DIV_N,
    n_uqual=tb.UNSEEN_QUAL_N,
):
    """Write synthetic EXP-028 CSVs (61-100 + unseen_cal) and matching EXP-027 CSV copies."""
    out_dir = Path(out_dir)
    all_items = tb.qual_items(tb.FRESH_N)
    e27_div = out_dir / "exp027_div.csv"
    e27_qual = out_dir / "exp027_qual.csv"
    div_path = out_dir / "diversity.csv"
    qual_path = out_dir / "quality.csv"

    with open(e27_div, "w", newline="", encoding="utf-8") as fd27, \
            open(e27_qual, "w", newline="", encoding="utf-8") as fq27, \
            open(div_path, "w", newline="", encoding="utf-8") as fd28, \
            open(qual_path, "w", newline="", encoding="utf-8") as fq28:

        dw27, qw27 = csv.DictWriter(fd27, fieldnames=tc.DIV_FIELDS), csv.DictWriter(fq27, fieldnames=tc.QUAL_FIELDS)
        dw28, qw28 = csv.DictWriter(fd28, fieldnames=tb.DIV_FIELDS), csv.DictWriter(fq28, fieldnames=tb.QUAL_FIELDS)
        dw27.writeheader()
        qw27.writeheader()
        dw28.writeheader()
        qw28.writeheader()

        for i, it in enumerate(all_items):
            spec = tbrg_fn(i)
            arm_specs = {
                "TBRG": spec,
                "STS2": sts2_fn(i) if sts2_fn else {
                    "cvf0": spec["cvf0"] - 0.002, "mel_spread": spec["mel_spread"] + 0.01, "slow_corr": 0.68,
                    "wer": spec["wer"], "similarity": spec["similarity"], "utmos": spec["utmos"] - 0.08,
                },
                "BLND": blnd_fn(i) if blnd_fn else {
                    "cvf0": spec["cvf0"] - 0.006, "mel_spread": spec["mel_spread"] + 0.03, "slow_corr": 0.45,
                    "wer": spec["wer"] + 0.001, "similarity": spec["similarity"], "utmos": spec["utmos"] - 0.14,
                },
                "SHUF": shuf_fn(i) if shuf_fn else {
                    "cvf0": 0.077 + spec.get("jitter", 0.0), "mel_spread": spec["mel_spread"] + 0.02, "slow_corr": 0.05,
                    "wer": spec["wer"] + 0.005, "similarity": spec["similarity"] - 0.01, "utmos": spec["utmos"] - 0.25,
                },
                "W6R2": w6r2_fn(i) if w6r2_fn else {
                    "cvf0": spec["cvf0"] - 0.001, "mel_spread": spec["mel_spread"] + 0.02, "slow_corr": 0.62,
                    "wer": spec["wer"], "similarity": spec["similarity"], "utmos": spec["utmos"] - 0.12,
                },
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
                d28_row = {"utt_id": it["utt_id"], "speaker_id": it["speaker_id"], "condition": "clean", "n_seeds": 20,
                           "case_seconds": 1.0, "utc": "2026-11-14T00:00:00Z"}
                for a, asp in arm_specs.items():
                    d28_row.update({
                        f"cvf0_{a}": asp["cvf0"],
                        f"voiced_{a}": 0.60,
                        f"mel_spread_{a}": asp["mel_spread"],
                        f"slow_corr_{a}": asp.get("slow_corr", 0.72),
                    })
                dw28.writerow(d28_row)

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
                q28_row = {
                    "utt_id": it["utt_id"], "speaker_id": it["speaker_id"], "condition": "clean",
                    "prompt_snr": 30.0, "gpu": "Fake-T4",
                    "d_1": 0.30 * a4, "d_2": 0.25 * a4, "d_3": (0.45 * a4) - d4, "d_4": d4, "A_4": a4,
                    "rho_4": d4 / max(1e-8, 0.45 * a4 - d4),
                    "case_seconds": 1.0, "utc": "2026-11-14T00:00:00Z",
                }
                for a, asp in arm_specs.items():
                    q28_row.update({
                        f"wer_{a}": asp["wer"],
                        f"similarity_{a}": asp["similarity"],
                        f"utmos_{a}": asp["utmos"],
                        f"hypothesis_{a}": "ok",
                    })
                qw28.writerow(q28_row)

    if with_unseen:
        unseen_list = tb.unseen_qual_items(tb.UNSEEN_QUAL_N)
        ud_path = out_dir / "unseen_diversity.csv"
        uq_path = out_dir / "unseen_quality.csv"
        with open(ud_path, "w", newline="", encoding="utf-8") as fdu, open(uq_path, "w", newline="", encoding="utf-8") as fqu:
            udw = csv.DictWriter(fdu, fieldnames=tb.UNSEEN_DIV_FIELDS)
            uqw = csv.DictWriter(fqu, fieldnames=tb.UNSEEN_QUAL_FIELDS)
            udw.writeheader()
            uqw.writeheader()
            for i, it in enumerate(unseen_list):
                if i < n_udiv:
                    ud_row = {"utt_id": it["utt_id"], "speaker_id": it["speaker_id"], "condition": "clean",
                              "n_seeds": tb.UNSEEN_SEEDS, "case_seconds": 1.0, "utc": "2026-11-14T00:00:00Z"}
                    for arm in tb.UNSEEN_ARMS:
                        ud_row.update({
                            f"cvf0_{arm}": 0.088 if arm in ("TBRG", "W8R2", "W6R2") else 0.077,
                            f"voiced_{arm}": 0.60,
                            f"mel_spread_{arm}": 0.940 if arm != "W4R2" else 1.030,
                            f"slow_corr_{arm}": 0.72 if arm in ("TBRG", "W8R2") else 0.08,
                            f"f0_corr_w8r2_{arm}": 0.45 if arm == "TBRG" else 0.05,
                            f"f0_shift_corr_w8r2_{arm}": 0.04,
                        })
                    udw.writerow(ud_row)
                if i < n_uqual:
                    uq_row = {
                        "utt_id": it["utt_id"], "speaker_id": it["speaker_id"], "condition": "clean",
                        "prompt_snr": 32.0, "gpu": "Fake-T4",
                        "d_1": 0.2, "d_2": 0.15, "d_3": 0.1, "d_4": 0.05, "A_4": 0.5, "rho_4": 0.5,
                        "case_seconds": 1.0, "utc": "2026-11-14T00:00:00Z",
                    }
                    for arm in tb.UNSEEN_ARMS:
                        uq_row.update({
                            f"wer_{arm}": 0.016,
                            f"similarity_{arm}": 0.932,
                            f"utmos_{arm}": 3.76 if arm in ("TBRG", "W4", "W8") else (3.40 if arm == "W4R2" else 3.65),
                            f"hypothesis_{arm}": "ok",
                        })
                    uqw.writerow(uq_row)

    return e27_div, e27_qual


class AnalyzeAndStacTests(unittest.TestCase):
    def test_confirmed_verdict_unseen_and_secondary_stac(self):
        def spec(i):
            j = 0.001 * ((i % 4) - 1.5)
            return {
                "cvf0": 0.088 + j, "mel_spread": 0.940 + j, "slow_corr": 0.75,
                "wer": 0.016, "similarity": 0.932, "utmos": 3.75 if i % 2 == 0 else 3.72,
                "r4": 0.04 if i % 2 == 0 else 0.18, "jitter": j,
            }

        with tempfile.TemporaryDirectory() as tmp:
            e27_d, e27_q = _write_synthetic_dataset(tmp, spec)
            res = tb.analyze(tmp, exp027_div_csv=e27_d, exp027_qual_csv=e27_q)
            self.assertEqual(res["verdict"], "confirmed")
            self.assertEqual(res["primary_prospective_verdict"], "confirmed")
            self.assertFalse(res["partial"])
            self.assertEqual(res["n_diversity"], 20)
            self.assertEqual(res["n_quality"], 40)
            self.assertIsNotNone(res["unseen_replication"])
            self.assertEqual(res["unseen_replication"]["verdict"], "confirmed")
            self.assertEqual(res["unseen_replication"]["n_diversity"], 20)
            self.assertEqual(res["unseen_replication"]["n_quality"], 30)
            self.assertGreater(res["unseen_replication"]["contrasts"]["TBRG_f0_corr_matched_minus_shifted"]["boot"]["mean"], 0.3)
            self.assertTrue((Path(tmp) / "TRANSMISSIBILITY_BRIDGE_REPORT.md").exists())
            self.assertTrue((Path(tmp) / "transmissibility_bridge_results.json").exists())
            self.assertTrue((Path(tmp) / "MANIFEST.json").exists())
            stac = res["stac_routing_secondary"]["relative_r4"]
            self.assertLessEqual(stac["mean_cost"], 20.0)
            self.assertGreaterEqual(stac["mean_cost"], 13.0)

        def split_spec(i):
            j = 0.001 * ((i % 4) - 1.5)
            settled = (i % 2 == 0)
            return {
                "cvf0": 0.088 + j, "mel_spread": 0.940 + j, "wer": 0.016, "similarity": 0.932,
                "utmos": 3.77 if settled else 3.40, "r4": 0.04 if settled else 0.22, "jitter": j,
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

    def test_supported_refuted_mixed_and_partial_verdicts(self):
        # supported_direction_only: H1 & H2 beat W4/W4R2, but H5 SHUF not beaten (or UTMOS mean d < -0.08)
        def sup(i):
            j = 0.001 * ((i % 4) - 1.5)
            return {"cvf0": 0.088 + j, "mel_spread": 0.940 + j, "wer": 0.016, "similarity": 0.932, "utmos": 3.55, "jitter": j}

        with tempfile.TemporaryDirectory() as tmp:
            e27_d, e27_q = _write_synthetic_dataset(tmp, sup, with_unseen=False)
            res = tb.analyze(tmp, exp027_div_csv=e27_d, exp027_qual_csv=e27_q)
            self.assertEqual(res["verdict"], "supported_direction_only")
            self.assertIsNone(res["unseen_replication"])

        # refuted: TBRG fails W8R2 CVf0 non-inferiority margin (-0.008) / W4
        def ref(i):
            j = 0.001 * ((i % 4) - 1.5)
            return {"cvf0": 0.060 + j, "mel_spread": 0.940 + j, "wer": 0.016, "similarity": 0.932, "utmos": 3.74, "jitter": j}

        with tempfile.TemporaryDirectory() as tmp:
            e27_d, e27_q = _write_synthetic_dataset(tmp, ref, with_unseen=False)
            self.assertEqual(tb.analyze(tmp, exp027_div_csv=e27_d, exp027_qual_csv=e27_q)["verdict"], "refuted")

        # mixed + partial + incomplete unseen check (CI crosses 0 vs W4, within -0.008 upper CI of W8R2 -> mixed)
        def mix(i):
            j = 0.001 * ((i % 4) - 1.5)
            return {"cvf0": 0.080 + (0.007 if i % 2 else -0.007), "mel_spread": 0.940, "wer": 0.016, "similarity": 0.932, "utmos": 3.72, "jitter": j}

        with tempfile.TemporaryDirectory() as tmp:
            e27_d, e27_q = _write_synthetic_dataset(tmp, mix, n_div=6, n_qual=10, with_unseen=True, n_udiv=4, n_uqual=5)
            with self.assertRaises(RuntimeError):
                tb.analyze(tmp, allow_partial=False, exp027_div_csv=e27_d, exp027_qual_csv=e27_q)
            res = tb.analyze(tmp, allow_partial=True, exp027_div_csv=e27_d, exp027_qual_csv=e27_q)
            self.assertTrue(res["partial"])
            self.assertEqual(res["verdict"], "mixed")
            self.assertIsNotNone(res["unseen_replication"])


class PreregistrationTests(unittest.TestCase):
    def test_repo_preregistration_valid(self):
        doc = tb.check_prereg()
        self.assertEqual(doc["experiment"], "EXP-028")
        self.assertEqual(doc["task_id"], "EVAL-021")
        self.assertIn("theory-motivated", doc["framing"])
        self.assertIn("cohort_hierarchy", doc)
        self.assertIn("multiplicity_rule", doc)
        self.assertIn("outcome_interpretation_matrix", doc)
        self.assertEqual(len(doc["pre_synthesis_amendment_chain"]), 2)
        self.assertEqual(doc["n_fresh_div"], 20)
        self.assertEqual(doc["n_fresh_qual"], 40)
        self.assertEqual(doc["n_unseen_div"], 20)
        self.assertEqual(doc["n_unseen_qual"], 30)
        self.assertEqual(doc["unseen_seeds"], 15)
        self.assertEqual(tuple(doc["new_arms"]), tb.NEW_ARMS)
        self.assertEqual(tuple(doc["unseen_arms"]), tb.UNSEEN_ARMS)

    def test_tampered_preregistration_code_or_input_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            copy_path = Path(tmp) / "preregistration.json"
            shutil.copy2(tb.PREREG_PATH, copy_path)
            tb.check_prereg(copy_path)

            data = json.loads(copy_path.read_text(encoding="utf-8"))
            data["n_fresh_qual"] = 99
            copy_path.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaises(RuntimeError):
                tb.check_prereg(copy_path)

            shutil.copy2(tb.PREREG_PATH, copy_path)
            data = json.loads(copy_path.read_text(encoding="utf-8"))
            payload = {k: v for k, v in data.items() if k not in ("sha256", "created_utc")}
            payload["code_hashes"]["run_transmissibility_bridge.py"] = "0" * 64
            payload_sha = tb._sha256_bytes(json.dumps(payload, sort_keys=True).encode("utf-8"))
            copy_path.write_text(json.dumps({**payload, "sha256": payload_sha, "created_utc": data["created_utc"]}),
                                 encoding="utf-8")
            with self.assertRaises(RuntimeError):
                tb.check_prereg(copy_path)

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
        self.assertIn('STAGE = "all"', all_src)
        self.assertIn("unseen_diversity.csv", all_src)
        self.assertIn("unseen_quality.csv", all_src)
        for cell in nb["cells"]:
            if cell["cell_type"] == "code":
                ast.parse("".join(cell["source"]))


if __name__ == "__main__":
    unittest.main()
