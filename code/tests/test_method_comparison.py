"""Unit tests for EXP-031 (single-formula handovers against the previous handover methods). They run without PyTorch or
the TTS models."""
import ast
import csv
import json
import re
import sys
import tempfile
import typing
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code/colab"))
import run_consistency_gate_prelim as cg  # noqa: E402
import run_followup_tests as rf  # noqa: E402
import run_method_comparison as mc  # noqa: E402
import run_selective_handover_prelim as sh  # noqa: E402
import run_timing_only_handover as toh  # noqa: E402


def smooth_signal(rng, frames, groups=10, pad=40):
    base = np.cumsum(rng.normal(size=(frames + 2 * pad, groups)), axis=0)
    base = np.stack([np.convolve(base[:, j], np.ones(9) / 9, mode="same") for j in range(groups)], axis=1)
    return np.repeat(base[pad:pad + frames], 10, axis=1)


class RandomMapTests(unittest.TestCase):
    def test_constraints_hold_by_construction(self):
        rng = np.random.default_rng(0)
        for _ in range(300):
            n, seed = int(rng.integers(20, 1200)), int(rng.integers(0, 10 ** 6))
            band = toh.band_width(n)
            w, shrink = mc.random_map(n, seed, band=band)
            self.assertTrue(mc.random_map_ok(w, n, band), (n, seed))
            self.assertTrue(0.0 < shrink <= 1.0)

    def test_deterministic_per_seed_and_different_across_seeds(self):
        a, _ = mc.random_map(400, 5)
        b, _ = mc.random_map(400, 5)
        c, _ = mc.random_map(400, 6)
        np.testing.assert_array_equal(a, b)
        self.assertFalse(np.array_equal(a, c))

    def test_size_matches_exp030_toh3_maps(self):
        rng = np.random.default_rng(1)
        rms, ratio = [], []
        for t in range(400):
            n = int(rng.integers(250, 750))
            r, m = toh.map_stats(mc.random_map(n, t)[0])
            rms.append(r / n)
            ratio.append(m / r)
        clean = [r for r in rf.read_csv(mc.EXP030_QUAL_CSV) if r["condition"] == "clean"]
        toh_frac = float(np.mean([float(r["warp_rms_3"]) / float(r["frames"]) for r in clean]))
        self.assertAlmostEqual(toh_frac, 0.0270, places=3)
        self.assertLess(abs(float(np.mean(rms)) - toh_frac), 0.003)
        self.assertTrue(1.7 <= float(np.mean(ratio)) <= 2.1)

    def test_short_inputs_give_the_identity(self):
        for n in (1, 2):
            w, _ = mc.random_map(n, 0)
            np.testing.assert_array_equal(w, np.arange(n, dtype=float))


class QuantileHandoverTests(unittest.TestCase):
    def test_values_are_the_students_and_order_is_the_teachers(self):
        rng = np.random.default_rng(7)
        S, T = rng.normal(size=(200, 100)), rng.normal(size=(200, 100))
        P = mc.quantile_handover(S, T)
        np.testing.assert_array_equal(np.sort(P, axis=0), np.sort(S, axis=0))
        np.testing.assert_array_equal(np.argsort(P, axis=0, kind="stable"), np.argsort(T, axis=0, kind="stable"))

    def test_draft_onto_itself_is_the_draft(self):
        rng = np.random.default_rng(8)
        S = rng.normal(size=(150, 100))
        np.testing.assert_array_equal(mc.quantile_handover(S, S), S)
        S[10:20, 3] = S[5, 3]   # ties are handled by the stable sort
        np.testing.assert_array_equal(mc.quantile_handover(S, S), S)

    def test_invariant_to_any_stationary_channel_and_noise_floor(self):
        """g_b(x) = log(h_b e^x + nu_b) is strictly increasing per band, so the payload is unchanged."""
        rng = np.random.default_rng(9)
        S, T = rng.normal(size=(300, 100)), rng.normal(size=(300, 100))
        h, nu = rng.uniform(0.2, 3.0, size=100), rng.uniform(0.0, 2.0, size=100)
        noisy = np.log(h * np.exp(T) + nu)
        np.testing.assert_array_equal(mc.quantile_handover(S, noisy), mc.quantile_handover(S, T))

    def test_payload_keeps_prompt_frames(self):
        rng = np.random.default_rng(10)
        s0, tk = rng.normal(size=(1, 140, 100)), rng.normal(size=(1, 140, 100))
        out = mc.quantile_payload(s0, tk, 40)
        np.testing.assert_array_equal(out[:, :40, :], s0[:, :40, :])
        np.testing.assert_array_equal(np.sort(out[0, 40:], axis=0), np.sort(s0[0, 40:], axis=0))

    def test_map_helpers(self):
        w = np.linspace(0, 99, 120)
        self.assertEqual(len(mc.resample_map(w, 100)), 100)
        np.testing.assert_array_equal(mc.resample_map(np.arange(50.0), 50), np.arange(50.0))
        self.assertTrue(mc.toh_map_ok(np.arange(80, dtype=float), 80, 8))
        self.assertFalse(mc.toh_map_ok(np.arange(80, dtype=float) * 0.5, 80, 8))


class _FT(np.ndarray):
    """A numpy array with the few tensor methods the blind-teacher code calls."""
    def clone(self):
        return self.copy().view(_FT)

    def unsqueeze(self, d):
        return np.expand_dims(self, d).view(_FT)

    def to(self, device):
        return self


class _FakeTorch:
    float32 = np.float32

    @staticmethod
    def zeros(*shape, dtype=None, device=None):
        return np.zeros(shape).view(_FT)

    @staticmethod
    def randn(*shape, dtype=None, generator=None):
        return generator.standard_normal(shape).view(_FT)

    @staticmethod
    def tensor(x, device=None, dtype=None):
        return float(x)

    @staticmethod
    def inference_mode():
        import contextlib
        return contextlib.nullcontext()


class _FakePipe:
    """Stands in for rf.Pipeline: records what reaches the input routine and the teacher."""
    def __init__(self):
        import math as _m
        self.torch, self.device, self.calls, self.cond_shapes = _FakeTorch(), "cpu", [], []
        self.convert_char_to_pinyin = lambda texts: texts
        grid = [1 - _m.cos(_m.pi / 2 * i / 16) for i in range(17)]
        self.sampler = type("S", (), {"CFG_STRENGTH": 2.0, "SWAY_TEACHER_GRID": grid,
                                      "make_local_generator": staticmethod(lambda seed: np.random.default_rng(seed))})()
        pipe = self

        class TTS:
            @staticmethod
            def _prepare_inputs(cond, text, n):
                pipe.cond_shapes.append(tuple(cond.shape))
                return cond, "ids", np.zeros((1, n, 100)).view(_FT), None, n, n

            @staticmethod
            def teacher(drop_audio_cond, drop_text, x, cond, text, time, mask):
                pipe.calls.append((drop_audio_cond, drop_text))
                return 0.1 * x + (0.0 if drop_text else 0.05)

        self.tts = TTS()


class BlindTeacherTests(unittest.TestCase):
    def test_blind_teacher_never_receives_the_prompt(self):
        """Amendment 2 regression: the placeholder is a one-frame mel (no STFT), and both guidance branches drop the
        audio condition, so nothing from the prompt reaches the teacher."""
        pipe = _FakePipe()
        bctx = mc.blind_context(pipe, {"gt_text": "hello world"}, 120, 5)
        self.assertEqual(pipe.cond_shapes, [(1, 1, 100)])
        self.assertEqual(bctx["m"], 120)
        self.assertEqual(tuple(bctx["y0"].shape), (1, 120, 100))
        ep = mc.blind_teacher_endpoint(pipe, bctx)
        self.assertEqual(tuple(ep.shape), (1, 120, 100))
        self.assertEqual(len(pipe.calls), 2 * (mc.HANDOVER_STEP + 1))
        self.assertTrue(all(drop_audio for drop_audio, _ in pipe.calls))
        self.assertEqual({drop_text for _, drop_text in pipe.calls}, {True, False})
        again = mc.blind_teacher_endpoint(pipe, mc.blind_context(pipe, {"gt_text": "hello world"}, 120, 5))
        np.testing.assert_array_equal(ep, again)


class PolicyTests(unittest.TestCase):
    def test_constants_match_their_sources(self):
        swsr = (ROOT / "code/colab/run_swsr_handover_prelim.py").read_text(encoding="utf-8")
        found = re.search(r"^DEFAULT_STAC_LAMS = (\{.*\})$", swsr, re.M)
        self.assertEqual(ast.literal_eval(found.group(1)), mc.STAC_LAMS)
        sampler = (ROOT / "code/scheduling/sampler.py").read_text(encoding="utf-8")
        self.assertIn("CANDIDATE_STEPS: Tuple[int, ...] = (4, 5, 6)", sampler)
        self.assertIn("REFERENCE_STEP: int = 8", sampler)
        self.assertIn("snr_gate_threshold: float = 20.0", sampler)
        self.assertEqual(cg.THETA, mc.THETA)
        self.assertEqual(tuple(cg.CANDS), mc.CH3_CANDIDATES)

    def test_ch3_matches_the_released_policy_function(self):
        """scheduling/sampler.evaluate_handover_policy, extracted from its source (the module needs PyTorch)."""
        tree = ast.parse((ROOT / "code/scheduling/sampler.py").read_text(encoding="utf-8"))
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "evaluate_handover_policy")
        ns = {"Dict": typing.Dict, "Tuple": typing.Tuple, "CANDIDATE_STEPS": (4, 5, 6), "REFERENCE_STEP": 8}
        exec(compile(ast.Module(body=[fn], type_ignores=[]), "sampler_policy", "exec"), ns)
        released = ns["evaluate_handover_policy"]
        rng = np.random.default_rng(2)
        for _ in range(2000):
            drifts = {j: float(rng.uniform(0.0, 1.0)) for j in range(1, 9)}
            snr = float(rng.uniform(5.0, 45.0))
            for lam in (mc.CH3_LAMBDA, mc.CH3_LAMBDA_LTT, float(rng.uniform(0.0, 1.0))):
                self.assertEqual(mc.ch3_step(drifts, snr, lam), released(drifts, snr, lam))

    def test_stac_semantics(self):
        self.assertEqual(mc.stac_step({j: 0.4 for j in range(1, 9)}), 4)     # A_4 = 1.6 >= 1.513
        self.assertEqual(mc.stac_step({j: 0.3 for j in range(1, 9)}), 6)     # A_5 = 1.5 < 1.647, A_6 = 1.8 >= 1.716
        self.assertEqual(mc.stac_step({j: 0.1 for j in range(1, 9)}), 8)

    def test_routing(self):
        low = {j: 0.1 for j in range(1, 9)}
        self.assertEqual(mc.routed_arm("SNR_TOH", 20.0, low), "TOH3")
        self.assertEqual(mc.routed_arm("SNR_TOH", 19.99, low), "STU")
        self.assertEqual(mc.routed_arm("SNR_W4", 30.0, low), "W4")
        self.assertEqual(mc.routed_arm("SNR_RND", 10.0, low), "STU")
        self.assertEqual(mc.routed_arm("CH3", 30.0, low), "W4")
        self.assertEqual(mc.routed_arm("CH3", 10.0, low), "W8")
        self.assertEqual(mc.routed_arm("CH3", 30.0, {j: 0.9 for j in range(1, 9)}), "W8")
        self.assertEqual(mc.routed_arm("CH3_LTT", 30.0, {j: 0.9 for j in range(1, 9)}), "W4")
        self.assertEqual(mc.routed_arm("STAC", 10.0, low), "W8")
        for x in mc.CANDIDATES:
            self.assertEqual(mc.routed_arm(x, 5.0, low), x)   # candidates are applied to every input
        for method in mc.METHODS:
            self.assertIn(mc.routed_arm(method, 25.0, low), mc.COST)


class AlignedVarietyTests(unittest.TestCase):
    def _seeds(self, warps):
        rng = np.random.default_rng(3)
        n = 300
        mel = smooth_signal(rng, n)
        t = np.arange(n + 1, dtype=float)
        f0 = 150.0 + 25.0 * np.sin(2 * np.pi * t / 70.0) + 10.0 * np.sin(2 * np.pi * t / 23.0)
        tracks, mels = [], []
        for w in warps(n):
            mels.append(toh.warp_frames(mel, w))
            tracks.append(np.interp(np.concatenate([w, [n]]), t, f0))
        return tracks, mels

    def test_timing_differences_vanish_after_alignment(self):
        rng = np.random.default_rng(4)

        def warps(n):
            i = np.arange(n, dtype=float)
            return [i] + [i + rng.uniform(5, 12) * np.sin(np.pi * i / (n - 1) * k) for k in (1, 2, 1, 2, 3)]

        tracks, mels = self._seeds(warps)
        frame_cv, _ = sh.cvf0(tracks)
        aligned_cv, voiced, spread = mc.aligned_cvf0(tracks, mels)
        self.assertGreater(frame_cv, 4 * aligned_cv)
        self.assertGreater(spread, 2.0)
        self.assertGreater(voiced, 0.9)

    def test_identical_seeds_give_zero(self):
        tracks, mels = self._seeds(lambda n: [np.arange(n, dtype=float)] * 4)
        cv, _, spread = mc.aligned_cvf0(tracks, mels)
        self.assertAlmostEqual(cv, 0.0, places=12)
        self.assertEqual(spread, 0.0)


class DesignTests(unittest.TestCase):
    def test_costs(self):
        for k in (4, 5, 6, 8):
            self.assertEqual(mc.COST[f"W{k}"], 2 * k + 5)
        for x in mc.CANDIDATES:
            self.assertEqual(mc.COST[x], 2 * mc.HANDOVER_STEP + 6)
        self.assertEqual((mc.COST["STU"], mc.COST["RND"]), (4, 4))
        self.assertTrue(set(mc.DIV_ARMS) <= set(mc.ARMS))
        self.assertTrue(set(mc.CANDIDATES) <= set(mc.DIV_ARMS))

    def test_cohorts_are_new_to_toh_and_outside_calibration_and_test(self):
        q = [it["utt_id"] for it in mc.qual_items()]
        d = [it["utt_id"] for it in mc.div_items()]
        self.assertEqual(len(q), 30)
        self.assertEqual(d, q[:20])
        self.assertFalse(set(q) & {it["utt_id"] for it in toh.qual_items()})
        pilot = {r["utt_id"] for r in rf.read_csv(mc.EXP011_PILOT_CSV)}
        self.assertEqual(set(q[:20]), pilot)
        cal = json.loads((ROOT / "research/protocols/splits/calibration_manifest.json").read_text(encoding="utf-8"))
        test = json.loads((ROOT / "research/protocols/splits/test_manifest.json").read_text(encoding="utf-8"))
        spk = {it["speaker_id"] for it in mc.qual_items()}
        self.assertFalse(spk & {r["speaker_id"] for r in cal})
        self.assertFalse(spk & {r["speaker_id"] for r in test})

    def test_conditions_and_dependencies(self):
        for c in mc.CONDS:
            self.assertIn(c, rf.CONDITIONS)
        self.assertTrue(set(mc.HIDDEN) <= set(mc.DEGRADED))
        for name in mc.DEPENDENCIES:
            self.assertTrue((mc.CODE_DIR / name).exists(), name)
        self.assertIn("run_transmissibility_test.py", mc.DEPENDENCIES)   # the blind teacher's grid comes from it


SNR = {"clean": 35.0, "gauss25": 27.0, "noisy15": 18.0, "reverb03": 28.0}


def write_synthetic(tmp, profile, with_div=True):
    """Quality and diversity CSVs whose arms behave as `profile` says (TOH3, QH3 and BTH3 alike unless noted)."""
    rng = np.random.default_rng(11)
    drifts = {j: 0.3 for j in range(1, 9)}   # STAC -> W6; CH3 -> W4 above the gate, W8 below
    q = Path(tmp) / "quality.csv"
    with open(q, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=mc.QUAL_FIELDS)
        w.writeheader()
        for idx, it in enumerate(mc.qual_items()):
            for cond in mc.CONDS:
                snr = SNR.get(cond, 22.0 if idx % 2 else 18.0)
                row = {"utt_id": it["utt_id"], "speaker_id": it["speaker_id"], "condition": cond, "prompt_snr": snr,
                       "gpu": "UnitTest", "frames": 400, "seq_len": 800, "blind_seq_len": 400, "warp_band": 40,
                       "toh_rms": 11.0, "toh_max": 22.0, "bth_rms": 11.5, "bth_max": 23.0, "qh_shift": 1.0,
                       "rnd_rms": 10.5, "rnd_max": 20.0, "rnd_shrink": 1.0, **{f"d_{j}": v for j, v in drifts.items()},
                       "stac_step": mc.stac_step(drifts), "ch3_step": mc.ch3_step(drifts, snr, mc.CH3_LAMBDA),
                       "ch3_ltt_step": mc.ch3_step(drifts, snr, mc.CH3_LAMBDA_LTT), "case_seconds": 1.0,
                       "utc": "2026-11-15T00:00:00+00:00"}
                for arm in mc.ARMS:
                    student_like = arm in ("STU", "RND") + mc.CANDIDATES
                    if cond == "clean":
                        utm = 3.9
                    elif cond in mc.HIDDEN and profile == "no_safety":
                        utm = 2.0
                    else:
                        utm = 3.3 if student_like or (arm == "W4" and profile == "w4_robust") else 1.9
                    if arm == "TOH3" and cond == "clean" and profile == "toh_clean_loss":
                        utm = 3.6
                    if arm == "RND" and cond == "clean":
                        utm = 3.95 if profile == "random_equal" else 3.6   # random at least as good: no chance pass
                    row.update({f"wer_{arm}": 0.0, f"similarity_{arm}": 0.93,
                                f"utmos_{arm}": utm + float(rng.normal(0.0, 0.02)), f"hypothesis_{arm}": "x"})
                w.writerow(row)
    if with_div:
        d = Path(tmp) / "diversity.csv"
        with open(d, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=mc.DIV_FIELDS)
            w.writeheader()
            for it in mc.div_items():
                row = {"utt_id": it["utt_id"], "speaker_id": it["speaker_id"], "condition": "clean", "n_seeds": 12,
                       "toh_rms_mean": 11.0, "bth_rms_mean": 11.5, "rnd_rms_mean": 10.5, "case_seconds": 1.0,
                       "utc": "x"}
                for arm in mc.DIV_ARMS:
                    cv = 0.05 if arm == "STU" else 0.09
                    if arm == "RND":
                        cv = 0.095 if profile == "random_equal" else 0.05
                    row.update({f"cvf0_{arm}": cv + float(rng.normal(0.0, 0.001)), f"voiced_{arm}": 0.6,
                                f"mel_spread_{arm}": 0.3, f"cvf0_aligned_{arm}": 0.04 + float(rng.normal(0.0, 0.001)),
                                f"voiced_aligned_{arm}": 0.6, f"timing_spread_{arm}": 8.0})
                w.writerow(row)


class AnalysisTests(unittest.TestCase):
    def _run(self, profile, with_div=True, allow_partial=False):
        with tempfile.TemporaryDirectory() as tmp:
            prereg = Path(tmp) / "preregistration.json"
            prereg.write_text(json.dumps(mc.build_prereg(), indent=2) + "\n", encoding="utf-8")
            write_synthetic(tmp, profile, with_div)
            res = mc.analyze(tmp, prereg_path=prereg, allow_partial=allow_partial)
            self.assertTrue((Path(tmp) / "METHOD_COMPARISON_REPORT.md").exists())
            self.assertTrue((Path(tmp) / "MANIFEST.json").exists())
            self.assertEqual(res["policy_steps"]["mismatches_with_recorded"], 0)
            self.assertAlmostEqual(res["candidate_contrasts"]["bth_teacher_sequence_share"], 0.5)
            return res["verdicts"]

    def test_verdicts(self):
        for x, v in self._run("works").items():
            self.assertEqual((v["teacher"], v["method"], v["condition"]),
                             ("teacher_needed", "beats_all_previous", "condition_safety_shown"), x)
        for v in self._run("random_equal").values():
            self.assertEqual(v["teacher"], "teacher_not_shown_needed")
        for v in self._run("no_safety").values():
            self.assertEqual((v["method"], v["condition"]), ("beats_all_previous", "condition_safety_not_shown"))
        v = self._run("toh_clean_loss")
        self.assertEqual(v["TOH3"]["method"], "beats_no_previous")
        self.assertEqual((v["QH3"]["method"], v["BTH3"]["method"]), ("beats_all_previous", "beats_all_previous"))
        for v in self._run("w4_robust").values():
            self.assertEqual(v["method"], "beats_some_previous")
            self.assertEqual(v["previous_beaten"], ["W8", "STAC", "CH3", "CH3_LTT"])
        for v in self._run("works", with_div=False, allow_partial=True).values():
            self.assertEqual(v["teacher"], "partial")
        with self.assertRaises(RuntimeError):
            self._run("works", with_div=False, allow_partial=False)


class PreregistrationAndNotebookTests(unittest.TestCase):
    def test_tampering_is_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "preregistration.json"
            doc = mc.build_prereg()
            p.write_text(json.dumps(doc), encoding="utf-8")
            mc.check_prereg(p)
            doc["routing"]["theta_db"] = 15.0
            p.write_text(json.dumps(doc), encoding="utf-8")
            with self.assertRaises(RuntimeError):
                mc.check_prereg(p)

    def test_committed_preregistration_and_notebook(self):
        if mc.PREREG_PATH.exists():
            doc = mc.check_prereg()
            self.assertEqual(doc["experiment"], "EXP-031")
            self.assertEqual(doc["task_id"], "EVAL-024")
            self.assertEqual(doc["candidates"], list(mc.CANDIDATES))
            self.assertEqual(doc["amendments"][0]["first_freeze_sha256"], "ec06ed0568b4")
        nb_path = ROOT / "code/colab/DMOSpeech2_EXP031_SNR_TOH_Method_Comparison.ipynb"
        nb = json.loads(nb_path.read_text(encoding="utf-8"))
        code = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
        for i, src in enumerate(code):
            compile("\n".join(l for l in src.splitlines() if not l.lstrip().startswith(("!", "%"))), f"cell{i}", "exec")
        joined = "\n".join(code)
        self.assertIn("run_method_comparison.py", joined)
        self.assertIn("EXP-031-snr-toh-method-comparison", joined)
        self.assertIn("--demand_dir", joined)
        self.assertIn("PCAFETER_16k.zip", joined)
        self.assertNotIn("EXP-030", joined)
        self.assertIn('STAGE = "all"', joined)
        md = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "markdown")
        self.assertIn("QH3", md)
        self.assertIn("BTH3", md)


if __name__ == "__main__":
    unittest.main()
