"""Tests for the prototype's SNR + drift file (code/prototype/snr_drift.py) and the engine's pure helpers. They check the
prototype against the thesis's own code: code/scheduling/sampler.py (functions extracted from its source, so PyTorch is
not needed) and the experiments' Chapter 3 policy. The PyTorch comparison of the drift runs only where PyTorch exists."""
import ast
import json
import math
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code/prototype"))
sys.path.insert(0, str(ROOT / "code/colab"))
import engine as eng  # noqa: E402
import run_followup_tests as rf  # noqa: E402
import run_grid32_prelim as gl  # noqa: E402
import run_method_comparison as mc  # noqa: E402
import snr_drift as sd  # noqa: E402

SAMPLER = ROOT / "code/scheduling/sampler.py"


def sampler_function(name, namespace):
    """One function from code/scheduling/sampler.py, compiled without importing the module (which needs PyTorch)."""
    tree = ast.parse(SAMPLER.read_text(encoding="utf-8"))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    src = "from typing import Dict, Tuple\n" + ast.get_source_segment(SAMPLER.read_text(encoding="utf-8"), node)
    ns = dict(namespace)
    exec(compile(src, str(SAMPLER), "exec"), ns)
    return ns[name]


def speech_like(rng, n, sr):
    t = np.arange(n) / sr
    env = (np.sin(2 * np.pi * 2.3 * t) > 0).astype(np.float32) * 0.9 + 0.05
    return (0.3 * env * np.sin(2 * np.pi * 180 * t) * (1 + 0.3 * rng.normal(size=n))).astype(np.float32)


class ConstantsTests(unittest.TestCase):
    def test_grids_costs_and_presets(self):
        self.assertEqual(len(sd.TEACHER_GRID), 17)
        for a, b in zip(sd.TEACHER_GRID, gl.sway_grid(16)):
            self.assertAlmostEqual(a, b)
        for a, b in zip(sd.STUDENT_TAUS, (0.07612046748871326, 0.2928932188134524, 0.6173165676349102)):
            self.assertAlmostEqual(a, b)
        for k in (4, 5, 6, 8):
            self.assertEqual(sd.passes(k), mc.COST[f"W{k}"])
        self.assertEqual(sd.REFERENCE_PASSES, 21)
        self.assertEqual((sd.SNR_GATE_DB, sd.CANDIDATE_STEPS, sd.REFERENCE_STEP), (mc.THETA, mc.CH3_CANDIDATES, mc.CH3_REFERENCE))
        self.assertEqual(sd.LAMBDA_PRESETS["cv"]["value"], mc.CH3_LAMBDA)
        self.assertEqual(sd.LAMBDA_PRESETS["ltt"]["value"], mc.CH3_LAMBDA_LTT)
        self.assertEqual(sd.DEFAULT_LAMBDA, 0.35)


class SnrTests(unittest.TestCase):
    def test_matches_the_thesis_sampler(self):
        ref = sampler_function("estimate_prompt_snr", {"np": np})
        rng = np.random.default_rng(0)
        for sr in (16000, 24000, 44100):
            clean = speech_like(rng, sr * 4, sr)
            for x in (clean, rf.add_gaussian(clean, 25.0, 7), rf.add_gaussian(clean, 15.0, 7), clean[: sr // 100]):
                self.assertEqual(sd.estimate_prompt_snr(x, sr), ref(x, sample_rate=sr))
        self.assertEqual(sd.estimate_prompt_snr(np.zeros(10, dtype=np.float32), 24000), 0.0)

    def test_white_noise_lowers_it(self):
        rng = np.random.default_rng(1)
        clean = speech_like(rng, 24000 * 4, 24000)
        snr = [sd.estimate_prompt_snr(rf.add_gaussian(clean, db, 3), 24000) for db in (40, 25, 15)]
        self.assertGreater(snr[0], snr[1])
        self.assertGreater(snr[1], snr[2])


class DriftTests(unittest.TestCase):
    def test_numpy_formula(self):
        rng = np.random.default_rng(2)
        a, b = rng.normal(size=(1, 50, 100)), rng.normal(size=(1, 50, 100))
        plen = 20
        want = np.linalg.norm((a - b)[:, plen:]) / (np.linalg.norm(a[:, plen:]) + 1e-8)
        self.assertAlmostEqual(sd.stabilization_drift(a, b, plen), want)
        self.assertEqual(sd.stabilization_drift(a, a, plen), 0.0)
        c = a.copy()
        c[:, :plen] += 100.0                      # prompt frames do not count
        self.assertAlmostEqual(sd.stabilization_drift(c, b, plen), want)

    @unittest.skipUnless(__import__("importlib").util.find_spec("torch"), "PyTorch not installed")
    def test_torch_matches_the_thesis_sampler(self):
        import torch
        ref = sampler_function("compute_stabilization_drift", {"torch": torch})
        g = torch.Generator().manual_seed(0)
        a, b = torch.randn(1, 60, 100, generator=g), torch.randn(1, 60, 100, generator=g)
        self.assertEqual(sd.stabilization_drift(a, b, 25), ref(a, b, 25))


class RuleTests(unittest.TestCase):
    def cases(self, n=3000):
        rng = np.random.default_rng(3)
        for _ in range(n):
            trace = {k: float(rng.uniform(0.05, 1.5)) for k in range(1, 9)}
            yield trace, float(rng.uniform(5.0, 45.0)), float(rng.choice([0.2, 0.35, 0.5, 1.2, rng.uniform(0, 1.5)]))

    def test_matches_chapter_3_policy_and_the_sampler(self):
        ref = sampler_function("evaluate_handover_policy", {"CANDIDATE_STEPS": (4, 5, 6), "REFERENCE_STEP": 8})
        for trace, snr, lam in self.cases():
            k = sd.decide(trace, snr, lam)
            self.assertEqual(k, mc.ch3_step(trace, snr, lam))
            self.assertEqual(k, ref(trace, snr, lam))

    def test_online_monitor_gives_the_same_step(self):
        for trace, snr, lam in self.cases():
            mon = sd.HandoverMonitor(snr, lam)
            stopped_at = None
            for i in range(9):
                if mon.observe(i, trace.get(i)):
                    stopped_at = i
                    break
            self.assertEqual(stopped_at, mon.step)
            self.assertEqual(mon.step, sd.decide(trace, snr, lam))
            s = mon.summary()
            self.assertEqual(s["passes"], sd.passes(mon.step))
            self.assertAlmostEqual(s["saving"], 1 - sd.passes(mon.step) / 21)
            self.assertTrue(s["reason"])

    def test_gate_and_threshold(self):
        low = {k: 0.01 for k in range(1, 9)}
        self.assertEqual(sd.decide(low, 19.9), 8)                    # gate closed: reference whatever the drift
        self.assertEqual(sd.decide(low, 20.0), 4)
        self.assertEqual(sd.decide({4: 0.5, 5: 0.3, 6: 0.1}, 30.0, 0.35), 5)
        self.assertEqual(sd.decide({4: 0.5, 5: 0.4, 6: 0.36}, 30.0, 0.35), 8)
        mon = sd.HandoverMonitor(12.0)
        self.assertFalse(any(mon.observe(i, 0.01 if i else None) for i in range(8)))
        self.assertTrue(mon.observe(8, 0.01))
        self.assertIn("gate closed", mon.reason)


class EngineHelperTests(unittest.TestCase):
    def test_prompt_text_rule(self):
        self.assertEqual(eng.ensure_sentence_end("Hello there"), "Hello there. ")
        self.assertEqual(eng.ensure_sentence_end("Hello there."), "Hello there. ")
        self.assertEqual(eng.ensure_sentence_end("Hello there. "), "Hello there. ")
        self.assertEqual(eng.ensure_sentence_end("你好。"), "你好。")

    def test_examples_are_development_speakers_with_experiment_seeds(self):
        items = eng.development_examples()
        self.assertEqual(len(items), 100)
        dev = json.loads((ROOT / "research/protocols/splits/development_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual({it["utt_id"] for it in items}, {r["utt_id"] for r in dev})
        seeds = {r["utt_id"]: int(r["input_seed"]) for r in dev}
        self.assertTrue(all(it["input_seed"] == seeds[it["utt_id"]] for it in items))

    def test_degradations_are_the_experiments(self):
        rng = np.random.default_rng(4)
        x = speech_like(rng, 24000 * 3, 24000)
        np.testing.assert_array_equal(eng.degrade_prompt(x, 24000, "clean", 5), x)
        np.testing.assert_array_equal(eng.degrade_prompt(x, 24000, "noisy15", 5), rf.add_gaussian(x, 15.0, 5))
        with self.assertRaises(ValueError):
            eng.degrade_prompt(x, 24000, "babble15", 5)


class InterfaceTests(unittest.TestCase):
    def test_page_uses_the_presentation_design_and_the_api(self):
        html = (ROOT / "code/prototype/web/index.html").read_text(encoding="utf-8")
        for token in ("--blue-grad", "--orange-grad", "--dark-grad", "Google+Sans+Flex", "/api/jobs", "/api/status",
                      "/api/examples", "SNR + Invariant Residual"):
            self.assertIn(token, html)
        pres = (ROOT / "presentation/index.html").read_text(encoding="utf-8")
        for token in ("#0284C7", "#EA580C", "#F8FAFC"):
            self.assertIn(token, pres)
            self.assertIn(token, html)


if __name__ == "__main__":
    unittest.main()
