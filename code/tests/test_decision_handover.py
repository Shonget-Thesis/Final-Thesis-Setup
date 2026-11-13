import csv
import hashlib
import importlib.util
import json
import math
import random
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code/colab"))
import run_decision_handover_pilot as dh  # noqa: E402
import run_slow_part_handover_prelim as sp  # noqa: E402

HAVE_NP = importlib.util.find_spec("numpy") is not None


class DesignTests(unittest.TestCase):
    def test_forward_passes(self):
        self.assertEqual({o: dh.cost(o) for o in dh.OUTS}, {"DH_2": 10, "DH_3": 12, "DHc_2": 10, "DHc_3": 12, "DHu_2": 7, "DHu_3": 8})
        self.assertEqual((dh.cost("plain_8"), dh.cost("STT_4"), dh.cost("student4")), (21, 14, 4))
        self.assertEqual((dh.cost("W3"), dh.cost("W4")), (11, 13))
        for k in dh.STEPS:
            self.assertEqual(dh.cost(f"STS_{k}"), sp.cost(f"STS_{k}"))

    def test_every_pilot_case_has_stored_labels(self):
        stored = dh.load_stored()
        cases = [(it["utt_id"], c) for it in dh.ex.items()[:dh.N_SPEAKERS] for c in dh.CONDS]
        self.assertEqual(len(cases), 60)
        self.assertTrue(all(k in stored for k in cases))
        self.assertEqual(sorted(stored[cases[0]]["m"]), sorted(["plain_8", "student4", "STT_4", "STS_2", "STS_3"]))
        self.assertEqual(len(dh.stored_sts3()), 360)

    def test_diversity_reuses_the_exp022_seeds_and_speakers(self):
        self.assertEqual(dh.DIV_SEED_BASE, sp.DIV_SEED_BASE)
        self.assertEqual(dh.DIV_SEED_BASE, dh.sh.DIV_SEED_BASE)
        cv = dh.stored_cvf0()
        self.assertTrue(all(it["speaker_id"] in cv for it in dh.ex.items()[:dh.DIV_N]))
        seed_check = {dh.SEED_BASE + m for m in range(dh.SEED_M)}
        diversity = {dh.DIV_SEED_BASE + m for m in range(dh.DIV_SEEDS)}
        self.assertFalse(seed_check & diversity)

    def test_completion_weights_come_from_the_stored_pitch_variety(self):
        rho, gain = dh.committed_shares(), dh.completion_gains()
        self.assertEqual(sorted(rho), list(dh.STEPS))
        for k in dh.STEPS:
            self.assertTrue(0.0 < rho[k] < 1.0)
            self.assertAlmostEqual(gain[k], math.sqrt(1.0 - rho[k]))
        self.assertLess(rho[2], rho[3])
        self.assertEqual(dh.completion_gains({2: -0.5, 3: 1.5}), {2: 1.0, 3: 0.0})


@unittest.skipUnless(HAVE_NP, "numpy is needed")
class FormulaTests(unittest.TestCase):
    def setUp(self):
        import numpy as np
        rng = np.random.default_rng(0)
        self.np, self.plen = np, 12
        self.T0, self.Tk, self.S = (rng.normal(size=(1, 60, 100)) for _ in range(3))

    def test_g0_is_the_slow_part_handover_of_exp022(self):
        x = dh.decision_handover(self.Tk, self.T0, self.S, self.plen, 0.0)
        self.np.testing.assert_allclose(x, dh.sh.selective_handover(self.Tk, self.S, self.plen, "STS"), atol=1e-12)

    def test_g1_adds_the_teachers_decision_to_the_draft(self):
        x = dh.decision_handover(self.Tk, self.T0, self.S, self.plen, 1.0)
        p = self.plen
        expected = self.S[0, p:] + dh.sh.decompose(self.Tk[0, p:] - self.T0[0, p:])[1]
        self.np.testing.assert_allclose(x[0, p:], expected, atol=1e-12)
        self.np.testing.assert_array_equal(x[0, :p], self.Tk[0, :p])

    def test_anything_every_seed_shares_cancels(self):
        # A component the teacher adds identically at Step 0 and Step k (for example the prompt's recording
        # conditions) leaves the handover unchanged; with g < 1 part of the teacher's own slow part remains.
        C = self.np.random.default_rng(1).normal(size=self.T0.shape) * 3.0
        a = dh.decision_handover(self.Tk, self.T0, self.S, self.plen, 1.0)
        b = dh.decision_handover(self.Tk + C, self.T0 + C, self.S, self.plen, 1.0)
        self.np.testing.assert_allclose(a[0, self.plen:], b[0, self.plen:], atol=1e-10)
        c = dh.decision_handover(self.Tk + C, self.T0 + C, self.S, self.plen, 0.5)
        self.assertGreater(float(abs(c[0, self.plen:] - dh.decision_handover(self.Tk, self.T0, self.S, self.plen, 0.5)[0, self.plen:]).max()), 1e-3)

    def test_a_decision_only_in_the_band_averages_hands_over_the_draft(self):
        Tk = self.T0 + 0.7 * self.np.arange(100.0)[None, None, :]
        x = dh.decision_handover(Tk, self.T0, self.S, self.plen, 1.0)
        self.np.testing.assert_allclose(x[0, self.plen:], self.S[0, self.plen:], atol=1e-10)


class _NoContext:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _FakeTTS:
    """A linear stand-in for the teacher: conditional and unconditional velocities differ, and both depend on x and t."""

    def __init__(self, np):
        self.np, self.calls = np, []

    def teacher(self, x, cond, text, time, mask, drop_audio_cond, drop_text):
        self.calls.append("uncond" if drop_audio_cond else "cond")
        if drop_audio_cond:
            return -0.4 * x + 0.05 * time
        return -0.5 * x + 0.1 * cond + 0.2 * time


def reference_single_pass(tts, y0, cond, grid, kmax, cfg):
    """sampler.run_teacher_single_pass written out again, for comparison."""
    state, ends = y0.copy(), {}
    for i in range(kmax + 1):
        t = grid[i]
        pred = tts.teacher(x=state, cond=cond, text=None, time=t, mask=None, drop_audio_cond=False, drop_text=False)
        null = tts.teacher(x=state, cond=cond, text=None, time=t, mask=None, drop_audio_cond=True, drop_text=True)
        v = pred + (pred - null) * cfg
        ends[i] = state + (1.0 - t) * v
        if i < kmax:
            state = state + (grid[i + 1] - t) * v
    return ends


class _FakePipe:
    gpu = "Tesla T4 (fake)"

    def __init__(self):
        import numpy as np
        self.np = np
        self.torch = type("T", (), {"inference_mode": staticmethod(_NoContext), "tensor": staticmethod(lambda v, device=None, dtype=None: v),
                                    "float32": None})
        grid = [1.0 - math.cos(math.pi / 2.0 * i / 16.0) for i in range(17)]
        drift = staticmethod(lambda a, b, plen: float(np.linalg.norm(a[0, plen:] - b[0, plen:]) / (np.linalg.norm(a[0, plen:]) + 1e-8)))
        self.sampler = type("S", (), {"STUDENT_TAUS": [0.0761, 0.2929, 0.6173], "SWAY_TEACHER_GRID": grid, "CFG_STRENGTH": 2.0,
                                      "compute_stabilization_drift": drift})
        self.snr = 30.0
        self.device = type("D", (), {"type": "cpu"})()
        self.tts = _FakeTTS(np)
        self.teacher_calls = []

    def prepare(self, item, cond, seed, out_dir, noise_seed=None):
        rng = self.np.random.default_rng(int(seed))
        return {"prompt_len": 10, "prompt_snr": self.snr, "seed": seed, "step_cond": self.np.ones((1, 50, 100)),
                "text_ids": None, "y0": rng.normal(size=(1, 50, 100))}

    def teacher(self, ctx, steps):
        self.teacher_calls.append(tuple(steps))
        return reference_single_pass(self.tts, ctx["y0"], ctx["step_cond"], self.sampler.SWAY_TEACHER_GRID, max(steps), 2.0), {}

    def ground_truth(self, item):
        return self.np.zeros(10), 24000

    def score(self, audio, item, gt):
        return {"wer": 0.1, "similarity": 0.8, "utmos": 3.0, "hypothesis": "text"}


@unittest.skipUnless(HAVE_NP, "numpy is needed")
class TeacherLoopTests(unittest.TestCase):
    def setUp(self):
        self.pipe = _FakePipe()
        self.ctx = self.pipe.prepare({}, "clean", 7, None)

    def test_the_guided_loop_is_the_released_single_pass(self):
        ends, passes = dh.teacher_loop(self.pipe, self.ctx, 4, guided=True)
        ref = reference_single_pass(self.pipe.tts, self.ctx["y0"], self.ctx["step_cond"], self.pipe.sampler.SWAY_TEACHER_GRID, 4, 2.0)
        self.assertEqual(passes, 10)
        for k in range(5):
            self.pipe.np.testing.assert_allclose(ends[k], ref[k], atol=1e-12)

    def test_the_unguided_loop_calls_only_the_conditional_branch(self):
        self.pipe.tts.calls.clear()
        ends, passes = dh.teacher_loop(self.pipe, self.ctx, 3, guided=False)
        self.assertEqual(passes, 4)
        self.assertEqual(self.pipe.tts.calls, ["cond"] * 4)
        ref = reference_single_pass(self.pipe.tts, self.ctx["y0"], self.ctx["step_cond"], self.pipe.sampler.SWAY_TEACHER_GRID, 3, 0.0)
        for k in range(4):
            self.pipe.np.testing.assert_allclose(ends[k], ref[k], atol=1e-12)


@unittest.skipUnless(HAVE_NP, "numpy is needed")
class SignalTests(unittest.TestCase):
    def test_decision_signals(self):
        import numpy as np
        rng = np.random.default_rng(2)
        T = {k: rng.normal(size=(1, 40, 100)) for k in range(4)}
        S = rng.normal(size=(1, 40, 100))
        sig = dh.decision_signals(T, S, 8, "g")
        sl = {k: dh.sh.decompose(T[k][0, 8:])[1] for k in range(4)}
        d2, d1 = sl[2] - sl[0], sl[1] - sl[0]
        self.assertAlmostEqual(sig["dn_g_2"], float(np.linalg.norm(d2)), places=8)
        self.assertAlmostEqual(sig["dc_g_2"], float((d2 * d1).sum() / (np.linalg.norm(d2) * np.linalg.norm(d1))), places=8)
        scale = float(np.linalg.norm(dh.sh.decompose(S[0, 8:])[1] - sl[0]))
        self.assertAlmostEqual(sig["dr_g_3"], float(np.linalg.norm(sl[3] - sl[0])) / scale, places=8)
        self.assertAlmostEqual(sig["vg_g"], dh.sh.voice_rms(T[0], S, 8), places=12)
        self.assertEqual(sorted(sig), sorted(f for f in dh.SIG_FIELDS if f.split("_")[1] == "g"))

    def test_seed_statistics_tell_shared_from_seed_specific(self):
        import numpy as np
        rng = np.random.default_rng(3)
        T0 = rng.normal(size=(1, 300, 100))
        common = rng.normal(size=(1, 300, 100))
        own = [rng.normal(size=(1, 300, 100)) for _ in range(4)]
        specific = [{k: T0 + 0.25 * k * own[m] for k in range(5)} for m in range(4)]
        shared = [{k: T0 + 0.25 * k * common + 1e-6 * own[m] for k in range(5)} for m in range(4)]
        a = dh.seed_statistics(specific, 10, "g")
        b = dh.seed_statistics(shared, 10, "g")
        self.assertEqual(a["spread_g_slow_0"], 0.0)
        self.assertGreater(a["spread_g_slow_3"], 0.1)
        self.assertAlmostEqual(a["shared_g_3"], 0.5, delta=0.05)          # 1/sqrt(4) for independent decisions
        self.assertGreater(b["shared_g_3"], 0.99)
        self.assertEqual((a["len_min"], a["len_max"]), (290, 290))


@unittest.skipUnless(HAVE_NP, "numpy is needed")
class RunLoopTests(unittest.TestCase):
    def setUp(self):
        import numpy as np
        self.saved = (dh.sh.student_sample, dh.gl.student, dh.ng.decode, dh.sh.f0_track)
        self.pipe = _FakePipe()
        self.student_calls = 0
        dh.sh.student_sample = lambda pipe, ctx: np.random.default_rng(int(ctx["seed"]) + 1009).normal(size=(1, 50, 100))

        def fake_student(pipe, ctx, x, taus):
            self.student_calls += 1
            return x
        dh.gl.student = fake_student
        dh.ng.decode = lambda pipe, ctx, x: x[0, ctx["prompt_len"]:, 0]
        dh.sh.f0_track = lambda audio: np.abs(audio) * 10 + 100

    def tearDown(self):
        dh.sh.student_sample, dh.gl.student, dh.ng.decode, dh.sh.f0_track = self.saved

    def test_quality_loop_records_every_signal_and_output_and_resumes(self):
        item = dh.ex.items()[0]
        with tempfile.TemporaryDirectory() as tmp:
            dh.run_quality(self.pipe, [item], ("clean", "gauss20"), tmp)
            with open(Path(tmp) / "pilot.csv", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            calls = self.student_calls
            dh.run_quality(self.pipe, [item], ("clean", "gauss20"), tmp)
            self.assertEqual(self.student_calls, calls)
        self.assertEqual(len(rows), 2)
        self.assertEqual(list(rows[0]), dh.FIELDS)
        self.assertTrue(all(rows[0][k] != "" for k in dh.FIELDS))
        self.assertEqual(calls, 2 * len(dh.OUTS))
        self.assertIn(tuple(range(dh.KMAX + 1)), self.pipe.teacher_calls)

    def test_smoke_quality_checks_the_teacher_loop_and_reproduces_sts3(self):
        item = dh.ex.items()[0]
        stored = {(item["utt_id"], "clean"): {"wer": 0.1, "similarity": 0.8}}
        with tempfile.TemporaryDirectory() as tmp:
            repro, loop = dh.run_quality(self.pipe, [item], ("clean",), tmp, smoke=True, stored=stored)
            with open(Path(tmp) / "pilot.csv", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
        self.assertEqual(list(rows[0]), dh.FIELDS + dh.SMOKE_FIELDS)
        self.assertEqual(len(repro), 1)
        self.assertEqual(dh.sh.check_reproduction(repro), (0.0, 0.0))
        self.assertEqual(loop, [0.0])
        self.assertEqual(self.student_calls, len(dh.OUTS) + 1)

    def test_seed_loop(self):
        with tempfile.TemporaryDirectory() as tmp:
            dh.run_seeds(self.pipe, dh.ex.items()[:1], ("clean", "noisy15"), tmp, seeds=3)
            with open(Path(tmp) / "seeds.csv", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 2)
        self.assertEqual(list(rows[0]), dh.SEED_FIELDS)
        self.assertTrue(all(rows[0][k] != "" for k in dh.SEED_FIELDS))
        self.assertEqual(self.student_calls, 0)

    def test_diversity_loop(self):
        with tempfile.TemporaryDirectory() as tmp:
            dh.run_diversity(self.pipe, dh.ex.items()[:1], dh.DIV_CONDS, tmp, seeds=3)
            with open(Path(tmp) / "diversity.csv", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 1)
        r = rows[0]
        self.assertEqual(list(r), dh.DIV_FIELDS)
        self.assertEqual(self.student_calls, 3 * (len(dh.OUTS) + len(dh.TA_OUTS)))
        self.assertIn(tuple(range(dh.DIV_KMAX + 1)), self.pipe.teacher_calls)
        for o in ("DH_3", "W3", "W4", "TA"):
            self.assertFalse(math.isnan(float(r[f"cvf0_{o}"])), o)
        self.assertEqual(r["ta_branch"], "whole")
        share = float(r["ta_step3_share"])
        self.assertTrue(0.0 <= share <= 1.0)
        if share in (0.0, 1.0):                       # every seed picked the same handover
            self.assertEqual(r["cvf0_TA"], r["cvf0_W3" if share == 1.0 else "cvf0_W4"])

    def test_diversity_loop_leaves_ta_to_the_stored_value_off_the_whole_branch(self):
        self.pipe.snr = 5.0
        with tempfile.TemporaryDirectory() as tmp:
            dh.run_diversity(self.pipe, dh.ex.items()[:1], dh.DIV_CONDS, tmp, seeds=2)
            with open(Path(tmp) / "diversity.csv", encoding="utf-8") as f:
                r = list(csv.DictReader(f))[0]
        self.assertEqual(r["ta_branch"], "student")
        self.assertTrue(math.isnan(float(r["cvf0_TA"])))
        self.assertFalse(math.isnan(float(r["cvf0_W3"])))


class AnalyzeTests(unittest.TestCase):
    """Synthetic outputs on the committed pilot cases. By default every output has the lower of the reference's and the
    stored STS's word errors (on these speakers STS is slightly better than the reference on degraded prompts) and the
    reference's similarity, and DHu_2 loses pitch variety, so the cheapest output that meets G2 is DHu_3."""

    def _write(self, out, step0_spread=0.05, worse_degraded=(), low_cvf0=("DHu_2",), with_div=True, ta_offset=0.0, ta_branch="whole",
               gaps=None):
        stored = dh.load_stored()
        snr = {k: float(r["prompt_snr"]) for k, r in dh._steps_rows().items()}
        cases = [(it["utt_id"], c) for it in dh.ex.items()[:dh.N_SPEAKERS] for c in dh.CONDS]
        rng = random.Random(6)
        with open(out / "pilot.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=dh.FIELDS)
            w.writeheader()
            for u, c in cases:
                ref = stored[(u, c)]["m"]["plain_8"]
                row = {"utt_id": u, "speaker_id": stored[(u, c)]["speaker_id"], "condition": c, "prompt_snr": snr[(u, c)], "gpu": "Tesla T4",
                       "case_seconds": 1, "utc": ""}
                row.update({k: rng.uniform(0.1, 0.9) for k in dh.SIG_FIELDS})
                if gaps is not None:                       # (clean range, degraded range) of the Step-0 voice gap
                    lo, hi = gaps[0] if c == "clean" else gaps[1]
                    row["vg_g"] = rng.uniform(lo, hi)
                for o in dh.OUTS:
                    sts = stored[(u, c)]["m"][f"STS_{o.rsplit('_', 1)[1]}"]
                    base = min(min(1.0, ref["wer"]), min(1.0, sts["wer"]))
                    extra = 0.2 if (o in worse_degraded and c in dh.DEGRADED) else 0.0
                    row.update({f"wer_{o}": base + extra, f"similarity_{o}": ref["similarity"], f"utmos_{o}": ref["utmos"], f"hypothesis_{o}": ""})
                w.writerow(row)
        with open(out / "seeds.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=dh.SEED_FIELDS)
            w.writeheader()
            for c in dh.SEED_CONDS:
                for it in dh.ex.items()[:dh.SEED_N]:
                    row = {"utt_id": it["utt_id"], "speaker_id": it["speaker_id"], "condition": c, "n_seeds": dh.SEED_M,
                           "len_min": 100, "len_max": 100, "case_seconds": 1, "utc": ""}
                    for t in dh.TAGS:
                        for p in dh.PARTS:
                            for k in range(dh.SEED_KMAX + 1):
                                row[f"spread_{t}_{p}_{k}"] = step0_spread if k == 0 else 0.5
                        for k in range(1, dh.SEED_KMAX + 1):
                            row[f"shared_{t}_{k}"] = 0.5
                    w.writerow(row)
        if with_div:
            cv = dh.stored_cvf0()
            with open(out / "diversity.csv", "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=dh.DIV_FIELDS)
                w.writeheader()
                for it in dh.ex.items()[:dh.DIV_N]:
                    row = {"utt_id": it["utt_id"], "speaker_id": it["speaker_id"], "condition": "clean", "n_seeds": 20, "case_seconds": 1, "utc": ""}
                    for o in dh.OUTS:
                        row[f"cvf0_{o}"] = cv[it["speaker_id"]]["plain_8"] - (0.03 if o in low_cvf0 else 0.0)
                        row[f"voiced_{o}"] = 0.6
                    ref = cv[it["speaker_id"]]["plain_8"]
                    row.update({"prompt_snr": 40.0, "cvf0_W3": ref - 0.005, "cvf0_W4": cv[it["speaker_id"]]["plain_4"],
                                "cvf0_TA": (ref + ta_offset) if ta_branch == "whole" else float("nan"),
                                "voiced_W3": 0.6, "voiced_W4": 0.6, "voiced_TA": 0.6, "ta_branch": ta_branch, "ta_step3_share": 0.9})
                    w.writerow(row)

    def test_the_cheapest_output_that_keeps_quality_and_pitch_variety_is_carried(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            self._write(out)
            res = dh.analyze(out)
            report = (out / "PILOT_REPORT.md").read_text(encoding="utf-8")
        self.assertEqual(res["have"], res["expected"])
        self.assertTrue(res["criteria"]["G1_first_estimate_shared"])
        self.assertFalse(res["criteria"]["G2_variant"]["DHu_2"])
        self.assertTrue(res["criteria"]["G2_variant"]["DHu_3"])
        self.assertTrue(res["advance"])
        self.assertEqual(res["carried_variant"], "DHu_3")
        self.assertIn("Decision Handover", report)
        self.assertAlmostEqual(res["assumptions"]["clean"]["g"]["slow_ratio_0_to_3"], 0.1)

    def test_a_first_estimate_that_varies_with_the_seed_stops_the_pilot(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            self._write(out, step0_spread=0.4)
            res = dh.analyze(out)
        self.assertFalse(res["criteria"]["G1_first_estimate_shared"])
        self.assertFalse(res["advance"])
        self.assertIsNone(res["carried_variant"])

    def test_more_word_errors_than_the_slow_part_handover_on_degraded_prompts_fails_g2(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            self._write(out, worse_degraded=("DHu_3",))
            res = dh.analyze(out)
        self.assertFalse(res["criteria"]["G2_variant"]["DHu_3"])
        self.assertGreater(res["quality"]["DHu_3"]["vs_sts_degraded"]["wer"]["mean"], 0.0)
        self.assertEqual(res["carried_variant"], "DH_2")

    def test_the_two_axis_handover_advances_on_pitch_variety_alone(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            self._write(out, step0_spread=0.4)                 # Decision Handover stops at G1 ...
            res = dh.analyze(out)
            report = (out / "PILOT_REPORT.md").read_text(encoding="utf-8")
        self.assertFalse(res["advance"])
        self.assertTrue(res["criteria"]["G3_two_axis_pitch_variety"])   # ... the Two-Axis Handover does not depend on it
        self.assertTrue(res["advance_two_axis"])
        self.assertAlmostEqual(res["two_axis"]["pitch"]["TA"]["mean"], 0.0, places=12)
        self.assertEqual(res["two_axis"]["pitch"]["w4_vs_stored_plain_4_max_abs"], 0.0)
        self.assertEqual(res["two_axis"]["params"], dh.two_axis_params())
        self.assertEqual(res["two_axis"]["replay"]["cases"], 360)
        self.assertIn("Two-Axis Handover", report)

    def test_lost_pitch_variety_stops_the_two_axis_handover(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            self._write(out, ta_offset=-0.03)
            res = dh.analyze(out)
        self.assertFalse(res["criteria"]["G3_two_axis_pitch_variety"])
        self.assertFalse(res["advance_two_axis"])
        self.assertTrue(res["advance"])                        # Decision Handover's decision is unaffected

    def test_off_the_whole_branch_the_stored_cvf0_is_used(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            self._write(out, ta_branch="student")              # every prompt below the Voice-Swap band: student alone
            res = dh.analyze(out)
        cv = dh.stored_cvf0()
        spk = [it["speaker_id"] for it in dh.ex.items()[:dh.DIV_N]]
        expected = sum(cv[s]["student4"] - cv[s]["plain_8"] for s in spk) / len(spk)
        self.assertAlmostEqual(res["two_axis"]["pitch"]["TA"]["mean"], expected, places=12)
        self.assertFalse(res["criteria"]["G3_two_axis_pitch_variety"])   # the student alone loses pitch variety

    def test_agreement_routing_follows_the_step0_voice_gap(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            self._write(out, gaps=((0.15, 0.15), (0.85, 0.85)))   # the models agree on clean prompts only
            res = dh.analyze(out)
            report = (out / "PILOT_REPORT.md").read_text(encoding="utf-8")
        a = res["agreement_routed"]
        self.assertEqual(a["cases"], 60)
        self.assertEqual(a["routes"]["clean"], {"W4": 20, "STT_4": 0, "student4": 0})
        for c in dh.DEGRADED:
            self.assertEqual(a["routes"][c], {"W4": 0, "STT_4": 0, "student4": 20})
        self.assertAlmostEqual(a["mean_passes"], (20 * 14 + 40 * 6) / 60)
        self.assertEqual(a["misroutes"], 0)
        self.assertLessEqual(a["snr_misroutes"], 60)
        self.assertEqual(a["pitch"]["n"], dh.DIV_N)
        self.assertEqual(res["criteria"]["G4_agreement_routed"], dh.arh_passes(a))
        self.assertEqual(res["advance_agreement_routed"], bool(dh.arh_passes(a)))
        self.assertIn("Agreement-Routed Handover", report)

    def test_a_gap_that_misses_the_noise_hands_noisy_prompts_to_the_teacher(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            self._write(out, gaps=((0.85, 0.85), (0.15, 0.15)))   # inverted: the gap does not see the noise
            res = dh.analyze(out)
        a = res["agreement_routed"]
        for c in dh.DEGRADED:
            self.assertEqual(a["routes"][c], {"W4": 20, "STT_4": 0, "student4": 0})
        self.assertGreater(a["degraded"]["wer"]["mean"], 0.0)          # the whole estimate at Step 4 copies the noise
        self.assertEqual(a["misroutes"], 40)
        self.assertLess(a["snr_misroutes"], 39)                        # the prompt SNR does see white noise
        self.assertFalse(res["criteria"]["G4_agreement_routed"])

    def test_partial_runs_need_the_flag_and_cannot_advance(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            self._write(out, with_div=False)
            with self.assertRaises(SystemExit):
                dh.analyze(out)
            res = dh.analyze(out, allow_partial=True)
        self.assertTrue(res["partial"])
        self.assertTrue(all(v is None for v in res["criteria"]["G2_variant"].values()))
        self.assertFalse(res["advance"])
        self.assertIsNone(res["criteria"]["G3_two_axis_pitch_variety"])
        self.assertFalse(res["advance_two_axis"])
        self.assertIsNone(res["criteria"]["G4_agreement_routed"])
        self.assertFalse(res["advance_agreement_routed"])


class TwoAxisTests(unittest.TestCase):
    def test_parameters_come_from_the_stored_data(self):
        p = dh.two_axis_params()
        snrs = sorted(float(r["prompt_snr"]) for r in dh._steps_rows().values() if r["condition"] == "clean")
        self.assertEqual(len(snrs), 60)
        self.assertEqual(sum(v >= p["snr_threshold"] for v in snrs), 58)          # 95% coverage, as in the screen
        self.assertEqual(p["band_db"], 2.0)
        self.assertIsNotNone(p["lambda3"])
        self.assertEqual(p, dh.two_axis_params())                                    # deterministic

    def test_routing(self):
        p = {"snr_threshold": 26.0, "band_db": 2.0, "lambda3": 1.2}
        self.assertEqual(dh.ta_route(30.0, 1.3, p), "W3")
        self.assertEqual(dh.ta_route(30.0, 1.1, p), "W4")
        self.assertEqual(dh.ta_route(25.0, 9.9, p), "STT_4")
        self.assertEqual(dh.ta_route(23.9, 9.9, p), "student4")
        self.assertEqual(dh.ta_route(30.0, 9.9, {**p, "lambda3": None}), "W4")

    def test_in_sample_replay_keeps_the_rule_it_was_fitted_with(self):
        r = dh.two_axis_replay(dh.two_axis_params(), dh.gl.thresholds())
        self.assertEqual(r["cases"], 360)
        self.assertEqual(sum(r["routes"].values()), 360)
        self.assertLess(r["mean_passes"], 8.0)
        self.assertLessEqual(r["clean"]["wer"]["mean"], dh.TA_RULE["clean_wer"])
        self.assertGreaterEqual(r["clean"]["similarity"]["mean"], -dh.TA_RULE["similarity"])
        self.assertGreaterEqual(r["clean"]["utmos"]["mean"], -dh.TA_RULE["utmos"])
        self.assertLessEqual(r["failures_clean"], 3)


class AgreementRoutedTests(unittest.TestCase):
    def test_cuts_and_branches(self):
        gaps = [i / 100 for i in range(1, 41)]                       # 40 clean gaps 0.01 .. 0.40
        lo, hi = dh.arh_cuts(gaps)
        self.assertEqual((lo, hi), (gaps[38], 0.40))                  # index int(0.95 * 40) = 38
        self.assertEqual(sum(g <= lo for g in gaps), 39)
        self.assertEqual(dh.arh_branch(0.05, lo, hi), "whole")
        self.assertEqual(dh.arh_branch(0.395, lo, hi), "vs")
        self.assertEqual(dh.arh_branch(0.41, lo, hi), "student")
        self.assertEqual({dh.ARH_OUT[b] for b in ("whole", "vs", "student")}, {"W4", "STT_4", "student4"})
        self.assertEqual(dh.ARH_COST, {"W4": 14, "STT_4": 14, "student4": 6})

    def test_folds_split_speakers_evenly_and_reproducibly(self):
        spk = [it["speaker_id"] for it in dh.ex.items()[:dh.N_SPEAKERS]]
        f = dh.arh_folds(set(spk))
        self.assertEqual(sorted(collections_counter(f.values())), [4, 4, 4, 4, 4])
        self.assertEqual(f, dh.arh_folds(set(spk)))

    def test_the_g4_rule(self):
        base = {"cases": 60, "mean_passes": 9.0, "misroutes": 2, "snr_misroutes": 1,
                "pitch": {"mean": -0.005, "lo": -0.01, "hi": 0.0, "n": 10}}
        self.assertTrue(dh.arh_passes(base))                          # one misroute more than the SNR rule is allowed
        self.assertFalse(dh.arh_passes({**base, "misroutes": 3}))
        self.assertFalse(dh.arh_passes({**base, "mean_passes": 14.8}))
        self.assertFalse(dh.arh_passes({**base, "pitch": {**base["pitch"], "mean": -0.011}}))
        self.assertIsNone(dh.arh_passes({**base, "mean_passes": None}))
        self.assertIsNone(dh.arh_passes({**base, "cases": 0}))


def collections_counter(values):
    out = {}
    for v in values:
        out[v] = out.get(v, 0) + 1
    return list(out.values())


class PreregistrationTests(unittest.TestCase):
    def test_committed_preregistration_matches_this_code(self):
        pre = dh.check_prereg()
        self.assertEqual(pre["experiment"], "EXP-023")
        self.assertEqual(pre["code_hashes"], dh.code_hashes())
        rho = {int(k): v for k, v in pre["completion"]["rho"].items()}
        self.assertEqual(rho, dh.committed_shares())
        self.assertEqual(pre["two_axis"]["params"], dh.two_axis_params())

    def test_the_amendments_are_recorded(self):
        pre = dh.check_prereg()
        self.assertEqual([a["replaces"] for a in pre["amendments"]], [dh.ORIGINAL_PREREG, dh.SECOND_PREREG, dh.PREVIOUS_PREREG])
        self.assertTrue(all(a["when"] == "before any synthesis" for a in pre["amendments"]))
        self.assertNotIn(pre["sha256"], (dh.ORIGINAL_PREREG, dh.SECOND_PREREG, dh.PREVIOUS_PREREG))
        self.assertIn("G3_two_axis_pitch_variety", pre["criteria"])
        self.assertIn("G4_agreement_routed", pre["criteria"])
        self.assertEqual(pre["agreement_routed"]["passes"], dh.ARH_COST)

    def test_an_edited_preregistration_is_rejected(self):
        pre = json.loads(dh.PREREG_PATH.read_text(encoding="utf-8"))
        pre["decision_rule"] = "anything passes"
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "preregistration.json"
            p.write_text(json.dumps(pre), encoding="utf-8")
            with self.assertRaises(SystemExit):
                dh.check_prereg(p)

    def test_input_hashes_do_not_depend_on_line_endings(self):
        with tempfile.TemporaryDirectory() as tmp:
            a, b = Path(tmp) / "a.json", Path(tmp) / "b.json"
            a.write_bytes(b'{\n  "x": 1\n}\n')
            b.write_bytes(b'{\r\n  "x": 1\r\n}\r\n')
            self.assertEqual(dh.input_hash(a), dh.input_hash(b))

    def test_changed_code_is_rejected(self):
        pre = json.loads(dh.PREREG_PATH.read_text(encoding="utf-8"))
        pre["code_hashes"]["code/colab/run_decision_handover_pilot.py"] = "0" * 64
        body = {k: v for k, v in pre.items() if k not in ("sha256", "created_utc")}
        pre["sha256"] = hashlib.sha256(json.dumps(body, sort_keys=True).encode("utf-8")).hexdigest()
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "preregistration.json"
            p.write_text(json.dumps(pre), encoding="utf-8")
            with self.assertRaises(SystemExit):
                dh.check_prereg(p)


if __name__ == "__main__":
    unittest.main()
