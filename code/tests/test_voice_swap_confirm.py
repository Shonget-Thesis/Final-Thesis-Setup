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
import run_voice_swap_confirm as vc  # noqa: E402

HAVE_NP = importlib.util.find_spec("numpy") is not None
EXP019_PREREG = ROOT / "research/experiments/EXP-019-guidance-decomposition-prelim/preregistration.json"


class DesignTests(unittest.TestCase):
    def test_forward_passes_match_the_exp020_accounting(self):
        self.assertEqual(vc.PASSES["ref"], vc.sh.cost("plain_8"))
        self.assertEqual(vc.PASSES["vs"], vc.sh.cost("STT_4"))
        self.assertEqual(vc.PASSES["plain4"], vc.sh.cost("plain_4"))
        self.assertEqual(vc.PASSES["student"], vc.sh.cost("student4"))
        self.assertEqual((vc.PASSES["ref"], vc.PASSES["vs"]), (21, 14))

    def test_voice_swap_is_the_exp020_primary_combination(self):
        self.assertEqual(vc.CODE, vc.sh.PRIMARY)
        self.assertEqual(vc.CODE, "STT")
        self.assertEqual(vc.K, vc.sh.K)

    def test_fresh_speakers_are_the_40_after_the_design_speakers(self):
        items = vc.fresh_items()
        self.assertEqual(len(items), 40)
        design = {it["utt_id"] for it in vc.ex.items()}
        self.assertEqual(len(design), 60)
        self.assertFalse(design & {it["utt_id"] for it in items})
        self.assertEqual(len({it["speaker_id"] for it in items}), 40)

    @unittest.skipUnless(EXP019_PREREG.exists(), "EXP-019 preregistration not present")
    def test_fresh_speakers_are_the_same_40_that_exp019_names(self):
        named = json.loads(EXP019_PREREG.read_text(encoding="utf-8"))["data"]["fresh_utt_ids"]
        self.assertEqual([it["utt_id"] for it in vc.fresh_items()], named)

    def test_second_noise_draw_does_not_reuse_any_seed_of_the_first(self):
        first = {0, 7, 29, 61, vc.sh.STUDENT_SAMPLE_OFFSET}
        second = {vc.REPLICATE_OFFSET + o for o in first}
        self.assertFalse(first & second)
        diversity = {vc.DIV_SEED_BASE + m + o for m in range(vc.DIV_SEEDS) for o in first}
        self.assertFalse(diversity & second)

    def test_conditions(self):
        self.assertTrue(set(vc.CONDS) <= set(vc.rf.CONDITIONS))
        self.assertNotIn("clean", vc.DEGRADED)
        self.assertEqual(len(vc.DEGRADED), 5)
        self.assertTrue(set(vc.DIV_CONDS) <= set(vc.CONDS))
        self.assertIn("babble15", vc.CONDS)

    def test_stored_labels_cover_the_exp020_cases(self):
        stored = vc.stored_labels()
        self.assertEqual(len(stored), 360)
        one = next(iter(stored.values()))
        self.assertEqual(set(one), {"ref", "plain4", "vs", "student"})
        self.assertTrue(all(0.0 <= v["similarity"] <= 1.0 for s in stored.values() for v in s.values()))


class _NoContext:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _FakePipe:
    """The parts of rf.Pipeline that the run loops touch, on numpy arrays, deterministic per seed."""
    gpu = "Tesla T4 (fake)"

    def __init__(self):
        import numpy as np
        self.np = np
        self.torch = type("T", (), {"inference_mode": staticmethod(_NoContext)})
        self.sampler = type("S", (), {"STUDENT_TAUS": [0.0761, 0.2929, 0.6173]})
        self.device = type("D", (), {"type": "cpu"})()
        self.prepared, self.teacher_calls = 0, []

    def prepare(self, item, cond, seed, out_dir, noise_seed=None):
        self.prepared += 1
        return {"prompt_len": 10, "prompt_snr": 30.0, "seed": seed, "step_cond": self.np.zeros((1, 50, 100))}

    def teacher(self, ctx, steps):
        self.teacher_calls.append(tuple(steps))
        return {i: self.np.random.default_rng(int(ctx["seed"]) * 10 + i).normal(size=(1, 50, 100)) for i in steps}, {}

    def ground_truth(self, item):
        return self.np.zeros(10), 24000

    def score(self, audio, item, gt):
        return {"wer": 0.1, "similarity": 0.8, "utmos": 3.0, "hypothesis": "text"}


@unittest.skipUnless(HAVE_NP, "numpy is needed")
class RunLoopTests(unittest.TestCase):
    """Exercises the run loops with a fake model, since torch is not installed for the local checks."""

    def setUp(self):
        import numpy as np
        self.saved = (vc.sh.student_sample, vc.gl.student, vc.ng.decode, vc.sh.f0_track)
        self.pipe = _FakePipe()
        self.student_calls = []
        vc.sh.student_sample = lambda pipe, ctx: np.random.default_rng(int(ctx["seed"]) + 1009).normal(size=(1, 50, 100))

        def fake_student(pipe, ctx, x, taus):
            self.student_calls.append((int(ctx["seed"]), list(taus)))
            return x + 0.001 * int(ctx["seed"] % 97)
        vc.gl.student = fake_student
        vc.ng.decode = lambda pipe, ctx, x: x[0, ctx["prompt_len"]:, 0]
        vc.sh.f0_track = lambda audio: np.abs(audio) * 10 + 100

    def tearDown(self):
        vc.sh.student_sample, vc.gl.student, vc.ng.decode, vc.sh.f0_track = self.saved

    def test_quality_loop_writes_every_column_alternates_order_and_resumes(self):
        item = vc.fresh_items()[0]
        stored = {(item["utt_id"], "clean"): {o: {"wer": 0.1, "similarity": 0.8} for o in ("ref", "plain4", "vs", "student")}}
        with tempfile.TemporaryDirectory() as tmp:
            repro = vc.run_quality(self.pipe, [item], ("clean", "gauss20"), tmp, stored=stored)
            with open(Path(tmp) / "confirm.csv", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            prepared = self.pipe.prepared
            vc.run_quality(self.pipe, [item], ("clean", "gauss20"), tmp)
            self.assertEqual(self.pipe.prepared, prepared)            # finished cases are not run again
        self.assertEqual(prepared, 3)                                  # one warm-up case plus the two recorded cases
        self.assertEqual(len(rows), 2)
        self.assertEqual(set(rows[0]), set(vc.FIELDS))
        self.assertTrue(all(rows[0][k] != "" for k in vc.FIELDS))
        self.assertEqual([r["order"] for r in rows], ["ref_first", "vs_first"])
        self.assertEqual(float(rows[0]["e4_max_abs_diff"]), 0.0)        # the two timed paths share the Step-4 estimate
        self.assertEqual(len(repro), 4)
        self.assertEqual(vc.sh.check_reproduction(repro), (0.0, 0.0))

    def test_paths_and_draws(self):
        item = vc.fresh_items()[0]
        with tempfile.TemporaryDirectory() as tmp:
            vc.run_quality(self.pipe, [item], ("clean",), tmp)
        seed, taus = int(item["input_seed"]), [0.0761, 0.2929, 0.6173]
        recorded = self.student_calls[len(self.student_calls) // 2:]   # the second half is the recorded case
        self.assertEqual(len(recorded), 6)
        self.assertEqual(sum(s == seed + vc.REPLICATE_OFFSET for s, _ in recorded), 2)   # ref2 and vs2
        self.assertEqual(sum(t == [0.0] + taus for _, t in recorded), 1)                 # student only
        self.assertIn((4,), self.pipe.teacher_calls)                                     # Voice-Swap stops the teacher at Step 4
        self.assertIn((4, 8), self.pipe.teacher_calls)

    def test_voice_swap_input_is_exp020s_stt(self):
        import numpy as np
        ctx = self.pipe.prepare(None, "clean", 5, None)
        T = np.random.default_rng(1).normal(size=(1, 50, 100))
        h, S = vc.voice_swap_input(self.pipe, ctx, T)
        np.testing.assert_allclose(h, vc.sh.selective_handover(T, S, 10, "STT"))
        np.testing.assert_allclose(h[0, 10:].mean(axis=0), S[0, 10:].mean(axis=0), atol=1e-10)
        np.testing.assert_array_equal(h[0, :10], T[0, :10])

    def test_diversity_loop(self):
        items = vc.fresh_items()[:1]
        with tempfile.TemporaryDirectory() as tmp:
            vc.run_diversity(self.pipe, items, vc.DIV_CONDS, tmp, seeds=3)
            with open(Path(tmp) / "diversity.csv", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 2)
        self.assertEqual({r["condition"] for r in rows}, set(vc.DIV_CONDS))
        self.assertEqual(set(rows[0]), set(vc.DIV_FIELDS))
        self.assertEqual(len(self.student_calls), 2 * 3 * len(vc.DIV_OUTS))
        self.assertFalse(math.isnan(float(rows[0]["cvf0_vs"])))


class AnalyzeTests(unittest.TestCase):
    def _write(self, out, wer_shift=0.0, sim_shift=0.0, ratio=0.68, cv_vs=0.093, with_div=True):
        rng = random.Random(4)
        with open(out / "confirm.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=vc.FIELDS)
            w.writeheader()
            for it in vc.fresh_items():
                base_w, base_s = rng.uniform(0.0, 0.08), rng.uniform(0.70, 0.95)
                for c in vc.CONDS:
                    shift = (wer_shift if c in vc.DEGRADED else 0.0)
                    row = {"utt_id": it["utt_id"], "speaker_id": it["speaker_id"], "condition": c, "prompt_snr": 25.0, "gpu": "Tesla T4",
                           "order": "ref_first", "t_ref": 3.0, "t_vs": 3.0 * ratio, "audio_seconds": 5.0, "e4_max_abs_diff": 0.0,
                           "voice_rms": 0.3, "case_seconds": 20.0, "utc": ""}
                    for o in vc.OUTS:
                        noise = rng.uniform(-0.004, 0.004)
                        wer = base_w + noise + (shift if o in vc.VS_PAIR else 0.0)
                        sim = base_s + noise + (sim_shift if o in vc.VS_PAIR else 0.0)
                        row.update({f"wer_{o}": max(0.0, wer), f"similarity_{o}": sim, f"utmos_{o}": 3.0, f"hypothesis_{o}": ""})
                    w.writerow(row)
        if with_div:
            with open(out / "diversity.csv", "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=vc.DIV_FIELDS)
                w.writeheader()
                for c in vc.DIV_CONDS:
                    for it in vc.fresh_items()[:vc.DIV_N]:
                        j = rng.uniform(-0.003, 0.003)
                        row = {"utt_id": it["utt_id"], "speaker_id": it["speaker_id"], "condition": c, "n_seeds": 20, "case_seconds": 1, "utc": ""}
                        row.update({"cvf0_ref": 0.095 + j, "cvf0_vs": cv_vs + j, "cvf0_student": 0.052 + j})
                        row.update({f"voiced_{o}": 0.8 for o in vc.DIV_OUTS})
                        w.writerow(row)

    def test_a_voice_swap_as_good_as_the_reference_is_confirmed(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            self._write(out)
            res = vc.analyze(out)
            report = (out / "CONFIRMATION_REPORT.md").read_text(encoding="utf-8")
        self.assertEqual(res["cases"], 240)
        self.assertEqual(res["diversity_pairs"], 60)
        self.assertFalse(res["partial"])
        self.assertTrue(all(res["primary"]["criteria"].values()), res["primary"]["criteria"])
        self.assertTrue(res["primary"]["confirmed"])
        self.assertIn("Voice-Swap", report)
        self.assertIn("noise floor", report)
        self.assertAlmostEqual(res["primary"]["measured"]["time_ratio"]["mean"], 0.68)

    def test_more_word_errors_on_degraded_prompts_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            self._write(out, wer_shift=0.05)
            res = vc.analyze(out)
        self.assertFalse(res["primary"]["criteria"]["P1_word_errors"])
        self.assertTrue(res["primary"]["criteria"]["P2_voice_similarity"])
        self.assertFalse(res["primary"]["confirmed"])

    def test_lower_similarity_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            self._write(out, sim_shift=-0.03)
            res = vc.analyze(out)
        self.assertFalse(res["primary"]["criteria"]["P2_voice_similarity"])

    def test_too_little_time_saved_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            self._write(out, ratio=0.95)
            res = vc.analyze(out)
        self.assertFalse(res["primary"]["criteria"]["P4_time"])
        self.assertFalse(res["primary"]["confirmed"])

    def test_lost_pitch_variety_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            self._write(out, cv_vs=0.060)
            res = vc.analyze(out)
        self.assertFalse(res["primary"]["criteria"]["P3_pitch_variety"])

    def test_a_partial_run_needs_the_flag_and_cannot_confirm(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            self._write(out, with_div=False)
            with self.assertRaises(SystemExit):
                vc.analyze(out)
            res = vc.analyze(out, allow_partial=True)
        self.assertTrue(res["partial"])
        self.assertIsNone(res["primary"]["criteria"]["P3_pitch_variety"])
        self.assertFalse(res["primary"]["confirmed"])

    def test_the_failure_table_uses_ref_as_the_reference(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            self._write(out, wer_shift=0.3)
            res = vc.analyze(out)
        self.assertGreater(sum(res["failures"]["vs"].values()), sum(res["failures"]["ref2"].values()))
        self.assertEqual(res["failures"]["vs"]["clean"], 0)


class PreregistrationTests(unittest.TestCase):
    def test_committed_preregistration_matches_this_code(self):
        pre = vc.check_prereg()
        self.assertEqual(pre["experiment"], "EXP-021")
        self.assertEqual(pre["code_hashes"], vc.code_hashes())
        self.assertEqual(pre["data"]["fresh_utt_ids"], [it["utt_id"] for it in vc.fresh_items()])

    def test_an_edited_preregistration_is_rejected(self):
        pre = json.loads(vc.PREREG_PATH.read_text(encoding="utf-8"))
        pre["decision_rule"] = "anything passes"
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "preregistration.json"
            p.write_text(json.dumps(pre), encoding="utf-8")
            with self.assertRaises(SystemExit):
                vc.check_prereg(p)

    def test_input_hashes_do_not_depend_on_line_endings(self):
        # A Windows checkout may hold CRLF where Colab holds LF; the frozen hashes must match on both.
        with tempfile.TemporaryDirectory() as tmp:
            a, b = Path(tmp) / "a.json", Path(tmp) / "b.json"
            a.write_bytes(b'{\n  "x": 1\n}\n')
            b.write_bytes(b'{\r\n  "x": 1\r\n}\r\n')
            self.assertEqual(vc.input_hash(a), vc.input_hash(b))

    def test_changed_code_is_rejected_even_with_a_consistent_hash(self):
        pre = json.loads(vc.PREREG_PATH.read_text(encoding="utf-8"))
        pre["code_hashes"]["code/colab/run_selective_handover_prelim.py"] = "0" * 64
        body = {k: v for k, v in pre.items() if k not in ("sha256", "created_utc")}
        pre["sha256"] = hashlib.sha256(json.dumps(body, sort_keys=True).encode("utf-8")).hexdigest()
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "preregistration.json"
            p.write_text(json.dumps(pre), encoding="utf-8")
            with self.assertRaises(SystemExit):
                vc.check_prereg(p)


if __name__ == "__main__":
    unittest.main()
