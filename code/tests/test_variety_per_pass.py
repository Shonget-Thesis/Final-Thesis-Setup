"""Unit tests for EXP-032 (variety per teacher pass against the paper's t_switch = 0.25). They run without PyTorch or the
TTS models."""
import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code/colab"))
import run_followup_tests as rf  # noqa: E402
import run_grid32_prelim as gl  # noqa: E402
import run_method_comparison as mc  # noqa: E402
import run_variety_per_pass as vp  # noqa: E402

TAUS = [0.0761, 0.2929, 0.6173]


class DesignTests(unittest.TestCase):
    def test_costs_and_the_papers_grid(self):
        for k in vp.G16:
            self.assertEqual(vp.COST[f"W{k}"], 2 * k + 5)
        self.assertEqual((vp.COST["STU"], vp.COST["P25"], vp.COST["P25R"], vp.COST["TCH"]), (4, 30, 32, 64))
        self.assertEqual(vp.K_PAPER, 14)
        self.assertLessEqual(gl.GRID32[14], 0.25)
        self.assertGreater(gl.GRID32[15], 0.25)
        self.assertAlmostEqual(gl.ALG2_TIMES[0], gl.GRID32[14])
        self.assertAlmostEqual(gl.GRID32[14], 1.0 - np.cos(np.pi / 2 * 7 / 16))   # = Step 7 of the 16-interval grid
        self.assertEqual(vp.REFERENCE, "P25")
        self.assertTrue(set(vp.DIV_ARMS) <= set(vp.ARMS))
        self.assertNotIn("TCH", vp.DIV_ARMS)

    def test_cohorts_match_exp031_and_exp017(self):
        q = [it["utt_id"] for it in vp.qual_items()]
        self.assertEqual(q, [it["utt_id"] for it in mc.qual_items()])
        self.assertEqual([it["utt_id"] for it in vp.div_items()], q[:20])
        e17 = {(r["utt_id"], r["condition"]) for r in rf.read_csv(vp.EXP017_CSV)}
        e31 = {(r["utt_id"], r["condition"]) for r in rf.read_csv(vp.EXP031_QUAL_CSV)}
        for u in q:
            for c in vp.CONDS:
                self.assertIn((u, c), e17)
                self.assertIn((u, c), e31)
        d31 = {r["utt_id"] for r in rf.read_csv(vp.EXP031_DIV_CSV)}
        self.assertEqual(set(q[:20]), d31)

    def test_conditions_and_dependencies(self):
        for c in vp.CONDS:
            self.assertIn(c, rf.CONDITIONS)
        for name in vp.DEPENDENCIES:
            self.assertTrue((vp.CODE_DIR / name).exists(), name)
        for name in ("run_grid32_prelim.py", "run_endpoint_extrapolation_prelim.py", "run_method_comparison.py"):
            self.assertIn(name, vp.DEPENDENCIES)
        self.assertEqual(vp.EXP017_NAME["P25"], "alg2")
        self.assertEqual(vp.EXP017_NAME["P25R"], "paper")


class _Sampler:
    STUDENT_TAUS = TAUS


class _Pipe:
    sampler = _Sampler()


class SynthesisWiringTests(unittest.TestCase):
    """Which teacher estimate feeds which arm, with every model call replaced by a recorder."""

    def setUp(self):
        self.calls = {"traj": [], "g32": []}
        self.saved = (vp.dts.teacher_trajectory, vp.gl.grid_teacher, vp.gl.student, vp.gl.student_from_state)

        def traj(pipe, ctx, wanted):
            self.calls["traj"].append(tuple(wanted))
            return {k: f"E16_{k}" for k in wanted}, {f"d_{j}": 0.1 * j for j in range(1, 9)}

        def g32(pipe, ctx, grid, wanted):
            self.calls["g32"].append(set(wanted))
            self.assertIs(grid, gl.GRID32)
            return {k: f"E32_{k}" for k in wanted}, {}, {k: f"S32_{k}" for k in wanted}

        vp.dts.teacher_trajectory = traj
        vp.gl.grid_teacher = g32
        vp.gl.student = lambda pipe, ctx, x, taus: ("student", x, tuple(taus))
        vp.gl.student_from_state = lambda pipe, ctx, state, times: ("alg2", state, tuple(times))

    def tearDown(self):
        (vp.dts.teacher_trajectory, vp.gl.grid_teacher, vp.gl.student, vp.gl.student_from_state) = self.saved

    def test_arms(self):
        outs, parts = vp.synthesize(_Pipe(), {"step_cond": "COND"}, vp.ARMS)
        s0 = ("student", "COND", (0.0,))
        self.assertEqual(parts["s0"], s0)
        self.assertEqual(outs["STU"], ("student", s0, tuple(TAUS)))
        for k in vp.G16:
            self.assertEqual(outs[f"W{k}"], ("student", f"E16_{k}", tuple(TAUS)))
        self.assertEqual(outs["P25"], ("alg2", "S32_14", tuple(gl.ALG2_TIMES)))
        self.assertEqual(outs["P25R"], ("student", "E32_14", tuple(TAUS[1:])))
        self.assertEqual(outs["TCH"], "E32_31")
        self.assertEqual(self.calls["g32"], [{14, 31}])
        self.assertEqual(self.calls["traj"], [vp.G16])

    def test_diversity_arms_skip_the_teacher_alone(self):
        outs, _ = vp.synthesize(_Pipe(), {"step_cond": "COND"}, vp.DIV_ARMS)
        self.assertEqual(tuple(outs), vp.DIV_ARMS)
        self.assertEqual(self.calls["g32"], [{14}])


class BankTests(unittest.TestCase):
    def test_round_trip(self):
        rng = np.random.default_rng(0)
        n, plen = 90, 30
        arr = lambda: rng.normal(size=(1, n, 100))  # noqa: E731
        ctx = {"prompt_len": plen, "cond": arr()}
        parts = {"s0": arr(), "ends": {k: arr() for k in vp.G16}, "e32": {vp.K_PAPER: arr()}}
        outs = {a: arr() for a in vp.DIV_ARMS}
        tracks = {a: rng.normal(size=60) for a in vp.DIV_ARMS}
        with tempfile.TemporaryDirectory() as tmp:
            p = vp.bank_save(vp.bank_path(tmp, "noisy15", "utt", 3), ctx, parts, outs, tracks)
            self.assertTrue(str(p).endswith("utt__s03.npz"))
            chk = vp.bank_check(p)
            self.assertTrue(chk["ok"], chk)
            with np.load(p) as z:
                self.assertEqual(int(z["prompt_len"]), plen)
                self.assertEqual(z["prompt"].shape, (plen, 100))
                self.assertEqual(z["s0"].shape, (n - plen, 100))
                self.assertEqual(z["s0"].dtype, np.float16)
                np.testing.assert_allclose(z["T4"], parts["ends"][4][0, plen:].astype(np.float16))
                np.testing.assert_allclose(z["out_P25"], outs["P25"][0, plen:].astype(np.float16))
                self.assertEqual(z["f0_W2"].shape, (60,))
            idx = vp.bank_index(tmp)
            self.assertEqual(idx["files"], 1)
            self.assertEqual(idx["entries"][0]["path"], "bank/noisy15/utt__s03.npz")


def div_row(it, cond, shares, rng, noise=0.0005):
    row = {"utt_id": it["utt_id"], "speaker_id": it["speaker_id"], "condition": cond, "n_seeds": 12,
           "case_seconds": 1.0, "utc": "x"}
    for a in vp.DIV_ARMS:
        s = shares.get(a, 1.0 if a in ("P25", "P25R") else 0.0)
        row.update({f"cvf0_{a}": 0.05 + 0.04 * s + float(rng.normal(0.0, noise)),
                    f"cvf0_aligned_{a}": 0.04 + 0.02 * s + float(rng.normal(0.0, noise)),
                    f"voiced_{a}": 0.6, f"voiced_aligned_{a}": 0.6, f"mel_spread_{a}": 0.3,
                    f"timing_spread_{a}": 4.0 + 6.0 * s})
    return row


class ShareTests(unittest.TestCase):
    def test_share_and_ninety_percent_contrast(self):
        rng = np.random.default_rng(1)
        rows = [div_row(it, "clean", {"W2": 0.5, "W4": 0.97}, rng) for it in vp.div_items()]
        for m in vp.VARIETY:
            half = vp.share_ci(rows, "W2", m)
            self.assertAlmostEqual(half["point"], 0.5, delta=0.05)
            self.assertLess(half["lo"], 0.5)
            self.assertGreater(half["hi"], 0.5)
            self.assertEqual(half["undefined_resamples"], 0)
        curve = vp.variety_curve(rows)
        self.assertGreater(curve["clean"]["cvf0"]["L90"]["W4"]["lo"], 0.0)
        self.assertLess(curve["clean"]["cvf0"]["L90"]["W2"]["hi"], 0.0)

    def test_no_gain_gives_no_share(self):
        rows = [div_row(it, "clean", {"P25": 0.0}, np.random.default_rng(2), noise=0.0) for it in vp.div_items()]
        s = vp.share_ci(rows, "W4", "cvf0")
        self.assertIsNone(s["point"])
        self.assertEqual(s["undefined_resamples"], vp.BOOT_B)


PROFILES = {
    # shares of P25's gain per arm, clean UTMOS offsets per arm
    "w3_enough": ({"W1": 0.3, "W2": 0.7, "W3": 0.95, "W4": 1.0, "W8": 1.0}, {}),
    "quality_gate": ({"W1": 0.3, "W2": 0.7, "W3": 0.95, "W4": 1.0, "W8": 1.0}, {"W3": -0.2}),
    "w8_short": ({"W1": 0.1, "W2": 0.2, "W3": 0.3, "W4": 0.5, "W8": 0.6}, {}),
}


def write_synthetic(tmp, profile, with_div=True):
    shares, offsets = PROFILES[profile]
    rng = np.random.default_rng(7)
    with open(Path(tmp) / "quality.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=vp.QUAL_FIELDS)
        w.writeheader()
        for it in vp.qual_items():
            for cond in vp.CONDS:
                row = {"utt_id": it["utt_id"], "speaker_id": it["speaker_id"], "condition": cond,
                       "prompt_snr": 35.0 if cond == "clean" else 18.0, "gpu": "UnitTest", "frames": 400,
                       "seq_len": 800, **{f"d_{j}": 0.3 for j in range(1, 9)}, "case_seconds": 1.0, "utc": "x"}
                for a in vp.ARMS:
                    if cond == "clean":
                        utm = 3.9 + offsets.get(a, 0.0)
                    else:
                        utm = 3.3 if a == "STU" else 1.9      # every teacher arm copies the noise
                    row.update({f"wer_{a}": 0.0, f"similarity_{a}": 0.93,
                                f"utmos_{a}": utm + float(rng.normal(0.0, 0.01)), f"hypothesis_{a}": "x"})
                w.writerow(row)
    if with_div:
        with open(Path(tmp) / "diversity.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=vp.DIV_FIELDS)
            w.writeheader()
            for cond in vp.CONDS:
                for it in vp.div_items():
                    w.writerow(div_row(it, cond, shares, rng))


class AnalysisTests(unittest.TestCase):
    def _run(self, profile, with_div=True, allow_partial=False):
        with tempfile.TemporaryDirectory() as tmp:
            prereg = Path(tmp) / "preregistration.json"
            prereg.write_text(json.dumps(vp.build_prereg(), indent=2) + "\n", encoding="utf-8")
            write_synthetic(tmp, profile, with_div)
            res = vp.analyze(tmp, prereg_path=prereg, allow_partial=allow_partial)
            for name in ("VARIETY_PER_PASS_REPORT.md", "MANIFEST.json", "bank_index.json",
                         "variety_per_pass_results.json"):
                self.assertTrue((Path(tmp) / name).exists(), name)
            self.assertIn("clean", res["exp031_quality_reproduction"])
            return res

    def test_labels(self):
        res = self._run("w3_enough")
        lab = res["labels"]
        self.assertEqual((lab["cheapest_step_point"], lab["cheapest_step_confirmed"]), ("W3", "W3"))
        self.assertTrue(lab["w8_matches_paper"])
        self.assertEqual(lab["noisy_cheapest_step_point"], "W3")
        self.assertEqual(lab["noisy_quality_safe_arms"], [])
        self.assertIsNotNone(res["variety_noise_change"])
        self.assertEqual(self._run("quality_gate")["labels"]["cheapest_step_point"], "W4")
        lab = self._run("w8_short")["labels"]
        self.assertEqual(lab["cheapest_step_point"], "none")
        self.assertFalse(lab["w8_matches_paper"])

    def test_partial(self):
        lab = self._run("w3_enough", with_div=False, allow_partial=True)["labels"]
        self.assertEqual(set(lab.values()), {"partial"})
        with self.assertRaises(RuntimeError):
            self._run("w3_enough", with_div=False, allow_partial=False)


class PreregistrationAndNotebookTests(unittest.TestCase):
    def test_tampering_is_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "preregistration.json"
            doc = vp.build_prereg()
            p.write_text(json.dumps(doc), encoding="utf-8")
            vp.check_prereg(p)
            doc["bars"]["share"] = 0.8
            p.write_text(json.dumps(doc), encoding="utf-8")
            with self.assertRaises(RuntimeError):
                vp.check_prereg(p)

    def test_committed_preregistration_and_notebook(self):
        if vp.PREREG_PATH.exists():
            doc = vp.check_prereg()
            self.assertEqual((doc["experiment"], doc["task_id"]), ("EXP-032", "EVAL-025"))
            self.assertEqual(doc["reference"]["arm"], "P25")
            self.assertEqual(doc["conditions"], ["clean", "noisy15"])
        nb_path = ROOT / "code/colab/DMOSpeech2_EXP032_Variety_Per_Pass.ipynb"
        nb = json.loads(nb_path.read_text(encoding="utf-8"))
        code = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
        for i, src in enumerate(code):
            compile("\n".join(l for l in src.splitlines() if not l.lstrip().startswith(("!", "%"))), f"cell{i}", "exec")
        joined = "\n".join(code)
        self.assertIn("run_variety_per_pass.py", joined)
        self.assertIn("EXP-032-variety-per-pass", joined)
        self.assertIn("DMOSpeech2_EXP032_Tensor_Bank.zip", joined)
        self.assertNotIn("EXP-030", joined)
        self.assertNotIn("--demand_dir", joined)
        self.assertIn('STAGE = "all"', joined)
        md = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "markdown")
        self.assertIn("0.25", md)
        self.assertIn("noisy15", md)


if __name__ == "__main__":
    unittest.main()
