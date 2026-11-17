"""Tests for code/analysis/exp023_followups.py on synthetic pilot files (the EXP-023 test writer)."""
import csv
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code/analysis"))
sys.path.insert(0, str(ROOT / "code/tests"))
import exp023_followups as fu  # noqa: E402
import test_decision_handover as tdh  # noqa: E402


def _rewrite(path, change):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
        fields = list(rows[0])
    for r in rows:
        change(r)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


class FollowupTests(unittest.TestCase):
    def _files(self, out, vg_u=None, dr1=None, dc2=None):
        tdh.AnalyzeTests("test_partial_runs_need_the_flag_and_cannot_advance")._write(out, gaps=((0.15, 0.15), (0.85, 0.85)))

        def change(r):
            clean = r["condition"] == "clean"
            if vg_u is not None:
                r["vg_u"] = vg_u[0] if clean else vg_u[1]
            if dr1 is not None:
                r["dr_g_1"] = dr1[0] if clean else dr1[1]
            if dc2 is not None:
                r["dc_g_2"] = dc2
        _rewrite(out / "pilot.csv", change)

    def test_every_rule_reports_on_all_cases(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            self._files(out, vg_u=(0.05, 0.6))
            res = fu.analyze(out)
            self.assertTrue((out / "FOLLOWUPS_REPORT.md").exists())
        self.assertEqual(set(res), {"SNR rule", "ARH (vg_g)", "GAR-u (vg_u)", "GAR-ratio (vg_g / vg_u)", "CGH (ARH + dr_g_1 gate)", "DCH replay",
                                    "PAS (ARH + prosody-agreement safeguard)", "DH_3 alone", "DHu_3 alone", "GCH (guided or unguided decision)"})
        for r in res.values():
            self.assertEqual(r["cases"], 60)
        self.assertEqual(res["ARH (vg_g)"]["misroutes"], 0)
        self.assertEqual(res["GAR-u (vg_u)"]["misroutes"], 0)               # vg_u separates clean from degraded too
        self.assertAlmostEqual(res["GAR-u (vg_u)"]["mean_passes"], (20 * 14 + 40 * 5) / 60)

    def test_a_signal_that_does_not_separate_misroutes(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            self._files(out, vg_u=(0.6, 0.05))                              # inverted: degraded cases look clean
            res = fu.analyze(out)
        self.assertEqual(res["GAR-u (vg_u)"]["misroutes"], 40)
        self.assertFalse(res["GAR-u (vg_u)"]["promising"])

    def test_cgh_only_gates_teacher_branch_cases_and_dch_uses_the_decision_signals(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            self._files(out, vg_u=(0.05, 0.6), dr1=(0.9, 0.1), dc2=0.0)      # large clean decisions, small degraded ones
            res = fu.analyze(out)
        cgh = res["CGH (ARH + dr_g_1 gate)"]
        self.assertEqual(cgh["routes"]["clean"].get("student4", 0), 0)     # clean decisions are large: never gated
        dch = res["DCH replay"]
        self.assertEqual(sum(dch["routes"]["clean"].values()), 20)
        self.assertTrue(set(dch["routes"]["clean"]) <= {"student4", "DH_2", "DH_3"})

    def test_pas_sends_only_misaligned_student_cases_to_voice_swap(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            tdh.AnalyzeTests("test_partial_runs_need_the_flag_and_cannot_advance")._write(out, gaps=((0.15, 0.15), (0.85, 0.85)))
            flip = {"n": 0}

            def change(r):
                r["dn_g_1"] = 0.5
                if r["condition"] == "clean":
                    r["dr_g_1"] = 0.5                        # the student's draft sits at the teacher's average timing
                else:
                    flip["n"] += 1
                    r["dr_g_1"] = 0.05 if flip["n"] % 2 else 0.5   # every other degraded draft is 10x farther away
            _rewrite(out / "pilot.csv", change)
            res = fu.analyze(out)
        pz, arh = res["PAS (ARH + prosody-agreement safeguard)"], res["ARH (vg_g)"]
        nonzero = lambda d: {o: n for o, n in d.items() if n}
        self.assertEqual(nonzero(pz["routes"]["clean"]), nonzero(arh["routes"]["clean"]))
        moved = sum(pz["routes"][c].get("STT_4", 0) for c in ("gauss20", "noisy15"))
        # the 20 drafts 10x farther from T_0 always move; the cut keeps 95% of clean prompts, so about 5% of the
        # normal-gap degraded drafts can sit above it too
        self.assertTrue(20 <= moved <= 22, moved)
        self.assertAlmostEqual(pz["mean_passes"], arh["mean_passes"] + moved * 8 / 60)
        self.assertIn("promising", pz)

    def test_gch_chooses_between_the_two_decisions_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            self._files(out, vg_u=(0.05, 0.6))
            res = fu.analyze(out)
        g = res["GCH (guided or unguided decision)"]
        self.assertEqual(g["cases"], 60)
        for c in ("clean", "gauss20", "noisy15"):
            self.assertTrue(set(o for o, n in g["routes"][c].items() if n) <= {"DH_3", "DHu_3"})
        n_u = sum(g["routes"][c].get("DHu_3", 0) for c in ("clean", "gauss20", "noisy15"))
        self.assertAlmostEqual(g["mean_passes"], (9 * n_u + 12 * (60 - n_u)) / 60)
        self.assertEqual(res["DHu_3 alone"]["mean_passes"], 8)

    def test_misroute_definition(self):
        self.assertTrue(fu.misroute(("u", "clean"), "student4"))
        self.assertTrue(fu.misroute(("u", "noisy15"), "W4"))
        self.assertFalse(fu.misroute(("u", "noisy15"), "STT_4"))
        self.assertFalse(fu.misroute(("u", "clean"), "STT_4"))


if __name__ == "__main__":
    unittest.main()
