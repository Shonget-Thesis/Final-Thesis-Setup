"""Verify workflow gates catch missing evidence, source drift, and stale records."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from check_workspace import REQUIRED, check_progress_diff, validate
from render_progress import render


class WorkspaceCheckTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for name in REQUIRED:
            file = self.root / name
            file.parent.mkdir(parents=True, exist_ok=True)
            file.write_text("", encoding="utf-8")
        (self.root / "source.txt").write_text("original", encoding="utf-8")
        manifest = {"files": [{"destination": "source.txt", "sha256": hashlib.sha256(b"original").hexdigest()}]}
        (self.root / "docs/SOURCE_MANIFEST.json").write_text(json.dumps(manifest), encoding="utf-8")
        self.data = {"tasks": [{"id": "TEST-001", "title": "Fixture", "owner": "tester", "status": "done",
                               "updated": "2026-10-09", "acceptance": "Evidence exists", "evidence": ["source.txt"],
                               "validation": "Inspected fixture"}]}
        self.save_tracker()

    def save_tracker(self):
        (self.root / "docs/progress/tasks.json").write_text(json.dumps(self.data), encoding="utf-8")
        (self.root / "docs/progress/TASKS.md").write_text(render(self.data), encoding="utf-8")

    def test_valid_fixture_passes(self):
        self.assertEqual(validate(self.root), [])

    def test_done_requires_evidence(self):
        self.data["tasks"][0]["evidence"] = []
        self.save_tracker()
        self.assertTrue(any("done requires evidence" in e for e in validate(self.root)))

    def test_source_drift_is_reported(self):
        (self.root / "source.txt").write_text("changed", encoding="utf-8")
        self.assertTrue(any("Imported source changed" in e for e in validate(self.root)))

    def test_stale_board_is_reported(self):
        (self.root / "docs/progress/TASKS.md").write_text("stale", encoding="utf-8")
        self.assertTrue(any("Task board is stale" in e for e in validate(self.root)))

    def test_broken_link_and_citation_are_reported(self):
        (self.root / "README.md").write_text("[missing](missing.md)", encoding="utf-8")
        (self.root / "paper/main.tex").write_text(r"\cite{missing}", encoding="utf-8")
        errors = validate(self.root)
        self.assertTrue(any("broken local link" in e for e in errors))
        self.assertTrue(any("missing bibliography key" in e for e in errors))

    def test_submission_rejects_draft_markers(self):
        (self.root / "paper/chapters/01-introduction.tex").write_text("TBD: author", encoding="utf-8")
        self.assertEqual(validate(self.root), [])
        self.assertTrue(any("unfinished manuscript" in e for e in validate(self.root, submission=True)))


    def test_submission_allows_unfinished_drafts(self):
        (self.root / "paper/drafts/tanh-velocity-blending.tex").write_text("TBD: draft", encoding="utf-8")
        self.assertEqual(validate(self.root, submission=True), [])

    def test_citations_and_inputs_resolve_within_each_project(self):
        draft = self.root / "paper/drafts/tanh-velocity-blending.tex"
        draft.write_text(r"\cite{draft-only}\bibitem{draft-only}Draft", encoding="utf-8")
        self.assertEqual(validate(self.root), [])
        (self.root / "paper/main.tex").write_text(r"\cite{draft-only}", encoding="utf-8")
        self.assertTrue(any("missing bibliography key" in e for e in validate(self.root)))

    def test_progress_diff_requires_records_for_code_change(self):
        with patch("check_workspace.subprocess.run", return_value=SimpleNamespace(stdout="code/analysis/new.py\n")):
            self.assertEqual(len(check_progress_diff(self.root, "base")), 3)
        complete = "code/analysis/new.py\ndocs/progress/tasks.json\ndocs/progress/STATUS.md\ndocs/progress/log/2026-10-09-test.md\n"
        with patch("check_workspace.subprocess.run", return_value=SimpleNamespace(stdout=complete)):
            self.assertEqual(check_progress_diff(self.root, "base"), [])


if __name__ == "__main__":
    unittest.main()
