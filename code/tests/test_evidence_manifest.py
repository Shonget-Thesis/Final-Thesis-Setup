import hashlib
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BENCH = ROOT / "research/evidence/benchmarks"
FOLDERS = (BENCH, ROOT / "research/experiments/EXP-011-followup-t4/results",
           ROOT / "research/experiments/EXP-012-consistency-gate-prelim/results",
           ROOT / "research/experiments/EXP-013-endpoint-extrapolation-prelim/results")


class EvidenceManifestTests(unittest.TestCase):
    def test_imported_t4_results_match_recorded_hashes(self):
        for folder in FOLDERS:
            manifest = json.loads((folder / "MANIFEST.json").read_text(encoding="utf-8"))
            for entry in manifest["files"]:
                data = (folder / entry["file"]).read_bytes()
                self.assertEqual(hashlib.sha256(data).hexdigest(), entry["sha256"], entry["file"])
                self.assertEqual(len(data), entry["bytes"], entry["file"])

    def test_every_evidence_file_is_listed(self):
        for folder in FOLDERS:
            manifest = json.loads((folder / "MANIFEST.json").read_text(encoding="utf-8"))
            listed = {e["file"] for e in manifest["files"]} | {"MANIFEST.json"}
            present = {p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file()}
            self.assertEqual(present - listed, set(), folder)


if __name__ == "__main__":
    unittest.main()
