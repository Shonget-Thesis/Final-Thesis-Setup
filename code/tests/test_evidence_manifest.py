import hashlib
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BENCH = ROOT / "research/evidence/benchmarks"


class EvidenceManifestTests(unittest.TestCase):
    def test_imported_t4_results_match_recorded_hashes(self):
        manifest = json.loads((BENCH / "MANIFEST.json").read_text(encoding="utf-8"))
        for entry in manifest["files"]:
            data = (BENCH / entry["file"]).read_bytes()
            self.assertEqual(hashlib.sha256(data).hexdigest(), entry["sha256"], entry["file"])
            self.assertEqual(len(data), entry["bytes"], entry["file"])

    def test_every_evidence_file_is_listed(self):
        manifest = json.loads((BENCH / "MANIFEST.json").read_text(encoding="utf-8"))
        listed = {e["file"] for e in manifest["files"]} | {"MANIFEST.json"}
        present = {p.relative_to(BENCH).as_posix() for p in BENCH.rglob("*") if p.is_file()}
        self.assertEqual(present - listed, set())


if __name__ == "__main__":
    unittest.main()
