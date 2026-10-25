"""Download and verify DMOSpeech 2 and F5-TTS model checkpoints.

Downloads official HuggingFace checkpoints into the workspace `ckpts/` folder
and verifies them against their authoritative SHA-256 checksums:
- DMOSpeech 2 (Teacher & Student): model_85000.pt (4.0 GB)
- Duration Predictor (GRPO):        model_1500.pt  (270 MB)
"""
import hashlib
import os
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CKPTS_DIR = ROOT / "ckpts"

CHECKPOINTS = {
    "model_85000.pt": {
        "url": "https://huggingface.co/yl4579/DMOSpeech2/resolve/main/model_85000.pt",
        "sha256": "e143188f7c46b69abb60786937d79b89dac9777d3c4c1e91a97634bcd39ebf2c",
        "size_bytes": 4019822332,
    },
    "model_1500.pt": {
        "url": "https://huggingface.co/yl4579/DMOSpeech2/resolve/main/model_1500.pt",
        "sha256": "6bd50428f3b5f0f25c1a60a0053f40ee47c817bda03c8ee120100f22680ae4c2",
        "size_bytes": 270615794,
    },
}


def compute_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def download_and_verify(name: str, meta: dict, target_dir: Path):
    target_path = target_dir / name
    target_dir.mkdir(parents=True, exist_ok=True)

    if target_path.exists():
        print(f"[{name}] File exists ({target_path.stat().st_size:,} bytes). Verifying SHA-256...")
        digest = compute_sha256(target_path)
        if digest == meta["sha256"]:
            print(f"[{name}] Verification PASSED ({digest[:12]}...).")
            return
        print(f"[{name}] Hash mismatch! Expected {meta['sha256']}, got {digest}. Re-downloading...")

    url = meta["url"]
    print(f"[{name}] Downloading from {url}...")

    def progress(count, block_size, total_size):
        percent = int(count * block_size * 100 / total_size) if total_size > 0 else 0
        sys.stdout.write(f"\r  Downloading {name}: {percent}% ({count * block_size / 1024 / 1024:.1f} MB)")
        sys.stdout.flush()

    urllib.request.urlretrieve(url, target_path, reporthook=progress)
    print("\nDownload complete. Verifying SHA-256...")
    digest = compute_sha256(target_path)
    if digest != meta["sha256"]:
        raise ValueError(f"[{name}] SHA-256 verification FAILED: {digest}")
    print(f"[{name}] Verification PASSED ({digest[:12]}...).")


def main():
    print(f"Checkpoint directory: {CKPTS_DIR}")
    for name, meta in CHECKPOINTS.items():
        download_and_verify(name, meta, CKPTS_DIR)
    print("\nAll model checkpoints are ready.")


if __name__ == "__main__":
    main()
