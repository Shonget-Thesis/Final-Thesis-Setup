"""Build the LaTeX manuscript and reject unresolved citations/references."""
import argparse
from pathlib import Path
import re
import shutil
import subprocess
import sys

from manuscripts import DEFAULT_TARGET, MANUSCRIPTS

ROOT = Path(__file__).resolve().parents[1]


def build_one(latexmk, target):
    source = ROOT / MANUSCRIPTS[target]
    paper = source.parent
    output = paper / "build" / target
    output.mkdir(parents=True, exist_ok=True)
    result = subprocess.run([latexmk, "-pdf", "-interaction=nonstopmode", "-halt-on-error",
                             "-file-line-error", f"-outdir=build/{target}", source.name], cwd=paper)
    if result.returncode:
        return result.returncode
    log = (output / f"{source.stem}.log").read_text(encoding="utf-8", errors="replace")
    if re.search(r"(?:Citation|Reference) .* undefined|There were undefined (?:references|citations)", log):
        print("Paper build failed: unresolved references or citations remain.", file=sys.stderr)
        return 1
    if not (output / f"{source.stem}.pdf").is_file():
        print("Paper build failed: no PDF was generated.", file=sys.stderr)
        return 1
    print(f"Built {output / (source.stem + '.pdf')}; inspect rendered pages before submission.")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--target", choices=MANUSCRIPTS, default=DEFAULT_TARGET)
    selection.add_argument("--all", action="store_true", help="Build each independent draft and the final project")
    args = parser.parse_args()
    latexmk = shutil.which("latexmk")
    if not latexmk:
        print("Paper build unavailable: install a TeX distribution with latexmk and pdfLaTeX on PATH. See docs/SETUP.md.", file=sys.stderr)
        return 2
    results = [build_one(latexmk, target) for target in (MANUSCRIPTS if args.all else [args.target])]
    return next((result for result in results if result), 0)


if __name__ == "__main__":
    sys.exit(main())
