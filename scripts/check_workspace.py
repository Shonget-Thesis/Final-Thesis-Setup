"""Check repository records, source integrity, paper references, and PR updates."""
import argparse
from datetime import date
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
from urllib.parse import unquote

from render_progress import render
from manuscripts import MANUSCRIPTS

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = ["README.md", "AGENTS.md", "code/README.md", "paper/README.md", "docs/SETUP.md", "docs/WORKFLOW.md", "docs/RESEARCH_DIRECTION.md",
            "docs/progress/STATUS.md", "docs/progress/tasks.json", "docs/progress/TASKS.md",
            "docs/SOURCE_MANIFEST.json", "research/protocols/evaluation.md"]
REQUIRED += list(MANUSCRIPTS.values()) + ["paper/references.bib", "paper/citc.sty", "paper/frontmatter.tex"]
REQUIRED += [f"paper/chapters/{name}.tex" for name in
             ("01-introduction", "02-related-work", "03-methodology", "04-results", "05-conclusion")]
ARCHIVED = ("code/vendor/", "code/experiments/legacy/", "research/evidence/legacy/", "research/audits/2026-10-05/")


def local_file(root, value):
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        raise ValueError(f"Expected repository-relative path: {value!r}")
    resolved = (root / value).resolve()
    if not resolved.is_relative_to(root.resolve()) or not resolved.is_file():
        raise ValueError(f"Missing file or path outside repository: {value}")
    return resolved


def validate(root, submission=False):
    errors = []
    for name in REQUIRED:
        if not (root / name).is_file():
            errors.append(f"Missing required file: {name}")
    try:
        data = json.loads((root / "docs/progress/tasks.json").read_text(encoding="utf-8"))
        if not data["tasks"]:
            errors.append("Task tracker is empty")
        ids = set()
        for task in data["tasks"]:
            ident = task["id"]
            if ident in ids or not re.fullmatch(r"[A-Z]+-\d{3}", ident):
                errors.append(f"Invalid or duplicate task ID: {ident}")
            ids.add(ident)
            for field in ("title", "owner", "acceptance", "updated"):
                if not str(task[field]).strip():
                    errors.append(f"{ident}: empty {field}")
            date.fromisoformat(task["updated"])
            if task["status"] not in {"todo", "in_progress", "blocked", "review", "done"}:
                errors.append(f"{ident}: unknown status")
            if not isinstance(task["evidence"], list):
                raise ValueError(f"{ident}: evidence must be an array")
            for evidence in task["evidence"]:
                local_file(root, evidence)
            if task["status"] == "done" and (not task["evidence"] or not task.get("validation", "").strip()):
                errors.append(f"{ident}: done requires evidence and validation")
            if task["status"] == "blocked" and not task.get("blocker", "").strip():
                errors.append(f"{ident}: blocked requires a blocker")
        expected = render(data)
        actual = (root / "docs/progress/TASKS.md").read_text(encoding="utf-8")
        if actual != expected:
            errors.append("Task board is stale: run python scripts/render_progress.py")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        errors.append(f"Task tracker: {exc}")
    try:
        manifest = json.loads((root / "docs/SOURCE_MANIFEST.json").read_text(encoding="utf-8"))
        if not manifest["files"]:
            errors.append("Source manifest is empty")
        for item in manifest["files"]:
            file = local_file(root, item["destination"])
            if hashlib.sha256(file.read_bytes()).hexdigest() != item["sha256"]:
                errors.append(f"Imported source changed: {item['destination']}")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        errors.append(f"Source manifest: {exc}")
    for file in root.rglob("*.md"):
        relative = file.relative_to(root).as_posix()
        if relative.startswith(ARCHIVED) or any(p in {".git", ".venv", "venv", "build"} for p in file.relative_to(root).parts):
            continue
        text = file.read_text(encoding="utf-8")
        for link in re.findall(r"\[[^\]]*\]\(([^\s)]+)\)", text):
            if link.startswith(("https://", "http://", "mailto:", "#")):
                continue
            link_path = unquote(link.split("#", 1)[0])
            if link_path and not (file.parent / link_path).exists():
                errors.append(f"{relative}: broken local link: {link}")
    for target, entry in MANUSCRIPTS.items():
        source = root / entry
        paper = source.parent
        bib = (paper / "references.bib").read_text(encoding="utf-8") if (paper / "references.bib").exists() else ""
        keys = set(re.findall(r"@\w+\s*\{\s*([^,]+),", bib))
        files = [source, *(f for f in sorted(paper.glob("*.tex")) if f != source),
                 *sorted((paper / "chapters").glob("*.tex"))]
        for file in files:
            if not file.exists():
                continue
            content = file.read_text(encoding="utf-8")
            for ref in re.findall(r"\\(?:input|include)\{([^}]+)\}", content):
                if not (paper / (ref if ref.endswith(".tex") else ref + ".tex")).is_file():
                    errors.append(f"{file.name}: missing TeX input {ref}")
            # \cite, \parencite, \textcite, ...; \workcite{Name}{key} names the author first.
            cited = re.findall(r"\\(?!workcite)[A-Za-z]*cite[A-Za-z]*\*?(?:\[[^\]]*\])*\{([^}]+)\}", content)
            cited += re.findall(r"\\workcite\{[^}]*\}\{([^}]+)\}", content)
            for group in cited:
                for key in group.split(","):
                    if key.strip() not in keys:
                        errors.append(f"{file.name}: missing bibliography key {key}")
            if submission and re.search(r"\bTBD\b|\bTODO\b|\\pending\{|\\placeholder\{", content):
                errors.append(f"{file.name}: unfinished manuscript content")
    return errors


def check_progress_diff(root, base):
    diff = subprocess.run(["git", "diff", "--name-only", "--diff-filter=ACDMRT", f"{base}...HEAD"],
                          cwd=root, check=True, capture_output=True, text=True)
    paths = set(diff.stdout.splitlines())
    substantive = any(p.startswith(("code/", "paper/", "research/", "scripts/")) for p in paths)
    if not substantive:
        return []
    required = {"docs/progress/tasks.json", "docs/progress/STATUS.md"}
    errors = [f"PR needs a progress update: {p}" for p in sorted(required - paths)]
    if not any(re.fullmatch(r"docs/progress/log/\d{4}-\d{2}-\d{2}-.+\.md", p) for p in paths):
        errors.append("PR needs a dated docs/progress/log/YYYY-MM-DD-description.md entry")
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission", action="store_true", help="Reject unfinished markers in the final manuscript (paper/main.tex and chapters) only")
    parser.add_argument("--base", help="Git base revision for committed PR progress-update checks")
    args = parser.parse_args()
    errors = validate(ROOT, args.submission)
    if args.base:
        try:
            errors.extend(check_progress_diff(ROOT, args.base))
        except (OSError, subprocess.CalledProcessError) as exc:
            errors.append(f"Could not check PR diff: {exc}")
    if errors:
        print("Workspace check FAILED:\n" + "\n".join(f"- {e}" for e in errors))
        return 1
    print("Workspace check PASSED: structure, task evidence, source hashes, links, and manuscript references.")
    print("Draft status is allowed. This check does not validate scientific claims, inference, or PDF layout.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
