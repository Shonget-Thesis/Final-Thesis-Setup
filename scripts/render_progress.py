"""Render the task board from its single source of truth."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def render(data):
    counts = {status: sum(t["status"] == status for t in data["tasks"])
              for status in ("todo", "in_progress", "blocked", "review", "done")}
    lines = ["# Task board", "", "Generated from tasks.json; edit the JSON and run python scripts/render_progress.py.", "",
             " | ".join(f"{status}: {count}" for status, count in counts.items()), "",
             "| ID | Task | Owner | Status | Updated | Evidence |", "| --- | --- | --- | --- | --- | --- |"]
    for task in data["tasks"]:
        evidence = ", ".join(f"[{Path(p).name}](../../{p})" for p in task["evidence"]) or "Pending"
        cells = [task["id"], task["title"], task["owner"], task["status"], task["updated"], evidence]
        lines.append("| " + " | ".join(str(c).replace("|", "\\|") for c in cells) + " |")
    lines.extend(["", "Completion requires acceptance evidence and a recorded validation. Unassigned owners need team assignment.", ""])
    return "\n".join(lines)


if __name__ == "__main__":
    source = ROOT / "docs/progress/tasks.json"
    destination = ROOT / "docs/progress/TASKS.md"
    destination.write_text(render(json.loads(source.read_text(encoding="utf-8"))), encoding="utf-8")
    print(f"Updated {destination.relative_to(ROOT)}")
