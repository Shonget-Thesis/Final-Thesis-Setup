# Working process

## Every work session

1. Read STATUS.md and select a task in progress/tasks.json. Assign an owner and set it to in_progress.
2. Make the implementation or manuscript change. Record the task ID in an experiment or decision when relevant.
3. Run the applicable checks. Record the exact command, result, and limitations in a dated file under docs/progress/log/.
4. Update the task's status, updated date, evidence paths, and validation note. Update docs/progress/STATUS.md.
5. Run python scripts/render_progress.py to regenerate the readable task board, then python scripts/check_workspace.py. Commit the related code, paper, evidence, and documentation together.

Allowed task statuses: todo, in_progress, blocked, review, done. A blocked task needs a blocker. A done task needs at least one existing repository-relative evidence file and a nonempty validation description. "review" means a deliverable exists but still awaits a stated check or reviewer. Team members must judge whether the evidence is sufficient.

## Experiments

Create research/experiments/EXP-NNN-short-name/ from the experiment template. Record the hypothesis, protocol version, code revision, exact command, dependency snapshot, dataset/speaker splits, checkpoint and input hashes, seeds, runtime hardware, raw outputs, evaluator version, and uncertainty method. Store large assets externally and record their URI and SHA-256. Do not overwrite old runs or silently retune against test speakers.

## Paper edits

Edit the relevant standalone paper/drafts/<name>.tex file. Draft metadata, chapters, and bibliography belong in that file. Record final-manuscript sources in paper/references.bib and link reported results to experiment IDs. Keep unsupported claims marked as pending. Build the PDF and inspect layout before adviser submission; CI compilation alone does not establish layout quality. Update draft metadata inline, or the metadata block in paper/main.tex when the title, authors, adviser, and institution are confirmed. When selecting a submission version, transfer reviewed text into the separate paper/chapters/ files, record its source draft/revision, and preserve draft history. The --submission check applies to paper/main.tex and paper/chapters/ only.

## Weekly review

Review blocked tasks and unassigned owners, check evidence for newly completed items, compare manuscript claims to experiment records, inspect the compiled PDF, and add adviser feedback as a dated decision or meeting note. Record the next review date when agreed by the team; no meeting date is invented here.

## Pull requests

The PR workflow requires tasks.json, STATUS.md, and a dated work log to change when maintained code, scripts, paper, or research changes. The checker can be run locally as python scripts/check_workspace.py --base <base-commit>. CI verifies bookkeeping and compilation; it does not assess novelty, scientific validity, or adviser acceptance.
