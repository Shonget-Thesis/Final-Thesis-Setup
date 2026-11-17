# 2026-11-17: Remove slides 20, 21, and 22 from presentation/index.html

**Task:** `PRES-002` (`review`, updating `PRES-001`)  
**Targets:** [presentation/index.html](../../../presentation/index.html), [presentation/SCRIPT.md](../../../presentation/SCRIPT.md), [presentation/README.md](../../../presentation/README.md), [presentation/proposal-defense-slides.pdf](../../../presentation/proposal-defense-slides.pdf)

## Changes Applied
1. Removed modular slide files `presentation/slides/20-appendix-stac-architecture.html`, `presentation/slides/21-appendix-stac-results.html`, and `presentation/slides/22-appendix-two-axis-architecture.html` (recoverable from git history if needed).
2. Rebuilt [presentation/index.html](../../../presentation/index.html), synced [presentation/SCRIPT.md](../../../presentation/SCRIPT.md), and re-exported [presentation/proposal-defense-slides.pdf](../../../presentation/proposal-defense-slides.pdf) via `python presentation/build.py --sync-md --pdf`:
   - `presentation/index.html` now contains `19` scenes (`01`–`16` main proposal slides + appendix slides `17`–`19`).
   - `presentation/SCRIPT.md` is synced to `19` scenes (~21 minutes total at auto-play pace).
   - `presentation/proposal-defense-slides.pdf` was re-exported via headless Edge at `19` slides (`9,535,551` bytes), with zero leftover `slide_pdf_` Edge processes.
3. Updated [presentation/README.md](../../../presentation/README.md) to reflect the `19`-scene deck (`17`–`19` appendix slides) and updated `PRES-001` evidence + added `PRES-002` in [tasks.json](../tasks.json).

## Checks Run
- `python presentation/build.py --sync-md --pdf`: built `index.html` with `19` slides (`6.3 ms`), synced `SCRIPT.md`, exported `19` slides to `proposal-defense-slides.pdf` (`9,535,551` bytes).
- Verified `19` `<section class="scene` entries in `presentation/index.html` and `19` `/Type /Page` entries in `presentation/proposal-defense-slides.pdf`.
- `python scripts/render_progress.py`, `python scripts/check_workspace.py`, and `python -m unittest discover -s code/tests`: run after all final edits.

## Remaining Blockers
- Team confirmation before the defense (`PRES-001` / `PRES-002`).
