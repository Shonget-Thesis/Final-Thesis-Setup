# 2026-10-27: Presenter script rewritten as a slide walkthrough

Task: PRES-001. User request: update the script so it describes the content of each slide, with some explanation, as a walkthrough.

## Changed

- presentation/slides/*.html: every slide's presenter script was rewritten to walk through what is on screen (panels left to right, table columns, diagram regions, the zoom stages) and explain each point, using only figures that appear on the slides and in the manuscript. The last appendix slide ends with "Thank you; we welcome your questions." Illustrative and development figures are named as such.
- Auto-play durations (data-dur) now follow script length at about 145 words per minute: 45-80 s per slide, about 21 minutes in total, about 17 minutes for the 16 main slides.
- presentation/SCRIPT.md and index.html regenerated with build.py --sync-md; README updated. The PDF does not contain the scripts and was not regenerated.

## Checks run

- python presentation/build.py --sync-md: 19 slides, 19 script sections.
- python scripts/check_workspace.py and python -m unittest discover -s code/tests: see the session.

## Open

- The team has not reviewed the wording or rehearsed the timing; the defense time limit is not recorded here.
