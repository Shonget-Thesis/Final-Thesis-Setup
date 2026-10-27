# 2026-10-27: Citations for Appendix D (preliminary experiments)

User request: provide citations that support the preliminary testing results, especially Appendix D.

## Changed

- paper/references.bib: four entries, each located through a web search or the publisher/preprint page on 2026-10-27: Hanley and Lippman-Hand (1983, JAMA 249(13):1743-1745); Cawley and Talbot (2010, JMLR 11:2079-2107); Wang et al. (Interspeech 2024, arXiv:2406.05699, "Accepted to INTERSPEECH 2024" per its arXiv page); Kim and Stern (Interspeech 2008).
- paper/chapters/appendices.tex, Appendix D, four added citations and no change to any result:
  - D.2: noisy audio prompts degrade flow-matching zero-shot TTS (Wang et al.), next to the finding that almost all fixed-switch failures came from noisy prompts.
  - D.4: after zero observed failures the upper confidence limit shrinks as 1/n, about 3/n at 95% (Hanley and Lippman-Hand), next to the "about 3%" certifiable risk at delta = 0.10.
  - D.4: reusing data that guided earlier choices to select and then report performance is a known source of selection bias (Cawley and Talbot), next to the "may be optimistic" caveat.
  - D.5: blind SNR estimation from waveform statistics is established practice (Kim and Stern); the study's estimator uses a frame-energy ratio instead.
- Not cited: endpoint cushioning, the student restart level and the fixed-switch trend itself. No source found supports those; they remain the team's own preliminary results.

## Checks run

- latexmk -g -Werror -interaction=nonstopmode in paper/: exit 0, 75 pages, no undefined citations; the four new keys appear in main.bbl.
- python scripts/check_workspace.py and python -m unittest discover -s code/tests: see below.

## Remaining

- Primary-literature verification is incomplete. Hanley and Lippman-Hand: the 3/N statement was confirmed from a secondary quotation, not the paper's text. Kim and Stern: title and venue confirmed, no DOI or pages recorded. Wang et al.: the venue pages are not recorded. A team member should check each against the original before submission.
- Appendix D's own results still come from development data; the citations support the framing, not the findings.
