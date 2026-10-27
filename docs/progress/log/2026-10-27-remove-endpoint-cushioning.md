# 2026-10-27: Endpoint cushioning removed from Appendix D

User request: remove the endpoint cushioning experiment from the manuscript.

## Changed

- paper/chapters/appendices.tex: deleted the "Endpoint Cushioning" subsection (96 LibriSpeech pairs, tanh weight) and its row in Table D.1. Appendix D now has D.1 Fixed Switching Times, D.2 Student Restart Level, D.3 Stabilization Policy with a Cross-Validated Threshold, D.4 Development-Split Signal Diagnostic and D.5 Plan for Further Investigation of the Stabilization Signal.
- presentation/slides/17-two-signals.html: the source footer pointed to D.5 for the development diagnostic; it is now D.4. STATUS.md refers to the fixed-switching analysis as D.1.
- Earlier dated logs keep the old subsection numbers (D.2, D.4, D.5) as written at the time.

## Checks run

- latexmk -g -Werror -interaction=nonstopmode in paper/: exit 0, 75 pages, no undefined citations. No other file in paper/ mentioned cushioning.
- python presentation/build.py --sync-md --pdf: 22 slides, PDF regenerated.

## Open

- LibriSpeech is now cited only in Section 3.9 (Ethical Considerations) as a corpus the study uses; the cushioning experiment was its only use in the study. The team should decide whether to keep or drop that mention.
- The student-restart reasoning in Chapter 2 and Appendix D.2 was reviewed separately; see the discussion in the session. No text was changed for it.
