# 2026-11-17: Revision matrix R2, R21, R23 (dataset) and R24

User request (Mark Vincent D. Limpahan): close the revision-matrix items for sample inputs (R2), the unseen dataset (R21, R23) and the prototype (R24). For the unseen dataset the implementor, Rhenel Jhon R. Sajol, named LibriSpeech-PC (https://www.openslr.org/145/; mirror https://huggingface.co/datasets/yoom618/librispeech_pc). Both Seed-TTS test-en and LibriSpeech-PC are test sets only.

## Changed

- references.bib: meister2023librispeechpc (Meister et al., arXiv 2310.02943). Authors and arXiv number checked on 2026-10-27/11-17 against the arXiv and OpenSLR pages; OpenSLR's own BibTeX lists it as a preprint, so no conference venue is given. OpenSLR 145 holds only transcripts; the audio is LibriSpeech test-clean (panayotov2015librispeech, already in the bibliography).
- Chapter 1 (scope) and Chapter 2 (Section 2.6): the two R13 to-do notes are replaced by the dataset: LibriSpeech-PC test-clean as a second test set, used only for testing.
- Chapter 3 (Section 3.4.3): a paragraph on the second test set: audio and transcripts, different source (read audiobooks), one utterance per speaker chosen in the manifest before testing, and results reported as a check of transfer, not as part of the guarantee, because Proposition 3.6 covers only the calibration distribution.
- Appendix C: the prototype is described as existing (code/prototype, PROTO-001) and labeled a demonstration with development thresholds; new Table C.1 gives one development speaker under clean, 25 dB and 15 dB prompts with the prototype's decision (SNR 44.3 / 27.8 / 18.0 dB; Step 4 / Step 5 / Step 8), copied from code/prototype/README.md, which reports that they match EXP-031's T4 records.
- revision-matrix.tex: R2, R21 and R24 marked done with locations; R23 keeps one pending item, the moderate-noise level.

## Decisions not made here

- The moderate-noise level for the protocol (R23). Appendix D reports that 18-22 dB white noise passes the 20 dB gate, so the level matters; left to the team.
- The LibriSpeech-PC subset and size (for example F5-TTS's 4-10 s cross-sentence list) are not fixed in the manuscript; it says they are chosen in the manifest before testing.
- Table 3.4 (splits) was not given a LibriSpeech-PC row.

## Checks run

- latexmk -g -Werror -interaction=nonstopmode in paper/: exit 0, 90 pages, no undefined references; Table C.1 was widened-then-wrapped after an overfull box. One pre-existing 2.4 pt overfull line remains in Chapter 2 (line 262).
- pdflatex revision-matrix.tex (twice, after main): exit 0, no undefined references.
- python scripts/check_workspace.py and python -m unittest discover -s code/tests: see the session.
