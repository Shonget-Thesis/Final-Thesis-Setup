# Setup and reproducibility

## Lightweight workspace tools

Use Python 3.10 or newer. No pip packages are needed for the maintained replay, validation scripts, or tests. Python 3.13 is available on the setup machine; this does not validate model compatibility with that version.

~~~powershell
python scripts/check_workspace.py
python -m unittest discover -s code/tests
python code/analysis/replay_timing.py --output research/experiments/EXP-000-timing-replay/results.json
~~~

The replay reads repository-local evidence and resolves paths relative to its own file, so it also works when launched from another working directory.

## Model/inference environment (separate, not validated by setup)

The imported upstream README recommends Python 3.10. Use a separate environment, inspect [upstream requirements](../code/vendor/dmospeech2/requirements.txt), and select a PyTorch build compatible with the intended hardware before installing those requirements. They are historical, loosely constrained dependencies, not a tested lockfile. A resolved environment export and a GPU smoke test are tracked as open tasks.

From code/vendor/dmospeech2, follow its README for checkpoints and src/demo.ipynb. Keep working-directory-sensitive upstream scripts in that directory. The snapshot excludes sample audio and checkpoints; obtain authorized inputs and record their hashes before inference. Packaged scripts in code/experiments/legacy still contain workstation-specific assumptions and need porting before use. This setup does not claim they run unchanged.

Do not install the model requirements into the lightweight checking environment. After establishing a working model environment, save Python/CUDA/driver/GPU versions, pip freeze or conda export, checkpoint hashes, and the code revision in the experiment record.

## LaTeX

Installed and verified on this Windows machine on 2026-10-09: MiKTeX 25.12, Strawberry Perl 5.42.3.1, and latexmk 4.88. The manuscript compiles with latexmk in paper/ (the earlier drafts were removed on 2026-10-24, ADR-004). Restart VS Code and open a new terminal after installation so PATH includes MiKTeX. Confirm discovery with Get-Command pdflatex,latexmk,perl. See the [installation record](progress/log/2026-10-09-latex-installation.md).


Windows: install MiKTeX or TeX Live with latexmk, pdfLaTeX, BibTeX, and Perl available on PATH. MiKTeX may request missing packages on the first build. The final manuscript uses biblatex-apa with biber. Linux/CI: install latexmk, texlive-latex-extra, texlive-fonts-recommended, texlive-bibtex-extra, texlive-science (algorithm packages), cm-super (sharp small fonts), and biber.

~~~powershell
cd paper
latexmk
~~~

PDFs go to paper/build/. For editing with live preview, install the LaTeX Workshop VS Code extension (recommended in .vscode/extensions.json); saving any .tex file rebuilds it. CI runs latexmk -Werror, which fails on unresolved citations/references. For Overleaf, upload the paper/ folder as a zip; main.tex is the main document. The final manuscript uses the CITC style (paper/citc.sty); see the paper guide. See the [paper guide](../paper/README.md).

## Automation

The GitHub workflow runs lightweight checks/tests and compiles the manuscript, then uploads its PDF as an artifact. It is configured locally; it only runs after these changes reach GitHub. No branch protection, remote push, or adviser approval is implied. Review the first workflow result and record its URL in the tracker.
