# Thesis workspace

Working title: **Input-Adaptive Teacher-to-Student Transition Timing for Efficient Diffusion Speech Synthesis**. This is a research workspace; the title, authors, institutional format, and contribution still need team/adviser review.

Start with [current status](docs/progress/STATUS.md), the [task board](docs/progress/TASKS.md), and [research direction](docs/RESEARCH_DIRECTION.md).

## Repository layout

| Folder | Purpose |
| --- | --- |
| [code/](code/README.md) | Maintained analysis, tests, experiment configurations, and imported DMOSpeech 2 source |
| [paper/](paper/README.md) | Self-contained, Overleaf-compatible LaTeX project: final manuscript (main.tex + chapters/) and single-file drafts |
| [docs/](docs/README.md) | Setup, research decisions, workflow, progress, and reproducibility records |
| [research/](research/README.md) | Historical evidence, audits, protocols, literature notes, and experiment records |
| scripts/ | Workspace validation and progress rendering |
| .github/ | Pull-request checklist and automatic validation/build workflow |

## First run

Run from this repository root. The audit and workspace tools need only Python 3.10+ and its standard library; GPU inference has a separate environment.

~~~powershell
python scripts/check_workspace.py
python -m unittest discover -s code/tests
python code/analysis/replay_timing.py
python scripts/render_progress.py
~~~

Build the papers after installing a TeX distribution with latexmk and pdfLaTeX (or save any .tex file in VS Code with LaTeX Workshop):

~~~powershell
cd paper
latexmk
~~~

This builds paper/main.tex and every draft into paper/build/. See the [paper guide](paper/README.md) for the VS Code and Overleaf workflow, [setup](docs/SETUP.md) for toolchain installation, and [the workflow](docs/WORKFLOW.md) for the daily update process. MiKTeX and Strawberry Perl are installed locally and all three PDFs build. GitHub Actions is configured to compile them on push/PR, but has not been run yet.

## What is already here

The selected DMOSpeech 2 source, packaged experiment scripts, telemetry, and October 5 audit were copied from the sibling Shonget-DMOSpeech2 working tree. [SOURCE_MANIFEST.json](docs/SOURCE_MANIFEST.json) records the original revision, paths, and SHA-256 hashes. The sibling repository was not moved or edited. Audio, checkpoints, large datasets, and duplicate source copies were excluded.

The executable timing replay reproduces development-record arithmetic. It does not synthesize speech or establish held-out performance. The manuscript explicitly distinguishes this evidence from experiments that still need to be run. Imported reports can contain stronger historical claims; consult the [audit](research/audits/2026-10-05/FINDINGS.md) before using them.

## Completion rule

Every substantive change has a task, evidence, a dated work log, and recorded checks. Run the workspace check before handoff. CI checks structure, provenance, local documentation links, task evidence, manuscript references, analysis tests, and LaTeX compilation. Pull requests that change code, paper, or research also need progress updates. These checks assess record completeness, not scientific truth or institutional approval.
