# 2026-10-09: Thesis workspace setup

Task IDs: INFRA-001, PAPER-001. Author: Codex, acting on the requested repository organization.

## Changes

Created code, paper, docs, and research areas in Final-Thesis-Setup. Imported 107 selected historical files with source revision and SHA-256 provenance. Added a portable timing replay, tests, task board, work/experiment templates, repository instructions, CI checks, and a LaTeX manuscript scaffold.

The sibling Shonget-DMOSpeech2 repository was left unchanged. Its audit was untracked at import, which is recorded in the manifest. Imported historical reports keep their wording, and maintained documentation explains their evidential limits.

## Evidence and validation

Environment: Python 3.13 on Windows. Maintained tools use Python 3.10-compatible standard-library APIs; CI targets Python 3.10 but has not run remotely.

| Command | Actual outcome |
| --- | --- |
| python scripts/render_progress.py | Generated the readable task board from tasks.json |
| python code/analysis/replay_timing.py --output research/experiments/EXP-000-timing-replay/results.json | Saved reproducible 48-instance arithmetic and input hashes |
| python scripts/check_workspace.py | Passed structure, task evidence, 107 imported source hashes, maintained local links, and manuscript references |
| python -m unittest discover -s code/tests -v | All 11 tests passed, covering the independent audit expectations, thresholds, joins, source drift, missing evidence, stale task board, missing references, draft gate, and PR progress requirements |
| python scripts/build_paper.py | Build unavailable: latexmk is absent; no PDF was generated |

The replay matches the audited fixed-policy failure counts. SNR-only recorded mean NFE is 11.458333, with 0 relative failures and 4 strict failures (IDs 23, 24, 32, 44). All 48 packaged hybrid WER/SIM/latency records match selected fixed records; this does not establish independent runtime performance.

The Git attributes preserve imported bytes across operating systems so the SHA-256 checks remain meaningful. Changes are local and uncommitted. No GPU inference, new data collection, literature verification, or adviser approval was performed.

## Decisions and blockers

See ADR-001. No local LaTeX toolchain is available; compilation, rendered layout review, model inference, and remote CI remain unverified. The next team actions are to assign owners, confirm metadata/template, and recover generation provenance before new experiments.
