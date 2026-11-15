# 2026-11-15: EXP-029 notebook made lighter on RAM after the smoke test was killed

Task: EVAL-022 (still todo). Rhenel Jhon R. Sajol ran the EXP-029 notebook on Colab (Tesla T4).

## What happened

- **The failure:** the pre-flight smoke test (Step 4) was killed after about 3 minutes. The error was `CalledProcessError: ... died with <Signals.SIGKILL: 9>`.
- **Cause:** on Colab, signal 9 means the machine ran out of system RAM. The timing fits the runner loading its models: both TTS models, Whisper large-v3, WavLM and UTMOS.
- **Precedent:** EXP-023's first smoke test failed the same way and was fixed in commit `7d35e81`.
- **No results:** no smoke results were written, and nothing has been synthesized for EXP-029.

## What changed

All changes are in the notebook only. None can change a result.

`code/colab/DMOSpeech2_EXP029_Damped_Transmissibility_STAC.ipynb`:
- **GPU check:** it now uses `nvidia-smi` instead of PyTorch, so the notebook kernel no longer holds a PyTorch and CUDA context in RAM. The cell stops if the GPU is not a Tesla T4, because the smoke test reproduces EXP-027's T4 scores.
- **Smoke test and full run:**
  - They start the runner with `MALLOC_ARENA_MAX=2` and stream its output into the cell, so progress is visible.
  - They print the available RAM first.
  - On exit code -9 they say to delete the runtime and run all again. The full run resumes from the cases already saved, which Step 3 restores from Drive.
- **Step 4 markdown:** it now names the -9 exit and how to recover.

**What stays frozen:** the runner and its dependencies are hashed in the preregistration and were not changed. `check_prereg()` still accepts `e58f3856ad20...`.

## Checks run

- `check_prereg()`: accepts `e58f3856ad20...`.
- All notebook code cells compile.
- `python -m unittest discover -s code/tests -p "test_damped_transmissibility_stac.py"`: 5 tests, OK.
- `python scripts/check_workspace.py`: PASSED.
- `python -m unittest discover -s code/tests`: 282 tests, OK.
- LaTeX build: not run. The manuscript did not change.

## Remaining blockers

The GPU run itself. If a fresh runtime still runs out of RAM, the next step would be a change to the runner, such as loading the scoring models later or with lower memory use. That needs a dated amendment and a new freeze before any synthesis.
