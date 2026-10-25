# Code

| Path | Role |
| --- | --- |
| [scheduling/](scheduling/) | Maintained deterministic adaptive handover sampler, metrics, and checkpoint downloader |
| [colab/](colab/) | Google Colab GPU (Tesla T4 / A100) runners, timing benchmarks, and turnkey notebooks |
| [colab/run_followup_tests.py](colab/run_followup_tests.py) | Follow-up T4 tests (seed-noise floor, fresh held-out speakers, degraded prompts) driven by [DMOSpeech2_Followup_Tests.ipynb](colab/DMOSpeech2_Followup_Tests.ipynb); see [EXP-011](../research/experiments/EXP-011-followup-t4/README.md) |
| [experiments/phase2_calib/](experiments/phase2_calib/) | Learn-Then-Test (LTT) PAC statistical calibration algorithms |
| [experiments/phase3_test/](experiments/phase3_test/) | Held-out statistical benchmark runner and hypothesis tests |
| [analysis/replay_timing.py](analysis/replay_timing.py) | Maintained, standard-library timing replay using stored records |
| [analysis/ltt_sim.py](analysis/ltt_sim.py) | Learn-Then-Test finite-sample PAC simulation |
| [configs/evaluation_v1.json](configs/evaluation_v1.json) | Legacy evaluation threshold parameters |
| [tests/](tests/) | Automated unit tests and workspace integrity safeguards |
| [vendor/dmospeech2/](vendor/dmospeech2/) | Upstream DMOSpeech 2 and F5-TTS baseline model source |
| [experiments/legacy/](experiments/legacy/) | Imported packaged historical scripts |

Run `python code/analysis/replay_timing.py` from the repository root. GPU speech synthesis runs via Google Colab notebooks in `code/colab/` (see [COLAB_QUICKSTART_GUIDE.md](colab/COLAB_QUICKSTART_GUIDE.md)). See [setup](../docs/SETUP.md) for environment details.

The Colab notebooks clone Shonget-DMOSpeech2 and run its `Handover Scheduling/` code; `scheduling/` and the older `colab/` runners here are reference copies of that code and import `core.*` paths that exist only in that clone. Results imported from those runs are in [research/evidence/benchmarks/](../research/evidence/benchmarks/MANIFEST.json), with source commits and SHA-256 hashes. The thesis uses Colab T4 results only; CPU runs are not imported.
