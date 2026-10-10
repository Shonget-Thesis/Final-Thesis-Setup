# Code

| Path | Role |
| --- | --- |
| analysis/replay_timing.py | Maintained, standard-library timing replay using stored records |
| configs/evaluation_v1.json | Explicit legacy-replay criteria and units; not an approved held-out protocol |
| tests/ | Regression and boundary checks for the maintained analysis |
| vendor/dmospeech2/ | Selected historical source snapshot with original LICENSE and README |
| experiments/legacy/ | Imported packaged scripts, awaiting path/provenance reconciliation |

Run python code/analysis/replay_timing.py from the repository root. It recomputes flags rather than trusting stored pass/fail fields. Its result is descriptive development evidence.

New scheduler and evaluator implementations should live in maintained subfolders here. Keep the imported snapshot intact; document patches separately with tests and a new revision. See [setup](../docs/SETUP.md) before attempting model inference.
