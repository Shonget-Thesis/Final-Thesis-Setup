# Preliminary test: Selective Handover (EXP-020)

Development speakers only; 360 of 360 cases, 20 of 20 diversity pairs; preregistration `7afeef06fabefaa8…`. Every number is computed from the CSV files by `run_selective_handover_prelim.py analyze`.

Combination names give the source of (voice, slow, fine): T = teacher, S = student draft. TTT is `plain_4` and SSS is `student4`.

## Preregistered criteria (primary combination STT)

Selective Handover advances to fresh speakers only if every criterion passes.

| C1_noisy_failures | C2_clean_failures | C3_voice_is_the_cause | C4_diversity | C5_beats_calibrated_snr | C6_beats_ladder_without_selection | Advance |
|---|---|---|---|---|---|---|
| PASS | PASS | PASS | PASS | FAIL | FAIL | **NO** |

## The map: effect of taking each part from the student instead of the teacher (Step 4, mean [95% speaker CI])

| Part | Noisy similarity | Noisy capped WER | Noisy failure rate | Clean UTMOS | CVf0 |
|---|---|---|---|---|---|
| voice | +0.0752 [+0.0635, +0.0874] | -0.0290 [-0.0472, -0.0114] | -0.2267 [-0.2942, -0.1617] | -0.003 [-0.030, +0.025] | -0.0020 [-0.0086, +0.0019] |
| slow | -0.0028 [-0.0087, +0.0031] | -0.0173 [-0.0256, -0.0088] | +0.0150 [-0.0325, +0.0667] | +0.037 [-0.009, +0.082] | -0.0402 [-0.0589, -0.0249] |
| fine | +0.0064 [+0.0042, +0.0087] | -0.0068 [-0.0120, -0.0019] | -0.0150 [-0.0333, +0.0042] | -0.019 [-0.048, +0.009] | -0.0010 [-0.0027, +0.0011] |

## Added failures against Step 8 (Definition 3.1)

| Output | Passes | clean | noisy15 | gauss18 | gauss20 | gauss22 | gauss25 |
|---|---:|---:|---:|---:|---:|---:|---:|
| student4 | 4 | 8 | 17 | 5 | 5 | 11 | 13 |
| plain_4 | 13 | 2 | 40 | 33 | 26 | 19 | 8 |
| plain_5 | 15 | 2 | 30 | 27 | 24 | 19 | 8 |
| plain_6 | 17 | 0 | 31 | 19 | 16 | 14 | 5 |
| plain_7 | 19 | 0 | 25 | 15 | 14 | 11 | 4 |
| plain_8 | 21 | 0 | 0 | 0 | 0 | 0 | 0 |
| STT_4 | 14 | 2 | 20 | 19 | 14 | 10 | 11 |
| TST_4 | 14 | 6 | 35 | 32 | 28 | 30 | 20 |
| TTS_4 | 14 | 1 | 38 | 35 | 29 | 21 | 11 |
| SST_4 | 14 | 5 | 21 | 21 | 16 | 11 | 13 |
| STS_4 | 14 | 2 | 21 | 19 | 13 | 10 | 12 |
| TSS_4 | 14 | 7 | 39 | 31 | 25 | 32 | 22 |
| STT_5 | 16 | 3 | 22 | 17 | 11 | 10 | 13 |
| STT_6 | 18 | 0 | 15 | 17 | 10 | 8 | 11 |
| STT_7 | 20 | 0 | 18 | 14 | 11 | 8 | 10 |

## Mean WER (capped at 1 per utterance)

| Output | clean | noisy15 | gauss18 | gauss20 | gauss22 | gauss25 |
|---|---:|---:|---:|---:|---:|---:|
| student4 | 0.021 | 0.130 | 0.032 | 0.026 | 0.027 | 0.030 |
| plain_4 | 0.023 | 0.286 | 0.114 | 0.076 | 0.036 | 0.035 |
| plain_5 | 0.026 | 0.175 | 0.087 | 0.066 | 0.045 | 0.036 |
| plain_6 | 0.026 | 0.148 | 0.084 | 0.057 | 0.040 | 0.033 |
| plain_7 | 0.027 | 0.159 | 0.078 | 0.062 | 0.040 | 0.037 |
| plain_8 | 0.027 | 0.168 | 0.075 | 0.060 | 0.037 | 0.035 |
| STT_4 | 0.023 | 0.178 | 0.088 | 0.053 | 0.030 | 0.028 |
| TST_4 | 0.021 | 0.244 | 0.082 | 0.056 | 0.037 | 0.034 |
| TTS_4 | 0.025 | 0.260 | 0.122 | 0.070 | 0.041 | 0.036 |
| SST_4 | 0.021 | 0.143 | 0.096 | 0.056 | 0.040 | 0.030 |
| STS_4 | 0.025 | 0.172 | 0.101 | 0.060 | 0.033 | 0.029 |
| TSS_4 | 0.021 | 0.236 | 0.078 | 0.054 | 0.036 | 0.031 |
| STT_5 | 0.026 | 0.179 | 0.084 | 0.053 | 0.035 | 0.028 |
| STT_6 | 0.023 | 0.148 | 0.088 | 0.049 | 0.029 | 0.028 |
| STT_7 | 0.027 | 0.179 | 0.094 | 0.059 | 0.034 | 0.030 |

## Mean speaker similarity

| Output | clean | noisy15 | gauss18 | gauss20 | gauss22 | gauss25 |
|---|---:|---:|---:|---:|---:|---:|
| student4 | 0.937 | 0.802 | 0.849 | 0.868 | 0.889 | 0.904 |
| plain_4 | 0.936 | 0.684 | 0.727 | 0.777 | 0.821 | 0.876 |
| plain_5 | 0.935 | 0.687 | 0.740 | 0.782 | 0.821 | 0.877 |
| plain_6 | 0.936 | 0.689 | 0.739 | 0.782 | 0.827 | 0.879 |
| plain_7 | 0.936 | 0.691 | 0.742 | 0.785 | 0.825 | 0.880 |
| plain_8 | 0.936 | 0.698 | 0.740 | 0.785 | 0.827 | 0.877 |
| STT_4 | 0.936 | 0.789 | 0.832 | 0.851 | 0.868 | 0.896 |
| TST_4 | 0.936 | 0.682 | 0.729 | 0.769 | 0.807 | 0.866 |
| TTS_4 | 0.935 | 0.688 | 0.731 | 0.776 | 0.823 | 0.873 |
| SST_4 | 0.936 | 0.777 | 0.816 | 0.839 | 0.867 | 0.889 |
| STS_4 | 0.934 | 0.790 | 0.835 | 0.854 | 0.868 | 0.896 |
| TSS_4 | 0.936 | 0.677 | 0.729 | 0.766 | 0.807 | 0.866 |
| STT_5 | 0.935 | 0.795 | 0.832 | 0.850 | 0.867 | 0.895 |
| STT_6 | 0.935 | 0.800 | 0.831 | 0.854 | 0.869 | 0.895 |
| STT_7 | 0.936 | 0.795 | 0.833 | 0.853 | 0.871 | 0.894 |

## Mean UTMOS

| Output | clean | noisy15 | gauss18 | gauss20 | gauss22 | gauss25 |
|---|---:|---:|---:|---:|---:|---:|
| student4 | 3.93 | 1.78 | 2.29 | 2.69 | 3.13 | 3.55 |
| plain_4 | 3.92 | 1.25 | 1.25 | 1.27 | 1.47 | 2.16 |
| plain_5 | 3.92 | 1.25 | 1.25 | 1.27 | 1.48 | 2.14 |
| plain_6 | 3.92 | 1.25 | 1.25 | 1.27 | 1.49 | 2.12 |
| plain_7 | 3.93 | 1.25 | 1.25 | 1.27 | 1.47 | 2.09 |
| plain_8 | 3.92 | 1.25 | 1.25 | 1.27 | 1.45 | 2.08 |
| STT_4 | 3.90 | 1.92 | 2.26 | 2.59 | 2.83 | 3.08 |
| TST_4 | 3.93 | 1.25 | 1.25 | 1.27 | 1.43 | 2.11 |
| TTS_4 | 3.88 | 1.25 | 1.25 | 1.27 | 1.47 | 2.15 |
| SST_4 | 3.93 | 1.69 | 1.94 | 2.23 | 2.50 | 3.02 |
| STS_4 | 3.87 | 1.92 | 2.29 | 2.59 | 2.83 | 3.11 |
| TSS_4 | 3.93 | 1.25 | 1.25 | 1.27 | 1.42 | 2.12 |
| STT_5 | 3.88 | 1.88 | 2.31 | 2.60 | 2.83 | 3.10 |
| STT_6 | 3.89 | 1.82 | 2.28 | 2.60 | 2.85 | 3.10 |
| STT_7 | 3.90 | 1.77 | 2.20 | 2.56 | 2.83 | 3.06 |

## Pitch diversity (CVf0, clean prompts)

| Output | CVf0 |
|---|---:|
| student4 | 0.0515 |
| plain_4 | 0.0942 |
| plain_8 | 0.0948 |
| STT_4 | 0.0898 |
| TST_4 | 0.0535 |
| TTS_4 | 0.0938 |
| SST_4 | 0.0521 |
| STS_4 | 0.0902 |
| TSS_4 | 0.0502 |

## Out-of-fold policies (5-fold speaker CV, α = 0.05, δ = 0.1)

| Policy | Failures | Mean passes | Routed | Chosen per fold |
|---|---:|---:|---|---|
| reference | 0/360 | 21.00 | {'plain_8': 360} | [None, None, None, None, None] |
| plain4_all | 128/360 | 13.00 | {'plain_4': 360} | [None, None, None, None, None] |
| STT4_all | 76/360 | 14.00 | {'STT_4': 360} | [None, None, None, None, None] |
| snr_theta | 7/360 | 18.71 | {'plain_4': 103, 'plain_8': 257} | [26, 26, 26, 26, 26] |
| ladder_plain | 8/360 | 19.27 | {'plain_7': 23, 'plain_8': 246, 'plain_4': 50, 'plain_6': 34, 'plain_5': 7} | [(29, 1), (31, 2), (29, 1), (29, 1), (31, 2)] |
| ladder | 9/360 | 19.68 | {'plain_8': 271, 'plain_4': 44, 'STT_7': 24, 'STT_4': 7, 'STT_6': 9, 'STT_5': 5} | [(31, 1), (31, 1), (32, 1), (31, 1), (30, 1)] |

ladder = plain Step 4, then STT at Steps 4 to 7, then Step 8, by prompt SNR; ladder_plain has no selection.

Development evidence only, white-noise prompts only. A claim needs a frozen test on fresh speakers and real degradations.
