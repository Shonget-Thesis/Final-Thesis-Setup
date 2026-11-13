# Preliminary test: Slow-Part Handover (EXP-022)

Development speakers only; 360 of 360 cases, 40 of 40 diversity pairs; preregistration `c311aeb52d424344…`. Every number is computed from the CSV files by `run_slow_part_handover_prelim.py analyze`.

The teacher stops at the first step k >= 2 where its slow part has settled (d_slow_k <= lambda) and hands over only that part; the student's one-pass draft supplies the voice part and the fine detail. Otherwise Step 8 as today. Candidate steps after the pitch-variety check: [4, 5, 6, 7].

## Preregistered criteria

| C1_cheaper_than_voice_swap | C2_quality | C3_pitch_variety_at_step_3 | C4_slow_part_beats_whole_estimate | Advance |
|---|---|---|---|---|
| FAIL | PASS | FAIL | PASS | **NO** |

## Stop rules, out of fold (5-fold speaker CV; lambda chosen on training folds)

| Signal | Mean passes | Clean | Noisy | Stops | WER vs Step 8 | Similarity vs Step 8 | WER vs Voice-Swap | Failures |
|---|---:|---:|---:|---|---|---|---|---:|
| slow | 14.19 | 15.13 | 14.00 | {'STS_4': 333, 'STS_5': 20, 'STS_6': 7} | +0.0031 [-0.0129, +0.0177] | +0.0523 [+0.0404, +0.0651] | +0.0035 [-0.0035, +0.0113] | 78/360 |
| full | 14.19 | 15.17 | 14.00 | {'STS_4': 330, 'STS_5': 25, 'STS_6': 5} | +0.0031 [-0.0129, +0.0177] | +0.0523 [+0.0403, +0.0651] | +0.0035 [-0.0035, +0.0113] | 78/360 |
| voice | 18.56 | 19.75 | 18.32 | {'STS_4': 63, 'STS_5': 21, 'STS_6': 94, 'STS_7': 50, 'plain_8': 132} | +0.0072 [-0.0029, +0.0174] | +0.0401 [+0.0300, +0.0507] | +0.0077 [-0.0033, +0.0202] | 52/360 |

slow is the proposed rule; full uses the change of the whole estimate (the thesis's drift) and voice the change of the voice part, both calibrated the same way.

## Fixed handovers (descriptive)

| Output | Passes | WER vs Step 8 | Similarity vs Step 8 | UTMOS vs Step 8 | Failures |
|---|---:|---|---|---|---:|
| plain_8 | 21 | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.000 [+0.000, +0.000] | 0/360 |
| STT_4 | 14 | -0.0004 [-0.0160, +0.0134] | +0.0517 [+0.0401, +0.0636] | +0.895 [+0.771, +1.021] | 76/360 |
| STS_2 | 10 | +0.0164 [-0.0006, +0.0329] | +0.0433 [+0.0318, +0.0549] | +0.717 [+0.594, +0.847] | 94/360 |
| STS_3 | 12 | +0.0106 [-0.0079, +0.0276] | +0.0497 [+0.0379, +0.0623] | +0.901 [+0.768, +1.037] | 88/360 |
| STS_4 | 14 | +0.0031 [-0.0136, +0.0184] | +0.0524 [+0.0405, +0.0644] | +0.898 [+0.772, +1.029] | 77/360 |
| STS_5 | 16 | +0.0029 [-0.0127, +0.0174] | +0.0529 [+0.0418, +0.0644] | +0.881 [+0.753, +1.011] | 70/360 |
| STS_6 | 18 | +0.0018 [-0.0129, +0.0158] | +0.0555 [+0.0453, +0.0661] | +0.873 [+0.743, +1.007] | 61/360 |
| STS_7 | 20 | +0.0036 [-0.0117, +0.0180] | +0.0532 [+0.0425, +0.0644] | +0.852 [+0.731, +0.974] | 71/360 |

## Pitch variety (CVf0)

| Condition | STS_2 | STS_3 | STS_4 | plain_8 |
|---|---:|---:|---:|---:|
| clean | 0.0792 | 0.0923 | 0.0902 | 0.0948 |
| noisy15 | 0.1438 | 0.2540 | 0.1244 | n/a |

Step 2 minus Step 8: clean -0.0157 [-0.0278, -0.0071]; noisy15 n/a (keeps pitch variety: FAIL)

Step 3 minus Step 8: clean -0.0025 [-0.0152, +0.0072]; noisy15 n/a (keeps pitch variety: FAIL)

## Mean part change per teacher step (descriptive)

**voice**

| Condition | k=1 | k=2 | k=3 | k=4 | k=5 | k=6 | k=7 | k=8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| clean | 0.434 | 1.282 | 0.904 | 0.532 | 0.285 | 0.176 | 0.144 | 0.101 |
| noisy15 | 0.135 | 0.557 | 0.165 | 0.131 | 0.113 | 0.091 | 0.084 | 0.087 |
| gauss18 | 0.111 | 0.580 | 0.168 | 0.124 | 0.106 | 0.078 | 0.074 | 0.073 |
| gauss20 | 0.106 | 0.625 | 0.190 | 0.138 | 0.108 | 0.074 | 0.068 | 0.064 |
| gauss22 | 0.110 | 0.700 | 0.232 | 0.170 | 0.120 | 0.078 | 0.071 | 0.061 |
| gauss25 | 0.143 | 0.891 | 0.338 | 0.261 | 0.160 | 0.104 | 0.089 | 0.065 |

**slow**

| Condition | k=1 | k=2 | k=3 | k=4 | k=5 | k=6 | k=7 | k=8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| clean | 0.164 | 0.643 | 0.659 | 0.625 | 0.497 | 0.320 | 0.213 | 0.149 |
| noisy15 | 0.435 | 0.718 | 0.414 | 0.301 | 0.293 | 0.273 | 0.274 | 0.297 |
| gauss18 | 0.370 | 0.763 | 0.462 | 0.327 | 0.285 | 0.263 | 0.251 | 0.264 |
| gauss20 | 0.345 | 0.770 | 0.500 | 0.346 | 0.292 | 0.261 | 0.229 | 0.224 |
| gauss22 | 0.306 | 0.747 | 0.539 | 0.357 | 0.303 | 0.261 | 0.216 | 0.186 |
| gauss25 | 0.256 | 0.698 | 0.571 | 0.383 | 0.324 | 0.266 | 0.201 | 0.144 |

**fine**

| Condition | k=1 | k=2 | k=3 | k=4 | k=5 | k=6 | k=7 | k=8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| clean | 0.446 | 0.750 | 0.735 | 0.616 | 0.525 | 0.423 | 0.321 | 0.278 |
| noisy15 | 0.423 | 0.541 | 0.396 | 0.428 | 0.451 | 0.432 | 0.412 | 0.475 |
| gauss18 | 0.407 | 0.548 | 0.466 | 0.510 | 0.483 | 0.433 | 0.393 | 0.454 |
| gauss20 | 0.403 | 0.556 | 0.516 | 0.556 | 0.506 | 0.430 | 0.373 | 0.402 |
| gauss22 | 0.404 | 0.575 | 0.563 | 0.591 | 0.506 | 0.431 | 0.374 | 0.349 |
| gauss25 | 0.406 | 0.597 | 0.619 | 0.610 | 0.513 | 0.429 | 0.372 | 0.301 |

**full**

| Condition | k=1 | k=2 | k=3 | k=4 | k=5 | k=6 | k=7 | k=8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| clean | 0.300 | 0.930 | 0.713 | 0.595 | 0.447 | 0.293 | 0.203 | 0.146 |
| noisy15 | 0.189 | 0.585 | 0.215 | 0.175 | 0.172 | 0.157 | 0.155 | 0.171 |
| gauss18 | 0.173 | 0.617 | 0.244 | 0.196 | 0.180 | 0.159 | 0.149 | 0.155 |
| gauss20 | 0.172 | 0.660 | 0.287 | 0.226 | 0.198 | 0.168 | 0.145 | 0.141 |
| gauss22 | 0.172 | 0.713 | 0.351 | 0.265 | 0.224 | 0.184 | 0.151 | 0.129 |
| gauss25 | 0.186 | 0.809 | 0.454 | 0.336 | 0.274 | 0.218 | 0.166 | 0.120 |

Development evidence only (speakers that designed EXP-020), white-noise prompts. A claim needs a frozen test on new speakers.
