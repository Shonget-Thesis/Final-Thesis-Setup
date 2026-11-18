# EXP-034: the invariant residual against SNR only and Chapter 3's drift

Rows: quality 240 / 240, diversity 40 / 40, timing 10 / 10 (partial=False).

**Verdict (frozen rules): `snr_r_better`**

| Hypothesis | Holds |
| --- | --- |
| H1_snr_r_vs_snr_only | yes |
| H2_snr_r_vs_snr_drift | yes |
| H3_detection | yes |
| H4_wrong_clean_flags | yes |
| H5_clean_variety | yes |

## Policies against each other (speaker-level bootstrap, 95% intervals)

| Comparison | dUTMOS | dWER | dSIM | Better (frozen bars) |
| --- | --- | --- | --- | --- |
| SNR_R_minus_SNR_ONLY | +0.574 [+0.468, +0.677] | -0.0017 [-0.0047, +0.0013] | +0.0156 [+0.0096, +0.0223] | yes |
| SNR_R_minus_SNR_DRIFT | +0.431 [+0.340, +0.524] | -0.0022 [-0.0055, +0.0008] | +0.0091 [+0.0047, +0.0139] | yes |
| R_ONLY_minus_SNR_ONLY | +0.573 [+0.469, +0.673] | -0.0017 [-0.0047, +0.0013] | +0.0152 [+0.0090, +0.0219] | yes |
| SNR_DRIFT_minus_SNR_ONLY | +0.143 [+0.056, +0.247] | +0.0005 [+0.0000, +0.0016] | +0.0065 [+0.0024, +0.0118] | yes |
| SNR_R_minus_W8 | +0.582 [+0.484, +0.687] | -0.0072 [-0.0202, +0.0080] | +0.0242 [+0.0154, +0.0327] | yes |

## Detection of 'CPQH needed' (n = 117 needed, 123 not)

| Signal | AUC, all cases | AUC, SNR gate open |
| --- | ---: | ---: |
| r | 0.864 | 0.897 |
| drift8_better_direction | 0.697 | 0.645 |
| rough_share | 0.706 | 0.700 |
| snr_low | 0.538 | 0.751 |
| snr_sw_low | 0.866 | 0.907 |

Wrong clean flags: SNR_ONLY 0.000, SNR_DRIFT 0.000, SNR_R 0.067, R_ONLY 0.067

## Mean UTMOS by condition

| Condition | n | SNR_ONLY | SNR_DRIFT | SNR_R | R_ONLY | SW_SNR | ORACLE | STU | STUR2 | CPH8 | CPQH | W8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| clean | 30 | 3.884 | 3.884 | 3.891 | 3.891 | 3.884 | 3.969 | 3.924 | 3.958 | 3.884 | 3.879 | 3.964 |
| noisy15 | 30 | 2.199 | 2.199 | 2.199 | 2.199 | 2.199 | 2.199 | 1.598 | 2.876 | 1.277 | 2.199 | 1.247 |
| gauss25 | 30 | 1.828 | 2.260 | 3.742 | 3.742 | 3.742 | 3.742 | 3.607 | 3.845 | 1.828 | 3.742 | 1.916 |
| pink25 | 30 | 2.743 | 2.907 | 3.868 | 3.868 | 3.530 | 3.900 | 3.883 | 4.004 | 2.743 | 3.894 | 3.059 |
| hp25 | 30 | 2.098 | 2.593 | 3.611 | 3.611 | 3.611 | 3.611 | 3.529 | 3.743 | 2.098 | 3.611 | 2.304 |
| reverb09 | 30 | 1.416 | 1.416 | 1.422 | 1.410 | 1.399 | 1.436 | 1.385 | 1.428 | 1.397 | 1.426 | 1.401 |
| telephone | 30 | 1.488 | 1.539 | 1.513 | 1.513 | 1.488 | 1.611 | 1.595 | 1.567 | 1.488 | 1.606 | 1.588 |
| babble15 | 30 | 3.355 | 3.355 | 3.355 | 3.360 | 3.329 | 3.429 | 3.519 | 3.491 | 3.329 | 3.349 | 3.465 |
| all | 240 | 2.376 | 2.519 | 2.950 | 2.949 | 2.898 | 2.987 | 2.880 | 3.114 | 2.256 | 2.963 | 2.368 |

## Mean WER by condition

| Condition | SNR_ONLY | SNR_DRIFT | SNR_R | R_ONLY | SW_SNR | ORACLE | STU | STUR2 | CPH8 | CPQH | W8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| clean | 0.0312 | 0.0312 | 0.0312 | 0.0312 | 0.0312 | 0.0312 | 0.0240 | 0.0240 | 0.0312 | 0.0312 | 0.0336 |
| noisy15 | 0.1457 | 0.1457 | 0.1457 | 0.1457 | 0.1457 | 0.1557 | 0.1382 | 0.0352 | 3.9359 | 0.1457 | 0.1725 |
| gauss25 | 0.0423 | 0.0423 | 0.0386 | 0.0386 | 0.0386 | 0.0386 | 0.0390 | 0.0275 | 0.0423 | 0.0386 | 0.0376 |
| pink25 | 0.0453 | 0.0453 | 0.0363 | 0.0363 | 0.0369 | 0.0363 | 0.0247 | 0.0247 | 0.0453 | 0.0363 | 0.0424 |
| hp25 | 0.0377 | 0.0377 | 0.0401 | 0.0401 | 0.0401 | 0.0401 | 0.0240 | 0.0240 | 0.0377 | 0.0401 | 0.0363 |
| reverb09 | 0.0278 | 0.0278 | 0.0248 | 0.0248 | 0.0309 | 0.0309 | 0.0379 | 0.0481 | 0.0309 | 0.0248 | 0.0410 |
| telephone | 0.0289 | 0.0330 | 0.0289 | 0.0289 | 0.0289 | 0.0330 | 0.0361 | 0.0428 | 0.0289 | 0.0330 | 0.0337 |
| babble15 | 0.0278 | 0.0278 | 0.0278 | 0.0278 | 0.0326 | 0.0278 | 0.0271 | 0.0216 | 0.0326 | 0.0309 | 0.0339 |
| all | 0.0483 | 0.0489 | 0.0467 | 0.0467 | 0.0481 | 0.0492 | 0.0439 | 0.0310 | 0.5231 | 0.0476 | 0.0539 |

## Share of prompts sent to CPQH

| Condition | SNR_ONLY | SNR_DRIFT | SNR_R | R_ONLY |
| --- | ---: | ---: | ---: | ---: |
| clean | 0.00 | 0.00 | 0.07 | 0.07 |
| noisy15 | 1.00 | 1.00 | 1.00 | 1.00 |
| gauss25 | 0.00 | 0.17 | 1.00 | 1.00 |
| pink25 | 0.00 | 0.07 | 0.93 | 0.93 |
| hp25 | 0.00 | 0.20 | 1.00 | 1.00 |
| reverb09 | 0.67 | 0.67 | 0.83 | 0.47 |
| telephone | 0.00 | 0.47 | 0.20 | 0.20 |
| babble15 | 0.73 | 0.73 | 0.73 | 0.03 |
| all | 0.30 | 0.41 | 0.72 | 0.59 |

## Variety, `clean` (n = 20 speakers x 12 seeds)

| Arm or policy | CVf0 | Aligned CVf0 | Voiced | Melody vs P25 |
| --- | ---: | ---: | ---: | ---: |
| STU | 0.0514 | 0.0419 | 0.57 | 0.449 |
| STUR2 | 0.0482 | 0.0390 | 0.57 | 0.445 |
| CPH8 | 0.0857 | 0.0611 | 0.60 | 0.776 |
| CPQH | 0.0852 | 0.0586 | 0.62 | 0.764 |
| P25 | 0.0887 | 0.0574 | 0.62 | 1.000 |
| SNR_ONLY | 0.0857 | 0.0611 | 0.60 | 0.776 |
| SNR_DRIFT | 0.0858 | 0.0611 | 0.60 | 0.776 |
| SNR_R | 0.0876 | 0.0604 | 0.60 | 0.772 |
| R_ONLY | 0.0876 | 0.0604 | 0.60 | 0.772 |

Share of CPH8's gain over STU (cvf0): SNR_ONLY 1.00, SNR_DRIFT 1.00, SNR_R 1.05, R_ONLY 1.05, CPQH 0.99, STUR2 -0.09

Share of CPH8's gain over STU (cvf0_aligned): SNR_ONLY 1.00, SNR_DRIFT 1.00, SNR_R 0.96, R_ONLY 0.96, CPQH 0.87, STUR2 -0.15

## Variety, `gauss25` (n = 20 speakers x 12 seeds)

| Arm or policy | CVf0 | Aligned CVf0 | Voiced | Melody vs P25 |
| --- | ---: | ---: | ---: | ---: |
| STU | 0.0568 | 0.0447 | 0.53 | 0.394 |
| STUR2 | 0.0544 | 0.0441 | 0.58 | 0.353 |
| CPH8 | 0.1152 | 0.0858 | 0.11 | 0.759 |
| CPQH | 0.0882 | 0.0673 | 0.63 | 0.511 |
| P25 | 0.1056 | 0.0770 | 0.12 | 1.000 |
| SNR_ONLY | 0.1152 | 0.0858 | 0.11 | 0.759 |
| SNR_DRIFT | 0.1000 | 0.0759 | 0.20 | 0.632 |
| SNR_R | 0.0882 | 0.0673 | 0.63 | 0.511 |
| R_ONLY | 0.0882 | 0.0673 | 0.63 | 0.511 |

## Cost (n = 10 clean cases, each on its own)

| Arm or policy | Passes | Counted = table | Median seconds | Ratio to P25 |
| --- | ---: | --- | ---: | ---: |
| STU | 4 | yes | 0.659 | 0.14 |
| STUR2 | 3 | yes | 0.511 | 0.11 |
| CPH8 | 12 | yes | 1.952 | 0.40 |
| CPQH | 13 | yes | 2.115 | 0.44 |
| W8 | 21 | yes | 3.409 | 0.71 |
| P25 | 30 | yes | 4.829 | 1.00 |
| SNR_ONLY | 12 | yes | 1.939 | 0.40 |
| SNR_R | 13 | yes | 2.114 | 0.44 |

## Secondary: thresholds re-chosen on positions 31-45, applied to 46-60

```
{
  "n_cal": 120,
  "n_test": 120,
  "lambda_r": {
    "lambda": 0.27086185554102343,
    "mean": 2.9686011215051016,
    "wrong_clean": 0.13333333333333333,
    "constraint_met": true
  },
  "drift_high": {
    "lambda": Infinity,
    "mean": 2.4627663950125376,
    "wrong_clean": 0.0,
    "constraint_met": true
  },
  "drift_low": {
    "lambda": 0.27711567282676697,
    "mean": 2.75558224717776,
    "wrong_clean": 0.13333333333333333,
    "constraint_met": true
  },
  "test_utmos": {
    "SNR_ONLY": 2.2897056390841803,
    "SNR_R": 2.94689452846845,
    "SNR_DRIFT": 2.7232219497362773,
    "ORACLE": 2.9711923629045485
  }
}
```
