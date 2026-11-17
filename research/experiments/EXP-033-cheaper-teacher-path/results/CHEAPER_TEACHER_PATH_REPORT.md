# EXP-033: cheaper teacher paths to the paper's t_switch = 0.25

Rows: quality 60 / 60, diversity 20 / 20, timing 10 / 10 (partial=False). Reference: `P25`, the paper's Algorithm 2 at 0.25 (30 passes).

## Labels (frozen rules)

```
{
  "arm_labels": {
    "W8": "variety_short",
    "W8R2": "matches_paper",
    "W6R2": "variety_short",
    "W4R2": "both_short",
    "CPH12": "variety_short",
    "CPH8": "matches_paper",
    "CPH6": "quality_short",
    "GI": "matches_paper",
    "SST": "both_short",
    "CWH4": "variety_short"
  },
  "cheapest_match_point": "CPH8",
  "cheapest_match_confirmed": "none",
  "w8r2_matches_paper": true,
  "noisy_quality_safe_arms": []
}
```

## Variety, `clean` (n=20 speakers x 12 seeds)

| Arm | Passes | Saved vs P25 | CVf0 | Aligned CVf0 | Timing spread | Melody vs P25 | Share of P25's gain: CVf0 | Aligned CVf0 | 90% confirmed (both) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| `STU` | 4 | 87% | 0.0521 | 0.0410 | 3.94 | 0.462 | 0% | 0% |  |
| `W8` | 21 | 30% | 0.0912 | 0.0576 | 10.92 | 0.650 | 60% [41%, 80%] | 35% [9%, 53%] | no |
| `W8R2` | 20 | 33% | 0.1159 | 0.0879 | 11.21 | 0.835 | 98% [86%, 111%] | 100% [86%, 119%] | no |
| `W6R2` | 16 | 47% | 0.1006 | 0.0693 | 11.26 | 0.833 | 74% [56%, 88%] | 60% [41%, 87%] | no |
| `W4R2` | 12 | 60% | 0.1032 | 0.0694 | 11.27 | 0.734 | 78% [58%, 115%] | 60% [37%, 131%] | no |
| `P25` | 30 | 0% | 0.1173 | 0.0881 | 12.01 | 1.000 | 100% | 100% |  |
| `CPH12` | 16 | 47% | 0.1124 | 0.0805 | 10.98 | 0.820 | 92% [83%, 103%] | 84% [65%, 112%] | no |
| `CPH8` | 12 | 60% | 0.1136 | 0.0867 | 10.57 | 0.765 | 94% [81%, 115%] | 97% [84%, 124%] | no |
| `CPH6` | 10 | 67% | 0.1231 | 0.0991 | 9.52 | 0.622 | 109% [77%, 169%] | 124% [79%, 278%] | no |
| `GI` | 15 | 50% | 0.1188 | 0.0949 | 11.40 | 0.761 | 102% [89%, 119%] | 115% [93%, 183%] | no |
| `SST` | 13 | 57% | 0.0671 | 0.0576 | 4.05 | 0.541 | 23% [12%, 40%] | 35% [19%, 81%] | no |
| `CWH4` | 13 | 57% | 0.0983 | 0.0658 | 11.19 | 0.722 | 71% [57%, 82%] | 53% [32%, 68%] | no |

Handed-over estimate's distance from the 16-interval Step 8 estimate: `CPH12` 0.157, `CPH8` 0.270, `CPH6` 0.499, `GI` 0.307, `SST` 0.691. CWH4's mean kappa: 0.681.

## Quality, `clean` (n=30; prompt SNR median 36.9 dB)

| Arm | Passes | UTMOS | WER mean / median | Runaway | SIM | dUTMOS vs P25 | dWER hi / dSIM lo | Bars |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| `STU` | 4 | 3.940 | 0.0189 / 0.0000 | 0 | 0.9461 | +0.231 [+0.092, +0.411] | +0.0000 / -0.0005 | pass |
| `W8` | 21 | 3.885 | 0.0208 / 0.0000 | 0 | 0.9449 | +0.176 [+0.046, +0.348] | +0.0056 / +0.0005 | pass |
| `W8R2` | 20 | 3.712 | 0.0208 / 0.0000 | 0 | 0.9421 | +0.003 [-0.126, +0.112] | +0.0056 / +0.0012 | pass |
| `W6R2` | 16 | 3.745 | 0.0189 / 0.0000 | 0 | 0.9383 | +0.036 [-0.049, +0.122] | +0.0000 / -0.0002 | pass |
| `W4R2` | 12 | 3.450 | 0.0213 / 0.0000 | 0 | 0.9349 | -0.259 [-0.379, -0.137] | +0.0000 / -0.0061 | fail |
| `P25` | 30 | 3.709 | 0.0213 / 0.0000 | 0 | 0.9348 | (reference) | |  |
| `CPH12` | 16 | 3.719 | 0.0208 / 0.0000 | 0 | 0.9409 | +0.010 [-0.089, +0.109] | +0.0056 / -0.0000 | pass |
| `CPH8` | 12 | 3.687 | 0.0231 / 0.0000 | 0 | 0.9403 | -0.022 [-0.108, +0.055] | +0.0090 / +0.0012 | pass |
| `CPH6` | 10 | 3.418 | 0.0416 / 0.0000 | 0 | 0.9358 | -0.291 [-0.579, -0.046] | +0.0644 / -0.0055 | fail |
| `GI` | 15 | 3.652 | 0.0208 / 0.0000 | 0 | 0.9384 | -0.057 [-0.158, +0.029] | +0.0056 / -0.0021 | pass |
| `SST` | 13 | 3.870 | 0.0238 / 0.0000 | 0 | 0.9469 | +0.161 [+0.009, +0.345] | +0.0109 / +0.0020 | fail |
| `CWH4` | 13 | 3.679 | 0.0213 / 0.0000 | 0 | 0.9395 | -0.030 [-0.138, +0.079] | +0.0000 / -0.0034 | pass |

## Quality, `noisy15` (n=30; prompt SNR median 17.9 dB)

| Arm | Passes | UTMOS | WER mean / median | Runaway | SIM | dUTMOS vs STU | dWER hi / dSIM lo | Bars |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| `STU` | 4 | 1.970 | 0.1210 / 0.0812 | 0 | 0.8220 | (reference) | |  |
| `W8` | 21 | 1.250 | 0.1751 / 0.0955 | 1 | 0.7264 | -0.720 [-0.977, -0.464] | +0.1732 / -0.1235 | fail |
| `W8R2` | 20 | 1.261 | 1.4879 / 0.4808 | 5 | 0.6900 | -0.709 [-0.968, -0.451] | +3.4643 / -0.1748 | fail |
| `W6R2` | 16 | 1.255 | 0.3611 / 0.2183 | 3 | 0.7039 | -0.715 [-0.973, -0.458] | +0.3933 / -0.1541 | fail |
| `W4R2` | 12 | 1.267 | 6.1321 / 1.0000 | 10 | 0.6913 | -0.703 [-0.961, -0.447] | +10.2437 / -0.1653 | fail |
| `P25` | 30 | 1.252 | 0.5614 / 0.1667 | 1 | 0.7049 | -0.718 [-0.975, -0.461] | +1.1290 / -0.1494 | fail |
| `CPH12` | 16 | 1.263 | 8.0229 / 0.5962 | 10 | 0.6810 | -0.707 [-0.966, -0.449] | +13.2568 / -0.1814 | fail |
| `CPH8` | 12 | 1.275 | 4.3771 / 0.8056 | 7 | 0.6758 | -0.695 [-0.957, -0.434] | +9.1573 / -0.1836 | fail |
| `CPH6` | 10 | 1.280 | 8.0555 / 1.0000 | 8 | 0.6746 | -0.690 [-0.948, -0.433] | +13.9175 / -0.1791 | fail |
| `GI` | 15 | 1.303 | 4.7304 / 1.0000 | 11 | 0.6532 | -0.667 [-0.933, -0.407] | +8.4565 / -0.2142 | fail |
| `SST` | 13 | 2.232 | 0.0639 / 0.0000 | 0 | 0.8178 | +0.261 [-0.056, +0.590] | -0.0087 / -0.0335 | fail |
| `CWH4` | 13 | 1.264 | 5.3515 / 0.7639 | 8 | 0.7003 | -0.706 [-0.962, -0.450] | +9.7933 / -0.1523 | fail |

## Verdict per arm (clean prompts, against P25)

| Arm | Passes | Saved vs P25 | Variety >= 90% (point) | Confirmed | Quality bars | Label |
| --- | ---: | ---: | --- | --- | --- | --- |
| `W8` | 21 | 30% | no | no | pass | `variety_short` |
| `W8R2` | 20 | 33% | yes | no | pass | `matches_paper` |
| `W6R2` | 16 | 47% | no | no | pass | `variety_short` |
| `W4R2` | 12 | 60% | no | no | fail | `both_short` |
| `CPH12` | 16 | 47% | no | no | pass | `variety_short` |
| `CPH8` | 12 | 60% | yes | no | pass | `matches_paper` |
| `CPH6` | 10 | 67% | yes | no | fail | `quality_short` |
| `GI` | 15 | 50% | yes | no | pass | `matches_paper` |
| `SST` | 13 | 57% | no | no | fail | `both_short` |
| `CWH4` | 13 | 57% | no | no | pass | `variety_short` |

## Cost on the T4 (n=10 clean cases, each arm on its own)

| Arm | Passes (teacher + student) | Counted = table | Median seconds | Ratio to P25 |
| --- | --- | --- | ---: | ---: |
| `STU` | 4 (0 + 4) | yes | 0.636 | 0.14 |
| `W8` | 21 (18 + 3) | yes | 3.228 | 0.70 |
| `W8R2` | 20 (18 + 2) | yes | 3.070 | 0.67 |
| `W6R2` | 16 (14 + 2) | yes | 2.458 | 0.54 |
| `W4R2` | 12 (10 + 2) | yes | 1.849 | 0.40 |
| `P25` | 30 (28 + 2) | yes | 4.589 | 1.00 |
| `CPH12` | 16 (14 + 2) | yes | 2.452 | 0.53 |
| `CPH8` | 12 (10 + 2) | yes | 1.842 | 0.40 |
| `CPH6` | 10 (8 + 2) | yes | 1.536 | 0.33 |
| `GI` | 15 (13 + 2) | yes | 2.297 | 0.50 |
| `SST` | 13 (10 + 3) | yes | 1.994 | 0.43 |
| `CWH4` | 13 (10 + 3) | yes | 1.983 | 0.43 |

## SNR routing (descriptive): below 20 dB the student alone, otherwise the arm

Prompts below the gate: `clean` 0 of 30, `noisy15` 30 of 30.

| Arm above the gate | Mean passes (all cases) | clean UTMOS | noisy15 UTMOS | noisy15 WER |
| --- | ---: | ---: | ---: | ---: |
| `W8` | 12.50 | 3.885 | 1.970 | 0.1210 |
| `W8R2` | 12.00 | 3.712 | 1.970 | 0.1210 |
| `W6R2` | 10.00 | 3.745 | 1.970 | 0.1210 |
| `W4R2` | 8.00 | 3.450 | 1.970 | 0.1210 |
| `P25` | 17.00 | 3.709 | 1.970 | 0.1210 |
| `CPH12` | 10.00 | 3.719 | 1.970 | 0.1210 |
| `CPH8` | 8.00 | 3.687 | 1.970 | 0.1210 |
| `CPH6` | 7.00 | 3.418 | 1.970 | 0.1210 |
| `GI` | 9.50 | 3.652 | 1.970 | 0.1210 |
| `SST` | 8.50 | 3.870 | 1.970 | 0.1210 |
| `CWH4` | 8.50 | 3.679 | 1.970 | 0.1210 |

## Equal-cost contrasts (descriptive)

```
{
  "CPH12_minus_W6R2": {
    "passes": 16,
    "quality_clean": {
      "wer": {
        "n": 30,
        "mean": 0.0018518518518518517,
        "lo": 0.0,
        "hi": 0.005555555555555555,
        "sd": 0.01014301032416974
      },
      "similarity": {
        "n": 30,
        "mean": 0.002548805872599284,
        "lo": -0.002393074830373128,
        "hi": 0.009057760238647461,
        "sd": 0.01621442709189321
      },
      "utmos": {
        "n": 30,
        "mean": -0.0265660564104716,
        "lo": -0.12369786103566488,
        "hi": 0.06420574188232422,
        "sd": 0.2654401431934263
      }
    },
    "variety": {
      "cvf0": {
        "n": 20,
        "mean": 0.01175833255680635,
        "lo": 0.0012690204539640838,
        "hi": 0.02922686042616036,
        "sd": 0.03531802941882567
      },
      "cvf0_aligned": {
        "n": 20,
        "mean": 0.011213711362847614,
        "lo": 0.0006078188082494742,
        "hi": 0.029370775351599255,
        "sd": 0.036680363947971945
      },
      "timing_spread": {
        "n": 20,
        "mean": -0.28518760553342554,
        "lo": -0.6933479887187541,
        "hi": 0.0823827850517449,
        "sd": 0.9093564160637061
      },
      "melody": {
        "n": 20,
        "mean": -0.013684705921322187,
        "lo": -0.02379837910895562,
        "hi": -0.0022970023949147934,
        "sd": 0.025605314123822838
      }
    }
  },
  "CPH8_minus_W4R2": {
    "passes": 12,
    "quality_clean": {
      "wer": {
        "n": 30,
        "mean": 0.0018518518518518517,
        "lo": -0.005291005291005291,
        "hi": 0.008994708994708995,
        "sd": 0.021324763593550157
      },
      "similarity": {
        "n": 30,
        "mean": 0.005391561985015869,
        "lo": 0.0002725700537363688,
        "hi": 0.010611210266749065,
        "sd": 0.014892487078057222
      },
      "utmos": {
        "n": 30,
        "mean": 0.2370298941930135,
        "lo": 0.12279559771219889,
        "hi": 0.35006635983784995,
        "sd": 0.33301747545022564
      }
    },
    "variety": {
      "cvf0": {
        "n": 20,
        "mean": 0.010388271554911423,
        "lo": -0.005627351982672888,
        "hi": 0.027121035827123476,
        "sd": 0.038436292571414704
      },
      "cvf0_aligned": {
        "n": 20,
        "mean": 0.01729116779901032,
        "lo": -0.0041870734598706905,
        "hi": 0.039824186306432995,
        "sd": 0.051848127979972805
      },
      "timing_spread": {
        "n": 20,
        "mean": -0.7043388347895904,
        "lo": -1.098089177092498,
        "hi": -0.3459530049704263,
        "sd": 0.8735688245850406
      },
      "melody": {
        "n": 20,
        "mean": 0.030494619154277737,
        "lo": 0.004428405503370508,
        "hi": 0.05667814703255808,
        "sd": 0.06098268379597802
      }
    }
  }
}
```

## Reproductions

EXP-032 quality (STU, W8, P25):

```
{
  "clean": {
    "cases": 30,
    "max_abs_prompt_snr": 0.0,
    "max_abs_wer_STU": 0.0,
    "median_abs_wer_STU": 0.0,
    "max_abs_similarity_STU": 4.76837158203125e-07,
    "median_abs_similarity_STU": 0.0,
    "max_abs_utmos_STU": 9.393692016601562e-05,
    "median_abs_utmos_STU": 0.0,
    "max_abs_wer_W8": 0.0,
    "median_abs_wer_W8": 0.0,
    "max_abs_similarity_W8": 2.384185791015625e-06,
    "median_abs_similarity_W8": 0.0,
    "max_abs_utmos_W8": 0.00042939186096191406,
    "median_abs_utmos_W8": 0.0,
    "max_abs_wer_P25": 0.0,
    "median_abs_wer_P25": 0.0,
    "max_abs_similarity_P25": 2.1457672119140625e-06,
    "median_abs_similarity_P25": 0.0,
    "max_abs_utmos_P25": 0.00014019012451171875,
    "median_abs_utmos_P25": 0.0
  },
  "noisy15": {
    "cases": 30,
    "max_abs_prompt_snr": 0.0,
    "max_abs_wer_STU": 0.0,
    "median_abs_wer_STU": 0.0,
    "max_abs_similarity_STU": 1.1622905731201172e-05,
    "median_abs_similarity_STU": 0.0,
    "max_abs_utmos_STU": 0.00011801719665527344,
    "median_abs_utmos_STU": 0.0,
    "max_abs_wer_W8": 0.0,
    "median_abs_wer_W8": 0.0,
    "max_abs_similarity_W8": 5.602836608886719e-06,
    "median_abs_similarity_W8": 0.0,
    "max_abs_utmos_W8": 8.344650268554688e-07,
    "median_abs_utmos_W8": 0.0,
    "max_abs_wer_P25": 0.0,
    "median_abs_wer_P25": 0.0,
    "max_abs_similarity_P25": 1.7285346984863281e-06,
    "median_abs_similarity_P25": 0.0,
    "max_abs_utmos_P25": 5.960464477539062e-07,
    "median_abs_utmos_P25": 0.0
  }
}
```

EXP-032 clean variety:

```
{
  "speakers": 20,
  "max_abs_cvf0_STU": 0.0,
  "max_abs_cvf0_aligned_STU": 0.0,
  "max_abs_timing_spread_STU": 0.0,
  "max_abs_cvf0_W8": 0.0,
  "max_abs_cvf0_aligned_W8": 0.0,
  "max_abs_timing_spread_W8": 0.0,
  "max_abs_cvf0_P25": 0.0,
  "max_abs_cvf0_aligned_P25": 0.0,
  "max_abs_timing_spread_P25": 0.0
}
```

EXP-026 quality (W8R2, W4R2; recorded, not a gate):

```
{
  "clean": {
    "cases": 30,
    "max_abs_prompt_snr": 0.0,
    "max_abs_wer_W8R2": 0.0,
    "median_abs_wer_W8R2": 0.0,
    "max_abs_similarity_W8R2": 1.1563301086425781e-05,
    "median_abs_similarity_W8R2": 0.0,
    "max_abs_utmos_W8R2": 0.00046896934509277344,
    "median_abs_utmos_W8R2": 0.0,
    "max_abs_wer_W4R2": 0.0,
    "median_abs_wer_W4R2": 0.0,
    "max_abs_similarity_W4R2": 4.76837158203125e-07,
    "median_abs_similarity_W4R2": 0.0,
    "max_abs_utmos_W4R2": 4.76837158203125e-07,
    "median_abs_utmos_W4R2": 0.0
  }
}
```
