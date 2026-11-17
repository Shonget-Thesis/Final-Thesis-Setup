# EXP-032: variety per teacher pass against the paper's t_switch = 0.25

Rows: quality 60 / 60, diversity 40 / 40 (partial=False). Reference: `P25`, the paper's Algorithm 2 at 0.25 (30 passes).

## Labels (frozen rules)

```
{
  "cheapest_step_point": "none",
  "cheapest_step_confirmed": "none",
  "w8_matches_paper": false,
  "noisy_cheapest_step_point": "none",
  "noisy_quality_safe_arms": []
}
```

## Variety, `clean` (n=20 speakers x 12 seeds)

| Arm | Passes | CVf0 | Aligned CVf0 | Timing spread | Share of P25's gain: CVf0 | Aligned CVf0 | 90% confirmed (both) |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| `STU` | 4 | 0.0521 | 0.0410 | 3.94 | 0% | 0% |  |
| `W1` | 7 | 0.0717 | 0.0541 | 7.25 | 30% [8%, 66%] | 28% [4%, 93%] | no |
| `W2` | 9 | 0.0862 | 0.0574 | 9.55 | 52% [37%, 67%] | 35% [16%, 52%] | no |
| `W3` | 11 | 0.0908 | 0.0571 | 10.83 | 59% [42%, 80%] | 34% [13%, 49%] | no |
| `W4` | 13 | 0.0918 | 0.0584 | 11.01 | 61% [42%, 81%] | 37% [16%, 54%] | no |
| `W8` | 21 | 0.0912 | 0.0576 | 10.92 | 60% [41%, 80%] | 35% [9%, 53%] | no |
| `P25` | 30 | 0.1173 | 0.0881 | 12.01 | 100% | 100% |  |
| `P25R` | 32 | 0.1150 | 0.0889 | 11.91 | 97% [89%, 106%] | 102% [92%, 116%] | no |

## Variety, `noisy15` (n=20 speakers x 12 seeds)

| Arm | Passes | CVf0 | Aligned CVf0 | Timing spread | Share of P25's gain: CVf0 | Aligned CVf0 | 90% confirmed (both) |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| `STU` | 4 | nan | nan | 5.51 | 0% | 0% |  |
| `W1` | 7 | nan | nan | 10.98 | n/a | n/a | no |
| `W2` | 9 | nan | nan | 11.42 | n/a | n/a | no |
| `W3` | 11 | nan | nan | 12.94 | n/a | n/a | no |
| `W4` | 13 | nan | nan | 12.97 | n/a | n/a | no |
| `W8` | 21 | nan | nan | 13.42 | n/a | n/a | no |
| `P25` | 30 | nan | nan | 14.57 | 100% | 100% |  |
| `P25R` | 32 | nan | nan | 14.82 | n/a | n/a | no |

## Quality, `clean` (n=30; prompt SNR median 36.9 dB)

| Arm | Passes | UTMOS | WER mean / median | Runaway | SIM | dUTMOS vs P25 | dWER hi / dSIM lo | Bars |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| `STU` | 4 | 3.940 | 0.0189 / 0.0000 | 0 | 0.9461 | +0.231 [+0.092, +0.411] | +0.0000 / -0.0005 | pass |
| `W1` | 7 | 2.962 | 0.0289 / 0.0000 | 0 | 0.9300 | -0.747 [-1.117, -0.395] | +0.0217 / -0.0179 | fail |
| `W2` | 9 | 3.831 | 0.0263 / 0.0000 | 0 | 0.9442 | +0.122 [-0.014, +0.287] | +0.0185 / -0.0034 | fail |
| `W3` | 11 | 3.822 | 0.0249 / 0.0000 | 0 | 0.9459 | +0.112 [-0.003, +0.242] | +0.0144 / +0.0006 | fail |
| `W4` | 13 | 3.867 | 0.0208 / 0.0000 | 0 | 0.9459 | +0.158 [+0.018, +0.327] | +0.0056 / +0.0002 | pass |
| `W8` | 21 | 3.885 | 0.0208 / 0.0000 | 0 | 0.9449 | +0.176 [+0.046, +0.348] | +0.0056 / +0.0005 | pass |
| `P25` | 30 | 3.709 | 0.0213 / 0.0000 | 0 | 0.9348 | (reference) | |  |
| `P25R` | 32 | 3.726 | 0.0231 / 0.0000 | 0 | 0.9408 | +0.017 [-0.070, +0.097] | +0.0056 / +0.0014 | pass |
| `TCH` | 64 | 3.155 | 0.0171 / 0.0000 | 0 | 0.9324 | -0.554 [-0.688, -0.411] | +0.0000 / -0.0108 | fail |

## Quality, `noisy15` (n=30; prompt SNR median 17.9 dB)

| Arm | Passes | UTMOS | WER mean / median | Runaway | SIM | dUTMOS vs STU | dWER hi / dSIM lo | Bars |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| `STU` | 4 | 1.970 | 0.1210 / 0.0812 | 0 | 0.8220 | (reference) | |  |
| `W1` | 7 | 1.317 | 3.2700 / 1.0000 | 8 | 0.6468 | -0.653 [-0.909, -0.405] | +6.7440 / -0.2174 | fail |
| `W2` | 9 | 1.270 | 1.3896 / 0.4722 | 6 | 0.6885 | -0.700 [-0.959, -0.446] | +2.9517 / -0.1683 | fail |
| `W3` | 11 | 1.268 | 1.5073 / 0.3375 | 2 | 0.7076 | -0.702 [-0.963, -0.445] | +3.6213 / -0.1522 | fail |
| `W4` | 13 | 1.259 | 1.6978 / 0.2321 | 1 | 0.7116 | -0.712 [-0.971, -0.453] | +4.4257 / -0.1443 | fail |
| `W8` | 21 | 1.250 | 0.1751 / 0.0955 | 1 | 0.7264 | -0.720 [-0.977, -0.464] | +0.1732 / -0.1235 | fail |
| `P25` | 30 | 1.252 | 0.5614 / 0.1667 | 1 | 0.7049 | -0.718 [-0.975, -0.461] | +1.1290 / -0.1494 | fail |
| `P25R` | 32 | 1.255 | 0.3566 / 0.0955 | 3 | 0.7248 | -0.715 [-0.973, -0.456] | +0.5240 / -0.1360 | fail |
| `TCH` | 64 | 1.264 | 3.8607 / 0.5694 | 4 | 0.6883 | -0.707 [-0.963, -0.450] | +7.9247 / -0.1686 | fail |

## Variety change from clean to noisy15 (same speakers)

| Arm | CVf0 | Aligned CVf0 | Timing spread |
| --- | --- | --- | --- |
| `STU` | +0.0374 [-0.0095, +0.1100] | +0.0389 [-0.0051, +0.1115] | +1.57 [+0.78, +2.32] |
| `W1` | +nan [+nan, +nan] | +nan [+nan, +nan] | +3.73 [+1.96, +5.54] |
| `W2` | +nan [+nan, +nan] | +nan [+nan, +nan] | +1.87 [-0.45, +4.49] |
| `W3` | +nan [+nan, +nan] | +nan [+nan, +nan] | +2.11 [-0.53, +5.11] |
| `W4` | +nan [+nan, +nan] | +0.4704 [+0.4704, +0.4704] | +1.97 [-0.42, +4.75] |
| `W8` | +0.2622 [+0.2622, +0.2622] | +0.2739 [+0.2739, +0.2739] | +2.50 [+0.08, +5.24] |
| `P25` | +nan [+nan, +nan] | -0.3717 [-0.3717, -0.3717] | +2.56 [+0.24, +5.17] |
| `P25R` | -0.0310 [-0.0310, -0.0310] | -0.0032 [-0.0032, -0.0032] | +2.91 [+0.61, +5.54] |

## Handover mechanism: P25 (continue from state) - P25R (re-noise endpoint)

```
{
  "clean": {
    "quality": {
      "wer": {
        "n": 30,
        "mean": -0.0018518518518518517,
        "lo": -0.005555555555555555,
        "hi": 0.0,
        "sd": 0.01014301032416974
      },
      "similarity": {
        "n": 30,
        "mean": -0.005944812297821045,
        "lo": -0.011551642417907714,
        "hi": -0.0014301915963490805,
        "sd": 0.014167239329127179
      },
      "utmos": {
        "n": 30,
        "mean": -0.017415459950764975,
        "lo": -0.097015118598938,
        "hi": 0.06990865866343181,
        "sd": 0.2317347551795814
      }
    },
    "variety": {
      "cvf0": {
        "n": 20,
        "mean": 0.0022583061059924915,
        "lo": -0.0030782352828971973,
        "hi": 0.008127895741076788,
        "sd": 0.013233851559152075
      },
      "cvf0_aligned": {
        "n": 20,
        "mean": -0.0008101052722194969,
        "lo": -0.0055271826312852474,
        "hi": 0.004307498175888637,
        "sd": 0.011560302952260123
      },
      "timing_spread": {
        "n": 20,
        "mean": 0.10169697529000321,
        "lo": 0.0025396779326863684,
        "hi": 0.21172503591797498,
        "sd": 0.24247445230119877
      }
    }
  },
  "noisy15": {
    "quality": {
      "wer": {
        "n": 30,
        "mean": 0.20487216487216486,
        "lo": -0.23404521404521406,
        "hi": 0.8802537277537277,
        "sd": 1.6785654291751066
      },
      "similarity": {
        "n": 30,
        "mean": -0.01988442838191986,
        "lo": -0.042242502172787987,
        "hi": 0.005246872703234355,
        "sd": 0.0677207512411567
      },
      "utmos": {
        "n": 30,
        "mean": -0.0026674191157023113,
        "lo": -0.008555229504903157,
        "hi": 0.0026853521664937335,
        "sd": 0.015488232494370218
      }
    },
    "variety": {
      "cvf0": {
        "n": 0,
        "mean": NaN,
        "lo": NaN,
        "hi": NaN,
        "sd": NaN
      },
      "cvf0_aligned": {
        "n": 0,
        "mean": NaN,
        "lo": NaN,
        "hi": NaN,
        "sd": NaN
      },
      "timing_spread": {
        "n": 20,
        "mean": -0.25004366052200366,
        "lo": -0.514163475090575,
        "hi": -0.008703152488440048,
        "sd": 0.6060859758364171
      }
    }
  }
}
```

## Reproductions

EXP-031 quality (STU, W4, W8):

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
    "max_abs_wer_W4": 0.0,
    "median_abs_wer_W4": 0.0,
    "max_abs_similarity_W4": 2.3245811462402344e-06,
    "median_abs_similarity_W4": 0.0,
    "max_abs_utmos_W4": 1.811981201171875e-05,
    "median_abs_utmos_W4": 0.0,
    "max_abs_wer_W8": 0.0,
    "median_abs_wer_W8": 0.0,
    "max_abs_similarity_W8": 2.384185791015625e-06,
    "median_abs_similarity_W8": 0.0,
    "max_abs_utmos_W8": 0.00042939186096191406,
    "median_abs_utmos_W8": 0.0
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
    "max_abs_wer_W4": 0.0,
    "median_abs_wer_W4": 0.0,
    "max_abs_similarity_W4": 4.410743713378906e-06,
    "median_abs_similarity_W4": 0.0,
    "max_abs_utmos_W4": 1.0728836059570312e-06,
    "median_abs_utmos_W4": 0.0,
    "max_abs_wer_W8": 0.0,
    "median_abs_wer_W8": 0.0,
    "max_abs_similarity_W8": 5.662441253662109e-06,
    "median_abs_similarity_W8": 0.0,
    "max_abs_utmos_W8": 8.344650268554688e-07,
    "median_abs_utmos_W8": 0.0
  }
}
```

EXP-031 clean variety:

```
{
  "speakers": 20,
  "max_abs_cvf0_STU": 0.0,
  "max_abs_cvf0_aligned_STU": 0.0,
  "max_abs_timing_spread_STU": 0.0,
  "max_abs_cvf0_W4": 3.154898376628301e-06,
  "max_abs_cvf0_aligned_W4": 1.4497609831867764e-06,
  "max_abs_timing_spread_W4": 0.0,
  "max_abs_cvf0_W8": 0.0,
  "max_abs_cvf0_aligned_W8": 0.0,
  "max_abs_timing_spread_W8": 0.0
}
```

EXP-017 quality (STU = student4, W4 = g16_4, W8 = g16_8, P25 = alg2, P25R = paper, TCH = teacher32):

```
{
  "clean": {
    "cases": 30,
    "max_abs_prompt_snr": 0.0,
    "max_abs_wer_STU": 0.0,
    "median_abs_wer_STU": 0.0,
    "max_abs_similarity_STU": 1.6689300537109375e-06,
    "median_abs_similarity_STU": 0.0,
    "max_abs_utmos_STU": 4.38690185546875e-05,
    "median_abs_utmos_STU": 0.0,
    "max_abs_wer_W4": 0.0,
    "median_abs_wer_W4": 0.0,
    "max_abs_similarity_W4": 4.112720489501953e-06,
    "median_abs_similarity_W4": 5.960464477539063e-08,
    "max_abs_utmos_W4": 3.910064697265625e-05,
    "median_abs_utmos_W4": 0.0,
    "max_abs_wer_W8": 0.0,
    "median_abs_wer_W8": 0.0,
    "max_abs_similarity_W8": 1.823902130126953e-05,
    "median_abs_similarity_W8": 5.960464477539063e-08,
    "max_abs_utmos_W8": 0.002594470977783203,
    "median_abs_utmos_W8": 0.0,
    "max_abs_wer_P25": 0.0,
    "median_abs_wer_P25": 0.0,
    "max_abs_similarity_P25": 3.4570693969726562e-06,
    "median_abs_similarity_P25": 5.960464477539063e-08,
    "max_abs_utmos_P25": 0.0003361701965332031,
    "median_abs_utmos_P25": 0.0,
    "max_abs_wer_P25R": 0.0,
    "median_abs_wer_P25R": 0.0,
    "max_abs_similarity_P25R": 6.735324859619141e-06,
    "median_abs_similarity_P25R": 5.960464477539063e-08,
    "max_abs_utmos_P25R": 1.8596649169921875e-05,
    "median_abs_utmos_P25R": 0.0,
    "max_abs_wer_TCH": 0.0,
    "median_abs_wer_TCH": 0.0,
    "max_abs_similarity_TCH": 1.3113021850585938e-06,
    "median_abs_similarity_TCH": 5.960464477539063e-08,
    "max_abs_utmos_TCH": 1.7404556274414062e-05,
    "median_abs_utmos_TCH": 0.0
  },
  "noisy15": {
    "cases": 30,
    "max_abs_prompt_snr": 0.0,
    "max_abs_wer_STU": 0.0,
    "median_abs_wer_STU": 0.0,
    "max_abs_similarity_STU": 7.218122482299805e-05,
    "median_abs_similarity_STU": 5.960464477539063e-08,
    "max_abs_utmos_STU": 0.0006999969482421875,
    "median_abs_utmos_STU": 0.0,
    "max_abs_wer_W4": 0.0,
    "median_abs_wer_W4": 0.0,
    "max_abs_similarity_W4": 1.0192394256591797e-05,
    "median_abs_similarity_W4": 5.960464477539063e-08,
    "max_abs_utmos_W4": 1.430511474609375e-06,
    "median_abs_utmos_W4": 0.0,
    "max_abs_wer_W8": 0.0,
    "median_abs_wer_W8": 0.0,
    "max_abs_similarity_W8": 9.775161743164062e-06,
    "median_abs_similarity_W8": 5.960464477539063e-08,
    "max_abs_utmos_W8": 1.0728836059570312e-06,
    "median_abs_utmos_W8": 0.0,
    "max_abs_wer_P25": 0.0,
    "median_abs_wer_P25": 0.0,
    "max_abs_similarity_P25": 9.298324584960938e-06,
    "median_abs_similarity_P25": 5.960464477539063e-08,
    "max_abs_utmos_P25": 2.6226043701171875e-06,
    "median_abs_utmos_P25": 0.0,
    "max_abs_wer_P25R": 0.0,
    "median_abs_wer_P25R": 0.0,
    "max_abs_similarity_P25R": 9.655952453613281e-06,
    "median_abs_similarity_P25R": 5.960464477539063e-08,
    "max_abs_utmos_P25R": 0.00010478496551513672,
    "median_abs_utmos_P25R": 0.0,
    "max_abs_wer_TCH": 0.0,
    "median_abs_wer_TCH": 0.0,
    "max_abs_similarity_TCH": 3.039836883544922e-06,
    "median_abs_similarity_TCH": 5.960464477539063e-08,
    "max_abs_utmos_TCH": 3.5762786865234375e-07,
    "median_abs_utmos_TCH": 0.0
  }
}
```

Tensor bank: 120 files, 93.2 MiB (hashes in bank_index.json; the bank is not in the results zip).
