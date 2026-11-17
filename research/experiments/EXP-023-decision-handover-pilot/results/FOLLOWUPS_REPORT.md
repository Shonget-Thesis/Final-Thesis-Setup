# EXP-023 pre-specified follow-ups (exploratory)

Rules fixed in `code/analysis/exp023_followups.py` before the pilot ran. Out of fold over the pilot's 5 speaker folds; development speakers only. A promising result leads to a frozen test on new speakers, never to a claim.

| Rule | Mean passes | Misroutes | Failures clean / degraded | Clean WER | Clean UTMOS | Degraded WER | Clean CVf0 vs reference | Promising |
|---|---:|---:|---|---|---|---|---|---|
| SNR rule | 7.02 | 0 | 0 / 5 | +0.0000 [+0.0000, +0.0000] | -0.033 [-0.115, +0.035] | -0.0465 [-0.1143, +0.0110] | -0.0100 [-0.0302, +0.0012] | — |
| ARH (vg_g) | 11.47 | 23 | 0 / 13 | +0.0000 [+0.0000, +0.0000] | -0.026 [-0.110, +0.042] | +0.0129 [-0.0791, +0.1113] | -0.0210 [-0.0631, +0.0012] | — |
| GAR-u (vg_u) | 11.30 | 24 | 0 / 15 | +0.0000 [+0.0000, +0.0000] | -0.026 [-0.110, +0.042] | -0.0080 [-0.0591, +0.0419] | -0.0210 [-0.0631, +0.0012] | no |
| GAR-ratio (vg_g / vg_u) | 11.47 | 27 | 0 / 15 | -0.0028 [-0.0083, +0.0000] | -0.062 [-0.147, +0.016] | +0.0621 [-0.0265, +0.1613] | -0.0057 [-0.0180, +0.0013] | no |
| CGH (ARH + dr_g_1 gate) | 11.47 | 23 | 0 / 13 | +0.0000 [+0.0000, +0.0000] | -0.026 [-0.110, +0.042] | +0.0129 [-0.0791, +0.1113] | -0.0210 [-0.0631, +0.0012] | no |
| DCH replay | 12.00 | 0 | 4 / 21 | +0.0279 [-0.0011, +0.0625] | -0.310 [-0.498, -0.150] | +0.1586 [+0.0737, +0.2506] | -0.0110 [-0.0443, +0.0103] | no |
| PAS (ARH + prosody-agreement safeguard) | 11.47 | 23 | 0 / 13 | +0.0000 [+0.0000, +0.0000] | -0.026 [-0.110, +0.042] | +0.0129 [-0.0791, +0.1113] | -0.0210 [-0.0631, +0.0012] | no |
| DH_3 alone | 12.00 | 0 | 4 / 21 | +0.0279 [-0.0007, +0.0639] | -0.310 [-0.499, -0.147] | +0.1586 [+0.0732, +0.2585] | -0.0110 [-0.0443, +0.0103] | — |
| DHu_3 alone | 8.00 | 0 | 1 / 9 | +0.0072 [-0.0083, +0.0300] | -0.044 [-0.182, +0.100] | -0.0234 [-0.0863, +0.0349] | -0.0270 [-0.0687, +0.0029] | — |
| GCH (guided or unguided decision) | 12.00 | 0 | 4 / 21 | +0.0279 [-0.0011, +0.0625] | -0.310 [-0.498, -0.150] | +0.1586 [+0.0737, +0.2506] | -0.0110 [-0.0443, +0.0103] | no |

Routes per condition:

- SNR rule: {'clean': {'STT_4': 1, 'W4': 19, 'student4': 0}, 'gauss20': {'STT_4': 0, 'W4': 0, 'student4': 20}, 'noisy15': {'STT_4': 0, 'W4': 0, 'student4': 20}}
- ARH (vg_g): {'clean': {'W4': 19, 'student4': 1}, 'gauss20': {'W4': 15, 'student4': 5}, 'noisy15': {'W4': 7, 'student4': 13}}
- GAR-u (vg_u): {'clean': {'W4': 19, 'student4': 1}, 'gauss20': {'W4': 14, 'student4': 6}, 'noisy15': {'W4': 9, 'student4': 11}}
- GAR-ratio (vg_g / vg_u): {'clean': {'W4': 17, 'student4': 3}, 'gauss20': {'W4': 11, 'student4': 9}, 'noisy15': {'W4': 13, 'student4': 7}}
- CGH (ARH + dr_g_1 gate): {'clean': {'W4': 19, 'student4': 1}, 'gauss20': {'W4': 15, 'student4': 5}, 'noisy15': {'W4': 7, 'student4': 13}}
- DCH replay: {'clean': {'DH_3': 20}, 'gauss20': {'DH_3': 20}, 'noisy15': {'DH_3': 20}}
- PAS (ARH + prosody-agreement safeguard): {'clean': {'W4': 19, 'student4': 1}, 'gauss20': {'W4': 15, 'student4': 5}, 'noisy15': {'W4': 7, 'student4': 13}}
- DH_3 alone: {'clean': {'DH_3': 20}, 'gauss20': {'DH_3': 20}, 'noisy15': {'DH_3': 20}}
- DHu_3 alone: {'clean': {'DHu_3': 20}, 'gauss20': {'DHu_3': 20}, 'noisy15': {'DHu_3': 20}}
- GCH (guided or unguided decision): {'clean': {'DH_3': 20}, 'gauss20': {'DH_3': 20}, 'noisy15': {'DH_3': 20}}
